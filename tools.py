import base64
import datetime
import json
import math
import os
import random
from collections import deque

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
import time
from functools import wraps

import requests
from django.shortcuts import redirect
from loguru import logger
from scipy.spatial.distance import pdist

from onlineReading import settings


def login_required(func):
    @wraps(func)
    def inner(request, *args, **kwargs):
        if request.session.get('username'):
            return func(request, *args, **kwargs)
        return redirect("/go_login/")

    return inner


class Timer:
    def __init__(self, name):
        self.elapsed = 0
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = round((self.end - self.start) * 1000, 2)
        logger.info(f"执行{self.name}用时{self.elapsed}ms")


def translate(content):
    """翻译接口"""
    try:
        starttime = datetime.datetime.now()
        # 1. 准备参数
        appid = settings.APPID
        salt = "990503"
        secret = settings.SECRET
        sign = appid + content + salt + secret
        # 2. 将sign进行md5加密
        import hashlib

        Encry = hashlib.md5()
        Encry.update(sign.encode())
        sign = Encry.hexdigest()
        # 3. 发送请求
        url = f"http://api.fanyi.baidu.com/api/trans/vip/translate?q={content}&from=en&to=zh&appid={appid}&salt={salt}&sign={sign}"
        # 4. 解析结果
        response = requests.get(url)
        data = json.loads(response.text)
        endtime = datetime.datetime.now()
        logger.info(
            f"翻译接口执行时间为{round((endtime - starttime).microseconds / 1000 / 1000, 3)}ms"
        )
        return {"status": 200, "zh": data["trans_result"][0]["dst"]}
    except Exception:
        return {"status": 500, "zh": None}


def format_gaze(
        gaze_x: str, gaze_y: str, gaze_t: str, use_filter=True, begin_time: int = 500, end_time: int = 500
) -> list:
    """将gaze点格式化"""
    list_x = []
    list_y = []
    list_t = []
    if type(gaze_x) == list:
        list_x = gaze_x
        list_y = gaze_y
        list_t = gaze_t
    else:
        for item in gaze_x.split(","):
            if len(item) > 0:
                list_x.append(float(item))

        for item in gaze_y.split(","):
            if len(item) > 0:
                list_y.append(float(item))

        for item in gaze_t.split(","):
            if len(item) > 0:
                list_t.append(float(item))

    print(f'len(x):{len(list_x)}')
    # 时序滤波
    if use_filter:
        filters = [
            {"type": "median", "window": 7},
            {"type": "median", "window": 7},
            {"type": "mean", "window": 5},
            {"type": "mean", "window": 5},
        ]
        list_x = preprocess_data(list_x, filters)
        list_y = preprocess_data(list_y, filters)

    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))
    list_t = list(map(int, list_t))
    assert len(list_x) == len(list_y) == len(list_t)
    gaze_points = [[list_x[i], list_y[i], list_t[i]] for i in range(len(list_x))]

    begin = next(
        (
            i
            for i, gaze in enumerate(gaze_points)
            if gaze[2] - gaze_points[0][2] > begin_time
        ),
        0,
    )
    end = next(
        (
            i
            for i in range(len(gaze_points) - 1, 0, -1)
            if gaze_points[-1][2] - gaze_points[i][2] > end_time
        ),
        0,
    )
    assert begin < end
    print(f'len(end_x):{end - begin + 1}')
    return gaze_points[begin:end]


def preprocess_data(data, filters):
    """滤波接口"""
    for layer in filters:
        if layer["type"] == "median":
            data = signal.medfilt(data, kernel_size=layer["window"])
        if layer["type"] == "mean":
            data = meanFilter(data, layer["window"])
    return data


def meanFilter(data, win):
    """均值滤波"""
    length = len(data)
    res = np.zeros(length)
    for i in range(length):
        s, n = 0, 0
        for j in range(i - win // 2, i + win - win // 2):
            if j < 0 or j >= length:
                continue
            s += data[j]
            n += 1
        res[i] = s / n
    return res


def fixation_word_mapping(fixations, locations, rows):
    adjust_fixations = []
    for fix in fixations:
        index, find_near = get_item_index_x_y(location=locations, x=fix[0], y=fix[1])
        if index != -1:
            if find_near:
                # 此处是为了可视化看起来清楚
                loc = locations[index]
                fix[0] = (loc["left"] + loc["right"]) / 2
                fix[1] = (loc["top"] + loc["bottom"]) / 2
            row_index, index_in_row = word_index_in_row(rows, index)
            if index_in_row != -1:
                # x,y,duration,文章中第几个词，一行中第几个词，第几行，起始时间，结束时间
                adjust_fix = [fix[0], fix[1], fix[2], index, index_in_row, row_index, fix[3], fix[4]]
                adjust_fixations.append(adjust_fix)
    return adjust_fixations


def split_fixation_by_row(adjust_fixations, rows):
    """将fixaton按行切割"""
    # 宁可多切，不可少切 TODO 假设每个人每行都读，实际上不一定满足这种假设
    begin_index = 0
    sequence_fixations = []
    for i, fix in enumerate(adjust_fixations):
        sequence = adjust_fixations[begin_index:i]
        y_list = np.array([x[1] for x in sequence])
        y_mean = np.mean(y_list)
        row_ind = row_index_of_sequence(rows, y_mean)
        word_num_in_row = rows[row_ind]["end_index"] - rows[row_ind]["begin_index"] + 1
        for j in range(i, begin_index, -1):
            if adjust_fixations[j][4] - fix[4] > int(word_num_in_row / 2):
                tmp = adjust_fixations[begin_index: j + 1]
                mean_interval = 0
                for f in range(1, len(tmp)):
                    mean_interval = mean_interval + abs(tmp[f][0] - tmp[f - 1][0])
                mean_interval = mean_interval / (len(tmp) - 1)
                data = pd.DataFrame(
                    tmp, columns=["x", "y", "t", "index", "index_in_row", "row_index", "begin_time", "end_time"]
                )
                if len(set(data["row_index"])) > 1:
                    row_indexs = list(data["row_index"])
                    start = 0
                    for ind in range(start, len(row_indexs)):
                        if (
                                row_indexs[ind] < row_indexs[ind - 1]
                                and abs(tmp[ind][0] - tmp[ind - 1][0]) > mean_interval * 2
                        ):
                            if len(tmp[start:ind]) > 0:
                                sequence_fixations.append(tmp[start:ind])
                            start = ind
                    if 0 < start < len(row_indexs) - 1:
                        if len(tmp[start:-1]) > 0:
                            sequence_fixations.append(tmp[start:-1])
                    elif start == 0:
                        if len(tmp) > 0:
                            sequence_fixations.append(tmp)
                elif len(tmp) > 0:
                    sequence_fixations.append(tmp)
                # sequence_fixations.append(adjust_fixations[begin_index:i])
                begin_index = i
                break
    if begin_index != len(adjust_fixations) - 1:
        sequence_fixations.append(adjust_fixations[begin_index:-1])

    return sequence_fixations


def move_fixation_by_no_blank_row_assumption(sequence_fixations, rows, len_per_word):
    """利用无空行先验调整fixation"""
    result_rows = []
    result_fixations = []
    row_level_fix = []

    now_max_row = -1
    row_pass_time = [0 for _ in range(len(rows))]

    for i, sequence in enumerate(sequence_fixations):
        y_list = np.array([x[1] for x in sequence])
        y_mean = np.mean(y_list)
        row_index = row_index_of_sequence(rows, y_mean)
        rows_per_fix = []
        for y in y_list:
            row_index_this_fix = row_index_of_sequence(rows, y)
            rows_per_fix.append(row_index_this_fix)

        # 假设不会出现空行
        if row_index > now_max_row + 1:
            if row_pass_time[now_max_row] >= 2:
                random_number = random.randint(0, 1)
                if random_number == 0:
                    # 把上一行拉下来
                    # 这一行没定位错，上一行定位错了
                    row_pass_time[result_rows[-1]] -= 1
                    result_rows[-1] = now_max_row + 1
                    row_pass_time[row_index] += 1
                    result_rows.append(row_index)
                else:
                    # 把下一行拉上去，这一行定位错了
                    # 如果上一行是短行，则不进行调整
                    row_left = rows[now_max_row + 1]['left']
                    row_right = rows[now_max_row + 1]['right']

                    if row_right - row_left <= len_per_word * 5:
                        row_pass_time[row_index] += 1
                        result_rows.append(row_index)
                    else:
                        row_pass_time[now_max_row + 1] += 1
                        result_rows.append(now_max_row + 1)
            else:
                # 如果上一行是短行，则不进行调整
                row_left = rows[now_max_row + 1]['left']
                row_right = rows[now_max_row + 1]['right']

                if row_right - row_left <= len_per_word * 5:
                    row_pass_time[row_index] += 1
                    result_rows.append(row_index)
                else:
                    row_pass_time[now_max_row + 1] += 1
                    result_rows.append(now_max_row + 1)
        else:
            row_pass_time[row_index] += 1
            result_rows.append(row_index)
        now_max_row = max(result_rows)

    assert len(result_rows) == len(sequence_fixations)

    for i, sequence in enumerate(sequence_fixations):
        if result_rows[i] != -1:
            adjust_y = (rows[result_rows[i]]["top"] + rows[result_rows[i]]["bottom"]) / 2
            result_fixation = [[x[0], adjust_y, x[2], x[6], x[7]] for x in sequence]
            result_fixations.extend(result_fixation)
            row_level_fix.append(result_fixation)
    return result_fixations, result_rows, row_level_fix


def generate_fixations(gaze_points, texts, location):
    """生成fixation"""
    # 根据gaze点生成fixation，未校准
    fixations = detect_fixations(gaze_points)
    # 对fixation的y轴做滤波（时序性的假设，对于行两侧的数据不友好，该假设太简单，TODO 更新算法）
    fixations = keep_row(fixations)

    word_list, sentence_list = get_word_and_sentence_from_text(texts)  # 获取单词和句子对应的index
    border, rows, danger_zone, len_per_word = textarea(location)
    locations = json.loads(location)

    assert len(word_list) == len(locations)

    # 根据fix确定对应的单词，若无对应，则找最近
    # TODO 所有没有对应单词的fixation都被删除了，这有点问题
    # adjust_fixations:[x,y,duration,文章中第几个词，一行中第几个词，第几行，起始时间，结束时间]
    adjust_fixations = fixation_word_mapping(fixations, locations, rows)

    # 将fixation按行切割
    sequence_fixations = split_fixation_by_row(adjust_fixations, rows)

    # 根据行先验调整fixations
    result_fixations, result_rows, row_level_fix = move_fixation_by_no_blank_row_assumption(sequence_fixations, rows, len_per_word)

    return result_fixations, result_rows, row_level_fix, sequence_fixations


def detect_fixations(
        gaze_points: list, min_duration: int = 200, max_duration: int = 10000, max_dispersion: int = 80
) -> list:
    """
    根据gaze点生成fixations
    fixation的特征：
    + duration：
        + 200~400ms or 150~600ms
        + rarely shorter than 200ms,longer than 800ms
        + very short fixations is not meaningful for studying behavior
        + smooth pursuit is included without maximum duration
            + from other perspective，compensation for VOR（what？）
    + mean duration:150~300ms
    fixation的输出：
    + [[x, y, duration, begin, end],...]
    """
    # TODO dispersion的默认值
    # 1. 舍去异常的gaze点，confidence？
    # 2. 开始处理
    working_queue = deque()
    remaining_gaze = deque(gaze_points)

    fixation_list = []
    # format of gaze: [x,y,t]
    while remaining_gaze:
        # check for min condition (1. two gaze 2. reach min duration)
        if len(working_queue) < 2 or working_queue[-1][2] - working_queue[0][2] < min_duration:
            working_queue.append(remaining_gaze.popleft())
            continue
        # min duration reached,check for fixation
        dispersion = gaze_dispersion(list(working_queue))  # 值域：[0:pai]
        if dispersion > max_dispersion:
            working_queue.popleft()
            continue

        left_idx = len(working_queue)
        # minimal fixation found,collect maximal data
        while (
                remaining_gaze
                and not remaining_gaze[0][2] > working_queue[0][2] + max_duration
        ):
            working_queue.append(remaining_gaze.popleft())

        # check for fixation with maximum duration
        dispersion = gaze_dispersion(list(working_queue))
        if dispersion <= max_dispersion:
            fixation_list.append(gaze_2_fixation(list(working_queue)))  # generate fixation
            working_queue.clear()
            continue

        right_idx = len(working_queue)
        slicable = list(working_queue)

        # binary search
        while left_idx < right_idx - 1:
            middle_idx = (left_idx + right_idx) // 2
            dispersion = gaze_dispersion(slicable[: middle_idx + 1])
            if dispersion <= max_dispersion:
                left_idx = middle_idx
            else:
                right_idx = middle_idx

        final_base_data = slicable[:left_idx]
        to_be_placed_back = slicable[left_idx:]
        fixation_list.append(gaze_2_fixation(final_base_data))  # generate fixation
        working_queue.clear()
        remaining_gaze.extendleft(reversed(to_be_placed_back))

    return fixation_list


def gaze_dispersion(gaze_points: list) -> int:
    """计算gaze点的dispersion"""
    gaze_points = [[x[0], x[1]] for x in gaze_points]
    # TODO 为什么pupil lab中使用的cosine
    distances = pdist(gaze_points, metric="euclidean")  # 越相似，距离越小
    return distances.max()


def gaze_2_fixation(gaze_points: list) -> list:
    """将一组gaze点组合为fixation"""
    duration = gaze_points[-1][2] - gaze_points[0][2]
    x = np.mean([x[0] for x in gaze_points])
    y = np.mean([x[1] for x in gaze_points])
    begin = gaze_points[0][2]
    end = gaze_points[-1][2]
    return [x, y, duration, begin, end]


def keep_row(fixations: list, kernel_size=9):
    """利用窗口均值取代当前值，本质是做滤波"""
    y = [fix[1] for fix in fixations]

    y = signal.medfilt(y, kernel_size=kernel_size)
    y = meanFilter(y, kernel_size)

    assert len(y) == len(fixations)
    for i, fixation in enumerate(fixations):
        fixation[1] = y[i]
    return fixations


def get_item_index_x_y(location, x, y):
    """根据所有item的位置，当前给出的x,y,判断其在哪个item里 分为word level和row level"""

    flag = False
    index = -1
    # 先找是否正好在范围内
    for i, word in enumerate(location):
        if word["left"] <= x <= word["right"] and word["top"] <= y <= word["bottom"]:
            index = i
            return index, False

    # 如果不在范围内,找最近的单词
    min_dist = 50

    # for i, word in enumerate(location):
    #     point = np.array([x, y])
    #     segments = [
    #         [np.array([word["left"], word["top"]]), np.array([word["left"], word["bottom"]])],  # 左上 左下
    #         [np.array([word["left"], word["top"]]), np.array([word["right"], word["top"]])],  # 左上 右上
    #         [np.array([word["left"], word["bottom"]]), np.array([word["right"], word["bottom"]])],  # 左下 右下
    #         [np.array([word["right"], word["top"]]), np.array([word["right"], word["bottom"]])],  # 右上 右下
    #     ]
    #     dis_list = [point_to_segment_distance(point, segment[0], segment[1]) for segment in segments]
    #     distance = min(dis_list)
    #     if distance < min_dist:
    #         min_dist = distance
    #         index = i
    #         flag = True
    print(f"min_dist:{min_dist}")
    return index, flag


def get_euclid_distance(coor1, coor2):
    """计算欧式距离"""
    coor1 = (float(coor1[0]), float(coor1[1]))
    coor2 = (float(coor2[0]), float(coor2[1]))

    x_pow = math.pow(coor1[0] - coor2[0], 2)
    y_pow = math.pow(coor1[1] - coor2[1], 2)
    return math.sqrt(x_pow + y_pow)


def point_to_segment_distance(point, segment_start, segment_end):
    """计算点到线段的最短距离"""
    """
    Calculates the shortest distance between a point and a line segment defined by two endpoints.
    """
    # Calculate the vector between the segment start and end points
    segment_vector = segment_end - segment_start

    # Calculate the vector between the segment start point and the query point
    point_vector = point - segment_start

    # Calculate the projection of the point vector onto the segment vector
    projection = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)

    # If the projection is less than 0, the closest point is the segment start point
    if projection < 0:
        closest_point = segment_start
    # If the projection is greater than 1, the closest point is the segment end point
    elif projection > 1:
        closest_point = segment_end
    # Otherwise, the closest point is on the segment between the start and end points
    else:
        closest_point = segment_start + projection * segment_vector

    return np.linalg.norm(point - closest_point)


def get_word_and_sentence_from_text(content):
    """将文本切割成单词和句子"""
    # TODO 使用与模型一致的切割方法 模型=前端=后端
    sentences = content.replace("...", ".").replace("..", ".").split(".")
    sentence_list = []
    word_list = []
    cnt = 0
    begin = 0
    for sentence in sentences:
        if len(sentence) > 2:
            sentence = sentence.strip()
            words = sentence.split(" ")
            for word in words:
                if len(word) > 0:
                    # 根据实际情况补充，或者更改为正则表达式（是否有去除数字的风险？）
                    word = word.strip().replace('"', "").replace(",", "")
                if len(word) > 0:
                    word_list.append(word)
                    cnt += 1
            end = cnt
            sentence_list.append((sentence, begin, end, end - begin))  # (句子文本，开始的单词序号，结束的单词序号+1，长度)
            begin = cnt
    return word_list, sentence_list


def textarea(locations: str, danger_r: int = 8) -> tuple:
    """
    确定文本区域的边界
    确定每一行的边界
    """
    locations = json.loads(locations)
    assert len(locations) > 0
    rows = []

    danger_zone = []
    pre_top = locations[0]["top"]
    begin_left = locations[0]["left"]

    word_num_per_row = 0
    first_row = True
    begin_index = 0
    for i, loc in enumerate(locations):
        if first_row:
            word_num_per_row += 1
        if i == 0:
            continue
        if loc["top"] != pre_top:
            first_row = False
            # 发生了换行
            row = {
                "left": begin_left,
                "top": locations[i - 1]["top"],
                "right": locations[i - 1]["right"],
                "bottom": locations[i - 1]["bottom"],
                "begin_index": begin_index,
                "end_index": i - 1,
            }
            rows.append(row)

            row_length = row["right"] - row["left"]
            range_x = [row["left"] + (1 / 5) * row_length, row["right"]]
            range_y = [loc["top"] - danger_r, loc["top"] + danger_r]
            zone = [range_x, range_y]
            danger_zone.append(zone)

            pre_top = loc["top"]
            begin_left = loc["left"]
            begin_index = i

        if i == len(locations) - 1:
            # 最后一行不可能发生换行
            row = {
                "left": begin_left,
                "top": loc["top"],
                "right": loc["right"],
                "bottom": loc["bottom"],
                "begin_index": begin_index,
                "end_index": i,
            }
            rows.append(row)
    border = {
        "left": rows[0]["left"],
        "top": rows[0]["top"],
        "right": rows[0]["right"],  # 实际上right不完全相同
        "bottom": rows[-1]["bottom"],
    }
    # print("word_num")
    # print(word_num_per_row)
    return border, rows, danger_zone, (rows[0]["right"] - rows[0]["left"]) / word_num_per_row


def word_index_in_row(rows, word_index):
    """根据index和row确定单词在row中的位置"""
    return next(
        (
            (i, word_index - row["begin_index"])
            for i, row in enumerate(rows)
            if row["end_index"] >= word_index >= row["begin_index"]
        ),
        (-1, -1),
    )


def row_index_of_sequence(rows, y):
    """根据y轴坐标确定行号"""
    return next(
        (i for i, row in enumerate(rows) if row["bottom"] >= y >= row["top"]),
        -1,
    )

def generate_pic_by_base64(image_base64: str, path:str, isMac=False):
    """
    使用base64生成图片，并保存至指定路径
    """
    data = image_base64.split(",")[1]
    img_data = base64.b64decode(data)

    with open(path, "wb") as f:
        f.write(img_data)
    logger.info(f"background已在该路径下生成:{path}")

    # img = Image.open(save_path + filename)
    # new_img = img.resize((1920, 1080))
    # new_img.save(save_path + filename)

    # if isMac:
    #     img = Image.open(save_path + filename)
    #     width = img.size[0]
    #     times = width / 2880
    #     new_img = img.resize((int(1440 * times), int(900 * times)))
    #     new_img.save(save_path + filename)
    return path

def show_fixations(fixations: list, background: str):
    """根据fixation画图"""
    canvas = cv2.imread(background)
    canvas = paint_fixations(canvas, fixations)
    return canvas

def paint_fixations(canvas, fixations, interval=1, label=1, line=True):
    """根据fixation画图"""
    fixations = [x for i, x in enumerate(fixations) if i % interval == 0]
    for i, fix in enumerate(fixations):
        x = int(fix[0])
        y = int(fix[1])
        cv2.circle(
            canvas,
            (x, y),
            3,
            (0, 0, 255),
            -1,
        )
        if i % label == 0:
            cv2.putText(
                canvas,
                str(i),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
        if i > 0 and line:
            cv2.line(
                canvas,
                (x, y),
                (int(fixations[i - 1][0]), int(fixations[i - 1][1])),
                (0, 0, 255),  # GBR
                1,
            )
    return canvas

def get_word_location(location):
    """获取单词的位置范围"""
    word_location = []
    locations = json.loads(location)
    for loc in locations:
        tmp_tuple = (loc["left"], loc["top"], loc["right"], loc["bottom"])
        word_location.append(tmp_tuple)
    return word_location

def paint_on_word(image, target_words_index, word_locations, title, pic_path, alpha=0.1, color=255):
    """在指定位置上涂色"""
    blk = np.zeros(image.shape, np.uint8)
    blk[0:image.shape[0] - 1, 0:image.shape[1] - 1] = 255
    set_title(blk, title)
    for word_index in target_words_index:
        loc = word_locations[word_index]
        cv2.rectangle(
            blk,
            (int(loc[0]), int(loc[1])),
            (int(loc[2]), int(loc[3])),
            (color, 0, 0),
            -1,
        )
    image = cv2.addWeighted(blk, alpha, image, 1 - alpha, 0)
    plt.imshow(image)
    plt.title(title)
    cv2.imwrite(pic_path, image)
    logger.info(f"heatmap已经生成:{pic_path}")

def set_title(blk, title):
    """设置图片标题"""
    cv2.putText(
        blk,
        str(title),  # text内容必须是str格式的
        (600, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (189, 252, 201),
        2,
    )




class FeatureSet(object):

    def __init__(self, num):
        self.num = num
        # word level
        self.total_fixation_duration = [0 for _ in range(num)]
        self.number_of_fixation = [0 for _ in range(num)]
        self.reading_times = [0 for _ in range(num)]
        # sentence level
        self.total_dwell_time = [0 for _ in range(num)]
        self.saccade_times = [0 for _ in range(num)]
        self.forward_saccade_times = [0 for _ in range(num)]
        self.backward_saccade_times = [0 for _ in range(num)]


        # 句子长度
        self.sentence_length = [0 for _ in range(num)]

        # label
        self.word_understand = []
        self.sentence_understand = []
        self.mind_wandering = []

        self.page_data = [0 for _ in range(num)]
        self.sentence_index = [0 for _ in range(num)]

        # word_list
        self.word_list = []

        # watching
        self.is_watching = [0 for _ in range(num)]

    def clean(self):
        self.total_fixation_duration = [0 for _ in range(self.num)]
        self.number_of_fixation = [0 for _ in range(self.num)]
        self.reading_times = [0 for _ in range(self.num)]

        self.total_dwell_time = [0 for _ in range(self.num)]
        self.saccade_times = [0 for _ in range(self.num)]
        self.forward_saccade_times = [0 for _ in range(self.num)]
        self.backward_saccade_times = [0 for _ in range(self.num)]


    def clean_watching(self):
        self.is_watching = [0 for _ in range(self.num)]

    def setLabel(self, label1, label2, label3):
        self.word_understand = label1
        self.sentence_understand = label2
        self.mind_wandering = label3

    def setWordList(self, word_list):
        self.word_list = word_list

    def get_list_div(self, list_a, list_b):
        div_list = [0 for _ in range(self.num)]
        for i in range(len(list_b)):
            if list_b[i] != 0:
                div_list[i] = list_a[i] / list_b[i]

        return div_list

    def list_log(self, list_a):
        log_list = [0 for _ in range(self.num)]
        for i in range(len(list_a)):
            log_list[i] = math.log(list_a[i] + 1)
        return log_list

    def to_csv(self, filename, exp_id, user, article_id, page_id, time):
        df = pd.DataFrame(
            {
                # 1. 实验信息相关
                "experiment_id": [exp_id for _ in range(self.num)],
                "user": [user for _ in range(self.num)],
                "article_id": [article_id for _ in range(self.num)],
                "time": [time for _ in range(self.num)],
                "word": self.word_list,

                "page_id": [page_id for _ in range(self.num)],
                "sentence": self.sentence_index,
                "word_watching": self.is_watching,

                # # 2. label相关
                "word_understand": self.word_understand,
                "sentence_understand": self.sentence_understand,
                "mind_wandering": self.mind_wandering,
                # 3. 特征相关
                # word level
                "reading_times": self.reading_times,
                "number_of_fixations": self.number_of_fixation,
                "fixation_duration": self.total_fixation_duration,
                # sentence_level_raw
                "total_dwell_time_of_sentence": self.total_dwell_time,
                "saccade_times_of_sentence": self.saccade_times,
                "forward_times_of_sentence": self.forward_saccade_times,
                "backward_times_of_sentence": self.backward_saccade_times,



                # sentence_level_div_length
                "total_dwell_time_of_sentence_div_length": self.get_list_div(self.total_dwell_time,
                                                                             self.sentence_length),
                "saccade_times_of_sentence_div_length": self.get_list_div(self.saccade_times, self.sentence_length),
                "forward_times_of_sentence_div_length": self.get_list_div(self.forward_saccade_times,
                                                                          self.sentence_length),
                "backward_times_of_sentence_div_length": self.get_list_div(self.backward_saccade_times,
                                                                           self.sentence_length),

                # sentence_level_div_log
                "total_dwell_time_of_sentence_div_log": self.get_list_div(self.total_dwell_time,
                                                                          self.list_log(self.sentence_length)),
                "saccade_times_of_sentence_div_log": self.get_list_div(self.saccade_times,
                                                                       self.list_log(self.sentence_length)),
                "forward_times_of_sentence_div_log": self.get_list_div(self.forward_saccade_times,
                                                                       self.list_log(self.sentence_length)),
                "backward_times_of_sentence_div_log": self.get_list_div(self.backward_saccade_times,
                                                                        self.list_log(self.sentence_length)),



            }
        )

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode="a", header=False)
        else:
            df.to_csv(filename, index=False, mode="a")

def compute_label(wordLabels, sentenceLabels, wanderLabels, word_list):
    """
    计算单词的标签
    """
    word_understand = [1 for _ in word_list]
    if wordLabels:
        wordLabels = json.loads(wordLabels)
        for label in wordLabels:
            word_understand[label] = 0

    sentence_understand = [1 for _ in word_list]
    if sentenceLabels:
        sentenceLabels = json.loads(sentenceLabels)
        for label in sentenceLabels:
            for i in range(label[0], label[1]):
                sentence_understand[i] = 0

    mind_wandering = [1 for _ in word_list]
    if wanderLabels:
        wanderLabels = json.loads(wanderLabels)
        for label in wanderLabels:
            for i in range(label[0], label[1] + 1):
                mind_wandering[i] = 0

    return word_understand, sentence_understand, mind_wandering

def compute_sentence_label(sentenceLabels, wanderLabels, sentence_list):
    """
    计算单词的标签
    """

    sentence_understand = [1 for _ in sentence_list]
    if sentenceLabels:
        sentenceLabels = json.loads(sentenceLabels)
        for label in sentenceLabels:
            for i,sentence in enumerate(sentence_list):
                if label[0] == sentence[1] and label[1] == sentence[2]:
                    sentence_understand[i] = 0

    mind_wandering = [1 for _ in sentence_list]
    if wanderLabels:
        wanderLabels = json.loads(wanderLabels)
        for label in wanderLabels:
            for i, sentence in enumerate(sentence_list):
                if label[0] == sentence[1] and label[1] == sentence[2]:
                    mind_wandering[i] = 0

    return sentence_understand, mind_wandering

def get_fix_by_time(fixations, start,end):
    """获取在某个时间内的fixation点"""
    fixs = []
    for fix in fixations:
        if start<= fix[-1] <= end:
            fixs.append(fix)
        if fix[-1] > end: # fix是按照时间顺序，减少遍历的内容
            break
    return fixs
def is_watching(fixations, location, num):
    """计算哪些单词正在被关注"""
    watching = [0 for _ in range(num)]
    for fix in fixations:
        index, isAdjust = get_item_index_x_y(location, fix[0], fix[1])
        if index != -1:
            watching[index] = 1
    return watching
def get_sentence_by_word(word_index, sentence_list):
    """查询单词所在的句子"""
    if word_index == -1:
        return -1
    return next(
        (
            i
            for i, sentence in enumerate(sentence_list)
            if sentence[2] > word_index >= sentence[1]
        ),
        -1,
    )

def generate_exp_csv(filename,experiments):
    """生成exp的信息"""
    pd.DataFrame({
        'exp_id':[exp.id for exp in experiments],
        'user':[exp.user for exp in experiments],
        'article_id':[exp.article_id for exp in experiments]
    }).to_csv(filename, index=False)

def round_list(a,num):
    return list(np.round(np.array(a),num))

def div_list(list_a, list_b):
    assert len(list_a) == len(list_b)
    div_list = [0 for _ in range(len(list_a))]
    for i in range(len(list_b)):
        if list_b[i] != 0:
            div_list[i] = list_a[i] / list_b[i]
    return div_list

def coor_to_input(coordinates, window):
    """
    将每个gaze点赋值上标签
    :param gaze_x:
    :param gaze_y:
    :param gaze_t:
    :param window:
    :return:
    """
    for i, coordinate in enumerate(coordinates):
        # 确定计算的窗口
        begin = i
        end = i
        for _ in range(int(window / 2)):
            if begin == 0:
                break
            begin -= 1

        for _ in range(int(window / 2)):
            if end == len(coordinates) - 1:
                break
            end += 1

        assert begin >= 0
        assert end <= len(coordinates) - 1

        # 计算速度、方向，作为模型输入
        time = (coordinates[end][2] - coordinates[begin][2]) / 100
        speed = 0
        if time != 0:
            speed = (
                    get_euclid_distance((coordinates[begin][0],coordinates[begin][1]),(coordinates[end][0],
                                        coordinates[end][1]))
                    / time
            )
        else:
            speed = 0

        direction = math.atan2(coordinates[end][1] - coordinates[begin][1], coordinates[end][0] - coordinates[begin][0])
        coordinate.append(speed)
        coordinate.append(direction)
    # 计算加速度
    for i, coordinate in enumerate(coordinates):
        begin = i
        end = i
        for _ in range(int(window / 2)):
            if begin == 0:
                break
            begin -= 1

        for _ in range(int(window / 2)):
            if end == len(coordinates) - 1:
                break
            end += 1

        assert begin >= 0
        assert end <= len(coordinates) - 1
        time = (coordinates[end][2] - coordinates[begin][2]) / 100
        acc = (coordinates[end][3] - coordinates[begin][3]) / time if time != 0 else 0
        coordinate.append(acc * acc)

    speed = [x[3] for x in coordinates]
    direction = [x[4] for x in coordinates]
    acc = [x[5] for x in coordinates]
    return speed, direction, acc

if __name__ == '__main__':
    point = np.array([5, 4])
    segment_start = np.array([1, 3])
    segment_end = np.array([2, 3])
    distance = point_to_segment_distance(point, segment_start, segment_end)
    print(f"distance:{distance}")
