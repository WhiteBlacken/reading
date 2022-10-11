import base64
import json
import math
import os
import datetime

import numpy as np
import requests
from loguru import logger
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from scipy import signal

from onlineReading import settings


def get_fixations(coordinates):
    """
    根据gaze data(x,y,t1)计算fixation(x,y,r,t2) t2远小于t1
    :param coordinates: [(x,y,t),(x,y,t)]   # gaze点
    :return: [(x,y,duration),(x,y,duration)]  # fixation点
    """
    from collections import deque

    fixations = []
    min_duration = 100
    max_duration = 1200
    max_distance = 140
    # 先进先出队列
    working_queue = deque()
    remaining_gaze = deque(coordinates)
    print("gaze length:%d" % len(remaining_gaze))
    while remaining_gaze:
        # 逐个处理所有的gaze data
        if (
                len(working_queue) < 2
                or (working_queue[-1][2] - working_queue[0][2]) < min_duration
        ):
            # 如果当前无要处理的gaze或gaze间隔太短--再加一个gaze后再来处理
            datum = remaining_gaze.popleft()
            working_queue.append(datum)
            continue
        # 如果队列中两点任意距离超过max_distance，则不是一个fixation
        flag = False
        for i in range(len(working_queue) - 1):
            for j in range(i + 1, len(working_queue) - 1):
                if not with_distance(working_queue[i], working_queue[j], max_distance):
                    # not a fixation,move forward
                    working_queue.popleft()
                    flag = True
                    break
            if flag:
                break
        if flag:
            continue

        # minimal fixation found,collect maximal data
        while remaining_gaze:
            datum = remaining_gaze[0]
            if datum[2] > working_queue[0][2] + max_duration or with_distance(
                    working_queue[0], datum, max_distance
            ):
                fixations.append(from_gazes_to_fixation(list(working_queue)))
                working_queue.clear()
                break  # maximum data found
            working_queue.append(remaining_gaze.popleft())
    print("fixaions:%s" % fixations)
    return fixations


def with_distance(gaze1, gaze2, max_distance):
    """判断两个gaze点之间的距离是否满足fixation"""
    return get_euclid_distance(gaze1[0], gaze2[0], gaze1[1], gaze2[1]) < max_distance


def from_gazes_to_fixation(gazes):
    """
    通过gaze序列，计算fixation
    gazes：tuple(x,y,t)
    """
    # fixation 三要素：x,y,r r表示时长/半径
    sum_x = 0
    sum_y = 0
    for gaze in gazes:
        sum_x = sum_x + gaze[0]
        sum_y = sum_y + gaze[1]

    return int(sum_x / len(gazes)), int(sum_y / len(gazes)), gazes[-1][2] - gazes[0][2]


def get_euclid_distance(x1, x2, y1, y2):
    """计算欧式距离"""
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def add_fixations_to_location(fixations, locations):
    """
    给出fixations，将fixation与单词对应起来  word level/row level
    :param fixations: [(坐标x,坐标y,durations),(坐标x,坐标y,durations)]
    :param locations: '[{"left:220,"top":23,"right":222,"bottom":222},{"left:220,"top":23,"right":222,"bottom":222}]'
    :return: {"0":[(坐标x,坐标y,durations),(坐标x,坐标y,durations)],"3":...}
    """
    words_fixations = {}
    for fixation in fixations:
        index = get_item_index_x_y(locations, fixation[0], fixation[1])
        if index != -1:
            if index in words_fixations.keys():
                # 如果当前索引已经存在
                tmp = words_fixations[index]
                tmp.append(fixation)
            else:
                tmp = [fixation]
            words_fixations[index] = tmp
    print(words_fixations)
    return words_fixations


def reading_times(words_fixations):
    reading_times = {}
    for key in words_fixations:
        reading_times[key] = len(words_fixations[key])
    return reading_times


def get_item_index_x_y(location, x, y):
    """根据所有item的位置，当前给出的x,y,判断其在哪个item里 分为word level和row level"""
    # 解析location
    location = json.loads(location)

    index = 0
    for word in location:
        if word["left"] <= x <= word["right"] and word["top"] <= y <= word["bottom"]:
            return index
        index = index + 1
    return -1


def get_word_and_location(location):
    word_and_location_dict = {}
    locations = json.loads(location)
    for i, loc in enumerate(locations):
        word_and_location_dict[i] = [
            loc["left"],
            loc["top"],
            loc["right"],
            loc["bottom"],
        ]
    return word_and_location_dict


def get_word_location(location):
    word_location = []
    locations = json.loads(location)
    for i, loc in enumerate(locations):
        tmp_tuple = (loc["left"], loc["top"], loc["right"], loc["bottom"])
        word_location.append(tmp_tuple)
    return word_location


def get_sentence_location(location, sentence_list):
    sentence_location = {}
    locations = json.loads(location)
    for i, loc in enumerate(locations):
        for j, sentence in enumerate(sentence_list):
            if sentence[2] > i >= sentence[1]:
                if j in sentence_location.keys():
                    sentence_location[j].append([loc["left"], loc["top"], loc["right"], loc["bottom"]])
                else:
                    sentence_location[j] = [[loc["left"], loc["top"], loc["right"], loc["bottom"]]]
    print("sentence_location")
    print(sentence_location)
    return sentence_location


def x_y_t_2_coordinate(gaze_x, gaze_y, gaze_t):
    """
    将gaze的x,y,z组合起来
    :param gaze_x: 637.8593938654011,651.1341242564663,610.7001684978032,604.6007775691435,608.401340526109  # str类型
    :param gaze_y: 637.8593938654011,651.1341242564663,610.7001684978032,604.6007775691435,608.401340526109  # str类型
    :param gaze_t: 637.8593938654011,651.1341242564663,610.7001684978032,604.6007775691435,608.401340526109  # str类型
    :return: [(x,y,t),(x,y,t)]
    """
    # 1. 处理坐标
    list_x = gaze_x.split(",")
    list_y = gaze_y.split(",")
    list_t = gaze_t.split(",")

    coordinates = []
    for i, item in enumerate(list_x):
        coordinate = (
            # int(float(list_x[i]) * 1920 / 1534),
            # int(float(list_y[i]) * 1920 / 1534),
            int(float(list_x[i])),
            int(float(list_y[i])),
            int(float(list_t[i])),
        )
        coordinates.append(coordinate)
    return coordinates


def fixation_image(image_base64, username, fixations, page_data_id):
    """
    将fixation点绘制到图片上
    :param image_base64: 图片的base64编码
    :param username: 用户
    :param fixations: fixation点
    :param page_data_id: 该数据对应在数据库中的id
    :return:
    """
    # 1. 处理图片
    data = image_base64.split(",")[1]
    # 将str解码为byte
    image_data = base64.b64decode(data)
    # 获取名称

    filename = "fixation.png"
    print("filename:%s" % filename)
    # 存储地址
    path = "static/data/heatmap/" + str(username) + "/" + str(page_data_id) + "/"
    logger.info("fixations轨迹已在该路径下生成:%s" % (path + filename))
    # 如果目录不存在，则创建目录
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + filename, "wb") as f:
        f.write(image_data)
    paint_image(path + filename, fixations)


def paint_image(path, coordinates):
    """
    在图片上绘画
    :param path: 图片的路径
    :param coordinates: fixation点的信息
    :return:
    """
    import cv2

    img = cv2.imread(path)
    cnt = 0
    pre_coordinate = (0, 0, 0)
    for coordinate in coordinates:
        cv2.circle(
            img,
            (coordinate[0], coordinate[1]),
            int(float(coordinate[2] / 30)),
            (0, 0, 255),
            1,
        )
        if cnt > 0:
            cv2.line(
                img,
                (pre_coordinate[0], pre_coordinate[1]),
                (coordinate[0], coordinate[1]),
                (0, 0, 255),
                2,
            )
        cnt = cnt + 1
        # 标序号
        cv2.putText(
            img,
            str(cnt),
            (coordinate[0], coordinate[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        pre_coordinate = coordinate
    cv2.imwrite(path, img)


# 示例:The Coral Sea reserve would cover almost 990 000 square kilometers and stretch as far as 1100 kilometers from the coast. Unveiled recently by environment minister Tony Burke, the proposal would be the last in a series of proposed marine reserves around Australia's coast.
def get_word_by_index(content):
    text = content.replace(",", " ").replace(".", " ").strip()
    contents = text.split(" ")
    index_2_word = {}
    cnt = 0
    for item in contents:
        if len(item) > 0:
            index_2_word[cnt] = item.strip().lower()
            cnt = cnt + 1
    return index_2_word


def get_index_by_word(content, word):
    word_indexes = []
    text = content.replace(",", " ").replace(".", " ").strip()
    text = text.replace("  ", " ")
    contents = text.split(" ")
    print(contents)
    for i, item in enumerate(contents):
        if len(item) > 0 and word.lower() == item.lower():
            word_indexes.append(i)
    return word_indexes


# [{"left":330,"top":95,"right":408.15625,"bottom":326.984375},{"left":408.15625,"top":95,"right":445.5,"bottom":326.984375},{"left":445.5,"top":95,"right":518.6875,"bottom":326.984375},{"left":518.6875,"top":95,"right":589.140625,"bottom":326.984375},{"left":589.140625,"top":95,"right":645.03125,"bottom":326.984375},{"left":645.03125,"top":95,"right":725.46875,"bottom":326.984375},{"left":725.46875,"top":95,"right":780.046875,"bottom":326.984375},{"left":780.046875,"top":95,"right":836.4375,"bottom":326.984375},{"left":836.4375,"top":95,"right":942.625,"bottom":326.984375},{"left":942.625,"top":95,"right":979.171875,"bottom":326.984375},{"left":979.171875,"top":95,"right":1055.796875,"bottom":326.984375},{"left":1055.796875,"top":95,"right":1113.015625,"bottom":326.984375},{"left":1113.015625,"top":95,"right":1162.203125,"bottom":326.984375},{"left":1162.203125,"top":95,"right":1231.65625,"bottom":326.984375},{"left":1231.65625,"top":95,"right":1283.859375,"bottom":326.984375},{"left":1283.859375,"top":95,"right":1315.421875,"bottom":326.984375},{"left":1315.421875,"top":95,"right":1343.21875,"bottom":326.984375},{"left":1343.21875,"top":95,"right":1430.078125,"bottom":326.984375},{"left":1430.078125,"top":95,"right":1507.3125,"bottom":326.984375},{"left":1507.3125,"top":95,"right":1543.859375,"bottom":326.984375},{"left":1543.859375,"top":95,"right":1678.296875,"bottom":326.984375},{"left":1678.296875,"top":95,"right":1722.234375,"bottom":326.984375},{"left":330,"top":326.984375,"right":379.1875,"bottom":558.96875},{"left":379.1875,"top":326.984375,"right":526.671875,"bottom":558.96875},{"left":526.671875,"top":326.984375,"right":596.25,"bottom":558.96875},{"left":596.25,"top":326.984375,"right":632.796875,"bottom":558.96875},{"left":632.796875,"top":326.984375,"right":742.984375,"bottom":558.96875},{"left":742.984375,"top":326.984375,"right":772.65625,"bottom":558.96875},{"left":772.65625,"top":326.984375,"right":800.453125,"bottom":558.96875},{"left":800.453125,"top":326.984375,"right":906.015625,"bottom":558.96875},{"left":906.015625,"top":326.984375,"right":946.9375,"bottom":558.96875},{"left":946.9375,"top":326.984375,"right":996.125,"bottom":558.96875},{"left":996.125,"top":326.984375,"right":1114.328125,"bottom":558.96875},{"left":1114.328125,"top":326.984375,"right":1255.125,"bottom":558.96875},{"left":1255.125,"top":326.984375,"right":1320.3125,"bottom":558.96875},{"left":1320.3125,"top":326.984375,"right":1403.90625,"bottom":558.96875},{"left":1403.90625,"top":326.984375,"right":1453.09375,"bottom":558.96875},{"left":1453.09375,"top":326.984375,"right":1535.078125,"bottom":558.96875},{"left":1535.078125,"top":326.984375,"right":1584.953125,"bottom":558.96875},{"left":1584.953125,"top":326.984375,"right":1641.859375,"bottom":558.96875},{"left":1641.859375,"top":326.984375,"right":1739.9375,"bottom":558.96875},{"left":330,"top":558.96875,"right":379.1875,"bottom":790.953125},{"left":379.1875,"top":558.96875,"right":467.59375,"bottom":790.953125},{"left":467.59375,"top":558.96875,"right":552.46875,"bottom":790.953125}]
# 根据bottom可以判
# 做两件事：切割成不同的句子，切割成不同的行
def get_sentence_by_word_index(content):
    sentences = content.split(".")
    sentence_cnt = 0
    word_cnt = 0
    sentence_dict = {}
    index_2_word = {}
    for sentence in sentences:
        if len(sentence) > 0:
            sen_begin = word_cnt
            sen_end = 0
            # 分割每一个句子
            words = sentence.strip().replace(",", "").split(" ")
            for word in words:
                if len(word) > 0:
                    index_2_word[word_cnt] = word.strip().lower()
                    word_cnt = word_cnt + 1
                    sen_end = word_cnt
            dict = {
                "sentence": sentence,
                "begin_word_index": sen_begin,
                "end_word_index": sen_end,
            }
            sentence_dict[sentence_cnt] = dict
            sentence_cnt = sentence_cnt + 1
    return index_2_word, sentence_dict


def get_row_by_word_index(row_list, word_index):
    cnt = 0
    for row in row_list:
        if row["end_word"] > word_index >= row["begin_word"]:
            return cnt
        cnt += 1
    return -1


# 样例 ： [{"left":330,"top":95,"right":408.15625,"bottom":326.984375},{"left":408.15625,"top":95,"right":445.5,"bottom":326.984375},{"left":445.5,"top":95,"right":518.6875,"bottom":326.984375},{"left":518.6875,"top":95,"right":589.140625,"bottom":326.984375},{"left":589.140625,"top":95,"right":645.03125,"bottom":326.984375},{"left":645.03125,"top":95,"right":725.46875,"bottom":326.984375},{"left":725.46875,"top":95,"right":780.046875,"bottom":326.984375},{"left":780.046875,"top":95,"right":836.4375,"bottom":326.984375},{"left":836.4375,"top":95,"right":942.625,"bottom":326.984375},{"left":942.625,"top":95,"right":979.171875,"bottom":326.984375},{"left":979.171875,"top":95,"right":1055.796875,"bottom":326.984375},{"left":1055.796875,"top":95,"right":1113.015625,"bottom":326.984375},{"left":1113.015625,"top":95,"right":1162.203125,"bottom":326.984375},{"left":1162.203125,"top":95,"right":1231.65625,"bottom":326.984375},{"left":1231.65625,"top":95,"right":1283.859375,"bottom":326.984375},{"left":1283.859375,"top":95,"right":1315.421875,"bottom":326.984375},{"left":1315.421875,"top":95,"right":1343.21875,"bottom":326.984375},{"left":1343.21875,"top":95,"right":1430.078125,"bottom":326.984375},{"left":1430.078125,"top":95,"right":1507.3125,"bottom":326.984375},{"left":1507.3125,"top":95,"right":1543.859375,"bottom":326.984375},{"left":1543.859375,"top":95,"right":1678.296875,"bottom":326.984375},{"left":1678.296875,"top":95,"right":1722.234375,"bottom":326.984375},{"left":330,"top":326.984375,"right":379.1875,"bottom":558.96875},{"left":379.1875,"top":326.984375,"right":526.671875,"bottom":558.96875},{"left":526.671875,"top":326.984375,"right":596.25,"bottom":558.96875},{"left":596.25,"top":326.984375,"right":632.796875,"bottom":558.96875},{"left":632.796875,"top":326.984375,"right":742.984375,"bottom":558.96875},{"left":742.984375,"top":326.984375,"right":772.65625,"bottom":558.96875},{"left":772.65625,"top":326.984375,"right":800.453125,"bottom":558.96875},{"left":800.453125,"top":326.984375,"right":906.015625,"bottom":558.96875},{"left":906.015625,"top":326.984375,"right":946.9375,"bottom":558.96875},{"left":946.9375,"top":326.984375,"right":996.125,"bottom":558.96875},{"left":996.125,"top":326.984375,"right":1114.328125,"bottom":558.96875},{"left":1114.328125,"top":326.984375,"right":1255.125,"bottom":558.96875},{"left":1255.125,"top":326.984375,"right":1320.3125,"bottom":558.96875},{"left":1320.3125,"top":326.984375,"right":1403.90625,"bottom":558.96875},{"left":1403.90625,"top":326.984375,"right":1453.09375,"bottom":558.96875},{"left":1453.09375,"top":326.984375,"right":1535.078125,"bottom":558.96875},{"left":1535.078125,"top":326.984375,"right":1584.953125,"bottom":558.96875},{"left":1584.953125,"top":326.984375,"right":1641.859375,"bottom":558.96875},{"left":1641.859375,"top":326.984375,"right":1739.9375,"bottom":558.96875},{"left":330,"top":558.96875,"right":379.1875,"bottom":790.953125},{"left":379.1875,"top":558.96875,"right":467.59375,"bottom":790.953125},{"left":467.59375,"top":558.96875,"right":552.46875,"bottom":790.953125}]
def get_row_location(buttons_locations):
    """
    根据button的位置切割行
    :param buttons_locations: button的位置
    :return: 开始和结束的单词的index，整行涵盖的区域
    """
    buttons = json.loads(buttons_locations)
    begin_word = 0
    row_index = 0
    top = buttons[0]["top"]
    rows_location = []
    for i, button in enumerate(buttons):
        # 换行时或者最后一个单词时触发
        if button["top"] != top or i == len(buttons) - 1:
            top = button["top"]
            print(top)
            print(i)
            row_dict = {
                "begin_word": begin_word,
                "end_word": i,
                "left": buttons[begin_word]["left"],
                "top": buttons[begin_word]["top"],
                "right": buttons[i - 1]["right"],
                "bottom": buttons[begin_word]["bottom"],
            }
            rows_location.append(row_dict)
            row_index = row_index + 1
            begin_word = i
    return rows_location


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
        url = (
                "http://api.fanyi.baidu.com/api/trans/vip/translate?"
                + "q=%s&from=en&to=zh&appid=%s&salt=%s&sign=%s"
                % (content, appid, salt, sign)
        )
        # 4. 解析结果
        response = requests.get(url)
        data = json.loads(response.text)
        endtime = datetime.datetime.now()
        logger.info(
            "翻译接口执行时间为%sms" % round((endtime - starttime).microseconds / 1000 / 1000, 3)
        )
        return {"status": 200, "zh": data["trans_result"][0]["dst"]}
    except Exception as e:
        return {"status": 500, "zh": None}


def get_out_of_screen_times(coordinates):
    out_of_screen_times = 0
    for gaze in coordinates:
        if gaze[0] < 0 or gaze[1] < 0:
            out_of_screen_times = out_of_screen_times + 1
    return out_of_screen_times


def get_proportion_of_horizontal_saccades(fixations, locations, saccade_times):
    return (
        get_vertical_saccades(fixations, locations) / saccade_times
        if saccade_times != 0
        else 0
    )


def get_vertical_saccades(fixations, locations):
    pre_row = 0
    vertical_saccade = 0
    for fixation in fixations:
        now_row = get_item_index_x_y(locations, fixation[0], fixation[1])
        if pre_row != now_row and now_row != -1:
            vertical_saccade = vertical_saccade + 1
        pre_row = now_row
    return vertical_saccade


def get_saccade_angle(fixation1, fixation2):
    """获得saccade的角度"""
    vertical_dis = abs(fixation2[1] - fixation1[1])

    if abs(fixation2[0] - fixation1[0]) != 0:
        horizontal_dis = abs(fixation2[0] - fixation1[0])
    else:
        horizontal_dis = 0.001
    return math.atan(vertical_dis / horizontal_dis) * 180 / math.pi


def get_saccade_info(fixations):
    """根据fixation，获取和saccade相关的 saccade_time,mean_saccade_angle"""
    saccade_times = 0
    sum_angle = 0
    for i in range(len(fixations) - 1):
        if (
                get_euclid_distance(
                    fixations[i][0],
                    fixations[i + 1][0],
                    fixations[i][1],
                    fixations[i + 1][1],
                )
                > 500
        ):
            saccade_times = saccade_times + 1
            sum_angle = sum_angle + get_saccade_angle(fixations[i], fixations[i + 1])
    return (
        saccade_times,
        sum_angle / saccade_times if saccade_times != 0 else 0,
    )


def get_reading_times_of_word(fixations, locations):
    """获取区域的reading times"""
    location = json.loads(locations)
    pre_fixation = [-2 for x in range(0, len(location))]
    number_of_fixations = {}
    reading_times = {}  # dict
    reading_durations = {}  # dict.dict
    fixation_cnt = 0
    for fixation in fixations:
        index = get_item_index_x_y(locations, fixation[0], fixation[1])
        if index != -1:
            # 计算reading times
            if fixation_cnt - pre_fixation[index] > 1:
                if index in reading_times.keys():
                    tmp = reading_times[index] + 1
                    reading_times[index] = tmp
                else:
                    reading_times[index] = 1
            pre_fixation[index] = fixation_cnt
            # 计算reading duration
            if index in reading_durations.keys():
                if reading_times[index] in reading_durations[index].keys():
                    tmp = reading_durations[index][reading_times[index]] + fixation[2]
                    reading_durations[index][reading_times[index]] = tmp
                else:
                    reading_durations[index][reading_times[index]] = fixation[2]
            else:
                reading_durations[index] = {reading_times[index]: fixation[2]}
        fixation_cnt = fixation_cnt + 1
    return reading_times, reading_durations


def get_reading_times_and_dwell_time_of_sentence(
        fixations, buttons_location, sentence_dict
):
    pre_fixations = [-2 for x in range(0, len(sentence_dict))]
    fixation_cnt = 0
    reading_times = {}
    first_fixations = [[] for x in range(0, len(sentence_dict))]
    second_fixations = [[] for x in range(0, len(sentence_dict))]
    dwell_time_fixations = [first_fixations, second_fixations]
    number_of_word = 0  # 该句子中单词的数量
    word_list = []
    for fixation in fixations:
        index = get_item_index_x_y(buttons_location, fixation[0], fixation[1])
        sentence = get_sentence_by_word(index, sentence_dict)
        if index != -1:
            if index not in word_list:
                word_list.append(index)
                number_of_word += 1
        if sentence != -1:
            if fixation_cnt - pre_fixations[sentence] > 1:
                # 求reading_times
                if sentence in reading_times.keys():
                    tmp = reading_times[sentence] + 1
                    reading_times[sentence] = tmp
                else:
                    reading_times[sentence] = 1
            # 求dwell_times
            if reading_times[sentence] == 1:
                dwell_time_fixations[0][sentence].append(fixation)
            if reading_times[sentence] == 2:
                dwell_time_fixations[1][sentence].append(fixation)
            pre_fixations[sentence] = fixation_cnt
        fixation_cnt = fixation_cnt + 1
    # 计算dwell_time TODO：此处是按照fixation算的，应该用gaze
    dwell_time = []
    for times in dwell_time_fixations:
        sentence_dwell = []
        for sentence_fixation in times:
            sum_duration = 0
            for fix in sentence_fixation:
                sum_duration = sum_duration + fix[2]
                sentence_dwell.append(sum_duration)
        dwell_time.append(sentence_dwell)
    print("dwell time:%s" % dwell_time)

    return reading_times, dwell_time, number_of_word


def get_sentence_by_word(word_index, sentences):
    """判断单词在哪个句子中"""
    index = 0
    for key in sentences:
        if (
                sentences[key]["end_word_index"]
                > word_index
                >= sentences[key]["begin_word_index"]
        ):
            return index
        index = index + 1
    return -1


def get_saccade(fixations, location):
    """
    获取saccade
    :param fixations: 注视点 [[x,y,t],...]
    :param location: button的坐标
    :return: saccade、前向saccade，后向saccade的次数，saccade的平均长度，saccade的平均角度
    """
    saccade_times = 0  # saccade的次数
    forward_saccade_times = 0  # 前向saccde的次数
    backward_saccade_times = 0  # 后向saccde的次数
    sum_saccade_length = 0  # saccde的总长度，用于后面取平均
    sum_saccade_angle = 0  # saccade的总角度，用于后面取平均
    # pre fixation 设置为第一个点
    pre_word_index = get_item_index_x_y(location, fixations[0][0], fixations[0][1])
    # 获取row
    row = get_row_location(location)
    first_word_index = get_item_index_x_y(location, fixations[0][0], fixations[0][1])
    if first_word_index != -1:
        pre_row = get_row_by_word_index(row, first_word_index)
    else:
        pre_row = 0

    pre_fixation = fixations[0]
    qxy = 0
    for fixation in fixations:
        # 获得当前fixation所在的位置
        word_index = get_item_index_x_y(location, fixation[0], fixation[1])
        now_row = get_row_by_word_index(row, word_index)
        if now_row != pre_row:
            qxy += 1
        if (word_index != pre_word_index and word_index != -1) or (now_row != pre_row):
            # 1. 计算saccade的次数和长度
            # 只要前后fix的单词不一致就是一个fixations
            saccade_times = saccade_times + 1
            sum_saccade_length = sum_saccade_length + abs(word_index - pre_word_index)
            # 判断是前向还是后向
            if word_index > pre_word_index:
                forward_saccade_times = forward_saccade_times + 1
            if word_index < pre_word_index:
                backward_saccade_times = backward_saccade_times + 1

            # 2. 计算saccade angle
            sum_saccade_angle = sum_saccade_angle + get_saccade_angle(
                pre_fixation, fixation
            )
            # 3. 更新pre fixation
            pre_word_index = word_index
            pre_fixation = fixation
            # 4. 更新pre row
            pre_row = now_row
    print("qxy")
    print(qxy)
    return (
        saccade_times,
        forward_saccade_times / saccade_times if saccade_times != 0 else 0,
        backward_saccade_times / saccade_times if saccade_times != 0 else 0,
        sum_saccade_length / saccade_times if saccade_times != 0 else 0,
        sum_saccade_angle / saccade_times if saccade_times != 0 else 0,
    )


def corr(path):
    """
    计算变量之间的相关性
    :param path:
    :return:
    """
    import pandas

    data = pandas.read_csv(path)
    # print(data)

    # print(data[['is_understand','mean_fixations_duration','fixation_duration','second_pass_duration','number_of_fixations','reading_times']])
    print(
        data[
            [
                "is_understand",
                "mean_fixations_duration",
                "fixation_duration",
                "second_pass_duration",
                "number_of_fixations",
                "reading_times",
            ]
        ].corr()
    )


def kmeans_classifier(feature):
    """
    kmeans分类器
    :return:
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2).fit(feature)
    predicted = kmeans.labels_
    print(kmeans.cluster_centers_)
    # 输出的0和1与实际标签并不是对应的，假设我们认为1一定比0多
    is_0 = 0
    for predict in predicted:
        if predict == 0:
            is_0 += 1
    if is_0 / len(predicted) > 0.5:
        # 0的数量多，则标签是相反的
        for i, predict in enumerate(predicted):
            if predict == 1:
                predicted[i] = 0
            else:
                predicted[i] = 1
    return predicted


def evaluate(path, classifier, feature):
    """
    评估分类器的性能
    :return:
    """
    import numpy as np
    import pandas as pd

    reader = pd.read_csv(path)

    is_understand = reader["is_understand"]

    feature = reader[feature]
    feature = np.array(feature)
    feature = feature.reshape(-1, 1)

    # 分类器
    if classifier == "kmeans":
        predicted = kmeans_classifier(feature)
    else:
        # 默认使用kmeans分类器
        predicted = kmeans_classifier(feature)

    # 计算TP等
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(is_understand)):
        if is_understand[i] == 0 and predicted[i] == 0:
            tp += 1
        if is_understand[i] == 0 and predicted[i] == 1:
            fn += 1
        if is_understand[i] == 1 and predicted[i] == 1:
            tn += 1
        if is_understand[i] == 1 and predicted[i] == 0:
            fp += 1
    print("tp:%d fp:%d tn:%d fn:%d" % (tp, fp, tn, fn))
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    from sklearn.metrics import roc_auc_score

    y_true = is_understand
    y_pred = predicted
    auc = roc_auc_score(y_true, y_pred)
    auc = 1 - auc if auc < 0.5 else auc

    print("precision：%f" % precision)
    print("recall：%f" % recall)
    print("f1 score: %f" % f1)
    print("auc:%f" % auc)
    print("accuracy：%f" % accuracy)


def standard_deviation(data_list):
    sum = 0
    for data in data_list:
        sum = sum + data
    mean = sum / len(data_list) if len(data_list) > 0 else 0
    sum = 0
    for data in data_list:
        sum = sum + math.pow(data - mean, 2)
    sum = sum / len(data_list) if len(data_list) > 0 else 0

    return math.sqrt(sum)


def ocr(img_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    # 输出结果保存路径
    result = ocr.ocr(img_path, cls=True)
    for line in result:
        print(line)

    from PIL import Image

    image = Image.open(img_path).convert("RGB")
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.show()


def get_sematic_heat_map():
    import urllib
    from pyheatmap.heatmap import HeatMap

    url = "https://raw.github.com/oldj/pyheatmap/master/examples/test_data.txt"
    sdata = urllib.request.urlopen(url).read().decode().strip().split(",")
    print("s_data")
    print(sdata)
    data = []
    for ln in sdata:
        a = ln.split(",")
        if len(a) != 2:
            continue
        a = [int(i) for i in a]
        data.append(a)

    # start painting
    hm = HeatMap(data)
    hm.clickmap(save_as="hit.png")
    hm.heatmap(save_as="heat.png")

    pass


def meanFilter(data, win):
    length = len(data)
    res = np.zeros(length)
    for i in range(length):
        s, n = 0, 0
        for j in range(i - win // 2, i + win - win // 2):
            if j < 0 or j >= length:
                continue
            else:
                s += data[j]
                n += 1
        res[i] = s / n
    return res


def preprocess_data(data, win):
    data = signal.medfilt(data, kernel_size=win)
    data = signal.medfilt(data, kernel_size=win)
    data = meanFilter(data, 5)
    data = meanFilter(data, 5)
    # data = meanFilter(data, win)
    return data


def get_importance(text):
    """获取单词的重要性"""
    from keybert import KeyBERT

    kw_model = KeyBERT()

    importance = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=100
    )
    return importance


def get_word_and_sentence_from_text(content):
    sentences = content.split("..")
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
                    word = word.strip().lower().replace('"', "")
                    word_list.append(word)
                    cnt += 1
            end = cnt
            sentence_list.append((sentence, begin, end))
            begin = cnt
    return word_list, sentence_list


def topk_tuple(data_list, k=10):
    # 笨拙的top_k的方法 针对tuple [('word',num)]
    top_k = []
    data_list_copy = [x for x in data_list]
    while k > 0:
        max_data = 0
        max_index = 0
        for i, data in enumerate(data_list_copy):
            if data[1] > max_data:
                max_data = data[1]
                max_index = i
        k -= 1
        top_k.append(data_list_copy[max_index])
        del data_list_copy[max_index]
    print(top_k)
    return top_k


if __name__ == "__main__":
    import re

    texts = "It is not controversial to say that an unhealthy diet causes bad health..Nor are the basic elements of healthy eating disputed..Obesity raises susceptibility to cancer, and Britain is the six most obese country on Earth..That is a public health emergency..But naming the problem is the easy part..No one disputes the costs in quality of life and depleted health budgets of an obese population, but the quest for solutions gets diverted by ideological arguments around responsibility and choice..And the water is muddied by lobbying from the industries that profit from consumption of obesity-inducing products..Historical precedent suggests that science and politics can overcome resistance from businesses that pollute and poison but it takes time, and success often starts small..So it is heartening to note that a programme in Leeds has achieved a reduction in childhood obesity, becoming the first UK city to reverse a fattening trend..The best results were among younger children and in more deprived areas..When 28% of English children aged two to 15 are obese, a national shift on the scale achieved by Leeds would lengthen hundreds of thousands of lives..A significant factor in the Leeds experience appears to Many members of parliament are uncomfortable even with their own government's anti-obesity strategy, since it involves a 'sugar tax' and a ban on the sale of energy drinks to under-16s..Bans and taxes can be blunt instruments, but their harshest critics can rarely suggest better methods..These critics just oppose regulation itself.."

    word_list, sentence_list = get_word_and_sentence_from_text(texts)
    print(word_list)
    print(sentence_list)
    print(len(word_list))
    location = '[{"left":330,"top":95,"right":362,"bottom":147},{"left":362,"top":95,"right":395.5,"bottom":147},{"left":395.5,"top":95,"right":450.5,"bottom":147},{"left":450.5,"top":95,"right":614.15625,"bottom":147},{"left":614.15625,"top":95,"right":654.15625,"bottom":147},{"left":654.15625,"top":95,"right":707.234375,"bottom":147},{"left":707.234375,"top":95,"right":769.171875,"bottom":147},{"left":769.171875,"top":95,"right":813.234375,"bottom":147},{"left":813.234375,"top":95,"right":943.28125,"bottom":147},{"left":943.28125,"top":95,"right":1003.578125,"bottom":147},{"left":1003.578125,"top":95,"right":1095.515625,"bottom":147},{"left":1095.515625,"top":95,"right":1155.125,"bottom":147},{"left":1155.125,"top":95,"right":1248.671875,"bottom":147},{"left":1248.671875,"top":95,"right":1308.609375,"bottom":147},{"left":1308.609375,"top":95,"right":1360.328125,"bottom":147},{"left":1360.328125,"top":95,"right":1413.671875,"bottom":147},{"left":1413.671875,"top":95,"right":1487.46875,"bottom":147},{"left":1487.46875,"top":95,"right":1608.046875,"bottom":147},{"left":1608.046875,"top":95,"right":1647.15625,"bottom":147},{"left":1647.15625,"top":95,"right":1747.625,"bottom":147},{"left":330,"top":147,"right":418.359375,"bottom":199},{"left":418.359375,"top":147,"right":540.8125,"bottom":199},{"left":540.8125,"top":147,"right":644.46875,"bottom":199},{"left":644.46875,"top":147,"right":725.125,"bottom":199},{"left":725.125,"top":147,"right":890.578125,"bottom":199},{"left":890.578125,"top":147,"right":930.578125,"bottom":199},{"left":930.578125,"top":147,"right":1025.265625,"bottom":199},{"left":1025.265625,"top":147,"right":1084.671875,"bottom":199},{"left":1084.671875,"top":147,"right":1174.671875,"bottom":199},{"left":1174.671875,"top":147,"right":1208.171875,"bottom":199},{"left":1208.171875,"top":147,"right":1261.515625,"bottom":199},{"left":1261.515625,"top":147,"right":1307.1875,"bottom":199},{"left":1307.1875,"top":147,"right":1380.984375,"bottom":199},{"left":1380.984375,"top":147,"right":1465.921875,"bottom":199},{"left":1465.921875,"top":147,"right":1570.65625,"bottom":199},{"left":1570.65625,"top":147,"right":1616.703125,"bottom":199},{"left":1616.703125,"top":147,"right":1698.578125,"bottom":199},{"left":1698.578125,"top":147,"right":1765.328125,"bottom":199},{"left":330,"top":199,"right":363.5,"bottom":251},{"left":363.5,"top":199,"right":392.765625,"bottom":251},{"left":392.765625,"top":199,"right":479.03125,"bottom":251},{"left":479.03125,"top":199,"right":566.796875,"bottom":251},{"left":566.796875,"top":199,"right":714.015625,"bottom":251},{"left":714.015625,"top":199,"right":768.8125,"bottom":251},{"left":768.8125,"top":199,"right":871.890625,"bottom":251},{"left":871.890625,"top":199,"right":925.234375,"bottom":251},{"left":925.234375,"top":199,"right":1038.46875,"bottom":251},{"left":1038.46875,"top":199,"right":1071.96875,"bottom":251},{"left":1071.96875,"top":199,"right":1125.3125,"bottom":251},{"left":1125.3125,"top":199,"right":1192.015625,"bottom":251},{"left":1192.015625,"top":199,"right":1260.90625,"bottom":251},{"left":1260.90625,"top":199,"right":1311.6875,"bottom":251},{"left":1311.6875,"top":199,"right":1371.359375,"bottom":251},{"left":1371.359375,"top":199,"right":1483.796875,"bottom":251},{"left":1483.796875,"top":199,"right":1537.140625,"bottom":251},{"left":1537.140625,"top":199,"right":1611.59375,"bottom":251},{"left":1611.59375,"top":199,"right":1648.78125,"bottom":251},{"left":1648.78125,"top":199,"right":1742.609375,"bottom":251},{"left":1742.609375,"top":199,"right":1781.71875,"bottom":251},{"left":330,"top":251,"right":380.71875,"bottom":303},{"left":380.71875,"top":251,"right":440.125,"bottom":303},{"left":440.125,"top":251,"right":558.140625,"bottom":303},{"left":558.140625,"top":251,"right":645.90625,"bottom":303},{"left":645.90625,"top":251,"right":756.40625,"bottom":303},{"left":756.40625,"top":251,"right":795.515625,"bottom":303},{"left":795.515625,"top":251,"right":839.578125,"bottom":303},{"left":839.578125,"top":251,"right":924.515625,"bottom":303},{"left":924.515625,"top":251,"right":1072.03125,"bottom":303},{"left":1072.03125,"top":251,"right":1127.09375,"bottom":303},{"left":1127.09375,"top":251,"right":1180.4375,"bottom":303},{"left":1180.4375,"top":251,"right":1260.25,"bottom":303},{"left":1260.25,"top":251,"right":1309,"bottom":303},{"left":1309,"top":251,"right":1429.03125,"bottom":303},{"left":1429.03125,"top":251,"right":1494.0625,"bottom":303},{"left":1494.0625,"top":251,"right":1605.5,"bottom":303},{"left":1605.5,"top":251,"right":1649.53125,"bottom":303},{"left":330,"top":303,"right":471.6875,"bottom":355},{"left":471.6875,"top":303,"right":610.875,"bottom":355},{"left":610.875,"top":303,"right":709.15625,"bottom":355},{"left":709.15625,"top":303,"right":877.71875,"bottom":355},{"left":877.71875,"top":303,"right":937.125,"bottom":355},{"left":937.125,"top":303,"right":1033.015625,"bottom":355},{"left":1033.015625,"top":303,"right":1096.046875,"bottom":355},{"left":1096.046875,"top":303,"right":1149.390625,"bottom":355},{"left":1149.390625,"top":303,"right":1229.140625,"bottom":355},{"left":1229.140625,"top":303,"right":1262.640625,"bottom":355},{"left":1262.640625,"top":303,"right":1381.984375,"bottom":355},{"left":1381.984375,"top":303,"right":1426.015625,"bottom":355},{"left":1426.015625,"top":303,"right":1543.546875,"bottom":355},{"left":1543.546875,"top":303,"right":1614.4375,"bottom":355},{"left":1614.4375,"top":303,"right":1667.78125,"bottom":355},{"left":330,"top":355,"right":457.65625,"bottom":407},{"left":457.65625,"top":355,"right":519.59375,"bottom":407},{"left":519.59375,"top":355,"right":598.1875,"bottom":407},{"left":598.1875,"top":355,"right":669.078125,"bottom":407},{"left":669.078125,"top":355,"right":836.25,"bottom":407},{"left":836.25,"top":355,"right":875.359375,"bottom":407},{"left":875.359375,"top":355,"right":1084.96875,"bottom":407},{"left":1084.96875,"top":355,"right":1208.390625,"bottom":407},{"left":330,"top":422,"right":453.296875,"bottom":474},{"left":453.296875,"top":422,"right":585.421875,"bottom":474},{"left":585.421875,"top":422,"right":702.8125,"bottom":474},{"left":702.8125,"top":422,"right":764.75,"bottom":474},{"left":764.75,"top":422,"right":864.34375,"bottom":474},{"left":864.34375,"top":422,"right":923.75,"bottom":474},{"left":923.75,"top":422,"right":1021.59375,"bottom":474},{"left":1021.59375,"top":422,"right":1077.6875,"bottom":474},{"left":1077.6875,"top":422,"right":1207.234375,"bottom":474},{"left":1207.234375,"top":422,"right":1336.9375,"bottom":474},{"left":1336.9375,"top":422,"right":1407.828125,"bottom":474},{"left":1407.828125,"top":422,"right":1546.796875,"bottom":474},{"left":1546.796875,"top":422,"right":1608.734375,"bottom":474},{"left":1608.734375,"top":422,"right":1705.25,"bottom":474},{"left":1705.25,"top":422,"right":1764.65625,"bottom":474},{"left":330,"top":474,"right":424.140625,"bottom":526},{"left":424.140625,"top":474,"right":479.203125,"bottom":526},{"left":479.203125,"top":474,"right":510.53125,"bottom":526},{"left":510.53125,"top":474,"right":586.015625,"bottom":526},{"left":586.015625,"top":474,"right":659.234375,"bottom":526},{"left":659.234375,"top":474,"right":718.640625,"bottom":526},{"left":718.640625,"top":474,"right":820.453125,"bottom":526},{"left":820.453125,"top":474,"right":897.1875,"bottom":526},{"left":897.1875,"top":474,"right":976.46875,"bottom":526},{"left":976.46875,"top":474,"right":1057.890625,"bottom":526},{"left":1057.890625,"top":474,"right":1103,"bottom":526},{"left":1103,"top":474,"right":1134.328125,"bottom":526},{"left":1134.328125,"top":474,"right":1167.828125,"bottom":526},{"left":1167.828125,"top":474,"right":1309.078125,"bottom":526},{"left":1309.078125,"top":474,"right":1349.078125,"bottom":526},{"left":1349.078125,"top":474,"right":1417.484375,"bottom":526},{"left":1417.484375,"top":474,"right":1479.421875,"bottom":526},{"left":1479.421875,"top":474,"right":1508.6875,"bottom":526},{"left":1508.6875,"top":474,"right":1660.484375,"bottom":526},{"left":1660.484375,"top":474,"right":1697.671875,"bottom":526},{"left":1697.671875,"top":474,"right":1779.6875,"bottom":526},{"left":330,"top":526,"right":385.15625,"bottom":578},{"left":385.15625,"top":526,"right":502.65625,"bottom":578},{"left":502.65625,"top":526,"right":531.921875,"bottom":578},{"left":531.921875,"top":526,"right":657.921875,"bottom":578},{"left":657.921875,"top":526,"right":695.109375,"bottom":578},{"left":695.109375,"top":526,"right":826.703125,"bottom":578},{"left":826.703125,"top":526,"right":930.546875,"bottom":578},{"left":930.546875,"top":526,"right":1061.8125,"bottom":578},{"left":1061.8125,"top":526,"right":1115.15625,"bottom":578},{"left":1115.15625,"top":526,"right":1175.265625,"bottom":578},{"left":1175.265625,"top":526,"right":1224.421875,"bottom":578},{"left":1224.421875,"top":526,"right":1280.5,"bottom":578},{"left":1280.5,"top":526,"right":1320.5,"bottom":578},{"left":1320.5,"top":526,"right":1419.078125,"bottom":578},{"left":1419.078125,"top":526,"right":1448.34375,"bottom":578},{"left":1448.34375,"top":526,"right":1568.546875,"bottom":578},{"left":1568.546875,"top":526,"right":1651.859375,"bottom":578},{"left":1651.859375,"top":526,"right":1710.015625,"bottom":578},{"left":1710.015625,"top":526,"right":1775.015625,"bottom":578},{"left":330,"top":578,"right":420.78125,"bottom":630},{"left":420.78125,"top":578,"right":491.671875,"bottom":630},{"left":491.671875,"top":578,"right":588.828125,"bottom":630},{"left":588.828125,"top":578,"right":700.375,"bottom":630},{"left":700.375,"top":578,"right":808.546875,"bottom":630},{"left":808.546875,"top":578,"right":867.953125,"bottom":630},{"left":867.953125,"top":578,"right":905.140625,"bottom":630},{"left":905.140625,"top":578,"right":981.328125,"bottom":630},{"left":981.328125,"top":578,"right":1098.59375,"bottom":630},{"left":1098.59375,"top":578,"right":1180.453125,"bottom":630},{"left":1180.453125,"top":578,"right":1264.0625,"bottom":630},{"left":1264.0625,"top":578,"right":1329.5625,"bottom":630},{"left":1329.5625,"top":578,"right":1368.671875,"bottom":630},{"left":1368.671875,"top":578,"right":1466.671875,"bottom":630},{"left":1466.671875,"top":578,"right":1574.84375,"bottom":630},{"left":1574.84375,"top":578,"right":1648.4375,"bottom":630},{"left":1648.4375,"top":578,"right":1707.515625,"bottom":630},{"left":1707.515625,"top":578,"right":1747.515625,"bottom":630},{"left":330,"top":630,"right":374.15625,"bottom":682},{"left":374.15625,"top":630,"right":425.875,"bottom":682},{"left":425.875,"top":630,"right":516.59375,"bottom":682},{"left":516.59375,"top":630,"right":545.859375,"bottom":682},{"left":545.859375,"top":630,"right":654.953125,"bottom":682},{"left":654.953125,"top":630,"right":720.96875,"bottom":682},{"left":720.96875,"top":630,"right":767.015625,"bottom":682},{"left":767.015625,"top":630,"right":820.359375,"bottom":682},{"left":820.359375,"top":630,"right":892.78125,"bottom":682},{"left":892.78125,"top":630,"right":1010.28125,"bottom":682},{"left":1010.28125,"top":630,"right":1054.3125,"bottom":682},{"left":1054.3125,"top":630,"right":1136.328125,"bottom":682},{"left":1136.328125,"top":630,"right":1223,"bottom":682},{"left":1223,"top":630,"right":1341.28125,"bottom":682},{"left":1341.28125,"top":630,"right":1465.890625,"bottom":682},{"left":1465.890625,"top":630,"right":1505,"bottom":682},{"left":1505,"top":630,"right":1640.390625,"bottom":682},{"left":1640.390625,"top":630,"right":1679.5,"bottom":682},{"left":1679.5,"top":630,"right":1751.234375,"bottom":682},{"left":1751.234375,"top":630,"right":1784.125,"bottom":682},{"left":330,"top":697,"right":463.765625,"bottom":749},{"left":463.765625,"top":697,"right":546.5625,"bottom":749},{"left":546.5625,"top":697,"right":583.75,"bottom":749},{"left":583.75,"top":697,"right":637.09375,"bottom":749},{"left":637.09375,"top":697,"right":719.109375,"bottom":749},{"left":719.109375,"top":697,"right":859.453125,"bottom":749},{"left":859.453125,"top":697,"right":966.71875,"bottom":749},{"left":966.71875,"top":697,"right":1006.71875,"bottom":749},{"left":330,"top":749,"right":410.21875,"bottom":801},{"left":410.21875,"top":749,"right":534.21875,"bottom":801},{"left":534.21875,"top":749,"right":573.328125,"bottom":801},{"left":573.328125,"top":749,"right":712.625,"bottom":801},{"left":712.625,"top":749,"right":764.34375,"bottom":801},{"left":764.34375,"top":749,"right":950.734375,"bottom":801},{"left":950.734375,"top":749,"right":1021.203125,"bottom":801},{"left":1021.203125,"top":749,"right":1086.265625,"bottom":801},{"left":1086.265625,"top":749,"right":1155.15625,"bottom":801},{"left":1155.15625,"top":749,"right":1220.15625,"bottom":801},{"left":1220.15625,"top":749,"right":1393,"bottom":801},{"left":1393,"top":749,"right":1546.109375,"bottom":801},{"left":1546.109375,"top":749,"right":1659.5,"bottom":801},{"left":1659.5,"top":749,"right":1733.4375,"bottom":801},{"left":1733.4375,"top":749,"right":1764.765625,"bottom":801},{"left":330,"top":801,"right":438.4375,"bottom":853},{"left":438.4375,"top":801,"right":467.703125,"bottom":853},{"left":467.703125,"top":801,"right":557.015625,"bottom":853},{"left":557.015625,"top":801,"right":617.84375,"bottom":853},{"left":617.84375,"top":801,"right":677.25,"bottom":853},{"left":677.25,"top":801,"right":706.515625,"bottom":853},{"left":706.515625,"top":801,"right":765.5625,"bottom":853},{"left":765.5625,"top":801,"right":811.609375,"bottom":853},{"left":811.609375,"top":801,"right":864.953125,"bottom":853},{"left":864.953125,"top":801,"right":925.34375,"bottom":853},{"left":925.34375,"top":801,"right":964.453125,"bottom":853},{"left":964.453125,"top":801,"right":1059.359375,"bottom":853},{"left":1059.359375,"top":801,"right":1145.234375,"bottom":853},{"left":1145.234375,"top":801,"right":1185.234375,"bottom":853},{"left":1185.234375,"top":801,"right":1322.734375,"bottom":853},{"left":1322.734375,"top":801,"right":1392.96875,"bottom":853},{"left":1392.96875,"top":801,"right":1452.375,"bottom":853},{"left":1452.375,"top":801,"right":1527.28125,"bottom":853},{"left":1527.28125,"top":801,"right":1583.375,"bottom":853},{"left":1583.375,"top":801,"right":1628.328125,"bottom":853},{"left":1628.328125,"top":801,"right":1704.5625,"bottom":853},{"left":330,"top":853,"right":487.90625,"bottom":905},{"left":487.90625,"top":853,"right":542.96875,"bottom":905},{"left":542.96875,"top":853,"right":611.859375,"bottom":905},{"left":611.859375,"top":853,"right":724.796875,"bottom":905},{"left":724.796875,"top":853,"right":806.859375,"bottom":905},{"left":806.859375,"top":853,"right":862.953125,"bottom":905},{"left":862.953125,"top":853,"right":942.921875,"bottom":905},{"left":942.921875,"top":853,"right":1049.203125,"bottom":905},{"left":1049.203125,"top":853,"right":1134.625,"bottom":905},{"left":1134.625,"top":853,"right":1257.953125,"bottom":905},{"left":1257.953125,"top":853,"right":1340.84375,"bottom":905},{"left":1340.84375,"top":853,"right":1422.90625,"bottom":905},{"left":1422.90625,"top":853,"right":1480.15625,"bottom":905},{"left":1480.15625,"top":853,"right":1582.0625,"bottom":905},{"left":1582.0625,"top":853,"right":1715.671875,"bottom":905},{"left":330,"top":905,"right":404.921875,"bottom":957}]'
    location = json.loads(location)
    print("length of location:%d" % len(location))

    pass
