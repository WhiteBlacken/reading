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
    import time

    filename = str(page_data_id) + ".png"
    print("filename:%s" % filename)
    # 存储地址
    print("session.username:%s" % username)
    path = "static/data/" + str(username) + "/"
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
    text = content.replace(",", "").replace(".", "").strip()
    contents = text.split(" ")
    index_2_word = {}
    cnt = 0
    for item in contents:
        if len(item) > 0:
            index_2_word[cnt] = item.strip().lower()
            cnt = cnt + 1
    return index_2_word


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
    data = meanFilter(data, win)
    data = meanFilter(data, win)
    data = meanFilter(data, win)
    return data


if __name__ == "__main__":
    location = (
        '[{"left":330,"top":95,"right":408.15625,"bottom":326.984375},{"left":408.15625,"top":95,'
        '"right":445.5,"bottom":326.984375}] '
    )

    data_x = [500.00482294599493,497.7410671077958,489.1817925202071,472.2135058006135,437.71755208923724,389.0151587732582,330.0664146565909,274.29258617679517,238.07254478720944,224.55918086037616,209.71899414564862,192.08841562481334,189.0245549219029,184.0024996571721,182.44631560296455,218.8139982511963,203.67133840432575,230.0974843624001,232.72217236099746,215.39329957112713,229.8655333467839,242.93022651515827,252.08872738895272,262.58040246444716,260.20534980614605,260.5365997321989,262.9930523094231,269.8956371469332,272.1917141884806,286.17733334143253,297.80528269637415,305.6603942250675,318.2935384500643,322.1202449221305,330.38141295699734,342.46096251412337,351.9948907102214,324.986402479509,301.94835003468296,309.5689761282732,282.70270374451616,288.3953048877491,292.3834788600465,325.90223017226253,338.8724349719994,330.90105238520897,358.6246956719383,354.88416343873206,345.1802376209365,362.77868992859817,391.53812549137126,395.5133560132687,395.5570471357961,399.1446430200257,396.9814887536509,396.78956274945807,396.0617104326679,403.08056470211915,380.6508094661336,390.4462215538108,416.56735931259567,403.79412865891453,411.7186108184453,390.1699675052009,381.4257150820969,391.5772997784395,412.1742869787501,432.5513099333351,461.81644950601685,479.2403094989221,500.33426570760776,501.11781136724323,514.556577602103,505.18321937323674,521.3842260669668,534.2809706653632,542.1333143862108,536.996257419594,536.5516917462895,552.3815365813853,525.6422071085151,487.9377938938245,487.2944319287807,484.5472202460993,470.79891445321215,490.0898712993016,526.9145843150687,527.9851703799759,534.3995194254481,554.5767965416624,554.7450158649277,554.6497346868434,552.8726021396303,555.261959305144,562.0915722762112,536.8647695979689,514.9562881417787,533.0076435507461,542.4703723018081,544.7895613043881,552.9104322405474,557.3068999973278,538.2698567817312,533.0412252027469,544.629512400883,541.2782698202177,579.4481949838366,571.0613419171507,587.8840756222945,591.3058215100348,595.4141709396189,610.0896324086829,594.2476785643024,569.8379401228715,575.166277135479,579.8068558486209,580.3533156571638,589.6414488971393,621.2811479817698,647.9579512668355,657.3506215328694,662.7007265056295,654.1474126828817,651.6428375126546,663.2120551219913,666.2164948810757,686.4297986967048,677.8745688076314,727.5818570454966,723.4480323707887,714.4072623097751,698.8507428096786,686.968608232286,696.6462939085643,717.8426125741023,759.1534750510007,749.195203729039,742.4158012051835,768.7864713052068,780.3947324497872,755.54314661547,717.4438787465159,701.8958269591507,725.3347872281508,763.0308558587974,786.4634282645166,781.4254450063612,799.5163975923222,807.918098471413,792.4404354789117,815.022139431331,820.627709779545,836.098706896558,830.6704218522019,849.9942412018011,835.8903485668055,812.5162104312897,812.5228799762972,801.9983890382196,801.6368418092458,794.1352399631809,814.8360720751797,832.6619739995119,864.40938565196,864.0046366715173,866.6803361829237,870.0248898074519,892.8524227193807,905.5939230343098,902.4060060848245,898.6888293789227,876.4073861551014,869.1511896557607,864.4073002525153,860.3736372769495,862.7821648801216,865.4887360506707,880.9502110264418,927.35010519856,930.4279496421561,931.2639683143472,936.5743073459887,938.48141030117,954.0117432878662,969.1642742448067,973.364614194062,973.8817525312093,953.8064461721865,949.0672452352231,960.4370423876238,946.8915444845842,955.7091872050552,963.1421102409939,966.9026023495162,982.6262304289571,974.2146105918949,977.5221092114168,997.2964675904243,998.0824121426419,1004.1638641218548,992.578637570946,977.7301586139419,984.7277952679599,999.5961181837863,1034.3814603575124,1047.9108452340572,1069.0059559158176,1070.001532126091,1069.6663811387143,1060.5102789406596,1072.5439739293047,1124.5989239049,1082.573751227357,1063.7026359538709,1060.9984504208967,1068.2499962420645,1074.6084707879606,1119.0924824479741,1159.0020813049,1156.6208547253686,1181.3608456302672,1206.9151038260604,1216.9643334525174,1218.071979857847,1206.1662051600736,1191.7958471791483,1191.443118110744,1177.4843238089404,1198.1159573285368,1177.0095575852001,1173.2299694575186,1172.3104241476044,1198.236076952733,1183.658101116465,1178.6999822526816,1155.2716284210887,1157.8570094669485,1139.2453175327744,1149.360771200399,1164.2766564091432,1176.9976312819726,1195.6678751497514,1193.2684229830168,1199.1085471936003,1191.130417255067,1192.302555318576,1196.2173735431413,1234.6698359923116,1270.2546665421514,1289.1778810994795,1292.4180408372906,1295.0321311674445,1294.399581374515,1286.7080438197527,1276.6007620887258,1330.9310453715127,1326.4604649949938,1327.4175493599864,1335.349179116163,1341.230685683164,1341.7013669257362,1354.616078370503,1398.3802690324978,1408.8464479796203,1407.9267574471082,1421.9716536759938,1430.120820446085,1435.8152801517851,1435.022454871992,1440.052133208996,1408.0735747552037,1376.3586402809667,1370.9278990358223,1346.2689210324413,1366.0207038227181,1378.601793326298,1425.7283417830245,1463.748436794855,1451.607045711449,1467.4848017239717,1443.9810287161188,1441.5658857776775,1452.6567988410477,1443.6713349070117,1453.0802395144187,1470.7616750779498,1464.99552343411,1459.4704395302408,1499.882709776163,1542.0769466065483,1533.582986570707,1558.6340991613122,1587.4447443588235,1556.9060945000797,1541.3302994795322,1509.92827047647,1518.0779389026657,1517.7936583800793,1544.2087914804058,1555.1190070857815,1580.6725389327246,1599.9780240851208,1620.1713310695961,1617.9012374967472,1615.206957773032,1600.6078717403568,1592.0594752834006,1576.7291988106258,1595.6326852748507,1615.4075531720896,1619.1080954631873,1616.1810351961035,1630.414903474981,1631.8886137530762,1639.3499517053126,1634.1691453201188,1437.604459859533,1270.1088410565164,1337.9998937863309,1287.4928585834323,1331.980046918871,1374.329047082675,1364.0736506009505,1271.6365120745218,1283.3576372400164,1219.5756932853872,1272.121513617536,1240.3260865882066,1134.9915753976404,1200.649982303357,1182.5897604256406,1139.3804800862656,1119.0700528129237,1055.0901449155406,985.925155768019,907.0775219108061,843.750637608852,733.1481720941658,709.6900805474461,603.1015367400955,534.735916476032,467.11954277894085,446.64412894559325,492.02979591595914,453.74494349466624,388.70582810554566,319.2250230083227,176.78558824653032,86.62266704458378,7.7020030600675735,-48.876937644830406,-50.910497145136276,-45.24174874674419,-12.649691825589656,20.315759744386014,97.79075377354113,158.55936871155524,127.49597847438808,116.11655032954886,139.29294914454508,115.2911168618499,56.20835108599471,74.7483009812134,142.40713002229427,162.30981180220107,173.84148339879616,222.7382139535019,267.973825631383,218.8301075255551,245.69115862625992,261.35332994617517,183.1615316583661,164.0821010273672,242.83558874132078,263.1867741123359,285.8727122050061,318.18621066386453,328.1067200794763,353.45425282510115,362.42541516654165,320.64447978531774,367.2239559716543,389.10944619476913,414.1844784291551,417.99235024476536,392.0147293193811,380.6977742568283,417.8487630326184,435.8026525435291,468.8212253079285,480.74390327622626,489.2788181785791,457.5474877269616,433.03149407957767,434.5145713162375,411.2485335579817,385.5296734411515,380.91755157390526,387.31859030625947,362.75721647787304,357.4755986878761,369.1974975853848,343.78631635949796,321.8315746667351,347.4116398117509,323.8785201229694,346.9910382889355,367.4235807324382,371.9814371176781,375.74345015058066,373.27255672024955,340.33996488627974,324.50958070652325,327.7213733627196,316.4377476503896,311.52458739585296,346.9530483491632,343.38246320152155,355.1740452778252,417.00229584459777,424.32943615557764,413.3641670940819,410.2047114389558,447.0205652474872,477.68764738543126,486.1110511635745,483.13358745140164,481.5417852003116,479.5606825463342,454.941478627038,447.37396007988895,431.01826340424014,405.5242012171772,405.5276083309371,405.88161847795953,420.83576076514976,433.9830801709238,428.2631794281202,536.3426941033417,712.7471457133342,795.9583434616835,836.2376917626003,840.0866770346418,771.729969024683,776.1885522599098,820.8187624645805,917.9760271509607,952.3296625389429,999.3230063780776,1036.9220364795933,1020.269264832827,1109.8109704132044,1198.6673347656765,1275.4446332738355,1376.9501655440595,1428.2848153314505,1454.9504510411314,1471.2304884393834,1475.5337121375353,1445.405341602004,1457.658965948305,1506.8753055016457,1487.0459048252978,1523.571282276725,1455.482995396721,1400.0466884909101,1402.4811931252902,1404.457618510135,1401.5681742009347,1389.2926106941084,1369.6310667458797,1330.2989602595692,1348.88567734251,1357.6297967489224,1356.4852921000884,1380.1994607233653,1399.6246869696254,1417.0488942342581,1409.6935555310229,1429.9889366295818,1439.991159886553,1417.4119508291585,1420.991812075878,1437.6254553168096,1396.2418933887718,1427.791293378027,1423.9730898629864,1420.4022459395774,1426.0542446898028,1435.4511244339858,1435.9143034621188,1392.2739019643548,1355.4156223832922,1312.541614196203,1312.4749478485267,1347.53285574527,1358.0285288987425,1390.5518672197832,1391.5466156853859,1464.8607694707014,1494.4647581138283,1472.8862660818318,1497.0302231984367,1521.4120029475328,1526.9117586563266,1507.515906882552,1529.3914439135679,1534.5485202895115,1574.4605851048805,1581.9689791152628,1580.2568973016564,1589.1354579991742,1588.1583888327307,1585.90077648632,1580.6538046159592,1575.4066610722239,1552.5699532186131,1543.7472772098256,1506.194004739226,1486.2364599838627,1482.6290337040011,1461.9923894157787,1486.4610920718776,1503.1569430810482,1533.336317329715,1529.5955249897218,1560.6050546154854,1591.4355926690519,1600.4256271615498,1633.6230614535245,1636.6441158705265,1652.3562691461502,1642.9164126200067,1641.6067875095948,1654.6466470124362,1646.0378288865015,1645.3683379581587,1655.1385846325404,1655.5727315296228,1672.327905509523,1695.716220689218,1717.7361960620028,1705.8551244621726,1678.8096367319445,1559.4359984456742,1492.765458330194,1231.6456562261403,983.7993946621698,789.4676284240699,621.0424272941503,534.8019332916558,492.0295007916303,414.81079074588735,328.3760263753486,215.93032797281288,162.34291684363365,153.59260041583894,126.99053844122966,97.17245608075397,74.99846766875982,63.34441652458907,71.69653519654682,83.708284710751,131.85131753332263,177.60212517368012,182.5052128482419,167.25174925342577,201.85445339187794,218.75086394872858,238.95062868695967,256.1246327997934,258.08299284859646,324.86214990364454,361.9986865007101,391.2066431901271,410.40564196979284,418.80569593967493,425.0949358833617,422.9698330704713,416.5018345677363,431.51198632638744,450.4564988798685,481.30151736751316,482.86853633794715,482.5245716910513,490.72901612125844,457.22084609608106,458.7048224917263,494.606014042728,507.982661888616,511.3472255439879,520.7733673426736,521.3171590399952,520.5723979893193,534.8315303364509,561.9804996510208,546.5554910094115,530.721646305406,527.015132940014,535.0410446139587,541.9002279091255,513.1179359559683,518.617699770863,545.6929621883817,567.1754828470044,590.5064227990587,572.971659832047,578.5164067289321,588.6884547691341,591.7926191906121,608.6939027443595,628.2100510180161,641.6416077591686,606.2454444420682,587.1096594517462,587.2711008749604,600.0216805423746,591.7946360881755,598.4907370712853,604.1678735902576,612.2701761262443,602.7275421371372,585.1456765110183,563.0321331508933,596.006031743439,631.1334691834301,641.5385050463888,627.9908615843481,636.4066515013841,645.2523668232972,652.2881546085569,623.1609775591143,654.8627148387067,681.8000694157936,675.3624812384022,708.4911451922466,738.5987207528062,722.7558394229324,707.8192093575653,739.1605781249061,713.4037399870346,681.5893879528714,699.3936854988856,709.2060977116548,693.052811508605,705.1972576130898,730.9843876429057,726.2836515956367,721.6922493984185,702.9685826739136,671.0650575328093,628.8410897970897,622.080874815551,635.0980531277255,631.878339224789,668.6379793093481,672.320163693872,659.4569526652906,685.0779138033619,709.0960425320698,682.353144861749,659.7344947646781,622.3317132380495,586.6334001368205,608.4601676705295,638.0447259344653,618.4262719459124,622.0572329357024,631.103794179619,618.1879315374125,630.892527083322,642.0201165270385,634.0067490299745,612.1415517922004,628.1237085536056,650.7348110626874,722.8834055451953,702.2232556476947,694.1492839745791,687.1446173829515,676.0913788316234,682.4898871461652,687.2257333790008,705.6472510699318,697.188106576718,698.0663980920924,712.6076149580138,719.4319304420388,751.9500073087061,724.3497892463566,731.0226381667543,731.332025406891,733.5064841348641,740.7308381234384,757.2418057552943,779.7473294040173,796.9267069246548,782.0753394553897,766.2794756718141,768.9395954485989,757.8195627869549,717.4493960522127,683.0695347724461,678.2918506934134,649.6983521533695,645.7884126363276,659.0018928405789,676.9482974806425,676.5248967238101,698.0097763693058,731.2624831246176,737.4629108600184,765.3950969604416,794.2559090101064,817.7724027194181,844.0013596211254,852.7131976704719,867.0005920858351,847.4103983692203,849.8028087976728,846.8583155618442,864.3599418054723,854.9399137003862,838.4160077074206,850.7009204628761,860.6304046891504,868.6494937819903,871.5379221824679,870.2705358709577,881.2132403993796,911.5377326287445,922.6245565927178,931.0623449879824,958.4098463495806,966.7729666479526,947.6958894324658,930.2153811047807,938.6852206232221,952.6118678865873,958.7841280669672,961.3349550453329,952.4187635371741,937.9924464602649,931.158130173171,925.3136029783927,926.1894305223415,935.4812024130417,944.8196300466834,862.9397500506526,883.6962942920995,886.5943844385489,878.224910906081,908.2824072214662,971.8520772663762,996.5707349920782,1014.407664232439,994.9945456827031,1013.9593740923176,1029.190846186143,1029.5464432658298,1017.4233451986684,1046.9473690873842,1078.7300295198688,1069.10712380633,1059.5016464612456,1071.751301644668,1081.438886782028,1083.0930308578038,1109.5477082866619,1088.6148393453268,1090.142491594016,1087.2041665260363,1060.900834264076,1069.5636914783665,1077.0466399526256,1077.0804845061452,1100.4991291151848,1120.7890606552576,1114.7506021226739,1106.7422103682813,1080.5293085053784,1070.9924342712272,1057.2823653383534,1063.7181611693638,1046.637778727927,1057.0194636165245,1052.5159433000058,1061.7970949055311,1068.4343372788414,1086.7520249492509,1112.2522184336956,1121.8271781847495,1154.3854061121333,1145.4437127351334,1136.6448823497133,1133.5347793053495,1186.3153739959391,1220.531448677863,1235.2610471142202,1226.75741615474,1268.6481623055415,1268.711230636312,1237.4106513706279,1213.7415461082321,1219.041700929577,1214.1492856321763,1224.6175377828024,1196.9719837253938,1173.4163795639786,1182.1199528303496,1208.6690917852948,1210.6341773247666,1219.4674207275793,1254.0220948829667,1281.0052021451006,1299.9650217000858,1323.0286593860178,1294.7357522023513,1274.416125512275,1259.225420765452,1276.3405640139038,1287.2009496284418,1304.4977321165736,1312.7341174643595,1326.1297744678745,1328.4317087769862,1327.2036523138586,1315.733128902533,1342.2763558763056,1382.837845095214,1370.7924089681915,1412.9063049343104,1430.532439945149,1441.1604880873751,1408.8614137652937,1391.1629021599986,1391.622081888154,1391.6986274842243,1353.8010411568757,1352.235305186689,1337.737694946949,1323.4607198926635,1326.04454431145,1341.5436825162117,1345.3574949796885,1356.946248550187,1359.340779005102,1361.2932575791724,1399.2219751063908,1427.6726342442719,1456.865588262381,1458.0550216472866,1482.487084021063,1500.9644824078687,1480.6538712412512,1475.6998790738035,1479.407375068498,1487.0458387523508,1520.5848349161415,1500.2167023997865,1502.6608837299277,1534.1810082133231,1524.5206447337614,1529.4006852336813,1534.9228882001096,1542.3260181597648,1530.92494543989,1540.6820840836733,1554.5915952340608,1562.5048264246693,1525.3798822057472,1525.837012013999,1535.1589245779533,1568.7540377440803,1610.520844585995,1647.6940270625485,1639.791662793687,1638.9400858603851,1617.780935395521,1616.4780492517184,1626.2997816894094,1603.8128888694164,1597.737208504801,1614.7161933092586,1593.0467221731049,1582.7270584730802,1575.920761926267,1568.3295830279844,1574.79942482148,1571.3110992754134,1577.764957095458,1589.287798447181,1579.8424404475359,1580.8864886853698,1599.133892996731,1609.4661955821823,1619.55770795111,1604.7070045597761,1634.2363187048916,1640.576112779602,1658.3993710573209,1689.3782800349395,1736.3115605516425,1774.951770375867,1758.827226378493,1750.7389515003972,1737.5303339259872,1746.5242502201925,1758.5761383110553,1749.9070011222789,1721.9786120389313,1711.2052854811686,1716.8936311837554,1711.1196492345575,1715.1166964376926,1724.1141066887185,1702.2075094091858,1560.176902141189,1236.0737671909167,980.4590202886852,1003.0314967199138,934.6818031858551,895.8916996148173,813.2133098009729,781.2972256142598,808.6853181594138,747.5941539158448,669.6264229321894,661.2064927062388,672.7922626004967,613.1346568408047,541.5976592966477,449.02700187418264,506.2911753784123,471.30819355423654,377.99888302471913,366.97205925144686,343.3486555242755,305.0264977381509,285.85400493493785,264.12193056288834,315.2722862971337,280.27505661724706,263.0089947600233,300.32236036242,324.5302029052864,292.62704085678365,284.1116330960702,257.42155126991906,280.9337163450383,317.29538587874094,332.8160206614489,387.19596951055405,435.0729201611867,466.84491390052744,492.65406548517615,512.5939214054015,517.2030185020415,495.4174023339705,505.3573593957537,503.14055127865987,486.68389040708064,478.75992153548265,429.5767168652567,418.5625607199459,429.62624977478316,450.66436567119746,442.47024163615566,442.51751563246074,467.25309247226113,470.2991524496404,456.04735489974604,473.1568129644507,463.5659872692829,447.7473978578582,432.3444818714432,482.4896423086918,505.8858519505637,509.03670086507725,478.1700287764389,506.3983996921359,537.6029705350404,523.505295131561,563.6316367307279,584.5520083704788,556.8926808764651,560.2646840202275,563.7352206936595,562.6280434835555,565.2599771510413,573.0143026068599,577.8262493509893,594.2102649291807,562.8504902722281,549.5967942740062,576.1929273327577,571.7319781359474,549.0343233121777,546.291683508877,564.0672488027817,588.3059747788437,626.863578274783,635.584455113802,637.9945205845545,650.4973242806199,670.6527601486453,689.1309164343701,688.9829118339674,658.7368805498926,668.8027033509377,676.1304394852704,694.3754460894171,698.3042909121273,694.2929324134802,689.8691186321363,698.101093199717,704.6897678884037,726.2928737992368,755.1251751150293,773.3277122261588,770.3511610734894,766.5147710559935,771.3560592932365,776.9942478552335,805.8624814968707,845.0797075471137,856.8648522731838,865.25380316557,854.3141859171673,839.3426191877659,843.0769064274666,831.4597557654163,839.0216283747661,836.7434874058133,809.533984095211,829.5421623377922,846.4877186183456,875.1176628242239,866.8317036451139,883.0164067731066,898.5486636236469,910.9848039621216,911.2201263069791,901.7969631510346,898.9654503934379,925.3178201264689,897.5689830788111,929.4473291133013,910.2695100083571,924.1304793931608,935.6267591850616,950.4178717922357,982.0207792779545,1007.0789681899347,1004.6921246468548,1017.203267160688,987.4106878333472,1012.1795320663629,962.313278123546,953.3515043630415,953.177741781429,951.9715802249264,936.5048144535535,934.8142769517822,935.6434433512652,933.866334504855,930.0612748659381,942.8343590127308,955.8469023532682,981.0990788112022,1002.1318673531465,983.3198379981523,1015.6904509572734,1007.7400351385837,986.749211113637,1015.1387220596715,1043.5539259353766,1063.1604299943551,1056.3095573644873,1049.297441158404,1052.6515278080196,1074.4936236844671,1092.462264236286,1091.4172186663989,1065.9244081394345,1065.3662328859523,1064.5829325812533,1066.0200112411012,1017.835968499968,869.1116038297349,744.3826213605628,700.8357426157748,648.3812740781292,614.5503938743451,585.9904242745882,571.9476862269921,555.878265330824,546.2785571851266,578.4774381681596,573.8328456776471,593.0924703319184,610.422262590578,603.2351929110176,571.3030161402385,545.0212640470126,567.3992938770047,525.6562715417064,496.8449484732974,499.74751194220596,556.0448984035664,662.1291151574698,769.8448902530489,816.3913322711101,852.015579070325,815.4928307870819,820.4206294935591,789.2413590152992,782.8714225647219,736.1479526608185,712.476819092647,718.499367023955,719.0231584934905,721.5184599908322,723.8157258580372,688.4488560779508,669.1673679404818,655.7751964684622,667.2479104179779,646.0365540591724,705.5790366587603,820.6842072212404,852.3241333751354,862.400689752697,856.5560163282588,874.2690640526426,847.2674001612584,879.5829316190204,901.0208889393147,899.4774840261906,919.8688961057312]
    print(len(data_x))
    data_x = preprocess_data(data_x,9)
    print(data_x)
    print(len(data_x))
    pass
