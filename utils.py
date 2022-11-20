import base64
import datetime
import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from loguru import logger
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from PIL import Image
from scipy import signal

from onlineReading import settings


def get_fixations(coordinates, min_duration=100, max_duration=1200, max_distance=140):
    """
    根据gaze data(x,y,t1)计算fixation(x,y,r,t2) t2远小于t1
    :param coordinates: [(x,y,t),(x,y,t)]   # gaze点
    :return: [(x,y,duration),(x,y,duration)]  # fixation点
    """
    from collections import deque

    fixations = []
    # 先进先出队列
    working_queue = deque()
    remaining_gaze = deque(coordinates)

    while remaining_gaze:
        # 逐个处理所有的gaze data
        if len(working_queue) < 2 or (working_queue[-1][2] - working_queue[0][2]) < min_duration:
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
            if datum[2] > working_queue[0][2] + max_duration or with_distance(working_queue[0], datum, max_distance):
                fixations.append(from_gazes_to_fixation(list(working_queue)))
                working_queue.clear()
                break  # maximum data found
            working_queue.append(remaining_gaze.popleft())
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
    # TODO 改成 list不知道有没有问题
    return [int(sum_x / len(gazes)), int(sum_y / len(gazes)), gazes[-1][2] - gazes[0][2]]


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
        # coordinate = (
        #     # int(float(list_x[i]) * 1920 / 1534),
        #     # int(float(list_y[i]) * 1920 / 1534),
        #     int(float(list_x[i])),
        #     int(float(list_y[i])),
        #     int(float(list_t[i])),
        # )
        if i % 2 == 0:
            coordinate = (
                # int(float(list_x[i]) * 1920 / 1534),
                # int(float(list_y[i]) * 1920 / 1534),
                int(float(list_x[i])),
                int(float(list_y[i])),
                int(float(list_t[i])),
            )
            coordinates.append(coordinate)
    return coordinates


def generate_pic_by_base64(image_base64: str, save_path: str, filename: str):
    """
    使用base64生成图片，并保存至指定路径
    """
    data = image_base64.split(",")[1]
    img_data = base64.b64decode(data)
    # 如果目录不存在，则创建目录
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(save_path + filename, "wb") as f:
        f.write(img_data)
    logger.info("background已在该路径下生成:%s" % (save_path + filename))
    return save_path + filename


def paint_gaze_on_pic(coordinates: list, background: str, save_path: str) -> None:
    """
    根据坐标绘制fixation轨迹
    """
    import matplotlib.pyplot as plt

    img = cv2.imread(background)
    # plt.rc('pdf', fonttype=42)
    # c1 = coordinates[0:6]
    # c2 = coordinates[7:-1]
    # coordinates = c1 + c2
    # coordinates[0][1] = coordinates[0][1] - 14
    for i, coordinate in enumerate(coordinates):
        if i > 0:
            if coordinates[i][1] < coordinates[i - 1][1] - 22:
                coordinates[i][1] = coordinates[i][1] + 10
                if coordinates[i][1] < coordinates[i - 1][1] - 44:
                    coordinates[i][1] = coordinates[i - 1][1] - 20
            elif coordinates[i][1] > coordinates[i - 1][1] + 22:
                coordinates[i][1] = coordinates[i][1] - 10
                if coordinates[i][1] > coordinates[i - 1][1] + 44:
                    coordinates[i][1] = coordinates[i - 1][1] + 20
            if coordinates[i - 1][0] > coordinates[i][0] > coordinates[i - 1][0] - 30:
                coordinates[i][0] = coordinates[i][0] - 30
            elif coordinates[i - 1][0] < coordinates[i][0] < coordinates[i - 1][0] + 30:
                coordinates[i][0] = coordinates[i][0] + 30

    coordinates_tmp = []
    for i, coordinate in enumerate(coordinates):
        coordinates[i][1] = coordinates[i][1] - 30
        if i % 2 == 0:
            coordinates_tmp.append(coordinates[i])
    coordinates = coordinates_tmp
    coordinates.pop(0)

    img = cv2.imread(background)
    for i, coordinate in enumerate(coordinates):
        # print(coordinate[2])
        cv2.circle(
            img,
            (coordinate[0], coordinate[1]),
            3,
            (0, 0, 255),
            -1,
        )
        if i > 0:
            cv2.line(
                img,
                (coordinates[i - 1][0], coordinates[i - 1][1]),
                (coordinate[0], coordinate[1]),
                (0, 0, 255),
                1,
            )
        # if i % 2 == 0:
        #     # 标序号 间隔着标序号
        #     cv2.putText(
        #         img,
        #         str(i),
        #         (coordinate[0], coordinate[1]),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.7,
        #         (0, 0, 0),
        #         2,
        #     )
    s = save_path.split('.')[0] + '.png'
    cv2.imwrite(s, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    img = plt.imread(s)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(img)
    plt.axis('off')
    for i, coordinate in enumerate(coordinates):
        if i % 3 == 0:
            if i == 12:
                ax.text(coordinate[0],
                        coordinate[1] + cal_fix_radius(coordinate[2]) + 15, str(i),
                        family='Times New Roman', fontsize=7, verticalalignment='center',
                        horizontalalignment='center', color="black")
            # if i == 12:
            #     ax.text(coordinate[0] - cal_fix_radius(coordinate[2]) - 15,
            #             coordinate[1] + cal_fix_radius(coordinate[2]) + 15, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 15:
            #     ax.text(coordinate[0],
            #             coordinate[1] + cal_fix_radius(coordinate[2]) + 20, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 18:
            #     ax.text(coordinate[0] + cal_fix_radius(coordinate[2]) + 15,
            #             coordinate[1] - cal_fix_radius(coordinate[2]) - 15, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 21:
            #     ax.text(coordinate[0],
            #             coordinate[1] - cal_fix_radius(coordinate[2]) - 20, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # if i == 0:
            #     ax.text(coordinate[0] - cal_fix_radius(coordinate[2]) - 10, coordinate[1] - cal_fix_radius(coordinate[2]) - 10, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 3:
            #     ax.text(coordinate[0], coordinate[1] + cal_fix_radius(coordinate[2]) + 20, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 9:
            #     ax.text(coordinate[0] - cal_fix_radius(coordinate[2]) - 15, coordinate[1] + cal_fix_radius(coordinate[2]) + 15, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 12:
            #     ax.text(coordinate[0],
            #             coordinate[1] - cal_fix_radius(coordinate[2]) - 20, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 15:
            #     ax.text(coordinate[0] - cal_fix_radius(coordinate[2]) - 15,
            #             coordinate[1] - cal_fix_radius(coordinate[2]) - 15, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # if i == 0:
            #     ax.text(coordinate[0], coordinate[1] - cal_fix_radius(coordinate[2]) - 10, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # if i == 18:
            #     ax.text(coordinate[0] - cal_fix_radius(coordinate[2]) - 24, coordinate[1], str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 21:
            #     ax.text(coordinate[0] + cal_fix_radius(coordinate[2]) + 22, coordinate[1] - 5, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 3 or i == 9:
            #     ax.text(coordinate[0], coordinate[1] + cal_fix_radius(coordinate[2]) + 20, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            # elif i == 6 or i == 9:
            #     ax.text(coordinate[0], coordinate[1] + cal_fix_radius(coordinate[2]) + 20, str(i),
            #             family='Times New Roman', fontsize=7, verticalalignment='center',
            #             horizontalalignment='center', color="black")
            else:
                ax.text(cal_annotate_loc(i, coordinates)[0], cal_annotate_loc(i, coordinates)[1], str(i),
                        family='Times New Roman', fontsize=7, verticalalignment='center',
                        horizontalalignment='center', color="black")
        # else:
        #     ax.text(coordinate[0], coordinate[1], str(i + 1), fontsize=5, verticalalignment='center',
        #             horizontalalignment='center', color="tomato")
    plt.show()
    fig.savefig(save_path, dpi=1000)


def cal_fix_radius(fix_duration):
    rad = (fix_duration - 100) / 5
    if rad > 22:
        return 22
    if rad < 4:
        return 4
    else:
        return round(rad)


def cal_annotate_loc(i, coordinates):
    rad = cal_fix_radius(coordinates[i][2])
    if i == 0:
        return [coordinates[0][0] - rad - 15, coordinates[0][1] - rad - 15]
    if i == len(coordinates) - 1:
        return [coordinates[i][0] + rad + 15, coordinates[i][1] + rad + 15]
    if coordinates[i + 1][0] > coordinates[i][0] > coordinates[i - 1][0]:
        if coordinates[i + 1][1] <= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0] + rad + 15, coordinates[i][1] + rad + 15]
        elif coordinates[i + 1][1] >= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0], coordinates[i][1] - rad - 15]
        elif coordinates[i + 1][1] <= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0], coordinates[i][1] + rad + 15]
        elif coordinates[i + 1][1] >= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0] + rad + 15, coordinates[i][1] - rad + 15]

    if coordinates[i + 1][0] <= coordinates[i][0] >= coordinates[i - 1][0]:
        if coordinates[i + 1][1] <= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0] + rad + 15, coordinates[i][1]]
        elif coordinates[i + 1][1] >= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0] + rad + 15, coordinates[i][1] - rad - 15]
        elif coordinates[i + 1][1] <= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0] + rad + 15, coordinates[i][1] + rad + 15]
        elif coordinates[i + 1][1] >= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0] + rad + 15, coordinates[i][1]]

    if coordinates[i + 1][0] >= coordinates[i][0] <= coordinates[i - 1][0]:
        if coordinates[i + 1][1] <= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0] - rad - 15, coordinates[i][1]]
        elif coordinates[i + 1][1] >= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0] - rad - 15, coordinates[i][1] - rad - 15]
        elif coordinates[i + 1][1] <= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0] - rad - 15, coordinates[i][1] + rad + 15]
        elif coordinates[i + 1][1] >= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0] - rad - 15, coordinates[i][1]]

    if coordinates[i + 1][0] <= coordinates[i][0] <= coordinates[i - 1][0]:
        if coordinates[i + 1][1] <= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0] - rad - 15, coordinates[i][1] - rad - 15]
        elif coordinates[i + 1][1] >= coordinates[i][1] <= coordinates[i - 1][1]:
            return [coordinates[i][0], coordinates[i][1] - rad - 15]
        elif coordinates[i + 1][1] <= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0], coordinates[i][1] + rad + 15]
        elif coordinates[i + 1][1] >= coordinates[i][1] >= coordinates[i - 1][1]:
            return [coordinates[i][0] - rad - 15, coordinates[i][1] - rad - 15]


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
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate?" + "q=%s&from=en&to=zh&appid=%s&salt=%s&sign=%s" % (
            content,
            appid,
            salt,
            sign,
        )
        # 4. 解析结果
        response = requests.get(url)
        data = json.loads(response.text)
        endtime = datetime.datetime.now()
        logger.info("翻译接口执行时间为%sms" % round((endtime - starttime).microseconds / 1000 / 1000, 3))
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
    return get_vertical_saccades(fixations, locations) / saccade_times if saccade_times != 0 else 0


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


def get_reading_times_and_dwell_time_of_sentence(fixations, buttons_location, sentence_dict):
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
        if sentences[key]["end_word_index"] > word_index >= sentences[key]["begin_word_index"]:
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
            sum_saccade_angle = sum_saccade_angle + get_saccade_angle(pre_fixation, fixation)
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

    importance = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=1000)
    return importance


def get_word_and_sentence_from_text(content):
    print(content)
    sentences = content.replace("...", ".").replace("..", ".").split(".")
    print(sentences)
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


def topk_tuple(data_list, k=10):
    # 笨拙的top_k的方法 针对tuple [('word',num)]
    top_k = []
    data_list_copy = [x for x in data_list]
    print(data_list)
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


def calculate_similarity(word_dict, level="word"):
    visual = word_dict.get("visual")
    topic = word_dict.get("topic_relevant")
    word_attention = word_dict.get("word_attention")
    sentence_attention = word_dict.get("sentence_attention")
    difficulty_word = word_dict.get("word_difficulty")
    difficulty_sentence = word_dict.get("sentence_difficulty")

    result = []

    if level == "word":
        result = [
            len(set(visual).intersection(set(topic))) / len(set(visual).union(set(topic))),
            len(set(visual).intersection(set(word_attention))) / len(set(visual).union(set(word_attention))),
            len(set(visual).intersection(set(difficulty_word))) / len(set(visual).union(set(difficulty_word))),
        ]
    elif level == "sentence":
        result = [
            len(set(visual).intersection(set(topic))) / len(set(visual).union(set(topic))),
            len(set(visual).intersection(set(sentence_attention))) / len(set(visual).union(set(sentence_attention))),
            len(set(visual).intersection(set(difficulty_sentence))) / len(set(visual).union(set(difficulty_sentence))),
        ]
    return result


def calculate_identity(word_dict, level="word"):
    visual = word_dict.get("visual")
    topic = word_dict.get("topic_relevant")
    word_attention = word_dict.get("word_attention")
    word_dict.get("sentence_attention")
    difficulty_word = word_dict.get("word_difficulty")
    word_dict.get("sentence_difficulty")

    result = []

    if level == "word":
        data = [visual, topic, word_attention, difficulty_word]

        for attr in data[1:]:
            total = len(attr)
            ident = 0
            a = data[0]
            for i in range(total):
                if a[i] == attr[i]:
                    ident += 1
            result.append(ident / total)

    return result


def paint_bar_graph(data_dict, base_path, attribute="similarity"):
    size = 3
    # x轴坐标
    x = np.arange(1, size + 1)
    # k的取值的个数
    array_num = len(data_dict)

    # 收集visual attention与topic-relavant word、word attention以及difficult word之间的相似性或一致性
    datalist = []

    x_labels = [
        "viusal&topic-relevant",
        "visual&word attention",
        "visual&difficult word",
    ]

    # 有array_num种类型的数据，n设置为array_num
    total_width, n = 0.8, array_num
    # 每种类型的柱状图宽度
    width = total_width / n

    # 重新设置x轴的坐标
    x = x - (total_width - width) / size

    fig, ax = plt.subplots()

    for data in data_dict:
        data_ = [data.get("k")]
        if attribute == "similarity":
            data_.append(data.get("similarity"))
        else:
            data_.append(data.get("identity"))
        datalist.append(data_)

    # haul柱状图
    for i in range(len(datalist)):
        ax.bar(
            x + width * (i + 1 - array_num / 2),
            [j for i in datalist[i][1:] for j in i],
            width=width,
            label="k=" + str(datalist[i][0]),
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(x_labels)

    plt.ylim(0, 0.5)

    # 标注X轴，标注Y轴
    # plt.xlabel("groups")
    ax.set_ylabel(attribute)
    fig.tight_layout()
    # 显示图例
    plt.legend()
    # 保存图片
    plt.savefig(base_path + attribute + ".png")
    # 显示柱状图
    plt.show()


def get_top_k(data_list, k=50):
    tag = [0 for x in data_list]
    top_k_index = []
    while k > 0:
        max_data = -1
        max_index = -1
        for i, data in enumerate(data_list):
            if tag[i] == 0 and data > max_data:
                max_data = data
                max_index = i
        if max_index == -1:
            break
        tag[max_index] = 1
        top_k_index.append(max_index)
        k -= 1
    return top_k_index


def get_word_by_one_gaze(word_locations, gaze):
    for i, loc in enumerate(word_locations):
        if loc[0] < gaze[0] < loc[2] and loc[1] < gaze[1] < loc[3]:
            return i
    return -1


def apply_heatmap(background, data, heatmap_name, alpha, title):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from pyheatmap.heatmap import HeatMap

    image = cv2.imread(background)
    background = Image.new("RGB", (image.shape[1], image.shape[0]), color=0)
    # 开始绘制热度图
    hm = HeatMap(data)
    hit_img = hm.heatmap(base=background, r=40)  # background为背景图片，r是半径，默认为10
    hit_img = cv2.cvtColor(np.asarray(hit_img), cv2.COLOR_RGB2BGR)  # Image格式转换成cv2格式
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 255, 255), -1)  # 设置蓝色为热度图基本色蓝色
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)  # 将背景热度图覆盖到原图
    image = cv2.addWeighted(hit_img, alpha, image, 1 - alpha, 0)  # 将热度图覆盖到原图
    plt.imshow(image)
    plt.title(title)
    plt.show()
    cv2.imwrite(heatmap_name, image)
    logger.info("heatmap已经生成:%s" % heatmap_name)


def find_threshold(df):
    d = df["color"]
    Percentile = np.percentile(d, [0, 25, 50, 75, 100])
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR * 1.5
    DownLimit = Percentile[1] - IQR * 1.5
    return Percentile[1], Percentile[3], DownLimit, UpLimit


# 处理两个图片的拼接
def join_two_image(img_1, img_2, save_path, flag="horizontal"):  # 默认是水平参数
    # 1、首先使用open创建Image对象，open()需要图片的路径作为参数
    # 2、然后获取size，size[0]代表宽，size[1]代表长，分别代表坐标轴的x,y
    # 3、使用Image.new创建一个新的对象
    # 4、设置地点，两个图片分别在大图的什么位置粘贴
    # 5、粘贴进大图，使用save()保存图像
    img1 = Image.open(img_1)
    img2 = Image.open(img_2)
    size1, size2 = img1.size, img2.size
    if flag == "horizontal":
        joint = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_path)


# 处理n张图的垂直拼接
def join_images_vertical(img_list, save_path):
    vertical_size = 0
    horizontal_size = 0

    for img in img_list:
        image = Image.open(img)
        horizontal_size = image.size[0]
        vertical_size += image.size[1]
    joint = Image.new("RGB", (horizontal_size, vertical_size))

    x = 0
    y = 0
    for img in img_list:
        image = Image.open(img)
        loc = (x, y)
        joint.paste(image, loc)
        y += image.size[1]
    joint.save(save_path)


def pixel_2_cm(pixel):
    """像素点到距离的转换"""
    cmPerPix = 23.8 * 2.54 / math.sqrt(math.pow(16, 2) + math.pow(9, 2)) * 16 / 1534
    return pixel * cmPerPix


def get_test_heatmap():
    import pandas as pd

    csv = pd.read_csv("static\\data\\dataset\\cnn.csv")
    gaze_x = csv["gaze_x"]
    gaze_y = csv["gaze_y"]
    gaze_x_filter = csv["gaze_x_filter"]
    gaze_y_filter = csv["gaze_y_filter"]

    coordinate1 = []
    coordinate2 = []
    for i in range(len(gaze_x)):
        tmp1 = (int(gaze_x[i]), int(gaze_y[i]))
        tmp2 = (int(gaze_x_filter[i]), int(gaze_y_filter[i]))
        coordinate1.append(tmp1)
        coordinate2.append(tmp2)

    apply_heatmap(
        "static\\background\\img.png",
        coordinate1,
        "static\\background\\gaze_before.png",
        0.4,
        "gaze_before",
    )
    apply_heatmap(
        "static\\background\\img.png",
        coordinate2,
        "static\\background\\gaze_after.png",
        0.4,
        "gaze_after",
    )


def split_csv(exp_id):
    filename = "static\\data\\dataset\\10-31-43.csv"
    df = pd.read_csv(filename)
    df = df[df["experiment_id"] == exp_id]
    df.to_csv("static\\data\\dataset\\" + str(exp_id) + ".csv")
    print("ok")


def get_para_from_txt(path, tar_page):
    dict = {}
    for line in open(path):
        print(line)
        line = line.split(" ")
        article_id = int(line[0])
        pages = line[1].split("|")
        word_num = 0
        para_1 = 0
        for i, page in enumerate(pages):
            paras = page.split(",")
            if i == 0:
                para_1 = int(paras[0])
                word_num += int(paras[-1])
        dict[article_id] = {"para_1": para_1, "word_num": word_num + 1}
    return dict


if __name__ == "__main__":
    path = "static\\data\\other\\paraLoc.txt"
    dict = get_para_from_txt(path, 1)
    print(dict)
