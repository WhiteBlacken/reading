import base64
import datetime
import json
import math
import os
import random
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from loguru import logger
from nltk.corpus import stopwords
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from PIL import Image
from scipy import signal
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from feature.utils import detect_fixations, keep_row, textarea, word_index_in_row, row_index_of_sequence
from onlineReading import settings
from semantic_attention import get_word_difficulty, get_word_familiar_rate, generate_word_attention


def in_danger_zone(x: int, y: int, danger_zone: list):
    for zone in danger_zone:
        if zone[0][0] < x < zone[0][1] and zone[1][0] < y < zone[1][1]:
            return True
    return False


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


def get_index_in_row_only_use_x(row: list, x: int):
    for i, word in enumerate(row):
        if word["left"] <= x <= word["right"]:
            return i
    return -1


def get_row(index, rows):
    for i, row in enumerate(rows):
        if row['begin_index'] <= index <= row['end_index']:
            return i
    return -1


def get_item_index_x_y(location, x, y, word_list=[], rows=[], remove_horizontal_drift=False):
    """根据所有item的位置，当前给出的x,y,判断其在哪个item里 分为word level和row level"""
    location = json.loads(location)

    flag = False
    index = -1
    # 先找是否正好在范围内
    for i, word in enumerate(location):
        if word["left"] <= x <= word["right"] and word["top"] <= y <= word["bottom"]:
            index = i
            break

    # 如果不在范围内,找最近的单词
    min_dist = 100
    if index == -1:
        for i, word in enumerate(location):
            center_x = (word["left"] + word["right"]) / 2
            center_y = (word["top"] + word["bottom"]) / 2
            dist = get_euclid_distance(x, center_x, y, center_y)
            weight_dist = dist - math.log(word["right"] - word["left"])
            if weight_dist < min_dist:
                min_dist = weight_dist
                index = i
                flag = True

    if remove_horizontal_drift and index != -1:
        word = word_list[index]
        stop_words = set(stopwords.words('english'))

        if word.lower() in stop_words:
            print(f"stop words:{word}")
            # 如果该词是停用词，则进行调整
            now_row = get_row(index, rows)
            if now_row != -1:
                left_row = get_row(index - 1, rows)
                right_row = get_row(index + 2, rows)

                left_index = rows[now_row]['begin_index'] if left_row != now_row else index - 1
                right_index = rows[now_row]['end_index'] if right_row != now_row else index + 2

                candidates_with_stop = [i for i in range(left_index, right_index + 1) if i != index]

                candidates = []
                # 将stop words删掉
                for can in candidates_with_stop:
                    if word_list[can].lower() not in stop_words:
                        candidates.append(can)
                if len(candidates) != 0:
                    difficulty_list = [1000 - get_word_familiar_rate(word_list[index]) for index in candidates]
                    if sum(difficulty_list) == 0:
                        diff_pro = [1 / (len(difficulty_list)) for _ in difficulty_list]
                    else:
                        diff_pro = [diff / sum(difficulty_list) for diff in difficulty_list]  # 分配到该单词上的概率
                    rand_num = random.randint(0, 1)

                    cnt = 0
                    diff_sum = 0
                    for i, diff in enumerate(diff_pro):
                        diff_sum += diff
                        if i == 0:
                            continue
                        if rand_num <= diff_sum:
                            cnt = i - 1
                            break
                    print("----------------")
                    print(f"index有{index}转到{candidates[cnt]}上")
                    print([word_list[i] for i in candidates])
                    print(diff_pro)
                    print(index)
                    # print(word_list[index])
                    # print(word_list[candidates[cnt]])
                    # print(f'由单词{word_list[index]}转到{word_list[candidates[cnt]]}上')
                    index = candidates[cnt]
                    flag = True

    return index, flag


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


def generate_pic_by_base64(image_base64: str, save_path: str, filename: str, isMac=False):
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

    # img = Image.open(save_path + filename)
    # new_img = img.resize((1920, 1080))
    # new_img.save(save_path + filename)

    if isMac:
        img = Image.open(save_path + filename)
        width = img.size[0]
        times = width / 2880
        new_img = img.resize((int(1440 * times), int(900 * times)))
        new_img.save(save_path + filename)
    return save_path + filename


def paint_gaze_on_pic(fixations: list, background: str, save_path: str) -> None:
    """
    根据坐标绘制fixation轨迹
    """
    import matplotlib.pyplot as plt
    key_fixations = [(x,i) for i, x in enumerate(fixations) if x[2] >= 500]
    print("key_fixations: {k}".format(k=key_fixations))

    canvas = cv2.imread(background)
    for i, fix in enumerate(fixations):
        dur = fix[2]
        l = 0
        if int(dur / 40) > 18:
            l = 18
        else:
            l = int(dur / 40)
        x = int(fix[0])
        y = int(fix[1])
        cv2.circle(
            canvas,
            (x, y),
            l,
            (0, 0, 255),
            2,
        )

        if i > 0:
            cv2.line(
                canvas,
                (x, y),
                (int(fixations[i - 1][0]), int(fixations[i - 1][1])),
                (0, 0, 255),  # GBR
                1,
            )

    cv2.imwrite(save_path, canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    img = plt.imread(save_path)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(img)
    plt.axis("off")
    for i, tuple in enumerate(key_fixations):
        fix = tuple[0]
        index = tuple[1]
        ax.text(
            fix[0],
            fix[1]-20,
            str(index),
            family="Times New Roman",
            fontsize=7,
            verticalalignment="center",
            horizontalalignment="center",
            color="black",
        )
    plt.show()
    fig.savefig(save_path, dpi=200)


def cal_fix_radius(fix_duration):
    rad = (fix_duration - 100) / 18
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
    return (
        saccade_times,
        forward_saccade_times / saccade_times if saccade_times != 0 else 0,
        backward_saccade_times / saccade_times if saccade_times != 0 else 0,
        sum_saccade_length / saccade_times if saccade_times != 0 else 0,
        sum_saccade_angle / saccade_times if saccade_times != 0 else 0,
    )


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


def normalize(data: list) -> list:
    return [(x - min(data)) / (max(data) - min(data)) for x in data]


def get_importance(text):
    """获取单词的重要性"""
    from keybert import KeyBERT

    kw_model = KeyBERT()

    importance = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=2000)
    return importance


def get_word_and_sentence_from_text(content):
    # print(content)
    sentences = content.replace("...", ".").replace("..", ".").split(".")
    # print(sentences)
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


def get_word_count(content):
    all_word_list, all_sentence_list = get_word_and_sentence_from_text(content)
    return len(all_word_list)


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
def join_images_vertical(img_list, save_path, is_filename=True):
    vertical_size = 0
    horizontal_size = 0

    for i, img in enumerate(img_list):
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


def preprocess_data(data, filters):
    cnt = 0
    for filter in filters:
        if filter["type"] == "median":
            data = signal.medfilt(data, kernel_size=filter["window"])
            cnt += 1
        if filter["type"] == "mean":
            data = meanFilter(data, filter["window"])
            cnt += 1
    # data = meanFilter(data, win)
    return data


def format_gaze(
        gaze_x: str, gaze_y: str, gaze_t: str, use_filter=True, begin_time: int = 500, end_time: int = 500
) -> list:
    list_x = []
    list_y = []
    list_t = []
    # gaze_x = "509.7145637731967, 520.4559880127492, 535.9621321867301, 566.3998924535039, 608.1176125209665, 659.1997062663372, 713.4659156361064, 765.227489931538, 810.3376093113789, 843.9298380337143, 858.7391996423017, 855.6349671374526, 835.7027658667373, 801.0116629923248, 756.1242095016527, 708.4275991009375, 659.9617177985868, 610.2421193116016, 556.0995439861697, 495.3609659225425, 431.98913207833647, 368.1923079568565, 310.2933766810511, 264.7162945586175, 237.2970251777884, 224.8648830796645, 226.19023013475848, 237.4651874164063, 255.89039958654266, 275.9973302459488, 295.90738409827696, 314.29468853784357, 330.89202112070086, 345.1059746353558, 357.20561111642303, 369.4213161951955, 382.2946338211349, 395.6223256992315, 409.31061930508474, 423.3743309415954, 433.25926674899955, 437.8712020847862, 436.63140306446985, 429.56769920714504, 417.3410748715575, 404.8854189779222, 393.4555350080161, 384.798208140481, 379.37841417732386, 377.3930531507446, 378.41325204688803, 383.2746773873928, 390.90407113131715, 401.39572020645835, 414.48352484935447, 427.49961309709386, 438.175149789273, 446.62949638841565, 452.2091032369202, 454.43817463709263, 454.31622332636954, 452.9776668849528, 450.43291694175997, 447.00874832559185, 443.667451593423, 441.07787323329165, 439.42791376844417, 438.7820486376801, 439.23140328862485, 440.52551640093435, 442.41078169988697, 444.5113981389897, 446.6984148406435, 448.7895809095977, 450.4512287305036, 451.1548993164054, 450.8519530391547, 449.60686533755086, 447.5107616592196, 444.7304758118348, 442.33749233977585, 441.2748963218126, 441.65927981784523, 443.79340919181806, 449.08692229189967, 458.7213368675555, 471.5930271496962, 488.1000688393211, 507.69769687117406, 529.9331340509731, 551.2980511412159, 572.1166050493512, 591.2428681936797, 608.8580716128812, 621.6388561510137, 630.6190306282857, 635.4342773455501, 636.7158561236067, 634.5245346250879, 631.2268113520486, 629.0505483160331, 629.933013703467, 634.4329630918725, 644.3843107194505, 659.7870565862013, 678.1342048511488, 696.9809999668274, 715.8009377336315, 730.9261896751577, 742.3567557914065, 751.4660944967816, 760.2534537644098, 768.7614395149607, 779.0930529534965, 791.5011980030246, 804.834889223798, 816.867032733319, 826.8569069789792, 834.2663380270569, 838.5895180315376, 840.63264429591, 841.568291723998, 842.6736392631623, 845.2088837335561, 849.4660973024087, 855.5486790398825, 863.4566289459777, 872.7519249268116, 881.4523472758013, 889.4795595045027, 896.3856345497716, 902.1705724116076, 905.8764731977599, 908.49444676152, 909.0634846183178, 907.8477384535287, 904.847208267152, 900.8465013519835, 896.4955612201127, 894.3130494880634, 894.6047005713383, 897.6530060604173, 904.4523252384073, 913.7206979372659, 923.5511698120347, 933.332272031707, 942.8236743970165, 950.0610725946565, 956.0219007631329, 961.3287525905669, 966.4473423636306, 971.7688612752232, 978.2406216050813, 985.6411191108231, 993.3821538002949, 1001.2043914278087, 1010.0668037725624, 1019.9902420825762, 1030.3645436230101, 1041.5378306311004, 1053.6022398768998, 1063.0314875059453, 1070.3730108431769, 1076.1289932927161, 1080.3116819227464, 1083.4945121621756, 1088.0281285510387, 1093.0594326446208, 1099.4504827906715, 1107.4564377576967, 1115.891454004628, 1124.3057909957377, 1132.768678152964, 1139.441552241208, 1143.14193425015, 1144.4783891824234, 1143.6839725248742, 1140.996897477781, 1137.4433717460813, 1134.5115442068388, 1132.2235362880392, 1130.5793479896818, 1129.578979311767, 1129.1079409317595, 1128.6813356843552, 1128.2799891606169, 1127.507805851607, 1125.351415480175, 1121.9085232373084, 1117.0969041089024, 1110.9327854848466, 1104.336696396421, 1100.1204959369513, 1098.2353315109442, 1098.6812031183993, 1101.4499970643724, 1105.8889418131164, 1109.4144300094308, 1112.0264616533157, 1114.7166801820156, 1117.4850855955308, 1120.4600159072663, 1124.426589656247, 1129.3848068424725, 1133.5208165158572, 1136.834618676401, 1139.3969514434825, 1141.40402526127, 1142.9967217687736, 1144.9905812976947, 1147.4019699500225, 1150.4291125921343, 1153.9487172820297, 1157.6927394968366, 1161.5411446277699, 1166.0081942093798, 1170.8221265053403, 1176.696294818535, 1184.3758116760189, 1194.3843586392268, 1205.6473822456755, 1217.9701735130886, 1229.6302902719285, 1239.3781144806633, 1247.9920791138832, 1256.0130420849491, 1263.4991165049892, 1271.2672423515878, 1279.9490880230846, 1284.7056537199346, 1283.923444545035, 1277.5576415045439, 1265.3387397837455, 1247.1732977812212, 1228.6448056304703, 1212.9863889508615, 1200.5664749735183, 1192.1625341016747, 1189.3392505037732, 1190.3876716843247, 1194.9148061065202, 1203.3162992868663, 1215.4305487003007, 1228.5237221328684, 1242.4053240347878, 1255.2894797614124, 1266.0618912432353, 1275.0643045897627, 1283.617510781222, 1291.6141761568456, 1298.9553361872943, 1306.1300096153843, 1311.953490873986, 1316.1941659778302, 1318.944321915031, 1321.8670559901552, 1325.0077883978283, 1329.3690273940376, 1336.64491711812, 1346.8908680063937, 1360.2064760457038, 1376.9638263299953, 1396.7189895419133, 1417.5465716993872, 1439.5796852002256, 1459.8078132027026, 1480.11904899086, 1500.0610220930998, 1518.922326011466, 1536.2395687019452, 1553.1846492291957, 1563.330380455221, 1566.1086426349416, 1562.2630639609501, 1551.829897179025, 1535.455200045644, 1518.5282619481036, 1503.2045590046305, 1490.228972815361, 1480.0679195842554, 1475.2184835732944, 1474.8616790007495, 1477.9197678075068, 1484.0203091934984, 1493.4848718386138, 1502.356500482999, 1510.155337613442, 1518.643764870258, 1529.716125440023, 1544.0128651951666, 1563.1528980114676, 1587.2407970604595, 1612.7517990615104, 1635.8972176414698, 1653.7318307098692, 1665.8990414426244, 1672.3285613148698, 1674.7827719669226, 1675.3751471494913, 1675.8556862987448, 1676.6005481264183, 1677.6520106534995, 1679.0100738799877, 1680.2364766776159, 1681.3312190463844, 1681.7017348479817, 1676.7654360414358, 1665.353269940208, 1644.4349545251694, 1613.7743492476307, 1569.1236734938466, 1517.4388649071122, 1460.1487738423466, 1397.7704369156472, 1326.5405305069676, 1251.9151133008131, 1170.8997862735816, 1080.0993140530554, 984.8569095346759, 890.8180921168681, 800.0996549883018, 715.3359812310434, 643.1040819467249, 582.4499241963788, 532.9074239434485, 493.30395109197053, 462.4431261803041, 437.4864951348506, 419.977096073506, 410.14453609774245, 406.34147004713066, 407.7509437225252, 412.79673317556245, 414.5580114335465, 409.1546045123808, 396.29054692928105, 375.85200672602593, 349.08369683047687, 321.9418924992547, 299.3677851113436, 281.95330563231175, 270.618014953319, 265.791801091142, 266.1235554708396, 270.55282244682223, 280.58548375400994, 295.1987822815639, 313.6912217090125, 334.2944384039789, 354.3370274776819, 370.2152944947132, 381.67139883615675, 388.8231066045325, 392.5545996160289, 394.43679889380473, 396.6503924869408, 401.6642345631843, 409.2427929174954, 419.3860675498743, 431.32587090473635, 444.41190351098703, 455.1101964946505, 463.53851595824653, 469.69686190177526, 474.31389258289045, 478.45028894508266, 484.3371291102382, 491.9869191896445, 501.5604899059152, 513.8297786122054, 527.1089585093124, 539.5618384916668, 551.1634063366939, 561.3717084146734, 567.8324893782467, 571.4423537228438, 573.3463979351927, 574.7831644084471, 576.9143008192905, 581.0519982815771, 587.1962567953067, 595.3470763604797, 602.7208516214089, 607.9768252234339, 611.1149971665548, 612.1353674507714, 612.2223352323, 614.9592123399891, 620.9312369333975, 630.1384090125251, 642.580728577372, 655.8893973155058, 665.5702281051842, 671.5732097974571, 673.8983423923241, 672.5456258897857, 668.6994594460579, 666.1140953090819, 666.2649365158345, 671.8593473885803, 683.6357036211632, 701.5940052135832, 723.9368708070272, 747.7635054764911, 767.0801404677683, 777.4118496098592, 778.0130636883335, 768.1761433310547, 749.2930459327894, 724.1491015639858, 699.4790354851104, 676.7739861250237, 658.472379183173, 647.1539014052785, 645.9825523519844, 653.0428401422207, 668.6106296352912, 691.754615807158, 717.6384786764282, 742.5421630772136, 765.4470662900114, 784.4348851057273, 799.6569175156128, 813.7020627132164, 827.101769904041, 838.7091870068944, 849.7242690568268, 858.9806965942786, 865.7830845321205, 871.809792742862, 878.093506182513, 884.383251251746, 891.744014901109, 901.3268463855163, 911.9103946418352, 923.5797241904323, 937.5468656893388, 953.4683803490996, 970.9529807557107, 989.6945207656441, 1009.7711851538509, 1028.3824992499415, 1046.017042985792, 1061.901259365834, 1075.7007774214276, 1087.1403906248179, 1097.5488563031793, 1105.5567753316013, 1110.537520484871, 1113.134618327101, 1113.1282329385451, 1110.245930098878, 1104.8276976502063, 1098.3651603960448, 1090.9896433362233, 1083.4649066599943, 1076.7509675776532, 1072.5081568146056, 1070.4095052861896, 1071.1931232489937, 1075.069920093236, 1081.9921436883283, 1091.1719882656598, 1102.6868245512446, 1115.3159192494018, 1128.4979017577393, 1141.3757215390674, 1152.8006476820365, 1162.3099163086379, 1169.9336757407236, 1175.0162987285428, 1177.662218752305, 1178.4314578497258, 1177.422868789106, 1175.0232637877714, 1171.7549678133607, 1168.2512396187915, 1165.0372407026528, 1162.5825869480032, 1161.4160940676697, 1163.165757451632, 1169.3157589070147, 1179.9741211089956, 1194.753211016557, 1212.5953972040452, 1230.550339090478, 1243.428582317609, 1249.2112259630733, 1248.0001534265298, 1240.324180420805, 1227.6075355365606, 1216.2003004974072, 1210.1177216894644, 1209.762303770761, 1215.134046741297, 1226.2329506010724, 1238.738895067127, 1248.4074415530122, 1254.1921272374043, 1256.0929521203036, 1254.0404307690417, 1249.396976123595, 1244.8283127655995, 1241.2198523636164, 1238.6410803503143, 1237.5076370827976, 1237.832341434477, 1239.5142189070227, 1244.2250954464944, 1252.7470692902357, 1264.5900296704629, 1279.9361474181987, 1297.6620501064485, 1314.0342368156203, 1327.847457149645, 1340.0377122525356, 1351.3956565006013, 1362.5444848090115, 1375.2113606629714, 1389.1825484119122, 1402.7412438060858, 1412.9939209494169, 1418.7772074356458, 1420.1841612478338, 1417.782184762605, 1412.4960369290436, 1407.997291060593, 1406.3996275960324, 1407.9445211149327, 1412.6319716172936, 1420.4619791031153, 1428.6215897329153, 1435.2326358120824, 1439.8121681814762, 1442.3601868410963, 1442.8766917909425, 1442.5248542493919, 1442.5031426801736, 1444.2005543855626, 1449.5932223132222, 1458.8000552517306, 1472.3392410667584, 1489.8742790353047, 1509.7052282449883, 1528.3742141418838, 1545.8701800518402, 1560.8150236124968, 1572.650115444791, 1581.332769205374, 1587.3992415437617, 1590.5149194425212, 1591.4258085213353, 1590.5693472073708, 1588.6136597307857, 1586.9852768837907, 1586.1974442134817, 1586.5587095109963, 1588.0977907693114, 1591.1400300238245, 1595.1744292189492, 1599.6280190665063, 1603.9901502074024, 1608.203386655683, 1611.7325220575015, 1614.868272648407, 1618.3845148722703, 1622.4475275305037, 1627.7392339968624, 1634.3963115716892, 1641.9786652526795, 1649.632253185325, 1657.7885901627506, 1665.1443093325347, 1671.4660873012479, 1677.0512058115921, 1682.2607145652812, 1686.2326060525102, 1689.6139978357921, 1692.5375018565708, 1693.9332588638968, 1693.6663133627142, 1692.2029808362447, 1689.546305193621, 1685.7971652078038, 1682.6599940846597, 1680.3007621282934, 1678.719469338705, 1677.9161157158937, 1677.8907012598604, 1675.1260035893388, 1667.0551746444967, 1645.948222388392, 1602.6728343343316, 1527.0536638109963, 1420.1620115270812, 1283.4490054984317, 1126.908534221398, 962.3895594179361, 807.4203204319529, 667.8655372718861, 548.4472547174412, 452.3676718867447, 383.3258027959615, 339.22007083083554, 316.165924530492, 310.531624703179, 316.85107577136176, 328.63687312390005, 338.208838642866, 345.28020074252277, 349.8509594228704, 352.02923320258253, 352.4773535846052, 353.6241826285358, 355.46972033437424, 358.0139667021206, 361.0406846944276, 363.3058489793669, 364.68654259569104, 364.03887872705116, 361.362857373447, 356.7665970535525, 351.0381193612332, 345.56907727699, 343.7388288173014, 345.54790021286146, 351.6746382298572, 361.98445646311717, 374.0627998353816, 384.5826127627407, 393.54284278380635, 399.1990887467256, 401.55135065149824, 401.74544755613067, 400.8729638444032, 399.87988963092323, 400.5199285799101, 403.40116363423476, 408.95005381892213, 417.16659913397234, 426.15987181155896, 435.69439019829815, 444.7354681192663, 452.6638320703535, 460.11164561467024, 468.16689116891985, 475.739924744822, 485.3271794775684, 498.77066090146945, 515.0754639332242, 535.4391254757363, 561.236379068371, 588.1513876632744, 612.7312769977394, 635.0955068860555, 655.11287534927, 671.4256163066864, 686.389126088008, 701.7279355070323, 717.6591859326414, 730.2825964828696, 739.8726032149798, 745.3524016285517, 746.2695882982682, 742.389523596917, 736.4036349497235, 728.7726868580958, 721.3168182986682, 716.7285938323366, 716.7870050138285, 722.2848597295157, 733.1194523674827, 748.979683420003, 766.0750912222882, 781.630032557576, 794.0588916531226, 803.3376964242983, 808.383198564153, 810.0680340852398, 809.42337482185, 806.7671841038397, 801.8387107485856, 795.1795789095634, 788.3034224074534, 781.6627608262042, 776.2072832788477, 773.2006249669705, 773.0620900099972, 775.2576862707334, 780.2103019053362, 787.7608363249919, 797.0536646540183, 807.7065798257808, 820.4613835412623, 834.4234772602464, 849.4711211116507, 864.6156081418306, 879.3634401257829, 891.5625393421026, 900.9290406710188, 906.7591919515893, 909.6901288336187, 909.9839311428101, 908.1688251392285, 904.2517513730256, 898.1344540002003, 890.2239039243944, 880.7507032603468, 870.9062063250251, 861.7766792946975, 854.5781299860115, 849.0336340896454, 845.2649514156537, 842.8670482917332, 841.3601362062707, 840.1362112509429, 839.3934021449593, 840.5087148069922, 844.1773260128223, 851.889169952245, 864.3936314200552, 882.6138546158614, 903.5724047364301, 926.2670902304695, 948.1535243095677, 968.3058379975166, 984.653988673784, 998.5477064539724, 1009.5245571409639, 1015.1079457948484, 1014.4356737361207, 1008.5427622750464, 997.5218696023023, 982.015349033064, 968.3134040413997, 958.5385845692938, 952.9739956766492, 951.8693829934906, 955.4014190441696, 960.7268453901931, 966.3548516377471, 972.6157161788018, 979.6561057148799, 987.4919696361651, 996.1233079426571, 1005.8726689301445, 1015.2625331779743, 1023.2503303930287, 1030.9431158477887, 1039.528970975644, 1049.2264308747124, 1061.1436717642707, 1077.0901083392857, 1094.097091786835, 1109.92403478995, 1124.3801030359098, 1136.948144846976, 1145.3016467563561, 1151.109580389956, 1155.3095433304018, 1157.7540374558444, 1158.8496415645277, 1160.7321781665157, 1163.7592216125609, 1168.0494394485734, 1173.8420518709538, 1180.952549409755, 1187.435800511642, 1192.5766564751086, 1196.3469315613852, 1197.8248967097468, 1196.2149559181457, 1192.2629644225246, 1186.3137459311922, 1178.3917269379624, 1169.8140754718333, 1162.1883610964844, 1155.9680048933635, 1151.599714710844, 1149.0834905489248, 1147.9888655963437, 1147.5549083904152, 1147.4447512809784, 1146.8032304986202, 1145.6303460433403, 1144.3473044786238, 1142.9541058044701, 1141.7815855603064, 1141.9170960794493, 1144.0985951929408, 1148.3810640230527, 1155.339767230153, 1165.5504946926424, 1178.6080667357212, 1193.0365676973051, 1207.8836222058812, 1221.9987009407153, 1233.6190816428789, 1241.1551579222112, 1245.201943537018, 1245.5481115305597, 1241.407463042082, 1233.3853863226395, 1223.020709720011, 1209.1573311713935, 1192.8762109366535, 1176.9002760580336, 1162.0105222560926, 1148.1881018557913, 1139.1203440336235, 1135.0328002727133, 1136.6603922455208, 1144.099571403241, 1157.2207956847276, 1173.0896879099632, 1190.762322903082, 1206.0459302769193, 1218.0410251898902, 1227.245922315786, 1233.8065171000612, 1238.047737618086, 1243.5097002550071, 1250.720998053779, 1258.4996067174204, 1267.245724939541, 1277.1888835280133, 1285.4416200997098, 1291.5788772043957, 1296.222566148796, 1299.7222476374143, 1301.6188600545074, 1303.3561345916385, 1305.213439025402, 1306.5271101010653, 1306.1564510275605, 1304.2656903132984, 1300.8548279582792, 1295.923863962503, 1290.8001248354356, 1286.7163220456985, 1284.1068385069684, 1283.0433323857476, 1283.80840881913, 1285.7384045523827, 1288.3917442034476, 1291.2305418213116, 1295.1429842066539, 1299.9039752904835, 1305.9026701085136, 1314.0765375309322, 1324.4601149800076, 1335.1815602890445, 1346.0005884599345, 1356.1635012514103, 1363.9756500768356, 1369.9143585378524, 1374.772117899804, 1378.7776609256882, 1382.361193343476, 1386.0996057158768, 1389.7388196373163, 1394.138792818394, 1399.0502733179346, 1404.3173242609446, 1410.1202348011643, 1415.2693831050606, 1418.2838656197762, 1419.366977453495, 1418.608992958793, 1416.025109038685, 1414.9213535905444, 1418.8406854439295, 1429.0780459049813, 1445.6334349736997, 1468.4764588440546, 1493.8642573451948, 1515.9957919588178, 1531.3416235293919, 1539.247035585347, 1539.3885039175855, 1532.882548894552, 1522.8466436219485, 1512.4548424924183, 1503.0165784490996, 1495.2092937162192, 1490.372940436366, 1489.4160484887223, 1491.977292197035, 1497.8869238191996, 1507.9914671402942, 1521.6043955831824, 1537.2405172519898, 1553.7433704127222, 1570.4970455918897, 1585.1310529951093, 1596.812196109606, 1605.7628041205176, 1612.5611078948411, 1616.984020191161, 1620.216785906668, 1622.6620302861877, 1623.987745753358, 1624.1939323081788, 1623.6346179360905, 1621.8974692695344, 1619.2386011991755, 1616.5355340787696, 1613.788267908317, 1610.9968026878173, 1608.9858051523886, 1607.7075710666536, 1607.4081526227328, 1610.610811596297, 1617.4370342785182, 1628.721413281505, 1645.5288699133066, 1668.2373701685506, 1691.8003904958955, 1716.145904640008, 1738.7800606415524, 1757.9951816221103, 1771.3668142569556, 1781.4182203217574, 1787.9289934536675, 1792.1460596323534, 1794.2590832032106, 1795.7583777178486, 1795.1956091232344, 1792.5723924709696, 1787.787148427376, 1779.7875183555323, 1765.0466907927978, 1740.126722492468, 1700.9834110672605, 1641.891850577112, 1559.30652299141, 1455.6181614421055, 1331.2955516530121, 1189.0392643029977, 1037.3370426649108, 882.338077745695, 729.8413376693691, 595.0164169405496, 485.2721064649816, 401.0685465686694, 346.72447479408464, 317.5770013478981, 302.8740233014164, 296.71971273271737, 296.1380795998963, 295.7018547502804, 296.4243040494838, 299.90812597269326, 309.0259527775755, 324.8178820485174, 347.7014751934454, 375.6502004811314, 405.45866096120216, 431.38159211832397, 450.8176230922071, 462.79257666331677, 468.31971869726704, 467.7095280301704, 463.8346369196938, 458.3412694836408, 451.925095937301, 444.8008555214804, 439.9268446269113, 438.62257554787374, 440.65768915926185, 445.9927235955658, 454.1982003751731, 463.23418563384183, 470.4616547830119, 475.8806078226835, 479.3078181532188, 481.1846098105946, 482.9769222613539, 486.0042677997766, 490.31113422853275, 496.5207342887825, 504.17989839018446, 511.88614815976655, 519.7390994852424, 529.9269011697154, 541.6549088901471, 555.2439486204405, 570.9842384235268, 589.039435596842, 605.1808448282421, 619.4313844023335, 631.602571961651, 640.9530770834128, 646.856737509606, 651.0285462445279, 653.6323264926012, 654.6614812039788, 654.6340386061867, 653.8071984547234, 652.4620264516625, 650.4416583048294, 647.9477704713834, 645.4673232540002, 643.2116952874092, 641.1808865716099, 639.8252739995826, 639.1213671750479, 638.825685946668, 638.8325409970782, 639.1419323262787, 639.5286714877793, 639.9646811291016, 640.7351884088547, 641.8953410097487, 644.6902822963023, 650.968340649637, 661.2875559553473, 675.077473896215, 692.2277991068204, 710.2482448581255, 725.3619561510438, 735.4490789229557, 740.7269998832572, 741.2508667146585, 738.2658227816781, 733.7805929391252, 730.0863359038686, 728.6106148210822, 730.1550164504245, 734.7195407918955, 742.2239896086512, 751.8897478810266, 760.2636239066535, 764.8963923194254, 764.9144632209369, 760.3178366111879, 751.1065124901783, 740.1755466239733, 731.698417229133, 729.5076291699819, 735.7216858495958, 752.6190616345123, 779.4925216511377, 812.1364385260382, 845.5065722257816, 875.3659159442162, 897.063169062202, 909.7179936571674, 914.8893483733698, 914.2200859092775, 909.6668626934995, 902.5720594627704, 894.4132964040388, 885.4322317554343, 876.5138102550467, 867.9817258518093, 862.1777363852805, 858.6031304992484, 857.1370790746485, 857.7533080410851, 860.2980994068233, 863.5806372784406, 868.22545983766, 875.4172770849306, 885.3201948469945, 897.9179551583877, 912.7830502368188, 928.0876353361491, 940.7977989581905, 949.2508642879336, 952.5835467603135, 950.7120036324268, 944.646618674458, 936.9010848814378, 928.6436452085605, 922.3936285819038, 919.2905550930294, 919.3344247419375, 921.8607460313367, 926.7836486956505, 931.6568529609482, 935.3738809288892, 937.9347325994738, 939.3394079727019, 939.5879070485737, 941.753426033583, 947.9709112279012, 958.2403626315281, 972.561780244464, 990.9351640667086, 1008.832713041741, 1022.7001583726317, 1032.5375000593804, 1038.9325800382594, 1042.1797728836787, 1044.8115812708272, 1048.9650780252348, 1055.4225964493685, 1063.008452670683, 1071.1801496743956, 1079.4486329878746, 1087.753439808576, 1094.7051353288066, 1100.8915614848381, 1106.514588583009, 1111.9795624562844, 1116.3275934285593, 1120.3567347696178, 1124.2293775019425, 1127.9917737595692, 1131.1822615764097, 1134.5282161922578, 1137.5431451001662, 1139.902266255169, 1141.6055796572662, 1142.9189224826994, 1143.70637235505, 1143.7571151632403, 1143.0372713591582, 1141.546840942805, 1139.28582391418, 1136.3221814614922, 1133.9695539452255, 1132.6204825065681, 1132.8073529721182, 1134.5346751707893, 1138.1633221835466, 1143.3702074771932, 1151.1771113130822, 1162.5094592440703, 1177.3582316123297, 1195.0016822559319, 1214.0998397542858, 1231.8358078643662, 1244.05834916257, 1250.7719734778107, 1252.19510521679, 1249.1486918277446, 1243.6123733008862, 1238.8143400087706, 1234.7545919513982, 1231.7180264772946, 1229.774293583529, 1228.542598690969, 1227.8860045217127, 1228.6856838106828, 1231.9142861543469, 1238.6335568630482, 1250.7287140537949, 1267.6943614941767, 1288.8708413406189, 1312.7247424726029, 1339.131791416313, 1365.2737689565106, 1391.1364509161842, 1415.395634777715, 1437.7728479699467, 1453.9000820029785, 1462.825492816834, 1464.670873770429, 1460.5389124900344, 1451.1263943965932, 1441.5840183741225, 1436.3509297153116, 1435.734897058846, 1440.2000938399729, 1449.746520058691, 1462.1910457675474, 1473.3944133143534, 1482.741085421736, 1489.6454535154649, 1494.3167482043548, 1498.111551081705, 1502.5781051878741, 1508.0241791615488, 1514.2284698454514, 1520.4798794692747, 1523.3283151800165, 1522.8104914550236, 1518.926408294296, 1512.018803994095, 1503.4348228121876, 1497.684713120028, 1494.7684749176162, 1494.940897455142, 1498.3246217329663, 1503.6976319116798, 1508.6759543072944, 1513.318260410146, 1517.1149717198548, 1520.8728162413886, 1525.1781933392278, 1531.1537662972173, 1539.0757901318168, 1549.1990540932152, 1559.542179170396, 1569.509942320204, 1577.6477755559793, 1582.8482688775107, 1583.9343130925558, 1580.739593111003, 1573.3063997418978, 1562.3620169785706, 1549.0578138107248, 1535.748008622845, 1524.8692516065323, 1518.1928461016837, 1517.4869149826536, 1522.8182365819544, 1533.4009223936314, 1548.0932327713726, 1564.571539108711, 1579.6537161742638, 1593.410957452969, 1606.0939905104465, 1617.8887677292953, 1630.1408384177462, 1644.0115257382192, 1659.4639421606528, 1675.0998376260468, 1690.3175510207416, 1704.0280142518955, 1716.3824651914013, 1725.2939127580746, 1731.2417916278985, 1734.5652250824471, 1735.9033287856614, 1735.886105292418, 1736.3962001561765, 1737.741303586498, 1740.0871959615233, 1744.4611414564438, 1749.9701831559553, 1756.227602919461, 1763.171754590053, 1770.4710774114487, 1775.9068774842501, 1779.2263521815999, 1773.4088415879814, 1755.1722547687, 1715.7257818069315, 1647.6187904716494, 1544.6689682754884, 1415.3557144939043, 1260.14699504473, 1091.043350474088, 922.6464823903376, 768.3120066422301, 632.4005474362142, 523.914914976072, 445.7237990544562, 394.71414786935577, 365.0262948763673, 351.10364768000534, 348.62974310299387, 351.69194110152347, 357.44365382447705, 365.18844276487664, 374.308130199544, 380.55464123605066, 383.7550998543003, 383.3718035029229, 379.1007881462725, 371.79921601506527, 364.43904010058856, 357.3660124430353, 351.95950218079116, 349.91105319763847, 353.15075323545045, 361.3311381951473, 374.84428150254524, 392.9064775716532, 413.3749313320268, 431.9833033336707, 449.6403342194546, 465.52310319477147, 479.6506527097939, 492.513431164885, 505.7100363841124, 518.0051632156546, 530.0347262086507, 541.5057994066632, 552.6854359744037, 564.0696834303019, 574.5316784255153, 583.0085564619366, 589.4932340621774, 593.7402818132199, 595.0505014638286, 593.986738856419, 591.0551861276101, 586.4788492600297, 580.6780942831781, 574.575983883939, 569.6967853921749, 566.8492635978039, 566.0334185008255, 567.0720271419916, 569.5921696574845, 572.180328436462, 573.7551613822218, 574.3166684947643, 573.864849774089, 572.3997052201961, 572.2676420855236, 574.8656797146632, 580.4609731416563, 589.2912572945285, 602.3145606983663, 616.6020724125885, 630.902627028116, 644.6819144768662, 657.4644649027878, 667.5938775332081, 675.9579007924867, 682.2217268684929, 686.5397874553581, 689.1498174811082, 690.4905329158285, 690.8186083886916, 690.7743042441444, 690.5830671620063, 690.2448971422771, 690.1073475075685, 690.3179795108643, 691.6598019399862, 694.0200914550245, 697.4166938021061, 702.0272528857713, 708.1912608369637, 714.5827922112564, 722.1326378808254, 732.3104344658498, 745.1448008298614, 759.5140689520206, 775.7210633577164, 791.9042023025975, 805.0706753079248, 814.2787247525705, 820.1629652734466, 823.4810581885279, 825.6973044938808, 828.317032301938, 832.0135913093658, 837.1004806485471, 842.8314357836313, 849.2740771290122, 856.5036688243456, 864.8919884349322, 873.9531930007845, 883.7120479042339, 892.4327819448217, 902.413002427386, 913.336162474674, 925.2334506095128, 938.2238529277565, 953.4420100915798, 966.0669150723392, 975.5133213737503, 981.7732276841334, 984.6669340047732, 984.1944403356695, 982.4777536231471, 979.9401335971809, 976.8798934906861, 973.8263210429066, 970.7794162538424, 968.472440588683, 968.4601693096158, 970.9138024022493, 977.0418893542059, 987.1956871200267, 1000.8870586828082, 1015.2494660867935, 1029.6421961278495, 1041.1188620914884, 1049.778400743329, 1059.8308045201316, 1072.9690064688027, 1090.0799642177003, 1113.2335183754835, 1141.2540846068282, 1165.883686417294, 1186.5219914792838, 1202.0357977402152, 1211.2318106972573, 1214.6833008464987, 1215.988996481287, 1213.37438473633, 1207.0021486517153, 1196.9497057565768, 1182.973654282443, 1166.134514137058, 1150.5816433786017, 1137.122877979483, 1127.681122939818, 1123.7974872315394, 1125.4817094964233, 1131.4621313455434, 1143.967821456316, 1161.0448558023581, 1180.970798255064, 1202.6656516231374, 1224.5233746262525, 1241.643720527109, 1254.2830483417813, 1262.921051021408, 1268.1479720989907, 1270.7668322146922, 1272.6791648522496, 1275.1140482294134, 1278.1686384392638, 1281.754160506693, 1285.8706144317002, 1290.8837268042275, 1294.8828135791932, 1298.4637091822337, 1301.653228751507, 1304.451372287014, 1306.8581397887538, 1309.8490268165326, 1312.1015679848574, 1313.135874559292, 1312.9519465398362, 1311.5497839264906, 1308.9293867192548, 1306.0836635030275, 1303.995821545875, 1302.9037103190506, 1303.8099281681828, 1307.1874835129788, 1312.663760436472, 1319.961980194542, 1329.492550774576, 1340.4094740701955, 1352.6057547417415, 1366.9260203654546, 1384.4100020952797, 1403.5233355426833, 1422.9502218835376, 1441.4856265380367, 1457.1553059855955, 1467.235323248587, 1472.611785256904, 1474.4438905954248, 1475.0306809654412, 1477.3851033655144, 1482.976538729415, 1491.804987057143, 1503.8704483486981, 1515.356822652298, 1523.0945010256835, 1527.083483468855, 1527.296191775223, 1523.676532973459, 1519.3858618197828, 1515.6810899944307, 1512.5622174974021, 1510.0429827409544, 1508.235571667743, 1506.4254938145364, 1505.00972175245, 1504.3193271281857, 1504.7961734061612, 1506.3994759801894, 1510.110673239224, 1515.627079110956, 1522.4870698333275, 1530.238174145918, 1539.4970405818322, 1548.17948434387, 1556.6571302326392, 1566.0805776950972, 1577.2394353611717, 1588.8544810692326, 1601.8774156185987, 1615.6478254098997, 1626.4949341198515, 1632.5262615904376, 1634.3890730850435, 1632.143629669857, 1627.9296491265172, 1626.346704093076, 1628.757649241325, 1636.0711877047809, 1648.602191070408, 1662.7779949195851, 1676.1310740883243, 1688.2310226400648, 1697.2604343077753, 1702.5895659175253, 1705.9352133606958, 1709.7332040333863, 1714.7082567338214, 1721.7690745955165, 1731.2305292054366, 1743.1389781088685, 1750.5489174226757, 1744.5621145200323, 1721.5350189339465, 1677.6508984257132, 1610.8153522054213, 1523.1284640548242, 1425.6141456549517, 1323.97926125326, 1217.7466597380492, 1108.3584978549875, 1001.6075651210983, 897.6724823238332, 792.420503122903, 693.6457471955272, 602.3100891903939, 515.3111240121732, 431.2694240909981, 355.90409123304414, 290.32526837379083, 236.6603391385052, 198.56005418457744, 176.55109968068732, 167.13036468204518, 167.39244124430115, 175.7102620343681, 188.74358374975517, 203.58254957912902, 221.18090948443995, 241.3718603626484, 263.40573635543836, 287.2939427565924, 314.14233280139314, 341.71048695380034, 365.2170974660593, 383.731655364929, 397.69916320011686, 406.80293775874117, 412.0763350207858, 416.7624945191004, 421.7316441886178, 425.56699852981603, 428.1702314442622, 429.5747601665403, 428.5511011820669, 424.58496358613445, 418.7662017850951, 411.3324218108122, 401.8019483846024, 392.3168420513707, 383.90568462053204, 376.36638178807846, 369.69893355401, 365.08670464957487, 361.9340245287136, 361.0141499893557, 362.4383303468237, 367.7324476100594, 377.31258759222845, 390.9996995262674, 406.77307258909303, 424.4988021927232, 441.4571404318185, 455.78135119566696, 467.29665497297265, 476.8173906125301, 484.2776193443366, 491.067096553142, 498.1305154901894, 505.54950720014085, 514.172295416025, 525.5277478506862, 539.2244091478599, 554.9155607234636, 572.4700348376629, 589.737676226774, 603.8379375504339, 614.7846658669937, 622.6819769948877, 627.5571527389887, 630.5876636669922, 633.2137834490793, 635.4666099713862, 637.4750558338694, 639.2895955383161, 640.6525955843405, 641.5640559719425, 642.1745976245468, 640.7111848947814, 636.6027536916753, 629.8381980094482, 620.4175178481001, 608.3407132076312, 597.1277428349581, 588.0205450961515, 582.3569369105254, 580.7603493546496, 583.964201133748, 591.2056291126366, 604.3490489795711, 621.7849791949714, 642.2665576056977, 664.326946801302, 685.980618487845, 701.3016974783019, 710.1612987770575, 713.1828534606818, 711.0997802343993, 704.9048432451791, 699.5948187759702, 696.7925148395344, 696.5655526574512, 698.9791095973812, 704.042306704139, 709.3591762136181, 714.3687855278285, 719.0893385638013, 723.3904805862146, 727.2746364509541, 731.540287045457, 735.1610928773346, 737.8977824477873, 739.8155331244765, 740.8821320611875, 741.4436504246477, 742.3253590799513, 744.2950060380481, 748.1114712017334, 754.8783919257346, 765.8180257609681, 780.6339078457956, 798.1466840357031, 816.8385945251005, 834.8964020195092, 848.837377916914, 858.8097546481338, 865.3293355499139, 869.3963608718408, 871.5909919808895, 874.1863913113715, 877.2619946590819, 880.9115404833944, 885.8214899011116, 892.1847533732522, 899.2838057999364, 906.9597755895733, 915.1236841374321, 921.6785281695329, 926.0975476796539, 928.7395052177355, 929.6838365795726, 928.9750310675308, 927.782270491996, 927.3049052455132, 927.9678355163942, 930.8473492899659, 935.9434465662283, 943.2884111927244, 952.2813654213705, 962.0835634246301, 970.5638886203554, 977.7223410085469, 983.9571972299476, 988.7347207977434, 992.4577028020728, 996.1595124512496, 999.8401497452736, 1002.6062098600298, 1004.3300382557991, 1005.0226894816684, 1004.7056229261441, 1003.3788385892263, 1003.220808902006, 1005.926879011958, 1012.1221945934951, 1021.9773814479638, 1035.4924395753642, 1049.2361127851716, 1061.286189230063, 1070.3585612977452, 1075.7784443791638, 1077.4969636418307, 1077.191549287151, 1075.635941460538, 1073.522918363339, 1071.6901718096235, 1070.235451464367, 1069.2351531152842, 1069.3447266056928, 1070.7470380868228, 1073.2797974648624, 1078.0933115577254, 1085.3366152441408, 1093.7804647281748, 1103.3273412958183, 1113.9584355738082, 1123.5208034200864, 1131.7496268361313, 1139.1370438838571, 1145.22804474719, 1149.5465194350083, 1152.7454035228045, 1153.9562804545844, 1153.2608061210487, 1150.9948266375836, 1147.482018456025, 1143.623615659129, 1141.205114542575, 1140.3806226554211, 1141.9160387789123, 1147.1976532514268, 1155.108022335617, 1165.346008361287, 1178.4063566262716, 1192.7871774121008, 1206.1926496079025, 1219.7824613359767, 1232.8473409402916, 1245.5850359604594, 1259.093638721513, 1273.8059204300655, 1288.4086151445954, 1302.7020553680484, 1314.1900799314622, 1322.118578215198, 1327.5194474134637, 1331.0633019162335, 1333.2600575820954, 1337.0761385511607, 1343.0120369104764, 1351.4776980637423, 1362.4866460220758, 1375.968342583398, 1390.0863602916415, 1404.6237334363734, 1417.131072772881, 1427.58133027893, 1435.6975817115206, 1441.2020919098231, 1444.2033437290538, 1446.1497289342321, 1447.1839847430851, 1447.51424005164, 1447.9377941191378, 1448.5853801560916, 1450.2104255257607, 1452.5545038149244, 1457.167699673158, 1464.839276917158, 1476.3487261807593, 1492.36841660985, 1515.822785487035, 1545.8768314135152, 1581.6755466926213, 1622.7758977575993, 1665.6806075698973, 1704.7992279775258, 1737.1515078293548, 1761.3468095487842, 1775.0413731660312, 1778.7528790159668, 1774.8821510248788, 1764.5034762671671, 1749.5993497804473, 1731.2421706570974, 1711.9837717224698, 1692.4841763243924, 1675.125146114773, 1659.4316871921214, 1647.3116026841192, 1638.0401819777687, 1630.057900174125, 1615.9141950384205, 1585.8049862084324, 1530.0150747251405, 1444.7231739895378, 1332.3455674724082, 1203.363203269882, 1075.2369114990158, 962.3783536297957, 867.8066258540559, 790.4060954543769, 727.3018377938, 670.1811499674726, 614.732958140782, 562.565503412892, 509.6793337905152, 452.912642279175, 395.6694793544102, 337.5804791258164, 280.6892070463622, 233.27461682744556, 199.65971102844335, 177.0516086172309, 164.60240958592163, 160.3567033538602, 160.1195813255123, 161.51007930997858, 164.7622454572217, 171.048102477186, 181.14678831192438, 195.02131263907648, 214.95286020824133, 240.73164106843373, 270.2178286086388, 300.39679729024294, 331.9858789570909, 361.4906770903057, 390.92435970626093, 420.4874201766976, 450.32985599105615, 479.13230808354047, 506.5888408327166, 528.4059447497408, 545.1467841536848, 557.1889920185222, 564.7105413581642, 569.0488868519205, 572.3887826177556, 574.0587811768437, 573.715096443021, 570.3601727473645, 564.0491293769177, 554.8469595142631, 543.6415566326747, 532.272980901859, 524.4923653979325, 520.2997101208957, 519.6950150707482, 522.3942251573653, 527.4773102958937, 531.5437784067165, 534.6549708410806, 536.8108875989865, 538.0277371268517, 538.4614134364614, 539.1981481014543, 540.2421077124371, 541.9750948145684, 544.4567867384756, 548.1233778238426, 553.026981996633, 560.5600913681712, 571.8807081605605, 587.340703197816, 606.1798949049596, 628.5297062937584, 651.8658211313133, 673.7822843575068, 694.3001664797251, 713.0076715817311, 729.2071536783693, 743.4463957828026, 755.2999220336333, 762.7465652726854, 765.6404554968391, 762.5560836457713, 753.5036402017934, 739.4698283888008, 722.6306503732836, 704.4376244409744, 688.6307577155022, 676.7678276304043, 670.2825719822196, 669.2809375913195, 673.1226446237853, 680.86474056771, 690.9384748513213, 701.6422370124426, 711.6584916224058, 721.3095382593106, 730.5987256363757, 740.9057170868566, 752.9815289571704, 767.5357056782747, 783.8993552559068, 799.5224756552701, 813.0074752728435, 823.9829854026406, 832.7048458593785, 839.2020246614932, 844.8562755392913, 849.8599629249329, 854.4062347739825, 857.39648115456, 859.0628471554444, 859.6431813179995, 859.4823633028133, 858.5803931098851, 857.9873624761909, 857.9562444930905, 858.5394786536912, 859.7611149524571, 862.6654834395222, 868.1806615402171, 877.2089145101506, 889.6531014202889, 905.6513434299993, 923.9783243381044, 942.0314646548273, 957.3851117954289, 970.1283297162618, 980.1335842864295, 986.8392159331831, 991.6107871617353, 996.7901163804625, 1002.2807404913326, 1007.8271643001808, 1014.0513353555004, 1020.9532536572918, 1025.0942853380832, 1026.0296507081773, 1023.9802079388049, 1019.066655205264, 1011.2889925075549, 1004.1724778682416, 1000.3901560843727, 1001.5791227346639, 1007.8572850218477, 1019.2246429459246, 1034.9642386258731, 1051.54026999048, 1065.79801275652, 1077.6012952636834, 1087.497176462369, 1094.7286427909803, 1100.4383015207375, 1106.1256305669976, 1112.2336599061703, 1117.9296130156647, 1123.4043056846394, 1128.9881594171359, 1134.5980089172556, 1139.537225758826, 1143.9342818810896, 1148.0592237707122, 1151.952134097136, 1156.8987247335056, 1162.9183132305104, 1170.0640494758438, 1178.046095151693, 1186.3414947670667, 1192.5814567457778, 1196.859726880546, 1199.169959844539, 1199.4720293894857, 1197.8064230275052, 1195.3575365466909, 1192.4429690749034, 1189.6125968319575, 1187.2365106322086, 1186.5685659515204, 1187.7710475132824, 1190.6851557535642, 1194.5424614490041, 1199.2721853533217, 1203.8835464188644, 1208.061838679282, 1212.3186753098655, 1217.1638972255655, 1222.8283448492336, 1228.9799676252032, 1236.7351130252189, 1245.9878516803565, 1256.7201554060089, 1268.1275448737797, 1280.4760128126898, 1291.8278432847412, 1300.86005550191, 1306.1843093168523, 1308.171341576655, 1306.824247181409, 1303.1168158403275, 1297.9663445350684, 1293.4802107568826, 1289.964568874872, 1289.390497918491, 1292.915388193252, 1301.310286796523, 1314.1425862562405, 1331.0658855268825, 1348.5246742714837, 1364.763914845717, 1378.5579382096441, 1389.00824731016, 1396.2437247212467, 1401.9776843243956, 1407.288747999589, 1413.2893052123673, 1421.0566813230766, 1430.5908763317168, 1441.891890238288, 1453.4380309937544, 1463.5973383973826, 1471.1067841840736, 1475.9663683538283, 1478.025916288006, 1478.3261454944743, 1477.8412486509974, 1477.2130517863914, 1476.441554900657, 1475.7575422914683, 1475.161013958825, 1474.6519699027272, 1474.2235355256635, 1473.8757108276338, 1474.1645950901695, 1475.3431223187465, 1479.0349912025263, 1486.3651287916532, 1499.2042430829565, 1517.1978454263465, 1539.840067810871, 1564.2720066715478, 1588.8084984158231, 1609.7578539846668, 1626.125184606835, 1638.1634242878038, 1646.9938708935297, 1653.6539111308623, 1660.7325339612312, 1668.8057119747386, 1677.8734451713856, 1687.775053757559, 1696.897104459798, 1703.653854545016, 1706.8317663988887, 1703.8613132141188, 1694.3060239209317, 1679.0300950254314, 1658.1597010008602, 1633.573804424741, 1609.0900126135632, 1585.9235361416297, 1564.7129906516202, 1546.841496864766, 1532.6018300809912, 1522.067356089211, 1518.991275633966, 1523.0070596754947, 1533.423147853181, 1549.4180357745677, 1569.6702771415487, 1582.7583394978387, 1588.200178164961, 1579.542677038489, 1555.843334960021, 1513.2104043399236, 1445.623812327026, 1338.4188937591684, 1198.5365155439777, 1023.023788938616, 819.1685550239193, 616.453048260905, 445.5201035476369, 310.71819210728916, 220.78059490074506, 171.63140992063057, 146.97265456779004, 131.44083324648557, 121.32886720519214, 113.00113488077002, 110.9506177938935, 116.09225042530811, 129.13196766315951, 148.9782706910897, 174.12535029558188, 200.01215706968338, 224.89621778025602, 248.6349415832172, 271.8599692948826, 294.29493009964165, 314.79786641248774, 334.10892125746085, 349.99758327322303, 359.62804923603073, 362.6565587728445, 360.97294587111986, 354.66460625920956, 345.2660777926998, 336.9844346497553, 331.4231561078594, 329.6410320544804, 334.3604743854072, 346.5399822714647, 364.5243757855591, 388.27476118846414, 416.45710634482447, 444.14698278194635, 469.4273921581804, 490.94661385566326, 507.34786189618734, 518.1228308880585, 524.8310291497064, 528.1344765800798, 527.7818708538074, 524.4890833165039, 519.0396616077508, 512.1982269256129, 504.5577378138429, 498.5534875546382, 494.46195724409046, 492.2831468821999, 491.94995750712803, 493.16590984699917, 494.6583432829617, 495.8742956228327, 496.8687808443645, 497.8097105740414, 498.69708481186353, 499.7076859126217, 501.1273572827364, 503.22324763290425, 505.4944917768994, 508.29178248931447, 512.5033519757285, 518.7227230051699, 526.3886620959668, 537.0641362305608, 551.6700772489801, 568.8427233989809, 587.3899923144681, 607.9718953006757, 628.0183775566977, 644.6354970787011, 657.4540212240337, 667.0400971764846, 672.3712838859071, 674.7326087527538, 675.7463891662379, 676.135618938637, 676.0418337089459, 676.1089881684481, 676.3370823171439, 677.8146864900363, 681.8228692606975, 688.1447342925552, 697.2944572555633, 710.1774562470367, 724.8965334075752, 739.1970081472581, 753.4488725837998, 767.4430221426446, 779.4729640294267, 790.381461153395, 800.9143316725009, 811.1301959821933, 819.8116157514476, 827.6914083044106, 834.4213600607707, 840.1496021800991, 843.9240751844145, 846.1218503857624, 846.4070767584046, 844.8361101314607, 841.2197417385571, 835.7660982586633, 826.3691989331716, 813.446825680701, 794.7138588914565, 769.8322162052643, 739.5125858259082, 708.8366756824107, 676.9326270416348, 646.5840950495551, 617.1649611725608, 587.1808512219147, 553.4441901760496, 517.4206802319904, 481.2456117394886, 448.2609547070841, 420.97831724639974, 401.3816155109341, 389.651997250468, 384.37174963812356, 383.64551478217584, 386.1591016423302, 390.63328368667436, 396.7161098134908, 402.67444629039977, 409.1217113084011, 416.0579048674946, 425.5696907109012, 436.71236786600895, 450.2606793724851, 464.3471695748357, 478.97183847306076, 487.7803791332922, 490.14192538683153, 482.15535387522425, 464.45495023384393, 437.04071446269046, 406.3612691998374, 374.5819475197594, 350.7582648118566, 338.7376547119376, 339.39328724779773, 351.6189497120418, 374.92984251688506, 403.5084247112489, 431.5214176211286, 457.5148982549338, 479.33933313260275, 495.9113622995199, 507.30401492345993, 515.2663957301636, 518.9982385991592, 519.5084295279967, 515.0215730361972, 505.74528580227815, 492.32252020998476, 477.2228973638495, 460.57817874883375, 447.02251528051073, 437.32091310656153, 431.47337222698604, 428.6110006759708, 428.6679177110349, 429.3270478743918, 431.4287854855218, 436.8974979584926, 448.1188347369624, 465.6828737178845, 489.58961490125876, 517.3932635004437, 545.2450846873038, 566.4087797705927, 577.4193406333156, 576.78729806546, 565.6233560723139, 544.4872210772512, 518.8157019139819, 493.7685811256338, 472.32479713223364, 454.7087367098467, 443.6497218398602, 438.2765143387949, 436.40595309082903, 438.39064594678894, 444.149528945762, 452.4812539385049, 463.17990977961546, 476.1795900633256, 487.8361693942182, 498.6603529407774, 508.4890272427242, 516.7974542918, 523.5029972769471, 530.5621028569083, 538.8509551006484, 549.7248098668094, 563.1123507797372, 578.9698058497902, 596.9483497023742, 611.4418851087365, 619.502875068428, 621.3749378593674, 615.9746928436354, 601.7104187555057, 584.751355984415, 566.7627005891994, 547.9918841584619, 530.7869159664456, 518.7200926454599, 511.039468395486, 508.75537525471447, 512.2506809765566, 520.3513809238909, 531.2713267805627, 542.7680205477442, 552.674569552654, 559.3203963894991, 562.7055010582793, 562.8298835589948, 560.7068258652395, 558.6826322194458, 558.8247059157615, 561.4328944713641, 566.6541738874463, 574.7218807817793, 583.1102993422794, 590.6700210071791, 596.8013507421222, 601.2588243103949, 603.7324123898455, 605.4849728865156, 606.9005575917444, 609.493755546994, 614.7341862609026, 622.6414142203839, 633.3796990834028, 647.3074897907393, 664.4819146460493, 682.1642952212941, 700.9451629413813, 720.5736315699739, 740.781984255912, 757.8117398190313, 772.9783735303878, 785.1155632791771, 793.141718818024, 794.7793609039251, 792.0766925445574, 785.0857856237873, 774.4395590449784, 762.3968133800332, 752.1915985771236, 746.184830438662, 745.0218511320334, 749.1848912318704, 758.0021384923168, 770.5525779515034, 783.8079022864663, 797.214131425582, 809.8068042195857, 820.7483776678795, 829.5101707733471, 835.9007732515414, 838.3420314646582, 837.3161759873299, 833.2419783198557, 826.3837789607935, 820.0390084609726, 816.543124623511, 816.2939757210245, 820.1127860935164, 828.1025085929375, 837.1816133180457, 846.4593997293099, 855.2269583675618, 862.3817830708082, 867.718946575794, 873.0979098042978, 878.6187636383271, 884.5057821783709, 890.5003047284041, 896.7547445520253, 901.7127988055929, 905.2018095782796, 907.308563956149, 908.5730044572153, 908.8932749362464, 909.4584673587844, 910.4527123070714, 911.9837842727214, 914.3502328975994, 917.6034754746443, 920.9216309164128, 924.1656406729728, 926.9607682689415, 928.7099144205895, 929.4130791279171, 929.4812029346452, 928.9838151157406, 928.4397559111551, 928.3808013718788, 930.7513502629579, 937.7444455519359, 949.3600872388124, 965.7304453236569, 986.3890669882182, 1007.4471547024044, 1024.5186225311288, 1036.8927825364622, 1043.7088474418676, 1045.2000436564715, 1043.310769945319, 1040.234069275954, 1037.8314742275907, 1037.2759444771318, 1038.5674800245781, 1041.706080869929, 1046.6917470131846, 1052.1457503221427, 1056.7226976366178, 1060.6352881501014, 1063.883521862594, 1067.1184702670746, 1070.4112458407812, 1074.6149591197816, 1079.766789366105, 1086.8006417703764, 1095.460727561923, 1106.8010474135547, 1121.3016669010108, 1138.6384622319424, 1157.6842489152398, 1178.0234584611185, 1198.018882267624, 1216.085948767488, 1231.910570985348, 1244.9454023315504, 1254.1846601257682, 1260.1268813419215, 1263.638212104269, 1264.603362121286, 1262.5508726644134, 1258.5068121814568, 1252.7413116951573, 1245.2887930402828, 1238.2344795648478, 1234.002540506248, 1232.592975864483, 1233.9770965857158, 1238.1549026699468, 1241.08536611819, 1241.4547435221382, 1239.263034881791, 1234.5102401971487, 1227.1963594682109, 1221.1622690637626, 1216.9707491124227, 1214.7238124264823, 1215.1636252688966, 1219.1461922089022, 1228.0945558879296, 1242.5382143836632, 1262.8881629589378, 1288.140798815712, 1317.71722861888, 1346.0084688573827, 1370.7011544336397, 1390.667256385112, 1405.7428146916068, 1414.6725338651338, 1420.2609056612228, 1424.3437990640307, 1427.5362349609718, 1430.2082767347322, 1433.183177089663, 1436.4609360257648, 1438.8340561838183, 1440.0171674235662, 1440.0656029176012, 1439.1342942154304, 1437.2232413170548, 1435.1807327458914, 1433.5775087824545, 1433.0711673782873, 1433.9523012298052, 1436.6484283513942, 1441.0056342755845, 1446.7385488621192, 1452.5319762079125, 1455.8037306055874, 1453.070668144796, 1443.4597417187317, 1426.9709513273935, 1404.1664981109861, 1379.1998496794808, 1357.6513045851993, 1341.266957041757, 1330.0468070491545, 1324.1816482300687, 1321.9487317010255, 1321.0302552560952, 1321.8655934092035, 1326.0853395081085, 1333.6753005652583, 1344.7745229229274, 1359.2223536931137, 1374.7695234208654, 1388.154845410666, 1400.6015929387854, 1412.8072007222513, 1425.433882514637, 1439.2090654999743, 1455.768719336868, 1473.529850959278, 1490.2630747597166, 1504.6831474114454, 1518.2417482836652, 1531.1212182465813, 1542.4929083083193, 1553.6105973015285, 1568.257188186048, 1586.5770408110614, 1609.2827528431756, 1637.481489240249, 1671.1732500022813, 1703.664581138161, 1730.7738336561451, 1749.7147543854303, 1759.737538244512, 1753.465137762855, 1733.6775595124072, 1696.3689698269675, 1640.3637493002466, 1559.607254480617, 1461.538569070061, 1341.5368145964417, 1210.54914547831, 1072.7891105759052, 940.604342620312, 817.8911895065005, 715.5413530229023, 630.4002168358431, 563.013296830664, 509.8484782260904, 470.3040040083791, 439.288531897326, 417.2248841939634, 403.42014344822735, 395.8632402434381, 391.1975474669227, 388.0677266326216, 385.30768491400784, 381.91283274500154, 377.88317012560265, 373.21869705581116, 369.80065735499863, 367.6684984534139, 366.94555768164673, 367.6318350396972, 369.7273305275652, 372.11919826659977, 374.2317364570271, 376.86635378215306, 380.64203253802503, 385.5587727246429, 391.7313903213169, 400.04509314336485, 409.3486437094622, 418.40407742751404, 427.21139429752054, 436.18402463818654, 445.3308680191741, 454.6513832141759, 464.80966667725863, 475.9497122965507, 487.51863939690884, 497.7735124536093, 506.4406812891765, 513.429917587572, 519.0598131902987, 523.1958384915279, 526.9077381699984, 531.0154748950416, 536.1237811368153, 542.10644880718, 550.0160100343104, 561.3220144954709, 576.4020939255797, 594.7624973311101, 615.1238898113195, 635.9291670343915, 654.3730983675828, 669.0829027283196, 679.3672271672391, 686.2467588816878, 691.0550985694457, 696.2801951249266, 704.1067797888472, 716.1986244165771, 732.9804216767769, 755.0823065054531, 780.3336117465208, 805.282415923886, 827.852146588312, 846.9602285881895, 859.5244370008058, 866.2063342804025, 868.6666450317282, 865.7319627428496, 857.8793572491201, 847.1626794604608, 834.4294716244733, 820.2444041014862, 809.967528185485, 803.8050092954684, 801.6000379763506, 802.9288431043303, 806.8222830617159, 808.9158871835069, 809.0620989170899, 806.5563332348556, 800.2644379054152, 790.4985845758282, 778.7543316403445, 765.1206267851925, 751.0066400655903, 739.3221154319193, 732.6027654745235, 730.8283748465872, 734.0416895833623, 742.252364031798, 753.1526799481597, 763.5972545411234, 773.6315081816985, 783.0810011131523, 790.5172545863683, 796.363082163228, 800.9579501634364, 803.4333250253715, 803.8764266273995, 803.0014943440789, 800.9178411382705, 799.1194248786709, 798.1846634305296, 798.3943206671746, 800.4126898756773, 805.3070119545163, 812.9078860026179, 823.4013706281055, 836.2447999547509, 851.7974488272621, 867.9391325570756, 885.024979343648, 903.6753031173654, 924.1331440106978, 943.6870724730167, 963.375735186012, 980.0093741136428, 990.4647833805215, 994.4604183424169, 993.684140418179, 988.1502467162021, 980.1343834210338, 973.2051079872796, 968.2330623581471, 965.7294368962146, 966.4841504962833, 970.9142553556555, 978.4811640259835, 988.866539193984, 1001.5759365575534, 1017.1095597183918, 1035.9568832522154, 1057.4684635072435, 1082.0177077436877, 1109.0599334780204, 1135.2249768224385, 1157.9205281020713, 1175.8287881451909, 1187.1774191716927, 1192.4943576046294, 1193.8596448353037, 1193.918814317632, 1194.751492416161, 1197.5047115355178, 1202.1784716757031, 1208.7727728367167, 1215.1266446267416, 1219.9936450195796, 1223.0879990994217, 1224.4097068662677, 1223.9587683201178, 1222.8156686568805, 1222.8483350217236, 1224.7554117335762, 1229.046261486412, 1236.7751742158819, 1248.82508846027, 1262.485169226959, 1277.2154526255179, 1291.9972132679989, 1304.721871283101, 1313.6235495942542, 1319.7392953550268, 1322.3701511031647, 1321.7919112950399, 1318.5113573645765, 1313.41142785006, 1306.7043696466505, 1300.0422866534773, 1294.6988161389595, 1291.7689751065486, 1291.3387197159816, 1293.536390282888, 1297.5359348577035, 1304.3132836066113, 1313.3209280278847, 1324.3869558020501, 1337.2546862978484, 1351.6761654997451, 1361.79991201447, 1367.5511488300788, 1369.0158321063086, 1366.3223021587887, 1359.9664670185878, 1356.0622511600104, 1356.0886727024524, 1360.3314686354497, 1369.7728286244976, 1384.6836646065149, 1401.8870178386444, 1419.0273780750063, 1435.6034283314443, 1450.597265881713, 1465.9163048983285, 1484.9730404183304, 1508.3405466230665, 1536.164246512241, 1567.5333765418604, 1595.909076675743, 1615.2345402848387, 1625.8879623262267, 1627.558920196468, 1621.1938905003083, 1612.0728869827467, 1604.400643578088, 1598.1771602863312, 1594.1635963041865, 1592.3599516316538, 1592.0421652306388, 1592.5677640893693, 1594.06821559514, 1596.456686521854, 1599.7620394808923, 1603.2024940132517, 1607.5784670329626, 1614.615722238484, 1625.225907931907, 1641.4139752497679, 1663.4133667229912, 1689.5649499264248, 1716.0227953012695, 1738.738521300229, 1751.9987943399124, 1755.6716844835587, 1750.2550870025286, 1737.737700369868, 1722.3063572848623, 1709.2574210591604, 1699.3975927210154, 1693.8360006763578, 1694.6994588652785, 1700.6441214614395, 1710.0531449886653, 1722.0238125978856, 1734.0571102209058, 1739.0289045087218, 1735.5973923240108, 1721.3160804284048, 1691.5956866216154, 1643.4383952944038, 1580.8673244335641, 1505.8201652286148, 1421.1584898233425, 1335.112030270847, 1254.8352278157058, 1181.7284159855517, 1114.9190605686458, 1052.020485765115, 987.1337809660836, 917.5164642295792, 846.1609782355124, 774.3665303880954, 708.107728809188, 654.1819709182442, 616.8321708262781, 592.3776493021184, 579.6865903071462, 569.8936772642652, 558.6004026354967, 541.3603047769379, 518.2843995889546, 487.97421139090767, 459.24957075904433, 432.4954412224668, 408.75317653287857, 388.58705822678894, 375.121802047323, 365.51357566937406, 360.0000393470659, 358.14772518520397, 359.9255229280287, 363.71644730700336, 368.19789264236545, 372.5095748967651, 376.47707610888824, 379.3259017253673, 379.65351380352456, 377.6052465958083, 373.6112421208933, 367.7587093594367, 360.45038921206736, 354.55493609000587, 350.94237584920154, 350.5280078262234, 353.7870343909437, 360.7559802558412, 369.905150426508, 380.28466044187326, 390.0639116287986, 398.29249924754015, 404.87671570787955, 409.8801395356821, 412.846776188702, 414.69192500350835, 416.25071259997384, 418.49728522919884, 423.87133215442674, 433.4362951705176, 447.6586899814911, 465.85485051270336, 486.17019185217197, 503.72533547341015, 517.5922611331628, 526.8017549980326, 531.6001642180156, 532.9147812491032, 533.0437666201253, 532.4511304527101, 531.5362884160088, 530.0984095848614, 527.6349944909359, 524.1080863251863, 518.6075810643031, 511.3762260540946, 503.0116847434657, 494.51895606908073, 486.39853985227063, 480.95695716707996, 479.16154181205957, 481.3617748873148, 487.32483841239645, 497.1081817777464, 508.8290749052041, 520.2822314272513, 530.5465579670823, 539.3453787252545, 545.9217663651405, 550.7620339141656, 555.034980979532, 559.5888801243601, 564.1680384838854, 569.1509197264219, 574.5375238519694, 580.1789160121436, 583.7870200153441, 585.6245230378947, 585.6914250797953, 583.9877261410461, 580.5920491462139, 578.1226869032455, 576.8410965210562, 576.9062553971512, 578.8761563833983, 582.8963700618931, 588.1088193750238, 594.1468717220548, 601.3512579783035, 609.6163179969824, 620.7629847316057, 634.483163349302, 651.1076879824739, 669.7098195976101, 689.052526003404, 705.0572295465396, 717.8882274571067, 726.9379208368459, 732.4016855368063, 733.9675668972893, 732.9356780601706, 729.2273181747144, 723.152566493933, 714.9430779274657, 706.5665346964416, 699.6468587525214, 694.4925716142359, 691.2473952909153, 689.9113297825592, 689.6133470139637, 689.5414860092985, 689.3478546261317, 688.7180305328072, 686.8177430908653, 683.646992300306, 679.205778161129, 673.8813634396683, 669.2334051355626, 667.1995568872763, 669.3321938967322, 675.9595416878398, 686.8879688774323, 699.7258454472047, 713.3439695097198, 724.8470210368671, 734.1068177271012, 741.1233595804223, 746.9711809424874, 751.9986863247856, 758.035908247823, 764.7166569815266, 772.0409325258966, 780.0087348809327, 787.7315993244107, 794.0259200919587, 798.9936812467704, 802.6348827888456, 804.9495247181845, 806.5034861329552, 810.0088700904696, 816.4304058875523, 825.7680935242031, 838.067777950969, 854.3948629422093, 871.0937034119324, 886.2830111103342, 900.7311222131708, 914.3674693203706, 925.2184805253712, 935.1083146865163, 945.0472144485718, 953.521037522278, 960.9186034065602, 968.1892311364139, 974.5891983040923, 979.8416571661871, 984.7254275688341, 988.4911929122114, 990.8994157485773, 992.509719726068, 995.1541608919148, 999.3656314191381, 1006.2266596344983, 1016.0399795665126, 1029.111358494148, 1042.6447891704343, 1055.5535545272724, 1067.092915881397, 1077.246287190374, 1085.586605674572, 1092.9516971746475, 1100.2466610956544, 1106.869048475109, 1112.538988947174, 1117.1933062342123, 1121.205452191443, 1123.7062117955566, 1125.3454639159636, 1126.3153174773674, 1126.989932955988, 1127.3693103518258, 1127.938376445582, 1131.5540142499344, 1141.9781194459235, 1160.7010894329728, 1187.722924211082, 1223.1576472227666, 1261.319650915047, 1295.1193938277524, 1321.0070150682805, 1338.9825146366318, 1348.817845647775, 1353.3135741696362, 1354.9288461775254, 1354.4101038985707, 1351.4427363144512, 1345.6358292254397, 1336.0845979273515, 1322.9472386279008, 1308.470416463349, 1293.28335347034, 1278.3959249333582, 1265.6740172055245, 1258.4698079564316, 1255.8080205205256, 1258.4722988571139, 1266.0204065132511, 1277.5540292424382, 1292.0818860191773, 1309.5032287187348, 1328.9316379683653, 1350.5913081191143, 1374.4779402437139, 1395.7930276707218, 1414.293898941874, 1428.4586278697734, 1437.6507221844374, 1441.7748929502195, 1443.2877696114479, 1442.1744141474187, 1439.7105604057392, 1436.2458051160493, 1431.5198527173218, 1426.9152715678083, 1422.8526650697595, 1419.4007466929866, 1417.5457815868606, 1418.1115275889933, 1420.4067005202592, 1424.936584434435, 1431.9814634073714, 1439.568807140325, 1447.0945612856904, 1454.699130824335, 1461.2418049753956, 1466.437808307885, 1471.2734059711747, 1475.7864511231642, 1479.6961338021188, 1483.1371183050585, 1484.1302946953565, 1482.6756629730128, 1478.8409209840024, 1472.7664737091923, 1465.5528759752176, 1462.3400691529391, 1463.9199942820962, 1471.6085432716777, 1485.405716121683, 1503.724672521745, 1522.2827831572158, 1540.179621919824, 1554.7834049915932, 1566.6000932725306, 1576.4430762789882, 1585.1410668772708, 1593.7823132978326, 1604.1582133499348, 1615.3338643201566, 1629.0812686612887, 1645.6276103610194, 1662.4387554585385, 1679.2201745946338, 1696.8793037744767, 1712.6115150852515, 1726.2580735075812, 1739.2645015061103, 1750.7933400984432, 1759.8105818144716, 1765.7820200607275, 1768.88672375542, 1769.2335412327932, 1767.0778874795662, 1762.9752757740903, 1759.1034288271724, 1754.9574990528695, 1745.6347398120051, 1728.1446687979028, 1701.951447769261, 1665.4790133165745, 1617.6204060649052, 1567.2412756645103, 1521.5257890925852, 1481.5456228317344, 1449.0158757774668, 1427.4416337747002, 1413.581927957061, 1402.466117812612, 1390.1138196155043, 1371.2448190268256, 1342.1516180491274, 1298.1958308149108, 1233.7028769087417, 1149.0919451772968, 1053.4756986404548, 950.2136345555227, 846.2754667355359, 755.7609721735569, 688.1654096298722, 637.6099094610602, 603.1514223902752, 582.817785443214, 569.686824967964, 557.0301553879762, 546.0121433293323, 536.0462616728033, 526.0514113024499, 516.161046684804, 506.8921347281301, 499.71208349210394, 494.7659539869802, 492.05374621275894, 491.117006073224, 491.69725011424305, 491.5506499678048, 486.8345042264001, 470.8858653280264, 443.70473327268394, 405.2911080603725, 357.38880008651637, 309.6266203258479, 276.1057753254678, 258.02365068200027, 255.49091517290634, 268.17888790561517, 289.56483072804224, 312.65715204911095, 336.7048278057322, 361.48652044298404, 385.91578135058387, 408.797154487267, 423.82677458433443, 428.91453297809187, 424.171098446, 410.13969529320036, 390.15672460886196, 377.9594766746818, 379.0928793980448, 395.656372659026, 427.7245068683136, 473.07035511680607, 522.2473119191316, 567.4530157208392, 604.4885867617786, 633.2049242205736, 653.0259812165102, 663.6313262098474, 668.3272413690754, 669.0968020016438, 664.6983893862831, 655.916420783716, 644.828449929531, 632.5655337359609, 619.2189714767879, 607.4005408291466, 598.2991975546781, 590.865448548696, 584.6218758129401, 579.7135638632788, 574.8654647413173, 569.3224484664715, 564.8202582426128, 561.3001537645035, 558.6635264695617, 557.0777503291583, 557.7169227350699, 559.704184470617, 563.7851651142416, 570.5301315756617, 580.0849743731031, 590.3387291761966, 601.2913959849418, 611.4517156424541, 619.6363330204116, 624.9896634079043, 628.4839270182624, 630.8957631760046, 632.9708014595733, 635.3603972508936, 638.8539077982174, 643.9197532389942, 649.9658936589153, 657.3295292116381, 667.2880256091805, 679.7073855830165, 693.6507688582466, 708.6691268866145, 724.4505146686329, 738.3558410366152, 749.8503076864515, 760.1323171384493, 768.9265230032702, 775.8566752553116, 782.1301478228368, 788.1481386647607, 792.7482999393912, 796.3134331048155, 799.2235919629902, 801.5734502235566, 803.3630078865147, 805.4915948336218, 809.3049173114603, 814.9777522790667, 822.6021786696772, 832.6258402506907, 843.8390259364587, 854.4610331019916, 864.8179912928501, 875.1934801254334, 884.692212064944, 894.0350990898875, 903.21740991839, 910.1435336772316, 913.748703181211, 913.4211736863101, 909.1024878616448, 901.2889801464729, 892.3269498758816, 882.3234345457861, 871.1982617028152, 858.927518983903, 845.7119809696912, 830.7816646126978, 815.5220833449214, 804.3635079507998, 797.9175889340147, 797.1264047122543, 802.1582225848931, 813.3979470273537, 826.5101045030303, 841.0327570990627, 856.5431397443923, 874.1040002416303, 893.1172495311142, 915.8315880068409, 941.8330495112762, 968.8397670625087, 994.1894306086718, 1013.5398924378562, 1023.7232756606512, 1025.0626486070785, 1018.7914121426544, 1006.3301903677532, 993.4041588141029, 982.6902646029048, 974.2792869638837, 969.5872577131273, 970.5241864032857, 977.5265659770519, 991.0487483140157, 1011.9912961973976, 1039.4849177626497, 1069.7095939044725, 1097.845767681628, 1121.5587215172702, 1138.220110401803, 1146.6122454772203, 1148.645136296172, 1146.728561329277, 1142.0298793649588, 1137.4126151087614, 1134.3997220032106, 1133.51176155832, 1134.9624809129223, 1138.7518800670173, 1143.2541350459735, 1148.0460067931756, 1152.2868096110476, 1155.5490492219249, 1158.2417719837406, 1161.074148351216, 1164.0786324234457, 1167.44149595031, 1172.6117830987182, 1179.482401529442, 1187.8156507558626, 1199.65750624673, 1215.183260080043, 1232.576862663203, 1250.8253596008658, 1269.6107161439081, 1284.7436190575438, 1296.1940724588044, 1304.034753042473, 1309.3136700707778, 1312.7896505676974, 1317.125205816502, 1322.18942377741, 1328.3439117263429, 1334.914651892123, 1340.6336888594578, 1344.0368773585785, 1345.2181853707132, 1344.3240814896778, 1341.089683339748, 1336.6481906644392, 1332.2730111542965, 1328.0151454117336, 1324.2187856575133, 1322.0877144142628, 1321.2891169914024, 1321.4828337801112, 1323.503832802333, 1326.8690677379163, 1331.8469134480572, 1338.7234334087439, 1347.9669734230608, 1358.6093536121414, 1370.7152381629605, 1382.708933217509, 1394.0183118238094, 1403.644491090834, 1412.0064538673669, 1419.6043211235265, 1426.978682207804, 1434.4156005961895, 1442.435157485773, 1450.2317744987142, 1456.9115712835805, 1462.663247311721, 1467.770762659739, 1472.6867244481298, 1477.5804735198342, 1482.9864911215511, 1488.857055084598, 1494.6242452557685, 1499.3520237817713, 1503.1206021705407, 1505.707786543197, 1507.2903814543963, 1508.23064863597, 1509.0286992960364, 1509.9668629149169, 1511.4745066197709, 1515.330192475424, 1521.6794742440438, 1530.7419954634483, 1543.2223914754693, 1558.5732518776058, 1572.7786221292627, 1585.3124897404182, 1595.6547025971718, 1602.2832657765853, 1605.4718844799097, 1607.3050077126, 1608.084792547281, 1608.1772367742203, 1608.4804746799073, 1608.9945062643415, 1610.238140393921, 1612.3792872363997, 1615.7375650198774, 1620.301545883302, 1626.4071704317416, 1633.1177420990434, 1640.1265636606458, 1646.8506933577737, 1653.402494838246, 1659.3377033266092, 1665.0390189031052, 1670.6619181793358, 1676.54332611783, 1682.1904667224142, 1689.3467653302937, 1700.1278689482394, 1714.4712753557146, 1731.7112658568906, 1751.9502656858272, 1771.5666539739116, 1785.3329118190236, 1792.2825177159616, 1791.7060612832834, 1776.826982128642, 1742.5895066871224, 1692.4264315079351, 1627.567800626011, 1550.3162620603503, 1474.567067642724, 1413.810543293495, 1366.3563550162867, 1327.767388715926, 1293.2046591131395, 1254.0849690229377, 1203.062352373475, 1141.801038240907, 1076.5581485189016, 1013.7547873683947, 955.7175723947399, 904.5181893335293, 861.3783354450492, 825.453747751555, 793.1775799091201, 762.5144767043661, 730.8910320730289, 696.6676448640304, 659.8243227974066, 620.7435361240009, 581.7955919649473, 545.6494317832303, 512.0564850261813, 480.45268307517017, 453.93183002826134, 432.73076320742484, 416.387815474283, 407.33059161915344, 404.94969785589217, 406.66002269733383, 411.19873057750254, 417.5231901857819, 422.6598083115688, 426.6085849548628, 429.3695201156643, 430.942613793973, 431.5628767226617, 432.0590870656126, 433.3757913577154, 437.3764247880152, 444.39229909462, 454.4234142775298, 467.5938229224824, 482.0144319596984, 493.9583710110877, 502.6689020350558, 507.87764858307366, 508.82432522585594, 505.6001534746818, 499.9660789317389, 491.5639621887377, 480.52891354286123, 468.37111494227474, 455.80188083233605, 441.0353145063261, 425.73266195317535, 410.42883192410574, 394.38431681064174, 378.73646279289994, 368.94638452615106, 366.79237997550473, 371.9552697311941, 385.33472796771895, 405.9354190832156, 428.6027622347286, 448.02480093772954, 462.50540072238175, 469.38734425389885, 468.40475646492223, 461.1394898367128, 450.2475226115348, 438.8737341587363, 429.829076042231, 423.6844673150398, 422.28706359009476, 426.0643891995161, 434.5087921564668, 446.7096599183251, 462.7366154966204, 478.9769875569047, 495.3670328096994, 510.3307102282364, 523.9268825648124, 535.7680869726171, 547.8058901264907, 558.8852056090298, 569.0874471654225, 577.6397724585763, 584.6970184925561, 590.23516514437, 595.1989958114019, 600.4009879840003, 606.6230639569842, 614.3720666285609, 626.4304113493738, 642.9017567856506, 663.4482575714412, 687.7736299858437, 714.8192594438324, 738.6960146351504, 759.2605484169915, 776.7098148098685, 790.944427891531, 801.7927058529671, 812.142512187999, 821.5574967583561, 829.4990161071726, 836.1408323699874, 841.863535488966, 846.8410972703999, 851.8508730065287, 857.6825359699822, 866.5395562381485, 879.2606466915793, 895.2388458972997, 913.3627207899968, 932.5218280733983, 947.7129397666195, 957.0921330109247, 960.688120159303, 959.1575395770551, 953.1592897785749, 946.089963887154, 940.184835085418, 937.1936899761351, 938.0120061221094, 943.1049241767453, 953.6303352069499, 969.4282430461883, 988.913086030075, 1010.2939090330019, 1032.64043074816, 1052.4447586545061, 1067.5073045655108, 1077.565357001479, 1083.1830259431779, 1084.825452044011, 1084.2465815645007, 1082.7767580524876, 1081.0044375952102, 1079.5923553567432, 1078.5405113370857, 1078.0030200463812, 1077.9798814846295, 1078.4422250150571, 1079.1165961623312, 1080.0029949264513, 1080.7931922871321, 1081.5805415765353, 1083.0979452794543, 1085.2917155415225, 1088.1618523627396, 1091.8624702532484, 1096.3542799237098, 1100.171476404536, 1103.2476960843508, 1105.5829389631533, 1107.1852939447592, 1112.0653042920417, 1121.0294154462495, 1134.1397657664224, 1151.3963552525604, 1172.7830060970343, 1190.0230863134686, 1202.969509988954, 1211.6222771234911, 1215.9813877170795, 1216.357114391026, 1216.9614818260106, 1218.6427397039963, 1221.9164053017262, 1228.3660449895476, 1238.102754565362, 1251.3785062268219, 1266.8319226530825, 1283.4867016200214, 1298.793216332621, 1312.2182759265888, 1324.245852460032, 1337.0068267060985, 1352.5747934622784, 1371.4599181333872, 1393.188796287482, 1416.3984633622254, 1438.2339740720508, 1458.0708830033643, 1477.7574503428586, 1499.4824100856788, 1523.5357741856656, 1550.8067665503409, 1577.6511161271974, 1601.1224816567405, 1617.7340538352025, 1627.7989532212298, 1631.7386910150533, 1632.2091738326642, 1630.9790893353052, 1630.2335653844452, 1629.9964897741781, 1630.2678625045028, 1630.9925530727737, 1631.7411383418203, 1632.2198815396355, 1632.4287826662185, 1634.1899177750638, 1639.102389968851, 1648.1485075228684, 1661.3480542313985, 1679.6662514542106, 1699.6741197854694, 1718.8672078229363, 1736.3738100732764, 1752.4708910750426, 1765.228008108696, 1776.0368918254237, 1785.3296577322537, 1791.490467946589, 1792.0362041524252, 1784.8834698516555, 1761.9367096904648, 1719.1640515586791, 1651.3256078760398, 1558.416341418976, 1444.6933450826614, 1326.153241893285, 1210.994868580176, 1108.9690957920122, 1024.261115851521, 954.0631750320251, 887.4387275495226, 817.279711486895, 742.8522465097057, 661.1725662315336, 575.5298257952392, 489.9602910493351, 410.02884149734274, 336.5890402631286, 273.68830983437226, 221.38631881406778, 181.98732102728172, 157.43523782941006, 147.4096044873799, 151.6184303698989, 168.4695569310693, 195.62806066777517, 227.92316131791713, 261.8055564389163, 291.7712192243409, 317.73692529076345, 339.3354926084811, 358.9001701980306, 376.30265547776696, 395.15291287689786, 415.14523163054537, 434.74913972506954, 450.9487946067362, 464.5450722904503, 473.7683219932798, 478.85009235295263, 481.6424743202144, 483.95950142458696, 486.1042654564567, 488.9750289900265, 492.57179202529653, 495.4757900320971, 497.38978297240465, 498.19395794569573, 497.8883149519705, 496.47285399122876, 494.6569573285554, 493.035105039998, 492.0924600468408, 492.58831288378343, 494.6492738590383, 498.440015641267, 503.89967823260065, 510.54377401572253, 518.5946254228131, 528.6690155257181, 540.7086357095232, 555.8379601214708, 575.1963344545109, 596.6284343281327, 618.5208626740068, 640.5344842092247, 659.7779682976818, 674.1045286029419, 685.5446818655241, 696.388751290197, 706.8312426128764, 718.3358745034213, 731.8928751655338, 745.2214418574874, 755.0358889519681, 761.118674378059, 763.5365559146077, 762.4092663613391, 759.1608031330912, 756.3573864991629, 755.5051301374282, 756.6040340478871, 759.6540982305398, 764.655322685386, 770.588879087457, 775.187600890594, 778.4211607357614, 778.7770149853119, 776.2551636392454, 771.2357474663733, 764.7579887820406, 757.3542488117341, 752.2152923699484, 749.8158308444899, 750.1566639786275, 753.2200624441723, 758.6536917409443, 765.9672715514837, 774.9651522474553, 786.2198144496455, 799.721993048636, 814.7005828044673, 828.6140244591672, 840.4294831059848, 848.5923770651328, 853.1073388913203, 854.595774458235, 854.4113021644625, 853.5897801060094, 853.301974246447, 853.5478845857755, 854.8539212616895, 859.8023000760925, 869.4087664022136, 883.5721244273993, 902.4819403694935, 925.0853939531077, 946.2180535744337, 963.07082558036, 975.3119292471395, 982.5622321390844, 985.3481443938894, 986.7808392692741, 988.4733318986221, 992.5014345884432, 999.0547135565814, 1008.5628721076002, 1019.9679953298679, 1032.9979276793997, 1043.863735115649, 1052.7156995855225, 1058.6944144798933, 1062.3288372545771, 1063.8096128721984, 1065.03120835303, 1065.9613835504756, 1067.0298417690992, 1068.2624029403808, 1069.988628630273, 1072.2879541771165, 1074.7740140333829, 1077.4468081990722, 1080.1752614728834, 1082.2395270603101, 1083.4049426656024, 1083.755450602463, 1082.93574727213, 1080.9614075113655, 1078.2024753278554, 1074.8899692980785, 1071.3926522890636, 1068.9879369730543, 1068.0105057298824, 1072.0453551291146, 1081.2360330370404, 1095.9000123407993, 1114.5483784921892, 1136.860243369066, 1155.6656138322971, 1170.5258109111644, 1179.9352716437068, 1184.3518201325055, 1183.6514122213073, 1180.7281500750992, 1175.527109147796, 1169.5538524013596, 1163.0263425800676, 1156.3444403471701, 1151.8784536144171, 1151.1017624325, 1155.59092964595, 1165.2369738826285, 1179.8399648109105, 1196.7319698210354, 1211.8677859754757, 1221.0905789438598, 1224.4003487261878, 1221.7970953224594, 1214.2705116316058, 1207.3949208831102, 1203.914496436634, 1205.6713777630066, 1213.1855126766786, 1226.4545535849695, 1241.943262795948, 1258.3201279206628, 1271.506865994527, 1279.3635985292688, 1281.4068754462562, 1276.28840684964, 1264.3768927037022, 1247.0330536806407, 1225.8657337131206, 1200.5479431934323, 1175.3902646747288, 1150.9868106173983, 1129.482430687489, 1112.5016312206888, 1103.2114757061913, 1101.2409070712663, 1107.1955123820335, 1122.1232055436271, 1144.0584241375523, 1170.1905699401684, 1198.3300308480798, 1229.0740345637132, 1259.2601221222449, 1289.0902291775976, 1318.4154395151688, 1346.2622747302132, 1368.6063166485087, 1384.8204609508332, 1395.0549148072278, 1399.7417999440522, 1400.2599726656026, 1399.2560130277852, 1398.1832328329497, 1396.735944011713, 1395.2580319748174, 1392.7904837518101, 1389.2112664622648, 1384.5582015703287, 1379.4426652147672, 1375.6392340112318, 1377.4519033221197, 1385.320023538878, 1399.5782481028782, 1420.5438464321753, 1445.1638862166524, 1467.9473075278595, 1488.8749316347455, 1510.214288096659, 1531.477465388667, 1553.7006010859434, 1579.0074808812149, 1607.0234631376159, 1632.1527962161122, 1653.5024307042581, 1670.3728110979634, 1682.5686291998031, 1688.5353571986234, 1691.1400265663865, 1690.0847817696263, 1685.8714512775719, 1677.2597155772462, 1665.6875900344555, 1651.0464350733255, 1635.048956065493, 1618.1732791643012, 1603.2436529438532, 1591.0059879600344, 1582.0004426674247, 1575.9691832729745, 1574.3469714073533, 1576.065032598785, 1581.9801866433688, 1592.5311507787303, 1608.4929785137247, 1628.8207069100908, 1654.4110179638199, 1682.350713085724, 1711.1963139892039, 1738.7202580787223, 1763.1301284916767, 1781.8580692046119, 1796.2114777154056, 1807.588412447493, 1817.7282261716616, 1827.9987053157631, 1839.9484604170575, 1853.5753005266856, 1867.3631049293574, 1879.819591453977, 1889.5997416537498, 1893.5301323288754, 1884.926295780581, 1858.8636240516503, 1810.8522348578103, 1739.9062787264072, 1647.2945998498242, 1543.7698014164077, 1439.9644744339196, 1344.9546056291206, 1261.6696929918814, 1195.6246518773871, 1142.578795665823, 1095.2738080485974, 1049.968789086654, 1005.3564444040878, 958.3867526772526, 910.9637192245142, 866.6987184185573, 825.5471213318842, 787.449555026756, 748.8377190008628, 704.0818672148957, 651.832169242644, 592.9442735356265, 529.6052290073997, 466.91004327101916, 410.6451326153917, 362.8875780921519, 323.42148475968196, 291.5660169017209, 267.3089972650713, 253.49575171878314, 248.38553438242064, 251.36262939995203, 261.775613935336, 278.6365876650348, 296.1729851039896, 313.92790552934764, 332.3020391295512, 351.0114136487656, 367.1487610420724, 383.1854725288284, 401.11493655295635, 420.4985279095601, 441.41705530917307, 464.39855541638445, 487.7417125424896, 508.5096457918136, 527.1321926731782, 543.3963554935947, 558.3131454580022, 571.2003124668984, 582.6212992578128, 591.338608984062, 598.2086252583252, 603.5155796188685, 608.5410359514324, 614.2942391554949, 622.0220791203674, 631.9948468955781, 645.7114292283793, 663.3588646132391, 685.6061476752993, 711.6592719967703, 740.189129782531, 767.7629770141777, 794.8168839776621, 819.8928136450122, 844.1271867554793, 868.270066259602, 894.0193933166786, 918.925791756986, 942.1293958558775, 962.0614518962681, 974.7814048493681, 979.9792736563628, 979.2272956779516, 973.4981839127125, 963.5987948832811, 954.7665377190281, 947.6303484144235, 942.1902269694677, 939.3713878234403, 939.2579120234184, 940.7003152228324, 944.4899500067655, 950.6268163752172, 958.0263425594458, 966.5203664652963, 975.0271212655704, 981.4419301845979, 985.9533354830253, 989.7284065040118, 993.903236717813, 999.2802680508852, 1007.9211643532424, 1019.9294803922483, 1036.5452033755657, 1055.6643084568375, 1076.763678610349, 1097.8246629117427, 1117.9063039282148, 1131.791003517546, 1140.4892083379723, 1144.2624769023519, 1143.8117532579806, 1139.9541189257898, 1136.3951033255344, 1133.2178379870988, 1131.0501336935745, 1130.8326204713953, 1133.0225958643798, 1137.8210126476001, 1145.6158035634799, 1155.5342107114798, 1167.676764939618, 1181.2518473648695, 1195.5352552780453, 1209.6679916644139, 1223.619084066638, 1235.5991063689053, 1245.314740357401, 1252.1704987384182, 1256.5958800193227, 1258.4292924398148, 1258.7273826561366, 1257.9495691051727, 1256.7342442191243, 1255.0814079979907, 1253.7591390210841, 1252.841319228399, 1253.168330019308, 1255.7110994873055, 1261.4968019777248, 1271.1566139164415, 1284.8830972287462, 1301.4660965315625, 1319.5207929492367, 1337.4295369913448, 1353.4581184821454, 1367.382393797571, 1379.455764391781, 1389.5350837356018, 1398.0193108656526, 1405.4916351231093, 1411.7603948811807, 1416.601074394775, 1421.4734578747891, 1428.0268080252404, 1438.1331499957294, 1451.7383230586915, 1471.891557887526, 1497.120977795741, 1524.2659065004714, 1549.9608480314419, 1574.7448805990803, 1593.187972794373, 1605.5256372916108, 1613.460803778112, 1618.9266367516188, 1622.0589433098341, 1626.6789229446038, 1633.4952691024905, 1642.5265904652902, 1653.724324049208, 1665.9122326069803, 1677.60315781506, 1688.3568015934968, 1698.2871238359967, 1707.0448942211267, 1715.5815437224137, 1723.881768351345, 1732.8299353327936, 1743.658970203199, 1756.36842915944, 1770.3334105712117, 1785.647955459799, 1800.3639182796037, 1811.1533545036978, 1813.7072200486396, 1798.0670915303567, 1753.8171963515047, 1677.8141643794515, 1567.5130434479922, 1430.0591294979927, 1281.1301162700242, 1140.2656892473121, 1015.3067521945986, 913.9539370297658, 831.9616613112896, 768.5647086832995, 716.3301352057725, 670.0568836573973, 628.0060599162128, 594.2565777374264, 563.8945405935852, 534.5180053144792, 510.0958558460321, 489.4793740490335, 470.7935467170667, 455.13488147597093, 445.2778472575994, 439.5780420338783, 438.42891160012533, 441.7753573229453, 448.9748660806881, 458.86757358294034, 470.0695633059042, 481.1827095079054, 491.09902249331105, 499.5868877254904, 505.2497903169834, 508.7796885296889, 510.8756452344443, 512.2635150181226, 513.6608534240404, 516.5343259296578, 521.0730951621578, 527.5167327355375, 535.4641816289953, 544.4137034396729, 553.7722694857239, 563.2001870580658, 572.9030081425988, 583.5252042076735, 596.6764605747337, 611.6688201284873, 630.5117500872238, 652.7505678798694, 677.7205468970458, 701.3005386044387, 723.9860624865363, 742.0206117252765, 754.7386090491184, 761.3089284680696, 764.0911507993376, 763.0301438986123, 760.0234774475666, 755.7462876889181, 751.950000125178, 751.2512164606295, 754.0478838650756, 760.3400023385163, 770.1275718809516, 782.8629755998627, 793.4686276572497, 801.7947056903847, 809.1492734157433, 818.3755769510065, 830.139871107774, 847.4519737903772, 870.1206012510663, 895.5296260568912, 917.9925559724891, 935.694223119772, 946.1397407837674, 949.3766729633683, 945.596597247582, 937.6197892390413, 925.7784254186699, 913.5713954751358, 900.832287377942, 889.7940733820245, 880.5026945174793, 874.2698517855106, 869.6853695527291, 868.0732533222975, 869.2835065873253, 875.1597001945017, 887.9075863541073, 907.7691072837302, 932.8807808718648, 962.0298078217381, 991.4831054098743, 1015.5174682145083, 1033.3766460382685, 1045.8378507208265, 1052.3365953289012, 1054.7394212242302, 1055.390641330248, 1054.3303154863738, 1051.5584436926079, 1047.976439223526, 1043.5843020791278, 1039.416611834311, 1037.9123412594768, 1042.6941296431442, 1053.6713603202152, 1070.8440332906898, 1093.694858767119, 1116.9466248568588, 1133.3134630308948, 1142.7953732892274, 1145.3923556318562, 1141.0506192143248, 1133.697279984774, 1127.0361571356752, 1114.6908677331937, 1094.6904901832233, 1061.0668156741556, 1013.1918286051387, 949.1109335684878, 881.4421545876837, 814.0055749084662, 758.8989846874229, 716.7506146075713, 691.3884755803294, 676.7056683602397, 670.974791238139, 666.6102949610517, 662.0084979112177, 653.9666740962465, 644.1736832492185, 631.2183516375588, 624.1581142536068, 616.5060548232573"
    # gaze_y = "504.4765819844874, 513.060819603218, 527.2754000742335, 556.9899969632528, 598.4849471289574, 648.6722201516556, 701.0397570587133, 746.8214085941618, 776.5434310554328, 787.5362296217306, 778.2328950332553, 748.855602090144, 704.4691764858655, 651.1584898216054, 594.9821595143032, 540.8746526818991, 493.49899819792154, 452.23967173455213, 415.4940121629732, 377.79219984285436, 338.9374262510301, 298.4191277870226, 258.6602416220299, 227.35498204749643, 211.86617018243555, 211.9727457229879, 227.6322705571219, 257.07059449990163, 290.6390441408199, 321.1083173318767, 345.8933012997183, 360.69181692099835, 362.27878716922, 351.26014025205325, 329.02359356498334, 296.54149749269334, 258.8192342632845, 221.03683819575016, 188.48826435041974, 161.82759224391808, 141.6952338802326, 126.30084572456198, 114.76716652294036, 104.49674489018548, 97.46911174809736, 93.36406109468228, 92.37812460201205, 93.38550030768852, 96.35317078172339, 98.89585937346214, 100.2452790053044, 99.83958896888066, 97.67878926419095, 93.76287989123529, 89.32425418037865, 86.7031229789075, 86.82231827901987, 89.68184008071573, 95.28168838399515, 102.84980135527444, 108.24762900637147, 110.19134806125956, 108.68095851993876, 103.71646038240904, 95.63178468350172, 89.12257261666028, 84.95291908657285, 83.12282409323942, 83.84490352742947, 86.94243679887924, 89.92038348410465, 92.53437704975607, 94.7844174958335, 96.24527304079791, 97.00530397978123, 97.43053576132023, 97.64315165208973, 97.64315165208973, 97.7058154933668, 97.90771387249595, 98.43768664179568, 101.25198337902788, 106.50202014639224, 114.06246926133463, 123.78018933070503, 135.27750064986643, 144.64190406329504, 150.96346758240358, 154.30485504846916, 154.74263715806677, 152.0266976398545, 148.11328607159413, 144.10646804457923, 140.00624355880979, 135.8126126142857, 131.91305719764915, 127.80855799764447, 123.41511554864762, 116.72194972688199, 107.54050524615596, 96.35424698224384, 83.62203223038901, 68.55760934083908, 55.18253856114734, 43.873930463697015, 34.258233831655204, 27.299139117870432, 25.920391412401678, 28.356378283654948, 35.31848960531639, 47.86080447524341, 64.67544734703155, 81.48903239197418, 98.00411315943145, 112.42079932964798, 123.00188095675158, 130.13170515031055, 135.73731607924182, 139.67810356827704, 142.8832618771076, 145.48965503813648, 147.4972830513637, 148.80074195719143, 149.59292068934536, 149.06629967714622, 147.6161571943729, 145.2424932410254, 141.99800979690252, 137.88270686200434, 134.42387627824797, 131.80661608545316, 130.03092628362, 128.97650033138495, 127.71936548943033, 125.46890846230238, 121.01384315252847, 114.35416956010867, 105.73050076776997, 96.99078225414765, 89.67428626324549, 86.1778180393948, 87.81522855060902, 94.4662112555247, 105.20679341482415, 117.0832063137063, 128.65805069619643, 137.26739472221954, 142.91123839177564, 145.179175409189, 145.33493720766916, 143.43116020370422, 140.7509182267994, 137.29421127695468, 133.87828083617404, 130.80953164521594, 128.06740422877425, 124.99388093678529, 121.58896176924902, 117.44938264918471, 112.53940498104525, 107.22023173359692, 103.0451492386837, 99.95123607145916, 97.93492112257579, 96.91789520881215, 96.66620950942615, 96.35160238519364, 98.52873336923611, 103.78764223411372, 112.1283289798265, 123.55079360637441, 138.0550361137575, 150.7205017102723, 160.27677070312728, 166.71429959294974, 168.53113847810556, 165.47892071480283, 159.79587278894064, 152.25271476842136, 142.86853365199062, 134.64722924291652, 128.6586817667931, 126.2745470433586, 127.40448492494188, 132.03895191217015, 138.67599810340934, 145.14296553393055, 148.98037692644795, 150.11282790164267, 148.50580774335026, 144.1593164515708, 139.20249585414754, 135.2016938365552, 133.16487295494943, 133.1610546416591, 135.1902388966841, 138.47442827510807, 142.72568654653315, 145.68582440501746, 147.32033113439655, 147.58967966383656, 146.4938699933375, 144.03290212289946, 142.00088451321488, 140.43734423511768, 139.55858747122068, 139.36461422152388, 139.85542448602735, 140.56254093314368, 141.04655020350862, 140.99342114439773, 139.008036048575, 134.15681201229387, 126.19571217273533, 114.70739349956412, 99.80370905723468, 84.27489426021907, 69.74025324468755, 56.601932361558454, 46.85474946626007, 40.549502511774165, 36.291073790864694, 34.05889752477011, 35.04820231137772, 38.995143877851795, 47.536082447713035, 62.27500554507492, 83.75474237071701, 109.3545441273855, 137.2469222876268, 164.1083584514178, 186.7308775705316, 201.77222328695515, 210.10882134333747, 212.13816757253198, 207.696356778197, 197.47831013575563, 184.12729261556942, 168.42527415487496, 151.4047516154187, 136.7170537899073, 126.27507674638726, 119.56982721806826, 116.21032023633204, 115.68030737030529, 115.94683369649422, 115.91130612487663, 115.57372465545257, 114.93408928822203, 113.99240002318501, 113.18160970178192, 113.04683995604, 113.58809078595922, 114.80536219153964, 116.6986541727812, 119.02393262822484, 119.99115312072126, 119.6003156502704, 117.85142021687236, 114.74446682052705, 110.77632455416658, 108.1760075080056, 106.94351568204411, 107.07884907628211, 108.5588818447762, 110.32260646655868, 110.93113094426901, 110.34135495383802, 108.16856338519142, 104.05950013663994, 98.70068833280047, 92.77856393876979, 85.6264400679486, 76.94650420191957, 67.51464608189136, 57.22685279844192, 46.358944670179454, 38.10358515596771, 34.946141379603304, 36.8291219604842, 44.1594611729946, 56.655481997246284, 71.04253962870567, 85.05847983427725, 98.01926978801308, 109.43775028181759, 119.11768022225151, 126.9126080085335, 133.00556635950582, 137.7385716881425, 140.8275958030362, 142.24382986453622, 142.9210472334265, 143.13552992865525, 142.88727795022243, 142.1312037855598, 140.90814376195667, 139.22699147208465, 137.08774691594377, 133.44012860550788, 128.8933377552361, 122.96666828165114, 114.66343584799681, 103.98364045427304, 93.0278450765321, 81.84321298336312, 74.23209647623943, 78.51023754277018, 96.90448885390319, 128.36456892161237, 172.6434539005366, 223.81959662171158, 265.46063138294187, 289.7919729691791, 296.8136213804233, 286.52557661667436, 261.5809380538182, 234.03837595608348, 212.89160377797998, 198.32044845037436, 190.37623074501488, 190.12772534000575, 194.4529085651417, 199.78069847275674, 205.7514412011178, 211.85263291115916, 215.7797876998822, 217.34537480038847, 216.29993688951413, 212.45163370091922, 206.6715105974903, 200.36221535091187, 194.24323881694755, 189.06369771643836, 186.29071561496198, 185.9125216744782, 188.00482778938468, 192.2078885317996, 198.14714554130245, 204.00335327835728, 208.98032882790622, 212.3501268663859, 213.17514854280407, 211.41670078140365, 207.40317732122926, 201.2018917917186, 193.54078951643507, 186.29506819736298, 179.54211398601691, 174.07271774663684, 170.9844398878271, 170.86288461743158, 172.77045308445813, 176.66845221314946, 182.1614865713856, 188.04676882623156, 191.90249817800378, 193.20435410330876, 191.81699921798923, 186.99092623778023, 177.63936478832096, 166.1211585218835, 153.48494848525485, 140.00140944674965, 127.52649359810103, 121.15565780852583, 120.30914746100136, 125.1942466028772, 135.83213992729202, 150.7594449035743, 164.3729039231019, 176.38826309203736, 185.3642905454836, 191.08482509331708, 193.90680435874103, 195.84147345847987, 196.90071752641583, 197.77218658011412, 197.3299162695467, 195.57390659471358, 192.52348830507833, 188.26548927599862, 183.9196464535583, 181.84236497951335, 182.1343256878048, 184.87586173998295, 190.6490601708478, 197.84807422520907, 205.22002104365274, 212.5635389582968, 219.71796164604058, 225.51911503728417, 229.88541363392355, 232.0892189190443, 231.47664232448014, 228.1060082221873, 221.61902440303592, 212.1841614385797, 200.64374241907262, 188.5069061487291, 175.81767020673743, 164.45679308095728, 155.2666787958802, 149.45889349076577, 146.27886776350687, 145.7511717161386, 147.41840134676534, 152.6392711376092, 159.88925434424374, 169.16835096666904, 180.38340322162622, 192.5684606250471, 202.163690187996, 208.48332515185788, 210.72746724191512, 208.942695349797, 203.5000168998296, 195.63630486782912, 187.09711118056333, 179.48223238746783, 172.79166848854265, 167.2493551192038, 163.9413793167549, 162.1271866835013, 161.4005324184077, 162.20673903337732, 164.4497116993743, 169.69021457372918, 177.85050054020658, 188.14326265144146, 199.67785588362736, 211.73998078040063, 218.73661214533564, 220.70662353655013, 218.04366842772646, 211.1480198544099, 201.2579937008876, 193.08177009859835, 185.77836793576964, 178.51997682575868, 171.09468050585883, 163.10525362189802, 153.1748279418711, 142.67915885015987, 133.25761832552752, 125.46918732246168, 119.17120177853874, 114.84592016924356, 112.26477506112994, 112.96598356565008, 117.07893118650327, 124.7174544276744, 135.7861391892557, 150.60211965730457, 165.60915943132235, 180.5679399255097, 195.25078813189685, 209.19982654134918, 221.16837310342527, 230.38058457841157, 235.50240177600858, 236.64766120020113, 234.04530160555655, 228.31866401729553, 222.30527809657391, 218.96058552668032, 218.46832323105096, 227.42130881396778, 245.8195422754308, 272.8077948909239, 305.2776316123546, 339.6204403830358, 361.35269526621335, 368.31058070609345, 357.88565731402826, 328.81533684020053, 285.3479886882529, 236.67221191884855, 186.9409555475458, 141.37109835164034, 105.07775767432804, 79.65508318463965, 64.04064297183979, 57.42645360044488, 58.262755353161836, 65.66269318264679, 77.42255279628539, 93.07206648819069, 110.58103256627476, 127.96274192391475, 142.08518050180783, 152.52048900633733, 159.65149701179655, 164.86420475298226, 169.72022561800392, 175.75804539496008, 183.191593730659, 191.7241793243541, 200.12448878161905, 206.70813541842958, 210.83158655449498, 211.89718219373694, 209.8573440010071, 203.6983845387952, 194.3991925955937, 182.80591428362945, 170.11386959505893, 156.71555642994926, 146.02707931819307, 138.47883468742646, 133.53050928088982, 130.84496757427866, 130.2259606175594, 129.9654361457857, 129.7276820803561, 130.0337473648179, 131.49944870815727, 135.55315672796092, 142.1948714242289, 151.37905198427305, 162.89791219206268, 174.82207169428096, 184.2947892557543, 191.31606487648264, 196.26762708933524, 199.04399938282603, 200.78225717565516, 202.91077108540935, 205.42954111208869, 207.8068706581831, 210.34173591764107, 212.85061633296272, 216.1775080749434, 220.32241114358317, 225.27184435432673, 230.5333318307632, 234.99188173839343, 236.92119686473794, 236.27227977019476, 232.7817161797578, 225.74422401227602, 217.52037714469867, 209.03078148959884, 200.37343192618025, 192.44759631553535, 187.15631469637748, 183.3534893376132, 180.40555342545787, 178.26350952030967, 176.4723200096993, 174.03036860300872, 170.18971598666266, 166.1408860464527, 161.88387878237876, 157.42950155828706, 152.87794691670024, 149.7250934847689, 147.37567931959728, 145.82970442118537, 145.1551736914919, 145.3019908592555, 145.5222166109009, 145.81585094642816, 146.492618644185, 147.44629973135244, 148.67689420793036, 150.36970665080602, 152.8176585267566, 156.52554728134356, 161.83863891249538, 169.1331400694975, 178.03844159857567, 187.96870056617541, 196.98514774613017, 204.4222767255134, 209.52767420575387, 212.33442200845352, 212.2851509011469, 209.56961651540828, 204.54087065517325, 197.57511996972727, 188.97680996964112, 181.33933038130994, 177.1088263900162, 176.28529799575998, 178.86874519854118, 184.70694524307456, 188.75849351288758, 187.97262912597378, 182.34935208233313, 171.88866238196562, 156.59056002487122, 144.8133015279351, 141.09001697524172, 147.33693965053203, 167.15326989982017, 200.53900772310618, 239.77974852666938, 279.84939090064523, 315.028843269145, 337.13445986442446, 345.8064715330579, 343.6879304969572, 332.27136492434204, 316.99500549708307, 303.42854271262644, 292.29151487782383, 283.606724881162, 278.45327317257625, 275.4470399810267, 273.90300854489834, 273.70393787315066, 275.34700065505865, 279.6212062742346, 286.60602262127355, 296.6391652840372, 309.9882377513489, 327.1730612795596, 345.61953613830866, 364.66622130903954, 382.73700067372596, 398.5690827675654, 409.205310074034, 414.4400958614025, 414.6041606389491, 410.63567662273726, 403.2873036263612, 394.8795963797421, 388.3902821376881, 383.81936090019906, 381.1668326672751, 380.4326974389163, 380.89397274721193, 380.5997746835279, 379.48878079343615, 376.87537305428594, 372.4725207082291, 366.9426034845536, 360.9756953633429, 352.1950364059769, 340.95324374989497, 327.488902250157, 310.498954962621, 289.5208010629618, 269.49192779170403, 251.76395494192107, 236.72080507703726, 226.30621235604917, 221.84383674028103, 221.17299733098116, 224.66030046065057, 231.97026946865307, 242.1310372754903, 253.08571501324542, 264.1568547844691, 272.5740061084351, 277.9009919888534, 279.62122848572596, 278.66475485722265, 275.3702950520682, 271.1230743106256, 266.79544662547534, 263.4205798766131, 260.998474064039, 259.62885385365456, 259.3117192454599, 259.63751279321053, 260.29381160482683, 261.28061568030887, 262.42583107368154, 263.72945778494494, 265.18634534959983, 266.3881716718093, 267.33493675157325, 268.2019858226413, 268.9893188850134, 269.6273782175514, 270.32032486817354, 270.2324139857188, 269.1303382770065, 266.98228133591556, 262.0325728157532, 253.3106688792829, 242.48805922882693, 229.4060145219452, 214.12816757087975, 199.91005918327, 188.53920618014092, 179.17986371033135, 171.7236111415832, 165.68991535011912, 159.97897166751432, 153.92737796342857, 147.53513423786185, 141.88626380116443, 140.02103923583863, 141.58748338495454, 146.432025395064, 154.55466526616692, 165.10462152220347, 173.34749915113778, 179.2832981529698, 182.91201852769956, 184.6588227588706, 185.2457784568077, 187.23924324993953, 190.63921713826608, 195.5444033622672, 201.23515961385442, 206.76006911211556, 211.39242422028155, 215.1322249383524, 217.7820647853683, 219.50574092687557, 220.69684201337404, 221.04405069813174, 220.13801989368397, 215.98222222835813, 207.9373921421071, 195.6914601186952, 180.35310983693856, 162.74103547176622, 147.04569824748347, 136.4502755786442, 132.3395058523069, 134.48079450216153, 142.46479444074362, 154.19627505590043, 165.61872617566115, 174.8972116508268, 181.50856626090982, 185.45279000591023, 186.66241529505538, 186.78072310480792, 186.57164863733323, 186.29677450287508, 185.90521115626538, 185.53189377904937, 185.17682237122708, 184.8383290090093, 184.51641369239604, 184.31285551172354, 183.93486797027964, 183.3824510680643, 182.65560480507753, 181.7543291813193, 181.07737729751972, 181.8068102246932, 185.25871246239188, 191.45716191711597, 200.40215858886546, 211.1944171858439, 222.14577228384118, 230.62405488375308, 236.3419378902084, 237.9914658003595, 236.02228126010465, 231.165807528579, 224.7381291053348, 217.2423528794031, 211.40855531596912, 207.23673641503296, 204.72689617659458, 203.87903460065394, 202.6679320961613, 199.54649455488007, 194.51472197681034, 187.57261436195205, 178.72017171030524, 171.52743138585987, 166.5984421382181, 164.60910384354597, 165.55941650184354, 169.44938011311083, 174.49431920424746, 179.74689025425573, 183.7474516473958, 186.49600338366767, 187.99254546307134, 188.23707788560682, 187.7491737342218, 187.4204166118978, 187.8132254940371, 188.9671341497116, 191.3641697633624, 194.9017220036831, 199.4719490072659, 203.98929595341966, 208.3746953040007, 211.942067528986, 215.0602291277884, 217.7291801004081, 220.43277316202006, 223.21054208169625, 225.6710530323995, 227.10696009905692, 227.51826328166845, 226.9442457103477, 225.38490738509472, 223.80240121479906, 222.65888502832746, 221.9543588256799, 221.68882260685646, 221.2187286138179, 219.23165730029103, 215.6552854187224, 210.48961296911202, 203.73463995145983, 197.06196798908005, 192.8530682007434, 192.295282016398, 195.40768210310966, 202.70825386951873, 213.97684340562736, 227.23861150139095, 240.1188752969134, 252.57948945806297, 263.584483167559, 269.27618813169863, 269.1395546869096, 264.3619242631399, 254.9623695274553, 241.45887588849632, 230.07676485920084, 223.92629607019467, 223.03971888067903, 227.4170332906539, 237.0582393001192, 247.46959283596334, 254.74848115442492, 258.5686489636301, 258.93009626357883, 255.83282305427124, 250.59502023430665, 245.9078166587251, 242.84121426741385, 241.72077054972206, 242.5464855056498, 245.318359135197, 249.19178626130793, 252.27374058220917, 253.9131071192024, 252.39093084120827, 247.7072117482267, 239.86194984025764, 230.5749662726759, 220.17181853483066, 212.59882527516044, 208.22378684163567, 208.0741529451942, 211.8584825917304, 222.29044617952871, 236.63427150495036, 254.15435787205456, 272.7958058589656, 291.80989392799165, 305.76928128256384, 314.59072108442297, 318.64201368153954, 318.9506087848514, 315.7915004031377, 311.87835893468286, 308.3944952285644, 304.94100543200267, 301.3626133970417, 296.4053615786396, 289.3503792121659, 278.15023822117166, 262.6476569928448, 242.87876087558348, 221.9646197503498, 201.34297514640448, 183.9253723675683, 172.60869358953875, 167.9533719180555, 168.76757011885783, 174.98259090474653, 186.29637016575634, 198.98260380004675, 212.433154142768, 225.6222263150058, 237.24947336189797, 244.48021641426922, 248.7000628638541, 250.07586806932062, 249.1205294701259, 246.4411303261446, 244.2418171494749, 242.30699758912766, 240.16006124470314, 237.72701899547232, 235.09405127654819, 231.74206750347935, 228.10225237824426, 224.58335339884516, 221.3333488067401, 218.30914838437258, 215.30654300183787, 212.1099403081467, 209.3316765084939, 206.8977624821504, 204.80819822911613, 203.46323080836714, 202.86286021990333, 202.46261316092745, 202.26248963143948, 202.02782202325164, 201.55848680687603, 200.85448398231262, 199.9158135495613, 200.1870984995874, 203.24580658472763, 209.091937804982, 217.72549216035054, 229.1464696508332, 239.47771137652435, 245.59747075191498, 247.0950161579107, 243.92778418590933, 236.09577483591107, 227.01943693483182, 219.14853082943023, 213.83248907314942, 211.37049177665995, 212.6750111574169, 216.75813429744485, 222.94904729134515, 229.78107988951484, 236.78356209641908, 242.20999300561607, 246.0603726171058, 249.21794668355423, 255.09668410739815, 266.4015730578798, 285.3507595957403, 313.43417029600926, 348.8853136533547, 385.9321904934116, 419.5929310646282, 445.9747059064631, 461.58479779563794, 464.15662243808856, 455.5447851369139, 438.24022076788987, 415.7977926083672, 390.733155379814, 370.592262162552, 357.4379015247486, 351.2700734664038, 351.04262679895373, 354.27824405873605, 352.9852238005141, 344.8590599593935, 329.84843614980616, 306.61134471164706, 278.81049163451706, 251.33036918969714, 226.71720093880867, 205.0736196529878, 189.0836406524447, 177.31521212930653, 170.01819319964812, 166.18749388714775, 166.05205692632177, 168.7656000345411, 174.29803427831717, 181.32941366733627, 189.66912245340964, 198.78049921566017, 207.71893190595583, 217.88800323803366, 229.48232237939396, 242.44160009485245, 258.4801138320252, 277.99991155474765, 297.48549226215664, 315.41275525249637, 330.7960500107941, 340.69576942411203, 345.15875219927057, 346.0608047655995, 345.8864548977813, 346.2102253016238, 348.5138479432417, 352.797322822635, 358.2435001144904, 362.0910863270734, 363.2810600155688, 361.8134211799766, 357.68816982029693, 352.5396055871564, 347.76872578655195, 344.281595284211, 342.07821408013353, 341.1585821743196, 340.54864667350284, 339.8517528136497, 338.6148681618963, 336.43145723806947, 333.3015200421691, 329.5388627101012, 325.3247210776932, 320.65909514494473, 316.3550558722028, 312.41260325946723, 308.6748342387851, 306.57497130048057, 306.46949209810276, 308.6661563542318, 313.1706345380859, 320.6832571399693, 330.2360339816272, 341.1160097559611, 352.049239313534, 363.4588749402903, 373.9442556556213, 382.3321896214913, 388.9791544914495, 394.2901549561099, 397.401875035929, 398.5436110792272, 398.61630957368527, 397.07100462988103, 394.06234150388417, 388.39143499649754, 380.7417132740087, 371.1309061892293, 360.5723619571412, 349.0660805777444, 339.878818898195, 331.64701545001657, 324.36180530680303, 317.6433897068572, 311.49176865017887, 304.2735637131902, 298.5497499019167, 294.32032721635824, 291.6043471698198, 290.7659064662305, 291.9367768276708, 294.49856696018634, 299.1088810726074, 305.56044901059977, 313.12507736630494, 321.53922269556216, 329.1780435162467, 334.3099403372912, 335.392890491835, 332.7833057438962, 326.56361128728753, 316.4383809921919, 303.8865587162726, 292.19945994758535, 281.3924545659523, 271.5642356279088, 265.01451476786957, 262.5085392161292, 262.40065122865974, 264.8834156553636, 269.90748596797295, 276.3020018761167, 281.73801426633315, 286.2155231386223, 289.33402891335743, 292.568741332306, 298.196758704461, 307.515191523332, 320.5240397889191, 337.42355329103555, 355.2633125461463, 369.4891209362655, 379.82426408985515, 386.2687420069154, 388.6397733531868, 388.41256787043716, 387.8642238676593, 386.72032600842084, 381.52390681321265, 372.3497936367685, 358.74879875078483, 340.72092215526163, 318.9941130323288, 300.3519861839983, 285.1645838234966, 274.3302814074309, 266.15778391289047, 260.27913517729826, 253.30974045190982, 245.01615858310774, 234.94920184258854, 226.49146027617377, 220.92399746934524, 218.91485737947733, 220.44974334654685, 225.68505004639957, 233.12254967699695, 240.63140230074518, 247.17968243862737, 252.79545641845417, 257.24832726323245, 260.3792044587847, 262.9593458958076, 265.4105805221854, 267.6337244221153, 269.62038752204603, 271.109549187716, 271.56138306258526, 270.9758891466536, 268.6600536548118, 264.69626896175816, 258.91821416084156, 252.55308836403893, 245.60089157135047, 239.3663169955451, 233.84936463662285, 229.83676416301986, 226.8434354135041, 224.86937838807552, 225.98847767685584, 230.68668694624785, 238.65639010700312, 250.77303451514285, 269.2842314967367, 289.1272560453995, 309.5378851901094, 330.380892564237, 349.7633941954572, 363.19016743163075, 373.0951404292934, 378.9054318111609, 380.63424807558215, 377.10975444873674, 369.95398380617166, 358.1513259564894, 342.44167561282035, 321.85611621939825, 300.6312012961891, 280.01808774423864, 262.0479959463414, 247.74891180227934, 239.13121067020538, 234.07661579013666, 231.95954871155027, 231.6123673505394, 231.95633413398053, 231.48415514600282, 230.02181713629517, 227.5693201048576, 224.1429815805953, 219.90480920562317, 215.7585696417425, 212.05228938957535, 208.78596844912167, 206.38306744009975, 204.84358636250954, 205.60638715558312, 208.9187170458883, 214.84594670906714, 223.12069745043527, 234.5992857657052, 247.08171752639691, 259.7254717787525, 272.3998071714879, 285.0639885817442, 296.0053830180962, 303.81106237327424, 308.4171176550171, 309.88891953896683, 308.2468355865528, 304.34718229348766, 300.59261565262113, 297.95347460223354, 296.4297591423249, 296.02146927289516, 295.82735554724775, 294.97870080600626, 292.9903355800303, 289.86225986932016, 285.59447367387554, 282.1497937156213, 279.8091854045667, 278.6306054211018, 278.98327893200633, 280.86720593728023, 283.0605013331637, 285.4781563757138, 288.00425770415023, 289.90035498491375, 291.15408975173557, 291.9161770310249, 292.13987598575613, 291.60426477634485, 290.27107364109423, 287.144846133327, 282.244787857288, 275.6962544755821, 268.05700302815904, 260.14202337197094, 253.9793037991793, 249.61237626305075, 247.3772822318217, 247.0985892413758, 248.37543709672178, 250.18765241864435, 252.4289656963656, 254.53451794184667, 256.5221499796793, 258.43729813796256, 260.77972803997613, 263.60257444110914, 266.9868861839315, 270.7305804215072, 274.72288029718345, 277.9642545644009, 281.9922365657956, 287.8870426203269, 295.58553172465855, 305.14640967385924, 317.06944209120877, 327.57264541374997, 334.3390238992514, 334.73619986170496, 328.7641733011105, 316.42294421746817, 300.66387970878407, 284.7951013001782, 274.4657556959934, 269.8049458259636, 270.8126716900889, 276.7905318397227, 284.2384626149477, 290.4702324950338, 295.4058432169046, 299.0452947805601, 301.4311000062344, 304.0900121391102, 307.0305560184689, 310.1896008095967, 316.53512312600463, 325.99559292388955, 340.27782267247295, 358.7084437171624, 381.13730079768004, 401.6284406870032, 420.16332329588124, 433.3071177535051, 441.46886280961036, 444.5102301974227, 444.7867835265055, 441.3527068880718, 435.24018987488296, 425.2029524272554, 411.3583242125781, 393.92671376438796, 373.60254088965996, 350.94358128887825, 328.65000869478814, 307.39469285793956, 288.5740557231017, 273.7538310540293, 263.99432249013313, 257.9467320997945, 255.72725734741852, 256.42401989118434, 258.89828251405527, 262.73291334244493, 267.8852978127578, 273.72899469289916, 280.51068373536367, 288.10899262779594, 296.69698877249584, 306.25222768570285, 316.21593026990234, 325.8171196224932, 333.44938812546286, 337.2526447272254, 337.2560572714863, 334.0409589006202, 328.63063024380415, 325.5178906553841, 325.91951299435294, 330.05927880910076, 338.3243777429565, 349.5317645727047, 359.3177198874864, 367.63976102594603, 374.6699640229789, 380.04662892140783, 384.4011711575922, 390.13857355668114, 397.5362916164253, 405.5788286218641, 413.8280155573646, 422.2040667734236, 428.3492133040472, 431.2342607111006, 431.2474847133887, 428.2461625811822, 422.1450939850841, 414.8312326589019, 406.45180650710654, 396.63516891834365, 384.85902580125173, 371.37356346412787, 355.70673573976507, 339.62681346533446, 325.85750220675163, 318.36173712770795, 317.01442507405505, 321.81556604579276, 331.622641572341, 343.02863400741245, 351.7573483855599, 357.80878470678334, 361.182942971083, 361.6229345265183, 359.92351061399887, 357.6144423348426, 354.6957296890495, 350.24849823131626, 344.37574568020347, 338.0606137630848, 331.16428733097, 323.68676638385904, 317.8765793976786, 314.3067075369975, 313.73912339163536, 316.2658566858034, 321.88690741950165, 328.86184197678665, 336.76325948060037, 343.0840730239299, 346.15017785403654, 345.9615739709204, 342.92904095990144, 337.06088946684866, 329.61066294526864, 323.97351175629785, 320.1989223201714, 318.2868946368892, 318.2374287064514, 320.0505245288579, 321.59007674320645, 322.34986734638414, 322.32989633839105, 321.53016371922706, 319.9506694888923, 318.07642551389097, 317.84923593299027, 319.2691007461901, 322.3360199534904, 327.0499935548914, 333.01495915661815, 335.72282878929656, 335.1736024529265, 331.3672801475081, 324.3038618730413, 314.2895269024421, 306.3046013478085, 300.8716985960576, 298.84484786046715, 300.22404914103697, 304.93720372041474, 311.014878221612, 317.4118458707945, 321.80053722983007, 324.01984889290867, 323.92705564208217, 321.404063118776, 316.9734847099071, 312.72837165190697, 308.99093075639564, 306.04661245926957, 305.04808273667044, 306.0539426560694, 309.5590349053777, 316.9720464969852, 328.15025221294354, 341.17897121181875, 355.941001358669, 370.0724939711475, 379.4055063311616, 383.9400384387112, 384.480852573771, 381.086549803812, 375.141766395063, 370.2730165291099, 365.1079431282134, 359.11663618101784, 352.1219254436155, 343.5773773808052, 332.45463011089373, 321.4983977893602, 312.030909305718, 304.5820810377404, 301.0135962150318, 302.32643488241104, 307.6897477194774, 316.86214361909686, 329.31530055744577, 342.5597288308083, 354.59346834954664, 361.73260718315186, 362.39565601755424, 355.51155543517336, 341.70525575282954, 321.9777370153418, 302.07229981171133, 283.76154832380865, 270.55260117220234, 264.57929777524356, 267.2946944411101, 276.76323399062653, 292.33034340723566, 310.8694895401087, 329.35543877990966, 344.88207851028267, 356.11873025791635, 363.06539402281084, 366.31581685117726, 367.17554134312275, 367.09762380682514, 366.74740347894016, 366.12488035946774, 363.3820293156531, 357.11487572687406, 347.30479230895673, 333.9194843135664, 316.9011013323021, 300.3562004565984, 286.6650939710932, 275.4672109443602, 266.68518349304213, 260.4347124339409, 254.98817198110874, 249.79686142713595, 245.77976186742384, 243.33299844016946, 242.51355843749093, 243.39385342831824, 245.5664764850766, 248.47890367901294, 252.18984592002872, 256.79973417565526, 261.64653854574374, 267.06761860017684, 272.94649905683224, 278.4483279564183, 283.02773026231495, 287.0157209245964, 289.5850253363267, 290.849387644065, 291.0342118961772, 290.4696044612328, 289.1555653392319, 287.6536449182727, 286.27453424642033, 286.3779428415561, 289.1099924494273, 294.74082152898325, 303.53888991434695, 315.5280827725574, 329.11804840641616, 342.0165433244281, 353.683290608695, 363.39763983081787, 369.7847026996662, 372.2802057402092, 371.7013141420513, 368.22563835666097, 362.00044426946, 353.10620866369425, 342.53347608717485, 330.20985788964725, 316.3204100860725, 301.29125133400646, 287.49427337465374, 275.7897137039898, 267.30921929095234, 261.9602621280611, 259.52978288653816, 258.6568540667789, 258.5498837738649, 258.8008375536183, 260.8084790163008, 265.74202444742673, 274.72257040732285, 288.5672366540866, 306.6314885709927, 326.11779893751753, 344.68773518263254, 360.0277906570749, 369.7976066880644, 374.31945058396377, 374.9920859550346, 372.9847290867914, 369.45413330386555, 367.22084216834077, 366.96466821187335, 369.18165137094854, 374.0068715235254, 381.6635403926023, 391.81612814624594, 403.1050097211443, 414.5381052443266, 425.8452549598745, 436.47697853086476, 441.20453100599906, 440.7077249169338, 435.4826002001543, 425.6642367336196, 411.6819600221821, 400.51675003121755, 392.1951486703349, 386.7171559395342, 384.16045067605427, 384.421975988968, 384.9571460818103, 385.86105592911144, 387.47673917947367, 389.6488381584196, 393.3362758027746, 398.26454434613237, 404.1638281106059, 410.3480597989901, 416.89491824852377, 421.88655758555643, 425.36372124365744, 427.3445259635525, 427.81542934141646, 426.7764313772491, 425.13563517089835, 422.9573810221197, 420.5017930369583, 418.48202332026966, 417.1652256864876, 416.6530398095674, 416.9454656895088, 417.91244127328935, 418.40056740796734, 417.1595588602892, 414.1385957932774, 409.33767820693186, 402.75680610125266, 396.11655912067744, 391.6653569848747, 389.465517630832, 389.5170410585492, 391.81992726802633, 395.50963876113934, 398.7715765405757, 401.29495224315195, 403.0797658688681, 404.12601741772386, 404.9358096203413, 406.0584531731837, 407.92857099165616, 410.66320309031346, 414.26234946915565, 417.4172140861315, 420.12779694124083, 422.2079455452749, 423.4235798691242, 423.63978986771326, 423.01193817735975, 421.5400247980636, 419.22404972982497, 416.18105298719837, 412.71306289298525, 410.0043027058106, 408.26218750674843, 407.4867172957991, 407.7908207358762, 408.9751713166035, 410.50970630858836, 411.97959554968173, 413.85051657718816, 415.92454370009057, 418.47167987843966, 421.62730236778134, 425.59882624919027, 429.7088535583833, 434.07299374861094, 438.21565736507335, 441.67994319691934, 444.4658512441488, 446.5311448234147, 447.7399254581684, 448.32998787580937, 447.44672871766636, 443.2527336962929, 435.2733266142819, 423.6605355144774, 408.4143603968796, 391.7629580896026, 377.38115716753884, 367.1790137687951, 361.1287957423145, 359.4817888459671, 361.1239146656957, 364.29825586569353, 367.728309914006, 371.53656995870887, 376.3162888004289, 382.07053461982446, 388.9684916904122, 396.78975197208456, 406.0686369706944, 414.8647838113787, 423.1720561328201, 430.4105945330681, 436.8536646744391, 440.989104739455, 443.91273904448263, 445.5982290582152, 446.22391726870666, 445.553765684692, 443.52089557290765, 439.5461275555579, 433.2268460897948, 424.86672255310066, 414.9168069200956, 405.18353130363244, 396.6848307442148, 390.22578482012375, 385.5641858109444, 382.6327139793474, 380.37046020765933, 378.47909254826055, 376.6739253209938, 375.1357025892059, 373.75895418624333, 373.1035128631197, 373.0289549047469, 373.4469086678146, 374.3015497412122, 375.59287812493966, 376.6743075007098, 377.57891258290425, 378.81102950646584, 380.9160918291941, 383.9406747543092, 388.2340949305707, 393.7302029292151, 399.4203264803572, 404.10995560670864, 407.70593990182925, 409.8960104828343, 410.71324206410554, 410.66197078058576, 410.35672543120035, 409.86159265178173, 409.4865001473816, 409.23144791799984, 409.09643596363634, 409.08146428429126, 409.06466261718987, 407.052754723061, 402.6427200323143, 395.7456081595129, 386.23120842379933, 374.2823599607112, 363.5338690032454, 354.61698772971977, 347.7096169110082, 343.0721779088258, 340.63455750997247, 338.7993854390495, 338.14402098363723, 339.5420007447423, 343.6769797132627, 351.8032164216693, 363.77402230922877, 378.15300454418804, 393.0151891536601, 406.7328447941335, 416.33164430847955, 421.16642477216135, 421.50338719800004, 418.00209236921785, 411.47640595757076, 404.08710540337813, 397.124516555714, 391.3178412598272, 387.43387584175076, 385.5171782731221, 385.1284643474529, 385.74656148029464, 387.00686874902283, 388.4451496631608, 389.97228827943366, 391.5687011951444, 393.07047767655257, 394.47761772365806, 395.68953150712434, 396.7507769985891, 398.2665126787468, 400.65987498454626, 403.9308639159878, 408.12081431514673, 413.2297261820231, 418.05497242726295, 421.33471958306274, 422.8855594324062, 422.2550565382932, 418.4902515277172, 410.75717434476763, 399.5757882763746, 385.3129097565709, 368.720849899901, 351.705527452379, 337.16129667650205, 326.2167972821977, 318.68862105244966, 314.5367917532524, 312.80835001159886, 312.12169631160214, 311.7068450809115, 311.5637963195269, 311.66527769939256, 312.01128922050844, 312.6716496376649, 314.05493209048984, 316.1611365789832, 318.71577596272135, 321.71885024170433, 323.2595983336682, 322.5208739593575, 319.5026771187721, 314.3064891917779, 306.9323101783747, 300.7902848863886, 296.57519405585754, 296.143349762727, 299.5766105092393, 306.87497629539456, 316.3135438297092, 327.3198979113628, 334.25656876887564, 336.8917666112782, 335.2254914385702, 329.3419196811906, 319.52725893954937, 311.48751252877, 306.7657149304898, 308.50494415169857, 316.853358039236, 331.81095659310233, 351.45289419370846, 373.06103724171874, 390.49150807101233, 403.4479909879095, 411.9304859924102, 415.93899308451455, 416.79854267064803, 417.3676560620829, 417.89350399937314, 418.39788739777697, 418.8808062572945, 419.3422605779259, 420.27594141726615, 422.13658655749293, 424.9898040930205, 429.83516335819934, 437.26897522230786, 446.5884142658732, 456.5869662433978, 467.06801222027815, 474.6407144716851, 477.57480348755143, 476.22169197761343, 471.23414351147704, 462.72136801407333, 456.71440836693375, 455.35440743783386, 459.1918366568542, 468.22669602399503, 482.458985539256, 496.00085527665954, 507.37558455374256, 515.4362948925885, 519.2469222693373, 517.8785098932685, 512.2606660564703, 501.50937742974554, 486.2669866786854, 468.40562185101055, 449.783196528161, 435.6138741742217, 429.47921297442343, 431.7317543196638, 441.8503959765923, 458.90618115448905, 476.42618435369576, 491.30793855776955, 501.5967820508773, 506.4627912519991, 505.9059661611347, 502.50353595669486, 498.2491245160346, 496.166664333447, 497.7287844323852, 503.1171879269964, 512.3318748172805, 523.134514580297, 533.7473193403493, 542.0549546315511, 546.9208077264886, 547.1472753359947, 543.7053445913917, 536.703206363806, 526.8579693470572, 513.9297125981809, 500.3136426955124, 486.2165402020855, 471.506914559473, 456.53150670818997, 442.8817940953398, 429.07600100154997, 415.1451006911957, 401.2378044595716, 387.68230604301806, 375.6533487151397, 365.7192773363459, 357.7905162093754, 352.8588630998987, 350.5899418701838, 350.13782889089657, 352.23967627092225, 356.89548401026093, 363.21073101268684, 371.18541727819996, 378.01731976683527, 381.44283581740814, 381.46196542991856, 378.1620604155125, 371.54312077418984, 365.7618150036853, 362.854605484508, 362.8214922166581, 365.6624752001353, 371.3775544349398, 377.2948025142706, 382.0468045043034, 385.6335604050381, 388.51278737721793, 390.7200021772427, 393.60779306171315, 398.13118565961315, 404.2901799709428, 411.1693416742157, 418.7540369254689, 426.103490181032, 432.45411153375716, 437.995462267738, 443.1852595437176, 448.164175725229, 452.5203419559272, 457.0063154127605, 461.40202616635077, 466.3964233422172, 472.3153902122879, 479.69980763400827, 487.92794166622855, 496.882721145019, 505.4862247165991, 512.7192740628855, 518.559345389794, 523.1376199095017, 526.5902046016276, 529.0060947971754, 530.2364435371926, 529.6717933435228, 527.3742853985891, 522.3112230883387, 514.1101050139262, 502.53884313702395, 488.0983654215667, 470.9039705139061, 453.0439973013373, 435.8634023760671, 421.69446827964555, 411.58485500049585, 405.4769132154421, 402.56022945144923, 402.1623254124137, 402.8635933242508, 403.7986172067003, 404.78376414717866, 405.3515222044611, 405.50189137854767, 404.48581841899147, 401.0387787275169, 395.5280381292909, 388.1873525949258, 379.0167221244216, 369.76393114654707, 365.6511676538794, 368.22696911937146, 378.0233122504207, 395.04019704702716, 418.0292144029939, 440.33956212799364, 458.50689945095337, 470.69181479459274, 475.53409557312443, 473.14579825776934, 465.69070121974954, 454.6141158468664, 441.46473457723357, 427.27128920525195, 412.30902264422986, 397.6366557437749, 383.82790649935697, 371.4784560254633, 362.9069339459721, 358.01524909956953, 356.60866369121084, 358.6660913807517, 364.15578359226834, 371.6090327329519, 382.0930424880614, 394.9385675980785, 409.6140627478218, 424.2064096352413, 436.6055797946122, 444.5103847483395, 447.5207722367445, 445.9025149174182, 440.7635409856166, 434.47312819842216, 429.0266253703136, 426.56839881844644, 427.4911848943666, 431.9786663968231, 440.3593962226536, 451.07977595794927, 462.05509763671114, 472.49988855584786, 482.0467831178613, 488.47424242776725, 492.52887297042406, 494.2607436617471, 493.7092701307115, 490.8251491635905, 486.7327282667171, 481.4442026632275, 476.20946759785204, 471.73516451573215, 468.6083981053175, 467.01378085438466, 467.1499669786572, 468.3920088557701, 470.3865857631525, 472.658446361332, 474.8202615963011, 476.0865480301906, 475.93693707886735, 474.3423369587389, 471.4238803333043, 467.37523172956674, 463.1531025413895, 459.7982299370393, 457.3687974837001, 456.61312975212604, 457.5312267423172, 459.7471086709879, 462.7404069540047, 466.4820298077757, 469.4753280907926, 471.1503817474495, 471.47868359468004, 470.06743040353706, 466.8266559178763, 462.46812470765224, 457.89514996707686, 453.0021357896421, 446.7831702582491, 439.26085605005653, 430.3197874838159, 419.30614815152705, 406.5166514154045, 395.1415307964888, 385.4054797088946, 377.6489895175183, 372.7494336781474, 370.574188816917, 369.3710946001755, 370.79709300622847, 374.7533867821795, 380.68302574163545, 388.5260360870448, 398.64103016524473, 407.39946434936536, 414.62188177950264, 421.30267854711593, 427.47184155098074, 432.2940719733964, 437.583641627798, 443.2293159155167, 447.24230265363394, 449.62260184214983, 450.80754191058577, 450.79712285894175, 450.20185604854214, 450.83654934775876, 452.70120275659167, 455.79581627504075, 460.1203899031061, 464.47597891242987, 467.221759749187, 468.3577324133773, 467.8838969050009, 466.0190247965561, 464.133323575268, 463.04720501804934, 462.76066912489995, 463.16327515890146, 462.46299309746, 460.0790118490045, 456.011331413535, 450.25995179105155, 443.1438959115136, 437.73078794062656, 435.4845351231758, 436.4882672228238, 440.7419842395708, 447.93896252425355, 456.4449575232487, 463.3321547469865, 468.21750912562874, 471.04131173757906, 471.2669278518859, 468.57810663672126, 462.87065673918175, 454.0673029104655, 442.28746299376564, 428.80068936322976, 414.51924135854006, 403.3583385981873, 396.5306919647511, 394.1212082553501, 395.72844188421936, 401.7460917957618, 412.0822254465455, 426.0734733745388, 445.85009778762, 472.288416441752, 503.68877210844676, 534.7617654098483, 564.9215503847023, 589.0996966763522, 604.1053365364584, 609.8055803707014, 609.1316782038975, 602.2373571164969, 592.0219885377994, 581.3050322913608, 571.8924938541126, 563.9902949773925, 558.02464179188, 553.7333868241425, 550.3622904409478, 546.5409223469594, 541.653645145993, 535.0813030992557, 527.1942667309488, 517.9914173109679, 508.72647183344645, 500.42478333941426, 494.08591697535286, 489.9500002422633, 489.3793103265936, 491.6672732611795, 496.3012125255062, 502.73448059643897, 509.87379397033754, 514.9977156675878, 516.708593258614, 514.7383892567219, 509.00498607221436, 500.4802492421525, 490.6046920905584, 480.31088214047423, 470.1348943652879, 460.24096394439465, 450.35905832027385, 440.58378204248913, 433.24295735368315, 428.0685467671621, 425.0559865029193, 424.26866634774404, 425.38890600249397, 426.5539681310596, 427.8571838055399, 429.9123457195916, 432.6505614444595, 437.26927836433305, 445.1643454929834, 457.48639389591256, 472.77517675673397, 490.7901524918698, 509.4541066320838, 525.5636306513417, 536.5374692023431, 543.144522598127, 545.5681160373301, 544.9651142885753, 543.8005842140917, 543.4118190235795, 544.0183002692545, 545.6200279511168, 547.9804870011576, 550.1963732179354, 551.4725590820651, 551.1843746354207, 547.7285788265858, 541.2234291895647, 531.9147225758325, 521.3927140241594, 510.2482987941481, 501.7475325993332, 497.5555812160528, 497.7696301177777, 501.78819752036134, 510.0105503940824, 520.7143004661209, 530.5691161838018, 539.3806266001826, 546.1399877739605, 549.127496028774, 548.402724975325, 545.5808966811463, 540.1720886048327, 533.613051423274, 527.3970413841182, 520.9691752525683, 514.1616135311014, 507.3161102515667, 499.05900750779585, 489.0405267200199, 478.2797885980598, 467.2623032633748, 457.06588669689665, 449.1947497693107, 445.7437656969863, 446.339342764673, 450.7137540572385, 458.034537576779, 466.8979424260566, 473.2792083590233, 477.08768961590584, 478.32338619670435, 476.98629810141864, 474.3797162538388, 472.68406930622734, 471.99908006227406, 472.8832949313866, 475.3367139135652, 478.7076915469148, 482.4776489552886, 485.6817703065156, 484.185859953415, 477.9899178959869, 467.09394413423126, 451.7572281062215, 435.15021605002556, 426.4605989325528, 427.4969847960616, 439.2467019304262, 461.73370039753684, 493.0075656842952, 524.8613171607938, 553.6777387425157, 577.4821738497128, 596.2267223586044, 606.7387144681604, 611.1639097271626, 611.2292849559444, 607.9221684443797, 600.9359235419772, 593.3322052815204, 585.4004062095677, 575.0121118425047, 562.167322180331, 547.5272106478106, 531.5267388117171, 514.9346596349012, 502.83321822827537, 495.4809149009012, 493.950908959559, 498.8080790508235, 510.7736831718771, 526.4113433641678, 545.5319241051686, 565.6356315523079, 585.5400080505104, 602.8703434678176, 618.6076762155916, 632.9128001252849, 646.5740648245535, 658.2858395680107, 668.8068721557662, 677.2870989467268, 682.9383376093222, 686.0682863393051, 687.4077344949403, 687.2424202534744, 685.2189461722471, 681.8953359648424, 677.2715896312595, 671.6724891282217, 665.0980344557286, 659.1214280649207, 653.4317722591529, 648.0290670384253, 643.6033128651894, 640.154509739445, 636.896056435622, 634.4497483470109, 632.8155854736117, 632.486631925686, 634.6495092164816, 642.540345945451, 656.3856548917284, 676.3524665106368, 700.6939986011805, 727.0370081368646, 748.9092379187838, 765.235866995379, 775.682834456004, 780.9010637728582, 780.9220437477978, 777.6140246654455, 770.8557062906086, 758.1220299611987, 737.9280281309423, 710.054543882998, 677.234491081761, 640.7852911491759, 606.0911223200645, 576.3250138869116, 555.3935205828373, 542.3770933270904, 536.6170214086985, 535.4358880461177, 539.5986248255018, 546.974202932739, 556.9179538159276, 571.4604491571505, 590.942710903009, 610.6618465887107, 629.8305813585864, 648.6257133406815, 663.2774721762504, 673.0597969644898, 680.4948156882901, 685.9505264213956, 689.2282354534315, 691.7757684556404, 691.9819369611115, 689.0145506360601, 682.6770790044153, 673.0000694356016, 659.5983069279052, 645.4875265145578, 631.5351546586592, 615.540364104291, 597.3069356389176, 576.3221391238545, 552.1909807376925, 524.992451553607, 499.77191898509363, 477.4934123452026, 461.2121559327056, 451.19512811161064, 449.09190708746274, 452.90954848790136, 462.0707725084051, 474.7096729960211, 490.1397442901501, 505.20259320938004, 518.994097102478, 531.5142559694443, 544.6072653528223, 557.7800190426316, 572.469510601685, 591.2013150820717, 614.7164285133474, 639.4144851879774, 665.0247455560659, 690.4104534021028, 710.684727303618, 724.3655752015005, 731.8507457713339, 731.9260774651846, 724.8311065647947, 709.5977289067083, 686.9600694780668, 658.1013547084242, 626.53828155343, 593.6391148007584, 566.5968711845529, 545.4252927896421, 532.4155522911693, 525.8093012113101, 524.9512914766656, 526.5542851375739, 530.611411151621, 534.336305459934, 537.8369858519425, 541.171335733087, 544.0641354535448, 546.5153850133156, 550.2209173937135, 555.2374003482755, 561.9038438250323, 569.8615879955079, 579.1109673878774, 587.6010889266788, 594.8945637365487, 600.0241844233512, 603.2955133956909, 603.6227591442041, 601.327386209043, 596.6820979226043, 590.2283240051142, 581.9854456370326, 574.1240422525616, 568.1871294947946, 564.5348806573857, 563.1672957403355, 564.0840402154679, 566.1999916297702, 568.5410646897901, 570.3869128082188, 571.7317154942044, 572.427785459299, 572.4751227035027, 571.5439988589974, 569.6049139689864, 566.6695090151726, 563.0331585744526, 558.6958626468263, 555.3535907400949, 553.785689355161, 553.9863380011734, 555.8078493896835, 559.1980622277488, 562.0476179420906, 563.1577691873048, 561.2962986479513, 556.3484192024047, 548.4184534365506, 538.4963300365605, 528.2001971923411, 520.1092766563975, 514.5195440303171, 511.912578556837, 512.2457020653908, 514.7153550032974, 519.1795744747707, 525.3907706415126, 532.2814624321627, 539.7134869779744, 547.6758151944036, 553.6107346312209, 557.2103738361157, 559.0084733447676, 559.0741145915506, 557.4128121187363, 555.4100790813202, 554.1415721585487, 553.6072913504219, 555.2733765774556, 559.13982783965, 565.17380289096, 572.3468068048335, 580.3430502124969, 586.2302532729188, 590.0084159860991, 591.5352824222182, 590.9759762921587, 588.7463617386127, 585.2257048241399, 580.1779545191665, 573.8538869881309, 565.4443156705718, 554.5075735718889, 543.0489286019549, 531.5404828199179, 519.9074516863595, 510.2541703634117, 503.4866656987244, 499.0163754410275, 496.84551120945196, 497.9899898789854, 501.3166501950268, 506.2539297191755, 512.3106446561778, 518.7564835623499, 523.078953938755, 524.36103091167, 521.8121637821525, 515.59370454481, 506.38435323429, 496.06013713790145, 486.4551060030907, 479.16461685060693, 473.6159788667296, 469.6896757183953, 466.9725200470436, 464.5474869789515, 462.88094593984897, 463.5056055439636, 466.68631112807634, 474.4839349960509, 487.1603962734745, 502.89447587419426, 520.338829010917, 538.8294951968699, 554.2447298243253, 566.0048952425241, 573.9602858154569, 578.2002494352016, 579.0567663451446, 578.5907088491497, 577.0710892255244, 574.6345063773051, 571.8769184939096, 568.6498050741005, 564.2702508063087, 558.3388215464526, 551.4057802062529, 543.1731476910011, 533.9379650031713, 525.0660627659028, 517.5582274075366, 512.568439796449, 510.2903573613179, 510.71331353263713, 513.3667586576204, 517.6980466988723, 522.0295387165604, 526.524433053818, 530.9070218471827, 534.7525737790882, 538.0610888495341, 542.5951155430679, 547.4472848873893, 552.7554508142296, 558.7319789823719, 563.6701159185128, 565.05853880384, 563.1998661404352, 558.0940979282984, 549.3818961531384, 540.4767677615616, 532.7447320658198, 526.6815784627273, 522.0630722057037, 519.6078893233325, 516.165036037177, 511.4729667491398, 504.36902960348453, 495.09823014420374, 483.21231667418806, 471.1316950241495, 459.2323125198221, 449.3436834758647, 441.6485010440324, 437.51846547902886, 437.5935572187157, 442.19619823126123, 450.9370630349064, 464.56459646265523, 480.8226194724668, 497.13598563553296, 512.4839036897827, 525.9148162010708, 535.6247505805917, 542.9232613977904, 549.4419787562056, 556.7175499305501, 566.9415870060673, 580.8663088138519, 597.5989107364136, 615.60506277889, 631.546468926941, 640.2729096552142, 641.134907729737, 634.6891056687111, 621.655795986159, 605.6328157797305, 591.5248392031306, 580.830510675381, 573.5498301964815, 569.9656134517681, 568.7920677142538, 569.4401186614007, 571.3730148357729, 575.7442562289561, 582.4124349982827, 591.3775511437523, 600.7778714004007, 608.8418649745597, 613.2625318830583, 614.0398721258967, 611.1738857030748, 605.7959792079233, 600.8115019715517, 597.3739539855455, 595.4833352499047, 595.1396457646291, 595.9188612155691, 596.4745923132075, 596.8068390575444, 596.9156014485797, 596.7267903490407, 596.486834504682, 596.1957339155039, 595.8534885815059, 594.9363585261232, 593.5840882058186, 591.3378603079123, 585.8384940170758, 577.0859893333093, 566.1278262097429, 552.9067831452695, 540.7471607589638, 534.3673206814831, 533.7672629128272, 538.4232474764309, 548.3268405542118, 557.8649506545822, 564.6783969622137, 568.7671794771062, 570.13129819926, 569.731303042616, 571.5003439227739, 578.1523351048323, 589.6872765887911, 606.10516837465, 625.4849106345268, 645.1268123644038, 659.5743245674996, 667.7987971459927, 669.7885984849376, 665.9847287138571, 656.773262753867, 644.210099560253, 630.2709317018916, 613.5124872042009, 594.4570930739062, 575.45602364076, 557.911470924141, 540.9580000797612, 527.5170499016205, 518.1026157295228, 511.9283613205148, 508.278830431615, 507.07241543605653, 506.8425811293667, 507.0725549494344, 506.0514704725741, 503.779327698786, 500.23163990841397, 495.3103496176773, 489.01545682657627, 483.35900563902777, 476.2263284673239, 467.66639875077664, 457.875331456947, 446.85312658583496, 435.32500172943185, 428.41885546615504, 426.15444106288186, 429.98692300376405, 441.4867076515925, 460.23034304763667, 482.4362614339754, 508.08229850479637, 534.2524574531152, 557.8892101697132, 575.9517377116167, 589.0781502102471, 597.5099854790527, 602.6450221887033, 605.8870974692636, 609.0874692128768, 612.3763640553332, 615.160239341041, 617.178640277884, 618.5148514822249, 619.136440932223, 619.0434086278785, 618.2870989246034, 616.5441868055411, 613.814672270692, 610.4170385848989, 606.3512857481621, 602.6235385659054, 600.331348992839, 599.4747170289629, 599.8944010418553, 601.3515274938163, 602.669601192029, 602.9272457719118, 602.1244612334647, 600.240710915616, 596.7024181150442, 591.9585380251526, 586.7397749845542, 581.0461289932488, 574.5164110629469, 569.0143952133919, 564.5400814445836, 560.7498622409756, 557.643737602568, 556.5965284931586, 557.5451145115662, 560.7440037684935, 566.1786998281008, 573.849202690388, 582.1715840369145, 589.1694371126006, 594.3337456960401, 597.6645097872329, 598.7024421284812, 598.0383757237887, 595.6288348771233, 591.7283276991882, 586.3196643837284, 580.3214194461391, 573.7335928864206, 568.2406580128402, 563.8426148253973, 560.2981180550282, 557.1403595361256, 554.3004202021855, 551.0385782724213, 547.2166711572629, 542.7657418455232, 537.7008321530204, 531.9404565419688, 525.9915940712214, 520.1305699199187, 515.2880931426486, 511.4566428315019, 509.4052204938562, 509.7781366449434, 513.3623711639724, 519.5547070828618, 528.3638246100766, 539.2828092864268, 550.3399460648949, 559.691413140545, 567.5346442310182, 573.8522789193846, 578.0216790855546, 580.6415158941486, 582.6240055487002, 582.6431606469047, 579.9704873063338, 575.1169412031788, 568.5886379094853, 560.3920405578758, 552.6100638860331, 546.7758231866435, 543.539041809882, 542.706430595302, 544.7697792823744, 548.8488937413856, 554.1037024413765, 559.2347586819975, 564.330726205399, 569.0087341239016, 573.1631831223918, 576.9419937497148, 580.9948893560454, 585.6460484219018, 590.8031340868037, 596.4316031250664, 602.3092383155966, 608.4360396583945, 613.316253609581, 615.7663767491574, 615.8523846946161, 613.6927044473714, 609.2873360074229, 604.2798732663996, 599.5132709942371, 594.9414704152798, 590.5134842753714, 586.2293125745119, 581.3533053305861, 577.083026707213, 573.5105942557036, 570.7379824843704, 568.8820075212769, 568.2843150754273, 570.3766905575545, 575.1130751920026, 582.4424817246157, 592.1312778992669, 603.9953784718767, 613.4240176112468, 620.0875453405894, 623.9859616599048, 625.2360826972563, 623.8644332317981, 622.1763961791296, 620.8658230171682, 619.9327137459138, 619.3770683653665, 619.1856244859493, 619.358382107662, 619.4965882050323, 618.5113715364206, 615.7900326034261, 610.9064724947623, 601.1507232669834, 586.2238078267155, 567.8569901908984, 547.2756693563341, 525.3320431455957, 507.44604744557444, 497.2574947362911, 495.4028424656525, 501.9755673621351, 520.2051965711662, 547.6426608806954, 578.5123723550385, 611.3983940514372, 644.8883735161364, 671.6710586357091, 691.2246519473634, 705.0428237659714, 712.4134251025384, 713.7134759665936, 710.8714898436028, 703.4928724048539, 690.9777156965313, 674.0169029217129, 652.0633945985504, 628.571415869538, 604.8519528548926, 584.5282715665642, 568.0710276867139, 557.5617688310858, 551.2901093980701, 548.9990301093305, 548.6408066652414, 549.9031358891082, 552.5697345122665, 556.6071485954405, 561.2184305750864, 566.4035804512038, 571.9279066143511, 573.9651651809666, 572.5320831206878, 568.0271342152873, 560.4503184647646, 550.2710190880041, 542.464968397696, 537.2937471550462, 536.7278012653786, 541.5694137483437, 551.8524185071176, 565.2895698789334, 582.6212347743246, 600.0087792192768, 616.0856924596285, 630.3149234701427, 642.9004264427192, 652.0822009289668, 659.6261771609413, 666.9564034709704, 674.7110214478638, 682.6377379758803, 690.989130175379, 699.8674558829945, 707.2835378733778, 712.0895241191126, 714.0441065571065, 712.2368967896081, 706.6678948166174, 698.4979380102452, 688.913882196749, 680.377482118731, 676.3273808320369, 676.7635783366661, 681.7343157634863, 691.0033083663011, 700.962053377978, 708.980856848959, 715.059718779244, 718.8668602891057, 720.0576111208718, 718.7525393433834, 715.558559904397, 710.4756728039129, 703.9282701750489, 695.4727523510344, 687.7198438827784, 680.6695447702808, 674.2161947764529, 668.1874587724582, 663.8152063495104, 660.271534990931, 657.5564446967196, 656.2449153493413, 656.5897727413998, 658.4990285547522, 661.6543762835478, 666.0558159277865, 670.8703684338064, 675.4998301396887, 678.8963080905048, 680.7609659564378, 681.0938037374872, 680.2584808419399, 678.6929272158227, 677.0662541724254, 676.1128707009266, 676.1380312440772, 677.1417358018773, 679.031432297615, 681.516791059639, 683.4214481643569, 684.1205463668388, 683.4066949450138, 680.8926069213893, 676.097118004332, 670.365074392761, 664.030427248284, 657.5079580150419, 652.4602825518605, 650.7587602829817, 651.7993362737176, 655.5676621646401, 661.8563472336784, 668.5020206956593, 671.3478109834296, 670.3438072902338, 665.490009616072, 656.7864179609445, 645.1210742286912, 635.6989597907293, 628.7272909429324, 624.8292530706244, 624.5825036903069, 628.8565805231617, 635.3583434681423, 644.3465985843739, 654.6232174451434, 665.6584522991252, 676.3603998145632, 687.1809552018672, 697.180500682186, 706.8857369529751, 715.6231869673804, 722.1720941360654, 725.6286680382094, 725.9589582135825, 722.8213191980533, 716.722168960643, 709.9005797905673, 701.8322583240351, 693.1172497905429, 684.2143520337388, 675.2639547710389, 665.4701079131391, 656.3354918029889, 647.6654792376809, 639.667468565707, 632.0546572171824, 623.6178673714585, 614.2399510178631, 604.7398536370713, 594.8859362334157, 585.3072320450442, 578.4220967132529, 573.8615395273443, 571.5787662162193, 572.2761544398305, 575.590233125156, 580.3118244515475, 586.4005753716273, 592.6162607947405, 597.4647866241654, 600.4400987885058, 601.5421972877615, 600.7710821219323, 598.7468658363457, 596.2165954793625, 594.192379193776, 593.4564054068748, 594.3743198814233, 597.4792869437151, 603.268815877908, 611.3183733332825, 620.5481226719302, 630.3281447689931, 639.5921109718843, 648.365233710471, 658.0099749766232, 670.589999718574, 686.5675312581279, 706.475733921579, 728.7716549967196, 750.4858083377774, 764.902060985771, 770.6787318844042, 766.4325093542125, 752.7361492932466, 728.1533749296234, 700.1330218812916, 670.6632299810375, 642.510622587791, 616.5701499018147, 598.8635831388956, 588.2071141781089, 584.0366153231319, 585.4321362156146, 591.9981890740162, 600.7392718302881, 609.8655993534339, 618.8166095636952, 626.6655798188418, 633.3085354816932, 638.0068895056661, 640.5963181962933, 641.3571025934536, 640.7526040182624, 638.8348097893099, 637.2069856974857, 636.5083292925985, 636.7388405746482, 637.898519543635, 639.9873661995584, 640.3635628532116, 639.0271095045946, 635.6140653396776, 630.1244303584609, 622.5582045609444, 615.922918620484, 610.677347100305, 608.0441560469169, 608.5715259860202, 612.3446169781943, 618.0685483523534, 624.8257709820471, 631.262775216345, 636.2832000038459, 639.7167252233912, 641.0421831085561, 640.1353519656856, 636.6665990928686, 630.7898586809079, 621.6898576348834, 609.8667732623129, 596.372603805771, 582.8561835059805, 570.1060050327375, 559.9229346970401, 552.469918621717, 547.3919490922599, 543.8646089883068, 541.4936519749602, 539.3786448967205, 538.8079115904527, 540.8463163002216, 545.5012142458561, 552.8966413737425, 563.0673495014514, 573.9680833010364, 583.2411257419957, 590.8717663846713, 597.0404087703504, 601.6775492638911, 604.9830625802399, 608.1358072346471, 611.143138446942, 613.2721412953924, 613.0238486515118, 610.456554573836, 605.5702590623654, 598.3649621170996, 589.269139172098, 581.7324797113761, 576.6934575853909, 574.9951191112795, 576.6572218307999, 581.6797657439522, 588.3045632113333, 595.7250876930633, 602.2552465548694, 607.9968051583451, 613.0159248531796, 616.6147882503628, 619.0182550929956, 621.2803294999217, 623.1837620044445, 624.5962299071861, 626.0577676986018, 627.5683753786914, 628.6605834622742, 629.3845646309284, 628.7923273053518, 626.3966217028416, 622.1099103573792, 615.2333244881072, 605.3821249560435, 593.7714408342575, 579.1609126101723, 561.7256152158246, 543.421863461643, 525.1102433891241, 508.0420882373951, 497.60234509871333, 493.90817107032007, 497.0920487274775, 507.0205937363013, 522.7057235939741, 539.4535322802691, 561.2909236383539, 588.1985089489285, 619.5824708382647, 653.8752229472926, 691.7074230035159, 724.4111796308209, 748.6250195037293, 764.645851309105, 773.6635643383036, 776.0840690870707, 776.3436585250931, 776.6897777757895, 775.6568288802612, 772.281173078504, 766.5300278643602, 758.2416635710175, 746.0575348966661, 732.6768025608883, 716.3511391934375, 694.7278427008447, 664.969087371461, 629.7919638089054, 588.1949444507105, 546.2194810666845, 508.95188436030486, 483.7288003640514, 470.2926173960441, 469.964056635111, 482.53882871199283, 508.2566697004795, 542.5496789022282, 584.1228953912819, 630.0530259291114, 674.8111023231411, 714.9264698724646, 749.6875049516864, 778.7838337433462, 805.933696393215, 832.5598510375255, 856.499361547106, 878.692099696973, 899.2399614202008, 913.2653206326604, 920.8523747142011, 923.9432841905818, 921.0221777987341, 908.6406505614939, 888.462908055357, 859.9360153291024, 824.5617086360901, 785.3768542918612, 750.0832726315714, 720.4123981019168, 700.0415273708371, 688.4245612878939, 684.0430666953168, 683.0461334255277, 684.5680442551787, 686.1541570761348, 687.3949339359121, 688.1714197624209, 686.6386373542293, 682.7965867113372, 676.7858348186816, 668.811150652504, 659.7677449795852, 655.1152159509026, 655.3321221185831, 660.5170901812528, 673.6651787842308, 694.8791683661051, 719.2112222132062, 745.7999497271543, 774.5010853746018, 799.3245118649098, 817.8548563089207, 830.9889064867697, 838.2602838890458, 838.7092482520442, 834.7629625551635, 827.8369770080957, 816.7750484988044, 802.1975042622362, 785.8433404486265, 767.3298546419147, 746.8986199344761, 727.7349785183618, 711.0501810847904, 696.3293847646713, 686.6065229235353, 683.1677229762855, 684.4203138268965, 689.3819281048654, 697.8337916274756, 705.7914802823059, 710.0682486167783, 709.4297980812771, 703.7415730155597, 693.003573419626, 680.4531354155675, 668.2430080252683, 660.4035436769492, 657.6300350931829, 659.9224822739694, 665.9446379260879, 675.3235845191417, 683.0474577451757, 687.3394902550523, 687.5965805438603, 683.7543736515157, 675.8128695780182, 666.6419338530793, 658.2474456482264, 653.6032498969996, 652.8380565195687, 655.9615098727478, 662.4941100103237, 671.646008047577, 679.2788186321579, 685.3281868039819, 689.9451743667, 692.9558367248644, 694.5101248789972, 696.6634074549353, 699.7551135429508, 703.4541864652981, 707.5848499504284, 711.8472019972968, 715.6657892016699, 718.3057708841644, 719.9374975620601, 720.7361349482057, 720.8516340431233, 720.57172154893, 720.3477915535753, 720.1798440570594, 719.9592468603371, 719.4050773340608, 718.4606840357691, 717.0700844666228, 714.694589751021, 711.5514642870537, 708.2025533334162, 704.7611597750323, 701.3222193618515, 698.9631098450757, 697.5751990256597, 696.8775642742557, 696.8135541484023, 696.4995361655845, 692.8412089004385, 683.8175848862032, 669.4286641228787, 649.674446610465, 626.0373900641432, 603.9634982371916, 588.0485775569653, 578.7016919008271, 576.2182817160168, 580.9342074542699, 590.5734119800279, 602.0072448388644, 614.4175782760533, 627.2135313971145, 638.4176027232076, 646.4663454646649, 652.1065480864415, 655.9360191012227, 658.2501989562487, 660.3249513318224, 664.6075612067718, 671.9174368324498, 684.1392828575798, 701.6474374420264, 723.861198656248, 748.1467912503698, 773.0335787385806, 794.1859179174659, 810.8551324672969, 823.1369339498564, 831.6675538561669, 835.4570538486344, 836.7056665851633, 834.1272008399962, 826.9737061550435, 814.7592213900477, 797.9611592706141, 776.1271106024026, 751.5629436851276, 726.0039605847663, 700.1266445963327, 676.440265039196, 656.0355240864278, 639.2822732954664, 625.4247801044628, 616.3363611886865, 611.0889758079862, 609.2356620132701, 610.6506177084659, 615.3855423009525, 623.1834342155431, 634.7109983802948, 649.709495258856, 666.4001481003106, 684.5303334617143, 700.8891505579746, 710.7902184601635, 714.3209445919161, 712.1201786172314, 703.9998923851234, 692.3260071045094, 681.8923427410277, 672.6421702044524, 664.8541247725873, 659.1568861903479, 654.953996476324, 651.1314633562824, 647.9286320442968, 645.539848119138, 643.650771708348, 643.1134283043345, 643.6572345126706, 645.1709059792679, 648.0090957187552, 652.8502856811901, 658.5853260502112, 665.2142168258184, 672.6782191575738, 679.8231427259899, 685.2920236309516, 687.5971726913092, 685.0532814263564, 677.1643581030746, 664.0199064631364, 645.9807311154816, 626.3615628062358, 608.5330184968126, 593.5458205036857, 582.6250057810275, 576.4059290110737, 573.8867940993096, 573.620929401528, 575.5003160586706, 578.9699241339733, 585.371663730755, 595.0980616538907, 608.3878987850678, 624.7871089969508, 645.0861439383364, 666.7070392054613, 689.1788931019393, 711.3082333548972, 732.7511653214699, 751.1296847880261, 765.9403862868758, 774.0413719631148, 774.8591155935999, 768.698261247641, 756.796312841054, 741.2556247427596, 726.2316112343949, 714.29499077942, 705.0781094737622, 698.0753044606565, 690.8353585343432, 682.0390675232954, 670.4137368340803, 656.9225974799809, 643.5402711526538, 636.9194157792429, 637.3007910684616, 644.7113967026052, 658.8109517173971, 678.3413804046191, 695.4308723108003, 709.9403586521487, 721.7989538796782, 730.6480145069454, 735.9406435508897, 740.0774880145043, 741.9763787587957, 740.7072149761335, 735.3695403963491, 725.1080677731173, 710.9795290715632, 694.4020832633081, 677.306817512598, 661.8532878462136, 651.8227721702461, 647.1159240369318, 647.2449759829993, 651.2734057846011, 658.1214354283467, 664.3033115592389, 670.8291183040908, 679.4752083805433, 691.084064991535, 706.0398103401294, 725.4058540276272, 746.0573236098852, 764.4415136516212, 778.786393672133, 788.1425084730419, 791.1281924859956, 789.2931811616411, 782.2201396737677, 770.8095939418641, 755.8080877534969, 739.9789522453709, 723.7158236905074, 711.4060771766106, 703.3238814221811, 699.3905664495982, 698.905386227287, 701.6715226187364, 706.2340286728866, 712.0445669527365, 719.303900795209, 727.010253450099, 735.2240880452825, 742.4679233951756, 749.6870496451374, 757.2025584622263, 765.1039917746, 773.3334237874307, 782.6295950935103, 791.650262839122, 798.7407904208571, 804.0037460859058, 807.3735920402951, 807.5540761022165, 805.1693616300157, 800.1169534708196, 791.744895668085, 780.1161886827364, 767.8233368783904, 754.9602563920732, 743.7924190991242, 737.4563940380059, 736.3219291133516, 740.5278293204816, 750.0271365908825, 763.6871149868848, 777.5767354154317, 789.0039604543017, 793.8892698408046, 790.752633657431, 778.2071318459982, 757.8279274557229, 733.8898516171488, 710.2467733242717, 690.163154676786, 677.0718747010815, 672.0831437180447, 673.2444201147204, 679.3462979826755, 688.5817835142482, 698.2458788312081, 706.1181632917817, 711.9167962567683, 715.0589394758717, 715.2853139353668, 713.2549585452778, 710.0780836264912, 706.9893922386187, 705.1545608822526, 704.8554301965935, 706.4302896125084, 710.2797620139814, 714.9088334983308, 720.6784829248882, 727.7762584086661, 735.538105586917, 743.5328510226407, 752.4650545849897, 760.9471512038061, 768.6570027029345, 775.9078495152677, 782.3601698628563, 787.5602766729959, 791.4518487571186, 793.9008812334853, 794.9161125537129, 794.2276651633259, 792.126817748146, 789.1136394572713, 785.6723621225977, 781.8105578388653, 778.869170063633, 777.1893964651912, 776.7712370435403, 777.3990549096663, 779.0690640161996, 781.0192271043492, 782.2259511692434, 782.6386795304968, 782.1259269378586, 780.2144892808826, 776.9654102462431, 772.0639826835743, 765.5959902316049, 756.7035927890227, 745.9770738863156, 733.4164335234834, 721.35707434271, 709.779099107696, 701.2785147444217, 696.4900359999308, 696.9246576763999, 701.9043666559871, 713.8687572273423, 730.7252768434411, 751.3763128651183, 772.9119632997078, 794.4090694235173, 810.9577832151614, 823.043975647246, 831.0786571803603, 836.3486473933062, 838.719402621283, 841.3811661165525, 844.3339378791154, 847.3974829273542, 849.9480626332576, 852.3830923479236, 851.9006588720055, 848.4896973086863, 841.2695783340239, 830.9897138659237, 817.606866010746, 803.815930304045, 789.6390365394549, 777.0675293510756, 766.8100636221941, 758.5131915934641, 752.2183883788996, 746.9943043958408, 741.1744170406491, 733.9644333365097, 726.1755269532381, 717.1711176456823, 708.7917747855279, 702.2921176875956, 697.8162511825278, 694.9751227832903, 693.8226173016012, 693.4384500516173, 693.3925496951433, 694.2539957190272, 695.997352593473, 698.4484578017668, 701.6073113439085, 706.1986344171686, 710.9401632172089, 715.8318977440292, 720.9609192559869, 726.327227753082, 730.1073690290226, 732.9424749859782, 733.3969166588505, 730.7318166120886, 724.5417863057303, 715.334195002267, 702.9413847871148, 690.2346135904693, 678.6916362834334, 668.949024266725, 661.8821096533323, 657.8262082724232, 655.3456911588999, 654.0448294297122, 653.8666459033092, 654.4059066096343, 656.4385240565298, 662.5404778508015, 672.0533423165672, 685.118518097452, 701.6711410205811, 720.1255450615403, 735.3297710067174, 747.6504095082003, 756.4562479203278, 761.7797183295377, 763.9614415173795, 765.5773970906596, 766.4970865628438, 767.036116256763, 767.1944861724171, 767.2736711302439, 767.2736711302439, 767.2157843175773, 767.1000106922442, 766.9263502542444, 766.694803003578, 766.4053689402454, 766.173821689579, 766.3057594197104, 766.9397288175513, 768.075729883102, 769.7137626163624, 771.7959402046657, 772.786110437403, 771.9777289068186, 768.5984074615577, 762.0555099245878, 752.3490362959088, 741.6257505685566, 730.8831014973059, 721.6658653848674, 715.1593145853054, 711.3477787942531, 708.8187936128569, 706.4283859764043, 703.4041677335401, 699.153502707232, 693.7077315062131, 688.0681089508655, 683.9608328728914, 681.8527737025236, 681.743931439762, 683.6186357802401, 686.9718863518723, 690.4786750966979, 693.2052611542515, 695.1516445245334, 695.7038608312798, 694.8619100744911, 692.904405178969, 690.2982165749465, 686.981758137824, 684.1829586201285, 681.9018180218595, 680.2239534943908, 679.001884639915, 677.8955382951849, 675.5107621270301, 670.9450115866645, 664.147388854952, 655.412854727507, 645.6063139046216, 636.2881423001105, 630.0369375295664, 628.7678089375191, 633.1352881247993, 643.0856111876113, 658.3766927427125, 676.954324419147, 695.8253136728412, 713.9240808975842, 730.4316630760759, 744.271855060987, 755.4274589425951, 764.1395317754584, 770.7352121207356, 776.5439844681559, 783.6110950600266, 792.4721376488249, 803.9641383795363, 817.4215109539236, 831.0042494098283, 841.814363054981, 849.5713708121708, 854.2752726813975, 856.5280840073125, 857.2498077709948, 857.7149943510564, 857.3770102203491, 854.8978761959048, 850.2775922777234, 843.5161584658048, 834.3386218832651, 824.0543345963619, 815.3392549710318, 808.1933830072747, 802.4594079038106, 796.5578146619579, 789.7302326834675, 780.6386827853714, 769.2831649676692, 755.9783008329221, 745.7461343192747, 740.659644685468, 742.4765763742374, 751.31401911017, 767.2378829645107, 786.9135073237685, 806.6670075289428, 823.9669918676312, 838.9781792853064, 851.2541280369176, 860.6723351392371, 868.9695985902687, 875.6395286482054, 880.0014182483412, 880.6293904291058, 877.708398585097, 871.1605372826876, 861.5688032962101, 848.5942767343622, 835.5351532653337, 822.8332944582405, 810.6445111803362, 800.9849061556622, 795.3301159561173, 792.0310427476068, 791.3512505445852, 793.2128339134258, 795.9706479558754, 798.8868743859842, 801.9615132037527, 804.6436215418832, 808.6145508974894, 814.2003836819242, 821.4011198951875, 830.2167595372796, 839.7746643767448, 845.9404259020399, 848.3054179190046, 846.3711978186375, 839.7805106489863, 829.9489635643088, 819.1701176127135, 807.5100864008439, 795.9657551467038, 785.2516337541969, 774.9357200833016, 766.8598503656525, 762.442013244398, 761.1837661105358, 762.7278540121137, 767.0297582346368, 771.7670135721496, 775.5398066950677, 778.348137603391, 780.3417771809369, 782.5144003678993, 787.1549993723884, 794.5879114683581, 804.8131366558082, 817.5311331671044, 829.960295598323, 837.982425916626, 841.3488839360758, 839.7018319349276, 833.1766666355703, 822.7774669038529, 811.5528683307508, 800.0001512881402, 788.8349912195096, 778.0861364477165, 770.1155456920573, 764.4740589581816, 760.8786741073321, 758.9715534177647, 758.4443196803206, 757.7188657735843, 756.2348726803409, 754.0610643062291, 751.197440651249, 748.2320078108594, 745.3332319467787, 743.3614614309233, 742.2823343104741, 742.1150484932114, 742.565600931406, 743.2970593016219, 743.8792494179003, 744.3121712802413, 744.0209083423854, 741.5768632442415, 737.1485021475279, 730.7358250522445, 722.3388319583914, 715.66724026347, 713.7330605674842, 716.9083718789728, 726.2667565756718, 741.9036894568959, 758.1554477637554, 773.4576101168224, 787.0720974316529, 797.9927281968867, 806.4091709490795, 814.9431133227681, 823.4017557171617, 832.1450192755316, 839.9645198873898, 846.1944960816827, 850.0661500026865, 851.6943141712978, 849.1262268848377, 842.9522098107398, 832.9033649775756, 820.0736431643218, 804.5809948097976, 790.3431011846286, 778.4612854421719, 770.2499423828014, 765.6336107318867, 766.531129765914, 770.9836588495706, 778.4405364061774, 788.2215211548239, 799.109506792169, 806.042099185819, 808.8209509689415, 806.8857953733442, 799.6254907987465, 788.0797606341995, 774.7251853127032, 759.8917750692171, 743.8963342437979, 727.6895801067052, 710.7760547447193, 694.8464631159826, 678.347220048903, 662.3255171679433, 646.7594324948743, 632.58086048588, 619.3391539757707, 611.3089448798806, 609.0806133309349, 613.6331408314567, 624.9210064200596, 643.734416896967, 666.4458461788705, 690.2916244853604, 712.7092066021903, 732.5482166281975, 746.9641237840248, 756.1101804719408, 761.3313982656057, 763.9090497721427, 764.4183229421332, 764.2814831652556, 765.0265585478808, 767.5253004646031, 772.0560705580923, 778.6342301562444, 787.3897749131728, 796.5420544853936, 805.2421476495632, 813.0965183692311, 820.6097756701386, 827.5249196851953, 835.6262298293539, 844.0699410586083, 852.8080405178509, 860.7852261719097, 867.5611895646495, 870.0793421869064, 868.5751056639226, 863.1857512828061, 854.36809001706, 843.0554785038576, 832.5118074870109, 823.8471787400313, 817.0983926570782, 812.4224906542124, 809.6517060232902, 807.7876503695053, 806.440306581464, 805.628853285594, 805.4110082365547, 806.0339754867998, 807.681577518545, 810.270253231967, 813.7549080388576, 817.8630650138363, 821.7037615321624, 824.6448773152999, 826.633489529078, 827.6194094610362, 827.6777518910424, 827.1883129093434, 826.3993323216891, 824.5220662734121, 821.4287089697891, 817.1636522013616, 811.290946264878, 803.8521501931285, 796.3728586845866, 789.1822570162042, 781.7041724549947, 774.2847092976929, 767.1103640219886, 759.5755630461917, 751.6662341680209, 745.0724996667589, 740.8519988066179, 739.4147405821279, 740.8599561445174, 746.1113104193049, 755.2469740784734, 767.2442870633586, 783.2410055401467, 803.5998971693924, 826.2294580238884, 848.3524539979971, 870.0307300784677, 886.6487540587771, 893.499219119253, 889.8320971712732, 877.249233994478, 855.8093360252102, 829.1021453261595, 805.0303136757566, 789.0914021192493, 783.0901045525436, 787.523185860586, 802.6677169799879, 825.4287891966272, 850.1986205766864, 873.3678233283536, 893.9428676817349, 910.031652793003, 918.5265549383382, 920.4110728321399, 917.4337066726348, 909.6450429507813, 897.8042836577973, 884.4915891257618, 871.3781286098705, 858.5762895054808, 846.9784286005703, 836.9582427565305, 830.8693743241062, 828.4213934457954, 829.82904518665, 835.0211731719417, 843.8109289709744, 852.452258168272, 860.2025216188371, 864.5631489913903, 862.8939264667657, 855.1948540449632, 842.718556281006, 825.7494351485944, 808.4718150220948, 796.9693107323542, 791.9573498344874, 793.7372466084774, 802.3350273735985, 816.7019214051725, 831.0912524219624, 844.0721653137391, 855.0420315205363, 863.9487984038066, 868.771111813647, 870.9310638618663, 870.7431280267025, 867.6825302966148, 860.0862975096336, 849.1420350681567, 835.2040883172247, 818.2880265543474, 800.0460263625725, 783.8560867043869, 769.7753709001427, 758.9350185185542, 753.9671826564031, 756.5621531660034, 766.9936031359948, 786.0280366361418, 812.9818247211142, 844.1477696123266, 874.4931150221033, 900.0925158106777, 919.1502905305467, 930.7537026784988, 934.9387783341839, 933.3421207539677, 926.9744524880086, 915.7989977454912, 900.1950629368428, 882.7616841546075, 865.1664999585562, 849.3134103881382, 837.0716484724878, 828.9166847372965, 824.5483867209739, 824.1742457936707, 827.0557570820253, 832.2583040714705, 839.5441514991608, 847.4140695367284, 853.956056616826, 858.7432224467268, 861.4964927803028, 861.7851946032594, 860.234057494531, 857.6908556702277, 853.8521611837314, 849.2685963315723, 844.5426048603507, 839.6741867700666, 834.6094113201574, 830.3820246965867, 826.6494295898568, 823.4987575496518, 820.7292575757344, 818.5230437812903, 816.3632430733375, 814.3994801657905, 812.3728527766601, 810.6848629064209, 809.4217664831692, 809.0507148032576, 811.2607897108394, 816.0519912059144, 823.2235682882452, 832.6299539659021, 843.1629933018672, 851.1626505423352, 856.1948868111492, 858.2597021083099, 857.3570964338167, 854.3019257726457, 851.1112897354899, 848.7139345797226, 847.5360585095243, 847.9252455985825, 850.2074373087842, 854.3201460671933, 859.8019505419434, 865.8004543246732, 871.620489268008, 876.262467759551, 879.6019226334545, 881.5116132859563, 882.2135047285767, 881.4836915140247, 879.5618607343233, 875.1406458393406, 868.0123224589834, 858.5853569785719, 848.0027284400627, 836.7846502718079, 827.6703227399175, 822.6248635133736, 822.3078003869459, 826.147643839656, 833.8842871573281, 844.1481302069071, 853.9139744239441, 861.2158582221979, 865.2695694807069, 864.9756030858132, 860.1659011299241, 851.8528446895689, 841.3770739749398, 830.3070132279605, 820.8416726759463, 813.6665468775258, 809.4803933195844, 808.7968898138629, 811.0138129474731, 815.0316576067573, 819.9836083972368, 825.7960459039157, 830.9647352805498, 835.1256991109924, 838.6056480422579, 841.7539608177891, 844.9242037337597, 848.9353387679575, 854.391512276442, 860.6393029651838, 867.6787108341829, 874.0022595299679, 879.7442775633979, 884.0604496385016, 888.545801578126, 893.646497556974, 900.2137540796814, 907.7783804605324, 916.7625343475127, 922.4487324940559, 923.8522372968185, 920.4168235761106, 911.8234334171208, 896.0828300174413, 878.8250343427077, 860.6810290754963, 842.8171330236197, 826.8726948465916, 817.8924713526338, 813.6956096468457, 814.1897004752904, 818.8537765977912, 827.1131952735852, 834.8575178310397, 842.7798884614112, 852.2584918176963, 865.0164062615455, 880.7579291690956, 900.8938946910679, 924.0380144449497, 947.4339191247478, 967.4488755159716, 980.699721921869, 986.4081537068605, 982.892096077078, 971.529733685518, 953.8316489902627, 935.075554131904, 915.7389473892988, 900.5722667326958, 889.575512162095, 883.4224441395529, 880.0365844613441, 879.6940104725194, 880.2663045761041, 881.8769851177858, 884.1929058625423, 887.8975507624913, 892.3562711621471, 897.4963740082743, 903.0708226094973, 909.0721489738055, 914.1333851969624, 918.4514961660701, 921.4314638231483, 923.1968065138848, 923.0112328755185, 921.5582268601678, 919.0230587671618, 915.8266383195656, 911.9689655173795, 908.9300910781369, 907.5314801540986, 907.6943597410884, 909.100469495045, 911.4982786230433, 914.1477617663171, 915.4059886203434, 915.0486015902707, 913.7121213642218, 911.8996095280463, 908.3023261490191, 903.7417363794015, 898.6665554088966, 892.7585228934431, 886.4532566661273, 884.9011778224834, 888.8573643463158, 898.9368199744924, 915.1395447070132, 936.0912412918558, 955.4172873361262, 969.1772507403775, 975.6924088411706, 974.962761638506, 967.6754577583946, 956.3634384309207, 947.3058658941133, 941.3421014796913, 938.4721451876552, 938.6959970180047, 940.9125464927693, 941.2421540032141, 939.2570518553533, 934.733774455876, 927.63863181703, 919.0147280720414, 909.2573890387615, 898.6790238610944, 887.7043274289324, 876.4006797177796, 865.9226319670119, 856.0731823264712, 847.5108155903059, 840.0551762451812, 832.7748414304172, 824.6358381794665, 817.3259866939412, 810.3021607297737, 804.4439444036152, 804.3939593669177, 815.4103457056738, 841.3089452978386, 889.4370321936258, 961.9851831816944, 1057.160933089036, 1164.3228578874973, 1273.6386484552572, 1370.4137566918882, 1446.318110512473, 1487.8097122864901, 1494.372569069989, 1468.9547964109693, 1417.4876378349434, 1346.3234039025435, 1273.8416631526648, 1211.7158255013633, 1163.8819689744587, 1133.9077373212417, 1120.8258118451931, 1117.8461980324216, 1119.1321909249, 1122.7157515097174, 1125.709683862397, 1127.957713204563, 1127.5692071979909, 1123.779083335471, 1116.3333576440637, 1107.0133001415222, 1094.5865568048005, 1088.0398906367864, 1081.5549717658203"
    # gaze_t = "18.5, 10245.800000000745, 10320.400000002235, 10382.60000000149, 10429.10000000149, 10481.10000000149, 10549.20000000298, 10602.800000000745, 10647.300000000745, 10688.400000002235, 10733.300000000745, 10764.70000000298, 10795.5, 10828.800000000745, 10860.400000002235, 10891.60000000149, 10924.900000002235, 10960.400000002235, 10991.10000000149, 11020.20000000298, 11071.10000000149, 11104.5, 11135.300000000745, 11184.900000002235, 11211.5, 11243.900000002235, 11272.900000002235, 11304.800000000745, 11332.900000002235, 11361.900000002235, 11388.60000000149, 11426.800000000745, 11453.300000000745, 11489.900000002235, 11520.800000000745, 11551.0, 11579.800000000745, 11608.400000002235, 11641.400000002235, 11669.900000002235, 11702.60000000149, 11733.10000000149, 11761.10000000149, 11792.5, 11823.20000000298, 11857.0, 11891.70000000298, 11922.300000000745, 11948.400000002235, 11974.0, 12007.60000000149, 12038.20000000298, 12071.400000002235, 12099.5, 12132.60000000149, 12161.70000000298, 12192.800000000745, 12220.300000000745, 12249.60000000149, 12281.900000002235, 12314.70000000298, 12341.10000000149, 12375.800000000745, 12407.60000000149, 12445.20000000298, 12472.800000000745, 12501.70000000298, 12531.5, 12563.800000000745, 12591.400000002235, 12629.300000000745, 12660.70000000298, 12696.0, 12722.0, 12748.70000000298, 12776.70000000298, 12811.20000000298, 12837.60000000149, 12867.5, 12900.900000002235, 12930.400000002235, 12960.0, 12988.300000000745, 13023.400000002235, 13051.900000002235, 13079.5, 13115.20000000298, 13144.400000002235, 13173.10000000149, 13204.0, 13235.300000000745, 13263.0, 13300.70000000298, 13327.300000000745, 13360.900000002235, 13386.900000002235, 13427.300000000745, 13455.5, 13486.20000000298, 13520.5, 13565.5, 13593.70000000298, 13625.400000002235, 13672.60000000149, 13700.800000000745, 13732.300000000745, 13764.5, 13792.10000000149, 13819.900000002235, 13854.900000002235, 13887.10000000149, 13935.5, 13979.900000002235, 14011.300000000745, 14059.60000000149, 14090.10000000149, 14119.0, 14156.70000000298, 14187.400000002235, 14220.800000000745, 14259.300000000745, 14288.400000002235, 14319.400000002235, 14345.70000000298, 14380.70000000298, 14412.0, 14440.5, 14489.400000002235, 14519.5, 14549.70000000298, 14583.70000000298, 14615.0, 14647.5, 14675.300000000745, 14710.5, 14752.900000002235, 14787.70000000298, 14818.0, 14848.60000000149, 14875.900000002235, 14912.20000000298, 14944.60000000149, 14971.10000000149, 15005.10000000149, 15033.70000000298, 15063.60000000149, 15096.400000002235, 15128.400000002235, 15160.0, 15195.20000000298, 15222.5, 15279.10000000149, 15304.70000000298, 15340.10000000149, 15370.70000000298, 15396.900000002235, 15430.70000000298, 15463.60000000149, 15491.300000000745, 15524.20000000298, 15571.10000000149, 15600.900000002235, 15628.300000000745, 15662.60000000149, 15687.10000000149, 15726.60000000149, 15754.800000000745, 15785.300000000745, 15813.60000000149, 15847.900000002235, 15875.60000000149, 15910.60000000149, 15941.0, 15973.400000002235, 16005.10000000149, 16039.10000000149, 16065.900000002235, 16096.10000000149, 16123.10000000149, 16154.800000000745, 16186.300000000745, 16218.300000000745, 16264.300000000745, 16290.10000000149, 16324.5, 16357.900000002235, 16393.400000002235, 16421.70000000298, 16449.5, 16475.10000000149, 16511.20000000298, 16536.800000000745, 16571.5, 16602.400000002235, 16632.0, 16664.10000000149, 16694.900000002235, 16724.800000000745, 16756.900000002235, 16787.20000000298, 16819.300000000745, 16846.900000002235, 16879.5, 16907.900000002235, 16936.400000002235, 16968.70000000298, 16998.400000002235, 17025.60000000149, 17060.300000000745, 17086.60000000149, 17121.70000000298, 17151.10000000149, 17178.60000000149, 17212.300000000745, 17244.10000000149, 17272.5, 17299.70000000298, 17331.900000002235, 17378.0, 17408.800000000745, 17434.70000000298, 17472.800000000745, 17498.400000002235, 17541.60000000149, 17569.400000002235, 17602.400000002235, 17629.60000000149, 17658.900000002235, 17690.60000000149, 17721.800000000745, 17753.900000002235, 17801.5, 17831.800000000745, 17859.400000002235, 17895.800000000745, 17925.60000000149, 17971.10000000149, 18001.900000002235, 18033.70000000298, 18063.0, 18089.5, 18123.60000000149, 18155.70000000298, 18187.300000000745, 18212.60000000149, 18245.70000000298, 18273.20000000298, 18306.300000000745, 18331.70000000298, 18366.20000000298, 18394.400000002235, 18426.60000000149, 18457.400000002235, 18488.60000000149, 18523.0, 18551.10000000149, 18583.60000000149, 18610.60000000149, 18646.10000000149, 18690.0, 18716.900000002235, 18749.800000000745, 18782.800000000745, 18829.60000000149, 18860.10000000149, 18886.60000000149, 18922.400000002235, 18950.70000000298, 19000.70000000298, 19029.800000000745, 19062.300000000745, 19093.5, 19124.0, 19172.0, 19199.800000000745, 19232.60000000149, 19265.400000002235, 19289.60000000149, 19327.10000000149, 19357.20000000298, 19383.20000000298, 19419.400000002235, 19444.70000000298, 19481.800000000745, 19529.60000000149, 19558.900000002235, 19590.20000000298, 19636.0, 19668.20000000298, 19699.70000000298, 19727.5, 19760.900000002235, 19793.300000000745, 19825.20000000298, 19870.20000000298, 19900.10000000149, 19946.60000000149, 19973.60000000149, 20007.0, 20055.400000002235, 20086.800000000745, 20119.20000000298, 20146.800000000745, 20178.10000000149, 20210.300000000745, 20238.800000000745, 20270.70000000298, 20301.300000000745, 20331.400000002235, 20366.60000000149, 20394.20000000298, 20427.400000002235, 20453.5, 20491.20000000298, 20517.800000000745, 20548.20000000298, 20583.70000000298, 20611.900000002235, 20638.70000000298, 20672.300000000745, 20705.60000000149, 20737.20000000298, 20766.0, 20792.60000000149, 20838.0, 20864.70000000298, 20900.0, 20926.400000002235, 20960.70000000298, 20991.70000000298, 21021.5, 21056.10000000149, 21083.10000000149, 21116.400000002235, 21144.300000000745, 21179.0, 21208.800000000745, 21256.800000000745, 21282.5, 21319.5, 21365.70000000298, 21391.400000002235, 21427.10000000149, 21455.20000000298, 21488.70000000298, 21521.0, 21552.900000002235, 21590.20000000298, 21626.20000000298, 21673.60000000149, 21706.5, 21738.5, 21769.70000000298, 21802.60000000149, 21834.70000000298, 21865.900000002235, 21894.70000000298, 21925.10000000149, 21957.400000002235, 21986.900000002235, 22031.0, 22063.0, 22109.400000002235, 22139.20000000298, 22170.70000000298, 22196.20000000298, 22232.60000000149, 22261.10000000149, 22291.70000000298, 22325.5, 22355.400000002235, 22402.70000000298, 22432.400000002235, 22462.10000000149, 22489.20000000298, 22523.300000000745, 22553.5, 22599.60000000149, 22646.400000002235, 22677.900000002235, 22706.800000000745, 22733.70000000298, 22770.300000000745, 22798.0, 22824.300000000745, 22861.10000000149, 22891.70000000298, 22920.70000000298, 22954.0, 22982.10000000149, 23017.400000002235, 23045.300000000745, 23077.70000000298, 23106.5, 23137.5, 23164.400000002235, 23199.900000002235, 23227.800000000745, 23262.20000000298, 23307.5, 23335.5, 23367.800000000745, 23396.400000002235, 23431.400000002235, 23462.60000000149, 23495.20000000298, 23524.900000002235, 23562.800000000745, 23597.400000002235, 23634.0, 23663.800000000745, 23693.10000000149, 23727.800000000745, 23756.800000000745, 23787.10000000149, 23814.400000002235, 23849.5, 23879.300000000745, 23907.60000000149, 23938.5, 23984.60000000149, 24013.900000002235, 24047.800000000745, 24075.0, 24109.20000000298, 24155.900000002235, 24189.0, 24219.400000002235, 24253.300000000745, 24280.70000000298, 24312.20000000298, 24339.60000000149, 24369.5, 24396.60000000149, 24431.70000000298, 24458.60000000149, 24493.900000002235, 24524.0, 24571.5, 24600.900000002235, 24628.5, 24664.20000000298, 24707.900000002235, 24739.300000000745, 24764.800000000745, 24799.900000002235, 24827.0, 24858.900000002235, 24890.20000000298, 24928.70000000298, 24955.900000002235, 24983.70000000298, 25010.70000000298, 25045.10000000149, 25071.60000000149, 25111.5, 25140.900000002235, 25171.10000000149, 25209.60000000149, 25255.900000002235, 25286.70000000298, 25316.800000000745, 25348.10000000149, 25377.300000000745, 25411.10000000149, 25454.800000000745, 25486.300000000745, 25516.900000002235, 25549.5, 25576.400000002235, 25611.0, 25642.900000002235, 25690.800000000745, 25721.60000000149, 25749.400000002235, 25786.5, 25817.60000000149, 25845.10000000149, 25878.300000000745, 25907.70000000298, 25935.900000002235, 25963.900000002235, 25998.20000000298, 26028.900000002235, 26054.70000000298, 26090.5, 26135.400000002235, 26168.20000000298, 26210.0, 26241.800000000745, 26271.20000000298, 26301.300000000745, 26328.0, 26365.0, 26397.0, 26427.10000000149, 26463.10000000149, 26491.20000000298, 26520.10000000149, 26548.800000000745, 26582.300000000745, 26642.20000000298, 26670.800000000745, 26718.300000000745, 26765.300000000745, 26798.400000002235, 26826.10000000149, 26859.60000000149, 26886.800000000745, 26923.70000000298, 26956.800000000745, 26983.60000000149, 27010.5, 27043.300000000745, 27082.900000002235, 27122.900000002235, 27149.5, 27177.70000000298, 27211.800000000745, 27239.5, 27271.60000000149, 27298.400000002235, 27336.5, 27363.900000002235, 27396.20000000298, 27442.0, 27472.0, 27501.900000002235, 27548.5, 27580.400000002235, 27608.10000000149, 27641.0, 27667.0, 27705.10000000149, 27748.900000002235, 27775.300000000745, 27812.0, 27838.400000002235, 27876.70000000298, 27904.5, 27930.800000000745, 27966.0, 27994.60000000149, 28025.60000000149, 28072.300000000745, 28119.60000000149, 28166.400000002235, 28198.900000002235, 28227.10000000149, 28258.10000000149, 28282.900000002235, 28324.70000000298, 28352.5, 28381.5, 28408.0, 28445.5, 28490.0, 28534.800000000745, 28583.300000000745, 28626.800000000745, 28661.300000000745, 28722.60000000149, 28769.0, 28810.400000002235, 28839.70000000298, 28878.900000002235, 28906.0, 28950.10000000149, 28980.300000000745, 29011.60000000149, 29045.60000000149, 29072.400000002235, 29104.70000000298, 29131.800000000745, 29167.400000002235, 29191.900000002235, 29227.900000002235, 29259.0, 29306.70000000298, 29333.800000000745, 29368.800000000745, 29400.800000000745, 29428.70000000298, 29463.10000000149, 29494.10000000149, 29520.70000000298, 29559.5, 29589.70000000298, 29620.70000000298, 29653.5, 29685.800000000745, 29715.400000002235, 29746.70000000298, 29777.300000000745, 29824.60000000149, 29863.900000002235, 29906.60000000149, 29947.5, 29979.800000000745, 30014.10000000149, 30044.900000002235, 30087.900000002235, 30113.800000000745, 30149.60000000149, 30178.60000000149, 30212.0, 30248.400000002235, 30277.60000000149, 30316.10000000149, 30355.10000000149, 30392.0, 30437.900000002235, 30484.300000000745, 30511.300000000745, 30543.70000000298, 30575.10000000149, 30609.10000000149, 30639.0, 30666.10000000149, 30701.0, 30727.0, 30762.0, 30793.70000000298, 30824.400000002235, 30855.60000000149, 30900.10000000149, 30925.400000002235, 30962.300000000745, 30991.900000002235, 31020.10000000149, 31053.70000000298, 31084.70000000298, 31116.800000000745, 31144.10000000149, 31178.20000000298, 31222.10000000149, 31256.20000000298, 31282.70000000298, 31313.900000002235, 31345.400000002235, 31371.70000000298, 31408.900000002235, 31434.5, 31468.60000000149, 31496.0, 31531.5, 31561.0, 31589.5, 31622.20000000298, 31649.20000000298, 31682.0, 31711.400000002235, 31738.70000000298, 31773.70000000298, 31803.70000000298, 31850.5, 31878.400000002235, 31914.20000000298, 31945.800000000745, 31976.5, 32005.800000000745, 32037.70000000298, 32063.900000002235, 32098.60000000149, 32128.5, 32159.300000000745, 32196.300000000745, 32224.60000000149, 32253.0, 32281.10000000149, 32313.0, 32357.800000000745, 32391.5, 32419.5, 32450.20000000298, 32480.20000000298, 32512.0, 32539.5, 32571.5, 32603.400000002235, 32651.60000000149, 32684.20000000298, 32711.60000000149, 32742.0, 32776.60000000149, 32821.60000000149, 32848.10000000149, 32886.400000002235, 32932.5, 32961.5, 32988.0, 33024.10000000149, 33053.10000000149, 33098.20000000298, 33126.10000000149, 33160.70000000298, 33188.800000000745, 33223.0, 33268.5, 33295.10000000149, 33329.400000002235, 33362.20000000298, 33387.900000002235, 33421.400000002235, 33451.20000000298, 33481.300000000745, 33513.400000002235, 33541.10000000149, 33574.800000000745, 33601.800000000745, 33638.900000002235, 33665.800000000745, 33699.20000000298, 33728.800000000745, 33755.800000000745, 33790.20000000298, 33816.0, 33850.400000002235, 33882.800000000745, 33910.800000000745, 33942.300000000745, 33989.60000000149, 34020.900000002235, 34054.5, 34082.5, 34116.10000000149, 34143.60000000149, 34176.900000002235, 34204.900000002235, 34235.5, 34262.5, 34296.800000000745, 34343.900000002235, 34383.60000000149, 34411.900000002235, 34440.10000000149, 34467.70000000298, 34496.5, 34531.20000000298, 34561.70000000298, 34589.800000000745, 34620.400000002235, 34646.70000000298, 34685.900000002235, 34712.5, 34744.400000002235, 34775.70000000298, 34804.5, 34839.800000000745, 34869.20000000298, 34900.70000000298, 34924.60000000149, 34959.20000000298, 34987.20000000298, 35019.10000000149, 35051.800000000745, 35096.400000002235, 35127.10000000149, 35158.900000002235, 35189.0, 35219.400000002235, 35250.0, 35282.10000000149, 35311.10000000149, 35338.900000002235, 35372.5, 35404.900000002235, 35430.70000000298, 35464.800000000745, 35493.70000000298, 35527.300000000745, 35571.60000000149, 35602.800000000745, 35633.20000000298, 35662.0, 35696.60000000149, 35723.0, 35749.300000000745, 35778.300000000745, 35823.300000000745, 35868.20000000298, 35914.10000000149, 35944.300000000745, 35972.300000000745, 36007.10000000149, 36050.20000000298, 36076.900000002235, 36110.60000000149, 36138.5, 36173.5, 36203.300000000745, 36249.900000002235, 36278.400000002235, 36312.0, 36343.10000000149, 36382.0, 36412.300000000745, 36438.900000002235, 36468.300000000745, 36499.900000002235, 36529.800000000745, 36560.20000000298, 36604.5, 36632.800000000745, 36666.900000002235, 36711.70000000298, 36738.0, 36774.10000000149, 36803.900000002235, 36833.900000002235, 36859.10000000149, 36883.60000000149, 36913.60000000149, 36938.5, 36975.5, 37008.5, 37037.400000002235, 37082.20000000298, 37109.10000000149, 37142.900000002235, 37173.300000000745, 37205.70000000298, 37237.10000000149, 37285.5, 37332.400000002235, 37365.5, 37410.60000000149, 37436.800000000745, 37469.20000000298, 37502.300000000745, 37534.900000002235, 37580.300000000745, 37605.900000002235, 37641.800000000745, 37671.70000000298, 37699.20000000298, 37733.300000000745, 37761.900000002235, 37792.5, 37821.0, 37853.300000000745, 37879.10000000149, 37914.10000000149, 37943.70000000298, 37970.400000002235, 38002.400000002235, 38035.900000002235, 38062.60000000149, 38097.800000000745, 38129.0, 38155.300000000745, 38188.60000000149, 38219.10000000149, 38251.5, 38283.70000000298, 38311.0, 38343.900000002235, 38373.70000000298, 38406.300000000745, 38450.0, 38480.0, 38507.10000000149, 38543.300000000745, 38581.60000000149, 38611.10000000149, 38639.400000002235, 38666.5, 38694.5, 38730.300000000745, 38776.0, 38807.10000000149, 38849.900000002235, 38876.60000000149, 38919.0, 38945.800000000745, 38975.20000000298, 39003.0, 39035.60000000149, 39066.10000000149, 39092.800000000745, 39125.60000000149, 39165.300000000745, 39193.60000000149, 39220.70000000298, 39255.20000000298, 39290.900000002235, 39322.400000002235, 39369.400000002235, 39400.800000000745, 39427.60000000149, 39459.900000002235, 39507.300000000745, 39535.10000000149, 39571.900000002235, 39613.20000000298, 39643.800000000745, 39672.70000000298, 39703.20000000298, 39730.20000000298, 39765.70000000298, 39795.900000002235, 39822.800000000745, 39857.300000000745, 39903.20000000298, 39947.0, 39976.900000002235, 40004.5, 40038.20000000298, 40065.400000002235, 40102.60000000149, 40151.60000000149, 40180.60000000149, 40211.900000002235, 40238.900000002235, 40277.400000002235, 40314.900000002235, 40344.70000000298, 40372.900000002235, 40401.60000000149, 40427.5, 40464.800000000745, 40491.10000000149, 40524.10000000149, 40568.20000000298, 40594.900000002235, 40630.5, 40659.800000000745, 40706.60000000149, 40731.400000002235, 40757.900000002235, 40786.5, 40815.60000000149, 40846.900000002235, 40875.400000002235, 40906.800000000745, 40940.5, 40985.400000002235, 41016.10000000149, 41045.400000002235, 41072.300000000745, 41105.800000000745, 41139.400000002235, 41165.5, 41197.20000000298, 41229.20000000298, 41257.300000000745, 41290.900000002235, 41321.0, 41352.70000000298, 41398.60000000149, 41430.10000000149, 41457.0, 41484.5, 41520.20000000298, 41554.60000000149, 41582.800000000745, 41609.60000000149, 41637.10000000149, 41671.300000000745, 41701.300000000745, 41730.60000000149, 41763.800000000745, 41792.800000000745, 41818.10000000149, 41842.5, 41871.0, 41894.900000002235, 41932.300000000745, 41958.60000000149, 41994.300000000745, 42025.60000000149, 42058.400000002235, 42087.60000000149, 42134.0, 42162.5, 42196.5, 42227.300000000745, 42258.900000002235, 42291.10000000149, 42320.70000000298, 42351.0, 42382.70000000298, 42409.70000000298, 42444.20000000298, 42476.300000000745, 42503.900000002235, 42539.5, 42568.900000002235, 42600.300000000745, 42628.400000002235, 42662.0, 42706.0, 42731.70000000298, 42765.900000002235, 42799.70000000298, 42829.900000002235, 42864.400000002235, 42892.900000002235, 42921.60000000149, 42966.20000000298, 42993.5, 43026.900000002235, 43055.5, 43087.5, 43131.10000000149, 43158.5, 43187.400000002235, 43237.10000000149, 43263.70000000298, 43291.0, 43321.800000000745, 43348.900000002235, 43378.800000000745, 43406.70000000298, 43436.10000000149, 43468.900000002235, 43514.70000000298, 43546.70000000298, 43574.400000002235, 43608.10000000149, 43639.900000002235, 43677.10000000149, 43704.60000000149, 43750.60000000149, 43782.5, 43813.800000000745, 43844.900000002235, 43872.5, 43899.5, 43933.900000002235, 43961.400000002235, 43995.0, 44025.400000002235, 44065.5, 44095.0, 44122.300000000745, 44167.300000000745, 44194.800000000745, 44228.20000000298, 44254.10000000149, 44288.20000000298, 44314.900000002235, 44347.800000000745, 44376.20000000298, 44408.70000000298, 44440.0, 44476.70000000298, 44506.10000000149, 44532.900000002235, 44564.0, 44593.400000002235, 44624.900000002235, 44654.0, 44679.400000002235, 44717.5, 44749.900000002235, 44780.0, 44825.70000000298, 44857.10000000149, 44882.70000000298, 44919.10000000149, 44964.5, 44993.0, 45025.60000000149, 45056.0, 45093.0, 45120.70000000298, 45148.400000002235, 45174.0, 45211.60000000149, 45244.5, 45274.0, 45320.0, 45368.60000000149, 45414.70000000298, 45443.300000000745, 45474.10000000149, 45505.60000000149, 45551.400000002235, 45580.400000002235, 45608.5, 45641.300000000745, 45673.800000000745, 45703.10000000149, 45735.60000000149, 45784.5, 45813.70000000298, 45845.10000000149, 45877.5, 45909.900000002235, 45937.70000000298, 45969.5, 45999.5, 46026.70000000298, 46061.300000000745, 46093.70000000298, 46123.70000000298, 46155.400000002235, 46184.60000000149, 46232.900000002235, 46271.900000002235, 46298.5, 46334.0, 46363.60000000149, 46392.400000002235, 46421.300000000745, 46448.10000000149, 46480.300000000745, 46506.5, 46542.300000000745, 46570.60000000149, 46603.20000000298, 46629.300000000745, 46663.900000002235, 46693.5, 46721.400000002235, 46751.60000000149, 46800.300000000745, 46827.70000000298, 46860.20000000298, 46904.400000002235, 46951.10000000149, 46978.10000000149, 47011.70000000298, 47043.20000000298, 47071.70000000298, 47110.20000000298, 47138.5, 47167.900000002235, 47205.10000000149, 47237.20000000298, 47273.60000000149, 47319.5, 47363.5, 47394.5, 47421.0, 47456.800000000745, 47501.400000002235, 47527.5, 47563.0, 47604.60000000149, 47641.5, 47672.900000002235, 47705.5, 47732.70000000298, 47770.900000002235, 47798.300000000745, 47827.900000002235, 47858.300000000745, 47890.5, 47915.60000000149, 47952.0, 47978.0, 48010.900000002235, 48042.10000000149, 48090.800000000745, 48121.5, 48152.900000002235, 48199.800000000745, 48229.300000000745, 48260.400000002235, 48291.70000000298, 48324.800000000745, 48370.60000000149, 48400.20000000298, 48440.20000000298, 48471.10000000149, 48508.10000000149, 48533.900000002235, 48569.0, 48612.900000002235, 48660.0, 48691.900000002235, 48721.20000000298, 48752.10000000149, 48783.300000000745, 48815.10000000149, 48841.900000002235, 48874.300000000745, 48905.800000000745, 48937.60000000149, 48983.10000000149, 49009.900000002235, 49044.300000000745, 49089.0, 49118.900000002235, 49145.10000000149, 49180.70000000298, 49210.300000000745, 49240.900000002235, 49272.0, 49302.300000000745, 49346.800000000745, 49377.400000002235, 49415.5, 49453.5, 49477.900000002235, 49513.400000002235, 49546.10000000149, 49591.60000000149, 49623.300000000745, 49652.400000002235, 49686.10000000149, 49732.800000000745, 49760.0, 49795.5, 49840.400000002235, 49872.300000000745, 49897.900000002235, 49933.20000000298, 49970.900000002235, 49997.400000002235, 50025.300000000745, 50062.300000000745, 50090.10000000149, 50115.5, 50149.20000000298, 50178.400000002235, 50211.60000000149, 50254.60000000149, 50289.300000000745, 50313.60000000149, 50348.60000000149, 50379.400000002235, 50418.20000000298, 50454.900000002235, 50486.20000000298, 50512.70000000298, 50550.5, 50596.900000002235, 50625.800000000745, 50658.900000002235, 50683.5, 50722.900000002235, 50753.900000002235, 50784.800000000745, 50807.70000000298, 50835.5, 50874.10000000149, 50900.20000000298, 50935.60000000149, 50962.5, 50996.5, 51029.400000002235, 51058.70000000298, 51096.20000000298, 51123.60000000149, 51154.70000000298, 51180.60000000149, 51209.800000000745, 51236.400000002235, 51273.5, 51307.800000000745, 51335.70000000298, 51378.0, 51410.400000002235, 51437.800000000745, 51468.5, 51499.900000002235, 51530.70000000298, 51562.60000000149, 51593.70000000298, 51625.70000000298, 51657.5, 51694.60000000149, 51720.400000002235, 51764.5, 51797.0, 51823.300000000745, 51857.5, 51887.300000000745, 51935.400000002235, 51979.20000000298, 52009.60000000149, 52038.0, 52071.60000000149, 52102.20000000298, 52133.5, 52179.70000000298, 52207.70000000298, 52241.10000000149, 52266.60000000149, 52303.400000002235, 52347.400000002235, 52378.10000000149, 52421.300000000745, 52454.900000002235, 52486.0, 52514.300000000745, 52547.0, 52578.10000000149, 52608.5, 52637.400000002235, 52673.5, 52718.70000000298, 52764.400000002235, 52796.400000002235, 52824.300000000745, 52856.300000000745, 52883.60000000149, 52918.70000000298, 52963.0, 52991.900000002235, 53019.10000000149, 53054.5, 53102.5, 53127.5, 53160.300000000745, 53190.300000000745, 53227.60000000149, 53254.60000000149, 53286.60000000149, 53322.0, 53347.20000000298, 53378.20000000298, 53407.60000000149, 53437.20000000298, 53463.5, 53498.70000000298, 53530.300000000745, 53560.400000002235, 53587.10000000149, 53620.5, 53667.5, 53703.10000000149, 53743.5, 53773.10000000149, 53802.5, 53834.10000000149, 53861.20000000298, 53895.0, 53923.400000002235, 53969.0, 53999.800000000745, 54029.5, 54062.70000000298, 54097.400000002235, 54124.900000002235, 54158.400000002235, 54188.0, 54219.5, 54245.5, 54280.400000002235, 54311.5, 54357.300000000745, 54386.0, 54418.0, 54447.20000000298, 54472.300000000745, 54507.60000000149, 54539.20000000298, 54584.10000000149, 54610.300000000745, 54647.0, 54693.0, 54723.10000000149, 54751.400000002235, 54791.10000000149, 54819.800000000745, 54852.10000000149, 54878.60000000149, 54907.0, 54936.20000000298, 54965.5, 54996.20000000298, 55024.10000000149, 55058.70000000298, 55089.60000000149, 55135.400000002235, 55166.900000002235, 55199.20000000298, 55228.800000000745, 55254.800000000745, 55291.70000000298, 55323.900000002235, 55368.5, 55400.60000000149, 55448.300000000745, 55492.10000000149, 55522.20000000298, 55548.400000002235, 55583.900000002235, 55610.400000002235, 55645.70000000298, 55675.20000000298, 55722.900000002235, 55751.20000000298, 55782.60000000149, 55813.5, 55839.800000000745, 55874.0, 55904.10000000149, 55952.800000000745, 55981.5, 56012.900000002235, 56043.70000000298, 56071.5, 56105.60000000149, 56137.5, 56168.900000002235, 56201.5, 56247.900000002235, 56293.400000002235, 56323.900000002235, 56354.300000000745, 56381.60000000149, 56414.60000000149, 56442.20000000298, 56475.70000000298, 56522.60000000149, 56568.400000002235, 56613.300000000745, 56641.10000000149, 56674.60000000149, 56703.60000000149, 56740.10000000149, 56767.0, 56796.10000000149, 56821.800000000745, 56854.800000000745, 56886.400000002235, 56913.800000000745, 56948.0, 56978.10000000149, 57004.300000000745, 57038.10000000149, 57070.800000000745, 57117.20000000298, 57145.5, 57178.800000000745, 57210.10000000149, 57237.60000000149, 57271.800000000745, 57318.300000000745, 57345.300000000745, 57378.800000000745, 57404.70000000298, 57440.70000000298, 57473.900000002235, 57502.400000002235, 57531.10000000149, 57562.800000000745, 57593.60000000149, 57620.300000000745, 57657.20000000298, 57689.800000000745, 57718.0, 57743.300000000745, 57778.5, 57810.60000000149, 57837.60000000149, 57870.10000000149, 57896.0, 57931.20000000298, 57957.800000000745, 57991.400000002235, 58035.60000000149, 58064.900000002235, 58097.0, 58123.70000000298, 58156.10000000149, 58185.900000002235, 58232.70000000298, 58259.400000002235, 58295.60000000149, 58342.400000002235, 58372.5, 58419.400000002235, 58448.60000000149, 58476.20000000298, 58509.800000000745, 58544.10000000149, 58571.10000000149, 58620.400000002235, 58651.0, 58682.5, 58729.20000000298, 58757.0, 58789.60000000149, 58819.60000000149, 58851.400000002235, 58876.70000000298, 58914.20000000298, 58943.400000002235, 58974.20000000298, 59004.800000000745, 59029.300000000745, 59066.0, 59092.300000000745, 59129.70000000298, 59162.20000000298, 59187.900000002235, 59223.10000000149, 59253.70000000298, 59298.5, 59329.900000002235, 59377.20000000298, 59403.800000000745, 59436.5, 59483.60000000149, 59532.400000002235, 59560.900000002235, 59586.900000002235, 59622.20000000298, 59647.300000000745, 59681.0, 59711.5, 59740.5, 59790.0, 59816.5, 59844.300000000745, 59882.400000002235, 59912.400000002235, 59938.10000000149, 59974.300000000745, 60004.900000002235, 60030.900000002235, 60063.60000000149, 60089.20000000298, 60125.400000002235, 60158.300000000745, 60183.800000000745, 60217.20000000298, 60248.70000000298, 60278.300000000745, 60309.400000002235, 60337.10000000149, 60370.400000002235, 60400.20000000298, 60429.400000002235, 60460.800000000745, 60491.300000000745, 60536.900000002235, 60568.20000000298, 60600.10000000149, 60628.5, 60661.900000002235, 60691.900000002235, 60725.70000000298, 60754.800000000745, 60787.5, 60818.400000002235, 60849.20000000298, 60881.20000000298, 60909.10000000149, 60942.20000000298, 60968.400000002235, 61001.800000000745, 61033.70000000298, 61064.10000000149, 61096.70000000298, 61123.20000000298, 61158.60000000149, 61188.800000000745, 61221.300000000745, 61249.0, 61282.0, 61310.5, 61341.20000000298, 61387.20000000298, 61432.10000000149, 61461.0, 61489.0, 61522.0, 61550.60000000149, 61596.400000002235, 61625.800000000745, 61658.60000000149, 61688.900000002235, 61720.300000000745, 61747.400000002235, 61784.0, 61808.10000000149, 61843.5, 61874.900000002235, 61904.10000000149, 61931.400000002235, 61965.0, 61996.800000000745, 62022.70000000298, 62058.60000000149, 62090.20000000298, 62135.10000000149, 62164.5, 62198.0, 62228.10000000149, 62259.800000000745, 62307.400000002235, 62339.5, 62370.400000002235, 62401.300000000745, 62428.60000000149, 62457.70000000298, 62488.300000000745, 62517.400000002235, 62550.800000000745, 62579.20000000298, 62611.800000000745, 62639.800000000745, 62671.800000000745, 62704.0, 62729.5, 62764.0, 62796.20000000298, 62822.10000000149, 62856.5, 62881.300000000745, 62911.400000002235, 62937.400000002235, 62966.60000000149, 62995.400000002235, 63028.300000000745, 63054.900000002235, 63094.10000000149, 63120.900000002235, 63147.5, 63174.400000002235, 63214.20000000298, 63240.0, 63276.900000002235, 63305.60000000149, 63336.10000000149, 63381.70000000298, 63414.400000002235, 63442.5, 63472.60000000149, 63504.900000002235, 63541.5, 63570.70000000298, 63603.900000002235, 63631.0, 63658.5, 63687.5, 63713.60000000149, 63746.300000000745, 63783.70000000298, 63836.10000000149, 63869.800000000745, 63915.60000000149, 63942.60000000149, 63975.5, 64006.800000000745, 64046.5, 64073.0, 64100.800000000745, 64125.900000002235, 64160.400000002235, 64187.10000000149, 64219.400000002235, 64251.400000002235, 64298.5, 64329.300000000745, 64360.400000002235, 64405.900000002235, 64450.0, 64480.10000000149, 64512.70000000298, 64557.70000000298, 64589.400000002235, 64615.70000000298, 64650.60000000149, 64681.800000000745, 64706.300000000745, 64757.70000000298, 64805.400000002235, 64830.800000000745, 64865.20000000298, 64896.900000002235, 64924.800000000745, 64958.300000000745, 64986.800000000745, 65033.5, 65060.5, 65095.20000000298, 65124.10000000149, 65156.5, 65189.800000000745, 65220.10000000149, 65244.5, 65272.400000002235, 65303.70000000298, 65329.900000002235, 65358.10000000149, 65385.10000000149, 65421.800000000745, 65449.800000000745, 65498.20000000298, 65524.800000000745, 65559.0, 65583.90000000224, 65620.30000000075, 65665.60000000149, 65694.10000000149, 65722.20000000298, 65757.80000000075, 65804.40000000224, 65834.0, 65866.40000000224, 65893.30000000075, 65926.10000000149, 65957.30000000075, 66005.5, 66031.60000000149, 66065.0, 66091.70000000298, 66127.80000000075, 66152.90000000224, 66194.40000000224, 66222.5, 66250.5, 66282.30000000075, 66311.20000000298, 66344.0, 66375.20000000298, 66404.20000000298, 66439.40000000224, 66485.20000000298, 66530.10000000149, 66562.10000000149, 66589.10000000149, 66615.60000000149, 66651.70000000298, 66700.5, 66726.80000000075, 66762.5, 66792.80000000075, 66833.10000000149, 66867.5, 66898.10000000149, 66928.0, 66960.5, 66997.80000000075, 67022.10000000149, 67068.30000000075, 67100.70000000298, 67130.60000000149, 67162.5, 67193.10000000149, 67224.20000000298, 67254.60000000149, 67286.0, 67318.40000000224, 67346.0, 67382.30000000075, 67410.5, 67445.0, 67483.90000000224, 67523.70000000298, 67560.30000000075, 67599.40000000224, 67624.5, 67658.10000000149, 67691.0, 67736.20000000298, 67762.80000000075, 67796.90000000224, 67826.5, 67860.0, 67906.80000000075, 67939.80000000075, 67978.0, 68014.0, 68039.30000000075, 68074.5, 68123.70000000298, 68150.80000000075, 68184.70000000298, 68233.30000000075, 68262.20000000298, 68295.30000000075, 68332.60000000149, 68371.10000000149, 68395.20000000298, 68431.5, 68458.10000000149, 68491.90000000224, 68525.70000000298, 68554.0, 68585.30000000075, 68610.30000000075, 68648.0, 68674.90000000224, 68709.30000000075, 68740.80000000075, 68772.20000000298, 68817.5, 68849.30000000075, 68878.90000000224, 68912.70000000298, 68939.10000000149, 68971.20000000298, 69018.30000000075, 69063.10000000149, 69090.0, 69125.70000000298, 69155.10000000149, 69184.5, 69220.30000000075, 69253.70000000298, 69289.90000000224, 69320.10000000149, 69348.10000000149, 69385.5, 69432.40000000224, 69464.20000000298, 69510.30000000075, 69539.90000000224, 69567.20000000298, 69596.40000000224, 69628.90000000224, 69656.60000000149, 69691.10000000149, 69727.40000000224, 69756.20000000298, 69787.60000000149, 69814.80000000075, 69844.40000000224, 69877.5, 69906.70000000298, 69938.90000000224, 69978.0, 70008.10000000149, 70037.0, 70072.5, 70108.80000000075, 70141.40000000224, 70170.80000000075, 70217.60000000149, 70249.60000000149, 70279.5, 70312.10000000149, 70338.60000000149, 70373.5, 70406.70000000298, 70431.10000000149, 70465.0, 70496.80000000075, 70523.5, 70553.60000000149, 70599.80000000075, 70629.90000000224, 70657.30000000075, 70689.70000000298, 70715.30000000075, 70757.10000000149, 70785.5, 70816.20000000298, 70848.70000000298, 70877.80000000075, 70904.90000000224, 70939.0, 70972.10000000149, 71019.20000000298, 71045.60000000149, 71080.30000000075, 71126.70000000298, 71154.5, 71189.0, 71215.90000000224, 71250.90000000224, 71279.60000000149, 71312.30000000075, 71343.60000000149, 71375.60000000149, 71403.60000000149, 71435.70000000298, 71466.5, 71498.70000000298, 71537.30000000075, 71566.30000000075, 71592.0, 71624.5, 71654.70000000298, 71680.0, 71715.5, 71747.90000000224, 71773.70000000298, 71808.40000000224, 71839.10000000149, 71871.20000000298, 71897.40000000224, 71933.20000000298, 71959.60000000149, 71992.40000000224, 72042.0, 72069.90000000224, 72094.90000000224, 72130.90000000224, 72159.90000000224, 72193.5, 72224.70000000298, 72260.30000000075, 72288.60000000149, 72320.80000000075, 72347.80000000075, 72378.20000000298, 72405.70000000298, 72438.10000000149, 72463.80000000075, 72498.60000000149, 72526.20000000298, 72562.0, 72594.5, 72622.80000000075, 72669.0, 72713.60000000149, 72744.5, 72773.80000000075, 72806.5, 72845.70000000298, 72874.70000000298, 72905.10000000149, 72943.20000000298, 72988.0, 73014.0, 73049.0, 73076.90000000224, 73110.5, 73153.90000000224, 73188.40000000224, 73213.5, 73248.40000000224, 73279.70000000298, 73305.70000000298, 73339.70000000298, 73366.70000000298, 73403.70000000298, 73450.5, 73478.90000000224, 73510.70000000298, 73556.80000000075, 73584.10000000149, 73637.70000000298, 73664.70000000298, 73694.5, 73725.60000000149, 73758.60000000149, 73789.10000000149, 73819.20000000298, 73865.0, 73891.10000000149, 73923.20000000298, 73954.80000000075, 74001.90000000224, 74032.30000000075, 74058.30000000075, 74092.70000000298, 74123.90000000224, 74154.40000000224, 74184.5, 74211.10000000149, 74246.90000000224, 74291.20000000298, 74323.40000000224, 74360.40000000224, 74387.80000000075, 74414.10000000149, 74441.0, 74478.30000000075, 74507.40000000224, 74538.70000000298, 74577.80000000075, 74605.30000000075, 74634.70000000298, 74661.70000000298, 74700.0, 74731.5, 74757.0, 74785.60000000149, 74831.5, 74860.20000000298, 74892.80000000075, 74924.0, 74956.20000000298, 75004.5, 75034.5, 75060.70000000298, 75095.20000000298, 75120.30000000075, 75161.90000000224, 75188.60000000149, 75221.0, 75247.10000000149, 75276.80000000075, 75311.60000000149, 75338.10000000149, 75374.20000000298, 75398.90000000224, 75435.60000000149, 75484.60000000149, 75531.60000000149, 75563.5, 75595.60000000149, 75625.80000000075, 75665.20000000298, 75702.20000000298, 75734.40000000224, 75780.5, 75827.10000000149, 75856.10000000149, 75888.30000000075, 75917.0, 75948.60000000149, 75980.5, 76013.20000000298, 76057.5, 76089.60000000149, 76114.90000000224, 76152.60000000149, 76198.20000000298, 76224.5, 76261.0, 76290.0, 76327.5, 76354.10000000149, 76380.90000000224, 76416.90000000224, 76443.60000000149, 76477.5, 76521.40000000224, 76558.20000000298, 76588.10000000149, 76615.5, 76642.40000000224, 76678.30000000075, 76722.30000000075, 76754.5, 76781.20000000298, 76816.10000000149, 76842.10000000149, 76876.70000000298, 76914.90000000224, 76951.60000000149, 76984.60000000149, 77012.0, 77042.80000000075, 77075.10000000149, 77108.30000000075, 77137.10000000149, 77169.20000000298, 77200.70000000298, 77226.5, 77261.5, 77309.40000000224, 77342.20000000298, 77373.0, 77400.40000000224, 77434.5, 77480.90000000224, 77508.80000000075, 77540.70000000298, 77567.20000000298, 77603.70000000298, 77631.80000000075, 77665.10000000149, 77696.5, 77721.60000000149, 77755.0, 77781.40000000224, 77817.80000000075, 77845.40000000224, 77880.80000000075, 77909.5, 77941.90000000224, 77989.20000000298, 78030.30000000075, 78064.0, 78095.0, 78123.30000000075, 78148.60000000149, 78174.20000000298, 78200.80000000075, 78227.80000000075, 78262.5, 78289.80000000075, 78332.90000000224, 78369.40000000224, 78393.70000000298, 78429.30000000075, 78460.10000000149, 78486.30000000075, 78522.90000000224, 78554.20000000298, 78600.5, 78629.20000000298, 78657.40000000224, 78689.0, 78721.5, 78766.0, 78795.70000000298, 78828.60000000149, 78862.5, 78890.20000000298, 78920.70000000298, 78953.40000000224, 78978.5, 79013.10000000149, 79044.40000000224, 79077.20000000298, 79122.70000000298, 79153.10000000149, 79184.70000000298, 79233.10000000149, 79276.80000000075, 79307.10000000149, 79336.90000000224, 79384.90000000224, 79431.70000000298, 79459.40000000224, 79495.60000000149, 79526.0, 79557.80000000075, 79596.70000000298, 79628.10000000149, 79654.5, 79684.40000000224, 79711.10000000149, 79741.40000000224, 79767.5, 79803.70000000298, 79835.5, 79884.60000000149, 79929.70000000298, 79960.60000000149, 79988.5, 80021.60000000149, 80052.30000000075, 80079.30000000075, 80114.60000000149, 80152.80000000075, 80181.40000000224, 80207.30000000075, 80235.60000000149, 80265.20000000298, 80294.80000000075, 80324.10000000149, 80355.30000000075, 80385.60000000149, 80417.0, 80446.90000000224, 80479.40000000224, 80504.60000000149, 80540.30000000075, 80585.0, 80631.20000000298, 80657.80000000075, 80691.60000000149, 80725.90000000224, 80755.20000000298, 80786.40000000224, 80810.90000000224, 80845.30000000075, 80873.10000000149, 80905.20000000298, 80938.10000000149, 80971.40000000224, 81001.5, 81046.30000000075, 81074.70000000298, 81109.40000000224, 81140.20000000298, 81171.70000000298, 81211.70000000298, 81248.20000000298, 81274.90000000224, 81303.0, 81337.90000000224, 81386.80000000075, 81416.80000000075, 81447.80000000075, 81488.30000000075, 81524.70000000298, 81555.40000000224, 81587.0, 81648.30000000075, 81674.70000000298, 81709.40000000224, 81737.90000000224, 81770.60000000149, 81799.80000000075, 81832.5, 81860.80000000075, 81892.90000000224, 81922.70000000298, 81954.10000000149, 81991.40000000224, 82020.90000000224, 82049.30000000075, 82076.5, 82111.30000000075, 82146.70000000298, 82174.80000000075, 82205.10000000149, 82248.10000000149, 82277.10000000149, 82308.90000000224, 82340.20000000298, 82371.0, 82417.5, 82443.30000000075, 82479.60000000149, 82512.80000000075, 82542.80000000075, 82574.5, 82603.60000000149, 82635.90000000224, 82663.90000000224, 82699.5, 82730.40000000224, 82760.70000000298, 82788.40000000224, 82820.90000000224, 82846.60000000149, 82883.20000000298, 82916.70000000298, 82942.10000000149, 82977.10000000149, 83003.10000000149, 83038.80000000075, 83084.0, 83115.70000000298, 83148.60000000149, 83178.10000000149, 83210.70000000298, 83241.80000000075, 83271.60000000149, 83303.80000000075, 83330.0, 83365.60000000149, 83392.80000000075, 83428.5, 83459.80000000075, 83496.80000000075, 83524.0, 83549.5, 83577.60000000149, 83609.80000000075, 83642.70000000298, 83681.70000000298, 83718.70000000298, 83751.80000000075, 83781.5, 83814.70000000298, 83854.80000000075, 83894.40000000224, 83924.30000000075, 83959.70000000298, 83987.30000000075, 84016.40000000224, 84043.60000000149, 84089.60000000149, 84120.20000000298, 84153.0, 84199.0, 84228.20000000298, 84261.70000000298, 84290.5, 84327.70000000298, 84355.30000000075, 84385.0, 84409.30000000075, 84442.90000000224, 84470.5, 84504.40000000224, 84533.0, 84565.0, 84593.30000000075, 84626.10000000149, 84651.30000000075, 84684.90000000224, 84711.0, 84744.90000000224, 84792.70000000298, 84823.80000000075, 84854.60000000149, 84887.60000000149, 84917.80000000075, 84949.5, 84976.70000000298, 85012.30000000075, 85041.30000000075, 85070.60000000149, 85101.0, 85126.60000000149, 85163.10000000149, 85187.40000000224, 85225.5, 85257.30000000075, 85287.40000000224, 85318.0, 85344.40000000224, 85379.70000000298, 85409.40000000224, 85439.80000000075, 85470.20000000298, 85501.70000000298, 85534.20000000298, 85560.40000000224, 85594.40000000224, 85634.30000000075, 85669.20000000298, 85703.40000000224, 85735.20000000298, 85763.90000000224, 85791.30000000075, 85820.80000000075, 85856.20000000298, 85888.10000000149, 85913.30000000075, 85948.90000000224, 85977.10000000149, 86012.40000000224, 86059.20000000298, 86088.40000000224, 86114.20000000298, 86150.40000000224, 86177.5, 86209.60000000149, 86242.30000000075, 86268.70000000298, 86300.90000000224, 86326.90000000224, 86362.80000000075, 86394.90000000224, 86421.90000000224, 86457.5, 86489.0, 86522.10000000149, 86567.20000000298, 86595.40000000224, 86630.90000000224, 86677.5, 86708.30000000075, 86736.70000000298, 86785.5, 86814.70000000298, 86846.10000000149, 86875.20000000298, 86905.90000000224, 86954.5, 86980.40000000224, 87016.30000000075, 87044.40000000224, 87077.90000000224, 87108.5, 87141.20000000298, 87170.70000000298, 87204.40000000224, 87231.10000000149, 87267.40000000224, 87295.70000000298, 87328.5, 87374.30000000075, 87403.20000000298, 87435.90000000224, 87482.90000000224, 87514.60000000149, 87546.40000000224, 87574.70000000298, 87604.90000000224, 87638.70000000298, 87672.5, 87717.60000000149, 87762.70000000298, 87795.30000000075, 87823.80000000075, 87853.80000000075, 87892.10000000149, 87922.20000000298, 87954.10000000149, 87984.5, 88014.80000000075, 88060.80000000075, 88090.60000000149, 88119.80000000075, 88148.20000000298, 88182.30000000075, 88212.40000000224, 88241.5, 88271.5, 88319.80000000075, 88346.60000000149, 88380.70000000298, 88412.5, 88444.20000000298, 88471.80000000075, 88523.20000000298, 88551.40000000224, 88586.70000000298, 88619.10000000149, 88645.60000000149, 88689.20000000298, 88717.10000000149, 88749.70000000298, 88780.0, 88812.30000000075, 88844.70000000298, 88873.0, 88904.30000000075, 88930.70000000298, 88962.90000000224, 88988.80000000075, 89023.60000000149, 89049.40000000224, 89085.70000000298, 89117.40000000224, 89143.60000000149, 89181.20000000298, 89207.90000000224, 89240.20000000298, 89266.90000000224, 89300.90000000224, 89331.90000000224, 89363.30000000075, 89393.80000000075, 89421.5, 89455.5, 89481.70000000298, 89516.60000000149, 89544.30000000075, 89578.0, 89624.60000000149, 89653.30000000075, 89678.90000000224, 89715.60000000149, 89743.30000000075, 89779.30000000075, 89821.0, 89853.90000000224, 89881.5, 89915.20000000298, 89959.30000000075, 89985.70000000298, 90019.90000000224, 90044.80000000075, 90080.60000000149, 90105.60000000149, 90141.40000000224, 90171.40000000224, 90202.80000000075, 90234.0, 90265.0, 90298.5, 90324.30000000075, 90360.70000000298, 90386.70000000298, 90421.90000000224, 90454.10000000149, 90496.90000000224, 90522.70000000298, 90560.20000000298, 90586.70000000298, 90649.0, 90675.40000000224, 90710.70000000298, 90738.5, 90773.0, 90799.70000000298, 90833.60000000149, 90866.0, 90892.70000000298, 90925.0, 90951.5, 90988.30000000075, 91015.20000000298, 91049.60000000149, 91076.90000000224, 91108.30000000075, 91136.90000000224, 91170.40000000224, 91198.20000000298, 91233.60000000149, 91260.60000000149, 91287.10000000149, 91324.10000000149, 91355.80000000075, 91381.60000000149, 91415.20000000298, 91442.70000000298, 91476.80000000075, 91501.20000000298, 91538.30000000075, 91570.40000000224, 91595.80000000075, 91632.80000000075, 91658.60000000149, 91692.30000000075, 91720.0, 91784.70000000298, 91833.40000000224, 91864.5, 91891.0, 91923.60000000149, 91955.40000000224, 91999.80000000075, 92030.30000000075, 92058.90000000224, 92091.10000000149, 92123.60000000149, 92154.30000000075, 92186.60000000149, 92211.5, 92246.30000000075, 92271.60000000149, 92308.10000000149, 92336.60000000149, 92382.20000000298, 92412.70000000298, 92442.60000000149, 92471.40000000224, 92505.90000000224, 92540.20000000298, 92572.20000000298, 92602.70000000298, 92628.60000000149, 92660.5, 92688.5, 92716.30000000075, 92750.20000000298, 92777.20000000298, 92812.40000000224, 92837.40000000224, 92873.5, 92921.40000000224, 92953.20000000298, 92978.90000000224, 93013.70000000298, 93047.90000000224, 93074.80000000075, 93108.0, 93134.5, 93170.5, 93201.5, 93263.60000000149, 93289.5, 93327.60000000149, 93355.60000000149, 93390.5, 93417.40000000224, 93445.10000000149, 93482.20000000298, 93526.0, 93554.0, 93588.10000000149, 93613.80000000075, 93647.80000000075, 93679.5, 93711.80000000075, 93737.80000000075, 93770.60000000149, 93802.60000000149, 93829.20000000298, 93866.30000000075, 93892.20000000298, 93927.60000000149, 93952.80000000075, 93991.20000000298, 94020.70000000298, 94044.90000000224, 94079.40000000224, 94112.30000000075, 94139.70000000298, 94171.20000000298, 94203.5, 94231.5, 94266.40000000224, 94311.40000000224, 94339.5, 94372.40000000224, 94405.20000000298, 94430.80000000075, 94466.40000000224, 94491.40000000224, 94529.5, 94556.0, 94586.60000000149, 94619.5, 94648.10000000149, 94680.40000000224, 94711.40000000224, 94760.30000000075, 94787.0, 94820.90000000224, 94860.90000000224, 94895.90000000224, 94925.40000000224, 94951.70000000298, 94987.5, 95015.40000000224, 95049.60000000149, 95081.10000000149, 95113.0, 95145.30000000075, 95170.80000000075, 95206.0, 95231.20000000298, 95269.20000000298, 95294.70000000298, 95329.10000000149, 95354.10000000149, 95390.30000000075, 95420.40000000224, 95445.40000000224, 95480.70000000298, 95513.20000000298, 95542.5, 95584.5, 95622.40000000224, 95646.80000000075, 95681.0, 95706.80000000075, 95741.60000000149, 95773.0, 95804.70000000298, 95832.80000000075, 95860.40000000224, 95895.20000000298, 95923.0, 95955.20000000298, 95983.10000000149, 96015.90000000224, 96060.90000000224, 96091.20000000298, 96124.80000000075, 96155.20000000298, 96187.5, 96234.40000000224, 96267.30000000075, 96312.20000000298, 96339.20000000298, 96372.90000000224, 96398.30000000075, 96433.20000000298, 96464.5, 96490.30000000075, 96526.20000000298, 96571.30000000075, 96601.80000000075, 96649.30000000075, 96677.0, 96709.10000000149, 96757.60000000149, 96785.90000000224, 96811.10000000149, 96848.70000000298, 96879.20000000298, 96910.0, 96941.90000000224, 96973.10000000149, 97005.0, 97035.70000000298, 97061.10000000149, 97096.60000000149, 97124.70000000298, 97159.0, 97204.5, 97238.30000000075, 97265.40000000224, 97294.30000000075, 97327.60000000149, 97356.20000000298, 97390.30000000075, 97420.70000000298, 97452.40000000224, 97480.0, 97507.20000000298, 97540.0, 97566.70000000298, 97600.20000000298, 97632.70000000298, 97660.40000000224, 97689.70000000298, 97727.70000000298, 97759.40000000224, 97787.30000000075, 97818.70000000298, 97845.90000000224, 97880.40000000224, 97907.0, 97939.30000000075, 97973.80000000075, 98005.30000000075, 98034.70000000298, 98065.5, 98095.90000000224, 98121.70000000298, 98156.0, 98181.80000000075, 98216.10000000149, 98244.60000000149, 98277.60000000149, 98310.40000000224, 98340.5, 98380.90000000224, 98407.30000000075, 98434.80000000075, 98466.70000000298, 98495.0, 98528.5, 98575.80000000075, 98608.5, 98637.10000000149, 98669.80000000075, 98714.30000000075, 98742.60000000149, 98770.90000000224, 98803.30000000075, 98833.90000000224, 98860.5, 98898.30000000075, 98927.5, 98958.60000000149, 98988.30000000075, 99019.80000000075, 99052.40000000224, 99098.20000000298, 99125.40000000224, 99159.60000000149, 99190.20000000298, 99232.0, 99266.90000000224, 99291.80000000075, 99328.70000000298, 99353.40000000224, 99389.10000000149, 99415.90000000224, 99450.10000000149, 99482.20000000298, 99514.5, 99545.90000000224, 99571.60000000149, 99605.5, 99635.70000000298, 99669.0, 99694.80000000075, 99729.10000000149, 99755.20000000298, 99790.20000000298, 99821.90000000224, 99848.5, 99879.80000000075, 99907.80000000075, 99943.30000000075, 99970.70000000298, 100010.5, 100041.80000000075, 100071.0, 100110.0, 100144.10000000149, 100170.5, 100203.30000000075, 100248.10000000149, 100277.80000000075, 100308.0, 100353.70000000298, 100384.70000000298, 100411.60000000149, 100447.5, 100473.0, 100509.70000000298, 100542.30000000075, 100568.70000000298, 100601.60000000149, 100633.10000000149, 100665.30000000075, 100691.60000000149, 100725.80000000075, 100757.90000000224, 100789.20000000298, 100823.10000000149, 100852.10000000149, 100877.5, 100906.40000000224, 100942.70000000298, 100976.30000000075, 101005.20000000298, 101033.90000000224, 101060.10000000149, 101087.0, 101121.5, 101157.10000000149, 101184.40000000224, 101215.20000000298, 101244.60000000149, 101273.20000000298, 101305.0, 101331.10000000149, 101367.10000000149, 101394.10000000149, 101426.90000000224, 101457.20000000298, 101486.80000000075, 101523.30000000075, 101549.80000000075, 101584.5, 101615.40000000224, 101645.60000000149, 101678.40000000224, 101719.10000000149, 101752.60000000149, 101783.90000000224, 101813.0, 101839.90000000224, 101875.10000000149, 101907.60000000149, 101953.60000000149, 101985.5, 102018.70000000298, 102047.5, 102078.70000000298, 102109.30000000075, 102140.30000000075, 102200.90000000224, 102232.0, 102264.40000000224, 102293.0, 102324.90000000224, 102366.60000000149, 102404.60000000149, 102430.40000000224, 102465.30000000075, 102511.0, 102536.5, 102577.70000000298, 102604.40000000224, 102633.80000000075, 102660.40000000224, 102692.30000000075, 102718.80000000075, 102755.40000000224, 102780.10000000149, 102816.90000000224, 102854.20000000298, 102893.5, 102919.30000000075, 102953.80000000075, 102980.40000000224, 103017.60000000149, 103064.60000000149, 103099.90000000224, 103127.80000000075, 103160.40000000224, 103189.30000000075, 103220.90000000224, 103248.70000000298, 103280.30000000075, 103306.20000000298, 103340.20000000298, 103372.70000000298, 103419.80000000075, 103449.5, 103481.40000000224, 103514.5, 103559.80000000075, 103587.20000000298, 103620.40000000224, 103653.20000000298, 103680.20000000298, 103712.30000000075, 103738.30000000075, 103779.70000000298, 103806.60000000149, 103834.40000000224, 103864.40000000224, 103894.80000000075, 103927.0, 103970.70000000298, 104004.60000000149, 104033.90000000224, 104061.60000000149, 104093.90000000224, 104125.10000000149, 104158.80000000075, 104193.60000000149, 104219.90000000224, 104251.20000000298, 104283.10000000149, 104311.0, 104343.10000000149, 104388.10000000149, 104420.30000000075, 104452.20000000298, 104480.20000000298, 104516.5, 104544.10000000149, 104578.5, 104610.5, 104641.30000000075, 104671.0, 104696.70000000298, 104730.0, 104758.30000000075, 104785.10000000149, 104824.0, 104853.70000000298, 104901.60000000149, 104926.60000000149, 104963.0, 105003.20000000298, 105039.20000000298, 105086.90000000224, 105112.5, 105148.70000000298, 105174.60000000149, 105208.70000000298, 105257.40000000224, 105288.0, 105313.60000000149, 105346.90000000224, 105373.0, 105408.20000000298, 105436.60000000149, 105467.20000000298, 105512.80000000075, 105542.10000000149, 105573.30000000075, 105605.0, 105640.30000000075, 105668.0, 105703.10000000149, 105728.80000000075, 105773.70000000298, 105805.10000000149, 105835.80000000075, 105864.70000000298, 105890.30000000075, 105924.90000000224, 105952.30000000075, 105984.80000000075, 106011.60000000149, 106047.5, 106072.90000000224, 106108.20000000298, 106138.0, 106170.60000000149, 106198.0, 106226.80000000075, 106258.30000000075, 106284.80000000075, 106319.0, 106344.0, 106379.40000000224, 106427.60000000149, 106458.5, 106489.20000000298, 106531.0, 106564.90000000224, 106597.60000000149, 106628.0, 106660.20000000298, 106707.20000000298, 106737.30000000075, 106762.90000000224, 106799.20000000298, 106826.60000000149, 106859.10000000149, 106891.70000000298, 106939.5, 106974.40000000224, 107018.30000000075, 107045.20000000298, 107079.60000000149, 107126.20000000298, 107155.30000000075, 107184.30000000075, 107213.60000000149, 107246.60000000149, 107273.5, 107308.40000000224, 107352.30000000075, 107386.20000000298, 107433.30000000075, 107476.90000000224, 107517.80000000075, 107553.80000000075, 107586.60000000149, 107612.30000000075, 107647.40000000224, 107676.60000000149, 107709.5, 107755.70000000298, 107789.80000000075, 107815.5, 107847.70000000298, 107876.90000000224, 107908.40000000224, 107956.0, 107981.0, 108016.40000000224, 108041.30000000075, 108079.30000000075, 108105.20000000298, 108142.0, 108174.5, 108203.60000000149, 108234.80000000075, 108261.60000000149, 108296.20000000298, 108323.0, 108357.20000000298, 108383.60000000149, 108419.0, 108445.80000000075, 108480.80000000075, 108510.5, 108542.5, 108588.0, 108620.30000000075, 108666.60000000149, 108695.10000000149, 108721.5, 108756.60000000149, 108791.5, 108820.20000000298, 108864.30000000075, 108896.90000000224, 108923.90000000224, 108973.30000000075, 109000.0, 109034.80000000075, 109080.60000000149, 109105.90000000224, 109141.70000000298, 109170.10000000149, 109206.90000000224, 109234.60000000149, 109266.80000000075, 109290.60000000149, 109325.40000000224, 109372.20000000298, 109405.40000000224, 109431.10000000149, 109462.90000000224, 109494.10000000149, 109521.60000000149, 109555.60000000149, 109581.60000000149, 109617.5, 109650.5, 109694.90000000224, 109722.20000000298, 109756.60000000149, 109784.0, 109817.5, 109844.10000000149, 109878.5, 109912.10000000149, 109940.30000000075, 109988.40000000224, 110019.80000000075, 110052.30000000075, 110079.90000000224, 110107.80000000075, 110140.30000000075, 110172.0, 110231.80000000075, 110263.20000000298, 110289.90000000224, 110324.20000000298, 110348.80000000075, 110383.60000000149, 110414.60000000149, 110445.10000000149, 110491.60000000149, 110520.70000000298, 110546.70000000298, 110582.40000000224, 110610.10000000149, 110642.0, 110670.70000000298, 110703.40000000224, 110738.10000000149, 110765.80000000075, 110795.40000000224, 110826.60000000149, 110855.80000000075, 110889.5, 110915.90000000224, 110950.20000000298, 110976.60000000149, 111011.20000000298, 111036.70000000298, 111073.5, 111108.40000000224, 111137.30000000075, 111165.10000000149, 111191.5, 111228.60000000149, 111258.70000000298, 111290.70000000298, 111323.0, 111353.20000000298, 111383.80000000075, 111415.10000000149, 111446.60000000149, 111479.40000000224, 111510.40000000224, 111537.80000000075, 111572.5, 111605.0, 111632.10000000149, 111657.70000000298, 111692.20000000298, 111739.70000000298, 111768.80000000075, 111802.5, 111833.10000000149, 111861.20000000298, 111895.40000000224, 111941.40000000224, 111973.0, 111998.90000000224, 112035.70000000298, 112060.80000000075, 112097.20000000298, 112121.5, 112173.30000000075, 112202.40000000224, 112235.90000000224, 112282.90000000224, 112311.30000000075, 112345.60000000149, 112377.0, 112404.70000000298, 112437.30000000075, 112472.90000000224, 112500.80000000075, 112528.30000000075, 112555.5, 112594.40000000224, 112625.5, 112660.80000000075, 112686.30000000075, 112731.70000000298, 112759.10000000149, 112790.30000000075, 112820.5, 112857.60000000149, 112885.40000000224, 112920.30000000075, 112947.30000000075, 112977.70000000298, 113002.80000000075, 113033.20000000298, 113063.10000000149, 113094.60000000149, 113123.30000000075, 113154.10000000149, 113187.30000000075, 113215.40000000224, 113242.10000000149, 113276.30000000075, 113307.60000000149, 113340.60000000149, 113399.90000000224, 113427.80000000075, 113462.5, 113493.10000000149, 113524.20000000298, 113554.90000000224, 113580.80000000075, 113616.5, 113645.60000000149, 113676.90000000224, 113707.5, 113740.60000000149, 113771.90000000224, 113816.0, 113843.5, 113879.0, 113910.80000000075, 113937.0, 113973.90000000224, 114001.90000000224, 114026.90000000224, 114063.90000000224, 114089.30000000075, 114124.10000000149, 114156.5, 114195.20000000298, 114221.60000000149, 114252.70000000298, 114279.10000000149, 114311.30000000075, 114341.0, 114372.30000000075, 114399.80000000075, 114426.40000000224, 114461.80000000075, 114495.70000000298, 114523.20000000298, 114561.10000000149, 114588.30000000075, 114616.10000000149, 114648.5, 114677.70000000298, 114709.5, 114761.80000000075, 114789.80000000075, 114817.80000000075, 114846.30000000075, 114881.90000000224, 114909.40000000224, 114942.0, 114989.0, 115017.40000000224, 115050.70000000298, 115097.60000000149, 115130.40000000224, 115175.0, 115204.40000000224, 115235.20000000298, 115263.20000000298, 115298.40000000224, 115324.30000000075, 115361.0, 115393.30000000075, 115421.40000000224, 115454.10000000149, 115480.5, 115514.60000000149, 115549.30000000075, 115578.80000000075, 115610.10000000149, 115639.90000000224, 115678.10000000149, 115704.70000000298, 115731.5, 115767.30000000075, 115795.20000000298, 115827.10000000149, 115854.70000000298, 115893.20000000298, 115919.60000000149, 115946.10000000149, 115980.5, 116010.70000000298, 116045.5, 116089.80000000075, 116119.10000000149, 116150.70000000298, 116183.80000000075, 116211.90000000224, 116244.80000000075, 116292.90000000224, 116320.5, 116352.80000000075, 116387.0, 116423.80000000075, 116452.60000000149, 116478.90000000224, 116522.10000000149, 116552.10000000149, 116583.20000000298, 116615.0, 116640.80000000075, 116676.80000000075, 116703.80000000075, 116738.80000000075, 116771.30000000075, 116797.30000000075, 116830.70000000298, 116857.0, 116892.20000000298, 116934.80000000075, 116965.10000000149, 116999.60000000149, 117046.90000000224, 117075.90000000224, 117107.70000000298, 117156.60000000149, 117183.30000000075, 117223.20000000298, 117253.80000000075, 117282.70000000298, 117311.70000000298, 117343.60000000149, 117372.0, 117417.20000000298, 117445.70000000298, 117476.40000000224, 117506.60000000149, 117540.40000000224, 117568.20000000298, 117600.0, 117626.40000000224, 117657.5, 117691.80000000075, 117720.0, 117767.10000000149, 117799.0, 117826.30000000075, 117859.20000000298, 117888.80000000075, 117927.20000000298, 117954.10000000149, 117981.40000000224, 118012.70000000298, 118045.60000000149, 118071.90000000224, 118105.10000000149, 118130.80000000075, 118167.30000000075, 118196.20000000298, 118228.30000000075, 118274.30000000075, 118303.90000000224, 118329.5, 118364.10000000149, 118392.0, 118440.90000000224, 118470.80000000075, 118502.70000000298, 118534.60000000149, 118567.60000000149, 118593.70000000298, 118628.30000000075, 118674.30000000075, 118720.60000000149, 118750.80000000075, 118796.40000000224, 118827.70000000298, 118854.70000000298, 118888.10000000149, 118920.80000000075, 118947.40000000224, 118985.0, 119015.5, 119045.20000000298, 119077.0, 119107.80000000075, 119132.60000000149, 119167.5, 119214.30000000075, 119245.60000000149, 119274.70000000298, 119301.10000000149, 119333.90000000224, 119361.5, 119394.40000000224, 119420.70000000298, 119460.70000000298, 119488.20000000298, 119514.70000000298, 119545.5, 119577.70000000298, 119605.5, 119637.80000000075, 119668.80000000075, 119693.90000000224, 119730.60000000149, 119760.40000000224, 119786.5, 119821.10000000149, 119854.80000000075, 119880.80000000075, 119910.70000000298, 119938.10000000149, 119974.30000000075, 120002.5, 120033.10000000149, 120059.20000000298, 120092.5, 120121.10000000149, 120154.5, 120185.30000000075, 120211.20000000298, 120245.90000000224, 120271.90000000224, 120304.90000000224, 120336.70000000298, 120367.20000000298, 120399.5, 120426.20000000298, 120465.30000000075, 120494.70000000298, 120527.10000000149, 120553.30000000075, 120586.60000000149, 120617.40000000224, 120665.60000000149, 120694.90000000224, 120726.30000000075, 120771.40000000224, 120802.20000000298, 120827.40000000224, 120862.10000000149, 120893.60000000149, 120927.10000000149, 120954.60000000149, 120996.0, 121032.80000000075, 121059.20000000298, 121091.60000000149, 121123.20000000298, 121157.90000000224, 121187.20000000298, 121218.60000000149, 121244.40000000224, 121281.70000000298, 121308.70000000298, 121343.30000000075, 121369.70000000298, 121412.20000000298, 121439.60000000149, 121470.5, 121498.0, 121528.70000000298, 121554.80000000075, 121587.70000000298, 121618.30000000075, 121651.5, 121677.80000000075, 121712.70000000298, 121739.70000000298, 121774.80000000075, 121806.70000000298, 121832.70000000298, 121865.30000000075, 121893.60000000149, 121928.10000000149, 121975.20000000298, 122005.20000000298, 122033.0, 122066.40000000224, 122093.90000000224, 122142.30000000075, 122170.90000000224, 122197.90000000224, 122230.90000000224, 122258.30000000075, 122292.10000000149, 122337.40000000224, 122368.40000000224, 122394.70000000298, 122427.80000000075, 122458.0, 122490.10000000149, 122525.80000000075, 122552.40000000224, 122599.60000000149, 122632.5, 122663.0, 122690.0, 122726.30000000075, 122754.80000000075, 122783.70000000298, 122815.60000000149, 122843.20000000298, 122875.40000000224, 122924.70000000298, 122952.30000000075, 122987.40000000224, 123014.90000000224, 123050.20000000298, 123097.20000000298, 123126.60000000149, 123153.30000000075, 123188.20000000298, 123223.60000000149, 123252.5, 123282.40000000224, 123306.10000000149, 123341.40000000224, 123372.10000000149, 123433.90000000224, 123478.80000000075, 123505.80000000075, 123540.5, 123573.10000000149, 123605.10000000149, 123647.40000000224, 123692.80000000075, 123721.60000000149, 123754.70000000298, 123787.90000000224, 123815.20000000298, 123850.5, 123896.40000000224, 123927.30000000075, 123955.90000000224, 123986.20000000298, 124014.60000000149, 124043.30000000075, 124076.60000000149, 124119.70000000298, 124154.20000000298, 124183.5, 124218.5, 124256.80000000075, 124284.60000000149, 124320.5, 124346.80000000075, 124374.5, 124402.90000000224, 124427.5, 124462.40000000224, 124489.60000000149, 124524.0, 124570.0, 124597.90000000224, 124629.5, 124662.30000000075, 124691.70000000298, 124734.40000000224, 124768.60000000149, 124816.80000000075, 124843.10000000149, 124875.80000000075, 124906.60000000149, 124967.40000000224, 125015.80000000075, 125045.5, 125072.60000000149, 125108.30000000075, 125141.0, 125170.70000000298, 125210.30000000075, 125239.20000000298, 125266.90000000224, 125294.80000000075, 125322.70000000298, 125360.30000000075, 125403.60000000149, 125432.20000000298, 125460.5, 125494.80000000075, 125537.60000000149, 125569.60000000149, 125602.0, 125635.60000000149, 125662.5, 125696.5, 125722.70000000298, 125774.5, 125802.20000000298, 125834.90000000224, 125883.80000000075, 125912.90000000224, 125945.5, 125976.90000000224, 126004.70000000298, 126036.40000000224, 126068.40000000224, 126094.40000000224, 126132.90000000224, 126158.0, 126193.20000000298, 126237.20000000298, 126263.20000000298, 126298.20000000298, 126330.20000000298, 126361.70000000298, 126406.40000000224, 126437.10000000149, 126462.60000000149, 126498.40000000224, 126524.20000000298, 126559.60000000149, 126586.5, 126620.30000000075, 126653.5, 126681.90000000224, 126714.40000000224, 126745.40000000224, 126777.0, 126803.90000000224, 126838.30000000075, 126871.20000000298, 126915.0, 126946.40000000224, 126973.30000000075, 127007.0, 127038.40000000224, 127083.5, 127114.60000000149, 127140.5, 127177.5, 127204.60000000149, 127239.10000000149, 127271.10000000149, 127296.5, 127330.70000000298, 127360.5, 127392.20000000298, 127437.70000000298, 127468.5, 127493.20000000298, 127528.5, 127559.60000000149, 127605.60000000149, 127636.20000000298, 127667.90000000224, 127694.5, 127729.30000000075, 127756.40000000224, 127802.30000000075, 127836.10000000149, 127869.0, 127916.0, 127945.20000000298, 127972.40000000224, 128005.80000000075, 128039.40000000224, 128070.60000000149, 128097.80000000075, 128126.30000000075, 128160.70000000298, 128187.10000000149, 128220.40000000224, 128246.80000000075, 128282.0, 128326.0, 128355.60000000149, 128389.10000000149, 128419.20000000298, 128450.5, 128480.60000000149, 128513.60000000149, 128543.20000000298, 128575.60000000149, 128603.70000000298, 128637.10000000149, 128665.20000000298, 128699.5, 128727.60000000149, 128762.20000000298, 128795.0, 128821.0, 128855.60000000149, 128890.0, 128914.0, 128948.20000000298, 128975.0, 129010.40000000224, 129037.40000000224, 129071.0, 129116.80000000075, 129148.60000000149, 129177.10000000149, 129211.0, 129244.10000000149, 129271.90000000224, 129304.60000000149, 129335.30000000075, 129366.5, 129400.30000000075, 129446.70000000298, 129478.60000000149, 129510.5, 129556.10000000149, 129582.70000000298, 129618.20000000298, 129650.70000000298, 129696.60000000149, 129723.10000000149, 129759.40000000224, 129789.40000000224, 129838.10000000149, 129866.5, 129901.60000000149, 129947.5, 129979.10000000149, 130026.20000000298, 130054.80000000075, 130085.30000000075, 130116.5, 130143.80000000075, 130176.5, 130209.5, 130243.70000000298, 130269.60000000149, 130295.60000000149, 130329.5, 130360.60000000149, 130388.10000000149, 130421.30000000075, 130452.10000000149, 130483.70000000298, 130515.5, 130561.30000000075, 130590.30000000075, 130622.10000000149, 130654.90000000224, 130685.90000000224, 130730.90000000224, 130761.70000000298, 130787.5, 130822.0, 130854.0, 130887.5, 130913.90000000224, 130944.70000000298, 130975.40000000224, 131001.20000000298, 131038.40000000224, 131065.60000000149, 131097.0, 131143.0, 131189.80000000075, 131221.70000000298, 131250.0, 131282.80000000075, 131314.40000000224, 131344.70000000298, 131379.30000000075, 131423.1000000015, 131451.30000000075, 131488.30000000075, 131517.80000000075, 131563.1000000015, 131590.40000000224, 131625.70000000298, 131655.70000000298, 131682.70000000298, 131718.5, 131765.70000000298, 131795.0, 131827.70000000298, 131855.20000000298, 131889.0, 131916.1000000015, 131949.0, 131980.0, 132013.0, 132059.90000000224, 132091.5, 132121.90000000224, 132155.20000000298, 132187.30000000075, 132232.30000000075, 132258.80000000075, 132293.70000000298, 132324.70000000298, 132356.80000000075, 132389.40000000224, 132413.40000000224, 132450.80000000075, 132482.1000000015, 132508.80000000075, 132544.5, 132570.70000000298, 132610.20000000298, 132642.20000000298, 132671.6000000015, 132695.90000000224, 132733.90000000224, 132759.90000000224, 132795.6000000015, 132826.0, 132853.90000000224, 132890.30000000075, 132921.70000000298, 132966.20000000298, 132992.80000000075, 133026.40000000224, 133058.80000000075, 133103.80000000075, 133150.0, 133178.70000000298, 133212.0, 133236.90000000224, 133272.0, 133308.70000000298, 133337.30000000075, 133370.30000000075, 133395.90000000224, 133424.5, 133449.70000000298, 133486.30000000075, 133512.0, 133547.40000000224, 133580.40000000224, 133610.30000000075, 133637.30000000075, 133671.5, 133703.6000000015, 133730.20000000298, 133766.30000000075, 133794.20000000298, 133828.6000000015, 133874.6000000015, 133906.30000000075, 133937.40000000224, 133972.0, 133999.1000000015, 134026.20000000298, 134057.70000000298, 134104.80000000075, 134132.70000000298, 134163.6000000015, 134196.5, 134221.70000000298, 134256.40000000224, 134287.20000000298, 134319.1000000015, 134344.0, 134380.40000000224, 134425.6000000015, 134453.40000000224, 134487.5, 134518.0, 134550.6000000015, 134582.90000000224, 134627.30000000075, 134656.40000000224, 134688.5, 134716.0, 134765.40000000224, 134792.40000000224, 134827.20000000298, 134857.0, 134894.90000000224, 134921.80000000075, 134952.80000000075, 134998.70000000298, 135025.90000000224, 135057.6000000015, 135087.30000000075, 135134.0, 135160.40000000224, 135195.30000000075, 135229.1000000015, 135256.40000000224, 135290.6000000015, 135322.5, 135349.1000000015, 135387.6000000015, 135415.30000000075, 135448.30000000075, 135478.0, 135506.1000000015, 135545.1000000015, 135575.6000000015, 135604.6000000015, 135632.30000000075, 135663.30000000075, 135693.6000000015, 135725.6000000015, 135774.0, 135803.40000000224, 135834.30000000075, 135867.20000000298, 135896.1000000015, 135927.40000000224, 135958.5, 135987.6000000015, 136025.0, 136049.40000000224, 136083.20000000298, 136110.70000000298, 136144.6000000015, 136175.80000000075, 136203.6000000015, 136241.20000000298, 136266.30000000075, 136300.5, 136332.6000000015, 136371.30000000075, 136397.6000000015, 136429.1000000015, 136455.20000000298, 136487.20000000298, 136515.80000000075, 136543.1000000015, 136575.30000000075, 136600.90000000224, 136636.0, 136665.20000000298, 136695.80000000075, 136744.70000000298, 136775.20000000298, 136807.0, 136855.5, 136883.80000000075, 136919.20000000298, 136947.0, 136978.0, 137010.30000000075, 137039.20000000298, 137083.40000000224, 137115.0, 137162.30000000075, 137191.1000000015, 137219.1000000015, 137250.30000000075, 137282.90000000224, 137310.30000000075, 137344.30000000075, 137392.6000000015, 137420.1000000015, 137454.30000000075, 137501.6000000015, 137547.0, 137572.70000000298, 137606.30000000075, 137637.80000000075, 137666.30000000075, 137698.40000000224, 137728.5, 137766.0, 137792.20000000298, 137823.0, 137854.6000000015, 137884.90000000224, 137916.80000000075, 137943.20000000298, 137976.80000000075, 138024.5, 138054.1000000015, 138085.6000000015, 138113.70000000298, 138147.6000000015, 138173.20000000298, 138202.30000000075, 138235.90000000224, 138281.80000000075, 138308.6000000015, 138341.90000000224, 138373.20000000298, 138405.6000000015, 138431.80000000075, 138465.20000000298, 138495.5, 138524.1000000015, 138556.20000000298, 138582.30000000075, 138618.0, 138664.30000000075, 138694.20000000298, 138724.90000000224, 138786.6000000015, 138814.70000000298, 138848.90000000224, 138879.30000000075, 138909.70000000298, 138941.1000000015, 139001.80000000075, 139034.6000000015, 139079.20000000298, 139108.80000000075, 139141.80000000075, 139172.5, 139198.0, 139232.80000000075, 139262.0, 139312.5, 139344.40000000224, 139373.6000000015, 139421.0, 139451.70000000298, 139481.1000000015, 139516.5, 139543.0, 139576.40000000224, 139605.80000000075, 139639.1000000015, 139683.1000000015, 139708.5, 139742.80000000075, 139776.80000000075, 139804.40000000224, 139832.80000000075, 139865.6000000015, 139896.1000000015, 139928.1000000015, 139973.1000000015, 140004.6000000015, 140030.40000000224, 140064.70000000298, 140095.70000000298, 140121.80000000075, 140159.40000000224, 140187.20000000298, 140219.80000000075, 140247.70000000298, 140282.20000000298, 140312.20000000298, 140339.6000000015, 140373.90000000224, 140406.20000000298, 140437.30000000075, 140482.5, 140515.1000000015, 140540.80000000075, 140575.20000000298, 140603.6000000015, 140636.1000000015, 140661.30000000075, 140698.5, 140723.90000000224, 140761.6000000015, 140807.30000000075, 140838.1000000015, 140865.80000000075, 140893.1000000015, 140926.80000000075, 140952.1000000015, 140987.80000000075, 141014.6000000015, 141052.6000000015, 141078.90000000224, 141113.5, 141144.80000000075, 141173.1000000015, 141202.90000000224, 141234.70000000298, 141265.70000000298, 141291.70000000298, 141328.6000000015, 141357.1000000015, 141388.70000000298, 141451.80000000075, 141480.90000000224, 141507.40000000224, 141537.20000000298, 141571.80000000075, 141604.20000000298, 141630.70000000298, 141664.80000000075, 141698.0, 141726.6000000015, 141758.40000000224, 141788.90000000224, 141820.90000000224, 141882.5, 141911.0, 141942.6000000015, 141989.0, 142035.40000000224, 142064.0, 142095.6000000015, 142127.40000000224, 142160.40000000224, 142205.6000000015, 142236.40000000224, 142263.5, 142297.0, 142341.40000000224, 142370.70000000298, 142396.70000000298, 142433.80000000075, 142464.70000000298, 142491.40000000224, 142527.5, 142555.20000000298, 142589.0, 142622.0, 142649.0, 142677.70000000298, 142710.30000000075, 142773.6000000015, 142810.6000000015, 142838.20000000298, 142878.1000000015, 142907.30000000075, 142941.6000000015, 142983.80000000075, 143019.6000000015, 143046.90000000224, 143081.40000000224, 143106.5, 143140.0, 143173.70000000298, 143201.90000000224, 143230.5, 143262.1000000015, 143293.5, 143324.5, 143372.0, 143397.1000000015, 143432.0, 143458.5, 143494.1000000015, 143540.20000000298, 143566.90000000224, 143600.6000000015, 143638.40000000224, 143671.0, 143707.80000000075, 143740.70000000298, 143800.6000000015, 143848.6000000015, 143881.30000000075, 143907.6000000015, 143941.20000000298, 143969.80000000075, 144017.0, 144046.40000000224, 144077.6000000015, 144126.30000000075, 144156.90000000224, 144187.80000000075, 144219.70000000298, 144252.5, 144298.0, 144325.90000000224, 144361.0, 144407.30000000075, 144438.6000000015, 144469.30000000075, 144500.30000000075, 144529.6000000015, 144561.90000000224, 144593.1000000015, 144621.1000000015, 144654.5, 144684.90000000224, 144717.0, 144744.80000000075, 144778.0, 144807.90000000224, 144839.20000000298, 144871.6000000015, 144896.70000000298, 144932.80000000075, 144963.30000000075, 144990.90000000224, 145025.40000000224, 145056.1000000015, 145088.5, 145131.80000000075, 145158.90000000224, 145193.80000000075, 145225.1000000015, 145256.6000000015, 145287.5, 145318.5, 145349.6000000015, 145382.1000000015, 145407.6000000015, 145441.0, 145532.6000000015, 145582.20000000298, 145631.5, 145662.6000000015"
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

    # begin = 0
    # end = 0
    # for i, gaze in enumerate(gaze_points):
    #     if gaze[2] - gaze_points[0][2] > begin_time:
    #         begin = i
    #         break
    # for i in range(len(gaze_points) - 1, 0, -1):
    #     if gaze_points[-1][2] - gaze_points[i][2] > end_time:
    #         end = i
    #         break
    # assert begin < end
    # print(f'len(end_x):{end - begin + 1}')
    return gaze_points
    return gaze_points[begin:end]


def compute_corr(
        filename: str,
        feature_of_word: list,
        feature_of_sentence: list,
        show_word: bool = True,
        show_sentence: bool = True,
        show_wander: bool = True,
        user: str = None,
):
    data = pd.read_csv(filename)
    pc_list = []
    if user:
        data = data.loc[(data["word_watching"] == 1) & (data["user"] == user)]
    else:
        data = data.loc[(data["word_watching"] == 1)]
    print(f"有效数据数量{len(data)}条")
    if show_word:
        print("---------word_understand---------")
        for feature in feature_of_word:
            col_data = np.array(data[feature])
            word_understand = np.array(data["word_understand"])
            pc = pearsonr(col_data, word_understand)
            pc_list.append(pc[0])

    if show_sentence:
        print("\n---------sentence_understand---------")
        for feature in feature_of_sentence:
            col_data = np.array(data[feature])
            word_understand = np.array(data["sentence_understand"])
            pc = pearsonr(col_data, word_understand)
            print(f"{feature}与sentence understanding:")
            print(f"相关系数：{pc[0]}")
            print(f"p value：{pc[1]}")
            print("--------------------")

    if show_wander:
        print("\n---------mind_wandering---------")
        for feature in feature_of_sentence:
            col_data = np.array(data[feature])
            word_understand = np.array(data["mind_wandering"])
            pc = pearsonr(col_data, word_understand)
            print(f"{feature}与mind wandering:")
            print(f"相关系数：{pc[0]}")
            print(f"p value：{pc[1]}")
            print("--------------------")


def kmeans_classifier(feature):
    """
    kmeans分类器
    :return:
    """
    print("feature")
    print(feature)
    kmeans = KMeans(n_clusters=2).fit(feature)
    predicted = kmeans.labels_
    print(kmeans.cluster_centers_)
    # 输出的0和1与实际标签并不是对应的，假设我们认为1一定比0多

    if np.mean(predicted) > 0.5:
        predict_list = predicted
    else:
        predict_list = [1 - x for x in predicted]

    return predict_list


def gmm_classifier(feature):
    """
    kmeans分类器
    :return:
    """

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(feature)
    predicted = gmm.predict(feature)
    # 输出的0和1与实际标签并不是对应的，假设我们认为1一定比0多

    if np.mean(predicted) > 0.5:
        predict_list = predicted
    else:
        predict_list = [1 - x for x in predicted]

    return predict_list


def evaluate(data, features: list, label="word_understand", classifier="kmeans"):
    """
    评估分类器的性能
    :return:
    """
    import numpy as np

    is_understand = np.array(data[label].to_list())

    feature = [[row[features[0]], row[features[1]]] for index, row in data.iterrows()]
    feature = np.array(feature)
    for feat in feature:
        print(feat)

    # 分类器
    if classifier == "kmeans":
        predicted = np.array(kmeans_classifier(feature))
    if classifier == "gmm":
        predicted = np.array(gmm_classifier(feature))
    if classifier == "random":
        predicted = np.array([random.randint(0, 1) for i in range(len(feature))])
    # 计算TP等
    tp = np.sum((is_understand == 0) & (predicted == 0))
    print(np.sum((is_understand == 0) & (predicted == 0)))
    fp = np.sum((is_understand == 1) & (predicted == 0))
    tn = np.sum((is_understand == 1) & (predicted == 1))
    fn = np.sum((is_understand == 0) & (predicted == 1))

    print(f"fp:{fp}")
    print(f"fn:{fn}")

    print(type(tp))
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    y_true = is_understand
    y_pred = predicted
    auc = roc_auc_score(y_true, y_pred)
    auc = 1 - auc if auc < 0.5 else auc

    kappa = cohen_kappa_score(is_understand, predicted)
    # print("precision：%f" % precision)
    # print("recall：%f" % recall)
    # print("f1 score: %f" % f1)
    # print("auc:%f" % auc)
    # print("accuracy：%f" % accuracy)
    # print("kappa:%f" % kappa)
    return accuracy, precision, recall, f1, kappa, auc


def plot_bar(data_list: list, labels: list, features: list, x_label, y_label="Proportion", ylim: float = 1.3, title=""):
    matplotlib.rc("font", family="MicroSoft YaHei")

    data_list = [np.array(list(map(abs, x))) for x in data_list]

    length = len(features)
    x = np.arange(length)  # 横坐标范围

    plt.figure()
    total_width, n = 0.8, len(labels)  # 柱状图总宽度，有几组数据
    width = total_width / n  # 单个柱状图的宽度

    loc = [x - width / n]
    for i in range(n):
        if i == 0:
            continue
        loc.append(loc[-1] + width)

    plt.ylabel(y_label)  # 纵坐标label
    plt.xlabel(x_label)  # 纵坐标label
    for i, label in enumerate(labels):
        plt.bar(loc[i], data_list[i], width=width, label=label)

    plt.xticks(x, features)
    # plt.title("relevance between features and sentence understanding")
    plt.legend()  # 给出图例
    # plt.ylim(0.0, 1.3)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_distribution(data, feature, bin, y_label="Proportion", label="sentence_understand", title=None, labels=[]):
    dat = data[feature]

    step = int((max(dat) - min(dat)) / bin)

    assert step > 0
    range_list = []
    tmp = int(min(dat))
    for i in range(bin):
        range_list.append(tmp)
        tmp += step

    range_list.append(math.ceil(max(dat)))

    understand = []
    not_understand = []

    for index in data[label].index:
        if data[label][index] == 0:
            not_understand.append(data[feature][index])
        if data[label][index] == 1:
            understand.append(data[feature][index])

    dict = {labels[0]: understand, labels[1]: not_understand}
    result_list, x_ticks = draw_hist_bar(dict, range_list, 0.3, feature, y_label, title, labels=labels)
    return result_list, x_ticks


def draw_hist_bar(nums_dict, ranges, width, x_label=None, y_label=None, title=None, labels=[]):
    counts_dict = {k: [0 for _ in range(len(ranges) - 1)] for k in nums_dict}
    for k, nums in nums_dict.items():
        for num in nums:
            idx = -1
            for i, item in enumerate(ranges):
                if i == 0:
                    continue
                if ranges[i - 1] <= num < item:
                    idx = i - 1
            if idx != -1 and idx < len(ranges) - 1:
                counts_dict[k][idx] += 1
    x_ticks = ["[%s, %s)" % (ranges[i - 1], ranges[i]) for i in range(1, len(ranges))]
    g_count = len(counts_dict)

    print(counts_dict)
    nums_of_understand = counts_dict[labels[0]]
    nums_of_not_understand = counts_dict[labels[1]]
    understand = []
    not_understand = []
    for i in range(len(counts_dict[labels[0]])):
        a = counts_dict[labels[0]][i]
        b = counts_dict[labels[1]][i]
        if a + b != 0:
            understand.append(a / (a + b))
            not_understand.append(b / (a + b))
        else:
            understand.append(0)
            not_understand.append(0)
    counts_dict[labels[0]] = understand
    counts_dict[labels[1]] = not_understand

    x = np.arange(len(x_ticks)) - (g_count - 1) * width / 2
    for k, counts in counts_dict.items():
        plt.bar(x, counts, width, label=k)
        x += width
    print(np.arange(len(x_ticks)))
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return [nums_of_understand, nums_of_not_understand], x_ticks


def get_complete_gaze_data_index(dat):
    index_list = []
    experiment_id = dat["experiment_id"][0]
    for index, row in dat.iterrows():
        if row["experiment_id"] != experiment_id:
            index_list.append(index - 1)
            experiment_id = row["experiment_id"]
    return index_list


def process_fixations(gaze_points, texts, location, use_not_blank_assumption=True, use_nlp_assumption=False):
    fixations = detect_fixations(gaze_points)

    print(f'detect_fix:{fixations}')
    fixations = keep_row(fixations)
    print(f'fix after keep row:{fixations}')

    word_list, sentence_list = get_word_and_sentence_from_text(texts)  # 获取单词和句子对应的index
    border, rows, danger_zone, len_per_word = textarea(location)
    locations = json.loads(location)

    now_max_row = -1
    assert len(word_list) == len(locations)
    adjust_fixations = []
    # 确定初始的位置
    for i, fix in enumerate(fixations):
        index, find_near = get_item_index_x_y(location=location, x=fix[0], y=fix[1], word_list=word_list,
                                              rows=rows,
                                              remove_horizontal_drift=False)
        if index != -1:
            if find_near:
                # 此处是为了可视化看起来清楚
                loc = locations[index]
                fix[0] = (loc["left"] + loc["right"]) / 2
                fix[1] = (loc["top"] + loc["bottom"]) / 2
            row_index, index_in_row = word_index_in_row(rows, index)
            if index_in_row != -1:
                adjust_fix = [fix[0], fix[1], fix[2], index, index_in_row, row_index, fix[3], fix[4]]
                adjust_fixations.append(adjust_fix)

    # 切割子序列
    sequence_fixations = []
    begin_index = 0

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
                else:
                    if len(tmp) > 0:
                        sequence_fixations.append(tmp)
                # sequence_fixations.append(adjust_fixations[begin_index:i])
                begin_index = i
                break
    if begin_index != len(adjust_fixations) - 1:
        sequence_fixations.append(adjust_fixations[begin_index:-1])
    print(f"sequence len:{len(sequence_fixations)}")

    cnt = 0

    for item in sequence_fixations:
        # print(f"从{cnt}开始裁剪")
        cnt += len(item)
    # 按行调整fixation
    word_attention = generate_word_attention(texts)
    # importance = get_importance(texts)
    result_fixations = []
    row_level_fix = []

    result_rows = []
    row_pass_time = [0 for _ in range(len(rows))]
    for i, sequence in enumerate(sequence_fixations):
        y_list = np.array([x[1] for x in sequence])
        y_mean = np.mean(y_list)
        row_index = row_index_of_sequence(rows, y_mean)
        rows_per_fix = []
        for y in y_list:
            row_index_this_fix = row_index_of_sequence(rows, y)
            rows_per_fix.append(row_index_this_fix)

        if use_not_blank_assumption:
            # 假设不会出现空行
            if row_index > now_max_row + 1:
                if row_pass_time[now_max_row] >= 2:
                    print("执行了")
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
    print(f"row_pass_time:{row_pass_time}")

    # assert sum(row_pass_time) == len(result_rows)
    print(len(result_rows))
    print(len(sequence_fixations))

    assert len(result_rows) == len(sequence_fixations)

    for i, sequence in enumerate(sequence_fixations):
        if result_rows[i] != -1:
            adjust_y = (rows[result_rows[i]]["top"] + rows[result_rows[i]]["bottom"]) / 2
            result_fixation = [[x[0], adjust_y, x[2], x[6], x[7]] for x in sequence]
            result_fixations.extend(result_fixation)
            row_level_fix.append(result_fixation)
    print(f'result_fixations:{result_fixations}')
    print(f"result_rows:{result_rows}")
    print(f"max of result rows:{max(result_rows)}")
    print(f"len of rows:{len(rows)}")
    # assert (max(result_rows) == len(rows) - 1) or (max(result_rows) == len(rows) - 2)
    return result_fixations, result_rows, row_level_fix, sequence_fixations


if __name__ == "__main__":
    # list = [1,0.85,1,0.89,0.83,1,1,1,1,1,1,0.93,0.42,0.88,0.9,1,0.92,0.89,1,1,1,1,0.81,0.95]
    list = [1, 0.89, 1, 0.89, 0.99, 1, 1, 1, 1, 1, 0.96, 0.55, 0.89, 0.97, 1, 0.96, 0.94, 1, 1, 1, 1, 0.88, 0.96]
    print(np.sum(list) / len(list))
