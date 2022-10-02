import base64
import json
import math
import os
import datetime

import requests
from loguru import logger

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
    path = "static/user/" + str(username) + "/"
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
    print(index_2_word)
    print(sentence_dict)
    return index_2_word, sentence_dict


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
    rows_fixation = []
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
            rows_fixation.append(row_dict)
            row_index = row_index + 1
            begin_word = i
    return rows_fixation


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


def get_proportion_of_horizontal_saccades(fixations, locations):
    pre_row = 0
    vertical_saccade = 0
    for fixation in fixations:
        now_row = get_item_index_x_y(locations, fixation[0], fixation[1])
        if pre_row != now_row and now_row != -1:
            vertical_saccade = vertical_saccade + 1
    return (
        (len(fixations) - vertical_saccade) / len(fixations)
        if len(fixations) != 0
        else 0
    )


def get_saccade_angle(fixation1, fixation2):
    """获得saccade的角度"""
    vertical_dis = abs(fixation2[1] - fixation1[1])
    horizontal_dis = abs(fixation2[0] - fixation1[0])
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
    reading_times = {}
    fixation_cnt = 0
    for fixation in fixations:
        index = get_item_index_x_y(locations, fixation[0], fixation[1])
        if index != -1:
            if fixation_cnt - pre_fixation[index] > 1:
                if index in reading_times.keys():
                    tmp = reading_times[index] + 1
                    reading_times[index] = tmp
                else:
                    reading_times[index] = 1
            pre_fixation[index] = fixation_cnt
        fixation_cnt = fixation_cnt + 1
    return reading_times


def get_reading_times_and_dwell_time_of_sentence(fixations, buttons_location, sentence_dict):
    pre_fixations = [-2 for x in range(0, len(sentence_dict))]
    fixation_cnt = 0
    reading_times = {}
    first_fixations = [[] for x in range(0, len(sentence_dict))]
    second_fixations = [[] for x in range(0, len(sentence_dict))]
    dwell_time_fixations = [first_fixations, second_fixations]
    print(sentence_dict)
    for fixation in fixations:
        index = get_item_index_x_y(buttons_location, fixation[0], fixation[1])
        sentence = get_sentence_by_word(index, sentence_dict)
        if sentence != -1:
            if fixation_cnt - pre_fixations[sentence] > 1:
                # 求reading_times
                if sentence in reading_times.keys():
                    tmp = reading_times[sentence] + 1
                    reading_times[sentence] = tmp
                else:
                    reading_times[sentence] = 1
            else:
                # 求dwell_times
                if reading_times[sentence] == 1:
                    dwell_time_fixations[0][sentence].append(fixation)
                if reading_times[sentence] == 2:
                    dwell_time_fixations[1][sentence].append(fixation)
            pre_fixations[sentence] = fixation_cnt
        fixation_cnt = fixation_cnt + 1
    # 计算dwell_time TODO：此处是按照fixation算的，应该用gaze
    dwell_time = []
    for sentence in dwell_time_fixations:
        sentence_dwell = []
        for sentence_fixation in sentence:
            sum_dwell = 0
            for fix in sentence_fixation:
                sum_dwell = sum_dwell + fix[2]
            sentence_dwell.append(sum_dwell)
        dwell_time.append(sentence_dwell)

    return reading_times, dwell_time


def get_sentence_by_word(word_index, sentences):
    """判断单词在哪个句子中"""
    index = 0
    for key in sentences:
        print(sentences[key])
        if word_index < sentences[key]['end_word_index'] and word_index >= sentences[key]['begin_word_index']:
            return index
        index = index + 1
    return -1


if __name__ == "__main__":
    location = (
        '[{"left":330,"top":95,"right":408.15625,"bottom":326.984375},{"left":408.15625,"top":95,'
        '"right":445.5,"bottom":326.984375}] '
    )
    # index = get_word_index_x_y(location, 440, 322)
    # print(index)
    # content = "The Coral Sea, reserve would cover almost 990 000 square kilometers and stretch as far as 1100 kilometers from the coast. Unveiled recently by environment minister Tony Burke, the proposal would be the last in a series of proposed marine reserves around Australia's coast."
    # get_sentence_by_word_index(content)

    # bottom":326.984375},{"left":1055.796875,"top":95,"right":1113.015625,"bottom":326.984375},{"left":1113.015625,"top":95,"right":1162.203125,"bottom":326.984375},{"left":1162.203125,"top":95,"right":1231.65625,"bottom":326.984375},{"left":1231.65625,"top":95,"right":1283.859375,"bottom":326.984375},{"left":1283.859375,"top":95,"right":1315.421875,"bottom":326.984375},{"left":1315.421875,"top":95,"right":1343.21875,"bottom":326.984375},{"left":1343.21875,"top":95,"right":1430.078125,"bottom":326.984375},{"left":1430.078125,"top":95,"right":1507.3125,"bottom":326.984375},{"left":1507.3125,"top":95,"right":1543.859375,"bottom":326.984375},{"left":1543.859375,"top":95,"right":1678.296875,"bottom":326.984375},{"left":1678.296875,"top":95,"right":1722.234375,"bottom":326.984375},{"left":330,"top":326.984375,"right":379.1875,"bottom":558.96875},{"left":379.1875,"top":326.984375,"right":526.671875,"bottom":558.96875},{"left":526.671875,"top":326.984375,"right":596.25,"bottom":558.96875},{"left":596.25,"top":326.984375,"right":632.796875,"bottom":558.96875},{"left":632.796875,"top":326.984375,"right":742.984375,"bottom":558.96875},{"left":742.984375,"top":326.984375,"right":772.65625,"bottom":558.96875},{"left":772.65625,"top":326.984375,"right":800.453125,"bottom":558.96875},{"left":800.453125,"top":326.984375,"right":906.015625,"bottom":558.96875},{"left":906.015625,"top":326.984375,"right":946.9375,"bottom":558.96875},{"left":946.9375,"top":326.984375,"right":996.125,"bottom":558.96875},{"left":996.125,"top":326.984375,"right":1114.328125,"bottom":558.96875}'
    # content = '[{"left":330,"top":95,"right":360.2250003814697,"bottom":267},{"left":360.2250061035156,"top":95,"right":391.7750072479248,"bottom":267},{"left":391.7749938964844,"top":95,"right":442.4249954223633,"bottom":267},{"left":442.4250183105469,"top":95,"right":589.6500244140625,"bottom":267},{"left":589.6500244140625,"top":95,"right":626.9875259399414,"bottom":267},{"left":626.9874877929688,"top":95,"right":675.9499893188477,"bottom":267},{"left":675.9500122070312,"top":95,"right":732.7750129699707,"bottom":267},{"left":732.7750244140625,"top":95,"right":773.7125244140625,"bottom":267},{"left":773.7125244140625,"top":95,"right":891.0625228881836,"bottom":267},{"left":891.0625,"top":95,"right":946.4375,"bottom":267},{"left":946.4375,"top":95,"right":1029.912498474121,"bottom":267},{"left":1029.9124755859375,"top":95,"right":1084.674976348877,"bottom":267},{"left":1084.675048828125,"top":95,"right":1169.587547302246,"bottom":267},{"left":1169.5875244140625,"top":95,"right":1224.637523651123,"bottom":267},{"left":1224.6375732421875,"top":95,"right":1272.375072479248,"bottom":267},{"left":1272.375,"top":95,"right":1321.5625,"bottom":267},{"left":1321.5625,"top":95,"right":1388.9250030517578,"bottom":267},{"left":330,"top":267,"right":438.9250030517578,"bottom":439},{"left":438.9250183105469,"top":267,"right":475.46252059936523,"bottom":439},{"left":475.4624938964844,"top":267,"right":566.5374984741211,"bottom":439},{"left":566.5375366210938,"top":267,"right":646.8375396728516,"bottom":439},{"left":646.8375244140625,"top":267,"right":757.4500274658203,"bottom":439},{"left":757.4500122070312,"top":267,"right":851.337516784668,"bottom":439},{"left":851.3375244140625,"top":267,"right":924.8000259399414,"bottom":439},{"left":924.7999877929688,"top":267,"right":1073.6124877929688,"bottom":439},{"left":1073.612548828125,"top":267,"right":1110.950050354004,"bottom":439},{"left":1110.9500732421875,"top":267,"right":1196.8750762939453,"bottom":439},{"left":1196.875,"top":267,"right":1251.4500007629395,"bottom":439},{"left":1251.4500732421875,"top":267,"right":1333.2125778198242,"bottom":439},{"left":1333.2125244140625,"top":267,"right":1364.7625255584717,"bottom":439},{"left":330,"top":439,"right":379.1875,"bottom":611},{"left":379.1875,"top":439,"right":421.54999923706055,"bottom":611},{"left":421.5500183105469,"top":439,"right":488.9125213623047,"bottom":611},{"left":488.9125061035156,"top":439,"right":566.1750106811523,"bottom":611},{"left":566.1749877929688,"top":439,"right":661.0249862670898,"bottom":611},{"left":661.0250244140625,"top":439,"right":703.7375259399414,"bottom":611},{"left":703.7374877929688,"top":439,"right":778.2749862670898,"bottom":611},{"left":778.2750244140625,"top":439,"right":839.3750267028809,"bottom":611},{"left":839.375,"top":439,"right":870.9250011444092,"bottom":611},{"left":870.9249877929688,"top":439,"right":898.7249889373779,"bottom":611},{"left":898.7250366210938,"top":439,"right":977.1625366210938,"bottom":611},{"left":977.1625366210938,"top":439,"right":1056.9500350952148,"bottom":611},{"left":1056.9500732421875,"top":439,"right":1189.5625762939453,"bottom":611},{"left":1189.5625,"top":439,"right":1240.0375022888184,"bottom":611},{"left":1240.0374755859375,"top":439,"right":1333.4249801635742,"bottom":611},{"left":1333.425048828125,"top":439,"right":1382.612548828125,"bottom":611},{"left":330,"top":611,"right":432.4124984741211,"bottom":783},{"left":432.4125061035156,"top":611,"right":463.9625072479248,"bottom":783},{"left":463.9624938964844,"top":611,"right":513.1499938964844,"bottom":783},{"left":513.1500244140625,"top":611,"right":574.2125244140625,"bottom":783},{"left":574.2125244140625,"top":611,"right":637.2125244140625,"bottom":783}]'
    # content = '[{"left":330,"top":95,"right":408.15625,"bottom":266.984375},{"left":408.15625,"top":95,"right":445.5,"bottom":266.984375},{"left":445.5,"top":95,"right":518.6875,"bottom":266.984375},{"left":518.6875,"top":95,"right":589.140625,"bottom":266.984375},{"left":589.140625,"top":95,"right":645.03125,"bottom":266.984375},{"left":645.03125,"top":95,"right":725.46875,"bottom":266.984375},{"left":725.46875,"top":95,"right":780.046875,"bottom":266.984375},{"left":780.046875,"top":95,"right":836.4375,"bottom":266.984375},{"left":836.4375,"top":95,"right":942.625,"bottom":266.984375},{"left":942.625,"top":95,"right":979.171875,"bottom":266.984375},{"left":979.171875,"top":95,"right":1055.796875,"bottom":266.984375},{"left":1055.796875,"top":95,"right":1113.015625,"bottom":266.984375},{"left":1113.015625,"top":95,"right":1162.203125,"bottom":266.984375},{"left":1162.203125,"top":95,"right":1231.65625,"bottom":266.984375},{"left":1231.65625,"top":95,"right":1283.859375,"bottom":266.984375},{"left":1283.859375,"top":95,"right":1315.421875,"bottom":266.984375},{"left":1315.421875,"top":95,"right":1343.21875,"bottom":266.984375},{"left":1343.21875,"top":95,"right":1430.078125,"bottom":266.984375},{"left":1430.078125,"top":95,"right":1507.3125,"bottom":266.984375},{"left":1507.3125,"top":95,"right":1543.859375,"bottom":266.984375},{"left":1543.859375,"top":95,"right":1678.296875,"bottom":266.984375},{"left":1678.296875,"top":95,"right":1722.234375,"bottom":266.984375},{"left":1722.234375,"top":95,"right":1771.421875,"bottom":266.984375},{"left":330,"top":266.984375,"right":477.484375,"bottom":438.96875},{"left":477.484375,"top":266.984375,"right":547.0625,"bottom":438.96875},{"left":547.0625,"top":266.984375,"right":583.609375,"bottom":438.96875},{"left":583.609375,"top":266.984375,"right":693.796875,"bottom":438.96875},{"left":693.796875,"top":266.984375,"right":723.46875,"bottom":438.96875},{"left":723.46875,"top":266.984375,"right":751.265625,"bottom":438.96875},{"left":751.265625,"top":266.984375,"right":856.828125,"bottom":438.96875},{"left":856.828125,"top":266.984375,"right":897.75,"bottom":438.96875},{"left":897.75,"top":266.984375,"right":946.9375,"bottom":438.96875},{"left":946.9375,"top":266.984375,"right":1065.140625,"bottom":438.96875},{"left":1065.140625,"top":266.984375,"right":1205.9375,"bottom":438.96875},{"left":1205.9375,"top":266.984375,"right":1271.125,"bottom":438.96875},{"left":1271.125,"top":266.984375,"right":1354.71875,"bottom":438.96875},{"left":1354.71875,"top":266.984375,"right":1403.90625,"bottom":438.96875},{"left":1403.90625,"top":266.984375,"right":1485.890625,"bottom":438.96875},{"left":1485.890625,"top":266.984375,"right":1535.765625,"bottom":438.96875},{"left":1535.765625,"top":266.984375,"right":1592.671875,"bottom":438.96875},{"left":1592.671875,"top":266.984375,"right":1690.75,"bottom":438.96875},{"left":1690.75,"top":266.984375,"right":1739.9375,"bottom":438.96875},{"left":330,"top":438.96875,"right":418.40625,"bottom":610.953125},{"left":418.40625,"top":438.96875,"right":503.28125,"bottom":610.953125}]'
    # content = '[{"left":330,"top":95,"right":376.90625,"bottom":266.984375},{"left":376.90625,"top":95,"right":431.71875,"bottom":266.984375},{"left":431.71875,"top":95,"right":533.4375,"bottom":266.984375},{"left":533.4375,"top":95,"right":582.625,"bottom":266.984375},{"left":582.625,"top":95,"right":650.578125,"bottom":266.984375},{"left":650.578125,"top":95,"right":685.40625,"bottom":266.984375},{"left":685.40625,"top":95,"right":770.578125,"bottom":266.984375},{"left":770.578125,"top":95,"right":807.125,"bottom":266.984375},{"left":807.125,"top":95,"right":853.984375,"bottom":266.984375},{"left":853.984375,"top":95,"right":908.5625,"bottom":266.984375},{"left":908.5625,"top":95,"right":1015.234375,"bottom":266.984375},{"left":1015.234375,"top":95,"right":1095.03125,"bottom":266.984375},{"left":1095.03125,"top":95,"right":1195.015625,"bottom":266.984375},{"left":1195.015625,"top":95,"right":1231.5625,"bottom":266.984375},{"left":1231.5625,"top":95,"right":1272.5,"bottom":266.984375},{"left":1272.5,"top":95,"right":1349.765625,"bottom":266.984375},{"left":1349.765625,"top":95,"right":1482.65625,"bottom":266.984375},{"left":1482.65625,"top":95,"right":1533.375,"bottom":266.984375},{"left":1533.375,"top":95,"right":1582.5625,"bottom":266.984375},{"left":1582.5625,"top":95,"right":1655.28125,"bottom":266.984375},{"left":1655.28125,"top":95,"right":1700.390625,"bottom":266.984375},{"left":330,"top":266.984375,"right":438.46875,"bottom":438.96875},{"left":438.46875,"top":266.984375,"right":498.046875,"bottom":438.96875},{"left":498.046875,"top":266.984375,"right":598.859375,"bottom":438.96875},{"left":598.859375,"top":266.984375,"right":639.78125,"bottom":438.96875},{"left":639.78125,"top":266.984375,"right":767.484375,"bottom":438.96875},{"left":767.484375,"top":266.984375,"right":892.96875,"bottom":438.96875},{"left":892.96875,"top":266.984375,"right":982.09375,"bottom":438.96875},{"left":982.09375,"top":266.984375,"right":1133.6875,"bottom":438.96875},{"left":1133.6875,"top":266.984375,"right":1188.265625,"bottom":438.96875},{"left":1188.265625,"top":266.984375,"right":1275.265625,"bottom":438.96875},{"left":1275.265625,"top":266.984375,"right":1333.0625,"bottom":438.96875},{"left":1333.0625,"top":266.984375,"right":1382.25,"bottom":438.96875},{"left":1382.25,"top":266.984375,"right":1454.90625,"bottom":438.96875},{"left":1454.90625,"top":266.984375,"right":1486.46875,"bottom":438.96875},{"left":1486.46875,"top":266.984375,"right":1594.3125,"bottom":438.96875},{"left":1594.3125,"top":266.984375,"right":1635.234375,"bottom":438.96875},{"left":1635.234375,"top":266.984375,"right":1741.484375,"bottom":438.96875},{"left":330,"top":438.96875,"right":394.796875,"bottom":610.953125},{"left":394.796875,"top":438.96875,"right":443.984375,"bottom":610.953125},{"left":443.984375,"top":438.96875,"right":559.21875,"bottom":610.953125},{"left":559.21875,"top":438.96875,"right":616.046875,"bottom":610.953125},{"left":616.046875,"top":438.96875,"right":687.6875,"bottom":610.953125},{"left":687.6875,"top":438.96875,"right":752.484375,"bottom":610.953125},{"left":752.484375,"top":438.96875,"right":902.84375,"bottom":610.953125},{"left":902.84375,"top":438.96875,"right":939.390625,"bottom":610.953125},{"left":939.390625,"top":438.96875,"right":1127.46875,"bottom":610.953125},{"left":1127.46875,"top":438.96875,"right":1238.9375,"bottom":610.953125}]'
    content = '[{"left":330,"top":95,"right":360.21875,"bottom":266.984375},{"left":360.21875,"top":95,"right":391.78125,"bottom":266.984375},{"left":391.78125,"top":95,"right":442.4375,"bottom":266.984375},{"left":442.4375,"top":95,"right":589.671875,"bottom":266.984375},{"left":589.671875,"top":95,"right":627.015625,"bottom":266.984375},{"left":627.015625,"top":95,"right":675.984375,"bottom":266.984375},{"left":675.984375,"top":95,"right":732.8125,"bottom":266.984375},{"left":732.8125,"top":95,"right":773.75,"bottom":266.984375},{"left":773.75,"top":95,"right":891.109375,"bottom":266.984375},{"left":891.109375,"top":95,"right":946.484375,"bottom":266.984375},{"left":946.484375,"top":95,"right":1029.96875,"bottom":266.984375},{"left":1029.96875,"top":95,"right":1084.734375,"bottom":266.984375},{"left":1084.734375,"top":95,"right":1169.65625,"bottom":266.984375},{"left":1169.65625,"top":95,"right":1224.703125,"bottom":266.984375},{"left":1224.703125,"top":95,"right":1272.453125,"bottom":266.984375},{"left":1272.453125,"top":95,"right":1321.640625,"bottom":266.984375},{"left":1321.640625,"top":95,"right":1389,"bottom":266.984375},{"left":1389,"top":95,"right":1497.9375,"bottom":266.984375},{"left":1497.9375,"top":95,"right":1534.484375,"bottom":266.984375},{"left":1534.484375,"top":95,"right":1625.5625,"bottom":266.984375},{"left":1625.5625,"top":95,"right":1705.875,"bottom":266.984375},{"left":330,"top":266.984375,"right":440.625,"bottom":438.96875},{"left":440.625,"top":266.984375,"right":534.53125,"bottom":438.96875},{"left":534.53125,"top":266.984375,"right":608,"bottom":438.96875},{"left":608,"top":266.984375,"right":756.828125,"bottom":438.96875},{"left":756.828125,"top":266.984375,"right":794.171875,"bottom":438.96875},{"left":794.171875,"top":266.984375,"right":880.109375,"bottom":438.96875},{"left":880.109375,"top":266.984375,"right":934.6875,"bottom":438.96875},{"left":934.6875,"top":266.984375,"right":1016.453125,"bottom":438.96875},{"left":1016.453125,"top":266.984375,"right":1048.015625,"bottom":438.96875},{"left":1048.015625,"top":266.984375,"right":1097.203125,"bottom":438.96875},{"left":1097.203125,"top":266.984375,"right":1139.578125,"bottom":438.96875},{"left":1139.578125,"top":266.984375,"right":1206.953125,"bottom":438.96875},{"left":1206.953125,"top":266.984375,"right":1284.21875,"bottom":438.96875},{"left":1284.21875,"top":266.984375,"right":1379.078125,"bottom":438.96875},{"left":1379.078125,"top":266.984375,"right":1421.796875,"bottom":438.96875},{"left":1421.796875,"top":266.984375,"right":1496.34375,"bottom":438.96875},{"left":1496.34375,"top":266.984375,"right":1557.453125,"bottom":438.96875},{"left":1557.453125,"top":266.984375,"right":1589.015625,"bottom":438.96875},{"left":1589.015625,"top":266.984375,"right":1616.8125,"bottom":438.96875},{"left":1616.8125,"top":266.984375,"right":1695.265625,"bottom":438.96875},{"left":1695.265625,"top":266.984375,"right":1775.0625,"bottom":438.96875},{"left":330,"top":438.96875,"right":462.625,"bottom":610.953125},{"left":462.625,"top":438.96875,"right":513.109375,"bottom":610.953125},{"left":513.109375,"top":438.96875,"right":606.5,"bottom":610.953125},{"left":606.5,"top":438.96875,"right":655.6875,"bottom":610.953125},{"left":655.6875,"top":438.96875,"right":758.109375,"bottom":610.953125},{"left":758.109375,"top":438.96875,"right":789.671875,"bottom":610.953125},{"left":789.671875,"top":438.96875,"right":838.859375,"bottom":610.953125},{"left":838.859375,"top":438.96875,"right":899.921875,"bottom":610.953125},{"left":899.921875,"top":438.96875,"right":962.9375,"bottom":610.953125}]'
    result = get_row_location(content)
    print(result)
    pass
