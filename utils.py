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
    max_distance = 150
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
        # 如果队列中两点距离超过max_distance，则不是一个fixation
        if with_distance(working_queue[-1], working_queue[0], max_distance):
            # not a fixation,move forward
            working_queue.popleft()
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


def add_fixations_to_word(fixations, locations):
    """
    给出fixations，将fixation与单词对应起来
    :param fixations: [(坐标x,坐标y,durations),(坐标x,坐标y,durations)]
    :param locations: '[{"left:220,"top":23,"right":222,"bottom":222},{"left:220,"top":23,"right":222,"bottom":222}]'
    :return: {"0":[(坐标x,坐标y,durations),(坐标x,坐标y,durations)],"3":...}
    """
    words_fixations = {}
    for fixation in fixations:
        index = get_word_index_x_y(locations, fixation[0], fixation[1])
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


def get_word_index_x_y(location, x, y):
    """根据所有word的位置，当前给出的x,y,判断其在哪个word里"""
    # 解析location
    words = json.loads(location)

    index = 0
    for word in words:
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
            sentence_cnt = sentence_cnt + 1
            sentence_dict[sentence_cnt] = dict
    print(index_2_word)
    print(sentence_dict)
    return index_2_word, sentence_dict


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


if __name__ == "__main__":
    location = (
        '[{"left":330,"top":95,"right":408.15625,"bottom":326.984375},{"left":408.15625,"top":95,'
        '"right":445.5,"bottom":326.984375}] '
    )
    # index = get_word_index_x_y(location, 440, 322)
    # print(index)
    content = "The Coral Sea, reserve would cover almost 990 000 square kilometers and stretch as far as 1100 kilometers from the coast. Unveiled recently by environment minister Tony Burke, the proposal would be the last in a series of proposed marine reserves around Australia's coast."
    get_sentence_by_word_index(content)

    pass
