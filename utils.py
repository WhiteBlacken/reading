import base64
import json
import math
import os


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


if __name__ == "__main__":
    location = (
        '[{"left":330,"top":95,"right":408.15625,"bottom":326.984375},{"left":408.15625,"top":95,'
        '"right":445.5,"bottom":326.984375}] '
    )
    index = get_word_index_x_y(location, 440, 322)
    print(index)
    pass
