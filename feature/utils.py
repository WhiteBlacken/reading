"""
所有与eye gaze计算的函数都写在这里
TODO list
+ fixation和saccade的计算
    + 配合图的生成看一下
    + 与单词的位置无关
+ 参考链接
    + http://www.fudancisl.cn/Video/pupil/-/tree/ellseg
    + https://github.com/FJK22/Pupil-Labs-Detect-Saccades-And-Saccades-Mean-Velocity
    + http://wap.medix.cn/Module/Disease/Departments/Text.aspx?code=0813000019&ivk_sa=1024320u
    + https://blog.csdn.net/Kobe123brant/article/details/111264204
    + https://www.sciencedirect.com/science/article/pii/B9780080436425500322
"""
import json
import math
import os
from collections import deque

import cv2
import numpy as np
from PIL import Image
from scipy import signal
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

import onlineReading.utils

# from utils import get_item_index_x_y

# 设备相关信息（目前仅支持一种设备）


resolution = (1920, 1080)
inch = 23.8

# 操作信息
distance_to_screen = 60


def get_sentence_by_word(word_index, sentence_list):
    if word_index == -1:
        return -1
    for i, sentence in enumerate(sentence_list):
        if sentence[2] > word_index >= sentence[1]:
            return i
    return -1


def detect_fixations(
    gaze_points: list, min_duration: int = 200, max_duration: int = 10000, max_dispersion: int = 80
) -> list:
    """
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
        while remaining_gaze:
            if remaining_gaze[0][2] > working_queue[0][2] + max_duration:
                break  # maximum data found
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


def detect_saccades(fixations: list, min_velocity: int = 70) -> tuple:
    """
    saccade的特征
    + velocity
        + 两个fixation之间的速度大于一个阈值
        + 400-600deg
        + https://blog.csdn.net/Kobe123brant/article/details/111264204
    """
    print(f"fixations:{fixations}")
    saccades = []
    velocities = []
    # TODO 速度的阈值 实际测量的数据fixation之间的平均速度在50~60左右，暂取70 对应的大概是75%
    for i, fixation in enumerate(fixations):
        if i == 0:
            continue

        # order necessary，fix1 after fix2
        vel = get_velocity(fixations[i], fixations[i - 1])

        velocities.append(vel)
        if vel > min_velocity:
            saccades.append(
                {
                    "begin": (fixations[i - 1][0], fixations[i - 1][1]),
                    "end": (fixations[i][0], fixations[i][1]),
                    "velocity": vel,
                }
            )
    return saccades, velocities  # velocities用于数据分析


def get_velocity(fix1, fix2) -> float:
    # order necessary，fix1 after fix2
    time = math.fabs(fix1[3] - fix2[4])
    dis = get_euclid_distance_of_fix(fix1, fix2)
    dpi = screen_dpi(resolution, inch)
    degree = pixel_2_deg(dis, dpi, distance_to_screen)
    return degree / time * 1000  # 单位 ms -> s


def gaze_dispersion(gaze_points: list) -> int:
    gaze_points = [[x[0], x[1]] for x in gaze_points]
    # TODO 为什么pupil lab中使用的cosine
    distances = pdist(gaze_points, metric="euclidean")  # 越相似，距离越小
    return distances.max()


def gaze_2_fixation(gaze_points: list) -> list:
    duration = gaze_points[-1][2] - gaze_points[0][2]
    x = np.mean([x[0] for x in gaze_points])
    y = np.mean([x[1] for x in gaze_points])
    begin = gaze_points[0][2]
    end = gaze_points[-1][2]
    return [x, y, duration, begin, end]


def screen_dpi(resolution: tuple, inch: float) -> float:
    # todo 不同屏幕的像素点max值
    assert resolution[0] == 1920 and resolution[1] == 1080
    if resolution[0] == 1920 and resolution[1] == 1080:
        pixelDiag = math.sqrt(math.pow(resolution[0], 2) + math.pow(resolution[1], 2))
        dpi = pixelDiag * (1534 / 1920) / inch
        return dpi
    return 1


def pixel_2_cm(pixel, dpi):
    # https://blog.csdn.net/weixin_43796392/article/details/124610034
    return pixel / dpi * 2.54


def pixel_2_deg(pixel, dpi, distance):
    """像素点到度数的转换"""
    return math.atan(pixel_2_cm(pixel, dpi) / distance) * 180 / math.pi


def get_euclid_distance_of_fix(fix1, fix2):
    """计算欧式距离"""
    return math.sqrt(math.pow(fix1[0] - fix2[0], 2) + math.pow(fix1[1] - fix2[1], 2))


def gaze_map(gaze_points: list, background: str, base_path: str, filename: str) -> str:
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    canvas = cv2.imread(background)
    for gaze in gaze_points:
        cv2.circle(
            canvas, (gaze[0], gaze[1]), 1, (0, 0, 255), -1  # img  # location  # size  # color BGR  # fill or not
        )
    cv2.imwrite(base_path + filename, canvas)
    return base_path + filename


def show_fixations(fixations: list, background: str):
    canvas = cv2.imread(background)
    canvas = paint_fixations(canvas, fixations)
    return canvas


def show_fixations_and_saccades(fixations: list, saccades: list, background: str):
    canvas = cv2.imread(background)
    canvas = paint_fixations(canvas, fixations)
    canvas = paint_saccades(canvas, saccades)

    return canvas


def paint_line_on_fixations(fixations: list, lines: list, background: str):
    canvas = cv2.imread(background)
    canvas = paint_fixations(canvas, fixations, line=False)
    for i, line in enumerate(lines):
        cv2.line(
            canvas,
            (int(line[0][0]), int(line[0][1])),
            (int(line[1][0]), int(line[1][1])),
            (0, 0, 255),
            1,
        )
    return canvas


def paint_fixations(image, fixations, interval=1, label=1, line=True):
    # save_path = "pic\\pilot_study\\1065\\fix.png"
    canvas = image
    fixations = [x for i, x in enumerate(fixations) if i % interval == 0]
    # for i in range(len(fixations)):
    #     fixations[i][1] = fixations[0][1]
    key_fixations = [(i, x) for i, x in enumerate(fixations) if x[2] >= 500]
    # for i in range(len(fixations)):
    #     if
    print("key_fixations: {k}".format(k=key_fixations))
    for i, fix in enumerate(fixations):
        dur = fix[2]
        l = 0
        if int(dur/50) > 18:
            l = 18
        else:
            l = int(dur/50)
        x = int(fix[0])
        y = int(fix[1])
        cv2.circle(
            canvas,
            (x, y),
            4,
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
    # for i, fix in enumerate(key_fixations):
    #     x = int(fix[0])
    #     y = int(fix[1])
    #     if i % label == 0:
    #         cv2.putText(
    #             canvas,
    #             str(i),
    #             (x, y),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7,
    #             (0, 0, 0),
    #             2,
    #         )
    # cv2.imwrite(save_path, canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # img = canvas
    # img = plt.imread(save_path)
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.imshow(img)
    # plt.axis("off")
    # for i, fix in enumerate(key_fixations):
    #     #     # if i % 3 == 0:
    #     #     #     if i == 3 or i == 6:
    #     #     #         ax.text(coordinate[0],
    #     #     #                 coordinate[1] - cal_fix_radius(coordinate[2]) - 20, str(i),
    #     #     #                 family='Times New Roman', fontsize=7, verticalalignment='center',
    #     #     #                 horizontalalignment='center', color="black")
    #     #     #     elif i == 9 or i == 12:
    #     #     #         ax.text(coordinate[0],
    #     #     #                 coordinate[1] + cal_fix_radius(coordinate[2]) + 20, str(i),
    #     #     #                 family='Times New Roman', fontsize=7, verticalalignment='center',
    #     #     #                 horizontalalignment='center', color="black")
    #     #     #     else:
    #     if fix[0] == 37:
    #         ax.text(
    #             fix[1][0] - 10,
    #             fix[1][1] - 20,
    #             str(fix[0]),
    #             family="Times New Roman",
    #             fontsize=7,
    #             verticalalignment="center",
    #             horizontalalignment="center",
    #             color="black",
    #         )
    #     elif fix[0] == 17 or fix[0] == 18:
    #         ax.text(
    #             fix[1][0] + 10,
    #             fix[1][1] - 20,
    #             str(fix[0]),
    #             family="Times New Roman",
    #             fontsize=7,
    #             verticalalignment="center",
    #             horizontalalignment="center",
    #             color="black",
    #         )
    #     else:
    #         ax.text(
    #             fix[1][0],
    #             fix[1][1]-20,
    #             str(fix[0]),
    #             family="Times New Roman",
    #             fontsize=7,
    #             verticalalignment="center",
    #             horizontalalignment="center",
    #             color="black",
    #         )
    # # else:
    # #     ax.text(coordinate[0], coordinate[1], str(i + 1), fontsize=5, verticalalignment='center',
    # #             horizontalalignment='center', color="tomato")
    # # plt.show()
    # fig.savefig(save_path, bbox_inches='tight', pad_inches=-0.1, dpi=300)

    return canvas


def paint_saccades(image, saccades):
    canvas = image
    for i, saccade in enumerate(saccades):
        cv2.line(
            canvas,
            (int(saccade["begin"][0]), int(saccade["begin"][1])),
            (int(saccade["end"][0]), int(saccade["end"][1])),
            (0, 255, 0),
            1,
        )
    return canvas


def join_images(img1, img2, save_path, flag="horizontal"):  # 默认是水平参数

    size1, size2 = img1.size, img2.size
    if flag == "horizontal":
        joint = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_path)


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


def keep_row(fixations: list, kernel_size=9):
    y = [x[1] for x in fixations]

    y = signal.medfilt(y, kernel_size=kernel_size)
    y = meanFilter(y, kernel_size)

    assert len(y) == len(fixations)
    for i, fixation in enumerate(fixations):
        fixation[1] = y[i]
    return fixations


# [{"left":330,"top":95,"right":435.109375,"bottom":147},{"left":435.109375,"top":95,"right":506,"bottom":147},{"left":506,"top":95,"right":627.59375,"bottom":147},{"left":627.59375,"top":95,"right":734.171875,"bottom":147},{"left":734.171875,"top":95,"right":809.953125,"bottom":147},{"left":809.953125,"top":95,"right":865.109375,"bottom":147},{"left":865.109375,"top":95,"right":938.46875,"bottom":147},{"left":938.46875,"top":95,"right":1044.28125,"bottom":147},{"left":1044.28125,"top":95,"right":1118.265625,"bottom":147},{"left":1118.265625,"top":95,"right":1243.421875,"bottom":147},{"left":1243.421875,"top":95,"right":1282.53125,"bottom":147},{"left":1282.53125,"top":95,"right":1382.421875,"bottom":147},{"left":1382.421875,"top":95,"right":1440.578125,"bottom":147},{"left":1440.578125,"top":95,"right":1568.078125,"bottom":147},{"left":1568.078125,"top":95,"right":1638.1875,"bottom":147},{"left":1638.1875,"top":95,"right":1746.53125,"bottom":147},{"left":330,"top":147,"right":405.953125,"bottom":199},{"left":405.953125,"top":147,"right":444.171875,"bottom":199},{"left":444.171875,"top":147,"right":497.515625,"bottom":199},{"left":497.515625,"top":147,"right":613.734375,"bottom":199},{"left":613.734375,"top":147,"right":653.734375,"bottom":199},{"left":653.734375,"top":147,"right":791.0625,"bottom":199},{"left":791.0625,"top":147,"right":902.59375,"bottom":199},{"left":902.59375,"top":147,"right":948.640625,"bottom":199},{"left":948.640625,"top":147,"right":1001.984375,"bottom":199},{"left":1001.984375,"top":147,"right":1101.703125,"bottom":199},{"left":1101.703125,"top":147,"right":1170.203125,"bottom":199},{"left":1170.203125,"top":147,"right":1252.25,"bottom":199},{"left":1252.25,"top":147,"right":1299.375,"bottom":199},{"left":1299.375,"top":147,"right":1399.421875,"bottom":199},{"left":1399.421875,"top":147,"right":1464.40625,"bottom":199},{"left":1464.40625,"top":147,"right":1543.65625,"bottom":199},{"left":1543.65625,"top":147,"right":1637.765625,"bottom":199},{"left":1637.765625,"top":147,"right":1689.484375,"bottom":199},{"left":330,"top":199,"right":448.46875,"bottom":251},{"left":448.46875,"top":199,"right":541.5,"bottom":251},{"left":541.5,"top":199,"right":624.484375,"bottom":251},{"left":624.484375,"top":199,"right":733.171875,"bottom":251},{"left":733.171875,"top":199,"right":788.328125,"bottom":251},{"left":788.328125,"top":199,"right":849.375,"bottom":251},{"left":849.375,"top":199,"right":956.5,"bottom":251},{"left":956.5,"top":199,"right":1080.46875,"bottom":251},{"left":1080.46875,"top":199,"right":1126.515625,"bottom":251},{"left":1126.515625,"top":199,"right":1179.859375,"bottom":251},{"left":1179.859375,"top":199,"right":1294.625,"bottom":251},{"left":1294.625,"top":199,"right":1333.734375,"bottom":251},{"left":1333.734375,"top":199,"right":1387.078125,"bottom":251},{"left":1387.078125,"top":199,"right":1440.515625,"bottom":251},{"left":1440.515625,"top":199,"right":1545.203125,"bottom":251},{"left":1545.203125,"top":199,"right":1675.015625,"bottom":251},{"left":1675.015625,"top":199,"right":1748.71875,"bottom":251},{"left":330,"top":251,"right":485.25,"bottom":303},{"left":485.25,"top":251,"right":554.03125,"bottom":303},{"left":554.03125,"top":251,"right":624.921875,"bottom":303},{"left":624.921875,"top":251,"right":678.265625,"bottom":303},{"left":678.265625,"top":251,"right":738.375,"bottom":303},{"left":738.375,"top":251,"right":853.109375,"bottom":303},{"left":853.109375,"top":251,"right":951.796875,"bottom":303},{"left":951.796875,"top":251,"right":990.90625,"bottom":303},{"left":990.90625,"top":251,"right":1059.796875,"bottom":303},{"left":1059.796875,"top":251,"right":1118.875,"bottom":303},{"left":1118.875,"top":251,"right":1240.96875,"bottom":303},{"left":1240.96875,"top":251,"right":1289.71875,"bottom":303},{"left":1289.71875,"top":251,"right":1333.875,"bottom":303},{"left":1333.875,"top":251,"right":1415.5625,"bottom":303},{"left":1415.5625,"top":251,"right":1508.59375,"bottom":303},{"left":1508.59375,"top":251,"right":1565.375,"bottom":303},{"left":1565.375,"top":251,"right":1644.03125,"bottom":303},{"left":1644.03125,"top":251,"right":1725.9375,"bottom":303},{"left":330,"top":303,"right":453.328125,"bottom":355},{"left":453.328125,"top":303,"right":538.875,"bottom":355},{"left":538.875,"top":303,"right":626.28125,"bottom":355},{"left":626.28125,"top":303,"right":738.328125,"bottom":355},{"left":738.328125,"top":303,"right":778.328125,"bottom":355},{"left":778.328125,"top":303,"right":844.4375,"bottom":355},{"left":844.4375,"top":303,"right":921.46875,"bottom":355},{"left":921.46875,"top":303,"right":1003.609375,"bottom":355},{"left":1003.609375,"top":303,"right":1062.6875,"bottom":355},{"left":1062.6875,"top":303,"right":1156.78125,"bottom":355},{"left":1156.78125,"top":303,"right":1274.9375,"bottom":355},{"left":1274.9375,"top":303,"right":1318.96875,"bottom":355},{"left":1318.96875,"top":303,"right":1372.3125,"bottom":355},{"left":1372.3125,"top":303,"right":1475.0625,"bottom":355},{"left":1475.0625,"top":303,"right":1614.71875,"bottom":355},{"left":1614.71875,"top":303,"right":1709.421875,"bottom":355},{"left":1709.421875,"top":303,"right":1746.609375,"bottom":355},{"left":330,"top":355,"right":383.34375,"bottom":407},{"left":383.34375,"top":355,"right":455.53125,"bottom":407},{"left":455.53125,"top":355,"right":530.15625,"bottom":407},{"left":530.15625,"top":355,"right":657.9375,"bottom":407},{"left":657.9375,"top":355,"right":736.390625,"bottom":407},{"left":736.390625,"top":355,"right":769.890625,"bottom":407},{"left":769.890625,"top":355,"right":909.40625,"bottom":407},{"left":909.40625,"top":355,"right":949.40625,"bottom":407},{"left":949.40625,"top":355,"right":1010.953125,"bottom":407},{"left":1010.953125,"top":355,"right":1085.171875,"bottom":407},{"left":1085.171875,"top":355,"right":1124.28125,"bottom":407},{"left":1124.28125,"top":355,"right":1177.625,"bottom":407},{"left":1177.625,"top":355,"right":1253.265625,"bottom":407},{"left":1253.265625,"top":355,"right":1312.671875,"bottom":407},{"left":1312.671875,"top":355,"right":1402.78125,"bottom":407},{"left":1402.78125,"top":355,"right":1497.859375,"bottom":407},{"left":1497.859375,"top":355,"right":1535.046875,"bottom":407},{"left":1535.046875,"top":355,"right":1588.390625,"bottom":407},{"left":1588.390625,"top":355,"right":1721.59375,"bottom":407},{"left":1721.59375,"top":355,"right":1786.65625,"bottom":407},{"left":330,"top":407,"right":420.28125,"bottom":459},{"left":420.28125,"top":407,"right":502.875,"bottom":459},{"left":502.875,"top":407,"right":562.28125,"bottom":459},{"left":562.28125,"top":407,"right":645.25,"bottom":459},{"left":645.25,"top":407,"right":710.640625,"bottom":459},{"left":710.640625,"top":407,"right":768.796875,"bottom":459},{"left":768.796875,"top":407,"right":890,"bottom":459},{"left":890,"top":407,"right":920.078125,"bottom":459},{"left":920.078125,"top":407,"right":1044.078125,"bottom":459},{"left":1044.078125,"top":407,"right":1099.234375,"bottom":459},{"left":1099.234375,"top":407,"right":1198.65625,"bottom":459},{"left":1198.65625,"top":407,"right":1294.828125,"bottom":459},{"left":1294.828125,"top":407,"right":1377.421875,"bottom":459},{"left":1377.421875,"top":407,"right":1486.765625,"bottom":459},{"left":1486.765625,"top":407,"right":1523.953125,"bottom":459},{"left":1523.953125,"top":407,"right":1577.296875,"bottom":459},{"left":1577.296875,"top":407,"right":1716.984375,"bottom":459},{"left":330,"top":459,"right":444.8125,"bottom":511},{"left":444.8125,"top":459,"right":504.21875,"bottom":511},{"left":504.21875,"top":459,"right":557.5625,"bottom":511},{"left":557.5625,"top":459,"right":605.34375,"bottom":511},{"left":605.34375,"top":459,"right":715.328125,"bottom":511},{"left":715.328125,"top":459,"right":754.4375,"bottom":511},{"left":754.4375,"top":459,"right":847.453125,"bottom":511},{"left":847.453125,"top":459,"right":918.578125,"bottom":511},{"left":918.578125,"top":459,"right":962.578125,"bottom":511},{"left":962.578125,"top":459,"right":1058.84375,"bottom":511},{"left":1058.84375,"top":459,"right":1123.546875,"bottom":511},{"left":1123.546875,"top":459,"right":1228.328125,"bottom":511},{"left":1228.328125,"top":459,"right":1306.78125,"bottom":511},{"left":1306.78125,"top":459,"right":1426.53125,"bottom":511},{"left":1426.53125,"top":459,"right":1538.890625,"bottom":511},{"left":1538.890625,"top":459,"right":1629.859375,"bottom":511},{"left":1629.859375,"top":459,"right":1731.21875,"bottom":511},{"left":330,"top":511,"right":430.15625,"bottom":563},{"left":430.15625,"top":511,"right":467.34375,"bottom":563},{"left":467.34375,"top":511,"right":520.6875,"bottom":563},{"left":520.6875,"top":511,"right":660.375,"bottom":563},{"left":660.375,"top":511,"right":780.96875,"bottom":563},{"left":780.96875,"top":511,"right":812.96875,"bottom":563},{"left":812.96875,"top":511,"right":896.796875,"bottom":563},{"left":896.796875,"top":511,"right":984.40625,"bottom":563},{"left":984.40625,"top":511,"right":1037.75,"bottom":563},{"left":1037.75,"top":511,"right":1126.828125,"bottom":563},{"left":1126.828125,"top":511,"right":1232.71875,"bottom":563},{"left":1232.71875,"top":511,"right":1315.3125,"bottom":563},{"left":1315.3125,"top":511,"right":1354.421875,"bottom":563},{"left":1354.421875,"top":511,"right":1400.546875,"bottom":563},{"left":1400.546875,"top":511,"right":1440.546875,"bottom":563},{"left":1440.546875,"top":511,"right":1498.78125,"bottom":563},{"left":1498.78125,"top":511,"right":1634.484375,"bottom":563},{"left":1634.484375,"top":511,"right":1678.546875,"bottom":563},{"left":1678.546875,"top":511,"right":1752.3125,"bottom":563},{"left":330,"top":563,"right":428.734375,"bottom":615},{"left":428.734375,"top":563,"right":474.859375,"bottom":615},{"left":474.859375,"top":563,"right":553.421875,"bottom":615},{"left":553.421875,"top":563,"right":612.828125,"bottom":615},{"left":612.828125,"top":563,"right":710.15625,"bottom":615},{"left":710.15625,"top":563,"right":763.5,"bottom":615},{"left":763.5,"top":563,"right":837.8125,"bottom":615},{"left":837.8125,"top":563,"right":876.921875,"bottom":615},{"left":876.921875,"top":563,"right":1004.578125,"bottom":615},{"left":1004.578125,"top":563,"right":1102.9375,"bottom":615},{"left":1102.9375,"top":563,"right":1181,"bottom":615},{"left":1181,"top":563,"right":1256.515625,"bottom":615},{"left":1256.515625,"top":563,"right":1325.40625,"bottom":615},{"left":1325.40625,"top":563,"right":1408.625,"bottom":615},{"left":1408.625,"top":563,"right":1468.03125,"bottom":615},{"left":1468.03125,"top":563,"right":1521.375,"bottom":615},{"left":1521.375,"top":563,"right":1613.546875,"bottom":615},{"left":1613.546875,"top":563,"right":1665.265625,"bottom":615},{"left":1665.265625,"top":563,"right":1763.140625,"bottom":615},{"left":330,"top":615,"right":395.0625,"bottom":667},{"left":395.0625,"top":615,"right":473.828125,"bottom":667},{"left":473.828125,"top":615,"right":551.71875,"bottom":667},{"left":551.71875,"top":615,"right":692.71875,"bottom":667},{"left":692.71875,"top":615,"right":752.125,"bottom":667},{"left":752.125,"top":615,"right":844.859375,"bottom":667},{"left":844.859375,"top":615,"right":903.015625,"bottom":667},{"left":903.015625,"top":615,"right":1016.828125,"bottom":667},{"left":1016.828125,"top":615,"right":1144.609375,"bottom":667},{"left":1144.609375,"top":615,"right":1236.453125,"bottom":667},{"left":1236.453125,"top":615,"right":1269.953125,"bottom":667},{"left":1269.953125,"top":615,"right":1378.609375,"bottom":667},{"left":1378.609375,"top":615,"right":1440.546875,"bottom":667},{"left":1440.546875,"top":615,"right":1564.546875,"bottom":667},{"left":1564.546875,"top":615,"right":1643,"bottom":667},{"left":1643,"top":615,"right":1697.109375,"bottom":667},{"left":1697.109375,"top":615,"right":1788.234375,"bottom":667},{"left":330,"top":667,"right":400.46875,"bottom":719},{"left":400.46875,"top":667,"right":476.65625,"bottom":719},{"left":476.65625,"top":667,"right":589.015625,"bottom":719},{"left":589.015625,"top":667,"right":626.203125,"bottom":719},{"left":626.203125,"top":667,"right":679.546875,"bottom":719},{"left":679.546875,"top":667,"right":745.0625,"bottom":719},{"left":745.0625,"top":667,"right":838.484375,"bottom":719},{"left":838.484375,"top":667,"right":877.59375,"bottom":719},{"left":877.59375,"top":667,"right":946.03125,"bottom":719},{"left":946.03125,"top":667,"right":986.03125,"bottom":719},{"left":986.03125,"top":667,"right":1094.375,"bottom":719},{"left":1094.375,"top":667,"right":1123.640625,"bottom":719},{"left":1123.640625,"top":667,"right":1244.84375,"bottom":719},{"left":1244.84375,"top":667,"right":1274.921875,"bottom":719},{"left":1274.921875,"top":667,"right":1357.640625,"bottom":719},{"left":1357.640625,"top":667,"right":1417.046875,"bottom":719},{"left":1417.046875,"top":667,"right":1470.390625,"bottom":719},{"left":1470.390625,"top":667,"right":1553.109375,"bottom":719},{"left":1553.109375,"top":667,"right":1632.8125,"bottom":719},{"left":1632.8125,"top":667,"right":1712.640625,"bottom":719},{"left":1712.640625,"top":667,"right":1782.1875,"bottom":719},{"left":330,"top":719,"right":409.75,"bottom":771},{"left":409.75,"top":719,"right":489.3125,"bottom":771},{"left":489.3125,"top":719,"right":533.34375,"bottom":771},{"left":533.34375,"top":719,"right":609.53125,"bottom":771},{"left":609.53125,"top":719,"right":677.3125,"bottom":771},{"left":677.3125,"top":719,"right":736.390625,"bottom":771},{"left":736.390625,"top":719,"right":837.09375,"bottom":771},{"left":837.09375,"top":719,"right":900.125,"bottom":771},{"left":900.125,"top":719,"right":929.390625,"bottom":771},{"left":929.390625,"top":719,"right":992.75,"bottom":771},{"left":992.75,"top":719,"right":1080.640625,"bottom":771},{"left":1080.640625,"top":719,"right":1163.578125,"bottom":771},{"left":1163.578125,"top":719,"right":1287.015625,"bottom":771},{"left":1287.015625,"top":719,"right":1438.25,"bottom":771},{"left":1438.25,"top":719,"right":1477.359375,"bottom":771},{"left":1477.359375,"top":719,"right":1588.984375,"bottom":771},{"left":1588.984375,"top":719,"right":1714.90625,"bottom":771},{"left":330,"top":771,"right":428.28125,"bottom":823},{"left":428.28125,"top":771,"right":481.625,"bottom":823},{"left":481.625,"top":771,"right":562.671875,"bottom":823},{"left":562.671875,"top":771,"right":680.0625,"bottom":823},{"left":680.0625,"top":771,"right":775.609375,"bottom":823},{"left":775.609375,"top":771,"right":827.328125,"bottom":823},{"left":827.328125,"top":771,"right":958.59375,"bottom":823},{"left":958.59375,"top":771,"right":1016.828125,"bottom":823},{"left":1016.828125,"top":771,"right":1142.328125,"bottom":823},{"left":1142.328125,"top":771,"right":1225.921875,"bottom":823},{"left":1225.921875,"top":771,"right":1361.203125,"bottom":823},{"left":1361.203125,"top":771,"right":1505.3125,"bottom":823},{"left":1505.3125,"top":771,"right":1576.203125,"bottom":823},{"left":1576.203125,"top":771,"right":1765.3125,"bottom":823},{"left":330,"top":823,"right":451.8125,"bottom":875},{"left":451.8125,"top":823,"right":544.828125,"bottom":875},{"left":544.828125,"top":823,"right":653.765625,"bottom":875},{"left":653.765625,"top":823,"right":801.34375,"bottom":875},{"left":801.34375,"top":823,"right":951.484375,"bottom":875},{"left":951.484375,"top":823,"right":1057.71875,"bottom":875},{"left":1057.71875,"top":823,"right":1153.265625,"bottom":875},{"left":1153.265625,"top":823,"right":1190.453125,"bottom":875},{"left":1190.453125,"top":823,"right":1248.6875,"bottom":875},{"left":1248.6875,"top":823,"right":1370.78125,"bottom":875},{"left":1370.78125,"top":823,"right":1419.53125,"bottom":875},{"left":1419.53125,"top":823,"right":1472.875,"bottom":875},{"left":1472.875,"top":823,"right":1558.84375,"bottom":875},{"left":1558.84375,"top":823,"right":1621.078125,"bottom":875},{"left":1621.078125,"top":823,"right":1684.609375,"bottom":875},{"left":1684.609375,"top":823,"right":1717.5,"bottom":875},{"left":1717.5,"top":823,"right":1787.78125,"bottom":875},{"left":330,"top":875,"right":369.109375,"bottom":927},{"left":369.109375,"top":875,"right":527.84375,"bottom":927},{"left":527.84375,"top":875,"right":589.96875,"bottom":927},{"left":589.96875,"top":875,"right":656.015625,"bottom":927},{"left":656.015625,"top":875,"right":735.84375,"bottom":927},{"left":735.84375,"top":875,"right":800.984375,"bottom":927},{"left":800.984375,"top":875,"right":879.515625,"bottom":927},{"left":879.515625,"top":875,"right":948.40625,"bottom":927},{"left":948.40625,"top":875,"right":1022.203125,"bottom":927},{"left":1022.203125,"top":875,"right":1106.6875,"bottom":927},{"left":1106.6875,"top":875,"right":1155.4375,"bottom":927},{"left":1155.4375,"top":875,"right":1213.671875,"bottom":927},{"left":1213.671875,"top":875,"right":1281.453125,"bottom":927},{"left":1281.453125,"top":875,"right":1310.71875,"bottom":927},{"left":1310.71875,"top":875,"right":1402.984375,"bottom":927},{"left":1402.984375,"top":875,"right":1433.703125,"bottom":927},{"left":1433.703125,"top":875,"right":1499.75,"bottom":927},{"left":1499.75,"top":875,"right":1557.453125,"bottom":927},{"left":1557.453125,"top":875,"right":1599.5,"bottom":927},{"left":1599.5,"top":875,"right":1668.390625,"bottom":927},{"left":1668.390625,"top":875,"right":1774.75,"bottom":927}]
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
    return border, rows, danger_zone, (rows[0]["right"] - rows[0]["left"]) / word_num_per_row


def detect_wrap(fixations: list, rows: list) -> list:
    wrap_data = []
    cnt = 0
    begin = 0
    for i, fixation in enumerate(fixations):
        if i == 0:
            continue
        if get_euclid_distance_of_fix(fixation, fixations[i - 1]) > (rows[cnt]["right"] - rows[cnt]["left"]) / 3 * 2:
            wrap_data.append([(fixations[i - 1][0], fixations[i - 1][1]), (fixation[0], fixation[1]), begin, i - 1])
            begin = i
    return wrap_data


def eye_gaze_to_feature(gaze, word_list, sentence_list, words_location, begin, pre_word_index, danger_zone):
    """
    is watching仅计算当前窗口内的，eye feature过去所有时刻的
    :param gaze: 
    :param word_list: 
    :param sentence_list: 
    :param words_location: 
    :param begin: 
    :return: 
    """ ""
    is_watching = []
    # num of fixation
    num_of_fixation_this_time = [0 for _ in word_list]
    # reading times
    reading_times_this_time = [0 for _ in word_list]
    # 阅读时长
    fixation_duration_this_time = [0 for _ in word_list]
    # 分配到每个单词上
    reading_times_of_sentence_in_word = [0 for _ in word_list]
    second_pass_dwell_time_of_sentence_in_word = [0 for _ in word_list]
    total_dwell_time_of_sentence_in_word = [0 for _ in word_list]
    saccade_times_of_sentence_word_level = [0 for _ in word_list]
    forward_times_of_sentence_word_level = [0 for _ in word_list]
    backward_times_of_sentence_word_level = [0 for _ in word_list]

    fix_in_window = keep_row(detect_fixations(gaze[begin:-1]), kernel_size=3)
    pre_fix_word_index = pre_word_index
    if len(fix_in_window) > 0:
        """word level"""
        for fixation in fix_in_window:
            index = get_item_index_x_y(
                words_location, fixation[0], fixation[1], pre_fix_word_index, danger_zone=danger_zone
            )
            if index != -1:
                is_watching.append(index)
                pre_fix_word_index = index

    fixations = keep_row(detect_fixations(gaze))  # 历史的所有fixation
    if len(fixations) > 0:
        """word level"""
        for i, fixation in enumerate(fixations):
            index = get_item_index_x_y(
                words_location, fixation[0], fixation[1], pre_word_index, danger_zone=danger_zone
            )
            if index != -1:
                num_of_fixation_this_time[index] += 1
                fixation_duration_this_time[index] += fixation[2]
                if index != pre_word_index:
                    reading_times_this_time[index] += 1
                    pre_word_index = index

        """sentence level"""
        reading_times_of_sentence = [0 for x in sentence_list]
        second_pass_dwell_time_of_sentence = [0 for x in sentence_list]
        total_dwell_time_of_sentence = [0 for x in sentence_list]

        pre_sentence_index = -1
        for fixation in fixations:
            word_index = get_item_index_x_y(words_location, fixation[0], fixation[1])
            sentence_index = get_sentence_by_word(word_index, sentence_list)
            if sentence_index != -1:
                # 累积fixation duration
                total_dwell_time_of_sentence[sentence_index] += fixation[2]
                # 计算reading times
                if sentence_index != pre_sentence_index:
                    reading_times_of_sentence[sentence_index] += 1
                # 只有在reading times是2时，才累积fixation duration
                if reading_times_of_sentence[sentence_index] == 2:
                    second_pass_dwell_time_of_sentence[sentence_index] += fixation[2]
                pre_sentence_index = sentence_index

        saccade_times_of_sentence = [0 for x in sentence_list]
        forward_times_of_sentence = [0 for x in sentence_list]
        backward_times_of_sentence = [0 for x in sentence_list]

        saccades, velocity = detect_saccades(fixations)
        for saccade in saccades:
            begin = saccade["begin"]
            end = saccade["end"]
            begin_word = get_item_index_x_y(words_location, begin[0], begin[1])
            end_word = get_item_index_x_y(words_location, end[0], end[1])
            sentence_index = get_sentence_by_word(begin_word, sentence_list)
            saccade_times_of_sentence[sentence_index] += 1
            if end_word > begin_word:
                forward_times_of_sentence[sentence_index] += 1
            else:
                backward_times_of_sentence[sentence_index] += 1

        for i, sentence in enumerate(sentence_list):
            scale = math.log((sentence[3] + 1))
            begin = sentence[1]
            end = sentence[2]

            reading_times_of_sentence_in_word[begin:end] = [
                reading_times_of_sentence[i] / scale for _ in range(end - begin)
            ]
            second_pass_dwell_time_of_sentence_in_word[begin:end] = [
                second_pass_dwell_time_of_sentence[i] / scale for _ in range(end - begin)
            ]
            total_dwell_time_of_sentence_in_word[begin:end] = [
                total_dwell_time_of_sentence[i] / scale for _ in range(end - begin)
            ]
            saccade_times_of_sentence_word_level[begin:end] = [
                saccade_times_of_sentence[i] / scale for _ in range(end - begin)
            ]
            forward_times_of_sentence_word_level[begin:end] = [
                forward_times_of_sentence[i] / scale for _ in range(end - begin)
            ]
            backward_times_of_sentence_word_level[begin:end] = [
                backward_times_of_sentence[i] / scale for _ in range(end - begin)
            ]
    print("is_watch in function")
    print(is_watching)
    return (
        num_of_fixation_this_time,
        reading_times_this_time,
        fixation_duration_this_time,
        reading_times_of_sentence_in_word,
        second_pass_dwell_time_of_sentence_in_word,
        total_dwell_time_of_sentence_in_word,
        saccade_times_of_sentence_word_level,
        forward_times_of_sentence_word_level,
        backward_times_of_sentence_word_level,
        is_watching,
        pre_word_index,
    )


def word_index_in_row(rows, word_index):
    for i, row in enumerate(rows):
        if row["end_index"] >= word_index >= row["begin_index"]:
            return i, word_index - row["begin_index"]
    return -1, -1


def row_index_of_sequence(rows, y):
    for i, row in enumerate(rows):
        if row["bottom"] >= y >= row["top"]:
            return i
    return -1


if __name__ == "__main__":
    gaze_data = [[2, 3, 3], [2, 3, 3], [3, 3, 3]]
    gaze_data1 = [[2, 3], [2, 3], [3, 3]]
    # detect_fixations(gaze_data)
    dispersion1 = gaze_dispersion(gaze_data)
    dispersion2 = gaze_dispersion(gaze_data1)
    assert dispersion1 == dispersion2

    gaze_data2 = [[2, 3, 3], [2, 3, 3], [2.1, 3, 300], [2.1, 3, 100000]]
    fixations = detect_fixations(gaze_data2)

    # 测试像素点到cm转换
    resolution = (1920, 1080)
    inch = 23.8
    dpi = screen_dpi(resolution, inch)
    print("dpi:%s" % dpi)

    assert pixel_2_cm(3, dpi) == onlineReading.utils.pixel_2_cm(3)

    # 测试像素点到度数转换
    assert pixel_2_deg(3, dpi, 60) == onlineReading.utils.pixel_2_deg(3)

    # 测试欧式距离的计算
    fix1 = (1, 1)
    fix2 = (2, 3)
    assert get_euclid_distance_of_fix(fix1, fix2) == math.sqrt(5)

    # optimal_list = [
    #     [574, 580],
    #     [582],
    #     [585, 588],
    #     [590, 591],
    #     [595, 598],
    #     [600, 605],
    #     [609, 610],
    #     [613, 619],
    #     [622, 625],
    #     [628],
    #     [630, 631],
    #     [634],
    #     [636],
    #     [637, 641],
    # ]
    optimal_list = [[577, 578]]

    experiment_list_select = []
    for item in optimal_list:
        if len(item) == 2:
            for i in range(item[0], item[1] + 1):
                experiment_list_select.append(i)
        if len(item) == 1:
            experiment_list_select.append(item[0])

    # 测试文本区域的分割
    location = '[{"left":330,"top":95,"right":435.109375,"bottom":147},{"left":435.109375,"top":95,"right":506,"bottom":147},{"left":506,"top":95,"right":627.59375,"bottom":147},{"left":627.59375,"top":95,"right":734.171875,"bottom":147},{"left":734.171875,"top":95,"right":809.953125,"bottom":147},{"left":809.953125,"top":95,"right":865.109375,"bottom":147},{"left":865.109375,"top":95,"right":938.46875,"bottom":147},{"left":938.46875,"top":95,"right":1044.28125,"bottom":147},{"left":1044.28125,"top":95,"right":1118.265625,"bottom":147},{"left":1118.265625,"top":95,"right":1243.421875,"bottom":147},{"left":1243.421875,"top":95,"right":1282.53125,"bottom":147},{"left":1282.53125,"top":95,"right":1382.421875,"bottom":147},{"left":1382.421875,"top":95,"right":1440.578125,"bottom":147},{"left":1440.578125,"top":95,"right":1568.078125,"bottom":147},{"left":1568.078125,"top":95,"right":1638.1875,"bottom":147},{"left":1638.1875,"top":95,"right":1746.53125,"bottom":147},{"left":330,"top":147,"right":405.953125,"bottom":199},{"left":405.953125,"top":147,"right":444.171875,"bottom":199},{"left":444.171875,"top":147,"right":497.515625,"bottom":199},{"left":497.515625,"top":147,"right":613.734375,"bottom":199},{"left":613.734375,"top":147,"right":653.734375,"bottom":199},{"left":653.734375,"top":147,"right":791.0625,"bottom":199},{"left":791.0625,"top":147,"right":902.59375,"bottom":199},{"left":902.59375,"top":147,"right":948.640625,"bottom":199},{"left":948.640625,"top":147,"right":1001.984375,"bottom":199},{"left":1001.984375,"top":147,"right":1101.703125,"bottom":199},{"left":1101.703125,"top":147,"right":1170.203125,"bottom":199},{"left":1170.203125,"top":147,"right":1252.25,"bottom":199},{"left":1252.25,"top":147,"right":1299.375,"bottom":199},{"left":1299.375,"top":147,"right":1399.421875,"bottom":199},{"left":1399.421875,"top":147,"right":1464.40625,"bottom":199},{"left":1464.40625,"top":147,"right":1543.65625,"bottom":199},{"left":1543.65625,"top":147,"right":1637.765625,"bottom":199},{"left":1637.765625,"top":147,"right":1689.484375,"bottom":199},{"left":330,"top":199,"right":448.46875,"bottom":251},{"left":448.46875,"top":199,"right":541.5,"bottom":251},{"left":541.5,"top":199,"right":624.484375,"bottom":251},{"left":624.484375,"top":199,"right":733.171875,"bottom":251},{"left":733.171875,"top":199,"right":788.328125,"bottom":251},{"left":788.328125,"top":199,"right":849.375,"bottom":251},{"left":849.375,"top":199,"right":956.5,"bottom":251},{"left":956.5,"top":199,"right":1080.46875,"bottom":251},{"left":1080.46875,"top":199,"right":1126.515625,"bottom":251},{"left":1126.515625,"top":199,"right":1179.859375,"bottom":251},{"left":1179.859375,"top":199,"right":1294.625,"bottom":251},{"left":1294.625,"top":199,"right":1333.734375,"bottom":251},{"left":1333.734375,"top":199,"right":1387.078125,"bottom":251},{"left":1387.078125,"top":199,"right":1440.515625,"bottom":251},{"left":1440.515625,"top":199,"right":1545.203125,"bottom":251},{"left":1545.203125,"top":199,"right":1675.015625,"bottom":251},{"left":1675.015625,"top":199,"right":1748.71875,"bottom":251},{"left":330,"top":251,"right":485.25,"bottom":303},{"left":485.25,"top":251,"right":554.03125,"bottom":303},{"left":554.03125,"top":251,"right":624.921875,"bottom":303},{"left":624.921875,"top":251,"right":678.265625,"bottom":303},{"left":678.265625,"top":251,"right":738.375,"bottom":303},{"left":738.375,"top":251,"right":853.109375,"bottom":303},{"left":853.109375,"top":251,"right":951.796875,"bottom":303},{"left":951.796875,"top":251,"right":990.90625,"bottom":303},{"left":990.90625,"top":251,"right":1059.796875,"bottom":303},{"left":1059.796875,"top":251,"right":1118.875,"bottom":303},{"left":1118.875,"top":251,"right":1240.96875,"bottom":303},{"left":1240.96875,"top":251,"right":1289.71875,"bottom":303},{"left":1289.71875,"top":251,"right":1333.875,"bottom":303},{"left":1333.875,"top":251,"right":1415.5625,"bottom":303},{"left":1415.5625,"top":251,"right":1508.59375,"bottom":303},{"left":1508.59375,"top":251,"right":1565.375,"bottom":303},{"left":1565.375,"top":251,"right":1644.03125,"bottom":303},{"left":1644.03125,"top":251,"right":1725.9375,"bottom":303},{"left":330,"top":303,"right":453.328125,"bottom":355},{"left":453.328125,"top":303,"right":538.875,"bottom":355},{"left":538.875,"top":303,"right":626.28125,"bottom":355},{"left":626.28125,"top":303,"right":738.328125,"bottom":355},{"left":738.328125,"top":303,"right":778.328125,"bottom":355},{"left":778.328125,"top":303,"right":844.4375,"bottom":355},{"left":844.4375,"top":303,"right":921.46875,"bottom":355},{"left":921.46875,"top":303,"right":1003.609375,"bottom":355},{"left":1003.609375,"top":303,"right":1062.6875,"bottom":355},{"left":1062.6875,"top":303,"right":1156.78125,"bottom":355},{"left":1156.78125,"top":303,"right":1274.9375,"bottom":355},{"left":1274.9375,"top":303,"right":1318.96875,"bottom":355},{"left":1318.96875,"top":303,"right":1372.3125,"bottom":355},{"left":1372.3125,"top":303,"right":1475.0625,"bottom":355},{"left":1475.0625,"top":303,"right":1614.71875,"bottom":355},{"left":1614.71875,"top":303,"right":1709.421875,"bottom":355},{"left":1709.421875,"top":303,"right":1746.609375,"bottom":355},{"left":330,"top":355,"right":383.34375,"bottom":407},{"left":383.34375,"top":355,"right":455.53125,"bottom":407},{"left":455.53125,"top":355,"right":530.15625,"bottom":407},{"left":530.15625,"top":355,"right":657.9375,"bottom":407},{"left":657.9375,"top":355,"right":736.390625,"bottom":407},{"left":736.390625,"top":355,"right":769.890625,"bottom":407},{"left":769.890625,"top":355,"right":909.40625,"bottom":407},{"left":909.40625,"top":355,"right":949.40625,"bottom":407},{"left":949.40625,"top":355,"right":1010.953125,"bottom":407},{"left":1010.953125,"top":355,"right":1085.171875,"bottom":407},{"left":1085.171875,"top":355,"right":1124.28125,"bottom":407},{"left":1124.28125,"top":355,"right":1177.625,"bottom":407},{"left":1177.625,"top":355,"right":1253.265625,"bottom":407},{"left":1253.265625,"top":355,"right":1312.671875,"bottom":407},{"left":1312.671875,"top":355,"right":1402.78125,"bottom":407},{"left":1402.78125,"top":355,"right":1497.859375,"bottom":407},{"left":1497.859375,"top":355,"right":1535.046875,"bottom":407},{"left":1535.046875,"top":355,"right":1588.390625,"bottom":407},{"left":1588.390625,"top":355,"right":1721.59375,"bottom":407},{"left":1721.59375,"top":355,"right":1786.65625,"bottom":407},{"left":330,"top":407,"right":420.28125,"bottom":459},{"left":420.28125,"top":407,"right":502.875,"bottom":459},{"left":502.875,"top":407,"right":562.28125,"bottom":459},{"left":562.28125,"top":407,"right":645.25,"bottom":459},{"left":645.25,"top":407,"right":710.640625,"bottom":459},{"left":710.640625,"top":407,"right":768.796875,"bottom":459},{"left":768.796875,"top":407,"right":890,"bottom":459},{"left":890,"top":407,"right":920.078125,"bottom":459},{"left":920.078125,"top":407,"right":1044.078125,"bottom":459},{"left":1044.078125,"top":407,"right":1099.234375,"bottom":459},{"left":1099.234375,"top":407,"right":1198.65625,"bottom":459},{"left":1198.65625,"top":407,"right":1294.828125,"bottom":459},{"left":1294.828125,"top":407,"right":1377.421875,"bottom":459},{"left":1377.421875,"top":407,"right":1486.765625,"bottom":459},{"left":1486.765625,"top":407,"right":1523.953125,"bottom":459},{"left":1523.953125,"top":407,"right":1577.296875,"bottom":459},{"left":1577.296875,"top":407,"right":1716.984375,"bottom":459},{"left":330,"top":459,"right":444.8125,"bottom":511},{"left":444.8125,"top":459,"right":504.21875,"bottom":511},{"left":504.21875,"top":459,"right":557.5625,"bottom":511},{"left":557.5625,"top":459,"right":605.34375,"bottom":511},{"left":605.34375,"top":459,"right":715.328125,"bottom":511},{"left":715.328125,"top":459,"right":754.4375,"bottom":511},{"left":754.4375,"top":459,"right":847.453125,"bottom":511},{"left":847.453125,"top":459,"right":918.578125,"bottom":511},{"left":918.578125,"top":459,"right":962.578125,"bottom":511},{"left":962.578125,"top":459,"right":1058.84375,"bottom":511},{"left":1058.84375,"top":459,"right":1123.546875,"bottom":511},{"left":1123.546875,"top":459,"right":1228.328125,"bottom":511},{"left":1228.328125,"top":459,"right":1306.78125,"bottom":511},{"left":1306.78125,"top":459,"right":1426.53125,"bottom":511},{"left":1426.53125,"top":459,"right":1538.890625,"bottom":511},{"left":1538.890625,"top":459,"right":1629.859375,"bottom":511},{"left":1629.859375,"top":459,"right":1731.21875,"bottom":511},{"left":330,"top":511,"right":430.15625,"bottom":563},{"left":430.15625,"top":511,"right":467.34375,"bottom":563},{"left":467.34375,"top":511,"right":520.6875,"bottom":563},{"left":520.6875,"top":511,"right":660.375,"bottom":563},{"left":660.375,"top":511,"right":780.96875,"bottom":563},{"left":780.96875,"top":511,"right":812.96875,"bottom":563},{"left":812.96875,"top":511,"right":896.796875,"bottom":563},{"left":896.796875,"top":511,"right":984.40625,"bottom":563},{"left":984.40625,"top":511,"right":1037.75,"bottom":563},{"left":1037.75,"top":511,"right":1126.828125,"bottom":563},{"left":1126.828125,"top":511,"right":1232.71875,"bottom":563},{"left":1232.71875,"top":511,"right":1315.3125,"bottom":563},{"left":1315.3125,"top":511,"right":1354.421875,"bottom":563},{"left":1354.421875,"top":511,"right":1400.546875,"bottom":563},{"left":1400.546875,"top":511,"right":1440.546875,"bottom":563},{"left":1440.546875,"top":511,"right":1498.78125,"bottom":563},{"left":1498.78125,"top":511,"right":1634.484375,"bottom":563},{"left":1634.484375,"top":511,"right":1678.546875,"bottom":563},{"left":1678.546875,"top":511,"right":1752.3125,"bottom":563},{"left":330,"top":563,"right":428.734375,"bottom":615},{"left":428.734375,"top":563,"right":474.859375,"bottom":615},{"left":474.859375,"top":563,"right":553.421875,"bottom":615},{"left":553.421875,"top":563,"right":612.828125,"bottom":615},{"left":612.828125,"top":563,"right":710.15625,"bottom":615},{"left":710.15625,"top":563,"right":763.5,"bottom":615},{"left":763.5,"top":563,"right":837.8125,"bottom":615},{"left":837.8125,"top":563,"right":876.921875,"bottom":615},{"left":876.921875,"top":563,"right":1004.578125,"bottom":615},{"left":1004.578125,"top":563,"right":1102.9375,"bottom":615},{"left":1102.9375,"top":563,"right":1181,"bottom":615},{"left":1181,"top":563,"right":1256.515625,"bottom":615},{"left":1256.515625,"top":563,"right":1325.40625,"bottom":615},{"left":1325.40625,"top":563,"right":1408.625,"bottom":615},{"left":1408.625,"top":563,"right":1468.03125,"bottom":615},{"left":1468.03125,"top":563,"right":1521.375,"bottom":615},{"left":1521.375,"top":563,"right":1613.546875,"bottom":615},{"left":1613.546875,"top":563,"right":1665.265625,"bottom":615},{"left":1665.265625,"top":563,"right":1763.140625,"bottom":615},{"left":330,"top":615,"right":395.0625,"bottom":667},{"left":395.0625,"top":615,"right":473.828125,"bottom":667},{"left":473.828125,"top":615,"right":551.71875,"bottom":667},{"left":551.71875,"top":615,"right":692.71875,"bottom":667},{"left":692.71875,"top":615,"right":752.125,"bottom":667},{"left":752.125,"top":615,"right":844.859375,"bottom":667},{"left":844.859375,"top":615,"right":903.015625,"bottom":667},{"left":903.015625,"top":615,"right":1016.828125,"bottom":667},{"left":1016.828125,"top":615,"right":1144.609375,"bottom":667},{"left":1144.609375,"top":615,"right":1236.453125,"bottom":667},{"left":1236.453125,"top":615,"right":1269.953125,"bottom":667},{"left":1269.953125,"top":615,"right":1378.609375,"bottom":667},{"left":1378.609375,"top":615,"right":1440.546875,"bottom":667},{"left":1440.546875,"top":615,"right":1564.546875,"bottom":667},{"left":1564.546875,"top":615,"right":1643,"bottom":667},{"left":1643,"top":615,"right":1697.109375,"bottom":667},{"left":1697.109375,"top":615,"right":1788.234375,"bottom":667},{"left":330,"top":667,"right":400.46875,"bottom":719},{"left":400.46875,"top":667,"right":476.65625,"bottom":719},{"left":476.65625,"top":667,"right":589.015625,"bottom":719},{"left":589.015625,"top":667,"right":626.203125,"bottom":719},{"left":626.203125,"top":667,"right":679.546875,"bottom":719},{"left":679.546875,"top":667,"right":745.0625,"bottom":719},{"left":745.0625,"top":667,"right":838.484375,"bottom":719},{"left":838.484375,"top":667,"right":877.59375,"bottom":719},{"left":877.59375,"top":667,"right":946.03125,"bottom":719},{"left":946.03125,"top":667,"right":986.03125,"bottom":719},{"left":986.03125,"top":667,"right":1094.375,"bottom":719},{"left":1094.375,"top":667,"right":1123.640625,"bottom":719},{"left":1123.640625,"top":667,"right":1244.84375,"bottom":719},{"left":1244.84375,"top":667,"right":1274.921875,"bottom":719},{"left":1274.921875,"top":667,"right":1357.640625,"bottom":719},{"left":1357.640625,"top":667,"right":1417.046875,"bottom":719},{"left":1417.046875,"top":667,"right":1470.390625,"bottom":719},{"left":1470.390625,"top":667,"right":1553.109375,"bottom":719},{"left":1553.109375,"top":667,"right":1632.8125,"bottom":719},{"left":1632.8125,"top":667,"right":1712.640625,"bottom":719},{"left":1712.640625,"top":667,"right":1782.1875,"bottom":719},{"left":330,"top":719,"right":409.75,"bottom":771},{"left":409.75,"top":719,"right":489.3125,"bottom":771},{"left":489.3125,"top":719,"right":533.34375,"bottom":771},{"left":533.34375,"top":719,"right":609.53125,"bottom":771},{"left":609.53125,"top":719,"right":677.3125,"bottom":771},{"left":677.3125,"top":719,"right":736.390625,"bottom":771},{"left":736.390625,"top":719,"right":837.09375,"bottom":771},{"left":837.09375,"top":719,"right":900.125,"bottom":771},{"left":900.125,"top":719,"right":929.390625,"bottom":771},{"left":929.390625,"top":719,"right":992.75,"bottom":771},{"left":992.75,"top":719,"right":1080.640625,"bottom":771},{"left":1080.640625,"top":719,"right":1163.578125,"bottom":771},{"left":1163.578125,"top":719,"right":1287.015625,"bottom":771},{"left":1287.015625,"top":719,"right":1438.25,"bottom":771},{"left":1438.25,"top":719,"right":1477.359375,"bottom":771},{"left":1477.359375,"top":719,"right":1588.984375,"bottom":771},{"left":1588.984375,"top":719,"right":1714.90625,"bottom":771},{"left":330,"top":771,"right":428.28125,"bottom":823},{"left":428.28125,"top":771,"right":481.625,"bottom":823},{"left":481.625,"top":771,"right":562.671875,"bottom":823},{"left":562.671875,"top":771,"right":680.0625,"bottom":823},{"left":680.0625,"top":771,"right":775.609375,"bottom":823},{"left":775.609375,"top":771,"right":827.328125,"bottom":823},{"left":827.328125,"top":771,"right":958.59375,"bottom":823},{"left":958.59375,"top":771,"right":1016.828125,"bottom":823},{"left":1016.828125,"top":771,"right":1142.328125,"bottom":823},{"left":1142.328125,"top":771,"right":1225.921875,"bottom":823},{"left":1225.921875,"top":771,"right":1361.203125,"bottom":823},{"left":1361.203125,"top":771,"right":1505.3125,"bottom":823},{"left":1505.3125,"top":771,"right":1576.203125,"bottom":823},{"left":1576.203125,"top":771,"right":1765.3125,"bottom":823},{"left":330,"top":823,"right":451.8125,"bottom":875},{"left":451.8125,"top":823,"right":544.828125,"bottom":875},{"left":544.828125,"top":823,"right":653.765625,"bottom":875},{"left":653.765625,"top":823,"right":801.34375,"bottom":875},{"left":801.34375,"top":823,"right":951.484375,"bottom":875},{"left":951.484375,"top":823,"right":1057.71875,"bottom":875},{"left":1057.71875,"top":823,"right":1153.265625,"bottom":875},{"left":1153.265625,"top":823,"right":1190.453125,"bottom":875},{"left":1190.453125,"top":823,"right":1248.6875,"bottom":875},{"left":1248.6875,"top":823,"right":1370.78125,"bottom":875},{"left":1370.78125,"top":823,"right":1419.53125,"bottom":875},{"left":1419.53125,"top":823,"right":1472.875,"bottom":875},{"left":1472.875,"top":823,"right":1558.84375,"bottom":875},{"left":1558.84375,"top":823,"right":1621.078125,"bottom":875},{"left":1621.078125,"top":823,"right":1684.609375,"bottom":875},{"left":1684.609375,"top":823,"right":1717.5,"bottom":875},{"left":1717.5,"top":823,"right":1787.78125,"bottom":875},{"left":330,"top":875,"right":369.109375,"bottom":927},{"left":369.109375,"top":875,"right":527.84375,"bottom":927},{"left":527.84375,"top":875,"right":589.96875,"bottom":927},{"left":589.96875,"top":875,"right":656.015625,"bottom":927},{"left":656.015625,"top":875,"right":735.84375,"bottom":927},{"left":735.84375,"top":875,"right":800.984375,"bottom":927},{"left":800.984375,"top":875,"right":879.515625,"bottom":927},{"left":879.515625,"top":875,"right":948.40625,"bottom":927},{"left":948.40625,"top":875,"right":1022.203125,"bottom":927},{"left":1022.203125,"top":875,"right":1106.6875,"bottom":927},{"left":1106.6875,"top":875,"right":1155.4375,"bottom":927},{"left":1155.4375,"top":875,"right":1213.671875,"bottom":927},{"left":1213.671875,"top":875,"right":1281.453125,"bottom":927},{"left":1281.453125,"top":875,"right":1310.71875,"bottom":927},{"left":1310.71875,"top":875,"right":1402.984375,"bottom":927},{"left":1402.984375,"top":875,"right":1433.703125,"bottom":927},{"left":1433.703125,"top":875,"right":1499.75,"bottom":927},{"left":1499.75,"top":875,"right":1557.453125,"bottom":927},{"left":1557.453125,"top":875,"right":1599.5,"bottom":927},{"left":1599.5,"top":875,"right":1668.390625,"bottom":927},{"left":1668.390625,"top":875,"right":1774.75,"bottom":927}]'
    # location = json.loads(location)
    border, rows, danger_zone, len_per_word = textarea(location)

    print(len_per_word)
