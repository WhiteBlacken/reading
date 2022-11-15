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
"""
import math
import os
from collections import deque

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist

import onlineReading.utils

# 设备相关信息（目前仅支持一种设备）
resolution = (1920, 1080)
inch = 23.8

# 操作信息
distance_to_screen = 60


def detect_fixations(
    gaze_points: list, min_duration: int = 200, max_duration: int = 1000, max_dispersion: int = 100
) -> list:
    """
    fixation的特征：
    + duration：
        + 200~400ms or 150~600ms
        + rarely shorter than 200ms,longer than 800ms
        + very short fixations is not meaningful for studying behavior
        + smooth pursuit is included without maximum duration
            + from other perspective，compensation for VOR（what？）
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
        if dispersion <= dispersion:
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


def detect_saccades(fixations: list, min_velocity: int = 400) -> tuple:
    """
    saccade的特征
    + velocity
        + 两个fixation之间的速度大于一个阈值
        + 400-600deg
        + https://blog.csdn.net/Kobe123brant/article/details/111264204
    """
    saccades = []
    velocities = []
    # TODO 速度的阈值 saccade的计算未找到合适的参考，以400作为阈值
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


def show_fixations_and_saccades(fixations: list, saccades: list, background: str, base_path: str, filename: str):
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    canvas = cv2.imread(background)
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
        cv2.putText(
            canvas,
            str(i),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    for i, saccade in enumerate(saccades):
        cv2.line(
            canvas,
            (int(saccade["begin"][0]), int(saccade["begin"][1])),
            (int(saccade["end"][0]), int(saccade["end"][1])),
            (0, 0, 255),
            1,
        )
    # cv2.imwrite(base_path + filename, canvas)
    # return base_path + filename
    return canvas


def join_images(img1, img2, save_path, flag="horizontal"):  # 默认是水平参数

    size1, size2 = img1.size, img2.size
    print(size1)
    print(size2)
    if flag == "horizontal":
        joint = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_path)


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

    optimal_list = [
        [574, 580],
        [582],
        [585, 588],
        [590, 591],
        [595, 598],
        [600, 605],
        [609, 610],
        [613, 619],
        [622, 625],
        [628],
        [630, 631],
        [634],
        [636],
        [637, 641],
    ]

    experiment_list_select = []
    for item in optimal_list:
        if len(item) == 2:
            for i in range(item[0], item[1] + 1):
                experiment_list_select.append(i)
        if len(item) == 1:
            experiment_list_select.append(item[0])
    print(experiment_list_select)
    print(len(experiment_list_select))

    print(pixel_2_cm(100, dpi))
