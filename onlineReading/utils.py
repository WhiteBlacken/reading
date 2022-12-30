import datetime
import json
import math

import requests
from loguru import logger

from onlineReading import settings


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


def data_scale(datas, a, b):
    """将原本的数据缩放到【a，b】之间"""
    new_data = []
    max_data = max(datas)
    min_data = min(datas)
    # 计算缩放系数
    k = (b - a) / (max_data - min_data)
    for data in datas:
        new_data.append(a + k * (data - min_data))
    return new_data


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


def with_distance(gaze1, gaze2, max_distance):
    """判断两个gaze点之间的距离是否满足fixation"""
    return get_euclid_distance(gaze1[0], gaze2[0], gaze1[1], gaze2[1]) < max_distance


def get_euclid_distance(x1, x2, y1, y2):
    """计算欧式距离"""
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def pixel_2_deg(pixel):
    """像素点到度数的转换"""
    cmPerPix = 23.8 * 2.54 / math.sqrt(math.pow(16, 2) + math.pow(9, 2)) * 16 / 1534
    return math.atan(pixel * cmPerPix / 60) * 180 / math.pi


def pixel_2_cm(pixel):
    """像素点到距离的转换"""
    cmPerPix = 23.8 * 2.54 / math.sqrt(math.pow(16, 2) + math.pow(9, 2)) * 16 / 1534
    return pixel * cmPerPix


def cm_2_pixel(cm):
    """距离到像素点的转换"""
    cmPerPix = 23.8 * 2.54 / math.sqrt(math.pow(16, 2) + math.pow(9, 2)) * 16 / 1534
    return cm / cmPerPix


if __name__ == "__main__":
    print(pixel_2_cm(52 - 16))
