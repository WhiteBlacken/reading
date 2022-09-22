import datetime
import json
import math
from queue import Queue

import requests
from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
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


def get_fixations(coordinates):
    """
    根据gaze data(x,y,t1)计算fixation(x,y,r,t2) t2远小于t1
    :return:
    """
    from collections import deque
    fixations = []
    min_duration = 100
    max_duration = 800
    max_distance = 40
    # 先进先出队列
    working_queue = deque()
    remaining_gaze = deque(coordinates)

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
        if (
                get_euclid_distance(
                    working_queue[-1][0],
                    working_queue[0][0],
                    working_queue[-1][1],
                    working_queue[0][1],
                )
                > max_distance
        ):
            # not a fixation,move forward
            working_queue.popleft()
            continue

        # minimal fixation found,collect maximal data
        while remaining_gaze:
            datum = remaining_gaze[0]
            if datum[2] > working_queue[0][2] + max_duration:
                break  # maximum data found
            working_queue.append(remaining_gaze.popleft())

    return 1


def get_euclid_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


if __name__ == "__main__":
    pass
