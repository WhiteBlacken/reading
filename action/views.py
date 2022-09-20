import datetime
import json

import requests
from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from loguru import logger

from onlineReading import settings


def translate(content):
    """
    翻译接口
    :return:
    """
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
