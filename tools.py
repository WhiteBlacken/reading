import datetime
import json
import time
from functools import wraps

import requests
from django.shortcuts import redirect
from loguru import logger

from onlineReading import settings


def login_required(func):
    @wraps(func)
    def inner(request, *args, **kwargs):
        if request.session.get('username'):
            return func(request, *args, **kwargs)
        return redirect("/go_login/")
    return inner


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
        url = f"http://api.fanyi.baidu.com/api/trans/vip/translate?q={content}&from=en&to=zh&appid={appid}&salt={salt}&sign={sign}"
        # 4. 解析结果
        response = requests.get(url)
        data = json.loads(response.text)
        endtime = datetime.datetime.now()
        logger.info(
            f"翻译接口执行时间为{round((endtime - starttime).microseconds / 1000 / 1000, 3)}ms"
        )
        return {"status": 200, "zh": data["trans_result"][0]["dst"]}
    except Exception:
        return {"status": 500, "zh": None}

