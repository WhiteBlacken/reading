import base64
import os

import simplejson
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from action.models import Text, Dictionary, Dataset, WordLevelData
from onlineReading.utils import translate, get_fixations


def login_page(request):
    return render(request, "login.html")


def login(request):
    username = request.POST.get("username")
    print("username:%s" % username)
    request.session["username"] = username
    return render(request, "calibration.html")


def index(request):
    """首页"""
    return render(request, "onlineReading.html")


def get_text(request):
    words_dict = {}
    text = Text.objects.first()
    # 去除前后的空格
    text = text.content.strip()
    # 切成句子
    sentences = text.split(".")
    cnt = 0
    words_dict[0] = text
    for sentence in sentences:
        # 去除句子前后空格
        sentence = sentence.strip()
        if len(sentence) > 3:
            # 句子长度低于 3，不是空，就是切割问题，暂时不考虑
            response = translate(sentence)
            if response["status"] == 500:
                return HttpResponse("翻译句子:%s 时出现错误" % sentence)
            sentence_zh = response["zh"]
            # 切成单词
            words = sentence.split(" ")
            for word in words:
                word = word.strip().replace(",", "")
                # 全部使用小写匹配
                dictionaries = Dictionary.objects.filter(en=word.lower())
                if dictionaries:
                    # 如果字典查得到，就从数据库中取，减少接口使用（要付费呀）
                    zh = dictionaries.first().zh
                else:
                    # 字典没有，调用接口
                    response = translate(word)
                    if response["status"] == 500:
                        return HttpResponse("翻译单词：%s 时出现错误" % word)
                    zh = response["zh"]
                    # 存入字典
                    Dictionary.objects.create(en=word.lower(), zh=zh)
                cnt = cnt + 1
                words_dict[cnt] = {"en": word, "zh": zh, "sentence_zh": sentence_zh}
    # 将文本存入数据库
    dataset = Dataset.objects.create(texts=text)
    request.session["data_id"] = dataset.id
    return JsonResponse(words_dict, json_dumps_params={"ensure_ascii": False})


def get_image(data_id, username):
    """获取截图的图片+eye gaze，并生成眼动热点图"""
    print(data_id)
    if data_id:
        dataset = Dataset.objects.get(id=data_id)
        image_base64 = dataset.image
        x = dataset.gaze_x
        y = dataset.gaze_y
        t = dataset.gaze_t

        # 1. 处理坐标
        list_x = x.split(",")
        list_y = y.split(",")
        list_t = t.split(",")

        coordinates = []
        for i, item in enumerate(list_x):
            coordinate = (
                int(float(list_x[i]) * 1920 / 1534),
                int(float(list_y[i]) * 1920 / 1534),
                int(float(list_t[i])),
            )
            coordinates.append(coordinate)

        fixations = get_fixations(coordinates)

        # 2. 处理图片
        data = image_base64.split(",")[1]
        # 将str解码为byte
        image_data = base64.b64decode(data)
        # 获取名称
        import time

        filename = time.strftime("%Y%m%d%H%M%S") + ".png"
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

    return HttpResponse("1")


def get_data(request):
    image_base64 = request.POST.get("image")  # base64类型
    x = request.POST.get("x")  # str类型
    y = request.POST.get("y")  # str类型
    t = request.POST.get("t")  # str类型
    interventions = request.POST.get("interventions")

    data_id = request.session.get("data_id", None)
    print("data_id")
    print(data_id)
    if data_id:
        Dataset.objects.filter(id=data_id).update(
            gaze_x=str(x),
            gaze_y=str(y),
            gaze_t=str(t),
            interventions=str(interventions),
            user=request.session.get("username"),
            image=image_base64
        )
    return HttpResponse(1)


def get_labels(request):
    labels = request.POST.get("labels")
    data_id = request.session.get("data_id", None)
    if data_id:
        Dataset.objects.filter(id=data_id).update(labels=str(labels))
    print("labels:%s" % str(labels))
    labels = list(map(int, labels.split(",")))
    WordLevelData.objects.filter(data_id=data_id).filter(word_index_in_text__in=labels).update(
        is_understand=0
    )
    # 生成眼动图
    if data_id:
        get_image(data_id, request.session.get("username"))
    return render(request, "login.html")


def paint_image(path, coordinates):
    """在指定图片上绘图"""
    import cv2

    img = cv2.imread(path)
    cnt = 0
    for coordinate in coordinates:
        cv2.circle(
            img,
            (coordinate[0], coordinate[1]),
            int(float(coordinate[2] / 10)),
            (0, 0, 255),
            1,
        )
        print(int(float(coordinate[2] / 10)))
        cnt = cnt + 1
    cv2.imwrite(path, img)


def cal(request):
    return render(request, "calibration.html")


def reading(request):
    return render(request, "onlineReading.html")


def test_dispersion(request):
    return render(request, "testDispersion.html")


def label(request):
    return render(request, "label.html")


def get_word_level_data(request):
    gazes = simplejson.loads(request.body)  # list类型
    data_id = request.session.get("data_id", None)

    if data_id:
        dataset = Dataset.objects.get(id=data_id)
        interventions = dataset.interventions.split(',')
        words = get_word_from_text(dataset.texts)
        for gaze in gazes:
            WordLevelData.objects.create(
                data_id=data_id,
                word_index_in_text=gaze[0],
                gaze=gaze[1],
                word=words[gaze[0]],
                is_intervention=1 if str(gaze[0]) in interventions else 0,
                is_understand=1
            )
    return HttpResponse(1)


def get_word_from_text(text):
    get_word = []
    sentences = text.split(".")
    for sentence in sentences:
        words = sentence.split(" ")
        for word in words:
            word = word.strip().lower().replace(",", "")
            get_word.append(word)
    return get_word
