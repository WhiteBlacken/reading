import base64
import math
import os

import simplejson
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from action.models import Text, Dictionary, Dataset, WordLevelData, Dispersion
from onlineReading.utils import (
    translate,
    get_fixations,
    get_euclid_distance,
    pixel_2_cm,
    pixel_2_deg,
    cm_2_pixel,
)


def login_page(request):
    return render(request, "login.html")


def login(request):
    username = request.POST.get("username", None)
    print("username:%s" % username)
    if username:
        request.session["username"] = username
    return render(request, "calibration.html")


def index(request):
    """首页"""
    return render(request, "onlineReading.html")


def get_text(request):

    texts = Text.objects.filter(article_id=2)
    para_dict = {}
    para = 0
    for text in texts:
        words_dict = {}
        # 切成句子
        sentences = text.content.split(".")
        cnt = 0
        words_dict[0] = text.content
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
        para_dict[para] = words_dict
    # 将文本存入数据库
    dataset = Dataset.objects.create(texts="test")
    request.session["data_id"] = dataset.id
    return JsonResponse(para_dict, json_dumps_params={"ensure_ascii": False})


def get_image(data_id, username):
    """获取截图的图片+eye gaze，并生成眼动热点图"""
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

    if data_id:
        Dataset.objects.filter(id=data_id).update(
            gaze_x=str(x),
            gaze_y=str(y),
            gaze_t=str(t),
            interventions=str(interventions),
            user=request.session.get("username"),
            image=image_base64,
        )
    return HttpResponse(1)


def get_labels(request):
    labels = request.POST.get("labels")
    data_id = request.session.get("data_id", None)
    if data_id:
        Dataset.objects.filter(id=data_id).update(labels=str(labels))
    labels = list(map(int, labels.split(",")))
    WordLevelData.objects.filter(data_id=data_id).filter(
        word_index_in_text__in=labels
    ).update(is_understand=0)
    # 生成眼动图
    if data_id:
        get_image(data_id, request.session.get("username"))
    return render(request, "login.html")


def paint_image(path, coordinates):
    """在指定图片上绘图"""
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


def cal(request):
    return render(request, "calibration.html")


def reading(request):
    if not request.session.get("username", None):
        return render(request, "login.html")
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
        interventions = dataset.interventions.split(",")
        words = get_word_from_text(dataset.texts)
        for gaze in gazes:
            WordLevelData.objects.create(
                data_id=data_id,
                word_index_in_text=gaze[0],
                gaze=gaze[1],
                word=words[gaze[0]],
                is_intervention=1 if str(gaze[0]) in interventions else 0,
                is_understand=1,
            )
    return HttpResponse(1)


def get_word_from_text(text):
    get_word = []
    sentences = text.split(".")
    for sentence in sentences:
        sentence = sentence.strip()
        words = sentence.split(" ")
        for word in words:
            if len(word) > 0:
                word = word.strip().lower().replace(",", "")
                get_word.append(word)
    return get_word


def get_hot_map(request, id):
    datas = WordLevelData.objects.filter(data_id=id)
    words = {}
    dataset = Dataset.objects.get(id=id)
    get_word = get_word_from_text(dataset.texts)
    for data in datas:
        # [[1210.9457590013656, 159.23231147793268, 9945], [1217.6909072718718, 159.41882110020455, 9990.800000011921], [1230.8749738465856, 168.33542300501568, 10017.700000017881]]
        if words.get(get_word[data.word_index_in_text], -1) == -1:
            tmp = []
            # [[499.97320585558384, 500.07984655036023, 21.5]]
            # [[499.97320585558384, 500.07984655036023, 21.5]]
        else:
            tmp = words[get_word[data.word_index_in_text]]
        gaze = data.gaze[2:-2]
        gazes = gaze.split("], [")
        t = []
        for gaze in gazes:
            coordinate = gaze.split(", ")
            t.append(int(float(coordinate[2])))
        if t[-1] - t[0] > 30:
            tmp.append(t[-1] - t[0])
        if len(tmp) > 0:
            words[get_word[data.word_index_in_text]] = tmp
    # 补0画图
    max_length = 0
    words_name = []
    for key in words:
        if len(words[key]) > max_length:
            max_length = len(words[key])
        words_name.append(key)
    print(max_length)
    pic_data = []
    for key in words:
        if len(words[key]) < max_length:
            i = max_length - len(words[key])
            tmp = words[key]
            while i > 0:
                tmp.append(0)
                i = i - 1
            words[key] = tmp
        pic_data.append(words[key])
    print(pic_data)

    print(words_name)
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # harvest = np.array(pic_data)
    #
    # plt.yticks(np.arange(len(words_name)), labels=words_name)
    # print(words_name)
    # plt.title("Harvest of local farmers (in tons/year)")
    #
    # plt.imshow(harvest)
    # plt.tight_layout()
    # plt.show()

    return JsonResponse(words)


def get_dispersion(request):
    # 三个圆的坐标点
    target1 = request.POST.get("target1")
    print("target1,格式（x,y）:%s" % target1)
    target2 = request.POST.get("target2")
    print("target2,格式（x,y）:%s" % target2)
    target3 = request.POST.get("target3")
    print("target3,格式（x,y）:%s" % target3)
    # 三个圆对应的gaze点
    gaze_1_x = request.POST.get("gaze_1_x")
    print("gaze_1_x:%s" % gaze_1_x)
    gaze_1_y = request.POST.get("gaze_1_y")
    print("gaze_1_y:%s" % gaze_1_y)
    gaze_1_t = request.POST.get("gaze_1_t")
    print("gaze_1_t:%s" % gaze_1_t)

    gaze_2_x = request.POST.get("gaze_2_x")
    print("gaze_2_x:%s" % gaze_2_x)
    gaze_2_y = request.POST.get("gaze_2_y")
    print("gaze_2_y:%s" % gaze_2_y)
    gaze_2_t = request.POST.get("gaze_2_t")
    print("gaze_2_t:%s" % gaze_2_t)

    gaze_3_x = request.POST.get("gaze_3_x")
    print("gaze_3_x:%s" % gaze_3_x)
    gaze_3_y = request.POST.get("gaze_3_y")
    print("gaze_3_y:%s" % gaze_3_y)
    gaze_3_t = request.POST.get("gaze_3_t")
    print("gaze_3_t:%s" % gaze_3_t)
    # 把data記下來
    Dispersion.objects.create(
        gaze_1_x=gaze_1_x,
        gaze_1_y=gaze_1_y,
        gaze_1_t=gaze_1_t,
        gaze_2_x=gaze_2_x,
        gaze_2_y=gaze_2_y,
        gaze_2_t=gaze_2_t,
        gaze_3_x=gaze_3_x,
        gaze_3_y=gaze_3_y,
        gaze_3_t=gaze_3_t,
        user=request.session.get('username')
    )
    # 三个圆同样计算后，算均值
    # 以2为例
    offset1, dispersion1 = get_offset_and_dispersion(
        gaze_1_x, gaze_1_y, gaze_1_t, target1, 1
    )
    print("-----------output-----------------")
    print("-----------target 1-----------------")
    print("offset1:%s" % offset1)
    print("dispersion1:%s" % dispersion1)

    offset2, dispersion2 = get_offset_and_dispersion(
        gaze_2_x, gaze_2_y, gaze_2_t, target2, 1
    )
    print("-----------target 2-----------------")
    print("offset2:%s" % offset2)
    print("dispersion2:%s" % dispersion2)

    offset3, dispersion3 = get_offset_and_dispersion(
        gaze_3_x, gaze_3_y, gaze_3_t, target3, 1
    )
    print("-----------target 3-----------------")
    print("offset3:%s" % offset3)
    print("dispersion3:%s" % dispersion3)

    print("-----------mean-----------------")
    print("mean offset:%s" % ((offset1+offset2 +offset3) / 3))
    print("mean dispersion:%s" % ((dispersion1+dispersion2 + dispersion3) / 3))
    return HttpResponse(1)


def get_offset_and_dispersion(gaze_x, gaze_y, gaze_t, target, outlier):
    dispersion = 0
    offset = 0
    gaze_x = gaze_x.split(",")
    gaze_y = gaze_y.split(",")

    gaze_t = list(map(float, gaze_t.split(",")))
    target = list(map(float, target.split(",")))

    begin = 0
    for i in range(len(gaze_t)):
        if gaze_t[i] - gaze_t[0] > 500:
            begin = i
            break
    end = len(gaze_t)
    for i in range(len(gaze_t) - 1, -1, -1):
        if gaze_t[len(gaze_t) - 1] - gaze_t[i] > 500:
            end = i
            break

    gaze_x = gaze_x[begin:end]
    gaze_y = gaze_y[begin:end]
    # 计算distance数组
    distance = []
    for i in range(len(gaze_x)):
        distance.append(get_euclid_distance(gaze_x[i], target[0], gaze_y[i], target[1]))
    tmp_x = []
    tmp_y = []
    if outlier == 1:
        # 表示去除异常点
        outliners = get_outliers_by_z_score(distance)
        # outliners = get_outliers_by_iqr(distance)
        # outliners = get_outlier_by_knn(distance)
        print("outliers:%s" % outliners)
        for i in range(len(gaze_x)):
            if i not in outliners:
                tmp_x.append(gaze_x[i])
                tmp_y.append(gaze_y[i])
        gaze_x = tmp_x
        gaze_y = tmp_y
    gaze1_index = 0
    gaze2_index = 0
    # 2. 计算
    for i in range(len(gaze_x)):
        offset = offset + get_euclid_distance(
            gaze_x[i], target[0], gaze_y[i], target[1]
        )

        for j in range(i + 1, len(gaze_x)):
            dis = get_euclid_distance(gaze_x[i], gaze_x[j], gaze_y[i], gaze_y[j])
            if dis > dispersion:
                dispersion = dis
                gaze1_index = i
                gaze2_index = j
    print(
        "dispersion的两点:(%s,%s),(%s,%s)"
        % (
            gaze_x[gaze1_index],
            gaze_y[gaze1_index],
            gaze_x[gaze2_index],
            gaze_y[gaze2_index],
        )
    )
    offset_x = (float(gaze_x[gaze1_index]) + float(gaze_x[gaze2_index])) / 2
    offset_y = (float(gaze_y[gaze1_index]) + float(gaze_y[gaze2_index])) / 2
    offset = get_euclid_distance(offset_x, target[0], offset_y, target[1])
    return pixel_2_cm(offset), pixel_2_deg(dispersion)


def get_outliers_by_z_score(data):
    # 返回非离群点的索引
    # 远离标准差3倍距离以上的数据点视为离群点
    import numpy as np

    mean_d = np.mean(data)
    std_d = np.std(data)
    outliers = []

    for i in range(len(data)):
        z_score = (data[i] - mean_d) / std_d
        if np.abs(z_score) > 2:
            outliers.append(i)
    return outliers


def get_outliers_by_iqr(data):
    from pandas import Series

    data = Series(data)
    # 四分位点内距
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    outliers = []
    for i in range(len(data)):
        if data[i] < fence_low or data[i] > fence_high:
            outliers.append(i)
    return outliers


def get_outlier_by_knn(data):
    """通过knn算法寻找离群点"""
    array = []
    for i in range(len(data)):
        tmp = []
        tmp.append(data[i])
        array.append(tmp)

    # import kNN分类器
    from pyod.models.knn import KNN

    # 训练一个kNN检测器
    # 初始化检测器clf
    clf = KNN(
        method="mean",
        n_neighbors=50,
    )
    clf.fit(array)

    # 返回训练数据X_train上的异常标签和异常分值
    # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
    outliers = []
    y_train_pred = clf.labels_
    for i in range(len(y_train_pred)):
        if y_train_pred[i] == 1:
            outliers.append(i)

    return outliers


def cm_2_pixel_test(request, k):
    return HttpResponse(cm_2_pixel(k))

def get_content_from_txt(request):
    words_dict = {}
    f = open('static/texts/1.txt', 'rb')
    content = f.readlines()
    print(content)
    words_dict[0] = content
    return JsonResponse(words_dict, json_dumps_params={"ensure_ascii": False})
