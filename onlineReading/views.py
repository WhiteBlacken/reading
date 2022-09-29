import base64
import json
import math
import os

import simplejson
from django.core import serializers
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from action.models import (
    Text,
    Dictionary,
    PageData,
    WordLevelData,
    Dispersion,
    Paragraph,
    Experiment,
)
from onlineReading.utils import (
    translate,
    get_euclid_distance,
    pixel_2_cm,
    pixel_2_deg,
    cm_2_pixel,
)
from utils import (
    x_y_t_2_coordinate,
    get_fixations,
    add_fixations_to_word,
    fixation_image,
    reading_times, get_word_by_index,
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


def get_all_text_available(request):
    """获取所有可以展示的文章列表"""
    texts = Text.objects.filter(is_show=True)
    texts_json = serializers.serialize("json", texts)
    return JsonResponse(
        texts_json, json_dumps_params={"ensure_ascii": False}, safe=False
    )


def get_paragraph_and_translation(request):
    """根据文章id获取整篇文章的分段以及翻译"""
    # 获取整篇文章的内容和翻译
    article_id = request.GET.get("article_id", 2)
    paragraphs = Paragraph.objects.filter(article_id=article_id)
    print(len(paragraphs))
    para_dict = {}
    para = 0
    for paragraph in paragraphs:
        words_dict = {}
        # 切成句子
        sentences = paragraph.content.split(".")
        cnt = 0
        words_dict[0] = paragraph.content
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
        para = para + 1
    # 创建一次实验
    experiment = Experiment.objects.create(
        article_id=article_id, user=request.session.get("username")
    )
    request.session["experiment_id"] = experiment.id
    return JsonResponse(para_dict, json_dumps_params={"ensure_ascii": False})


def get_page_data(request):
    """存储每页的数据"""
    image_base64 = request.POST.get("image")  # base64类型
    x = request.POST.get("x")  # str类型
    y = request.POST.get("y")  # str类型
    t = request.POST.get("t")  # str类型
    interventions = request.POST.get("interventions")
    texts = request.POST.get("text")
    page = request.POST.get("page")

    location = request.POST.get("location")

    print("interventions:%s" % interventions)
    experiment_id = request.session.get("experiment_id", None)
    if experiment_id:
        PageData.objects.create(
            gaze_x=str(x),
            gaze_y=str(y),
            gaze_t=str(t),
            texts=texts,  # todo 前端发送过来
            interventions=str(interventions),
            image=image_base64,
            page=page,  # todo 前端发送过来
            experiment_id=experiment_id,
            location=location,
        )
    return HttpResponse(1)


def get_labels(request):
    """一次性获得所有页的label，分页存储"""
    labels = request.POST.get("labels")
    experiment_id = request.session.get("experiment_id", None)
    # 示例：labels:[{"page":1,"wordLabels":[],"sentenceLabels":[[27,57]],"wanderLabels":[[0,27]]},{"page":2,"wordLabels":[36],"sentenceLabels":[],"wanderLabels":[]},{"page":3,"wordLabels":[],"sentenceLabels":[],"wanderLabels":[[0,34]]}]
    labels = json.loads(labels)

    if experiment_id:
        for label in labels:
            PageData.objects.filter(experiment_id=experiment_id).filter(
                page=label["page"]
            ).update(
                wordLabels=label["wordLabels"],
                sentenceLabels=label["sentenceLabels"],
                wanderLabels=label["wanderLabels"],
            )
    return HttpResponse(1)


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


# def get_hot_map(request, id):
#     datas = WordLevelData.objects.filter(data_id=id)
#     words = {}
#     dataset = Dataset.objects.get(id=id)
#     get_word = get_word_from_text(dataset.texts)
#     for data in datas:
#         # [[1210.9457590013656, 159.23231147793268, 9945], [1217.6909072718718, 159.41882110020455, 9990.800000011921], [1230.8749738465856, 168.33542300501568, 10017.700000017881]]
#         if words.get(get_word[data.word_index_in_text], -1) == -1:
#             tmp = []
#             # [[499.97320585558384, 500.07984655036023, 21.5]]
#             # [[499.97320585558384, 500.07984655036023, 21.5]]
#         else:
#             tmp = words[get_word[data.word_index_in_text]]
#         gaze = data.gaze[2:-2]
#         gazes = gaze.split("], [")
#         t = []
#         for gaze in gazes:
#             coordinate = gaze.split(", ")
#             t.append(int(float(coordinate[2])))
#         if t[-1] - t[0] > 30:
#             tmp.append(t[-1] - t[0])
#         if len(tmp) > 0:
#             words[get_word[data.word_index_in_text]] = tmp
#     # 补0画图
#     max_length = 0
#     words_name = []
#     for key in words:
#         if len(words[key]) > max_length:
#             max_length = len(words[key])
#         words_name.append(key)
#     print(max_length)
#     pic_data = []
#     for key in words:
#         if len(words[key]) < max_length:
#             i = max_length - len(words[key])
#             tmp = words[key]
#             while i > 0:
#                 tmp.append(0)
#                 i = i - 1
#             words[key] = tmp
#         pic_data.append(words[key])
#     print(pic_data)
#
#     print(words_name)
#     # import matplotlib.pyplot as plt
#     # import numpy as np
#     #
#     # harvest = np.array(pic_data)
#     #
#     # plt.yticks(np.arange(len(words_name)), labels=words_name)
#     # print(words_name)
#     # plt.title("Harvest of local farmers (in tons/year)")
#     #
#     # plt.imshow(harvest)
#     # plt.tight_layout()
#     # plt.show()
#
#     return JsonResponse(words)


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
        user=request.session.get("username"),
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
    print("mean offset:%s" % ((offset1 + offset2 + offset3) / 3))
    print("mean dispersion:%s" % ((dispersion1 + dispersion2 + dispersion3) / 3))
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
    f = open("static/texts/1.txt", "rb")
    content = f.readlines()
    print(content)
    words_dict[0] = content
    return JsonResponse(words_dict, json_dumps_params={"ensure_ascii": False})


def get_utils_test(request):
    page_data_id = request.GET.get("id")
    pagedata = PageData.objects.get(id=page_data_id)
    gaze_coordinates = x_y_t_2_coordinate(
        pagedata.gaze_x, pagedata.gaze_y, pagedata.gaze_t
    )
    print("len(gaze):%d" % (len(gaze_coordinates)))
    fixations = get_fixations(gaze_coordinates)
    print("fixations:%s" % fixations)
    result = add_fixations_to_word(fixations, pagedata.location)
    username = Experiment.objects.get(id=pagedata.experiment_id).user
    # fixation_image(pagedata.image, username, fixations, page_data_id)
    times = reading_times(result)
    print(times)
    return HttpResponse(result)

def analysis(request):
    # 要分析的页
    page_data_id = request.GET.get("id")
    pagedata = PageData.objects.get(id=page_data_id)
    # 组合gaze点
    gaze_coordinates = x_y_t_2_coordinate(
        pagedata.gaze_x, pagedata.gaze_y, pagedata.gaze_t
    )
    print("len(gaze):%d" % (len(gaze_coordinates)))
    # 根据gaze点计算fixation
    fixations = get_fixations(gaze_coordinates)
    word_fixation = add_fixations_to_word(fixations, pagedata.location)
    times = reading_times(word_fixation)
    words_index = get_word_by_index(pagedata.texts)
    analysis_result = {}
    for key in word_fixation:
        result = {
            'word': words_index[key],
            'fixations': word_fixation[key],
            'reading_time': times[key]
        }
        analysis_result[key] = result
    return JsonResponse(analysis_result)


