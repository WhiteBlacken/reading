import base64
import copy
import datetime
import json
import math
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image

from action.models import Dictionary, Dispersion, Experiment, PageData, Paragraph, Text, Translation
from model.FeatureClassify import ModelWithoutFC
from onlineReading.utils import cm_2_pixel, get_euclid_distance, pixel_2_cm, pixel_2_deg, translate
from pyheatmap import myHeatmap
from semantic_attention import (  # generate_sentence_difficulty,
    generate_sentence_attention,
    generate_sentence_difficulty,
    generate_word_attention,
    generate_word_difficulty,
    nlp,
)
from utils import (
    add_fixations_to_location,
    calculate_identity,
    calculate_similarity,
    find_threshold,
    fixation_image,
    get_fixations,
    get_importance,
    get_item_index_x_y,
    get_out_of_screen_times,
    get_proportion_of_horizontal_saccades,
    get_reading_times_and_dwell_time_of_sentence,
    get_reading_times_of_word,
    get_row_location,
    get_saccade,
    get_saccade_info,
    get_sentence_by_word_index,
    get_word_and_sentence_from_text,
    get_word_by_one_gaze,
    get_word_location,
    join_images_vertical,
    join_two_image,
    paint_bar_graph,
    preprocess_data,
    x_y_t_2_coordinate,
)

# 加载模型
model = ModelWithoutFC()
model.load_state_dict(torch.load("model/EyeFeatureModel.pth"))
model.eval()


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


def choose_text(request):
    return render(request, "chooseTxt.html")


def get_all_text_available(request):
    """获取所有可以展示的文章列表"""
    texts = Text.objects.filter(is_show=True)
    text_dict = {}
    for text in texts:
        text_dict[text.id] = text.title
    return JsonResponse(text_dict, json_dumps_params={"ensure_ascii": False}, safe=False)


def get_paragraph_and_translation(request):
    """根据文章id获取整篇文章的分段以及翻译"""
    # 获取整篇文章的内容和翻译

    article_id = request.GET.get("article_id", 20)

    paragraphs = Paragraph.objects.filter(article_id=article_id)
    print(len(paragraphs))
    para_dict = {}
    para = 0
    logger.info("--实验开始--")
    for paragraph in paragraphs:
        words_dict = {}
        # 切成句子
        sentences = paragraph.content.split(".")
        cnt = 0
        words_dict[0] = paragraph.content
        sentence_id = 0
        for sentence in sentences:
            # 去除句子前后空格
            sentence = sentence.strip()
            if len(sentence) > 3:
                # 句子长度低于 3，不是空，就是切割问题，暂时不考虑
                # 句子翻译前先查表
                starttime = datetime.datetime.now()
                translations = (
                    Translation.objects.filter(article_id=article_id)
                    .filter(para_id=para)
                    .filter(sentence_id=sentence_id)
                )
                if translations:
                    sentence_zh = translations.first().txt
                    endtime = datetime.datetime.now()
                    logger.info("该翻译已经缓存，读取时间为%sms" % round((endtime - starttime).microseconds / 1000 / 1000, 3))
                else:
                    response = translate(sentence)
                    print(response)
                    if response["status"] == 500:
                        return HttpResponse("翻译句子:%s 时出现错误" % sentence)
                    sentence_zh = response["zh"]
                    Translation.objects.create(
                        txt=sentence_zh,
                        article_id=article_id,
                        para_id=para,
                        sentence_id=sentence_id,
                    )
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
                sentence_id = sentence_id + 1
        para_dict[para] = words_dict
        para = para + 1
    # 创建一次实验
    experiment = Experiment.objects.create(article_id=article_id, user=request.session.get("username"))
    request.session["experiment_id"] = experiment.id
    logger.info("--本次实验开始,实验者：%s，实验id：%d--" % (request.session.get("username"), experiment.id))
    return JsonResponse(para_dict, json_dumps_params={"ensure_ascii": False})


def get_gaze_data_pic(request):
    image_base64 = request.POST.get("image")  # base64类型
    x = request.POST.get("x")  # str类型
    y = request.POST.get("y")  # str类型
    t = request.POST.get("t")  # str类型
    pagedata = PageData.objects.create(
        gaze_x=str(x),
        gaze_y=str(y),
        gaze_t=str(t),
        texts="",  # todo 前端发送过来
        interventions="",
        image=image_base64,
        page=0,  # todo 前端发送过来
        experiment_id=0,
        location="",
        is_test=0,
    )
    logger.info("pagedata:%d 已保存" % pagedata.id)
    return HttpResponse(1)


def get_gaze_data_pic(request):
    image_base64 = request.POST.get("image")  # base64类型
    x = request.POST.get("x")  # str类型
    y = request.POST.get("y")  # str类型
    t = request.POST.get("t")  # str类型
    pagedata = PageData.objects.create(
        gaze_x=str(x),
        gaze_y=str(y),
        gaze_t=str(t),
        texts="",  # todo 前端发送过来
        interventions="",
        image=image_base64,
        page=0,  # todo 前端发送过来
        experiment_id=0,
        location="",
        is_test=0,
    )
    logger.info("pagedata:%d 已保存" % pagedata.id)
    return HttpResponse(1)


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
        pagedata = PageData.objects.create(
            gaze_x=str(x),
            gaze_y=str(y),
            gaze_t=str(t),
            texts=texts,  # todo 前端发送过来
            interventions=str(interventions),
            image=image_base64,
            page=page,  # todo 前端发送过来
            experiment_id=experiment_id,
            location=location,
            is_test=0,
        )
        logger.info("第%s页数据已存储,id为%s" % (page, str(pagedata.id)))
    return HttpResponse(1)


def get_labels(request):
    """一次性获得所有页的label，分页存储"""
    labels = request.POST.get("labels")
    experiment_id = request.session.get("experiment_id", None)
    # 示例：labels:[{"page":1,"wordLabels":[],"sentenceLabels":[[27,57]],"wanderLabels":[[0,27]]},{"page":2,"wordLabels":[36],"sentenceLabels":[],"wanderLabels":[]},{"page":3,"wordLabels":[],"sentenceLabels":[],"wanderLabels":[[0,34]]}]
    labels = json.loads(labels)

    paras = request.POST.get("sentence")
    paras = json.loads(paras)

    if experiment_id:
        for i, label in enumerate(labels):
            PageData.objects.filter(experiment_id=experiment_id).filter(page=label["page"]).update(
                wordLabels=label["wordLabels"],
                sentenceLabels=label["sentenceLabels"],
                wanderLabels=label["wanderLabels"],
                para=paras[i],
            )
        Experiment.objects.filter(id=experiment_id).update(is_finish=1)
    logger.info("已获得所有页标签")
    logger.info("--实验结束--")
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
    offset1, dispersion1 = get_offset_and_dispersion(gaze_1_x, gaze_1_y, gaze_1_t, target1, 1)
    print("-----------output-----------------")
    print("-----------target 1-----------------")
    print("offset1:%s" % offset1)
    print("dispersion1:%s" % dispersion1)

    offset2, dispersion2 = get_offset_and_dispersion(gaze_2_x, gaze_2_y, gaze_2_t, target2, 1)
    print("-----------target 2-----------------")
    print("offset2:%s" % offset2)
    print("dispersion2:%s" % dispersion2)

    offset3, dispersion3 = get_offset_and_dispersion(gaze_3_x, gaze_3_y, gaze_3_t, target3, 1)
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
        offset = offset + get_euclid_distance(gaze_x[i], target[0], gaze_y[i], target[1])

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


def add_word_feature_to_csv(analysis_result_by_word_level, path):
    """
    将word level的特征生成数据集
    :param analysis_result_by_word_level:
    :param path:
    :return:
    """
    import pandas as pd

    is_understand = []
    mean_fixations_duration = []
    fixation_duration = []
    second_pass_duration = []
    number_of_fixations = []
    reading_times = []
    first_reading_durations = []
    second_reading_durations = []
    third_reading_durations = []
    fourth_reading_durations = []
    for key in analysis_result_by_word_level:
        is_understand.append(analysis_result_by_word_level[key]["is_understand"])
        mean_fixations_duration.append(analysis_result_by_word_level[key]["mean_fixations_duration"])
        fixation_duration.append(analysis_result_by_word_level[key]["fixation_duration"])
        second_pass_duration.append(analysis_result_by_word_level[key]["second_pass_duration"])
        number_of_fixations.append(analysis_result_by_word_level[key]["number_of_fixations"])
        reading_times.append(analysis_result_by_word_level[key]["reading_times"])
        first_reading_durations.append(analysis_result_by_word_level[key]["first_reading_durations"])
        second_reading_durations.append(analysis_result_by_word_level[key]["second_reading_durations"])
        third_reading_durations.append(analysis_result_by_word_level[key]["third_reading_durations"])
        fourth_reading_durations.append(analysis_result_by_word_level[key]["fourth_reading_durations"])

    df = pd.DataFrame(
        {
            "is_understand": is_understand,
            "mean_fixations_duration": mean_fixations_duration,
            "fixation_duration": fixation_duration,
            "second_pass_duration": second_pass_duration,
            "number_of_fixations": number_of_fixations,
            "reading_times": reading_times,
            "first_reading_durations": first_reading_durations,
            "second_reading_durations": second_reading_durations,
            "third_reading_durations": third_reading_durations,
            "fourth_reading_durations": fourth_reading_durations,
        }
    )

    import os

    # model='a' 是追加模式
    if os.path.exists(path):
        df.to_csv(path, index=False, mode="a", header=False)
    else:
        df.to_csv(path, index=False, mode="a")


def add_sentence_feature_to_csv(analysis_result_by_sentence_level, path):
    """
    将sentence level的特征生成数据集
    :param analysis_result_by_sentence_level:
    :param path:
    :return:
    """
    import pandas as pd

    is_understand = []
    sum_dwell_time = []
    reading_times_of_sentence = []
    dwell_time = []
    second_pass_dwell_time = []
    for key in analysis_result_by_sentence_level:
        is_understand.append(analysis_result_by_sentence_level[key]["is_understand"])
        sum_dwell_time.append(analysis_result_by_sentence_level[key]["sum_dwell_time"])
        reading_times_of_sentence.append(analysis_result_by_sentence_level[key]["reading_times_of_sentence"])
        dwell_time.append(analysis_result_by_sentence_level[key]["dwell_time"])
        second_pass_dwell_time.append(analysis_result_by_sentence_level[key]["second_pass_dwell_time"])

    df = pd.DataFrame(
        {
            "is_understand": is_understand,
            "sum_dwell_time": sum_dwell_time,
            "reading_times_of_sentence": reading_times_of_sentence,
            "dwell_time": dwell_time,
            "second_pass_dwell_time": second_pass_dwell_time,
        }
    )

    import os

    # model='a' 是追加模式
    if os.path.exists(path):
        df.to_csv(path, index=False, mode="a", header=False)
    else:
        df.to_csv(path, index=False, mode="a")


def add_page_feature_to_csv(analysis_result_by_page_level, path):
    """
    将sentence level的特征生成数据集
    :param analysis_result_by_sentence_level:
    :param path:
    :return:
    """
    import pandas as pd

    page_wander = []
    saccade_times = []

    forward_saccade_times = []
    backward_saccade_times = []
    mean_saccade_length = []
    mean_saccade_angle = []
    out_of_screen_times = []
    proportion_of_horizontal_saccades = []
    number_of_fixations = []

    page_wander.append(analysis_result_by_page_level["page_wander"])
    saccade_times.append(analysis_result_by_page_level["saccade_times"])
    forward_saccade_times.append(analysis_result_by_page_level["forward_saccade_times"])
    backward_saccade_times.append(analysis_result_by_page_level["backward_saccade_times"])
    mean_saccade_length.append(analysis_result_by_page_level["mean_saccade_length"])
    mean_saccade_angle.append(analysis_result_by_page_level["mean_saccade_angle"])
    out_of_screen_times.append(analysis_result_by_page_level["out_of_screen_times"])
    proportion_of_horizontal_saccades.append(analysis_result_by_page_level["proportion of horizontal saccades"])
    number_of_fixations.append(analysis_result_by_page_level["number_of_fixations"])

    df = pd.DataFrame(
        {
            "page_wander": page_wander,
            "saccade_times": saccade_times,
            "forward_saccade_times": forward_saccade_times,
            "backward_saccade_times": backward_saccade_times,
            "mean_saccade_length": mean_saccade_length,
            "mean_saccade_angle": mean_saccade_angle,
            "out_of_screen_times": out_of_screen_times,
            "proportion_of_horizontal_saccades": proportion_of_horizontal_saccades,
            "number_of_fixations": number_of_fixations,
        }
    )

    import os

    # model='a' 是追加模式
    if os.path.exists(path):
        df.to_csv(path, index=False, mode="a", header=False)
    else:
        df.to_csv(path, index=False, mode="a")


def analysis(request):
    # 要分析的页
    experiment_id = request.GET.get("id")
    image = request.GET.get("image", False)
    csv = request.GET.get("csv", False)
    pagedatas = PageData.objects.filter(experiment_id=experiment_id)
    for pagedata in pagedatas:
        try:
            """
            准备工作：获取fixation
            """
            # 组合gaze点  [(x,y,t),(x,y,t)]
            gaze_coordinates = x_y_t_2_coordinate(pagedata.gaze_x, pagedata.gaze_y, pagedata.gaze_t)
            # 根据gaze点计算fixation list [(x,y,duration),(x,y,duration)]
            fixations = get_fixations(gaze_coordinates)
            """
                计算word level的特征
            """
            # 得到word对应的fixation dict
            word_fixation = add_fixations_to_location(fixations, pagedata.location)
            # 得到word在文章中下标 dict
            words_index, sentence_index = get_sentence_by_word_index(pagedata.texts)
            # 获取不懂的单词的label 使用", "来切割，#TODO 可能要改
            wordlabels = pagedata.wordLabels[1:-1].split(", ")
            # 获取每个单词的reading times dict  /dict.dict
            reading_times_of_word, reading_durations = get_reading_times_of_word(fixations, pagedata.location)
            # 需要输出的单词level分析结果
            analysis_result_by_word_level = {}
            # 因为所有的特征都依赖于fixation，所以遍历有fixation的word就足够了
            # TODO 那没有fixation的word怎么比较？
            for key in word_fixation:
                # 计算平均duration
                sum_t = 0
                for fixation in word_fixation[key]:
                    sum_t = sum_t + fixation[2]
                mean_fixations_duration = sum_t / len(word_fixation[key])

                result = {
                    "word": words_index[key],
                    "is_understand": 0 if str(key) in wordlabels else 1,  # 是否理解
                    "mean_fixations_duration": mean_fixations_duration,  # 平均阅读时长
                    "fixation_duration": word_fixation[key][0][2],  # 首次的fixation时长
                    "second_pass_duration": word_fixation[key][1][2]
                    if len(word_fixation[key]) > 1
                    else 0,  # second-pass duration
                    "number_of_fixations": len(word_fixation[key]),  # 在一个单词上的fixation次数
                    "fixations": word_fixation[key],  # 输出所有的fixation点
                    "reading_times": reading_times_of_word[key],
                    "first_reading_durations": reading_durations[key][1],
                    "second_reading_durations": reading_durations[key][2] if reading_times_of_word[key] > 1 else 0,
                    "third_reading_durations": reading_durations[key][3] if reading_times_of_word[key] > 2 else 0,
                    "fourth_reading_durations": reading_durations[key][4] if reading_times_of_word[key] > 3 else 0,
                }

                analysis_result_by_word_level[key] = result

            """
                获取sentence level的特征
            """
            analysis_result_by_sentence_level = {}
            # 获取不懂的句子
            json.loads(pagedata.sentenceLabels)

            # 需要输出的句子level的输出结果
            # 获取每隔句子的reading times dict/list 下标代表第几次 0-first pass
            (
                reading_times_of_sentence,
                dwell_time_of_sentence,
                number_of_word,
            ) = get_reading_times_and_dwell_time_of_sentence(fixations, pagedata.location, sentence_index)
            # for key in sentence_index:
            #     begin = sentence_index[key]["begin_word_index"]
            #     end = sentence_index[key]["end_word_index"]
            #     sum_duration = 0
            #     # 将在这个句子中所有单词的fixation duration相加
            #     for word_key in word_fixation:
            #         if begin <= word_key < end:
            #             for fixation_in_word in word_fixation[word_key]:
            #                 sum_duration = sum_duration + fixation_in_word[2]
            #     # 判断句子是否懂
            #     is_understand = 1
            #     for sentencelabel in sentencelabels:
            #         if begin == sentencelabel[0] and end == sentencelabel[1]:
            #             is_understand = 0
            #     result = {
            #         "sentence": sentence_index[key]["sentence"],  # 句子本身
            #         "is_understand": is_understand,  # 是否理解
            #         "sum_dwell_time": sum_duration / number_of_word
            #         if number_of_word != 0
            #         else 0,  # 在该句子上的fixation总时长
            #         "reading_times_of_sentence": reading_times_of_sentence[key],
            #         "dwell_time": dwell_time_of_sentence[0][key] / number_of_word
            #         if number_of_word != 0
            #         else 0,
            #         "second_pass_dwell_time": dwell_time_of_sentence[1][key] / number_of_word
            #         if number_of_word != 0
            #         else 0,
            #     }
            #     analysis_result_by_sentence_level[key] = result
            """
                计算row level的特征
            """
            # 需要输出行level的输出结果
            # row本身的信息
            row_info = get_row_location(pagedata.location)
            row_fixation = add_fixations_to_location(fixations, str(row_info).replace("'", '"'))
            analysis_result_by_row_level = {}
            wanderlabels = json.loads(pagedata.wanderLabels)
            # TODO 一旦有一行是wander,那么整页的标签就是wander
            page_wander = 0
            for key in row_fixation:
                saccade_times_of_row, mean_saccade_angle_of_row = get_saccade_info(row_fixation[key])
                # 判断该行是否走神
                begin_word = row_info[key]["begin_word"]
                end_word = row_info[key]["end_word"]
                is_wander = 0
                for wanderlabel in wanderlabels:
                    if begin_word == wanderlabel[0] and end_word == wanderlabel[1]:
                        is_wander = 1
                        page_wander = 1
                result = {
                    "is_wander": is_wander,
                    "saccade_times": saccade_times_of_row,
                    "mean_saccade_angle": mean_saccade_angle_of_row,
                }
                analysis_result_by_row_level[key] = result
            """
                计算page level的特征
            """
            # 获取page level的特征
            (
                saccade_time,
                forward_saccade_times,
                backward_saccade_times,
                mean_saccade_length,
                mean_saccade_angle,
            ) = get_saccade(fixations, pagedata.location)

            proportion_of_horizontal_saccades = get_proportion_of_horizontal_saccades(
                fixations, str(row_info).replace("'", '"'), saccade_time
            )

            analysis_result_by_page_level = {
                "page_wander": page_wander,
                "saccade_times": saccade_time,
                "forward_saccade_times": forward_saccade_times,
                "backward_saccade_times": backward_saccade_times,
                "mean_saccade_length": mean_saccade_length,
                "mean_saccade_angle": mean_saccade_angle,
                "out_of_screen_times": get_out_of_screen_times(gaze_coordinates),
                "proportion of horizontal saccades": proportion_of_horizontal_saccades,
                "number_of_fixations": len(fixations),
            }

            if image:
                # 输出图示
                print("输出图")
                fixation_image(
                    pagedata.image,
                    Experiment.objects.get(id=pagedata.experiment_id).user,
                    fixations,
                    pagedata.id,
                    "fixation.png",
                )

            # 将数据写入csv
            # 先看word level
            if csv:
                # word_path = "static/user/dataset/" + "word_level_2.csv"
                # add_word_feature_to_csv(analysis_result_by_word_level, word_path)

                sentence_path = "static/user/dataset/" + "sentence_level_lq.csv"
                add_sentence_feature_to_csv(analysis_result_by_sentence_level, sentence_path)

                # page_path = "static/user/dataset/" + "page_level_czh.csv"
                # add_page_feature_to_csv(analysis_result_by_page_level, page_path)

            # 返回结果
            analysis = {
                "word": analysis_result_by_word_level,
                "sentence": analysis_result_by_sentence_level,
                "row": analysis_result_by_row_level,
                "page": analysis_result_by_page_level,
            }
        except:
            logger.warning("page_data:%d 计算发生错误" % pagedata.id)
    # print(analysis_result_by_word_level)
    # return JsonResponse(analysis, json_dumps_params={"ensure_ascii": False})
    return HttpResponse(1)


def analysis_1(request):
    # 要分析的页
    page_data_id = request.GET.get("id")
    pagedata = PageData.objects.get(id=page_data_id)

    """
        准备工作：获取fixation
    """
    # 组合gaze点  [(x,y,t),(x,y,t)]
    gaze_coordinates = x_y_t_2_coordinate(pagedata.gaze_x, pagedata.gaze_y, pagedata.gaze_t)
    # 根据gaze点计算fixation list [(x,y,duration),(x,y,duration)]
    fixations = get_fixations(gaze_coordinates)
    """
        计算word level的特征
    """
    # 得到word对应的fixation dict
    word_fixation = add_fixations_to_location(fixations, pagedata.location)
    # 得到word在文章中下标 dict
    words_index, sentence_index = get_sentence_by_word_index(pagedata.texts)
    # 获取不懂的单词的label 使用", "来切割，#TODO 可能要改
    wordlabels = pagedata.wordLabels[1:-1].split(", ")
    # 获取每个单词的reading times dict  /dict.dict
    reading_times_of_word, reading_durations = get_reading_times_of_word(fixations, pagedata.location)
    # 需要输出的单词level分析结果
    analysis_result_by_word_level = {}
    # 因为所有的特征都依赖于fixation，所以遍历有fixation的word就足够了
    # TODO 那没有fixation的word怎么比较？
    for key in word_fixation:
        # 计算平均duration
        sum_t = 0
        for fixation in word_fixation[key]:
            sum_t = sum_t + fixation[2]
        mean_fixations_duration = sum_t / len(word_fixation[key])

        result = {
            "word": words_index[key],
            "is_understand": 0 if str(key) in wordlabels else 1,  # 是否理解
            "mean_fixations_duration": mean_fixations_duration,  # 平均阅读时长
            "fixation_duration": word_fixation[key][0][2],  # 首次的fixation时长
            "second_pass_duration": word_fixation[key][1][2]
            if len(word_fixation[key]) > 1
            else 0,  # second-pass duration
            "number_of_fixations": len(word_fixation[key]),  # 在一个单词上的fixation次数
            "fixations": word_fixation[key],  # 输出所有的fixation点
            "reading_times": reading_times_of_word[key],
            "first_reading_durations": reading_durations[key][1],
            "second_reading_durations": reading_durations[key][2] if reading_times_of_word[key] > 1 else 0,
            "third_reading_durations": reading_durations[key][3] if reading_times_of_word[key] > 2 else 0,
            "fourth_reading_durations": reading_durations[key][4] if reading_times_of_word[key] > 3 else 0,
        }

        analysis_result_by_word_level[key] = result

    """
        获取sentence level的特征
    """
    # 获取不懂的句子
    json.loads(pagedata.sentenceLabels)

    # 需要输出的句子level的输出结果
    # 获取每隔句子的reading times dict/list 下标代表第几次 0-first pass
    (
        reading_times_of_sentence,
        dwell_time_of_sentence,
        number_of_word,
    ) = get_reading_times_and_dwell_time_of_sentence(fixations, pagedata.location, sentence_index)
    # for key in sentence_index:
    #     begin = sentence_index[key]["begin_word_index"]
    #     end = sentence_index[key]["end_word_index"]
    #     sum_duration = 0
    #     # 将在这个句子中所有单词的fixation duration相加
    #     for word_key in word_fixation:
    #         if begin <= word_key < end:
    #             for fixation_in_word in word_fixation[word_key]:
    #                 sum_duration = sum_duration + fixation_in_word[2]
    #     # 判断句子是否懂
    #     is_understand = 1
    #     for sentencelabel in sentencelabels:
    #         if begin == sentencelabel[0] and end == sentencelabel[1]:
    #             is_understand = 0
    #     result = {
    #         "sentence": sentence_index[key]["sentence"],  # 句子本身
    #         "is_understand": is_understand,  # 是否理解
    #         "sum_dwell_time": sum_duration / number_of_word
    #         if number_of_word != 0
    #         else 0,  # 在该句子上的fixation总时长
    #         "reading_times_of_sentence": reading_times_of_sentence[key],
    #         "dwell_time": dwell_time_of_sentence[0][key] / number_of_word
    #         if number_of_word != 0
    #         else 0,
    #         "second_pass_dwell_time": dwell_time_of_sentence[1][key] / number_of_word
    #         if number_of_word != 0
    #         else 0,
    #     }
    #     analysis_result_by_sentence_level[key] = result
    """
        计算row level的特征
    """
    # 需要输出行level的输出结果
    # row本身的信息
    row_info = get_row_location(pagedata.location)
    row_fixation = add_fixations_to_location(fixations, str(row_info).replace("'", '"'))
    analysis_result_by_row_level = {}
    wanderlabels = json.loads(pagedata.wanderLabels)
    # TODO 一旦有一行是wander,那么整页的标签就是wander
    page_wander = 0
    for key in row_fixation:
        saccade_times_of_row, mean_saccade_angle_of_row = get_saccade_info(row_fixation[key])
        # 判断该行是否走神
        begin_word = row_info[key]["begin_word"]
        end_word = row_info[key]["end_word"]
        is_wander = 0
        for wanderlabel in wanderlabels:
            if begin_word == wanderlabel[0] and end_word == wanderlabel[1]:
                is_wander = 1
                page_wander = 1
        result = {
            "is_wander": is_wander,
            "saccade_times": saccade_times_of_row,
            "mean_saccade_angle": mean_saccade_angle_of_row,
        }
        analysis_result_by_row_level[key] = result
    """
        计算page level的特征
    """
    # 获取page level的特征
    (
        saccade_time,
        forward_saccade_times,
        backward_saccade_times,
        mean_saccade_length,
        mean_saccade_angle,
    ) = get_saccade(fixations, pagedata.location)

    proportion_of_horizontal_saccades = get_proportion_of_horizontal_saccades(
        fixations, str(row_info).replace("'", '"'), saccade_time
    )

    analysis_result_by_page_level = {
        "page_wander": page_wander,
        "saccade_times": saccade_time,
        "forward_saccade_times": forward_saccade_times,
        "backward_saccade_times": backward_saccade_times,
        "mean_saccade_length": mean_saccade_length,
        "mean_saccade_angle": mean_saccade_angle,
        "out_of_screen_times": get_out_of_screen_times(gaze_coordinates),
        "proportion of horizontal saccades": proportion_of_horizontal_saccades,
        "number_of_fixations": len(fixations),
    }

    # 返回结果
    analysis = {
        "word": analysis_result_by_word_level,
        # "sentence": analysis_result_by_sentence_level,
        # "row": analysis_result_by_row_level,
        # "page": analysis_result_by_page_level,
    }

    return JsonResponse(analysis, json_dumps_params={"ensure_ascii": False})


def test_motion(request):
    return render(request, "test_motion.html")


def takeSecond(elem):
    return elem[1]


def paint_on_word(image, target_words_index, word_locations, title, pic_path, alpha=0.4, color=255):
    blk = np.zeros(image.shape, np.uint8)
    set_title(blk, title)
    for word_index in target_words_index:
        loc = word_locations[word_index]
        cv2.rectangle(
            blk,
            (int(loc[0]), int(loc[1])),
            (int(loc[2]), int(loc[3])),
            (color, 0, 0),
            -1,
        )
    image = cv2.addWeighted(blk, alpha, image, 1 - alpha, 0)
    plt.imshow(image)
    plt.title(title)
    plt.show()
    cv2.imwrite(pic_path, image)
    logger.info("heatmap已经生成:%s" % pic_path)


def get_sentence_meaning(request):
    page_data_id = request.GET.get("id")
    page_data = PageData.objects.get(id=page_data_id)
    dict = {}
    texts = ""
    page_data_ls = PageData.objects.filter(experiment_id=page_data.experiment_id)
    for data in page_data_ls:
        texts += data.texts
    print(texts)
    attention = generate_sentence_attention(texts)
    diff = generate_sentence_difficulty(texts)
    dict[page_data_id] = {"attention": attention, "difficulty": diff}

    filename = str(id) + "-sen.txt"
    with open(filename, "w") as f:
        for key, value in dict.items():
            f.write(key)
            f.write(": ")
            f.write(str(value))
            f.write("\n")
    return HttpResponse(1)


def get_visual_heatmap(request):
    """
    仅生成 fixation、visual attention、二者组合、分行的fixation
    :param request:
    :return:
    """
    top_dict = {}

    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)
    exp = Experiment.objects.filter(id=pageData.experiment_id)
    # 准备工作，获取单词和句子的信息
    word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
    # 获取单词的位置
    word_locations = get_word_location(pageData.location)  # [(left,top,right,bottom),(left,top,right,bottom)]

    # 确保单词长度是正确的
    assert len(word_locations) == len(word_list)

    # 获取图片生成的路径
    exp = Experiment.objects.filter(id=pageData.experiment_id)

    base_path = "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\"

    # 创建图片存储的目录
    # 如果目录不存在，则创建目录
    path_levels = [
        "static\\data\\heatmap\\" + str(exp.first().user) + "\\",
        "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\",
    ]
    for path in path_levels:
        print("生成:%s" % path)
        if not os.path.exists(path):
            os.mkdir(path)

    # 创建背景图片
    background = base_path + "background.png"
    # 使用数据库的base64直接生成background
    data = pageData.image.split(",")[1]
    # 将str解码为byte
    image_data = base64.b64decode(data)
    with open(background, "wb") as f:
        f.write(image_data)

    kernel_size = int(request.GET.get("window", 0))

    get_visual_attention(
        page_data_id,
        exp.first().user,
        pageData.gaze_x,
        pageData.gaze_y,
        pageData.gaze_t,
        kernel_size,
        background,
        base_path,
        word_list,
        word_locations,
        top_dict,
        pageData.image,
    )

    # 将spatial和temporal结合
    save_path = base_path + "spatial_with_temporal_filter.png"
    heatmap_img = base_path + "visual.png"
    fixation_img = base_path + "fixation.png"
    join_two_image(heatmap_img, fixation_img, save_path)

    return HttpResponse(1)


def get_row_level_fixations_map(request):
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)
    exp = Experiment.objects.filter(id=pageData.experiment_id)
    kernel_size = int(request.GET.get("window", 0))

    gaze_x = pageData.gaze_x
    gaze_y = pageData.gaze_y
    gaze_t = pageData.gaze_t
    """
    1.
    清洗数据
    """
    list_x = list(map(float, gaze_x.split(",")))
    list_y = list(map(float, gaze_y.split(",")))
    list_t = list(map(float, gaze_t.split(",")))

    print("length of list")
    print(len(list_x))
    print("kernel_size:%d" % kernel_size)
    if kernel_size != 0:
        # 滤波
        list_x = preprocess_data(list_x, kernel_size)
        list_y = preprocess_data(list_y, kernel_size)

    print("length of list after filter")
    print(len(list_x))
    # 滤波完是浮点数，需要转成int
    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))

    # 组合
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i], list_t[i]]
        coordinates.append(coordinate)
    # 去除开始结束的gaze点
    coordinates = coordinates[2:-1]
    # 去除前后200ms的gaze点
    begin = 0
    end = -1
    for i, coordinate in enumerate(coordinates):
        if coordinate[2] - coordinates[0][2] > 500:
            begin = i
            break
    print("begin:%d" % begin)
    for i in range(len(coordinates) - 1, -1, -1):
        if coordinates[-1][2] - coordinates[i][2] > 500:
            end = i
            break
    coordinates = coordinates[begin:end]

    fixations = get_fixations(coordinates)
    # 按行切割gaze点
    pre_fixation = fixations[0]
    distance = 600
    row_fixations = []
    tmp = []

    row_cnt = 1
    for fixation in fixations:
        if get_euclid_distance(fixation[0], pre_fixation[0], fixation[1], pre_fixation[1]) > distance:
            row_cnt += 1
            # if row_cnt == 19:
            #     tmp.append(fixation)
            # else:
            row_fixations.append(tmp)
            tmp = [fixation]
        else:
            tmp.append(fixation)
        pre_fixation = fixation

    base_path = "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\"

    if not os.path.exists(base_path + "fixation\\"):
        os.mkdir(base_path + "fixation\\")
    for i, item in enumerate(row_fixations):
        name = "1"
        for j in range(i):
            name += "1"
        print(item)
        fixation_image(
            pageData.image,
            exp.first().user,
            item,
            page_data_id,
            "/fixation/" + name + ".png",
        )

    import glob

    print(base_path)
    img_list = glob.glob(base_path + "fixation\\" + "*.png")
    img_list.sort()
    print(img_list)
    join_images_vertical(img_list, base_path + "row_fix.png")

    shutil.rmtree(base_path + "fixation\\")
    return HttpResponse(1)


def get_row_level_heatmap(request):
    id = request.GET.get("id")
    pageData = PageData.objects.get(id=id)
    exp = Experiment.objects.get(id=pageData.experiment_id)

    kernel_size = int(request.GET.get("window", 7))

    gaze_x = pageData.gaze_x
    gaze_y = pageData.gaze_y
    gaze_t = pageData.gaze_t
    """
    1.
    清洗数据
    """
    list_x = list(map(float, gaze_x.split(",")))
    list_y = list(map(float, gaze_y.split(",")))
    list_t = list(map(float, gaze_t.split(",")))

    print("length of list")
    print(len(list_x))
    print("kernel_size:%d" % kernel_size)
    if kernel_size != 0:
        # 滤波
        list_x = preprocess_data(list_x, kernel_size)
        list_y = preprocess_data(list_y, kernel_size)

    print("length of list after filter")
    print(len(list_x))
    # 滤波完是浮点数，需要转成int
    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))

    # 组合
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i], list_t[i]]
        coordinates.append(coordinate)
    # 去除开始结束的gaze点
    coordinates = coordinates[2:-1]
    # 去除前后200ms的gaze点
    begin = 0
    end = -1
    for i, coordinate in enumerate(coordinates):
        if coordinate[2] - coordinates[0][2] > 500:
            begin = i
            break
    print("begin:%d" % begin)
    for i in range(len(coordinates) - 1, -1, -1):
        if coordinates[-1][2] - coordinates[i][2] > 500:
            end = i
            break
    coordinates = coordinates[begin:end]

    fixations = get_fixations(coordinates)
    # 按行切割gaze点
    pre_fixation = fixations[0]
    distance = 600
    row_fixations = []
    tmp = []
    sum_y = 0
    for fixation in fixations:
        if get_euclid_distance(fixation[0], pre_fixation[0], fixation[1], pre_fixation[1]) > distance:
            mean_y = int(sum_y / len(tmp))
            for item in tmp:
                if item[1] > mean_y + 20:
                    item[1] = mean_y + 20
                if item[1] < mean_y - 20:
                    item[1] = mean_y - 20
            row_fixations.append(tmp)
            tmp = [fixation]
            sum_y = 0
        else:
            tmp.append(fixation)
            sum_y += fixation[1]
        pre_fixation = fixation

    base_path = "static\\data\\heatmap\\" + str(exp.user) + "\\" + str(pageData.id) + "\\"

    if not os.path.exists(base_path + "heat\\"):
        os.mkdir(base_path + "heat\\")
    for i, item in enumerate(row_fixations):
        name = "1"
        for j in range(i):
            name += "1"
        print(item)
        myHeatmap.draw_heat_map(item, base_path + "heat\\" + name + ".png", base_path + "background.png")

    import glob

    print(base_path)
    img_list = glob.glob(base_path + "heat\\" + "*.png")
    img_list.sort()
    print(img_list)
    join_images_vertical(img_list, base_path + "row_heat.png")

    image = cv2.imread(base_path + "row_heat.png")
    cv2.putText(
        image,
        str(pageData.id),  # text内容必须是str格式的
        (700, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    print(pageData.id)
    cv2.imwrite("static\\data\\heatmap\\row_level_heat\\" + str(pageData.id) + ".png", image)
    shutil.rmtree(base_path + "heat\\")

    return HttpResponse(1)


def get_row_level_heatmap_of_all_user(request):
    exps = Experiment.objects.filter(is_finish=True).order_by("-id")[0:51]
    success = 0
    fail = 0
    for exp in exps:
        pageData_list = PageData.objects.filter(experiment_id=exp.id)
        try:
            for pageData in pageData_list:
                kernel_size = int(request.GET.get("window", 7))

                gaze_x = pageData.gaze_x
                gaze_y = pageData.gaze_y
                gaze_t = pageData.gaze_t
                """
                1.
                清洗数据
                """
                list_x = list(map(float, gaze_x.split(",")))
                list_y = list(map(float, gaze_y.split(",")))
                list_t = list(map(float, gaze_t.split(",")))

                print("length of list")
                print(len(list_x))
                print("kernel_size:%d" % kernel_size)
                if kernel_size != 0:
                    # 滤波
                    list_x = preprocess_data(list_x, kernel_size)
                    list_y = preprocess_data(list_y, kernel_size)

                print("length of list after filter")
                print(len(list_x))
                # 滤波完是浮点数，需要转成int
                list_x = list(map(int, list_x))
                list_y = list(map(int, list_y))

                # 组合
                coordinates = []
                for i in range(len(list_x)):
                    coordinate = [list_x[i], list_y[i], list_t[i]]
                    coordinates.append(coordinate)
                # 去除开始结束的gaze点
                coordinates = coordinates[2:-1]
                # 去除前后200ms的gaze点
                begin = 0
                end = -1
                for i, coordinate in enumerate(coordinates):
                    if coordinate[2] - coordinates[0][2] > 500:
                        begin = i
                        break
                print("begin:%d" % begin)
                for i in range(len(coordinates) - 1, -1, -1):
                    if coordinates[-1][2] - coordinates[i][2] > 500:
                        end = i
                        break
                coordinates = coordinates[begin:end]

                fixations = get_fixations(coordinates)
                # 按行切割gaze点
                pre_fixation = fixations[0]
                distance = 600
                row_fixations = []
                tmp = []
                sum_y = 0
                for fixation in fixations:
                    if get_euclid_distance(fixation[0], pre_fixation[0], fixation[1], pre_fixation[1]) > distance:
                        mean_y = int(sum_y / len(tmp))
                        for item in tmp:
                            if item[1] > mean_y + 20:
                                item[1] = mean_y + 20
                            if item[1] < mean_y - 20:
                                item[1] = mean_y - 20
                        row_fixations.append(tmp)
                        tmp = [fixation]
                        sum_y = 0
                    else:
                        tmp.append(fixation)
                        sum_y += fixation[1]
                    pre_fixation = fixation

                base_path = "static\\data\\heatmap\\" + str(exp.user) + "\\" + str(pageData.id) + "\\"

                if not os.path.exists(base_path + "heat\\"):
                    os.mkdir(base_path + "heat\\")
                for i, item in enumerate(row_fixations):
                    name = "1"
                    for j in range(i):
                        name += "1"
                    print(item)
                    myHeatmap.draw_heat_map(item, base_path + "heat\\" + name + ".png", base_path + "background.png")

                import glob

                print(base_path)
                img_list = glob.glob(base_path + "heat\\" + "*.png")
                img_list.sort()
                print(img_list)
                join_images_vertical(img_list, base_path + "row_heat.png")

                image = cv2.imread(base_path + "row_heat.png")
                cv2.putText(
                    image,
                    str(pageData.id),  # text内容必须是str格式的
                    (700, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite("static\\data\\heatmap\\row_level_heat\\" + str(pageData.id) + ".png", image)
                shutil.rmtree(base_path + "heat\\")
            success += 1
            logger.info("成功生成 %d条,失败 %d条" % (success, fail))
        except:
            fail += 1
            logger.info("成功生成 %d条,失败 %d条" % (success, fail))
    return HttpResponse(1)


def get_row_level_fixations(page_data_id, kernel_size):
    pageData = PageData.objects.get(id=page_data_id)
    exp = Experiment.objects.filter(id=pageData.experiment_id)

    gaze_x = pageData.gaze_x
    gaze_y = pageData.gaze_y
    gaze_t = pageData.gaze_t
    """
    1.
    清洗数据
    """
    list_x = list(map(float, gaze_x.split(",")))
    list_y = list(map(float, gaze_y.split(",")))
    list_t = list(map(float, gaze_t.split(",")))

    print("length of list")
    print(len(list_x))
    print("kernel_size:%d" % kernel_size)
    if kernel_size != 0:
        # 滤波
        list_x = preprocess_data(list_x, kernel_size)
        list_y = preprocess_data(list_y, kernel_size)

    print("length of list after filter")
    print(len(list_x))
    # 滤波完是浮点数，需要转成int
    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))

    # 组合
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i], list_t[i]]
        coordinates.append(coordinate)
    # 去除开始结束的gaze点
    coordinates = coordinates[2:-1]
    # 去除前后200ms的gaze点
    begin = 0
    end = -1
    for i, coordinate in enumerate(coordinates):
        if coordinate[2] - coordinates[0][2] > 500:
            begin = i
            break
    print("begin:%d" % begin)
    for i in range(len(coordinates) - 1, -1, -1):
        if coordinates[-1][2] - coordinates[i][2] > 500:
            end = i
            break
    coordinates = coordinates[begin:end]

    fixations = get_fixations(coordinates)
    # 按行切割gaze点
    pre_fixation = fixations[0]
    distance = 600
    row_fixations = []
    tmp = []
    for fixation in fixations:
        if get_euclid_distance(fixation[0], pre_fixation[0], fixation[1], pre_fixation[1]) > distance:
            row_fixations.append(tmp)
            tmp = [fixation]
        else:
            tmp.append(fixation)
        pre_fixation = fixation

    base_path = "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\"

    if not os.path.exists(base_path + "fixation\\"):
        os.mkdir(base_path + "fixation\\")
    for i, item in enumerate(row_fixations):
        name = "1"
        for j in range(i):
            name += "1"
        print(item)
        fixation_image(
            pageData.image,
            exp.first().user,
            item,
            page_data_id,
            "/fixation/" + name + ".png",
        )

    import glob

    print(base_path)
    img_list = glob.glob(base_path + "fixation\\" + "*.png")
    img_list.sort()
    print(img_list)
    join_images_vertical(img_list, base_path + "row_fix.png")

    shutil.rmtree(base_path + "fixation\\")
    return HttpResponse(1)


def get_all_heatmap(request):
    """
    根据page_data_id生成visual attention和nlp attention
    :param request: id,window
    :return:
    """
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)
    # 准备工作，获取单词和句子的信息
    word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
    # 获取单词的位置
    word_locations = get_word_location(pageData.location)  # [(left,top,right,bottom),(left,top,right,bottom)]

    # 确保单词长度是正确的
    assert len(word_locations) == len(word_list)
    # 获取图片生成的路径
    exp = Experiment.objects.filter(id=pageData.experiment_id)

    base_path = "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\"

    # 创建图片存储的目录
    # 如果目录不存在，则创建目录
    path_levels = [
        "static\\data\\heatmap\\" + str(exp.first().user) + "\\",
        "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\",
    ]
    for path in path_levels:
        print("生成:%s" % path)
        if not os.path.exists(path):
            os.mkdir(path)

    # 创建背景图片
    background = base_path + "background.png"
    # 使用数据库的base64直接生成background
    data = pageData.image.split(",")[1]
    # 将str解码为byte
    image_data = base64.b64decode(data)
    with open(background, "wb") as f:
        f.write(image_data)

    top_dict = {}  # 实际上是word的top dict
    # nlp attention

    """
    word level
    1. 生成3张图片
    2. 填充top_k
    """
    get_word_level_nlp_attention(
        pageData.texts,
        page_data_id,
        top_dict,
        background,
        word_list,
        word_locations,
        exp.first().user,
        base_path,
    )
    """
    sentence level
    """

    top_sentence_dict = {}
    # 生成图片
    get_sentence_level_nlp_attention(
        pageData.texts,
        sentence_list,
        background,
        word_locations,
        page_data_id,
        top_sentence_dict,
        exp.first().user,
        base_path,
    )

    """
    visual attention
    1. 生成2张图片：heatmap + fixation
    2. 填充top_k
    存在问题：生成top和展示图片实际上做了两个heatmap
    """
    # 滤波处理
    kernel_size = int(request.GET.get("window", 0))
    get_visual_attention(
        page_data_id,
        exp.first().user,
        pageData.gaze_x,
        pageData.gaze_y,
        pageData.gaze_t,
        kernel_size,
        background,
        base_path,
        word_list,
        word_locations,
        top_dict,
        pageData.image,
    )

    # 将spatial和temporal结合
    save_path = base_path + "spatial_with_temporal_filter.png"
    heatmap_img = base_path + "visual.png"
    fixation_img = base_path + "fixation.png"
    join_two_image(heatmap_img, fixation_img, save_path)

    """
    将单词不懂和句子不懂输出,走神的图示输出
    """
    image = cv2.imread(background)
    # 走神与否
    words_to_be_painted = []
    if pageData.wanderLabels:
        paras_wander = json.loads(pageData.wanderLabels)
    else:
        paras_wander = []

    for para in paras_wander:
        for i in range(para[0], para[1] + 1):  # wander label是到一段结尾，不是到下一段
            words_to_be_painted.append(i)

    title = str(page_data_id) + "-" + exp.first().user + "-" + "para_wander"
    pic_path = base_path + "para_wander" + ".png"
    # 画图
    paint_on_word(image, words_to_be_painted, word_locations, title, pic_path)

    # 单词 TODO 将这些整理为函数，复用
    # 找需要画的单词
    if pageData.wordLabels:
        words_not_understand = json.loads(pageData.wordLabels)
    else:
        words_not_understand = []
    title = str(page_data_id) + "-" + exp.first().user + "-" + "words_not_understand"
    pic_path = base_path + "words_not_understand" + ".png"
    # 画图
    paint_on_word(image, words_not_understand, word_locations, title, pic_path)

    # 句子
    if pageData.sentenceLabels:
        sentences_not_understand = json.loads(pageData.sentenceLabels)
    else:
        sentences_not_understand = []

    words_to_painted = []
    for sentence in sentences_not_understand:
        for i in range(sentence[0], sentence[1]):
            # 此处i代表的是单词
            words_to_painted.append(i)

    title = str(page_data_id) + "-" + exp.first().user + "-" + "sentences_not_understand"
    pic_path = base_path + "sentences_not_understand" + ".png"
    # 画图
    paint_on_word(image, words_to_painted, word_locations, title, pic_path)

    """
    不同k值下similarity和identity的计算以及图示
    """
    k_list = [3, 5, 7, 10, 20]

    k_dict = {}
    pic_list = []
    max_k = len(top_dict["visual"])
    for k in k_list:
        if k > max_k:
            break
        attention_type = {}
        for key in top_dict.keys():
            attention_type[key] = top_dict[key][0:k]
        k_dict[k] = attention_type

        pic_dict = {
            "k": k,
            "similarity": calculate_similarity(attention_type),
            "identity": calculate_identity(attention_type),
        }
        pic_list.append(pic_dict)

    paint_bar_graph(pic_list, base_path, "similarity")
    paint_bar_graph(pic_list, base_path, "identity")

    result_dict = {"word level": k_dict, "sentence level": top_sentence_dict}

    result_json = json.dumps(result_dict)
    path = base_path + "result.txt"
    with open(path, "w") as json_file:
        json_file.write(result_json)

    return JsonResponse(result_dict, json_dumps_params={"ensure_ascii": False}, safe=False)


def get_all_heatmap_of_all_user(request):
    """
    根据page_data_id生成visual attention和nlp attention
    :param request: id,window
    :return:
    """
    success = 0
    fail = 0
    exps = Experiment.objects.filter(is_finish=True).order_by("-id")[16:51]
    for exp in exps:
        pageDatas = PageData.objects.filter(experiment_id=exp.id)
        for pageData in pageDatas:
            try:
                # 准备工作，获取单词和句子的信息
                word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
                # 获取单词的位置
                word_locations = get_word_location(
                    pageData.location
                )  # [(left,top,right,bottom),(left,top,right,bottom)]

                # 确保单词长度是正确的
                assert len(word_locations) == len(word_list)
                # 获取图片生成的路径
                exp = Experiment.objects.filter(id=pageData.experiment_id)

                base_path = "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(pageData.id) + "\\"

                # 创建图片存储的目录
                # 如果目录不存在，则创建目录
                path_levels = [
                    "static\\data\\heatmap\\" + str(exp.first().user) + "\\",
                    "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(pageData.id) + "\\",
                ]
                for path in path_levels:
                    print("生成:%s" % path)
                    if not os.path.exists(path):
                        os.mkdir(path)

                # 创建背景图片
                background = base_path + "background.png"
                # 使用数据库的base64直接生成background
                data = pageData.image.split(",")[1]
                # 将str解码为byte
                image_data = base64.b64decode(data)
                with open(background, "wb") as f:
                    f.write(image_data)

                top_dict = {}  # 实际上是word的top dict
                # nlp attention

                """
                word level
                1. 生成3张图片
                2. 填充top_k
                """
                get_word_level_nlp_attention(
                    pageData.texts,
                    pageData.id,
                    top_dict,
                    background,
                    word_list,
                    word_locations,
                    exp.first().user,
                    base_path,
                )
                """
                sentence level
                """

                top_sentence_dict = {}
                # 生成图片
                get_sentence_level_nlp_attention(
                    pageData.texts,
                    sentence_list,
                    background,
                    word_locations,
                    pageData.id,
                    top_sentence_dict,
                    exp.first().user,
                    base_path,
                )

                """
                visual attention
                1. 生成2张图片：heatmap + fixation
                2. 填充top_k
                存在问题：生成top和展示图片实际上做了两个heatmap
                """
                # 滤波处理
                kernel_size = int(request.GET.get("window", 7))
                get_visual_attention(
                    pageData.id,
                    exp.first().user,
                    pageData.gaze_x,
                    pageData.gaze_y,
                    pageData.gaze_t,
                    kernel_size,
                    background,
                    base_path,
                    word_list,
                    word_locations,
                    top_dict,
                    pageData.image,
                )

                # 将spatial和temporal结合
                save_path = base_path + "spatial_with_temporal_filter.png"
                heatmap_img = base_path + "visual.png"
                fixation_img = base_path + "fixation.png"
                join_two_image(heatmap_img, fixation_img, save_path)

                """
                将单词不懂和句子不懂输出,走神的图示输出
                """
                image = cv2.imread(background)
                # 走神与否
                words_to_be_painted = []
                if pageData.wanderLabels:
                    paras_wander = json.loads(pageData.wanderLabels)
                else:
                    paras_wander = []

                for para in paras_wander:
                    for i in range(para[0], para[1] + 1):  # wander label是到一段结尾，不是到下一段
                        words_to_be_painted.append(i)

                title = str(pageData.id) + "-" + exp.first().user + "-" + "para_wander"
                pic_path = base_path + "para_wander" + ".png"
                # 画图
                paint_on_word(image, words_to_be_painted, word_locations, title, pic_path)

                # 单词 TODO 将这些整理为函数，复用
                # 找需要画的单词
                if pageData.wordLabels:
                    words_not_understand = json.loads(pageData.wordLabels)
                else:
                    words_not_understand = []
                title = str(pageData.id) + "-" + exp.first().user + "-" + "words_not_understand"
                pic_path = base_path + "words_not_understand" + ".png"
                # 画图
                paint_on_word(image, words_not_understand, word_locations, title, pic_path)

                # 句子
                if pageData.sentenceLabels:
                    sentences_not_understand = json.loads(pageData.sentenceLabels)
                else:
                    sentences_not_understand = []

                words_to_painted = []
                for sentence in sentences_not_understand:
                    for i in range(sentence[0], sentence[1]):
                        # 此处i代表的是单词
                        words_to_painted.append(i)

                title = str(pageData.id) + "-" + exp.first().user + "-" + "sentences_not_understand"
                pic_path = base_path + "sentences_not_understand" + ".png"
                # 画图
                paint_on_word(image, words_to_painted, word_locations, title, pic_path)

                """
                不同k值下similarity和identity的计算以及图示
                """
                k_list = [3, 5, 7, 10, 20]

                k_dict = {}
                pic_list = []
                max_k = len(top_dict["visual"])
                for k in k_list:
                    if k > max_k:
                        break
                    attention_type = {}
                    for key in top_dict.keys():
                        attention_type[key] = top_dict[key][0:k]
                    k_dict[k] = attention_type

                    pic_dict = {
                        "k": k,
                        "similarity": calculate_similarity(attention_type),
                        "identity": calculate_identity(attention_type),
                    }
                    pic_list.append(pic_dict)

                paint_bar_graph(pic_list, base_path, "similarity")
                paint_bar_graph(pic_list, base_path, "identity")

                result_dict = {"word level": k_dict, "sentence level": top_sentence_dict}

                result_json = json.dumps(result_dict)
                path = base_path + "result.txt"
                with open(path, "w") as json_file:
                    json_file.write(result_json)

                get_row_level_fixations(pageData.id, 7)
                success += 1
            except:
                fail += 1
            logger.info("成功生成 %d 条，失败%d,当前id是%d" % (success, fail, pageData.id))
    return HttpResponse(1)


def set_title(blk, title):
    """设置图片标题"""
    cv2.putText(
        blk,
        str(title),  # text内容必须是str格式的
        (600, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (189, 252, 201),
        2,
    )


def get_semantic_value_of_word(request):
    ids = request.GET.get("ids")
    ids = ids.split(",")
    dict = {}
    # sentence = "The BBC's Suranjana Tewari in Singapore says Asian investors are also dumping sterling and other currencies because the US Federal Reserve has hiked interest rates so quickly to fight inflation."
    # sentence = "This policy sets the stage for the Department of Commerce to take over the management of traffic in space."
    # sentence = "One impetus for the policy is that companies are already starting to build massive constellations, comprising hundreds or thousands of satellites with many moving parts among them"
    # sentence = "The personal grievance provisions of New Zealand’s Employment Relations Act 2000 (ERA) prevent an employer from firing an employee without good cause."
    sentence = "A local chief was also among the victims of the violence on Saturday in Turkana County."
    # 计算某一句的word attention
    doc = nlp(sentence)

    words = []
    for token in doc:
        words.append(token.text.strip().lower())

    for id in ids:
        dict[id] = {}
        texts = PageData.objects.filter(id=id).first().texts
        topics = get_importance(texts)

        attention = generate_word_attention(texts)
        diff = generate_word_difficulty(texts)

        topic_ls = []
        at_ls = []
        diff_ls = []
        for word in words:
            flag = True
            for topic in topics:
                if topic[0].lower() == word:
                    topic_ls.append(topic)
                    flag = False
                    break
            if flag:
                tmp = (word, 0)
                topic_ls.append(tmp)

            for at in attention:
                if at[0].lower() == word:
                    tmp = (at[0].lower(), at[1])
                    at_ls.append(tmp)
                    break

            for di in diff:
                if di[0].lower() == word:
                    tmp = (di[0].lower(), di[1])
                    diff_ls.append(tmp)
                    break
            print(diff_ls)
        dict[id] = {"topic": topic_ls, "attention": at_ls, "difficulty": diff_ls}

        filename = str(id) + ".txt"
        with open(filename, "w") as f:
            for key, value in dict.items():
                f.write(key)
                f.write(": ")
                f.write(str(value))
                f.write("\n")

    return HttpResponse(1)


def get_word_level_nlp_attention(
    texts,
    page_data_id,
    top_dict,
    background,
    word_list,
    word_locations,
    username,
    base_path,
):
    logger.info("word level attention正在分析....")
    nlp_attentions = ["topic_relevant", "word_attention", "word_difficulty"]
    for attention in nlp_attentions:
        data_list = [[]]
        # 获取数据
        if attention == "topic_relevant":
            data_list = get_importance(texts)  # [('word',1)]
            print(data_list)
        if attention == "word_attention":
            data_list = generate_word_attention(texts)
        if attention == "word_difficulty":
            data_list = generate_word_difficulty(texts)

        data_list.sort(reverse=True, key=takeSecond)

        print(data_list)
        top_dict[attention] = [x[0] for x in data_list]

        image = cv2.imread(background)
        color = 255
        alpha = 0.4  # 设置覆盖图片的透明度
        blk = np.zeros(image.shape, np.uint8)
        title = str(page_data_id) + "-" + str(username) + "-" + str(attention)
        # 设置标题
        set_title(blk, title)
        for data in data_list:
            # 该单词在页面上是否存在
            index_list = []
            for i, word in enumerate(word_list):
                if data[0].lower() == word.lower():
                    index_list.append(i)
            for index in index_list:
                loc = word_locations[index]
                cv2.rectangle(
                    blk,
                    (int(loc[0]), int(loc[1])),
                    (int(loc[2]), int(loc[3])),
                    (color, 0, 0),
                    -1,
                )
            color = color - 15
            if color - 5 < 50:
                break
        image = cv2.addWeighted(blk, alpha, image, 1 - alpha, 0)

        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.title(title)
        # plt.show()
        heatmap_name = base_path + str(attention) + ".png"
        cv2.imwrite(heatmap_name, image)
        logger.info("heatmap已经生成:%s" % heatmap_name)


def get_sentence_level_nlp_attention(
    texts,
    sentence_list,
    background,
    word_locations,
    page_data_id,
    top_sentence_dict,
    username,
    base_path,
):
    logger.info("sentence level attention正在分析....")
    sentence_attentions = ["sentence_attention", "sentence_difficulty"]

    for attention in sentence_attentions:
        sentence_attention = [[]]
        # TODO 切割句子需要 '. ' 可能之后出现问题
        if attention == "sentence_attention":
            sentence_attention = generate_sentence_attention(texts.replace("..", ". "))  # [('xx',数值),('xx',数值)]
        if attention == "sentence_difficulty":
            sentence_attention = generate_sentence_difficulty(texts.replace("..", ". "))  # [('xx',数值),('xx',数值)]
        # 确保句子长度是正确的
        # sentence有的拆的不对 3是个magic number，认为不会有长度在3以下的句子
        sentence_attention = [item for item in sentence_attention if len(item[0]) > 3]
        print(sentence_attention)
        print(sentence_list)
        assert len(sentence_attention) == len(sentence_list)
        # 获得句子的index
        index_list_by_weight = []  # 按照index排序
        for i in range(len(sentence_attention)):
            max_weight = -1
            max_index = -1
            for j, sentence in enumerate(sentence_attention):
                if sentence[1] > max_weight and j not in index_list_by_weight:
                    max_weight = sentence[1]
                    max_index = j
            index_list_by_weight.append(max_index)
        image = cv2.imread(background)
        color = 255
        alpha = 0.4  # 设置覆盖图片的透明度
        blk = np.zeros(image.shape, np.uint8)
        title = str(page_data_id) + "-" + str(username) + "-" + attention
        set_title(blk, title)
        for j, index in enumerate(index_list_by_weight):
            sentence = sentence_list[index]
            q = 0  # 标志句首
            for i in range(sentence[1], sentence[2]):
                # 此处i代表的是单词
                loc = word_locations[i]
                cv2.rectangle(
                    blk,
                    (int(loc[0]), int(loc[1])),
                    (int(loc[2]), int(loc[3])),
                    (color, 0, 0),
                    -1,
                )
                # 标序号，便于分析，只在句首标
                if q == 0:
                    cv2.putText(
                        blk,
                        str(j),  # text内容必须是str格式的
                        (int(loc[0]), int(loc[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                q = q + 1
            color = color - 30
            if color - 5 < 50:
                break
        image = cv2.addWeighted(blk, alpha, image, 1 - alpha, 0)
        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.title(title)
        # plt.show()
        heatmap_name = base_path + attention + ".png"
        cv2.imwrite(heatmap_name, image)
        logger.info("heatmap已经生成:%s" % heatmap_name)

        # 生成top_k
        sentence_attention.sort(reverse=True, key=takeSecond)
        top_sentence_dict[attention] = [x[0] for x in sentence_attention]


def valid_coordinates(coordinates):
    # 去除前两个gaze点
    coordinates = coordinates[2:-1]
    # 去除前后200ms的gaze点
    begin = 0
    end = -1
    for i, coordinate in enumerate(coordinates):
        if coordinate[2] - coordinates[0][2] > 200:
            begin = i
            break
    print("begin:%d" % begin)
    for i in range(len(coordinates) - 1, -1, -1):
        if coordinates[-1][2] - coordinates[i][2] > 200:
            end = i
            break
    coordinates = coordinates[begin:end]
    return coordinates


def get_visual_attention(
    page_data_id,
    username,
    gaze_x,
    gaze_y,
    gaze_t,
    kernel_size,
    background,
    base_path,
    word_list,
    word_locations,
    top_dict,
    image,
):
    logger.info("visual attention正在分析....")

    """
    1. 清洗数据
    """
    list_x = list(map(float, gaze_x.split(",")))
    list_y = list(map(float, gaze_y.split(",")))
    list_t = list(map(float, gaze_t.split(",")))

    print("length of list")
    print(len(list_x))
    print("kernel_size:%d" % kernel_size)
    if kernel_size != 0:
        # 滤波
        list_x = preprocess_data(list_x, kernel_size)
        list_y = preprocess_data(list_y, kernel_size)

    print("length of list after filter")
    print(len(list_x))
    # 滤波完是浮点数，需要转成int
    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))

    # 组合
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i], list_t[i]]
        coordinates.append(coordinate)
    # 去除开始结束的gaze点
    coordinates = valid_coordinates(coordinates)

    """
    2. 计算top_k
    为了拿到热斑的位置，生成了一次heatmap
    """
    # 计算top_k
    data_list = []
    for coordinate in coordinates:
        data = [coordinate[0], coordinate[1]]
        data_list.append(data)
    hotspot = myHeatmap.draw_heat_map(data_list, base_path + "visual.png", background)

    df = pd.DataFrame(
        {
            "x": [x[0] for x in hotspot],
            "y": [x[1] for x in hotspot],
            "color": [x[2] for x in hotspot],
        }
    )

    # 将坐标按color排序
    df = df.sort_values(by=["color"])
    Q1, Q3, ther_low, ther_up = find_threshold(df)
    hotpixel = []  # [(1232, 85, 240), (1233, 85, 240)]
    for ind, row in df.iterrows():
        if row[2] < Q1:
            hotpixel.append(np.array(row).tolist())

    img = Image.open(background)

    width = img.width

    height = img.height

    # 搜索热斑的范围
    heatspots = []

    graph = [[0] * width for _ in range(height)]

    visit = [[False] * width for _ in range(height)]

    for pix in hotpixel:
        graph[pix[1]][pix[0]] = 1

    V = [x for x in hotpixel]

    while V:
        U = []
        for pix in V:
            if not visit[pix[1]][pix[0]]:
                U.append(pix)
                V.remove(pix)
                break

        for pix in U:
            x = pix[0]
            y = pix[1]
            if not visit[y][x]:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if (graph[y + i][x + j] == 1) and not visit[y + i][x + j] and (i != 0 or j != 0):
                            flag = True
                            for p in U:
                                if (p[0] == x + j) & (p[1] == y + i):
                                    flag = False
                                    break
                            if not flag:
                                continue
                            for p in V:
                                if (p[0] == x + j) & (p[1] == y + i):
                                    U.append(p)
                                    V.remove(p)
                                    break
                visit[y][x] = True
        heatspots.append(U)

    # 计算每个热斑到附近单词的距离
    top_dict["visual"] = []

    # 找每个热斑的临近单词
    for heat in heatspots:
        near_word_idx = []
        for pix in heat:
            word_index = get_word_by_one_gaze(word_locations, pix[0:2])
            if word_index != -1:
                near_word_idx.append(word_index)
        near_word_idx = list(set(near_word_idx))

        words = []
        for idx in near_word_idx:
            word = {
                "idx": idx,
                "word": word_list[idx],
                "x": (word_locations[idx][0] + word_locations[idx][2]) / 2,
                "y": (word_locations[idx][1] + word_locations[idx][3]) / 2,
                "distance_to_heatspot": [],
            }
            words.append(word)
        for pix in heat:
            for word in words:
                word["distance_to_heatspot"].append(get_euclid_distance(word.get("x"), pix[0], word.get("y"), pix[1]))
        dis = []
        for word in words:
            word["distance_to_heatspot"] = sum(word["distance_to_heatspot"]) / len(word["distance_to_heatspot"])
            dis.append(word["distance_to_heatspot"])
        dis.sort()
        theta = 10
        for word in words:
            if word["distance_to_heatspot"] == dis[0]:
                top_dict["visual"].append(word["word"])
        for d in dis[1:]:
            if d - dis[0] < theta:
                for word in words:
                    if word["distance_to_heatspot"] == d:
                        top_dict["visual"].append(word["word"])

    list2 = list(set(top_dict["visual"]))
    list2.sort(key=top_dict["visual"].index)
    top_dict["visual"] = list2

    # 将排序后的结果更换为word

    """
    3. 生成热力图
    为了半透明，再次生成了一次heatmap
    """

    # 生成fixation图示
    fixations = get_fixations(coordinates)
    fixation_image(image, username, fixations, page_data_id, "fixation.png")


def get_center(location):
    return (location["left"] + location["right"]) / 2, (location["top"] + location["bottom"]) / 2


def is_saccade(pre_word_location, now_word_location, magic_saccade_dis):
    center1 = get_center(pre_word_location)
    center2 = get_center(now_word_location)
    return get_euclid_distance(center1[0], center2[0], center1[1], center2[1]) > magic_saccade_dis


def get_para_by_word_index(pre_fixation, para_list):
    for i, para in enumerate(para_list):
        if para[1] >= pre_fixation >= para[0]:
            return i
    return -1


def get_coordinate(gaze_x, gaze_y, gaze_t):
    # 超参
    times_for_remove = 500  # 删除开头结尾的长度，单位是ms
    # 转换格式
    list_x = list(map(float, gaze_x.split(",")))
    list_y = list(map(float, gaze_y.split(",")))
    list_t = list(map(float, gaze_t.split(",")))
    assert len(list_x) == len(list_y) == len(list_t)

    coordinates = []
    # 组合+舍去开头结尾
    for i in range(len(list_x)):
        if list_t[-1] - list_t[i] <= times_for_remove:
            # 舍去最后的一段时间的gaze点
            break
        if list_t[i] - list_t[0] > times_for_remove:
            # 只有开始一段时间后的gaze点才计入
            gaze = [list_x[i], list_y[i], list_t[i]]
            coordinates.append(gaze)
    return coordinates


def is_row_change(i, coordinates, gate):
    # 判断是否产生了换行
    sum_x = 0
    sum_y = 0
    gaze_to_remove = []
    print("i:%d" % i)
    for j in range(i + 1, len(coordinates) - 1, 1):
        sum_x = coordinates[j - 1][0] - coordinates[i][0]

        sum_y = coordinates[j][1] - coordinates[i][1]
        gaze_to_remove.append(j)
        # 如果中途就超过了，还是会产生误差
        if sum_x > gate and sum_y >= 0:
            return True, gaze_to_remove
    print(sum_x)
    return False, []


def split_coordinates_by_row(coordinates, words_location):
    """
    获取按行切割的POGs
    :param coordinates:
    :param words_location:
    :return:
    """
    # 超参
    gate_for_change_row = 0.4

    rows = []
    row = []
    pre_top = words_location[0].get("top")
    for loc in words_location:
        if loc.get("top") != pre_top:
            rows.append(row)
            row = [loc]
        else:
            row.append(loc)
        pre_top = loc.get("top")
    # 把最后一行加上
    rows.append(row)

    rows_info = []
    for item in rows:
        rows_info.append(
            {
                "left": item[0].get("left"),
                "top": item[0].get("top"),
                "right": item[-1].get("right"),
                "bottom": item[0].get("bottom"),
            }
        )
    print(rows_info)
    rows_gaze = []
    row_gaze = []
    row_index = 0
    row_length = rows_info[row_index].get("right") - rows_info[row_index].get("left")
    print("row_length")
    print(row_length)
    gaze_to_remove = []
    for i, coordinate in enumerate(coordinates):
        if i == 0 or i in gaze_to_remove:
            continue
        # 如何判断换行，连续4个点的位移超过阈值，且整体是向下偏移的
        row_change, gaze_to_remove = is_row_change(i, coordinates, row_length * gate_for_change_row)
        if row_change:
            rows_gaze.append(row_gaze)
            row_gaze = [coordinates[gaze_to_remove[-1]]]
            row_index += 1
            # 检测是否有切割错误
            print(row_index)
            print(len(rows_info))
            assert row_index <= len(rows_info) - 1
            row_length = rows_info[row_index].get("right") - rows_info[row_index].get("left")
        else:
            row_gaze.append(coordinate)
    # 检测是否切割错误
    assert row_index == len(rows_info)
    return rows_gaze


def eye_gaze_to_feature(gaze, word_list, sentence_list, words_location, begin):
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
    num_of_fixation = [0 for _ in word_list]
    # reading times
    reading_times = [0 for _ in word_list]
    # 分配到每个单词上
    reading_times_of_sentence_in_word = [0 for _ in word_list]
    second_pass_dwell_time_of_sentence_in_word = [0 for _ in word_list]
    total_dwell_time_of_sentence_in_word = [0 for _ in word_list]
    saccade_times_of_sentence_word_level = [0 for _ in word_list]
    forward_times_of_sentence_word_level = [0 for _ in word_list]
    backward_times_of_sentence_word_level = [0 for _ in word_list]

    loc = json.loads(words_location)
    fix_in_window = get_fixations(gaze[begin:-1])
    for fix in fix_in_window:
        index = get_item_index_x_y(words_location, fix[0], fix[1])
        if index != -1:
            is_watching.append(index)
    fixations = get_fixations(gaze)  # 历史的所有fixation
    if len(fixations) > 0:
        """word level"""
        pre_word_index = -1
        for fixation in fixations:
            index = get_item_index_x_y(words_location, fixation[0], fixation[1])
            if index != -1:
                num_of_fixation[index] += 1
                if index != pre_word_index:
                    reading_times[index] += 1
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

        saccade_times_of_sentence = [0 for x in sentence_list]
        forward_times_of_sentence = [0 for x in sentence_list]
        backward_times_of_sentence = [0 for x in sentence_list]

        pre_fixation = get_item_index_x_y(words_location, fixations[0][0], fixations[0][1])
        magic_saccade_dis = 300
        for fixation in fixations:
            index = get_item_index_x_y(words_location, fixation[0], fixation[1])
            if index != -1:
                now_word_location = loc[index]
                pre_word_location = loc[pre_fixation]
                if is_saccade(pre_word_location, now_word_location, magic_saccade_dis):
                    # 将saccade算入起点的句子
                    sentence_index = get_sentence_by_word(pre_fixation, sentence_list)
                    if sentence_index != -1:
                        saccade_times_of_sentence[sentence_index] += 1
                        # 计算回看
                        if index > pre_fixation:
                            forward_times_of_sentence[sentence_index] += 1
                        else:
                            backward_times_of_sentence[sentence_index] += 1
                pre_fixation = index

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
    return (
        num_of_fixation,
        reading_times,
        reading_times_of_sentence_in_word,
        second_pass_dwell_time_of_sentence_in_word,
        total_dwell_time_of_sentence_in_word,
        saccade_times_of_sentence_word_level,
        forward_times_of_sentence_word_level,
        backward_times_of_sentence_word_level,
        is_watching,
    )


def compute_label(wordLabels, sentenceLabels, wanderLabels, word_list):
    word_understand = [1 for x in word_list]
    if wordLabels:
        wordLabels = json.loads(wordLabels)
        for label in wordLabels:
            word_understand[label] = 0

    sentence_understand = [1 for x in word_list]
    if sentenceLabels:
        sentenceLabels = json.loads(sentenceLabels)
        for label in sentenceLabels:
            for i in range(label[0], label[1]):
                sentence_understand[i] = 0

    mind_wandering = [1 for x in word_list]
    if wanderLabels:
        wanderLabels = json.loads(wanderLabels)
        for label in wanderLabels:
            for i in range(label[0], label[1] + 1):
                mind_wandering[i] = 0

    return word_understand, sentence_understand, mind_wandering


def get_eye_feature_dataset(request):
    # experiments = Experiment.objects.filter(id=641)
    experiments = Experiment.objects.filter(is_finish=True).order_by("-id")[10:51]
    # experiments = Experiment.objects.filter(is_finish=True).filter(id=630)
    success = 0
    fail = 0

    experiment_ids = []
    times = []
    gaze_x = []
    gaze_y = []
    speed = []
    direction = []
    acc = []

    print("共有%d条记录：" % len(experiments))
    for experiment in experiments:
        try:
            page_data_list = PageData.objects.filter(experiment_id=experiment.id)
            word_num = 0
            word = []
            word_understand = []
            sentence_understand = []
            mind_wandering = []

            coors = []
            words = []
            sentences = []
            locations = []
            page_datas = []
            nums = []
            timestamp = 0
            for page_data in page_data_list:
                # 1. 将gaze组合+去除异常值
                coordinates = get_coordinate(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
                coors.append(coordinates)
                # 2.feature计算

                word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
                words_location = json.loads(
                    page_data.location
                )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
                assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
                nums.append(len(word_list))
                word_num += len(word_list)
                word.extend(word_list)

                words.append(word_list)
                sentences.append(sentence_list)
                locations.append(page_data.location)
                page_datas.append(page_data)

                # 生成标签
                word_understand_in_page, sentence_understand_in_page, mind_wander_in_page = compute_label(
                    page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
                )
                word_understand.extend(word_understand_in_page)
                sentence_understand.extend(sentence_understand_in_page)
                mind_wandering.extend(mind_wander_in_page)

            number_of_fixations = [0 for x in range(word_num)]
            reading_times = [0 for x in range(word_num)]
            reading_times_of_sentence = [0 for x in range(word_num)]  # 相对的
            second_pass_dwell_time_of_sentence = [0 for x in range(word_num)]  # 相对的
            total_dwell_time_of_sentence = [0 for x in range(word_num)]  # 相对的
            saccade_times_of_sentence = [0 for x in range(word_num)]
            forward_times_of_sentence = [0 for x in range(word_num)]
            backward_times_of_sentence = [0 for x in range(word_num)]

            for i, coor in enumerate(coors):
                begin = 0
                for j, coordinate in enumerate(coor):
                    if j == 0:
                        continue
                    if coor[j][2] - coor[begin][2] > 800:
                        (
                            num_of_fixation_this_page,
                            reading_times_this_page,
                            reading_times_of_sentence_in_word_this_page,
                            second_pass_dwell_time_of_sentence_in_word_this_page,
                            total_dwell_time_of_sentence_in_word_this_page,
                            saccade_times_of_sentence_word_level_this_page,
                            forward_times_of_sentence_word_level_this_page,
                            backward_times_of_sentence_word_level_this_page,
                            is_watching,
                        ) = eye_gaze_to_feature(coor[0:j], words[i], sentences[i], locations[i], begin)

                        if i == 0:
                            begin_index = 0
                        else:
                            begin_index = nums[i - 1]

                        cnt = 0

                        word_watching = [0 for _ in range(word_num)]
                        for item in is_watching:
                            word_watching[item + begin_index] = 1

                        for x in range(begin_index, nums[i]):
                            number_of_fixations[x] += num_of_fixation_this_page[cnt]
                            reading_times[x] += reading_times_this_page[cnt]
                            reading_times_of_sentence[x] += reading_times_of_sentence_in_word_this_page[cnt]  # 相对的
                            second_pass_dwell_time_of_sentence[
                                x
                            ] += second_pass_dwell_time_of_sentence_in_word_this_page[
                                cnt
                            ]  # 相对的
                            total_dwell_time_of_sentence[x] += total_dwell_time_of_sentence_in_word_this_page[
                                cnt
                            ]  # 相对的
                            saccade_times_of_sentence[x] += saccade_times_of_sentence_word_level_this_page[cnt]
                            forward_times_of_sentence[x] += forward_times_of_sentence_word_level_this_page[cnt]
                            backward_times_of_sentence[x] += backward_times_of_sentence_word_level_this_page[cnt]
                            cnt += 1

                        experiment_ids.append(experiment.id)
                        times.append(timestamp)
                        gaze_of_x = [x[0] for x in coor[begin:j]]
                        gaze_of_y = [x[1] for x in coor[begin:j]]
                        speed_now, direction_now, acc_now = coor_to_input(coor[begin:j], 8)
                        assert len(gaze_of_x) == len(gaze_of_y) == len(speed_now) == len(direction_now) == len(acc_now)
                        gaze_x.append(gaze_of_x)
                        gaze_y.append(gaze_of_y)
                        speed.append(speed_now)
                        direction.append(direction_now)
                        acc.append(acc_now)

                        begin = j
                        df = pd.DataFrame(
                            {
                                # 1. 实验信息相关
                                "experiment_id": [experiment.id for x in range(word_num)],
                                "user": [experiment.user for x in range(word_num)],
                                "article_id": [experiment.article_id for x in range(word_num)],
                                "time": [timestamp for x in range(word_num)],
                                "word": word,
                                "word_watching": word_watching,
                                # # 2. label相关
                                "word_understand": word_understand,
                                "sentence_understand": sentence_understand,
                                "mind_wandering": mind_wandering,
                                # 3. 特征相关
                                # word level
                                "reading_times": reading_times,
                                "number_of_fixations": number_of_fixations,
                                # sentence level
                                "second_pass_dwell_time_of_sentence": second_pass_dwell_time_of_sentence,
                                "total_dwell_time_of_sentence": total_dwell_time_of_sentence,
                                "reading_times_of_sentence": reading_times_of_sentence,
                                "saccade_times_of_sentence": saccade_times_of_sentence,
                                "forward_times_of_sentence": forward_times_of_sentence,
                                "backward_times_of_sentence": backward_times_of_sentence,
                            }
                        )
                        timestamp += 1
                        path = "static\\data\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + ".csv"

                        if os.path.exists(path):
                            df.to_csv(path, index=False, mode="a", header=False)
                        else:
                            df.to_csv(path, index=False, mode="a")
            success += 1
            logger.info("已成功生成%d条，失败%d条" % (success, fail))
        except Exception:
            fail += 1
    # 生成注视信息的数据集
    data = pd.DataFrame(
        {
            # 1. 实验信息相关
            "experiment_id": experiment_ids,
            "time": times,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "speed": speed,
            "direction": direction,
            "acc": acc,
        }
    )
    path = "static\\data\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + "-gaze.csv"
    if os.path.exists(path):
        data.to_csv(path, index=False, mode="a", header=False)
    else:
        data.to_csv(path, index=False, mode="a")
    logger.info("成功生成%d条，失败%d条" % (success, fail))
    return JsonResponse({"status": "ok"})


def get_dataset(request):
    # 要确定有几段，把所有段串联成，去计算每个单词的特征
    # 特征可以直接用raw data去算，也可以用smooth过的特征去算
    # label分别是按照单词，句子、段落给的，可以先试下单词怎么给

    experiments = Experiment.objects.filter(is_finish=True)
    success = 0
    fail = 0
    print("共有%d条记录：" % len(experiments))
    for experiment in experiments:
        if success == 10:
            break
        try:
            page_data_list = PageData.objects.filter(experiment_id=experiment.id)

            word = []
            # word level
            word_understand = []
            sentence_understand = []
            mind_wandering = []
            reading_times = []
            number_of_fixations = []
            # sentence level
            reading_times_of_sentence = []  # 相对的
            second_pass_dwell_time_of_sentence = []  # 相对的
            total_dwell_time_of_sentence = []  # 相对的
            saccade_times_of_sentence = []
            forward_times_of_sentence = []
            backward_times_of_sentence = []
            # para level
            saccade_times_of_para = []
            forward_saccade_times_of_para = []
            backward_saccade_times_of_para = []

            # 分页填充数据
            word_num = 0  # 记录总单词数
            for page_data in page_data_list:
                word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
                words_location = json.loads(
                    page_data.location
                )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
                assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
                word_num += len(word_list)
                word.extend(word_list)

                # 打标签
                # 单词不懂
                word_understand_this_page = [1 for x in word_list]
                if page_data.wordLabels:
                    wordLabels = json.loads(page_data.wordLabels)
                    for label in wordLabels:
                        word_understand_this_page[label] = 0
                word_understand.extend(word_understand_this_page)

                # 句子不懂,实际上也反应在单词上
                sentence_understand_this_page = [1 for x in word_list]
                if page_data.sentenceLabels:
                    sentenceLabels = json.loads(page_data.sentenceLabels)
                    for label in sentenceLabels:
                        for i in range(label[0], label[1]):
                            sentence_understand_this_page[i] = 0
                sentence_understand.extend(sentence_understand_this_page)

                # 走神，以一段为标签出现
                mind_wandering_this_page = [1 for x in word_list]
                if page_data.wanderLabels:
                    wanderLabels = json.loads(page_data.wanderLabels)
                    for label in wanderLabels:
                        for i in range(label[0], label[1] + 1):
                            mind_wandering_this_page[i] = 0
                mind_wandering.extend(mind_wandering_this_page)

                # 计算特征
                """
                word level
                """
                list_x = list(map(float, page_data.gaze_x.split(",")))
                list_y = list(map(float, page_data.gaze_y.split(",")))
                list_t = list(map(float, page_data.gaze_t.split(",")))

                # 组合
                coordinates = []
                for i in range(len(list_x)):
                    coordinate = [list_x[i], list_y[i], list_t[i]]
                    coordinates.append(coordinate)
                fixations = get_fixations(coordinates)

                # 计算number of fixation
                number_of_fixations_this_page = [0 for x in word_list]
                for fixation in fixations:
                    index = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
                    if index != -1:
                        number_of_fixations_this_page[index] += 1
                number_of_fixations.extend(number_of_fixations_this_page)
                # 计算reading times
                reading_times_this_page = [0 for x in word_list]
                pre_word_index = -1
                for fixation in fixations:
                    index = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
                    if index != pre_word_index and index != -1:
                        reading_times_this_page[index] += 1
                        pre_word_index = index
                reading_times.extend(reading_times_this_page)

                """
                句子level
                """
                # 先从句子角度去看fixation的相关特征，之后将其再分配回单词上
                reading_times_of_sentence_this_page_in_sentence_level = [0 for x in sentence_list]
                second_pass_dwell_time_of_sentence_this_page_in_sentence_level = [0 for x in sentence_list]
                total_dwell_time_of_this_page_in_sentence_level = [0 for x in sentence_list]
                pre_sentence_index = -1
                for fixation in fixations:
                    word_index = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
                    sentence_index = get_sentence_by_word(word_index, sentence_list)
                    if sentence_index != -1:
                        # 累积fixation duration
                        total_dwell_time_of_this_page_in_sentence_level[sentence_index] += fixation[2]
                        # 计算reading times
                        if sentence_index != pre_sentence_index:
                            reading_times_of_sentence_this_page_in_sentence_level[sentence_index] += 1
                        # 只有在reading times是2时，才累积fixation duration
                        if reading_times_of_sentence_this_page_in_sentence_level[sentence_index] == 2:
                            second_pass_dwell_time_of_sentence_this_page_in_sentence_level[sentence_index] += fixation[
                                2
                            ]
                # 分配到每个单词上
                reading_times_of_sentence_this_page_in_word_level = [0 for x in word_list]
                second_pass_dwell_time_of_sentence_this_page_in_word_level = [0 for x in word_list]
                total_dwell_time_of_this_page_in_word_level = [0 for x in word_list]

                for i, sentence in enumerate(sentence_list):
                    for j in range(sentence[1], sentence[2]):
                        reading_times_of_sentence_this_page_in_word_level[
                            j
                        ] = reading_times_of_sentence_this_page_in_sentence_level[i] / (math.log(sentence[3] + 1))
                        second_pass_dwell_time_of_sentence_this_page_in_word_level[
                            j
                        ] = second_pass_dwell_time_of_sentence_this_page_in_sentence_level[i] / (
                            math.log(sentence[3] + 1)
                        )
                        total_dwell_time_of_this_page_in_word_level[
                            j
                        ] = total_dwell_time_of_this_page_in_sentence_level[i] / (math.log(sentence[3] + 1))
                reading_times_of_sentence.extend(reading_times_of_sentence_this_page_in_word_level)
                second_pass_dwell_time_of_sentence.extend(second_pass_dwell_time_of_sentence_this_page_in_word_level)
                total_dwell_time_of_sentence.extend(total_dwell_time_of_this_page_in_word_level)

                saccade_times_of_sentence_in_sentence = [0 for x in sentence_list]
                forward_times_of_sentence_in_sentence = [0 for x in sentence_list]
                backward_times_of_sentence_in_sentence = [0 for x in sentence_list]

                pre_fixation = get_item_index_x_y(page_data.location, fixations[0][0], fixations[0][1])
                magic_saccade_dis = 300
                for fixation in fixations:
                    index = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
                    if index != -1:
                        now_word_location = words_location[index]
                        pre_word_location = words_location[pre_fixation]
                        if is_saccade(pre_word_location, now_word_location, magic_saccade_dis):
                            # 将saccade算入起点的句子
                            sentence_index = get_sentence_by_word(pre_fixation, sentence_list)
                            if sentence_index != -1:
                                saccade_times_of_sentence_in_sentence[sentence_index] += 1
                                # 计算回看
                                if index > pre_fixation:
                                    forward_times_of_sentence_in_sentence[sentence_index] += 1
                                else:
                                    backward_times_of_sentence_in_sentence[sentence_index] += 1
                        pre_fixation = index
                saccade_times_of_sentence_word_level = [0 for x in word_list]
                forward_times_of_sentence_word_level = [0 for x in word_list]
                backward_times_of_sentence_word_level = [0 for x in word_list]
                for i, sentence in enumerate(sentence_list):
                    scale = math.log((sentence[3] + 1))
                    for j in range(sentence[1], sentence[2]):
                        saccade_times_of_sentence_word_level[j] = saccade_times_of_sentence_in_sentence[i] / scale
                        forward_times_of_sentence_word_level[j] = forward_times_of_sentence_in_sentence[i] / scale
                        backward_times_of_sentence_word_level[j] = backward_times_of_sentence_in_sentence[i] / scale
                saccade_times_of_sentence.extend(saccade_times_of_sentence_word_level)
                forward_times_of_sentence.extend(forward_times_of_sentence_word_level)
                backward_times_of_sentence.extend(backward_times_of_sentence_word_level)
                """
                para level
                """
                if len(page_data.para) > 3:
                    para_list = json.loads(page_data.para)  # [[0,9],[10,17]
                else:
                    logger.warning("para data :%d 没有para标签" % page_data.id)
                    para_list = []
                saccade_times_of_para_in_para = [0 for i in para_list]
                forward_saccade_times_of_para_in_para = [0 for i in para_list]
                backward_saccade_times_of_para_in_para = [0 for i in para_list]

                # 计算saccade的magic number，当两个fixation的距离大于300，则为saccade
                magic_saccade_dis = 300
                pre_fixation = get_item_index_x_y(page_data.location, fixations[0][0], fixations[0][1])
                for fixation in fixations:
                    index = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
                    if index != -1:
                        now_word_location = words_location[index]
                        pre_word_location = words_location[pre_fixation]
                        if is_saccade(pre_word_location, now_word_location, magic_saccade_dis):
                            # 将saccade算入起点的段落
                            para_index = get_para_by_word_index(pre_fixation, para_list)
                            if para_list != -1:
                                saccade_times_of_para_in_para[para_index] += 1
                                # 计算回看
                                if index > pre_fixation:
                                    forward_saccade_times_of_para_in_para[para_index] += 1
                                else:
                                    backward_saccade_times_of_para_in_para[para_index] += 1
                        pre_fixation = index
                saccade_times_of_para_word_level = [0 for i in word_list]
                forward_saccade_times_word_level = [0 for i in word_list]
                backward_saccade_times_word_level = [0 for i in word_list]
                for i, para in enumerate(para_list):
                    for j in range(para[0], para[1] + 1):
                        assert len(word_list) > para[1]
                        scale = math.log((para[1] - para[0] + 1) + 1)
                        saccade_times_of_para_word_level[j] = saccade_times_of_para_in_para[i] / scale
                        forward_saccade_times_word_level[j] = forward_saccade_times_of_para_in_para[i] / scale
                        backward_saccade_times_word_level[j] = backward_saccade_times_of_para_in_para[i] / scale

                saccade_times_of_para.extend(saccade_times_of_para_word_level)
                forward_saccade_times_of_para.extend(forward_saccade_times_word_level)
                backward_saccade_times_of_para.extend(backward_saccade_times_word_level)

            df = pd.DataFrame(
                {
                    # 1. 实验信息相关
                    "experiment_id": [experiment.id for x in range(word_num)],
                    "user": [experiment.user for x in range(word_num)],
                    "article_id": [experiment.article_id for x in range(word_num)],
                    "word": word,
                    # # 2. label相关
                    "word_understand": word_understand,
                    "sentence_understand": sentence_understand,
                    "mind_wandering": mind_wandering,
                    # 3. 特征相关
                    # word level
                    "reading_times": reading_times,
                    "number_of_fixations": number_of_fixations,
                    # sentence level
                    "second_pass_dwell_time_of_sentence": second_pass_dwell_time_of_sentence,
                    "total_dwell_time_of_sentence": total_dwell_time_of_sentence,
                    "reading_times_of_sentence": reading_times_of_sentence,
                    "saccade_times_of_sentence": saccade_times_of_sentence,
                    "forward_times_of_sentence": forward_times_of_sentence,
                    "backward_times_of_sentence": backward_times_of_sentence,
                    # para level
                    "saccade_times_of_para": saccade_times_of_para,
                    "forward_saccade_times_of_para": forward_saccade_times_of_para,
                    "backward_saccade_times_of_para": backward_saccade_times_of_para,
                }
            )
            path = "static\\data\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + ".csv"
            # path = "static\\data\\dataset\\" + "test.csv"
            import os

            if os.path.exists(path):
                df.to_csv(path, index=False, mode="a", header=False)
            else:
                df.to_csv(path, index=False, mode="a")
            success += 1
        except Exception as e:
            logger.warning("experiment: %d 生成数据失败" % experiment.id)
            fail += 1
    logger.info("成功生成%d条，失败%d条" % (success, fail))
    return JsonResponse({"status": "ok"})


def get_sentence_by_word(word_index, sentence_list):
    if word_index == -1:
        return -1
    for i, sentence in enumerate(sentence_list):
        if sentence[2] > word_index >= sentence[1]:
            return i
    return -1


def article_2_csv(request):
    texts = Text.objects.all()
    ids = []
    contents = []
    for text in texts:
        para_list = Paragraph.objects.filter(article_id=text.id).order_by("para_id")
        content = ""
        for para in para_list:
            tmp = para.content.replace("...", ".").replace("..", ".").replace(". ", ".")
            tmp = tmp.replace(".", ". ")
            content = content + tmp
        contents.append(content)
        ids.append(text.id)
    df = pd.DataFrame(
        {
            "id": ids,
            "content": contents,
        }
    )
    path = "static\\data\\dataset\\" + "article.csv"
    df.to_csv(path, index=False, header=False)
    return JsonResponse({"status_code": 200, "status": "ok"})


def get_cnn_dataset(request):
    data_list = PageData.objects.filter(wordLabels__isnull=False).exclude(wordLabels=1)
    x = []
    y = []
    t = []
    for data in data_list:
        x.append(data.gaze_x)
        y.append(data.gaze_y)
        t.append(data.gaze_t)
    df = pd.DataFrame({"x": x, "y": y, "t": t})
    path = "static\\data\\dataset\\filter.csv"
    df.to_csv(path, index=False)

    return HttpResponse(1)


def filter_layer(data, kernel_size=7):
    data = preprocess_data(data, kernel_size)

    data = list(map(int, data))

    return data


def get_input(gaze_x, gaze_y, gaze_t):
    # 超参
    times_for_remove = 500  # 删除开头结尾的长度，单位是ms
    # 转换格式
    list_x = list(map(float, gaze_x.split(",")))
    list_y = list(map(float, gaze_y.split(",")))
    list_t = list(map(float, gaze_t.split(",")))
    assert len(list_x) == len(list_y) == len(list_t)

    coordinates = []
    # 组合+舍去开头结尾
    for i in range(len(list_x)):
        if list_t[-1] - list_t[i] <= times_for_remove:
            # 舍去最后的一段时间的gaze点
            break
        if list_t[i] - list_t[0] > times_for_remove:
            # 只有开始一段时间后的gaze点才计入
            gaze = [list_x[i], list_y[i], list_t[i]]
            coordinates.append(gaze)
    return coordinates


def test(request):
    page_data = PageData.objects.get(id=1319)
    # 1. 将gaze组合+去除异常值
    window = 8
    coordinates = get_coordinate(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
    # 需要使用深拷贝，不然会改变对象
    input = gaze_to_input(copy.deepcopy(coordinates), window)
    output = input_to_label(input).tolist()
    assert len(output) == len(input)
    for i, coor in enumerate(coordinates):
        coor.append(output[i])
    get_feature(coordinates)

    return HttpResponse(output)


def get_feature(coordinates):
    fixations = 0
    saccades = 0
    for i, coor in enumerate(coordinates):
        if i == 0:
            continue
        if coordinates[i][-1] != coordinates[i - 1][-1]:
            if coordinates[i - 1][-1] == 1:
                fixations += 1
            if coordinates[i - 1][-1] == 2:
                saccades += 1
    return [fixations, saccades]


def gaze_to_input(coordinates, window):
    """
    将每个gaze点赋值上标签
    :param gaze_x:
    :param gaze_y:
    :param gaze_t:
    :param window:
    :return:
    """

    for i, coordinate in enumerate(coordinates):
        # 确定计算的窗口
        begin = i
        end = i
        for j in range(int(window / 2)):
            if begin == 0:
                break
            begin -= 1

        for j in range(int(window / 2)):
            if end == len(coordinates) - 1:
                break
            end += 1

        assert begin >= 0
        assert end <= len(coordinates) - 1

        # 计算速度、方向，作为模型输入
        time = (coordinates[end][2] - coordinates[begin][2]) / 100
        speed = (
            get_euclid_distance(coordinates[begin][0], coordinates[end][0], coordinates[begin][1], coordinates[end][1])
            / time
        )
        direction = math.atan2(coordinates[end][1] - coordinates[begin][1], coordinates[end][0] - coordinates[begin][0])
        coordinate.append(speed)
        coordinate.append(direction)
    # 计算加速度
    for i, coordinate in enumerate(coordinates):
        begin = i
        end = i
        for j in range(int(window / 2)):
            if begin == 0:
                break
            begin -= 1

        for j in range(int(window / 2)):
            if end == len(coordinates) - 1:
                break
            end += 1

        assert begin >= 0
        assert end <= len(coordinates) - 1
        time = (coordinates[end][2] - coordinates[begin][2]) / 100
        acc = (coordinates[end][3] - coordinates[begin][3]) / time
        coordinate.append(acc * acc)
    input = [[x[0], x[1], x[3], x[4], x[5]] for x in coordinates]
    return input


def coor_to_input(coordinates, window):
    """
    将每个gaze点赋值上标签
    :param gaze_x:
    :param gaze_y:
    :param gaze_t:
    :param window:
    :return:
    """

    for i, coordinate in enumerate(coordinates):
        # 确定计算的窗口
        begin = i
        end = i
        for j in range(int(window / 2)):
            if begin == 0:
                break
            begin -= 1

        for j in range(int(window / 2)):
            if end == len(coordinates) - 1:
                break
            end += 1

        assert begin >= 0
        assert end <= len(coordinates) - 1

        # 计算速度、方向，作为模型输入
        time = (coordinates[end][2] - coordinates[begin][2]) / 100
        speed = (
            get_euclid_distance(coordinates[begin][0], coordinates[end][0], coordinates[begin][1], coordinates[end][1])
            / time
        )
        direction = math.atan2(coordinates[end][1] - coordinates[begin][1], coordinates[end][0] - coordinates[begin][0])
        coordinate.append(speed)
        coordinate.append(direction)
    # 计算加速度
    for i, coordinate in enumerate(coordinates):
        begin = i
        end = i
        for j in range(int(window / 2)):
            if begin == 0:
                break
            begin -= 1

        for j in range(int(window / 2)):
            if end == len(coordinates) - 1:
                break
            end += 1

        assert begin >= 0
        assert end <= len(coordinates) - 1
        time = (coordinates[end][2] - coordinates[begin][2]) / 100
        acc = (coordinates[end][3] - coordinates[begin][3]) / time
        coordinate.append(acc * acc)

    speed = [x[3] for x in coordinates]
    direction = [x[4] for x in coordinates]
    acc = [x[5] for x in coordinates]
    return speed, direction, acc


def input_to_label(input):
    """
    使用模型做预测，拿到真实的label
    :param input:
    :return:
    """
    input = torch.tensor(input)
    input = torch.unsqueeze(input, 0)
    output = model(input)
    output = torch.squeeze(output, 0)
    output = torch.argmax(output, dim=1)
    output = output.numpy()
    print("fixation占比:%f" % (np.sum(output == 1) / len(output)))
    print("saccade占比:%f" % (np.sum(output == 2) / len(output)))
    return output


def get_t(gaze_t):
    # 超参
    times_for_remove = 500  # 删除开头结尾的长度，单位是ms
    # 转换格式
    list_t = list(map(float, gaze_t.split(",")))

    new_list_t = []
    # 组合+舍去开头结尾
    for i in range(len(list_t)):
        if list_t[-1] - list_t[i] <= times_for_remove:
            # 舍去最后的一段时间的gaze点
            break
        if list_t[i] - list_t[0] > times_for_remove:
            # 只有开始一段时间后的gaze点才计入
            new_list_t.append(list_t[i])
    return new_list_t


def get_fixation_by_time(request):
    exp_id = request.GET.get("exp_id")
    time = request.GET.get("time")

    # 1. 准备底图
    # 先要把time按照page分页
    page_data_ls = PageData.objects.filter(experiment_id=exp_id)

    timestamp = 0
    timestamps = []  # 每页的time范围
    for page_data in page_data_ls:
        gaze_t = get_t(page_data.gaze_t)
        print(gaze_t)
        begin = 0
        for i, t in enumerate(gaze_t):
            if i == 0:
                continue
            if gaze_t[i] - gaze_t[begin] > 800:
                timestamp += 1
                begin = i
        timestamps.append(timestamp)

    print("timestamps")
    print(timestamps)
    page_index = -1
    for i, item in enumerate(timestamps):
        print(item)
        if item >= int(time):
            page_index = i
            break
    assert page_index >= 0
    assert len(page_data_ls) >= page_index

    page_data = page_data_ls[page_index]

    image = page_data.image.split(",")[1]
    image_data = base64.b64decode(image)

    filename = "static\\data\\timestamp\\origin.png"
    with open(filename, "wb") as f:
        f.write(image_data)

    # 2. 准备gaze点
    row = -1
    csv = "static\\dataset\\10-31-43-gaze.csv"
    file = pd.read_csv(csv)
    for i, r in file.iterrows():
        if str(r["experiment_id"]) == exp_id and str(r["time"]) == time:
            print(type(r["experiment_id"]))
            print(type(exp_id))
            row = i
            break
    if row == -1:
        return JsonResponse({"code": 404, "status": "未检索相关信息"}, json_dumps_params={"ensure_ascii": False}, safe=False)

    print(file["gaze_x"][row])
    print(type(file["gaze_x"][row]))

    gaze_x = file["gaze_x"][row]
    gaze_y = file["gaze_y"][row]

    list_x = list(map(float, gaze_x[1:-1].split(",")))
    list_y = list(map(float, gaze_y[1:-1].split(",")))
    list_t = []

    time_t = 0
    for i in range(len(list_x)):
        list_t.append(time_t)
        time_t += 30
    assert len(list_x) == len(list_y) == len(list_t)

    coordinates = []
    for i in range(len(list_x)):
        tmp = [int(list_x[i]), int(list_y[i]), int(list_t[i])]
        coordinates.append(tmp)
    fixations = get_fixations(coordinates)

    # 3. 画图
    img = cv2.imread(filename)
    cnt = 0
    pre_fix = (0, 0)
    for i, fix in enumerate(fixations):
        cv2.circle(
            img,
            (fix[0], fix[1]),
            3,
            (0, 0, 255),
            -1,
        )
        if cnt > 0:
            cv2.line(
                img,
                (pre_fix[0], pre_fix[1]),
                (fix[0], fix[1]),
                (0, 0, 255),
                1,
            )

        # 标序号 间隔着标序号
        cv2.putText(
            img,
            str(cnt),
            (fix[0], fix[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        cnt = cnt + 1
        pre_fix = fix
    result_filename = "static\\data\\timestamp\\" + str(exp_id) + "-" + str(time) + "-fix.png"
    cv2.imwrite(result_filename, img)
    logger.info("fixation 图片生成路径:%s" % result_filename)

    img = cv2.imread(filename)
    for i, fix in enumerate(coordinates):
        cv2.circle(
            img,
            (fix[0], fix[1]),
            3,
            (0, 0, 255),
            -1,
        )
    result_filename = "static\\data\\timestamp\\" + str(exp_id) + "-" + str(time) + "-gaze.png"
    cv2.imwrite(result_filename, img)
    logger.info("gaze 图片生成路径:%s" % result_filename)
    return JsonResponse({"code": 200, "status": "生成完毕"}, json_dumps_params={"ensure_ascii": False}, safe=False)


def get_speed(request):
    page_ids = request.GET.get("ids")
    index = request.GET.get("index")
    page_id_ls = page_ids.split(",")
    dict = {}
    for page_id in page_id_ls:
        if len(page_id) > 0:
            page_data = PageData.objects.filter(id=page_id).first()
            coors = x_y_t_2_coordinate(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
            time = 0
            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            for coor in coors:
                word_index = get_item_index_x_y(page_data.location, coor[0], coor[1])
                if word_index != -1:
                    sentence_index = get_sentence_by_word(word_index, sentence_list)
                    if sentence_index >= int(index):
                        time += 30
            dict[page_id] = time
    return JsonResponse(dict, json_dumps_params={"ensure_ascii": False}, safe=False)
