import datetime
import json
import random

import cv2
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger

from action.models import (
    Text,
    Dictionary,
    PageData,
    Dispersion,
    Paragraph,
    Experiment,
    Translation,
)
from heatmap import draw_heat_map

import pandas as pd
import numpy as np

from onlineReading.utils import (
    translate,
    get_euclid_distance,
    pixel_2_cm,
    pixel_2_deg,
    cm_2_pixel,
)
from semantic_attention import (
    generate_word_difficulty,
    generate_word_attention,
    generate_sentence_attention,
)
from utils import (
    x_y_t_2_coordinate,
    get_fixations,
    fixation_image,
    get_sentence_by_word_index,
    add_fixations_to_location,
    get_row_location,
    get_out_of_screen_times,
    get_proportion_of_horizontal_saccades,
    get_saccade_angle,
    get_saccade_info,
    get_reading_times_of_word,
    get_reading_times_and_dwell_time_of_sentence,
    get_saccade,
    preprocess_data,
    get_importance,
    get_word_by_index,
    get_word_and_location,
    get_index_by_word,
    topk_tuple,
    get_word_and_sentence_from_text,
    get_word_location,
    get_sentence_location,
    get_top_k,
    get_word_by_one_gaze,
    calculate_similarity,
    calculate_identity,
    paint_bar_graph, apply_heatmap
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

    article_id = request.GET.get("article_id", 6)

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
                    logger.info(
                        "该翻译已经缓存，读取时间为%sms"
                        % round((endtime - starttime).microseconds / 1000 / 1000, 3)
                    )
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
    experiment = Experiment.objects.create(
        article_id=article_id, user=request.session.get("username")
    )
    request.session["experiment_id"] = experiment.id
    logger.info(
        "--本次实验开始,实验者：%s，实验id：%d--" % (request.session.get("username"), experiment.id)
    )
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

    if experiment_id:
        for label in labels:
            PageData.objects.filter(experiment_id=experiment_id).filter(
                page=label["page"]
            ).update(
                wordLabels=label["wordLabels"],
                sentenceLabels=label["sentenceLabels"],
                wanderLabels=label["wanderLabels"],
            )
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
        mean_fixations_duration.append(
            analysis_result_by_word_level[key]["mean_fixations_duration"]
        )
        fixation_duration.append(
            analysis_result_by_word_level[key]["fixation_duration"]
        )
        second_pass_duration.append(
            analysis_result_by_word_level[key]["second_pass_duration"]
        )
        number_of_fixations.append(
            analysis_result_by_word_level[key]["number_of_fixations"]
        )
        reading_times.append(analysis_result_by_word_level[key]["reading_times"])
        first_reading_durations.append(
            analysis_result_by_word_level[key]["first_reading_durations"]
        )
        second_reading_durations.append(
            analysis_result_by_word_level[key]["second_reading_durations"]
        )
        third_reading_durations.append(
            analysis_result_by_word_level[key]["third_reading_durations"]
        )
        fourth_reading_durations.append(
            analysis_result_by_word_level[key]["fourth_reading_durations"]
        )

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
        reading_times_of_sentence.append(
            analysis_result_by_sentence_level[key]["reading_times_of_sentence"]
        )
        dwell_time.append(analysis_result_by_sentence_level[key]["dwell_time"])
        second_pass_dwell_time.append(
            analysis_result_by_sentence_level[key]["second_pass_dwell_time"]
        )

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
    backward_saccade_times.append(
        analysis_result_by_page_level["backward_saccade_times"]
    )
    mean_saccade_length.append(analysis_result_by_page_level["mean_saccade_length"])
    mean_saccade_angle.append(analysis_result_by_page_level["mean_saccade_angle"])
    out_of_screen_times.append(analysis_result_by_page_level["out_of_screen_times"])
    proportion_of_horizontal_saccades.append(
        analysis_result_by_page_level["proportion of horizontal saccades"]
    )
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
            gaze_coordinates = x_y_t_2_coordinate(
                pagedata.gaze_x, pagedata.gaze_y, pagedata.gaze_t
            )
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
            reading_times_of_word, reading_durations = get_reading_times_of_word(
                fixations, pagedata.location
            )
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
                    "second_reading_durations": reading_durations[key][2]
                    if reading_times_of_word[key] > 1
                    else 0,
                    "third_reading_durations": reading_durations[key][3]
                    if reading_times_of_word[key] > 2
                    else 0,
                    "fourth_reading_durations": reading_durations[key][4]
                    if reading_times_of_word[key] > 3
                    else 0,
                }

                analysis_result_by_word_level[key] = result

            """
                获取sentence level的特征
            """
            analysis_result_by_sentence_level = {}
            # 获取不懂的句子
            sentencelabels = json.loads(pagedata.sentenceLabels)

            # 需要输出的句子level的输出结果
            # 获取每隔句子的reading times dict/list 下标代表第几次 0-first pass
            (
                reading_times_of_sentence,
                dwell_time_of_sentence,
                number_of_word,
            ) = get_reading_times_and_dwell_time_of_sentence(
                fixations, pagedata.location, sentence_index
            )
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
            row_fixation = add_fixations_to_location(
                fixations, str(row_info).replace("'", '"')
            )
            analysis_result_by_row_level = {}
            wanderlabels = json.loads(pagedata.wanderLabels)
            # TODO 一旦有一行是wander,那么整页的标签就是wander
            page_wander = 0
            for key in row_fixation:
                saccade_times_of_row, mean_saccade_angle_of_row = get_saccade_info(
                    row_fixation[key]
                )
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
                )

            # 将数据写入csv
            # 先看word level
            if csv:
                # word_path = "static/user/dataset/" + "word_level_2.csv"
                # add_word_feature_to_csv(analysis_result_by_word_level, word_path)

                sentence_path = "static/user/dataset/" + "sentence_level_lq.csv"
                add_sentence_feature_to_csv(
                    analysis_result_by_sentence_level, sentence_path
                )

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
    gaze_coordinates = x_y_t_2_coordinate(
        pagedata.gaze_x, pagedata.gaze_y, pagedata.gaze_t
    )
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
    reading_times_of_word, reading_durations = get_reading_times_of_word(
        fixations, pagedata.location
    )
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
            "second_reading_durations": reading_durations[key][2]
            if reading_times_of_word[key] > 1
            else 0,
            "third_reading_durations": reading_durations[key][3]
            if reading_times_of_word[key] > 2
            else 0,
            "fourth_reading_durations": reading_durations[key][4]
            if reading_times_of_word[key] > 3
            else 0,
        }

        analysis_result_by_word_level[key] = result

    """
        获取sentence level的特征
    """
    analysis_result_by_sentence_level = {}
    # 获取不懂的句子
    sentencelabels = json.loads(pagedata.sentenceLabels)

    # 需要输出的句子level的输出结果
    # 获取每隔句子的reading times dict/list 下标代表第几次 0-first pass
    (
        reading_times_of_sentence,
        dwell_time_of_sentence,
        number_of_word,
    ) = get_reading_times_and_dwell_time_of_sentence(
        fixations, pagedata.location, sentence_index
    )
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
        saccade_times_of_row, mean_saccade_angle_of_row = get_saccade_info(
            row_fixation[key]
        )
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


def get_visual_heatmap(request):
    """
    绘制visual attention
    :param request:
    :return:
    """
    # 1. 获取gaze点数据
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)
    list_x = list(map(float, pageData.gaze_x.split(",")))
    list_y = list(map(float, pageData.gaze_y.split(",")))

    # 2. 滤波处理
    kernel_size = int(request.GET.get("window", 0))

    if kernel_size != 0:
        # 滤波
        list_x = preprocess_data(list_x, kernel_size)
        list_y = preprocess_data(list_y, kernel_size)

    # 3. 组合数据
    # 滤波完是浮点数，需要转成int
    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))
    # 组合
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i]]
        coordinates.append(coordinate)

    # 4. 画图
    exp = Experiment.objects.filter(id=pageData.experiment_id)
    if not exp:
        username = "tmp"
        base = "static\\background\\" + str(request.GET.get("background")) + ".jpg"
    else:
        username = exp.first().user
        base = (
                "static\\background\\"
                + str(exp.first().article_id)
                + "\\"
                + str(pageData.page)
                + ".jpg"
        )

    hit_pic_name = (
            "static\\data\\heatmap\\"
            + str(username)
            + "\\visual"
            + "\\hit_"
            + str(page_data_id)
            + "_"
            + str(kernel_size)
            + ".png"
    )
    heatmap_name = (
            "static\\data\\heatmap\\"
            + str(username)
            + "\\visual"
            + "\\heatmap_"
            + str(page_data_id)
            + "_"
            + str(kernel_size)
            + ".png"
    )
    draw_heat_map(coordinates, hit_pic_name, heatmap_name, base)

    return HttpResponse(1)


def test_motion(request):
    return render(request, "test_motion.html")


def get_nlp_heatmap(request):
    """
    绘制text attention
    :param request:
    :return:
    """
    # 1. 获取文本数据
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)

    # 2. 调用文本分析的接口
    atention_type = int(request.GET.get("type", 0))
    if atention_type == 0:
        # 单词在文中的重要性
        attention = get_importance(pageData.texts)
    elif atention_type == 1:
        # 单词之间的attention
        attention = generate_word_attention(pageData.texts)
    elif atention_type == 2:
        # 应该要是句子之间的attention
        attention = generate_sentence_attention(pageData.texts)
        print(attention)
        return HttpResponse(1)
    elif atention_type == 3:
        # 难度
        attention = generate_word_difficulty(pageData.texts)
        print("hard word--")
        for at in attention:
            if at[1] >= 1:
                print(at[0])
        print("hard word end")
    else:
        attention = []
    print("attention")
    print(attention)

    # 3. 获取单词的位置
    word_index = get_word_by_index(pageData.texts)
    print("word_index")
    print(len(word_index))
    word_and_location_dict = get_word_and_location(pageData.location)

    gaze_x = []
    gaze_y = []
    importance_list = [x for x in attention if x[1] > 0]

    importance_list.sort(reverse=True)

    # 数量多为 0.000x，将其最大的数扩大至100
    tmp = 1
    if importance_list:
        while importance_list[0][1] * tmp < 10:
            tmp = tmp * 10
    print("放大倍数是:%d" % tmp)

    for importance in importance_list:
        word_indexes = get_index_by_word(pageData.texts, importance[0].lower())  # list
        for word_index in word_indexes:
            if word_index in word_and_location_dict.keys():
                loc = word_and_location_dict[word_index]
                for i in range(int(importance[1] * tmp)):
                    gaze_x.append(random.randint(int(loc[0]), int(loc[2])))
                    gaze_y.append(random.randint(int(loc[1]), int(loc[3]) - 10))
    # gaze_x = preprocess_data(gaze_x, 7)
    # gaze_y = preprocess_data(gaze_y, 7)
    gaze_x = list(map(int, gaze_x))
    gaze_y = list(map(int, gaze_y))

    # 组合数据
    coordinates = []
    for i in range(len(gaze_x)):
        coordinate = [gaze_x[i], gaze_y[i]]
        coordinates.append(coordinate)

    exp = Experiment.objects.filter(id=pageData.experiment_id)
    if not exp:
        username = "tmp_text"
        base = "static\\background\\" + str(request.GET.get("background")) + ".jpg"
    else:
        username = exp.first().user
        base = (
                "static\\background\\"
                + str(exp.first().article_id)
                + "\\"
                + str(pageData.page)
                + ".jpg"
        )

    hit_pic_name = (
            "static\\data\\heatmap\\"
            + str(username)
            + "\\nlp_"
            + str(atention_type)
            + "\\hit_"
            + str(page_data_id)
            + ".png"
    )
    heatmap_name = (
            "static\\data\\heatmap\\"
            + str(username)
            + "\\nlp_"
            + str(atention_type)
            + "\\heatmap_"
            + str(page_data_id)
            + ".png"
    )
    draw_heat_map(coordinates, hit_pic_name, heatmap_name, base)
    return JsonResponse({"code": 200, "status": "success", "pic_path": heatmap_name})

def takeSecond(elem):
    return elem[1]

def get_heatmap(request):
    """
    根据page_data_id生成visual attention和nlp attention
    :param request: id,window
    :return:
    """
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)
    # 准备工作，获取单词和句子的信息
    word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
    # 获取单词和句子的位置
    word_locations = get_word_location(
        pageData.location
    )  # [(left,top,right,bottom),(left,top,right,bottom)]
    sentence_locations = get_sentence_location(pageData.location, sentence_list)
    # 获取图片生成的路径
    exp = Experiment.objects.filter(id=pageData.experiment_id)

    base_path = (
            "static\\data\\heatmap\\"
            + str(exp.first().user)
            + "\\"
            + str(page_data_id)
            + "\\"
    )
    backgound = (
            "static\\background\\"
            + str(exp.first().article_id)
            + "\\"
            + str(pageData.page)
            + ".jpg"
    )

    top_dict = {}
    # nlp attention

    """word level"""
    nlp_attentions = ["topic_relevant", "word_attention", "word_difficulty"]
    # nlp_attentions = []
    for attention in nlp_attentions:
        data_list = [[]]
        # 获取数据
        if attention == "topic_relevant":
            data_list = get_importance(pageData.texts)  # [('word',1)]
        if attention == "word_attention":
            data_list = generate_word_attention(pageData.texts)
        if attention == "word_difficulty":
            data_list = generate_word_difficulty(pageData.texts)

        data_list.sort(reverse=True,key=takeSecond)

        print(data_list)
        top_dict[attention] = [x[0] for x in data_list]

        image = cv2.imread(backgound)
        color = 255
        print(word_list)
        for data in data_list:
            # 该单词在页面上是否存在
            index_list = []
            for i, word in enumerate(word_list):
                if data[0].lower() == word.lower():
                    if attention == "word_attention":
                        print(word)
                    index_list.append(i)
            for index in index_list:
                loc = word_locations[index]
                cv2.rectangle(image, (int(loc[0]), int(loc[1])), (int(loc[2]), int(loc[3])),(color,0,0), 10)
            color = color - 15
            if color - 5 < 50:
                break

        import matplotlib.pyplot as plt

        heatmap_name = base_path + str(attention) + ".png"
        plt.imshow(image)
        plt.title(heatmap_name)
        plt.show()
        logger.info("heatmap已在该路径下生成:%s" % heatmap_name)


    # """sentence level"""
    # sentence_relation = generate_sentence_attention(pageData.texts.replace("..", ". "))
    # top_k = topk_tuple(sentence_relation, k=2)
    #
    # # top_dict["sentence_relation"] = top_k
    # loc_x = []
    # loc_y = []
    #
    # # 数量多为 0.000x，将其最大的数扩大至100
    # tmp = 1
    # if sentence_relation:
    #     while top_k[0][1] * tmp < 10:
    #         tmp = tmp * 10
    # print("放大倍数是:%d" % tmp)
    #
    # for i, sentence in enumerate(sentence_relation):
    #     for loc in sentence_locations[i]:
    #         for j in range(int(sentence[1] * tmp)):
    #             loc_x.append(random.randint(int(loc[0]), int(loc[2])))
    #             loc_y.append(random.randint(int(loc[1]), int(loc[3])))
    #
    # loc_x = list(map(int, loc_x))
    # loc_y = list(map(int, loc_y))
    #
    # # 组合数据
    # coordinates = []
    # for i in range(len(loc_x)):
    #     coordinate = [loc_x[i], loc_y[i]]
    #     coordinates.append(coordinate)
    #
    # heatmap_name = base_path + "sentence_relation" + ".png"
    # apply_heatmap(backgound, heatmap_name, coordinates)
    '''
>>>>>>> d6715d8 (save work)
    # visual attention
    list_x = list(map(float, pageData.gaze_x.split(",")))
    list_y = list(map(float, pageData.gaze_y.split(",")))
    list_t = list(map(float, pageData.gaze_t.split(",")))

    # 滤波处理
    kernel_size = int(request.GET.get("window", 0))
    if kernel_size != 0:
        # 滤波
        list_x = preprocess_data(list_x, kernel_size)
        list_y = preprocess_data(list_y, kernel_size)

        # 3. 组合数据
        # 滤波完是浮点数，需要转成int
    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))
    # 组合
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i]]
        coordinates.append(coordinate)

    # 生成热力图
    heatmap_name = base_path + "visual.png"
    apply_heatmap(backgound,heatmap_name,coordinates)

    hotspot = draw_heat_map(coordinates,heatmap_name,heatmap_name,backgound)
    # 去计算所有的gaze点
    words_cnt = [0 for x in word_list]
    for coordinate in coordinates:
        word_index = get_word_by_one_gaze(word_locations, coordinate)
        if word_index != -1:
            words_cnt[word_index] += 1

    round = 50
    to_be_deleted = [0 for x in word_list]
    top_k = []
    # while round:
    #     max_cnt = -1
    #     max_index = -1
    #     for i, cnt in enumerate(words_cnt):
    #         if cnt > max_cnt and to_be_deleted[i] == 0 and word_list[i] not in top_k:
    #             max_cnt = cnt
    #             max_index = i
    #     if max_index == -1:
    #         break
    #     to_be_deleted[max_index] = 1
    #     top_k.append(word_list[max_index])
    #     round -= 1
    #
    # top_dict["visual"] = top_k

    # 获取top_k
    # data_1 = [x for x in hotspot]
    df = pd.DataFrame({
        'x': [x[0] for x in hotspot],
        'y': [x[1] for x in hotspot],
        'color': [x[2] for x in hotspot]
    })
    df.sort_values(by=['color'])

    # data_index = get_top_k(data_1, k=5000)
    top_k_hotspot = []  # [(1232, 85, 240), (1233, 85, 240)]

    for ind, row in df.iterrows():
        if ind == 5000:
            break
        top_k_hotspot.append(row)

    # for i, data in enumerate(hotspot):
    #     if i in data_index:
    #         top_k_hotspot.append(data)
    top_k = []
    for item in top_k_hotspot:
        word_index = get_word_by_one_gaze(word_locations, item)
        if word_index != -1:
            top_k.append(word_list[word_index])
    print(top_k)
    new_top_k = list(set(top_k))
    new_top_k.sort(key=top_k.index)

    top_dict['visual'] = new_top_k
    # 生成fixation图示
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i], list_t[i]]
        coordinates.append(coordinate)
    fixations = get_fixations(coordinates)
    fixation_image(
        pageData.image,
        Experiment.objects.get(id=pageData.experiment_id).user,
        fixations,
        pageData.id,
    )
    '''
    k_list = [5, 10, 15, 20, 30]
    top_list = []
    k_dict = {}
    pic_list = []
    for k in k_list:

        attention_type = {}
        for key in top_dict.keys():
            attention_type[key] = top_dict[key][0:k]
        k_dict[k] = attention_type

        pic_dict = {
            "k": k,
            # "similarity": calculate_similarity(attention_type),
            # "identity": calculate_identity(attention_type),
        }
        pic_list.append(pic_dict)

    top_list.append(k_dict)

    # paint_bar_graph(pic_list, "similarity")
    # paint_bar_graph(pic_list, "identity")

    return JsonResponse(top_list, json_dumps_params={"ensure_ascii": False}, safe=False)

def visual_attention(request):
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)
    # 准备工作，获取单词和句子的信息
    word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
    # 获取单词和句子的位置
    word_locations = get_word_location(
        pageData.location
    )  # [(left,top,right,bottom),(left,top,right,bottom)]
    sentence_locations = get_sentence_location(pageData.location, sentence_list)
    # 获取图片生成的路径
    exp = Experiment.objects.filter(id=pageData.experiment_id)

    base_path = (
            "static\\data\\heatmap\\"
            + str(exp.first().user)
            + "\\"
            + str(page_data_id)
            + "\\"
    )
    backgound = (
            "static\\background\\"
            + str(exp.first().article_id)
            + "\\"
            + str(pageData.page)
            + ".jpg"
    )
    top_dict = {}
    # visual attention
    list_x = list(map(float, pageData.gaze_x.split(",")))
    list_y = list(map(float, pageData.gaze_y.split(",")))
    list_t = list(map(float, pageData.gaze_t.split(",")))

    # 滤波处理
    kernel_size = int(request.GET.get("window", 0))
    if kernel_size != 0:
        # 滤波
        list_x = preprocess_data(list_x, kernel_size)
        list_y = preprocess_data(list_y, kernel_size)

        # 3. 组合数据
        # 滤波完是浮点数，需要转成int
    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))
    # 组合
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i]]
        coordinates.append(coordinate)

    # 生成热力图
    heatmap_name = base_path + "visual.png"
    # apply_heatmap(backgound, heatmap_name, coordinates)

    hotspot = draw_heat_map(coordinates, heatmap_name, heatmap_name, backgound)
    # 去计算所有的gaze点
    words_cnt = [0 for x in word_list]
    for coordinate in coordinates:
        word_index = get_word_by_one_gaze(word_locations, coordinate)
        if word_index != -1:
            words_cnt[word_index] += 1

    round = 50
    to_be_deleted = [0 for x in word_list]
    top_k = []
    # while round:
    #     max_cnt = -1
    #     max_index = -1
    #     for i, cnt in enumerate(words_cnt):
    #         if cnt > max_cnt and to_be_deleted[i] == 0 and word_list[i] not in top_k:
    #             max_cnt = cnt
    #             max_index = i
    #     if max_index == -1:
    #         break
    #     to_be_deleted[max_index] = 1
    #     top_k.append(word_list[max_index])
    #     round -= 1
    #
    # top_dict["visual"] = top_k

    # 获取top_k
    # data_1 = [x for x in hotspot]
    df = pd.DataFrame({
        'x': [x[0] for x in hotspot],
        'y': [x[1] for x in hotspot],
        'color': [x[2] for x in hotspot]
    })

    df = df.sort_values(by=['color'])
    # data_index = get_top_k(data_1, k=5000)
    top_k_hotspot = []  # [(1232, 85, 240), (1233, 85, 240)]
    for ind, row in df.iterrows():
        top_k_hotspot.append(row)
    # for i, data in enumerate(hotspot):
    #     if i in data_index:
    #         top_k_hotspot.append(data)
    top_k = []
    for item in top_k_hotspot:
        word_index = get_word_by_one_gaze(word_locations, item)
        if word_index != -1:
            top_k.append(word_list[word_index])

    new_top_k = list(set(top_k))
    new_top_k.sort(key=top_k.index)

    top_dict['visual'] = new_top_k
    # 生成fixation图示
    coordinates = []
    for i in range(len(list_x)):
        coordinate = [list_x[i], list_y[i], list_t[i]]
        coordinates.append(coordinate)
    fixations = get_fixations(coordinates)
    fixation_image(
        pageData.image,
        Experiment.objects.get(id=pageData.experiment_id).user,
        fixations,
        pageData.id,
    )

    k_list = [5, 10, 15, 20, 30]
    top_list = []
    k_dict = {}
    pic_list = []
    for k in k_list:

        attention_type = {}
        for key in top_dict.keys():
            attention_type[key] = top_dict[key][0:k]
        k_dict[k] = attention_type

        pic_dict = {
            "k": k,
            # "similarity": calculate_similarity(attention_type),
            # "identity": calculate_identity(attention_type),
        }
        pic_list.append(pic_dict)

    top_list.append(k_dict)

    # paint_bar_graph(pic_list, "similarity")
    # paint_bar_graph(pic_list, "identity")

    return JsonResponse(top_list, json_dumps_params={"ensure_ascii": False}, safe=False)