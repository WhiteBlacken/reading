import base64
import json
import math
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image

from action.models import Dictionary, Experiment, PageData, Paragraph, Text, Translation
from feature.utils import detect_fixations, keep_row
from onlineReading.utils import get_euclid_distance, translate
from pyheatmap import myHeatmap
from semantic_attention import (  # generate_sentence_difficulty,
    generate_sentence_attention,
    generate_sentence_difficulty,
    generate_word_attention,
    generate_word_difficulty,
    nlp, generate_word_list,
)
from utils import (
    Timer,
    calculate_identity,
    calculate_similarity,
    find_threshold,
    format_gaze,
    generate_pic_by_base64,
    get_importance,
    get_item_index_x_y,
    get_para_from_txt,
    get_word_and_sentence_from_text,
    get_word_by_one_gaze,
    get_word_location,
    join_images_vertical,
    join_two_image,
    paint_bar_graph,
    paint_gaze_on_pic,
    preprocess_data,
    x_y_t_2_coordinate,
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


def choose_text(request):
    return render(request, "chooseTxt.html")


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
    para_dict = {}
    para = 0
    logger.info("--实验开始--")
    name = "读取文章及其翻译"
    with Timer(name):  # 开启计时
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
                    translations = (
                        Translation.objects.filter(article_id=article_id)
                        .filter(para_id=para)
                        .filter(sentence_id=sentence_id)
                    )
                    if translations:
                        sentence_zh = translations.first().txt
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


def takeSecond(elem):
    return elem[1]


def paint_on_word(image, target_words_index, word_locations, title, pic_path, alpha=0.1, color=255):
    blk = np.zeros(image.shape, np.uint8)
    blk[0:image.shape[0] - 1, 0:image.shape[1] - 1] = 255
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
            os.makedirs(path)

    # 创建背景图片
    background = base_path + "background.png"
    # 使用数据库的base64直接生成background
    data = pageData.image.split(",")[1]
    # 将str解码为byte
    image_data = base64.b64decode(data)
    with open(background, "wb") as f:
        f.write(image_data)

    get_visual_attention(
        page_data_id,
        exp.first().user,
        pageData.gaze_x,
        pageData.gaze_y,
        pageData.gaze_t,
        background,
        base_path,
        word_list,
        word_locations,
        top_dict,
    )

    # 将spatial和temporal结合
    save_path = base_path + "spatial_with_temporal_filter.png"
    heatmap_img = base_path + "visual.png"
    fixation_img = base_path + "fixation.png"
    join_two_image(heatmap_img, fixation_img, save_path)

    return HttpResponse(1)


def get_row_level_fixations_map(request):
    print("执行了")
    page_data_ids = request.GET.get("id").split(',')
    print(page_data_ids)
    for page_data_id in page_data_ids:
        pageData = PageData.objects.get(id=page_data_id)
        exp = Experiment.objects.filter(id=pageData.experiment_id)

        begin = request.GET.get("begin", 0)
        end = request.GET.get("end", -1)
        coordinates = format_gaze(pageData.gaze_x, pageData.gaze_y, pageData.gaze_t)[begin:end]
        fixations = detect_fixations(coordinates)

        # 按行切割gaze点
        pre_fixation = fixations[0]
        distance = 600
        row_fixations = []
        tmp = []
        # print("fixations: " + str(fixations))
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
        if len(tmp) != 0:
            row_fixations.append(tmp)
            # row_fixations[len(row_fixations) - 1] = row_fixations[len(row_fixations) - 1] + tmp
        base_path = "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\"

        page_data = PageData.objects.get(id=page_data_id)
        path = "static/data/heatmap/" + str(exp.first().user) + "/" + str(page_data_id) + "/"
        filename = "background.png"

        generate_pic_by_base64(page_data.image, path, filename)

        if not os.path.exists(base_path + "fixation\\"):
            os.makedirs(base_path + "fixation\\")
        for i, fixs in enumerate(row_fixations):
            name = "1"
            for j in range(i):
                name += "1"
            # for i, fix in enumerate(fixs):
            #     if i > 3:
            #         if fixs[i][1] < sum([fixs[j][1] for j in range(i)]) / i - 20:
            #             fixs[i][1] = fixs[i][1] + 8
            #             if fixs[i][1] < sum([fixs[j][1] for j in range(i)]) / i - 30:
            #                 fixs[i][1] = sum([fixs[j][1] for j in range(i)]) / i - 20
            #         elif fixs[i][1] > sum([fixs[j][1] for j in range(i)]) / i + 20:
            #             fixs[i][1] = fixs[i][1] - 8
            #             if fixs[i][1] > sum([fixs[j][1] for j in range(i)]) / i + 30:
            #                 fixs[i][1] = sum([fixs[j][1] for j in range(i)]) / i + 20
            #         if fixs[i - 1][0] > fixs[i][0] > fixs[i - 1][0] - 30:
            #             fixs[i][0] = coordinates[i][0] - 30
            #         elif fixs[i - 1][0] < fixs[i][0] < fixs[i - 1][0] + 30:
            #             fixs[i][0] = fixs[i][0] + 30
            # if i == 1:
            #     paint_gaze_on_pic(fixs, path + filename,
            #                       base_path + str(exp.first().user) + "_sentence_observation_5_" + str(i+1) + ".png")
            paint_gaze_on_pic(fixs, path + filename, path + "/fixation/" + name + ".png")

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
        filters = [
            {"type": "median", "window": 7},
            {"type": "median", "window": 7},
            {"type": "mean", "window": 5},
            {"type": "mean", "window": 5},
        ]
        # 滤波
        list_x = preprocess_data(list_x, filters)
        list_y = preprocess_data(list_y, filters)

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

    fixations = detect_fixations(coordinates)
    # 按行切割gaze点
    pre_fixation = fixations[0]
    distance = 600
    row_fixations = []
    tmp = []
    sum_y = 0
    row_cnt = 1
    for fixation in fixations:
        if get_euclid_distance(fixation[0], pre_fixation[0], fixation[1], pre_fixation[1]) > distance:
            mean_y = int(sum_y / len(tmp))
            # for item in tmp:
            #     if item[1] > mean_y + 20:
            #         item[1] = mean_y + 20
            #     if item[1] < mean_y - 20:
            #         item[1] = mean_y - 20
            for i, fix in enumerate(tmp):
                if i > 3:
                    if tmp[i][1] < sum([tmp[j][1] for j in range(i)]) / i - 20:
                        tmp[i][1] = tmp[i][1] + 8
                        if tmp[i][1] < sum([tmp[j][1] for j in range(i)]) / i - 30:
                            tmp[i][1] = sum([tmp[j][1] for j in range(i)]) / i - 20
                    elif tmp[i][1] > sum([tmp[j][1] for j in range(i)]) / i + 20:
                        tmp[i][1] = tmp[i][1] - 8
                        if tmp[i][1] > sum([tmp[j][1] for j in range(i)]) / i + 30:
                            tmp[i][1] = sum([tmp[j][1] for j in range(i)]) / i + 20
                    if tmp[i - 1][0] > tmp[i][0] > tmp[i - 1][0] - 30:
                        tmp[i][0] = tmp[i][0] - 30
                    elif tmp[i - 1][0] < tmp[i][0] < tmp[i - 1][0] + 30:
                        tmp[i][0] = tmp[i][0] + 30
            # row_fixations.append(tmp)
            # tmp = [fixation]
            if row_cnt == 2 or row_cnt == 4 or row_cnt == 6:
                row_fixations[len(row_fixations) - 1] = row_fixations[len(row_fixations) - 1] + tmp
            else:
                row_fixations.append(tmp)
            tmp = [fixation]
            row_cnt += 1
            sum_y = 0
        else:
            tmp.append(fixation)
            sum_y += fixation[1]
        pre_fixation = fixation
    if len(tmp) > 0:
        # row_fixations.append(tmp)
        row_fixations[len(row_fixations) - 1] = row_fixations[len(row_fixations) - 1] + tmp
    base_path = "static\\data\\heatmap\\" + str(exp.user) + "\\" + str(pageData.id) + "\\"

    if not os.path.exists(base_path + "heat\\"):
        os.mkdir(base_path + "heat\\")
    for i, item in enumerate(row_fixations):
        name = "1"
        for j in range(i):
            name += "1"
        print(len(item))
        dat = [[int(x[0]), int(x[1])] for x in item]
        myHeatmap.draw_heat_map(dat, base_path + "heat\\" + name + ".png", base_path + "background.png")

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
        filters = [
            {"type": "median", "window": 7},
            {"type": "median", "window": 7},
            {"type": "mean", "window": 5},
            {"type": "mean", "window": 5},
        ]
        # 滤波
        list_x = preprocess_data(list_x, filters)
        list_y = preprocess_data(list_y, filters)

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

    fixations = detect_fixations(coordinates)
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

    if len(tmp) != 0:
        row_fixations.append(tmp)

    base_path = "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\"

    if not os.path.exists(base_path + "fixation\\"):
        os.mkdir(base_path + "fixation\\")
    for i, fixs in enumerate(row_fixations):
        name = "1"
        for j in range(i):
            name += "1"

        page_data = PageData.objects.get(id=page_data_id)
        path = "static/data/heatmap/" + str(exp.first().user) + "/" + str(page_data_id) + "/"
        filename = "background.png"

        generate_pic_by_base64(page_data.image, path, filename)
        paint_gaze_on_pic(fixs, path + filename, path + "fixation.png")

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
    page_data_ids = request.GET.get("id").split(',')
    print(page_data_ids)
    for page_data_id in page_data_ids:
        pageData = PageData.objects.get(id=page_data_id)
        # 准备工作，获取单词和句子的信息
        word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
        # 获取单词的位置
        word_locations = get_word_location(pageData.location)  # [(left,top,right,bottom),(left,top,right,bottom)]

        # 确保单词长度是正确的
        assert len(word_locations) == len(word_list)
        # 获取图片生成的路径
        exp = Experiment.objects.filter(id=pageData.experiment_id)

        base_path = "pic\\" + str(page_data_id) + "\\"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        # 创建图片存储的目录
        # 如果目录不存在，则创建目录
        path_levels = [
            "static\\data\\heatmap\\" + str(exp.first().user) + "\\",
            "static\\data\\heatmap\\" + str(exp.first().user) + "\\" + str(page_data_id) + "\\",
        ]
        # for path in path_levels:
        #     print("生成:%s" % path)
        #     if not os.path.exists(path):
        #         os.makedirs(path)

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
        # get_word_level_nlp_attention(
        #     pageData.texts,
        #     page_data_id,
        #     top_dict,
        #     background,
        #     word_list,
        #     word_locations,
        #     exp.first().user,
        #     base_path,
        # )
        """
        sentence level
        """

        top_sentence_dict = {}
        # 生成图片
        # get_sentence_level_nlp_attention(
        #     pageData.texts,
        #     sentence_list,
        #     background,
        #     word_locations,
        #     page_data_id,
        #     top_sentence_dict,
        #     exp.first().user,
        #     base_path,
        # )

        """
        visual attention
        1. 生成2张图片：heatmap + fixation
        2. 填充top_k
        存在问题：生成top和展示图片实际上做了两个heatmap
        """
        # 滤波处理
        # kernel_size = int(request.GET.get("window", 0))
        # get_visual_attention(
        #     page_data_id,
        #     exp.first().user,
        #     pageData.gaze_x,
        #     pageData.gaze_y,
        #     pageData.gaze_t,
        #     background,
        #     base_path,
        #     word_list,
        #     word_locations,
        #     top_dict,
        # )

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
        # k_list = [3, 5, 7, 10, 20]
        #
        # k_dict = {}
        # pic_list = []
        # max_k = len(top_dict["visual"])
        # for k in k_list:
        #     if k > max_k:
        #         break
        #     attention_type = {}
        #     for key in top_dict.keys():
        #         attention_type[key] = top_dict[key][0:k]
        #     k_dict[k] = attention_type
        #
        #     pic_dict = {
        #         "k": k,
        #         "similarity": calculate_similarity(attention_type),
        #         "identity": calculate_identity(attention_type),
        #     }
        #     pic_list.append(pic_dict)
        #
        # paint_bar_graph(pic_list, base_path, "similarity")
        # paint_bar_graph(pic_list, base_path, "identity")
        #
        # result_dict = {"word level": k_dict, "sentence level": top_sentence_dict}
        #
        # result_json = json.dumps(result_dict)
        # path = base_path + "result.txt"
        # with open(path, "w") as json_file:
        #     json_file.write(result_json)

    return JsonResponse("", json_dumps_params={"ensure_ascii": False}, safe=False)


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
        alpha = 0.3  # 设置覆盖图片的透明度
        blk = np.zeros(image.shape, np.uint8)
        # (1080, 1920, 3)
        blk[0: image.shape[0] - 1, 0: image.shape[1] - 1] = 255
        print(image.shape)
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
        heatmap_name = base_path + str(attention) + ".png"
        cv2.imwrite(heatmap_name, image)
        logger.info("heatmap已经生成:%s" % heatmap_name)


def get_topic_relevant(request):
    article_id = request.GET.get('article_id')
    paras = Paragraph.objects.filter(article_id=article_id).order_by("para_id")
    text = ""
    for para in paras:
        text += para.content

    topic_value = get_importance(text)
    word_list, word4show_list = generate_word_list(text)
    topic_value_of_words = []

    print(len(word_list))
    print(len(topic_value))

    for item in topic_value:
        if item[0].lower() == 'a':
            print('yes')

    for word in word_list:
        for item in topic_value:
            if item[0].lower() == word.lower():
                tmp = (word, item[1])
                topic_value_of_words.append(tmp)
                break
    return JsonResponse({'topic value': topic_value_of_words})

def get_diff(request):
    article_id = request.GET.get('article_id')
    paras = Paragraph.objects.filter(article_id=article_id).order_by("para_id")
    text = ""
    for para in paras:
        text += para.content

    topic_value = generate_word_difficulty(text)
    word_list, word4show_list = generate_word_list(text)
    topic_value_of_words = []

    print(len(word_list))
    print(len(topic_value))

    for item in topic_value:
        if item[0].lower() == 'a':
            print('yes')

    for word in word_list:
        for item in topic_value:
            if item[0].lower() == word.lower():
                tmp = (word, item[1])
                topic_value_of_words.append(tmp)
                break
    return JsonResponse({'topic value': topic_value_of_words})

def get_att(request):
    article_id = request.GET.get('article_id')
    paras = Paragraph.objects.filter(article_id=article_id).order_by("para_id")
    text = ""
    for para in paras:
        text += para.content

    topic_value = generate_word_attention(text)
    word_list, word4show_list = generate_word_list(text)
    topic_value_of_words = []

    print(len(word_list))
    print(len(topic_value))

    for item in topic_value:
        if item[0].lower() == 'a':
            print('yes')

    for word in word_list:
        for item in topic_value:
            if item[0].lower() == word.lower():
                tmp = (word, item[1])
                topic_value_of_words.append(tmp)
                break
    print(topic_value_of_words)
    return HttpResponse(topic_value_of_words)

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
            sentence_attention = generate_sentence_attention(
                texts.replace("..", ".").replace(".", ". ")
            )  # [('xx',数值),('xx',数值)]
        if attention == "sentence_difficulty":
            sentence_attention = generate_sentence_difficulty(
                texts.replace("..", ". ").replace(".", ". ")
            )  # [('xx',数值),('xx',数值)]
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
        alpha = 0.3  # 设置覆盖图片的透明度
        blk = np.zeros(image.shape, np.uint8)
        blk[0: image.shape[0] - 1, 0: image.shape[1] - 1] = 255

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
        heatmap_name = base_path + attention + ".png"
        cv2.imwrite(heatmap_name, image)
        logger.info("heatmap已经生成:%s" % heatmap_name)

        # 生成top_k
        sentence_attention.sort(reverse=True, key=takeSecond)
        top_sentence_dict[attention] = [x[0] for x in sentence_attention]


def get_visual_attention(
        page_data_id,
        username,
        gaze_x,
        gaze_y,
        gaze_t,
        background,
        base_path,
        word_list,
        word_locations,
        top_dict,
):
    logger.info("visual attention正在分析....")

    """
    1. 清洗数据
    """
    coordinates = format_gaze(gaze_x, gaze_y, gaze_t)

    """
    2. 计算top_k
    """
    path = "static/data/heatmap/" + str(username) + "/" + str(page_data_id) + "/"
    filename = "background.png"

    data_list = [[x[0], x[1]] for x in coordinates]
    hotspot = myHeatmap.draw_heat_map(data_list, base_path + "visual.png", background)

    list2 = get_top_word_of_visual(hotspot, word_locations, path + filename, word_list)
    top_dict["visual"] = list2

    # 生成fixation图示
    fixations = detect_fixations(coordinates)
    print(len(fixations))
    page_data = PageData.objects.get(id=page_data_id)

    generate_pic_by_base64(page_data.image, path, filename)

    paint_gaze_on_pic(fixations, path + filename, path + "fixation.png")
    paint_gaze_on_pic(fixations, path + "visual.png", path + "fix_heat.png")


def get_top_word_of_visual(hotspot, word_locations, background, word_list):
    top_dict = {}
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
    return list2


def get_para_by_word_index(pre_fixation, para_list):
    for i, para in enumerate(para_list):
        if para[1] >= pre_fixation >= para[0]:
            return i
    return -1


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
    path = "jupyter\\dataset\\" + "article-230111.csv"
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
                get_euclid_distance(coordinates[begin][0], coordinates[end][0], coordinates[begin][1],
                                    coordinates[end][1])
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
                get_euclid_distance(coordinates[begin][0], coordinates[end][0], coordinates[begin][1],
                                    coordinates[end][1])
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


def get_t(gaze_t, remove=500):
    # 超参
    times_for_remove = remove  # 删除开头结尾的长度，单位是ms
    # 转换格式
    list_t = list(map(float, gaze_t.split(",")))

    new_list_t = []
    # 组合+舍去开头结尾
    for i in range(len(list_t)):
        if list_t[-1] - list_t[i] < times_for_remove:
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

    interval = 2 * 1000

    timestamp = 0
    timestamps = []  # 每页的time范围
    for page_data in page_data_ls:
        gaze_t = get_t(page_data.gaze_t)
        begin = 0
        for i, t in enumerate(gaze_t):
            if i == 0:
                continue
            if gaze_t[i] - gaze_t[begin] > interval:
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
    csv = "jupyter\\dataset\\all-gaze-time.csv"
    file = pd.read_csv(csv)
    for i, r in file.iterrows():
        if str(r["experiment_id"]) == exp_id and str(r["time"]) == time:
            print(type(r["experiment_id"]))
            print(type(exp_id))
            row = i
            break
    if row == -1:
        return JsonResponse({"code": 404, "status": "未检索相关信息"}, json_dumps_params={"ensure_ascii": False},
                            safe=False)

    print(file["gaze_x"][row])
    print(type(file["gaze_x"][row]))

    list_x = json.loads(file["gaze_x"][row])
    list_y = json.loads(file["gaze_y"][row])
    list_t = json.loads(file["gaze_t"][row])

    print(list_x)
    print(type(list_x))
    # 时序滤波
    if filter:
        filters = [
            {"type": "median", "window": 7},
            {"type": "median", "window": 7},
            {"type": "mean", "window": 5},
            {"type": "mean", "window": 5},
        ]
        list_x = preprocess_data(list_x, filters)
        list_y = preprocess_data(list_y, filters)

    list_x = list(map(int, list_x))
    list_y = list(map(int, list_y))
    list_t = list(map(int, list_t))
    assert len(list_x) == len(list_y) == len(list_t)
    gaze_points = [[list_x[i], list_y[i], list_t[i]] for i in range(len(list_x))]

    print(detect_fixations(gaze_points))
    fixations = keep_row(detect_fixations(gaze_points), kernel_size=3)
    print(fixations)
    # 3. 画图
    img = cv2.imread(filename)
    cnt = 0
    pre_fix = (0, 0)
    for i, fix in enumerate(fixations):
        fix[0] = int(fix[0])
        fix[1] = int(fix[1])
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
    for i, fix in enumerate(gaze_points):
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
    page_id_ls = page_ids.split(",")
    dict = {}
    path = "static\\data\\other\\paraLoc.txt"
    for page_id in page_id_ls:
        if len(page_id) > 0:
            time1 = 0
            time2 = 0
            page_data = PageData.objects.get(id=page_id)

            aticle_para_1 = get_para_from_txt(path, page_data.page - 1)
            article_id = Experiment.objects.get(id=page_data.experiment_id).article_id
            first_para_index = aticle_para_1[article_id]["para_1"]
            coors = x_y_t_2_coordinate(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)

            for coor in coors:
                word_index = get_item_index_x_y(page_data.location, coor[0], coor[1])
                if word_index != -1:
                    if word_index <= first_para_index:
                        time1 += 30
                    else:
                        time2 += 30

            para_1_speed = (first_para_index + 1) / time1 * 60000

            para_above_1_speed = (aticle_para_1[article_id]["word_num"] - first_para_index) / time2 * 60000

            sen_cnt = 0
            print("fist_para")
            print(first_para_index)
            for sentence in sentence_list:
                print(sentence)
                if sentence[2] - 1 <= first_para_index:
                    sen_cnt += 1

            wordLabels = json.loads(page_data.wordLabels)
            senLabels = json.loads(page_data.sentenceLabels)
            wanderLabels = json.loads(page_data.wanderLabels)

            wordLabel_num = np.sum(np.array(wordLabels) <= first_para_index)
            sens = [x[1] for x in senLabels]
            senLabel_num = np.sum(np.array(sens) <= first_para_index + 1)
            wanders = [x[1] for x in wanderLabels]
            wanderLabel_num = np.sum(np.array(wanders) <= first_para_index)

            dict[page_id] = {
                "para_1_speed": para_1_speed,
                "para_above_1_speed": para_above_1_speed,
                "para1_word_label": wordLabel_num / first_para_index,
                "para_1_sen_label": senLabel_num / sen_cnt,
                "para_1_mw_label": wanderLabel_num / sen_cnt,
            }
    return JsonResponse(dict, json_dumps_params={"ensure_ascii": False}, safe=False)


def check_article(request):
    exp_id = request.GET.get('exp_id')
    pageDatas = PageData.objects.filter(experiment_id=exp_id)
    print(len(pageDatas))
    for pageData in pageDatas:
        word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
        print(pageData.location)
        if len(pageData.location) >0 and pageData.location != 'undefined':
            # 获取单词的位置
            word_locations = get_word_location(pageData.location)  # [(left,top,right,bottom),(left,top,right,bottom)]

            # 确保单词长度是正确的
            print(len(word_locations))
            print(len(word_list))
            assert len(word_locations) == len(word_list)
    return HttpResponse('测试成功')