import base64
import json
import math
import os
import pickle
import shutil

import cv2

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger
from PIL import Image

from action.models import Dictionary, Experiment, PageData, Paragraph, Text, Translation
from feature.utils import detect_fixations, keep_row, textarea
from onlineReading.utils import get_euclid_distance, translate
from pyheatmap import myHeatmap
from semantic_attention import (  # generate_sentence_difficulty,
    generate_sentence_attention,
    generate_sentence_difficulty,
    generate_word_attention,
    generate_word_difficulty,
    generate_word_list,
)

from textstat import textstat

from transformers import XLNetTokenizerFast, XLNetModel
import numpy as np
import spacy
import torch

from nltk.tokenize import sent_tokenize
import pandas as pd
import json
import matplotlib.pyplot as plt
from pke.unsupervised import YAKE

nlp = spacy.load("en_core_web_lg")

xlnet_tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased', output_attentions=True)

kw_extractor = YAKE()

word_fam_map = {}
with open('mrc2.dct', 'r') as fp:
    i = 0
    for line in fp:
        line = line.strip()

        word, phon, dphon, stress = line[51:].split('|')

        w = {
            'wid': i,
            'nlet': int(line[0:2]),
            'nphon': int(line[2:4]),
            'nsyl': int(line[4]),
            'kf_freq': int(line[5:10]),
            'kf_ncats': int(line[10:12]),
            'kf_nsamp': int(line[12:15]),
            'tl_freq': int(line[15:21]),
            'brown_freq': int(line[21:25]),
            'fam': int(line[25:28]),
            'conc': int(line[28:31]),
            'imag': int(line[31:34]),
            'meanc': int(line[34:37]),
            'meanp': int(line[37:40]),
            'aoa': int(line[40:43]),
            'tq2': line[43],
            'wtype': line[44],
            'pdwtype': line[45],
            'alphasyl': line[46],
            'status': line[47],
            'var': line[48],
            'cap': line[49],
            'irreg': line[50],
            'word': word,
            'phon': phon,
            'dphon': dphon,
            'stress': stress
        }
        if word not in word_fam_map:
            word_fam_map[word] = w['fam']
        word_fam_map[word] = max(word_fam_map[word], w['fam'])
        i += 1

from utils import (
    Timer,
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
    paint_gaze_on_pic,
    preprocess_data,
    x_y_t_2_coordinate, process_fixations, get_row,
)

with open('model/wordSVM.pickle', 'rb') as f:
    wordSVM = pickle.load(f)

with open('model/sentSVM.pickle', 'rb') as f:
    sentSVM = pickle.load(f)


def login_page(request):
    return render(request, "login.html")


def login(request):
    username = request.POST.get("username", None)
    device = request.POST.get("device", None)
    print("device:" + device)
    print("username:%s" % username)
    if username:
        request.session["username"] = username
    request.session["device"] = device
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

    article_id = request.GET.get("article_id", 1)

    request.session['article_id'] = article_id

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
    experiment = Experiment.objects.create(article_id=article_id, user=request.session.get("username"),
                                           device=request.session.get("device"))
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
    print(target_words_index)
    print(len(word_locations))
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
    # plt.show()
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
        speed = 0
        if time != 0:
            speed = (
                    get_euclid_distance(coordinates[begin][0], coordinates[end][0], coordinates[begin][1],
                                        coordinates[end][1])
                    / time
            )
        else:
            speed = 0

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
        if time != 0:
            acc = (coordinates[end][3] - coordinates[begin][3]) / time
        else:
            acc = 0
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

    # 1. 准备gaze点

    csv = "jupyter\\paint-230112.csv"
    row = -1
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

    # 准备gaze
    list_x = json.loads(file["gaze_x"][row])
    list_y = json.loads(file["gaze_y"][row])
    list_t = json.loads(file["gaze_t"][row])

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

    # 准备fix
    fix_x = json.loads(file["fix_x"][row])
    fix_y = json.loads(file["fix_y"][row])

    assert len(fix_x) == len(fix_y)
    fixations = [[fix_x[i], fix_y[i]] for i in range(len(fix_x))]

    # 2. 准备底图
    page = file["page"][row]
    page_data = PageData.objects.filter(experiment_id=exp_id).filter(page=page + 1).first()
    image = page_data.image.split(",")[1]
    image_data = base64.b64decode(image)

    filename = "static\\visual\\origin.png"

    with open(filename, "wb") as f:
        f.write(image_data)
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
            0.6,
            (255, 0, 0),
            1,
        )
        cnt = cnt + 1
        pre_fix = fix
    result_filename = "static\\visual\\" + str(exp_id) + "-" + str(time) + "-fix.png"
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
    result_filename = "static\\visual\\" + str(exp_id) + "-" + str(time) + "-gaze.png"
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
        if len(pageData.location) > 0 and pageData.location != 'undefined':
            # 获取单词的位置
            word_locations = get_word_location(pageData.location)  # [(left,top,right,bottom),(left,top,right,bottom)]

            # 确保单词长度是正确的
            print(len(word_locations))
            print(len(word_list))
            assert len(word_locations) == len(word_list)
    return HttpResponse('测试成功')


def get_word_feature(wordFeature, result_fixations, location):
    pre_fixation = -1
    for fixation in result_fixations:
        index, flag = get_item_index_x_y(location, fixation[0],
                                         fixation[1])
        if index != -1:
            wordFeature.number_of_fixations[index] += 1
            wordFeature.fixation_duration[index] += fixation[2]
            if index != pre_fixation:
                wordFeature.reading_times[pre_fixation] += 1
                pre_fixation = index
    return wordFeature


def get_sent_feature(sentFeature, result_fixations, location, sent_list, rows):
    pre_word_index = -1

    for i, fixation in enumerate(result_fixations):
        index, flag = get_item_index_x_y(location, fixation[0],
                                         fixation[1])

        pre_row = get_row(pre_word_index, rows)
        now_row = get_row(index, rows)

        if index != -1:
            sent_index = get_sentence_by_word(index, sent_list)
            if sent_index != 0:
                sentFeature.total_dwell_time_of_sentence[sent_index] += fixation[2]
                sentFeature.saccade_times_of_sentence[sent_index] += 1
                if i != 0:
                    sentFeature.saccade_duration[sent_index] += result_fixations[i][3] - \
                                                                result_fixations[i - 1][4]
                    sentFeature.saccade_distance[sent_index] += get_euclid_distance(result_fixations[i][0],
                                                                                    result_fixations[i - 1][
                                                                                        0],
                                                                                    result_fixations[i][1],
                                                                                    result_fixations[i - 1][1])
                if index > pre_word_index:
                    sentFeature.forward_times_of_sentence[sent_index] += 1
                else:
                    sentFeature.backward_times_of_sentence[sent_index] += 1
                if pre_row == now_row:
                    sentFeature.horizontal_saccade[sent_index] += 1

            pre_word_index = index
    return sentFeature


def get_pred(request):
    """
    输入应该是
        * 当前2s内的gaze点（前端传输）
        * 历史的所有gaze点（后端存储--存储在哪？）
        * 该页的位置信息（后端存储--存储在哪？）
    """
    word_threshold = 0.2
    sent_threshold = 0.15

    x = request.POST.get("x")
    y = request.POST.get("y")
    t = request.POST.get("t")

    history_x = request.session.get('history_x', None)
    history_y = request.session.get('history_y', None)
    history_t = request.session.get('history_t', None)

    if len(x) > 0:
        if history_x is None:
            request.session['history_x'] = x
            request.session['history_y'] = y
            request.session['history_t'] = t
        else:
            request.session['history_x'] += "," + x
            request.session['history_y'] += "," + y
            request.session['history_t'] += "," + t

    print(f"x:{x}")
    print(f"y:{y}")

    print(f'history_x:{request.session["history_x"]}')
    page_text = request.session['page_text']
    word_list, sentence_list = get_word_and_sentence_from_text(page_text)
    location = request.session['location']

    wordFeature = WordFeature(len(word_list), word_list, sentence_list, request.session['semantic_feature'])
    sentFeature = SentFeature(len(sentence_list), sentence_list, word_list)

    border, rows, danger_zone, len_per_word = textarea(location)

    word_predicts = [0 for _ in range(len(word_list))]
    sent_predicts = [0 for _ in range(len(sentence_list))]
    # TODO 为了减少计算量，仅在当前的单词上计算特征
    if history_x and history_y and history_t:
        gaze_points = format_gaze(request.session['history_x'], request.session['history_y'],
                                  request.session['history_t'], begin_time=30, end_time=30)


        print(f'gaze_points:{gaze_points}')
        result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
            gaze_points, request.session['page_text'], location
        )
        print(f'fix:{result_fixations}')

        wordFeature = get_word_feature(wordFeature, result_fixations, location)
        wordFeature.update()
        word_feature = wordFeature.to_dataframe()
        word_predicts = wordSVM.predict_proba(word_feature)[:, 1]

        print(wordFeature.fixation_duration)
        print(f'word_predicts:{word_predicts}')

        sentFeature = get_sent_feature(sentFeature, result_fixations, location, sentence_list, rows)
        sentFeature.update()
        sentFeature = sentFeature.to_dataframe()

        sent_predicts = sentSVM.predict_proba(sentFeature)[:, 1]

        print(f'sent_predicts:{sent_predicts}')

    word_watching_list = []
    sent_watching_list = []
    if len(x) > 0:
        gaze_points = format_gaze(x, y,
                                  t, begin_time=30, end_time=30)
        result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
            gaze_points, request.session['page_text'], request.session['location']
        )
        # 单词fixation最多的句子，为需要判断的句子
        sent_fix = [0 for _ in range(len(sentence_list))]
        for fixation in result_fixations:
            index, flag = get_item_index_x_y(location, fixation[0],
                                             fixation[1])
            if index != -1:
                word_watching_list.append(index)
                sent_index = get_sentence_by_word(index, sentence_list)
                if sent_index != 0:
                    sent_fix[sent_index] += 1
        max_fix_sent = 0
        max_index_sent = 0
        for i, sent in enumerate(sent_fix):
            if sent > max_fix_sent:
                max_fix_sent = sent
                max_index_sent = i
        sent_watching_list.append(max_index_sent)

    word_not_understand_list = []
    sent_not_understand_list = []
    sent_mind_wandering_list = []

    for watching in word_watching_list:
        if word_predicts[watching] > word_threshold:
            word_not_understand_list.append(watching)

    for watching in sent_watching_list:
        if sent_predicts[watching] > sent_threshold:
            sent = sentence_list[watching]
            sent_not_understand_list.append([sent[1], sent[2] - 1])
    print(f'word_not_understand_list:{word_not_understand_list}')

    context = {
        "word": word_not_understand_list,
        "sentence": sent_not_understand_list,
        "wander": sent_mind_wandering_list
    }

    # 将系统的干预记下来，用于pilot study的分析
    return JsonResponse(context)


def get_page_info(request):
    page_text = request.POST.get("page_text")
    location = request.POST.get("location")
    print(f'page_text:{page_text}')
    print(f'location:{location}')

    request.session['page_text'] = page_text
    request.session['location'] = location

    readingArticle = ReadingArticle(page_text)
    transformer_feature = readingArticle.get_transformer_features()

    # print(f'feature:{readingArticle.get_original_features()}')

    request.session['semantic_feature'] = readingArticle.get_original_features()
    print(f"feature:{request.session['semantic_feature']}")
    logger.info("该页信息已加载")
    return HttpResponse(1)


# ['syllable', 'length', 'fam', 'ent_flag', 'topic_score', 'fixation_duration', 'number_of_fixations', 'reading_times', 'fixation_duration_diff', 'number_of_fixations_diff',
#  'reading_times_diff', 'fixation_duration_mean', 'fixation_duration_var', 'number_of_fixations_mean', 'number_of_fixations_var', 'reading_times_mean', 'reading_times_var',
#  'fixation_duration_div_syllable', 'fixation_duration_div_length',
#
#  'trf_62', 'trf_63', 'trf_64', 'trf_65', 'trf_66', 'trf_67',
#  'trf_68', 'trf_69', 'trf_70', 'trf_71', 'trf_72', 'trf_73', 'trf_74', 'trf_75', 'trf_76', 'trf_77', 'trf_78', 'trf_79', 'trf_80', 'trf_81', 'trf_82', 'trf_83', 'trf_84', 'trf_85', 'trf_86', 'trf_87', 'trf_88', 'trf_89', 'trf_90', 'trf_91', 'trf_92', 'trf_93', 'trf_94', 'trf_95', 'trf_96', 'trf_97', 'trf_98', 'trf_99', 'trf_100', 'trf_101', 'trf_102', 'trf_103', 'trf_104', 'trf_105', 'trf_106', 'trf_107', 'trf_108', 'trf_109', 'trf_110', 'trf_111', 'trf_112', 'trf_113', 'trf_114', 'trf_115', 'trf_116', 'trf_117', 'trf_118', 'trf_119', 'trf_120', 'trf_121', 'trf_122', 'trf_123', 'trf_124', 'trf_125', 'trf_126', 'trf_127', 'trf_128', 'trf_129', 'trf_130', 'trf_131', 'trf_132', 'trf_133', 'trf_134', 'trf_135', 'trf_136', 'trf_137', 'trf_138', 'trf_139', 'trf_140', 'trf_141', 'trf_142', 'trf_143', 'trf_144', 'trf_145', 'trf_146', 'trf_147', 'trf_148', 'trf_149', 'trf_150', 'trf_151', 'trf_152', 'trf_153', 'trf_154', 'trf_155', 'trf_156', 'trf_157', 'trf_158', 'trf_159', 'trf_160', 'trf_161', 'trf_162', 'trf_163', 'trf_164', 'trf_165', 'trf_166', 'trf_167', 'trf_168', 'trf_169', 'trf_170', 'trf_171', 'trf_172', 'trf_173', 'trf_174', 'trf_175', 'trf_176', 'trf_177', 'trf_178', 'trf_179', 'trf_180', 'trf_181', 'trf_182', 'trf_183', 'trf_184', 'trf_185', 'trf_186', 'trf_187', 'trf_188', 'trf_189', 'trf_190', 'trf_191', 'trf_192', 'trf_193', 'trf_194', 'trf_195', 'trf_196', 'trf_197', 'trf_198', 'trf_199', 'trf_200', 'trf_201', 'trf_202', 'trf_203', 'trf_204', 'trf_205', 'trf_206', 'trf_207', 'trf_208', 'trf_209', 'trf_210', 'trf_211', 'trf_212', 'trf_213', 'trf_214', 'trf_215', 'trf_216', 'trf_217', 'trf_218', 'trf_219', 'trf_220', 'trf_221', 'trf_222', 'trf_223', 'trf_224', 'trf_225', 'trf_226', 'trf_227', 'trf_228', 'trf_229', 'trf_230', 'trf_231', 'trf_232', 'trf_233', 'trf_234', 'trf_235', 'trf_236', 'trf_237', 'trf_238', 'trf_239', 'trf_240', 'trf_241', 'trf_242', 'trf_243', 'trf_244', 'trf_245', 'trf_246', 'trf_247', 'trf_248', 'trf_249', 'trf_250', 'trf_251', 'trf_252', 'trf_253', 'trf_254', 'trf_255', 'trf_256', 'trf_257', 'trf_258', 'trf_259', 'trf_260', 'trf_261', 'trf_262', 'trf_263', 'trf_264', 'trf_265', 'trf_266', 'trf_267', 'trf_268', 'trf_269', 'trf_270', 'trf_271', 'trf_272', 'trf_273', 'trf_274', 'trf_275', 'trf_276', 'trf_277', 'trf_278', 'trf_279', 'trf_280', 'trf_281', 'trf_282', 'trf_283', 'trf_284', 'trf_285', 'trf_286', 'trf_287', 'trf_288', 'trf_289', 'trf_290', 'trf_291', 'trf_292', 'trf_293', 'trf_294', 'trf_295', 'trf_296', 'trf_297', 'trf_298', 'trf_299', 'trf_300', 'trf_301', 'trf_302', 'trf_303', 'trf_304', 'trf_305', 'trf_306', 'trf_307', 'trf_308', 'trf_309', 'trf_310', 'trf_311', 'trf_312', 'trf_313', 'trf_314', 'trf_315', 'trf_316', 'trf_317', 'trf_318', 'trf_319', 'trf_320', 'trf_321', 'trf_322', 'trf_323', 'trf_324', 'trf_325', 'trf_326', 'trf_327', 'trf_328', 'trf_329', 'trf_330', 'trf_331', 'trf_332', 'trf_333', 'trf_334', 'trf_335', 'trf_336', 'trf_337', 'trf_338', 'trf_339', 'trf_340', 'trf_341', 'trf_342', 'trf_343', 'trf_344', 'trf_345', 'trf_346', 'trf_347', 'trf_348', 'trf_349', 'trf_350', 'trf_351', 'trf_352', 'trf_353', 'trf_354', 'trf_355', 'trf_356', 'trf_357', 'trf_358', 'trf_359', 'trf_360', 'trf_361', 'trf_362', 'trf_363', 'trf_364', 'trf_365', 'trf_366', 'trf_367', 'trf_368', 'trf_369', 'trf_370', 'trf_371', 'trf_372', 'trf_373', 'trf_374', 'trf_375', 'trf_376', 'trf_377', 'trf_378', 'trf_379', 'trf_380', 'trf_381', 'trf_382', 'trf_383', 'trf_384', 'trf_385', 'trf_386', 'trf_387', 'trf_388', 'trf_389', 'trf_390', 'trf_391', 'trf_392', 'trf_393', 'trf_394', 'trf_395', 'trf_396', 'trf_397', 'trf_398', 'trf_399', 'trf_400', 'trf_401', 'trf_402', 'trf_403', 'trf_404', 'trf_405', 'trf_406', 'trf_407', 'trf_408', 'trf_409', 'trf_410', 'trf_411', 'trf_412', 'trf_413', 'trf_414', 'trf_415', 'trf_416', 'trf_417', 'trf_418', 'trf_419', 'trf_420', 'trf_421', 'trf_422', 'trf_423', 'trf_424', 'trf_425', 'trf_426', 'trf_427', 'trf_428', 'trf_429', 'trf_430', 'trf_431', 'trf_432', 'trf_433', 'trf_434', 'trf_435', 'trf_436', 'trf_437', 'trf_438', 'trf_439', 'trf_440', 'trf_441', 'trf_442', 'trf_443', 'trf_444', 'trf_445', 'trf_446', 'trf_447', 'trf_448', 'trf_449', 'trf_450', 'trf_451', 'trf_452', 'trf_453', 'trf_454', 'trf_455', 'trf_456', 'trf_457', 'trf_458', 'trf_459', 'trf_460', 'trf_461', 'trf_462', 'trf_463', 'trf_464', 'trf_465', 'trf_466', 'trf_467', 'trf_468', 'trf_469', 'trf_470', 'trf_471', 'trf_472', 'trf_473', 'trf_474', 'trf_475', 'trf_476', 'trf_477', 'trf_478', 'trf_479', 'trf_480', 'trf_481', 'trf_482', 'trf_483', 'trf_484', 'trf_485', 'trf_486', 'trf_487', 'trf_488', 'trf_489', 'trf_490', 'trf_491', 'trf_492', 'trf_493', 'trf_494', 'trf_495', 'trf_496', 'trf_497', 'trf_498', 'trf_499', 'trf_500', 'trf_501', 'trf_502', 'trf_503', 'trf_504', 'trf_505', 'trf_506', 'trf_507', 'trf_508', 'trf_509', 'trf_510', 'trf_511', 'trf_512', 'trf_513', 'trf_514', 'trf_515', 'trf_516', 'trf_517', 'trf_518', 'trf_519', 'trf_520', 'trf_521', 'trf_522', 'trf_523', 'trf_524', 'trf_525', 'trf_526', 'trf_527', 'trf_528', 'trf_529', 'trf_530', 'trf_531', 'trf_532', 'trf_533', 'trf_534', 'trf_535', 'trf_536', 'trf_537', 'trf_538', 'trf_539', 'trf_540', 'trf_541', 'trf_542', 'trf_543', 'trf_544', 'trf_545', 'trf_546', 'trf_547', 'trf_548', 'trf_549', 'trf_550', 'trf_551', 'trf_552', 'trf_553', 'trf_554', 'trf_555', 'trf_556', 'trf_557', 'trf_558', 'trf_559', 'trf_560', 'trf_561', 'trf_562', 'trf_563', 'trf_564', 'trf_565', 'trf_566', 'trf_567', 'trf_568', 'trf_569', 'trf_570', 'trf_571', 'trf_572', 'trf_573', 'trf_574', 'trf_575', 'trf_576', 'trf_577', 'trf_578', 'trf_579', 'trf_580', 'trf_581', 'trf_582', 'trf_583', 'trf_584', 'trf_585', 'trf_586', 'trf_587', 'trf_588', 'trf_589', 'trf_590', 'trf_591', 'trf_592', 'trf_593', 'trf_594', 'trf_595', 'trf_596', 'trf_597', 'trf_598', 'trf_599', 'trf_600', 'trf_601', 'trf_602', 'trf_603', 'trf_604', 'trf_605', 'trf_606', 'trf_607', 'trf_608', 'trf_609', 'trf_610', 'trf_611', 'trf_612', 'trf_613', 'trf_614', 'trf_615', 'trf_616', 'trf_617', 'trf_618', 'trf_619', 'trf_620', 'trf_621', 'trf_622', 'trf_623', 'trf_624', 'trf_625', 'trf_626', 'trf_627', 'trf_628', 'trf_629', 'trf_630', 'trf_631', 'trf_632', 'trf_633', 'trf_634', 'trf_635', 'trf_636', 'trf_637', 'trf_638', 'trf_639', 'trf_640', 'trf_641', 'trf_642', 'trf_643', 'trf_644', 'trf_645', 'trf_646', 'trf_647', 'trf_648', 'trf_649', 'trf_650', 'trf_651', 'trf_652', 'trf_653', 'trf_654', 'trf_655', 'trf_656', 'trf_657', 'trf_658', 'trf_659', 'trf_660', 'trf_661', 'trf_662', 'trf_663', 'trf_664', 'trf_665', 'trf_666', 'trf_667', 'trf_668', 'trf_669', 'trf_670', 'trf_671', 'trf_672', 'trf_673', 'trf_674', 'trf_675', 'trf_676', 'trf_677', 'trf_678', 'trf_679', 'trf_680', 'trf_681', 'trf_682', 'trf_683', 'trf_684', 'trf_685', 'trf_686', 'trf_687', 'trf_688', 'trf_689', 'trf_690', 'trf_691', 'trf_692', 'trf_693', 'trf_694', 'trf_695', 'trf_696', 'trf_697', 'trf_698', 'trf_699', 'trf_700', 'trf_701', 'trf_702', 'trf_703', 'trf_704', 'trf_705', 'trf_706', 'trf_707', 'trf_708', 'trf_709', 'trf_710', 'trf_711', 'trf_712', 'trf_713', 'trf_714', 'trf_715', 'trf_716', 'trf_717', 'trf_718', 'trf_719', 'trf_720', 'trf_721', 'trf_722', 'trf_723', 'trf_724', 'trf_725', 'trf_726', 'trf_727', 'trf_728', 'trf_729', 'trf_730', 'trf_731', 'trf_732', 'trf_733', 'trf_734', 'trf_735', 'trf_736', 'trf_737', 'trf_738', 'trf_739', 'trf_740', 'trf_741', 'trf_742', 'trf_743', 'trf_744', 'trf_745', 'trf_746', 'trf_747', 'trf_748', 'trf_749', 'trf_750', 'trf_751', 'trf_752', 'trf_753', 'trf_754', 'trf_755', 'trf_756', 'trf_757', 'trf_758', 'trf_759', 'trf_760', 'trf_761', 'trf_762', 'trf_763', 'trf_764', 'trf_765', 'trf_766', 'trf_767', 'context_syllable', 'context_length', 'context_fam', 'context_ent_flag', 'context_topic_score', 'context_fixation_duration', 'context_number_of_fixations', 'context_reading_times', 'context_fixation_duration_diff', 'context_number_of_fixations_diff', 'context_reading_times_diff', 'context_fixation_duration_mean', 'context_fixation_duration_var', 'context_number_of_fixations_mean', 'context_number_of_fixations_var', 'context_reading_times_mean', 'context_reading_times_var', 'context_fixation_duration_div_syllable', 'context_fixation_duration_div_length', 'context_trf_0', 'context_trf_1', 'context_trf_2', 'context_trf_3', 'context_trf_4', 'context_trf_5', 'context_trf_6', 'context_trf_7', 'context_trf_8', 'context_trf_9', 'context_trf_10', 'context_trf_11', 'context_trf_12', 'context_trf_13', 'context_trf_14', 'context_trf_15', 'context_trf_16', 'context_trf_17', 'context_trf_18', 'context_trf_19', 'context_trf_20', 'context_trf_21', 'context_trf_22', 'context_trf_23', 'context_trf_24', 'context_trf_25', 'context_trf_26', 'context_trf_27', 'context_trf_28', 'context_trf_29', 'context_trf_30', 'context_trf_31', 'context_trf_32', 'context_trf_33', 'context_trf_34', 'context_trf_35', 'context_trf_36', 'context_trf_37', 'context_trf_38', 'context_trf_39', 'context_trf_40', 'context_trf_41', 'context_trf_42', 'context_trf_43', 'context_trf_44', 'context_trf_45', 'context_trf_46', 'context_trf_47', 'context_trf_48', 'context_trf_49', 'context_trf_50', 'context_trf_51', 'context_trf_52', 'context_trf_53', 'context_trf_54', 'context_trf_55', 'context_trf_56', 'context_trf_57', 'context_trf_58', 'context_trf_59', 'context_trf_60', 'context_trf_61', 'context_trf_62', 'context_trf_63', 'context_trf_64', 'context_trf_65', 'context_trf_66', 'context_trf_67', 'context_trf_68', 'context_trf_69', 'context_trf_70', 'context_trf_71', 'context_trf_72', 'context_trf_73', 'context_trf_74', 'context_trf_75', 'context_trf_76', 'context_trf_77', 'context_trf_78', 'context_trf_79', 'context_trf_80', 'context_trf_81', 'context_trf_82', 'context_trf_83', 'context_trf_84', 'context_trf_85', 'context_trf_86', 'context_trf_87', 'context_trf_88', 'context_trf_89', 'context_trf_90', 'context_trf_91', 'context_trf_92', 'context_trf_93', 'context_trf_94', 'context_trf_95', 'context_trf_96', 'context_trf_97', 'context_trf_98', 'context_trf_99', 'context_trf_100', 'context_trf_101', 'context_trf_102', 'context_trf_103', 'context_trf_104', 'context_trf_105', 'context_trf_106', 'context_trf_107', 'context_trf_108', 'context_trf_109', 'context_trf_110', 'context_trf_111', 'context_trf_112', 'context_trf_113', 'context_trf_114', 'context_trf_115', 'context_trf_116', 'context_trf_117', 'context_trf_118', 'context_trf_119', 'context_trf_120', 'context_trf_121', 'context_trf_122', 'context_trf_123', 'context_trf_124', 'context_trf_125', 'context_trf_126', 'context_trf_127', 'context_trf_128', 'context_trf_129', 'context_trf_130', 'context_trf_131', 'context_trf_132', 'context_trf_133', 'context_trf_134', 'context_trf_135', 'context_trf_136', 'context_trf_137', 'context_trf_138', 'context_trf_139', 'context_trf_140', 'context_trf_141', 'context_trf_142', 'context_trf_143', 'context_trf_144', 'context_trf_145', 'context_trf_146', 'context_trf_147', 'context_trf_148', 'context_trf_149', 'context_trf_150', 'context_trf_151', 'context_trf_152', 'context_trf_153', 'context_trf_154', 'context_trf_155', 'context_trf_156', 'context_trf_157', 'context_trf_158', 'context_trf_159', 'context_trf_160', 'context_trf_161', 'context_trf_162', 'context_trf_163', 'context_trf_164', 'context_trf_165', 'context_trf_166', 'context_trf_167', 'context_trf_168', 'context_trf_169', 'context_trf_170', 'context_trf_171', 'context_trf_172', 'context_trf_173', 'context_trf_174', 'context_trf_175', 'context_trf_176', 'context_trf_177', 'context_trf_178', 'context_trf_179', 'context_trf_180', 'context_trf_181', 'context_trf_182', 'context_trf_183', 'context_trf_184', 'context_trf_185', 'context_trf_186', 'context_trf_187', 'context_trf_188', 'context_trf_189', 'context_trf_190', 'context_trf_191', 'context_trf_192', 'context_trf_193', 'context_trf_194', 'context_trf_195', 'context_trf_196', 'context_trf_197', 'context_trf_198', 'context_trf_199', 'context_trf_200', 'context_trf_201', 'context_trf_202', 'context_trf_203', 'context_trf_204', 'context_trf_205', 'context_trf_206', 'context_trf_207', 'context_trf_208', 'context_trf_209', 'context_trf_210', 'context_trf_211', 'context_trf_212', 'context_trf_213', 'context_trf_214', 'context_trf_215', 'context_trf_216', 'context_trf_217', 'context_trf_218', 'context_trf_219', 'context_trf_220', 'context_trf_221', 'context_trf_222', 'context_trf_223', 'context_trf_224', 'context_trf_225', 'context_trf_226', 'context_trf_227', 'context_trf_228', 'context_trf_229', 'context_trf_230', 'context_trf_231', 'context_trf_232', 'context_trf_233', 'context_trf_234', 'context_trf_235', 'context_trf_236', 'context_trf_237', 'context_trf_238', 'context_trf_239', 'context_trf_240', 'context_trf_241', 'context_trf_242', 'context_trf_243', 'context_trf_244', 'context_trf_245', 'context_trf_246', 'context_trf_247', 'context_trf_248', 'context_trf_249', 'context_trf_250', 'context_trf_251', 'context_trf_252', 'context_trf_253', 'context_trf_254', 'context_trf_255', 'context_trf_256', 'context_trf_257', 'context_trf_258', 'context_trf_259', 'context_trf_260', 'context_trf_261', 'context_trf_262', 'context_trf_263', 'context_trf_264', 'context_trf_265', 'context_trf_266', 'context_trf_267', 'context_trf_268', 'context_trf_269', 'context_trf_270', 'context_trf_271', 'context_trf_272', 'context_trf_273', 'context_trf_274', 'context_trf_275', 'context_trf_276', 'context_trf_277', 'context_trf_278', 'context_trf_279', 'context_trf_280', 'context_trf_281', 'context_trf_282', 'context_trf_283', 'context_trf_284', 'context_trf_285', 'context_trf_286', 'context_trf_287', 'context_trf_288', 'context_trf_289', 'context_trf_290', 'context_trf_291', 'context_trf_292', 'context_trf_293', 'context_trf_294', 'context_trf_295', 'context_trf_296', 'context_trf_297', 'context_trf_298', 'context_trf_299', 'context_trf_300', 'context_trf_301', 'context_trf_302', 'context_trf_303', 'context_trf_304', 'context_trf_305', 'context_trf_306', 'context_trf_307', 'context_trf_308', 'context_trf_309', 'context_trf_310', 'context_trf_311', 'context_trf_312', 'context_trf_313', 'context_trf_314', 'context_trf_315', 'context_trf_316', 'context_trf_317', 'context_trf_318', 'context_trf_319', 'context_trf_320', 'context_trf_321', 'context_trf_322', 'context_trf_323', 'context_trf_324', 'context_trf_325', 'context_trf_326', 'context_trf_327', 'context_trf_328', 'context_trf_329', 'context_trf_330', 'context_trf_331', 'context_trf_332', 'context_trf_333', 'context_trf_334', 'context_trf_335', 'context_trf_336', 'context_trf_337', 'context_trf_338', 'context_trf_339', 'context_trf_340', 'context_trf_341', 'context_trf_342', 'context_trf_343', 'context_trf_344', 'context_trf_345', 'context_trf_346', 'context_trf_347', 'context_trf_348', 'context_trf_349', 'context_trf_350', 'context_trf_351', 'context_trf_352', 'context_trf_353', 'context_trf_354', 'context_trf_355', 'context_trf_356', 'context_trf_357', 'context_trf_358', 'context_trf_359', 'context_trf_360', 'context_trf_361', 'context_trf_362', 'context_trf_363', 'context_trf_364', 'context_trf_365', 'context_trf_366', 'context_trf_367', 'context_trf_368', 'context_trf_369', 'context_trf_370', 'context_trf_371', 'context_trf_372', 'context_trf_373', 'context_trf_374', 'context_trf_375', 'context_trf_376', 'context_trf_377', 'context_trf_378', 'context_trf_379', 'context_trf_380', 'context_trf_381', 'context_trf_382', 'context_trf_383', 'context_trf_384', 'context_trf_385', 'context_trf_386', 'context_trf_387', 'context_trf_388', 'context_trf_389', 'context_trf_390', 'context_trf_391', 'context_trf_392', 'context_trf_393', 'context_trf_394', 'context_trf_395', 'context_trf_396', 'context_trf_397', 'context_trf_398', 'context_trf_399', 'context_trf_400', 'context_trf_401', 'context_trf_402', 'context_trf_403', 'context_trf_404', 'context_trf_405', 'context_trf_406', 'context_trf_407', 'context_trf_408', 'context_trf_409', 'context_trf_410', 'context_trf_411', 'context_trf_412', 'context_trf_413', 'context_trf_414', 'context_trf_415', 'context_trf_416', 'context_trf_417', 'context_trf_418', 'context_trf_419', 'context_trf_420', 'context_trf_421', 'context_trf_422', 'context_trf_423', 'context_trf_424', 'context_trf_425', 'context_trf_426', 'context_trf_427', 'context_trf_428', 'context_trf_429', 'context_trf_430', 'context_trf_431', 'context_trf_432', 'context_trf_433', 'context_trf_434', 'context_trf_435', 'context_trf_436', 'context_trf_437', 'context_trf_438', 'context_trf_439', 'context_trf_440', 'context_trf_441', 'context_trf_442', 'context_trf_443', 'context_trf_444', 'context_trf_445', 'context_trf_446', 'context_trf_447', 'context_trf_448', 'context_trf_449', 'context_trf_450', 'context_trf_451', 'context_trf_452', 'context_trf_453', 'context_trf_454', 'context_trf_455', 'context_trf_456', 'context_trf_457', 'context_trf_458', 'context_trf_459', 'context_trf_460', 'context_trf_461', 'context_trf_462', 'context_trf_463', 'context_trf_464', 'context_trf_465', 'context_trf_466', 'context_trf_467', 'context_trf_468', 'context_trf_469', 'context_trf_470', 'context_trf_471', 'context_trf_472', 'context_trf_473', 'context_trf_474', 'context_trf_475', 'context_trf_476', 'context_trf_477', 'context_trf_478', 'context_trf_479', 'context_trf_480', 'context_trf_481', 'context_trf_482', 'context_trf_483', 'context_trf_484', 'context_trf_485', 'context_trf_486', 'context_trf_487', 'context_trf_488', 'context_trf_489', 'context_trf_490', 'context_trf_491', 'context_trf_492', 'context_trf_493', 'context_trf_494', 'context_trf_495', 'context_trf_496', 'context_trf_497', 'context_trf_498', 'context_trf_499', 'context_trf_500', 'context_trf_501', 'context_trf_502', 'context_trf_503', 'context_trf_504', 'context_trf_505', 'context_trf_506', 'context_trf_507', 'context_trf_508', 'context_trf_509', 'context_trf_510', 'context_trf_511', 'context_trf_512', 'context_trf_513', 'context_trf_514', 'context_trf_515', 'context_trf_516', 'context_trf_517', 'context_trf_518', 'context_trf_519', 'context_trf_520', 'context_trf_521', 'context_trf_522', 'context_trf_523', 'context_trf_524', 'context_trf_525', 'context_trf_526', 'context_trf_527', 'context_trf_528', 'context_trf_529', 'context_trf_530', 'context_trf_531', 'context_trf_532', 'context_trf_533', 'context_trf_534', 'context_trf_535', 'context_trf_536', 'context_trf_537', 'context_trf_538', 'context_trf_539', 'context_trf_540', 'context_trf_541', 'context_trf_542', 'context_trf_543', 'context_trf_544', 'context_trf_545', 'context_trf_546', 'context_trf_547', 'context_trf_548', 'context_trf_549', 'context_trf_550', 'context_trf_551', 'context_trf_552', 'context_trf_553', 'context_trf_554', 'context_trf_555', 'context_trf_556', 'context_trf_557', 'context_trf_558', 'context_trf_559', 'context_trf_560', 'context_trf_561', 'context_trf_562', 'context_trf_563', 'context_trf_564', 'context_trf_565', 'context_trf_566', 'context_trf_567', 'context_trf_568', 'context_trf_569', 'context_trf_570', 'context_trf_571', 'context_trf_572', 'context_trf_573', 'context_trf_574', 'context_trf_575', 'context_trf_576', 'context_trf_577', 'context_trf_578', 'context_trf_579', 'context_trf_580', 'context_trf_581', 'context_trf_582', 'context_trf_583', 'context_trf_584', 'context_trf_585', 'context_trf_586', 'context_trf_587', 'context_trf_588', 'context_trf_589', 'context_trf_590', 'context_trf_591', 'context_trf_592', 'context_trf_593', 'context_trf_594', 'context_trf_595', 'context_trf_596', 'context_trf_597', 'context_trf_598', 'context_trf_599', 'context_trf_600', 'context_trf_601', 'context_trf_602', 'context_trf_603', 'context_trf_604', 'context_trf_605', 'context_trf_606', 'context_trf_607', 'context_trf_608', 'context_trf_609', 'context_trf_610', 'context_trf_611', 'context_trf_612', 'context_trf_613', 'context_trf_614', 'context_trf_615', 'context_trf_616', 'context_trf_617', 'context_trf_618', 'context_trf_619', 'context_trf_620', 'context_trf_621', 'context_trf_622', 'context_trf_623', 'context_trf_624', 'context_trf_625', 'context_trf_626', 'context_trf_627', 'context_trf_628', 'context_trf_629', 'context_trf_630', 'context_trf_631', 'context_trf_632', 'context_trf_633', 'context_trf_634', 'context_trf_635', 'context_trf_636', 'context_trf_637', 'context_trf_638', 'context_trf_639', 'context_trf_640', 'context_trf_641', 'context_trf_642', 'context_trf_643', 'context_trf_644', 'context_trf_645', 'context_trf_646', 'context_trf_647', 'context_trf_648', 'context_trf_649', 'context_trf_650', 'context_trf_651', 'context_trf_652', 'context_trf_653', 'context_trf_654', 'context_trf_655', 'context_trf_656', 'context_trf_657', 'context_trf_658', 'context_trf_659', 'context_trf_660', 'context_trf_661', 'context_trf_662', 'context_trf_663', 'context_trf_664', 'context_trf_665', 'context_trf_666', 'context_trf_667', 'context_trf_668', 'context_trf_669', 'context_trf_670', 'context_trf_671', 'context_trf_672', 'context_trf_673', 'context_trf_674', 'context_trf_675', 'context_trf_676', 'context_trf_677', 'context_trf_678', 'context_trf_679', 'context_trf_680', 'context_trf_681', 'context_trf_682', 'context_trf_683', 'context_trf_684', 'context_trf_685', 'context_trf_686', 'context_trf_687', 'context_trf_688', 'context_trf_689', 'context_trf_690', 'context_trf_691', 'context_trf_692', 'context_trf_693', 'context_trf_694', 'context_trf_695', 'context_trf_696', 'context_trf_697', 'context_trf_698', 'context_trf_699', 'context_trf_700', 'context_trf_701', 'context_trf_702', 'context_trf_703', 'context_trf_704', 'context_trf_705', 'context_trf_706', 'context_trf_707', 'context_trf_708', 'context_trf_709', 'context_trf_710', 'context_trf_711', 'context_trf_712', 'context_trf_713', 'context_trf_714', 'context_trf_715', 'context_trf_716', 'context_trf_717', 'context_trf_718', 'context_trf_719', 'context_trf_720', 'context_trf_721', 'context_trf_722', 'context_trf_723', 'context_trf_724', 'context_trf_725', 'context_trf_726', 'context_trf_727', 'context_trf_728', 'context_trf_729', 'context_trf_730', 'context_trf_731', 'context_trf_732', 'context_trf_733', 'context_trf_734', 'context_trf_735', 'context_trf_736', 'context_trf_737', 'context_trf_738', 'context_trf_739', 'context_trf_740', 'context_trf_741', 'context_trf_742', 'context_trf_743', 'context_trf_744', 'context_trf_745', 'context_trf_746', 'context_trf_747', 'context_trf_748', 'context_trf_749', 'context_trf_750', 'context_trf_751', 'context_trf_752', 'context_trf_753', 'context_trf_754', 'context_trf_755', 'context_trf_756', 'context_trf_757', 'context_trf_758', 'context_trf_759', 'context_trf_760', 'context_trf_761', 'context_trf_762',
#  'context_trf_763', 'context_trf_764', 'context_trf_765', 'context_trf_766', 'context_trf_767']
class WordFeature:

    def __init__(self, num, word_list, sentence_list, semantic_feature):
        super().__init__()
        self.num = num
        self.word_list = word_list
        self.sent_list = sentence_list
        self.semantic_feature = semantic_feature
        # semantic feature
        # self.syllable, self.length, self.fam, self.ent_flag, self.topic_score = self.get_semantic_feature()
        # visual feature
        self.fixation_duration = [0 for _ in range(num)]
        self.number_of_fixations = [0 for _ in range(num)]
        self.reading_times = [0 for _ in range(num)]

        self.fixation_duration_diff = [0 for _ in range(num)]
        self.number_of_fixations_diff = [0 for _ in range(num)]
        self.reading_times_diff = [0 for _ in range(num)]

        self.fixation_duration_mean = [0 for _ in range(num)]
        self.fixation_duration_var = [0 for _ in range(num)]
        self.number_of_fixations_mean = [0 for _ in range(num)]
        self.number_of_fixations_var = [0 for _ in range(num)]
        self.reading_times_mean = [0 for _ in range(num)]
        self.reading_times_var = [0 for _ in range(num)]

        self.fixation_duration_div_syllable = [0 for _ in range(num)]
        self.fixation_duration_div_length = [0 for _ in range(num)]

        # bert feature

        # 辅助

    # def get_semantic_feature(self):
    #     syllable = [0 for _ in range(self.num)]
    #     length = [0 for _ in range(self.num)]
    #     fam = [0 for _ in range(self.num)]
    #     ent_flag = [0 for _ in range(self.num)]
    #     topic_score = [0 for _ in range(self.num)]
    #     for i, semantic in enumerate(self.semantic_feature):
    #         syllable[i] = semantic['syllable']
    #         length[i] = semantic['length']
    #         fam[i] = semantic['fam']
    #         ent_flag[i] = semantic['ent_flag']
    #         topic_score[i] = semantic['topic_score']
    #     return syllable, length, fam, ent_flag, topic_score

    def update(self):
        self.fixation_duration_diff = self.get_diff(self.fixation_duration)
        self.number_of_fixations_diff = self.get_diff(self.number_of_fixations)
        self.reading_times_diff = self.get_diff(self.reading_times)

        for i, word in enumerate(self.word_list):
            syllable_len = textstat.syllable_count(word)
            if syllable_len != 0:
                self.fixation_duration_div_syllable[i] = self.fixation_duration[i] / syllable_len
            else:
                self.fixation_duration_div_syllable[i] = 0

            if len(word) != 0:
                self.fixation_duration_div_length[i] = self.fixation_duration[i] / len(word)
            else:
                self.fixation_duration_div_length[i] = 0

        self.fixation_duration_mean, self.fixation_duration_var = self.get_sentence_statistic(self.sent_list,
                                                                                              self.fixation_duration)
        self.number_of_fixations_mean, self.number_of_fixations_var = self.get_sentence_statistic(self.sent_list,
                                                                                                  self.number_of_fixations)

        self.reading_times_mean, self.reading_times_var = self.get_sentence_statistic(self.sent_list,
                                                                                      self.reading_times)

    def get_diff(self, list1):
        results = [0 for _ in range(len(list1))]
        for i in range(len(list1)):
            if i == 0:
                continue
            results[i] = list1[i] - list1[i - 1]
        return results

    def get_syllable(self, word_list):
        syllable_len = []
        for word in word_list:
            syllable_len.append(textstat.syllable_count(word))
        return syllable_len

    def get_sentence_statistic(self, sentence_list, target_list):
        mean_list = [0 for _ in range(self.num)]
        var_list = [0 for _ in range(self.num)]
        for sent in sentence_list:
            sen_info = target_list[sent[1]:sent[2]]
            mean = np.mean(sen_info)
            var = np.var(sen_info)

            for i in range(sent[1], sent[2]):
                mean_list[i] = mean
                var_list[i] = var
        return mean_list, var_list

    def to_str(self):
        print(f"fixation_duration_mean:{self.fixation_duration_mean}")

    def to_dataframe(self):
        data = pd.DataFrame({
            'fixation_duration': self.fixation_duration,
            'number_of_fixations': self.number_of_fixations,
            'reading_times': self.reading_times,

            'fixation_duration_diff': self.fixation_duration_diff,
            'number_of_fixations_diff': self.number_of_fixations_diff,
            'reading_times_diff': self.reading_times_diff,

            'fixation_duration_mean': self.fixation_duration_mean,
            'fixation_duration_var': self.fixation_duration_var,
            'number_of_fixations_mean': self.number_of_fixations_mean,
            'number_of_fixations_var': self.number_of_fixations_var,
            'reading_times_mean': self.reading_times_mean,
            'reading_times_var': self.reading_times_var,

            'fixation_duration_div_syllable': self.fixation_duration_div_syllable,
            'fixation_duration_div_length': self.fixation_duration_div_length
        })
        return data


class SentFeature:

    def __init__(self, num, sent_list, word_list):
        super().__init__()
        self.num = num
        self.sent_list = sent_list
        self.word_list = word_list

        self.backward_times_of_sentence = [0 for _ in range(num)]
        self.forward_times_of_sentence = [0 for _ in range(num)]
        self.horizontal_saccade_proportion = [0 for _ in range(num)]
        self.saccade_duration = [0 for _ in range(num)]
        self.saccade_times_of_sentence = [0 for _ in range(num)]
        self.saccade_velocity = [0 for _ in range(num)]
        self.total_dwell_time_of_sentence = [0 for _ in range(num)]

        self.saccade_distance = [0 for _ in range(num)]

        self.horizontal_saccade = [0 for _ in range(num)]

        self.backward_times_of_sentence_div_syllable = [0 for _ in range(num)]
        self.forward_times_of_sentence_div_syllable = [0 for _ in range(num)]
        self.horizontal_saccade_proportion_div_syllable = [0 for _ in range(num)]
        self.saccade_duartion_div_syllable = [0 for _ in range(num)]
        self.saccade_times_of_sentence_div_syllable = [0 for _ in range(num)]
        self.saccade_velocity_div_syllable = [0 for _ in range(num)]
        self.total_dwell_time_of_sentence_div_syllable = [0 for _ in range(num)]

    def update(self):
        self.backward_times_of_sentence_div_syllable = self.div_syllable(self.backward_times_of_sentence)
        self.forward_times_of_sentence_div_syllable = self.div_syllable(self.forward_times_of_sentence)

        self.horizontal_saccade_proportion = self.get_list_div(self.horizontal_saccade, self.saccade_times_of_sentence)
        self.horizontal_saccade_proportion_div_syllable = self.div_syllable(self.horizontal_saccade_proportion)

        self.saccade_duration_div_syllable = self.div_syllable(self.saccade_duration)
        self.saccade_times_of_sentence_div_syllable = self.div_syllable(self.saccade_times_of_sentence)

        self.saccade_velocity = self.get_list_div(self.saccade_distance, self.saccade_duration)
        self.saccade_velocity_div_syllable = self.div_syllable(self.saccade_velocity)

        self.total_dwell_time_of_sentence_div_syllable = self.div_syllable(self.total_dwell_time_of_sentence)

    def get_list_div(self, list_a, list_b):
        div_list = [0 for _ in range(self.num)]
        for i in range(len(list_b)):
            if list_b[i] != 0:
                div_list[i] = list_a[i] / list_b[i]

        return div_list

    def div_syllable(self, feature):
        assert len(feature) == len(self.sent_list)
        results = []
        for i in range(len(feature)):
            sent = self.sent_list[i]
            syllable_len = self.get_syllable(self.word_list[sent[1]:sent[2]])
            if syllable_len > 0:
                results.append(feature[i] / syllable_len)
            else:
                results.append(0)
        return results

    def get_syllable(self, word_list):
        syllable_len = 0
        for word in word_list:
            syllable_len += textstat.syllable_count(word)
        return syllable_len

    def to_dataframe(self):
        data = pd.DataFrame({
            'backward_times_of_sentence_div_syllable': self.backward_times_of_sentence_div_syllable,
            'forward_times_of_sentence_div_syllable': self.forward_times_of_sentence_div_syllable,
            'horizontal_saccade_proportion_div_syllable': self.horizontal_saccade_proportion_div_syllable,
            'saccade_duration_div_syllable': self.saccade_duration_div_syllable,
            'saccade_times_of_sentence_div_syllable': self.saccade_times_of_sentence_div_syllable,
            'saccade_velocity_div_syllable': self.saccade_velocity_div_syllable,
            'total_dwell_time_of_sentence_div_syllable': self.total_dwell_time_of_sentence_div_syllable
        })
        return data


class ReadingArticle:
    all_values = {
        'syllable': [],
        'length': [],
        'fam': [],
        'ent_flag': [],
        'stop_flag': [],
        'topic_score': [],
    }
    mean_values = {}
    std_values = {}

    word_fam_map = {}
    with open('mrc2.dct', 'r') as fp:
        i = 0
        for line in fp:
            line = line.strip()

            word, phon, dphon, stress = line[51:].split('|')

            w = {
                'wid': i,
                'nlet': int(line[0:2]),
                'nphon': int(line[2:4]),
                'nsyl': int(line[4]),
                'kf_freq': int(line[5:10]),
                'kf_ncats': int(line[10:12]),
                'kf_nsamp': int(line[12:15]),
                'tl_freq': int(line[15:21]),
                'brown_freq': int(line[21:25]),
                'fam': int(line[25:28]),
                'conc': int(line[28:31]),
                'imag': int(line[31:34]),
                'meanc': int(line[34:37]),
                'meanp': int(line[37:40]),
                'aoa': int(line[40:43]),
                'tq2': line[43],
                'wtype': line[44],
                'pdwtype': line[45],
                'alphasyl': line[46],
                'status': line[47],
                'var': line[48],
                'cap': line[49],
                'irreg': line[50],
                'word': word,
                'phon': phon,
                'dphon': dphon,
                'stress': stress
            }
            if word not in word_fam_map:
                word_fam_map[word] = w['fam']
            word_fam_map[word] = max(word_fam_map[word], w['fam'])
            i += 1

    def __init__(self, article_text):
        self.text = article_text
        self._spacy_doc = nlp(article_text)
        self._word_list = [token.text for token in self._spacy_doc]
        self._transformer_features = self._generate_word_embedding(xlnet_model)
        self._difficulty_features = self._generate_word_difficulty()
        self._topic_features = self._generate_topic_feature(top_n=1000)
        self._sentence_word_mapping = self._generate_sentence_mapping()

    def _generate_word_embedding(self, language_model):
        inputs = xlnet_tokenizer(self.text, return_tensors='pt')
        word_token_mapping = self.generate_token_mapping(self._word_list, inputs.tokens())
        outputs = language_model(**inputs)
        token_embeddings = outputs[0].squeeze()
        word_embeddings = []
        for start_id, end_id in word_token_mapping:
            if start_id <= end_id:
                word_embeddings.append(
                    torch.mean(token_embeddings[start_id: end_id + 1, :], 0, dtype=torch.float32))
            else:
                word_embeddings.append(torch.zeros((token_embeddings.shape[1],), dtype=torch.float32))
        return word_embeddings

    def _generate_sentence_mapping(self):
        self._sentence_list = sent_tokenize(self.text)
        sentence_word_mapping = self.generate_token_mapping(self._sentence_list, self._word_list)
        return sentence_word_mapping

    @staticmethod
    def standardization(value, column):
        if ReadingArticle.std_values[column] != 0.:
            return (value - ReadingArticle.mean_values[column]) / ReadingArticle.std_values[column]
        else:
            return 0.

    @staticmethod
    def generate_token_mapping(string_list, token_list):
        string_pos = 0
        string_idx = 0
        token_string_idx_list = []
        max_cross_count = 3
        for token_idx, token in enumerate(token_list):
            original_token = token.replace('▁', '')
            flag = False
            while string_idx < len(string_list) and string_list[string_idx][
                                                    string_pos: string_pos + len(original_token)] != original_token:
                string_pos += 1
                if string_pos >= len(string_list[string_idx]):
                    cross_count = 1
                    prefix = string_list[string_idx]
                    pre_length = len(string_list[string_idx])
                    while cross_count <= max_cross_count and string_idx + cross_count < len(string_list):
                        prefix += string_list[string_idx + cross_count]
                        new_string_pos = 0
                        while new_string_pos + len(original_token) <= len(prefix) and new_string_pos < len(
                                string_list[string_idx]):
                            if prefix[new_string_pos: new_string_pos + len(
                                    original_token)] == original_token and new_string_pos + len(
                                original_token) > len(
                                string_list[string_idx]):
                                string_pos = new_string_pos + len(original_token) - pre_length
                                flag = True
                                break
                            new_string_pos += 1
                        if flag:
                            break
                        pre_length += len(string_list[string_idx + cross_count])
                        cross_count += 1
                    if flag:
                        for delta_idx in range(cross_count + 1):
                            token_string_idx_list.append((token_idx, string_idx + delta_idx))
                        string_idx += cross_count
                        if string_idx < len(string_list) and string_pos == len(string_list[string_idx]):
                            string_pos = 0
                            string_idx += 1
                        break
                    else:
                        string_pos = 0
                        string_idx += 1
            if flag:
                continue
            if string_idx < len(string_list) and string_pos == len(string_list[string_idx]):
                string_pos = 0
                string_idx += 1
            if string_idx >= len(string_list):
                continue
            token_string_idx_list.append((token_idx, string_idx))
            string_pos += len(original_token)

        # for token_idx, string_idx in token_string_idx_list:
        #     print(inputs.tokens()[token_idx], string_list[string_idx])

        string_token_mapping = [(float('inf'), 0)] * len(string_list)
        for token_idx, string_idx in token_string_idx_list:
            string_token_mapping[string_idx] = (
                min(string_token_mapping[string_idx][0], token_idx),
                max(string_token_mapping[string_idx][1], token_idx))

        return string_token_mapping

    @staticmethod
    def get_word_familiar_rate(word_text):
        capital_word = word_text.upper()
        return ReadingArticle.word_fam_map.get(capital_word, 0)

    def _generate_word_difficulty(self):
        word_difficulties = []
        for token in self._spacy_doc:
            if token.is_alpha:  # and not token.is_stop:
                fam = self.get_word_familiar_rate(token.text)
                if fam == 0:
                    fam = self.get_word_familiar_rate(token.lemma_)
                syllable = textstat.syllable_count(token.text)
                length = len(token.text)
                ent_flag = token.ent_iob != 2
                stop_flag = token.is_stop
                difficulty_feat = {
                    # float(textstat.syllable_count(token.text) > 2),
                    # float(len(token.text) > 7),
                    # float(fam < 482),
                    'syllable': float(syllable),
                    'length': float(length),
                    'fam': float(fam),
                    'ent_flag': float(ent_flag),
                }
                for column in difficulty_feat:
                    ReadingArticle.all_values[column].append(difficulty_feat[column])
            else:
                difficulty_feat = {
                    'syllable': 0.,
                    'length': 0.,
                    'fam': 645.,
                    'ent_flag': 0.,
                }
            word_difficulties.append(difficulty_feat)
        return word_difficulties

    def _generate_topic_feature(self, top_n):
        kw_extractor.load_document(input=self.text, language='en')
        kw_extractor.candidate_selection(1)
        kw_extractor.candidate_weighting()
        keywords = kw_extractor.get_n_best(n=top_n)
        keywords = {k: min(max(v, 0), 1) for k, v in keywords}

        topic_feature = []
        for token in self._spacy_doc:
            doc_level_score = keywords.get(token.text, 1.)
            topic_feature.append({
                'topic_score': doc_level_score
            })
            ReadingArticle.all_values['topic_score'].append(doc_level_score)
        return topic_feature

    def get_word_filter_id_set(self, only_alpha=True, filter_digit=True, filter_punctuation=True,
                               filter_stop_words=False):
        word_filter_id_set = set()
        for word in self._spacy_doc:
            if only_alpha and not word.is_alpha:
                word_filter_id_set.add(word.i)
            if filter_digit and word.is_digit:
                word_filter_id_set.add(word.i)
            if filter_punctuation and word.is_punct:
                word_filter_id_set.add(word.i)
            if filter_stop_words and word.is_stop:
                word_filter_id_set.add(word.i)
        return word_filter_id_set

    def get_word_list(self):
        return self._word_list

    def get_sentence_list(self):
        return self._sentence_list

    def get_transformer_features(self):
        return self._transformer_features

    def get_difficulty_features(self):
        rst = []
        if not ReadingArticle.mean_values or not ReadingArticle.std_values:
            for column, values in ReadingArticle.all_values.items():
                ReadingArticle.mean_values[column] = np.mean(values)
                ReadingArticle.std_values[column] = np.std(values)
        for difficulty_feature in self._difficulty_features:
            rst.append(torch.tensor(
                [self.standardization(value, column) for column, value in difficulty_feature.items()],
                dtype=torch.float32)
            )
        return rst

    def get_topic_features(self):
        rst = []
        if not ReadingArticle.mean_values or not ReadingArticle.std_values:
            for column, values in ReadingArticle.all_values.items():
                ReadingArticle.mean_values[column] = np.mean(values)
                ReadingArticle.std_values[column] = np.std(values)
        for topic_feature in self._topic_features:
            rst.append(
                torch.tensor([self.standardization(value, column) for column, value in topic_feature.items()],
                             dtype=torch.float32))
        return rst

    def get_original_features(self):
        rst = []
        for difficulty_feat, topic_feat in zip(self._difficulty_features, self._topic_features):
            line = {}
            for column, value in difficulty_feat.items():
                line[column] = value
            for column, value in topic_feat.items():
                line[column] = value
            rst.append(line)
        return rst

    def get_sentence_word_mapping(self):
        return self._sentence_word_mapping
