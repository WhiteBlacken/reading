import base64
import json
import math
import os
import pickle
import shutil

import cv2
from django.db.models import F

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger
from PIL import Image

from action.models import Dictionary, Experiment, PageData, Paragraph, Text, Translation, UserReadingInfo
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
import time
from nltk.tokenize import sent_tokenize
import pandas as pd
import json
import matplotlib.pyplot as plt
from pke.unsupervised import YAKE

nlp = spacy.load("en_core_web_lg")
# base = "D:\\qxy\\pre-trained-model\\"
base = ""

xlnet_tokenizer = XLNetTokenizerFast.from_pretrained(base + 'xlnet-base-cased')
xlnet_model = XLNetModel.from_pretrained(base + 'xlnet-base-cased', output_attentions=True)

kw_extractor = YAKE()

start = time.perf_counter()

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

with open('model/abnormalSVM.pickle', 'rb') as f:
    def login_page(request):
        return render(request, "login.html")


    abnormalSVM = pickle.load(f)





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

    request.session['page_info'] = None
    return render(request, "onlineReading.html")


def test_dispersion(request):
    return render(request, "testDispersion.html")


def label(request):
    request.session['page_info'] = None
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
    interventions = request.session.get("interventions")
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
        request.session['intervention'] = None
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
    # print(len(fixations))
    page_data = PageData.objects.get(id=page_data_id)

    generate_pic_by_base64(page_data.image, path, filename)

    paint_gaze_on_pic(fixations, path + filename, path + "fixation.png")
    paint_gaze_on_pic(fixations, path + "visual.png", path + "fix_heat.png")


# def get_semantic_attention(
#         page_data_id,
#         username,
#         gaze_x,
#         gaze_y,
#         gaze_t,
#         background,
#         base_path,
#         word_list,
#         word_locations,
#         top_dict,
# ):
#     logger.info("visual attention正在分析....")
#
#     """
#     1. 清洗数据
#     """
#     coordinates = format_gaze(gaze_x, gaze_y, gaze_t)
#
#     """
#     2. 计算top_k
#     """
#     path = "static/data/heatmap/" + str(username) + "/" + str(page_data_id) + "/"
#     filename = "background.png"
#
#     data_list = [[x[0], x[1]] for x in coordinates]
#     hotspot = myHeatmap.draw_heat_map(data_list, base_path + "visual.png", background)


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

                if index > pre_word_index:
                    sentFeature.forward_times_of_sentence[sent_index] += 1
                else:
                    sentFeature.backward_times_of_sentence[sent_index] += 1

            pre_word_index = index
    return sentFeature


def get_pred(request):
    """
    输入应该是
        * 当前2s内的gaze点（前端传输）
        * 历史的所有gaze点
        * 该页的位置信息
    """
    global start
    end = time.perf_counter()
    logger.info(f"两次接口调用用时{end - start}ms")
    start = end
    with Timer("pred"):  # 开启计时
        x = request.POST.get("x")
        y = request.POST.get("y")
        t = request.POST.get("t")

        page_id = request.session['page_id']
        page_data = PageData.objects.get(id=page_id)

        userInfo = UserReadingInfo.objects.filter(user=request.session['username']).first()

        gaze_x = ""
        gaze_y = ""
        gaze_t = ""

        if len(x) > 0:
            if page_data.gaze_x is None:
                gaze_x = x
                gaze_y = y
                gaze_t = t
            else:
                gaze_x = page_data.gaze_x + "," + x
                gaze_y = page_data.gaze_y + "," + y
                gaze_t = page_data.gaze_t + "," + t

        word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)

        border, rows, danger_zone, len_per_word = textarea(page_data.location)

        diff_list = generate_word_difficulty(page_data.texts)

        word_watching_list = []
        sent_watching_list = []

        # TODO 为了减少计算量，仅在当前的单词上计算特征

        if len(x) > 0:
            gaze_points = format_gaze(x, y, t, begin_time=0, end_time=0)
            result_fixations = detect_fixations(gaze_points)

            now_word_feature = WordFeature(len(word_list), word_list, sentence_list, 'test')
            now_word_feature = get_word_feature(now_word_feature, result_fixations, page_data.location)

            now_sent_feature = SentFeature(len(sentence_list), sentence_list, word_list)
            now_sent_feature = get_sent_feature(now_sent_feature, result_fixations, page_data.location, sentence_list,
                                                rows)
            now_sent_feature.update()  # 特征除以syllable

            # result_fixations = keep_row(result_fixations)
            # 单词fixation最多的句子，为需要判断的句子
            sent_fix = [0 for _ in range(len(sentence_list))]

            print(f"result_fixation:{result_fixations}")
            for fixation in result_fixations:
                index, flag = get_item_index_x_y(page_data.location, fixation[0],
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

        print(f'word_watching:{word_watching_list}')
        print(f'sent_watching:{sent_watching_list}')

        word_not_understand_list = []
        sent_not_understand_list = []
        sent_mind_wandering_list = []

        for watching in word_watching_list:
            print(f'now_word_feature:{now_word_feature.fixation_duration[watching]}')
            if now_word_feature.fixation_duration[watching] >= 3.7 * float(userInfo.fixation_duration_mean) and \
                    diff_list[watching][1] >= 1:
                for q in range(watching - 3, watching + 3):
                    if 0 <= q <= len(word_list) - 1:
                        word_not_understand_list.append(q)
        word_not_understand_list = list(set(word_not_understand_list))

        print(f"word_not_understand:{word_not_understand_list}")

        if len(word_not_understand_list) > 0:
            PageData.objects.filter(id=page_id).update(
                word_intervention=page_data.word_intervention + "," + str(word_not_understand_list)
            )

        for watching in sent_watching_list:
            if watching >= 0:
                sent = sentence_list[watching]
                if watching == request.session.get('pre_sent_inter', None):
                    print("重复干预")
                    continue
                print(f"total dwell time:{now_sent_feature.total_dwell_time_of_sentence_div_syllable[watching]}")
                if now_sent_feature.total_dwell_time_of_sentence_div_syllable[watching] > (1 / 2) * float(
                        userInfo.total_dwell_time_of_sentence_mean) and now_sent_feature.backward_times_of_sentence[
                    watching] > float(userInfo.backward_times_of_sentence_mean):
                    sent_not_understand_list.append([sent[1], sent[2] - 1])
                    PageData.objects.filter(id=page_id).update(
                        sent_intervention=page_data.sent_intervention + "," + str(watching)
                    )
                    request.session['pre_sent_inter'] = watching
                if watching - 1 > 0:
                    if now_sent_feature.total_dwell_time_of_sentence_div_syllable[watching] < (1 / 4) * float(
                            userInfo.total_dwell_time_of_sentence_mean) and len(word_not_understand_list) <= 9:
                        sent_mind_wandering_list.append([sent[1], sent[2] - 1])
                        PageData.objects.filter(id=page_id).update(
                            mind_wander_intervention=page_data.mind_wander_intervention + "," + str(watching)
                        )
                        request.session['pre_sent_inter'] = watching

            # if sent_predicts[watching]:
            #     sent = sentence_list[watching]
            #     # abnormal 再来判断原因
            #     if abnormal_predicts[watching] == 0:
            #         sent_mind_wandering_list.append([sent[1], sent[2] - 1])
            #         sent_not_understand_list.append([sent[1], sent[2] - 1])
            #     if abnormal_predicts[watching] == 1:
            #         sent_not_understand_list.append([sent[1], sent[2] - 1])
            #     if abnormal_predicts[watching] == 2:
            #         sent_mind_wandering_list.append([sent[1], sent[2] - 1])

        PageData.objects.filter(id=page_id).update(
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            gaze_t=gaze_t,
        )

    context = {
        "word": word_not_understand_list,
        "sentence": sent_not_understand_list,
        "wander": sent_mind_wandering_list
    }

    # 将系统的干预记下来，用于pilot study的分析
    return JsonResponse(context)


def get_pred_by_glass(request):
    import zmq
    import msgpack
    ctx = zmq.Context()

    pupil_remote = ctx.socket(zmq.REQ)

    ip = 'localhost'  # If you talk to a different machine use its IP.
    port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

    pupil_remote.connect(f'tcp://{ip}:{port}')

    # Request 'SUB_PORT' for reading data
    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()
    print(f'sub_port:{sub_port}')

    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(f'tcp://{ip}:{sub_port}')
    subscriber.subscribe('gaze.')  # receive all gaze messages

    # we need a serializer
    cnt = 0

    x = []
    y = []
    t = []

    while True: # 需要修改
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        print(f"{topic}: {message}")
        x.append(message['norm_pos'][0]*1920)
        y.append(message['norm_pos'][1]*1080)
        t.append(message['timestamp'])
        cnt += 1
        if cnt > 30:
            break

    with Timer("pred_by_glass"):  # 开启计时

        page_id = request.session['page_id']
        page_data = PageData.objects.get(id=page_id)

        userInfo = UserReadingInfo.objects.filter(user=request.session['username']).first()


        word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)

        border, rows, danger_zone, len_per_word = textarea(page_data.location)

        diff_list = generate_word_difficulty(page_data.texts)

        word_watching_list = []
        sent_watching_list = []

        # TODO 为了减少计算量，仅在当前的单词上计算特征

        if len(x) > 0:
            gaze_points = format_gaze(x, y, t, begin_time=0, end_time=0)
            result_fixations = detect_fixations(gaze_points)

            now_word_feature = WordFeature(len(word_list), word_list, sentence_list, 'test')
            now_word_feature = get_word_feature(now_word_feature, result_fixations, page_data.location)

            now_sent_feature = SentFeature(len(sentence_list), sentence_list, word_list)
            now_sent_feature = get_sent_feature(now_sent_feature, result_fixations, page_data.location, sentence_list,
                                                rows)
            now_sent_feature.update()  # 特征除以syllable

            # result_fixations = keep_row(result_fixations)
            # 单词fixation最多的句子，为需要判断的句子
            sent_fix = [0 for _ in range(len(sentence_list))]

            print(f"result_fixation:{result_fixations}")
            for fixation in result_fixations:
                index, flag = get_item_index_x_y(page_data.location, fixation[0],
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

        print(f'word_watching:{word_watching_list}')
        print(f'sent_watching:{sent_watching_list}')

        word_not_understand_list = []
        sent_not_understand_list = []
        sent_mind_wandering_list = []

        for watching in word_watching_list:
            print(f'now_word_feature:{now_word_feature.fixation_duration[watching]}')
            if now_word_feature.fixation_duration[watching] >= 3.7 * float(userInfo.fixation_duration_mean) and \
                    diff_list[watching][1] >= 1:
                for q in range(watching - 3, watching + 3):
                    if 0 <= q <= len(word_list) - 1:
                        word_not_understand_list.append(q)
        word_not_understand_list = list(set(word_not_understand_list))

        print(f"word_not_understand:{word_not_understand_list}")

        if len(word_not_understand_list) > 0:
            PageData.objects.filter(id=page_id).update(
                word_intervention=page_data.word_intervention + "," + str(word_not_understand_list)
            )

        for watching in sent_watching_list:
            if watching >= 0:
                sent = sentence_list[watching]
                if watching == request.session.get('pre_sent_inter', None):
                    print("重复干预")
                    continue
                print(f"total dwell time:{now_sent_feature.total_dwell_time_of_sentence_div_syllable[watching]}")
                if now_sent_feature.total_dwell_time_of_sentence_div_syllable[watching] > (1 / 2) * float(
                        userInfo.total_dwell_time_of_sentence_mean) and now_sent_feature.backward_times_of_sentence[
                    watching] > float(userInfo.backward_times_of_sentence_mean):
                    sent_not_understand_list.append([sent[1], sent[2] - 1])
                    PageData.objects.filter(id=page_id).update(
                        sent_intervention=page_data.sent_intervention + "," + str(watching)
                    )
                    request.session['pre_sent_inter'] = watching
                if watching - 1 > 0:
                    if now_sent_feature.total_dwell_time_of_sentence_div_syllable[watching] < (1 / 4) * float(
                            userInfo.total_dwell_time_of_sentence_mean) and len(word_not_understand_list) <= 9:
                        sent_mind_wandering_list.append([sent[1], sent[2] - 1])
                        PageData.objects.filter(id=page_id).update(
                            mind_wander_intervention=page_data.mind_wander_intervention + "," + str(watching)
                        )
                        request.session['pre_sent_inter'] = watching

            # if sent_predicts[watching]:
            #     sent = sentence_list[watching]
            #     # abnormal 再来判断原因
            #     if abnormal_predicts[watching] == 0:
            #         sent_mind_wandering_list.append([sent[1], sent[2] - 1])
            #         sent_not_understand_list.append([sent[1], sent[2] - 1])
            #     if abnormal_predicts[watching] == 1:
            #         sent_not_understand_list.append([sent[1], sent[2] - 1])
            #     if abnormal_predicts[watching] == 2:
            #         sent_mind_wandering_list.append([sent[1], sent[2] - 1])

        # PageData.objects.filter(id=page_id).update(
        #     gaze_x=gaze_x,
        #     gaze_y=gaze_y,
        #     gaze_t=gaze_t,
        # )

    context = {
        "word": word_not_understand_list,
        "sentence": sent_not_understand_list,
        "wander": sent_mind_wandering_list
    }

    # 将系统的干预记下来，用于pilot study的分析
    return JsonResponse(context)


def Test(request):
    data = pd.read_csv('jupyter/dataset/handcraft-div-duration.csv')
    users = ['pwt',
             'czh',
             'ys',
             'chenyuwang',
             'shiyubin',
             'luqi',
             'xuzhenyu',
             'qxy',
             'zhouyanglu',
             'zyf',
             'Zhenyu',
             'dongfang',
             'xuhailin',
             'wuyuting',
             'liuyiting',
             'XianweiWang',
             '梁胃寒',
             'ln',
             'luyutian',
             'gonghaotong']

    for user in users:
        dat = data[data.user == user]
        # backward_times_of_sentence_var = np.var(dat['backward_times_of_sentence_div_syllable'])
        # print(backward_times_of_sentence_var)

        UserReadingInfo.objects.create(
            user=user,
            backward_times_of_sentence_mean=np.mean(dat['backward_times_of_sentence_div_syllable']),
            backward_times_of_sentence_var=np.var(dat['backward_times_of_sentence_div_syllable']),

            forward_times_of_sentence_mean=np.mean(dat['forward_times_of_sentence_div_syllable']),
            forward_times_of_sentence_var=np.var(dat['forward_times_of_sentence_div_syllable']),

            saccade_duration_mean=np.mean(dat['saccade_duartion_div_syllable']),
            saccade_duration_var=np.var(dat['saccade_duartion_div_syllable']),

            saccade_times_of_sentence_mean=np.mean(dat['saccade_times_of_sentence_div_syllable']),
            saccade_times_of_sentence_var=np.var(dat['saccade_times_of_sentence_div_syllable']),

            total_dwell_time_of_sentence_mean=np.mean(dat['total_dwell_time_of_sentence_div_syllable']),
            total_dwell_time_of_sentence_var=np.var(dat['total_dwell_time_of_sentence_div_syllable']),

            fixation_duration_mean=np.mean(dat['fixation_duration']),
            fixation_duration_var=np.var(dat['fixation_duration']),

            number_of_fixations_mean=np.mean(dat['number_of_fixations']),
            number_of_fixations_var=np.var(dat['number_of_fixations']),

            reading_times_mean=np.mean(dat['reading_times']),
            reading_times_var=np.var(dat['reading_times']),
        )

    return HttpResponse(1)


def get_page_info(request):
    page_text = request.POST.get("page_text")
    location = request.POST.get("location")

    is_end = request.POST.get("is_end", 0)

    experiment_id = request.session.get("experiment_id", None)
    if experiment_id and page_text:
        page_num = request.session.get('page_num')

        if not page_num:
            page_num = 1
        print(f'page_num:{page_num}')

        page_data = PageData.objects.create(
            texts=page_text,
            page=page_num,
            image="",
            experiment_id=experiment_id,
            location=location,
            is_pilot_study=True
        )
        request.session['page_id'] = page_data.id
        logger.info("第%s页数据已存储,id为%s" % (page_num, str(page_data.id)))
        request.session['page_num'] = page_num + 1

    if int(is_end) == 1:  # 阅读结束
        request.session['page_num'] = None
        request.session['experiment_id'] = None

    request.session['pre_sent_inter'] = None
    return HttpResponse(1)


# def generate_point_on_word_by_semantic(page_data_id):
def go_marker(request):
    return render(request,'onlineReading_maker.html')
def get_semantic_attention_map(request):
    exp_ids = request.GET.get("exp_id").split(',')
    for exp_id in exp_ids:
        page_data_ls = PageData.objects.filter(experiment_id=exp_id)
        for page_data in page_data_ls:
            page_data_id = page_data.id
            pageData = PageData.objects.get(id=page_data_id)
            # 准备工作，获取单词和句子的信息
            word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
            # 获取单词的位置
            word_locations = get_word_location(pageData.location)  # [(left,top,right,bottom),(left,top,right,bottom)]

            # 确保单词长度是正确的
            assert len(word_locations) == len(word_list)
            # 获取图片生成的路径
            exp = Experiment.objects.filter(id=pageData.experiment_id)

            base_path = "semantic_attention_map\\" + str(exp.first().user) + "\\" + str(exp_id) + "\\" + \
                        str(page_data_id) + "\\"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            # 创建图片存储的目录
            # 如果目录不存在，则创建目录
            path_levels = [
                "semantic_attention_map\\" + str(exp.first().user) + "\\",
                "semantic_attention_map\\" + str(exp.first().user) + "\\" + str(exp_id) + "\\",
                "semantic_attention_map\\" + str(exp.first().user) + "\\" + str(exp_id) + "\\" +
                str(page_data_id) + "\\",
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

            # 读取word semantic feature，使用familiar score（[0,645]，越低越难）
            # 目前采取读csv方式获得semantic feature，之后需要修改为获取keybert模型输出
            word_fam_data = pd.read_csv("jupyter\\tmp_semantic_data\\word_train_processed_20230208.csv")
            word_fam_data = word_fam_data[(word_fam_map['exp_id'] == exp_id) &
                                          (word_fam_map['ent_flag'] < 0)]
            # for i in range(len(word_fam_data)):

            word_fam_data = dict(zip(word_fam_data['word'], word_fam_data['fam']))


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

    def add(self, word_feature):
        self.fixation_duration = np.sum([self.fixation_duration, word_feature.fixation_duration], axis=0).tolist()
        self.number_of_fixations = np.sum([self.number_of_fixations, word_feature.number_of_fixations], axis=0).tolist()
        self.reading_times = np.sum([self.reading_times, word_feature.reading_times], axis=0).tolist()

    def norm(self, userInfo):
        for i in range(self.num):
            self.fixation_duration[i] = (self.fixation_duration[
                                             i] - float(userInfo.fixation_duration_mean)) / float(
                userInfo.fixation_duration_var)
            self.number_of_fixations[i] = (self.number_of_fixations[
                                               i] - float(userInfo.number_of_fixations_mean)) / float(
                userInfo.number_of_fixations_var)
            self.reading_times[i] = (self.reading_times[i] - float(userInfo.reading_times_mean)) / float(
                userInfo.reading_times_var)

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

    def to_dataframe(self):
        data = pd.DataFrame({
            'fixation_duration': self.fixation_duration,
            'number_of_fixations': self.number_of_fixations,
            'reading_times': self.reading_times,
        })
        # print(data)
        return data


class SentFeature:

    def __init__(self, num, sent_list, word_list):
        super().__init__()
        self.num = num
        self.sent_list = sent_list
        self.word_list = word_list

        self.backward_times_of_sentence = [0 for _ in range(num)]
        self.forward_times_of_sentence = [0 for _ in range(num)]

        self.saccade_duration = [0 for _ in range(num)]
        self.saccade_times_of_sentence = [0 for _ in range(num)]

        self.total_dwell_time_of_sentence = [0 for _ in range(num)]

        self.backward_times_of_sentence_div_syllable = [0 for _ in range(num)]
        self.forward_times_of_sentence_div_syllable = [0 for _ in range(num)]
        self.saccade_duartion_div_syllable = [0 for _ in range(num)]
        self.saccade_times_of_sentence_div_syllable = [0 for _ in range(num)]
        self.total_dwell_time_of_sentence_div_syllable = [0 for _ in range(num)]

    def update(self):
        self.backward_times_of_sentence_div_syllable = self.div_syllable(self.backward_times_of_sentence)
        self.forward_times_of_sentence_div_syllable = self.div_syllable(self.forward_times_of_sentence)

        self.saccade_duration_div_syllable = self.div_syllable(self.saccade_duration)
        self.saccade_times_of_sentence_div_syllable = self.div_syllable(self.saccade_times_of_sentence)

        self.total_dwell_time_of_sentence_div_syllable = self.div_syllable(self.total_dwell_time_of_sentence)

    def add(self, sent_feature):
        self.backward_times_of_sentence_div_syllable = np.sum(
            [self.backward_times_of_sentence_div_syllable, sent_feature.backward_times_of_sentence_div_syllable],
            axis=0).tolist()
        self.forward_times_of_sentence_div_syllable = np.sum(
            [self.forward_times_of_sentence_div_syllable, sent_feature.forward_times_of_sentence_div_syllable],
            axis=0).tolist()
        self.saccade_duration_div_syllable = np.sum(
            [self.saccade_duration_div_syllable, sent_feature.saccade_duration_div_syllable], axis=0).tolist()
        self.saccade_times_of_sentence_div_syllable = np.sum(
            [self.saccade_times_of_sentence_div_syllable, sent_feature.saccade_times_of_sentence_div_syllable],
            axis=0).tolist()
        self.total_dwell_time_of_sentence_div_syllable = np.sum(
            [self.total_dwell_time_of_sentence_div_syllable, sent_feature.total_dwell_time_of_sentence_div_syllable],
            axis=0).tolist()

    def norm(self, userInfo):
        for i in range(self.num):
            self.backward_times_of_sentence_div_syllable[i] = (self.backward_times_of_sentence_div_syllable[
                                                                   i] - float(
                userInfo.backward_times_of_sentence_mean)) / float(userInfo.backward_times_of_sentence_var)
            self.forward_times_of_sentence_div_syllable[i] = (self.forward_times_of_sentence_div_syllable[
                                                                  i] - float(
                userInfo.forward_times_of_sentence_mean)) / float(userInfo.forward_times_of_sentence_var)
            self.saccade_duration_div_syllable[i] = (self.saccade_duration_div_syllable[
                                                         i] - float(userInfo.saccade_duration_mean)) / float(
                userInfo.saccade_duration_var)
            self.saccade_times_of_sentence_div_syllable[i] = (self.saccade_times_of_sentence_div_syllable[
                                                                  i] - float(
                userInfo.saccade_times_of_sentence_mean)) / float(userInfo.saccade_times_of_sentence_var)
            self.total_dwell_time_of_sentence_div_syllable[i] = (self.total_dwell_time_of_sentence_div_syllable[
                                                                     i] - float(
                userInfo.total_dwell_time_of_sentence_mean)) / float(userInfo.total_dwell_time_of_sentence_var)

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
            'saccade_duration_div_syllable': self.saccade_duration_div_syllable,
            'saccade_times_of_sentence_div_syllable': self.saccade_times_of_sentence_div_syllable,
            'total_dwell_time_of_sentence_div_syllable': self.total_dwell_time_of_sentence_div_syllable
        })
        # print(data)
        return data