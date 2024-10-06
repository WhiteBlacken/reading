import json
import logging
import math
import os
import time

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render
from loguru import logger

from analysis.feature import WordFeature, SentFeature, CNNFeature, FixationMap, RowArticle
from analysis.models import PageData, Experiment
from feature.utils import detect_fixations
from pyheatmap import myHeatmap
from tools import format_gaze, generate_fixations, generate_pic_by_base64, show_fixations, get_word_location, \
    paint_on_word, get_word_and_sentence_from_text, compute_label, textarea, get_fix_by_time, \
    get_item_index_x_y, get_item_index_x_y_new, is_watching, get_sentence_by_word, compute_sentence_label,\
    get_cnn_feature, get_row, get_euclid_distance, generate_fixations_in_skip_data, show_fixations_by_line, keep_row,  split_fixations
import cv2
from semantic_attention import get_word_familiar_rate, calculate_topic_related_score, calculate_keywords_score
import numpy as np

# Create your views here.


def dataset_new(request):
    start_time = time.time()

    experiments_id = available_exp_id()
    experiments = Experiment.objects.filter(id__in=experiments_id)

    experiments_time = time.time() - start_time

    datasets = {
        "uid": [],
        "page_id": [],
        "text_sequences": [],
        "eye_tracking_sequences": [],
        "labels": []
    }

    pages_search_time = []
    pages_process_time = []

    cnt = 0
    for experiment in experiments:
        start_time = time.time()

        pages = PageData.objects.filter(experiment_id=experiment.id)

        pages_search_time.append(time.time() - start_time)

        start_time = time.time()
        for page in pages:
            # text data
            text_sequence = text_data(page.texts, page.location)
            # eye tracking data
            # eye_tracking_sequence = eye_tracking_data(page.gaze_x, page.gaze_y, page.gaze_t)
            gaze_points = format_gaze(page.gaze_x, page.gaze_y, page.gaze_t)
            # 计算fixations
            fixations = detect_fixations(gaze_points)
            eye_tracking_seq = []
            for fix in fixations:
                eye_tracking_seq.append([round(fix[0], 2), round(fix[1], 2), round(fix[2], 2), round(fix[3], 2), round(fix[4], 2)])
            # labels 目前只处理word label
            try:
                wordLabels = json.loads(page.wordLabels)
            except:
                wordLabels = []

            labels = [1 if i in wordLabels else 0 for i in range(len(text_sequence))]

            datasets['uid'].append(experiment.user)
            datasets['page_id'].append(page.id)
            datasets['text_sequences'].append(text_sequence)
            datasets['eye_tracking_sequences'].append(eye_tracking_seq)
            datasets['labels'].append(labels)

            cnt += 1
            print(f"进度：已处理{cnt}条")
        pages_process_time.append(time.time() - start_time)

    start_time = time.time()
    pd.DataFrame(datasets).to_csv('raw_data_0414.csv')
    savefile_time = time.time() - start_time

    print(f"时间：查询experiment的时间为{experiments_time}")
    print(
        f"时间：查询page的总时间为{sum(pages_search_time)},平均时间为{sum(pages_search_time) / len(pages_search_time)}")
    print(
        f"时间：查询page的总时间为{sum(pages_process_time)},平均时间为{sum(pages_process_time) / len(pages_process_time)}")
    print(f"时间：存储csv的时间为{savefile_time}")

    return HttpResponse(1)


def eye_tracking_data(gaze_x, gaze_y, gaze_t):
    gaze_x = [int(float(item)) for item in gaze_x.split(",")]
    gaze_y = [int(float(item)) for item in gaze_y.split(",")]
    gaze_t = [int(float(item)) for item in gaze_t.split(",")]
    return [[gaze_x[i], gaze_y[i], gaze_t[i]] for i in range(len(gaze_x))]


def text_data(texts: str, locations: str):
    text_sequence = []
    words_list, _ = get_word_and_sentence_from_text(texts)
    locations = json.loads(locations)
    for i, word in enumerate(words_list):
        loc = locations[i]
        x, y, width, height = int(loc['left']), int(loc['top']), int(loc['right'] - loc['left']), int(
            loc['bottom'] - loc['top'])
        text_sequence.append([i, word, x, y, width, height])
    return text_sequence


def get_all_time_pic(request):
    exp_id = request.GET.get("exp_id")
    page_data_ls = PageData.objects.filter(experiment_id=exp_id)
    exp = Experiment.objects.get(id=exp_id)

    base_path = f"data/pic/all_time/{exp_id}/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    for page_data in page_data_ls:

        end = 500
        if page_data.id in [2818]:
            begin = 0
        if page_data.id in [2051, 2052, 2053, 2067, 1226, 1298, 1300, 2802, 2807, 2794]:
            end = 0

        print(f"page_id:{page_data.id}")
        # 拿到gaze point
        gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t, end_time=end)
        # 原始的fixations
        origin_fixations = detect_fixations(gaze_points)
        # 滤波数据
        keep_row_fixations = keep_row(origin_fixations)
        # 计算fixations
        result_fixations, _, _, _ = generate_fixations(
            gaze_points, page_data.texts, page_data.location, page_id=page_data.id
        )
        # 不使用空行假设
        _, row_level_fix_without_row_assumption, hit_rows = generate_fixations_in_skip_data(
            gaze_points, page_data.texts, page_data.location, page_id=page_data.id
        )
        
        print(f"row_level_fix:{row_level_fix_without_row_assumption}")
        word_locations = get_word_location(page_data.location)
        print(f"word_locations:{word_locations}")
        # 分行的单词
        word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
        word_features_by_line = get_word_feature_by_line(word_locations, word_list)
        adjust_fix_by_line(word_features_by_line, row_level_fix_without_row_assumption, hit_rows)

        # 生成图片
        path = f"{base_path}{page_data.id}/"

        # 如果目录不存在，则创建目录
        if not os.path.exists(path):
            os.mkdir(path)

        # 生成背景图
        background = generate_pic_by_base64(
            page_data.image, f"{path}background.png"
        )
        # 原始的fixation图
        fix_img = show_fixations(origin_fixations, background)
        cv2.imwrite(f"{path}fix-origin.png", fix_img)
        # 滤波后的图
        fix_img = show_fixations(keep_row_fixations, background)
        cv2.imwrite(f"{path}fix-lvbo.png", fix_img)
        # 生成调整后的fixation图
        fix_img = show_fixations(result_fixations, background)
        cv2.imwrite(f"{path}fix-adjust.png", fix_img)
        # 不使用空行假设的fixation图
        # fix_img = show_fixations(adjust_fixations_without_row_assumption, background)
        # cv2.imwrite(f"{path}fix-adjust-without-row-assumption.png", fix_img)
        fix_img = show_fixations_by_line(row_level_fix_without_row_assumption, background)
        cv2.imwrite(f"{path}fix-split.png", fix_img)

        # 序列切割
        fixations_seq_split_y_diff = split_fixations(gaze_points, page_data.location, "y_diff")
        print(f"size of fixations_seq_split_y_diff:{len(fixations_seq_split_y_diff)}")
        fix_img = show_fixations_by_line(fixations_seq_split_y_diff, background)
        cv2.imwrite(f"{path}fix-split-by-y-diff-all.png", fix_img)

        fix_seq_path = f"{path}fix_seq/"
        if not os.path.exists(fix_seq_path):
            os.mkdir(fix_seq_path)
        for i, fix_seq in enumerate(fixations_seq_split_y_diff):
            fix_img = show_fixations(fix_seq, background)
            cv2.imwrite(f"{fix_seq_path}fix-split-by-y-diff-{i}.png", fix_img)

        # 画热点图
        gaze_4_heat = [[x[0], x[1]] for x in result_fixations]
        myHeatmap.draw_heat_map(gaze_4_heat, f"{path}fix_heatmap.png", background)


        # 画duration图
        gaze_duration = []
        for fix in result_fixations:
            gaze_duration.extend([fix[0], fix[1]] for _ in range(fix[2] // 100))
        myHeatmap.draw_heat_map(gaze_duration, f"{path}duration_heatmap.png", background)

        # 画label TODO 合并成一个函数
        image = cv2.imread(background)
        
        # 1. 走神
        words_to_be_painted = []
        paras_wander = json.loads(page_data.wanderLabels) if page_data.wanderLabels else []
        for para in paras_wander:
            words_to_be_painted.extend(iter(range(para[0], para[1] + 1)))
        title = f"{str(page_data.id)}-{exp.user}-para_wander"
        pic_path = f"{path}para_wander.png"
        paint_on_word(image, words_to_be_painted, word_locations, title, pic_path)
        # 2. 单词不懂
        words_not_understand = json.loads(page_data.wordLabels) if page_data.wordLabels else []
        title = f"{str(page_data.id)}-{exp.user}-words_not_understand"
        word_pic_path = f"{path}words_not_understand.png"
        paint_on_word(image, words_not_understand, word_locations, title, word_pic_path)
        # 3. 句子不懂
        sentences_not_understand = json.loads(page_data.sentenceLabels) if page_data.sentenceLabels else []
        words_to_painted = []
        for sentence in sentences_not_understand:
            words_to_painted.extend(iter(range(sentence[0], sentence[1])))
        title = f"{str(page_data.id)}-{exp.user}-sentences_not_understand"
        pic_path = f"{path}sentences_not_understand.png"
        paint_on_word(image, words_to_painted, word_locations, title, pic_path)


        # 序列与三行映射
        # 切割的可以直接用
        if False:
            fix_seq_path = f"{path}fix_seq_in_three_domain/"
            if not os.path.exists(fix_seq_path):
                os.mkdir(fix_seq_path)
            print(f"fixations_seq_split_y_diff:{fixations_seq_split_y_diff}")
            border, rows, danger_zone, len_per_word = textarea(page_data.location)
            print(f"this rows:{rows}")
            for fix_id, fix_seq in enumerate(fixations_seq_split_y_diff):
                y_mean = sum([x[1] for x in fix_seq]) / len(fix_seq)
                # 查看匹配的行
                hit_row = 0
                for i, row in enumerate(rows):
                    if row['bottom'] >= y_mean >= row['top']:
                        hit_row = i
                        break
                    if y_mean < rows[0]['top']:
                        hit_row = 0
                    if y_mean > rows[-1]['bottom']:
                        hit_row = len(rows) - 1
                possible_rows = [x for x in range(hit_row-1,hit_row+2) if x >= 0 and x < len(rows)]
                print(f"this possible_rows:{possible_rows}")
                for i, possible_row in enumerate(possible_rows):
                    row_y = (rows[possible_row]['top'] + rows[possible_row]['bottom']) / 2
                    fix_to_pic = [[x[0],row_y,x[2]] for x in fix_seq]
                    gaze_duration = []
                    for fix in fix_to_pic:
                        gaze_duration.extend([fix[0], fix[1]] for _ in range(fix[2] // 100))
                    final_pic_path = f"{fix_seq_path}fix-split-by-y-diff-{fix_id}.png"
                    if i == 0:
                        myHeatmap.draw_heat_map(gaze_duration, final_pic_path, word_pic_path)
                    else:
                        myHeatmap.draw_heat_map(gaze_duration, final_pic_path, final_pic_path)

        # 画语义图
        print(f"sentence_list:{sentence_list}")
        print(f"word_list:{word_list}")
        print(f"word_location:{word_locations}")
        assert len(word_list) == len(word_locations)
        topic_score_dict = calculate_topic_related_score(page_data.texts)
        keywords_dict = calculate_keywords_score(page_data.texts)
        print(f"topic_score_dict:{topic_score_dict}")
        semantic_path = f"{path}semantic_path/"
        if not os.path.exists(semantic_path):
            os.mkdir(semantic_path)
        familiar_rate_seq = []
        topic_score_seq = []
        keyword_score_seq = []
        if len(word_locations) > 0:
            y_loc = word_locations[0][1]

        row_idx = 0
        for i, location in enumerate(word_locations):
            if location[1] != y_loc:
                semantic_familiar_pic_save_path = f"{semantic_path}semantic_familiar_rate_row_{row_idx}.png"
                semantic_topic_pic_save_path = f"{semantic_path}semantic_topic_rate_row_{row_idx}.png"
                semantic_keyword_pic_save_path = f"{semantic_path}semantic_keyword_row_{row_idx}.png"
                myHeatmap.draw_heat_map(familiar_rate_seq, semantic_familiar_pic_save_path, word_pic_path)
                myHeatmap.draw_heat_map(topic_score_seq, semantic_topic_pic_save_path, word_pic_path)
                myHeatmap.draw_heat_map(keyword_score_seq, semantic_keyword_pic_save_path, word_pic_path)
                familiar_rate_seq = []
                topic_score_seq = []
                keyword_score_seq = []
                row_idx += 1
            x, y = (word_locations[i][0] + word_locations[i][2]) // 2, (word_locations[i][1] + word_locations[i][3]) // 2
            familiar_rate_seq.extend([x,y] for _ in range(get_word_familiar_rate(word_list[i])//10))
            if word_list[i] in topic_score_dict:
                topic_score_seq.extend([x, y] for _ in range(int(topic_score_dict[word_list[i]]*20)))
            if word_list[i] in keywords_dict:
                keyword_score_seq.extend([x, y] for _ in range(int(keywords_dict[word_list[i]] * 20)))

            y_loc = location[1]

        if len(familiar_rate_seq) > 0:
            semantic_familiar_pic_save_path = f"{semantic_path}semantic_familiar_rate_row_{row_idx}.png"
            semantic_topic_pic_save_path = f"{semantic_path}semantic_topic_rate_row_{row_idx}.png"
            semantic_keyword_pic_save_path = f"{semantic_path}semantic_keyword_row_{row_idx}.png"
            myHeatmap.draw_heat_map(familiar_rate_seq, semantic_familiar_pic_save_path, word_pic_path)
            myHeatmap.draw_heat_map(topic_score_seq, semantic_topic_pic_save_path, word_pic_path)
            myHeatmap.draw_heat_map(keyword_score_seq, semantic_keyword_pic_save_path, word_pic_path)
    return HttpResponse(1)


def dataset_of_timestamp(request):
    """按照时间切割数据集"""
    filename = "native.txt"
    file = open(filename, 'r')
    lines = file.readlines()

    experiment_list_select = list(lines)

    # 获取切割的窗口大小
    interval = request.GET.get("interval", 8)
    interval = interval * 1000
    # 确定文件路径
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d")

    base_path = f"data\\dataset\\{now}\\"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    word_feature_path = f"{base_path}word-feature-{now}-{len(experiment_list_select)}.csv"
    sent_feature_path = f"{base_path}sent-feature-{now}-{len(experiment_list_select)}.csv"
    cnn_feature_path = f"{base_path}cnn-feature-{now}-{len(experiment_list_select)}.csv"
    fixations_map_path = f"{base_path}fixation-map-{now}-{len(experiment_list_select)}.csv"
    # 获取需要生成的实验
    # experiment_list_select = [1011,1792]
    experiments = Experiment.objects.filter(id__in=experiment_list_select)

    cnnFeature = CNNFeature()
    fixationMap = FixationMap()  # 用来记录画时刻图的信息

    success = 0
    fail = 0

    logger.info(f"本次生成{len(experiment_list_select)}条")
    for experiment in experiments:
        # try:
        time = 0  # 记录当前的时间
        page_data_list = PageData.objects.filter(experiment_id=experiment.id)
        # 创建不同页的信息
        word_feature_list = []
        sent_feature_list = []
        for page_data in page_data_list:
            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            # word_level
            wordFeature = WordFeature(len(word_list))
            wordFeature.word_list = word_list  # 填充单词
            wordFeature.word_understand, wordFeature.sentence_understand, wordFeature.mind_wandering = compute_label(
                page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
            )  # 填充标签
            for i, word in enumerate(word_list):
                sent_index = get_sentence_by_word(i, sentence_list)
                wordFeature.sentence_id[i] = sent_index  # 使用page_id,time,sentence_id可以区分
            word_feature_list.append(wordFeature)
            # sentence_level
            sentFeature = SentFeature(len(sentence_list))
            sentFeature.sentence = [sentence[0] for sentence in sentence_list]
            sentFeature.sentence_id = list(range(len(sentence_list)))  # 记录id
            # todo 句子标签的生成
            sentFeature.sentence_understand, sentFeature.mind_wandering = compute_sentence_label(
                page_data.sentenceLabels, page_data.wanderLabels, sentence_list)
            sent_feature_list.append(sentFeature)

        for p, page_data in enumerate(page_data_list):
            wordFeature = word_feature_list[p]  # 获取单词特征
            sentFeature = sent_feature_list[p]  # 获取句子特征

            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            border, rows, danger_zone, len_per_word = textarea(page_data.location)

            end = 500
            if page_data.id in [2051, 2052, 2053, 2067, 1226, 1298, 1300, 2807]:
                end = 0

            gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t, end_time=end)
            result_fixations, row_sequence, row_level_fix, sequence_fixations = generate_fixations(
                gaze_points, page_data.texts, page_data.location, page_id=page_data.id
            )

            pre_gaze = 0
            for g, gaze in enumerate(gaze_points):
                if g == 0:
                    continue
                if gaze[-1] - gaze_points[pre_gaze][-1] > interval:  # 按照interval切割gaze
                    # 把当前页的特征清空，因为要重新算一遍特征
                    wordFeature.clean()
                    sentFeature.clean()
                    # 目的是为了拿到gaze的时间，来切割fixation，为什么不直接gaze->fixation,会不准 todo 实时处理
                    fixations_before = get_fix_by_time(result_fixations, start=0, end=gaze[-1])
                    fixations_now = get_fix_by_time(result_fixations, gaze_points[pre_gaze][-1], gaze[-1])
                    # 计算特征

                    pre_word_index = -1
                    for f, fixation in enumerate(fixations_before):
                        word_index, isAdjust = get_item_index_x_y(json.loads(page_data.location), fixation[0],
                                                                  fixation[1])
                        if word_index != -1:
                            wordFeature.number_of_fixation[word_index] += 1
                            wordFeature.total_fixation_duration[word_index] += fixation[2]
                            if word_index != pre_word_index:  # todo reading times的计算
                                wordFeature.reading_times[word_index] += 1

                            sent_index = get_sentence_by_word(word_index, sentence_list)
                            if sent_index != -1:
                                sentFeature.total_dwell_time[sent_index] += fixation[2]
                                # if f!=0:
                                if pre_word_index != word_index:
                                    sentFeature.saccade_times[sent_index] += 1  # 将两个fixation之间都作为saccade

                                    if pre_word_index - word_index >= 0:  # 往后看,阈值暂时设为1个单词
                                        sentFeature.backward_saccade_times[sent_index] += 1
                                    if pre_word_index - word_index < 0:  # 往前阅读（正常阅读顺序)
                                        sentFeature.forward_saccade_times[sent_index] += 1

                                    sentFeature.saccade_duration[sent_index] += fixations_before[f][3] - \
                                                                                fixations_before[f - 1][4]  # 3是起始，4是结束
                                    sentFeature.saccade_velocity[sent_index] += get_euclid_distance(
                                        (fixations_before[f][0], fixations_before[f][1]),
                                        (fixations_before[f - 1][0], fixations_before[f - 1][1]))  # 记录的实际上是距离
                                    pre_row = get_row(pre_word_index, rows)
                                    now_row = get_row(word_index, rows)
                                    if pre_row == now_row:
                                        sentFeature.horizontal_saccade_proportion[sent_index] += 1  # 记录的实际上是次数

                            pre_word_index = word_index  # todo important
                    # 计算need prediction
                    wordFeature.need_prediction = is_watching(fixations_now, json.loads(page_data.location),
                                                              wordFeature.num)
                    # 生成数据
                    for feature in word_feature_list:
                        feature.to_csv(word_feature_path, experiment.id, page_data.id, time, experiment.user,
                                       experiment.article_id)

                    for feature in sent_feature_list:
                        feature.to_csv(sent_feature_path, experiment.id, page_data.id, time, experiment.user,
                                       experiment.article_id)

                    # cnn feature的生成 todo 暂时不变，之后修改
                    get_cnn_feature(time, cnnFeature, gaze_points[pre_gaze:g], experiment.id, fixations_now)

                    # 记录每个时刻的眼动，用于画图
                    fixationMap.update(time, experiment.id, page_data.id, fixations_now)

                    time += 1
                    pre_gaze = g  # todo important

        success += 1
        logger.info(f"成功生成{success}条,失败{fail}条")
        # except Exception:
        #     fail += 1

    # 生成exp相关信息
    cnnFeature.to_csv(cnn_feature_path)
    fixationMap.to_csv(fixations_map_path)

    return HttpResponse(1)


def get_part_time_pic(request):
    time = request.GET.get('time')
    exp_id = request.GET.get('exp_id')
    base_path = f"data\\pic\\part_time\\{exp_id}\\"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d")
    page_csv = pd.read_csv(f'results\\{now}\\fixation-map-{now}.csv')

    page_row = page_csv[(page_csv['exp_id'] == int(exp_id)) & (page_csv['time'] == int(time))]

    page_id = page_row['page_id'].iloc[0]
    fixations = json.loads(page_row['fixation'].iloc[0])

    base_path = f"{base_path}\\{page_id}\\"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    if page_datas := PageData.objects.filter(id=page_id):
        page_data = page_datas.first()
        background = generate_pic_by_base64(
            page_data.image, f"{base_path}background.png"
        )

        fix_img = show_fixations(fixations, background)

        cv2.imwrite(f"{base_path}fix-{time}.png", fix_img)

    return HttpResponse(1)


def dataset_of_all_time(request):
    # """按照时间切割数据集"""
    # filename = "exps/data1.txt"
    # file = open(filename, 'r')
    # lines = file.readlines()

    # experiment_list_select = list(lines)

    # filename = "exps/data2.txt"
    # file = open(filename, 'r')
    # lines1 = file.readlines()
    # experiment_list_select.extend(list(lines1))

    # filename = "exps/data3.txt"
    # file = open(filename, 'r')
    # lines2 = file.readlines()
    # experiment_list_select.extend(list(lines2))
    # print(f"lens:{len(experiment_list_select)}")

    skip_page_list = [1589, 1825, 1929, 1960, 1985, 2020, 2640, 2650, 2653, 2671, 2682, 2691, 2713, 2726, 2738, 2749, 2750, 2751, 2752, 2753, 2754, 2762, 2763, 2771, 2773, 2775, 2782, 2784, 2785, 2787, 2788]
    # experiments = Experiment.objects.filter(id__in=experiment_list_select)

    exp_ids = [x.experiment_id for x in PageData.objects.filter(id__in=skip_page_list)]
    exp_ids = list(set(exp_ids))
    experiments = Experiment.objects.filter(id__in=exp_ids)
    experiment_list_select = experiments
    #
    # experiment_list_select = [1889, 1890, 1892, 1896]
    # filename = "native.txt"
    # file = open(filename, 'r')
    # lines = file.readlines()

    # experiment_list_select = list(lines)
    # 确定文件路径
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d")

    logger.info(f"本次生成{len(experiment_list_select)}条")

    base_path = f"data/dataset/{now}/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    word_feature_path = f"{base_path}all-word-feature-{now}-{len(experiment_list_select)}.csv"
    sent_feature_path = f"{base_path}all-sent-feature-{now}-{len(experiment_list_select)}.csv"
    cnn_feature_path = f"{base_path}all-cnn-feature-{now}-{len(experiment_list_select)}.csv"
    # 获取需要生成的实验
    # experiment_list_select = [1011,1792]
    experiments = Experiment.objects.filter(id__in=experiment_list_select)

    cnnFeature = CNNFeature()

    success = 0
    fail = 0

    

    for experiment in experiments:
        # try:
        time = 0  # 记录当前的时间
        page_data_list = PageData.objects.filter(experiment_id=experiment.id)
        # 创建不同页的信息
        word_feature_list = []
        sent_feature_list = []
        for page_data in page_data_list:
            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            # word_level
            wordFeature = WordFeature(len(word_list))
            wordFeature.word_list = word_list  # 填充单词
            wordFeature.word_understand, wordFeature.sentence_understand, wordFeature.mind_wandering = compute_label(
                page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
            )  # 填充标签
            for i, word in enumerate(word_list):
                sent_index = get_sentence_by_word(i, sentence_list)
                wordFeature.sentence_id[i] = sent_index  # 使用page_id,time,sentence_id可以区分
            word_feature_list.append(wordFeature)
            # sentence_level
            sentFeature = SentFeature(len(sentence_list))
            sentFeature.sentence = [sentence[0] for sentence in sentence_list]
            sentFeature.sentence_id = list(range(len(sentence_list)))  # 记录id
            # todo 句子标签的生成
            sentFeature.sentence_understand, sentFeature.mind_wandering = compute_sentence_label(
                page_data.sentenceLabels, page_data.wanderLabels, sentence_list)
            sent_feature_list.append(sentFeature)

        cnn_gaze_points = []
        cnn_result_fixations = []
        for p, page_data in enumerate(page_data_list):
            wordFeature = word_feature_list[p]  # 获取单词特征
            sentFeature = sent_feature_list[p]  # 获取句子特征

            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            border, rows, danger_zone, len_per_word = textarea(page_data.location)

            end = 500
            if page_data.id in [2051, 2052, 2053, 2067, 1226, 1298, 1300]:
                end = 0
            gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t, end_time=end)
            cnn_gaze_points.extend(gaze_points)

            result_fixations, row_sequence, row_level_fix, sequence_fixations = generate_fixations(
                gaze_points, page_data.texts, page_data.location
            )
            cnn_result_fixations.extend(result_fixations)
            # 记录是否已过first pass
            first_pass = [0 for _ in sentence_list]
            reach_medium_times = [0 for _ in sentence_list]
            max_reach_index = [0 for _ in sentence_list]

            # 计算特征
            pre_word_index = -1
            for f, fixation in enumerate(result_fixations):
                word_index, isAdjust = get_item_index_x_y(json.loads(page_data.location), fixation[0], fixation[1])
                if word_index != -1:
                    wordFeature.number_of_fixation[word_index] += 1
                    wordFeature.total_fixation_duration[word_index] += fixation[2]
                    if word_index != pre_word_index:  # todo reading times的计算
                        wordFeature.reading_times[word_index] += 1

                    sent_index = get_sentence_by_word(word_index, sentence_list)
                    if sent_index != -1:
                        sentFeature.total_dwell_time[sent_index] += fixation[2]
                        if first_pass[sent_index] < 2:
                            sentFeature.first_pass_total_dwell_time[sent_index] += fixation[2]
                        # if f!=0:
                        if pre_word_index != word_index:
                            sentFeature.saccade_times[sent_index] += 1  # 将两个fixation之间都作为saccade
                            if first_pass[sent_index] < 2:
                                sentFeature.first_pass_saccade_times[sent_index] += 1

                            if pre_word_index - word_index > 0:  # 往后看,阈值暂时设为1个单词
                                sentFeature.backward_saccade_times[sent_index] += 1
                                if first_pass[sent_index] < 2:
                                    sentFeature.first_pass_backward_saccade_times[sent_index] += 1
                            if pre_word_index - word_index < 0:  # 往前阅读（正常阅读顺序)
                                sentFeature.forward_saccade_times[sent_index] += 1
                                if first_pass[sent_index] < 2:
                                    sentFeature.first_pass_forward_saccade_times[sent_index] += 1

                            sentFeature.saccade_duration[sent_index] += result_fixations[f][3] - \
                                                                        result_fixations[f - 1][4]  # 3是起始，4是结束
                            sentFeature.saccade_velocity[sent_index] += get_euclid_distance(
                                (result_fixations[f][0], result_fixations[f][1]),
                                (result_fixations[f - 1][0], result_fixations[f - 1][1]))  # 记录的实际上是距离
                            if first_pass[sent_index] < 2:
                                sentFeature.first_pass_saccade_duration[sent_index] += result_fixations[f][3] - \
                                                                                       result_fixations[f - 1][
                                                                                           4]  # 3是起始，4是结束
                                sentFeature.first_pass_saccade_velocity[sent_index] += get_euclid_distance(
                                    (result_fixations[f][0], result_fixations[f][1]),
                                    (result_fixations[f - 1][0], result_fixations[f - 1][1]))  # 记录的实际上是距离

                            pre_row = get_row(pre_word_index, rows)
                            now_row = get_row(word_index, rows)
                            if pre_row == now_row:
                                sentFeature.horizontal_saccade_proportion[sent_index] += 1  # 记录的实际上是次数
                                if first_pass[sent_index] < 2:
                                    sentFeature.first_pass_horizontal_saccade_proportion[sent_index] += 1  # 记录的实际上是次数

                            if word_index > max_reach_index[sent_index]:
                                max_reach_index[sent_index] = word_index

                            sentence_now = sentence_list[sent_index]
                            # 相关度
                            # words = word_list[sentence_now[1]:sentence_now[2]]
                            # diffs = [get_word_familiar_rate(word) for word in words]
                            # diffs = normalize_list(diffs)
                            # reading_times_norm = normalize_list(wordFeature.reading_times[sentence_now[1]:sentence_now[2]])
                            # number_of_fixations_norm = normalize_list(wordFeature.number_of_fixation[sentence_now[1]:sentence_now[2]])
                            # total_fixation_duration_norm = normalize_list(wordFeature.total_fixation_duration[sentence_now[1]:sentence_now[2]])
                            #
                            # sentFeature.reading_times_cor[sent_index] = multiply_and_sum_lists(diffs,reading_times_norm)
                            # sentFeature.number_of_fixation_cor[sent_index] = multiply_and_sum_lists(diffs,number_of_fixations_norm)
                            # sentFeature.total_fixation_duration_cor[sent_index] = multiply_and_sum_lists(diffs,total_fixation_duration_norm)
                            #
                            # if first_pass[sent_index]  < 2:
                            #     sentFeature.first_pass_reading_times_cor[sent_index] = multiply_and_sum_lists(diffs,
                            #                                                                        reading_times_norm)
                            #     sentFeature.first_pass_number_of_fixation_cor[sent_index] = multiply_and_sum_lists(diffs,
                            #                                                                             number_of_fixations_norm)
                            #     sentFeature.first_pass_total_fixation_duration_cor[sent_index] = multiply_and_sum_lists(diffs,
                            #                                                                                  total_fixation_duration_norm)
                            #

                            # 计算是否为first_pass
                            word_now_loc = (word_index - sentence_now[1]) / sentence_now[3]
                            if word_now_loc > 0.5:
                                reach_medium_times[sent_index] += 1
                            if reach_medium_times[sent_index] > 3 and word_now_loc < 0.3:
                                first_pass[sent_index] += 1

                    pre_word_index = word_index  # todo important
            # 计算need prediction
            wordFeature.need_prediction = is_watching(result_fixations, json.loads(page_data.location), wordFeature.num)
            # 生成数据
            wordFeature.to_csv(word_feature_path, experiment.id, page_data.id, time, experiment.user,
                               experiment.article_id)

            sentFeature.to_csv(sent_feature_path, experiment.id, page_data.id, time, experiment.user,
                               experiment.article_id)

        # cnn feature的生成 todo 暂时不变，之后修改
        get_cnn_feature(time, cnnFeature, cnn_gaze_points, experiment.id, cnn_result_fixations)

        time += 1

        success += 1
        logger.info(f"成功生成{success}条,失败{fail}条")
        # except Exception:
        #     fail += 1

    # 生成exp相关信息
    cnnFeature.to_csv(cnn_feature_path)

    return HttpResponse(1)


def count_label(request):
    filename = "exp.txt"
    file = open(filename, 'r')
    lines = file.readlines()

    pagedatas = PageData.objects.filter(experiment_id__in=list(lines))

    label = sum(len(json.loads(page.sentenceLabels)) for page in pagedatas)
    return HttpResponse(label)


def get_word_index(request):
    page_id = request.GET.get("page_id")
    page_data = PageData.objects.get(id=page_id)
    input = request.GET.get("word")
    word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
    res = "".join(str(i) + "," for i, word in enumerate(word_list) if word == input)
    return HttpResponse(res)


def sent_domain(request):
    page_id = request.GET.get("page_id")
    sent_id = int(request.GET.get("sent"))
    page_data = PageData.objects.get(id=page_id)
    word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
    res = f"[{sentence_list[sent_id][1]},{sentence_list[sent_id][2]}]"

    return HttpResponse(res)


def available_exp_id():
    filenames = ["exps/data1.txt", "exps/data2.txt", "exps/data3.txt"]
    experiments_id = []
    for filename in filenames:
        lines = open(filename, 'r').readlines()
        experiments_id.extend(list(lines))
    experiments_id = list(set(experiments_id))
    logger.info(f"实验id的数量为：{len(experiments_id)}")
    return experiments_id

def get_word_feature_by_line(word_locations, word_list):
    word_features_by_line = []
    tmp = []
    size = 0
    for i, loc in enumerate(word_locations):
        if len(tmp) == 0:
            tmp.append(WordFeatureByLine(loc[0], loc[1], loc[2], loc[3], word_list[i]))
            continue
        if loc[1] != tmp[0].y:
            size += len(tmp)
            word_features_by_line.append([x for x in tmp])
            tmp = []
        tmp.append(WordFeatureByLine(loc[0], loc[1], loc[2], loc[3], word_list[i]))
    if len(tmp) > 0:
        size += len(tmp)
        word_features_by_line.append([x for x in tmp])
    print(f"size:{size}")
    return word_features_by_line


class WordFeatureByLine:
    def __init__(self, x, y, width, height, word) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.word = word
        self.length = len(word)
        pass

def adjust_fix_by_line(word_features_by_line, row_level_fix_without_row_assumption, hit_rows):
    # 单词行级特征/fix行级特征/提前计算好的命中行
    win = 1
    for seqIdx, row_fix in enumerate(row_level_fix_without_row_assumption):
        # 确定最大可能性的row
        hitRow = hit_rows[seqIdx]
        domainRow = [i for i in range(hitRow-win, hitRow+2) if i >= 0 and i < len(word_features_by_line)]
        print(f"domainRow:{domainRow}")
        result_vals = []
        for row in domainRow:
            # 看row_fix落在哪个单词上, 只看x轴
            word_features = [x.length for x in word_features_by_line[row]]
            word_features = pooling(word_features, 2)
            print(f"pooling_word_feature:{word_features}")
            # break
            fix_features = [0 for x in word_features_by_line[row]]
            for fixIdx, fix in enumerate(row_fix): 
                for wordIdx, word in enumerate(word_features_by_line[row]):
                    if fix[0] >= word.x and fix[0] <= word.x + word.width:
                        if wordIdx == 0:
                            print(f"fix_expect:{fix, fixIdx}")
                            print(f"word:{word.x, word.width, word.word}")
                        fix_features[wordIdx] += 1
                        break
            # word_features, fix_features = normalize_list_numpy(word_features), normalize_list_numpy(fix_features)
            result_vals.append(sum([x * y for x, y in zip(word_features, fix_features)]))
            print(f"word_features_{row}:{word_features}")
            print(f"fix_features_{row}:{fix_features}")
        max_possible_row = domainRow[result_vals.index(max(result_vals))]
        print(f"result_vals:{result_vals}")
        print(f"max_possible_row:{max_possible_row}")

        # 调整fix
        adjust_y = (word_features_by_line[max_possible_row][0].y + word_features_by_line[max_possible_row][0].height)/2
        for i, fix in enumerate(row_level_fix_without_row_assumption[seqIdx]):
            row_level_fix_without_row_assumption[seqIdx][i][1] = adjust_y

def normalize_list_numpy(lst):
    arr = np.array(lst)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def pooling(data: list, window_size: int) -> list:
    pooled_features = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        window = data[start:end]
        # 这里以计算平均值作为池化操作，您可以根据需要修改为其他池化方法，如最大值、总和等
        pooled_value = sum(window) / len(window)
        pooled_features.append(pooled_value)
    return pooled_features


def dataset_of_all_time_for_skip(request):
    """按照时间切割数据集"""
    experiment_list_select = [1924]
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d")

    logger.info(f"本次生成{len(experiment_list_select)}条")

    base_path = f"data/dataset/{now}/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    skip_page_list = [1226, 1227, 1236, 1237, 1247, 1248, 1249, 1250, 1298, 1299, 1300, 1323, 1324, 1588, 1590, 1591, 1592, 1593, 1642, 1643, 1686, 1687, 1692, 1693, 1699, 1700, 1701, 1702, 1742, 1743, 1745, 1747, 1794, 1795, 1807, 1808, 1819, 1820, 1821, 1822, 1823, 1824, 1826, 1831, 1860, 1861, 1862, 1863, 1926, 1927, 1930, 1931, 1948, 1949, 1950, 1952, 1951, 1953, 1959, 1966, 1967, 1980, 1981, 1984, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2035, 2036, 2044, 2045, 2046, 2047, 2051, 2052, 2053, 2066, 2067, 2639, 2641, 2644, 2645, 2649, 2651, 2646, 2648, 2647, 2652, 2654, 2655, 2656, 2658, 2660, 2657, 2659, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2670, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2692, 2693, 2704, 2705, 2706, 2707, 2712, 2714, 2716, 2715, 2717, 2720, 2721, 2722, 2723, 2724, 2725, 2729, 2730, 2731, 2732, 2733, 2736, 2737, 2739, 2744, 2745, 2746, 2747, 2748, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2772, 2774, 2776, 2777, 2778, 2779, 2780, 2781, 2783, 2786, 2789, 2790, 2791]
    skip_page_list = [1589, 1825, 1929, 1960, 1985, 2020, 2640, 2650, 2653, 2671, 2682, 2691, 2713, 2726, 2738, 2749, 2750, 2751, 2752, 2753, 2754, 2762, 2763, 2771, 2773, 2775, 2782, 2784, 2785, 2787, 2788]
    # experiments = Experiment.objects.filter(id__in=experiment_list_select)

    exp_ids = [x.experiment_id for x in PageData.objects.filter(id__in=skip_page_list)]
    exp_ids = list(set(exp_ids))
    experiments = Experiment.objects.filter(id__in=exp_ids)

    word_feature_path = f"{base_path}all-word-feature-{now}-{len(experiments)}-row-formart.csv"
    sent_feature_path = f"{base_path}all-sent-feature-{now}-{len(experiments)}-row-formart.csv"
    cnn_feature_path = f"{base_path}all-cnn-feature-{now}-{len(experiments)}-row-formart.csv"

    row_article_path = f"{base_path}row_article-{now}-{len(experiments)}-row-formart.csv"


    for experiment in experiments:
        page_data_list = PageData.objects.filter(experiment_id=experiment.id)
        print(f"page_data_list:{[x.id for x in page_data_list]}")
        row_idx = 0
        fix_idx = 0
        # 创建不同页的信息
        for page_data in page_data_list:
            gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t, end_time=0)
            # 按行切割的眼动
            _, row_level_fix_without_row_assumption, hit_rows = generate_fixations_in_skip_data(
                gaze_points, page_data.texts, page_data.location, page_id=page_data.id
            )
            print(f"row_level_fix_without_row_assumption:{row_level_fix_without_row_assumption[:5]}")
            border, rows, danger_zone, len_per_word = textarea(page_data.location)
            if len(rows) <= 3: continue

            try:
                assert len(row_level_fix_without_row_assumption) == len(hit_rows)
            except Exception as e:
                continue
            # 按行切割的文本
            word_list, _ = get_word_and_sentence_from_text(page_data.texts)
            location =  json.loads(page_data.location)
            text_rows = split_text_by_row(location, word_list)
            # label
            word_label, _, _ = compute_label( # 全量的，需要按行截取
                page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
            )  # 填充标签
            word_label_by_rows = get_word_label_by_row(word_label, text_rows)
            word_list_by_rows = get_word_list_by_row(word_list, text_rows)

            # 生成文本集
            rowArticle = RowArticle(len(text_rows))
            rowArticle.article_id = experiment.article_id
            rowArticle.row_idx = [i+row_idx for i in range(len(text_rows))]
            rowArticle.row_text = word_list_by_rows
            rowArticle.row_label = word_label_by_rows
            rowArticle.experiment_id = experiment.id
            rowArticle.to_csv(row_article_path)

            fixations_seq_split_y_diff = split_fixations(gaze_points, page_data.location, "y_diff")
            hit_rows = get_hit_row_new(fixations_seq_split_y_diff, rows)
            assert len(fixations_seq_split_y_diff) == len(hit_rows)
            print(f"hit_rows:{hit_rows}")
            # 处理每一个fix_seq
            for fix_seq_id, fix_seq in enumerate(fixations_seq_split_y_diff):
                possible_rows = hit_rows[fix_seq_id]
                assert len(possible_rows) == 3
                for row in possible_rows:
                    adjust_y = (text_rows[row][0][2] + text_rows[row][0][4]) / 2 # word, left, top, right, bottom
                    fix_seq = [[fix[0], adjust_y, fix[2], fix[3], fix[4]] for fix in fix_seq]
                    # 调整fix到每一行
                    word_feature = get_word_feature(fix_seq, text_rows[row], word_label_by_rows[row])

                    fix_seq_id_list = [fix_seq_id+fix_idx for _ in range(word_feature.num)]
                    word_feature.fix_seq_id_list = fix_seq_id_list
                    row_idx_list = [row+row_idx for _ in range(word_feature.num)]
                    word_feature.row_idx_list = row_idx_list

                    word_feature.word_list = word_list_by_rows[row]

                    word_feature.to_csv(word_feature_path, experiment.id, page_data.id, time, experiment.user,
                               experiment.article_id)
                    
                    
            row_idx = rowArticle.row_idx[-1] + 1
            fix_idx = word_feature.fix_seq_id_list[-1] + 1 # 这个word_feature要注意
    return HttpResponse(1)
      

def split_text_by_row(locations, word_list):
    text_rows = []
    row_data = []
    assert len(locations) == len(word_list)
    if len(locations) == 0:
        return text_rows
    top = locations[0]["top"]
    for i, loc in enumerate(locations):
        if loc["top"] != top:
            text_rows.append(row_data)
            row_data = []
        row_data.append([word_list[i], loc["left"], loc["top"], loc["right"], loc["bottom"]])
        top = loc["top"]
    if len(row_data) > 0:
        text_rows.append(row_data)
    return text_rows
    
def get_word_feature(fix_seq, text_row, word_label):
    wordFeature = WordFeature(len(text_row))
    wordFeature.word_understand = word_label
    # 计算特征
    pre_word_index = -1
    for i, fixation in enumerate(fix_seq):
        word_index, _ = get_item_index_x_y_new(text_row, fixation[0], fixation[1])
        if word_index != -1:
            wordFeature.number_of_fixation[word_index] += 1
            wordFeature.total_fixation_duration[word_index] += fixation[2]
            if word_index != pre_word_index:
                wordFeature.reading_times[word_index] += 1
                # 句子级别
                wordFeature.saccade_times_of_sentence_one_word += 1
                if i > 0:
                    wordFeature.saccade_duration_one_word += fix_seq[i][3] - fix_seq[i-1][4]
                    print(fix_seq)
                    # raise KeyError
                    if fix_seq[i][3] - fix_seq[i-1][4] != 0:
                        if fix_seq[i][3] - fix_seq[i-1][4] < 0:
                            print("error----")
                            # print(fix_seq)
                            # raise KeyError
                        wordFeature.saccade_velocity_one_word += (get_fix_distance(fix_seq[i-1], fix_seq[i])) / (fix_seq[i][3] - fix_seq[i-1][4]) # error 是duration，不是timestamp
                if pre_word_index > word_index:
                    wordFeature.backward_times_of_sentence_one_word += 1
                if pre_word_index < word_index:
                    wordFeature.forward_times_of_sentence_one_word += 1
                pre_word_index = word_index
    wordFeature.horizontal_saccade_proportion_one_word = 1
    wordFeature.total_dwell_time_of_sentence_one_word = fix_seq[-1][3] - fix_seq[0][3]
    print(f"backward_times_of_sentence_one_word:{wordFeature.backward_times_of_sentence_one_word},forward_times_of_sentence_one_word:{wordFeature.forward_times_of_sentence_one_word}")
    return wordFeature



def get_word_label_by_row(word_label, text_rows):
    word_label_by_rows = []
    begin = 0
    for row in text_rows:
        end = begin + len(row)
        word_label_by_rows.append(word_label[begin: end])
        begin = end
    assert len(text_rows) == len(word_label_by_rows)
    assert len(text_rows[0]) == len(word_label_by_rows[0])
    return word_label_by_rows

def get_word_list_by_row(word_list, text_rows):
    word_list_by_rows = []
    begin = 0
    for row in text_rows:
        end = begin + len(row)
        word_list_by_rows.append(word_list[begin: end])
        begin = end
    assert len(text_rows) == len(word_list_by_rows)
    assert len(text_rows[0]) == len(word_list_by_rows[0])
    return word_list_by_rows

def get_fix_distance(pre_fix, fix):
    return math.sqrt(math.pow(pre_fix[0]-fix[0], 2) + math.pow(pre_fix[1]-fix[1], 2))


def get_pic_by_fix(request):
    """按照时间切割数据集"""
    experiment_list_select = [1924]
    from datetime import datetime
    request_exp_id = request.GET.get('exp_id')
    print(f"request_exp_id:{request_exp_id}, type(request):{type(request_exp_id)}")
    logger.info(f"本次生成{len(experiment_list_select)}条")

    file_path = f"data/reader/20240908-exp_fix_match_row.csv"
    
    data = pd.read_csv(file_path)

    fix_seq_max_id = 0
    fix_num = 0
    for idx, row in data.iterrows():
        exp_id, page_id, fix_seq_id, row_id = row['exp_id'], row['page_id'], row['fix_seq_id'], row['row_id']
        if exp_id != int(request_exp_id):
            continue

        page_data_in_exp = PageData.objects.filter(experiment_id=exp_id).order_by('id')
        text_rows_num_by_page = [0]
        for page in page_data_in_exp:
            word_list, _ = get_word_and_sentence_from_text(page.texts)
            location = json.loads(page.location)
            text_rows = split_text_by_row(location, word_list)
            text_rows_num_by_page.append(text_rows_num_by_page[-1]+len(text_rows))
        page_data = PageData.objects.get(id=page_id)
        gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t, end_time=0)
        # 按行切割的眼动
        _, row_level_fix_without_row_assumption, hit_rows = generate_fixations_in_skip_data(
            gaze_points, page_data.texts, page_data.location, page_id=page_data.id
        )
        assert len(row_level_fix_without_row_assumption) == len(hit_rows)
        # 按行切割的文本
        word_list, _ = get_word_and_sentence_from_text(page_data.texts)
        location =  json.loads(page_data.location)
        # label
        word_label, _, _ = compute_label( # 全量的，需要按行截取
            page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
        )  # 填充标签
        word_label_by_rows = get_word_label_by_row(word_label, text_rows)
        word_list_by_rows = get_word_list_by_row(word_list, text_rows)

        text_rows = split_text_by_row(json.loads(page_data.location), word_list)

        fixations_seq_split_y_diff = split_fixations(gaze_points, page_data.location, "y_diff")

        # 转换row_id
        convert_row_id = 0
        for row_max_id in text_rows_num_by_page:
            if row_id >= row_max_id:
                convert_row_id = row_id - row_max_id
        print(f"convert_row_id:{convert_row_id}")
        adjust_y = (text_rows[convert_row_id][0][2] + text_rows[convert_row_id][0][4]) / 2 # word, left, top, right, bottom
        try:
            fix_seq = [[x[0], adjust_y] for x in fixations_seq_split_y_diff[fix_seq_id-fix_seq_max_id]]
        except:
            fix_seq_max_id = fix_seq_id
            fix_seq = [[x[0], adjust_y] for x in fixations_seq_split_y_diff[fix_seq_id-fix_seq_max_id]]


        base_path = f"data/pic/all_time/{exp_id}/"
        if not os.path.exists(base_path):
            os.mkdir(base_path)

         # 生成图片
        path = f"{base_path}{page_data.id}/"

        # 如果目录不存在，则创建目录
        if not os.path.exists(path):
            os.mkdir(path)

        background_path = f"{path}background.png"
        fix_all_path = f"{path}fix-all-{page_data.id}.png"
        if not os.path.exists(background_path):
            # 生成背景图
            background = generate_pic_by_base64(
                page_data.image, f"{path}background.png"
            )
            image = cv2.imread(background)
            word_locations = get_word_location(page_data.location)
            words_not_understand = json.loads(page_data.wordLabels) if page_data.wordLabels else []
            title = ""
            paint_on_word(image, words_not_understand, word_locations, title, background_path)
        if not os.path.exists(fix_all_path):
            fix_all = generate_pic_by_base64(
                page_data.image, f"{path}fix-all-{page_data.id}.png"
            )
            image = cv2.imread(fix_all)
            word_locations = get_word_location(page_data.location)
            words_not_understand = json.loads(page_data.wordLabels) if page_data.wordLabels else []
            title = ""
            paint_on_word(image, words_not_understand, word_locations, title, fix_all)
        # 原始的fixation图
        fix_img = show_fixations(fix_seq, background_path)
        cv2.imwrite(f"{path}fix-origin-{fix_seq_id}.png", fix_img)

        fix_img = show_fixations(fix_seq, fix_all_path, begin=fix_num)
        print(f"fix_img:{fix_img}")
        cv2.imwrite( f"{path}fix-all-{page_data.id}.png", fix_img)

        gaze_4_heat = [[x[0], x[1]] for x in fix_seq]
        myHeatmap.draw_heat_map(gaze_4_heat, f"{path}fix_heatmap-{fix_seq_id}.png", background_path)

        fix_num += len(fix_seq)
        # print(f"fix_seq:{fix_seq}")
    return HttpResponse(1)

def get_hit_row(sequence_fixations, rows):
    result_rows = []
    for _, sequence in enumerate(sequence_fixations):
        y_list = np.array([x[1] for x in sequence])
        y_mean = np.mean(y_list)
        row_index = row_index_of_sequence(rows, y_mean)
        result_rows.append(row_index)
    return result_rows

def row_index_of_sequence(rows, y):
    """根据y轴坐标确定行号"""
    return next(
        (i for i, row in enumerate(rows) if row["bottom"] >= y >= row["top"]),
        -1,
    )

def get_hit_row_new(sequence_fixations, rows):
    result_rows = []
    for _, sequence in enumerate(sequence_fixations):
        y_list = np.array([x[1] for x in sequence])
        y_mean = np.mean(y_list)
        # 获取最接近 y_mean 的三个行号
        row_indices = find_closest_rows(rows, y_mean)
        print(f"row_indices:{row_indices}")
        result_rows.append(row_indices)
    return result_rows

def find_closest_rows(rows, y):
    distances = [abs(row["top"] - y) + abs(row["bottom"] - y) for row in rows]
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
    return sorted_indices[:3]