import json
import logging
import math
import os

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render
from loguru import logger

from analysis.feature import WordFeature, SentFeature, CNNFeature, FixationMap
from analysis.models import PageData, Experiment, Paragraph
from pyheatmap import myHeatmap
from tools import format_gaze, generate_fixations, generate_pic_by_base64, show_fixations, get_word_location, \
    paint_on_word, get_word_and_sentence_from_text, compute_label, textarea, get_fix_by_time, \
    get_item_index_x_y, is_watching, get_sentence_by_word, compute_sentence_label, coor_to_input, \
    get_cnn_feature, get_row, get_euclid_distance, normalize_list, multiply_and_sum_lists
import cv2
import re

# Create your views here.


def get_all_time_pic(request):
    exp_id = request.GET.get("exp_id")
    page_data_ls = PageData.objects.filter(experiment_id=exp_id)
    exp = Experiment.objects.get(id=exp_id)

    base_path = f"data\\pic\\all_time\\{exp_id}\\"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    for page_data in page_data_ls:

        end = 500
        if page_data.id in [2818]:
            begin = 0
        if page_data.id in [2051,2052,2053,2067,1226,1298,1300,2802,2807,2794]:
            end = 0

        print(f"page_id:{page_data.id}")
        # 拿到gaze point
        gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t,end_time=end)
        # 计算fixations
        result_fixations, _, _, _ = generate_fixations(
            gaze_points, page_data.texts, page_data.location,page_id=page_data.id
        )

        path = f"{base_path}{page_data.id}\\"

        # 如果目录不存在，则创建目录
        if not os.path.exists(path):
            os.mkdir(path)

        # 生成背景图
        background = generate_pic_by_base64(
            page_data.image, f"{path}background.png"
        )
        # 生成调整后的fixation图
        print(f"len of fixations:{len(result_fixations)}")
        fix_img = show_fixations(result_fixations, background)

        cv2.imwrite(f"{path}fix-adjust.png", fix_img)
        # 画热点图
        gaze_4_heat = [[x[0], x[1]] for x in result_fixations]
        myHeatmap.draw_heat_map(gaze_4_heat, f"{path}fix_heatmap.png", background)

        # 画duration图
        gaze_duration = []
        for fix in result_fixations:
            gaze_duration.extend([fix[0],fix[1]] for _ in range(fix[2]//100))
        myHeatmap.draw_heat_map(gaze_duration, f"{path}duration_heatmap.png", background)


        # 画label TODO 合并成一个函数
        image = cv2.imread(background)
        word_locations = get_word_location(page_data.location)
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
        pic_path = f"{path}words_not_understand.png"
        paint_on_word(image, words_not_understand, word_locations, title, pic_path)
        # 3. 句子不懂
        sentences_not_understand = json.loads(page_data.sentenceLabels) if page_data.sentenceLabels else []
        words_to_painted = []
        for sentence in sentences_not_understand:
            words_to_painted.extend(iter(range(sentence[0],sentence[1])))
        title = f"{str(page_data.id)}-{exp.user}-sentences_not_understand"
        pic_path = f"{path}sentences_not_understand.png"
        paint_on_word(image, words_to_painted, word_locations, title, pic_path)

    return HttpResponse(1)


def dataset_of_timestamp(request):
    """按照时间切割数据集"""
    filename = "native.txt"
    file = open(filename, 'r')
    lines = file.readlines()

    experiment_list_select = list(lines)

    # 获取切割的窗口大小
    interval = request.GET.get("interval",8)
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
    fixationMap = FixationMap() # 用来记录画时刻图的信息

    success = 0
    fail = 0

    logger.info(f"本次生成{len(experiment_list_select)}条")
    for experiment in experiments:
        # try:
        time = 0 # 记录当前的时间
        page_data_list = PageData.objects.filter(experiment_id=experiment.id)
        # 创建不同页的信息
        word_feature_list = []
        sent_feature_list = []
        for page_data in page_data_list:
            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            # word_level
            wordFeature = WordFeature(len(word_list))
            wordFeature.word_list = word_list # 填充单词
            wordFeature.word_understand, wordFeature.sentence_understand, wordFeature.mind_wandering = compute_label(
                page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
            ) # 填充标签
            for i, word in enumerate(word_list):
                sent_index = get_sentence_by_word(i, sentence_list)
                wordFeature.sentence_id[i] = sent_index # 使用page_id,time,sentence_id可以区分
            word_feature_list.append(wordFeature)
            # sentence_level
            sentFeature = SentFeature(len(sentence_list))
            sentFeature.sentence = [sentence[0] for sentence in sentence_list]
            sentFeature.sentence_id = list(range(len(sentence_list))) # 记录id
            # todo 句子标签的生成
            sentFeature.sentence_understand,sentFeature.mind_wandering = compute_sentence_label(page_data.sentenceLabels, page_data.wanderLabels,sentence_list)
            sent_feature_list.append(sentFeature)

        for p,page_data in enumerate(page_data_list):
            wordFeature = word_feature_list[p] # 获取单词特征
            sentFeature = sent_feature_list[p] # 获取句子特征

            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            border, rows, danger_zone, len_per_word = textarea(page_data.location)

            end = 500
            if page_data.id in [2051, 2052, 2053, 2067, 1226, 1298, 1300,2807]:
                end = 0

            gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t,end_time=end)
            result_fixations, row_sequence, row_level_fix, sequence_fixations = generate_fixations(
                gaze_points, page_data.texts, page_data.location,page_id=page_data.id
            )

            pre_gaze = 0
            for g,gaze in enumerate(gaze_points):
                if g == 0:
                    continue
                if gaze[-1] - gaze_points[pre_gaze][-1] > interval: # 按照interval切割gaze
                    # 把当前页的特征清空，因为要重新算一遍特征
                    wordFeature.clean()
                    sentFeature.clean()
                    # 目的是为了拿到gaze的时间，来切割fixation，为什么不直接gaze->fixation,会不准 todo 实时处理
                    fixations_before = get_fix_by_time(result_fixations, start=0,end=gaze[-1])
                    fixations_now = get_fix_by_time(result_fixations, gaze_points[pre_gaze][-1], gaze[-1])
                    # 计算特征

                    pre_word_index = -1
                    for f,fixation in enumerate(fixations_before):
                        word_index, isAdjust = get_item_index_x_y(json.loads(page_data.location), fixation[0], fixation[1])
                        if word_index != -1:
                            wordFeature.number_of_fixation[word_index] += 1
                            wordFeature.total_fixation_duration[word_index] += fixation[2]
                            if word_index != pre_word_index: # todo reading times的计算
                                wordFeature.reading_times[word_index] += 1

                            sent_index = get_sentence_by_word(word_index, sentence_list)
                            if sent_index != -1:
                                sentFeature.total_dwell_time[sent_index] += fixation[2]
                                # if f!=0:
                                if pre_word_index != word_index:
                                    sentFeature.saccade_times[sent_index] += 1 # 将两个fixation之间都作为saccade

                                    if pre_word_index - word_index >= 0: # 往后看,阈值暂时设为1个单词
                                        sentFeature.backward_saccade_times[sent_index] += 1
                                    if pre_word_index - word_index < 0: # 往前阅读（正常阅读顺序)
                                        sentFeature.forward_saccade_times[sent_index] += 1

                                    sentFeature.saccade_duration[sent_index] += fixations_before[f][3] - fixations_before[f-1][4] # 3是起始，4是结束
                                    sentFeature.saccade_velocity[sent_index] += get_euclid_distance((fixations_before[f][0],fixations_before[f][1]),(fixations_before[f-1][0],fixations_before[f-1][1])) # 记录的实际上是距离
                                    pre_row = get_row(pre_word_index, rows)
                                    now_row = get_row(word_index, rows)
                                    if pre_row == now_row:
                                        sentFeature.horizontal_saccade_proportion[sent_index] += 1 # 记录的实际上是次数

                            pre_word_index = word_index # todo important
                    # 计算need prediction
                    wordFeature.need_prediction = is_watching(fixations_now,json.loads(page_data.location),wordFeature.num)
                    # 生成数据
                    for feature in word_feature_list:
                        feature.to_csv(word_feature_path, experiment.id, page_data.id, time, experiment.user, experiment.article_id)

                    for feature in sent_feature_list:
                        feature.to_csv(sent_feature_path, experiment.id, page_data.id, time, experiment.user,experiment.article_id)


                    # cnn feature的生成 todo 暂时不变，之后修改
                    get_cnn_feature(time,cnnFeature,gaze_points[pre_gaze:g],experiment.id,fixations_now)

                    # 记录每个时刻的眼动，用于画图
                    fixationMap.update(time,experiment.id,page_data.id,fixations_now)

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

    page_row = page_csv[(page_csv['exp_id']==int(exp_id))&(page_csv['time']==int(time))]

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
    """按照时间切割数据集"""
    filename = "exps/data1.txt"
    file = open(filename, 'r')
    lines = file.readlines()

    experiment_list_select = list(lines)

    filename = "exps/data2.txt"
    file = open(filename, 'r')
    lines1 = file.readlines()
    experiment_list_select.extend(list(lines1))

    filename = "exps/data3.txt"
    file = open(filename, 'r')
    lines2 = file.readlines()
    experiment_list_select.extend(list(lines2))
    print(f"lens:{len(experiment_list_select)}")
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


    base_path = f"data\\dataset\\{now}\\"
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
        time = 0 # 记录当前的时间
        page_data_list = PageData.objects.filter(experiment_id=experiment.id)
        # 创建不同页的信息
        word_feature_list = []
        sent_feature_list = []
        for page_data in page_data_list:
            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            # word_level
            wordFeature = WordFeature(len(word_list))
            wordFeature.word_list = word_list # 填充单词
            wordFeature.word_understand, wordFeature.sentence_understand, wordFeature.mind_wandering = compute_label(
                page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
            ) # 填充标签
            for i, word in enumerate(word_list):
                sent_index = get_sentence_by_word(i, sentence_list)
                wordFeature.sentence_id[i] = sent_index # 使用page_id,time,sentence_id可以区分
            word_feature_list.append(wordFeature)
            # sentence_level
            sentFeature = SentFeature(len(sentence_list))
            sentFeature.sentence = [sentence[0] for sentence in sentence_list]
            sentFeature.sentence_id = list(range(len(sentence_list))) # 记录id
            # todo 句子标签的生成
            sentFeature.sentence_understand,sentFeature.mind_wandering = compute_sentence_label(page_data.sentenceLabels, page_data.wanderLabels,sentence_list)
            sent_feature_list.append(sentFeature)

        cnn_gaze_points = []
        cnn_result_fixations = []
        for p,page_data in enumerate(page_data_list):
            wordFeature = word_feature_list[p] # 获取单词特征
            sentFeature = sent_feature_list[p] # 获取句子特征

            word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
            border, rows, danger_zone, len_per_word = textarea(page_data.location)

            end = 500
            if page_data.id in [2051, 2052, 2053, 2067, 1226, 1298, 1300]:
                end = 0
            gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t,end_time=end)
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
            for f,fixation in enumerate(result_fixations):
                word_index, isAdjust = get_item_index_x_y(json.loads(page_data.location), fixation[0], fixation[1])
                if word_index != -1:
                    wordFeature.number_of_fixation[word_index] += 1
                    wordFeature.total_fixation_duration[word_index] += fixation[2]
                    if word_index != pre_word_index: # todo reading times的计算
                        wordFeature.reading_times[word_index] += 1

                    sent_index = get_sentence_by_word(word_index, sentence_list)
                    if sent_index != -1:
                        sentFeature.total_dwell_time[sent_index] += fixation[2]
                        if first_pass[sent_index] < 2:
                            sentFeature.first_pass_total_dwell_time[sent_index] += fixation[2]
                        # if f!=0:
                        if pre_word_index != word_index:
                            sentFeature.saccade_times[sent_index] += 1 # 将两个fixation之间都作为saccade
                            if first_pass[sent_index] < 2:
                                sentFeature.first_pass_saccade_times[sent_index] += 1

                            if pre_word_index - word_index > 0: # 往后看,阈值暂时设为1个单词
                                sentFeature.backward_saccade_times[sent_index] += 1
                                if first_pass[sent_index] < 2:
                                    sentFeature.first_pass_backward_saccade_times[sent_index] += 1
                            if pre_word_index - word_index < 0: # 往前阅读（正常阅读顺序)
                                sentFeature.forward_saccade_times[sent_index] += 1
                                if first_pass[sent_index] < 2:
                                    sentFeature.first_pass_forward_saccade_times[sent_index] += 1

                            sentFeature.saccade_duration[sent_index] += result_fixations[f][3] - result_fixations[f-1][4] # 3是起始，4是结束
                            sentFeature.saccade_velocity[sent_index] += get_euclid_distance((result_fixations[f][0],result_fixations[f][1]),(result_fixations[f-1][0],result_fixations[f-1][1])) # 记录的实际上是距离
                            if first_pass[sent_index] < 2:
                                sentFeature.first_pass_saccade_duration[sent_index] += result_fixations[f][3] - \
                                                                            result_fixations[f - 1][4]  # 3是起始，4是结束
                                sentFeature.first_pass_saccade_velocity[sent_index] += get_euclid_distance(
                                    (result_fixations[f][0], result_fixations[f][1]),
                                    (result_fixations[f - 1][0], result_fixations[f - 1][1]))  # 记录的实际上是距离

                            pre_row = get_row(pre_word_index, rows)
                            now_row = get_row(word_index, rows)
                            if pre_row == now_row:
                                sentFeature.horizontal_saccade_proportion[sent_index] += 1 # 记录的实际上是次数
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
                            word_now_loc = (word_index-sentence_now[1])/sentence_now[3]
                            if  word_now_loc > 0.5:
                                reach_medium_times[sent_index] += 1
                            if reach_medium_times[sent_index] > 3 and word_now_loc < 0.3:
                                first_pass[sent_index] += 1




                    pre_word_index = word_index # todo important
            # 计算need prediction
            wordFeature.need_prediction = is_watching(result_fixations,json.loads(page_data.location),wordFeature.num)
            # 生成数据
            wordFeature.to_csv(word_feature_path, experiment.id, page_data.id, time, experiment.user, experiment.article_id)

            sentFeature.to_csv(sent_feature_path, experiment.id, page_data.id, time, experiment.user,experiment.article_id)


        # cnn feature的生成 todo 暂时不变，之后修改
        get_cnn_feature(time,cnnFeature,cnn_gaze_points,experiment.id,cnn_result_fixations)


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
    res = "".join(str(i)+"," for i, word in enumerate(word_list) if word == input)
    return HttpResponse(res)


def sent_domain(request):
    page_id = request.GET.get("page_id")
    sent_id = int(request.GET.get("sent"))
    page_data = PageData.objects.get(id=page_id)
    word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
    res = f"[{sentence_list[sent_id][1]},{sentence_list[sent_id][2]}]"

    return HttpResponse(res)


def get_questionare(request):
    exp_ids = request.GET.get("exp_ids")
    exp_ids = exp_ids.split(',')
    word_list = []
    sent_list = []
    for exp_id in exp_ids:
        page_data_list = PageData.objects.filter(experiment_id=exp_id)
        for page_data in page_data_list:
            words, sentences = get_word_and_sentence_from_text(page_data.texts)
            word_inter = page_data.word_intervention
            sent_inter = page_data.sent_intervention
            word_inter = [int(s) for s in re.findall(r'-?\d+\.?\d*', word_inter)]
            sent = [int(s) for s in re.findall(r'-?\d+\.?\d*', sent_inter)]
            sent_inter = []
            print("experiment id: "+exp_id)

            for start, end in zip(sent[0::2], sent[1::2]):
                sent_inter.append([start, end])
            for w in word_inter:
                word_list.append(words[w])
                print(str(w)+": "+words[w])
            for sl in sent_inter:
                for sentence in sentences:
                    if sentence[1] == sl[0]:
                        sent_list.append(sentence[0])
    word_list = list(set(word_list))
    sent_list = list(set(sent_list))
    word_dic = {'word':word_list,
                }
    sent_dic = {'sentence':sent_list,
                }
    word_df = pd.DataFrame(word_dic)
    word_df['understand now?'] = ''
    word_df['understand before experiment?'] = ''
    sent_df = pd.DataFrame(sent_dic)
    sent_df['understand now?'] = ''
    sent_df['understand before experiment?'] = ''
    word_df.to_csv('questionaire/word_questionaire.csv')
    sent_df.to_csv('questionaire/sent_questionaire.csv')
    return HttpResponse(1)







