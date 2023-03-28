import json
import logging
import math
import os

from django.http import HttpResponse
from django.shortcuts import render
from loguru import logger

from analysis.feature import WordFeature, SentFeature, CNNFeature
from analysis.models import PageData, Experiment
from pyheatmap import myHeatmap
from tools import format_gaze, generate_fixations, generate_pic_by_base64, show_fixations, get_word_location, \
    paint_on_word, get_word_and_sentence_from_text, compute_label, textarea, get_fix_by_time, \
    get_item_index_x_y, is_watching, get_sentence_by_word, generate_exp_csv, compute_sentence_label, coor_to_input
import cv2

# Create your views here.


def get_all_time_pic(request):
    exp_id = request.GET.get("exp_id")
    page_data_ls = PageData.objects.filter(experiment_id=exp_id)
    exp = Experiment.objects.get(id=exp_id)

    base_path = f"data\\pic\\all_time\\{exp_id}\\"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    for page_data in page_data_ls:
        # 拿到gaze point
        gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
        # 计算fixations
        result_fixations, _, _, _ = generate_fixations(
            gaze_points, page_data.texts, page_data.location
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
    filename = "exp.txt"
    file = open(filename, 'r')
    lines = file.readlines()

    experiment_list_select = list(lines)
    # 获取切割的窗口大小
    interval = request.GET.get("interval",8)
    interval = interval * 1000
    # 确定文件路径
    base_path = "data\\dataset\\"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d")

    exp_path = f"{base_path}exp_info.csv"
    word_feature_path = f"{base_path}word-feature-{now}.csv"
    sent_feature_path = f"{base_path}sent-feature-{now}.csv"
    cnn_feature_path = f"{base_path}cnn-feature-{now}.csv"
    # 获取需要生成的实验
    # experiment_list_select = [1011,1792]
    experiments = Experiment.objects.filter(id__in=experiment_list_select)

    cnnFeature = CNNFeature()
    success = 0
    fail = 0

    for experiment in experiments:
        try:
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

                gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
                result_fixations, row_sequence, row_level_fix, sequence_fixations = generate_fixations(
                    gaze_points, page_data.texts, page_data.location
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
                        for fixation in fixations_before:
                            word_index, isAdjust = get_item_index_x_y(json.loads(page_data.location), fixation[0], fixation[1])
                            if word_index != -1:
                                wordFeature.number_of_fixation[word_index] += 1
                                wordFeature.total_fixation_duration[word_index] += fixation[2]
                                if word_index != pre_word_index: # todo reading times的计算
                                    wordFeature.reading_times[word_index] += 1

                                sent_index = get_sentence_by_word(word_index, sentence_list)
                                if sent_index != -1:
                                    sentFeature.total_dwell_time[sent_index] += fixation[2]
                                    sentFeature.saccade_times[sent_index] += 1 # 将两个fixation之间都作为saccade

                                    if pre_word_index - word_index > 1: # 往后看,阈值暂时设为1个单词
                                        sentFeature.backward_saccade_times[sent_index] += 1
                                    if pre_word_index - word_index < -1: # 往前阅读（正常阅读顺序)
                                        sentFeature.forward_saccade_times[sent_index] += 1

                                pre_word_index = word_index # todo important
                        # 计算need prediction
                        wordFeature.need_prediction = is_watching(fixations_now,json.loads(page_data.location),wordFeature.num)
                        # 生成数据
                        for feature in word_feature_list:
                            feature.to_csv(word_feature_path, experiment.id, page_data.id, time, experiment.user)

                        for feature in sent_feature_list:
                            feature.to_csv(sent_feature_path, experiment.id, page_data.id, time, experiment.user)


                        # cnn feature的生成 todo 暂时不变，之后修改
                        cnnFeature.experiment_ids.append(experiment.id)
                        cnnFeature.times.append(time)
                        gazes = gaze_points[pre_gaze:g]

                        fix_of_x = [x[0] for x in fixations_now]
                        fix_of_y = [x[1] for x in fixations_now]
                        cnnFeature.fix_x.append(fix_of_x)
                        cnnFeature.fix_y.append(fix_of_y)

                        gaze_of_x = [x[0] for x in gazes]
                        gaze_of_y = [x[1] for x in gazes]
                        gaze_of_t = [x[2] for x in gazes]
                        speed_now, direction_now, acc_now = coor_to_input(gazes, 8)
                        assert len(gaze_of_x) == len(gaze_of_y) == len(speed_now) == len(direction_now) == len(acc_now)
                        cnnFeature.gaze_x.append(gaze_of_x)
                        cnnFeature.gaze_y.append(gaze_of_y)
                        cnnFeature.gaze_t.append(gaze_of_t)
                        cnnFeature.speed.append(speed_now)
                        cnnFeature.direction.append(direction_now)
                        cnnFeature.acc.append(acc_now)

                        time += 1
                        pre_gaze = g  # todo important

            success += 1
            logger.info(f"成功生成{success}条,失败{fail}条")
        except Exception:
            fail += 1
    # 生成exp相关信息
    cnnFeature.to_csv(cnn_feature_path)
    generate_exp_csv(exp_path,experiments)
    return HttpResponse(1)