import datetime
import json
import os
import random

import cv2
import numpy as np
import pandas as pd
from django.http import HttpResponse, JsonResponse

# Create your views here.
from loguru import logger
from PIL import Image

from action.models import Experiment, PageData
from feature.utils import (
    detect_fixations,
    detect_saccades,
    eye_gaze_to_feature,
    gaze_map,
    join_images,
    keep_row,
    paint_fixations,
    row_index_of_sequence,
    show_fixations,
    show_fixations_and_saccades,
    textarea,
    word_index_in_row,
)
from onlineReading.views import compute_label, coor_to_input
from pyheatmap import myHeatmap
from semantic_attention import generate_word_attention, get_word_difficulty
from utils import (
    format_gaze,
    generate_pic_by_base64,
    get_importance,
    get_index_in_row_only_use_x,
    get_item_index_x_y,
    get_word_and_sentence_from_text,
    normalize,
)

"""
所有与eye gaze计算的函数都写在这里
TODO list
1. fixation和saccade的计算
    1.1 配合图的生成看一下
    1.2 与单词的位置无关
"""


def process_fixations(gaze_points, texts, location, use_not_blank_assumption=True, use_nlp_assumption=False):
    fixations = detect_fixations(gaze_points)
    fixations = keep_row(fixations)

    word_list, sentence_list = get_word_and_sentence_from_text(texts)  # 获取单词和句子对应的index
    border, rows, danger_zone, len_per_word = textarea(location)
    locations = json.loads(location)
    now_max_row = -1
    assert len(word_list) == len(locations)
    adjust_fixations = []
    # 确定初始的位置
    for i, fix in enumerate(fixations):
        index, find_near = get_item_index_x_y(location, fix[0], fix[1])
        if index != -1:
            if find_near:
                # 此处是为了可视化看起来清楚
                loc = locations[index]
                fix[0] = (loc["left"] + loc["right"]) / 2
                fix[1] = (loc["top"] + loc["bottom"]) / 2
            row_index, index_in_row = word_index_in_row(rows, index)
            if index_in_row != -1:
                adjust_fix = [fix[0], fix[1], fix[2], index, index_in_row, row_index]
                adjust_fixations.append(adjust_fix)

    # 切割子序列
    sequence_fixations = []
    begin_index = 0

    for i, fix in enumerate(adjust_fixations):
        sequence = adjust_fixations[begin_index:i]
        y_list = np.array([x[1] for x in sequence])
        y_mean = np.mean(y_list)
        row_ind = row_index_of_sequence(rows, y_mean)
        word_num_in_row = rows[row_ind]["end_index"] - rows[row_ind]["begin_index"] + 1
        for j in range(i, begin_index, -1):
            if adjust_fixations[j][4] - fix[4] > int(word_num_in_row / 2):
                tmp = adjust_fixations[begin_index : j + 1]
                mean_interval = 0
                for f in range(1, len(tmp)):
                    mean_interval = mean_interval + abs(tmp[f][0] - tmp[f - 1][0])
                mean_interval = mean_interval / (len(tmp) - 1)
                data = pd.DataFrame(tmp, columns=["x", "y", "t", "index", "index_in_row", "row_index"])
                if len(set(data["row_index"])) > 1:
                    row_indexs = list(data["row_index"])
                    start = 0
                    for ind in range(start, len(row_indexs)):
                        if (
                            row_indexs[ind] < row_indexs[ind - 1]
                            and abs(tmp[ind][0] - tmp[ind - 1][0]) > mean_interval * 2
                        ):
                            if len(tmp[start:ind]) > 0:
                                sequence_fixations.append(tmp[start:ind])
                            start = ind
                    if 0 < start < len(row_indexs) - 1:
                        if len(tmp[start:-1]) > 0:
                            sequence_fixations.append(tmp[start:-1])
                    elif start == 0:
                        if len(tmp) > 0:
                            sequence_fixations.append(tmp)
                else:
                    if len(tmp) > 0:
                        sequence_fixations.append(tmp)
                # sequence_fixations.append(adjust_fixations[begin_index:i])
                begin_index = i
                break
    if begin_index != len(adjust_fixations) - 1:
        sequence_fixations.append(adjust_fixations[begin_index:-1])
    print(f"sequence len:{len(sequence_fixations)}")
    cnt = 0
    for item in sequence_fixations:
        print(f"从{cnt}开始裁剪")
        cnt += len(item)
    # 按行调整fixation
    word_attention = generate_word_attention(texts)
    importance = get_importance(texts)
    result_fixations = []
    row_level_fix = []

    result_rows = []
    row_pass_time = [0 for _ in range(len(rows))]
    for i, sequence in enumerate(sequence_fixations):
        y_list = np.array([x[1] for x in sequence])
        y_mean = np.mean(y_list)
        row_index = row_index_of_sequence(rows, y_mean)
        rows_per_fix = []
        for y in y_list:
            row_index_this_fix = row_index_of_sequence(rows, y)
            rows_per_fix.append(row_index_this_fix)
        # print(f"fix偏移占比{1 - np.sum(np.array(rows_per_fix) == row_index) / len(rows_per_fix)}")

        if use_nlp_assumption:
            if np.sum(np.array(rows_per_fix) == row_index) / len(rows_per_fix) < 0.4:
                # 根据语义去调整fix的位置
                candidate_rows = (
                    [row_index - 1, row_index, row_index + 1] if row_index > 0 else [row_index, row_index + 1]
                )
                final_row = -1
                max_corr = -1
                for j, candidate_row in enumerate(candidate_rows):
                    row = rows[candidate_row]
                    words = word_list[row["begin_index"] : row["end_index"] + 1]
                    word_loc_in_row = locations[row["begin_index"] : row["end_index"] + 1]
                    # nlp feature
                    difficulty_level = [get_word_difficulty(x) for x in words]  # text feature
                    difficulty_level = normalize(difficulty_level)

                    importance_level = [0 for _ in words]
                    attention_level = [0 for _ in words]
                    for q, word in enumerate(words):
                        for impo in importance:
                            if impo[0] == word:
                                importance_level[q] = impo[1]
                        for att in word_attention:
                            if att[0] == word:
                                attention_level[q] = att[1]
                    importance_level = normalize(importance_level)
                    attention_level = normalize(attention_level)

                    number_of_fixations = [0 for _ in words]
                    for fix in sequence:
                        index = get_index_in_row_only_use_x(word_loc_in_row, fix[0])
                        if index != -1:
                            number_of_fixations[index] += 1
                    nlp_feature = [
                        difficulty_level[i] + importance_level[i] + attention_level[i] for i in range(len(words))
                    ]
                    corr = sum(np.multiply(nlp_feature, number_of_fixations))
                    print(corr)
                    if corr > max_corr:
                        final_row = candidate_row
                        max_corr = corr
                    tmp_list = []
                    for x in range(len(words)):
                        tmp = (words[x], nlp_feature[x], number_of_fixations[x])
                        tmp_list.append(tmp)
                    print(tmp_list)
                    print("------")
                if final_row != -1:
                    print(f"将行号右{row_index}改为{final_row}")
                    row_index = final_row
                print("------")
        if use_not_blank_assumption:
            # 假设不会出现空行
            if row_index > now_max_row + 1:
                if row_pass_time[now_max_row] >= 2:
                    print("执行了")
                    random_number = random.randint(0, 1)
                    if random_number == 0:
                        # 把上一行拉下来
                        # 这一行没定位错，上一行定位错了
                        row_pass_time[result_rows[-1]] -= 1
                        result_rows[-1] = now_max_row + 1
                        row_pass_time[row_index] += 1
                        result_rows.append(row_index)
                    else:
                        # 把下一行拉上去，这一行定位错了
                        row_pass_time[now_max_row + 1] += 1
                        result_rows.append(now_max_row + 1)
                else:
                    # 如果上一行没有回看，则直接把拉上来
                    row_pass_time[now_max_row + 1] += 1
                    result_rows.append(now_max_row + 1)
            else:
                row_pass_time[row_index] += 1
                result_rows.append(row_index)
            now_max_row = max(result_rows)
    print(f"row_pass_time:{row_pass_time}")
    assert sum(row_pass_time) == len(result_rows)
    assert len(result_rows) == len(sequence_fixations)
    for i, sequence in enumerate(sequence_fixations):
        if result_rows[i] != -1:
            adjust_y = (rows[result_rows[i]]["top"] + rows[result_rows[i]]["bottom"]) / 2
            result_fixation = [[x[0], adjust_y, x[2]] for x in sequence]
            result_fixations.extend(result_fixation)
            row_level_fix.append(result_fixation)

    return result_fixations, result_rows, row_level_fix, sequence_fixations


def add_fixation_to_word(request):
    page_data_id = request.GET.get("id")
    begin = request.GET.get("begin", 0)
    end = request.GET.get("end", -1)
    pageData = PageData.objects.get(id=page_data_id)

    gaze_points = format_gaze(pageData.gaze_x, pageData.gaze_y, pageData.gaze_t)[begin:end]

    result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
        gaze_points, pageData.texts, pageData.location
    )
    # 重要的就是把有可能的错的行挑出来
    base_path = "pic\\" + str(page_data_id) + "\\"
    background = generate_pic_by_base64(pageData.image, base_path, "background.png")
    fix_img = show_fixations(result_fixations, background)
    cv2.imwrite(base_path + "fix_adjust.png", fix_img)

    label = {
        # "1016":[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]],
        "1232": [[0], [1], [2], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [11], [11], [11], [11], [12], [13]],
        "1017": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [8], [8], [9], [10], [11], [12], [13], [14]],
        "1015": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]],
        "1018": [
            [0],
            [1],
            [2],
            [3],
            [4],
            [4],
            [4, 5],
            [5],
            [5],
            [6],
            [7],
            [8],
            [9],
            [10],
            [11],
            [12],
            [13],
            [14],
            [14],
        ],
    }

    assert len(label[page_data_id]) == len(row_sequence)

    # 找index
    cnt = 0
    sequences = []
    for sequence in sequence_fixations:
        tmp = [cnt, cnt + len(sequence) - 1]
        cnt = cnt + len(sequence)
        sequences.append(tmp)
    correct_num = 0
    for i, row in enumerate(row_sequence):
        if row in label[page_data_id][i]:
            correct_num += 1
        else:
            print(f"{sequences[i]}序列出错，label：{label[page_data_id][i]}，预测：{row}")
    correct_rate = correct_num / len(row_sequence)
    print(f"预测行：{row_sequence}")
    print(f"标签行：{label[page_data_id]}")
    print(f"成功率：{correct_rate}")

    row_level_pic = []
    for fix in row_level_fix:
        row_level_pic.append(show_fixations(fix, background))
    return HttpResponse(1)
    for i, fix in enumerate(fixations):
        # 取前三个fix坐标的均值
        j = i - 1
        pre_fixations = []
        while j > 0:
            if i - j > 3:
                break
            pre_fixations.append(fixations[j])
            j -= 1
        print(pre_fixations)
        index = get_item_index_x_y(pageData.location, fix[0], fix[1])
        if index != -1:
            word_index_list.append(word_list[index])
            fix_index_list.append(i)
            page_id_list.append(pageData.id)

    df = pd.DataFrame(
        {
            "word": word_index_list,
            "fix_index": fix_index_list,
            "page_id": page_id_list,
        }
    )
    path = "jupyter\\dataset\\" + "fix-word-map.csv"

    if os.path.exists(path):
        df.to_csv(path, index=False, mode="a", header=False)
    else:
        df.to_csv(path, index=False, mode="a")
    return JsonResponse({"status": "ok"})


def classify_gaze_2_label_in_pic(request):
    page_data_id = request.GET.get("id")
    begin = request.GET.get("begin", 0)
    end = request.GET.get("end", -1)
    pageData = PageData.objects.get(id=page_data_id)

    gaze_points = format_gaze(pageData.gaze_x, pageData.gaze_y, pageData.gaze_t)[begin:end]

    """
    生成示意图 要求如下：
    1. 带有raw gaze的图
    2. 带有fixation的图，圆圈代表duration的大小，给其中的saccade打上标签
    """

    base_path = "pic\\" + str(page_data_id) + "\\"

    background = generate_pic_by_base64(pageData.image, base_path, "background.png")

    gaze_map(gaze_points, background, base_path, "gaze.png")

    # heatmap
    gaze_4_heat = [[x[0], x[1]] for x in gaze_points]
    myHeatmap.draw_heat_map(gaze_4_heat, base_path + "heatmap.png", background)
    # generate fixations
    fixations = detect_fixations(gaze_points)  # todo:default argument should be adjust to optimal--fixed
    # 单独对y轴做滤波
    fixations = keep_row(fixations)

    # generate saccades
    saccades, velocities = detect_saccades(fixations)  # todo:default argument should be adjust to optimal
    # plt using fixations and saccade
    print("fixations: " + str(fixations[36][2]) + ", " + str(fixations[37][2]) + ", " + str(fixations[38][2]))
    fixation_map = show_fixations_and_saccades(fixations, saccades, background)

    # todo 减少IO操作
    heatmap = Image.open(base_path + "heatmap.png")
    # cv2->PIL.Image
    fixation_map = cv2.cvtColor(fixation_map, cv2.COLOR_BGR2RGB)
    fixation_map = Image.fromarray(fixation_map)

    join_images(heatmap, fixation_map, base_path + "heat_fix.png")

    # todo 修改此处的写法
    vel_csv = pd.DataFrame({"velocity": velocities})

    user = Experiment.objects.get(id=pageData.experiment_id).user

    vel_csv.to_csv("jupyter//data//" + str(user) + "-" + str(page_data_id) + ".csv", index=False)

    # 画换行
    # wrap_img = paint_line_on_fixations(fixations, wrap_data, background)
    # cv2.imwrite(base_path + "wrap_img.png", wrap_img)
    #
    # print("detect rows:%d" % len(wrap_data))
    # print("actual rows:%d" % len(rows))
    # assert len(wrap_data) == len(rows) - 1
    return JsonResponse({"code": 200, "status": "生成成功"}, json_dumps_params={"ensure_ascii": False})


def generate_tmp_pic(request):
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)

    gaze_points = format_gaze(pageData.gaze_x, pageData.gaze_y, pageData.gaze_t)

    base_path = "pic\\" + str(page_data_id) + "\\"

    background = generate_pic_by_base64(pageData.image, base_path, "background.png")

    gaze_4_heat = [[x[0], x[1]] for x in gaze_points]
    myHeatmap.draw_heat_map(gaze_4_heat, base_path + "heatmap.png", background)
    fixations = detect_fixations(gaze_points, max_dispersion=80)

    pd.DataFrame({"durations": [x[2] for x in fixations]}).to_csv(
        "D:\\qxy\\reading-new\\reading\\jupyter\\data\\duration.csv", index=False
    )

    canvas = paint_fixations(cv2.imread(base_path + "heatmap.png"), fixations, interval=1, label=3)
    cv2.imwrite(base_path + "fix_on_heat.png", canvas)

    return JsonResponse({"code": 200, "status": "生成成功"}, json_dumps_params={"ensure_ascii": False})


def get_dataset(request):
    # optimal_list = [
    #     [574, 580],
    #     [582],
    #     [585, 588],
    #     [590, 591],
    #     [595, 598],
    #     [600, 605],
    #     [609, 610],
    #     [613, 619],
    #     [622, 625],
    #     [628],
    #     [630, 631],
    #     [634],
    #     [636],
    #     [637, 641],
    # ]
    optimal_list = [[603, 604]]

    # users = ['luqi', 'qxy', 'zhaoyifeng', 'ln']
    # users = ['qxy']
    experiment_list_select = []
    for item in optimal_list:
        if len(item) == 2:
            for i in range(item[0], item[1] + 1):
                experiment_list_select.append(i)
        if len(item) == 1:
            experiment_list_select.append(item[0])
    # experiments = Experiment.objects.filter(is_finish=True).filter(id__in=experiment_list_select).filter(user__in=users)
    experiments = Experiment.objects.filter(is_finish=True).filter(id__in=experiment_list_select)
    print(len(experiments))
    # 超参
    interval = 2 * 1000
    # cnn相关的特征
    experiment_ids = []
    times = []
    gaze_x = []
    gaze_y = []
    gaze_t = []
    speed = []
    direction = []
    acc = []
    # 手工特征相关
    experiment_id_all = []
    user_all = []
    article_id_all = []
    time_all = []
    word_all = []
    word_watching_all = []
    word_understand_all = []
    sentence_understand_all = []
    mind_wandering_all = []
    reading_times_all = []
    number_of_fixations_all = []
    fixation_duration_all = []
    average_fixation_duration_all = []
    second_pass_dwell_time_of_sentence_all = []
    total_dwell_time_of_sentence_all = []
    reading_times_of_sentence_all = []
    saccade_times_of_sentence_all = []
    forward_times_of_sentence_all = []
    backward_times_of_sentence_all = []
    #
    success = 0
    fail = 0
    starttime = datetime.datetime.now()
    for experiment in experiments:
        try:
            page_data_list = PageData.objects.filter(experiment_id=experiment.id)

            # 全文信息
            words_per_page = []  # 每页的单词
            words_of_article = []  # 整篇文本的单词
            words_num_until_page = []  # 到该页为止的单词数量，便于计算
            sentences_per_page = []  # 每页的句子
            locations_per_page = []  # 每页的位置信息
            # 标签信息
            word_understand = []
            sentence_understand = []
            mind_wandering = []  # todo 走神了是0还是1？
            # 眼动信息
            gaze_points_list = []  # 分页存储的

            timestamp = 0
            # 收集信息
            for page_data in page_data_list:
                gaze_points_this_page = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
                gaze_points_list.append(gaze_points_this_page)

                word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
                words_location = json.loads(
                    page_data.location
                )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
                assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
                if len(words_num_until_page) == 0:
                    words_num_until_page.append(len(word_list))
                else:
                    words_num_until_page.append(words_num_until_page[-1] + len(word_list))

                words_per_page.append(word_list)
                words_of_article.extend(word_list)

                sentences_per_page.append(sentence_list)
                locations_per_page.append(page_data.location)
                # 生成标签
                word_understand_in_page, sentence_understand_in_page, mind_wander_in_page = compute_label(
                    page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
                )
                word_understand.extend(word_understand_in_page)
                sentence_understand.extend(sentence_understand_in_page)
                mind_wandering.extend(mind_wander_in_page)

            word_num = len(words_of_article)
            # 特征相关
            number_of_fixations = [0 for _ in range(word_num)]
            reading_times = [0 for _ in range(word_num)]
            fixation_duration = [0 for _ in range(word_num)]
            average_fixation_duration = [0 for _ in range(word_num)]
            reading_times_of_sentence = [0 for _ in range(word_num)]  # 相对的
            second_pass_dwell_time_of_sentence = [0 for _ in range(word_num)]  # 相对的
            total_dwell_time_of_sentence = [0 for _ in range(word_num)]  # 相对的
            saccade_times_of_sentence = [0 for _ in range(word_num)]
            forward_times_of_sentence = [0 for _ in range(word_num)]
            backward_times_of_sentence = [0 for _ in range(word_num)]

            pre_word_index = -1
            for i, gaze_points in enumerate(gaze_points_list):
                print("---正在处理第%d页---" % i)
                begin = 0
                border, rows, danger_zone = textarea(locations_per_page[i])
                for j, gaze in enumerate(gaze_points):
                    if gaze[2] - gaze_points[begin][2] > interval:
                        (
                            num_of_fixation_this_time,
                            reading_times_this_time,
                            fixation_duration_this_time,
                            reading_times_of_sentence_in_word_this_page,
                            second_pass_dwell_time_of_sentence_in_word_this_page,
                            total_dwell_time_of_sentence_in_word_this_page,
                            saccade_times_of_sentence_word_level_this_page,
                            forward_times_of_sentence_word_level_this_page,
                            backward_times_of_sentence_word_level_this_page,
                            is_watching,
                            pre_word,
                        ) = eye_gaze_to_feature(
                            gaze_points[0:j],
                            words_per_page[i],
                            sentences_per_page[i],
                            locations_per_page[i],
                            begin,
                            pre_word_index,
                            danger_zone,
                        )
                        pre_word_index = pre_word
                        word_watching = [0 for _ in range(word_num)]

                        begin_index = words_num_until_page[i - 1] if i > 0 else 0
                        # for item in is_watching:
                        #     word_watching[item + begin_index] = 1

                        for item in is_watching:
                            if num_of_fixation_this_time[item] > 0 and reading_times_this_time[item] > 0:
                                word_watching[item + begin_index] = 1

                        cnt = 0
                        for x in range(begin_index, words_num_until_page[i]):
                            number_of_fixations[x] = num_of_fixation_this_time[cnt]
                            reading_times[x] = reading_times_this_time[cnt]
                            fixation_duration[x] = fixation_duration_this_time[cnt]

                            average_fixation_duration[x] = (
                                fixation_duration[x] / number_of_fixations[x] if number_of_fixations[x] != 0 else 0
                            )
                            reading_times_of_sentence[x] = reading_times_of_sentence_in_word_this_page[cnt]  # 相对的
                            second_pass_dwell_time_of_sentence[
                                x
                            ] = second_pass_dwell_time_of_sentence_in_word_this_page[
                                cnt
                            ]  # 相对的
                            total_dwell_time_of_sentence[x] = total_dwell_time_of_sentence_in_word_this_page[cnt]  # 相对的
                            saccade_times_of_sentence[x] = saccade_times_of_sentence_word_level_this_page[cnt]
                            forward_times_of_sentence[x] = forward_times_of_sentence_word_level_this_page[cnt]
                            backward_times_of_sentence[x] = backward_times_of_sentence_word_level_this_page[cnt]
                            cnt += 1

                        experiment_id_all.extend([experiment.id for x in range(word_num)])
                        user_all.extend([experiment.user for x in range(word_num)])
                        time_all.extend([timestamp for x in range(word_num)])
                        article_id_all.extend([experiment.article_id for _ in range(word_num)])
                        word_all.extend(words_of_article)
                        word_watching_all.extend(word_watching)
                        word_understand_all.extend(word_understand)
                        sentence_understand_all.extend(sentence_understand)
                        mind_wandering_all.extend(mind_wandering)
                        reading_times_all.extend(reading_times)
                        number_of_fixations_all.extend(number_of_fixations)
                        fixation_duration_all.extend(fixation_duration)
                        average_fixation_duration_all.extend(average_fixation_duration)
                        # sentence level
                        second_pass_dwell_time_of_sentence_all.extend(second_pass_dwell_time_of_sentence)
                        total_dwell_time_of_sentence_all.extend(total_dwell_time_of_sentence)
                        reading_times_of_sentence_all.extend(reading_times_of_sentence)
                        saccade_times_of_sentence_all.extend(saccade_times_of_sentence)
                        forward_times_of_sentence_all.extend(forward_times_of_sentence)
                        backward_times_of_sentence_all.extend(backward_times_of_sentence)

                        experiment_ids.append(experiment.id)
                        times.append(timestamp)
                        timestamp += 1
                        gaze_of_x = [x[0] for x in gaze_points[begin:j]]
                        gaze_of_y = [x[1] for x in gaze_points[begin:j]]
                        gaze_of_t = [x[2] for x in gaze_points[begin:j]]
                        speed_now, direction_now, acc_now = coor_to_input(gaze_points[begin:j], 8)
                        assert len(gaze_of_x) == len(gaze_of_y) == len(speed_now) == len(direction_now) == len(acc_now)
                        gaze_x.append(gaze_of_x)
                        gaze_y.append(gaze_of_y)
                        gaze_t.append(gaze_of_t)
                        speed.append(speed_now)
                        direction.append(direction_now)
                        acc.append(acc_now)

                        begin = j
                # 生成手工数据集
                df = pd.DataFrame(
                    {
                        # 1. 实验信息相关
                        "experiment_id": experiment_id_all,
                        "user": user_all,
                        "article_id": article_id_all,
                        "time": time_all,
                        "word": word_all,
                        "word_watching": word_watching_all,
                        # # 2. label相关
                        "word_understand": word_understand_all,
                        "sentence_understand": sentence_understand_all,
                        "mind_wandering": mind_wandering_all,
                        # 3. 特征相关
                        # word level
                        "reading_times": reading_times_all,
                        "number_of_fixations": number_of_fixations_all,
                        "fixation_duration": fixation_duration_all,
                        "average_fixation_duration": average_fixation_duration_all,
                        # sentence level
                        "second_pass_dwell_time_of_sentence": second_pass_dwell_time_of_sentence_all,
                        "total_dwell_time_of_sentence": total_dwell_time_of_sentence_all,
                        "reading_times_of_sentence": reading_times_of_sentence_all,
                        "saccade_times_of_sentence": saccade_times_of_sentence_all,
                        "forward_times_of_sentence": forward_times_of_sentence_all,
                        "backward_times_of_sentence": backward_times_of_sentence_all,
                    }
                )
                path = "jupyter\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + "-test-all.csv"

                if os.path.exists(path):
                    df.to_csv(path, index=False, mode="a", header=False)
                else:
                    df.to_csv(path, index=False, mode="a")

                # 清空列表
                experiment_id_all = []
                user_all = []
                article_id_all = []
                time_all = []
                word_all = []
                word_watching_all = []
                word_understand_all = []
                sentence_understand_all = []
                mind_wandering_all = []
                reading_times_all = []
                number_of_fixations_all = []
                fixation_duration_all = []
                average_fixation_duration_all = []
                second_pass_dwell_time_of_sentence_all = []
                total_dwell_time_of_sentence_all = []
                reading_times_of_sentence_all = []
                saccade_times_of_sentence_all = []
                forward_times_of_sentence_all = []
                backward_times_of_sentence_all = []

                success += 1
                endtime = datetime.datetime.now()
                logger.info(
                    "成功生成%d条,失败%d条,耗时为%ss" % (success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
                )
        except:
            fail += 1
            endtime = datetime.datetime.now()
            logger.info(
                "成功生成%d条,失败%d条,耗时为%ss" % (success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
            )
    # 生成cnn的数据集
    data = pd.DataFrame(
        {
            # 1. 实验信息相关
            "experiment_id": experiment_ids,
            "time": times,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "gaze_t": gaze_t,
            "speed": speed,
            "direction": direction,
            "acc": acc,
        }
    )
    path = "jupyter\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + "-test-all-gaze.csv"
    if os.path.exists(path):
        data.to_csv(path, index=False, mode="a", header=False)
    else:
        data.to_csv(path, index=False, mode="a")
    logger.info("成功生成%d条，失败%d条" % (success, fail))
    return JsonResponse({"status": "ok"})


def get_all_time_dataset(request):
    optimal_list = [
        [574, 580],
        [582],
        [585, 588],
        [590, 591],
        [595, 598],
        [600, 605],
        [609, 610],
        [613, 619],
        [622, 625],
        [628],
        [630, 631],
        [634],
        [636],
        [637, 641],
    ]
    # optimal_list = [[630, 631]]

    users = ["chenyuwang"]
    # users = ['qxy']
    experiment_list_select = []
    for item in optimal_list:
        if len(item) == 2:
            for i in range(item[0], item[1] + 1):
                experiment_list_select.append(i)
        if len(item) == 1:
            experiment_list_select.append(item[0])
    experiments = Experiment.objects.filter(is_finish=True).filter(id__in=experiment_list_select).filter(user__in=users)
    print(len(experiments))
    # 超参
    success = 0
    fail = 0
    starttime = datetime.datetime.now()
    for experiment in experiments:
        try:
            page_data_list = PageData.objects.filter(experiment_id=experiment.id)

            # 全文信息
            words_per_page = []  # 每页的单词
            words_of_article = []  # 整篇文本的单词
            words_num_until_page = []  # 到该页为止的单词数量，便于计算
            locations_per_page = []  # 每页的位置信息
            # 标签信息
            word_understand = []
            # tmp
            texts = ""
            for page_data in page_data_list:
                texts += page_data.texts
            all_word_list, all_sentence_list = get_word_and_sentence_from_text(texts)  # 获取单词和句子对应的index
            # 收集信息
            word_num = len(all_word_list)
            # 特征相关
            number_of_fixations = [0 for _ in range(word_num)]
            reading_times = [0 for _ in range(word_num)]
            fixation_duration = [0 for _ in range(word_num)]

            for i, page_data in enumerate(page_data_list):

                word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
                words_location = json.loads(
                    page_data.location
                )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
                assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
                if len(words_num_until_page) == 0:
                    words_num_until_page.append(len(word_list))
                else:
                    words_num_until_page.append(words_num_until_page[-1] + len(word_list))

                words_per_page.append(word_list)
                words_of_article.extend(word_list)

                locations_per_page.append(page_data.location)
                # 生成标签
                word_understand_this_page, sentence_understand_in_page, mind_wander_in_page = compute_label(
                    page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
                )
                word_understand.extend(word_understand_this_page)

                gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
                result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
                    gaze_points, page_data.texts, page_data.location, use_not_blank_assumption=True
                )

                """word level"""
                begin = 0 if i == 0 else words_num_until_page[i - 1] - 1
                pre_word_index = -1
                for j, fixation in enumerate(result_fixations):

                    index, isAdjust = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
                    if index != -1:
                        number_of_fixations[index + begin] += 1
                        fixation_duration[index + begin] += fixation[2]
                        if index != pre_word_index:
                            reading_times[index + begin] += 1
                            pre_word_index = index

            # 生成手工数据集
            df = pd.DataFrame(
                {
                    # 1. 实验信息相关
                    "experiment_id": [experiment.id for _ in range(word_num)],
                    "word": all_word_list,
                    # # 2. label相关
                    "word_understand": word_understand,
                    # 3. 特征相关
                    # word level
                    "reading_times": reading_times,
                    "number_of_fixations": number_of_fixations,
                    "fixation_duration": fixation_duration,
                    # "average_fixation_duration": average_fixation_duration_all,
                }
            )
            path = "jupyter\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + "-test-chenyuwang-all.csv"

            if os.path.exists(path):
                df.to_csv(path, index=False, mode="a", header=False)
            else:
                df.to_csv(path, index=False, mode="a")

            success += 1
            endtime = datetime.datetime.now()
            logger.info(
                "成功生成%d条,失败%d条,耗时为%ss" % (success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
            )
        except:
            fail += 1
            endtime = datetime.datetime.now()
            logger.info(
                "成功生成%d条,失败%d条,耗时为%ss" % (success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
            )

    logger.info("成功生成%d条，失败%d条" % (success, fail))
    return JsonResponse({"status": "ok"})


def get_interval_dataset(request):
    optimal_list = [
        [574, 580],
        [582],
        [585, 588],
        [590, 591],
        [595, 598],
        [600, 605],
        [609, 610],
        [613, 619],
        [622, 625],
        [628],
        [630, 631],
        [634],
        [636],
        [637, 641],
    ]

    experiment_list_select = []
    for item in optimal_list:
        if len(item) == 2:
            for i in range(item[0], item[1] + 1):
                experiment_list_select.append(i)
        if len(item) == 1:
            experiment_list_select.append(item[0])
    experiments = Experiment.objects.filter(is_finish=True).filter(id__in=experiment_list_select)
    print(len(experiments))

    success = 0
    fail = 0
    interval_list = []
    for experiment in experiments:
        try:
            page_data_list = PageData.objects.filter(experiment_id=experiment.id)
            for page_data in page_data_list:
                gaze_points_this_page = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
                fixations = keep_row(detect_fixations(gaze_points_this_page))

                word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)

                words_location = json.loads(
                    page_data.location
                )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
                assert len(word_list) == len(words_location)  # 确保单词分割的是正确的

                word_understand_in_page, sentence_understand_in_page, mind_wander_in_page = compute_label(
                    page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
                )

                first_time = [0 for _ in word_list]
                last_time = [0 for _ in word_list]
                for fixation in fixations:
                    index = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
                    if index != -1:
                        if first_time[index] == 0:
                            first_time[index] = fixation[3]
                            last_time[index] = fixation[4]
                        else:
                            last_time[index] = fixation[4]

                interval = list(map(lambda x: x[0] - x[1], zip(last_time, first_time)))

                interval = [item for i, item in enumerate(interval) if item > 0 and word_understand_in_page[i] == 0]

                interval_list.extend(interval)
            success += 1
            print("成功%d条，失败%d条" % (success, fail))
        except:
            fail += 1
            print("成功%d条，失败%d条" % (success, fail))
    pd.DataFrame({"interval": interval_list}).to_csv(
        "D:\\qxy\\reading-new\\reading\\jupyter\data\\interval.csv", index=False
    )
    return JsonResponse({"status": "ok"})
