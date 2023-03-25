# import datetime
# import json
# import math
# import os
# import random
#
# import cv2
# import numpy as np
# import pandas as pd
# from django.http import HttpResponse, JsonResponse
#
# # Create your views here.
# from loguru import logger
# from PIL import Image
#
# from action.models import Experiment, PageData, Paragraph
# from feature.utils import (
#     detect_fixations,
#     detect_saccades,
#     eye_gaze_to_feature,
#     gaze_map,
#     get_sentence_by_word,
#     join_images,
#     keep_row,
#     paint_fixations,
#     row_index_of_sequence,
#     show_fixations,
#     show_fixations_and_saccades,
#     textarea,
#     word_index_in_row,
# )
# from onlineReading.views import compute_label, coor_to_input, paint_on_word
# from pyheatmap import myHeatmap
# from semantic_attention import generate_word_attention, get_word_difficulty, generate_word_list
# from utils import (
#     format_gaze,
#     generate_pic_by_base64,
#     get_importance,
#     get_index_in_row_only_use_x,
#     get_item_index_x_y,
#     get_word_and_sentence_from_text,
#     normalize, get_word_location, process_fixations, get_word_count, get_row, get_euclid_distance,
# )
#
# """
# 所有与eye gaze计算的函数都写在这里
# TODO list
# 1. fixation和saccade的计算
#     1.1 配合图的生成看一下
#     1.2 与单词的位置无关
# """
#
#
# def add_fixation_to_word(request):
#     exp_id = request.GET.get("exp_id")
#     begin = request.GET.get("begin", 0)
#     end = request.GET.get("end", -1)
#     csv = request.GET.get('csv', False)
#     heatmap = request.GET.get('heatmap', True)
#     check = request.GET.get("check", False)
#
#     page_data_ls = PageData.objects.filter(experiment_id=exp_id)
#     exp = Experiment.objects.get(id=exp_id)
#
#     path = "pic\\" + str(exp_id) + "\\"
#     if not os.path.exists(path):
#         os.mkdir(path)
#
#     for page_data in page_data_ls:
#         page_data_id = page_data.id
#
#         pageData = PageData.objects.get(id=page_data_id)
#
#         border, rows, danger_zone, len_per_word = textarea(pageData.location)
#
#         gaze_points = format_gaze(pageData.gaze_x, pageData.gaze_y, pageData.gaze_t)[begin:end]
#
#         result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
#             gaze_points, pageData.texts, pageData.location
#         )
#         # result_fixations = keep_row(detect_fixations(gaze_points))
#         # row_level_fix = []
#         # row_sequence = []
#         # sequence_fixations = []
#
#         # 重要的就是把有可能的错的行挑出来
#         base_path = path + str(page_data_id) + "\\"
#
#         isMac = True if exp.device == "mac" else False
#         # background = generate_pic_by_base64(pageData.image, base_path, "background.png")
#         background = base_path + "background.png"
#
#         fix_img = show_fixations(result_fixations, background)
#         cv2.imwrite(base_path + "fix-adjust.png", fix_img)
#
#         gaze_4_heat = [[x[0], x[1]] for x in result_fixations]
#         myHeatmap.draw_heat_map(gaze_4_heat, base_path + "fix_heatmap.png", background)
#
#         word_list, sentence_list = get_word_and_sentence_from_text(pageData.texts)
#
#         if csv:
#             word_index_list = []
#             fix_index_list = []
#             page_id_list = []
#             experiment_id_list = []
#             words = []
#
#             for i, x in enumerate(result_fixations):
#                 word_index, is_adjust = get_item_index_x_y(location=pageData.location, x=x[0], y=x[1],
#                                                            word_list=word_list,
#                                                            rows=rows,
#                                                            remove_horizontal_drift=False)
#                 word_index_list.append(word_index)
#                 fix_index_list.append(i)
#                 page_id_list.append(pageData.id)
#                 experiment_id_list.append(pageData.experiment_id)
#                 if word_index != -1:
#                     words.append(word_list[word_index])
#                 else:
#                     words.append("-1")
#
#             df = pd.DataFrame(
#                 {
#                     "word_index": word_index_list,
#                     "word": words,
#                     "fix_index": fix_index_list,
#                     "page_id": page_id_list,
#                     "exp_id": experiment_id_list,
#                 }
#             )
#             path = "jupyter\\dataset\\" + "fix-word-map-no-drift.csv"
#
#             if os.path.exists(path):
#                 df.to_csv(path, index=False, mode="a", header=False)
#             else:
#                 df.to_csv(path, index=False, mode="a")
#
#         if heatmap:
#             exp = Experiment.objects.get(id=pageData.experiment_id)
#             word_locations = get_word_location(pageData.location)
#             image = cv2.imread(background)
#             # 走神与否
#             words_to_be_painted = []
#             if pageData.wanderLabels:
#                 paras_wander = json.loads(pageData.wanderLabels)
#             else:
#                 paras_wander = []
#
#             for para in paras_wander:
#                 for i in range(para[0], para[1] + 1):  # wander label是到一段结尾，不是到下一段
#                     words_to_be_painted.append(i)
#
#             title = str(page_data_id) + "-" + exp.user + "-" + "para_wander"
#             pic_path = base_path + "para_wander" + ".png"
#             # 画图
#             paint_on_word(image, words_to_be_painted, word_locations, title, pic_path)
#
#             # 单词 TODO 将这些整理为函数，复用
#             # 找需要画的单词
#             if pageData.wordLabels:
#                 words_not_understand = json.loads(pageData.wordLabels)
#             else:
#                 words_not_understand = []
#             title = str(page_data_id) + "-" + exp.user + "-" + "words_not_understand"
#             pic_path = base_path + "words_not_understand" + ".png"
#             # 画图
#             paint_on_word(image, words_not_understand, word_locations, title, pic_path)
#
#             # 句子
#             if pageData.sentenceLabels:
#                 sentences_not_understand = json.loads(pageData.sentenceLabels)
#             else:
#                 sentences_not_understand = []
#
#             words_to_painted = []
#             for sentence in sentences_not_understand:
#                 for i in range(sentence[0], sentence[1]):
#                     # 此处i代表的是单词
#                     words_to_painted.append(i)
#
#             title = str(page_data_id) + "-" + exp.user + "-" + "sentences_not_understand"
#             pic_path = base_path + "sentences_not_understand" + ".png"
#             # 画图
#             paint_on_word(image, words_to_painted, word_locations, title, pic_path)
#
#         if check:
#             label = {
#                 # "1016":[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]],
#                 '1211': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]],
#
#                 '1230': [[0], [1], [2], [3], [4], [4], [5], [5], [6], [6], [7], [8], [9], [10], [11], [11], [12], [13],
#                          [14],
#                          [15]],
#                 '1231': [[0], [0]],
#
#                 "1232": [[0], [1], [2], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [11], [11], [11], [11],
#                          [12],
#                          [13]],
#
#                 '1236': [[0], [1], [2], [3], [4], [5], [6], [7], [7], [7], [8], [9], [10], [11], [12], [11, 12, 13],
#                          [13],
#                          [13],
#                          [14]],
#                 '1237': [[0], [1], [1], [2], [2], [2]],
#
#                 '1238': [[0], [0, 1], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]],
#
#                 '1247': [[0], [1], [2], [3], [3], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14],
#                          [15]],
#                 '1248': [[0]],
#
#                 '1249': [[0], [0, 1], [1], [2], [3], [4], [5], [5, 6], [5], [6], [7], [8], [9], [9], [9], [10], [11],
#                          [12],
#                          [13],
#                          [13],
#                          [14]],
#                 '1250': [[0], [1], [2], [3], [4], [5], [6], [7]],
#
#                 '1257': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [12], [13], [14]],
#                 '1258': [[0], [1], [2], [4], [5], [6], [7]],
#
#                 '1288': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
#                 '1289': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
#
#                 '1290': [[0], [1], [2], [3], [5], [6], [7], [8], [9], [9], [10], [11], [12]],
#                 '1291': [[0], [1], [2], [3], [5], [6], [7], [8], [9], [10], [11], [12], [13]],
#
#                 '1298': [[0], [1], [1], [2], [3], [4], [5], [6], [7], [7], [8], [9], [10], [11], [12], [13], [14],
#                          [14]],
#                 '1299': [[0], [1], [2], [3], [4], [5], [6], [6, 7], [6], [7], [8], [9], [10], [11], [11]],
#                 '1300': [[0], [0], [1], [2], [3], [4]],
#
#                 '1316': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [8, 9], [9], [10], [11], [12], [13], [14]],
#                 '1317': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [10]],
#
#                 '1323': [[0], [1], [2], [3], [4], [4], [5], [7], [8], [9], [10], [10], [10], [11], [11], [11]],
#                 '1324': [[0], [1], [2], [3], [5], [6], [7], [7], [8], [9], [9], [10], [11], [11], [11], [12], [13],
#                          [14],
#                          [14]],
#
#                 "1017": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [8], [8], [9], [10], [11], [12], [13], [14]],
#                 "1015": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]],
#                 "1018": [
#                     [0],
#                     [1],
#                     [2],
#                     [3],
#                     [4],
#                     [4],
#                     [4, 5],
#                     [5],
#                     [5],
#                     [6],
#                     [7],
#                     [8],
#                     [9],
#                     [10],
#                     [11],
#                     [12],
#                     [13],
#                     [14],
#                     [14],
#                 ],
#
#                 '1226': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [14]],
#                 '1227': [[0], [1], [2]],
#
#                 '1267': [[0], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
#                 '1268': [[0], [1]]
#             }
#
#             print(len(label[page_data_id]))
#             print(len(row_sequence))
#             assert len(label[page_data_id]) == len(row_sequence)
#
#             # 找index
#             cnt = 0
#             sequences = []
#
#             wrong_fix_num = 0
#
#             for sequence in sequence_fixations:
#                 tmp = [cnt, cnt + len(sequence) - 1]
#                 cnt = cnt + len(sequence)
#                 sequences.append(tmp)
#
#             correct_num = 0
#             for i, row in enumerate(row_sequence):
#                 if row in label[page_data_id][i]:
#                     correct_num += 1
#                 else:
#                     print(f"{sequences[i]}序列出错，label：{label[page_data_id][i]}，预测：{row}")
#                     wrong_fix_num += sequences[i][1] - sequences[i][0] + 1
#             correct_rate = correct_num / len(row_sequence)
#             print(f"预测行：{row_sequence}")
#             print(f"标签行：{label[page_data_id]}")
#             print(f"行成功率：{correct_rate}")
#             print(f'fix成功率:{(len(result_fixations) - wrong_fix_num) / len(result_fixations)}')
#
#         row_level_pic = []
#         for fix in row_level_fix:
#             row_level_pic.append(show_fixations(fix, background))
#
#     return HttpResponse(1)
#
#
# def classify_gaze_2_label_in_pic(request):
#     page_data_id = request.GET.get("id")
#     begin = request.GET.get("begin", 0)
#     end = request.GET.get("end", -1)
#     pageData = PageData.objects.get(id=page_data_id)
#
#     gaze_points = format_gaze(pageData.gaze_x, pageData.gaze_y, pageData.gaze_t)[begin:end]
#
#     """
#     生成示意图 要求如下：
#     1. 带有raw gaze的图
#     2. 带有fixation的图，圆圈代表duration的大小，给其中的saccade打上标签
#     """
#
#     base_path = "pic\\" + str(page_data_id) + "\\"
#
#     background = generate_pic_by_base64(pageData.image, base_path, "background.png")
#
#     gaze_map(gaze_points, background, base_path, "gaze.png")
#
#     # heatmap
#     gaze_4_heat = [[x[0], x[1]] for x in gaze_points]
#     myHeatmap.draw_heat_map(gaze_4_heat, base_path + "heatmap.png", background)
#     # generate fixations
#     fixations = detect_fixations(gaze_points)  # todo:default argument should be adjust to optimal--fixed
#     # 单独对y轴做滤波
#     fixations = keep_row(fixations)
#
#     # generate saccades
#     saccades, velocities = detect_saccades(fixations)  # todo:default argument should be adjust to optimal
#     # plt using fixations and saccade
#     # print("fixations: " + str(fixations[36][2]) + ", " + str(fixations[37][2]) + ", " + str(fixations[38][2]))
#     fixation_map = show_fixations_and_saccades(fixations, saccades, background)
#
#     cv2.imwrite(base_path + 'fix.png', fixation_map)
#     # todo 减少IO操作
#     heatmap = Image.open(base_path + "heatmap.png")
#     # cv2->PIL.Image
#     fixation_map = cv2.cvtColor(fixation_map, cv2.COLOR_BGR2RGB)
#     fixation_map = Image.fromarray(fixation_map)
#
#     join_images(heatmap, fixation_map, base_path + "heat_fix.png")
#
#     # todo 修改此处的写法
#     vel_csv = pd.DataFrame({"velocity": velocities})
#
#     user = Experiment.objects.get(id=pageData.experiment_id).user
#
#     vel_csv.to_csv("jupyter//data//" + str(user) + "-" + str(page_data_id) + ".csv", index=False)
#
#     # 画换行
#     # wrap_img = paint_line_on_fixations(fixations, wrap_data, background)
#     # cv2.imwrite(base_path + "wrap_img.png", wrap_img)
#     #
#     # print("detect rows:%d" % len(wrap_data))
#     # print("actual rows:%d" % len(rows))
#     # assert len(wrap_data) == len(rows) - 1
#     return JsonResponse({"code": 200, "status": "生成成功"}, json_dumps_params={"ensure_ascii": False})
#
#
# def generate_tmp_pic(request):
#     page_data_id = request.GET.get("id")
#     pageData = PageData.objects.get(id=page_data_id)
#
#     gaze_points = format_gaze(pageData.gaze_x, pageData.gaze_y, pageData.gaze_t)
#
#     base_path = "pic\\" + str(page_data_id) + "\\"
#
#     background = generate_pic_by_base64(pageData.image, base_path, "background.png")
#
#     gaze_4_heat = [[x[0], x[1]] for x in gaze_points]
#     myHeatmap.draw_heat_map(gaze_4_heat, base_path + "heatmap.png", background)
#     fixations = detect_fixations(gaze_points, max_dispersion=80)
#
#     pd.DataFrame({"durations": [x[2] for x in fixations]}).to_csv(
#         "D:\\qxy\\reading-new\\reading\\jupyter\\data\\duration.csv", index=False
#     )
#
#     canvas = paint_fixations(cv2.imread(base_path + "heatmap.png"), fixations, interval=1, label=3)
#     cv2.imwrite(base_path + "fix_on_heat.png", canvas)
#
#     return JsonResponse({"code": 200, "status": "生成成功"}, json_dumps_params={"ensure_ascii": False})
#
#
# def get_dataset(request):
#     # optimal_list = [
#     #     [574, 580],
#     #     [582],
#     #     [585, 588],
#     #     [590, 591],
#     #     [595, 598],
#     #     [600, 605],
#     #     [609, 610],
#     #     [613, 619],
#     #     [622, 625],
#     #     [628],
#     #     [630, 631],
#     #     [634],
#     #     [636],
#     #     [637, 641],
#     # ]
#     optimal_list = [[603, 604]]
#
#     # users = ['luqi', 'qxy', 'zhaoyifeng', 'ln']
#     # users = ['qxy']
#     experiment_list_select = []
#     for item in optimal_list:
#         if len(item) == 2:
#             for i in range(item[0], item[1] + 1):
#                 experiment_list_select.append(i)
#         if len(item) == 1:
#             experiment_list_select.append(item[0])
#     # experiments = Experiment.objects.filter(is_finish=True).filter(id__in=experiment_list_select).filter(user__in=users)
#     experiments = Experiment.objects.filter(is_finish=True).filter(id__in=experiment_list_select)
#     print(len(experiments))
#     # 超参
#     interval = 2 * 1000
#     # cnn相关的特征
#     experiment_ids = []
#     times = []
#     gaze_x = []
#     gaze_y = []
#     gaze_t = []
#     speed = []
#     direction = []
#     acc = []
#     # 手工特征相关
#     experiment_id_all = []
#     user_all = []
#     article_id_all = []
#     time_all = []
#     word_all = []
#     word_watching_all = []
#     word_understand_all = []
#     sentence_understand_all = []
#     mind_wandering_all = []
#     reading_times_all = []
#     number_of_fixations_all = []
#     fixation_duration_all = []
#     average_fixation_duration_all = []
#     second_pass_dwell_time_of_sentence_all = []
#     total_dwell_time_of_sentence_all = []
#     reading_times_of_sentence_all = []
#     saccade_times_of_sentence_all = []
#     forward_times_of_sentence_all = []
#     backward_times_of_sentence_all = []
#     #
#     success = 0
#     fail = 0
#     starttime = datetime.datetime.now()
#     for experiment in experiments:
#         try:
#             page_data_list = PageData.objects.filter(experiment_id=experiment.id)
#
#             # 全文信息
#             words_per_page = []  # 每页的单词
#             words_of_article = []  # 整篇文本的单词
#             words_num_until_page = []  # 到该页为止的单词数量，便于计算
#             sentences_per_page = []  # 每页的句子
#             locations_per_page = []  # 每页的位置信息
#             # 标签信息
#             word_understand = []
#             sentence_understand = []
#             mind_wandering = []  # todo 走神了是0还是1？
#             # 眼动信息
#             gaze_points_list = []  # 分页存储的
#
#             timestamp = 0
#             # 收集信息
#             for page_data in page_data_list:
#                 gaze_points_this_page = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
#                 gaze_points_list.append(gaze_points_this_page)
#
#                 word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
#                 words_location = json.loads(
#                     page_data.location
#                 )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
#                 assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
#                 if len(words_num_until_page) == 0:
#                     words_num_until_page.append(len(word_list))
#                 else:
#                     words_num_until_page.append(words_num_until_page[-1] + len(word_list))
#
#                 words_per_page.append(word_list)
#                 words_of_article.extend(word_list)
#
#                 sentences_per_page.append(sentence_list)
#                 locations_per_page.append(page_data.location)
#                 # 生成标签
#                 word_understand_in_page, sentence_understand_in_page, mind_wander_in_page = compute_label(
#                     page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
#                 )
#                 word_understand.extend(word_understand_in_page)
#                 sentence_understand.extend(sentence_understand_in_page)
#                 mind_wandering.extend(mind_wander_in_page)
#
#             word_num = len(words_of_article)
#             # 特征相关
#             number_of_fixations = [0 for _ in range(word_num)]
#             reading_times = [0 for _ in range(word_num)]
#             fixation_duration = [0 for _ in range(word_num)]
#             average_fixation_duration = [0 for _ in range(word_num)]
#             reading_times_of_sentence = [0 for _ in range(word_num)]  # 相对的
#             second_pass_dwell_time_of_sentence = [0 for _ in range(word_num)]  # 相对的
#             total_dwell_time_of_sentence = [0 for _ in range(word_num)]  # 相对的
#             saccade_times_of_sentence = [0 for _ in range(word_num)]
#             forward_times_of_sentence = [0 for _ in range(word_num)]
#             backward_times_of_sentence = [0 for _ in range(word_num)]
#
#             pre_word_index = -1
#             for i, gaze_points in enumerate(gaze_points_list):
#                 print("---正在处理第%d页---" % i)
#                 begin = 0
#                 border, rows, danger_zone = textarea(locations_per_page[i])
#                 for j, gaze in enumerate(gaze_points):
#                     if gaze[2] - gaze_points[begin][2] > interval:
#                         (
#                             num_of_fixation_this_time,
#                             reading_times_this_time,
#                             fixation_duration_this_time,
#                             reading_times_of_sentence_in_word_this_page,
#                             second_pass_dwell_time_of_sentence_in_word_this_page,
#                             total_dwell_time_of_sentence_in_word_this_page,
#                             saccade_times_of_sentence_word_level_this_page,
#                             forward_times_of_sentence_word_level_this_page,
#                             backward_times_of_sentence_word_level_this_page,
#                             is_watching,
#                             pre_word,
#                         ) = eye_gaze_to_feature(
#                             gaze_points[0:j],
#                             words_per_page[i],
#                             sentences_per_page[i],
#                             locations_per_page[i],
#                             begin,
#                             pre_word_index,
#                             danger_zone,
#                         )
#                         pre_word_index = pre_word
#                         word_watching = [0 for _ in range(word_num)]
#
#                         begin_index = words_num_until_page[i - 1] if i > 0 else 0
#                         # for item in is_watching:
#                         #     word_watching[item + begin_index] = 1
#
#                         for item in is_watching:
#                             if num_of_fixation_this_time[item] > 0 and reading_times_this_time[item] > 0:
#                                 word_watching[item + begin_index] = 1
#
#                         cnt = 0
#                         for x in range(begin_index, words_num_until_page[i]):
#                             number_of_fixations[x] = num_of_fixation_this_time[cnt]
#                             reading_times[x] = reading_times_this_time[cnt]
#                             fixation_duration[x] = fixation_duration_this_time[cnt]
#
#                             average_fixation_duration[x] = (
#                                 fixation_duration[x] / number_of_fixations[x] if number_of_fixations[x] != 0 else 0
#                             )
#                             reading_times_of_sentence[x] = reading_times_of_sentence_in_word_this_page[cnt]  # 相对的
#                             second_pass_dwell_time_of_sentence[
#                                 x
#                             ] = second_pass_dwell_time_of_sentence_in_word_this_page[
#                                 cnt
#                             ]  # 相对的
#                             total_dwell_time_of_sentence[x] = total_dwell_time_of_sentence_in_word_this_page[cnt]  # 相对的
#                             saccade_times_of_sentence[x] = saccade_times_of_sentence_word_level_this_page[cnt]
#                             forward_times_of_sentence[x] = forward_times_of_sentence_word_level_this_page[cnt]
#                             backward_times_of_sentence[x] = backward_times_of_sentence_word_level_this_page[cnt]
#                             cnt += 1
#
#                         experiment_id_all.extend([experiment.id for x in range(word_num)])
#                         user_all.extend([experiment.user for x in range(word_num)])
#                         time_all.extend([timestamp for x in range(word_num)])
#                         article_id_all.extend([experiment.article_id for _ in range(word_num)])
#                         word_all.extend(words_of_article)
#                         word_watching_all.extend(word_watching)
#                         word_understand_all.extend(word_understand)
#                         sentence_understand_all.extend(sentence_understand)
#                         mind_wandering_all.extend(mind_wandering)
#                         reading_times_all.extend(reading_times)
#                         number_of_fixations_all.extend(number_of_fixations)
#                         fixation_duration_all.extend(fixation_duration)
#                         average_fixation_duration_all.extend(average_fixation_duration)
#                         # sentence level
#                         second_pass_dwell_time_of_sentence_all.extend(second_pass_dwell_time_of_sentence)
#                         total_dwell_time_of_sentence_all.extend(total_dwell_time_of_sentence)
#                         reading_times_of_sentence_all.extend(reading_times_of_sentence)
#                         saccade_times_of_sentence_all.extend(saccade_times_of_sentence)
#                         forward_times_of_sentence_all.extend(forward_times_of_sentence)
#                         backward_times_of_sentence_all.extend(backward_times_of_sentence)
#
#                         experiment_ids.append(experiment.id)
#                         times.append(timestamp)
#                         timestamp += 1
#                         gaze_of_x = [x[0] for x in gaze_points[begin:j]]
#                         gaze_of_y = [x[1] for x in gaze_points[begin:j]]
#                         gaze_of_t = [x[2] for x in gaze_points[begin:j]]
#                         speed_now, direction_now, acc_now = coor_to_input(gaze_points[begin:j], 8)
#                         assert len(gaze_of_x) == len(gaze_of_y) == len(speed_now) == len(direction_now) == len(acc_now)
#                         gaze_x.append(gaze_of_x)
#                         gaze_y.append(gaze_of_y)
#                         gaze_t.append(gaze_of_t)
#                         speed.append(speed_now)
#                         direction.append(direction_now)
#                         acc.append(acc_now)
#
#                         begin = j
#                 # 生成手工数据集
#                 df = pd.DataFrame(
#                     {
#                         # 1. 实验信息相关
#                         "experiment_id": experiment_id_all,
#                         "user": user_all,
#                         "article_id": article_id_all,
#                         "time": time_all,
#                         "word": word_all,
#                         "word_watching": word_watching_all,
#                         # # 2. label相关
#                         "word_understand": word_understand_all,
#                         "sentence_understand": sentence_understand_all,
#                         "mind_wandering": mind_wandering_all,
#                         # 3. 特征相关
#                         # word level
#                         "reading_times": reading_times_all,
#                         "number_of_fixations": number_of_fixations_all,
#                         "fixation_duration": fixation_duration_all,
#                         "average_fixation_duration": average_fixation_duration_all,
#                         # sentence level
#                         "second_pass_dwell_time_of_sentence": second_pass_dwell_time_of_sentence_all,
#                         "total_dwell_time_of_sentence": total_dwell_time_of_sentence_all,
#                         "reading_times_of_sentence": reading_times_of_sentence_all,
#                         "saccade_times_of_sentence": saccade_times_of_sentence_all,
#                         "forward_times_of_sentence": forward_times_of_sentence_all,
#                         "backward_times_of_sentence": backward_times_of_sentence_all,
#                     }
#                 )
#                 path = "jupyter\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + "-test-all.csv"
#
#                 if os.path.exists(path):
#                     df.to_csv(path, index=False, mode="a", header=False)
#                 else:
#                     df.to_csv(path, index=False, mode="a")
#
#                 # 清空列表
#                 experiment_id_all = []
#                 user_all = []
#                 article_id_all = []
#                 time_all = []
#                 word_all = []
#                 word_watching_all = []
#                 word_understand_all = []
#                 sentence_understand_all = []
#                 mind_wandering_all = []
#                 reading_times_all = []
#                 number_of_fixations_all = []
#                 fixation_duration_all = []
#                 average_fixation_duration_all = []
#                 second_pass_dwell_time_of_sentence_all = []
#                 total_dwell_time_of_sentence_all = []
#                 reading_times_of_sentence_all = []
#                 saccade_times_of_sentence_all = []
#                 forward_times_of_sentence_all = []
#                 backward_times_of_sentence_all = []
#
#                 success += 1
#                 endtime = datetime.datetime.now()
#                 logger.info(
#                     "成功生成%d条,失败%d条,耗时为%ss" % (
#                         success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
#                 )
#         except:
#             fail += 1
#             endtime = datetime.datetime.now()
#             logger.info(
#                 "成功生成%d条,失败%d条,耗时为%ss" % (
#                     success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
#             )
#     # 生成cnn的数据集
#     data = pd.DataFrame(
#         {
#             # 1. 实验信息相关
#             "experiment_id": experiment_ids,
#             "time": times,
#             "gaze_x": gaze_x,
#             "gaze_y": gaze_y,
#             "gaze_t": gaze_t,
#             "speed": speed,
#             "direction": direction,
#             "acc": acc,
#         }
#     )
#     path = "jupyter\\dataset\\" + datetime.datetime.now().strftime("%Y-%m-%d") + "-test-all-gaze.csv"
#     if os.path.exists(path):
#         data.to_csv(path, index=False, mode="a", header=False)
#     else:
#         data.to_csv(path, index=False, mode="a")
#     logger.info("成功生成%d条，失败%d条" % (success, fail))
#     return JsonResponse({"status": "ok"})
#
#
# def get_all_time_dataset(request):
#     # experiment_list_select = [
#     #     590,
#     #     601,
#     #     586,
#     #     587,
#     #     588,
#     #     597,
#     #     598,
#     #     625,
#     #     630,
#     #     631,
#     #     641,
#     #     639,
#     #     622,
#     #     623,
#     #     624,
#     #     628,
#     #     617,
#     #     609,
#     #     636,
#     #     638,
#     #     640,
#     #     577,
#     #     591,
#     # ]
#     experiment_list_select = [590, 597, 598, 630]
#     experiment_failed_list = [586, 624, 639]
#     user_remove_list = ["shiyubin"]
#     # experiment_list_select = [630]
#     path = "jupyter\\dataset\\" + "optimal-data.csv"
#     experiments = (
#         Experiment.objects.filter(is_finish=True)
#         .filter(id__in=experiment_list_select)
#         .exclude(id__in=experiment_failed_list)
#         .exclude(user__in=user_remove_list)
#     )
#     print(f"一共会生成{len(experiments)}条数据")
#
#     # 超参
#     success = 0
#     fail = 0
#     starttime = datetime.datetime.now()
#     for experiment in experiments:
#         try:
#             # page_data_list = PageData.objects.filter(id__in=[1300])
#             page_data_list = PageData.objects.filter(experiment_id=experiment.id)
#
#             # 全文信息
#             words_per_page = []  # 每页的单词
#             words_of_article = []  # 整篇文本的单词
#             words_num_until_page = []  # 到该页为止的单词数量，便于计算
#             locations_per_page = []  # 每页的位置信息
#             # 标签信息
#             word_understand = []
#             sentence_understand = []
#             mind_wandering = []
#             # tmp
#             texts = ""
#             for page_data in page_data_list:
#                 texts += page_data.texts
#             all_word_list, all_sentence_list = get_word_and_sentence_from_text(texts)  # 获取单词和句子对应的index
#             # 收集信息
#             word_num = len(all_word_list)
#             # 特征相关
#             number_of_fixations = [0 for _ in range(word_num)]
#             reading_times = [0 for _ in range(word_num)]
#             fixation_duration = [0 for _ in range(word_num)]
#             first_fixation_duration = [0 for _ in range(word_num)]
#             background_regression = [0 for _ in range(word_num)]
#
#             for i, page_data in enumerate(page_data_list):
#
#                 word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
#                 words_location = json.loads(
#                     page_data.location
#                 )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
#                 assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
#                 if len(words_num_until_page) == 0:
#                     words_num_until_page.append(len(word_list))
#                 else:
#                     words_num_until_page.append(words_num_until_page[-1] + len(word_list))
#
#                 words_per_page.append(word_list)
#                 words_of_article.extend(word_list)
#
#                 locations_per_page.append(page_data.location)
#                 # 生成标签
#                 word_understand_this_page, sentence_understand_in_page, mind_wander_in_page = compute_label(
#                     page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
#                 )
#                 word_understand.extend(word_understand_this_page)
#                 sentence_understand.extend(sentence_understand_in_page)
#                 mind_wandering.extend(mind_wander_in_page)
#
#                 gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
#                 result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
#                     gaze_points, page_data.texts, page_data.location, use_not_blank_assumption=True
#                 )
#
#                 # fixations = detect_fixations(gaze_points)
#                 # result_fixations = keep_row(fixations)
#
#                 """word level"""
#                 begin = 0 if i == 0 else words_num_until_page[i - 1]
#                 print(f"words_num_until_page:{words_num_until_page}")
#                 pre_word_index = -1
#                 for j, fixation in enumerate(result_fixations):
#
#                     index, isAdjust = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
#                     if index != -1:
#                         number_of_fixations[index + begin] += 1
#                         fixation_duration[index + begin] += fixation[2]
#                         if first_fixation_duration[index + begin] == 0:
#                             first_fixation_duration[index + begin] = fixation[2]
#                         if pre_word_index > index:
#                             background_regression[pre_word_index + begin] += 1
#                         if index != pre_word_index:
#                             reading_times[index + begin] += 1
#                             pre_word_index = index
#
#             # 生成手工数据集
#             df = pd.DataFrame(
#                 {
#                     # 1. 实验信息相关
#                     "experiment_id": [experiment.id for _ in range(word_num)],
#                     "user": [experiment.user for _ in range(word_num)],
#                     "word": all_word_list,
#                     # # 2. label相关
#                     "word_understand": word_understand,
#                     "sentence_understand": sentence_understand,
#                     "mind_wandering": mind_wandering,
#                     # 3. 特征相关
#                     # word level
#                     "reading_times": reading_times,
#                     "number_of_fixations": number_of_fixations,
#                     "fixation_duration": fixation_duration,
#                     # "average_fixation_duration": average_fixation_duration_all,
#                     "first_fixation_duration": first_fixation_duration,
#                     "background_regression": background_regression
#
#                 }
#             )
#
#             if os.path.exists(path):
#                 df.to_csv(path, index=False, mode="a", header=False)
#             else:
#                 df.to_csv(path, index=False, mode="a")
#
#             success += 1
#             endtime = datetime.datetime.now()
#             logger.info(
#                 "成功生成%d条,失败%d条,耗时为%ss" % (
#                     success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
#             )
#         except:
#             fail += 1
#             endtime = datetime.datetime.now()
#             logger.info(
#                 "成功生成%d条,失败%d条,耗时为%ss" % (
#                     success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
#             )
#             experiment_failed_list.append(experiment.id)
#
#     logger.info("成功生成%d条，失败%d条" % (success, fail))
#     logger.info(f"失败的experiment有:{experiment_failed_list}")
#     return JsonResponse({"status": "ok"})
#
#
# def get_gazes(fixation, page_data):
#     begin = fixation[3]
#     end = fixation[4]
#
#     return_gaze = []
#     gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
#     for gaze in gaze_points:
#         if gaze[2] >= begin and gaze[2] <= end:
#             return_gaze.append(gaze)
#
#     return return_gaze
#
#
# def get_fix_by_time(result_fixations, param):
#     fixs = []
#     for fix in result_fixations:
#         if fix[-1] <= param:
#             fixs.append(fix)
#         else:
#             break
#     return fixs
#
#
# def get_fix_this_time(result_fixations, pre_gaze_t, now_gaze_t):
#     fixs = []
#     for fix in result_fixations:
#         if fix[-2] >= pre_gaze_t and fix[-1] <= now_gaze_t:
#             fixs.append(fix)
#         if fix[-1] > now_gaze_t:
#             break
#     return fixs
#
#
# def get_sentence_interval(request):
#     filename = "exp.txt"
#     file = open(filename, 'r')
#     lines = file.readlines()
#
#     experiment_list_select = []
#     for line in lines:
#         experiment_list_select.append(line)
#
#     experiment_failed_list = []
#
#     experiments = (
#         Experiment.objects.filter(is_finish=True)
#         .filter(id__in=experiment_list_select)
#         .exclude(id__in=experiment_failed_list)
#     )
#     print(f"一共会生成{len(experiments)}条数据")
#     # 超参
#
#     exp_list = []
#     page_list = []
#     word_dis = []
#     sen_index_dis = []
#     word_dis_positive = []
#     sen_index_dis_positive = []
#
#     success = 0
#     fail = 0
#     for experiment in experiments:
#
#         page_data_list = PageData.objects.filter(experiment_id=experiment.id)
#
#         for page_data in page_data_list:
#             word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
#
#             gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
#             result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
#                 gaze_points, page_data.texts, page_data.location, use_not_blank_assumption=True
#             )
#             pre_word = 0
#             pre_sentence = 0
#             for j, fixation in enumerate(result_fixations):
#                 index, isAdjust = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
#                 sentence_index = get_sentence_by_word(index, sentence_list)
#                 if j == 0:
#                     continue
#                 if sentence_index != -1:
#                     if sentence_index != pre_sentence:
#                         exp_list.append(experiment.id)
#                         page_list.append(page_data.id)
#                         word_dis.append(index - pre_word)
#                         sen_index_dis.append(sentence_index - pre_sentence)
#                         word_dis_positive.append((index - pre_word) if index - pre_word > 0 else pre_word - index)
#                         sen_index_dis_positive.append((
#                                                               sentence_index - pre_sentence) if sentence_index - pre_sentence > 0 else pre_sentence - sentence_index)
#
#                     pre_word = index
#                     pre_sentence = sentence_index
#         success += 1
#         logger.info(
#             "成功生成%d条,失败%d条" % (
#                 success, fail)
#         )
#     path = "jupyter\\sentence_interval.csv"
#
#     df = pd.DataFrame(
#         {
#             # 1. 实验信息相关
#             "exp": exp_list,
#             "page": page_list,
#             "word_dis": word_dis,
#             "sen_index_dis": sen_index_dis,
#             "word_dis_positive": word_dis_positive,
#             "sen_index_dis_positive": sen_index_dis_positive
#
#         }
#     )
#
#     df.to_csv(path, index=False)
#
#     return JsonResponse({"status": "ok"})
#
#
# def get_timestamp_dataset(request):
#     # filename = "exp.txt"
#     # file = open(filename, 'r')
#     # lines = file.readlines()
#     #
#     # experiment_list_select = []
#     # for line in lines:
#     #     experiment_list_select.append(line)
#
#     experiment_list_select = [597, 598]
#
#     experiment_failed_list = []
#
#     experiments = (
#         Experiment.objects.filter(is_finish=True)
#         .filter(id__in=experiment_list_select)
#         .exclude(id__in=experiment_failed_list)
#     )
#     print(f"一共会生成{len(experiments)}条数据")
#
#     # 文件路径及超参
#     interval_time = 8000
#     hand_path = "jupyter\\dataset\\" + "handcraft-data-luqi.csv"
#     cnn_path = "jupyter\\dataset\\" + "cnn-data-230127-3s.csv"
#     pic_path = "jupyter\\dataset\\" + "paint-230127-3s.csv"
#
#     # cnn相关的特征
#     experiment_ids = []
#     times = []
#     pages = []
#     gaze_x = []
#     gaze_y = []
#     gaze_t = []
#     speed = []
#     direction = []
#     acc = []
#
#     fix_x = []
#     fix_y = []
#
#     # 超参
#     success = 0
#     fail = 0
#     starttime = datetime.datetime.now()
#     for experiment in experiments:
#         # handcrafted-feature
#         experiment_id_all = []
#         user_all = []
#         article_id_all = []
#
#         time_all = []
#         word_all = []
#         word_watching_all = []
#         sentence_index = []
#
#         word_understand_all = []
#         sentence_understand_all = []
#         mind_wandering_all = []
#
#         reading_times_all = []
#         number_of_fixations_all = []
#         fixation_duration_all = []
#
#         total_dwell_time_of_sentence_all = []
#         saccade_times_of_sentence_all = []
#         forward_times_of_sentence_all = []
#         backward_times_of_sentence_all = []
#
#         saccade_times_of_sentence_vel_all = []
#         forward_times_of_sentence_vel_all = []
#         backward_times_of_sentence_vel_all = []
#
#         # 全文信息
#         words_per_page = []  # 每页的单词
#         words_of_article = []  # 整篇文本的单词
#         words_num_until_page = []  # 到该页为止的单词数量，便于计算
#         locations_per_page = []  # 每页的位置信息
#         # 标签信息
#         word_understand = []
#         sentence_understand = []
#         mind_wandering = []
#
#         try:
#             page_data_list = PageData.objects.filter(experiment_id=experiment.id)
#
#             texts = ""
#             for page_data in page_data_list:
#                 texts += page_data.texts
#             all_word_list, all_sentence_list = get_word_and_sentence_from_text(texts)  # 获取单词和句子对应的index
#             # 收集信息
#             word_num = len(all_word_list)
#             # 特征相关
#             number_of_fixations = [0 for _ in range(word_num)]
#             reading_times = [0 for _ in range(word_num)]
#             fixation_duration = [0 for _ in range(word_num)]
#
#             total_dwell_time_of_sentence_tmp = [0 for _ in range(word_num)]
#             saccade_times_of_sentence_tmp = [0 for _ in range(word_num)]
#             forward_times_of_sentence_tmp = [0 for _ in range(word_num)]
#             backward_times_of_sentence_tmp = [0 for _ in range(word_num)]
#
#             saccade_times_of_sentence_vel_tmp = [0 for _ in range(word_num)]
#             forward_times_of_sentence_vel_tmp = [0 for _ in range(word_num)]
#             backward_times_of_sentence_vel_tmp = [0 for _ in range(word_num)]
#
#             number_of_fixations_pre_page = [0 for _ in range(word_num)]
#             reading_times_pre_page = [0 for _ in range(word_num)]
#             fixation_duration_pre_page = [0 for _ in range(word_num)]
#
#             reading_times_of_sentence_tmp_pre_page = [0 for _ in range(word_num)]
#             second_pass_dwell_time_of_sentence_tmp_pre_page = [0 for _ in range(word_num)]
#             total_dwell_time_of_sentence_tmp_pre_page = [0 for _ in range(word_num)]
#             saccade_times_of_sentence_tmp_pre_page = [0 for _ in range(word_num)]
#             forward_times_of_sentence_tmp_pre_page = [0 for _ in range(word_num)]
#             backward_times_of_sentence_tmp_pre_page = [0 for _ in range(word_num)]
#
#             time = 0
#
#             for i, page_data in enumerate(page_data_list):
#
#                 word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
#
#                 if len(words_num_until_page) == 0:
#                     words_num_until_page.append(len(word_list))
#                 else:
#                     words_num_until_page.append(words_num_until_page[-1] + len(word_list))
#
#             for i, page_data in enumerate(page_data_list):
#
#                 word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)  # 获取单词和句子对应的index
#
#                 # 生成标签
#                 word_label_page, sent_label_page, wander_label_page = compute_label(
#                     page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
#                 )
#                 word_understand.extend(word_label_page)
#                 sentence_understand.extend(sent_label_page)
#                 mind_wandering.extend(wander_label_page)
#
#                 # 位置信息
#                 words_location = json.loads(
#                     page_data.location
#                 )
#                 locations_per_page.append(page_data.location)
#
#                 total_dwell_time_sent = [0 for _ in sentence_list]
#                 saccade_times_sent = [0 for _ in sentence_list]
#                 forward_times_sent = [0 for _ in sentence_list]
#                 backward_times_sent = [0 for _ in sentence_list]
#
#                 saccade_times_sent_vel = [0 for _ in sentence_list]
#                 forward_times_sent_vel = [0 for _ in sentence_list]
#                 backward_times_sent_vel = [0 for _ in sentence_list]
#
#                 assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
#
#                 words_per_page.append(word_list)
#                 words_of_article.extend(word_list)
#
#                 gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
#                 result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
#                     gaze_points, page_data.texts, page_data.location, use_not_blank_assumption=True
#                 )
#
#                 pre_word_index = -1
#                 pre_sentence_index = -1
#
#                 """word level"""
#                 begin = 0 if i == 0 else words_num_until_page[i - 1]
#                 print(f"words_num_until_page:{words_num_until_page}")
#
#                 pre_gaze = 0
#                 for m, gaze in enumerate(gaze_points):
#
#                     if m == 0:
#                         continue
#                     if gaze[-1] - gaze_points[pre_gaze][-1] > interval_time:
#
#                         number_of_fixations = [0 for _ in range(word_num)]
#                         reading_times = [0 for _ in range(word_num)]
#                         fixation_duration = [0 for _ in range(word_num)]
#
#                         total_dwell_time_sent_tmp = [0 for _ in range(word_num)]
#                         saccade_times_sent_tmp = [0 for _ in range(word_num)]
#                         forward_times_sent_tmp = [0 for _ in range(word_num)]
#                         backward_times_sent_tmp = [0 for _ in range(word_num)]
#
#                         saccade_times_sent_vel_tmp = [0 for _ in range(word_num)]
#                         forward_times_sent_vel_tmp = [0 for _ in range(word_num)]
#                         backward_times_sent_vel_tmp = [0 for _ in range(word_num)]
#
#                         fixations_before = get_fix_by_time(result_fixations, gaze[-1])
#                         fixations_now = get_fix_this_time(result_fixations, gaze_points[pre_gaze][-1], gaze[-1])
#                         for j, fixation in enumerate(fixations_before):
#
#                             index, isAdjust = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
#                             if index != -1:
#                                 number_of_fixations[index + begin] += 1
#                                 fixation_duration[index + begin] += fixation[2]
#                                 if index != pre_word_index:
#                                     reading_times[index + begin] += 1
#
#                                     # 计算sentence相关
#                                     sentence_index = get_sentence_by_word(pre_word_index, sentence_list)
#                                     saccade_times_sent[sentence_index] += 1
#                                     if index > pre_word_index:
#                                         forward_times_sent[sentence_index] += 1
#                                     else:
#                                         backward_times_sent[sentence_index] += 1
#
#                                     pre_word_index = index
#
#                         """sentence level"""
#                         for j, fixation in enumerate(fixations_before):
#                             index, isAdjust = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
#                             sentence_index = get_sentence_by_word(index, sentence_list)
#                             if sentence_index != -1:
#                                 # 累积fixation duration
#                                 total_dwell_time_sent[sentence_index] += fixation[2]
#
#                         saccades_vel, velocity = detect_saccades(fixations_before)
#                         for saccade in saccades_vel:
#                             sac_begin = saccade["begin"]
#                             sac_end = saccade["end"]
#                             begin_word, isAdjust = get_item_index_x_y(page_data.location, sac_begin[0], sac_begin[1])
#                             end_word, isAdjust = get_item_index_x_y(page_data.location, sac_end[0], sac_end[1])
#                             sentence_index = get_sentence_by_word(begin_word, sentence_list)
#                             saccade_times_sent_vel[sentence_index] += 1
#                             if end_word > begin_word:
#                                 forward_times_sent_vel[sentence_index] += 1
#                             else:
#                                 backward_times_sent_vel[sentence_index] += 1
#
#                         for q, sentence in enumerate(sentence_list):
#                             # scale = math.log((sentence[3] + 1))
#                             # scale = sentence[3] + 1
#                             scale = 1
#                             begin_index = sentence[1] + begin
#                             end_index = sentence[2] + begin
#                             length = end_index - begin_index
#
#                             total_dwell_time_of_sentence_tmp[begin_index:end_index] = [
#                                 total_dwell_time_sent[q] / scale for _ in range(length)
#                             ]
#                             saccade_times_of_sentence_tmp[begin_index:end_index] = [
#                                 saccade_times_sent[q] / scale for _ in range(length)
#                             ]
#                             forward_times_of_sentence_tmp[begin_index:end_index] = [
#                                 forward_times_sent[q] / scale for _ in range(length)
#                             ]
#                             backward_times_of_sentence_tmp[begin_index:end_index] = [
#                                 backward_times_sent[q] / scale for _ in range(length)
#                             ]
#
#                             saccade_times_of_sentence_vel_tmp[begin_index:end_index] = [
#                                 saccade_times_sent_vel[q] / scale for _ in range(length)
#                             ]
#                             forward_times_of_sentence_vel_tmp[begin_index:end_index] = [
#                                 forward_times_sent_vel[q] / scale for _ in range(length)
#                             ]
#                             backward_times_of_sentence_vel_tmp[begin_index:end_index] = [
#                                 backward_times_sent_vel[q] / scale for _ in range(length)
#                             ]
#
#                         experiment_id_all.extend([experiment.id for _ in range(word_num)])
#                         user_all.extend([experiment.user for _ in range(word_num)])
#                         article_id_all.extend([experiment.article_id for _ in range(word_num)])
#                         time_all.extend([time for _ in range(word_num)])
#                         word_all.extend(all_word_list)
#
#                         tmp = [0 for i in range(word_num)]
#                         record = []
#                         for fix in fixations_now:
#                             index, isAdjust = get_item_index_x_y(page_data.location, fix[0], fix[1])
#                             if index != -1:
#                                 tmp[index + begin] = 1
#                                 record.append(index)
#                         print(f'当前关注的是:{record}')
#
#                         word_watching_all.extend(tmp)
#
#                         word_understand_all.extend(word_understand)
#                         sentence_understand_all.extend(sentence_understand)
#                         mind_wandering_all.extend(mind_wandering)
#
#                         reading_times_all.extend(np.sum([reading_times, reading_times_pre_page], axis=0).tolist())
#                         number_of_fixations_all.extend(
#                             np.sum([number_of_fixations, number_of_fixations_pre_page], axis=0).tolist())
#                         fixation_duration_all.extend(
#                             np.sum([fixation_duration, fixation_duration_pre_page], axis=0).tolist())
#
#                         total_dwell_time_of_sentence_all.extend(
#                             np.sum([total_dwell_time_of_sentence_tmp, total_dwell_time_of_sentence_tmp_pre_page],
#                                    axis=0).tolist())
#
#                         saccade_times_of_sentence_all.extend(
#                             np.sum([saccade_times_of_sentence_tmp, saccade_times_of_sentence_tmp_pre_page],
#                                    axis=0).tolist())
#                         forward_times_of_sentence_all.extend(
#                             np.sum([forward_times_of_sentence_tmp, forward_times_of_sentence_tmp_pre_page],
#                                    axis=0).tolist())
#                         backward_times_of_sentence_all.extend(
#                             np.sum([backward_times_of_sentence_tmp, backward_times_of_sentence_tmp_pre_page],
#                                    axis=0).tolist())
#
#                         experiment_ids.append(experiment.id)
#                         times.append(time)
#                         pages.append(i)
#
#                         # 挑出对应的gaze点
#                         gazes = gaze_points[pre_gaze:m]
#
#                         fix_of_x = [x[0] for x in fixations_now]
#                         fix_of_y = [x[1] for x in fixations_now]
#                         fix_x.append(fix_of_x)
#                         fix_y.append(fix_of_y)
#
#                         gaze_of_x = [x[0] for x in gazes]
#                         gaze_of_y = [x[1] for x in gazes]
#                         gaze_of_t = [x[2] for x in gazes]
#                         speed_now, direction_now, acc_now = coor_to_input(gazes, 8)
#                         assert len(gaze_of_x) == len(gaze_of_y) == len(speed_now) == len(direction_now) == len(acc_now)
#                         gaze_x.append(gaze_of_x)
#                         gaze_y.append(gaze_of_y)
#                         gaze_t.append(gaze_of_t)
#                         speed.append(speed_now)
#                         direction.append(direction_now)
#                         acc.append(acc_now)
#
#                         time += 1
#                         pre_gaze = m
#
#                 number_of_fixations_pre_page = np.sum([number_of_fixations, number_of_fixations_pre_page],
#                                                       axis=0).tolist()
#                 reading_times_pre_page = np.sum([reading_times, reading_times_pre_page], axis=0).tolist()
#                 fixation_duration_pre_page = np.sum([fixation_duration, fixation_duration_pre_page], axis=0).tolist()
#
#                 print("reading_times_pre_page")
#                 print(reading_times_pre_page)
#
#                 second_pass_dwell_time_of_sentence_tmp_pre_page = np.sum(
#                     [second_pass_dwell_time_of_sentence_tmp, second_pass_dwell_time_of_sentence_tmp_pre_page],
#                     axis=0).tolist()
#                 total_dwell_time_of_sentence_tmp_pre_page = np.sum(
#                     [total_dwell_time_of_sentence_tmp, total_dwell_time_of_sentence_tmp_pre_page], axis=0).tolist()
#                 reading_times_of_sentence_tmp_pre_page = np.sum(
#                     [reading_times_of_sentence_tmp, reading_times_of_sentence_tmp_pre_page], axis=0).tolist()
#                 saccade_times_of_sentence_tmp_pre_page = np.sum(
#                     [saccade_times_of_sentence_tmp, saccade_times_of_sentence_tmp_pre_page], axis=0).tolist()
#                 forward_times_of_sentence_tmp_pre_page = np.sum(
#                     [forward_times_of_sentence_tmp, forward_times_of_sentence_tmp_pre_page], axis=0).tolist()
#                 backward_times_of_sentence_tmp_pre_page = np.sum(
#                     [backward_times_of_sentence_tmp, backward_times_of_sentence_tmp_pre_page], axis=0).tolist()
#
#                 number_of_fixations = [0 for _ in range(word_num)]
#                 reading_times = [0 for _ in range(word_num)]
#                 fixation_duration = [0 for _ in range(word_num)]
#
#                 reading_times_of_sentence_tmp = [0 for _ in range(word_num)]
#                 second_pass_dwell_time_of_sentence_tmp = [0 for _ in range(word_num)]
#                 total_dwell_time_of_sentence_tmp = [0 for _ in range(word_num)]
#                 saccade_times_of_sentence_tmp = [0 for _ in range(word_num)]
#                 forward_times_of_sentence_tmp = [0 for _ in range(word_num)]
#                 backward_times_of_sentence_tmp = [0 for _ in range(word_num)]
#
#             print(len(experiment_id_all))
#             print(len(user_all))
#             print(len(article_id_all))
#             print(len(time_all))
#             print(len(word_all))
#             print(len(word_watching_all))
#             print(len(word_understand_all))
#             print(len(sentence_understand_all))
#             print(len(mind_wandering_all))
#             print(len(reading_times_all))
#             print(len(number_of_fixations_all))
#             print(len(fixation_duration_all))
#             print(len(total_dwell_time_of_sentence_all))
#             print(len(saccade_times_of_sentence_all))
#             print(len(forward_times_of_sentence_all))
#             print(len(backward_times_of_sentence_all))
#
#             # 生成手工数据集
#             df = pd.DataFrame(
#                 {
#                     # 1. 实验信息相关
#                     "experiment_id": experiment_id_all,
#                     "user": user_all,
#                     "article_id": article_id_all,
#                     "time": time_all,
#                     "word": word_all,
#                     "word_watching": word_watching_all,
#                     # # 2. label相关
#                     "word_understand": word_understand_all,
#                     "sentence_understand": sentence_understand_all,
#                     "mind_wandering": mind_wandering_all,
#                     # 3. 特征相关
#                     # word level
#                     "reading_times": reading_times_all,
#                     "number_of_fixations": number_of_fixations_all,
#                     "fixation_duration": fixation_duration_all,
#                     # sentence level
#                     "total_dwell_time_of_sentence": total_dwell_time_of_sentence_all,
#                     "saccade_times_of_sentence": saccade_times_of_sentence_all,
#                     "forward_times_of_sentence": forward_times_of_sentence_all,
#                     "backward_times_of_sentence": backward_times_of_sentence_all,
#
#                 }
#             )
#
#             if os.path.exists(hand_path):
#                 df.to_csv(hand_path, index=False, mode="a", header=False)
#             else:
#                 df.to_csv(hand_path, index=False, mode="a")
#
#             success += 1
#             endtime = datetime.datetime.now()
#             logger.info(
#                 "成功生成%d条,失败%d条,耗时为%ss" % (
#                     success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
#             )
#
#         except Exception as e:
#             fail += 1
#             endtime = datetime.datetime.now()
#             logger.info(
#                 "成功生成%d条,失败%d条,耗时为%ss" % (
#                     success, fail, round((endtime - starttime).microseconds / 1000 / 1000, 3))
#             )
#             experiment_failed_list.append(experiment.id)
#
#     # 生成cnn的数据集
#     data = pd.DataFrame(
#         {
#             # 1. 实验信息相关
#             "experiment_id": experiment_ids,
#             "time": times,
#             "gaze_x": gaze_x,
#             "gaze_y": gaze_y,
#             "gaze_t": gaze_t,
#             "speed": speed,
#             "direction": direction,
#             "acc": acc,
#         }
#     )
#
#     if os.path.exists(cnn_path):
#         data.to_csv(cnn_path, index=False, mode="a", header=False)
#     else:
#         data.to_csv(cnn_path, index=False, mode="a")
#
#     data1 = pd.DataFrame(
#         {
#             # 1. 实验信息相关
#             "experiment_id": experiment_ids,
#             "time": times,
#             "page": pages,
#             "gaze_x": gaze_x,
#             "gaze_y": gaze_y,
#             "gaze_t": gaze_t,
#
#             "fix_x": fix_x,
#             "fix_y": fix_y
#
#         }
#     )
#
#     if os.path.exists(pic_path):
#         data1.to_csv(pic_path, index=False, mode="a", header=False)
#     else:
#         data1.to_csv(pic_path, index=False, mode="a")
#     logger.info("成功生成%d条，失败%d条" % (success, fail))
#     logger.info("成功生成%d条，失败%d条" % (success, fail))
#     logger.info(f"失败的experiment有:{experiment_failed_list}")
#     return JsonResponse({"status": "ok"})
#
#
# def get_interval_dataset(request):
#     optimal_list = [
#         [574, 580],
#         [582],
#         [585, 588],
#         [590, 591],
#         [595, 598],
#         [600, 605],
#         [609, 610],
#         [613, 619],
#         [622, 625],
#         [628],
#         [630, 631],
#         [634],
#         [636],
#         [637, 641],
#     ]
#
#     experiment_list_select = []
#     for item in optimal_list:
#         if len(item) == 2:
#             for i in range(item[0], item[1] + 1):
#                 experiment_list_select.append(i)
#         if len(item) == 1:
#             experiment_list_select.append(item[0])
#     experiments = Experiment.objects.filter(is_finish=True).filter(id__in=experiment_list_select)
#     print(len(experiments))
#
#     success = 0
#     fail = 0
#     interval_list = []
#     for experiment in experiments:
#         try:
#             page_data_list = PageData.objects.filter(experiment_id=experiment.id)
#             for page_data in page_data_list:
#                 gaze_points_this_page = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
#                 fixations = keep_row(detect_fixations(gaze_points_this_page))
#
#                 word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
#
#                 words_location = json.loads(
#                     page_data.location
#                 )  # [{'left': 330, 'top': 95, 'right': 435.109375, 'bottom': 147},...]
#                 assert len(word_list) == len(words_location)  # 确保单词分割的是正确的
#
#                 word_understand_in_page, sentence_understand_in_page, mind_wander_in_page = compute_label(
#                     page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
#                 )
#
#                 first_time = [0 for _ in word_list]
#                 last_time = [0 for _ in word_list]
#                 for fixation in fixations:
#                     index = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
#                     if index != -1:
#                         if first_time[index] == 0:
#                             first_time[index] = fixation[3]
#                             last_time[index] = fixation[4]
#                         else:
#                             last_time[index] = fixation[4]
#
#                 interval = list(map(lambda x: x[0] - x[1], zip(last_time, first_time)))
#
#                 interval = [item for i, item in enumerate(interval) if item > 0 and word_understand_in_page[i] == 0]
#
#                 interval_list.extend(interval)
#             success += 1
#             print("成功%d条，失败%d条" % (success, fail))
#         except:
#             fail += 1
#             print("成功%d条，失败%d条" % (success, fail))
#     pd.DataFrame({"interval": interval_list}).to_csv(
#         "D:\\qxy\\reading-new\\reading\\jupyter\data\\interval.csv", index=False
#     )
#     return JsonResponse({"status": "ok"})
#
#
# def get_nlp_sequence(request):
#     experiment_id = request.GET.get("exp_id")
#     article_id = Experiment.objects.get(id=experiment_id).article_id
#
#     texts = ""
#     paras = Paragraph.objects.filter(article_id=article_id).order_by("para_id")
#     for para in paras:
#         texts += para.content
#
#     word_list, sentence_list = get_word_and_sentence_from_text(texts)
#     difficulty_level = [get_word_difficulty(x) for x in word_list]  # text feature
#     difficulty_level = normalize(difficulty_level)
#     word_attention = generate_word_attention(texts)
#     importance = get_importance(texts)
#
#     importance_level = [0 for _ in word_list]
#     attention_level = [0 for _ in word_list]
#     for q, word in enumerate(word_list):
#         for impo in importance:
#             if impo[0] == word:
#                 importance_level[q] = impo[1]
#         for att in word_attention:
#             if att[0] == word:
#                 attention_level[q] = att[1]
#     importance_level = normalize(importance_level)
#     attention_level = normalize(attention_level)
#
#     nlp_word_list, word4show_list = generate_word_list(texts)
#     nlp_feature = [(
#         (1 / 3) * difficulty_level[i] + (1 / 3) * importance_level[i] + (1 / 3) * attention_level[i]
#         , nlp_word_list[i]) for i in range(len(word_list))
#     ]
#     print(nlp_feature)
#     print(len(nlp_feature))
#
#     import matplotlib.pyplot as plt
#     import pandas as pd
#
#     columns1 = ["index1", "value1"]
#
#     fig = plt.figure(figsize=(50, 4), dpi=100)
#
#     data1 = []
#     csv = pd.read_csv(r"D:\qxy\reading-new\reading\jupyter\dataset\data_after_norm.csv")
#     csv = csv[csv["experiment_id"] == 577]
#     columns = ["index", "value"]
#     print(len(csv))
#
#     data = []
#     data_not_understand = []
#     line = []
#     i = 0
#
#     word_feature = []
#     for index, row in csv.iterrows():
#         word_feature.append(row["word"])
#         for feature in nlp_feature:
#             if feature[1].lower().strip() == str(row["word"]).lower().strip():
#                 data1.append([i, feature[0]])
#
#                 value = 0.4 * row["number_of_fixations"] + 0.4 * row["fixation_duration"] + 0.2 * row["reading_times"]
#                 if row["word_understand"] == 0:
#                     data_not_understand.append([i, value])
#                 data.append([i, value])
#                 line.append([i, 0.5])
#                 i += 1
#                 break
#     df1 = pd.DataFrame(data=data1, columns=columns1)
#     df = pd.DataFrame(data=data, columns=columns)
#     line = pd.DataFrame(data=line, columns=columns)
#     df_1 = pd.DataFrame(data=data_not_understand, columns=columns)
#
#     plt.plot(df1["index1"], df1["value1"], lw=3, ls="-", color="green", zorder=0, alpha=0.3)
#     plt.plot(df["index"], df["value"], lw=3, ls="-", color="black", zorder=0, alpha=0.3)
#     plt.plot(line["index"], line["value"], lw=3, ls="-", color="orange", zorder=0, alpha=0.3)
#     plt.scatter(df_1["index"], df_1["value"], color="red", zorder=1, s=60)
#     plt.show()
#     return JsonResponse({"visual": word_feature, "nlp": nlp_word_list})
#
#
# def get_label_num(request):
#     filename = "exp.txt"
#     file = open(filename, 'r')
#     lines = file.readlines()
#
#     exp_list = []
#     for line in lines:
#         exp_list.append(line)
#
#     print(f"exp_list:{exp_list}")
#
#     exps = Experiment.objects.filter(id__in=exp_list)
#
#     word_label_num = 0
#     sen_label_num = 0
#     wander_label_num = 0
#
#     for exp in exps:
#         page_data_ls = PageData.objects.filter(experiment_id=exp.id)
#         for page in page_data_ls:
#             word_label_num += len(json.loads(page.wordLabels))
#             sen_label_num += len(json.loads(page.sentenceLabels))
#             wander_label_num += len(json.loads(page.wanderLabels))
#
#     print(f"单词标签的数量：{word_label_num}")
#     print(f"句子标签的数量：{sen_label_num}")
#     print(f"走神标签的数量：{wander_label_num}")
#
#     return HttpResponse(1)
#
#
# def get_handcraft_feature(request):
#     filename = "exp.txt"
#     file = open(filename, 'r')
#     lines = file.readlines()
#
#     experiment_list_select = []
#     for line in lines:
#         experiment_list_select.append(line)
#
#     # experiment_list_select = [506]
#     experiment_failed_list = []
#
#     experiments = (
#         Experiment.objects.filter(is_finish=True)
#         .filter(id__in=experiment_list_select)
#         .exclude(id__in=experiment_failed_list)
#     )
#     print(f"一共会生成{len(experiments)}条数据")
#
#     interval = 2 * 1000
#
#     success = 0
#     for experiment in experiments:
#         time = 0
#         # 一次实验保存一次
#         page_data_list = PageData.objects.filter(experiment_id=experiment.id)
#
#         # 先生成不同page的feature
#         feature_list = []
#         for page_data in page_data_list:
#             featureSet = FeatureSet(get_word_count(page_data.texts))
#
#             word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
#             word_understand, sentence_understand, mind_wander = compute_label(
#                 page_data.wordLabels, page_data.sentenceLabels, page_data.wanderLabels, word_list
#             )
#
#             featureSet.setLabel(word_understand, sentence_understand, mind_wander)
#             featureSet.setWordList(word_list)
#
#             feature_list.append(featureSet)
#
#         # 不是第一页，要把上一页的信息加上
#         for p, page_data in enumerate(page_data_list):
#             featureSet = feature_list[p]
#
#             word_list, sentence_list = get_word_and_sentence_from_text(page_data.texts)
#             border, rows, danger_zone, len_per_word = textarea(page_data.location)
#
#             gaze_points = format_gaze(page_data.gaze_x, page_data.gaze_y, page_data.gaze_t)
#             result_fixations, row_sequence, row_level_fix, sequence_fixations = process_fixations(
#                 gaze_points, page_data.texts, page_data.location, use_not_blank_assumption=True
#             )
#
#             pre_gaze = 0
#             for g, gaze in enumerate(gaze_points):
#                 if g == 0:
#                     continue
#                 if gaze[-1] - gaze_points[pre_gaze][-1] > interval:
#                     fixations_before = get_fix_by_time(result_fixations, gaze[-1])
#                     fixations_now = get_fix_this_time(result_fixations, gaze_points[pre_gaze][-1], gaze[-1])
#
#                     pre_gaze = g
#                     pre_word_index = -1
#                     featureSet.clean()
#                     for i, fixation in enumerate(fixations_before):
#                         word_index, isAdjust = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
#                         if word_index != -1:
#                             featureSet.number_of_fixation[word_index] += 1
#                             featureSet.total_fixation_duration[word_index] += fixation[2]
#
#                             pre_row = get_row(pre_word_index, rows)
#                             now_row = get_row(word_index, rows)
#
#                             if word_index != pre_word_index:
#                                 featureSet.reading_times[word_index] += 1
#
#                                 # 计算sentence的saccade相关
#                                 sent_index = get_sentence_by_word(pre_word_index, sentence_list)
#                                 sentence = sentence_list[sent_index]
#                                 for j in range(sentence[1], sentence[2]):
#                                     featureSet.saccade_times[j] += 1
#                                     if i != 0:
#                                         featureSet.saccade_duration[j] += result_fixations[i][3] - \
#                                                                           result_fixations[i - 1][4]
#                                         featureSet.saccade_distance[j] += get_euclid_distance(result_fixations[i][0],
#                                                                                               result_fixations[i - 1][
#                                                                                                   0],
#                                                                                               result_fixations[i][1],
#                                                                                               result_fixations[i - 1][1])
#                                 if word_index > pre_word_index:
#                                     for j in range(sentence[1], sentence[2]):
#                                         featureSet.forward_saccade_times[j] += 1
#                                 else:
#                                     for j in range(sentence[1], sentence[2]):
#                                         featureSet.backward_saccade_times[j] += 1
#
#                                 if pre_row == now_row:
#                                     for j in range(sentence[1], sentence[2]):
#                                         featureSet.horizontal_saccade[j] += 1
#
#                                 pre_word_index = word_index
#                     for i, fixation in enumerate(fixations_before):
#                         word_index, isAdjust = get_item_index_x_y(page_data.location, fixation[0], fixation[1])
#                         sentence_index = get_sentence_by_word(word_index, sentence_list)
#
#                         if sentence_index != -1:
#                             sentence = sentence_list[sentence_index]
#                             # 累积fixation duration
#                             for j in range(sentence[1], sentence[2]):
#                                 featureSet.total_dwell_time[j] += fixation[2]
#
#                     saccades_vel, velocity = detect_saccades(fixations_before)
#
#                     for saccade in saccades_vel:
#                         sac_begin = saccade["begin"]
#                         sac_end = saccade["end"]
#                         begin_word, isAdjust = get_item_index_x_y(page_data.location, sac_begin[0], sac_begin[1])
#                         end_word, isAdjust = get_item_index_x_y(page_data.location, sac_end[0], sac_end[1])
#                         sent_index = get_sentence_by_word(begin_word, sentence_list)
#                         sentence = sentence_list[sent_index]
#
#                         for j in range(sentence[1], sentence[2]):
#                             featureSet.saccade_times_vel[j] += 1
#
#                         if end_word > begin_word:
#                             for j in range(sentence[1], sentence[2]):
#                                 featureSet.forward_saccade_times_vel[j] += 1
#                         else:
#                             for j in range(sentence[1], sentence[2]):
#                                 featureSet.backward_saccade_times_vel[j] += 1
#
#                     for sentence in sentence_list:
#                         for i in range(sentence[1], sentence[2]):
#                             featureSet.sentence_length[i] = sentence[3]
#
#                     # 生成单词所在句子
#                     for i, word in enumerate(word_list):
#                         sent_index = get_sentence_by_word(i, sentence_list)
#                         featureSet.sentence_index[i] = sent_index
#
#                     for fix in fixations_now:
#                         index, isAdjust = get_item_index_x_y(page_data.location, fix[0], fix[1])
#                         if index != -1:
#                             featureSet.is_watching[index] = 1
#
#                     for feature in feature_list:
#                         feature.to_csv('jupyter/dataset/handcraft-0212-2s.csv', experiment.id, experiment.user,
#                                        experiment.article_id, page_data.id, time)
#
#                     time += 1
#                     featureSet.clean_watching()
#
#         success += 1
#         logger.info("成功生成%d条" % (success))
#     return HttpResponse(1)
#
#
# class FeatureSet(object):
#
#     def __init__(self, num):
#         self.num = num
#         # 特征
#         self.total_fixation_duration = [0 for _ in range(num)]
#         self.number_of_fixation = [0 for _ in range(num)]
#         self.reading_times = [0 for _ in range(num)]
#
#         self.total_dwell_time = [0 for _ in range(num)]
#         self.saccade_times = [0 for _ in range(num)]
#         self.forward_saccade_times = [0 for _ in range(num)]
#         self.backward_saccade_times = [0 for _ in range(num)]
#
#         self.saccade_times_vel = [0 for _ in range(num)]
#         self.forward_saccade_times_vel = [0 for _ in range(num)]
#         self.backward_saccade_times_vel = [0 for _ in range(num)]
#
#         self.saccade_duration = [0 for _ in range(num)]
#         self.saccade_amplitude = [0 for _ in range(num)]
#         self.saccade_velocity = [0 for _ in range(num)]
#         self.number_of_saccade = [0 for _ in range(num)]
#
#         self.horizontal_saccade = [0 for _ in range(num)]
#         self.saccade_distance = [0 for _ in range(num)]
#
#         # 句子长度
#         self.sentence_length = [0 for _ in range(num)]
#
#         # label
#         self.word_understand = []
#         self.sentence_understand = []
#         self.mind_wandering = []
#
#         self.page_data = [0 for _ in range(num)]
#         self.sentence_index = [0 for _ in range(num)]
#
#         # word_list
#         self.word_list = []
#
#         # watching
#         self.is_watching = [0 for _ in range(num)]
#
#     def clean(self):
#         self.total_fixation_duration = [0 for _ in range(self.num)]
#         self.number_of_fixation = [0 for _ in range(self.num)]
#         self.reading_times = [0 for _ in range(self.num)]
#
#         self.total_dwell_time = [0 for _ in range(self.num)]
#         self.saccade_times = [0 for _ in range(self.num)]
#         self.forward_saccade_times = [0 for _ in range(self.num)]
#         self.backward_saccade_times = [0 for _ in range(self.num)]
#
#         self.saccade_times_vel = [0 for _ in range(self.num)]
#         self.forward_saccade_times_vel = [0 for _ in range(self.num)]
#         self.backward_saccade_times_vel = [0 for _ in range(self.num)]
#
#         self.saccade_duration = [0 for _ in range(self.num)]
#         self.saccade_amplitude = [0 for _ in range(self.num)]
#         self.saccade_velocity = [0 for _ in range(self.num)]
#         self.number_of_saccade = [0 for _ in range(self.num)]
#
#         self.horizontal_saccade = [0 for _ in range(self.num)]
#         self.saccade_distance = [0 for _ in range(self.num)]
#
#     def clean_watching(self):
#         self.is_watching = [0 for _ in range(self.num)]
#
#     def setLabel(self, label1, label2, label3):
#         self.word_understand = label1
#         self.sentence_understand = label2
#         self.mind_wandering = label3
#
#     def setWordList(self, word_list):
#         self.word_list = word_list
#
#     def get_list_div(self, list_a, list_b):
#         div_list = [0 for _ in range(self.num)]
#         for i in range(len(list_b)):
#             if list_b[i] != 0:
#                 div_list[i] = list_a[i] / list_b[i]
#
#         return div_list
#
#     def list_log(self, list_a):
#         log_list = [0 for _ in range(self.num)]
#         for i in range(len(list_a)):
#             log_list[i] = math.log(list_a[i] + 1)
#         return log_list
#
#     def to_csv(self, filename, exp_id, user, article_id, page_id, time):
#         df = pd.DataFrame(
#             {
#                 # 1. 实验信息相关
#                 "experiment_id": [exp_id for _ in range(self.num)],
#                 "user": [user for _ in range(self.num)],
#                 "article_id": [article_id for _ in range(self.num)],
#                 "time": [time for _ in range(self.num)],
#                 "word": self.word_list,
#
#                 "page_id": [page_id for _ in range(self.num)],
#                 "sentence": self.sentence_index,
#                 "word_watching": self.is_watching,
#
#                 # # 2. label相关
#                 "word_understand": self.word_understand,
#                 "sentence_understand": self.sentence_understand,
#                 "mind_wandering": self.mind_wandering,
#                 # 3. 特征相关
#                 # word level
#                 "reading_times": self.reading_times,
#                 "number_of_fixations": self.number_of_fixation,
#                 "fixation_duration": self.total_fixation_duration,
#                 # sentence_level_raw
#                 "total_dwell_time_of_sentence": self.total_dwell_time,
#                 "saccade_times_of_sentence": self.saccade_times,
#                 "forward_times_of_sentence": self.forward_saccade_times,
#                 "backward_times_of_sentence": self.backward_saccade_times,
#
#                 "saccade_times_of_sentence_vel": self.saccade_times_vel,
#                 "forward_times_of_sentence_vel": self.forward_saccade_times_vel,
#                 "backward_times_of_sentence_vel": self.backward_saccade_times_vel,
#                 "saccade_duartion": self.saccade_duration,
#
#                 "horizontal_saccade_proportion": self.get_list_div(self.horizontal_saccade, self.saccade_times),
#                 "saccade_velocity": self.get_list_div(self.saccade_distance, self.saccade_duration),
#
#                 # sentence_level_div_length
#                 "total_dwell_time_of_sentence_div_length": self.get_list_div(self.total_dwell_time,
#                                                                              self.sentence_length),
#                 "saccade_times_of_sentence_div_length": self.get_list_div(self.saccade_times, self.sentence_length),
#                 "forward_times_of_sentence_div_length": self.get_list_div(self.forward_saccade_times,
#                                                                           self.sentence_length),
#                 "backward_times_of_sentence_div_length": self.get_list_div(self.backward_saccade_times,
#                                                                            self.sentence_length),
#
#                 "saccade_times_of_sentence_vel_div_length": self.get_list_div(self.saccade_times_vel,
#                                                                               self.sentence_length),
#                 "forward_times_of_sentence_vel_div_length": self.get_list_div(self.forward_saccade_times_vel,
#                                                                               self.sentence_length),
#                 "backward_times_of_sentence_vel_div_length": self.get_list_div(self.backward_saccade_times_vel,
#                                                                                self.sentence_length),
#
#                 "saccade_duartion_div_length": self.get_list_div(self.saccade_duration, self.sentence_length),
#                 "horizontal_saccade_proportion_div_length": self.get_list_div(self.get_list_div(self.horizontal_saccade,
#                                                                                                 self.saccade_times),
#                                                                               self.sentence_length),
#                 "saccade_velocity_div_length": self.get_list_div(
#                     self.get_list_div(self.saccade_distance, self.saccade_duration), self.sentence_length),
#
#                 # sentence_level_div_log
#                 "total_dwell_time_of_sentence_div_log": self.get_list_div(self.total_dwell_time,
#                                                                           self.list_log(self.sentence_length)),
#                 "saccade_times_of_sentence_div_log": self.get_list_div(self.saccade_times,
#                                                                        self.list_log(self.sentence_length)),
#                 "forward_times_of_sentence_div_log": self.get_list_div(self.forward_saccade_times,
#                                                                        self.list_log(self.sentence_length)),
#                 "backward_times_of_sentence_div_log": self.get_list_div(self.backward_saccade_times,
#                                                                         self.list_log(self.sentence_length)),
#
#                 "saccade_times_of_sentence_vel_div_log": self.get_list_div(self.saccade_times_vel,
#                                                                            self.list_log(self.sentence_length)),
#                 "forward_times_of_sentence_vel_div_log": self.get_list_div(self.forward_saccade_times_vel,
#                                                                            self.list_log(self.sentence_length)),
#                 "backward_times_of_sentence_vel_div_log": self.get_list_div(self.backward_saccade_times_vel,
#                                                                             self.list_log(self.sentence_length)),
#
#                 "saccade_duartion_div_log": self.get_list_div(self.saccade_duration,
#                                                               self.list_log(self.sentence_length)),
#                 "horizontal_saccade_proportion_div_log": self.get_list_div(self.get_list_div(self.horizontal_saccade,
#                                                                                              self.saccade_times),
#                                                                            self.list_log(self.sentence_length)),
#                 "saccade_velocity_div_log": self.get_list_div(
#                     self.get_list_div(self.saccade_distance, self.saccade_duration),
#                     self.list_log(self.sentence_length)),
#
#             }
#         )
#
#         if os.path.exists(filename):
#             df.to_csv(filename, index=False, mode="a", header=False)
#         else:
#             df.to_csv(filename, index=False, mode="a")
