import json
import math
import os

from django.http import HttpResponse
from django.shortcuts import render

from analysis.models import PageData, Experiment
from pyheatmap import myHeatmap
from tools import format_gaze, generate_fixations, generate_pic_by_base64, show_fixations, get_word_location, \
    paint_on_word
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


