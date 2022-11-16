import cv2
import pandas as pd
from django.http import JsonResponse
from PIL import Image

# Create your views here.
from action.models import Experiment, PageData
from feature.utils import detect_fixations, detect_saccades, gaze_map, join_images, show_fixations_and_saccades, \
    preprocess_data, paint_fixations, format_gaze
from pyheatmap import myHeatmap
from utils import generate_pic_by_base64

"""
所有与eye gaze计算的函数都写在这里
TODO list
1. fixation和saccade的计算
    1.1 配合图的生成看一下
    1.2 与单词的位置无关
"""


def classify_gaze_2_label_in_pic(request):
    page_data_id = request.GET.get("id")
    begin = request.GET.get("begin", 0)
    end = request.GET.get("end", -1)
    pageData = PageData.objects.get(id=page_data_id)

    gaze_points = format_gaze(pageData, filter=True)[begin:end]

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
    # generate saccades
    saccades, velocities = detect_saccades(fixations)  # todo:default argument should be adjust to optimal
    # plt using fixations and saccade
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

    return JsonResponse({"code": 200, "status": "生成成功"}, json_dumps_params={"ensure_ascii": False})


def generate_tmp_pic(request):
    page_data_id = request.GET.get("id")
    pageData = PageData.objects.get(id=page_data_id)

    gaze_points = format_gaze(pageData, filter=True)

    base_path = "pic\\" + str(page_data_id) + "\\"

    background = generate_pic_by_base64(pageData.image, base_path, "background.png")

    gaze_4_heat = [[x[0], x[1]] for x in gaze_points]
    myHeatmap.draw_heat_map(gaze_4_heat, base_path + "heatmap.png", background)
    fixations = detect_fixations(gaze_points)
    canvas = paint_fixations(cv2.imread(base_path + "heatmap.png"), fixations, interval=3, label=3)
    cv2.imwrite(base_path + "fix_on_heat.png", canvas)

    return JsonResponse({"code": 200, "status": "生成成功"}, json_dumps_params={"ensure_ascii": False})