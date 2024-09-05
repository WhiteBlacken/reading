import json

import pandas as pd
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.db.models import QuerySet
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger
from django.shortcuts import redirect

from analysis.feature import WordFeature, SentFeature
from analysis.models import Text, Paragraph, Translation, Dictionary, Experiment, PageData
from tools import login_required, translate, Timer, simplify_word, simplify_sentence, get_word_and_sentence_from_text, \
    format_gaze, detect_fixations, get_item_index_x_y, get_sentence_by_word, get_euclid_distance, textarea, \
    get_row

# from tools import freq_dist


# from autogluon.multimodal import MultiModalPredictor

# wordPredictor = MultiModalPredictor.load('model/word')
# sentPredictor = MultiModalPredictor.load('model/sent')
# wanderPredictor = MultiModalPredictor.load('model/wander')


def go_login(request):
    """
    跳转去登录页面
    """
    request.session.flush()
    return render(request, "login.html")


def my_login(request):
    """
    登录
    ：简单的登录逻辑，记下用户名
    """
    username = request.POST.get("username", None)
    psw = request.POST.get("psw", None)
    device = request.POST.get("device", None)
    request.session["device"] = device
    needCorrection = request.POST.get("needCorrection", None)
    print(f"needCorrection:{needCorrection}")

    user = authenticate(username=username, password=psw)
    if user is None:
        user = User.objects.create_user(username=username, password=psw)
        user.save()
    login(request, user)
    
    if needCorrection == "true":
        return render(request, "calibration.html")
    else:
        return redirect(choose_text)


def choose_text(request):
    """
    选择读的文章
    """
    texts = Text.objects.filter(is_show=True)
    return render(request, "chooseTxt.html", {"texts": texts})


@login_required
def reading(request):
    """
    进入阅读页面,分为数据收集/阅读辅助两种
    """
    reading_type = request.GET.get('type', '1')
    native = request.GET.get('native', '1')
    request.session['role'] = 'native' if native == '1' else "nonnative"

    if reading_type == '1':
        return render(request, "reading_for_data_1.html")
    else:
        return render(request, "reading_for_aid.html")

def get_text_dict(paragraphs: QuerySet, article_id: int) -> dict:
    # sourcery skip: raise-specific-error
    """将文章及翻译返回"""
    para_dict = {}
    para = 0
    for paragraph in paragraphs:
        # 切成句子
        sentences = paragraph.content.split(".")
        cnt = 0
        words_dict = {0: paragraph.content}
        for sentence in sentences:
            # 去除句子前后空格
            sentence = sentence.strip()
            if len(sentence) > 3:
                # 切成单词
                words = sentence.split(" ")
                for word in words:
                    cnt = cnt + 1
                    words_dict[cnt] = {"en": word, "zh": "", "sentence_zh": ""}
        para_dict[para] = words_dict
        para = para + 1
    return para_dict

def get_translation_sentence(paragraphs: QuerySet, article_id: int) -> dict:
    # sourcery skip: raise-specific-error
    """将文章及翻译返回"""
    para_dict = {}
    para = 0
    for paragraph in paragraphs:
        # 切成句子
        sentences = paragraph.content.split(".")
        cnt = 0
        words_dict = {0: paragraph.content}
        sentence_id = 0
        for sentence in sentences:
            # 去除句子前后空格
            sentence = sentence.strip()
            if len(sentence) > 3:
                if translation := (
                        Translation.objects.filter(article_id=article_id)
                                .filter(para_id=para)
                                .filter(sentence_id=sentence_id).first()
                ):
                    if translation.txt:
                        sentence_zh = translation.txt
                    else:
                        response = translate(sentence)

                        if response["status"] == 500:
                            raise Exception("百度翻译接口访问失败")

                        sentence_zh = response["zh"]
                        translation.txt = sentence_zh
                        translation.save()
                else:
                    response = translate(sentence)

                    if response["status"] == 500:
                        raise Exception("百度翻译接口访问失败")

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
                    if dictionary := Dictionary.objects.filter(
                            en=word.lower()
                    ).first():
                        # 如果字典查得到，就从数据库中取，减少接口使用
                        if dictionary.zh:
                            zh = dictionary.zh
                        else:
                            response = translate(word)
                            if response["status"] == 500:
                                raise Exception("百度翻译接口访问失败")
                            zh = response["zh"]
                            dictionary.zh = zh
                            dictionary.save()
                    else:
                        # 字典没有，调用接口
                        response = translate(word)
                        if response["status"] == 500:
                            raise Exception("百度翻译接口访问失败")
                        zh = response["zh"]
                        # 存入字典
                        Dictionary.objects.create(en=word.lower(), zh=zh)
                    cnt = cnt + 1
                    words_dict[cnt] = {"en": word, "zh": zh, "sentence_zh": sentence_zh}
                sentence_id = sentence_id + 1
        para_dict[para] = words_dict
        para = para + 1
    return para_dict


def get_simplified_sentence(paragraphs: QuerySet, article_id: int) -> dict:
    """将文章及其简化后的返回"""
    para_dict = {}
    para = 0
    for paragraph in paragraphs:
        # 切成句子
        sentences = paragraph.content.split(".")
        cnt = 0
        words_dict = {0: paragraph.content}
        sentence_id = 0
        for sentence in sentences:
            # 去除句子前后空格
            sentence = sentence.strip()
            if len(sentence) > 3:
                if translation := (
                        Translation.objects.filter(article_id=article_id)
                                .filter(para_id=para)
                                .filter(sentence_id=sentence_id).first()
                ):
                    if translation.simplify:
                        sentence_zh = translation.simplify
                    else:
                        sentence_zh = simplify_sentence(sentence)
                        translation.simplify = sentence_zh
                        translation.save()
                else:
                    sentence_zh = simplify_sentence(sentence)

                    Translation.objects.create(
                        simplify=sentence_zh,
                        article_id=article_id,
                        para_id=para,
                        sentence_id=sentence_id,
                    )
                # 切成单词
                words = sentence.split(" ")
                for word in words:
                    word = word.strip().replace(",", "")
                    if dictionary := Dictionary.objects.filter(
                            en=word.lower()
                    ).first():
                        # 如果字典查得到，就从数据库中取，减少接口使用（要付费呀）
                        if dictionary.synonym:
                            zh = dictionary.synonym
                        else:
                            zh = simplify_word(word)
                            dictionary.synonym = zh
                            dictionary.save()
                    else:
                        # 如果没有该条记录
                        # 存入字典
                        zh = simplify_word(word)
                        Dictionary.objects.create(en=word.lower(), synonym=zh)
                    cnt = cnt + 1
                    words_dict[cnt] = {"en": word, "zh": zh, "sentence_zh": sentence_zh}
                sentence_id = sentence_id + 1
        para_dict[para] = words_dict
        para = para + 1
    return para_dict


def get_para(request):
    """根据文章id获取整篇文章的分段以及翻译"""
    # 获取整篇文章的内容和翻译

    article_id = request.GET.get("article_id", 1)

    request.session['article_id'] = article_id

    paragraphs = Paragraph.objects.filter(article_id=article_id)
    logger.info("--实验开始--")
    name = "读取文章及其翻译"
    with Timer(name):  # 开启计时
        print('role:' + request.session.get('role', 'native'))
        try:
            para_dict = get_text_dict(paragraphs, article_id)
        except Exception:
            logger.warning("百度翻译接口访问失败")

    # 创建一次实验
    # requesert
    experiment = Experiment.objects.create(article_id=article_id, user=1,
                                           device=request.session.get("device"), is_finish=0)
    request.session["experiment_id"] = experiment.id
    logger.info("--本次实验开始,实验者：%s，实验id：%d--" % (request.user.username, experiment.id))
    return JsonResponse(para_dict, json_dumps_params={"ensure_ascii": False})


def collect_page_data(request):
    """存储每页的数据"""
    image_base64 = request.POST.get("image")  # base64类型
    x = request.POST.get("x")  # str类型
    y = request.POST.get("y")  # str类型
    t = request.POST.get("t")  # str类型
    texts = request.POST.get("text")
    page = request.POST.get("page")

    location = request.POST.get("location")

    if experiment_id := request.session.get("experiment_id", None):
        pagedata = PageData.objects.create(
            gaze_x=str(x),
            gaze_y=str(y),
            gaze_t=str(t),
            texts=texts,  # todo 前端发送过来
            image=image_base64,
            page=page,  # todo 前端发送过来
            experiment_id=experiment_id,
            location=location,
            is_test=0,
        )
        logger.info(f"第{page}页数据已存储,id为{str(pagedata.id)}")
    return HttpResponse(1)


def go_label_page(request):
    return render(request, "label_1.html")


def collect_labels(request):
    """一次性获得所有页的label，分页存储"""
    # 示例：labels:[{"page":1,"wordLabels":[],"sentenceLabels":[[27,57]],"wanderLabels":[[0,27]]},{"page":2,"wordLabels":[36],"sentenceLabels":[],"wanderLabels":[]},{"page":3,"wordLabels":[],"sentenceLabels":[],"wanderLabels":[[0,34]]}]
    labels = request.POST.get("labels")
    labels = json.loads(labels)

    if experiment_id := request.session.get("experiment_id", None):
        for label in labels:
            PageData.objects.filter(experiment_id=experiment_id).filter(page=label["page"]).update(
                wordLabels=label["wordLabels"],
                sentenceLabels=label["sentenceLabels"],
                wanderLabels=label["wanderLabels"],

            )
        Experiment.objects.filter(id=experiment_id).update(is_finish=1)
    logger.info("已获得所有页标签,实验结束")
    # 提交后清空缓存
    request.session.flush()
    return HttpResponse(1)


def get_word_feature_by_fixations(fixations: list, word_list: list, location: str) -> WordFeature:
    wordFeature = WordFeature(len(word_list))
    pre_fix = -1

    for fix in fixations:
        index, _ = get_item_index_x_y(location, fix[0], fix[1])
        if index != -1:
            wordFeature.number_of_fixation[index] += 1
            wordFeature.total_fixation_duration[index] += fix[2]
            if index != pre_fix:
                wordFeature.reading_times[pre_fix] += 1
            pre_fix = index
    wordFeature.word_list = word_list
    return wordFeature


def get_sent_feature_by_fixations(fixations: list, sent_list: list, location: str) -> SentFeature:
    pre_fix = -1
    sentFeature = SentFeature(len(sent_list))
    for i, fix in enumerate(fixations):
        index, _ = get_item_index_x_y(location, fix[0], fix[1])
        if index == -1:
            continue
        sent_index = get_sentence_by_word(index, sent_list)
        if sent_index == -1:
            continue
        sentFeature.total_dwell_time[sent_index] += fix[2]
        if i == 0:
            continue
        sentFeature.saccade_times[sent_index] += 1
        sentFeature.saccade_duration[sent_index] += fixations[i][3] - fixations[i - 1][4]
        if index >= pre_fix:
            sentFeature.forward_saccade_times[sent_index] += 1
        else:
            sentFeature.backward_saccade_times[sent_index] += 1
        sentFeature.saccade_velocity[sent_index] += get_euclid_distance(
            (fixations[i][0], fixations[i][1]),
            (fixations[i - 1][0], fixations[i - 1][1]))  # 记录的实际上是距离
        pre_fix = index

    return sentFeature


def get_word_not_understand(wordFeature,rows) -> list:
    data = pd.DataFrame({
        'reading_times': wordFeature.reading_times,
        'number_of_fixations': wordFeature.number_of_fixation,
        'fixation_duration': wordFeature.total_fixation_duration,
        'word': wordFeature.word_list
    })
    # results = wordPredictor.predict(data)
    # results = []
    max_duration = 0
    max_index = -1
    for i, item in enumerate(wordFeature.total_fixation_duration):
        if item > max_duration:
            max_duration = item
            max_index = i

    if max_index == -1:
        return []
    results = (
        [max_index]
        if len([wordFeature.word_list[max_index]]) < 4
        and wordFeature.total_fixation_duration[max_index] > 900
        else []
    )

    words_index = []
    for result in results:
        row = get_row(result,rows)
        if row != -1:
            words_index = [i for i in range(result-2,result+3) if rows[row]['begin_index'] <= i <= rows[row]['end_index']]

    return words_index

def get_sent_not_understand(sentFeature,sent_list):
    data = pd.DataFrame({
        'total_dwell_time_of_sentence': sentFeature.total_dwell_time,
        'saccade_times_of_sentence': sentFeature.saccade_times,
        'saccade_duration': sentFeature.saccade_duration,
        'forward_times_of_sentence': sentFeature.forward_saccade_times,
        'backward_times_of_sentence': sentFeature.backward_saccade_times
    })
    # results = sentPredictor.predict(data)
    max_duration = 0
    max_index = -1
    for i, item in enumerate(sentFeature.total_dwell_time):
        if item > max_duration:
            max_duration = item
            max_index = i
    if max_index == -1:
        return []

    if sentFeature.backward_saccade_times[max_index] > 0.55 * sentFeature.forward_saccade_times[max_index] and sentFeature\
            and sentFeature.total_dwell_time[max_index] > 2800:
        return [[sent_list[max_index][1],sent_list[max_index][2]+1]]
    return []


def get_mind_wadering(sentFeature, sent_list):
    data = pd.DataFrame({
        'total_dwell_time_of_sentence': sentFeature.total_dwell_time,
        'saccade_times_of_sentence': sentFeature.saccade_times,
        'saccade_duration': sentFeature.saccade_duration,
        'forward_times_of_sentence': sentFeature.forward_saccade_times,
        'backward_times_of_sentence': sentFeature.backward_saccade_times
    })
    # results = wanderPredictor.predict(data)
    max_duration = 0
    max_index = -1
    for i, item in enumerate(sentFeature.total_dwell_time):
        if item > max_duration:
            max_duration = item
            max_index = i
    if max_index == -1:
        return []

    if sentFeature.backward_saccade_times[max_index] < (1/4) * sentFeature.forward_saccade_times[max_index] \
        and sentFeature.saccade_velocity[max_index] / sentFeature.saccade_duration[max_index] > 2:
            return [[sent_list[max_index][1],sent_list[max_index][2]]]
    return []


def get_pred(request) -> JsonResponse:
    """根据眼动获取预测结果"""
    x = request.POST.get("x")
    print(f"len(x):{len(x)}")
    y = request.POST.get("y")
    t = request.POST.get("t")

    logger.info("执行了get_pred")
    page_id = request.session['page_id']
    page_data = PageData.objects.get(id=page_id)

    word_list, sent_list = get_word_and_sentence_from_text(page_data.texts)
    gaze_points = format_gaze(x, y, t, begin_time=0, end_time=0)
    fixations = detect_fixations(gaze_points)

    # 通过fixations计算feature
    location = json.loads(page_data.location)
    wordFeature = get_word_feature_by_fixations(fixations, word_list, location)
    sentFeature = get_sent_feature_by_fixations(fixations, sent_list, location)

    border, rows, danger_zone, len_per_word = textarea(page_data.location)
    word_not_understand_list = get_word_not_understand(wordFeature,rows)
    sent_not_understand_list = get_sent_not_understand(sentFeature,sent_list)
    mind_wander_list = get_mind_wadering(sentFeature, sent_list)

    if len(mind_wander_list) > 0:
        sent_not_understand_list = []
        word_not_understand_list = []
    elif len(sent_not_understand_list) > 0:
        word_not_understand_list = []

    PageData.objects.filter(id=page_id).update(
        word_intervention=page_data.word_intervention + "," + str(word_not_understand_list),
        sent_intervention=page_data.sent_intervention + "," + str(sent_not_understand_list),
        mind_wander_intervention=page_data.mind_wander_intervention + "," + str(mind_wander_list)
    )

    print(f"word_list:{word_not_understand_list}")
    context = {
        "word": word_not_understand_list,
        "sentence": sent_not_understand_list,
        "wander": mind_wander_list
    }
    print(f"context:{context}")

    return JsonResponse(context)


def page_info(request):
    page_text = request.POST.get("page_text")
    location = request.POST.get("location")

    is_end = request.POST.get("is_end", 0)
    experiment_id = request.session.get("experiment_id", None)
    page_num = request.session.get('page_num', 1)

    print(f'page_num:{page_num}')

    try:
        page_data = PageData.objects.create(
            texts=page_text,
            page=page_num,
            experiment_id=experiment_id,
            location=location,
            is_pilot_study=True
        )

        request.session['page_id'] = page_data.id
        logger.info(f"第{page_num}页数据已存储,id为{str(page_data.id)}")
        request.session['page_num'] = page_num + 1
    except Exception:
        print("末尾页")

    return HttpResponse(1)


def count_data_num(request):
    exps = []
    with open('exp.txt', 'r') as f:
        exps.extend(int(f.readline()) for _ in f)
    visited = {
        page.experiment_id
        for page in PageData.objects.all()
        if page.experiment_id in exps
    }
    print(len(visited))
    print(visited)
    return HttpResponse(exps)