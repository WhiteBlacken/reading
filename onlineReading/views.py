import json

from django.db.models import QuerySet
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger
from analysis.models import Text, Paragraph, Translation, Dictionary, Experiment, PageData
from tools import login_required, translate, Timer, simplify_word, simplify_sentence


def go_login(request):
    """
    跳转去登录页面
    """
    return render(request, "login.html")


def login(request):
    """
    登录
    ：简单的登录逻辑，记下用户名
    """
    username = request.POST.get("username", None)
    device = request.POST.get("device", None)
    print(f"device:{device}")
    print(f"username:{username}")
    if username:
        request.session["username"] = username
        request.session["device"] = device
    return render(request, "calibration.html")


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
    native = request.GET.get('native','1')
    request.session['role'] = 'native' if native == '1' else "nonnative"

    if reading_type == '1':
        return render(request, "reading_for_data_1.html")
    else:
        return render(request, "reading_for_aid.html")


def get_translation_sentence(paragraphs:QuerySet,article_id:int) -> dict:
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


def get_simplified_sentence(paragraphs:QuerySet,article_id:int) -> dict:
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
                        print(sentence_zh)
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
                            dictionary.synonym=zh
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
        print('role:'+request.session.get('role','native'))
        if request.session.get('role','native') == 'native':
            try:
                para_dict = get_simplified_sentence(paragraphs,article_id)
            except Exception:
                logger.warning("简化模型调用失败")
        else:
            try:
                para_dict = get_translation_sentence(paragraphs,article_id)
            except Exception:
                logger.warning("百度翻译接口访问失败")

    # 创建一次实验
    experiment = Experiment.objects.create(article_id=article_id, user=request.session.get("username"),
                                           device=request.session.get("device"))
    request.session["experiment_id"] = experiment.id
    logger.info("--本次实验开始,实验者：%s，实验id：%d--" % (request.session.get("username"), experiment.id))
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
