import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from loguru import logger
from analysis.models import Text, Paragraph, Translation, Dictionary, Experiment, PageData
from tools import login_required, translate, Timer


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

    if reading_type == '1':
        return render(request, "reading_for_data_1.html")
    else:
        return render(request, "reading_for_aid.html")


def get_para(request):
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
            # 切成句子
            sentences = paragraph.content.split(".")
            cnt = 0
            words_dict = {0: paragraph.content}
            sentence_id = 0
            for sentence in sentences:
                # 去除句子前后空格
                sentence = sentence.strip()
                if len(sentence) > 3:
                    if translations := (
                            Translation.objects.filter(article_id=article_id)
                                    .filter(para_id=para)
                                    .filter(sentence_id=sentence_id)
                    ):
                        sentence_zh = translations.first().txt
                    else:
                        response = translate(sentence)
                        print(response)
                        if response["status"] == 500:
                            return HttpResponse(f"翻译句子:{sentence} 时出现错误")
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
                        if dictionaries := Dictionary.objects.filter(
                                en=word.lower()
                        ):
                            # 如果字典查得到，就从数据库中取，减少接口使用（要付费呀）
                            zh = dictionaries.first().zh
                        else:
                            # 字典没有，调用接口
                            response = translate(word)
                            if response["status"] == 500:
                                return HttpResponse(f"翻译单词：{word} 时出现错误")
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
