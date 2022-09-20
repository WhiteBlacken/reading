from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from action.models import Text, Dictionary
from action.views import translate


def runoob(request):
    """首页测试"""
    context = {"hello": "Hello World!"}
    return render(request, "hello.html", context)


def get_text(request):
    words_dict = {}
    text = Text.objects.first()
    # 去除前后的空格
    text = text.content.strip()
    # 切成句子
    sentences = text.split(".")
    cnt = 0
    for sentence in sentences:
        # 去除句子前后空格
        sentence = sentence.strip()
        if len(sentence) > 3:
            # 句子长度低于 3，不是空，就是切割问题，暂时不考虑
            response = translate(sentence)
            if response["status"] == 500:
                return HttpResponse("翻译句子:%s 时出现错误" % sentence)
            sentence_zh = response["zh"]
            # 切成单词
            words = sentence.split(" ")
            for word in words:
                word = word.strip().replace(",", "")
                # 全部使用小写匹配
                dictionaries = Dictionary.objects.filter(en=word.lower())
                if dictionaries:
                    # 如果字典查得到，就从数据库中取，减少接口使用（要付费呀）
                    zh = dictionaries.first().zh
                else:
                    # 字典没有，调用接口
                    response = translate(word)
                    if response["status"] == 500:
                        return HttpResponse("翻译单词：%s 时出现错误" % word)
                    zh = response["zh"]
                    # 存入字典
                    Dictionary.objects.create(en=word.lower(), zh=zh)
                words_dict[cnt] = {"en": word, "zh": zh, "sentence_zh": sentence_zh}
                cnt = cnt + 1
    return JsonResponse(words_dict, json_dumps_params={"ensure_ascii": False})
