import django.utils.timezone as timezone
from django.db import models


class Text(models.Model):
    title = models.CharField(max_length=200)
    is_show = models.BooleanField()

    class Meta:
        db_table = "material_text"

    def toJson(self):
        import json
        return json.dumps(dict([(attr, getattr(self, attr)) for attr in [f.name for f in self._meta.fields]]))


class PilotStudy(models.Model):
    exp_id = models.IntegerField()
    user = models.CharField(max_length=100)
    article_id = models.IntegerField()

    word_intervention = models.CharField(max_length=1000)
    sent_intervention = models.CharField(max_length=1000)
    mind_wander_intervention = models.CharField(max_length=1000)

    class Meta:
        db_table = "pilot_study"


class Paragraph(models.Model):
    article_id = models.BigIntegerField()
    para_id = models.IntegerField()
    content = models.TextField()

    class Meta:
        db_table = "material_paragraph"


class Translation(models.Model):
    # 记录下句子翻译
    txt = models.CharField(max_length=2000)
    simplify = models.CharField(max_length=2000)
    article_id = models.BigIntegerField()
    para_id = models.IntegerField()
    sentence_id = models.IntegerField()

    class Meta:
        db_table = "material_translation"


class Dictionary(models.Model):
    # 记录下单词翻译
    en = models.CharField(max_length=100)
    zh = models.CharField(max_length=800)
    synonym = models.CharField(max_length=100)
    class Meta:
        db_table = "material_dictionary"


class Experiment(models.Model):
    # 一人次是一个实验
    article_id = models.BigIntegerField()
    user = models.CharField(max_length=200)
    is_finish = models.BooleanField()
    device = models.CharField(max_length=96, default="not detect")
    create_time = models.DateTimeField(default=timezone.now)  # 创建的时间

    class Meta:
        db_table = "data_experiment"


class PageData(models.Model):
    gaze_x = models.TextField()
    gaze_y = models.TextField()
    gaze_t = models.TextField()
    texts = models.CharField(max_length=4000)

    wordLabels = models.CharField(max_length=1000)
    sentenceLabels = models.CharField(max_length=1000)
    wanderLabels = models.CharField(max_length=1000)
    image = models.TextField()
    experiment_id = models.BigIntegerField()
    page = models.IntegerField()
    created_time = models.DateTimeField(default=timezone.now)
    location = models.TextField()
    is_test = models.BooleanField(default=False)
    para = models.CharField(max_length=1000)

    word_intervention = models.CharField(max_length=1000)
    sent_intervention = models.CharField(max_length=1000)
    mind_wander_intervention = models.CharField(max_length=1000)

    is_pilot_study = models.BooleanField(default=False)

    class Meta:
        db_table = "data_page"
