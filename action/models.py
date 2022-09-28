from django.db import models
import django.utils.timezone as timezone

# Create your models here.


class Text(models.Model):
    title = models.CharField(max_length=200)
    is_show = models.BooleanField()

    class Meta:
        db_table = "material_text"

    def toJson(self):
        import json
        return json.dumps(dict([(attr, getattr(self, attr)) for attr in [f.name for f in self._meta.fields]]))


class Paragraph(models.Model):
    article_id = models.BigIntegerField()
    para_id = models.IntegerField()
    content = models.TextField()

    class Meta:
        db_table = "material_paragraph"


class Dictionary(models.Model):
    en = models.CharField(max_length=100)
    zh = models.CharField(max_length=800)

    class Meta:
        db_table = "dictionary"


class Experiment(models.Model):
    # 一人次是一个实验
    article_id = models.BigIntegerField()
    user = models.CharField(max_length=200)

    class Meta:
        db_table = "data_experiment"


class PageData(models.Model):
    gaze_x = models.TextField()
    gaze_y = models.TextField()
    gaze_t = models.TextField()
    texts = models.TextField()
    interventions = models.CharField(max_length=1000)
    labels = models.CharField(max_length=1000)
    image = models.TextField()
    experiment_id = models.BigIntegerField()
    page = models.IntegerField()
    created_time = models.DateTimeField(default=timezone.now)
    class Meta:
        db_table = "data_page"


class WordLevelData(models.Model):
    # 单词对应的记录
    data_id = models.BigIntegerField()
    # 单词在句中的位置
    word_index_in_text = models.IntegerField()
    # 单词本身的text
    word = models.CharField(max_length=100)
    # 单词对应的eye gaze点的集合
    gaze = models.TextField()
    # 单词是否给过干预
    is_intervention = models.BooleanField()
    # 单词是否不懂
    is_understand = models.BooleanField()

    class Meta:
        db_table = "word_level_data"


class Dispersion(models.Model):
    gaze_1_x = models.TextField()
    gaze_1_y = models.TextField()
    gaze_1_t = models.TextField()
    gaze_2_x = models.TextField()
    gaze_2_y = models.TextField()
    gaze_2_t = models.TextField()
    gaze_3_x = models.TextField()
    gaze_3_y = models.TextField()
    gaze_3_t = models.TextField()

    user = models.CharField(max_length=100)

    class Meta:
        db_table = "dispersion"
