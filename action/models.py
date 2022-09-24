from django.db import models


# Create your models here.


class Text(models.Model):
    content = models.CharField(max_length=1600)

    class Meta:
        db_table = "text"


class Dictionary(models.Model):
    en = models.CharField(max_length=100)
    zh = models.CharField(max_length=800)

    class Meta:
        db_table = "dictionary"


class Dataset(models.Model):
    gaze_x = models.TextField()
    gaze_y = models.TextField()
    gaze_t = models.TextField()
    texts = models.TextField()
    interventions = models.CharField(max_length=1000)
    labels = models.CharField(max_length=1000)
    image = models.TextField()
    user = models.CharField(max_length=200)

    class Meta:
        db_table = "dataset"


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
