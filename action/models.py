import django.utils.timezone as timezone
from django.db import models


# Create your models here.


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
    txt = models.TextField()
    article_id = models.BigIntegerField()
    para_id = models.IntegerField()
    sentence_id = models.IntegerField()

    class Meta:
        db_table = "material_translation"


class Dictionary(models.Model):
    # 记录下单词翻译
    en = models.CharField(max_length=100)
    zh = models.CharField(max_length=800)

    class Meta:
        db_table = "material_dictionary"


class Experiment(models.Model):
    # 一人次是一个实验
    article_id = models.BigIntegerField()
    user = models.CharField(max_length=200)
    is_finish = models.BooleanField()
    device = models.CharField(max_length=96, default="not detect")

    class Meta:
        db_table = "data_experiment"


class PageData(models.Model):
    gaze_x = models.TextField()
    gaze_y = models.TextField()
    gaze_t = models.TextField()
    texts = models.TextField()
    interventions = models.CharField(max_length=1000)
    wordLabels = models.CharField(max_length=1000)
    sentenceLabels = models.CharField(max_length=1000)
    wanderLabels = models.CharField(max_length=1000)
    image = models.TextField()
    experiment_id = models.BigIntegerField()
    page = models.IntegerField()
    created_time = models.DateTimeField(default=timezone.now)
    location = models.TextField()
    is_test = models.BooleanField()
    para = models.CharField(max_length=1000)

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


class UserReadingInfo(models.Model):
    user = models.CharField(max_length=100)

    backward_times_of_sentence_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    backward_times_of_sentence_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    forward_times_of_sentence_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    forward_times_of_sentence_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    horizontal_saccade_proportion_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    horizontal_saccade_proportion_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    saccade_duartion_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    saccade_duartion_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    saccade_times_of_sentence_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    saccade_times_of_sentence_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    saccade_velocity_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    saccade_velocity_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    total_dwell_time_of_sentence_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    total_dwell_time_of_sentence_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    fixation_duration_mean = models.DecimalField(max_digits=9, decimal_places=2)
    fixation_duration_var = models.DecimalField(max_digits=9, decimal_places=2)

    number_of_fixations_mean = models.DecimalField(max_digits=9, decimal_places=2)
    number_of_fixations_var = models.DecimalField(max_digits=9, decimal_places=2)

    reading_times_mean = models.DecimalField(max_digits=9, decimal_places=2)
    reading_times_var = models.DecimalField(max_digits=9, decimal_places=2)

    fixation_duration_diff_mean = models.DecimalField(max_digits=9, decimal_places=2)
    fixation_duration_diff_var = models.DecimalField(max_digits=9, decimal_places=2)

    number_of_fixations_diff_mean = models.DecimalField(max_digits=9, decimal_places=2)
    number_of_fixations_diff_var = models.DecimalField(max_digits=9, decimal_places=2)

    reading_times_diff_mean = models.DecimalField(max_digits=9, decimal_places=2)
    reading_times_diff_var = models.DecimalField(max_digits=9, decimal_places=2)

    fixation_duration_mean_mean = models.DecimalField(max_digits=9, decimal_places=2)
    fixation_duration_mean_var = models.DecimalField(max_digits=9, decimal_places=2)

    fixation_duration_var_mean = models.DecimalField(max_digits=9, decimal_places=2)
    fixation_duration_var_var = models.DecimalField(max_digits=9, decimal_places=2)

    number_of_fixations_mean_mean = models.DecimalField(max_digits=9, decimal_places=2)
    number_of_fixations_mean_var = models.DecimalField(max_digits=9, decimal_places=2)

    number_of_fixations_var_mean = models.DecimalField(max_digits=9, decimal_places=2)
    number_of_fixations_var_var = models.DecimalField(max_digits=9, decimal_places=2)

    reading_times_mean_mean = models.DecimalField(max_digits=9, decimal_places=2)
    reading_times_mean_var = models.DecimalField(max_digits=9, decimal_places=2)

    reading_times_var_mean = models.DecimalField(max_digits=9, decimal_places=2)
    reading_times_var_var = models.DecimalField(max_digits=9, decimal_places=2)

    fixation_duration_div_syllable_mean = models.DecimalField(max_digits=9, decimal_places=2)
    fixation_duration_div_syllable_var = models.DecimalField(max_digits=9, decimal_places=2)

    fixation_duration_div_length_mean = models.DecimalField(max_digits=9, decimal_places=2)
    fixation_duration_div_length_var = models.DecimalField(max_digits=9, decimal_places=2)

    class Meta:
        db_table = "user_reading_info"
