import math
import os

import pandas as pd
import numpy as np
from textstat import textstat

from tools import div_list, round_list

class RowArticle:
    def __init__(self, num):
        self.num = num
        self.article_id = 0
        self.row_idx = []
        self.row_text = []
        self.row_label = []
        self.experiment_id = 0


    def to_csv(self, filename):
        df = pd.DataFrame({
            "experiment_id": [self.experiment_id for _ in range(self.num)],
            "article_id": [self.article_id for _ in range(self.num)],
            "row_idx": self.row_idx,
            "row_text": self.row_text,
            "row_label": self.row_label
        })
        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode="a", header=False)
        else:
            df.to_csv(filename, index=False, mode="a")


    

class WordFeature(object):
    def __init__(self, num):
        self.num = num
        # 特征1
        self.total_fixation_duration = [0 for _ in range(num)]
        self.number_of_fixation = [0 for _ in range(num)]
        self.reading_times = [0 for _ in range(num)]
        # 特征2
        self.fixation_duration_div_syllable = [0 for _ in range(num)]
        self.fixation_duration_div_length = [0 for _ in range(num)]
        # 特征3
        self.fixation_duration_diff = [0 for _ in range(num)]
        self.number_of_fixations_diff = [0 for _ in range(num)]
        self.reading_times_diff = [0 for _ in range(num)]

        #
        self.backward_times_of_sentence_one_word = 0 # 反正每行都一样
        self.forward_times_of_sentence_one_word = 0
        self.horizontal_saccade_proportion_one_word = 0
        self.saccade_duration_one_word = 0
        self.saccade_times_of_sentence_one_word = 0
        self.saccade_velocity_one_word = 0
        self.total_dwell_time_of_sentence_one_word = 0


        # 实验相关信息
        self.word_list = ["" for _ in range(num)]  # 单词列表
        self.sentence_id = [0 for _ in range(num)]  # 不同时刻的同一个句子，为不同的句子
        self.need_prediction = [0 for _ in range(num)]  # 是否需要预测
        # label
        self.word_understand = [0 for _ in range(num)]
        self.sentence_understand = [0 for _ in range(num)]
        self.mind_wandering = [0 for _ in range(num)]
        
        # 标识
        self.fix_seq_id_list = [0 for _ in range(num)]
        self.row_idx_list = [0 for _ in range(num)]



    def clean(self):
        # 特征1
        self.total_fixation_duration = [0 for _ in range(self.num)]
        self.number_of_fixation = [0 for _ in range(self.num)]
        self.reading_times = [0 for _ in range(self.num)]
        # 特征2
        self.fixation_duration_div_syllable = [0 for _ in range(self.num)]
        self.fixation_duration_div_length = [0 for _ in range(self.num)]
        # 特征3
        self.fixation_duration_diff = [0 for _ in range(self.num)]
        self.number_of_fixations_diff = [0 for _ in range(self.num)]
        self.reading_times_diff = [0 for _ in range(self.num)]

    def get_syllable(self):
        # return -1
        return [textstat.syllable_count(word) for word in self.word_list]

    def get_len(self):
        return [len(word) for word in self.word_list]

    def diff(self,list_a):
        results = [0 for _ in range(len(list_a))]
        for i,item in enumerate(list_a):
            if i == 0:
                results[i] = 0
                continue
            results[i] = round(list_a[i] - list_a[i-1], 3)
        return results

    def to_csv(self, filename, exp_id, page_id, time, user, article_id):
        self.sum_word_len = sum([len(x) for x in self.word_list])
        self.sum_word_syllable = sum(self.get_syllable())

        df = pd.DataFrame(
            {
                # 1. 实验信息相关
                "exp_id": [exp_id for _ in range(self.num)],
                "page_id": [page_id for _ in range(self.num)],
                "fix_seq_id": self.fix_seq_id_list,
                "row_idx": self.row_idx_list,
                "article_id": [article_id for _ in range(self.num)],
                # "time": [time for _ in range(self.num)],
                "user": [user for _ in range(self.num)],
                "sentence_id": self.sentence_id,
                "word": self.word_list,
                # "need_prediction": self.need_prediction,

                "need_prediction": [1 if x > 0 else 0 for x in self.number_of_fixation],
                # # 2. label相关
                "word_understand": self.word_understand,
                "sentence_understand": self.sentence_understand,
                "mind_wandering": self.mind_wandering,
                # 单词level特征1
                "reading_times": self.reading_times,
                "number_of_fixations": self.number_of_fixation,
                "fixation_duration": self.total_fixation_duration,

                # 单词level特征2
                "fixation_duration_diff": self.diff(self.total_fixation_duration),
                "number_of_fixations_diff": self.diff(self.number_of_fixation),
                "reading_times_diff": self.diff(self.reading_times),

                # 单词level特征3
                "fixation_duration_mean": [round(sum(self.total_fixation_duration)/self.num, 3) for _ in range(self.num)],
                "number_of_fixations_mean": [round(sum(self.number_of_fixation)/self.num, 3) for _ in range(self.num)],
                "reading_times_mean": [round(sum(self.reading_times)/self.num, 3) for _ in range(self.num)],

                # 单词level特征4
                "fixation_duration_var": [round(np.var(self.total_fixation_duration), 3) for _ in range(self.num)],
                "number_of_fixations_var": [round(np.var(self.number_of_fixation), 3) for _ in range(self.num)],
                "reading_times_var": [round(np.var(self.reading_times), 3) for _ in range(self.num)],

                'fixation_duration_div_syllable': [round(x, 3) for x in div_list(self.total_fixation_duration, self.get_syllable())],
                'fixation_duration_div_length': [round(x, 3) for x in div_list(self.total_fixation_duration, [len(w) for w in self.word_list])],
                
                "backward_times_of_sentence": [self.backward_times_of_sentence_one_word for _ in range(self.num)],
                "forward_times_of_sentence": [self.forward_times_of_sentence_one_word for _ in range(self.num)],
                "horizontal_saccade_proportion": [self.horizontal_saccade_proportion_one_word for _ in range(self.num)],
                "saccade_duration": [self.saccade_duration_one_word for _ in range(self.num)],
                "saccade_times_of_sentence": [self.saccade_times_of_sentence_one_word for _ in range(self.num)],
                "saccade_velocity": [round(self.saccade_velocity_one_word, 3) for _ in range(self.num)],
                "total_dwell_time_of_sentence": [self.total_dwell_time_of_sentence_one_word for _ in range(self.num)],

                "backward_times_of_sentence_div_log": [round(self.backward_times_of_sentence_one_word/math.log(self.sum_word_len),3) for _ in range(self.num)],
                "forward_times_of_sentence_div_log": [round(self.forward_times_of_sentence_one_word/math.log(self.sum_word_len),3) for _ in range(self.num)],
                "horizontal_saccade_proportion_div_log": [round(self.horizontal_saccade_proportion_one_word/math.log(self.sum_word_len),3) for _ in range(self.num)],
                "saccade_duration_div_log": [round(self.saccade_duration_one_word/math.log(self.sum_word_len),3) for _ in range(self.num)],
                "saccade_times_of_sentence_div_log": [round(self.saccade_times_of_sentence_one_word/math.log(self.sum_word_len),3) for _ in range(self.num)],
                "saccade_velocity_div_log": [round(self.saccade_velocity_one_word/math.log(self.sum_word_len), 3) for _ in range(self.num)],
                "total_dwell_time_of_sentence_div_log": [round(self.total_dwell_time_of_sentence_one_word/math.log(self.sum_word_len),3) for _ in range(self.num)],
                
                "backward_times_of_sentence_div_syllable": [round(self.backward_times_of_sentence_one_word/self.sum_word_syllable,3) for _ in range(self.num)],
                "forward_times_of_sentence_div_syllable": [round(self.forward_times_of_sentence_one_word/self.sum_word_syllable,3) for _ in range(self.num)],
                "horizontal_saccade_proportion_div_syllable": [round(self.horizontal_saccade_proportion_one_word/self.sum_word_syllable,3) for _ in range(self.num)],
                "saccade_duration_div_syllable": [round(self.saccade_duration_one_word/self.sum_word_syllable,3) for _ in range(self.num)],
                "saccade_times_of_sentence_div_syllable": [round(self.saccade_times_of_sentence_one_word/self.sum_word_syllable,3) for _ in range(self.num)],
                "saccade_velocity_div_syllable": [round(self.saccade_velocity_one_word/self.sum_word_syllable, 3) for _ in range(self.num)],
                "total_dwell_time_of_sentence_div_syllable": [round(self.total_dwell_time_of_sentence_one_word/self.sum_word_syllable,3) for _ in range(self.num)],
                

            }
        )

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode="a", header=False)
        else:
            df.to_csv(filename, index=False, mode="a")


class SentFeature(object):
    def __init__(self, num):
        self.num = num
        # 特征1
        self.total_dwell_time = [0 for _ in range(num)]
        self.saccade_times = [0 for _ in range(num)]
        self.forward_saccade_times = [0 for _ in range(num)]
        self.backward_saccade_times = [0 for _ in range(num)]
        self.horizontal_saccade_proportion = [0 for _ in range(num)]
        self.saccade_velocity = [0 for _ in range(num)]
        self.saccade_duration = [0 for _ in range(num)]

        # first pass特征
        self.first_pass_total_dwell_time = [0 for _ in range(num)]
        self.first_pass_saccade_times = [0 for _ in range(num)]
        self.first_pass_forward_saccade_times = [0 for _ in range(num)]
        self.first_pass_backward_saccade_times = [0 for _ in range(num)]
        self.first_pass_horizontal_saccade_proportion = [0 for _ in range(num)]
        self.first_pass_saccade_velocity = [0 for _ in range(num)]
        self.first_pass_saccade_duration = [0 for _ in range(num)]

        # 相似度特征
        self.reading_times_cor = [0 for _ in range(num)]
        self.number_of_fixation_cor = [0 for _ in range(num)]
        self.total_fixation_duration_cor = [0 for _ in range(num)]

        self.first_pass_reading_times_cor = [0 for _ in range(num)]
        self.first_pass_number_of_fixation_cor = [0 for _ in range(num)]
        self.first_pass_total_fixation_duration_cor = [0 for _ in range(num)]

        # 特征4
        self.length = [0 for _ in range(self.num)]
        # 实验相关信息
        self.sentence_understand = []
        self.mind_wandering = []
        self.sentence_id = []
        self.sentence = [] # 记录句子
        self.need_prediction = [0 for _ in range(num)] # 这个句子当前在看

        # page_id,exp_id都是需要的

    def clean(self):
        # 特征1
        self.total_dwell_time = [0 for _ in range(self.num)]
        self.saccade_times = [0 for _ in range(self.num)]
        self.forward_saccade_times = [0 for _ in range(self.num)]
        self.backward_saccade_times = [0 for _ in range(self.num)]
        self.horizontal_saccade_proportion = [0 for _ in range(self.num)]
        self.saccade_velocity = [0 for _ in range(self.num)]
        self.saccade_duration = [0 for _ in range(self.num)]

        # first pass特征
        self.first_pass_total_dwell_time = [0 for _ in range(self.num)]
        self.first_pass_saccade_times = [0 for _ in range(self.num)]
        self.first_pass_forward_saccade_times = [0 for _ in range(self.num)]
        self.first_pass_backward_saccade_times = [0 for _ in range(self.num)]
        self.first_pass_horizontal_saccade_proportion = [0 for _ in range(self.num)]
        self.first_pass_saccade_velocity = [0 for _ in range(self.num)]
        self.first_pass_saccade_duration = [0 for _ in range(self.num)]

        # 相似度特征
        self.reading_times_cor = [0 for _ in range(self.num)]
        self.number_of_fixation_cor = [0 for _ in range(self.num)]
        self.total_fixation_duration_cor = [0 for _ in range(self.num)]

        self.first_pass_reading_times_cor = [0 for _ in range(self.num)]
        self.first_pass_number_of_fixation_cor = [0 for _ in range(self.num)]
        self.first_pass_total_fixation_duration_cor = [0 for _ in range(self.num)]

        # 特征4
        self.length = [0 for _ in range(self.num)]

    def get_syllable(self):
        syllable_len = [0 for _ in range(self.num)]
        for i,sent in enumerate(self.sentence):
            words = sent.split()
            for word in words:
                syllable_len[i] += textstat.syllable_count(word)
        return syllable_len

    def get_log(self):
        log = [0 for _ in range(self.num)]
        for i, sent in enumerate(self.sentence):
            words = sent.split()
            for word in words:
                log[i] += math.log(len(word))
        return log

    def get_len(self):
        length = [0 for _ in range(self.num)]
        for i, sent in enumerate(self.sentence):
            words = sent.split()
            for word in words:
                length[i] += len(word)
        return length

    def to_csv(self, filename, exp_id, page_id, time, user,article_id):

        # 获取每句的字节数
        syllable_list = self.get_syllable()
        log_list = self.get_log()
        self.length = self.get_len()

        self.saccade_velocity = div_list(self.saccade_velocity,self.saccade_duration)
        self.horizontal_saccade_proportion = div_list(self.horizontal_saccade_proportion,self.saccade_times)

        df = pd.DataFrame(
            {
                # 1. 实验信息相关
                "exp_id": [exp_id for _ in range(self.num)],
                "article_id": [article_id for _ in range(self.num)],
                "user": [user for _ in range(self.num)],
                "time": [time for _ in range(self.num)],
                "page_id": [page_id for _ in range(self.num)],
                "sentence_id": self.sentence_id,
                "word": self.sentence,
                "need_prediction": self.need_prediction,

                # # 2. label相关
                "sentence_understand": self.sentence_understand,
                "mind_wandering": self.mind_wandering,
                # 3. 特征相关
                # 3.1 特征1
                "total_dwell_time_of_sentence": self.total_dwell_time,
                "saccade_times_of_sentence": self.saccade_times,
                "forward_times_of_sentence": self.forward_saccade_times,
                "backward_times_of_sentence": self.backward_saccade_times,
                "horizontal_saccade_proportion": self.horizontal_saccade_proportion, # todo
                "saccade_velocity": self.saccade_velocity, # todo
                "saccade_duration": self.saccade_duration, # todo
                # # 3.2 特征2
                # "total_dwell_time_of_sentence_div_syllable": round_list(div_list(self.total_dwell_time,syllable_list),3),
                # "saccade_times_of_sentence_div_syllable": round_list(div_list(self.saccade_times,syllable_list),3),
                # "forward_times_of_sentence_div_syllable": round_list(div_list(self.forward_saccade_times,syllable_list),3),
                # "backward_times_of_sentence_div_syllable": round_list(div_list(self.backward_saccade_times,syllable_list),3),
                # "horizontal_saccade_proportion_div_syllable": round_list(div_list(self.horizontal_saccade_proportion,syllable_list),3),
                # "saccade_velocity_div_syllable": round_list(div_list(self.saccade_velocity,syllable_list),3),
                # "saccade_duration_div_syllable": round_list(div_list(self.saccade_duration,syllable_list),3),
                # # 3.2 特征3
                # "total_dwell_time_of_sentence_div_log": round_list(div_list(self.total_dwell_time, log_list),3),
                # "saccade_times_of_sentence_div_log": round_list(div_list(self.saccade_times, log_list), 3),
                # "forward_times_of_sentence_div_log": round_list(
                #     div_list(self.forward_saccade_times, log_list), 3),
                # "backward_times_of_sentence_div_log": round_list(
                #     div_list(self.backward_saccade_times, log_list), 3),
                # "horizontal_saccade_proportion_div_log": round_list(
                #     div_list(self.horizontal_saccade_proportion, log_list), 3),
                # "saccade_velocity_div_log": round_list(div_list(self.saccade_velocity, log_list), 3),
                # "saccade_duration_div_log": round_list(div_list(self.saccade_duration, log_list), 3),

                # 特征2
                "total_dwell_time_of_sentence_div_syllable": div_list(self.total_dwell_time, syllable_list),
                "saccade_times_of_sentence_div_syllable": div_list(self.saccade_times, syllable_list),
                "forward_times_of_sentence_div_syllable": div_list(self.forward_saccade_times, syllable_list),
                "backward_times_of_sentence_div_syllable": div_list(self.backward_saccade_times, syllable_list),
                "horizontal_saccade_proportion_div_syllable": div_list(self.horizontal_saccade_proportion, syllable_list),
                "saccade_velocity_div_syllable": div_list(self.saccade_velocity, syllable_list),
                "saccade_duration_div_syllable": div_list(self.saccade_duration, syllable_list),
                # 特征3
                "total_dwell_time_of_sentence_div_log": div_list(self.total_dwell_time, log_list),
                "saccade_times_of_sentence_div_log": div_list(self.saccade_times, log_list),
                "forward_times_of_sentence_div_log":
                    div_list(self.forward_saccade_times, log_list),
                "backward_times_of_sentence_div_log":
                    div_list(self.backward_saccade_times, log_list),
                "horizontal_saccade_proportion_div_log":
                    div_list(self.horizontal_saccade_proportion, log_list),
                "saccade_velocity_div_log": div_list(self.saccade_velocity, log_list),
                "saccade_duration_div_log": div_list(self.saccade_duration, log_list),

                # # first pass
                # "first_pass_total_dwell_time_of_sentence": self.first_pass_total_dwell_time,
                # "first_pass_saccade_times_of_sentence": self.first_pass_saccade_times,
                # "first_pass_forward_times_of_sentence": self.first_pass_forward_saccade_times,
                # "first_pass_backward_times_of_sentence": self.first_pass_backward_saccade_times,
                # "first_pass_horizontal_saccade_proportion": self.first_pass_horizontal_saccade_proportion,  # todo
                # "first_pass_saccade_velocity": self.first_pass_saccade_velocity,  # todo
                # "first_pass_saccade_duration": self.first_pass_saccade_duration,  # todo
                #
                # "first_pass_total_dwell_time_of_sentence_div_syllable": div_list(self.first_pass_total_dwell_time, syllable_list),
                # "first_pass_saccade_times_of_sentence_div_syllable": div_list(self.first_pass_saccade_times, syllable_list),
                # "first_pass_forward_times_of_sentence_div_syllable": div_list(self.first_pass_forward_saccade_times, syllable_list),
                # "first_pass_backward_times_of_sentence_div_syllable": div_list(self.first_pass_backward_saccade_times, syllable_list),
                # "first_pass_horizontal_saccade_proportion_div_syllable": div_list(self.first_pass_horizontal_saccade_proportion,
                #                                                        syllable_list),
                # "first_pass_saccade_velocity_div_syllable": div_list(self.first_pass_saccade_velocity, syllable_list),
                # "first_pass_saccade_duration_div_syllable": div_list(self.first_pass_saccade_duration, syllable_list),
                #
                # "first_pass_total_dwell_time_of_sentence_div_log": div_list(self.first_pass_total_dwell_time, log_list),
                # "first_pass_saccade_times_of_sentence_div_log": div_list(self.first_pass_saccade_times, log_list),
                # "first_pass_forward_times_of_sentence_div_log":
                #     div_list(self.first_pass_forward_saccade_times, log_list),
                # "first_pass_backward_times_of_sentence_div_log":
                #     div_list(self.first_pass_backward_saccade_times, log_list),
                # "first_pass_horizontal_saccade_proportion_div_log":
                #     div_list(self.first_pass_horizontal_saccade_proportion, log_list),
                # "first_pass_saccade_velocity_div_log": div_list(self.first_pass_saccade_velocity, log_list),
                # "first_pass_saccade_duration_div_log": div_list(self.first_pass_saccade_duration, log_list),
                #
                # # 相似度特征
                # "reading_times_cor": self.reading_times_cor,
                # "number_of_fixation_cor": self.number_of_fixation_cor,
                # "total_fixation_duration_cor": self.total_fixation_duration_cor,
                #
                # "first_pass_reading_times_cor": self.first_pass_reading_times_cor,
                # "first_pass_number_of_fixation_cor": self.first_pass_number_of_fixation_cor,
                # "first_pass_total_fixation_duration_cor": self.first_pass_total_fixation_duration_cor,
                # 特征4
                "length": self.length
            }
        )

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode="a", header=False)
        else:
            df.to_csv(filename, index=False, mode="a")

class CNNFeature(object):
    def __init__(self):
        self.experiment_ids = []
        self.times = []
        self.pages = []
        self.gaze_x = []
        self.gaze_y = []
        self.gaze_t = []
        self.speed = []
        self.direction = []
        self.acc = []

        self.fix_x = []
        self.fix_y = []


    def to_csv(self, filename):

        df = pd.DataFrame(
            {
                "experiment_id": self.experiment_ids,
                "time": self.times,
                "gaze_x": self.gaze_x,
                "gaze_y": self.gaze_y,
                "gaze_t": self.gaze_t,
                "speed": self.speed,
                "direction": self.direction,
                "acc": self.acc,
            }
        )

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode="a", header=False)
        else:
            df.to_csv(filename, index=False, mode="a")

class FixationMap(object):
    def __init__(self):
        self.times = []
        self.exp_ids = []
        self.page_ids = []
        self.fixations = []

    def update(self,time,exp_id,page_id,fixation):
        self.times.append(time)
        self.exp_ids.append(exp_id)
        self.page_ids.append(page_id)
        self.fixations.append(fixation)

    def to_csv(self,filename):
        df = pd.DataFrame(
            {
                "exp_id": self.exp_ids,
                "page_id": self.page_ids,
                "time": self.times,
                "fixation": self.fixations,
            }
        )

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode="a", header=False)
        else:
            df.to_csv(filename, index=False, mode="a")
