import os

import pandas as pd


class WordFeature(object):
    def __init__(self, num):
        self.num = num
        # 特征
        self.total_fixation_duration = [0 for _ in range(num)]
        self.number_of_fixation = [0 for _ in range(num)]
        self.reading_times = [0 for _ in range(num)]
        # 实验相关信息
        self.word_list = []  # 单词列表
        self.sentence_id = [0 for _ in range(num)]  # 不同时刻的同一个句子，为不同的句子
        self.need_prediction = [0 for _ in range(num)]  # 是否需要预测
        # label
        self.word_understand = [0 for _ in range(num)]
        self.sentence_understand = [0 for _ in range(num)]
        self.mind_wandering = [0 for _ in range(num)]

    def clean(self):
        self.total_fixation_duration = [0 for _ in range(self.num)]
        self.number_of_fixation = [0 for _ in range(self.num)]
        self.reading_times = [0 for _ in range(self.num)]

    def to_csv(self, filename, exp_id, page_id, time):
        print(f"num:{self.num}")
        print(f"word_list:{len(self.word_list)}")
        print(f"sentence_id:{len(self.sentence_id)}")
        print(f"need_prediction:{len(self.need_prediction)}")
        print(f"label:{len(self.word_understand)}")
        print(f"feature:{len(self.reading_times)}")
        df = pd.DataFrame(
            {
                # 1. 实验信息相关
                "exp_id": [exp_id for _ in range(self.num)],
                "time": [time for _ in range(self.num)],
                "page_id": [page_id for _ in range(self.num)],
                "sentence_id": self.sentence_id,
                "word": self.word_list,
                "need_prediction": self.need_prediction,

                # # 2. label相关
                "word_understand": self.word_understand,
                "sentence_understand": self.sentence_understand,
                "mind_wandering": self.mind_wandering,
                # 3. 特征相关
                # word level
                "reading_times": self.reading_times,
                "number_of_fixations": self.number_of_fixation,
                "fixation_duration": self.total_fixation_duration,
            }
        )

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode="a", header=False)
        else:
            df.to_csv(filename, index=False, mode="a")


class SentFeature(object):
    def __init__(self, num):
        self.num = num
        # 特征
        self.total_dwell_time = [0 for _ in range(num)]
        self.saccade_times = [0 for _ in range(num)]
        self.forward_saccade_times = [0 for _ in range(num)]
        self.backward_saccade_times = [0 for _ in range(num)]
        # 实验相关信息
        self.sentence_understand = []
        self.mind_wandering = []
        self.sentence_id = []
        self.sentence = [] # 记录句子

        # page_id,exp_id都是需要的

    def clean(self):
        self.total_dwell_time = [0 for _ in range(self.num)]
        self.saccade_times = [0 for _ in range(self.num)]
        self.forward_saccade_times = [0 for _ in range(self.num)]
        self.backward_saccade_times = [0 for _ in range(self.num)]

    def to_csv(self, filename, exp_id, page_id, time):
        # print(f"num:{self.num}")
        # print(f"word_list:{len(self.word_list)}")
        # print(f"sentence_id:{len(self.sentence_id)}")
        # print(f"need_prediction:{len(self.need_prediction)}")
        # print(f"label:{len(self.word_understand)}")
        # print(f"feature:{len(self.reading_times)}")
        # df = pd.DataFrame(
        #     {
        #         # 1. 实验信息相关
        #         "exp_id": [exp_id for _ in range(self.num)],
        #         "time": [time for _ in range(self.num)],
        #         "page_id": [page_id for _ in range(self.num)],
        #         "sentence_id": self.sentence_id,
        #         "word": self.word_list,
        #         "need_prediction": self.need_prediction,
        #
        #         # # 2. label相关
        #         "word_understand": self.word_understand,
        #         "sentence_understand": self.sentence_understand,
        #         "mind_wandering": self.mind_wandering,
        #         # 3. 特征相关
        #         # word level
        #         "reading_times": self.reading_times,
        #         "number_of_fixations": self.number_of_fixation,
        #         "fixation_duration": self.total_fixation_duration,
        #     }
        # )
        #
        # if os.path.exists(filename):
        #     df.to_csv(filename, index=False, mode="a", header=False)
        # else:
        #     df.to_csv(filename, index=False, mode="a")
        pass