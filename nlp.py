# 绘制折线图


# nlp feature
from semantic_attention import generate_word_attention, get_word_difficulty
from utils import get_importance, get_word_and_sentence_from_text, normalize

texts = ""

word_list, sentence_list = get_word_and_sentence_from_text(texts)
difficulty_level = [get_word_difficulty(x) for x in word_list]  # text feature
difficulty_level = normalize(difficulty_level)
word_attention = generate_word_attention(texts)
importance = get_importance(texts)

importance_level = [0 for _ in word_list]
attention_level = [0 for _ in word_list]
for q, word in enumerate(word_list):
    for impo in importance:
        if impo[0] == word:
            importance_level[q] = impo[1]
    for att in word_attention:
        if att[0] == word:
            attention_level[q] = att[1]
importance_level = normalize(importance_level)
attention_level = normalize(attention_level)

nlp_feature = [difficulty_level[i] + importance_level[i] + attention_level[i] for i in range(len(word_list))]

import matplotlib.pyplot as plt
import pandas as pd

dat = pd.read_csv("dataset/step-1-max-min-norm-np-filter.csv")
dat = dat.loc[(dat["experiment_id"] == 577)]

columns = ["index", "value"]

data = []
data_not_understand = []
line = []
i = 0
for index, row in dat.iterrows():
    value = 0.4 * row["number_of_fixations"] + 0.4 * row["fixation_duration"] + 0.2 * row["reading_times"]
    if row["word_understand"] == 0:
        data_not_understand.append([i, value])
    data.append([i, value])
    line.append([i, 0.5])
    i += 1
df = pd.DataFrame(data=data, columns=columns)
line = pd.DataFrame(data=line, columns=columns)
df_1 = pd.DataFrame(data=data_not_understand, columns=columns)

fig = plt.figure(figsize=(50, 4), dpi=100)

plt.plot(df["index"], df["value"], lw=3, ls="-", color="black", zorder=0, alpha=0.3)
plt.plot(line["index"], line["value"], lw=3, ls="-", color="orange", zorder=0, alpha=0.3)
plt.scatter(df_1["index"], df_1["value"], color="red", zorder=1, s=60)

plt.show()
