# import nltk
#
# from nltk.corpus import wordnet as wn
#
# # 加载WordNet词典
# wn.ensure_loaded()
#
# # 获取单词synset
# synsets = wn.synsets('happy')
#
# # 获取同义词
# synonyms = set()
# for synset in synsets:
#     for lemma in synset.lemmas():
#         synonyms.add(lemma.name())
#
# print(synonyms)


import nltk
nltk.download('brown')
from nltk.corpus import brown

# 加载brown语料库，并分词
words = brown.words()
freq_dist = nltk.FreqDist(words)

# 统计单词'example'出现的频率
word = 'a'
freq = freq_dist[word]

print("Frequency of word '{0}' is {1}".format(word, freq))

