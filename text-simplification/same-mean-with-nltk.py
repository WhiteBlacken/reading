import nltk

from nltk.corpus import wordnet as wn

# 加载WordNet词典
wn.ensure_loaded()

word = 'happy'
# 获取单词synset
synsets = wn.synsets(word)

# 获取同义词
synonyms = set()
for synset in synsets:
    for lemma in synset.lemmas():
        synonyms.add(lemma.name())

print(synonyms)
print(type(synonyms))



nltk.download('brown')
from nltk.corpus import brown

# 加载brown语料库，并分词
words = brown.words()
freq_dist = nltk.FreqDist(words)

max_freq = 0
candidate = ""

for item in synonyms:
    if item == word:
        continue
    if freq_dist[item] > max_freq:
        max_freq = freq_dist[item]
        candidate = item

print(candidate)

