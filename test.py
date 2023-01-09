# 绘制折线图


# nlp feature
from semantic_attention import generate_word_attention, get_word_difficulty
from utils import get_importance, get_word_and_sentence_from_text, normalize

text1 = "feature of 'folk psychology' anyway. If a first-class scientific account of intentional facts and phenomena can't be"

text2 = "given, that is because scientific reductionism is not the right line to take in metaphysics,but rather it is"


def get_nlp_feature(text):
    word_list, sentence_list = get_word_and_sentence_from_text(text)
    difficulty_level = [get_word_difficulty(x) for x in word_list]  # text feature
    # difficulty_level = normalize(difficulty_level)
    word_attention = generate_word_attention(text)
    importance = get_importance(text)

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

    nlp_feature = [(1 / 3) * difficulty_level[i] + (1 / 3) * importance_level[i] + (1 / 3) * attention_level[i] for i in
                   range(len(word_list))]
    # return nlp_feature, word_list
    return [get_word_difficulty(x) for x in word_list],word_list


nlp_feature1, word_list1 = get_nlp_feature(text1)
nlp_feature2, word_list2 = get_nlp_feature(text2)
print(nlp_feature2)
print(word_list2)

# ['feature', 'of', "'folk", "psychology'", 'anyway', 'If', 'a', 'first-class', 'scientific', 'account', 'of', 'intentional', 'facts', 'and', 'phenomena', "can't", 'be']
visual_feature1 = [1, 0, 0, 2, 1, 0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 0]

visual_feature2 = []
for i, feature in enumerate(nlp_feature2):
    print(f'{word_list1[i]}:{feature}')
