from transformers import XLNetTokenizerFast, XLNetModel
from docx import Document
import numpy as np

tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
model = XLNetModel.from_pretrained("xlnet-base-cased", output_attentions=True)


def generate_word_list(input_text):
    doc = nlp(input_text)
    word_list = []
    word4show_list = []
    for token in doc:
        word_list.append(token.text)
        word4show_list.append(token.text.strip())
    return word_list, word4show_list


def generate_phrase_att(input_text, phrase_list, phrase4show_list):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    attention = outputs[-1]
    att = attention[4][0][0].detach().numpy()

    phrase_pos = 0
    phrase_idx = 0
    token_phrase_idx_list = []
    max_cross_count = 3
    for token_idx, token in enumerate(inputs.tokens()):
        original_token = token.replace('▁', '')
        flag = False
        while phrase_idx < len(phrase_list) and phrase_list[phrase_idx][phrase_pos : phrase_pos + len(original_token)] != original_token:
            phrase_pos += 1
            if phrase_pos >= len(phrase_list[phrase_idx]):
                cross_count = 1
                prefix = phrase_list[phrase_idx]
                pre_length = len(phrase_list[phrase_idx])
                while cross_count <= max_cross_count and phrase_idx + cross_count < len(phrase_list):
                    prefix += phrase_list[phrase_idx + cross_count]
                    new_phrase_pos = 0
                    while new_phrase_pos + len(original_token) <= len(prefix) and new_phrase_pos < len(phrase_list[phrase_idx]):
                        if prefix[new_phrase_pos : new_phrase_pos + len(original_token)] == original_token and new_phrase_pos + len(original_token) > len(phrase_list[phrase_idx]):
                            print(prefix[new_phrase_pos : new_phrase_pos + len(original_token)], original_token)
                            phrase_pos = new_phrase_pos + len(original_token) - pre_length
                            print(phrase_pos)
                            flag = True
                            break
                        new_phrase_pos += 1
                    if flag:
                        break
                    pre_length += len(phrase_list[phrase_idx + cross_count])
                    cross_count += 1
                if flag:
                    for delta_idx in range(cross_count + 1):
                        token_phrase_idx_list.append((token_idx, phrase_idx + delta_idx))
                    phrase_idx += cross_count
                    # print('哈哈哈哈')
                    # print(phrase_list[phrase_idx], phrase_pos)
                    if phrase_idx < len(phrase_list) and phrase_pos == len(phrase_list[phrase_idx]):
                        phrase_pos = 0
                        phrase_idx += 1
                    break
                else:
                    phrase_pos = 0
                    phrase_idx += 1
        if flag:
            continue
        # print(original_token, phrase_list[phrase_idx])
        if phrase_idx < len(phrase_list) and phrase_pos == len(phrase_list[phrase_idx]):
            phrase_pos = 0
            phrase_idx += 1
        if phrase_idx >= len(phrase_list):
            token_phrase_idx_list.append((token_idx, -1))
            continue
        token_phrase_idx_list.append((token_idx, phrase_idx))
        phrase_pos += len(original_token)

    # for token_idx, phrase_idx in token_phrase_idx_list:
    #     print(inputs.tokens()[token_idx], phrase_list[phrase_idx])

    phrase_start_end = {}
    for phrase_idx in range(-1, len(phrase_list)):
        phrase_start_end[phrase_idx] = (99999, 0)
    for token_idx, phrase_idx in token_phrase_idx_list:
        phrase_start_end[phrase_idx] = (min(phrase_start_end[phrase_idx][0], token_idx), max(phrase_start_end[phrase_idx][1], token_idx))
    phrase_start_end.pop(-1)

    # for idx, (phrase_id, (phrase_start, phrase_end)) in enumerate(phrase_start_end.items()):
    #     print(phrase_id, phrase_list[phrase_id], [inputs.tokens()[i] for i in range(phrase_start, phrase_end + 1)])

    rows = []
    for phrase_idx, (token_start_idx, token_end_idx) in phrase_start_end.items():
        if token_start_idx > token_end_idx:
            rows.append(np.zeros((len(inputs.tokens()), )))
        else:
            rows.append(np.mean(att[token_start_idx : token_end_idx + 1, :], axis=0))
    rows = np.array(rows)
    cols = []
    for phrase_idx, (token_start_idx, token_end_idx) in phrase_start_end.items():
        if token_start_idx > token_end_idx:
            cols.append(np.zeros((len(phrase_list), 1)))
        else:
            cols.append(np.mean(rows[:, token_start_idx : token_end_idx + 1], axis=1, keepdims=True))
    cols = np.concatenate(cols, axis=1)
    phrase_att = cols

    return phrase_att

def generate_word_attention(input_text):
    doc = nlp(input_text)
    word_list, word4show_list = generate_word_list(input_text)
    # print(phrase_list)
    word_att_mat = generate_phrase_att(read_text, word_list, word4show_list)
    word_att_value = np.average(word_att_mat, axis=0)
    # print(word_att_mat)
    word_att_list = []
    for token, value in zip(doc, word_att_value):
        if token.is_alpha and not token.is_stop:
            score = value
        else:
            score = 0.
        word_att_list.append((token.text, score))
    return word_att_list

if __name__ == "__main__":
    list = generate_word_attention('It is not controversial to say that an unhealthy diet causes bad health. Nor are the basic elements of healthy eating disputed. Obesity raises susceptibility to cancer, and Britain is the six most obese country on Earth. That is a public health emergency. But naming the problem is the easy part. No one disputes the costs in quality of life and depleted health budgets of an obese population, but the quest for solutions gets diverted by ideological arguments around responsibility and choice. And the water is muddied by lobbying from the industries that profit from consumption of obesity-inducing products.')
    print(list)