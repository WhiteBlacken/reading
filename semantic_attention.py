import numpy as np
import spacy
import torch
from docx import Document
from nltk.tokenize import sent_tokenize
from textstat import textstat
from transformers import BertForMaskedLM, BertTokenizer, XLNetModel, XLNetTokenizerFast

nlp = spacy.load("en_core_web_lg")
tokenizer = XLNetTokenizerFast.from_pretrained(r"D:\qxy\pre-trained-model\xlnet-base-cased")
model = XLNetModel.from_pretrained(r"D:\qxy\pre-trained-model\xlnet-base-cased", output_attentions=True)
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained(r"D:\qxy\pre-trained-model\bert-base-cased")
# bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained(r"D:\qxy\pre-trained-model\bert-base-cased")

f = open("mrc2.dct", "r")
word_fam_map = {}

i = 0
for line in f:
    line = line.strip()

    # see wordmodel.py for blurbs of each variable
    # or even better, mrc2.doc

    word, phon, dphon, stress = line[51:].split("|")

    w = {
        "wid": i,
        "nlet": int(line[0:2]),
        "nphon": int(line[2:4]),
        "nsyl": int(line[4]),
        "kf_freq": int(line[5:10]),
        "kf_ncats": int(line[10:12]),
        "kf_nsamp": int(line[12:15]),
        "tl_freq": int(line[15:21]),
        "brown_freq": int(line[21:25]),
        "fam": int(line[25:28]),
        "conc": int(line[28:31]),
        "imag": int(line[31:34]),
        "meanc": int(line[34:37]),
        "meanp": int(line[37:40]),
        "aoa": int(line[40:43]),
        "tq2": line[43],
        "wtype": line[44],
        "pdwtype": line[45],
        "alphasyl": line[46],
        "status": line[47],
        "var": line[48],
        "cap": line[49],
        "irreg": line[50],
        "word": word,
        "phon": phon,
        "dphon": dphon,
        "stress": stress,
    }
    word_fam_map[word] = w["fam"]
    i += 1


def get_word_familiar_rate(word_text):
    capital_word = word_text.upper()
    return word_fam_map.get(capital_word, 0)


def get_docx_text(docx_path):
    document = Document(docx_path)
    read_text = ""
    for pa in document.paragraphs:
        read_text += pa.text + " "
    return read_text


def generate_word_list(input_text):
    doc = nlp(input_text)
    word_list = []
    word4show_list = []
    for token in doc:
        word_list.append(token.text)
        word4show_list.append(token.text.strip())
    return word_list, word4show_list


def generate_phrase_att(input_text, phrase_list):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    attention = outputs[-1]
    att = attention[4][0][0].detach().numpy()

    phrase_pos = 0
    phrase_idx = 0
    token_phrase_idx_list = []
    max_cross_count = 3
    for token_idx, token in enumerate(inputs.tokens()):
        original_token = token.replace("▁", "")
        flag = False
        while (
            phrase_idx < len(phrase_list)
            and phrase_list[phrase_idx][phrase_pos : phrase_pos + len(original_token)] != original_token
        ):
            phrase_pos += 1
            if phrase_pos >= len(phrase_list[phrase_idx]):
                cross_count = 1
                prefix = phrase_list[phrase_idx]
                pre_length = len(phrase_list[phrase_idx])
                while cross_count <= max_cross_count and phrase_idx + cross_count < len(phrase_list):
                    prefix += phrase_list[phrase_idx + cross_count]
                    new_phrase_pos = 0
                    while new_phrase_pos + len(original_token) <= len(prefix) and new_phrase_pos < len(
                        phrase_list[phrase_idx]
                    ):
                        if prefix[
                            new_phrase_pos : new_phrase_pos + len(original_token)
                        ] == original_token and new_phrase_pos + len(original_token) > len(phrase_list[phrase_idx]):
                            phrase_pos = new_phrase_pos + len(original_token) - pre_length
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
                    if phrase_idx < len(phrase_list) and phrase_pos == len(phrase_list[phrase_idx]):
                        phrase_pos = 0
                        phrase_idx += 1
                    break
                else:
                    phrase_pos = 0
                    phrase_idx += 1
        if flag:
            continue
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
        phrase_start_end[phrase_idx] = (
            min(phrase_start_end[phrase_idx][0], token_idx),
            max(phrase_start_end[phrase_idx][1], token_idx),
        )
    phrase_start_end.pop(-1)

    # for idx, (phrase_id, (phrase_start, phrase_end)) in enumerate(phrase_start_end.items()):
    #     print(phrase_id, phrase_list[phrase_id], [inputs.tokens()[i] for i in range(phrase_start, phrase_end + 1)])

    rows = []
    for phrase_idx, (token_start_idx, token_end_idx) in phrase_start_end.items():
        if token_start_idx > token_end_idx:
            rows.append(np.zeros((len(inputs.tokens()),)))
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
    word_att_mat = generate_phrase_att(input_text, word_list)
    word_att_value = np.average(word_att_mat, axis=0)
    # print(word_att_mat)
    word_att_list = []
    for token, value in zip(doc, word_att_value):
        if token.is_alpha and not token.is_stop:
            score = value
        else:
            score = 0.0
        word_att_list.append((token.text, score))
    return word_att_list


def generate_sentence_attention(input_text):
    sentence_list = sent_tokenize(input_text)
    # print(phrase_list)
    sentence_att_mat = generate_phrase_att(input_text, sentence_list)
    # print(sentence_att_mat.shape, len(sentence_list))
    sentence_att_value = np.average(sentence_att_mat, axis=0)
    # print(word_att_mat)
    sentence_att_list = []
    for sentence, score in zip(sentence_list, sentence_att_value):
        sentence_att_list.append((sentence, score))
    return sentence_att_list


def generate_word_difficulty(input_text):
    doc = nlp(input_text)
    word_difficulty_list = []
    for token in doc:
        score = 0
        if token.is_alpha and not token.is_stop:
            fam = get_word_familiar_rate(token.text)
            if fam == 0:
                fam = get_word_familiar_rate(token.lemma_)
            # print(token.text, token.lemma_, textstat.syllable_count(token.text), len(token.text), fam)
            if textstat.syllable_count(token.text) > 2:
                score += 1
            if len(token.text) > 7:
                score += 1
            if fam < 482:
                score += 1
        word_difficulty_list.append((token.text, score))
    return word_difficulty_list


def generate_sentence_difficulty(input_text):
    bert_model.eval()
    sentence_list = sent_tokenize(input_text)
    sentence_difficulty_list = []
    for sentence in sentence_list:
        tokenize_input = bert_tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([bert_tokenizer.convert_tokens_to_ids(tokenize_input)])
        sen_len = len(tokenize_input)
        sent_loss = 0.0
        for i, word in enumerate(tokenize_input):
            tokenize_input[i] = bert_tokenizer.mask_token
            mask_input = torch.tensor([bert_tokenizer.convert_tokens_to_ids(tokenize_input)])
            output = bert_model(mask_input)
            pred_scores = output[0]
            ps = torch.log_softmax(pred_scores[0, i], dim=0)
            word_loss = ps[tensor_input[0, i]]
            sent_loss += word_loss.item()
            tokenize_input[i] = word  # restore
        ppl = np.exp(-sent_loss / sen_len)
        sentence_difficulty_list.append((sentence, ppl))
    return sentence_difficulty_list


def getAttention(output, layer, head):
    res = output[2]
    # print(type(res))#
    # print(len(res)) # 12 。解释一下是12的原因：Number of hidden layers in the Transformer encoder 是12
    # print(res) #(tensor()...,tensor()...)

    layer_attention_score = res[layer]  # 得到 attention 中的最后一个tuple，也就是最后一层的attention 值
    # print(type(res)) # 去掉一维之后仍然是tuple
    # print(attention_score.size()) # torch.Size([1, 12, 28, 28])。 这个size = [batch_size,num_head,seq_len,seq_len]

    layer_attention_score.squeeze_(0)  # 去掉第一维的1
    # print(attention_score.size()) # 因为有12个head，这里只取第一个
    layer_head_attention_score = layer_attention_score[head, :, :]
    # print(layer_head_attention_score.size())
    # print(layer_head_attention_score)
    return layer_head_attention_score


# print(first_attention_score)
"""可视化attention 的结果
01.可视化某个head
02.或者把所有的head平均下来看看
"""


def drawAttention(inputs, first_attention_score, head):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import pandas as pd

    # 将first_attention_score 的grad属性去除 => 使用detach()，并转为numpy
    res2 = first_attention_score.detach().numpy()
    temp = inputs["input_ids"][0]  # 得到input_ids，为了将其转换成tokens
    # print(temp)
    tokens = tokenizer.convert_ids_to_tokens(temp)
    # print(type(tokens)) #
    # print(tokens) # ['[CLS]', 'it', 'is', ...]

    print(res2)
    # tokens就是我们的横纵坐标(标签)
    df = pd.DataFrame(res2, columns=tokens, index=tokens)
    fig = plt.figure(figsize=(10, 10))  # 画图，这里调整了图片的大小，否则会因为word太长导致文字重叠
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(df, interpolation="nearest", cmap="hot_r")
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticklabels([""] + list(df.columns))
    ax.set_yticklabels([""] + list(df.index))
    str(head) + ".png"
    # plt.savefig(name) # 存储图片
    plt.show()


def process(text, tokenizer, bert):
    inputs = tokenizer(text, return_tensors="pt")
    # print(inputs)
    output = bert(**inputs, output_attentions=True)  # 因为需要输出attention score 的值，所以这里用了 output_attentions 参数
    return output


def att_pic(tokens, att, head, layer):
    import pandas as pd
    import seaborn as sns

    remove_list = ["[CLS]", "[SEP]", ".", "sitting", "in", "a", "car", "but", "that", "once", "to", "be", "the", "is"]

    data = []
    for i, row in enumerate(att):
        if tokens[i] in remove_list:
            continue
        tmp = []
        for j, item in enumerate(row):
            if tokens[j] in remove_list:
                continue
            tmp.append(att[i][j])
        data.append(tmp)

    tokens = [x for x in tokens if x not in remove_list]
    print(len(tokens))
    print(len(data))
    df = pd.DataFrame(data, index=tokens, columns=tokens)

    sns.set_theme()
    ax = sns.heatmap(df, linewidths=0.5, annot=True, fmt=".2f")
    plt.tight_layout()
    plt.savefig("static\\data\\atts\\" + str(layer) + "-" + str(head) + ".png")
    plt.show()


if __name__ == "__main__":

    sentence = "Sitting in a more comfortable car in a different traffic jam is pleasant but hardly the liberation that once seemed to be promised."

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(r"D:\qxy\pre-trained-model\bert-base-cased")

    tokens = tokenizer.tokenize(sentence)
    print(tokens)

    import matplotlib.pyplot as plt
    from transformers import BertModel

    tokenizer = BertTokenizer.from_pretrained(r"D:\qxy\pre-trained-model\bert-base-cased")
    bert = BertModel.from_pretrained(r"D:\qxy\pre-trained-model\bert-base-cased")
    # text ="It is in this spirit that a majority of American governments have passed new laws since 2008 making the registration or voting process more difficult."
    text = "Sitting in a more comfortable car in a different traffic jam is pleasant but hardly the liberation that once seemed to be promised."
    # figure = plt.figure(30,30)
    output = process(text, tokenizer, bert)

    # 根据每个head，画出一张attention图
    layer = 4
    heads = [1]
    # pass 12 11 1
    # heads = [2]
    for head in heads:
        layer_head_attention_score = getAttention(output, layer, head)
        res2 = layer_head_attention_score.detach().numpy()

        inputs = tokenizer(text, return_tensors="pt")
        temp = inputs["input_ids"][0]  # 得到input_ids，为了将其转换成tokens
        tokens = tokenizer.convert_ids_to_tokens(temp)
        print(tokens)
        att_pic(tokens, res2, head, layer)
