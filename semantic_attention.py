from docx import Document
from nltk.tokenize import sent_tokenize
from textstat import textstat
from transformers import XLNetTokenizerFast, XLNetModel
import spacy
import numpy as np

nlp = spacy.load("en_core_web_lg")
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
model = XLNetModel.from_pretrained("xlnet-base-cased", output_attentions=True)

f = open('mrc2.dct', 'r')
word_fam_map = {}

i = 0
for line in f:
    line = line.strip()

    # see wordmodel.py for blurbs of each variable
    # or even better, mrc2.doc

    word, phon, dphon, stress = line[51:].split('|')

    w = {
        'wid': i,
        'nlet': int(line[0:2]),
        'nphon': int(line[2:4]),
        'nsyl': int(line[4]),
        'kf_freq': int(line[5:10]),
        'kf_ncats': int(line[10:12]),
        'kf_nsamp': int(line[12:15]),
        'tl_freq': int(line[15:21]),
        'brown_freq': int(line[21:25]),
        'fam': int(line[25:28]),
        'conc': int(line[28:31]),
        'imag': int(line[31:34]),
        'meanc': int(line[34:37]),
        'meanp': int(line[37:40]),
        'aoa': int(line[40:43]),
        'tq2': line[43],
        'wtype': line[44],
        'pdwtype': line[45],
        'alphasyl': line[46],
        'status': line[47],
        'var': line[48],
        'cap': line[49],
        'irreg': line[50],
        'word': word,
        'phon': phon,
        'dphon': dphon,
        'stress': stress
    }
    word_fam_map[word] = w['fam']
    i += 1


def get_word_familiar_rate(word_text):
    capital_word = word_text.upper()
    return word_fam_map.get(capital_word, 0)


def get_docx_text(docx_path):
    document = Document(docx_path)
    read_text = ''
    for pa in document.paragraphs:
        read_text += pa.text + ' '
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
    word_att_mat = generate_phrase_att(input_text, word_list)
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


if __name__ == '__main__':
    # rst = generate_word_attention(get_docx_text('/home/wtpan/memx4edu-code/exp_data/1009/2.docx'))
    texts = "It is not controversial to say that an unhealthy diet causes bad health..Nor are the basic elements of healthy eating disputed..Obesity raises susceptibility to cancer, and Britain is the six most obese country on Earth..That is a public health emergency..But naming the problem is the easy part..No one disputes the costs in quality of life and depleted health budgets of an obese population, but the quest for solutions gets diverted by ideological arguments around responsibility and choice..And the water is muddied by lobbying from the industries that profit from consumption of obesity-inducing products..Historical precedent suggests that science and politics can overcome resistance from businesses that pollute and poison but it takes time, and success often starts small..So it is heartening to note that a programme in Leeds has achieved a reduction in childhood obesity, becoming the first UK city to reverse a fattening trend..The best results were among younger children and in more deprived areas..When 28% of English children aged two to 15 are obese, a national shift on the scale achieved by Leeds would lengthen hundreds of thousands of lives..A significant factor in the Leeds experience appears to Many members of parliament are uncomfortable even with their own government's anti-obesity strategy, since it involves a 'sugar tax' and a ban on the sale of energy drinks to under-16s..Bans and taxes can be blunt instruments, but their harshest critics can rarely suggest better methods..These critics just oppose regulation itself.."
    texts = texts.replace('..','.')
    rst = generate_sentence_attention(texts)
    # rst = generate_word_difficulty(get_docx_text('/home/wtpan/memx4edu-code/exp_data/1009/2.docx'))
    print(rst)
