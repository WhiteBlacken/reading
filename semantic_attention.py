import numpy as np
import spacy
import torch
from docx import Document
from nltk.tokenize import sent_tokenize
from textstat import textstat
from transformers import BertForMaskedLM, BertTokenizer, XLNetModel, XLNetTokenizerFast

# base = "D:\\qxy\\pre-trained-model\\"
base = ""

nlp = spacy.load("en_core_web_lg")
tokenizer = XLNetTokenizerFast.from_pretrained(base+"xlnet-base-cased")
model = XLNetModel.from_pretrained(base+"xlnet-base-cased", output_attentions=True)
bert_tokenizer = BertTokenizer.from_pretrained(base+"bert-base-cased")
bert_model = BertForMaskedLM.from_pretrained(base+"bert-base-cased")

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
    return 700 - word_fam_map.get(capital_word, 0)