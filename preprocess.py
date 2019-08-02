import pandas as pd
import itertools
import re
import torch
from tqdm import tqdm_notebook
import sys


PAD = 0
SOS = 1
EOS = 2
UNK = 3

class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.word2count = {}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addToken(word)

    def addToken(self, word):

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def replaceRareword(self, min_count):

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.word2count = {}
        self.n_words = 4

        for word in keep_words:
            self.addToken(word)


def normalizeText(text):
    #     if re.findall('[a-zA-Z]', text):
    #         text = translator.translate(text, dest='ko').text

    result = re.sub('[^a-zA-Zㄱ-ㅣ가-힣.?! ]', ' ', text)
    result = re.sub(' +', ' ', result)

    return result


def prepareText(csv_dir, input_col, output_col, voc_name, tokenize_by):
    inputs = list(pd.read_csv(csv_dir, encoding='utf-8')[input_col])
    outputs = list(pd.read_csv(csv_dir, encoding='utf-8')[output_col])

    normalized_inputs = [normalizeText(text) for text in inputs]
    normalized_outputs = [normalizeText(text) for text in outputs]

    if tokenize_by == 'morph':
        normalized_inputs = [' '.join(okt.morphs(text)) for text in normalized_inputs]
        normalized_outputs = [' '.join(okt.morphs(text)) for text in normalized_outputs]

    voc_dic = Dictionary(voc_name)

    for q, a in tqdm_notebook(zip(normalized_inputs, normalized_outputs),
                              desc="Loading", file=sys.stdout):
        voc_dic.addSentence(q)
        voc_dic.addSentence(a)

    return normalized_inputs, normalized_outputs, voc_dic


def pairUp(inputs, outputs):
    pair = []
    for i in range(len(inputs)):
        pair.append([inputs[i], outputs[i]])
    return pair


def indexesFromSentence(voc_dic, sentence):
    seq = []
    for word in sentence.split(' '):
        if word not in voc_dic.word2index:
            seq.append(UNK)
        else:
            seq.append(voc_dic.word2index[word])
    seq.append(EOS)

    return seq


def zeroPadding(l, pad_with=PAD):
    return list(itertools.zip_longest(*l, fillvalue=pad_with))


def binaryMatrix(l, value=PAD):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(l, voc_dic):
    indexes_batch = [indexesFromSentence(voc_dic, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l, voc_dic):
    indexes_batch = [indexesFromSentence(voc_dic, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc_dic, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc_dic)
    output, mask, max_target_len = outputVar(output_batch, voc_dic)
    return inp, lengths, output, mask, max_target_len
