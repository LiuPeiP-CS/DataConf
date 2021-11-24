#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 上午10:01
# @Author  : PeiP Liu
# @FileName: data_utils.py
# @Software: PyCharm
import os
import torch
import torch.nn as nn
import datetime
import numpy as np
import json
from numpy import random
from collections import Counter
import csv
import pandas as pd


def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds = seconds))


def read_csv(file_path, name):
    infor = pd.read_csv(file_path).fillna(value='')
    sentences = infor['text']
    if name == 'train':
        sentences_labels = infor['BIO_anno']

    sentence_list = []
    sentence_label_list = []

    for i_data in range(infor.shape[0]):

        sentence = list()
        i_sentence = sentences[i_data]
        i_data_length = len(i_sentence)
        for i_char in range(i_data_length):
            sentence.append(i_sentence[i_char])

        if name == 'train':
            sentence_label = sentences_labels[i_data].strip().split(' ')
        else:
            sentence_label = ['O']*len(sentence)  # 对于测试数据，添加虚拟标签

        # print((sentence,sentence_label))
        # print((len(sentence), len(sentence_label)))
        assert len(sentence) == len(sentence_label)

        sentence_list.append(sentence)
        sentence_label_list.append(sentence_label)

    return sentence_list, sentence_label_list


def del_digit(sentence_list, sentence_label_list):
    new_sents = []
    new_sents_labels = []

    for i_sentence in range(len(sentence_list)):
        new_sentence = []
        new_sentence_label = []

        sentence = sentence_list[i_sentence]
        sentence_label = sentence_label_list[i_sentence]
        sent_len = len(sentence)
        token_iter = 0
        while token_iter < sent_len:
            token_post = token_iter+1
            if sentence[token_iter].isdigit():
                while token_post < sent_len and sentence[token_post].isdigit():
                    token_post = token_post + 1
            new_sentence.append(sentence[token_iter])
            new_sentence_label.append(sentence_label[token_iter])
            token_iter = token_post

        new_sents.append(new_sentence)
        new_sents_labels.append(new_sentence_label)

    return new_sents, new_sents_labels


def del_repretition(sentence_list, sentence_label_list):

    new_sents = []
    new_sents_labels = []

    for i_sentence in range(len(sentence_list)):
        new_sentence = []
        new_sentence_label = []

        sentence = sentence_list[i_sentence]
        sentence_label = sentence_label_list[i_sentence]
        sent_len = len(sentence)
        token_iter = 0
        while token_iter < sent_len:
            token_post = token_iter+1
            while token_post < sent_len and sentence[token_post] == sentence[token_iter]:
                token_post = token_post + 1
            new_sentence.append(sentence[token_iter])
            new_sentence_label.append(sentence_label[token_iter])
            token_iter = token_post

        new_sents.append(new_sentence)
        new_sents_labels.append(new_sentence_label)

    return new_sents, new_sents_labels


def maxlen_padding(sentences, sentences_labels, sent_maxlen, word_padding_value=0, label_padding_value=0):  # 输入是batch采样后的ids
    padded_sents = []
    padded_sentlabels = []

    for sent_iter, sent in enumerate(sentences):
        padded_sentence = []
        padded_sentence_label = []
        if len(sent) < sent_maxlen:
            padded_sentence = sent + [word_padding_value] * (sent_maxlen - len(sent))
            padded_sentence_label = sentences_labels[sent_iter] + [label_padding_value] * (sent_maxlen - len(sent))
        else:
            padded_sentence = sent[:sent_maxlen]
            padded_sentence_label = sentences_labels[sent_iter][:sent_maxlen]

        padded_sents.append(padded_sentence)
        padded_sentlabels.append(padded_sentence_label)

    return padded_sents, padded_sentlabels


def gen_batch_data(data, batch_size, shuffle=True):  # 此处的输入是全部的原始数据data，返回的也是一个batch的原始类型的data
    data = np.array(data)
    data_idx = np.arange(len(data))
    if shuffle:
        random.shuffle(data_idx)
    i = 0
    while True:
        if i + batch_size >= len(data):
            batch_idx = data_idx[i:]
            yield list(data[batch_idx])  # 将数据恢复成为列表格式
            break
        else:
            batch_idx = data_idx[i: i+batch_size]
            yield list(data[batch_idx])  # 将数据恢复成为列表格式
            i = i + batch_size

def batch_decomposition(batch_data):
    word_sentences = []
    wordid_sentences = []
    label_sentences = []
    labelid_sentences = []
    for each_instance in batch_data:
        word_sentences.append(each_instance[0])
        wordid_sentences.append(each_instance[1])
        label_sentences.append(each_instance[2])
        labelid_sentences.append(each_instance[3])

    return word_sentences, wordid_sentences, label_sentences, labelid_sentences

