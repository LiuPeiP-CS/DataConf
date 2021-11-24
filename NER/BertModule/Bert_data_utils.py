#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 下午2:57
# @Author  : PeiP Liu
# @FileName: Bert_data_utils.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
# from keras_preprocessing.sequence import pad_sequences


class InputFeature():
    def __init__(self, input_ids, input_mask, seg_ids, true_label_ids, true_label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seg_ids = seg_ids
        self.true_label_ids = true_label_ids
        self.true_label_mask = true_label_mask


class DataProcessor():
    def __init__(self, sentences, sentence_labels, tokenizer, max_seq_len, label_alphabet):
        self.sentences = sentences
        self.sentence_labels = sentence_labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_alphabet = label_alphabet

    def sentence2feature(self, sentence, sentence_label):

        tokens = ['[CLS]']  # the beginning of a sentence
        true_label_ids = []

        for i, i_word in enumerate(sentence):
            tokens.extend(i_word)
            true_label_ids.append(self.label_alphabet.get_index(sentence_label[i]))

        # truncating before filling
        if len(tokens) > self.max_seq_len-1:
            tokens = tokens[:self.max_seq_len-1]

        tokens = tokens + ['[SEP]']
        input_mask = len(tokens) * [1]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(input_ids) < self.max_seq_len:
            input_mask.append(0)
            input_ids.append(0)

        if len(true_label_ids) > self.max_seq_len:
            true_label_ids = true_label_ids[:self.max_seq_len]
            # word_token_num = word_token_num[:self.max_seq_len]

        true_label_mask = len(true_label_ids) * [1]

        while len(true_label_ids) < self.max_seq_len:
            true_label_ids.append(0)
            true_label_mask.append(0)

        seg_ids = self.max_seq_len * [0]

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(seg_ids) == self.max_seq_len
        assert len(true_label_ids) == self.max_seq_len
        assert len(true_label_mask) == self.max_seq_len
        # print(tokens)
        # print(input_ids)
        # print(true_label_ids)


        return input_ids, input_mask, seg_ids, true_label_ids, true_label_mask

    def get_features(self):
        features = []
        for sentence, sentence_label in zip(self.sentences, self.sentence_labels):
            ii, im, si, tli, tlm = self.sentence2feature(sentence, sentence_label)
            features.append(InputFeature(ii, im, si, tli, tlm))
        return features


class BertCRFData(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        return feature.input_ids, feature.input_mask, feature.seg_ids,\
               feature.true_label_ids, feature.true_label_mask

    @classmethod
    def seq_tensor(cls, batch):

        list2tensor = lambda x: torch.tensor([feature[x] for feature in batch], dtype=torch.long)
        input_ids = list2tensor(0)
        input_mask = list2tensor(1)
        seg_ids = list2tensor(2)
        true_label_ids = list2tensor(3)
        true_label_mask = list2tensor(4)
        return input_ids, input_mask, seg_ids, true_label_ids, true_label_mask

