#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 下午10:49
# @Author  : PeiP Liu
# @FileName: BertData.py
# @Software: PyCharm
import torch
import torch.nn
import numpy as np
import sys
sys.path.append('..')
from NER.alphabet import Alphabet


def load_pretrained_emb(emb_file_path):
    emb_dim = -1
    emb_dict = dict()
    with open(emb_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if emb_dim < 0:
                emb_dim = len(tokens) - 1
            else:
                assert emb_dim+1 == len(tokens)

            embedding = [float(x) for x in tokens[1:]]
            emb_dict[tokens[0]] = np.array(embedding)
            return emb_dict, emb_dim

def emb_norm(embedidng):
    root_sum_square = np.sqrt(np.sum(np.square(embedidng)))
    return embedidng / root_sum_square


class PreData:
    def __init__(self):
        self.label_alphabet = Alphabet('label', True)
        self.word_alphabet = Alphabet('word')

    def build_alphabet(self, sentences, name, sentences_labels=None):
        for sent_i in range(len(sentences)):
            for word_j in range(len(sentences[sent_i])):
                self.word_alphabet.add(sentences[sent_i][word_j])
                if name == 'train':
                    self.label_alphabet.add(sentences_labels[sent_i][word_j])

    def build_embedding_table(self, emb_file):
        emb_dict, emb_dim = load_pretrained_emb(emb_file)
        scale = np.sqrt(3.0 / emb_dim)
        emb_table = np.empty([self.word_alphabet.size(), emb_dim], dtype=np.float32)
        emb_table[:2, :] = np.random.uniform(-scale, scale, [2, emb_dim])  # for UNK and PAD
        # emb_table[0, :] = np.random.uniform(-scale, scale, [1, emb_dim])  # for UNK
        for instance, index in self.word_alphabet.items():
            if instance in emb_dict:
                instance_emb = emb_norm(emb_dict[instance])
            else:
                instance_emb = np.random.uniform(-scale, scale, [1, emb_dim])  # 对于不在向量字典里的向量，进行随机化
            emb_table[index, :] = instance_emb

        return emb_table, emb_dim

    def text2ids(self, sentences, sentences_labels):
        instances = []
        for sent_iter, sentence in enumerate(sentences):
            sentence_label = sentences_labels[sent_iter]
            assert len(sentence) == len(sentence_label)
            word_ids = [self.word_alphabet.get_index(word) for word in sentence]
            labels_ids = [self.label_alphabet.get_index(label) for label in sentence_label]

            instances.append([sentence, word_ids, sentence_label, labels_ids])

        return instances
