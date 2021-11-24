#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 下午6:14
# @Author  : PeiP Liu
# @FileName: augmentation.py
# @Software: PyCharm
import torch
import torch.nn as nn
import json
from collections import OrderedDict
from gensim.models import KeyedVectors
from annoy import AnnoyIndex
import numpy as np


class SemanticAug(nn.Module):
    def __init__(self, vector_file, word_alphabet, semantic_emb_dim):  # 打开预训练好的词向量文件
        super(SemanticAug, self).__init__()
        self.word_alphabet = word_alphabet
        self.semantic_emb_dim = semantic_emb_dim

        tc_wv_model = KeyedVectors.load_word2vec_format(vector_file, binary=False)

        self.word2index = OrderedDict()  # 构建数据字典，word: id
        for word_iter, word in enumerate(tc_wv_model.vocab.keys()):
            self.word2index[word] = word_iter

        self.tc_index = AnnoyIndex(200)
        i = 0
        for word in tc_wv_model.vocab.keys():
            word_vector = tc_wv_model[word]  # 获取单词向量
            self.tc_index.add_item(i, word_vector)  # 构建索引向量
            i += 1

        self.tc_index.build(10)
        self.index2word = dict([(word_id, word) for (word, word_id) in self.word2index.items()])

    def word_augmentation(self, word, nb_num=4):  # 为每个字符提供语义增强信息
        word_item = self.word2index[word]  # 获取当前词的id
        distances = []  # 与相似语义word的距离
        similar_vectors = []  # 获取相似词的相应向量
        for item in self.tc_index.get_nns_by_item(word_item, nb_num):
            distances.append(self.tc_index.get_distance(word_item, item))
            similar_vectors.append(self.tc_index.get_item_vector(item))

        sim_scores = np.array(distances, dtype=np.float32)
        sim_word_embeddings = np.array(similar_vectors, dtype=np.float32)
        e_x = np.exp(sim_scores - np.max(sim_scores))
        f_x = e_x / e_x.sum()

        return np.sum(f_x[:, None] * sim_word_embeddings, axis=0)

    def semantic_emb_talbe(self):
        scale = np.sqrt(3.0 / self.semantic_emb_dim)
        emb_table = np.empty([self.word_alphabet.size(), self.semantic_emb_dim], dtype=np.float32)
        emb_table[:2, :] = np.random.uniform(-scale, scale, [2, self.semantic_emb_dim])  # for UNK and PAD
        for instance, index in self.word_alphabet.items():
            if instance in self.word2index:  # 训练数据集中的词是否在外部数据中
                instance_emb = self.word_augmentation(instance)
            else:
                instance_emb = np.random.uniform(-scale, scale, [1, self.semantic_emb_dim])  # 对于不在向量字典里的向量，进行随机化
            emb_table[index, :] = instance_emb

        return emb_table



