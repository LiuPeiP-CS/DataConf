#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 上午11:01
# @Author  : PeiP Liu
# @FileName: security_augmentation.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import gensim


def read_numpy(file_addr):
    np_data = np.load(file_addr)
    return np_data


class InternalAugmentation:

    def __init__(self, args, word_alphabet, semantic_emb_dim):
        self.args = args
        self.word_alphabet = word_alphabet
        self.semantic_emb_dim = semantic_emb_dim
        self.word2vec = read_numpy(self.args.word2vec)
        self.wv_model = gensim.models.Word2Vec.load(self.args.word2vec_model_path)

    def consine_sim(self, word):
        sim_scores = []
        sim_words = self.wv_model.wv.most_similar(word, topn=self.args.sim_num)
        for sim_word in sim_words:
            sim_scores.append(sim_word[1])
        return sim_scores

    def most_k_index(self, word):
        words_index = []
        sim_words = self.wv_model.wv.most_similar(word, topn=self.args.sim_num)
        for sim_word in sim_words:
            words_index.append(self.wv_model.wv.vocab[sim_word[0]].index)
        return words_index

    def compute_atten_embedding(self, word):
        sim_scores = np.array(self.consine_sim(word), dtype=np.float32)
        e_x = np.exp(sim_scores - np.max(sim_scores))
        f_x = e_x / e_x.sum()  # the softmax weight

        similar_words_embedding = self.word2vec[self.most_k_index(word)] # the index of most similar K words and vectors
        hard_atten_embedding = np.sum(f_x[:, None] * similar_words_embedding, axis=0)
        return hard_atten_embedding

    def semantic_emb_talbe(self):
        attention_embedding = np.zeros([self.word_alphabet.size(), self.semantic_emb_dim], dtype=np.float32)
        for word, ind in self.word_alphabet.items():
            if word in self.wv_model.wv.index2word:
                attention_embedding[ind, :] = self.compute_atten_embedding(word)
            else:
                # get the random augmentation embedding
                scale = np.sqrt(3.0 / self.semantic_emb_dim)
                attention_embedding[ind, :] = np.random.uniform(-scale, scale, [1, self.semantic_emb_dim])

        # attention_embedding = torch.tensor(attention_embedding, dtype=torch.float32).to(self.args.device)
        return attention_embedding
