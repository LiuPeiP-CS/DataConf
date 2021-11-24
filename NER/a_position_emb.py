#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 下午6:15
# @Author  : PeiP Liu
# @FileName: position_emb.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn
import numpy as np
# from torch.autograd import Variable

class APosEmb(nn.Module):
    # rf https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    def __init__(self, model_dim, dropout_rate, maxseq_len):
        super(APosEmb, self).__init__()
        self.register_buffer('position_embedding', self.get_sinusoid_encoding_table(maxseq_len, model_dim))  # 相当于属性赋值
        self.dropout = nn.Dropout(p=dropout_rate)

    def get_sinusoid_encoding_table(self, maxseq_len, model_dim):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(model_dim_j//2)/model_dim) for model_dim_j in range(model_dim)]

        sinusoid_table = np.array([get_position_angle_vec(i_position) for i_position in range(maxseq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim = 2i, : for all tokens of sequence but not the batch
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim = 2i+1
        return torch.tensor(sinusoid_table, dtype=torch.float32)

    def forward(self, x):
        return self.dropout(x + self.position_embedding[:, :x.size(1)].clone().detach())
