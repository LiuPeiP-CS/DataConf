#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/8 下午5:15
# @Author  : PeiP Liu
# @FileName: transformer.py
# @Software: PyCharm
import torch
import torch.nn as nn
import copy, math, time
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')


class attention(nn.Module):
    def __init__(self, dropout):
        super(attention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attention_mask):
        scale = torch.tensor(k.size(-1), dtype=torch.float32)
        attention_matrix = q.bmm(k.permute(0,2,1).contiguous())
        attention_matrix = attention_matrix / torch.sqrt(scale)

        if attention_mask is not None:
            attention_matrix.masked_fill_(attention_mask, -np.inf)

        soft_attention = F.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = soft_attention.bmm(v)
        return output

def clones(module, N):
    """
    :param module: the single model layer
    :param N: the copy number of the single layer
    :return: the copy for original model layer
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, model_dim, dropout=0.1):
        """
        :param head_num: the number of head
        :param model_dim: the hidden dim of the model
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert model_dim % head_num == 0
        self.h_size = model_dim
        self.num_head = head_num
        self.d_k = model_dim // head_num

        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)

        self.attention = attention(dropout)  # we only have the statement for this argument
        self.fc = nn.Linear(self.h_size, self.h_size)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, q, k, v, attention_mask=None):
        batch_size = q.size(0)

        residual = q

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(batch_size * self.num_head, -1, self.d_k)
        k = k.view(batch_size * self.num_head, -1, self.d_k)
        v = v.view(batch_size * self.num_head, -1, self.d_k)

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(self.num_head, 1, 1)
            #

        context = self.attention(q, k, v, attention_mask)

        context = context.contiguous().view(batch_size, -1, self.h_size)

        output = self.dropout(self.fc(context))
        output = self.layer_norm(residual + output)

        return output

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim

        self.Linear1 = nn.Linear(self.model_dim, self.ff_dim)
        self.Linear2 = nn.Linear(self.ff_dim, self.model_dim)
        self.norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output1 = self.Linear2(F.relu(self.Linear1(x)))
        output2 = self.dropout(output1)
        output = self.norm(x + output2)
        return output


class AttentionModel(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, num_head, num_layer, dropout):
        super(AttentionModel, self).__init__()
        self.input2model = nn.Linear(input_dim, model_dim)
        # self.position_emb = APosEmb(seq_len, model_dim, dropout)
        self.num_layer = num_layer
        for layer_iter in range(self.num_layer):
            self.__setattr__('MultiHeadAttention_{}'.format(layer_iter), MultiHeadAttention(
                head_num=num_head,
                model_dim=model_dim,
                dropout=dropout
            ))

            self.__setattr__('FeedForward_{}'.format(layer_iter), FeedForward(
                model_dim=model_dim,
                ff_dim=ff_dim,
                dropout=dropout
            ))

    def forward(self, input_feature, attention_mask):
        # input_feature.shape = (batch_size, seq_len, input_dim)
        # initial linear transformation for input feature(after fusing augmentation), and add the position feature
        # x = self.position_emb(self.input2model(input_feature))
        x = self.input2model(input_feature)

        for layer_iter in range(self.num_layer):
            x = self.__getattr__('MultiHeadAttention_{}'.format(layer_iter))(x, x, x, attention_mask)
            x = self.__getattr__('FeedForward_{}'.format(layer_iter))(x)

        # the output.shape = (batch_size, seq_len, model_dim)
        return x
