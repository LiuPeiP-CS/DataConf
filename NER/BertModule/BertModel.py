#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 上午10:04
# @Author  : PeiP Liu
# @FileName: BertModel.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.nn import LayerNorm as BertLayerNorm
import sys
sys.path.append("../..")
# from torchcrf import CRF
from NER.crf import CRF
import torch.nn.functional as F

def decode_tag(emission, valid_mask):
    """
    :param emission: (batch_size, sent_len, num_labels)
    :param valid_mask: (batch_size, sent_lem)
    :return:
    """
    valid_sentlen = valid_mask.sum(1)
    pre_tag = emission.argmax(-1)
    pre_valid_tag = [pre_tag[i_sent][:valid_sentlen[i_sent].item()].detach().tolist() for i_sent in range(emission.size(0))]

    return pre_valid_tag

class BERT_CRF_NER(nn.Module):
    def __init__(self, bert_model, label_alphabet, hidden_size=768, batch_size=64, max_seq_len=256, device='cpu'):
        super(BERT_CRF_NER, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_label = label_alphabet.size()
        self.label_alphabet = label_alphabet
        self.device = device
        # create the dropout layer
        self.dropout = nn.Dropout(0.5)
        self.bert_sigmod = nn.Sigmoid()
        # create the final nn layer to convert the output feature to emission
        self.hid2label = nn.Linear(self.hidden_size, self.num_label)
        # init the weight and bias of feature-emission layer
        nn.init.xavier_uniform_(self.hid2label.weight)
        nn.init.constant_(self.hid2label.bias, 0.0)
        self.crf = CRF(self.num_label, self.label_alphabet.get_index('<PAD>'), self.label_alphabet.get_index('<BOS>'),self.label_alphabet.get_index('<EOS>'), 'cuda')


    def get_bert_features(self, input_ids, seg_ids, input_mask):
        # rf https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        outputs = self.bert_model(input_ids, token_type_ids=seg_ids,
                                  attention_mask=input_mask, output_hidden_states=True, output_attentions=True)
        token_features = outputs[0] # (batch_size, seq_length, hidden_size)

        batch_size, seq_len, feature_dim = token_features.shape

        wv = torch.zeros(token_features.shape, dtype=torch.float32).to(self.device)
        for batch_iter in range(batch_size):
            # get the valid information except for [CLS] and [SEP]
            valid_token_input = token_features[batch_iter][input_mask[batch_iter].bool()][1:-1]
            i_word_vector = 0
            while i_word_vector < len(valid_token_input):
                wv[batch_iter][i_word_vector] = valid_token_input[i_word_vector]
                i_word_vector+=1

        dropout_feature = self.dropout(wv)
        return dropout_feature

    def neg_log_likehood(self, input_ids, input_mask, seg_ids, true_label_ids, true_label_mask):
        dropout_feature = self.get_bert_features(input_ids, seg_ids, input_mask)
        wv2emission = self.hid2label(dropout_feature)
        wv2emission = self.bert_sigmod(wv2emission)
        object_score = self.crf(wv2emission, true_label_ids, true_label_mask != 0)
        _, path = self.crf.viterbi_decode(wv2emission, true_label_mask != 0)
        print(path)
        # object_score = F.cross_entropy(wv2emission.view(-1, 10), true_label_ids.view(-1), ignore_index=0)
        return object_score

    def forward(self, input_ids, input_mask, seg_ids, true_label_mask):
        dropout_feature = self.get_bert_features(input_ids, seg_ids, input_mask)
        word2emission = self.hid2label(dropout_feature)
        word2emission = self.bert_sigmod(word2emission)
        _, path = self.crf.viterbi_decode(word2emission, true_label_mask != 0) # 注意该包的特殊需求 
        return path
        # return decode_tag(word2emission, true_label_mask)
