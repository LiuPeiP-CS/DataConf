#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 下午9:14
# @Author  : PeiP Liu
# @FileName: ner_model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from NER.BertModule.Bert_Feature import GetBertFeature
from NER.a_position_emb import APosEmb
from NER.semantic_augmentation import SemanticAug
from NER.InternalAugmentation import InternalAugmentation
from NER.transformer import AttentionModel
from data_utils import *
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

class NER_model(nn.Module):
    def __init__(self, basic_args, label_alphabet, word_alphabet, emb_table, emb_dim):
        super(NER_model,self).__init__()
        self.args = basic_args
        self.label_alphabet = label_alphabet
        self.word_alphabet = word_alphabet
        self.emb_table = emb_table
        self.emb_dim = emb_dim  # the input word emb_dim
        self.word_pad_indx = self.word_alphabet.get_index('<PAD>')
        self.label_pad_index = self.label_alphabet.get_index('<PAD>')
        self.num_labels = self.label_alphabet.size()
        # self.crf = CRF(self.num_labels, batch_first=True)
        self.crf = CRF(self.num_labels,self.label_pad_index, self.label_alphabet.get_index('<BOS>'),self.label_alphabet.get_index('<EOS>'), 'cuda')

        # the input word embedding table
        self.word_embedding = nn.Embedding(self.word_alphabet.size(), self.emb_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.emb_table))
        # the position embedding
        self.aposition_module = APosEmb(self.emb_dim, self.args.dropout_rate, self.args.max_seq_len)
        # the transformer encoder
        self.transformer_encoder = AttentionModel(self.emb_dim, self.args.transformer_mod_dim,
                                                  self.args.transformer_ff_dim, self.args.transformer_num_heads,
                                                  self.args.transformer_num_layer, self.args.dropout_rate)
        self.transformer2unify = nn.Linear(self.args.transformer_mod_dim, self.args.unified_encoder_output_dim)

        # the bert embedding
        self.BertEncoder = GetBertFeature(self.args, self.label_alphabet)
        self.bert2unify = nn.Linear(self.args.bert_emb_dim, self.args.unified_encoder_output_dim)

        # the semantic augmentation embedding table
        # self.semantic_aug = SemanticAug(self.args.GazVec, self.word_alphabet, self.args.semantic_emb_dim)# 此处是外部增强，维度是200
        
        # 以下和上面的语义增强模块作用一致，只是来自于内部预训练的word2vec.
        self.semantic_aug = InternalAugmentation(self.args, self.word_alphabet, self.args.semantic_emb_dim)
        
        self.semantic_emb_table = nn.Embedding(self.word_alphabet.size(), self.args.semantic_emb_dim)
        self.semantic_emb_table.weight.data.copy_(torch.from_numpy(self.semantic_aug.semantic_emb_talbe()))
        self.semantic2unify = nn.Linear(self.args.semantic_emb_dim, self.args.unified_encoder_output_dim)

        self.gate1_w = nn.Parameter(torch.empty(2 * self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim), requires_grad=True)
        nn.init.xavier_normal_(self.gate1_w)
        self.gate1_linear = nn.Linear(self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim)

        self.gate2_w = nn.Parameter(torch.empty(2 * self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim), requires_grad=True)
        nn.init.xavier_normal_(self.gate2_w)
        self.gate2_linear = nn.Linear(self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim)

        self.unify2emission = nn.Linear(self.args.unified_encoder_output_dim, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.args.dropout_rate)
        """****************The above module belongs to encoder which is for getting feature from input sequence**************"""


    def encoder_output(self, wordids_tensor):# return the fused feature after encoder
        encoder_word_embedding = self.word_embedding(wordids_tensor)
        encoder_input = self.aposition_module(encoder_word_embedding)
        attention_mask = wordids_tensor.eq(self.word_pad_indx).unsqueeze(1).expand(-1,wordids_tensor.size(-1),-1)  # batch_size, seq_len, seq_len()
        transformer_output = self.transformer_encoder(encoder_input, attention_mask) # (batch_size, seq_len, model_dim)

        return transformer_output
        
    def fused_feature(self, word_sentences, wordids_sentences, label_sentences):
        transformer_unify_output = self.transformer2unify(self.encoder_output(wordids_sentences))
        bert_unify_output = self.bert2unify(self.BertEncoder.get_bert_feature(word_sentences, label_sentences, self.args.device))
        # semantic_unify_output = self.semantic2unify(self.semantic_emb_table(wordids_sentences))

        
        transformer_bert_fusion = transformer_unify_output + bert_unify_output
        transformer_bert_fusion = self.sigmoid(transformer_bert_fusion)
        return self.dropout(transformer_bert_fusion)


        """
        # CatFusion, 串接特征融合
        transformer_bert_cat = torch.cat([transformer_unify_output, bert_unify_output], dim=-1)
        transformer_bert_gate = transformer_bert_cat.matmul(self.gate1_w)
        transformer_bert_fusion = self.sigmoid(transformer_bert_gate)
        return self.dropout(transformer_bert_fusion)
        """
        

        """
        # GatedFusion，门特征融合
        transformer_bert_cat = torch.cat([transformer_unify_output, bert_unify_output], dim=-1)
        transformer_bert_gate = transformer_bert_cat.matmul(self.gate1_w)
        transformer_bert_gate = self.sigmoid(transformer_bert_gate)
        transformer_bert_ones = torch.ones(transformer_bert_gate.shape).to(self.args.device)
        transformer_bert_fusion = transformer_bert_gate.mul(transformer_unify_output)+\
                                  (transformer_bert_ones-transformer_bert_gate).mul(bert_unify_output)
        transformer_bert_fusion = self.gate1_linear(transformer_bert_fusion)

        return self.dropout(transformer_bert_fusion)
        """
        
        

        """
        # 以下代码，本次不适用
        twos_semantic_cat = torch.cat([transformer_bert_fusion, semantic_unify_output], dim=-1)
        twos_semantic_gate = twos_semantic_cat.matmul(self.gate2_w)
        twos_semantic_gate = self.sigmoid(twos_semantic_gate)
        twos_semantic_ones = torch.ones(twos_semantic_gate.shape).to(self.args.device)
        all_fused_feature = twos_semantic_gate.mul(transformer_bert_fusion)+\
                                  (twos_semantic_ones-twos_semantic_gate).mul(semantic_unify_output)
        all_fused_feature = self.gate2_linear(all_fused_feature)

        # 除了门，我们可以使用简单的加法运算
        # fused_feature = semantic_unify_output + transformer_bert_fusion
        # all_fused_feature = self.gate2_linear(fused_feature)

        return self.dropout(all_fused_feature)  # this should be the fused feature with unified dim
        """

    def ner_train(self, word_sentences, padded_wordids, label_sentences, padded_labelids):
        emission_feature = self.unify2emission(self.fused_feature(word_sentences, padded_wordids, label_sentences))
        valid_sent_mask = padded_labelids != self.label_pad_index
        # object_score = self.crf(emission_feature, padded_labelids, valid_sent_mask)  # pytorchcrf
        
        # object_score = F.cross_entropy(emission_feature.view(-1, self.num_labels), 
        #                                padded_labelids.view(-1), ignore_index=self.label_pad_index)
        
        object_score = self.crf(emission_feature, padded_labelids, valid_sent_mask)  # come from crf file

        return object_score

    def ner_test(self, word_sentences, padded_wordids, label_sentences, padded_labelids):
        emission_feature = self.unify2emission(self.fused_feature(word_sentences, padded_wordids, label_sentences))
        valid_sent_mask = padded_labelids != self.label_pad_index
        # return decode_tag(emission_feature, valid_sent_mask)
        # return self.crf.decode(emission_feature, valid_sent_mask)
        _, path = self.crf.viterbi_decode(emission_feature, valid_sent_mask)  # come from crf file
        return path
