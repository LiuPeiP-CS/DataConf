#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 下午7:25
# @Author  : PeiP Liu
# @FileName: config.py
# @Software: PyCharm
import os
import torch
import numpy as np
import sys
sys.path.append('..')
from data_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # bert——1， transformer——0

class BasicArgs:
    BertPath = '/root/bigdata/liupei/DataConf/NER/BERT_Chinese/'
    GazVec = '/root/bigdata/liupei/DataConf/NER/Pre_Emb/Tencent_AILab_ChineseEmbedding.txt'
    WordVec = '/root/bigdata/liupei/DataConf/NER/Pre_Emb/gigaword_chn.all.a2b.uni.ite50.vec'  # 预训练的中文字符向量
    Bert_saved_path = '/root/bigdata/liupei/DataConf/NER/Bert_output/'  # 预训练bert模型的存储位置
    Model_output_dir = '/root/bigdata/liupei/DataConf/NER/FusedModel_output/'  # 模型的存储位置
    train_file = '/root/bigdata/liupei/DataConf/train_data_public.csv'
    test_file = '/root/bigdata/liupei/DataConf/test_public.csv'
    word2vec_model_path = '/root/bigdata/liupei/DataConf/NER/Word2Vec/word2vec_embedding.bin'  # word2vec模型的存储位置
    word2vec = '/root/bigdata/liupei/DataConf/NER/Word2Vec/word2vec_embedding.npy'  # 模型训练时要保证和semantic_emb_dim相同。作用等同于GazVec
    result = '/root/bigdata/liupei/DataConf/result.csv'

    train_sents, train_sentlabels = read_csv(train_file, 'train')
    test_sents, test_sentlabels = read_csv(test_file, 'test')

    total_epoch = 200  
    batch_size = 64
    max_seq_len = 64
    learning_rate = 5e-5  # bert下学习率设置为5e-5, 主模型下设置学习率为5e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    weight_decay_finetune = 1e-5
    lr_crf_fc = 1e-5
    weight_decay_crf_fc = 1e-5
    warmup_proportion = 0.002

    word_emb_dim = 50   # gigaword_chn ite50
    bert_emb_dim = 768
    semantic_emb_dim = 200  # 200 for GazVec, but if we use the ctb vector, the emb_dim should be 50. The same with word2vec

    transformer_num_layer = 4
    transformer_num_heads = 8
    transformer_mod_dim = 128
    transformer_ff_dim = 256

    unified_encoder_output_dim = 128

    semantic_num_sim = 3
    sim_num = 3
    dropout_rate = 0.5


    lr = 5e-4
    weight_decay = 0.001
    min_lr = 5e-5
    lr_decay_factor = 0.5

    gradient_accumulation_steps = 40

    reload_checkpoint = False  # whether the first training ?
