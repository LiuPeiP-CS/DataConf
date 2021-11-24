#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 上午10:00
# @Author  : PeiP Liu
# @FileName: Bert_Feature.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig

import sys
sys.path.append("..")
from NER.BertModule.Bert_data_utils import DataProcessor, BertCRFData
from NER.BertModule.BertModel import BERT_CRF_NER


class GetBertFeature(nn.Module):
    def __init__(self, args, label_alphabet):
        super().__init__()
        self.args = args
        self.label_alphabet = label_alphabet
        self.tokenizer = BertTokenizer.from_pretrained(self.args.BertPath, do_lower_case=False)
        config = BertConfig.from_pretrained(self.args.BertPath, output_hidden_states=True)
        bert_model = BertModel.from_pretrained(self.args.BertPath, config=config)
        self.model = BERT_CRF_NER(bert_model, self.label_alphabet, batch_size=self.args.batch_size, max_seq_len=self.args.max_seq_len,
                             device=self.args.device)
        checkpoint = torch.load(self.args.Bert_saved_path + 'bert_crf_ner.checkpoint.pt', map_location='cpu')
        # parser the model params
        pretrained_model_dict = checkpoint['model_state']
        # get the model param names
        model_state_dict = self.model.state_dict()
        # get the params interacting between model_state_dict and pretrained_model_dict
        selected_model_state = {k: v for k, v in pretrained_model_dict.items() if k in model_state_dict}
        model_state_dict.update(selected_model_state)
        # load the params into model
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.args.device)  # gpu
        self.model.eval()

    def get_bert_feature(self, batch_sents, batch_labels, device):
        batch_dp = DataProcessor(batch_sents, batch_labels, self.tokenizer, self.args.max_seq_len, self.label_alphabet)
        batch_bert_data = BertCRFData(batch_dp.get_features())
        batch_dataloader = DataLoader(dataset=batch_bert_data, batch_size=self.args.batch_size, shuffle=False,
                                      collate_fn=BertCRFData.seq_tensor)  # return the iterator object of batch_data

        with torch.no_grad():
            for i_batch_data in batch_dataloader:  # in fact, there is only one batch_data
                batch_data = tuple(t.to(device) for t in i_batch_data)  # gpu
                input_ids, input_mask, seg_ids, label_ids, label_mask = batch_data
                bert_feature = self.model.get_bert_features(input_ids, seg_ids, input_mask)
                return bert_feature  # gpu


