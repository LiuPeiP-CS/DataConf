#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 下午5:12
# @Author  : PeiP Liu
# @FileName: Bert_retrain.py
# @Software: PyCharm
# rf https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py
import os
import torch
import time
import datetime
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

import sys
sys.path.append("../..")  # 访问上上级目录
from NER.PreData import PreData
from config import BasicArgs
from data_utils import *
from NER.BertModule.BertModel import BERT_CRF_NER
from NER.BertModule.Bert_data_utils import DataProcessor, BertCRFData
from NER.BertModule.BertEval import bert_evaluate as evaluate

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def warmup_linear(x, warmup = 0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


if __name__ == "__main__":
    Bert_output_dir = BasicArgs.Bert_saved_path
    device = BasicArgs.device
    batch_size = BasicArgs.batch_size

    train_sents = BasicArgs.train_sents
    train_sentlabels = BasicArgs.train_sentlabels
    test_sents = BasicArgs.test_sents
    test_sentlabels = BasicArgs.test_sentlabels

    pre_data = PreData()
    pre_data.build_alphabet(train_sents, 'train', train_sentlabels)
    pre_data.build_alphabet(test_sents, 'test')

    label_alphabet = pre_data.label_alphabet
    for iter_label, iter_id in label_alphabet.items():
        print({iter_label:iter_id})

    load_checkpoint = BasicArgs.reload_checkpoint
    max_seq_len = BasicArgs.max_seq_len
    learning_rate = BasicArgs.learning_rate
    weight_decay_finetune = BasicArgs.weight_decay_finetune
    lr_crf_fc = BasicArgs.lr_crf_fc
    weight_decay_crf_fc = BasicArgs.weight_decay_crf_fc
    total_train_epoch = BasicArgs.total_epoch
    warmup_proportion = BasicArgs.warmup_proportion
    gradient_accumulation_steps = BasicArgs.gradient_accumulation_steps

    tokenizer = BertTokenizer.from_pretrained(BasicArgs.BertPath, do_lower_case=False)
    config = BertConfig.from_pretrained(BasicArgs.BertPath, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(BasicArgs.BertPath, config=config)

    model = BERT_CRF_NER(bert_model, label_alphabet, batch_size=batch_size, max_seq_len=max_seq_len, device=device)

    if load_checkpoint and os.path.exists(Bert_output_dir + 'bert_crf_ner.checkpoint.pt'):
        checkpoint = torch.load(Bert_output_dir + 'bert_crf_ner.checkpoint.pt', map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        """
        valid_acc_pre = checkpoint['valid_acc']
        valid_f1_pre = checkpoint['valid_f1']
        """
        pretrained_model_state = checkpoint['model_state']
        cur_model_state = model.state_dict()
        selected_pretrained_model_state = {k: v for k, v in pretrained_model_state.items() if k in cur_model_state}
        cur_model_state.update(selected_pretrained_model_state)
        model.load_state_dict(cur_model_state)
        """
        print("Load the pretrained model, epoch:", checkpoint['epoch'], 'valid_acc:', checkpoint['valid_acc'],
              'valid_f1:', checkpoint['valid_f1'])
        """
    else:
        start_epoch = 0
        """
        valid_acc_pre = 0
        valid_f1_pre = 0
        """
        if not os.path.exists(Bert_output_dir + 'bert_crf_ner.checkpoint.pt'):
            os.mknod(Bert_output_dir + 'bert_crf_ner.checkpoint.pt')

    model.to(device=device)
    params_list = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params_list if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in params_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # the total_train_steps referring to the loss computing
    total_train_steps = int(len(train_sents)/batch_size/gradient_accumulation_steps)*total_train_epoch
    warmup_steps = int(warmup_proportion*total_train_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    # i_th step in all steps we have planed(from 0). And here, the ideal batch_size we want is batch_size*grad_acc_steps
    global_step_th = int(len(train_sents)/batch_size/gradient_accumulation_steps * start_epoch)

    train_dp = DataProcessor(train_sents, train_sentlabels, tokenizer, max_seq_len, label_alphabet)
    train_bert_crf_data = BertCRFData(train_dp.get_features())

    """
    valid_dp = DataProcessor(BasicArgs.valid_sents, BasicArgs.valid_sentlabels, tokenizer, max_seq_len, label_alphabet)
    valid_bert_crf_data = BertCRFData(valid_dp.get_features())
    """

    train_average_loss = []
    """
    valid_acc_score = []
    valid_f1_score = []
    """
    for epoch in trange(start_epoch, total_train_epoch, desc='Epoch'):
        train_loss = 0
        train_start = time.time()
        model.train()
        # clear the gradient
        model.zero_grad()
        train_dataloader = DataLoader(dataset=train_bert_crf_data, batch_size=batch_size, shuffle=True, collate_fn=BertCRFData.seq_tensor)
        batch_start = time.time()
        for step, batch in enumerate(train_dataloader):
            # we show the time cost ten by ten batches
            if step % 10 == 0 and step != 0:
                print('Ten batches cost time : {}'.format(time_format(time.time()-batch_start)))
                batch_start = time.time()

            # input and output
            batch_data = tuple(cat_data.to(device) for cat_data in batch)
            train_input_ids, train_atten_mask, train_seg_ids, true_labels_ids, true_label_mask = batch_data
            object_loss = model.neg_log_likehood(train_input_ids, train_atten_mask, train_seg_ids,
                                                 true_labels_ids,true_label_mask)
            # loss regularization
            if gradient_accumulation_steps > 1:
                object_loss = object_loss / gradient_accumulation_steps

            # Implementation of backpropagation
            object_loss.backward()
            train_loss = train_loss + object_loss.cpu().item()
            if (step+1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
                global_step_th = global_step_th + 1
            print("Epoch:{}-{}/{}, Object-loss:{}".format(epoch, step, len(train_dataloader), object_loss))
        ave_loss = train_loss / len(train_dataloader)
        train_average_loss.append(ave_loss)

        print("Epoch: {} is completed, the average loss is: {}, spend: {}".format(epoch, ave_loss, time_format(time.time()-train_start)))

        
        torch.save({'model_state': model.state_dict()},os.path.join(Bert_output_dir+ 'bert_crf_ner.checkpoint.pt'))
        

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    # Next, we will test the model on stranger dataset
    # load the pretrained model
    checkpoint = torch.load(Bert_output_dir + 'bert_crf_ner.checkpoint.pt', map_location='cpu')
    pretrained_model_dict = checkpoint['model_state']
    # get the model param names
    model_state_dict = model.state_dict()
    # get the params interacting between model_state_dict and pretrained_model_dict
    selected_model_state = {k: v for k, v in pretrained_model_dict.items() if k in model_state_dict}
    model_state_dict.update(selected_model_state)
    # load the params into model
    model.load_state_dict(model_state_dict)
    # show the details about loaded model
    """
    print('Loaded the pretrained NER_BERT_CRF model, epoch:', checkpoint['epoch'],
          'valid_acc:', checkpoint['valid_acc'], 'valid_f1:',checkpoint['valid_f1'])
    """
    model.to(device)

    test_dp = DataProcessor(test_sents, test_sentlabels, tokenizer, max_seq_len, label_alphabet)
    test_bert_crf_data = BertCRFData(test_dp.get_features())
    """
    test_dataloader = DataLoader(dataset=test_bert_crf_data, batch_size=batch_size, shuffle=True,
                                  collate_fn=BertCRFData.seq_tensor)
    test_acc, test_f1 = evaluate(model, test_dataloader, epoch, device, 'Test')
    """
    model.eval()
    result_data = []

    with torch.no_grad():
        demon_dataloader = DataLoader(dataset=test_bert_crf_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=BertCRFData.seq_tensor)
        for demon_batch in demon_dataloader:
            demon_batch_data = tuple(t.to(device) for t in demon_batch)
            demon_input_ids, demon_atten_mask, demon_seg_ids, demon_label_ids, demon_label_mask = demon_batch_data
            demon_predicted_labels_seq_ids = model(demon_input_ids, demon_atten_mask, demon_seg_ids, demon_label_mask)

            for each_sent_pred_ids in demon_predicted_labels_seq_ids:
                sent_pred_tag = [label_alphabet.get_instance(label_id) for label_id in each_sent_pred_ids]
                print(sent_pred_tag)
                result_data.append(' '.join(sent_pred_tag))
    submit=pd.DataFrame(result_data)  # 写入的每行，都是最外层列表的一个元素
    submit.to_csv(BasicArgs.result,index=False)
