#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 下午3:39
# @Author  : PeiP Liu
# @FileName: main.py
# @Software: PyCharm


import os
import torch
import pickle
import time
from tqdm import trange
import pandas as pd
import torch.optim as optim
from config import BasicArgs
from data_utils import time_format, maxlen_padding, gen_batch_data, batch_decomposition
from NER.PreData import PreData
from NER.model import NER_model
from NER.BertModule.Bert_Feature import GetBertFeature
from NER.BertModule.BertEval import lc_cal_f1, lc_cal_acc
import random

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    if not os.path.exists(BasicArgs.Model_output_dir):
        os.makedirs(BasicArgs.Model_output_dir)

    tagged_sents = BasicArgs.train_sents
    tagged_sentlabels = BasicArgs.train_sentlabels
    test_sents = BasicArgs.test_sents
    test_sentlabels = BasicArgs.test_sentlabels

    PreData = PreData()
    PreData.build_alphabet(tagged_sents, 'train', tagged_sentlabels)
    PreData.build_alphabet(test_sents, 'test', test_sentlabels)

    word_alphabet = PreData.word_alphabet
    label_alphabet = PreData.label_alphabet
    emb_table, emb_dim = PreData.build_embedding_table(BasicArgs.WordVec)

    tagged_ids = PreData.text2ids(sentences=tagged_sents, sentences_labels=tagged_sentlabels)
    test_ids = PreData.text2ids(sentences=test_sents, sentences_labels=test_sentlabels)

    model = NER_model(BasicArgs, label_alphabet, word_alphabet, emb_table, emb_dim).to(BasicArgs.device)

    optimizer = optim.Adam(model.parameters(), lr=BasicArgs.lr, weight_decay=BasicArgs.weight_decay)
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=BasicArgs.lr_decay_factor,
                                                    verbose=True, patience=3, min_lr=BasicArgs.min_lr)

    tagged_piece_num = len(tagged_ids) // 10  # 将有标签数据分成10份，选取90%进行训练train，10%进行验证valid

    train_num = tagged_piece_num*9
    valid_num = len(tagged_sents) - train_num

    print("*****************************Starting Training*****************************")
    num_batch = train_num // BasicArgs.batch_size if train_num % BasicArgs.batch_size == 0 else train_num // BasicArgs.batch_size + 1

    valid_f1_prev = 0

    train_ave_loss = []
    valid_acc_score = []
    valid_f1_score = []
    valid_loss_score = []

    train_ids = tagged_ids
    
    for epoch in trange(BasicArgs.total_epoch, desc='Epoch'):
        """
        train_ids = random.sample(tagged_ids, train_num)  # 每次将带标签数据随机选90%做训练
        valid_ids = []
        for instance_ids in tagged_ids:
            if instance_ids not in train_ids:  # 剩下10%做验证
                valid_ids.append(instance_ids)
        # list(set(tagged_ids) - set(train_ids))
        """

        train_loss = 0

        # compute the training time, and initiate the time
        train_start = time.time()
        batch_start = time.time()

        # setting the training mode and clear the grad
        model.train()
        model.zero_grad()
        for i_batch, batch_train_ids in enumerate(gen_batch_data(train_ids, BasicArgs.batch_size)):
            word_sentences, wordid_sentences, label_sentences, labelid_sentences = batch_decomposition(batch_train_ids)
            padded_wordids, padded_labelids = maxlen_padding(wordid_sentences, labelid_sentences, BasicArgs.max_seq_len,
                                                             word_padding_value=word_alphabet.get_index('<PAD>'), label_padding_value=label_alphabet.get_index('<PAD>'))
            padded_wordids = torch.tensor(padded_wordids, dtype=torch.long).to(BasicArgs.device)  # the padded data to gpu
            padded_labelids = torch.tensor(padded_labelids, dtype=torch.long).to(BasicArgs.device)  # the padded data to gpu

            batch_train_loss = model.ner_train(word_sentences, padded_wordids, label_sentences, padded_labelids)

            # backpropagation and clear the grad
            batch_train_loss.backward()
            train_loss = train_loss + batch_train_loss.cpu().item()
            optimizer.step()
            optimizer.zero_grad()

            # compute the training time
            if i_batch % 40 == 0 and i_batch != 0:
                print('Ten batches cost time : {}'.format(time.time() - batch_start))
                # print the training infor
                print("Epoch:{}-{}/{}, Loss:{}".format(epoch, i_batch, num_batch, batch_train_loss))
                batch_start = time.time()

        ave_loss = train_loss / num_batch  # the average loss of each epoch
        train_ave_loss.append(ave_loss)
        print("Epoch: {} is completed, the average train_loss is: {}, spend: {}".format(epoch, ave_loss,
                                                                                        time.time() - train_start))
        torch.save({'epoch': epoch, 'model_state': model.state_dict()}, os.path.join(BasicArgs.Model_output_dir, 'Fusion.checkpoint.pt'))

        lr_decay.step(ave_loss)

        """
        print("********************Let us begin the validation of epoch {}***************************".format(epoch))

        # evaluate the model
        model.eval()
        valid_true, valid_pre = [], []
        valid_acml_loss = 0
        for j_batch, batch_valid_ids in enumerate(gen_batch_data(valid_ids, BasicArgs.batch_size)):
            valid_word_sentences, valid_wordid_sentences, valid_label_sentences, valid_labelid_sentences = batch_decomposition(batch_valid_ids)
            valid_padded_wordids, valid_padded_labelids = maxlen_padding(valid_wordid_sentences, valid_labelid_sentences, BasicArgs.max_seq_len,
                                                                         word_padding_value=word_alphabet.get_index('<PAD>'), label_padding_value=label_alphabet.get_index('<PAD>'))
            valid_padded_wordids = torch.tensor(valid_padded_wordids, dtype=torch.long).to(BasicArgs.device)  # the padded data to gpu
            valid_padded_labelids = torch.tensor(valid_padded_labelids, dtype=torch.long).to(BasicArgs.device)  # the padded data to gpu

            batch_valid_loss = model.ner_train(valid_word_sentences, valid_padded_wordids, valid_label_sentences, valid_padded_labelids)

            # input and output
            batch_valid_preds = model.ner_test(valid_word_sentences, valid_padded_wordids, valid_label_sentences, valid_padded_labelids)

            batch_valid_labels_flatten = [each_label for each_sent in valid_labelid_sentences for each_label in each_sent]
            batch_valid_preds_flatten = [each_pred_label for each_pre_sent in batch_valid_preds for each_pred_label in each_pre_sent]

            valid_true.extend(batch_valid_labels_flatten)  # array is also well
            valid_pre.extend(batch_valid_labels_flatten)

            valid_acml_loss = valid_acml_loss + batch_valid_loss.detach().cpu().item() * len(valid_word_sentences)

        valid_avg_loss = valid_acml_loss / len(valid_ids)
        valid_loss_score.append(valid_avg_loss)
        each_epoch_valid_f1 = lc_cal_f1(valid_true, valid_pre)
        valid_f1_score.append(each_epoch_valid_f1)
        each_epoch_valid_acc = lc_cal_acc(true_tags=valid_true, pred_tags=valid_pre)
        valid_acc_score.append(each_epoch_valid_acc)
        print('Validation: Epoch-{}, Val_loss-{}, Val_acc-{}, Val_f1-{}'.format(epoch, valid_avg_loss,
                                                                                each_epoch_valid_acc,
                                                                                each_epoch_valid_f1))

        if each_epoch_valid_f1 > valid_f1_prev:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': each_epoch_valid_acc,
                        'valid_f1': each_epoch_valid_f1},
                       os.path.join(BasicArgs.Model_output_dir, 'Fusion.checkpoint.pt'))
            valid_f1_prev = each_epoch_valid_f1

        lr_decay.step(valid_avg_loss)  # when there is no change about loss within patience step , lr will decay
        
        """

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    test_checkpoint = torch.load(os.path.join(BasicArgs.Model_output_dir, 'Fusion.checkpoint.pt'),
                                 map_location='cpu')

    # test_valid_f1 = test_checkpoint['valid_f1']
    # test_valid_acc = test_checkpoint['valid_acc']
    trained_model_dict = test_checkpoint['model_state']
    # get the model param names
    test_model_state_dict = model.state_dict()
    # get the params interacting between model_state_dict and pretrained_model_dict
    selected_model_state = {k: v for k, v in trained_model_dict.items() if k in test_model_state_dict}
    test_model_state_dict.update(selected_model_state)
    # load the params into model
    model.load_state_dict(test_model_state_dict)
    # show the details about loaded model
    # print('Load the best trained model, epoch:', test_checkpoint['epoch'], 'valid_acc:', test_checkpoint['valid_acc'],
    #       'valid_f1:', test_checkpoint['valid_f1'])
    model.to(BasicArgs.device)
    # evaluate the model
    model.eval()
    # test_true, test_pre = [], []

    result_data=[]

    for k_batch, batch_test_ids in enumerate(gen_batch_data(test_ids, BasicArgs.batch_size, shuffle=False)):
        batch_test_word_sentences, batch_test_wordid_sentences, batch_test_label_sentences, \
                                batch_test_labelid_sentences = batch_decomposition(batch_test_ids)
        batch_test_padded_wordids, batch_test_padded_labelids = maxlen_padding(batch_test_wordid_sentences, batch_test_labelid_sentences,
                                BasicArgs.max_seq_len, word_padding_value=word_alphabet.get_index('<PAD>'), label_padding_value=label_alphabet.get_index('<PAD>'))
        batch_test_padded_wordids = torch.tensor(batch_test_padded_wordids, dtype=torch.long).to(BasicArgs.device)  # the padded data to gpu
        batch_test_padded_labelids = torch.tensor(batch_test_padded_labelids, dtype=torch.long).to(BasicArgs.device)  # the padded data to gpu

        batch_test_preds = model.ner_test(batch_test_word_sentences, batch_test_padded_wordids,
                                           batch_test_label_sentences, batch_test_padded_labelids)

        for iter, each_pred_sent in enumerate(batch_test_preds):
            sent_pred_tag = [label_alphabet.get_instance(pred_tag_id) for pred_tag_id in each_pred_sent]
            print(batch_test_word_sentences[iter])
            print(sent_pred_tag)

            result_data.append(' '.join(sent_pred_tag))
    submit=pd.DataFrame(result_data)  # 写入的每行，都是最外层列表的一个元素
    submit.to_csv(BasicArgs.result,index=False)
