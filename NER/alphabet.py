#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/6 下午5:45
# @Author  : PeiP Liu
# @FileName: alphabet.py
# @Software: PyCharm
import os
import json
import torch


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.__name = name
        self.UNKWOWN = '<UNK>'
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        self.default_index = 0
        self.add('<PAD>')
        if not self.label:  # 构造词字典时
            self.add(self.UNKWOWN)
        else:  # 构造标签字典时
            self.add('<BOS>')
            self.add('<EOS>')

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.default_index = 0
        # self.next_index = 1

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = len(self.instance2index)

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                self.add(instance)
                return self.instance2index[instance]
            else:
                return self.instance2index[self.UNKWOWN]

    def get_instance(self, index):
        # if index == 0:
        #     return None
        try:
            return self.instances[index]  # index from 0
        except IndexError:
            print('We can not find the element at this index :' % index)
            return self.instance2index[0]

    def size(self):
        return len(self.instance2index)

    def items(self):
        return self.instance2index.items()

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data['instances']
        self.instance2index = data['instance2index']

    def save(self, output_dictionary, name=None):
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_dictionary, saving_name+'.json'), 'w'))
        except Exception as e:
            print('Exception: Alphabet is not saved:' % repr(e))

    def load(self, input_dictionary, name=None):
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_dictionary, loading_name+'.json'), 'r')))