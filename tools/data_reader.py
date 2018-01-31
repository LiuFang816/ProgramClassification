# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import collections
import numpy as np
import tensorflow.contrib.keras as kr

#seq length 为num_step-1
def batch_iter(x,y,batch_size):
    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        if end_id-start_id<batch_size:
            break
        yield x[start_id:end_id],y[start_id:end_id]

def read_data(file,seq_len):
    with open(file,'r',encoding='utf-8',errors='ignore') as f:
        contents,labels=[],[]
        lines=f.readlines()
        for line in lines:
            try:
                label,content=line.split('\t')
                content=content.replace('\n','')
                content=content.split(' ')
                if len(content)>seq_len:
                    continue
                labels.append(int(label)-1)#label从0开始
                contents.append(content)
            except:
                pass
        return contents,labels
# contents,labels=read_data('../data/pro_cla/val.txt',500)
# print(contents[0])
# print(labels[0])

def file_to_id(word_to_id,data,num_steps):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=word_to_id[data[i][j]] if data[i][j] in word_to_id else word_to_id['UNK']
        if num_steps:
            for _ in range(num_steps - len(data[i])):
                data[i].append(word_to_id['PAD'])
            # print(len(data[i]))
    return data


def raw_data(data_path=None, word_to_id=None, num_steps=None,num_classes=None):
    train_path = os.path.join(data_path,"pro_cla/train.txt")
    val_path=os.path.join(data_path,'pro_cla/val.txt')
    test_path = os.path.join(data_path, "pro_cla/test.txt")

    train_data,train_label=read_data(train_path,num_steps)
    test_data,test_label=read_data(test_path,num_steps)
    val_data,val_label=read_data(val_path,num_steps)
    train_data=np.asarray(file_to_id(word_to_id,train_data,num_steps))
    val_data=np.asarray(file_to_id(word_to_id,val_data,num_steps))
    test_data=np.asarray(file_to_id(word_to_id,test_data,num_steps))
    train_label=kr.utils.to_categorical(train_label,num_classes)
    val_label=kr.utils.to_categorical(val_label,num_classes)
    test_label=kr.utils.to_categorical(test_label,num_classes)
    left_id = word_to_id['{']
    right_id = word_to_id['}']
    PAD_ID = word_to_id['PAD']
    return train_data,train_label,val_data,val_label,test_data,test_label,left_id, right_id, PAD_ID

# def get_pair(data):
#     labels=data[:,1:]
#     inputs=data[:,:-1]
#     return inputs,labels


def _build_vocab(filename,vocab_size,seq_len):
    data,label = read_data(filename,seq_len)
    words=[]
    for content in data:
        words.extend(content)
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, values = list(zip(*count_pairs))
    words = words[0:vocab_size-2]
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['UNK'] = len(word_to_id)
    word_to_id['PAD'] = len(word_to_id)
    return word_to_id


def reverseDic(curDic):
    newmaplist = {}
    for key, value in curDic.items():
        newmaplist[value] = key
    return newmaplist


