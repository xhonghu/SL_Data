import glob
import os
import pickle
import pandas as pd
from collections import Counter

import numpy as np

annotations_dir = 'annotations/csl-daily/'

with open(os.path.join(annotations_dir, 'csl2020ct_v2.pkl'), 'rb') as f:
    data = pickle.load(f)
info = data['info']
gloss_map = data['gloss_map']
char_map = data['char_map']
word_map = data['word_map']
postag_map = data['postag_map']

info = pd.DataFrame(info)

split = pd.read_csv(os.path.join(annotations_dir, 'split_1.txt'), sep='|', header=0)

train_list = split[split['split'] == 'train'].name.tolist()
dev_list = split[split['split'] == 'dev'].name.tolist()
test_list = split[split['split'] == 'test'].name.tolist()

train_info = info[info['name'].isin(train_list)]
dev_info = info[info['name'].isin(dev_list)]
test_info = info[info['name'].isin(test_list)]

train_list_ = []
train_word_counts = Counter(sum(train_info.label_gloss.tolist(), []))
for k,v in dict(train_word_counts).items():
    train_list_.append(k)
print(len(train_word_counts))
print(len(train_list_))

dev_list_ = []
dev_word_counts = Counter(sum(dev_info.label_gloss.tolist(), []))
for k,v in dict(dev_word_counts).items():
    dev_list_.append(k)
print(len(dev_word_counts))
print(len(dev_list_))

test_list_ = []
test_word_counts = Counter(sum(test_info.label_gloss.tolist(), []))
for k,v in dict(test_word_counts).items():
    test_list_.append(k)
print(len(test_word_counts))
print(len(test_list_))
