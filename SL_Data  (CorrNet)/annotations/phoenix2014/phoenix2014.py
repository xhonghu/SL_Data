import glob
import os
import pandas as pd
from collections import Counter

import numpy as np


annotations_dir ="annotations/phoenix2014"


train_corpus = pd.read_csv(os.path.join(annotations_dir, "train.corpus.csv"), sep='|', header=0, index_col='id')
dev_corpus = pd.read_csv(os.path.join(annotations_dir, "dev.corpus.csv"), sep='|', header=0, index_col='id')
test_corpus = pd.read_csv(os.path.join(annotations_dir, "test.corpus.csv"), sep='|', header=0, index_col='id')
all_corpus = pd.concat([train_corpus, dev_corpus, test_corpus])

print(all_corpus)


train_list = []
train_annotations = train_corpus.annotation.tolist()
train_word_counts = Counter(sum([a.split() for a in train_annotations], []))
for k,v in dict(train_word_counts).items():
    train_list.append(k)
print(len(train_list))


dev_zero = []
dev_list = []
dev_annotations = dev_corpus.annotation.tolist()
dev_word_counts = Counter(sum([a.split() for a in dev_annotations], []))
for k,v in dict(dev_word_counts).items():
    dev_list.append(k)
print(len(dev_list))
for i in dev_list:
    if i not in train_list:
        dev_zero.append(i)
print(dev_zero)
print(len(dev_zero))
for k,v in dict(dev_word_counts).items():
    if k in dev_zero:
        print(k,v)

test_zero = []
test_list = []
test_annotations = test_corpus.annotation.tolist()
test_word_counts = Counter(sum([a.split() for a in test_annotations], []))
for k,v in dict(test_word_counts).items():
    test_list.append(k)
print(len(test_list))
for i in test_list:
    if i not in train_list:
        test_zero.append(i)
print(test_zero)
print(len(test_zero))
for k,v in dict(test_word_counts).items():
    if k in test_zero:
        print(k,v)


all_annotations = all_corpus.annotation.tolist()
all_word_counts = Counter(sum([a.split() for a in all_annotations], []))
print(len(all_word_counts))

print(len(train_word_counts)-len(all_word_counts))