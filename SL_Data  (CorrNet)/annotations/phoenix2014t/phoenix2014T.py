import glob
import os
import pandas as pd
from collections import Counter

import numpy as np

annotations_dir = 'annotations/phoenix2014t/'

train_corpus = pd.read_csv(os.path.join(annotations_dir, "PHOENIX-2014-T.train.corpus.csv"), sep='|', header=0,
                           index_col='name')
train_complex_corpus = pd.read_csv(os.path.join(annotations_dir, "PHOENIX-2014-T.train-complex-annotation.corpus.csv"),
                                   sep='|', header=0, index_col='name')
dev_corpus = pd.read_csv(os.path.join(annotations_dir, "PHOENIX-2014-T.dev.corpus.csv"), sep='|', header=0,
                         index_col='name')
test_corpus = pd.read_csv(os.path.join(annotations_dir, "PHOENIX-2014-T.test.corpus.csv"), sep='|', header=0,
                          index_col='name')
all_corpus = pd.concat([train_corpus, dev_corpus, test_corpus])


train_list = []
train_annotations = train_corpus.orth.tolist()
train_word_counts = Counter(sum([a.split() for a in train_annotations], []))
for k,v in dict(train_word_counts).items():
    train_list.append(k)
print(len(train_word_counts))
print(len(train_list))


dev_zero = []
dev_list = []
dev_annotations = dev_corpus.orth.tolist()
dev_word_counts = Counter(sum([a.split() for a in dev_annotations], []))
for k,v in dict(dev_word_counts).items():
    dev_list.append(k)
print(len(dev_word_counts))
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
test_annotations = test_corpus.orth.tolist()
test_word_counts = Counter(sum([a.split() for a in test_annotations], []))
for k,v in dict(test_word_counts).items():
    test_list.append(k)
print(len(test_word_counts))
print(len(test_list))
for i in test_list:
    if i not in train_list:
        test_zero.append(i)
print(test_zero)
print(len(test_zero))

for k,v in dict(test_word_counts).items():
    if k in test_zero:
        print(k,v)

all_annotations = all_corpus.orth.tolist()
all_word_counts = Counter(sum([a.split() for a in all_annotations], []))
print(len(all_word_counts))

print(len(train_word_counts)-len(all_word_counts))