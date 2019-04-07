from glob import glob
import os
import pickle
import re

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics.scorer import make_scorer

import numpy as np


CLASSES = ['FNSZ', 'GNSZ', 'SPSZ', 'CPSZ', 'ABSZ', 'TNSZ', 'TCSZ']


def evaluate(model, X, y, file_offsets, cv=None, **kwargs):
    if cv is None:
        cv = split_cv5(file_offsets)
    cv_scores = cross_val_score(
        model, X, y,
        cv=cv,
        verbose=True,
        scoring=weighted_f1,
        **kwargs,
    )
    print('cv_scores:', cv_scores)
    print('mean:', np.mean(cv_scores))
    

def get_Xy(dirname):
    
    X, y = [], []
    file_offsets = {}
    
    saved_dir = os.getcwd()
    os.chdir(dirname)
    
    for file_name in glob('*.pkl'):
        d = pickle.load(open(file_name, 'rb'))
        if d.seizure_type == "MYSZ":
            continue
        X.append(d.data.astype('float32'))
        file_offsets[file_name] = (len(y), len(y) + d.data.shape[0])
        y.extend([CLASSES.index(d.seizure_type)] * d.data.shape[0])
    
    os.chdir(saved_dir)
    
    X = np.concatenate(X)
    
    return X, np.array(y, dtype='uint8'), file_offsets


def parse_cv5():
    folds = open('data/cv-5.txt').read().split(":")[1:]
    return [re.findall(r'seiz_[0-9]+.pkl', i) for i in folds]


def split_cv5(file_offsets, sample=None):
    folds = parse_cv5()
    for n in range(5):
        train = list(folds)
        test = train.pop(n)
        train = [j for i in train for j in i]
        train_idx = []
        for i in train:
            train_idx.extend(range(*file_offsets[i]))
        test_idx = []
        for i in test:
            test_idx.extend(range(*file_offsets[i]))
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        if sample is not None:
            train_idx = train_idx[:int(len(train_idx) * sample)]
        yield train_idx, test_idx
        

def kfold_split(file_offsets):
    files = list(file_offsets.keys())
    for train, test in KFold(5).split(files):
        train_idx = []
        for i in train:
            train_idx.extend(range(*file_offsets[files[i]]))
        test_idx = []
        for i in test:
            test_idx.extend(range(*file_offsets[files[i]]))
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        yield train_idx, test_idx


weighted_f1 = make_scorer(f1_score, average="weighted")