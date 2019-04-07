#!/usr/bin/env python
# coding: utf-8

"""RNK Team Kazan Neurohackaton Task #2 EEG Classifier Solution.

Using the pp_2 dataset.

(c) Andrew Grigorev <andrew@ei-grad.ru>
(c) Valeria Vorobyova <vvalerika@gmail.com>

2019, April 7
"""

from glob import glob
import os
import pandas as pd
import pickle

from utils import get_Xy, evaluate, CLASSES
from nmslibclf import NMSLIBClassifier


# Loading the train data
X, y, file_offsets = get_Xy('data/pp_2')


###############################################
# CALCULATING THE CV-5 CROSS VALIDATION SCORE #
###############################################

evaluate(NMSLIBClassifier(space='l1', index_params={'efConstruction': 500}), X, y, file_offsets)

# Output:
#
# cv_scores: [0.77876278 0.80540095 0.85690264 0.89508698 0.88094148]
# mean: 0.8434189643669201
# CPU times: user 2h 53min 52s, sys: 23.4 s, total: 2h 54min 16s
# Wall time: 16min 57s


#######################################
# PRODUCING THE TEST DATASET SOLUTION #
#######################################

clf = NMSLIBClassifier(space='l1', index_params={'efConstruction': 500}).fit(X, y)

os.chdir('../../test/pp_2')

files = []
results = []

for file_name in glob('*.pkl'):
    X = pickle.load(open(file_name, 'rb'))
    y_pred = clf.predict(X)
    files.append(file_name)
    results.append(int(pd.Series(y_pred).mode()))

df = pd.DataFrame({'id': files, 'label_id': results})

df['label'] = df.label_id.map(CLASSES.__getitem__)
df['id'] = df['id'].map(lambda x: x.split('.')[0])
df['num'] = df['id'].map(lambda x: int(x[4:]))

# Write the CSV result
df = df.sort_values('num')[['id', 'label']]
df.to_csv('RNK-submit.csv', header=True, index=False)
# MD5: 3e9308d76de95a47d04c98d65c7ffed9


########################################
# SAVING THE MODEL FOR THE WEB SERVICE #
########################################

clf.index.saveIndex('index.dat')
# MD5: f61957e21ed4eddbd2d51e37c4604376
