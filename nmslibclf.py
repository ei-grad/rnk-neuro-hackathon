from collections import defaultdict

import nmslib

from sklearn.base import BaseEstimator, ClassifierMixin


def weighted_class(ids, dists):
    d = defaultdict(list)
    for id_, dist in zip(ids, dists):
        d[id_].append(1. / (dist ** 2))
    return max(d.keys(), key=lambda id_: sum(d[id_]))


class NMSLIBClassifier(ClassifierMixin, BaseEstimator):
    
    def __init__(self, method='hnsw', space='l2',
                 index_params=None):
        self.method = method
        self.space = space
        if index_params is None:
            index_params = {'efConstruction': 100}
        self.index_params = index_params
    
    def fit(self, X, y):
        self.index = nmslib.init(method=self.method, space=self.space)
        self.index.addDataPointBatch(X, y)
        self.index.createIndex(self.index_params)
        return self
        
    def predict(self, X):
        y_pred = self.index.knnQueryBatch(X, k=10)
        return [weighted_class(*i) for i in y_pred]