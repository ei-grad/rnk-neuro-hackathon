import pickle

import pandas as pd

import nmslib

from flask import request, render_template
from flask import Flask

from utils import CLASSES, get_Xy
from nmslibclf import weighted_class


app = Flask(__name__)
app.debug = True

print("Loading data...")
X, y, _ = get_Xy('data/pp_2')

index = nmslib.init(method='hnsw', space='l1')

print("Adding data points...")
index.addDataPointBatch(X, y)

print("Loading index...")
index.loadIndex('index.dat')


def predict(data):
    best_classes = []
    for class_ids, distances in index.knnQueryBatch(data, k=10):
        best_classes.append(weighted_class(class_ids, distances))
    return CLASSES[int(pd.Series(best_classes).mode())]


@app.route('/', methods=['GET', 'POST'])
def handler():
    message = ""
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename[-4:] == '.pkl':
            result = predict(pickle.load(file))
            message = f"File {file.filename} is {result}"
        else:
            message = "ERROR: File should have .pkl extension"
    return render_template('web.html', message=message)
