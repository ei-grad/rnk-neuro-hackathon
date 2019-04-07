import seaborn as sns
import pandas as pd
import random
from matplotlib import pyplot as plt

from keras.models import Model
from keras.callbacks import Callback

from utils import CLASSES


class DrawCallback(Callback):
    
    def __init__(self, model, X, y, idx, num_points=1000):
        self.draw_model = Model(model.inputs, model.get_layer("vis").output)
        self.X = X
        self.y = y
        self.idx = idx
        self.num_points = num_points
        
    def on_epoch_end(self, *args):
        idx = random.sample(self.idx, self.num_points)
        coords = self.draw_model.predict(self.X[idx])
        series = {
            'y': pd.Series(self.y[idx]).map(CLASSES.__getitem__),
        }
        for i in range(coords.shape[-1]):
            series[f'dim{i}'] = coords[:,i]
        try:
            g = sns.pairplot(pd.DataFrame(series), hue='y', plot_kws={"s": 15})
            plt.show()
        except Exception as e:
            print(e)
            plt.clf()