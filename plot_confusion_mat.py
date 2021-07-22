from dataclasses import dataclass
from utils import ConfigBase
import pandas as pd
import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import numpy as np
import sklearn.metrics
import pickle
from data_prep import data_prep
from data_prep import Config as DataConfig
from model import select_model
from model import Config as ModelConfig
from pathlib import Path
from data_prep import format_hdf5
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sn
import matplotlib.pyplot as plt
import pdb


@dataclass
class Config(DataConfig, ModelConfig):
    # ipath2model: str = "model/vanilla_cnn-{epoch:02d}.hdf5"
    ipath2model: str = "model/vanilla_cnn-01.hdf5"
    fig_dir: str = "figure"


def format_label(y):
    ans = [[] for _ in range(len(y))]
    for col in range(0, len(y)):
        for row in y[col]:
            if row[0] >= row[1]:
                ans[col].append(0)
            else:
                ans[col].append(1)
    return ans

# TN [[96638 28329] FN
# FP  [  227  2225]] TP
# 
# [[94856 27672]
#  [  612  4279]]
# 
# [[103414  23727]
#  [    22    256]]
# 
# [[91652 34575]
#  [  550   642]]
# 
# [[90410 33519]
#  [  934  2556]]
def plot_confusion_mat(config):
    # data preparation
    # each is a tuple of (x_train, y_train)
    _, _, test = data_prep(config)
    x_test, y_test = test
    print("Tensorlfow version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
    model = tf.keras.models.load_model(config.ipath2model)
    results = model.evaluate(x_test, y_test, batch_size=128)
    for num in range(0, len(model.metrics_names)):
        print(model.metrics_names[num] + ': ' + str(results[num]))
    predicted = model.predict(x_test)

    pred_test = format_label(predicted)
    y_test = format_label(y_test)

    for col in range(0, len(predicted)):
        print(pd.value_counts(pred_test[col]))

    for col in range(1, len(predicted)+1):
        print('\nThis is evaluation for column', col)
        confusion = sklearn.metrics.confusion_matrix(
            y_true=y_test[col - 1], y_pred=pred_test[col - 1])
        print(confusion)
        df_cm = pd.DataFrame(confusion, index = ["Neg", "Pos"],
                  columns = ["Neg", "Pos"])
        fig = plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)
        fig.savefig(f'{config.fig_dir}/conf_mat_raw_col{col}.png')
        plt.close()


if __name__ == "__main__":
    config = Config().parse_args()
    plot_confusion_mat(config)
