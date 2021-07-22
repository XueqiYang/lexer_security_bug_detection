from dataclasses import dataclass
from utils import ConfigBase
import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn.metrics
import pickle
from data_prep import data_prep
from data_prep import Config as DataConfig
from model import select_model
from pathlib import Path
import pdb


@dataclass
class Config(DataConfig):
    ipath2model: str = "model/vanilla_cnn-01.hdf5"
    log_filename: str = "log/model_1.log"


def format_label(y):
    ans = [[] for _ in range(len(y))]
    for col in range(0, len(y)):
        for row in y[col]:
            if row[0] >= row[1]:
                ans[col].append(0)
            else:
                ans[col].append(1)
    return ans


def test_model(config):
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

    if config.only_single_label >= 0:
        predicted = [predicted]
    pred_test = format_label(predicted)
    y_test = format_label(y_test)

    import sys
    log_writer = open(config.log_filename, 'w')
    org_stdout = sys.stdout
    sys.stdout = log_writer
    for col in range(0, len(predicted)):
        print(pd.value_counts(pred_test[col]))

    for col in range(1, len(predicted)+1):
        print('\nThis is evaluation for column', col)
        confusion = sklearn.metrics.confusion_matrix(
            y_true=y_test[col - 1], y_pred=pred_test[col - 1])
        print(confusion)
        tn, fp, fn, tp = confusion.ravel()
        print('\nTP:', tp)
        print('FP:', fp)
        print('TN:', tn)
        print('FN:', fn)

        # Performance measure
        print('\nAccuracy: ' + str(sklearn.metrics.accuracy_score(
            y_true=y_test[col - 1], y_pred=pred_test[col-1])))
        print('Precision: ' + str(sklearn.metrics.precision_score(
            y_true=y_test[col - 1], y_pred=pred_test[col-1])))
        print('Recall: ' + str(sklearn.metrics.recall_score(
            y_true=y_test[col - 1], y_pred=pred_test[col-1])))
        print('F-measure: ' + str(sklearn.metrics.f1_score(
            y_true=y_test[col - 1], y_pred=pred_test[col-1])))
        print('Precision-Recall AUC: ' + str(sklearn.metrics.average_precision_score(
            y_true=y_test[col - 1], y_score=predicted[col-1][:, 1])))
        print('AUC: ' + str(sklearn.metrics.roc_auc_score(
            y_true=y_test[col - 1], y_score=predicted[col-1][:, 1])))
        print('MCC: ' + str(sklearn.metrics.matthews_corrcoef(
            y_true=y_test[col - 1], y_pred=pred_test[col-1])))
    sys.stdout = org_stdout
    return True


if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    
    tf_config = ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = InteractiveSession(config=tf_config)

    config = Config().parse_args()
    test_model(config)
