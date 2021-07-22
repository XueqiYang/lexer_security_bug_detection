import h5py
import pandas as pd
from dataclasses import dataclass
from utils import ConfigBase
import tensorflow as tf
import numpy as np
import pickle
import sklearn.metrics
from pathlib import Path
from lex_text import lex_data_in_parallel
import pdb


@dataclass
class Config(ConfigBase):
    # inputs
    ipath2train_hdf5: str = "VDISC_train.hdf5"
    ipath2valid_hdf5: str = "VDISC_validate.hdf5"
    ipath2test_hdf5: str = "VDISC_test.hdf5"
    # outputs
    iopath2train_pickle: str = "VDISC_train.pickle"
    iopath2valid_pickle: str = "VDISC_validate.pickle"
    iopath2test_pickle: str = "VDISC_test.pickle"
    # misc
    recompute: bool = False
    word_capacity: int = 10000
    max_seq_len: int = 500
    only_single_label: int = -1
    num_jobs: int = 10
    # use lexer
    use_lexer: bool = True
    opath2train_error_info: str = "VDISC_train.err"
    opath2valid_error_info: str = "VDISC_valid.err"
    opath2test_error_info: str = "VDISC_test.err"
 

def encode_texts(train, valid, test, word_capacity=10000, max_seq_len=500):
    # Create source code sdata for tokenization
    x_all = train['functionSource']
    x_all = x_all.append(valid['functionSource'])
    x_all = x_all.append(test['functionSource'])

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
          filters='', lower=False, split=' ',
          oov_token='<OOV>', char_level=False)
    tokenizer.fit_on_texts(list(x_all))
    # del(x_all)
    print('Number of tokens: ', len(tokenizer.word_counts))
    # Reducing to top N words
    tokenizer.num_words = word_capacity

    # Tokenizing train data and create matrix
    list_tokenized_train = tokenizer.texts_to_sequences(
        train['functionSource'])
    x_train = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_train,
                                                            maxlen=max_seq_len,
                                                            padding='post')
    x_train = x_train.astype(np.int64)

    # Tokenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences(test['functionSource'])
    x_test = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_test,
                                                           maxlen=max_seq_len,
                                                           padding='post')
    x_test = x_test.astype(np.int64)

    # Tokenizing validate data and create matrix
    list_tokenized_validate = tokenizer.texts_to_sequences(
        valid['functionSource'])
    x_validate = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_validate,
                                                               maxlen=max_seq_len,
                                                               padding='post')
    x_validate = x_validate.astype(np.int64)
    return x_train, x_validate, x_test


def encode_labels(train, valid, test, num_classes=2):
    y_train = []
    y_test = []
    y_validate = []

    for col in range(1, 6):
        y_train.append(tf.keras.utils.to_categorical(
            train.iloc[:, col], num_classes=2).astype(np.int64))
        y_test.append(tf.keras.utils.to_categorical(
            test.iloc[:, col], num_classes=2).astype(np.int64))
        y_validate.append(tf.keras.utils.to_categorical(
            valid.iloc[:, col], num_classes=2).astype(np.int64))
    return y_train, y_validate, y_test


def format_hdf5(ipath2hdf5):
    data = h5py.File(ipath2hdf5,'r')
    mydf = pd.DataFrame(list(data['functionSource']))
    mydf['CWE-119']=list(data['CWE-119'])
    mydf['CWE-120']=list(data['CWE-120'])
    mydf['CWE-469']=list(data['CWE-469'])
    mydf['CWE-476']=list(data['CWE-476'])
    mydf['CWE-other']=list(data['CWE-other'])
    mydf.rename(columns={0:'functionSource'},inplace=True)
    return mydf


def lex_and_select_data(dataset, error_info_file, num_jobs=10):
    errs = lex_data_in_parallel(dataset, num_jobs=num_jobs, opath2error_info=error_info_file)
    if errs:
        dataset.drop(dataset.index[errs], inplace=True)


def data_prep(config):
    recompute = config.recompute or False==all([Path(i).is_file() and Path(i).exists() for i in [config.iopath2train_pickle, config.iopath2valid_pickle, config.iopath2test_pickle]])
    if recompute:
        train, valid, test = [format_hdf5(x) for x in [
            config.ipath2train_hdf5, config.ipath2valid_hdf5, config.ipath2test_hdf5]]
        for dataset in [train, valid, test]:
            for col in range(1, 6):
                dataset.iloc[:, col] = dataset.iloc[:, col].map(
                    {False: 0, True: 1})

        if config.use_lexer:
            print(">> pre-lexing texts")
            print("lexing train texts")
            lex_and_select_data(train, config.opath2train_error_info, num_jobs=config.num_jobs)
            print("lexing valid texts")
            lex_and_select_data(valid, config.opath2valid_error_info, num_jobs=config.num_jobs)
            print("lexing test texts")
            lex_and_select_data(test, config.opath2test_error_info, num_jobs=config.num_jobs)
        
        print(">> encoding texts")
        x_train, x_valid, x_test = encode_texts(
            train, valid, test,
            config.word_capacity, config.max_seq_len)

        print(">> encoding labels")
        y_train, y_valid, y_test = encode_labels(
            train, valid, test,
            num_classes=2)
        train = (x_train, y_train)
        valid = (x_valid, y_valid)
        test = (x_test, y_test)
        with open(config.iopath2train_pickle, 'wb') as fout:
            pickle.dump(train, fout)
        with open(config.iopath2valid_pickle, 'wb') as fout:
            pickle.dump(valid, fout)
        with open(config.iopath2test_pickle, 'wb') as fout:
            pickle.dump(test, fout)
    else:
        print(">> preprocessed data exist. try reading instead.")
        train = pickle.load(open(config.iopath2train_pickle, 'rb'))
        valid = pickle.load(open(config.iopath2valid_pickle, 'rb'))
        test = pickle.load(open(config.iopath2test_pickle, 'rb'))
    # unpack
    if config.only_single_label >= 0:
        only_single_label = config.only_single_label
        x_train, y_train = train
        train = (x_train, [y_train[only_single_label]])
        x_valid, y_valid = valid
        valid = (x_valid, [y_valid[only_single_label]])
        x_test, y_test = test
        test = (x_test, [y_test[only_single_label]])
    return train, valid, test


if __name__ == '__main__':
    config = Config().parse_args()
    data_prep(config)
