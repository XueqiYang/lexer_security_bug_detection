from dataclasses import dataclass
from utils import ConfigBase
import tensorflow as tf
import numpy as np
import pickle
from data_prep import data_prep
from data_prep import Config as DataConfig
from model import select_model
from model import Config as ModelConfig
from pathlib import Path
import pdb


@dataclass
class Config(DataConfig, ModelConfig):
    pos_weight: float = 5.0
    opath2model: str = "model/vanilla_cnn-{epoch:02d}.hdf5"
    lr: float = 0.005
    log_dir: str = 'log'
    num_epochs: int = 20
    batch_size: int = 128
    opath2history: str = "history/vanilla_cnn-20epoch.pickle"


def train_model(config):
    # data preparation
    # each is a tuple of (x_train, y_train)
    train, valid, test = data_prep(config)
    x_train, y_train = train
    x_valid, y_valid = valid
    x_test, y_test = test
    # x_train, y_train = x_train[:1000], [y[:1000] for y in y_train]
    # x_valid, y_valid  = x_valid[:1000], [y[:1000] for y in y_valid]
    print("Tensorlfow version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
    # model initialization
    model = select_model(config)
    # optimizer configuration
    solver = tf.keras.optimizers.Adam(
        lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=1e-5, amsgrad=False)
    model.compile(optimizer=solver,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # mkdirs if necessary
    Path(config.opath2model).parent.mkdir(parents=True, exist_ok=True)
    Path(config.opath2history).parent.mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    # log configuration
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.log_dir,
                                                 histogram_freq=1,
                                                 embeddings_freq=1,
                                                 write_graph=True,
                                                 write_images=True)
    tb_callback.set_model(model)
    mdl_callback = tf.keras.callbacks.ModelCheckpoint(filepath=config.opath2model,
                                                      monitor='val_loss',
                                                      save_best_only=True,
                                                      model='auto',
                                                      save_freq="epoch",
                                                      verbose=1)
    # train configuration
    class_weights = []
    num_head = len(y_test)
    for i in range(num_head):
        neg, pos = y_test[i].sum(axis=0)
        # class_weights.append({0: 1., 1: neg/pos})
        class_weights.append({0: 1., 1: config.pos_weight})
    # class_weights = [{0: 1., 1: config.pos_weight} for _ in range(5)]
    print(f"class_weights: {class_weights}")
    history = model.fit(x=x_train,
                        y=[y_train[i] for i in range(num_head)],
                        validation_data=(x_valid, [y_valid[i]
                                                   for i in range(num_head)]),
                        epochs=config.num_epochs,
                        batch_size=config.batch_size,
                        verbose=1,
                        class_weight=class_weights,
                        callbacks=[mdl_callback, tb_callback]
                        )
    with open(config.opath2history, 'wb') as fout:
        pickle.dump(history.history, fout)


if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    
    tf_config = ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = InteractiveSession(config=tf_config)

    config = Config().parse_args()
    train_model(config)
