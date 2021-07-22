import tensorflow as tf
import numpy as np


def vanilla_cnn(word_capacity, max_seq_len, rand_seed=71926):
    """ multi-label CNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), padding='same', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.MaxPool1D(pool_size=5)(mid_layers)
    # mid_layers = tf.keras.layers.Dropout(0.5)(mid_layers)
    mid_layers = tf.keras.layers.Flatten()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output2 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output3 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output4 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output5 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    model = tf.keras.Model(
        inp_layer, [output1, output2, output3, output4, output5])
    return model


def multihead_cnn(word_capacity, max_seq_len, rand_seed=71926):
    """ multi-label CNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), padding='same', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.MaxPool1D(pool_size=5)(mid_layers)
    # mid_layers = tf.keras.layers.Dropout(0.5)(mid_layers)
    mid_layers = tf.keras.layers.Flatten()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    # mid_layers = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    output2 = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output2 = tf.keras.layers.Dense(2, activation='softmax')(output2)
    output3 = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output3 = tf.keras.layers.Dense(2, activation='softmax')(output3)
    output4 = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output4 = tf.keras.layers.Dense(2, activation='softmax')(output4)
    output5 = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output5 = tf.keras.layers.Dense(2, activation='softmax')(output5)
    model = tf.keras.Model(
        inp_layer, [output1, output2, output3, output4, output5])
    return model


def onehead_cnn(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label CNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), padding='same', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.Dropout(0.2)(mid_layers)
    mid_layers = tf.keras.layers.GlobalMaxPool1D()(mid_layers)
    # mid_layers = tf.keras.layers.Dropout(0.2)(mid_layers)
    mid_layers = tf.keras.layers.Flatten()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def onehead_cnn_v1(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label CNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=64, kernel_size=(
        9), padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    mid_layers = tf.keras.layers.Convolution1D(filters=128, kernel_size=(
        9), padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    mid_layers = tf.keras.layers.Convolution1D(filters=128, kernel_size=(
        9), padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.GlobalMaxPool1D()(mid_layers)
    mid_layers = tf.keras.layers.Dropout(0.2)(mid_layers)
    mid_layers = tf.keras.layers.Flatten()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def onehead_cnn_v2(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label CNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=64, kernel_size=(
        9), padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    mid_layers = tf.keras.layers.Convolution1D(filters=128, kernel_size=(
        9), padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.GlobalMaxPool1D()(mid_layers)
    mid_layers = tf.keras.layers.Dropout(0.2)(mid_layers)
    mid_layers = tf.keras.layers.Flatten()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def onehead_cnn_v3(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label CNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=64, kernel_size=(
        9), strides=3, padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    mid_layers = tf.keras.layers.Convolution1D(filters=128, kernel_size=(
        9), strides=3, padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), strides=3, padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.GlobalMaxPool1D()(mid_layers)
    mid_layers = tf.keras.layers.Dropout(0.2)(mid_layers)
    mid_layers = tf.keras.layers.Flatten()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def onehead_cnn_v4(word_capacity, max_seq_len, rand_seed=71926):
    """ multi-scale CNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layer1 = tf.keras.layers.Convolution1D(filters=64, kernel_size=(
        9), strides=1, padding='valid', activation='relu')(mid_layers)
    mid_layer1 = tf.keras.layers.BatchNormalization()(mid_layer1)
    mid_layer2 = tf.keras.layers.Convolution1D(filters=128, kernel_size=(
        9), strides=2, padding='valid', activation='relu')(mid_layer1)
    mid_layer2 = tf.keras.layers.BatchNormalization()(mid_layer2)
    mid_layer3 = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), strides=3, padding='valid', activation='relu')(mid_layer2)
    
    mid_layer1_out = tf.keras.layers.GlobalMaxPool1D()(mid_layer1)
    mid_layer2_out = tf.keras.layers.GlobalMaxPool1D()(mid_layer2)
    mid_layer3_out = tf.keras.layers.GlobalMaxPool1D()(mid_layer3)
    mid_layer_outs = tf.keras.layers.concatenate([mid_layer1_out, mid_layer2_out, mid_layer3_out])
    mid_layers = tf.keras.layers.Dropout(0.2)(mid_layer_outs)
    mid_layers = tf.keras.layers.Flatten()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def vanilla_rnn(word_capacity, max_seq_len, rand_seed=71926):
    """ multi-label RNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256)(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output2 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output3 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output4 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output5 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    model = tf.keras.Model(
        inp_layer, [output1, output2, output3, output4, output5])
    return model


def onehead_rnn(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label RNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GlobalAveragePooling1D()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def onehead_rnn_v1(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label RNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GlobalMaxPool1D()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def onehead_rnn_v2(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label RNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), padding='valid', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.MaxPool1D(pool_size=5)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GlobalMaxPool1D()(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(output1)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model


def vanilla_crnn(word_capacity, max_seq_len, rand_seed=71926):
    """ multi-label RNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), padding='same', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.MaxPool1D(pool_size=5)(mid_layers)
    # mid_layers = tf.keras.layers.Dropout(0.5)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256)(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.Dense(16, activation='relu')(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output2 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output3 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output4 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    output5 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    model = tf.keras.Model(
        inp_layer, [output1, output2, output3, output4, output5])
    return model


def onehead_crnn(word_capacity, max_seq_len, rand_seed=71926):
    """ one-label RNN
    """
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    # Create a random weights matrix
    random_weights = np.random.normal(size=(word_capacity, 13), scale=0.01)
    # Must use non-sequential model building to create branches in the output layer
    inp_layer = tf.keras.layers.Input(shape=(max_seq_len,))
    mid_layers = tf.keras.layers.Embedding(input_dim=word_capacity,
                                           output_dim=13,
                                           weights=[random_weights],
                                           input_length=max_seq_len)(inp_layer)
    mid_layers = tf.keras.layers.Convolution1D(filters=512, kernel_size=(
        9), padding='same', activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.MaxPool1D(pool_size=5)(mid_layers)
    # mid_layers = tf.keras.layers.Dropout(0.5)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256, return_sequences=True)(mid_layers)
    mid_layers = tf.keras.layers.GRU(256)(mid_layers)
    mid_layers = tf.keras.layers.Dense(64, activation='relu')(mid_layers)
    mid_layers = tf.keras.layers.BatchNormalization()(mid_layers)
    output1 = tf.keras.layers.Dense(2, activation='softmax')(mid_layers)
    model = tf.keras.Model(
        inp_layer, [output1])
    return model
