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
from pathlib import Path
from data_prep import format_hdf5
import h5py
import pdb


@dataclass
class Config(DataConfig):
    # ipath2model: str = "model/vanilla_cnn-01.hdf5"
    ipath2model: str = "model/singlelabel_search.hdf5"
    fig_dir: str = "figure"
    layername: str = "conv1d"
    only_single_label: int = 1
    sample_idx: int = 89  # 49  select the sample to inspect
    cam_thresh: float = 0.9


# similar to PostLexer in lex_text.py but the function is to for visualization
class PostMapper:
    def __init__(self):
        self.func_mapper = {
            'keyword': self.handle_keyword,
            'identifier': self.handle_identifier,
            'integer': self.handle_integer,
            'floating': self.handle_floating,
            'string': self.handle_string,
            'character': self.handle_character,
            'operator': self.handle_operator,
            'preprocessor': self.handle_preprocessor,
            'filename': self.handle_filename,
            'newline': self.handle_newline,
        }
        
    def process_oneline(self, row_idx, col_idx, category, word):
        # if ',' in word:
        #     print(f"row_idx: {row_idx}, col_idx:{col_idx}, {category}, {word}")
        return self.func_mapper[category](row_idx, col_idx, word)
        
    def handle_keyword(self, row_idx, col_idx, word):
        yield int(row_idx), int(col_idx), word

    def handle_identifier(self, row_idx, col_idx, word):
        yield int(row_idx), int(col_idx), word
    
    def handle_integer(self, row_idx, col_idx, word):
        for i, ch in enumerate(word):
            yield int(row_idx), int(col_idx)+i, ch

    def handle_floating(self, row_idx, col_idx, word):
        yield int(row_idx), int(col_idx), word

    def handle_string(self, row_idx, col_idx, word):
        yield int(row_idx), int(col_idx), word

    def handle_character(self, row_idx, col_idx, word):
        yield int(row_idx), int(col_idx), word
     
    def handle_operator(self, row_idx, col_idx, word):
        if word == "\",\"":
            word = ','
        yield int(row_idx), int(col_idx), word

    def handle_preprocessor(self, row_idx, col_idx, word):
        yield int(row_idx), int(col_idx), word

    def handle_newline(self, row_idx, col_idx, word):
        yield int(row_idx), int(col_idx), word

    def handle_filename(self, word):
        yield int(row_idx), int(col_idx), word


def plot_cam(config):
    sample_idx = config.sample_idx
    _, _, test = data_prep(config)
    x_test, y_test = test
    data = h5py.File(config.ipath2test_hdf5, 'r')['functionSource']
    print("Tensorlfow version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
    model = tf.keras.models.load_model(config.ipath2model)
    model.summary()
    if sample_idx < 0:
        from random import choice
        sample_idx = choice(np.where(y_test[0][:, 1] == 1)[0])
        print(f"randomly choice positive sample {sample_idx}")
    pos_prob = model.predict(x_test)[sample_idx, 1]
    # if the prediction should be positive, which region of the input should it focus on ?
    # heatmap gives the answer
    heatmap = make_gradcam_heatmap(x_test[sample_idx], model, config.layername, class_index=1)
    # plot 
    org_text = data[sample_idx]
    tokens = x_test[sample_idx]
    if 1 == y_test[0][sample_idx, 1]:
        print(f"This sample is positive, the predicted positive prob is {pos_prob}")
    else:
        print(f"This sample is negative, the predicted positive prob is {pos_prob}")
    visualize(org_text, tokens, heatmap, config.cam_thresh)


def visualize(org_text, tokens, heatmap, thresh=0.9):
    import copy
    from termcolor import colored
    formated_text = get_lexer_output(org_text)
    num_nonzeros = get_num_nonzero_tokens(tokens)
    if not len(formated_text) == num_nonzeros:
        print(f"#token doesn't match. lexer output is {len(formated_text)}, while quantized output is {num_nonzeros}")
        pdb.set_trace()
    for i in range(num_nonzeros):
        formated_text[i].append(heatmap[i])
    lines = org_text.splitlines()
    lines_format = [[] for _ in lines]
    print("Analyzing error position candidates")
    for row, col, category, word, prob in formated_text:
        if prob < thresh: continue
        row, col = int(row) - 1, int(col)
        word_len = len(word)
        lines_format[row].append((col, col+word_len))
        print(f"{row} {col} {category} {word}")
    # format lines for visualization
    out_lines = []
    for i, line in enumerate(lines):
        # if len(lines_format[i]) == 0: continue
        # modify from right to left
        # there could exsist bugs in some extreme cases, like the code contains a character ','
        # will validate later
        for start, end in reversed(lines_format[i]):
            line = line[:start] + colored(line[start:end], 'white', 'on_red') + line[end:]
        out_lines.append(line)
    print('\n'.join(out_lines))
        

def get_lexer_output(text):
    from utils import run_command
    mapper = PostMapper()
    out = run_command("tokenize -l C++ -m csv", text)
    ans = []
    for line in out.splitlines()[1:]:
        row, col, category, word = line.rstrip().split(',', 3)
        for row, col, word in mapper.process_oneline(row, col, category, word):
            ans.append([row, col, category, word])
        # token = lexer.process_oneline(category, word)
    return ans


def get_num_nonzero_tokens(tokens):
    start, end = 0, len(tokens)
    while start < end:
        mid = (start + end) // 2
        if tokens[mid] == 0:
            end = mid
        else:
            start = mid + 1
    return start


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, class_index=0):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    img_array = tf.expand_dims(img_array, 0)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # class_channel = tf.squeeze(preds[class_index])[-1]
        class_channel = tf.squeeze(preds)[class_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



if __name__ == "__main__":
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    
    tf_config = ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = InteractiveSession(config=tf_config)

    config = Config().parse_args()
    plot_cam(config)
