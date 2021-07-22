from dataclasses import dataclass
from utils import ConfigBase
import modules


@dataclass
class Config(ConfigBase):
    word_capacity: int = 10000
    model_select: int = 0


def select_model(config):
    model_select = config.model_select
    if model_select == 0:
        print("> selecting vanilla cnn")
        return modules.vanilla_cnn(config.word_capacity, config.max_seq_len)
    elif model_select == 1:
        print("> selecting vanilla rnn")
        return modules.vanilla_rnn(config.word_capacity, config.max_seq_len)
    elif model_select == 2:
        print("> selecting vanilla crnn")
        return modules.vanilla_crnn(config.word_capacity, config.max_seq_len)
    elif model_select == 3:
        print("> selecting multi-head cnn")
        return modules.multihead_cnn(config.word_capacity, config.max_seq_len)
    elif model_select == 4:
        print("> selecting one-head cnn")
        return modules.onehead_cnn(config.word_capacity, config.max_seq_len)
    elif model_select == 401:
        print("> selecting one-head cnn_v1")
        return modules.onehead_cnn_v1(config.word_capacity, config.max_seq_len)
    elif model_select == 402:
        print("> selecting one-head cnn_v2")
        return modules.onehead_cnn_v2(config.word_capacity, config.max_seq_len)
    elif model_select == 403:
        print("> selecting one-head cnn_v3")
        return modules.onehead_cnn_v3(config.word_capacity, config.max_seq_len)
    elif model_select == 404:
        print("> selecting one-head cnn_v4")
        return modules.onehead_cnn_v4(config.word_capacity, config.max_seq_len)
    elif model_select == 5:
        print("> selecting one-head rnn")
        return modules.onehead_rnn(config.word_capacity, config.max_seq_len)
    elif model_select == 501:
        print("> selecting one-head rnn_v1")
        return modules.onehead_rnn_v1(config.word_capacity, config.max_seq_len)
    elif model_select == 502:
        print("> selecting one-head rnn_v2")
        return modules.onehead_rnn_v2(config.word_capacity, config.max_seq_len)
    elif model_select == 503:
        print("> selecting one-head rnn_v3")
        return modules.onehead_rnn_v3(config.word_capacity, config.max_seq_len)
    elif model_select == 6:
        print("> selecting one-head crnn")
        return modules.onehead_crnn(config.word_capacity, config.max_seq_len)
    else:
        print(f"Unexpected model_select: {model_select}. Exit!")
        return None


if __name__ == '__main__':
    config = Config().parse_args()
    select_model(config)
