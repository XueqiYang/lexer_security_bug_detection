import h5py
import pandas as pd
from dataclasses import dataclass
from utils import ConfigBase, run_command
import numpy as np
from tqdm import trange
import yaml
import pdb


@dataclass
class Config(ConfigBase):
    ipath2hdf5: str = "VDISC_test.hdf5"
    opath2error_info: str = "VDISC_test.err"
    num_jobs: int = 1


class PostLexer:
    def __init__(self):
        self.id2idx = {}
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
        
    def process_oneline(self, category, word):
        return self.func_mapper[category](word)
        
    def handle_keyword(self, word):
        return f"kw_{word}"

    def handle_identifier(self, word):
        if word not in self.id2idx:
            self.id2idx[word] = len(self.id2idx)
        idx = self.id2idx[word]
        return f"id_{idx}"
    
    def handle_integer(self, word):
        return ' '.join([ch for ch in word])

    def handle_floating(self, word):
        return 'const_float'

    def handle_string(self, word):
        return 'const_string'

    def handle_character(self, word):
        return 'const_char'
     
    def handle_operator(self, word):
        return f"op_{word}"

    def handle_preprocessor(self, word):
        return f"pp_{word}"

    def handle_newline(self, word):
        return "newline"

    def handle_filename(self, word):
        return "startoffile"

    def reset(self):
        self.id2idx = {}


def lex_data(dataset, opath2error_info="", max_error_num=1000):
    data = dataset['functionSource']
    lexer = PostLexer()
    # for i in range(10):
    errors = []
    iters = trange(len(data), desc='Lexing data', leave=True)

    for i in iters:
        lexer.reset()
        tokens = []
        inp = data.iloc[i]
        # out = run_command("tokenize -l C++ -n -m csv", inp)
        out = run_command("tokenize -l C++ -m csv", inp)
        try:
            for line in out.splitlines()[1:]:
                _, _, category, word = line.rstrip().split(',', 3)
                token = lexer.process_oneline(category, word)
                tokens.append(token)
            tokens = ' '.join(tokens)
            data.at[i] = tokens
        except:
            errors.append(i)
            iters.set_description(f"Lexing data, found {len(errors)} error files", refresh=True)
    if len(errors) > 0:
        print(f"In processing the data, {len(errors)} code files are failed to pass the lexing.")
        if len(errors) < 20:
          print(f"Their indexs are {errors}")
        if opath2error_info:
            print(f"Storing errors in file {opath2error_info}...")
            with open(opath2error_info, 'w') as file:
                yaml.dump(errors, file)
    else:
        print(f"error free ! Successfully processing all the data...")
    return errors


def lex_data_in_parallel(dataset, num_jobs=10, opath2error_info=""):
    data = dataset['functionSource']
    errors = []
    def helper(inp):
        lexer = PostLexer()
        tokens = []
        out = run_command("tokenize -l C++ -m csv", inp)
        try:
            for line in out.splitlines()[1:]:
                _, _, category, word = line.rstrip().split(',', 3)
                token = lexer.process_oneline(category, word)
                tokens.append(token)
            tokens = ' '.join(tokens)
            return tokens
        except:
            return ''

    from joblib import Parallel, delayed
    strs = Parallel(n_jobs=num_jobs)(delayed(helper)(data.iloc[idx]) for idx in range(len(data)))
    errors = []
    for idx, tokens in enumerate(strs):
        if len(tokens) <= 0:
            errors.append(idx)
    dataset['functionSource'] = strs
    if len(errors) > 0:
        print(f"In processing the data, {len(errors)} code files are failed to pass the lexing.")
        if len(errors) < 20:
          print(f"Their indexs are {errors}")
        if opath2error_info:
            print(f"Storing errors in file {opath2error_info}...")
            with open(opath2error_info, 'w') as file:
                yaml.dump(errors, file)
    else:
        print(f"error free ! Successfully processing all the data...")
    return errors

        
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


def debug():
    mydf = pd.read_hdf("debug.hdf", 'mydf')
    return mydf


if __name__ == "__main__":
    from datetime import datetime
    config = Config().parse_args()
    starttime = datetime.now()
    dataset = format_hdf5(config.ipath2hdf5)
    pdb.set_trace()
    # dataset = debug()
    print(f"The data loading consumes {datetime.now()-starttime}")
    starttime = datetime.now()
    errs = lex_data_in_parallel(dataset[:20000], num_jobs=config.num_jobs, opath2error_info=config.opath2error_info)
    print(f"The code lexing consumes {datetime.now()-starttime}")
