import json
import numpy as np
import tqdm
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from load_dataset import *
MAX_H = 52
MAX_TH = 25
MAX_VAL = 21
MAX_METRIC = 28

def load_numericNLG_dataset(tokenizer,path_input, path_gold,  split=False, encoding="utf8", tag_token=None):
    if tag_token is None:
        tag_token = tokenizer.vocab["<|endoftext|>"]
    token_chunks = []
    id_list = []
    for (dirpath, _, fnames) in os.walk(path_input):
        for index, fname in enumerate(fnames):
            path_gold_i = os.path.join(path_gold,str(index))
            path_input_i = os.path.join(path_input,str(index))
            fp_gold_i = open(path_gold_i, 'r', encoding=encoding)
            fp_input_i = open(path_input_i, 'r', encoding=encoding)
            if split:
                text = fp_input_i.readline() + "\n"+ fp_gold_i.readline()
            else:
                text = fp_input_i.readline() + "\n"
            tokens_item = tokenizer(text)['input_ids']
            if split:
                start = 0
                end = 1023
                while end < len(tokens_item):
                    token_chunks.append(tokens_item[start:end])
                    id_list.append(str(index))
                    start += 512
                    end += 512
                if len(tokens_item) > start:
                    token_chunks.append(tokens_item[start:]+(end - len(tokens_item))*[tag_token])
                    id_list.append(str(index))
            else:
                token_chunks.append(tokens_item)
                id_list.append(str(index))
            # token_chunks.append(tokens_item)
            # id_list.append(str(index))

    # max_len = max([len(i) for i in token_chunks])
    # for i, token_chunk in enumerate(token_chunks):
    #     token_chunks[i] = token_chunk + (max_len + 1 - len(token_chunk))*tokenizer('<|endoftext|>')['input_ids']
    return id_list, token_chunks


def merge_table(table_list, tag_token):
    H, TH, VAL, METRIC = table_list
    result = [[0 for _ in range(len(H[0]))] for _ in range(len(H))]
    for i in range(len(H)):
        for j in range(len(H[0])):
            result[i][j] = expend(H[i][j], MAX_H, tag_token) + expend(TH[i][j], MAX_TH, tag_token) + expend(VAL[i][j], MAX_VAL, tag_token) + expend(METRIC[i][j], MAX_METRIC, tag_token)
    return np.array(result)

def load_numericNLG_tables(tokenizer, path, tag_token=None):
    tables = json.loads(open(path, "r").readline())
    semantic_table_chunks = []
    numerical_table_chunks = []
    H, TH, VAL, NUM, METRIC, TARGET = tables # H[1084] TH[1084*56*24] VAL[1084*56*24] NUM[1084*56*24] METRIC[1084*56*24] TARGET[1084*56*24]
    if tag_token is None:
        tag_token = tokenizer.vocab["<|endoftext|>"]
    for index, table in enumerate(zip(H, TH, VAL, NUM, METRIC)):
        h, th, val, num, metric = table
        h = table_encode(h, tokenizer)
        th = table_encode(th, tokenizer)
        val = table_encode(val, tokenizer)
        num = num
        metric = table_encode(metric, tokenizer)
        #TARGET = TARGET
        semantic_table_chunks.append(merge_table([h, th, val, metric], tag_token))
        numerical_table_chunks.append(num)
    return semantic_table_chunks, numerical_table_chunks