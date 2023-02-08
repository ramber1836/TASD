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

def load_numericNLG_dataset(tokenizer, path, split=False, encoding="utf8", tag_token=None):
    if tag_token is None:
        tag_token = tokenizer.vocab["<|endoftext|>"]
    raw_text = []
    raw_text_len = 0
    # path_gold = path.replace("rewrite", "data") + '_gold'
    path_gold = re.sub(r"rewrite/?.*/", r"data/", path) + '_gold'
    path_input = path + '_input'
    token_chunks = []
    id_list = []
    for (dirpath, _, fnames) in os.walk(path_input):
        for index, fname in enumerate(fnames):
            path_gold_i = os.path.join(path_gold,str(index))
            path_input_i = os.path.join(path_input,str(index))
            fp_gold_i = open(path_gold_i, 'r', encoding=encoding)
            fp_input_i = open(path_input_i, 'r', encoding=encoding)
            if split:
                text = fp_input_i.readline() + fp_gold_i.readline()
            else:
                text = fp_input_i.readline()
                # text = fp_input_i.readline()
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

def load_numericNLGLarge_dataset(tokenizer, path, split=False, encoding="utf8"):
    raw_text = []
    raw_text_len = 0
    # path_gold = path.replace("rewrite", "data") + '_gold'
    path_gold = re.sub(r"rewrite/?.*/", r"data/", path) + '_gold'
    path_input = path + '_input'
    token_chunks = []
    id_list = []
    for (dirpath, _, fnames) in os.walk(path_input):
        for index, fname in enumerate(fnames):
            path_gold_i = os.path.join(path_gold,str(index))
            path_input_i = os.path.join(path_input,str(index))
            fp_gold_i = open(path_gold_i, 'r', encoding=encoding)
            fp_input_i = open(path_input_i, 'r', encoding=encoding)
            if split:
                text = fp_input_i.readline() + fp_gold_i.readline()
            else:
                text = fp_input_i.readline()
                # text = fp_input_i.readline()
            tokens_item = tokenizer(text)['input_ids']
            token_chunks.append(tokens_item)
            id_list.append(str(index))

    max_len = max([len(i) for i in token_chunks])
    for i, token_chunk in enumerate(token_chunks):
        token_chunks[i] = token_chunk + (max_len + 1 - len(token_chunk))*tokenizer('<|endoftext|>')['input_ids']
    return id_list, token_chunks

def load_numericNLGbart_dataset(tokenizer, path, split=False, encoding="utf8", tag_token=None):
    if tag_token is None:
        tag_token = tokenizer.vocab["<pad>"]
    raw_text = []
    raw_text_len = 0
    # path_gold = path.replace("rewrite", "data") + '_gold'
    path_gold = re.sub(r"rewrite/?.*/", r"data/", path) + '_gold'
    path_input = path + '_input'
    token_chunks = []
    target_chunks = []
    for (dirpath, _, fnames) in os.walk(path_input):
        for index, fname in enumerate(fnames):
            path_gold_i = os.path.join(path_gold,str(index))
            path_input_i = os.path.join(path_input,str(index))
            fp_gold_i = open(path_gold_i, 'r', encoding=encoding)
            fp_input_i = open(path_input_i, 'r', encoding=encoding)
            text = fp_input_i.readline()
            target = fp_gold_i.readline()

            tokens_item = tokenizer(text)['input_ids'][:1024]
            tokens_item = tokens_item + (1024-len(tokens_item)) * [tag_token]
            target_item = tokenizer(target)['input_ids'][:1024]
            target_item = target_item + (1024-len(target_item)) * [tag_token]
            token_chunks.append(tokens_item)
            target_chunks.append(target_item)

    return token_chunks, target_chunks


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

def count_numericNLG_tables(tokenizer, path):
    tables = json.loads(open(path, "r").readline())
    semantic_table_chunks = []
    numerical_table_chunks = []
    H, TH, VAL, NUM, METRIC, TARGET = tables # H[1084] TH[1084*56*24] VAL[1084*56*24] NUM[1084*56*24] METRIC[1084*56*24] TARGET[1084*56*24]
    len_all = {"h":[], "th":[], "val":[], "metric":[]}
    for index, table in enumerate(zip(H, TH, VAL, NUM, METRIC)):
        h, th, val, num, metric = table
        h = table_encode(h, tokenizer)
        th = table_encode(th, tokenizer)
        val = table_encode(val, tokenizer)
        num = num
        metric = table_encode(metric, tokenizer)
        #TARGET = TARGET
        len_all["h"].append(max(len(j) for i in h for j in i))
        len_all["th"].append(max(len(j) for i in th for j in i))
        len_all["val"].append(max(len(j) for i in val for j in i))
        len_all["metric"].append(max(len(j) for i in metric for j in i))
    return max(len_all["h"]), max(len_all["th"]), max(len_all["val"]), max(len_all["metric"])

# from transformer import AutoTokenizer

# max_len_val = count_numericNLG_tables(AutoTokenizer.from_pretrained("afs/model/facebook/bart-large-cnn"), "afs/numericNLGbart/data/table_val.json")
# print(max_len_val)
# max_len_test = count_numericNLG_tables(AutoTokenizer.from_pretrained("afs/model/facebook/bart-large-cnn"), "afs/numericNLGbart/data/table_test.json")
# print(max_len_test)


# max_len_train = count_numericNLG_tables(AutoTokenizer.from_pretrained("afs/model/facebook/bart-large-cnn"), "afs/numericNLGbart/data/table_train.json")
# print(max_len_train)