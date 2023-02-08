import json
import numpy as np
import tqdm
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from load_dataset import *


def load_Totto_dataset(tokenizer, path_input, path_gold, split=False, encoding="utf8"):
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
                # text = fp_input_i.readline() + "table"
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
                    token_chunks.append(tokens_item[start:]+(end - len(tokens_item))*tokenizer('<|endoftext|>')['input_ids'])
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

def merge_table(table_list, tag_token, MAX_H, MAX_VAL):
    H, VAL = table_list
    result = [[0 for _ in range(len(H[0]))] for _ in range(len(H))]
    for i in range(len(H)):
        for j in range(len(H[0])):
            result[i][j] = expend(H[i][j], MAX_H, tag_token) + expend(VAL[i][j], MAX_VAL, tag_token)
    return np.array(result)

def load_Totto_tables(tokenizer, path):
    tables = json.loads(open(path, "r").readline())
    H, VAL = tables
    semantic_table_chunks = []
    numerical_table_chunks = []
    tag_token = tokenizer('<|endoftext|>')['input_ids'][0]
    MAX_H = 29
    MAX_VAL = 116
    for index, table in enumerate(zip(H, VAL)):
        h, val = table
        h = table_encode(h, tokenizer)
        val = table_encode(val, tokenizer)
        semantic_table_chunks.append([h, val])
        numerical_table_chunks.append(np.array([index]))
        MAX_H = max([max([len(j) for j in i]) for i in h]+[MAX_H])
        MAX_VAL = max([max([len(j) for j in i]) for i in val]+[MAX_VAL])
    
    print(MAX_H, MAX_VAL)
    
    for i, chunk in enumerate(semantic_table_chunks):
        semantic_table_chunks[i] = merge_table(chunk, tag_token, MAX_H, MAX_VAL)

    return semantic_table_chunks, numerical_table_chunks