import glob
import numpy as np
import os
import tqdm
import json

def table_encode(t, tokenizer, type="", index=-1):
    for i in range(len(t)):
        for j in range(len(t[0])):
            try:
                t[i][j] = tokenizer(t[i][j])['input_ids']
            except:
                print(type, i, j, index)
                print(t[i][j])
                exit(1)
    return t

def expend(tokens, number, tag_token):
    return tokens + [tag_token] * (number - len(tokens))

def load_numericNLG_tables(tokenizer, path):

    def merge_table(table_list, tag_token):
        H, TH, VAL, METRIC = table_list
        result = [[0 for _ in range(len(H[0]))] for _ in range(len(H))]
        for i in range(len(H)):
            for j in range(len(H[0])):
                result[i][j] = expend(H[i][j], MAX_H, tag_token) + expend(TH[i][j], MAX_TH, tag_token) + \
                    expend(VAL[i][j], MAX_VAL, tag_token) + expend(METRIC[i][j], MAX_METRIC, tag_token)
        return np.array(result)

    tables = json.loads(open(path, "r").readline())
    semantic_table_chunks = []
    numerical_table_chunks = []
    H, TH, VAL, NUM, METRIC, TARGET = tables
    tag_token = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    MAX_H = 51
    MAX_TH = 24
    MAX_VAL = 20
    MAX_METRIC = 27
    for index, table in enumerate(zip(H, TH, VAL, NUM, METRIC)):
        h, th, val, num, metric = table
        h = table_encode(h, tokenizer, "H", index)
        th = table_encode(th, tokenizer, "TH", index)
        val = table_encode(val, tokenizer, "VAL", index)
        num = num
        metric = table_encode(metric, tokenizer, "METRIC", index)
        #TARGET = TARGET
        semantic_table_chunks.append(merge_table([h, th, val, metric], tag_token))
        numerical_table_chunks.append(num)
    return semantic_table_chunks, numerical_table_chunks


def load_Totto_tables(tokenizer, path):

    def merge_table(table_list, tag_token):
        H, VAL = table_list
        result = [[0 for _ in range(len(H[0]))] for _ in range(len(H))]
        for i in range(len(H)):
            for j in range(len(H[0])):
                result[i][j] = expend(H[i][j], MAX_H, tag_token) + expend(VAL[i][j], MAX_VAL, tag_token)
        return np.array(result)

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
    
    for i, chunk in enumerate(semantic_table_chunks):
        semantic_table_chunks[i] = merge_table(chunk, tag_token)

    return semantic_table_chunks, numerical_table_chunks
