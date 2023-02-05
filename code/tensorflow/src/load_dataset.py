import glob
import numpy as np
import os
import tensorflow.compat.v1 as tf
import tqdm
import json

MAX_H = 51
MAX_TH = 24
MAX_VAL = 20
MAX_METRIC = 27

def load_dataset(enc, path, combine, encoding=None, split=False):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for index, fname in enumerate(fnames):
                paths.append(os.path.join(dirpath, str(index)))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = []
    raw_text_len = 0
    if split is False:
        for path in tqdm.tqdm(paths):
            if path.endswith('.npz'):
                # Pre-encoded
                with np.load(path) as npz:
                    for item in npz.files:
                        token_chunks.append(npz[item])
            else:
                # Plain text
                with open(path, 'r', encoding=encoding) as fp:
                    text = fp.read()
                    raw_text.append(text)
                    raw_text_len += len(text)
                if raw_text_len >= combine:
                    tokens = np.hstack([enc.encode(entry)+[enc.encoder['<|endoftext|>']] for entry in raw_text])
                    token_chunks.append(tokens)
                    raw_text = []
                    raw_text_len = 0
        if raw_text:
            tokens = np.hstack([enc.encode(entry)+[enc.encoder['<|endoftext|>']] for entry in raw_text])
            token_chunks.append(tokens)
    else:
        for path in tqdm.tqdm(paths):
            if path.endswith('.npz'):
                # Pre-encoded
                with np.load(path) as npz:
                    for item in npz.files:
                        token_chunks.append(npz[item])
            else:
                # Plain text
                path_split = path.split('/')
                if 'rewrite' in path:
                    path1 = path
                else:
                    path1 = 'afs/data/TD_train_rewrite_input/' + path_split[-1]
                path2 = 'afs/data/TD_train_rewrite_gold/' + path_split[-1]
                fp1 = open(path1, 'r', encoding=encoding)
                fp2 = open(path2, 'r', encoding=encoding)
                text = fp1.readline() + ' ' + fp2.readline()
                raw_text.append(text)
                tokens_list = []
                for entry in raw_text:
                    tokens_item = enc.encode(entry)+[enc.encoder['<|endoftext|>']]
                    if len(tokens_item) < 1024:
                        tokens_list.append(tokens_item + [enc.encoder['<|endoftext|>']] * (1024 - len(tokens_item)))
                    if len(tokens_item) >= 1024:
                        tokens_list.append(tokens_item + [enc.encoder['<|endoftext|>']] * (1024 - (len(tokens_item) % 1024)))
                tokens = np.hstack(tokens_list)
                token_chunks.append(tokens)
                raw_text = []
    print(len(token_chunks))
    return token_chunks

def table_encode(t, enc, type="", index=-1):
    for i in range(len(t)):
        for j in range(len(t[0])):
            try:
                t[i][j] = enc.encode(t[i][j])
            except:
                print(type, i, j, index)
                print(t[i][j])
                exit(1)
    return t

def expend(tokens, number, tag_token):
    return [tag_token] * (number - len(tokens))+ tokens


def merge_table(table_list, tag_token):
    H, TH, VAL, METRIC = table_list
    result = [[0 for _ in range(len(H[0]))] for _ in range(len(H))]
    for i in range(len(H)):
        for j in range(len(H[0])):
            result[i][j] = expend(H[i][j], MAX_H, tag_token) + expend(TH[i][j], MAX_TH, tag_token) + \
                expend(VAL[i][j], MAX_VAL, tag_token) + expend(METRIC[i][j], MAX_METRIC, tag_token)
    return np.array(result)

def load_tables(enc, path):
    tables = json.loads(open(path, "r").readline())
    semantic_table_chunks = []
    numerical_table_chunks = []
    H, TH, VAL, NUM, METRIC, TARGET = tables
    tag_token = enc.encoder['<|endoftext|>']
    for index, table in enumerate(zip(H, TH, VAL, NUM, METRIC)):
        h, th, val, num, metric = table
        h = table_encode(h, enc, "H", index)
        th = table_encode(th, enc, "TH", index)
        val = table_encode(val, enc, "VAL", index)
        num = num
        metric = table_encode(metric, enc, "METRIC", index)
        #TARGET = TARGET
        semantic_table_chunks.append(merge_table([h, th, val, metric], tag_token))
        numerical_table_chunks.append(num)
    return semantic_table_chunks, numerical_table_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

class Iterater(object):
    def __init__(self, chunks, enc, seed=None):
        self.chunks = chunks
        self.chunks_index = 0
        self.last_undone = False
        self.last_chunk_index = -1
        self.enc = enc
    
    def iterate(self, length):
        chunk = self.chunks[self.chunks_index]
        if not self.last_undone:
            if len(chunk) == length:
                self.chunks_index += 1
                self.chunks_index = self.chunks_index % len(self.chunks)
                return chunk
            else:
                self.last_chunk_index = length // 2
                self.last_undone = True
                return chunk[:length]
        else:
            if len(chunk) - self.last_chunk_index <= length:
                self.chunks_index += 1
                self.chunks_index = self.chunks_index % len(self.chunks)
                self.last_undone = False
                if len(chunk) - self.last_chunk_index == length:
                    return chunk[self.last_chunk_index:]
                else:
                    return chunk[-length:]
            else:
                self.last_chunk_index = self.last_chunk_index + length // 2
                return chunk[self.last_chunk_index - length // 2:self.last_chunk_index + length // 2]
        
class Iterater_table(object):
    def __init__(self, chunks, tables, seed=None):
        self.chunks = chunks
        self.chunks_index = 0
        self.last_undone = False
        self.last_chunk_index = -1
        self.tables = tables
        self.total_size = len(chunks)
        print(len(self.chunks), len(self.tables[0]))
        assert len(self.chunks) == len(self.tables[0])
    
    def iterate(self, length):
        chunk = self.chunks[self.chunks_index]
        table_chunk = [self.tables[0][self.chunks_index], self.tables[1][self.chunks_index]]
        if not self.last_undone:
            if len(chunk) == length:
                self.chunks_index += 1
                self.chunks_index = self.chunks_index % len(self.chunks)
                return chunk, table_chunk
            else:
                self.last_chunk_index = length // 2
                self.last_undone = True
                return chunk[:length], table_chunk
        else:
            if len(chunk) - self.last_chunk_index <= length:
                self.chunks_index += 1
                self.chunks_index = self.chunks_index % len(self.chunks)
                self.last_undone = False
                if len(chunk) - self.last_chunk_index == length:
                    return chunk[self.last_chunk_index:], table_chunk
                else:
                    return chunk[-length:], table_chunk
            else:
                self.last_chunk_index = self.last_chunk_index + length // 2
                return chunk[self.last_chunk_index - length // 2:self.last_chunk_index + length // 2], table_chunk
        
