import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def table_encode(t, tokenizer):
    for i in range(len(t)):
        for j in range(len(t[0])):
            t[i][j] = tokenizer(t[i][j])['input_ids']
    return t

def expend(tokens, number, tag_token):
    return tokens + [tag_token] * (number - len(tokens))

class myDataset(Dataset):
    def __init__(self, table_tensor, num_tensor, text_tensor, target_tensor):
        self.semantic_table_tensor = np.array(table_tensor)
        self.numerical_table_tensor = np.array(num_tensor)
        self.text_tensor = text_tensor
        self.target_tensor = target_tensor
    def __len__(self):
        return len(self.text_tensor)
    def __getitem__(self, index):
        return self.semantic_table_tensor[index], self.numerical_table_tensor[index], np.array(self.text_tensor[index]), np.array(self.target_tensor[index])

def path2data(text_data_path, table_data_path, length=-1):
    id_chunks, target_chunks = torch.load(text_data_path)
    chunks_table, chunks_table_num = torch.load(table_data_path)
    chunks_table = [chunks_table[int(i)] for i in id_chunks]
    chunks_table_num = [chunks_table_num[int(i)] for i in id_chunks]
    if length == -1:
        return chunks_table, chunks_table_num, target_chunks, target_chunks
    else:
        return chunks_table[:length], chunks_table_num[:length], target_chunks[:length], target_chunks[:length]

def path2data_bart(text_data_path, table_data_path, length=-1):
    token_chunks, target_chunks = torch.load(text_data_path)
    chunks_table, chunks_table_num = torch.load(table_data_path)
    if length == -1:
        return chunks_table, chunks_table_num, token_chunks, target_chunks
    else:
        return chunks_table[:length], chunks_table_num[:length], token_chunks[:length], target_chunks[:length]

def myDataloader(chunks_table_new, chunks_table_new_num, token_chunks, target_chunks, batch_size, shuffle=False):
    dataset = myDataset(chunks_table_new, chunks_table_new_num, token_chunks, target_chunks)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
