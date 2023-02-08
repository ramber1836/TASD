import argparse
import os
import torch
import json
from generation import generation
from transformer import GPT2Tokenizer
from transformer.models.gpt2.gpt2_tasd import GPT2TASDLMHeadModel
from load_dataset import myDataloader, path2data
import threading
import pandas as pd

def rewrite(mode, best_model_idx, args):
    out_path = "./{}/generate/{}/{}/{}/epochs{}_save{}_model{}_beam{}_generate{}_lr{}_{}_json".format(root_dir, mode, args.model_size, args.turn, args.epochs, args.save_every, best_model_idx, args.beam_num, args.generate_length, args.lr, args.table)
    in_path = "{}/rewrite/{}_epochs{}_save{}_beam{}_generate{}_lr{}_{}/TD_{}_input".format(root_dir, args.model_size, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr, args.table, mode)
    idxs_path = "./{}/bleu/{}/{}/{}/epochs{}_save{}/model{}_beam{}_generate{}_lr{}_{}.csv".format(root_dir, mode, args.model_size, args.turn, args.epochs, args.save_every, best_model_idx, args.beam_num, args.generate_length, args.lr, args.table)
    if not os.path.exists("{}/rewrite".format(root_dir)):
        os.mkdir("{}/rewrite".format(root_dir))
    if not os.path.exists("{}/rewrite/{}_epochs{}_save{}_beam{}_generate{}_lr{}_{}".format(root_dir, args.model_size, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr, args.table)):
        os.mkdir("{}/rewrite/{}_epochs{}_save{}_beam{}_generate{}_lr{}_{}".format(root_dir, args.model_size, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr, args.table))
    if not os.path.exists(in_path):
        os.mkdir(in_path)
    f = open(out_path, "r")
    idxs = pd.DataFrame(pd.read_csv(idxs_path))['idx']
    for i, line in enumerate(f):
        w = open(in_path+"/"+str(i), "w")
        text = json.loads(line)[idxs[i]].replace("<|endoftext|>",".").replace("\n",",")
        w.write(text)
        w.close()


def run(i):
    locks[i].acquire()
    device = "cuda:{}".format(i)
    train_dataloader = myDataloader(chunks_table_new[length * i :length*(i+1)], chunks_table_new_num[length * i :length*(i+1)], token_chunks[length * i :length*(i+1)], batch_size=1)
    # 根据模型生成训练集文本
    print("Generate the train text[{}:{}]".format(length * i,length*(i+1)))
    candidates_list[i] = generation(model_path, train_dataloader, device, args.turn, args.beam_num, args.generate_length)
    # candidates_list[i] = generate(model_path, train_dataloader, device, args.turn, 1, args.generate_length)
    locks[i].release()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=21, type=int, help='')
    parser.add_argument('--save_every', default=2, type=int, help='')
    parser.add_argument('--beam_num', default=2, type=int, help='')
    parser.add_argument('--generate_length', default=256, type=int, help='')
    parser.add_argument('--model_size', default="medium", type=str, help='')
    parser.add_argument('--turn', default="first", type=str, help='')
    parser.add_argument('--lr', default=1e-5, type=float, help='')
    parser.add_argument('--cuda_num', default=4, type=int, help='')
    parser.add_argument('--length', default=-1, type=int, help='')
    parser.add_argument('--mode', default="test", type=str, help='')
    parser.add_argument('--data_name', default="numericNLG", type=str, help='')
    parser.add_argument('--table', default="NT", type=str, help='')
    args = parser.parse_args()
    print(args)
    root_dir = "afs/{}".format(args.data_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 先从val中找到最优的模型id
    best_model_idx = 0
    best_model_score = 0
    m = open("./{}/bleu/{}/{}/{}/epochs{}_save{}/beam{}_generate{}_lr{}_{}_json".format(root_dir, "val", args.model_size, args.turn, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr, args.table), "r") 
    for l in m:
        idx, metric = l.split('\t')
        metric = json.loads(metric)
        best_model_idx = idx if metric['BLEU'] > best_model_score else best_model_idx
        best_model_score = max(best_model_score, metric['BLEU'])
    print("The best model id is {}".format(best_model_idx))


    # 重写所有数据集的文件夹
    print("Rewrite to the files")
    rewrite(args.mode, best_model_idx, args)
