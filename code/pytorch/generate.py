import argparse
import numpy as np
from tqdm import tqdm
import os
from load_dataset import myDataloader, path2data
import json
from generation import generation
import threading

def run(cuda_i, length, model_path):
    i = cuda_i
    locks[i].acquire()
    device = "cuda:{}".format(i)
    dataloader = myDataloader(chunks_table_new[length * i :length*(i+1)], chunks_table_new_num[length * i :length*(i+1)], token_chunks[length * i :length*(i+1)],target_chunks[length * i :length*(i+1)], batch_size=1)
    # 根据模型生成文本
    print("Generate the {} text[{}:{}]".format(args.mode, length * i,length*(i+1)))
    candidates_list[i] = generation(model_path, dataloader, device, args)
    # print(candidates_list[i])
    locks[i].release()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="test", type=str, help='')
    parser.add_argument('--model_size', default="medium", type=str, help='')
    parser.add_argument('--beam_num', default=2, type=int, help='')
    parser.add_argument('--epochs', default=21, type=int, help='')
    parser.add_argument('--save_every', default=1, type=int, help='')
    parser.add_argument('--cuda_num', default=8, type=int, help='')
    parser.add_argument('--generate_length', default=256, type=int, help='')
    parser.add_argument('--turn', default="first", type=str, help='')
    parser.add_argument('--lr', default=1e-5, type=float, help='')
    parser.add_argument('--length', default=-1,type=int, help='')
    parser.add_argument('--start_epoch', default=20,type=int, help='')
    parser.add_argument('--end_epoch', default=21,type=int, help='')
    parser.add_argument('--data_name', default="numericNLG",type=str, help='')
    parser.add_argument('--table', default="NT",type=str, help='')
    parser.add_argument('--model_pt', default="", type=str, help='')
    
    args = parser.parse_args()

    print("Generate the text")
    print(args)
    root_dir = "afs/{}".format(args.data_name)
    table_data = "../../data/{}/table_{}.pth".format(args.data_name, args.mode)

    if args.turn == "rewrite":
        text_data = "./{}/rewrite/{}_epochs{}_save{}_beam{}_generate{}_lr{}_{}/text_{}.pth".format(root_dir, args.model_size, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr,args.table, args.mode)
    else:
        text_data = "../../data/{}/text_{}.pth".format(args.data_name, args.mode)

    chunks_table_new, chunks_table_new_num, token_chunks, target_chunks = path2data(text_data, table_data, length=args.length)
    
    if not os.path.exists("./{}/generate".format(root_dir)):
        os.mkdir("./{}/generate".format(root_dir))
    if not os.path.exists("./{}/generate/{}".format(root_dir, args.mode)):
        os.mkdir("./{}/generate/{}".format(root_dir, args.mode))
    if not os.path.exists("./{}/generate/{}/{}".format(root_dir, args.mode, args.model_size)):
        os.mkdir("./{}/generate/{}/{}".format(root_dir, args.mode, args.model_size))
    if not os.path.exists("./{}/generate/{}/{}/{}".format(root_dir, args.mode, args.model_size, args.turn)):
        os.mkdir("./{}/generate/{}/{}/{}".format(root_dir, args.mode, args.model_size, args.turn))

    locks = [threading.Lock() for i in range(args.cuda_num)] # 实例化锁对象

    if args.mode == "train":
        # 先从val中找到最优的模型id
        best_model_idx = 0
        best_model_score = 0
        m = open("./{}/bleu/{}/{}/{}/epochs{}_save{}/beam{}_generate{}_lr{}_{}_json".format(root_dir, "val", args.model_size, args.turn, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr, args.table), "r") 
        for l in m:
            idx, metric = l.split('\t')
            metric = json.loads(metric)
            best_model_idx = idx if metric['BLEU'] > best_model_score else best_model_idx
            best_model_score = max(best_model_score, metric['BLEU'] )
        print("The best model id is {}".format(best_model_idx))
        # best_model_idx = 30

    for e in range(args.start_epoch, args.end_epoch, args.save_every):
        if args.mode == "train" and e != int(best_model_idx):
            continue
        print("generate the model-{} text".format(e))
        model_path = "{}/model/{}/{}/checkpoint_{}_{}_{}_{}/{}".format(root_dir, args.model_size, args.turn, args.epochs, args.save_every, args.lr,args.table, e)
        # model_path = "{}/model/{}/{}/checkpoint_{}_{}_{}_{}/{}".format(root_dir, args.model_size, args.turn, args.epochs, args.save_every, args.lr,args.table, e)
        length = len(token_chunks) // args.cuda_num + 1
        threads = []
        candidates_list = [[]for i in range(args.cuda_num)]

        # run(0, length, model_path)
        
        for i in range(args.cuda_num):
            t = threading.Thread(target=run, args=(i, length, model_path,))
            t.start()
            threads.append(t)

        candidates = []
        for i, thread in enumerate(threads):
            thread.join()
        for candidate in candidates_list:
            candidates = candidates + candidate

        out_path = "./{}/generate/{}/{}/{}/epochs{}_save{}_model{}_beam{}_generate{}_lr{}_{}_json".format(root_dir, args.mode, args.model_size, args.turn, args.epochs, args.save_every, e, args.beam_num, args.generate_length, args.lr, args.table)
        # out_path = "afs/numericNLG/generate/test/temp_json"
        w = open(out_path, "w")
        for i in candidates:
            w.write(json.dumps(i) + "\n")
        w.close()