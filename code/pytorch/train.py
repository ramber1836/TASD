import os
import torch
import argparse
import numpy as np
from transformer import AutoTokenizer
from transformer.models.gpt2.gpt2_tasd import GPT2TASDLMHeadModel
from tqdm import tqdm
from load_dataset import myDataloader, path2data
import random

def saver(checkpoint_path, model,tokenizer, epoch):
    path = "{}/{}".format(checkpoint_path, epoch)
    if not os.path.exists(path):
        os.makedirs(path)
    model.module.save_pretrained(path)
    tokenizer.save_pretrained(path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_train', default=False, type=bool, help='')
    parser.add_argument('--epochs', default=1, type=int, help='')
    parser.add_argument('--save_every', default=1, type=int, help='')
    parser.add_argument('--model_size', default="small", type=str, help='')
    parser.add_argument('--turn', default="first", type=str, help='')
    parser.add_argument('--lr', default=1e-5, type=float, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--length', default=-1, type=int, help='')
    parser.add_argument('--beam_num', default=2, type=int, help='')
    parser.add_argument('--generate_length', default=128, type=int, help='')
    parser.add_argument('--cudas', default="0", type=str, help='')
    parser.add_argument('--data_name', default="numericNLG", type=str, help='')
    parser.add_argument('--seed', default=970903, type=int, help='')
    args = parser.parse_args()
    print("Train the model")
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    root_dir = "afs/{}".format(args.data_name)
    table_train_data = "../../data/{}/table_train.pth".format(args.data_name)
    
    if args.turn == "rewrite":
        text_train_data = "./{}/rewrite/{}_epochs{}_save{}_beam{}_generate{}_lr{}/text_train_split.pth".format(root_dir, args.model_size, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr)
    else:
        text_train_data = "../../data/{}/text_train_split.pth".format(args.data_name)
    
    print("the text data path:{}, the table data path:{}".format(text_train_data, table_train_data))
    chunks_table_new, chunks_table_new_num, token_chunks, target_chunks = path2data(text_train_data, table_train_data, length=args.length)
    train_dataloader = myDataloader(chunks_table_new, chunks_table_new_num, token_chunks, target_chunks, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "../../models/pytorch/{}/gpt2-{}".format(args.data_name, args.model_size)

    checkpoint_path = "{}/model/{}/{}/checkpoint_{}_{}_{}".format(root_dir, args.model_size, args.turn, args.epochs, args.save_every, args.lr)

    # model_path = checkpoint_path + "/16"
    print("the checkpoint_path is {}".format(checkpoint_path))

    if not os.path.exists("{}/model".format(root_dir)):
        os.mkdir("{}/model".format(root_dir))
    if not os.path.exists("{}/model/{}".format(root_dir, args.model_size)):
        os.mkdir("{}/model/{}".format(root_dir, args.model_size))
    if not os.path.exists("{}/model/{}/{}".format(root_dir, args.model_size, args.turn)):
        os.mkdir("{}/model/{}/{}".format(root_dir, args.model_size, args.turn))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2TASDLMHeadModel.from_pretrained(model_path)

    optimizer = torch.optim.Adam(lr=args.lr, params=filter(lambda p: p.requires_grad, model.parameters()),eps=1e-7)

    model.train()

    model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.cudas.split(",")]).cuda()
    
    # 训练部分
    epoch_loss = []
    for e in range(args.epochs):
        with tqdm(train_dataloader) as bar:
            loss_list = []
            for semantic_table, numerical_table, text, target in bar: #然而实际上并没有用到numerical的信息
                semantic_table, text, target = semantic_table.to(device), text.to(device), target.to(device)
                inputs = {
                    'input_ids': text,
                    'semantic_table_ids':semantic_table,
                }
                outputs = model(**inputs, labels=target)
                if len(args.cudas.split(",")) > 1:
                    loss = torch.sum(outputs.loss,-1) / outputs.loss.shape[-1]
                else:
                    loss = torch.sum(outputs.loss,-1)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_list.append(loss)
                bar.set_description("epoch:{} loss:{}".format(e, loss))
            epoch_loss.append(sum(loss_list)/len(loss_list))
            if e % args.save_every == 0:
                saver(checkpoint_path, model, tokenizer, e)
                
    for i, loss in enumerate(epoch_loss):
        print("epoch{} loss: {}".format(i, loss))
