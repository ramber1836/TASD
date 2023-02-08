import paddle
from modeling_tasd import GPTForPretraining
#from paddlenlp.transformers.gpt.modeling import GPTForPretraining
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer
import time
import argparse
from tqdm import tqdm
from paddlenlp.data import Dict, Pad
from paddlenlp.datasets import load_dataset
import os
import pickle
import json
import os
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.gpt.modeling import GPTPretrainingCriterion
import numpy as np
from utils import save_model, attention_mask, mkdir_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned GPT-2 on your custom dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=30, type=int, help='')
    parser.add_argument('--every', default=3, type=int, help='')
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='')
    parser.add_argument('--model_type', default="afs/numericNLG/models/gpt2-en", type=str, help='')
    parser.add_argument('--checkpoint_path', default="afs/numericNLG/checkpoint/rewrite_gpt2-en_30_3_1e-6", type=str, help='')
    parser.add_argument('--pickle_file_path', default="afs/numericNLG/data/table_data/tokens_train.pkl", type=str, help='')
    parser.add_argument('--input_data_path', default='afs/numericNLG/data/gpt2-en_30_3_1e-6/TD_train_input', type=str, help='')
    parser.add_argument('--gold_data_path', default='afs/numericNLG/data/origin/TD_train_gold', type=str, help='')
    args = parser.parse_args()

    paddle.device.set_device("gpu:0")

    tokenizer = GPTTokenizer.from_pretrained(args.model_type)
    model = GPTForPretraining.from_pretrained(args.model_type, eol_token_id=tokenizer.eol_token_id)

    mkdir_files(args.checkpoint_path)

    pickle_file = open(args.pickle_file_path, "rb")
    train_chunks_tables = pickle.load(pickle_file)
    pickle_file.close()

    pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print("table data is loaded.")
    train_tuple_list = []
    path_list_input =[]
    path_list_gold = []
    list_table = []
    for (dirpath, _, fnames) in os.walk(args.input_data_path):
        for index, fname in enumerate(fnames):
            path_list_input.append(os.path.join(dirpath, str(index)))
            path_list_gold.append(os.path.join(args.gold_data_path, str(index)))
            list_table.append(train_chunks_tables[0][index])

    for path_ipnut, path_gold, table_item in zip(path_list_input, path_list_gold, list_table):
        with open(path_ipnut, 'r', encoding='utf-8') as f_input:
            f_gold = open(path_gold, 'r', encoding='utf-8')
            ori_text_input = f_input.readline()
            ori_text_gold = f_gold.readline()
            tokens_ori_text = tokenizer(ori_text_input + ' ' + ori_text_gold)["input_ids"]
            len_ori_text = len(tokens_ori_text)
            if len_ori_text > 1023:
                start = 0
                end = 1024
                while end < len_ori_text:
                    input_ids = paddle.to_tensor([tokens_ori_text[start:end]])
                    labels = paddle.to_tensor([tokens_ori_text[start + 1:end + 1]])
                    table_item_tensor = paddle.unsqueeze(paddle.to_tensor(table_item), axis=0)
                    train_tuple_list.append((input_ids, labels, table_item_tensor))
                    start += 512
                    end += 512
                if start >= len_ori_text:
                    continue
                input_ids = paddle.to_tensor([tokens_ori_text[len_ori_text - 1024:len_ori_text]])
                labels = paddle.to_tensor([tokens_ori_text[len_ori_text - 1023:len_ori_text] + [pad_token_id]])
                table_item_tensor = paddle.unsqueeze(paddle.to_tensor(table_item), axis=0)
                train_tuple_list.append((input_ids, labels, table_item_tensor))
                continue

            d = tokenizer(ori_text_input + ' ' + ori_text_gold)
            d["input_ids"] = tokens_ori_text + [pad_token_id] * (1024 - len_ori_text)
            d["labels"] = d["input_ids"][1:] + [pad_token_id]
            input_ids = paddle.to_tensor([d["input_ids"]])
            labels = paddle.to_tensor([d["labels"]])
            table_item_tensor = paddle.unsqueeze(paddle.to_tensor(table_item), axis=0) # [1,56,24,122]
            train_tuple_list.append((input_ids, labels, table_item_tensor))

    print("model start to run.")
    learning_rate = args.learning_rate
    epochs = args.epochs
    every = args.every
    checkpoint_path = args.checkpoint_path

    num_training_steps = len(train_tuple_list) * epochs
    optimizer = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-7,
            parameters=model.parameters())
    model.train()
    with tqdm(range(epochs)) as bar:
        for epoch in bar:
            epoch_loss = 0.
            start_time = time.time()
            for step, batch in enumerate(train_tuple_list):
                input_ids, labels, table_ids = batch
                position_ids = paddle.to_tensor(list(range(1024)))
                att_mask = attention_mask().reshape([1, 1, 1024, 1024])
                att_mask.stop_gradient = True
                logits = model(input_ids, position_ids, att_mask, table_ids=table_ids)
                #logits = model(input_ids, position_ids, att_mask)
                masked_lm_loss = paddle.nn.CrossEntropyLoss(reduction="none")(logits, labels.unsqueeze(2))
                out = masked_lm_loss.reshape([-1])
                loss = paddle.mean(out)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
            end_time = time.time()
            epoch_loss /= step
            bar.set_description("loss:{}".format(epoch_loss))
            end_time = time.time()
            if epoch % args.every == args.every - 1:
                save_model(os.path.join(checkpoint_path, "%d" % epoch), model, tokenizer)


