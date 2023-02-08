import paddle
from modeling_tasd import GPTForPretraining
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer
import time
import argparse
from paddlenlp.data import Dict, Pad
from paddlenlp.datasets import load_dataset
import os
import pickle
import numpy as np
import json
import os
import paddle.nn.functional as F
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.gpt.modeling import GPTPretrainingCriterion
from utils import encode, mkdir_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned GPT-2 on your custom dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path', default="afs/Totto/checkpoint/gpt2-en_30_3_1e-6", type=str, help='')
    parser.add_argument('--evaluated_path', default="afs/Totto/evaluated_result/gpt2-en_30_3_1e-6", type=str, help='')
    parser.add_argument('--pickle_file_path', default="afs/Totto/data/table_data/tokens_train.pkl", type=str, help='')
    parser.add_argument('--data_path', default='afs/Totto/data/origin/TD_train_input', type=str, help='')
    parser.add_argument('--generated_path', default='afs/Totto/data/gpt2-en_30_3_1e-6', type=str, help='')
    args = parser.parse_args()
    mkdir_files(args.generated_path)

    paddle.device.set_device("gpu:0")

    evaluated_result = [json.load(open(f"{args.evaluated_path}/{i}/val.metric", "r")) for i in os.listdir(args.evaluated_path)]

    best_model_id = "0"
    best_evaluation_result = {"Bleu_4":0}

    for i in os.listdir(args.evaluated_path):
        evaluated_result = json.load(open(f"{args.evaluated_path}/{i}/val.metric", "r"))
        best_model_id = i if best_evaluation_result["Bleu_4"] < evaluated_result["Bleu_4"] else best_model_id
        best_evaluation_result["Bleu_4"] = max(best_evaluation_result["Bleu_4"], evaluated_result["Bleu_4"])

    print(f"best model is {best_model_id}")

    tokenizer = GPTTokenizer.from_pretrained(f"{args.checkpoint_path}/{best_model_id}")
    model = GPTForPretraining.from_pretrained(f"{args.checkpoint_path}/{best_model_id}", eol_token_id=tokenizer.eol_token_id)

    #train_chunks_tables = load_tables(tokenizer, './table_train.json')
    #test_chunks_tables = load_tables(tokenizer, './table_test.json')
    #val_chunks_tables = load_tables(tokenizer, './table_val.json')

    pickle_file = open(args.pickle_file_path, "rb")
    train_chunks_tables = pickle.load(pickle_file)
    pickle_file.close()
    pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    train_path_list = []
    train_list = []
    train_list_table = []
    for (dirpath, _, fnames) in os.walk(args.data_path):
        for index, fname in enumerate(fnames):
            train_path_list.append(os.path.join(dirpath, str(index)))
            train_list_table.append(train_chunks_tables[0][index])
    for path, table_item in zip(train_path_list, train_list_table):
        with open(path, 'r', encoding='utf-8') as f_input:
            train_text_input = f_input.readline()
            tokens_train_text = encode(train_text_input, tokenizer)
            table_item = paddle.unsqueeze(paddle.to_tensor(table_item), axis=0)
            train_list.append((tokens_train_text, table_item))
    model.eval()
    
    for i, train_input in enumerate(train_list):
        train_input_ids, train_table_ids = train_input
        fg = open(f"{args.generated_path}/{i}" ,'w')
        if len(train_input_ids) > 1024:
            fg.write(json.dumps([""] * 5) + '\n')
            fg.flush()
            continue
        ids, scores = model.generate(
            input_ids=train_input_ids,
            table_ids = train_table_ids,
            max_length=128, #最大生成文本的长度
            eos_token_id=pad_token_id,
            pad_token_id=pad_token_id,
            decode_strategy="beam_search",
            num_beams=2,
            num_return_sequences=2,
            cache=None
            )
        # print(scores) 
        response = []
        for sequence_ids in ids.numpy().tolist():
            if tokenizer.pad_token_id in sequence_ids:
                sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.pad_token_id)]
            text = tokenizer.convert_ids_to_string(sequence_ids)
            response.append(text)
        fg.write(response[0])
        fg.flush()


