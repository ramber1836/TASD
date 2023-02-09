import paddle
from modeling_tasd import GPTForPretraining
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer
import argparse
import os
import pickle
import json
import os
from utils import encode, mkdir_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned GPT-2 on your custom dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path', default="afs/numericNLG/checkpoint/rewrite_gpt2-en_30_3_1e-6/17", type=str, help='')
    parser.add_argument('--table_data_path', default="afs/numericNLG/data/table_data/tokens_test.pkl", type=str, help='')
    parser.add_argument('--data_path', default='afs/numericNLG/data/origin/TD_test_input', type=str, help='')
    parser.add_argument('--generate_path', default='afs/numericNLG/generated_result/rewrite_gpt2-en_30_3_1e-6/17/test.out', type=str, help='')
    args = parser.parse_args()
    print(args)
    mkdir_files(args.generate_path)

    paddle.device.set_device("gpu:0")

    tokenizer = GPTTokenizer.from_pretrained(args.checkpoint_path)
    model = GPTForPretraining.from_pretrained(args.checkpoint_path, eol_token_id=tokenizer.eol_token_id)

    test_chunks_tables = pickle.load(open(args.table_data_path, "rb"))

    pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    test_path_list = []
    test_list = []
    test_list_table = []
    for (dirpath, _, fnames) in os.walk(args.data_path):
        for index, fname in enumerate(fnames):
            test_path_list.append(os.path.join(dirpath, str(index)))
            test_list_table.append(test_chunks_tables[0][index])
    for path, table_item in zip(test_path_list, test_list_table):
        with open(path, 'r', encoding='utf-8') as f_input:
            test_text_input = f_input.readline()
            if test_text_input == '':
                test_text_input = " "
            tokens_test_text = encode(test_text_input, tokenizer)
            table_item = paddle.unsqueeze(paddle.to_tensor(table_item), axis=0)
            test_list.append((tokens_test_text, table_item))
    model.eval()
    fg = open(args.generate_path ,'w')
    for test_input in test_list:
        test_input_ids, test_table_ids = test_input
        if len(test_input_ids) > 1024:
            fg.write(json.dumps([""] * 5) + '\n')
            fg.flush()
            continue
        ids, scores = model.generate(
            input_ids=test_input_ids,
            table_ids = test_table_ids,
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
        fg.write(json.dumps(response) + '\n')
        fg.flush()


