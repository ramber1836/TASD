import argparse
from transformers import AutoTokenizer
from load_numericNLG import load_numericNLG_tables, load_numericNLG_dataset
from load_Totto import load_Totto_tables, load_Totto_dataset
import torch

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', default="medium", type=str, help='')
    parser.add_argument('--beam_num', default=1, type=int, help='')
    parser.add_argument('--epochs', default=31, type=int, help='')
    parser.add_argument('--save_every', default=5, type=int, help='')
    parser.add_argument('--generate_length', default=10, type=int, help='')
    parser.add_argument('--turn', default="first", type=str, help='')
    parser.add_argument('--lr', default=1e-5, type=float, help='')
    parser.add_argument('--data_name', default="numericNLG", type=str, help='')
    parser.add_argument('--table', default="NT", type=str, help='')

    args = parser.parse_args()
    print("Process the data")
    print(args)

    gold_root_dir = "../../data/{}".format(args.data_name)

    tokenizer = AutoTokenizer.from_pretrained("afs/{}/model/gpt2-{}".format(args.data_name, args.model_size))
    
    if args.table == "T":
        if "numericNLG" in args.data_name:
            train_chunks_tables = load_numericNLG_tables(tokenizer, '{}/table_train.json'.format(gold_root_dir)) #[1084*56*24*122], [1084*56*24]
            val_chunks_tables = load_numericNLG_tables(tokenizer, '{}/table_val.json'.format(gold_root_dir))
            test_chunks_tables = load_numericNLG_tables(tokenizer, '{}/table_test.json'.format(gold_root_dir))
        elif "Totto" in args.data_name:
            train_chunks_tables = load_Totto_tables(tokenizer, '{}/table_train.json'.format(gold_root_dir)) #[1084*56*24*122], [1084*56*24]
            val_chunks_tables = load_Totto_tables(tokenizer, '{}/table_val.json'.format(gold_root_dir))
            test_chunks_tables = load_Totto_tables(tokenizer, '{}/table_test.json'.format(gold_root_dir))
        else:
            print("Error")
        torch.save(train_chunks_tables, '{}/table_train.pth'.format(gold_root_dir))
        torch.save(val_chunks_tables, '{}/table_val.pth'.format(gold_root_dir))
        torch.save(test_chunks_tables, '{}/table_test.pth'.format(gold_root_dir))


    if args.turn == "rewrite":
        input_root_dir = "afs/{}/rewrite/{}_epochs{}_save{}_beam{}_generate{}_lr{}_{}".format(args.data_name, args.model_size, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr, args.table)
    else:
        input_root_dir = gold_root_dir

    if "numericNLG" == args.data_name:
        chunks_train = load_numericNLG_dataset(tokenizer, "{}/TD_train_input".format(input_root_dir), "{}/TD_train_gold".format(gold_root_dir))#[1084*1024]
        chunks_val = load_numericNLG_dataset(tokenizer, "{}/TD_val_input".format(input_root_dir), "{}/TD_val_gold".format(gold_root_dir))#[1084*1024]
        chunks_test = load_numericNLG_dataset(tokenizer, "{}/TD_test_input".format(input_root_dir), "{}/TD_test_gold".format(gold_root_dir))#[1084*1024]    
        chunks_train_split = load_numericNLG_dataset(tokenizer, "{}/TD_train_input".format(input_root_dir), "{}/TD_train_gold".format(gold_root_dir), split=True)#[1084*1024]    
    else:
        chunks_train = load_Totto_dataset(tokenizer, "{}/TD_train_input".format(input_root_dir), "{}/TD_train_gold".format(gold_root_dir))#[1084*1024]
        chunks_val = load_Totto_dataset(tokenizer, "{}/TD_val_input".format(input_root_dir), "{}/TD_val_gold".format(gold_root_dir))#[1084*1024]
        chunks_test = load_Totto_dataset(tokenizer, "{}/TD_test_input".format(input_root_dir), "{}/TD_test_gold".format(gold_root_dir))#[1084*1024]    
        chunks_train_split = load_Totto_dataset(tokenizer, "{}/TD_train_input".format(input_root_dir), "{}/TD_train_gold".format(gold_root_dir), split=True)#[1084*1024] 

    torch.save(chunks_train_split, "{}/text_train_split.pth".format(input_root_dir))
    torch.save(chunks_val, "{}/text_val.pth".format(input_root_dir))
    torch.save(chunks_test, "{}/text_test.pth".format(input_root_dir))
    torch.save(chunks_train, "{}/text_train.pth".format(input_root_dir))

