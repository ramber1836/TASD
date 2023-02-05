#!/usr/bin/env python3

import os
import argparse
from posixpath import expanduser
import tqdm
import json
from nlgeval import NLGEval
from utils import get_median
from table_generate_conditional_samples import inference_model as inference_model_table
from generate_conditional_samples import inference_model as inference_model

parser = argparse.ArgumentParser(
    description='Evaluate fine-tuned GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--merge', default=False, help='merge.')
parser.add_argument('--save_every', type=int, default=400, help='Pretrained model name')
parser.add_argument('--epochs', type=int, default=2000, help='Pretrained model name')

parser.add_argument('--ori_path_val', type=str, default=False, help='merge.')
parser.add_argument('--ori_path_test', type=str, default=False, help='merge.')
parser.add_argument('--ckpt_dir', type=str, default=False, help='merge.')

parser.add_argument('--gen_path', type=str, default='', help='merge.')
parser.add_argument('--type', type=str, default='val', help='merge.')
parser.add_argument('--metric', type=str, default='ROUGE_L', help='merge.')

parser.add_argument('--model_name', type=str, default='', help='Pretrained model name')
parser.add_argument('--data_dir', type=str, default='./afs/models', help='Path to models directory')
parser.add_argument('--out_dir', type=str, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--table', type=bool, default=False, help='use table representation or not.')
parser.add_argument('--header_num', type=int, default=3, help='header number')
parser.add_argument('--mode', type=str, default='ori', help='mode')


def strategy(input_string):
    if len(input_string) == 0:
        return input_string
    while input_string[0] == "\n":
        input_string = input_string[1:]
        if len(input_string) == 0:
            return ""
    split_string = input_string.split('\n')
    string_set = set()
    res_string = ""
    for string in split_string:
        if string not in string_set:
            res_string += string
            string_set.add(string)
    split_string = res_string.split('. ')
    string_set = set()
    res_string = ""
    for string in split_string:
        if string not in string_set:
            res_string += string + '. '
            string_set.add(string)
    return res_string

def evaluate(ori_path, gen_path, size, eval):
    average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
    with open(gen_path + "_generation_json", 'r') as f_hypothesis:
        for i in range(size):
            max_value = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
            with open(ori_path + "/" + str(i)) as f_references:
                hypothesis = json.loads(f_hypothesis.readline().strip())
                for j in range(len(hypothesis)):
                    hypothesis[j] = hypothesis[j].split("<|endoftext|>")[0].strip()
                    #hypothesis[i] = hypothesis[i]
                references = f_references.readline().strip()
                for h in hypothesis:
                    #print(h)
                    metrics_dict = eval.compute_individual_metrics([references], h)
                    for key in max_value.keys():
                        if metrics_dict[key] > max_value[key]:
                            max_value[key] = metrics_dict[key]
            for key in average_metrics_dict.keys():
                average_metrics_dict[key] += (max_value[key] / float(size))
    print(average_metrics_dict)            

def evaluate_strategy(ori_path, gen_path, size, eval, metric="METEOR"):
    average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0}
    metrics_dict_list = {'Bleu_1': [], 'Bleu_2': [], 'Bleu_3': [], 'Bleu_4': [], 'METEOR': [], 'ROUGE_L': []}
    keys = list(average_metrics_dict.keys())
    with open(gen_path + "_generation_json", 'r') as f_hypothesis:
        for i in range(size):
            max_value = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0}
            with open(ori_path + "/" + str(i)) as f_references:
                hypothesis = json.loads(f_hypothesis.readline().strip())
                for j in range(len(hypothesis)):
                    hypothesis[j] = hypothesis[j].split("<|endoftext|>")[0].strip()
                    if len(hypothesis[j]) == 0:
                        continue
                    hypothesis[j] = strategy(hypothesis[j])
                    #hypothesis[i] = hypothesis[i]
                references = f_references.readline().strip()
                for h in hypothesis:
                    #print(h)
                    metrics_dict = eval.compute_individual_metrics([references], h)
                    if metrics_dict[metric] > max_value[metric]:
                        max_value = metrics_dict
            for key in keys:
                average_metrics_dict[key] += (max_value[key] / float(size))
                metrics_dict_list[key].append(max_value[key])
        for key in keys:
            average_metrics_dict[key + '_median'] = get_median(metrics_dict_list[key])
    return average_metrics_dict


if __name__ == "__main__":
    #args = parser.parse_args()
    args = parser.parse_args()
    eval = NLGEval(no_skipthoughts=True, no_glove=True)
    if args.mode == 'ori':
        metric_max = 0
        index_max = -1
        for i in range(1, args.epochs // args.save_every + 1):
        #for i in range(1, 14):
            average_metrics_dict = evaluate_strategy('./afs/data/'+args.ori_path_val, './afs/data/'+args.gen_path + '_' + str(args.save_every * i) ,136, eval)
            print(average_metrics_dict)
            if average_metrics_dict[args.metric] > metric_max:
                metric_max = average_metrics_dict[args.metric]
                index_max = i
        if args.table is True:
            inference_model_table('-1', ckpt_dir='afs/checkpoint/'+args.ckpt_dir, ckpt_list=[str(args.save_every * index_max)], model_name=args.model_name,
                data_dir='afs/data/'+args.data_dir, out_dir='afs/data/'+args.out_dir, table_path='afs/data/table_test.json', header_num=args.header_num)
        else:
            inference_model('-1', ckpt_dir='afs/checkpoint/'+args.ckpt_dir, ckpt_list=[str(args.save_every * index_max)], model_name=args.model_name,
                data_dir='afs/data/'+args.data_dir, out_dir='afs/data/'+args.out_dir)
        test_average_metrics_dict = evaluate_strategy('./afs/data/'+args.ori_path_test, './afs/data/'+args.out_dir + '_' + str(args.save_every * index_max) ,135, eval)
        with open('afs/data/'+'result_'+args.out_dir, 'w') as f:
            f.write(json.dumps(test_average_metrics_dict))
    elif args.mode == 'eva_mid':
        metric_max = 0
        index_max = -1
        for i in range(1, args.epochs // args.save_every + 1):
        #for i in range(1, 14):
            average_metrics_dict = evaluate_strategy('./afs/data/'+args.ori_path_val, './afs/data/'+args.gen_path + '_' + str(args.save_every * i) ,136, eval)
            print(average_metrics_dict)
            if average_metrics_dict[args.metric] > metric_max:
                metric_max = average_metrics_dict[args.metric]
                index_max = i
        test_average_metrics_dict = evaluate_strategy('./afs/data/'+args.ori_path_test, './afs/data/'+args.out_dir + '_' + str(args.save_every * index_max) ,135, eval)
        with open('afs/data/'+'result_'+args.out_dir, 'w') as f:
            f.write(json.dumps(test_average_metrics_dict))

    #evaluate_strategy('./data/test_gold', './rewrite/data/TD_test_rewrite_generated_345M_2400', './data/evaluate' ,135, eval)
    