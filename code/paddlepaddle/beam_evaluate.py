import os
from posixpath import expanduser
import tqdm
import json
from nlgeval import NLGEval
import argparse
from utils import mkdir_files

def evaluate(ori_path, gen_path, size, eval):
    average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
    with open(gen_path, 'r') as f_hypothesis:
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

def evaluate_strategy(ori_path, gen_path, eval, evaluate_path):
    average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
    for i, hypothesis in enumerate(open(gen_path, 'r').readlines()):
        max_value = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
        with open(ori_path + "/" + str(i)) as f_references:
            hypothesis = json.loads(hypothesis.strip())
            # if hypothesis[0] == "" and hypothesis[-1] == "":
            #     size -= 1
            #     continue
            for j in range(len(hypothesis)):
                hypothesis[j] = hypothesis[j].split("<|endoftext|>")[0].strip()
                if len(hypothesis[j]) == 0:
                    continue
                if hypothesis[j][0] == '\n':
                    hypothesis[j] = hypothesis[j][1:]
                hypothesis[j] = hypothesis[j].split('\n')[0]
            references = f_references.readline().strip()
            for h in hypothesis:
                metrics_dict = eval.compute_individual_metrics([references], h)
                # print(metrics_dict)
                for key in max_value.keys():
                    if metrics_dict[key] > max_value[key]:
                        max_value[key] = metrics_dict[key]
        for key in average_metrics_dict.keys():
            average_metrics_dict[key] += max_value[key]
    for key in average_metrics_dict.keys():
        average_metrics_dict[key] /= float(i + 1)
    
    with open(evaluate_path, "w") as w:
        w.write(json.dumps(average_metrics_dict))
    print(average_metrics_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned GPT-2 on your custom dataset.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--groundtruth_path', default='afs/numericNLG/data/origin/TD_test_gold', type=str, help='')
    parser.add_argument('--generate_path', default='afs/Totto/generated_result/online/epoch_30_gpt2_small_run_extra_1e-6_30_3', type=str, help='')
    parser.add_argument('--evaluate_path', default='afs/numericNLG/evaluated_result/test.metric', type=str, help='')
    args = parser.parse_args()
    mkdir_files(args.evaluate_path)
    eval = NLGEval(no_skipthoughts=True, no_glove=True)
    print(f"==================== start evaluate {args.generate_path} ====================")
    #evaluate('./data/TD_val_gold', './rewrite/data/TD_val_rewrite_generated_345M_with_origin_' + str(i * 400), './data/evaluate' ,135, eval)
    evaluate_strategy(args.groundtruth_path, args.generate_path, eval, args.evaluate_path)
    print("==================== evaluate finished. ====================")
