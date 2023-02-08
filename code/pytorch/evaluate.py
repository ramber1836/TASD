import argparse
import os
from nlgeval import NLGEval
from nlgeval import compute_metrics
import json
import pandas as pd
import math
# from bert_score import score
# from table_text_eval import table_text_eval

def strategy(input_string):
    while len(input_string) > 0 and input_string[0] in ["\n"]:
        input_string = input_string[1:]
        if len(input_string) == 0:
            return ""
    input_string = input_string.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
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

def geo_ave(l):
    for i in l:
        if i == 0:
            return 0
    return math.pow(math.e, sum([math.log(i) for i in l]) / len(l))
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="test", type=str, help='')
    parser.add_argument('--epochs', default=21, type=int, help='')
    parser.add_argument('--save_every', default=1, type=int, help='')
    parser.add_argument('--beam_num', default=2, type=int, help='')
    parser.add_argument('--generate_length', default=256, type=int, help='')
    parser.add_argument('--model_size', default="medium", type=str, help='')
    parser.add_argument('--turn', default="first", type=str, help='')
    parser.add_argument('--data_name', default="numericNLG", type=str, help='')
    parser.add_argument('--table', default="NT", type=str, help='')
    parser.add_argument('--lr', default=1e-5, type=float, help='')
    parser.add_argument('--start_epoch', default=20, type=int, help='')
    parser.add_argument('--end_epoch', default=21, type=int, help='')
    args = parser.parse_args()
    print(args)
    root_dir = "afs/{}".format(args.data_name)
    reference_path = "../../data/{}/TD_{}_gold".format(args.data_name, args.mode)
    if not os.path.exists("{}/bleu".format(root_dir)):
        os.mkdir("{}/bleu".format(root_dir))
    if not os.path.exists("{}/bleu/{}".format(root_dir, args.mode)):
        os.mkdir("{}/bleu/{}".format(root_dir, args.mode))
    if not os.path.exists("{}/bleu/{}/{}".format(root_dir, args.mode, args.model_size)):
        os.mkdir("{}/bleu/{}/{}".format(root_dir, args.mode, args.model_size))
    if not os.path.exists("{}/bleu/{}/{}/{}".format(root_dir, args.mode, args.model_size, args.turn)):
        os.mkdir("{}/bleu/{}/{}/{}".format(root_dir, args.mode, args.model_size, args.turn))
    if not os.path.exists("{}/bleu/{}/{}/{}/epochs{}_save{}".format(root_dir, args.mode, args.model_size, args.turn, args.epochs, args.save_every)):
        os.mkdir("{}/bleu/{}/{}/{}/epochs{}_save{}".format(root_dir, args.mode, args.model_size, args.turn, args.epochs, args.save_every))
    result = open("./{}/bleu/{}/{}/{}/epochs{}_save{}/beam{}_generate{}_lr{}_{}_json".format(root_dir, args.mode, args.model_size, args.turn, args.epochs, args.save_every, args.beam_num, args.generate_length, args.lr, args.table), "w")
    print("the bleu result will be written into {}".format(result))
    print("reference_path is {}".format(reference_path))

    tables_json = json.load(open("../../data/{}/table_{}.json".format(args.data_name, args.mode), "r"))
    tables = []
    for i , table in enumerate(zip(*tables_json)):
        records = []
        for j, rows in enumerate(zip(*table)):
            for k, columns in enumerate(zip(*rows)):
                if args.data_name == "Totto":
                    records.append([columns[0].split(' '), columns[1].split(' ')])
                if args.data_name == "numericNLG":
                    if columns[5] != 0:
                        records.append([columns[0].split(' ')+columns[1].split(' ')+columns[2].split(' ')+columns[4].split(' '), columns[0].split(' ')+columns[1].split(' ')+columns[2].split(' ')+columns[4].split(' ')])

        tables.append(records)


    if args.mode == "train":
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
    
    all_candidates = []
    references = []
    for _, _, nums in os.walk(reference_path):
        for i in range(len(nums)):
            with open(reference_path + "/" + str(i), "r") as f:
                reference = f.readline()
            references.append(reference)

    for e in range(args.start_epoch, args.end_epoch, args.save_every):
        if args.mode == "train" and e != int(best_model_idx):
            continue
        candidate_path = "./{}/generate/{}/{}/{}/epochs{}_save{}_model{}_beam{}_generate{}_lr{}_{}_json".format(root_dir, args.mode, args.model_size, args.turn, args.epochs, args.save_every, e, args.beam_num, args.generate_length, args.lr, args.table)
        # candidate_path = "afs/numericNLG/generate/test/temp_json"
        # candidate_path = "./{}/generate/{}/{}/{}/epochs{}_save{}_model{}_beam{}_gengeate{}_lr{}_json".format(root_dir, args.mode, args.model_size, args.turn, args.epochs, args.save_every, e, args.beam_num, args.generate_length, args.lr)
        with open(candidate_path, "r") as f:
            candidates_raw = f.readlines()
        candidates = []
        for i, candidate in enumerate(candidates_raw):
            candidates.append([strategy(j.split("<|endoftext|>")[0].strip()) for j in json.loads(candidate)])
        all_candidates.append(candidates)

    # all_bert_score = []
    # for i in zip(*sum(all_candidates,[])):
    #     all_bert_score.append(score(list(i), references * len(all_candidates), lang="en", verbose=True, device="cuda")[2])

    for e in range(args.start_epoch, args.end_epoch, args.save_every):
        if args.mode == "train" and e != int(best_model_idx):
            continue
        if args.mode == "train":
            # myBERTScore = all_bert_score
            candidates = all_candidates[0]
        else:
            # myBERTScore = [i[e//args.save_every*len(all_candidates[0]):(e//args.save_every+1)*len(all_candidates[0])] for i in all_bert_score]
            candidates = all_candidates[(e - args.start_epoch)//args.save_every]
        average_metric = {'Bleu_1': 0, 'Bleu_2': 0, 'Bleu_3': 0, 'Bleu_4': 0, 'ROUGE_L': 0, 'BLEU': 0}
        metrics_dict = {'Bleu_1': [], 'Bleu_2': [], 'Bleu_3': [], 'Bleu_4': [], 'ROUGE_L': [], 'BLEU': []}
        metrics = []
        idxs = []
        eval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR", "CIDEr"])

        for i, candidate in enumerate(candidates):
            metric = []
            reference = references[i]
            idx = 0
            for k ,j in enumerate(candidate):
                metric_j = eval.compute_individual_metrics([reference], j)
                # if len(tables[i]) == 0:
                if len(tables[i]) >= 0:
                    parent_score = 0
                else:
                    _, _, parent_score, _ = table_text_eval.parent([j.split()], [reference.split()],[tables[i]],lambda_weight=None)
                BP = math.pow(math.e, int(1- len(reference) / (len(j) + 1))) if len(j) < len(reference) else 1
                metric_j["BLEU"] = BP * geo_ave([metric_j["Bleu_1"], metric_j["Bleu_2"], metric_j["Bleu_3"], metric_j["Bleu_4"]])
                metric_j["BERTScore_F1"] = 0
                # metric_j["BERTScore_F1"] = float(myBERTScore[k][i])
                metric_j["PARENT"] = parent_score
                metric.append(metric_j)
                # idx = k if metric[k]["BLEU"] > metric[idx]["BLEU"] else idx
            metrics.append(metric)
            idxs.append(idx)

        metrics_dict = {key: [metrics[i][j][key] for i, j in enumerate(idxs)] for key in average_metric}
        metrics_dict['idx'] = idxs 
        average_metric =  {key: round(sum(metrics_dict[key])/len(idxs) * 100,2) for key in average_metric}
        print(average_metric)
        result.write("{}\t{}\n".format(e, json.dumps(average_metric)))
        result.flush()
        metrics_csv_path = "./{}/bleu/{}/{}/{}/epochs{}_save{}/model{}_beam{}_generate{}_lr{}_{}.csv".format(root_dir, args.mode, args.model_size, args.turn, args.epochs, args.save_every, e, args.beam_num, args.generate_length, args.lr, args.table)
        df = pd.DataFrame(metrics_dict)
        df.to_csv(metrics_csv_path)