import os
from posixpath import expanduser
import tqdm
import json
from nlgeval import NLGEval

def strategy_rewrite(input_string):
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
        if string not in string_set and string != '':
            res_string += string + '. '
            string_set.add(string)
    return res_string
    
def train_output2input_extra(ori_path, gen_path, input_path, size, eval, point='ROUGE_L'):
    #average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    with open(gen_path + "_generation_json", 'r') as f_hypothesis:
        input_list = []
        for i in range(size):
            max_value = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
            with open(ori_path + "/" + str(i)) as f_references:
                hypothesis = json.loads(f_hypothesis.readline().strip())
                for j in range(len(hypothesis)):
                    hypothesis[j] = strategy_rewrite(hypothesis[j].split("<|endoftext|>")[0].strip())
                references = f_references.readline().strip()
                max_index = -1
                for k, h in enumerate(hypothesis):
                    #print(h)
                    metrics_dict = eval.compute_individual_metrics([references], h)

                    if metrics_dict[point] > max_value[point]:
                        max_value[point] = metrics_dict[point]
                        max_index = k
                input_list.append(hypothesis[max_index].strip() + '\n')
        
        for i in range(size):
            f_input = open(input_path + '/' + str(i), 'w')
            if len(input_list[i]) != 0:
                f_input.write(input_list[i])
            else:
                f_input.write('<|endoftext|> ')    

def train_output2input(ori_path, gen_path, input_path, size, eval, point='ROUGE_L'):
    #average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    with open(gen_path + "_generation_json", 'r') as f_hypothesis:
        input_list = []
        for i in range(size):
            max_value = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
            with open(ori_path + "/" + str(i)) as f_references:
                hypothesis = json.loads(f_hypothesis.readline().strip())
                for j in range(len(hypothesis)):
                    hypothesis[j] = strategy_rewrite(hypothesis[j].split("<|endoftext|>")[0].strip())
                    #hypothesis[i] = hypothesis[i]
                references = f_references.readline().strip()
                max_index = -1
                for k, h in enumerate(hypothesis):
                    #print(h)
                    metrics_dict = eval.compute_individual_metrics([references], h)

                    if metrics_dict[point] > max_value[point]:
                        max_value[point] = metrics_dict[point]
                        max_index = k
                input_list.append(hypothesis[max_index] + ' ' + references)
        
        for i in range(size):
            f_input = open(input_path + '/' + str(i), 'w')
            if len(input_list[i]) != 0:
                f_input.write(input_list[i])
            else:
                f_input.write('<|endoftext|> ')
            
def test_output2input(out_path, in_path):
    with open(out_path, 'r') as fo:
        index = 0
        if not os.path.exists(in_path):
            os.makedirs(in_path)
        for line in fo:
            g_list = json.loads(line)
            with open(in_path + '/'+ str(index), 'w') as fi:
                write_string = strategy_rewrite(g_list[0].split('<|endoftext|>')[0])
                if len(write_string) != 0:
                    fi.write(write_string)
                else:
                    fi.write('<|endoftext|> ')
            index += 1


def train_output2input_with_origin(ori_input_path, ori_path, gen_path, input_path, size, eval, point='ROUGE_L'):
    #average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    with open(gen_path + "_generation_json", 'r') as f_hypothesis:
        input_list = []
        for i in range(size):
            max_value = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
            with open(ori_path + "/" + str(i)) as f_references:
                f_ori_input = open(ori_input_path + '/' + str(i))
                ori_input = f_ori_input.readline().strip()
                hypothesis = json.loads(f_hypothesis.readline().strip())
                for j in range(len(hypothesis)):
                    hypothesis[j] = strategy_rewrite(hypothesis[j].split("<|endoftext|>")[0].strip())
                    #hypothesis[i] = hypothesis[i]
                references = f_references.readline().strip()
                max_index = -1
                for k, h in enumerate(hypothesis):
                    #print(h)
                    metrics_dict = eval.compute_individual_metrics([references], h)

                    if metrics_dict[point] > max_value[point]:
                        max_value[point] = metrics_dict[point]
                        max_index = k
                #input_list.append(hypothesis[max_index] + ' ' + references)
                input_list.append(ori_input + ' draft ' + hypothesis[max_index] + ' ' + references)
        
        for i in range(size):
            f_input = open(input_path + '/' + str(i), 'w')
            if len(input_list[i]) != 0:
                f_input.write(input_list[i])
            else:
                f_input.write('<|endoftext|> ')
            
def test_output2input_with_origin(ori_input_path, out_path, in_path):
    with open(out_path, 'r') as fo:
        index = 0
        if not os.path.exists(in_path):
            os.makedirs(in_path)
        for line in fo:
            foi = open(ori_input_path + '/' + str(index), 'r')
            oi = foi.readline().strip()
            g_list = json.loads(line)
            with open(in_path + '/'+ str(index), 'w') as fi:
                write_string = oi + ' draft ' + strategy_rewrite(g_list[0].split('<|endoftext|>')[0])
                if len(write_string) != 0:
                    fi.write(write_string)
                else:
                    fi.write('<|endoftext|> ')
            index += 1

            
def val_output2input_with_origin(ori_input_path, out_path, gold_path, in_path):
    with open(out_path, 'r') as fo:
        #fg = open(gold_path, 'r')
        index = 0
        if not os.path.exists(in_path):
            os.makedirs(in_path)
        for line in fo:
            foi = open(ori_input_path + '/' + str(index), 'r')
            oi = foi.readline().strip()
            fg = open(gold_path + '/'+ str(index), 'r')
            gold = fg.readline().strip()
            g_list = json.loads(line)
            with open(in_path + '/'+ str(index), 'w') as fi:
                write_string = oi + ' draft ' + strategy_rewrite(g_list[0].split('<|endoftext|>')[0])
                if len(write_string) != 0:
                    fi.write(write_string + ' ' + gold)
                else:
                    fi.write('<|endoftext|> ' + gold)
            index += 1
'''

def train_output2input_with_origin(ori_input_path, ori_path, gen_path, input_path, size, eval, point='ROUGE_L'):
    #average_metrics_dict = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
    with open(gen_path + "_generation_json", 'r') as f_hypothesis:
        input_list = []
        for i in range(size):
            max_value = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0,}
            with open(ori_path + "/" + str(i)) as f_references:
                f_ori_input = open(ori_input_path + '/' + str(i))
                ori_input = f_ori_input.readline().strip()
                hypothesis = json.loads(f_hypothesis.readline().strip())
                for j in range(len(hypothesis)):                    
                    hypothesis[j] = hypothesis[j].split("<|endoftext|>")[0].strip()
                    #hypothesis[i] = hypothesis[i]
                references = f_references.readline().strip()
                max_index = -1
                for k, h in enumerate(hypothesis):
                    #print(h)
                    metrics_dict = eval.compute_individual_metrics([references], h)

                    if metrics_dict[point] > max_value[point]:
                        max_value[point] = metrics_dict[point]
                        max_index = k
                input_list.append(ori_input + ' ' + hypothesis[max_index] + ' ' + references)
        
        for i in range(size):
            f_input = open(input_path + '/' + str(i), 'w')
            f_input.write(input_list[i] + '\n')

def test_output2input_with_origin(ori_input_path, out_path, in_path):
    with open(out_path, 'r') as fo:
        index = 0
        for line in fo:
            foi = open(ori_input_path + '/' + str(index), 'r')
            oi = foi.readline().strip()
            g_list = json.loads(line)
            with open(in_path + str(index), 'w') as fi:
                fi.write(oi + ' ' + g_list[0].split('<|endoftext|>')[0])
            index += 1
'''
if __name__ == "__main__":
    #args = parser.parse_args()
    eval = NLGEval(no_skipthoughts=True, no_glove=True)
    train_output2input('./data/TD_train_rewrite_gold', '../data/TD_train_generated_345M_3200', './data/TD_train_output_input' ,1084, eval)
    #train_output2input_with_origin('./data/TD_train_rewrite_input', './data/TD_train_rewrite_gold', '../data/TD_train_generated_345M_3200', './data/TD_train_output_input_with_origin' ,1084, eval)
    #test_output2input('../data/TD_val_generated_345M_3200_generation_json', './data/TD_val_generated_output_345M_3200/')
    #test_output2input_with_origin('../data/TD_test_input', '../data/TD_test_generated_345M_3200_generation_json', './data/TD_test_generated_output_345M_with_origin_3200/')
    
