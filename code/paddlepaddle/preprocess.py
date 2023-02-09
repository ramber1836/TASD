import pickle
from sys import argv
from load_dataset import load_numericNLG_tables
from load_dataset import load_Totto_tables
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer

dataset = argv[1]

tokenizer = GPTTokenizer.from_pretrained("gpt2-en")

load_tables = {
    "numericNLG": load_numericNLG_tables,
    "Totto": load_Totto_tables
}

for mode in ["test", "val", "train"]:
    chunks_tables = load_tables[dataset](tokenizer, f"../../data/{dataset}/table_{mode}.json")
    pickle.dump(chunks_tables, open(f"../../data/{dataset}/tokens_{mode}.pkl", "wb"))
    print(f"{dataset} {mode} preprocess over")