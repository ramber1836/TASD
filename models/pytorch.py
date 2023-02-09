from transformers import AutoModel, AutoTokenizer
import json

AutoModel.from_pretrained("gpt2").save_pretrained("pytorch/numericNLG/gpt2-small")
AutoModel.from_pretrained("gpt2-medium").save_pretrained("pytorch/numericNLG/gpt2-medium")
AutoModel.from_pretrained("gpt2-large").save_pretrained("pytorch/numericNLG/gpt2-large")
AutoTokenizer.from_pretrained("gpt2").save_pretrained("pytorch/numericNLG/gpt2-small")
AutoTokenizer.from_pretrained("gpt2-medium").save_pretrained("pytorch/numericNLG/gpt2-medium")
AutoTokenizer.from_pretrained("gpt2-large").save_pretrained("pytorch/numericNLG/gpt2-large")

AutoModel.from_pretrained("gpt2").save_pretrained("pytorch/Totto/gpt2-small")
AutoModel.from_pretrained("gpt2-medium").save_pretrained("pytorch/Totto/gpt2-medium")
AutoModel.from_pretrained("gpt2-large").save_pretrained("pytorch/Totto/gpt2-large")
AutoTokenizer.from_pretrained("gpt2").save_pretrained("pytorch/Totto/gpt2-small")
AutoTokenizer.from_pretrained("gpt2-medium").save_pretrained("pytorch/Totto/gpt2-medium")
AutoTokenizer.from_pretrained("gpt2-large").save_pretrained("pytorch/Totto/gpt2-large")

config_json = {
    "numericNLG":{
        "n_stx":122,
      "n_ttx_row":56,
      "n_ttx_col":24,
      "num_head":3
    },
    
    "Totto":{
        "n_stx": 145,
        "n_ttx_row": 8,
        "n_ttx_col": 8
    }
}

for i in ["small", "medium", "large"]:
    for j in ["numericNLG", "Totto"]:
        with open(f"pytorch/{j}/gpt2-{i}/config.json", "r") as r:
            config = json.load(r)
        for k in ["n_stx", "n_ttx_row", "n_ttx_col"]:
            config[k] = config_json[j][k]
        with open(f"pytorch/{j}/gpt2-{i}/config.json", "w") as w:
            json.dump(config, w)