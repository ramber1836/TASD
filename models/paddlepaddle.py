from paddlenlp.transformers import AutoTokenizer, AutoModel
import json

AutoModel.from_pretrained("gpt2-en").save_pretrained("paddlepaddle/numericNLG/gpt2-en")
AutoTokenizer.from_pretrained("gpt2-en").save_pretrained("paddlepaddle/numericNLG/gpt2-en")

AutoModel.from_pretrained("gpt2-en").save_pretrained("paddlepaddle/Totto/gpt2-en")
AutoTokenizer.from_pretrained("gpt2-en").save_pretrained("paddlepaddle/Totto/gpt2-en")

config_json = {
    "numericNLG":{
        "n_stx":122,
        "n_ttx_row":56,
        "n_ttx_col":24,
        "num_head":3
    },
    "Totto":{
        "n_stx":145, 
        "n_ttx_row":8, 
        "n_ttx_col":8,
        "num_head":3
    }
}

for i in ["en"]:
    for j in ["numericNLG", "Totto"]:
        with open(f"paddlepaddle/{j}/gpt2-{i}/model_config.json", "r") as r:
            config = json.load(r)
        for k in ["n_stx", "n_ttx_row", "n_ttx_col"]:
            config[k] = config_json[j][k]
        with open(f"paddlepaddle/{j}/gpt2-{i}/model_config.json", "w") as w:
            json.dump(config, w)