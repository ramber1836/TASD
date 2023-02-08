from tqdm import tqdm
import torch
from transformer import AutoTokenizer
from transformer.models.gpt2.gpt2_tasd import GPT2TASDLMHeadModel

def generation(model_path, dataloader, device, args):
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPT2TASDLMHeadModel.from_pretrained(model_path).to(device)

        if len(args.model_pt) !=0 :
            params = torch.load(args.model_pt, map_location=device)
            model.load_state_dict({p[6:]:params[p] for p in params if p[:6] == 'model.'}, strict=False)
        
        model.eval()
        candidates = []
        for semantic_table, numerical_table, text, _ in dataloader: #然而实际上并没有用到numerical的信息
            semantic_table, numerical_table, text = semantic_table.to(device), numerical_table.to(device), text.to(device)
            start_idx = text.shape[1]
            inputs = {
                # 'inputs': text[:, :512],
                'inputs': text,
                'semantic_table_ids': semantic_table,
                'max_length': text.shape[1] + args.generate_length,
                'do_sample': True,
                'num_beams': args.beam_num,
                "early_stopping": True,
                "pad_token_id": 50256,
                "eos_token_id": 50256,
                "num_return_sequences": args.beam_num
            }
            outputs = model.generate(**inputs)
            if "numericNLG" in args.data_name:
                pred_text = [tokenizer.decode(outputs[i][start_idx:].tolist()) for i in range(len(outputs))]
            elif "Totto" in args.data_name:
                pred_text = [tokenizer.decode(outputs[i][start_idx:].tolist()) for i in range(len(outputs))]
            # print(device)
            # print(pred_text)
            candidates.append(pred_text)
    return candidates