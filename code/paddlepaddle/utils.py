import os
import paddle

def mkdir_files(dir_path):
    path = "."
    for i in dir_path.split("/"):
        if '.' in i:
            break
        path = os.path.join(path, i)
        if not os.path.exists(path):
            os.mkdir(path)

def save_model(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def loss_calculation(batch):
    input_ids, labels = batch
    logits = model(input_ids)
    masked_lm_loss = paddle.nn.CrossEntropyLoss(reduction="none")(logits,
                                labels.unsqueeze(2))
    loss = paddle.sum(masked_lm_loss.reshape([-1]))
    loss = loss / logits.shape[0]
    del input_ids, labels
    return loss

def encode(text, tokenizer):
    tokens = tokenizer(text)
    input_ids = tokens['input_ids']

    input_ids = paddle.reshape(paddle.to_tensor(input_ids), [1, -1])
    return input_ids

def decode(ids, tokenizer):
    return tokenizer.convert_ids_to_string(ids.numpy().tolist()[0])

def attention_mask():
    i = paddle.arange(1024)[:,None]
    j = paddle.arange(1024)
    m = (i >= j)
    att_mask = (m.astype('float32') - 1) * 1e9
    return att_mask