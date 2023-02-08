#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import tqdm
import argparse
from output2input import test_output2input

import model, sample, encoder


os.environ["PL_GLOBAL_SEED"] = str(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)


parser = argparse.ArgumentParser(
    description='use GPT-2 to generate samples on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ckpt_dir', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--save_every', type=int, default=400, help='Pretrained model name')
parser.add_argument('--epochs', type=int, default=2000, help='Pretrained model name')
parser.add_argument('--model', type=str, default='normal', help='Pretrained model name')

parser.add_argument('--model_name', type=str, default='', help='Pretrained model name')
parser.add_argument('--data_dir', type=str, default='./afs/models', help='Path to models directory')
parser.add_argument('--out_dir', type=str, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')



def move_files(run_num, fsrc, ftgt, path='/data/cmer/repos/table2text/gpt-2-finetuning/'):
    fpsrc = path + 'checkpoint/' + run_num
    fptgt = path + 'models/' + ftgt
    print('rename ' + fpsrc + '/' + fsrc + ' ' + fpsrc + '/model.ckpt ' + fpsrc + '/' + fsrc + '*')
    print('mv ' + fpsrc + '/model.ckpt* ' + fptgt + '/')
    os.system('rename ' + fpsrc + '/' + fsrc + ' ' + fpsrc + '/model.ckpt ' + fpsrc + '/' + fsrc + '*')
    os.system('cp ' + fpsrc + '/model.ckpt* ' + fptgt + '/')

def write_to_files(path, data):
    assert os.path.isdir(path)
    for i in range(len(data)):
        with open(path + '/' + str(i), 'w') as f:
            f.write(data[i])

def write_to_files_json(path, data):
    #assert os.path.isdir(path)
    with open(path + '_generation_json', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_raw_text(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for index, fname in enumerate(fnames):
                paths.append(os.path.join(dirpath, str(index)))
    context_data = []
    for path in tqdm.tqdm(paths):
        # Plain text
        with open(path, 'r', encoding='utf-8') as fp:
            raw_text = fp.readline()
            if len(raw_text) == 0:
                raw_text = '<|endoftext|>'
            context_data.append(raw_text)
    return context_data
'''
def interact_model(
    gpu,
    model_name='345M_test',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=0.9,
    top_k=20,
    top_p=0,
    models_dir='models',
    data_dir=None,
    out_dir=None
):
    """
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    context_data = load_raw_text(data_dir)
    generation = []
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence_beam_search(
            hparams=hparams, length=length,
            context=context,
            beam_width=5,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)
        generated = 0
        for raw_text in context_data:
            context_tokens = enc.encode(raw_text)
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
                    generation.append(text.lower())
            print("=" * 80)
        write_to_files(out_dir, generation)
'''

def inference_model(
    gpu,
    model_name='774M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=0.9,
    top_k=20,
    top_p=0,
    models_dir='afs/models',
    ckpt_dir='afs/checkpoint',
    ckpt_list=[],
    data_dir=None,
    out_dir=None,
    beam_width=5
):
    """
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    if gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    context_data = load_raw_text(data_dir)
    for ckpt_item in ckpt_list:
        generation = []
        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence_beam_search(
                hparams=hparams, length=length,
                context=context,
                beam_width=beam_width,
                temperature=temperature, top_k=top_k, top_p=top_p
            )
            saver = tf.train.Saver()
            ckpt = os.path.join(ckpt_dir, "model-" + ckpt_item)
            saver.restore(sess, ckpt)
            generated = 0
            for raw_text in context_data:
                context_tokens = enc.encode(raw_text)
                generation_item = []
                for _ in range(nsamples // batch_size):
                    '''
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    '''
                    out1, out2 = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })
                    
                    for i in range(beam_width):
                        generated += 1
                        text = enc.decode(out2[i])
                        print("=" * 40 + " SAMPLE " + str(generated) + " in model " + ckpt_item + " " + "=" * 40)
                        print(text)
                        
                        generation_item.append(text.lower().strip() + '\n')
                    generation.append(generation_item)
            write_to_files_json(out_dir + '_' + ckpt_item, generation)

if __name__ == '__main__':
    args = parser.parse_args()
    #move_files('run1', 'model-760', '345M_test')
    #interact_model('-1', ckpt_dir='afs/checkpoint/run4_774_400_10000', ckpt_list=[str(400 * i) for i in range(7, 26)], model_name="774M",
    # data_dir='afs/data/TD_test_input', out_dir='afs/data/TD_test_generated_774M')
    if args.model == 'normal':
        inference_model('-1', ckpt_dir='afs/checkpoint/'+args.ckpt_dir, ckpt_list=[str(args.save_every * i) for i in range(1, args.epochs // args.save_every + 1)], model_name=args.model_name,
        data_dir='afs/data/'+args.data_dir, out_dir='afs/data/'+args.out_dir)
    elif args.model == 'rewrite':
        inference_model('-1', ckpt_dir='afs/rewrite/checkpoint/'+args.ckpt_dir, ckpt_list=[str(args.save_every * i) for i in range(1, args.epochs // args.save_every + 1)], model_name=args.model_name,
        data_dir='afs/rewrite/'+args.data_dir, out_dir='afs/rewrite/'+args.out_dir)
        

