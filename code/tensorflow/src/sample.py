import tensorflow.compat.v1 as tf
import numpy as np
import model
import table_model
import math
import os

os.environ["PL_GLOBAL_SEED"] = str(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
        min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens


def sample_sequence_beam_search(*, hparams, beam_width, length, context=None, temperature=1, top_k=0, top_p=0.0):
    beam_width_max = beam_width
    def step(hparams, tokens, past=None):
        beam_width = tokens.shape[0]
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        shape = model.past_shape(hparams=hparams, batch_size=beam_width)
        presents.set_shape(shape)
        return {
            'logits': logits,
            'presents': presents,
        }
    def raw_body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]
    def init_body(past, prev, beam_width, log_beam_probs, beam_results):
        
        next_outputs = step(hparams, prev, past=past)
        #outputs['logits']: [batch, sequence, hparams.n_vocab]
        logits = next_outputs['logits'][:, -1, :]
        '''
        #得到beam_width个序列的下一个词的概率分布
        if top_p > 0.0:
            logits = top_p_logits(logits, p=top_p)
        else:
            logits = top_k_logits(logits, k=top_k)
        '''
        probs = tf.log(tf.nn.softmax(logits) + 1e-12)
        #取log，方便后续计算
        best_probs, indices = tf.nn.top_k(probs, beam_width)
        indices = tf.reshape(tf.stop_gradient(indices), [beam_width, -1])
        best_probs = tf.reshape(tf.stop_gradient(best_probs), [beam_width, -1])
        symbols = indices
        beam_parent = tf.constant([[0], [0], [0], [0], [0]])

        #解码到<EOS>符号，终止解码
        partition = tf.cast(tf.cast(symbols-hparams.n_vocab+1, tf.bool), tf.int32)

        prob_group = tf.dynamic_partition(best_probs, partition, 2)
        symbols_group = tf.dynamic_partition(symbols, partition, 2)
        parent_group = tf.dynamic_partition(beam_parent, partition, 2)
        new_indices = parent_group[1]
        
        #取出含<EOS>符号的beam，并保存对应结果
        #eos_prev = symbols_group[0]
        #eos_output = tf.concat([tf.gather(output, parent_group[0]), eos_prev], axis=1)
        #new_beam_results_eos = tf.reshape(symbols_group[0], eos_num)

        #不含<EOS>符号的beam, 找到对应的之前的序列，拼接符号序列以及对应的概率
        
        eos_num = tf.shape(prob_group[0])
        beam_width -= eos_num

        new_past = tf.gather(next_outputs['presents'], new_indices)
        
        new_prev = tf.reshape(symbols_group[1], [-1, 1])
        
        new_beam_results_eos = tf.reshape(symbols_group[0], beam_width_max-beam_width)
        new_beam_results = tf.concat([symbols_group[1], new_beam_results_eos], 0)


        new_log_beam_probs_eos = tf.reshape(prob_group[0], beam_width_max-beam_width)
        new_log_beam_probs = tf.concat([prob_group[1], new_log_beam_probs_eos], 0)

        log_beam_probs = tf.reshape(new_log_beam_probs, [-1, 1])
        beam_results = tf.reshape(new_beam_results, [-1, 1])

        

        # 否则，则更新上面的值，进入下个循环 

        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        '''
        emb_prev = embedding_ops.embedding_lookup(embedding, _symbols)
        emb_prev = tf.reshape(emb_prev,[beam_size,embedding_size])
        # emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
        '''
        #samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
        return [
            new_past,
            new_prev,
            beam_width,
            log_beam_probs,
            beam_results
        ]
    with tf.name_scope('sample_sequence'):
        def body(past, prev, beam_width, log_beam_probs, beam_results):
            beam_results_eos = beam_results[beam_width[0]:,:]
            next_outputs = step(hparams, prev, past=past)
            #outputs['logits']: [batch, sequence, hparams.n_vocab]
            #logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = next_outputs['logits'][:, -1, :]
            #得到beam_width个序列的下一个词的概率分布
            '''
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            '''
            probs = tf.log(tf.nn.softmax(logits) + 1e-12)
            #取log，方便后续计算

            probs = tf.reshape(probs + tf.reshape(log_beam_probs[:beam_width[0],-1],[beam_width[0],-1]), [-1, beam_width[0] * hparams.n_vocab])
            #和上次的log结果相加（即原结果相乘）
            best_probs, indices = tf.nn.top_k(probs, beam_width[0])
            indices = tf.stop_gradient(indices)
            best_probs = tf.stop_gradient(best_probs)
            # 将所有beam平铺（flatten），查找当前5个所有Beam当中的概率最大的5只beam，并保存其位置
            symbols = indices % (hparams.n_vocab)
            # 取余数的方法获得当前的字符
            beam_parent = indices // (hparams.n_vocab)
                # 取商数的方法获得当前的字符原本属于哪个beam
            # 现阶段概率值的对数=当前logit的对数+之前Beampath路径上的prob对数


            #解码到<EOS>符号，终止此beam解码
            partition = tf.cast(tf.cast(symbols-hparams.n_vocab+1, tf.bool), tf.int32)

            prob_group = tf.dynamic_partition(best_probs, partition, 2)
            symbols_group = tf.dynamic_partition(symbols, partition, 2)
            parent_group = tf.dynamic_partition(beam_parent, partition, 2)
            beam_width = tf.shape(prob_group[1])

            new_indices = parent_group[1]
            new_beam_results = tf.reshape(symbols_group[1], [beam_width[0], -1])

            eos_num = beam_width_max - tf.shape(prob_group[1])[0]
            
            
            #取出含<EOS>符号的beam，并保存对应结果
            #产生<EOS>
            new_beam_results_eos = tf.fill([eos_num, 1], hparams.n_vocab-1)
            new_beam_results = tf.concat([new_beam_results, new_beam_results_eos], 0)
                
            #不含<EOS>符号的beam, 找到对应的之前的序列，拼接符号序列以及对应的概率
            
            new_past = tf.gather(tf.concat([past, next_outputs['presents']], axis=-2), new_indices)
            new_prev = tf.reshape(symbols_group[1], [beam_width[0], -1])
            
            new_log_eos = tf.fill([eos_num, 1], -math.inf)
            best_probs = tf.transpose(best_probs)
            new_log_beam_probs = tf.gather(best_probs, new_indices)
            
            new_log_beam_probs = tf.concat([new_log_beam_probs, new_log_eos], 0)
            log_beam_probs = tf.concat([log_beam_probs, new_log_beam_probs], -1)

            beam_eos = tf.concat([tf.gather(beam_results, parent_group[0]), beam_results_eos], 0)
            #new_beam_eos = tf.concat([beam_eos, tf.fill([eos_num, 1], hparams.n_vocab-1)], -1)

            new_beam = tf.concat([tf.gather(beam_results, new_indices), beam_eos], 0)

            beam_results = tf.concat([new_beam, new_beam_results], -1)
            # 否则，则更新上面的值，进入下个循环

            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            '''
            emb_prev = embedding_ops.embedding_lookup(embedding, _symbols)
            emb_prev = tf.reshape(emb_prev,[beam_size,embedding_size])
            # emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
            if not update_embedding:
                emb_prev = array_ops.stop_gradient(emb_prev)
            return emb_prev
            '''
            #samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                new_past,
                new_prev,
                beam_width,
                tf.reshape(log_beam_probs, [beam_width_max, -1]),
                tf.reshape(beam_results, [beam_width_max, -1])
            ]
        #context = tf.tile(context, [beam_width, -1])
        #log_beam_probs, beam_results, out = raw_body(None, context, context)
        past, prev, beam_width, log_beam_probs, beam_results = init_body(None, context, tf.constant(beam_width), None, None)
        '''
        for i in range(99):
            if beam_width[0] == 0:
                break
            past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        '''
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
        #return log_beam_probs, beam_results

        def cond(past,
                prev,
                beam_width, log_beam_probs, beam_results):
            return (beam_width[0] > 0)
        #'''
        _, _, beam_width, log_beam_probs, beam_results = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                beam_width, log_beam_probs,  beam_results
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=None)),
                tf.TensorShape([None, None]),
                tf.TensorShape([1]),
                tf.TensorShape([beam_width_max, None]),
                tf.TensorShape([beam_width_max, None]),
            ],
            back_prop=False,
        )
        #'''
        
        return log_beam_probs, beam_results


def table_sample_sequence_beam_search(*, hparams, beam_width, length, context=None, tables=None, keras_layers=[], temperature=1, top_k=0, top_p=0.0):
    beam_width_max = beam_width
    
    def step(hparams, tokens, past=None):
        beam_width = tokens.shape[0]
        lm_output = table_model.model(hparams=hparams, X=tokens, T=tables, keras_layers=keras_layers, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        shape = table_model.past_shape(hparams=hparams, batch_size=beam_width)
        presents.set_shape(shape)
        return {
            'logits': logits,
            'presents': presents,
        }
    def raw_body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]
    def init_body(past, prev, beam_width, log_beam_probs, beam_results):
        
        next_outputs = step(hparams, prev, past=past)
        #outputs['logits']: [batch, sequence, hparams.n_vocab]
        logits = next_outputs['logits'][:, -1, :]
        '''
        #得到beam_width个序列的下一个词的概率分布
        if top_p > 0.0:
            logits = top_p_logits(logits, p=top_p)
        else:
            logits = top_k_logits(logits, k=top_k)
        '''
        probs = tf.log(tf.nn.softmax(logits) + 1e-12)
        #取log，方便后续计算
        best_probs, indices = tf.nn.top_k(probs, beam_width)
        indices = tf.reshape(tf.stop_gradient(indices), [beam_width, -1])
        best_probs = tf.reshape(tf.stop_gradient(best_probs), [beam_width, -1])
        symbols = indices
        beam_parent = tf.constant([[0], [0], [0], [0], [0]])

        #解码到<EOS>符号，终止解码
        partition = tf.cast(tf.cast(symbols-hparams.n_vocab+1, tf.bool), tf.int32)

        prob_group = tf.dynamic_partition(best_probs, partition, 2)
        symbols_group = tf.dynamic_partition(symbols, partition, 2)
        parent_group = tf.dynamic_partition(beam_parent, partition, 2)
        new_indices = parent_group[1]
        
        #取出含<EOS>符号的beam，并保存对应结果
        #eos_prev = symbols_group[0]
        #eos_output = tf.concat([tf.gather(output, parent_group[0]), eos_prev], axis=1)
        #new_beam_results_eos = tf.reshape(symbols_group[0], eos_num)

        #不含<EOS>符号的beam, 找到对应的之前的序列，拼接符号序列以及对应的概率
        
        eos_num = tf.shape(prob_group[0])
        beam_width -= eos_num

        new_past = tf.gather(next_outputs['presents'], new_indices)
        
        new_prev = tf.reshape(symbols_group[1], [-1, 1])
        
        new_beam_results_eos = tf.reshape(symbols_group[0], beam_width_max-beam_width)
        new_beam_results = tf.concat([symbols_group[1], new_beam_results_eos], 0)


        new_log_beam_probs_eos = tf.reshape(prob_group[0], beam_width_max-beam_width)
        new_log_beam_probs = tf.concat([prob_group[1], new_log_beam_probs_eos], 0)

        log_beam_probs = tf.reshape(new_log_beam_probs, [-1, 1])
        beam_results = tf.reshape(new_beam_results, [-1, 1])

        

        # 否则，则更新上面的值，进入下个循环 

        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        '''
        emb_prev = embedding_ops.embedding_lookup(embedding, _symbols)
        emb_prev = tf.reshape(emb_prev,[beam_size,embedding_size])
        # emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
        '''
        #samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
        return [
            new_past,
            new_prev,
            beam_width,
            log_beam_probs,
            beam_results
        ]
    
    def body(past, prev, beam_width, log_beam_probs, beam_results):
        beam_results_eos = beam_results[beam_width[0]:,:]
        next_outputs = step(hparams, prev, past=past)
        #outputs['logits']: [batch, sequence, hparams.n_vocab]
        #logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
        logits = next_outputs['logits'][:, -1, :]
        '''
        if top_p > 0.0:
            logits = top_p_logits(logits, p=top_p)
        else:
            logits = top_k_logits(logits, k=top_k)
        '''
        probs = tf.log(tf.nn.softmax(logits) + 1e-12)

        probs = tf.reshape(probs + tf.reshape(log_beam_probs[:beam_width[0],-1],[beam_width[0],-1]), [-1, beam_width[0] * hparams.n_vocab])
        best_probs, indices = tf.nn.top_k(probs, beam_width[0])
        indices = tf.stop_gradient(indices)
        best_probs = tf.stop_gradient(best_probs)
        symbols = indices % (hparams.n_vocab)
        beam_parent = indices // (hparams.n_vocab)


        partition = tf.cast(tf.cast(symbols-hparams.n_vocab+1, tf.bool), tf.int32)

        prob_group = tf.dynamic_partition(best_probs, partition, 2)
        symbols_group = tf.dynamic_partition(symbols, partition, 2)
        parent_group = tf.dynamic_partition(beam_parent, partition, 2)
        beam_width = tf.shape(prob_group[1])

        new_indices = parent_group[1]
        new_beam_results = tf.reshape(symbols_group[1], [beam_width[0], -1])

        eos_num = beam_width_max - tf.shape(prob_group[1])[0]
        
        
        new_beam_results_eos = tf.fill([eos_num, 1], hparams.n_vocab-1)
        new_beam_results = tf.concat([new_beam_results, new_beam_results_eos], 0)
            
        
        new_past = tf.gather(tf.concat([past, next_outputs['presents']], axis=-2), new_indices)
        new_prev = tf.reshape(symbols_group[1], [beam_width[0], -1])
        
        new_log_eos = tf.fill([eos_num, 1], -math.inf)
        best_probs = tf.transpose(best_probs)
        new_log_beam_probs = tf.gather(best_probs, new_indices)
        
        new_log_beam_probs = tf.concat([new_log_beam_probs, new_log_eos], 0)
        log_beam_probs = tf.concat([log_beam_probs, new_log_beam_probs], -1)

        beam_eos = tf.concat([tf.gather(beam_results, parent_group[0]), beam_results_eos], 0)
        #new_beam_eos = tf.concat([beam_eos, tf.fill([eos_num, 1], hparams.n_vocab-1)], -1)

        new_beam = tf.concat([tf.gather(beam_results, new_indices), beam_eos], 0)

        beam_results = tf.concat([new_beam, new_beam_results], -1)

        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        '''
        emb_prev = embedding_ops.embedding_lookup(embedding, _symbols)
        emb_prev = tf.reshape(emb_prev,[beam_size,embedding_size])
        # emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
        '''
        #samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
        return [
            new_past,
            new_prev,
            beam_width,
            tf.reshape(log_beam_probs, [beam_width_max, -1]),
            tf.reshape(beam_results, [beam_width_max, -1])
        ]
        #context = tf.tile(context, [beam_width, -1])
        #log_beam_probs, beam_results, out = raw_body(None, context, context)
    past, prev, beam_width, log_beam_probs, beam_results = init_body(None, context, tf.constant(beam_width), None, None)
    '''
    for i in range(99):
        if beam_width[0] == 0:
            break
        past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    '''
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #past, prev, beam_width, log_beam_probs, beam_results = body(past, prev, beam_width, log_beam_probs, beam_results)
    #return log_beam_probs, beam_results

    def cond(past,
            prev,
            beam_width, log_beam_probs, beam_results):
        return (beam_width[0] > 0)
    #'''
    _, _, beam_width, log_beam_probs, beam_results = tf.while_loop(
        cond=cond, body=body,
        maximum_iterations=length - 1,
        loop_vars=[
            past,
            prev,
            beam_width, log_beam_probs,  beam_results
        ],
        shape_invariants=[
            tf.TensorShape(table_model.past_shape(hparams=hparams, batch_size=None)),
            tf.TensorShape([None, None]),
            tf.TensorShape([1]),
            tf.TensorShape([beam_width_max, None]),
            tf.TensorShape([beam_width_max, None]),
        ],
        back_prop=False,
    )
    #'''
    
    return log_beam_probs, beam_results
