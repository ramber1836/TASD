import numpy as np
import tensorflow.compat.v1 as tf

class HParams(object):
    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def override_from_dict(self, kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)


def default_hparams():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        n_stx=122,
        n_ttx_row=56,
        n_ttx_col=24
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        n_state = shape_list(x)[-1]
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        nx = shape_list(x)[-1]
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def seq_self_attn(x, mha1, scope):
    with tf.variable_scope(scope):
        batch, row, col, seq_len, dim = shape_list(x)
        #dim_num = len(x.shape)
        #layer = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=256, attention_axes=(dim_num-2, dim_num-1))
        x = tf.reshape(x, [batch * row * col, seq_len, dim])
        x = mha1(x, x)
        x = tf.reshape(x, [batch, row, col, seq_len, dim])
        x = tf.reduce_mean(x, -2)
        return x

def table_self_attn(x, mha15, scope):
    with tf.variable_scope(scope):
        batch, row, col, dim = shape_list(x)
        x_reshape = tf.reshape(x, [batch, row*col, dim])
        x = mha15(x_reshape, x_reshape)
        x = tf.reshape(x, [batch, row, col, dim])
        
        #x_reshape = tf.reshape(x, [batch, row*col, dim])
        #x = mha2(q, x_reshape)
        return x

def table_attn(q, x, mha2, scope):
    with tf.variable_scope(scope, reuse=True):
        batch, row, col, dim = shape_list(x)
        #layer = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=256)
        x_reshape = tf.reshape(x, [batch, row*col, dim])
        x = mha2(q, x_reshape)
        return x

#def model_table(h, t, tp, mha1, mha2, scope="extra_model_table"):
def model_table(h, t, tp, mha1, mha2, mha15, scope="extra_model_table"):
    with tf.variable_scope(scope, reuse=True):
        #nx = shape_list(t)[-1]
        sa = seq_self_attn(norm(t, 'ln_3'), mha1, 'extra_seq_self_attn')
        #tsa = table_self_attn(norm(tf.reduce_mean(t, -2) + sa, 'ln_35'), mha15, 'extra_table_self_attn')
        #ta = table_attn(h, norm(tsa + tp, "ln_4"), mha2, "extra_table_attn")
        
        tsa = table_self_attn(norm(tf.reduce_mean(t, -2) + sa + tp, 'ln_35'), mha15, 'extra_table_self_attn')
        ta = table_attn(h, norm(tsa, "ln_4"), mha2, "extra_table_attn")

        #sa = seq_self_attn(norm(t, 'ln_3'), mha1, 'extra_seq_self_attn')
        #ta = table_attn(h, norm(sa + tp, "ln_4"), mha2, "extra_table_attn")
        return ta

def table_mask(inputs, mske, mask_rate):
    embeddings = tf.identity(inputs)
    distrib = tf.distributions.Bernoulli(probs=[mask_rate])
    masked_onehot = distrib.sample(embeddings.shape[:-1])
    masked_indices = tf.where(tf.squeeze(masked_onehot, [-1]))
    updates = tf.tile(mske, [tf.reduce_sum(masked_onehot),1])
    #labels[~masked_indices] = -1
    print(embeddings.shape, masked_indices.shape, updates.shape)
    masked_embeddings = tf.tensor_scatter_nd_update(embeddings, masked_indices, updates)
    return masked_embeddings, masked_indices

def model_table_recover(t, tp, mske, mha1, mha15, mask_rate, mharcv, scope="extra_model_table"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #nx = shape_list(t)[-1]
        sa = seq_self_attn(norm(t, 'ln_3'), mha1, 'extra_seq_self_attn')
        sa = norm(tf.reduce_mean(t, -2) + sa + tp, 'ln_35')
        masked_embeddings, masked_indices = table_mask(sa, mske, mask_rate)
        tsa = table_self_attn(masked_embeddings, mha15, 'extra_table_self_attn')
        recovered_embeddings = mharcv(tsa, tsa)
        return tf.gather_nd(sa, masked_indices), tf.gather_nd(recovered_embeddings, masked_indices)

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length, hparams):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile((past_length + tf.range(nsteps)) % hparams.n_ctx, batch_size)

def positions_for_table(batch_size, twper, twpec, hparams):
    tr = tf.tile(tf.reshape(tf.range(hparams.n_ttx_row), [1, hparams.n_ttx_row, 1]), [batch_size, 1, hparams.n_ttx_col])
    tc = tf.tile(tf.reshape(tf.range(hparams.n_ttx_col), [1, 1, hparams.n_ttx_col]), [batch_size, hparams.n_ttx_row, 1])

    return tf.gather(twper, tr) + tf.gather(twpec, tc)

def positions_for_seq_in_table(batch_size, swpe, hparams):
    tsp = tf.tile(tf.reshape(tf.range(hparams.n_stx), [1, 1, 1, hparams.n_stx]), [batch_size, hparams.n_ttx_row, hparams.n_ttx_col, 1])
    return tf.gather(swpe, tsp)


    
def model(hparams, X, T, keras_layers, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        mha1, mha2, mha15 = keras_layers
        #mha1, mha2 = keras_layers
        results = {}
        batch, sequence = shape_list(X)
        
        twper = tf.get_variable('extra_twper', [hparams.n_ttx_row, hparams.n_embd],
                                initializer=tf.random_normal_initializer(stddev=0.01))
        twpec = tf.get_variable('extra_twpec', [hparams.n_ttx_col, hparams.n_embd],
                                initializer=tf.random_normal_initializer(stddev=0.01))
        swpe = tf.get_variable('extra_swpe', [hparams.n_stx, hparams.n_embd],
                                initializer=tf.random_normal_initializer(stddev=0.01))

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))

        mske = tf.get_variable('extra_mske', [1, hparams.n_embd],
                            initializer=tf.random_normal_initializer(stddev=0.01))

        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length, hparams))
        
        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')
        
        sh = tf.gather(wte, T) + positions_for_seq_in_table(batch, swpe, hparams)
        #masked_sh = table_mask(sh, mske)
        #target, masked_th = model_table_recover(sh, mske, mha1, mha15)

        tp = positions_for_table(batch, twper, twpec, hparams)
        th = model_table(h, sh, tp, mha1, mha2, mha15)
        #th = model_table(h, sh, tp, mha1, mha2)

        h = norm(h + th, 'extra_ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results

def model_mtl(hparams, X, T, keras_layers, mask_rate, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        mha1, mha2, mha15, mharcv = keras_layers
        #mha1, mha2 = keras_layers
        results = {}
        batch, sequence = shape_list(X)
        
        twper = tf.get_variable('extra_twper', [hparams.n_ttx_row, hparams.n_embd],
                                initializer=tf.random_normal_initializer(stddev=0.01))
        twpec = tf.get_variable('extra_twpec', [hparams.n_ttx_col, hparams.n_embd],
                                initializer=tf.random_normal_initializer(stddev=0.01))
        swpe = tf.get_variable('extra_swpe', [hparams.n_stx, hparams.n_embd],
                                initializer=tf.random_normal_initializer(stddev=0.01))

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))

        mske = tf.get_variable('extra_mske', [1, hparams.n_embd],
                            initializer=tf.random_normal_initializer(stddev=0.01))

        #past_length = 0 if past is None else tf.shape(past)[-2]
        #h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length, hparams))
        #gather
        # Transformer

        sh = tf.gather(wte, T) + positions_for_seq_in_table(batch, swpe, hparams)
        #masked_sh = table_mask(sh, mske)
        tp = positions_for_table(batch, twper, twpec, hparams)
        target, masked_th = model_table_recover(sh, tp, mske, mha1, mha15, mask_rate, mharcv)

        results['true'] = target
        results['recover'] = masked_th
        return results

if __name__ == '__main__':
    test = tf.zeros([4,4,4,4,256])
    