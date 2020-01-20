# MIT License

# Copyright (c) 2020 Streack, Jayakrishna Sahit

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LayerNormalization, Dense
from .TFefficient_attention import TFLSHAttention, TFLSHSelfAttention
from .TFattention import TFSelfAttention, TFFeedForward
from .TFutils import cache_fn, Chunk, WithNorm
from .blocks import ReversibleBlock, ReversibleSequence

class TFReformer(tf.keras.Model):
    def __init__(self, emb, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., lsh_attend_across_buckets = True, lsh_allow_duplicate_attention = True, random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.emb = emb
        self.depth = depth

        get_full_attn = lambda: TFSelfAttention(emb, heads, causal = causal)
        get_lsh_attn = lambda: TFLSHSelfAttention(emb, heads, bucket_size, n_hashes, causal = causal, dropout = lsh_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head)

        get_attn = get_full_attn if use_full_attn else get_lsh_attn
        get_ff = lambda: TFFeedForward(emb)

        if weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if use_scale_norm else LayerNormalization

        for _ in range(depth):
            attn = get_attn()
            parallel_net = get_attn() if twin_attention else get_ff()
            f = WithNorm(norm_type, emb, attn)
            g = WithNorm(norm_type, emb, parallel_net)

            if not twin_attention and ff_chunks > 1:
                g = Chunk(ff_chunks, g, along_axis = -2)

            blocks.append(ReversibleBlock(f, g, split_along_axis=-1))

        self.model_layers = ReversibleSequence(blocks)

    def call(self, x):
        x = tf.concat([x, x], axis = -1)
        x = self.model_layers(x)
        return tf.stack(tf.reduce_sum(tf.split(x, 2, axis=-1), axis=0))

class TFReformerLM(tf.keras.Model):
    def __init__(self, num_tokens, emb, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.token_emb = Embedding(num_tokens, emb)
        self.pos_emb = Embedding(max_seq_len, emb)
        self.reformer = TFReformer(emb, depth, max_seq_len, heads = heads, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, use_full_attn = use_full_attn)
        self.to_logits = Dense(num_tokens)

    def call(self, inputs):
        print(inputs.shape)
        inputs = self.token_emb(inputs) + self.pos_emb(tf.range(inputs.shape[1]))
        inputs = self.reformer(inputs)
        return self.to_logits(inputs)