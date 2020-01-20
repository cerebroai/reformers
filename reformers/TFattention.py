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

import math
import tensorflow as tf
from tensorflow.keras.layers import Dense

def mask_fill_inf(matrix, mask):
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (matrix * mask) + (-((negmask * num + num) - num))

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)
        self.dense = Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)
        return outputs


class TFSelfAttention(tf.keras.Model):
    def __init__(self, emb, heads = 8, causal = False):
        super().__init__()
        assert emb % heads == 0, 'dimensions must be divisible by number of heads'
        self.attn = MultiheadAttention(emb, heads)
        self.to_out = Dense(emb, emb)
        self.causal = causal

    def call(self, inputs):
        b, t, e = inputs.shape
        inputs = tf.transpose(inputs, (0, 1))

        attn_mask = tf.zeros(t, t)
        if self.causal:
            causal_mask = tf.triu(tf.ones(t, t) == 1, 1)
            mask_fill_inf(attn_mask, causal_mask)

        output = self.attn({'query' : x, 'key' : x, 'value' : x, 'mask' : attn_mask})
        return self.to_out(tf.transpose(output, (0, 1)))


class TFFeedForward(tf.keras.Model):
    def __init__(self, emb, mult = 4):
        super().__init__()
        self.emb = emb
        self.proj_in = Dense(emb * mult)
        self.proj_out = Dense(emb)

    def call(self, inputs):
        inputs = self.proj_in(inputs)
        inputs = tf.keras.activations.relu(inputs)
        inputs = self.proj_out(inputs)
        return inputs