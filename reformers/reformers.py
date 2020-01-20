# MIT License

# Copyright (c) 2020 Phillip Wang, 2020 Streack, Jayakrishna Sahit

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from revtorch import ReversibleBlock, ReversibleSequence
from .attention import SelfAttention, FeedForward
from .efficient_attention import LSHSelfAttention
from .utils import WithNorm, Chunk

class Reformer(nn.Module):
    def __init__(self, emb, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., lsh_attend_across_buckets = True, lsh_allow_duplicate_attention = True, random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.emb = emb
        self.depth = depth

        get_full_attn = lambda: SelfAttention(emb, heads, causal = causal)
        get_lsh_attn = lambda: LSHSelfAttention(emb, heads, bucket_size, n_hashes, causal = causal, dropout = lsh_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head)

        get_attn = get_full_attn if use_full_attn else get_lsh_attn
        get_ff = lambda: FeedForward(emb)

        if weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        for _ in range(depth):
            attn = get_attn()
            parallel_net = get_attn() if twin_attention else get_ff()

            f = WithNorm(norm_type, emb, attn)
            g = WithNorm(norm_type, emb, parallel_net)

            if not twin_attention and ff_chunks > 1:
                g = Chunk(ff_chunks, g, along_dim = -2)

            blocks.append(ReversibleBlock(f, g, split_along_dim=-1))

        self.layers = ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x):
        x = torch.cat([x, x], dim = -1)
        x = self.layers(x)
        return torch.stack(x.chunk(2, dim=-1)).sum(dim=0)

class ReformerLM(nn.Module):
    def __init__(self, num_tokens, emb, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, emb)
        self.pos_emb = nn.Embedding(max_seq_len, emb)
        self.reformer = Reformer(emb, depth, max_seq_len, heads = heads, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, use_full_attn = use_full_attn)
        self.to_logits = nn.Linear(emb, num_tokens)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.reformer(x)
        return self.to_logits(x)

