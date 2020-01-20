from reformers import ReformerLM, TFReformerLM
import torch
import tensorflow as tf

model = ReformerLM(
    num_tokens= 20000,
    emb = 512,
    depth = 12,
    max_seq_len = 32000,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 1600,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    use_full_attn = False   # use full self attention, for comparison
)

model_tf = TFReformerLM(
    num_tokens= 20000,
    emb = 512,
    depth = 1,
    max_seq_len = 32768,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 1600,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    use_full_attn = False   # use full self attention, for comparison
)

x = tf.random.uniform((1, 32000))
y = model_tf(x)

# x = torch.randint(0, 20000, (1, 32768)).long()
# y = model(x)
# y.sum().backward()