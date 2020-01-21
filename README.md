# Reformers - Efficient Transformers

This repository containes implementations of reformers as described in the following paper - [https://openreview.net/pdf?id=rkgNKkHtvB](https://openreview.net/pdf?id=rkgNKkHtvB). The following repository contains implementations of reformers in PyTorch as well as Tensorflow Keras. We will be performing more experiments on these over the course of time. 

## Install 

clone the repository locally and install dependencies using pip install. The dependencies are present in the `requirements.txt` file. 

## Usage

### PyTorch

```python
import torch
from reformers import ReformerLM

model = ReformerLM(
    num_tokens= 20000,
    emb = 512,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    use_full_attn = False   # use full self attention, for comparison
).cuda()

x = torch.randint(0, 20000, (1, 8192)).long().cuda()
y = model(x) # (1, 8192, 20000)
```

```python
import torch
from reformers import Reformer

model = Reformer(
    emb = 512,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
).cuda()

x = torch.randn(1, 8192, 512).cuda()
y = model(x) # (1, 8192, 512)
```

### Tensorflow

```python
model_tf = TFReformerLM(
    num_tokens= 20000,
    emb = 512,
    depth = 1,
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

model_tf.build(input_shape=(1,32000))
model_tf.summary()

# Model: "tf_reformer_lm"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        multiple                  10240000  
# _________________________________________________________________
# embedding_1 (Embedding)      multiple                  16384000  
# _________________________________________________________________
# tf_reformer (TFReformer)     multiple                  2888704   
# _________________________________________________________________
# dense_5 (Dense)              multiple                  10260000  
# =================================================================
# Total params: 39,772,704
# Trainable params: 39,772,704
# Non-trainable params: 0
```

Source for PyTorch code - [https://github.com/lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)