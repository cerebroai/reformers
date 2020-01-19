import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def make_unit_length(x, epsilon=1e-6):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    return x.div(norm + epsilon)

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: x.chunk(chunks, dim=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return outputs

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


class ScaleNorm(nn.Module):
    def __init__(self, emb, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, requires_grad=True))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

class WithNorm(nn.Module):
    def __init__(self, norm_class, emb, fn):
        super().__init__()
        self.emb = emb
        self.norm = norm_class(emb)
        self.fn = fn
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)
