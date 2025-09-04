import torch
from torch.cuda import nvtx

def push_nvtx(name: str):
    if torch.cuda.is_available(): nvtx.range_push(name)

def pop_nvtx():
    if torch.cuda.is_available(): nvtx.range_pop()