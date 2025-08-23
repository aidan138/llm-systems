import torch
from torch import Tensor
from typing import Iterable
import math


def softmax(x: Tensor, dim: int = -1):
    norm_x = x-x.max(dim=dim, keepdim= True).values # Subtract the max for numerical stability
    softmax = norm_x.exp() / torch.sum(norm_x.exp(), dim=dim, keepdim=True)
    return softmax


def cross_entropy_loss(logits: Tensor, targets: Tensor):
    # logits can be [... , N, V]
    # targets are [..., N]
    logits = logits - logits.max(dim=-1, keepdim=True).values # Subtract for numerical stability
    targ_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1).long())
    prob_sum = logits.exp().sum(dim=-1, keepdim=True).log()
    return (prob_sum - targ_logits).mean()

def lr_cosine_scheduling(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int):
    if t < t_w:
        return (t / t_w) * lr_max
    elif t <= t_c:
        return lr_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.data
        l2 = grad.square().sum().sqrt()
        if l2 > max_l2_norm:
            grad *= max_l2_norm / (l2 + eps)

