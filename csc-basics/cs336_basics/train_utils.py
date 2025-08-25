import torch
import torch.nn as nn
from os import PathLike
from typing import BinaryIO
import numpy.typing as npt
import numpy as np

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | PathLike | BinaryIO):

    state_dict = {
        'Model' : model.state_dict(),
        'Optimizer' : optimizer.state_dict(),
        'Iteration' : iteration
    }

    torch.save(state_dict, out)


def load_checkpoint(src, model: nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['Model'])
    optimizer.load_state_dict(state_dict['Optimizer'])
    return state_dict['Iteration']

def get_batch(dataset: npt.NDArray, batch_size, context_length, device):
    #dataset = torch.from_numpy(dataset).to(device)
    
    dataset_len = dataset.shape[0] # Document length
    #idxs = np.arange(0, batch_size)
    idxs = np.random.randint(0, dataset_len-context_length-1, batch_size)
    X = torch.Tensor(np.stack([dataset[i: i+context_length] for i in idxs], axis=0)).cpu().to(device).int()
    y = torch.Tensor(np.stack([dataset[i+1: i+context_length+1] for i in idxs], axis=0)).cpu().to(device).long()
    return X, y