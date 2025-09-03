import torch
import torch.nn as nn
import torch.distributed as dist
import logging


class DDPIndividualParameters(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        def hook(param: torch.Tensor):
            with torch.no_grad():
                handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)

        for param in module.parameters():
            
            with torch.no_grad():
                dist.broadcast(tensor=param, src=0, async_op=False) # Broadcast the parameters
            
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook) # Register reduction hook
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()
        
