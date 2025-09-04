import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Iterable

SIZEOFMB = 1024**2

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



class DDPBucketed(nn.Module):

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.handles = []
        self.buckets = self._initialize_model_buckets(bucket_size_mb=bucket_size_mb, parameters=list(module.parameters()))
        self.flat_grads = []
        self.curr_bucket_idx = 0

    def forward(self, *inputs, **kwargs):
        # Reset the bucket idx for backward pass functionality
        self.curr_bucket_idx = 0
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for i, handle in enumerate(self.handles):
            handle.wait() # Wait for comm call to finish
            bucket_params = self.buckets[i] # List of the buckets params
            bucket_flat_grad = self.flat_grads[i] # Flat gradients for the parameters
            bucket_grads_list = torch._utils._unflatten_dense_tensors(bucket_flat_grad, bucket_params)
            for param, grad in zip(bucket_params, bucket_grads_list):
                # Copy the reduce-all avg gradients in place
                param.grad.copy_(grad)

        # Clear handles for next set of communication calls
        self.handles.clear()
        self.flat_grads.clear()

    def _initialize_model_buckets(self, bucket_size_mb: int, parameters: Iterable[nn.Parameter]):
        def bucket_hook(param: torch.Tensor):
            curr_bucket = self.buckets[self.curr_bucket_idx]

            # Check if param is the last tensor in the current bucket
            if param is curr_bucket[-1]:
                with torch.no_grad():
                    # Flatten the entire param buckets gradients to one tensor
                    flat_grad = torch._utils._flatten_dense_tensors(list(param.grad for param in curr_bucket))
                    # Asynchronously all reduce the flat gradients across ranks
                    self.handles.append(dist.all_reduce(tensor=flat_grad, op=dist.ReduceOp.AVG, async_op=True))
                    self.flat_grads.append(flat_grad) # Track the avg flat gradients
                self.curr_bucket_idx += 1
        
        # Initialize buckets
        buckets = []
        curr_bucket_size = 0
        curr = [] # Current bucket
        
        # Reverse order to roughly follow computation of gradients
        for param in parameters[::-1]:

            # Synchronize model parameters across ranks
            with torch.no_grad():
                dist.broadcast(tensor=param, src=0)

            # Don't assign parameters without grads to buckets
            if not param.requires_grad:
                continue
            
            # Create parameter buckets for batched grad communication
            param.register_post_accumulate_grad_hook(bucket_hook)
            param_size_mb = param.nelement() * param.element_size()

            # Check whether the parameter can be added to the bucket
            # NOTE if a single parameter exceeds bucket size it gets its own bucket
            if curr and param_size_mb + curr_bucket_size > bucket_size_mb:
                buckets.append(curr)
                curr = []
                curr_bucket_size = 0
            curr.append(param)
            curr_bucket_size += param_size_mb

        # Make sure to add partially filled buckets
        if curr:
            buckets.append(curr)

        return buckets
        