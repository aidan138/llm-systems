import torch
from torch.optim import Optimizer
from torch.nn import Parameter
from typing import Iterable, Any
import torch.distributed as dist

class OSDP(Optimizer):
    def __init__(self, params: Iterable[Parameter], optimizer_cls: Optimizer, **kwargs: Any):
        
        # Distributed must be initialized for valid sharding
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        self.param_to_rank = {} # Tracks which parameter is assigned to which rank, allows optimizer to update all model params in its process through broadcasts
        self.rank_to_numel = {i: 0 for i in range(self.world_size)} # Tracks the current number of elements assigned to each rank
        self.optimizer_cls = optimizer_cls # Store the class for later initialization
        
        # Super call iterates through params assigning param_groups with the defaults if they aren't set
        super().__init__(params=params, defaults=dict(**kwargs))

    def step(self, closure=None, **kwargs: Any):

        # Step the optimizer if present on the rank
        if hasattr(self, "optimizer"):
            self.optimizer.step(closure, **kwargs)
        
        with torch.no_grad():
            for p, owner in self.param_to_rank.items():
                # Broadcast the updated parameters
                dist.broadcast(tensor=p.data, src=owner)

    def add_param_group(self, param_group: dict[str, Any]):
        local_params = []
        for p in param_group["params"]:
            # Add the parameter group to the one with currently the fewest number of parameters
            owner = min(self.rank_to_numel.keys(), key=lambda k: self.rank_to_numel[k])
            self.param_to_rank[p] = owner
            self.rank_to_numel[owner] += p.numel()
            
            # Track the local params
            if owner == self.rank:
                local_params.append(p)
        
        if local_params:
            param_group["params"] = local_params # Keeps param group in tact with only local params

            # Required for the optimizer to track the param group for standard optimizer practices
            super().add_param_group(param_group)

            if hasattr(self, "optimizer"):
                # Adds the parameter group to optimizer
                self.optimizer.add_param_group(param_group)
            else:
                # Creates the optimizer with the single param group and kwargs
                self.optimizer = self.optimizer_cls([param_group], **self.defaults)
