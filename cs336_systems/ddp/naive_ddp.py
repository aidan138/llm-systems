import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.functional as f
import argparse
from cs336_basics.nn.layers import Linear, silu
from cs336_basics.nn.utils import cross_entropy_loss
from torch.optim import SGD

from copy import deepcopy

DEFAULT_WORLD_SIZE = 4
d_model = 2
seq_len = 1
d_ff = 4
seed=42

model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)
save_path = f"{model_dir}/naive_ddp.pt"

class ToyModel(torch.nn.Module):
    def __init__(self, d_model=d_model, d_ff=d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.w2(silu(self.w1(x)))

def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "29502"
    if torch.cuda.is_available():
        local_rank = None
        num_devices = torch.cuda.device_count()
        if num_devices > 0:
            local_rank = rank % num_devices
        else:
            raise ValueError
        device = f"cuda:{local_rank}"
        
    else:
        device = 'cpu'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def cleanup():
    dist.destroy_process_group()

def naive_ddp(rank: int, world_size: int, backend: str, data: torch.Tensor, targets: torch.tensor, num_steps: int, model: torch.nn.Module):
    device = setup(rank, world_size, backend)
    device = 'cpu'
    
    #print(f"Rank {rank} model parameters before broadcast: {list(model.parameters())}")
    torch.manual_seed(seed)

    model = model.to(device)
    # Unify the parameters
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(tensor=param, src=0, async_op=False)
    
    dist.barrier()
    if rank == 0:
        print(f"\nRank {rank} model parameters after broadcast: {list(model.parameters())}")
    batch_size = data.size(0)
    optimizer = SGD(model.parameters(), lr=0.1)
    torch.manual_seed(seed)
    for step in range(num_steps):
        local_batch_size = batch_size // world_size
        start = rank*local_batch_size
        local_data = data[start:start+local_batch_size, ...]
        local_targets = targets[start:start+local_batch_size, ...]
        logits = model(local_data)
        loss = cross_entropy_loss(logits, local_targets)
        print(f"loss is {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                # Perform all reduce average reduction to average gradients
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Step with the averaged gradients
        optimizer.step()
    
    dist.barrier()
    if rank == 0:
        print(f"\nRank {rank} final model parameters: {list(model.parameters())}")
        torch.save(model, save_path)
    cleanup()

def demo_naive_ddp(batch_size: int, model_cls: torch.nn.Module = ToyModel):
    if torch.cuda.is_available():
        backend = "nccl"
        world_size = torch.cuda.device_count()
    else:
        backend = "gloo"
        world_size = DEFAULT_WORLD_SIZE
    backend,  world_size = "gloo", DEFAULT_WORLD_SIZE

    assert batch_size % world_size == 0, "Batch size must be divisible by the world size"
    torch.manual_seed(seed)
    serial_model = model_cls()
    ddp_model = deepcopy(serial_model) # Identical copy of the model
    data = torch.randn((batch_size, 1, d_model)) # dummy data
    with torch.inference_mode():
        output_shape = serial_model(data).shape
    labels = torch.randint(0, d_model, output_shape[:-1]) # Dummy output
    num_steps = 1
    
    print(f"Optimizing the serial model with parameters: {list(serial_model.parameters())}")
    serial_optim = SGD(serial_model.parameters(), lr=0.1)
    torch.manual_seed(seed)
    for iter in range(num_steps):
        logits = serial_model(data)
        loss = cross_entropy_loss(logits, labels)
        print(f"loss is {loss.item()}")
        serial_optim.zero_grad()
        loss.backward()
        print()
        serial_optim.step()

        for serial_param, ddp_param in zip(
            serial_model.parameters(),
            ddp_model.parameters()
        ):
            if serial_param.requires_grad and ddp_param.requires_grad:
                assert not torch.allclose(serial_param.cpu(), ddp_param.cpu(), rtol=1e-3, atol=1e-4)
            else:
                assert torch.allclose(serial_param.cpu(), ddp_param.cpu(), rtol=1e-3, atol=1e-4)
    
    print(f"Finished serial training, final parameters: {list(serial_model.parameters())}")
    print("Starting ddp training")
    mp.spawn(naive_ddp, args=(world_size, backend, data, labels, num_steps, ddp_model), nprocs=world_size)
    print("Finished ddp training")
    ddp_model = torch.load(save_path, weights_only=False)
    for serial_param, ddp_param in zip(
        serial_model.parameters(),
        ddp_model.parameters()
    ):
        assert torch.allclose(serial_param.cpu(), ddp_param.cpu(), rtol=1e-2, atol=1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)    
    args = parser.parse_args()
    demo_naive_ddp(args.batch_size, ToyModel)
    

if __name__ == "__main__":
    main()