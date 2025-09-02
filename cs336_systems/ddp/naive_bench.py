import os
import torch
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.nn.layers import TransformerLM
from cs336_basics.args import ModelArgs
from cs336_basics.nn.utils import cross_entropy_loss
from cs336_basics.nn.optim import AdamW
from timeit import default_timer

SAVE_DIR = "./distributed_logs"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SPEC_PATH = './cs336_systems/model_specs.csv'
model_specs = pd.read_csv(MODEL_SPEC_PATH).set_index('model')
model_size = 'xl'


def extend_dataframe(save_file, df):
    assert '.csv' in save_file
    if os.path.exists(save_file):
        old = pd.read_csv(save_file)
        df = pd.concat([old, df])
    return df


def ddp_train_step(rank, world_size, data, targets, model, optimizer):
    start = default_timer()
    batch_size = data.size(0)
    local_batch_size = batch_size // world_size
    offset = rank*local_batch_size
    local_data = data[offset:offset+local_batch_size, ...]
    local_targets = targets[offset:offset+local_batch_size, ...]
    logits = model(local_data)
    loss = cross_entropy_loss(logits, local_targets)
    print(f"loss is {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    if rank == 0: print(loss.item())
    comm_start = default_timer()
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # Perform all reduce average reduction to average gradients
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
    dist.barrier()
    comm_end = default_timer()

    # Step with the averaged gradients
    optimizer.step()
    end = default_timer()
    synchronize()
    return comm_end-comm_start, end-start

def ddp_reduced_com_step(rank, world_size, data, targets, model, optimizer):
    start = default_timer()
    batch_size = data.size(0)
    local_batch_size = batch_size // world_size
    offset = rank*local_batch_size
    local_data = data[offset:offset+local_batch_size, ...]
    local_targets = targets[offset:offset+local_batch_size, ...]
    logits = model(local_data)
    loss = cross_entropy_loss(logits, local_targets)
    print(f"loss is {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    if rank == 0: print(loss.item())
    comm_start = default_timer()
    with torch.no_grad():
        grads_list = list(param.grad for param in model.parameters())
        flat_grads = torch._utils._flatten_dense_tensors(grads_list)
        # Perform all reduce average reduction to average gradients
        dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.AVG, async_op=False)
        grads_list = torch._utils._unflatten_dense_tensors(flat_grads, grads_list)
        
        for param, grad in zip(model.parameters(), grads_list):
            param.grad.copy_(grad)

    dist.barrier()
    comm_end = default_timer()

    # Step with the averaged gradients
    optimizer.step()
    end = default_timer()
    synchronize()
    return comm_end-comm_start, end-start

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "29502"
    if backend == 'nccl':
        local_rank = None
        num_devices = torch.cuda.device_count()
        if num_devices > 0:
            local_rank = rank % num_devices
        else:
            raise ValueError
        device = f"cuda:{local_rank}"
        
    else:
        device = 'cpu'
    dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=rank)
    return device

def cleanup():
    dist.destroy_process_group()

def timed_naive_ddp(rank: int, world_size: int, backend: str, data: torch.Tensor, targets: torch.Tensor, model_cls: torch.nn.Module, model_args: ModelArgs, save_file=None, num_steps: int=10, num_warmups: int = 5):
    device = setup(rank, world_size, backend)
    
    model = model_cls(model_args).to(device)
    data, targets = data.to(device), targets.to(device)
    optimizer = AdamW(model.parameters())
    print(f"Rank {rank} model parameters before broadcast: {list(model.parameters())[0]}")

    # Unify the parameters
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(tensor=param, src=0, async_op=False)
    print(f"\nRank {rank} model parameters after broadcast: {list(model.parameters())[0]}")

    for _ in range(num_warmups):
        ddp_train_step(rank=rank, world_size=world_size, data=data, targets=targets, model=model, optimizer=optimizer)

    dist.barrier()
    
    iter_times, comm_times = [], []
    for step in range(num_steps):
        comm_time, iter_time = ddp_train_step(rank=rank, world_size=world_size, data=data, targets=targets, model=model, optimizer=optimizer)
        iter_times.append(iter_time)
        comm_times.append(comm_time)

    dist.barrier()
    
    # Take mean across iterations
    iter_times = torch.tensor(iter_times, device=device).mean(dim=-1)
    comm_times = torch.tensor(comm_times, device=device).mean(dim=-1)

    # Get full benchmark data
    all_ranks_iter_times = [torch.empty_like(iter_times) for _ in range(world_size)]
    all_ranks_comm_times = [torch.empty_like(comm_times) for _ in range(world_size)]
    dist.all_gather(tensor_list=all_ranks_iter_times, tensor=iter_times)
    dist.all_gather(tensor_list=all_ranks_comm_times, tensor=comm_times)
    
    # Take mean across ranks 
    rank_avg_iter = torch.tensor(all_ranks_iter_times, device=device).mean(dim=0) 
    rank_avg_comm = torch.tensor(all_ranks_comm_times, device=device).mean(dim=0)
    if rank == 0:
        row = dict(
            DDP_Type="Naive",
            Model_Size=model_size,
            Avg_Comm_Time_s=rank_avg_comm.item(),
            Avg_Iter_Time_s=rank_avg_iter.item()
        )
        df = pd.DataFrame([row])
        if save_file:
            df = extend_dataframe(save_file, df)
            df.to_csv(save_file)

        print("Results from training:\n")
        print(df.to_markdown(index=False))

    cleanup()


def main():
    print(model_specs)
    model_dict = model_specs.loc[model_size].to_dict()
    max_batch_size = 16
    max_seq_len = 32
    vocab_size=10000
    data_model_args = dict(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model_dict.update(data_model_args)
    model_args = ModelArgs(**model_dict)
    save_file = './distributed_logs/ddp_benchmarks.csv'
    world_size=2
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        backend = "nccl"
    else:
        backend = "gloo"
    print(backend)
    data = torch.randint(0, vocab_size, (max_batch_size*world_size, max_seq_len+1))
    # construct dummy training batch
    X = data[:, :-1]
    y = data[:, 1:]
    mp.spawn(timed_naive_ddp, args=(world_size, backend, X, y, TransformerLM, model_args, save_file), nprocs=world_size)
    

if __name__ == '__main__':
    main()
