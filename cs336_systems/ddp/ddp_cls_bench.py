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
from cs336_systems.ddp.ddp import DDPIndividualParameters
from torch.cuda import nvtx
from cs336_systems.ddp.nvtx import push_nvtx, pop_nvtx
import logging

SAVE_DIR = "./distributed_logs"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SPEC_PATH = './cs336_systems/model_specs.csv'
model_specs = pd.read_csv(MODEL_SPEC_PATH).set_index('model')
model_size = 'xl'

logging.basicConfig(level=logging.DEBUG)


def extend_dataframe(save_file, df):
    assert '.csv' in save_file
    if os.path.exists(save_file):
        old = pd.read_csv(save_file)
        df = pd.concat([old, df])
    return df

@nvtx.range("Training step")
def ddp_step(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer, local_batch: torch.Tensor, local_targets: torch.Tensor):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    iter_start = default_timer()
    
    # Forward and backward pass
    push_nvtx("forward"); logits = ddp_model(local_batch); pop_nvtx()
    push_nvtx("loss"); loss = cross_entropy_loss(logits, local_targets); pop_nvtx()
    optimizer.zero_grad()
    push_nvtx("backward pass"); loss.backward(); pop_nvtx()
    push_nvtx("optimizer step"); optimizer.step(); pop_nvtx
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    iter_end = default_timer()

    return iter_end - iter_start


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
        torch.cuda.set_device(local_rank)
        
    else:
        device = 'cpu'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def cleanup():
    dist.destroy_process_group()

def timed_ddp(rank: int, world_size: int, backend: str, data: torch.Tensor, targets: torch.Tensor, model_cls: torch.nn.Module, model_args: ModelArgs, save_file=None, num_steps: int=10, num_warmups: int = 5):
    device = setup(rank, world_size, backend)
    
    model = model_cls(model_args).to(device)
    ddp_model = DDPIndividualParameters(model)
    data, targets = torch.chunk(data.to(device), world_size, dim=0), torch.chunk(targets.to(device), world_size, dim=0)
    optimizer = AdamW(ddp_model.parameters())
    
    local_batch = data[rank]
    local_targets = targets[rank]
    logging.info("Warming up")
    for _ in range(num_warmups):
        ddp_step(ddp_model=ddp_model, optimizer=optimizer, local_batch=local_batch, local_targets=local_targets)

    dist.barrier()
    
    logging.info("Benchmarking")
    iter_times = []
    for step in range(num_steps):
        iter_times.append(
            ddp_step(ddp_model=ddp_model, optimizer=optimizer, local_batch=local_batch, local_targets=local_targets)
        )
            
    # Take mean across iterations
    iter_times = torch.tensor(iter_times, device=device).mean(dim=-1)

    # Get full benchmark data
    all_ranks_iter_times = [torch.empty_like(iter_times) for _ in range(world_size)]
    dist.all_gather(tensor_list=all_ranks_iter_times, tensor=iter_times)
    
    # Take mean across ranks 
    rank_avg_iter = torch.tensor(all_ranks_iter_times, device=device).mean(dim=0) 
    if rank == 0:
        row = dict(
            DDP_Type="DDP Wrapped",
            Model_Size=model_size,
            Avg_Comm_Time_s=float('nan'),
            Avg_Iter_Time_s=rank_avg_iter.item()
        )
        df = pd.DataFrame([row])
        if save_file:
            df = extend_dataframe(save_file, df)
            df.to_csv(save_file, index=False)

        logging.info("Results from benchmarking:\n")
        logging.info(df.to_markdown(index=False))

    dist.barrier()
    cleanup()


def main():
    save_file = './distributed_logs/ddp_benchmarks.csv'
    world_size=2

    # NOTE these specifications are for 2 GPUs (80GB/device) setup to fit within memory
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        backend = "nccl"
        model_size = "xl"
        max_batch_size = 16
        max_seq_len = 32
        vocab_size=10000
        num_layers=model_dict[model_size]['num_layers']
    else:
        # NOTE these constraints were derived for local testing
        backend = "gloo"
        model_size = "small"
        max_batch_size = 2
        max_seq_len = 4
        vocab_size=100
        num_layers = 2

    logging.info(model_specs)
    model_dict = model_specs.loc[model_size].to_dict()
    
    data_model_args = dict(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        num_layers=num_layers
    )
    model_dict.update(data_model_args)
    model_args = ModelArgs(**model_dict)

    logging.debug(backend)
    data = torch.randint(0, vocab_size, (max_batch_size*world_size, max_seq_len+1))
    # construct dummy training batch
    X = data[:, :-1]
    y = data[:, 1:]
    mp.spawn(timed_ddp, args=(world_size, backend, X, y, TransformerLM, model_args, save_file), nprocs=world_size)
    

if __name__ == '__main__':
    main()
