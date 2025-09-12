import os
import torch
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.nn.layers import TransformerLM
from cs336_basics.args import ModelArgs
from cs336_basics.nn.utils import cross_entropy_loss
from cs336_basics.nn.optim import AdamW
from cs336_systems.ddp.sdp import OSDP
from timeit import default_timer
from cs336_systems.ddp.ddp import DDPIndividualParameters, DDPBucketed
from torch.cuda import nvtx
from cs336_systems.ddp.nvtx import push_nvtx, pop_nvtx
import logging
from torch.profiler import ProfilerActivity
from argparse import ArgumentParser
from typing import Any

SAVE_DIR = "./distributed_logs"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SPEC_PATH = './cs336_systems/model_specs.csv'
model_specs = pd.read_csv(MODEL_SPEC_PATH).set_index('model')
model_size = 'xl'
SIZES = [1, 10, 100, 1000]

logging.basicConfig(level=logging.DEBUG)


def extend_dataframe(save_file, df):
    assert '.csv' in save_file
    if os.path.exists(save_file):
        old = pd.read_csv(save_file)
        df = pd.concat([old, df])
    return df

def print_mem(prefix=""):
    if not torch.cuda.is_available():
        print(prefix + " (cpu only)")
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()/1024**2
    reserved = torch.cuda.memory_reserved()/1024**2
    peak = torch.cuda.max_memory_allocated()/1024**2
    print(f"{prefix} alloc={alloc:,} Mbytes | reserved={reserved:,} Mbytes | peak={peak:,} Mbytes")
    torch.cuda.reset_peak_memory_stats()
    return alloc, reserved, peak

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
    ddp_model.finish_gradient_synchronization()
    _, _, peak_pre_optim = print_mem("Pre-optimizer step")
    push_nvtx("optimizer step"); optimizer.step(); pop_nvtx()
    _, _, peak_post_optim = print_mem("Post-optimizer step")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    iter_end = default_timer()

    return (iter_end - iter_start), peak_pre_optim, peak_post_optim


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

def timed_ddp(
        rank: int,
        world_size: int,
        backend: str,
        data: torch.Tensor,
        targets: torch.Tensor,
        model_cls: torch.nn.Module,
        model_args: ModelArgs,
        config: dict[str, Any]
        ):
        
    # Setup the distributed backend
    device = setup(rank, world_size, backend)

    bucket_size_mb = config.get("bucket_size_mb")
    shard_optimizer = config.get("shard_optimizer", False)
    save_file = config.get("save_file")
    track_mem = config.get("track_mem", False)
    torch_profile = config.get("torch_profile", False)
    num_steps = config.get("num_steps", 10)
    num_warmups = config.get("num_warmups", 5)
    
    # Initialize the model and optimizer
    model = model_cls(model_args).to(device)
    ddp_model = (DDPBucketed(model, bucket_size_mb=bucket_size_mb) if bucket_size_mb else DDPIndividualParameters(model))
    data, targets = torch.chunk(data.to(device), world_size, dim=0), torch.chunk(targets.to(device), world_size, dim=0)
    optimizer = OSDP(ddp_model.parameters(), AdamW) if shard_optimizer else AdamW(ddp_model.parameters())
    logging.info(f"Created {type(ddp_model)} model with {sum(param.numel() for param in ddp_model.parameters())} Parameters and {type(optimizer)} optimizer")

    # To track peak memory usage
    peak_init, peak_pre_optim, peak_post_optim = 0, 0, 0

    _,_, peak_init = print_mem("Initialized model")
    
    # For DDP
    local_batch = data[rank]
    local_targets = targets[rank]
    logging.info("Warming up")
    for i in range(num_warmups):
        ddp_step(ddp_model=ddp_model, optimizer=optimizer, local_batch=local_batch, local_targets=local_targets)
    dist.barrier()
    
    logging.info("Benchmarking")
    iter_times = []
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step in range(num_steps):
            iter_time, tmp_pre_optim, tmp_post_optim = ddp_step(
                                                            ddp_model=ddp_model,
                                                            optimizer=optimizer,
                                                            local_batch=local_batch,
                                                            local_targets=local_targets
                                                        )
            
            peak_pre_optim = max(tmp_pre_optim, peak_pre_optim)
            peak_post_optim = max(tmp_post_optim, peak_post_optim)

            iter_times.append(
                iter_time
            )

    # Print out profiling details        
    if torch_profile and rank == 0:
        ka = prof.key_averages()
        table = ka.table(sort_by="cuda_time_total",
                                        max_name_column_width=80,
                                        row_limit=50)
        
        print("\nRank", rank, "-Table:")
        print(table)
        prof.export_chrome_trace(f"./trace_{bucket_size_mb}mb.json")
        print("="*80)

    # Take mean across iterations
    iter_times = torch.tensor(iter_times, device=device).mean(dim=-1)

    # Get full benchmark data
    all_ranks_iter_times = [torch.empty_like(iter_times) for _ in range(world_size)]
    dist.all_gather(tensor_list=all_ranks_iter_times, tensor=iter_times)
    
    # Take mean across ranks 
    rank_avg_iter = torch.tensor(all_ranks_iter_times, device=device).mean(dim=0) 
    DDP_Type = f"{backend} DDP Wrapped"
    DDP_Type += (f" ({bucket_size_mb} MB buckets)" if bucket_size_mb else "")
    DDP_Type += (f" Sharded Optimizer" if shard_optimizer else "")
    if rank == 0:
        row = dict(
            DDP_Type=DDP_Type,
            Model_Size=model_size,
            Avg_Comm_Time_s=float('nan'),
            Avg_Iter_Time_s=rank_avg_iter.item(),
        )
        row = (row | dict(
                Peak_Init_Mem_mb=peak_init,
                Peak_Pre_Optim_Mem_mb=peak_pre_optim,
                Peak_Post_Optim_Mem_mb=peak_post_optim,
            ) if track_mem else row)
        df = pd.DataFrame([row])
        
        # Save the dataframe
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
    parser = ArgumentParser()
    parser.add_argument("--shard-optim", type=bool, default=True)
    parser.add_argument("--buckets", type=bool, default=False)
    parser.add_argument("--track-mem", type=bool, default=True)
    parser.add_argument("--torch-prof", type=bool, default=False)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--num-warmups", type=int, default=5)

    args = parser.parse_args()

    # NOTE these specifications are for 2 GPUs (80GB/device) setup to fit within memory
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        backend = "nccl"
        model_size = "xl"
        max_batch_size = 16
        max_seq_len = 32
        vocab_size=10000
        num_layers=model_specs.loc[model_size]['num_layers']
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
    config = {
        "bucket_size_mb" : None, # Determines whether to use bucketed DDP
        "shard_optimizer" : args.shard_optim,
        "save_file": save_file,
        "track_mem": args.track_mem,
        "torch_profile": args.torch_prof,
        "num_steps": args.num_steps,
        "num_warmups": args.num_warmups
    }
    if args.buckets:
        for size_mb in SIZES:
            config["bucket_size_mb"] = size_mb
            mp.spawn(timed_ddp, args=(world_size, backend, X, y, TransformerLM, model_args, config), nprocs=world_size)
    else:
        mp.spawn(timed_ddp, args=(world_size, backend, X, y, TransformerLM, model_args, config), nprocs=world_size)


if __name__ == '__main__':
    main()
