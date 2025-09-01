import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import product
from timeit import default_timer
from statistics import mean
import pandas as pd
from tqdm import tqdm

MB_SIZE = 1024**2
SAVE_DIR = './distributed_logs' 
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = f"{SAVE_DIR}/all_scatter.csv"

backends = ["gloo", "nccl"]
num_processes = [2, 4, 6]
data_mem_size = [MB_SIZE, MB_SIZE*10, MB_SIZE*100, MB_SIZE*1000]


def sync_devices():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

def setup(rank, world_size, protocol):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"
    if protocol == 'nccl':
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    dist.init_process_group(protocol, rank=rank, world_size=world_size)
    return device

def cleanup():
    dist.destroy_process_group()

def run_dist_all_reduce(rank, world_size, protocol, memory, num_warmups=5):
    device = setup(rank, world_size, protocol)
    dtype = torch.float32
    tensor_numel = memory // dtype.itemsize
    data = torch.randint(0, tensor_numel, (tensor_numel,), device=device)

    for _ in range(num_warmups):
        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    sync_devices()

    print(f"Rank {rank} data before all-reduce: {data}")
    start = default_timer()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    end = default_timer()
    sync_devices()
    print(f"Rank {rank} data after all-reduce({end-start:.2f}ms): {data}")
    res = [torch.empty((1,), device=device) for _ in range(world_size)]
    dist.all_gather(tensor_list=res, tensor=torch.tensor([end-start]))
    if rank == 0:
        avg_time = mean([t.item() for t in res])
        print(f"Mean all-reduce time: {avg_time:.2f}ms")
        res = dict(
            backend=protocol,
            num_processes=world_size,
            data_size_mb=memory//(1024**2),
            rank_avg=avg_time
        )
        df = pd.DataFrame([res])
        if os.path.exists(save_path):
            old = pd.read_csv(save_path)
            df = pd.concat([old, df])
        df.to_csv(save_path, index=False)
    
    cleanup()

def main():
    num_iters = len(backends) * len(num_processes) * len(data_mem_size)
    for backend, world_size, memory in tqdm(product(backends, num_processes, data_mem_size), desc="Distributed Workloads", total=num_iters):
        if memory == data_mem_size[-1] and backend == backends[0]:
            continue
        print(f"\nRunning {backend} distributed {memory//1024**2}MB workload with world_size: {world_size}")
        mp.spawn(run_dist_all_reduce, args=(world_size, backend, memory), nprocs=world_size, join=True)
    
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()