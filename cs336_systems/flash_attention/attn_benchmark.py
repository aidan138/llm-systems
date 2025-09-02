from cs336_basics.nn.layers import sdp_attention
import itertools
import argparse
import torch
from timeit import default_timer
from statistics import mean, stdev
import pandas as pd
import os
from tqdm import tqdm
import gc

ATTN_DIR = './attn_stats'
os.makedirs(ATTN_DIR, exist_ok=True)

bsz = 8
head_dims = [16,32,64,128]
seq_lens = [256, 1024, 4096, 8192, 16384] 

def print_mem(msg=""):
    torch.cuda.synchronize()
    print(msg)
    print("allocated:", torch.cuda.memory_allocated() / (1024**3), "GB")
    print("reserved: ", torch.cuda.memory_reserved() / (1024**3), "GB")
    print("max allocated:", torch.cuda.max_memory_allocated() / (1024**3), "GB")
    print(torch.cuda.memory_summary()[:1000])
    gc.collect()

def synchronize():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def benchmark_attn(head_dim: int, seq_len: int, bsz: int=8, warmup_iters: int=5, num_trials: int=100, device='cuda'):
    Q, K, V = tuple(torch.randn((bsz, seq_len, head_dim), requires_grad=True, device=device) for _ in range(3))
    mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device)).bool()
    print(f'Benchmarking attention with head_dim: {head_dim} and seq_len: {seq_len}')

    for _ in range(warmup_iters):
        attn_output = sdp_attention(Q, K, V, mask)
        attn_output.backward(torch.ones_like(attn_output))
    synchronize()

    forward_times, backward_times = [], []
    mem_before_backward = None
    for trial in tqdm(range(num_trials), total=num_trials, desc="Trials"):

        f_start = default_timer()
        attn_output = sdp_attention(Q,K,V,mask)
        synchronize()
        f_end = default_timer()
        forward_times.append((f_end-f_start))

        if mem_before_backward is None:
            mem_before_backward = torch.cuda.memory_allocated()
            print_mem("Finished first forward pass")

        b_start = default_timer()
        attn_output.backward(torch.ones_like(attn_output))
        synchronize()
        b_end = default_timer()
        backward_times.append((b_end - b_start))

    return {'Head Dimension': head_dim, 'Sequence Length': seq_len, 'Forward Memory (MB)': mem_before_backward / (1024**2),
            'Total Forward Time': sum(forward_times), 'Avg Forward Pass': mean(forward_times), 'Forward Pass Std': stdev(forward_times),
            'Total Backward Time': sum(backward_times), 'Avg Backward Pass': mean(backward_times), 'Backward Pass Std': stdev(backward_times)}


def main(save_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    product = itertools.product(head_dims, seq_lens)
    out_res = []
    for dim, seq in product:
        try:
            results = benchmark_attn(dim, seq, device=device)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                results = {'Head Dimension': dim, 'Sequence Length': seq,
                            'Avg Forward Pass': 'OOM', 'Forward Pass Std': 'OOM',
                            'Avg Backward Pass': 'OOM', 'Backward Pass Std': 'OOM'}
            else:
                raise e
            
        out_res.append(results)

    df = pd.DataFrame(out_res)
    if save_path and os.path.exists(save_path):
        prev_res = pd.read_csv(save_path)
        df = pd.concat([prev_res, df], ignore_index=True)
    if save_path:
        df.to_csv(save_path)
    print('Results\n', df.to_markdown())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str)
    args  = parser.parse_args()

    main(
        save_path= f'{ATTN_DIR}/{args.o}.csv' if args.o else None
    )
