import argparse
import torch
from cs336_systems.flash_attention.flash_attention import FlashAttention
from cs336_basics.nn.layers import sdp_attention
from triton.testing import do_bench
import itertools
import pandas as pd
import os

ATTN_DIR = './triton_stats'
os.makedirs(ATTN_DIR, exist_ok=True)
torch._functorch.config.donated_buffer = False
torch.set_float32_matmul_precision('high')

flash = torch.compile(FlashAttention.apply)

def causal_unfused(q, k, v):
    seq_len = q.shape[-2]
    seq = torch.arange(0, seq_len, device='cuda')
    qi = seq.view(-1, 1)
    kj = seq.view(1, -1)
    mask = qi >= kj
    return sdp_attention(q,k,v, mask)
    
def get_bench_res(fn):
    try:
        torch.cuda.synchronize()
        res = do_bench(fn)
        torch.cuda.synchronize()

    except torch.OutOfMemoryError:
        res = float('nan')
    return res


def run_benchmark(precision, num_heads, seq_len, head_dim):
    q, k, v = torch.randn((3, num_heads, seq_len, head_dim), dtype=precision, device='cuda', requires_grad=True)
    o = torch.empty_like(q, dtype=precision, device='cuda', requires_grad=True)

    def reset_grads():
        q.grad, k.grad, v.grad = None, None, None

    def run_backward():
        reset_grads()
        loss = o.sum()
        loss.backward(retain_graph=True)

    def unfused_forward():
        reset_grads()
        causal_unfused(q,k,v)

    def unfused_forward_backward():
        reset_grads()
        o = causal_unfused(q,k,v)
        loss = o.sum()
        loss.backward()
    
    o = causal_unfused(q,k,v)
    unfused_fw_ms = get_bench_res(unfused_forward)
    unfused_bw_ms = get_bench_res(run_backward)
    unfused_fwbw_ms = get_bench_res(unfused_forward_backward)

    def flash_forward():
        reset_grads()
        flash(q, k, v, True)
    
    def flash_forward_backward():
        reset_grads()
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    o = flash(k,q,v, True)
    flash_fw_ms = get_bench_res(flash_forward)
    flash_bw_ms = get_bench_res(run_backward)
    flash_fwbw_ms = get_bench_res(flash_forward_backward)

    return dict(
        unfused_fw_ms=unfused_fw_ms,
        unfused_bw_ms=unfused_bw_ms,
        unfused_fwbw_ms=unfused_fwbw_ms,
        flash_fw_ms=flash_fw_ms,
        flash_bw_ms=flash_bw_ms,
        flash_fwbw_ms=flash_fwbw_ms
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim-min", type=int, default=16)
    parser.add_argument("--head-dim-max", type=int, default=128)
    parser.add_argument("--seq-len-min", type=int, default=128)
    parser.add_argument("--seq-len-max", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16, help="# of attention heads")
    parser.add_argument("--repeats", type=int, default=30 )
    parser.add_argument("--precisions", type=str, choices=["full", "both"], default="both")
    parser.add_argument("-o", type=str, default=None)

    assert torch.cuda.is_available(), "Need a CUDA capable device for flash attention"
    args = parser.parse_args()
    save_path = f'{ATTN_DIR}/{args.o}.csv' if args.o else None

    seq_lens = []
    s = args.seq_len_min
    while s <= args.seq_len_max:
        seq_lens.append(s)
        s *= 2
    
    head_dims = []
    d = args.head_dim_min
    while d <= args.head_dim_max:
        head_dims.append(d)
        d *= 2
    
    dtypes = [torch.float32] + ([] if args.precisions == "full" else [torch.bfloat16])

    results = []
    i=0
    for precision, seq_len, head_dim in itertools.product(dtypes, seq_lens, head_dims):
        row = dict(
            precision=precision,
            seq_len=seq_len,
            head_dim=head_dim
        )
        
        row.update(run_benchmark(precision, num_heads=args.heads, seq_len=seq_len, head_dim=head_dim))
        results.append(row)


    try:
        df = pd.DataFrame(results).sort_values(['precision', 'seq_len','head_dim'])
        if save_path:
            df.to_csv(save_path, index=False)
        print("\nOutput latencies in ms")
        print(df.to_markdown(index=False))
        
    except IOError:
        pass

if __name__ == '__main__':
    main()