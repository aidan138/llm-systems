import argparse
import torch
from torch import nn
from timeit import default_timer
from cs336_basics.nn.layers import TransformerLM
from  cs336_basics.nn.utils import cross_entropy_loss, softmax
from cs336_basics.nn.optim import AdamW
from cs336_basics.args import ModelArgs
import cs336_basics
from statistics import mean, stdev
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool
from torch import Tensor
import math
import pandas as pd
import os
from contextlib import nullcontext

MODEL_SPECS = {
    'small': {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    'medium': {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    'large': {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    'xl': {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    '2.7B': {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

CSV_PATH = './model_csvs'

os.makedirs(CSV_PATH, exist_ok=True)

@nvtx.range("Scaled dot product attention")
def annotated_sdp(Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,):
    batch, *rest, seq_len, feat_dim = Q.shape # 
    with nvtx.range("Computing attention scores"):
        sim_matrix = Q @ K.transpose(-2,-1)/math.sqrt(feat_dim) # (B, *rest, N, D) @ (B, *rest, D, M) -> (B, *rest, N, M)

    if mask is not None:
        sim_matrix = torch.where(condition=mask, input=sim_matrix, other=float('-inf'))
    
    with nvtx.range("computing softmax"):
        prob_matrix = softmax(sim_matrix, dim=-1)
    
    with nvtx.range("final matmul"):
        output = (prob_matrix @ V).view(batch, *rest, -1, feat_dim) # Returns B, ..., n, feat_dim

    return output

@nvtx.range('Benchmarking model')
def benchmark_model(mode: str, model_args: ModelArgs, x: torch.Tensor, use_optim: bool =False, num_warmups: int = 5, num_trials: int = 10, mixed_prec: bool = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x.to(device)
    targs = torch.randint(low=0, high=x.max().item(), size=(x.shape[0],1)).to(device) 
    backward = True if mode == 'forward-backward' else False
    with nvtx.range('Initializing model'):
        transformer = TransformerLM(model_args).to(device)
    optimizer = AdamW(transformer.parameters()) if use_optim else None

    prec_context = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_prec else nullcontext()
    # Warmup trials
    for _ in range(num_warmups):

        if optimizer:
            optimizer.zero_grad()
        else:
            transformer.zero_grad(set_to_none=True)
        
        with prec_context:
            logits = transformer(x) # B, N, V
            loss = cross_entropy_loss(logits, targs)

        if backward:
            loss.backward()

        if optimizer:
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize() # Wait for CUDA threads to finish
    
    # Actual trials
    times = []
    #torch.cuda.cudart().cudaProfilerStart()
    #torch.cuda.memory._record_memory_history(max_entries=1000000)
    for trial in range(num_trials):
        nvtx.range_push(f"step_{trial}")
        
        if optimizer:
            optimizer.zero_grad()
        else:
            transformer.zero_grad(set_to_none=True)

        start = default_timer()

        with nvtx.range("forward pass"):
            with prec_context:
                logits = transformer(x)
                loss = cross_entropy_loss(logits, targs)

            
        if backward:
            with nvtx.range("backward pass"):
                loss.backward()

        if optimizer:
            with nvtx.range("optimizer step"):
                optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        nvtx.range_pop()
        end = default_timer()
        times.append((end - start))

    #torch.cuda.cudart().cudaProfilerStop()
    # torch.cuda.memory._dump_snapshot().pickle("memory_snapshot.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)
    return {f'Mean ({mode.capitalize()})': mean(times), f'Std ({mode.capitalize()})':stdev(times)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking Language Model")

    # Main arguments
    parser.add_argument(
        "--size", type=str, default=None, choices=MODEL_SPECS.keys(), help="Model size"
    )
    parser.add_argument(
        "--mode", type=str, choices=['forward', 'forward-backward'], default='forward'
    )
    parser.add_argument("--seq-len", type=int)
    
    # Typically defaulted arguments
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    
    # For specific experimentation
    parser.add_argument("--num-warmups", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("-o", type=str, default=None)
    parser.add_argument("--prec", type=str, choices=['full', 'bfloat16'], default='full')

    args = parser.parse_args()

    save_path = f"{CSV_PATH}/{args.o}.csv" if args.o else None
    if args.size:
        kwargs = dict(args._get_kwargs()) | MODEL_SPECS[args.size]
    else:
        kwargs = dict(args._get_kwargs())

    model_args = ModelArgs(**kwargs)
    cs336_basics.nn.layers.sdp_attention = annotated_sdp

    with nvtx.range("define input"):
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    
    result = {
        'Size': args.size,
        'Context Length': args.seq_len
    }

    try:
        stats = benchmark_model(
            args.mode, model_args, x=x, num_warmups=args.num_warmups, num_trials=args.num_trials, use_optim=args.mode=='forward-backward', mixed_prec=args.prec=='bfloat16'
        )
        result |= stats
    
        
    except RuntimeError as e:
        
        if 'out of memory' in str(e):
            print(f"{args.size} model with context length {args.seq_len} tokens ran out of memory")
        result |= {'Mean (Forward)': 'Out of memory', 'Std (Forward)': 'Out of memory'}

    df = pd.DataFrame([result])
    if save_path and os.path.exists(save_path):
        old = pd.read_csv(save_path)
        df = pd.concat([old, df], ignore_index=True)
    if save_path:
        df.to_csv(save_path, index=False)

    print('\nResults:')
    print(df.to_markdown(index=False))
        
