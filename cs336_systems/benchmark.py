import argparse
import torch
from torch import nn
from timeit import default_timer
from cs336_basics.nn.layers import TransformerLM
from  cs336_basics.nn.utils import cross_entropy_loss
from cs336_basics.args import ModelArgs
from statistics import mean, stdev
import torch.cuda.nvtx as nvtx

def run_forward_pass(model: nn.Module, x: torch.Tensor):
    return model(x)


def benchmark_model(mode: str, model: nn.Module, x: torch.Tensor, num_warmups: int = 5, num_trials: int = 10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    targs = torch.randint(low=0, high=x.max().item(), size=(x.shape[0],1)).to(device) if mode == 'forward-backward' else None
    model.to(device)

    # Warmup trials
    for _ in range(num_warmups):
        logits = model(x) # B, N, V
        if targs is not None:
            loss = cross_entropy_loss(logits, targs)
            loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize() # Wait for CUDA threads to finish
    
    # Actual trials
    times = []
    for trial in range(num_trials):
        start = default_timer()

        logits = model(x)

        if targs is not None:
            loss =cross_entropy_loss(logits, targs)
            loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = default_timer()
        times.append((end - start))

    return {'mean': mean(times), 'std':stdev(times)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-warmups", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--num-heads", type=int, default=4)

    parser.add_argument(
        "--mode", type=str, choices=['forward', 'forward-backward'], default='forward'
    )

    args = parser.parse_args()
    kwargs = dict(args._get_kwargs())
    model_args = ModelArgs(**kwargs)

    # with nvtx.range('define model'):
    transformer = TransformerLM(model_args)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    stats_dict = benchmark_model(
            args.mode, transformer, x=x, num_warmups=args.num_warmups, num_trials=args.num_trials
        )

    print(f"The avg {args.mode} time: {stats_dict['mean']} | std {stats_dict['std']}")
