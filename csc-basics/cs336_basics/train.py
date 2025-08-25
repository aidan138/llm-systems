import torch
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch.nn as nn
from nn.layers import TransformerLM
from args import ModelArgs, TrainingArgs
from cs336_basics.nn.optim import AdamW
from cs336_basics.nn.utils import cross_entropy_loss, gradient_clipping, lr_cosine_scheduling
from cs336_basics.train_utils import load_checkpoint, save_checkpoint, get_batch
import numpy as np
import numpy.typing as npt
import wandb
from pydantic import ValidationError
import time
from dotenv import load_dotenv
import os
import argparse

load_dotenv()
#wandb_username = os.getenv('WANDBUSERNAME')
project_name = os.getenv('WANDBPROJECTNAME')
os.makedirs('models', exist_ok=True)

total_tokens_processed = 327680000
batch_size = 16
context_length = 256

iterations = total_tokens_processed // (batch_size*context_length) + 1
warmup_iters = int(0.0125 * iterations)
cos_iters = iterations - warmup_iters -1000


# Current args
# See args.py for more options
try:
    model_args = ModelArgs(
        # LM config
        d_model=512,
        d_ff=1344,
        vocab_size=10000,
        rope_theta=10000,
        
        # Attention config
        num_layers=4,
        num_heads=16,

        # Inference time parameters
        max_batch_size=batch_size,
        max_seq_len=context_length, # Consideration for inference time
    )



    train_args = TrainingArgs(
        # Train Loop
        iterations=iterations,
        checkpoint_freq=int(1000 * 32/batch_size), # Save every 1000 iters at batch size of 32
        batch_size=batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float32,
        save_path=r'models\Larger_lr.pt',
        train_path=r'data\TinyStoriesV2-GPT4-train.npy',
        cv_path=r'data\TinyStoriesV2-GPT4-valid.npy',
        load_path=None,

        # Optimizer
        lr_max=1.4e-3 * (batch_size/32), # Scale the lr based on batch size (Used 32 to find this value)
        weight_decay=0.1,
        betas=(0.9,0.95),

        # Learning rate scheduler
        lr_min=1.4e-4 * (batch_size/32),
        warmup_iterations=warmup_iters,
        cos_iterations=cos_iters, # Perform 1000 iterations at mins

        # Gradient Clipping
        max_l2_norm=1.0,

        # Logging Parameters
        log_cv_iterations=int(5000 * 32/batch_size),
        log_train_iterations=10,
        train_loss_alpha=0.1,

        context_length=context_length # For the model
    )
except ValidationError as e:
    print(f"Validation error: {e.errors()}")


def get_cv_loss(model: nn.Module, val_set: npt.ArrayLike, batch_size: int, context_length: int, device:torch.device):
    loss_total = 0
    n = 0
    # Evaluate over the entire validation set in batches
    for i in range(0, len(val_set)-(context_length+1) * batch_size, batch_size*context_length):
        X_cv, y_cv = get_batch(val_set[i : i+batch_size*context_length+1],
                               batch_size,
                               context_length,
                               device)
        logits = model(X_cv)
        loss_total += cross_entropy_loss(logits, y_cv).item()
        n+= 1
        model.current_pos = 0
        
    return loss_total / n


def train(model: nn.Module, train_args: TrainingArgs, run: wandb.Run = None):
    
    optimizer = AdamW(
        params = model.parameters(),
        lr = train_args.lr_max,
        betas = train_args.betas,
        weight_decay = train_args.weight_decay
    )

    # Reload the provided checkpoint
    current_iter = 0 if train_args.load_path is None else load_checkpoint(train_args.load_path, model, optimizer)
    print(f"Model initialized on iteration {current_iter}")

    # mmep the file into memory for lazy batching
    train_set = np.load(train_args.train_path, mmap_mode='r')
    val_set = np.load(train_args.cv_path, mmap_mode='r')
    iterations, checkpoint_freq = train_args.iterations, train_args.checkpoint_freq
    log_cv_iterations, log_train_iterations = train_args.log_cv_iterations, train_args.log_train_iterations
    best_cv = float('inf')
    running_loss = 0
    batch_size = train_args.batch_size
    context_length = train_args.context_length
    device = train_args.device
    alpha = train_args.train_loss_alpha
    max_l2_norm = train_args.max_l2_norm if train_args.max_l2_norm else False

    # For loading optimizer from checkpointing
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    model.to(device)
    print(f"Model is on device: {device}")

    iterations -= current_iter
    start_time = time.perf_counter()
    for i in range(iterations):
        full_iters = i + current_iter
        # Forward pass
        X, y = get_batch(train_set, batch_size, context_length, device)
        logits = model(X)
        loss = cross_entropy_loss(logits, y)
        running_loss = alpha * running_loss + (1 - alpha) * loss.item()

        # Perform an update step
        optimizer.zero_grad()

        # lr scheduling
        if train_args.lr_min:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_cosine_scheduling(t=full_iters,
                                                        lr_max=train_args.lr_max,
                                                        lr_min=train_args.lr_min,
                                                        t_w=train_args.warmup_iterations,
                                                        t_c=train_args.cos_iterations)

        # Backprop
        loss.backward()

        # Gradient clipping
        if max_l2_norm:
            gradient_clipping(model.parameters(), max_l2_norm)

        # Parameter update
        optimizer.step()


        if i != 0 and i % checkpoint_freq == 0:
            print(f"Checkpointing at iteration {full_iters}")
            save_checkpoint(model, optimizer, full_iters, train_args.save_path)

        # Logging statistics
        if i!= 0 and full_iters % log_cv_iterations == 0:
            model.eval()
            with torch.no_grad():
                cv_loss = get_cv_loss(model, val_set, batch_size, context_length, device)
            print(f"CV loss at iteration {full_iters} is {cv_loss:.6f}")
            if run:
                run.log({'CV Loss': cv_loss, 'Step': full_iters, 'Clock Time': time.perf_counter() - start_time})
            best_cv = best_cv if best_cv >= cv_loss else cv_loss
            model.current_pos = 0 # reset the cache idx each time during training to not cache
            model.train()

        if i % log_train_iterations == 0:
            iter_end = time.perf_counter()
            print(f"Iteration {i}({full_iters}) / {iterations}")

            print(f"Training loss at iteration {full_iters} is {running_loss:.6f}")
            if run:
                run.log({'Train Loss': running_loss, 'Step': full_iters, 'Clock Time': time.perf_counter() - start_time})
            iter_start = time.perf_counter()   

    end_time = time.perf_counter()

    total_duration = end_time - start_time
    secs = total_duration
    hours = secs // 3600
    secs -= hours * 3600
    mins = secs // 60
    secs -= mins * 60
    print(f"Finished training\nTraining took {hours} hour(s) {mins} minutes and {secs:.2f} seconds")
    print(f"Final loss was {running_loss:.6f}")
    run.log({'Train Loss': running_loss, 'Step': full_iters, 'Clock Time': end_time - start_time})      

        
            


def main(model_args: ModelArgs, train_args: TrainingArgs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None, help='Type anything to enable logging to wandb')
    args = parser.parse_args()
    # Set the random seeds
    seed = 32
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU    
    assert model_args.max_seq_len >= train_args.context_length
    transformer = TransformerLM(
        model_args=model_args
    )
    print(f'Total number of parameters: {sum([param.nelement() for param in transformer.parameters()])}')

    if args.log:
        run = wandb.init(
            project = project_name,
            config = {'training': train_args.model_dump(), 'model': model_args.model_dump()},
            
        )
    else:
        run = None

    print(f"Starting training")
    train(transformer,
            train_args,
            run
            )

if __name__ == '__main__':
    assert model_args.max_batch_size >= train_args.batch_size
    main(model_args, train_args)