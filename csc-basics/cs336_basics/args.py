from enum import Enum
from typing import Optional, IO

from pydantic import BaseModel, model_validator, field_serializer
import torch
from pathlib import Path

class ModelArgs(BaseModel):
    d_model: int
    vocab_size: int
    d_ff: Optional[int] = None
    rope_theta: float = 10000
    
    # Attention config
    num_layers: int
    num_heads: int = -1
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # Inference time parameters
    max_batch_size: int = 32
    max_seq_len: int = 2048 # Will be used at train as well but should be scaled down considerably

    @model_validator(mode='after')
    def validate(self) -> "ModelArgs":
        # Esssentially validating all attention heads
        self.num_kv_heads = self.num_heads if self.num_kv_heads is None else self.num_kv_heads
        assert self.num_kv_heads <= self.num_heads, f"Number of kv_heads: {self.num_kv_heads} must be less than n_heads: {self.num_heads}"
        assert self.num_heads % self.num_kv_heads == 0, f"n_heads {self.num_heads} must be divisible by {self.num_kv_heads}"
        assert self.d_model % self.num_heads == 0, f"d_model: {self.d_model} must be divisible by n_heads: {self.num_heads}"
        return self
    

class TrainingArgs(BaseModel):
    model_config = {
        'arbitrary_types_allowed': True
    }
    # Train Loop
    iterations: int
    checkpoint_freq: int
    batch_size: int
    context_length: int
    save_path: str | Path
    train_path: str | Path
    cv_path: str | Path
    load_path: Optional[str | Path] = None
    device: Optional[str] = 'cpu'
    dtype: Optional[torch.dtype] = torch.float32

    # Logging
    log_cv_iterations: int
    log_train_iterations: int
    train_loss_alpha: float

    # Optimizer
    lr_max : float
    weight_decay: Optional [float] = None
    betas: Optional[tuple] = (0.9, 0.999)

    # Learning rate scheduler
    lr_min : Optional[float] = None
    warmup_iterations: Optional[float] = None
    cos_iterations: Optional[float] = None

    # Gradient Clipping
    max_l2_norm : Optional[float] = None
    
    @field_serializer(mode='before')
    def parse_dtype(cls, v):
        if isinstance(v, str):
            getattr(torch, v)

    @model_validator(mode='after')
    def validate(self) -> "TrainingArgs":
        if self.lr_min or self.warmup_iterations or self.cos_iterations:
            assert None not in [self.lr_min, self.warmup_iterations, self.cos_iterations], f"If using annealing lr_min ({self.lr_min}), \
                  warmup_iterations ({self.warmup_iterations}), and cos_iterations ({self.cos_iterations}) must be set"

        return self