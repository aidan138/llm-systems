import torch
from torch import nn, Tensor
import math
from cs336_basics.nn.utils import softmax
from cs336_basics.args import ModelArgs

class Linear(nn.Module):
    """
    Basic dense layer where every input feature has a corresponding weight within its output.
    """
    def __init__(self, in_features: int, output_features: int, bias: bool = False, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        dtype = torch.float32 if dtype is None else dtype
        weight_stdv = math.sqrt(2/ (in_features + output_features))
        weights = nn.init.trunc_normal_(
            torch.zeros((output_features, in_features)),
            0, # Mean
            weight_stdv,
            -3 * weight_stdv, # Lower bound
            3 * weight_stdv # Uppper bound
        ).type(dtype)
        self.weight = nn.Parameter(weights).to(device)
        self.b = nn.Parameter(torch.zeros(output_features)).to(device) if bias else None
        self.out = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        return out if self.b is None else out + self.b
    
    def get_num_parameters(self):
        return sum(param.numel() for param in self.parameters())

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        dtype = torch.float32 if dtype is None else dtype
        embeddings = nn.init.trunc_normal_(
            torch.zeros((num_embeddings, embedding_dim)),
            0, # Mean
            1, # std
            -3, # Lower bound
            3 # Upper bound
        ).type(dtype)
        self.weight = nn.Parameter(embeddings).to(device)

    def forward(self, token_ids: Tensor):
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, epsilon: float=1e-5, device = None, dtype= None):
        super().__init__()
        dtype = torch.float32 if dtype is None else dtype
        self.gain = nn.Parameter(torch.ones(d_model))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        assert D == self.gain.shape[0]
        input_dtype = x.dtype
        x = x.to(torch.float32) # Up cast to prevent overflow in lower dtypes
        square_sum_x = (x.square() + self.epsilon).sum(-1, keepdim=True) # B, N, 1
        rms_x = (square_sum_x / D).sqrt() # B, N, 1 
        # rms_x will be broadcasted across the feature dim and gain will be broadcasted across batch and sequence dims
        rms_norm_x = (x / rms_x) * self.gain # Element wise multiply and division
        return rms_norm_x.to(input_dtype)
    

def silu(x: Tensor):
    return x * torch.sigmoid(x)

class SwiGLU_Feedforward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        # Ensure the chosen dff is a multiple of 64 for CUDA
        d_ff = (d_ff // 64 + 1) * 64

        # The dimensionality of the weights
        dim_dict = {'W1': (d_model, d_ff),'W2': (d_ff, d_model), 'W3': (d_model, d_ff)}
        self.linears = nn.ModuleDict({
            layer_name: Linear(*dims, device=device, dtype=dtype) for layer_name, dims in dim_dict.items()
        })

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3
        xpreact = self.linears['W1'](x) # B, N, FF
        x_gate = self.linears['W3'](x)# B, N, FF
        # Apply elementwise multiplier with sigmoid activation
        # Note: sigmoid between 0 and 1
        xact = silu(xpreact)
        xact = xact * x_gate # Apply the linear transform gate
        return self.linears['W2'](xact)


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        assert d_k % 2 == 0, "Input dim must be divisible by 2"
        sin, cos = self._compute_theta_mat(theta, d_k, max_seq_len, device)
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)

    def _compute_theta_mat(self, theta: float, d_k: int, max_seq_len: int, device: torch.device):
        angle_vals = [
            i/(theta**(2*k/d_k)) for i in range(max_seq_len) for k in range(0, d_k//2) # 
        ]
        #positions = torch.arange(max_seq_len, device=device)
        angle_vals = torch.Tensor(angle_vals).view((max_seq_len, d_k//2, -1)).to(device) # N, D/2, 1
        sin, cos = angle_vals.sin(), angle_vals.cos() # N, D/2, 1 for both
        return sin, cos

    def forward(self, x: Tensor, token_positions: torch.Tensor = None):
        *batch_dim, N, D = x.shape # Handle arbitrary batch dims
        x = x.view((*batch_dim, N, D//2, 2, 1)) # ..., N, D/2, 2, 1
        token_positions = token_positions if token_positions is not None else torch.arange(N, device=x.device)
        sin, cos = self.sin[token_positions], self.cos[token_positions]

        R_mats = torch.stack( # seq_len, d_features/2,  -> seq_len, d_model/2, 2, 2
            (torch.concat([cos, -sin], dim=-1),
             torch.concat([sin, cos], dim=-1)), dim=-2
        ) 

        for _ in batch_dim:
            R_mats = R_mats.unsqueeze(0) # 1*B, N, D/2, 2,2

        return (R_mats @ x).squeeze(-1).view(*batch_dim, N, D)

def sdp_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None):
    batch, *rest, seq_len, feat_dim = Q.shape # 
    # Note works for a two different sequence lengths
    # Therefore you can decide to determine attention attending from two diff sequences
    # For a autoregressive SA it will always be same source though
    sim_matrix = Q @ K.transpose(-2,-1)/math.sqrt(feat_dim) # (B, *rest, N, D) @ (B, *rest, D, M) -> (B, *rest, N, M)
    if mask is not None:
        sim_matrix = torch.where(condition=mask, input=sim_matrix, other=float('-inf'))
    prob_matrix = softmax(sim_matrix, dim=-1)
    
    return  (prob_matrix @ V).view(batch, *rest, -1, feat_dim) # Returns B, ..., n, feat_dim

class Multiheaded_Self_Attention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_length: int = 0, positional_embeddings: RoPE = None):
        assert d_model % num_heads == 0, 'Model embeddings must be divisible by number of heads'
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.RoPE = positional_embeddings

    def forward(self, x: Tensor, token_positions: Tensor | None = None):
        batch_size, seq_len, d_model = x.shape
        # We get an output from the matmul of batch_size, seq_len, num_heads * head_dim
        # We then view for batch_size, seq_len, num_heads, head_dim
        # Finally permute the head dim to the front so that you are processing all heads in parallel as if separate modules
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, -1).permute((2, 0, 1, 3))
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, -1).permute((2, 0, 1, 3))
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, -1).permute((2, 0, 1, 3))
        # Apply positional embeddings if applicable
        if self.RoPE is not None:
            query = self.RoPE(query, token_positions)
            key = self.RoPE(key, token_positions)
        mask_shape = (*query.shape[:-1], key.shape[-2]) # ..., N, M
        mask = torch.full(mask_shape, True, device=x.device).tril()
        attended = sdp_attention(query, key, value, mask = mask) # Apply attention -> num_heads, batch, seq_len, head_dim
        attended = attended.permute(1, 2, 0, 3).reshape(batch_size, seq_len, -1) # undo the permutation
        return self.output_proj(attended).view(batch_size, seq_len, d_model)



class Parallel_Multiheaded_Self_Attention(nn.Module):
    
    def __init__(self, model_args: ModelArgs, use_cache = False):
        super().__init__()
        self.head_dim = model_args.d_model // model_args.num_heads
        self.num_heads = model_args.num_heads
        self.kqv_proj = nn.Parameter(torch.stack([Linear(self.head_dim*self.num_heads, model_args.d_model).weight.data for _ in range(3)], dim=0))
        self.output_proj = Linear(model_args.d_model, model_args.d_model)
        self.num_kv_heads = self.num_heads if model_args.num_kv_heads is None else model_args.num_kv_heads
        self.max_batch_size = model_args.max_batch_size
        self.max_seq_len = model_args.max_seq_len
        self.use_cache = use_cache
        
        # init caches to None to save mem while training
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    # KV cache init
    def _init_cache(self, max_bsz, device, dtype):
        if self.k_cache is None or self.k_cache.size(1) < max_bsz:
            self.k_cache = torch.zeros(
                (self.num_kv_heads, max_bsz, self.max_seq_len, self.head_dim), device=device, dtype=dtype
            )
            self.v_cache = torch.zeros_like(self.k_cache)
            

    def forward(self, x: Tensor, positional_embeddings: RoPE = None, start_pos:int = 0):
        batch_size, seq_len, d_model = x.shape
        # Combined K,Q,V into a single KQV (3, num_head * head_dim, d_model) matrix reducing everything to a single matrix multiply
        # by leveraging broadcasting to perform serial actions
        # 1st permute performs the transpose of all the matrices
        # Unsqueezing the input sequence dimension(batch, 1, seq_length, d_model) allowed for the seq_length, d_model matrices
        # to be broadcasted to the 3 KQV dims.
        # Unsqueezing the KQV (1, 3, num_head * head_dim, d_model) allowed for the KQV to be broadcasted across all batches
        transformed = (x.unsqueeze(-3) @ self.kqv_proj.permute((0,2,1)).unsqueeze(0)).permute(1,0,2,3)
        
        # We get an output from the matmul of batch_size, seq_len, num_heads * head_dim
        # We then view for batch_size, seq_len, num_heads, head_dim
        # Finally permute the head dim to the front so that you are processing all heads in parallel as if separate modules
        query = transformed[0].reshape(batch_size, seq_len, self.num_heads, -1).permute((2,0,1,3))
        key = transformed[1].reshape(batch_size, seq_len, self.num_kv_heads, -1).permute((2,0,1,3))
        value = transformed[2].reshape(batch_size, seq_len, self.num_kv_heads, -1).permute((2,0,1,3))

        last_pos = start_pos + seq_len
        token_positions = torch.arange(start_pos, last_pos)
        # Apply positional embeddings if applicable
        if positional_embeddings is not None:
            query = positional_embeddings(query, token_positions)
            key = positional_embeddings(key, token_positions)


        if not self.training and self.use_cache:
            self._init_cache(batch_size, key.device, key.dtype)
            # Save to kv_cache
            self.k_cache[:, :batch_size, start_pos:last_pos] = key
            self.v_cache[:, :batch_size, start_pos:last_pos] = value

            # Extract all the cached values
            value = self.v_cache[:, :batch_size, :last_pos]
            key = self.k_cache[:, :batch_size, :last_pos]

        mask = torch.tril(torch.ones(seq_len, key.shape[-2], device=x.device)).bool()
        attended = sdp_attention(query, key, value, mask = mask) # Apply attention -> num_heads, batch, seq_len, head_dim
        attended = attended.permute(1, 2, 0, 3).reshape(batch_size, seq_len, -1) # undo the permutation -> batch, seq_len, d_model
        attended = self.output_proj(attended).view(batch_size, seq_len, d_model)
        return attended

class Transformer_Block(nn.Module):

    def __init__(self, model_args):
        super().__init__()
        d_ff = model_args.d_ff if model_args.d_ff is not None else int(4 * 2/3 * model_args.d_model) # Based on convention for GLUs
        self.attn = Parallel_Multiheaded_Self_Attention(model_args)
        self.ln1 = RMSNorm(model_args.d_model)

        self.ffn = SwiGLU_Feedforward(model_args.d_model, d_ff)
        self.ln2 = RMSNorm(model_args.d_model)
    
    def forward(self, x: Tensor, start_pos: int = 0, positional_embeddings: RoPE | None = None):
        attended_x = self.attn(self.ln1(x), positional_embeddings, start_pos)
        r_x = x + attended_x
        ffn_x = self.ffn(self.ln2(r_x))
        return ffn_x + r_x
        

class TransformerLM(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        d_model = model_args.d_model if model_args.d_model is not None else 128 * model_args.num_layers
        num_heads = model_args.num_heads
        head_dim = model_args.head_dim if model_args.head_dim is not None else d_model // num_heads
        vocab_size = model_args.vocab_size
        self.embeddings = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([Transformer_Block(model_args) for _ in range(model_args.num_layers)])
        self.rope = RoPE(model_args.rope_theta, head_dim, model_args.max_seq_len)
        self.ln = RMSNorm(d_model)
        self.output_layer = Linear(d_model, vocab_size)
        self.current_pos = 0 # For kv caching

    def forward(self, x: Tensor):
        seq_len = x.shape[1]
        x = self.embeddings(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, positional_embeddings=self.rope, start_pos=self.current_pos)
        x = self.ln(x)
        x = self.output_layer(x)
        if not self.training:
            self.current_pos += seq_len
        return x

    def activate_cache(self):
        for block in self.transformer_blocks:
            block.attn.use_cache = True
