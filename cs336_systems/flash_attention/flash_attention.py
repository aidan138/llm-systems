import torch
import math
import triton
import triton.language as tl

def cdiv(x, y):
    return x // y + (x % y > 0)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, M_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    query_tile_index = tl.program_id(0)

    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(M_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1,0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(M_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1,0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_running = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_running = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)
    q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')

    for i in tl.range(tl.cdiv(M_KEYS, K_TILE_SIZE)):
        
        k, v = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero'), tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        
        scores = tl.dot(q, tl.trans(k, (1,0))) * scale
        if is_causal:
            n = Q_TILE_SIZE * query_tile_index
            m = K_TILE_SIZE * i
            q_range, k_range = tl.arange(0, Q_TILE_SIZE), tl.arange(0, K_TILE_SIZE)
            q_range, k_range = q_range + n, k_range + m

            mask = tl.where((q_range[:, None] < k_range[None, :]), -1e6, 0)
            scores += mask

        m_new  = tl.maximum(m_running, tl.max(scores, axis=-1))
        
        rescale_factor = tl.exp(m_running - m_new)
        p = tl.exp(scores - m_new[:, None])
    
        l_running = l_running * rescale_factor + p.sum(axis=-1)
        output *= rescale_factor[:, None]
        output = tl.dot(p.to(v.dtype), v, acc=output)
        
        m_running  = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    L = m_running + tl.log(l_running)
    output = output / l_running[:, None]
    tl.store(L_block_ptr, L.to(L_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0,1))

@torch.compile()
def flash_backward(Q, K, V, O, grad_O, L, is_causal):
    scale = 1 / math.sqrt(Q.shape[-1])
    S = Q @ K.transpose(-1,-2) * scale
    if is_causal:
        mask = torch.full(S.shape[-2:], float('-inf'), device=S.device).triu(diagonal=1)
        S +=  mask
    P = torch.exp(S - L.unsqueeze(-1))
    dV = P.transpose(-1,-2) @ grad_O
    dP = grad_O @ V.transpose(-1,-2)
    D = torch.sum(O * grad_O, axis=-1, keepdim=True)
    dS = P * (dP - D)
    dQ = dS @ K * scale
    dK = dS.transpose(-1,-2) @ Q * scale
    return dQ, dK, dV


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        n, m = Q.shape[-2], K.shape[-2]
        D = Q.shape[-1]

        assert Q.shape[-1] == K.shape[-1] == V.shape[-1] == D, "Embedding dimension mismatch"
        assert K.shape[-2] == V.shape[-2] == m, "Key and value must have same sequence length"
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "All tensors must be contiguous"

        ctx.Q_TILE_SIZE = 16 # TODO According to flashAttention2 these are tuneable
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal
        Q_shape, KV_shape = Q.shape, K.shape
        ctx.orig_shapes = (Q_shape, KV_shape)
        
        batch_dim = Q.size(0)
        Q, K, V = Q.view(-1, n, D), K.view(-1, m, D), V.view(-1,m, D)

        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty(Q.shape[:-1], device=Q.device, requires_grad=False) # Log sum across rows
        scale = 1/math.sqrt(D)

        flash_fwd_kernel[(cdiv(n, ctx.Q_TILE_SIZE), batch_dim)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n, m,
            scale,
            D = D,
            Q_TILE_SIZE = ctx.Q_TILE_SIZE,
            K_TILE_SIZE = ctx.K_TILE_SIZE,
            is_causal = is_causal,
        )
        
        ctx.save_for_backward(L, Q, K, V, O)
        return O.reshape(Q_shape)

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dO = dO.reshape(O.shape)
        Q_shape, KV_shape = ctx.orig_shapes
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ.reshape(Q_shape), dK.reshape(KV_shape), dV.reshape(KV_shape), None
    