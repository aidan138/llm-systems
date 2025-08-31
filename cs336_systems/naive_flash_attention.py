import torch
import math

def cdiv(x, y):
    return x // y + (x % y > 0)

class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        n, m = Q.shape[-2], K.shape[-2]
        batch_dims = Q.shape[:-2]
        output_dims = Q.shape
        d = Q.shape[-1]

        ctx.save_for_backward(Q,K,V)
        
        
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1] == d, "Embedding dimension mismatch"
        assert K.shape[-2] == V.shape[-2] == m, "Key and value must have same sequence length"
        #assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous
        Q_TILE_SIZE = 16 # TODO According to flashAttention2.0 these are tuneable
        K_TILE_SIZE = 16
        Tq = cdiv(n, Q_TILE_SIZE) # number of query tiles
        Tk = cdiv(m, K_TILE_SIZE) # number of key/value tiles
        
        O = torch.empty(output_dims, device=Q.device)
        L = torch.empty(Q.shape[:-1], device=Q.device) # Log sum across rows
        scale = math.sqrt(d)

        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        for i in range(0, Tq):
            i = i * Q_TILE_SIZE
            qi = Q[..., i:i+Q_TILE_SIZE, :]
            oi = torch.zeros_like(qi, device=Q.device)
            li = torch.zeros(qi.shape[:-1], device=Q.device).unsqueeze(-1)
            mi = torch.fill(
                torch.empty(qi.shape[:-1], device=Q.device).unsqueeze(-1), float('-inf')
            )
            #print(f"Starting shapes qi: {qi.shape} oi: {oi.shape} li: {li.shape} mi {mi.shape}")

            for j in range(Tk):
                j = j* K_TILE_SIZE
                kj, vj = K[..., j:j+K_TILE_SIZE, :], V[..., j:j+K_TILE_SIZE, :] # Each bath, tile_size, :
                scores = qi @ kj.transpose(-1, -2) / scale
                mij = torch.maximum(mi, scores.max(dim=-1, keepdim=True)[0]) # Get the running max
                pij = (scores - mij).exp() # Numerically stable exponent with running max
                li = torch.exp(mi - mij) * li + pij.sum(dim=-1, keepdim=True) # Scale down the previous running exp sum and add new exp values
                oi = torch.exp(mi - mij) * oi + pij @ vj
                mi = mij
            

            oi = (li**-1) * oi
            Li = mi + torch.log(li)
            L[..., i:i+Q_TILE_SIZE] = Li.squeeze()
            O[..., i:i+Q_TILE_SIZE, :] = oi

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    def backward():
        raise NotImplementedError