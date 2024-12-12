import einops
import math 
import torch

from jaxtyping import Float
from torch import Tensor, nn

class Attention(nn.Module):
    W_Q: Float[Tensor, "n_heads d_model d_head"]
    b_Q: Float[Tensor, "n_heads d_head"]

    W_K: Float[Tensor, "n_heads d_model d_head"]
    b_K: Float[Tensor, "n_heads d_head"]

    W_V: Float[Tensor, "n_heads d_model d_head"]
    b_V: Float[Tensor, "n_heads d_head"]

    W_O: Float[Tensor, "n_heads d_model d_head"]
    b_O: Float[Tensor, "n_heads d_head"]

    def __init__(self, cfg):
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_head, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)

        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_K = nn.Parameter(torch.zeros((cfg.n_head, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)

        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_V = nn.Parameter(torch.zeros((cfg.n_head, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_O = nn.Parameter(torch.zeros((cfg.n_head, cfg.d_head)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32))

    
    def forward(self, normalized_resid_pre: Float[Tensor, "batch position d_model"]):
        q = einops.einsum(normalized_resid_pre, self.W_Q, "batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head") + self.b_Q
        k = einops.einsum(normalized_resid_pre, self.W_K, "batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head") + self.b_K

        attn_scores = einops.einsum(q, k, "batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos")
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]
        
        v = einops.einsum(normalized_resid_pre, self.W_V, "batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head") + self.b_V
        z = einops.einsum(pattern, v, "batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head")

        attn_out = einops.einsum(z, self.W_O, "batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model") + self.b_O

        return attn_out
    
    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]):
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores





