import einops
import math 
import torch

from jaxtyping import Float
from torch import Tensor, nn

class MLP(nn.Module):
    W_in: Float[Tensor, "d_model d_mlp"]
    b_in: Float[Tensor, "d_mlp"]

    W_out: Float[Tensor, "d_mlp d_model"]
    b_out: Float[Tensor, "d_model"]

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg 
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)

        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
    
    def forward(self, normalized_resid_mid: Float[Tensor, "batch position d_model"]):
        pre = einops.einsum(normalized_resid_mid, self.W_in, "batch position d_model, d_model d_mlp -> batch position d_mlp") + self.b_in 
        post = torch.nn.functional.gelu(pre)        
        mlp_out = einops.einsum(post, self.W_out, "batch position d_mlp, d_mlp d_model -> batch position d_model") + self.b_out
        return mlp_out


