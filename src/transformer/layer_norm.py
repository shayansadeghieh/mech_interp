import einops
import torch

from jaxtyping import Float
from torch import Tensor, nn

class LayerNorm(nn.Module):
    W: Float[Tensor, "d_model"]
    b: Float[Tensor, "d_model"]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg         
        self.W = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.ones(cfg.d_model))
    
    def forward(self, residual: Float[Tensor, "batch position d_model"]):
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        scale = (einops.reduce(residual.power(2), "batch position d_model -> batch position 1", "mean") + self.cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.W + self.b 
        return normalized 
