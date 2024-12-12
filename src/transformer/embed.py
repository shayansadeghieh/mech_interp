import einops
import torch

from jaxtyping import Float
from torch import Tensor, nn

class Embed(nn.Module):
    W_E: Float[Tensor, "d_vocab d_model"]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
    
    def forward(self, tokens: Float[Tensor, "batch position"]):
        # embed should be shape [batch, position, d_model]
        embed = self.W_E[tokens, :]        
        return embed 


class PosEmbed(nn.Module):
    W_pos: Float[Tensor, "n_ctx d_model"]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))        
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Float[Tensor, "batch position"]):
        pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        return pos_embed 
    
class Unembed(nn.Module):
    W_U: Float[Tensor, "d_model d_vocab"]
    b_U: Float[Tensor, "d_vocab"]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab)), requires_grad=False)
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
    
    def forward(self, normalized_resid_final: Float[Tensor, "batch position d_model"]):        
        logits = einops.einsum(normalized_resid_final, self.W_U, "batch position d_model, d_model d_vocab -> batch position d_vocab") + self.b_U
        return logits 
