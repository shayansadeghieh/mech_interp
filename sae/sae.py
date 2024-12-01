import einops
import torch 

from dataclasses import dataclass 
from jaxtyping import Float 
from torch import Tensor, nn 
from torch.nn import functional as F
from typing import Literal


@dataclass
class SAEConfig:
    """
    d_in: The input size to the SAE (equivalent to dimension of the layer from the model)
    d_sae: The SAE's latent dimension size     
    l1_coeff: L1 regularizer used in the loss function to encourage sparsity
    weight_normalize_eps: Epsilon used when normalizing wieghts
    tied_weights: Whether encoder and decoder weights are tied 
    architecture: 
    """
    d_in: int 
    d_sae: int
    l1_coeff: float = 0.2
    wieght_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated"] = "standard"

class ModelConfig:
    """
    d_hidden: The dimension of the activation layer that you're training on 
    """
    d_hidden: int 

class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(Model, self).__init__()
        self.cfg = cfg 

class SAE(nn.Module):
    """
    W_enc: d_in x d_sae
    """
    W_enc: Float[Tensor, "d_in d_sae"]
    b_enc: Float[Tensor, "d_sae"]    
    W_dec: Float[Tensor, "d_sae d_in"] 
    b_dec: Float[Tensor, "d_in"]

    def __init__(self, cfg: SAEConfig, model: Model):
        super().__init__()

        assert cfg.d_in == model.cfg.d_hidden, "Model's hidden dim doesn't match SAE input dim"

        self.cfg = cfg
        self.model = model.requires_grad_(False)

        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((self.cfg.d_in, self.cfg.d_sae))))
        self.b_enc = nn.Parameter(torch.zeros((cfg.d_sae)))

        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((self.cfg.d_sae, self.cfg.d_in))))
        self.b_dec = nn.Parameter(torch.zeros((cfg.d_in)))  

    def forward(self, h: Float[Tensor, "batch d_in"]):
        """
        Args:
            h: hidden layer activations of the model 
        """
        h_cent = h - self.b_enc 

        activations = einops.einsum(h_cent, self.W_enc, "batch d_in, d_in d_sae -> batch d_sae")
        activations = F.relu(activations + self.b_enc)

        # Compute reconstructed input 
        h_reconstructed = einops.einsum(activations, self.W_dec, "batch d_sae, d_sae d_in -> batch d_in") + self.b_dec 

        # Compute loss terms 
        loss_reconstruction = (h_reconstructed - h).pow(2).mean(-1)
        loss_sparsity = activations.abs().sum(-1)
        loss_dict = {
            "loss_reconstruction": loss_reconstruction,
            "loss_sparsity": loss_sparsity
        }
        loss = (loss_reconstruction + self.cfg.l1_coeff * loss_sparsity).mean(0).sum()

        return loss_dict, loss, activations, h_reconstructed

        


