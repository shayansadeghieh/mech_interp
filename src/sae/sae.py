import einops
import torch 

from dataclasses import dataclass 
from jaxtyping import Float 
from torch import Tensor, nn 
from torch.nn import functional as F
from tqdm import tqdm 
from typing import Literal

from toy_model.toy_model import Model 


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
    d_in: int = 2
    d_sae: int = 5
    l1_coeff: float = 0.2
    wieght_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated"] = "standard"

class SAE(nn.Module):
    """
    W_enc: d_in x d_sae
    """
    W_enc: Float[Tensor, "d_in d_sae"]
    b_enc: Float[Tensor, "d_sae"]    
    W_dec: Float[Tensor, "d_sae d_in"] 
    b_dec: Float[Tensor, "d_in"]

    def __init__(self, model: Model, cfg: SAEConfig = SAEConfig()):
        super().__init__()

        assert cfg.d_in == model.cfg.d_hidden, "Model's hidden dim doesn't match SAE input dim"

        self.cfg = cfg
        self.model = model.requires_grad_(False)

        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((self.cfg.d_in, self.cfg.d_sae))))
        self.b_enc = nn.Parameter(torch.zeros((cfg.d_sae)))

        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((self.cfg.d_sae, self.cfg.d_in))))
        self.b_dec = nn.Parameter(torch.zeros((cfg.d_in)))  
    
    def generate_batch(self, batch_size: int):
        return einops.einsum(
            self.model.generate_batch(batch_size),
            self.model.W,
            "batch feats, d_in feats -> batch d_in"
        )

    def forward(self, h: Float[Tensor, "batch d_in"]):
        """
        Args:
            h: hidden layer activations of the model 
        """
        h_cent = h - self.b_dec

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

    def optimize(self, lr: float = 1e-3, steps: int = 10_000, batch_size: int = 1024):
        """
        Args:
            lr: Learning rate 
            steps: Number of optimization steps 
        """

        optimizer = torch.optim.Adam(list(self.parameters()), lr=lr, betas=(0.0, 0.999))
        progress_bar = tqdm(range(steps))
        frac_active_list = []

        data_log = {"steps": [], "W_enc": [], "W_dec": [], "frac_active": []}
        
        for step in progress_bar: 

            # Get a batch of hidden activations from the model 
            with torch.inference_mode():
                h = self.generate_batch(batch_size)
            
            loss_dict, loss, activations, _ = self.forward(h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate the mean sparsities over batch dim for each feature
            frac_active = (activations.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            if  (step + 1 == steps):
                progress_bar.set_postfix(
                    lr=lr,
                    frac_active=frac_active.mean().item(),
                    **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                )
                data_log["W_enc"].append(self.W_enc.detach().cpu().clone())
                data_log["W_dec"].append(self.W_dec.detach().cpu().clone())
                data_log["frac_active"].append(frac_active.detach().cpu().clone())
                data_log["steps"].append(step)
        
        return data_log
            

if __name__ == "__main__":
    toy_model = Model()
    toy_model.optimize(steps=10_000)

    sae = SAE(model=toy_model)
    data_log = sae.optimize(steps=25_000)

    print("this is data_log", data_log)

