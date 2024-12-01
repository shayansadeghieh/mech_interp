import einops
import torch

from dataclasses import dataclass 
from jaxtyping import Float 
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

@dataclass
class ModelConfig:
    """
    d_hidden: The dimension of the activation layer that you're training on 
    """
    features: int = 5
    d_hidden: int = 2    


class Model(nn.Module):
    W: Float[Tensor, "d_hidden features"]
    b_final: Float[Tensor, "features"]

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg 

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.cfg.d_hidden, self.cfg.features))))

        self.b_final = nn.Parameter(torch.zeros((self.cfg.features)))
    
    def forward(self, features: Float[Tensor, "... features"]):
        h = einops.einsum(features, self.W.T, "... features, features hidden -> ... hidden")
        output = einops.einsum(h, self.W, "... hidden, hidden features -> ... features")
        return F.relu(output + self.b_final)

    def generate_batch(self, batch_size: int):
        batch_shape = (batch_size, self.cfg.features)
        feat_mag = torch.rand(batch_shape, device = self.W.device)  
        return feat_mag  
    
    def calculate_loss(self, output, batch):        
        loss = (batch - output).pow(2).mean()
        return loss 
        
    def optimize(self, batch_size: int = 1024, steps: int = 10_000, lr: float = 1e-3):
        
        optimizer = torch.optim.Adam(list(self.parameters()), lr = lr)
        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            # Reset the gradient of all optimized torch tensors
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)            
            output = self(batch)
            loss = self.calculate_loss(output, batch)
            loss.backward()
            optimizer.step()

            if (step + 1 == steps):
                progress_bar.set_postfix(loss = loss.item(), lr = lr)


if __name__ == "__main__":
    model = Model()
    model.optimize(steps=10000)
