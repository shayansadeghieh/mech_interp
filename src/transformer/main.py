import os
# We set TOKENIZERS_PARALLELISM to false to silence a warning from HuggingFace 
# I also don't think we need parallelism right now for our tokenizer? 
# See https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np 
import plotly_express as px 
import torch

# Running locally on my mac 
if torch.backends.mps.is_available():
    torch.set_default_device("mps")

from transformer.transformer_block import TransformerBlock
from transformer.config import Config
from transformer.embed import Embed, PosEmbed, Unembed
from transformer.layer_norm import LayerNorm
from transformer.attention import Attention

from datetime import datetime 
from jaxtyping import Float
from pathlib import Path
from torch import nn
from torch import Tensor
from transformers import AutoTokenizer
from tqdm import tqdm 

class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens: Float[Tensor, "batch position"]):        
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed        
        for block in self.blocks:
            residual = block(residual)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        return logits

def lm_cross_entropy_loss(logits, tokens):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

if __name__ == "__main__":
    cfg = Config()    
    model = DemoTransformer(cfg=cfg)

    batch_size = 8
    num_epochs = 1
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    
    num_repeats = 1000000
    dataset = ["The cat in the hat", "The dog in the house"] * num_repeats
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer(dataset, return_tensors='pt')

    tensor_dataset = torch.utils.data.TensorDataset(tokens['input_ids'])
    data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size)
    
    losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for c, batch in tqdm(enumerate(data_loader)):
            tokens = batch[0]                
            logits = model(tokens)
            loss = lm_cross_entropy_loss(logits, tokens)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if c % log_every == 0:
                print(f"Step: {c}, Loss: {loss.item():.4f}")
            if c > max_steps:
                break 
    
    fig = px.line(y=losses, x=np.arange(len(losses))*(cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training Curve")
    fig.show()
    

    # Save the model weights with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join('trained_models', f'model_weights_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
