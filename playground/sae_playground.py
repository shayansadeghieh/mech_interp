import torch 

from sae_lens import HookedSAETransformer, SAE
from transformer_lens import utils 


def main():
    device = str(torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"The device is {device}")

    gpt2: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    gpt2_sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    device=device)

    prompt = "The cat in the"
    answer = " hat"    

    utils.test_prompt(prompt, answer, gpt2)

    with gpt2.saes(saes=[gpt2_sae]):
        utils.test_prompt(prompt, answer, gpt2)
    
    _, cache = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])



if __name__ == "__main__":
    main()
