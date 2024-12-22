import torch 

def get_layers_dict(model) -> dict:
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param.size()            
    return params_dict 

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("No GPU found. Using CPU.")
    return device