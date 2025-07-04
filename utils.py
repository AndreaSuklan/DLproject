import torch
import os

def setup_device():
    if torch.cuda.is_available():
        print("Using cuda")
        return torch.device("cuda")
    
    else :
        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)
        print("Using cpu")
        return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

