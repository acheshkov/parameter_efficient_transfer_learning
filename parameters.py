import torch

def disable_parameters(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

def print_params_stat(model: torch.nn.Module) -> None:
    c_rg = 0
    c_nrg = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            c_rg += parameter.numel()
        else:
            c_nrg += parameter.numel()

    print("Parameters require grad:", c_rg)
    print("Prameters not require grad:", c_nrg)
