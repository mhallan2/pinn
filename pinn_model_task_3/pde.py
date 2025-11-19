import torch


def u_exact(x, t, L=1.0, alpha=1.0):
    return torch.exp(-alpha * (torch.pi / L)**2 * t) * torch.sin(torch.pi * x / L)

def f_func(x, y):
    return torch.zeros_like(x)

def g_func(x, y):
    return torch.zeros_like(x)
