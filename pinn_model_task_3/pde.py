import torch
from config import Config


l = Config.L
alpha = Config.alpha

def u_exact(x, t):
    return torch.exp(-alpha * (torch.pi / l)**2 * t) * torch.sin(torch.pi * x / l)

def g(x, y):
    return torch.zeros_like(x)

def f(x):
    return torch.sin(torch.pi * x / l)
