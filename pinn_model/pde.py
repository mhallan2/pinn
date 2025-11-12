import torch


def u_exact(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def f_func(x, y):
    return -2 * (torch.pi**2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def g_func(x, y):
    return torch.zeros_like(x)
