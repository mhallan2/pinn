import torch


def generate_domain_points(N_f, device="cpu"):
    """Генерирует случайные точки в расчетной области [0,1] x [0,1]."""
    x = torch.rand(N_f, 1, device=device)
    y = torch.rand(N_f, 1, device=device)
    return x, y


def generate_boundary_points(N_b, device="cpu"):
    """Генерирует точки на границах квадрата [0,1] x [0,1]."""
    t = torch.linspace(0, 1, N_b, device=device).view(-1, 1)
    x_b = torch.cat([torch.zeros_like(t), torch.ones_like(t), t, t], dim=0)
    y_b = torch.cat([t, t, torch.zeros_like(t), torch.ones_like(t)], dim=0)
    return x_b, y_b
