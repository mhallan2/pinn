import torch


class DataGenerator:
    @staticmethod
    def domain_points(N_f, device="cpu"):
        x = torch.rand(N_f, 1, device=device)
        y = torch.rand(N_f, 1, device=device)
        return x, y

    @staticmethod
    def boundary_points(N_b, device="cpu"):
        t = torch.linspace(0, 1, N_b, device=device).view(-1, 1)
        x_b = torch.cat([torch.zeros_like(t), torch.ones_like(t), t, t], dim=0)
        y_b = torch.cat([t, t, torch.zeros_like(t), torch.ones_like(t)], dim=0)
        return x_b, y_b
