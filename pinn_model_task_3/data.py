import torch

class DataGenerator:
    @staticmethod
    def domain_points(N_f, L=1.0, T=1.0, device="cuda"):
        x = torch.rand(N_f, 1, device=device) * L
        t = torch.rand(N_f, 1, device=device) * T
        return x, t

    @staticmethod
    def boundary_points(N_b, L=1.0, T=1.0, device="cuda"):
        t = torch.rand(N_b, 1, device=device) * T
        x_left = torch.zeros_like(t)
        x_right = torch.ones_like(t) * L
        x_b = torch.cat([x_left, x_right], dim=0)
        t_b = torch.cat([t, t], dim=0)
        return x_b, t_b

    @staticmethod
    def initial_points(N_i, L=1.0, device="cuda"):
        x = torch.rand(N_i, 1, device=device) * L
        t0 = torch.zeros_like(x)
        return x, t0
