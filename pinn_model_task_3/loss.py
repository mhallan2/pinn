from abc import ABC, abstractmethod
import torch
from pde import f, g, alpha


class Losses(ABC):

    @staticmethod
    @abstractmethod
    def pde_residual(model, xt):
        pass

    @staticmethod
    @abstractmethod
    def boundary_residual(model, x_b, t_b):
        pass

    @staticmethod
    @abstractmethod
    def initial_residual(model, x_i, t_i):
        pass

    @classmethod
    def pde_loss(cls, model, xt):
        r = cls.pde_residual(model, xt)
        return torch.mean(r**2)

    @classmethod
    def boundary_loss(cls, model, x_b, t_b):
        r = cls.boundary_residual(model, x_b, t_b)
        return torch.mean(r**2)

    @classmethod
    def initial_loss(cls, model, x_i, t_i):
        r = cls.initial_residual(model, x_i, t_i)
        return torch.mean(r**2)

class HeatLosses(Losses):

    @staticmethod
    def pde_residual(model, xt):
        xt = xt.clone().requires_grad_(True)

        u = model(xt)

        grads = torch.autograd.grad(
            u, xt,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, xt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]

        return u_t - alpha * u_xx

    @staticmethod
    def boundary_residual(model, x_b, t_b):
        xt = torch.cat([x_b, t_b], dim=1)
        return model(xt) - g(x_b, t_b)

    @staticmethod
    def initial_residual(model, x_i, t_i):
        xt = torch.cat([x_i, t_i], dim=1)
        return model(xt) - f(x_i)
