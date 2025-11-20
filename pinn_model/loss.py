from abc import ABC, abstractmethod
import torch
from pde import f, g


class Losses(ABC):
    """Универсальный базовый класс для PDE вида N[u] = 0."""

    @staticmethod
    @abstractmethod
    def pde_residual(model, xy):
        pass

    @staticmethod
    @abstractmethod
    def boundary_residual(model, x_b, y_b):
        pass

    @classmethod
    def pde_loss(cls, model, xy, as_tensor=False):
        residual = cls.pde_residual(model, xy)
        mse = torch.mean(residual**2)
        return residual if as_tensor else mse

    @classmethod
    def boundary_loss(cls, model, x_b, y_b, as_tensor=False):
        residual = cls.boundary_residual(model, x_b, y_b)
        mse = torch.mean(residual**2)
        return residual if as_tensor else mse


class PoissonLosses(Losses):

    @staticmethod
    def pde_residual(model, xy):
        xy = xy.clone().requires_grad_(True)

        x = xy[:, 0:1]
        y = xy[:, 1:2]

        u = model(xy)

        grads = torch.autograd.grad(
            u, xy,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, xy,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]

        u_yy = torch.autograd.grad(
            u_y, xy,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:, 1:2]

        return u_xx + u_yy - f(x, y)

    @staticmethod
    def boundary_residual(model, x_b, y_b):
        xy_b = torch.cat([x_b, y_b], dim=1)
        return model(xy_b) - g(x_b, y_b)
