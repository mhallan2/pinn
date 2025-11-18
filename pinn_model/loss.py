from abc import ABC, abstractmethod
import torch
from pde import f_func, g_func


class Losses(ABC):
    """Абстрактный базовый класс для уравнений в частных производных."""

    @staticmethod
    @abstractmethod
    def f(x, y):
        """Правая часть уравнения (функция f)."""
        pass

    @staticmethod
    @abstractmethod
    def g(x, y):
        """Граничное условие (функция g)."""
        pass

    @classmethod
    def pde_loss(cls, model, xy, as_tensor=False):
        """Вычисление невязки PDE."""
        xy.requires_grad_(True)
        u = model(xy)
        grads = torch.autograd.grad(
            u, xy, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_x, u_y = grads[:, 0:1], grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x,
            xy,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y,
            xy,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        residual = u_xx + u_yy - cls.f(xy[:, 0:1], xy[:, 1:2])
        mse_pde = torch.mean(residual ** 2)
        return residual if as_tensor else mse_pde

    @classmethod
    def boundary_loss(cls, model, x_b, y_b, as_tensor=False):
        """Вычисление невязки граничных условий."""
        xy_b = torch.cat([x_b, y_b], dim=1)
        residual = model(xy_b) - cls.g(x_b, y_b)
        mse_boundary = torch.mean(residual ** 2)
        return residual if as_tensor else mse_boundary

class PoissonLosses(Losses):
    """Потери на двумерноем уравнении Пуассона."""

    @staticmethod
    def f(x, y):
        return f_func(x, y)

    @staticmethod
    def g(x, y):
        return g_func(x, y)
