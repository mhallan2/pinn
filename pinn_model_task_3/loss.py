# loss.py
import torch
from abc import ABC, abstractmethod

class Losses(ABC):
    @staticmethod
    @abstractmethod
    def f(x, t):
        pass

    @staticmethod
    @abstractmethod
    def g_b(x, t):
        pass

    @staticmethod
    @abstractmethod
    def g_i(x):
        pass

    @classmethod
    def pde_loss(cls, model, x_t, alpha, as_tensor=False):
        # x_t: [N,2] with columns [x, t]
        x_t.requires_grad_(True)
        u = model(x_t)
        grads = torch.autograd.grad(
            u, x_t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, x_t, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
        )[0][:, 0:1]

        residual = u_t - alpha * u_xx - cls.f(x_t[:,0:1], x_t[:,1:2])
        mse_pde = torch.mean(residual ** 2)
        return residual if as_tensor else mse_pde

    @classmethod
    def boundary_loss(cls, model, x_b, t_b, as_tensor=False):
        xt_b = torch.cat([x_b, t_b], dim=1)
        residual = model(xt_b) - cls.g_b(x_b, t_b)
        mse_bc = torch.mean(residual ** 2)
        return residual if as_tensor else mse_bc

    @classmethod
    def initial_loss(cls, model, x_i, t_i, as_tensor=False):
        xt_i = torch.cat([x_i, t_i], dim=1)
        residual = model(xt_i) - cls.g_i(x_i)
        mse_ic = torch.mean(residual ** 2)
        return residual if as_tensor else mse_ic

# Конкретная реализация
class HeatLosses(Losses):
    @staticmethod
    def f(x, t):
        # Если уравнение однородное, f=0
        return torch.zeros_like(x)

    @staticmethod
    def g_b(x, t):
        # граничные 0
        return torch.zeros_like(x)

    @staticmethod
    def g_i(x):
        # начальное условие f(x) - пример: sin(pi x / L)
        # здесь предполагаем, что L будет подставлено через замыкание/конфиг или на глобальном уровне
        return torch.sin(torch.pi * x)  # можно масштабировать по L
