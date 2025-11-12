import torch


class Losses:
    @staticmethod
    def pde_loss(model, xy, f_func, as_tensor=False):
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

        residual = u_xx + u_yy - f_func(xy[:, 0:1], xy[:, 1:2])
        rmse_pde = torch.mean(residual ** 2)
        return residual if as_tensor else rmse_pde 

    @staticmethod
    def boundary_loss(model, x_b, y_b, g_func, as_tensor=False):
        xy_b = torch.cat([x_b, y_b], dim=1)
        residual = model(xy_b) - g_func(x_b, y_b)
        rmse_boundary = torch.mean(residual ** 2)
        return residual if as_tensor else rmse_boundary
