import torch


def pde_residual(model, xy, f_func):
    """
    Потери на физическом дифференциальном уравнении
    ∇²u(x,y) = f(x,y)
    """
    #xy = xy.detach().clone().requires_grad_(True)
    xy.requires_grad_(True)
    u = model(xy)

    # Первые производные
    grads = torch.autograd.grad(
        u, xy, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    # Вторые производные
    u_xx = torch.autograd.grad(
        u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True
    )[0][:, 1:2]

    f_val = f_func(xy[:, 0:1], xy[:, 1:2])
    residual = u_xx + u_yy - f_val

    return residual


def boundary_loss(model, x_b, y_b, g_func):
    """Потери на граничных условиях."""
    xy_b = torch.cat([x_b, y_b], dim=1)
    u_pred = model(xy_b)
    return torch.mean((u_pred - g_func(x_b, y_b)) ** 2)
