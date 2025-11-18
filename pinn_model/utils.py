import torch
import random
import numpy as np


def set_seed(seed=69):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_errors(model, device, u_exact, Nx=100, Ny=100, return_fields=False):
    """Вычисление RMSE и относительной L2 ошибки."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    xy = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        U_pred = model(xy).cpu().numpy().reshape(Nx, Ny)
        U_true = (
            u_exact(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
            )
            .numpy()
            .reshape(Nx, Ny)
        )

    error = U_pred - U_true
    mse = np.mean(error**2)
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(U_true)
    if return_fields:
        return mse, rel_l2, X, Y, U_pred, U_true
    else:
        return mse, rel_l2
