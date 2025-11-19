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


def compute_errors_time(model, device, u_exact, L=1.0, T=1.0, Nx=100, Nt=100, return_fields=False):
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    X, Tt = np.meshgrid(x, t)

    x_t = torch.tensor(
        np.stack([X.ravel(), Tt.ravel()], axis=1),
        dtype=torch.float32
    ).to(device)

    model.eval()
    with torch.no_grad():
        U_pred = model(x_t).cpu().numpy().reshape(Nt, Nx)
        U_true = u_exact(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Tt, dtype=torch.float32)
        ).numpy()

    error = U_pred - U_true
    mse = np.mean(error**2)
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(U_true)

    if return_fields:
        return mse, rel_l2, X, Tt, U_pred, U_true
    return mse, rel_l2
