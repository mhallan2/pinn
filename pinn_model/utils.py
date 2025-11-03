import torch
import numpy as np


def set_seed(seed=69):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def u_exact(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def f_func(x, y):
    return -2 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def g_func(x, y):
    return torch.zeros_like(x)

def compute_errors(model, device, u_exact, N=1000, return_fields=False):
    """Вычисление RMSE и относительной L2 ошибки."""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    xy = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        U_pred = model(xy).cpu().numpy().reshape(N, N)
        U_true = u_exact(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        ).numpy().reshape(N, N)

    error = U_pred - U_true
    rmse = np.sqrt(np.mean(error**2))
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(U_true)
    if return_fields:
        return rmse, rel_l2, X, Y, U_pred, U_true
    else:
        return rmse, rel_l2


# def plot_solution_UNUSED(model, device="cpu", N=60):
#     """Графики в двумерном и трехмерном пространствах для аналитического и численного решений."""
#     x = np.linspace(0, 1, N)
#     y = np.linspace(0, 1, N)
#     X, Y = np.meshgrid(x, y)
#     xy = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32).to(device)

#     with torch.no_grad():
#         U_pred = model(xy).cpu().numpy().reshape(N, N)
#         U_true = (
#             (torch.sin(torch.pi * xy[:, 0]) * torch.sin(torch.pi * xy[:, 1]))
#             .cpu()
#             .numpy()
#             .reshape(N, N)
#         )

#     # Канвас на 4 графика: два 2D и два 3D
#     fig = plt.figure(figsize=(16, 10))

#     # 2D графики
#     ax1 = fig.add_subplot(2, 2, 1)
#     im1 = ax1.contourf(X, Y, U_pred, 100, cmap="viridis")
#     fig.colorbar(im1, ax=ax1)
#     ax1.set_title("PINN Prediction (2D)")
#     ax1.set_xlabel("x")
#     ax1.set_ylabel("y")

#     ax2 = fig.add_subplot(2, 2, 2)
#     im2 = ax2.contourf(X, Y, U_true, 100, cmap="viridis")
#     fig.colorbar(im2, ax=ax2)
#     ax2.set_title("Exact Solution (2D)")
#     ax2.set_xlabel("x")
#     ax2.set_ylabel("y")

#     # 3D поверхности
#     ax3 = fig.add_subplot(2, 2, 3, projection="3d")
#     surf1 = ax3.plot_surface(
#         X, Y, U_pred, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True
#     )
#     ax3.set_title("PINN Prediction (3D)")
#     ax3.set_xlabel("x")
#     ax3.set_ylabel("y")
#     ax3.set_zlabel("u(x,y)")
#     fig.colorbar(surf1, ax=ax3, shrink=0.5, aspect=5, location="left")

#     ax4 = fig.add_subplot(2, 2, 4, projection="3d")
#     surf2 = ax4.plot_surface(
#         X, Y, U_true, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True
#     )
#     ax4.set_title("Exact Solution (3D)")
#     ax4.set_xlabel("x")
#     ax4.set_ylabel("y")
#     ax4.set_zlabel("u(x,y)")
#     fig.colorbar(surf2, ax=ax4, shrink=0.5, aspect=5, location="left")

#     plt.tight_layout()
#     plt.show()

#     # График ошибки
#     plot_error(X, Y, U_pred, U_true)


# def plot_error(X, Y, U_pred, U_true):
#     """Функция для отображения ошибки между предсказанием и точным решением."""
#     error = np.abs(U_pred - U_true)

#     fig = plt.figure(figsize=(15, 4))

#     # 2D ошибка
#     ax1 = fig.add_subplot(1, 3, 1)
#     im1 = ax1.contourf(X, Y, error, 100, cmap="hot")
#     fig.colorbar(im1, ax=ax1)
#     ax1.set_title("Absolute Error (2D)")
#     ax1.set_xlabel("x")
#     ax1.set_ylabel("y")

#     # 3D ошибка
#     ax2 = fig.add_subplot(1, 3, 2, projection="3d")
#     surf = ax2.plot_surface(X, Y, error, cmap="hot", alpha=0.8)
#     ax2.set_title("Absolute Error (3D)")
#     ax2.set_xlabel("x")
#     ax2.set_ylabel("y")
#     ax2.set_zlabel("Error")
#     fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, location="left")

#     # Статистика ошибки
#     ax3 = fig.add_subplot(1, 3, 3)
#     ax3.axis("off")
#     stats_text = f"""Error statistics:
#     Max error: {np.max(error):.2e}
#     Mean error: {np.mean(error):.2e}
#     MSE: {np.sqrt(np.mean(error**2)):.2e}
#     Relative L2: {np.linalg.norm(error) / np.linalg.norm(U_true):.2e}"""
#     ax3.text(
#         0.1,
#         0.9,
#         stats_text,
#         fontsize=12,
#         verticalalignment="top",
#         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
#     )
#     ax3.set_title("Error Statistics")

#     plt.tight_layout()
#     plt.show()
