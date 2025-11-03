"""
Модуль визуализации решений PINN:
 - 2D-графики предсказаний, точного решения и ошибок
 - 3D-графики поверхностей для тех же данных

Предназначен для анализа качества решения PDE.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_unique_path(base_dir="results", prefix="plot", ext="png"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(base_dir, f"{prefix}_{timestamp}.{ext}")

# ==============================================================
# === 2D ВИЗУАЛИЗАЦИЯ ===
# ==============================================================

def plot_2d_solution_and_error(X, Y, U_pred, U_true, save_path=None, prefix="solution_2d"):
    """
    Строит 2D графики предсказания, точного решения и ошибки.
    """
    error = np.abs(U_pred - U_true)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Предсказание ---
    im0 = axes[0].contourf(X, Y, U_pred, 100, cmap="viridis")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("PINN Prediction (2D)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # --- Точное решение ---
    im1 = axes[1].contourf(X, Y, U_true, 100, cmap="viridis")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Exact Solution (2D)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    # --- Ошибка ---
    im2 = axes[2].contourf(X, Y, error, 100, cmap="hot")
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title("Absolute Error (2D)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    plt.tight_layout()

    if save_path is None:
        save_path = get_unique_path(base_dir="results", prefix=prefix)
    fig.savefig(save_path, dpi=300)
    print(f"Saved 2D plot to {save_path}")

    plt.show()


# ==============================================================
# === 3D ВИЗУАЛИЗАЦИЯ ===
# ==============================================================

def plot_3d_solution_and_error(X, Y, U_pred, U_true, save_path=None, prefix="solution_3d"):
    """
    Строит 3D графики предсказания, точного решения и ошибки.
    """
    error = np.abs(U_pred - U_true)

    fig = plt.figure(figsize=(18, 5))

    # --- Предсказание ---
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(X, Y, U_pred, cmap="viridis", alpha=0.7)
    fig.colorbar(surf1, ax=ax1, shrink=0.3, location="left")
    ax1.set_title("PINN Prediction (3D)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u_pred")

    # --- Точное решение ---
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(X, Y, U_true, cmap="viridis", alpha=0.7)
    fig.colorbar(surf2, ax=ax2, shrink=0.3, location="left")
    ax2.set_title("Exact Solution (3D)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u_true")

    # --- Ошибка ---
    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(X, Y, error, cmap="hot", alpha=0.9)
    fig.colorbar(surf3, ax=ax3, shrink=0.3, location="left")
    ax3.set_title("Absolute Error (3D)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("|Error|")

    plt.tight_layout()

    if save_path is None:
        save_path = get_unique_path(base_dir="results", prefix=prefix)
    fig.savefig(save_path, dpi=300)
    print(f"Saved 3D plot to {save_path}")

    plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import torch


# def plot_solution(model, u_exact, device="cpu", N=60, show_error=True):
#     x = np.linspace(0, 1, N)
#     y = np.linspace(0, 1, N)
#     X, Y = np.meshgrid(x, y)
#     xy = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32).to(device)

#     model.eval()
#     with torch.no_grad():
#         U_pred = model(xy).cpu().numpy().reshape(N, N)
#         U_true = u_exact(
#             torch.tensor(X, dtype=torch.float32),
#             torch.tensor(Y, dtype=torch.float32),
#         ).numpy().reshape(N, N)

#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#     axs[0].contourf(X, Y, U_pred, 100, cmap="viridis")
#     axs[0].set_title("PINN Prediction")
#     axs[1].contourf(X, Y, U_true, 100, cmap="viridis")
#     axs[1].set_title("Exact Solution")
#     plt.show()

#     if show_error:
#         plot_error(X, Y, U_pred, U_true)


# def plot_error(X, Y, U_pred, U_true):
#     error = np.abs(U_pred - U_true)
#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.contourf(X, Y, error, 100, cmap="hot")
#     fig.colorbar(im)
#     ax.set_title(f"Error map (mean={np.mean(error):.2e}, max={np.max(error):.2e})")
#     plt.show()
