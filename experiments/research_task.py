import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN
from data import generate_domain_points, generate_boundary_points
from utils import f_func, g_func, u_exact
from loss import pde_residual, boundary_loss


def compute_errors(model, device, N=100):
    """Вычисляет RMSE и Relative L2 ошибку"""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    xy = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32).to(
        device
    )

    with torch.no_grad():
        U_pred = model(xy).cpu().numpy().reshape(N, N)
        U_true = (
            u_exact(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
            )
            .numpy()
            .reshape(N, N)
        )

    error = U_pred - U_true
    rmse = np.sqrt(np.mean(error**2))
    relative_l2 = np.linalg.norm(error) / np.linalg.norm(U_true)

    return rmse, relative_l2


def train_with_history(
    model,
    f_func,
    g_func,
    generate_domain_points,
    generate_boundary_points,
    N_f=5000,
    N_b=1000,
    epochs=2000,
    lr=1e-4,
    lam_pde=1.0,
    lam_bc=1.0,
    device="cpu",
    verbose_every=1000,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_b, y_b = generate_boundary_points(N_b, device=device)

    # Для сохранения истории
    history = {
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "rmse": [],
        "relative_l2": [],
    }

    # Вычисляем начальную ошибку
    rmse, rel_l2 = compute_errors(model, device)
    history["rmse"].append(rmse)
    history["relative_l2"].append(rel_l2)
    x_f, y_f = generate_domain_points(N_f, device=device)
    xy_f = torch.cat([x_f, y_f], dim=1)

    for ep in range(epochs + 1):
        opt.zero_grad()

        loss_pde = torch.mean(pde_residual(model, xy_f, f_func) ** 2)
        loss_bc = boundary_loss(model, x_b, y_b, g_func)
        loss = lam_pde * loss_pde + lam_bc * loss_bc

        # Проверка на NaN
        if torch.isnan(loss):
            print(f"NaN detected at epoch {ep}. Stopping training.")
            break

        loss.backward()
        opt.step()

        # Сохраняем потери
        history["total_loss"].append(loss.item())
        history["pde_loss"].append(loss_pde.item())
        history["bc_loss"].append(loss_bc.item())

        if ep % verbose_every == 0:
            # Вычисляем ошибки решения
            rmse, rel_l2 = compute_errors(model, device)
            history["rmse"].append(rmse)
            history["relative_l2"].append(rel_l2)
            print(
                f"Epoch {ep:5d} | Loss: {loss.item():.3e} | RMSE: {rmse:.3e}"
            )

    # Гарантируем, что у нас есть хотя бы одно значение ошибки
    if not history["rmse"]:
        rmse, rel_l2 = compute_errors(model, device)
        history["rmse"].append(rmse)
        history["relative_l2"].append(rel_l2)

    return model, history


def run_lambda_experiment(layers=(32, 64, 32), lambda_values=[0.1, 1.0, 10.0, 50.0, 100.0]):
    """Запуск эксперимента с разными весами граничных условий"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = {}

    for lam_bc in lambda_values:
        print(f"\n{'=' * 50}")
        print(f"Эксперимент с λ_bc = {lam_bc}")
        print(f"{'=' * 50}")

        # Создаем новую модель для каждого эксперимента
        model = PINN(hidden_sizes=layers).to(device)

        # Обучаем с текущим λ
        trained_model, history = train_with_history(
            model,
            f_func,
            g_func,
            generate_domain_points,
            generate_boundary_points,
            lam_bc=lam_bc,
            lam_pde=1.0,
            epochs=10000,
            device=device,
            verbose_every=1000,
        )

        # Безопасное извлечение финальных ошибок
        final_rmse = history["rmse"][-1] if history["rmse"] else float("nan")
        final_relative_l2 = (
            history["relative_l2"][-1]
            if history["relative_l2"]
            else float("nan")
        )

        # Сохраняем результаты
        results[lam_bc] = {
            "model": trained_model,
            "history": history,
            "final_rmse": final_rmse,
            "final_relative_l2": final_relative_l2,
        }

        print(
            f"Завершено: λ_bc = {lam_bc}, Final Relative L2 = {final_relative_l2:.2e}"
        )

    return results


def plot_lambda_comparison(results):
    """Сравнительный анализ влияния λ"""

    lambda_values = sorted(results.keys())
    final_errors = []
    valid_lambdas = []

    # Собираем только валидные результаты
    for lam in lambda_values:
        error = results[lam]["final_relative_l2"]
        if not np.isnan(error):
            final_errors.append(error)
            valid_lambdas.append(lam)

    if not valid_lambdas:
        print("Нет валидных результатов для построения графиков")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Final error vs lambda
    axes[0, 0].semilogx(
        valid_lambdas, final_errors, "o-", linewidth=2, markersize=8
    )
    axes[0, 0].set_xlabel("λ (log scale)")
    axes[0, 0].set_ylabel("Final relative L2 norm")
    axes[0, 0].set_title("Точность решения")
    axes[0, 0].grid(True)

    # 2. Loss trajectories for selected lambdas
    # selected_lambdas = valid_lambdas
    for lam in valid_lambdas:
        history = results[lam]["history"]
        axes[0, 1].semilogy(history["total_loss"], label=f"λ={lam}", alpha=0.7)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Total Loss")
    axes[0, 1].set_title("Обучаемость модели")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. PDE vs BC loss balance
    for lam in valid_lambdas:
        history = results[lam]["history"]
        pde_loss = history["pde_loss"]
        bc_loss = history["bc_loss"]
        n_points = len(pde_loss)
        ratio = [
            p / b if b > 1e-10 else 0
            for p, b in zip(pde_loss[:n_points], bc_loss[:n_points])
        ]
        axes[1, 0].semilogy(ratio, label=f"λ={lam}", alpha=0.7)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("PDE Loss / BC Loss")
    axes[1, 0].set_title("Потери на PDE против потерь на BC")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Error convergence
    for lam in valid_lambdas:
        history = results[lam]["history"]
        errors = history["relative_l2"]
        epochs = [i * 1000 for i in range(len(errors))]
        axes[1, 1].semilogy(epochs, errors, "o-", label=f"λ={lam}")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Relative L2 Error")
    axes[1, 1].set_title("Сходимость ошибки")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("\n=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===")
    for lam in lambda_values:
        if lam in valid_lambdas:
            error = results[lam]["final_relative_l2"]
            print(f"λ = {lam:6.1f} | Relative L2 Error = {error:.2e}")
        else:
            print(f"λ = {lam:6.1f} | No valid result")


if __name__ == "__main__":
    import random

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(69)

    results = run_lambda_experiment()
    plot_lambda_comparison(results)
