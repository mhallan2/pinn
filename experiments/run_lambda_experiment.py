"""
Модуль для исследования влияния веса граничных условий (λ_bc)
на сходимость и точность Physics-Informed Neural Network (PINN).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..pinn_model.model import PINN
from config import CONFIG
from pinn_model.data import generate_domain_points, generate_boundary_points
from pinn_model.train import train
from pinn_model.utils import f_func, g_func, set_seed, compute_errors


def plot_lambda_results(results):
    """Построение графиков сравнения между λ_bc."""

    lambdas = sorted(results.keys())
    rel_errors = [results[lam]["final_relative_l2"] for lam in lambdas]
    rmse_errors = [results[lam]["final_rmse"] for lam in lambdas]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Ошибка L2 от λ_bc
    axs[0, 0].semilogx(lambdas, rel_errors, "o-", linewidth=2, markersize=8)
    axs[0, 0].set_xlabel("λ_bc (log scale)")
    axs[0, 0].set_ylabel("Relative L2 Error")
    axs[0, 0].set_title("Зависимость точности от λ_bc")
    axs[0, 0].grid(True)

    # 2. RMSE от λ_bc
    axs[0, 1].semilogx(lambdas, rmse_errors, "s--", linewidth=2, markersize=8, color="orange")
    axs[0, 1].set_xlabel("λ_bc (log scale)")
    axs[0, 1].set_ylabel("RMSE")
    axs[0, 1].set_title("RMSE при разных λ_bc")
    axs[0, 1].grid(True)

    # 3. Графики потерь
    for lam in lambdas:
        hist = results[lam]["history"]
        axs[1, 0].semilogy(hist["total_loss"], label=f"λ={lam}")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Total Loss")
    axs[1, 0].set_title("Динамика обучения")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Отношение PDE/BC потерь
    for lam in lambdas:
        hist = results[lam]["history"]
        ratio = np.array(hist["pde_loss"]) / (np.array(hist["bc_loss"]) + 1e-8)
        axs[1, 1].semilogy(ratio, label=f"λ={lam}")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("PDE / BC Loss ratio")
    axs[1, 1].set_title("Баланс между PDE и BC потерями")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Таблица результатов
    print("\n=== Итоговые ошибки ===")
    for lam in lambdas:
        r = results[lam]
        print(f"λ_bc = {lam:6.2f} | RMSE = {r['final_rmse']:.3e} | Rel L2 = {r['final_relative_l2']:.3e}")


# ======================================
# === Основной эксперимент ============
# ======================================

def run_lambda_experiment(
    lambda_values=None,
    layers=(64, 64, 64),
    output_dir="results_lambda",
):
    """
    Запуск серии экспериментов с разными λ_bc.

    Parameters
    ----------
    lambda_values : list[float]
        Список значений λ_bc для тестирования.
    layers : tuple[int]
        Архитектура скрытых слоёв.
    output_dir : str
        Папка для сохранения результатов.
    """
    if lambda_values is None:
        lambda_values = [0.1, 1.0, 10.0, 50.0, 100.0]

    cfg = CONFIG.copy()
    set_seed(cfg["seed"])
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print(f"Запуск экспериментов на устройстве: {cfg['device']}")
    print(f"Тестируем λ_bc ∈ {lambda_values}")

    for lam_bc in lambda_values:
        print(f"\n{'='*50}")
        print(f"Обучение с λ_bc = {lam_bc}")
        print(f"{'='*50}")

        cfg["lam_bc"] = lam_bc
        model = PINN(hidden_sizes=layers).to(cfg["device"])

        trained_model, history = train(
            model,
            f_func,
            g_func,
            generate_domain_points,
            generate_boundary_points,
            cfg,
        )

        rmse, rel_l2 = compute_errors(trained_model, cfg["device"])
        results[lam_bc] = {
            "model": trained_model,
            "history": history,
            "final_rmse": rmse,
            "final_relative_l2": rel_l2,
        }

        print(f" λ_bc = {lam_bc:.2f} | RMSE = {rmse:.3e} | Rel L2 = {rel_l2:.3e}")

    # Построить графики
    plot_lambda_results(results)

    # Сохранить результаты
    np.save(os.path.join(output_dir, "lambda_experiment.npy"), results, allow_pickle=True)
    print(f"\n Результаты сохранены в {output_dir}/lambda_experiment.npy")

    return results


if __name__ == "__main__":
    # Пример запуска
    run_lambda_experiment(lambda_values=[0.1, 1.0, 10.0, 50.0, 100.0])
