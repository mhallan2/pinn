import os
from abc import ABC
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(ABC):
    """Базовый класс для визуализации с общими методами."""
    
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_unique_path(self, prefix="plot"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.base_dir, f"{prefix}_{timestamp}.png")

    def _save_plot(self, fig, prefix="plot"):
        path = self._get_unique_path(prefix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")


class SolutionVisualizer(Visualizer):
    """Визуализация решений PDE."""
    
    def plot_2d(self, X, Y, U_pred, U_true):
        """2D график решения и ошибки."""
        error = np.abs(U_pred - U_true)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax, data, title in zip(axes, [U_pred, U_true, error], 
                                 ["Prediction", "Exact", "Error"]):
            im = ax.contourf(X, Y, data, 100, cmap="viridis" if title != "Error" else "hot")
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{title} (2D)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        
        plt.tight_layout()
        self._save_plot(fig, "solution_2d")
        plt.show()

    def plot_3d(self, X, Y, U_pred, U_true):
        """3D график решения и ошибки."""
        error = np.abs(U_pred - U_true)
        
        fig = plt.figure(figsize=(18, 5))
        
        for i, (data, title) in enumerate(zip([U_pred, U_true, error], 
                                            ["Prediction", "Exact", "Error"])):
            ax = fig.add_subplot(1, 3, i+1, projection="3d")
            surf = ax.plot_surface(X, Y, data, cmap="viridis" if title != "Error" else "hot", 
                                 alpha=0.7)
            fig.colorbar(surf, ax=ax, shrink=0.5)
            ax.set_title(f"{title} (3D)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        
        plt.tight_layout()
        self._save_plot(fig, "solution_3d")
        plt.show()


class LambdaVisualizer(Visualizer):
    """Визуализация экспериментов с λ."""
    
    def plot_results(self, results):
        """Все графики для экспериментов с λ."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Финальные ошибки
        lambdas = [r.lam_bc for r in results]
        errors = [r.rel_l2_final for r in results]
        axes[0,0].semilogx(lambdas, errors, 'o-', linewidth=2)
        axes[0,0].set_xlabel("λ")
        axes[0,0].set_ylabel("Final L2 Error")
        axes[0,0].grid(True)
        
        # Сходимость по эпохам
        for r in results:
            epochs = range(len(r.rel_l2_per_epoch))
            axes[0,1].semilogy(epochs, r.rel_l2_per_epoch, label=f"λ={r.lam_bc}")
        axes[0,1].legend()
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("L2 Error")
        axes[0,1].grid(True)
        
        # Loss
        for r in results:
            axes[1,0].semilogy(r.history.total_loss, label=f"λ={r.lam_bc}")
        axes[1,0].legend()
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Total Loss")
        axes[1,0].grid(True)
        
        # Соотношение loss-ов
        for r in results:
            ratio = np.array(r.history.pde_loss) / np.array(r.history.bc_loss)
            axes[1,1].semilogy(ratio, label=f"λ={r.lam_bc}")
        axes[1,1].legend()
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("PDE/BC Loss Ratio")
        axes[1,1].grid(True)
        
        plt.tight_layout()
        self._save_plot(fig, "lambda_results")
        plt.show()
