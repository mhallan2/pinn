import os
from abc import ABC
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(ABC):
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

    def plot_2d(self, X, Tt, U_pred, U_true):
        error = np.abs(U_pred - U_true)
        fig, axes = plt.subplots(1,3,figsize=(18,5))

        for ax, data, title in zip(axes, [U_pred,U_true,error], ["Prediction","Exact","Error"]):
            im = ax.contourf(X, Tt, data, 100, cmap="viridis" if title!="Error" else "hot")
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{title} (2D)")
            ax.set_xlabel("x")
            ax.set_ylabel("t")

        plt.tight_layout()
        self._save_plot(fig, "solution_2d")
        plt.show()

    def plot_3d(self, X, Tt, U_pred, U_true):

        error = np.abs(U_pred - U_true)
        fig = plt.figure(figsize=(18,5))

        for i, (data, title) in enumerate(zip([U_pred,U_true,error], ["Prediction","Exact","Error"])):
            ax = fig.add_subplot(1,3,i+1, projection="3d")
            surf = ax.plot_surface(X, Tt, data, cmap="viridis" if title!="Error" else "hot", alpha=0.7)
            fig.colorbar(surf, ax=ax, shrink=0.5)
            ax.set_title(f"{title} (3D)")
            ax.set_xlabel("x")
            ax.set_ylabel("t")

        plt.tight_layout()
        self._save_plot(fig, "solution_3d")
        plt.show()
