# python solve.py
import torch
from config import Config
from model import PINN
from training.trainer import Trainer
from data import DataGenerator
from visualization import SolutionVisualizer
from utils import compute_errors, set_seed
from pde import f_func, g_func, u_exact


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PINN(hidden_sizes=cfg.layers).to(device)
    data_gen = DataGenerator()
    trainer = Trainer(model, cfg, f_func, g_func, data_gen)
    model, history = trainer.train()

    rmse, rel_l2, X, Y, U_pred, U_true = compute_errors(
        model, device, u_exact, return_fields=True
    )
    print(f"RMSE = {rmse:.3e}, Rel L2 = {rel_l2:.3e}")

    # --- 4. Визуализация ---
    sv = SolutionVisualizer()
    sv.plot_2d(X, Y, U_pred, U_true)
    sv.plot_3d(X, Y, U_pred, U_true)


if __name__ == "__main__":
    main()
