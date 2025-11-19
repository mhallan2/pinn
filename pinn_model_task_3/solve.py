import torch
from config import Config
from model import PINN
from data import DataGenerator
from loss import HeatLosses
from training.trainer import Trainer
from visualization import SolutionVisualizer
from utils import set_seed, compute_errors_time
from pde import u_exact

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PINN(hidden_sizes=cfg.layers).to(device)
    data_gen = DataGenerator
    loss_class = HeatLosses

    trainer = Trainer(model, cfg, loss_class, data_gen)
    trained_model, history = trainer.train()

    mse, rel_l2, X, Tt, U_pred, U_true = compute_errors_time(
        trained_model, device, u_exact, L=cfg.L, T=cfg.T, return_fields=True
    )

    sv = SolutionVisualizer()
    sv.plot_2d(X, Tt, U_pred, U_true)
    sv.plot_3d(X, Tt, U_pred, U_true)

if __name__ == "__main__":
    main()
