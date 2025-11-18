import torch
from config import Config
from model import PINN
from pinn_model.loss import PoissonLosses
from training.trainer import Trainer
from data import DataGenerator
from visualization import SolutionVisualizer
from utils import compute_errors, set_seed
from pde import u_exact


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    model = PINN(hidden_sizes=cfg.layers).to(device)

    data_gen = DataGenerator
    loss_class = PoissonLosses

    trainer = Trainer(
        model=model,
        config=cfg,
        loss_class=loss_class,
        data_gen=data_gen,
    )

    trained_model, history = trainer.train()

    mse, rel_l2, X, Y, U_pred, U_true = compute_errors(
        trained_model, device, u_exact, return_fields=True
    )

    print(f"MSE={mse:.3e}, RelL2={rel_l2:.3e}")

    sv = SolutionVisualizer()
    sv.plot_2d(X, Y, U_pred, U_true)
    sv.plot_3d(X, Y, U_pred, U_true)


if __name__ == "__main__":
    main()
