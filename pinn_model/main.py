import torch
from config import CONFIG
from model import PINN
from data import generate_domain_points, generate_boundary_points
from train import train
from utils import f_func, g_func, u_exact, set_seed, compute_errors
from visualization import plot_2d_solution_and_error, plot_3d_solution_and_error


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    model = PINN(hidden_sizes=cfg["layers"]).to(cfg["device"])
    model, history = train(
        model,
        f_func,
        g_func,
        generate_domain_points,
        generate_boundary_points,
        cfg,
    )

    rmse, rel_l2, X, Y, U_pred, U_true = compute_errors(model, device, u_exact, return_fields=True)
    plot_2d_solution_and_error(X, Y, U_pred, U_true)
    plot_3d_solution_and_error(X, Y, U_pred, U_true)
    #plot_solution(model, u_exact, device=cfg["device"], N=80)


if __name__ == "__main__":
    main()
