from model import PINN
from data import generate_domain_points, generate_boundary_points
from train import train
from utils import f_func, g_func, plot_solution
import torch
import random
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

layers = (128, 128, 128)
model = PINN(hidden_sizes=layers).to(device)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(69)
train(
    model,
    f_func,
    g_func,
    generate_domain_points,
    generate_boundary_points,
    epochs=10000,
    lr=1e-4,
    lam_pde=1.0,
    lam_bc=10.0,
    device=device,
)
plot_solution(model, device=device)
