import torch
from loss import pde_residual, boundary_loss
from tqdm import trange


# scheduler менят learning rate, возможно, стоит отказаться
def train(model, f_func, g_func, gen_domain, gen_boundary, config):
    """Процесс обучения PINN с сохранением данных."""
    device = config["device"]
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=500)
    model.train()

    x_b, y_b = gen_boundary(config["N_b"], device=device)

    history = {"total_loss": [], "pde_loss": [], "bc_loss": []}

    for ep in trange(config["epochs"], desc="Training", leave=False):
        opt.zero_grad()

        x_f, y_f = gen_domain(config["N_f"], device=device)
        xy_f = torch.cat([x_f, y_f], dim=1)

        loss_pde = torch.mean(pde_residual(model, xy_f, f_func) ** 2)
        loss_bc = boundary_loss(model, x_b, y_b, g_func)
        loss = config["lam_pde"] * loss_pde + config["lam_bc"] * loss_bc

        loss.backward()
        opt.step()
        scheduler.step(loss)

        history["total_loss"].append(loss.item())
        history["pde_loss"].append(loss_pde.item())
        history["bc_loss"].append(loss_bc.item())

        if ep % config["verbose_every"] == 0:
            print(f"[{ep:5d}] Total={loss.item():.3e} PDE={loss_pde.item():.3e} BC={loss_bc.item():.3e}")

    return model, history


def train_UNUSED(
    model,
    f_func,
    g_func,
    generate_domain_points,
    generate_boundary_points,
    N_f=2000,
    N_b=200,
    epochs=5000,
    lr=1e-3,
    lam_pde=1.0,
    lam_bc=10.0,
    device="cpu",
    verbose_every=500,
):
    """Обучение нейросети."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Фиксируем граничные точки
    x_b, y_b = generate_boundary_points(N_b, device=device)

    for ep in range(epochs + 1):
        opt.zero_grad()

        # Генерируем новые точки
        x_f, y_f = generate_domain_points(N_f, device=device)
        xy_f = torch.cat([x_f, y_f], dim=1)
        xy_f.requires_grad_(True)

        # Считаем потери
        loss_pde = torch.mean(pde_residual(model, xy_f, f_func) ** 2)
        loss_bc = boundary_loss(model, x_b, y_b, g_func)
        loss = lam_pde * loss_pde + lam_bc * loss_bc

        loss.backward()
        opt.step()

        # Откладка
        if ep % verbose_every == 0:
            print(
                f"Epoch {ep:5d} | total={loss.item():.3e}, PDE={loss_pde.item():.3e}, BC={loss_bc.item():.3e}"
            )

    return model
