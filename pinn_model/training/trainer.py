import torch
from training.history import TrainingHistory
from loss import Losses


class Trainer:
    def __init__(self, model, config, f_func, g_func, data_gen):
        self.model = model
        self.cfg = config
        self.f_func = f_func
        self.g_func = g_func
        self.data_gen = data_gen
        self.history = TrainingHistory()
        self.device = config.device

    def train(self, epoch_callback=None):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=500
        )

        x_b, y_b = self.data_gen.boundary_points(self.cfg.N_b, device=self.device)
        self.model.train()

        for epoch in range(self.cfg.epochs + 1):
            opt.zero_grad()
            x_f, y_f = self.data_gen.domain_points(self.cfg.N_f, device=self.device)
            xy_f = torch.cat([x_f, y_f], dim=1)

            loss_pde = Losses.pde_loss(self.model, xy_f, self.f_func)
            loss_bc = Losses.boundary_loss(self.model, x_b, y_b, self.g_func)
            loss = self.cfg.lam_pde * loss_pde + self.cfg.lam_bc * loss_bc

            loss.backward()
            opt.step()
            scheduler.step(loss.item())

            self.history.total_loss.append(loss.item())
            self.history.pde_loss.append(loss_pde.item())
            self.history.bc_loss.append(loss_bc.item())

            if epoch_callback:
                epoch_callback(self.model, epoch)

            if epoch % self.cfg.verbose_every == 0:
                print(
                    f"[{epoch:5d}] Total={loss.item():.3e} PDE={loss_pde.item():.3e} BC={loss_bc.item():.3e}"
                )

        return self.model, self.history
