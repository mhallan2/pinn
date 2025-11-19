import torch
from training.history import TrainingHistory

class Trainer:
    def __init__(self, model, config, loss_class, data_gen):
        self.model = model
        self.cfg = config
        self.losses = loss_class
        self.data_gen = data_gen
        self.history = TrainingHistory()
        self.device = config.device

    def train(self, epoch_callback=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        # Предсгенерированные точки для границ и начального условия
        x_b, t_b = self.data_gen.boundary_points(self.cfg.N_b, self.cfg.L, self.cfg.T, self.device)
        x_i, t_i = self.data_gen.initial_points(self.cfg.N_i, self.cfg.L, self.device)

        self.model.train()
        for epoch in range(self.cfg.epochs + 1):
            optimizer.zero_grad()

            # Внутренние точки (domain)
            x_f, t_f = self.data_gen.domain_points(self.cfg.N_f, self.cfg.L, self.cfg.T, self.device)
            x_t = torch.cat([x_f, t_f], dim=1)

            # Вычисление потерь
            loss_pde = self.losses.pde_loss(self.model, x_t, alpha=self.cfg.alpha)
            loss_bc = self.losses.boundary_loss(self.model, x_b, t_b)
            loss_ic = self.losses.initial_loss(self.model, x_i, t_i)

            loss_total = self.cfg.lam_pde * loss_pde + self.cfg.lam_bc * loss_bc + self.cfg.lam_ic * loss_ic
            loss_total.backward()
            optimizer.step()

            # Сохранение истории
            self.history.total_loss.append(loss_total.item())
            self.history.pde_loss.append(loss_pde.item())
            self.history.bc_loss.append(loss_bc.item())
            self.history.ic_loss.append(loss_ic.item())

            if epoch_callback:
                epoch_callback(self.model, epoch)

            if epoch % self.cfg.verbose_every == 0:
                print(f"[{epoch:5d}] Total={loss_total:.3e} PDE={loss_pde:.3e} BC={loss_bc:.3e} IC={loss_ic:.3e}")

        return self.model, self.history
