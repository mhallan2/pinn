from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 53
    device: str = "cuda"
    epochs: int = 5000
    lr: float = 1e-3
    N_f: int = 1000        # внутренние точки (x,t)
    N_b: int = 1000        # точки на границе x=0,L
    N_i: int = 1000        # точки начального условия t=0
    layers: tuple = (128, 128, 128, 128)
    lam_pde: float = 1e1
    lam_bc: float = 1e0
    lam_ic: float = 1e0
    L: float = 10.0         # длина по x
    T: float = 1.0         # конечное время
    alpha: float = 0.5     # коэффициент теплопроводности
    verbose_every: int = 1000
