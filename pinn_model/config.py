from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 69
    device: str = "cuda"
    epochs: int = 5000
    lr: float = 1e-4
    N_f: int = 5000
    N_b: int = 1000
    layers: tuple = (64, 64, 64)
    lam_pde: float = 1.0
    lam_bc: float = 1e3
    verbose_every: int = 1000


cfg = Config()
