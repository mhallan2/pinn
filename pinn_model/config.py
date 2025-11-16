from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 69                  # сид для воспроизводимости
    device: str = "cuda"            # по умолчанию ставим видеокарту, в качестве вычислительного средства
    epochs: int = 5000              # количество итераций обучения
    lr: float = 1e-4                # степень обучаемости
    N_f: int = 5000                 # количество точек внутри области
    N_b: int = 1000                 # количество точек на границе области
    layers: tuple = (64, 64, 64)    # количество слоев и нейронов
    lam_pde: float = 1.0            # весовые коэффициенты в функции потерь
    lam_bc: float = 1e3             # ...
    verbose_every: int = 1000       # вывод прогресса обучения


cfg = Config()
