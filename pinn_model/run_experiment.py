from model import PINN
from config import Config
from data import DataGenerator
from training.trainer import Trainer
from experiments.experiment import LambdaExperiment
from visualization import LambdaVisualizer
from utils import set_seed
from loss import PoissonLosses


def main():
    
    # Конфигурация эксперимента
    cfg = Config()
    set_seed(cfg.seed)

    device = "cuda" if cfg.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Генератор данных и модель
    data_gen = DataGenerator()
    model = PINN(hidden_sizes=cfg.layers).to(cfg.device)
    
    # Список значений λ для эксперимента
    lambda_values = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

    results = []
    for lam in lambda_values:
        exp = LambdaExperiment(
            lam_bc=lam,
            cfg=cfg,
            model=model,
            loss_class=PoissonLosses,
            data_gen=data_gen,
        )
        result = exp.run(Trainer)
        results.append(result)

    
    # Визуализация результатов
    lv = LambdaVisualizer(base_dir="results")
    lv.plot_results(results)


if __name__ == "__main__":
    main()
