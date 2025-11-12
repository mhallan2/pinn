from abc import ABC, abstractmethod
from utils import compute_errors
from pde import u_exact



class Experiment(ABC):
    """Абстрактный базовый класс для всех экспериментов."""
    
    def __init__(self, cfg, model_cls, f_func, g_func, data_gen):
        self.cfg = cfg
        self.model = model_cls(hidden_sizes=cfg.layers).to(cfg.device)
        self.f_func = f_func
        self.g_func = g_func
        self.data_gen = data_gen
        self.history = None
        self.rmse_per_epoch = []
        self.rel_l2_per_epoch = []

    def epoch_callback(self, model, epoch):
        """Вызывается на каждой эпохе для сбора метрик."""
        rmse, rel_l2 = compute_errors(model, self.cfg.device, u_exact)
        self.rmse_per_epoch.append(rmse)
        self.rel_l2_per_epoch.append(rel_l2)

    @abstractmethod
    def get_experiment_name(self):
        pass

    @abstractmethod
    def prepare_training_config(self):
        pass

    @abstractmethod
    def create_result(self, history, rmse_final, rel_l2_final):
        pass

    def run(self, trainer_cls):
        """Основной метод запуска эксперимента."""
        print(f"\n=== {self.get_experiment_name()} ===")

        # Подготовка конфига для обучения
        training_cfg = self.prepare_training_config()

        # Запуск тренировки
        trainer = trainer_cls(
            self.model, training_cfg, self.f_func, self.g_func, self.data_gen
        )
        trained_model, history = trainer.train(epoch_callback=self.epoch_callback)
        self.history = history

        # Конечные метрики
        rmse_final, rel_l2_final = compute_errors(
            trained_model, training_cfg.device, u_exact
        )

        print(f"{self.get_experiment_name()} | RMSE={rmse_final:.3e} | Rel L2={rel_l2_final:.3e}")

        return self.create_result(history, rmse_final, rel_l2_final)
    