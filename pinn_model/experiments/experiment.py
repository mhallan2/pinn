from abc import ABC, abstractmethod
from dataclasses import replace
from utils import compute_errors
from pde import u_exact
from experiments.result import LambdaRunResult


class Experiment(ABC):
    """Абстрактный базовый класс для экспериментов с гиперпараметрами."""
    
    def __init__(self, cfg, model, loss_class, data_gen):
        self.cfg = cfg
        self.model = model
        self.loss_class = loss_class
        self.data_gen = data_gen
        self.history = None
        self.mse_per_epoch = []
        self.rel_l2_per_epoch = []

    def epoch_callback(self, model, epoch):
        """Вызывается на каждой эпохе для сбора метрик."""
        mse, rel_l2 = compute_errors(model, self.cfg.device, u_exact)
        self.mse_per_epoch.append(mse)
        self.rel_l2_per_epoch.append(rel_l2)

    @abstractmethod
    def get_experiment_name(self):
        pass

    @abstractmethod
    def prepare_training_config(self):
        pass

    @abstractmethod
    def create_result(self, history, mse_final, rel_l2_final):
        pass

    def run(self, trainer_cls):
        """Основной метод запуска эксперимента."""
        print(f"\n=== {self.get_experiment_name()} ===")

        # Подготовка конфига для обучения
        training_cfg = self.prepare_training_config()

        # Запуск тренировки
        trainer = trainer_cls(
            self.model, training_cfg, self.loss_class, self.data_gen
        )
        trained_model, history = trainer.train(epoch_callback=self.epoch_callback)
        self.history = history

        # Конечные метрики
        mse_final, rel_l2_final = compute_errors(
            trained_model, training_cfg.device, u_exact
        )

        print(f"{self.get_experiment_name()} | MSE={mse_final:.3e} | Rel L2={rel_l2_final:.3e}")

        return self.create_result(history, mse_final, rel_l2_final)

class LambdaExperiment(Experiment):
    """
    Эксперимент по изучению влияния весового коэффициента λ
    перед слагаемым с граничными условиями в функции потерь
    на точность и эффективность решения.
    """

    def __init__(self, lam_bc, cfg, model, loss_class, data_gen):
        super().__init__(cfg, model, loss_class, data_gen)
        self.lam_bc = lam_bc

    def get_experiment_name(self):
        return f"LambdaExperiment: λ = {self.lam_bc}"

    def prepare_training_config(self):
        return replace(self.cfg, lam_bc=self.lam_bc)

    def create_result(self, history, mse_final, rel_l2_final):
        """Создает объект результата"""
        return LambdaRunResult(
            lam_bc=self.lam_bc,
            history=history,
            mse_final=mse_final,
            rel_l2_final=rel_l2_final,
            mse_per_epoch=self.mse_per_epoch,
            rel_l2_per_epoch=self.rel_l2_per_epoch,
        )
