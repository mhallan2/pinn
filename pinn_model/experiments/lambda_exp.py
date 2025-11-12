from dataclasses import replace
from experiments.result import LambdaRunResult
from experiments.base_exp import Experiment


class LambdaExperiment(Experiment):
    """
    Эксперимент по изучению влияния весового коэффициента λ
    перед слагаемым с граничными условиями в функции потерь 
    на точность и эффективность решения.
    """

    def __init__(self, lam_bc, cfg, model_cls, f_func, g_func, data_gen):
        super().__init__(cfg, model_cls, f_func, g_func, data_gen)
        self.lam_bc = lam_bc

    def get_experiment_name(self):
        return f"LambdaExperiment: λ = {self.lam_bc}"

    def prepare_training_config(self):
        return replace(self.cfg, lam_bc=self.lam_bc)

    def create_result(self, history, rmse_final, rel_l2_final):
        """Создает объект результата"""
        return LambdaRunResult(
            lam_bc=self.lam_bc,
            history=history,
            rmse_final=rmse_final,
            rel_l2_final=rel_l2_final,
            rmse_per_epoch=self.rmse_per_epoch,
            rel_l2_per_epoch=self.rel_l2_per_epoch,
        )
