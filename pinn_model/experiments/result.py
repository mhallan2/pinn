from dataclasses import dataclass, field
from training.history import TrainingHistory


@dataclass
class LambdaRunResult:
    lam_bc: float
    history: TrainingHistory
    mse_final: float
    rel_l2_final: float
    mse_per_epoch: list[float] = field(default_factory=list)
    rel_l2_per_epoch: list[float] = field(default_factory=list)
