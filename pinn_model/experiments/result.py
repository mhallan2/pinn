from dataclasses import dataclass, field


@dataclass
class LambdaRunResult:
    lam_bc: float
    history: any
    rmse_final: float
    rel_l2_final: float
    rmse_per_epoch: list[float] = field(default_factory=list)
    rel_l2_per_epoch: list[float] = field(default_factory=list)
