from dataclasses import dataclass, field

@dataclass
class TrainingHistory:
    total_loss: list[float] = field(default_factory=list)
    pde_loss: list[float] = field(default_factory=list)
    bc_loss: list[float] = field(default_factory=list)
    ic_loss: list[float] = field(default_factory=list)
