"""
Mean Absolute Error (L1) Loss criterion for regression tasks.

This module provides a configurable L1 loss implementation for use in training models.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType

@dataclass
class L1LossConfig(CriterionBaseConfig):
    """
    Configuration for L1 Loss criterion.

    Attributes:
        reduction (ReductionType): Reduction method. Default: ReductionType.MEAN
    """
    _target_: str = "src.criterions.l1_loss.L1Loss"
    name: str = "l1_loss"
    reduction: ReductionType = ReductionType.MEAN

    def __post_init__(self):
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(f"reduction must be a ReductionType enum, got {type(self.reduction)}")

    def to_dict(self) -> dict:
        return {
            'reduction': self.reduction.value,
            'device': self.device,
            'name': self.name,
            'log_loss': self.log_loss
        }

    def __str__(self) -> str:
        return (f"L1LossConfig(reduction='{self.reduction.value}', enabled={self.enabled})")

class L1Loss(BaseCriterion):
    """
    Mean Absolute Error (L1) Loss for regression tasks.
    """
    def __init__(
        self,
        reduction: ReductionType = ReductionType.MEAN,
        device: Optional[torch.device] = None,
        name: str = "l1_loss",
        log_loss: bool = False,
        enabled: bool = True
    ):
        # Call parent constructor with explicit parameters
        super().__init__(
            reduction=reduction,
            device=device,
            name=name,
            log_loss=log_loss
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the L1 loss between input and target.
        """
        loss = F.l1_loss(input, target, reduction=self.reduction.value)
        return loss

# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_l1_loss", node=L1LossConfig)
