"""
Mean Squared Error (MSE) Loss criterion for regression tasks.

This module provides a configurable MSE loss implementation for use in training models.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType

@dataclass
class MSELossConfig(CriterionBaseConfig):
    """Pure data container for MSE Loss configuration"""
    _target_: str = "src.criterions.mse_loss.MSELoss"
    name: str = "mse_loss"
    reduction: ReductionType = ReductionType.MEAN

class MSELoss(BaseCriterion):
    """Mean Squared Error (MSE) Loss for regression tasks"""
    
    def __init__(
        self,
        reduction: ReductionType = ReductionType.MEAN,
        device: Optional[torch.device] = None,
        name: str = "mse_loss",
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
        """Compute the MSE loss between input and target"""
        loss = F.mse_loss(input, target, reduction=self.reduction.value)
        return loss

# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_mse_loss", node=MSELossConfig)
