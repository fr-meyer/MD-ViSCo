"""Mean Squared Error (MSE) Loss criterion for regression tasks.

This module provides a configurable MSE loss implementation for use in training models.
Registers config with Hydra ConfigStore at import time.
"""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import torch
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional
from hydra.core.config_store import ConfigStore

# Local imports
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType


@dataclass
class MSELossConfig(CriterionBaseConfig):
    """Configuration for MSE Loss criterion.

    Attributes:
        reduction: Reduction method. Default: ReductionType.MEAN.
    """

    _target_: str = "src.criterions.mse_loss.MSELoss"
    name: str = "mse_loss"
    reduction: ReductionType = ReductionType.MEAN

    def __post_init__(self):
        """Validate configuration after initialization.

        Raises:
            ValueError: If reduction is not a ReductionType.
        """
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(
                f"reduction must be a ReductionType enum, got {type(self.reduction)}"
            )


class MSELoss(BaseCriterion):
    """Mean Squared Error (MSE) Loss for regression tasks."""

    def __init__(
        self,
        reduction: ReductionType = ReductionType.MEAN,
        device: torch.device | None = None,
        name: str = "mse_loss",
        log_loss: bool = False,
    ):
        """Initialize MSE Loss.

        Args:
            reduction: Reduction method for the loss.
            device: Device to compute the loss on.
            name: Name for logging purposes.
            log_loss: Whether to log loss values.
        """
        super().__init__(
            reduction=reduction, device=device, name=name, log_loss=log_loss
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the MSE loss between input and target.

        Args:
            input: Predictions tensor (same shape as target).
            target: Ground truth tensor.

        Returns:
            Loss tensor, scalar if reduction is 'mean' or 'sum'.
        """
        loss = F.mse_loss(input, target, reduction=self.reduction)
        return loss


cs = ConfigStore.instance()
cs.store(group="criterion", name="base_mse_loss", node=MSELossConfig)
