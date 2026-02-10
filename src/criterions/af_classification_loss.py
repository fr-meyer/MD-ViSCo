"""AF Classification Loss Function.

This module implements a binary classification loss function optimized for
atrial fibrillation detection using BCEWithLogitsLoss for numerical stability.
"""

# Standard library imports
from dataclasses import dataclass

import torch
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig


@dataclass
class AFClassificationLossConfig(CriterionBaseConfig):
    """Configuration for AF Classification Loss."""

    _target_: str = "src.criterions.af_classification_loss.AFClassificationLoss"
    name: str = "af_classification_loss"

    # Loss parameters
    pos_weight: float | None = (
        None  # Weight for positive class (handles class imbalance)
    )
    reduction: str = "mean"


class AFClassificationLoss(BaseCriterion):
    """Binary classification loss using BCEWithLogitsLoss.

    This loss function is optimized for binary classification and matches
    sklearn's internal behavior for MLPClassifier binary classification.

    BCEWithLogitsLoss combines sigmoid activation and binary cross entropy
    in a numerically stable way, which is more stable than using sigmoid
    followed by BCE separately.

    Args:
        pos_weight (Optional[float]): Weight for positive class. Useful for handling
            class imbalance. If None, no weighting is applied.
        reduction (str): Reduction method for the loss ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        pos_weight: float | None = None,
        reduction: str = "mean",
        *args,
        **kwargs,
    ):
        """Initialize AF Classification Loss.

        Args:
            pos_weight: Weight for positive class to handle imbalance.
            reduction: Reduction method ('mean', 'sum', 'none').
            *args: Variable positional arguments passed to parent.
            **kwargs: Variable keyword arguments passed to parent.
        """
        super().__init__(*args, **kwargs)

        # Use BCEWithLogitsLoss for numerical stability
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight))
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight, reduction=reduction
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            predictions: Logits from model [N, 1]
            targets: Class labels [N] (0 or 1)

        Returns:
            torch.Tensor: Binary cross-entropy loss value.
        """
        # BCEWithLogitsLoss expects float and [N, 1] or [N]
        targets_float = targets.float().unsqueeze(1)  # [N, 1]

        return self.loss_fn(predictions, targets_float)

    def extra_repr(self) -> str:
        """Return string representation of the criterion's parameters."""
        return (
            f"pos_weight={self.loss_fn.pos_weight}, reduction={self.loss_fn.reduction}"
        )


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_af_classification_loss",
    node=AFClassificationLossConfig,
    group="criterion",
)
