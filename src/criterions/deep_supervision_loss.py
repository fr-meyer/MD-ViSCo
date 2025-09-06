"""
Deep Supervision Loss criterion for models with multiple output scales.

This module provides a configurable deep supervision loss implementation that handles
models with multiple outputs at different scales (e.g., PPG2ABP, NABNet). It supports
both tuple outputs (fixed-length) and list outputs (variable-length) with automatic
weight normalization and interpolation.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType
from omegaconf import MISSING



@dataclass
class DeepSupervisionLossConfig(CriterionBaseConfig):
    """
    Configuration for Deep Supervision Loss criterion.

    Attributes:
        base_criterion (BaseCriterion): Base criterion instance to use
        loss_weights (Optional[List[float]]): Custom loss weights. If None, auto-determined
        reduction (ReductionType): Reduction method. Default: ReductionType.MEAN
        return_individual_losses (bool): Whether to return individual scale losses. Default: False
    """
    _target_: str = "src.criterions.deep_supervision_loss.DeepSupervisionLoss"
    name: str = "deep_supervision_loss"
    base_criterion: CriterionBaseConfig = MISSING
    loss_weights: Optional[List[float]] = None
    reduction: ReductionType = ReductionType.MEAN
    return_individual_losses: bool = False

    def __post_init__(self):
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(f"reduction must be a ReductionType enum, got {type(self.reduction)}")
        if self.loss_weights is not None and any(w < 0 for w in self.loss_weights):
            raise ValueError("All loss weights must be non-negative")

    def to_dict(self) -> dict:
        return {
            'base_criterion': self.base_criterion,
            'loss_weights': self.loss_weights,
            'reduction': self.reduction.value,
            'device': self.device,
            'name': self.name,
            'log_loss': self.log_loss,
            'return_individual_losses': self.return_individual_losses
        }

    def __str__(self) -> str:
        return (f"DeepSupervisionLossConfig(base_criterion='{self.base_criterion}', "
                f"reduction='{self.reduction.value}', enabled={self.enabled})")


class DeepSupervisionLoss(BaseCriterion):
    """
    Deep Supervision Loss for models with multiple output scales.

    This criterion handles models that produce multiple outputs at different scales
    (e.g., PPG2ABP with tuple outputs, NABNet with list outputs). It automatically
    interpolates outputs to match the target size and applies weighted loss computation.

    Args:
        base_criterion (BaseCriterion): Base criterion instance to use
        loss_weights (Optional[List[float]]): Custom loss weights. If None, auto-determined
        reduction (ReductionType): Reduction method. Default: ReductionType.MEAN
        device (Optional[torch.device]): Device to compute the loss on. Default: None
        name (Optional[str]): Name for logging. Default: None
        log_loss (bool): Whether to log loss values. Default: False
        return_individual_losses (bool): Whether to return individual scale losses. Default: False
    """
    def __init__(
        self,
        base_criterion: BaseCriterion,
        loss_weights: Optional[List[float]] = None,
        reduction: ReductionType = ReductionType.MEAN,
        return_individual_losses: bool = False,
        device: Optional[torch.device] = None,
        name: str = "deep_supervision_loss",
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
        
        # Store additional fields specific to this criterion
        self.base_criterion = base_criterion
        self.loss_weights = loss_weights
        self.return_individual_losses = return_individual_losses

    def _normalize_weights(self, num_outputs: int) -> List[float]:
        """Normalize loss weights based on number of outputs."""
        if self.loss_weights is not None:
            if len(self.loss_weights) != num_outputs:
                raise ValueError(f"Expected {num_outputs} weights, got {len(self.loss_weights)}")
            return self.loss_weights
        
        # Default to linear decay: [1.0, 0.9, ..., max(0.1, 1 - 0.1 * (n-1))]
        # This ensures weights are always positive and decreasing
        return [max(1.0 - 0.1 * i, 0.1) for i in range(num_outputs)]

    def _interpolate_output(self, output: torch.Tensor, target_size: int) -> torch.Tensor:
        """Interpolate output to match target size."""
        if output.shape[-1] != target_size:
            return F.interpolate(
                output,
                size=target_size,
                mode='linear',
                align_corners=False
            )
        return output

    def _compute_scale_loss(self, output: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        """Compute base criterion loss for a single scale with weight."""
        # Interpolate output to match target size
        resized_output = self._interpolate_output(output, target.shape[-1])
        
        # Compute base criterion loss (assumes it returns a scalar)
        scale_loss = self.base_criterion(resized_output, target)
        
        # Apply weight
        return scale_loss * weight

    def compute_loss(self, outputs, targets, **kwargs) -> torch.Tensor:
        """
        Override compute_loss to handle multi-output cases properly.
        
        Args:
            outputs: Model outputs (single tensor, tuple, or list)
            targets: Target tensor
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Computed loss
        """
        # For single tensor case, use parent's compute_loss
        if not isinstance(outputs, (tuple, list)):
            return super().compute_loss(outputs, targets, **kwargs)
        
        # For multi-output case, bypass parent validation and use forward directly
        result = self.forward(outputs, targets)
        
        # Handle case where forward returns a tuple (when return_individual_losses=True)
        if isinstance(result, tuple):
            return result[0]  # Return only the combined loss
        else:
            return result

    def forward(
        self, 
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]], 
        targets: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Compute deep supervision loss.

        Args:
            outputs: Model outputs. Can be:
                - Single tensor (no deep supervision)
                - Tuple of tensors (fixed-length, e.g., PPG2ABP)
                - List of tensors (variable-length, e.g., NABNet)
            targets: Target tensor

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
                Combined loss value and optionally per-scale losses.
        """
        # Handle single output case
        if not isinstance(outputs, (tuple, list)):
            return self.base_criterion(outputs, targets)

        # Convert to list for uniform handling
        output_list = list(outputs) if isinstance(outputs, tuple) else outputs
        num_outputs = len(output_list)

        # Normalize weights
        weights = self._normalize_weights(num_outputs)

        # Compute individual losses
        individual_losses = []
        for output, weight in zip(output_list, weights):
            scale_loss = self._compute_scale_loss(output, targets, weight)
            individual_losses.append(scale_loss)

        # Combine losses
        if self.reduction == ReductionType.NONE:
            # For 'none' reduction, stack and sum across scales
            combined_loss = torch.stack(individual_losses).sum(dim=0)
        else:
            # For 'mean' or 'sum', sum the scalar losses as tensors
            combined_loss = torch.stack(individual_losses).sum()

        # Return based on flag
        if self.return_individual_losses:
            return combined_loss, individual_losses
        else:
            return combined_loss

    def extra_repr(self) -> str:
        """Return string representation of the criterion's parameters."""
        return (f"base_criterion={self.base_criterion}, "
                f"reduction={self.reduction.value}, "
                f"return_individual_losses={self.return_individual_losses}")

# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_deep_supervision_loss", node=DeepSupervisionLossConfig)

