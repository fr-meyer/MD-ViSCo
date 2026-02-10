"""Multi-Regression Loss criterion for handling multiple scalar regression
targets.

This module provides a configurable multi-regression loss implementation
that can handle multiple regression loss terms with configurable weights
and enabling/disabling per term. Reuses existing criterion classes
(L1Loss, MSELoss, etc.) for modularity.
"""

# Standard library imports
from dataclasses import dataclass
from dataclasses import field
from typing import Any

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType


@dataclass
class MultiRegressionLossConfig(CriterionBaseConfig):
    """Pure data container for Multi Regression Loss configuration."""

    _target_: str = "src.criterions.multi_regression_loss.MultiRegressionLoss"
    return_breakdown: bool = True
    loss_terms: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate that loss_terms is not empty."""
        if not self.loss_terms:
            raise ValueError("loss_terms cannot be empty")

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "return_breakdown": self.return_breakdown,
            "name": self.name,
            "enabled": self.enabled,
            "num_terms": len(self.loss_terms),
        }

    def get_term_names(self) -> list[str]:
        """Get list of term names."""
        return [
            term.get("name", getattr(term.get("criterion"), "name", f"term_{i}"))
            for i, term in enumerate(self.loss_terms)
        ]


class MultiRegressionLoss(BaseCriterion):
    """Multi-Regression Loss for handling multiple scalar regression targets.

    This class can handle multiple regression loss terms with configurable weights
    and enabling/disabling per term. Reuses existing criterion classes for modularity.

    Args:
        config: MultiRegressionLossConfig containing loss terms and settings
    """

    def __init__(
        self,
        return_breakdown: bool = True,
        loss_terms: list[dict[str, Any]] | None = None,
        reduction: ReductionType = ReductionType.MEAN,
        device: torch.device | None = None,
        name: str = "multi_regression_loss",
        log_loss: bool = False,
        enabled: bool = True,
    ):
        """Initialize Multi Regression Loss.

        Args:
            return_breakdown: Whether to return detailed loss breakdown.
            loss_terms: List of loss term configurations.
            reduction: Reduction method for the loss.
            device: Device to compute the loss on.
            name: Name for logging purposes.
            log_loss: Whether to log loss values.
            enabled: Whether the criterion is enabled.
        """
        super().__init__(
            reduction=reduction, device=device, name=name, log_loss=log_loss
        )
        self.enabled = enabled

        # Store additional fields specific to this criterion
        self.return_breakdown = return_breakdown
        # Type narrowing for Pyright
        # (e.g. from Hydra ListConfig/DictConfig).
        raw_terms = loss_terms or []
        self.loss_terms: list[dict[str, Any]] = (
            [dict(t) for t in raw_terms] if raw_terms else []
        )

    def compute_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, float | bool]]:
        """Compute the multi-regression loss, bypassing BaseCriterion's
        tensor-only validation.

        This override skips _validate_inputs since this criterion expects
        dict inputs (model_outputs and batch) rather than tensor arguments.

        Args:
            model_outputs: Dictionary containing processor-managed predictions/
                extras. Trainers now pass `processed["extras"]` alongside scalar
                tensors from `processed["predictions"]`.
            batch: Dictionary containing target values
            **kwargs: Additional keyword arguments

        Returns:
            If return_breakdown is False: Total loss tensor
            If return_breakdown is True: Dictionary with total loss and individual terms
        """
        # Directly call forward with dict inputs, bypassing tensor validation
        result = self.forward(model_outputs, batch)

        if isinstance(result, torch.Tensor):
            result = self._apply_reduction(result)
            if self.reduction_enum != ReductionType.NONE:
                loss_value = result.item()
                self._update_stats(loss_value)
        elif isinstance(result, dict) and "total_loss" in result:
            if isinstance(result["total_loss"], torch.Tensor):
                result["total_loss"] = self._apply_reduction(result["total_loss"])
                if self.reduction_enum != ReductionType.NONE:
                    loss_value = result["total_loss"].item()
                    self._update_stats(loss_value)

        return result

    def forward(
        self, model_outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, float | bool]]:
        """Compute the multi-regression loss.

        Args:
            model_outputs: Dictionary containing processor-managed predictions/
                extras. Trainers now pass `processed["extras"]` alongside scalar
                tensors from `processed["predictions"]`.
            batch: Dictionary containing target values

        Returns:
            If return_breakdown is False: Total loss tensor
            If return_breakdown is True: Dictionary with total loss and individual terms
        """
        if not self.enabled:
            device = (
                next(iter(model_outputs.values())).device
                if model_outputs
                else torch.device("cpu")
            )
            if self.return_breakdown:
                return {"total_loss": torch.tensor(0.0, device=device), "breakdown": {}}
            else:
                return torch.tensor(0.0, device=device)

        device = next(iter(model_outputs.values())).device
        total_loss = torch.tensor(0.0, device=device)
        loss_breakdown = {}

        for term in self.loss_terms:
            if not term["enabled"]:
                continue

            if term["prediction_key"] not in model_outputs:
                raise KeyError(
                    f"Prediction key '{
                        term['prediction_key']
                    }' not found in model_outputs"
                )
            if term["target_key"] not in batch:
                raise KeyError(f"Target key '{term['target_key']}' not found in batch")

            prediction = model_outputs[term["prediction_key"]]
            target = batch[term["target_key"]]

            # Use the existing criterion directly
            loss = term["criterion"](prediction, target)
            weighted_loss = term["weight"] * loss

            total_loss = total_loss + weighted_loss

            # Store breakdown if requested
            if self.return_breakdown:
                loss_breakdown[term["name"]] = {
                    "raw_loss": float(loss.item()),
                    "weighted_loss": float(weighted_loss.item()),
                    "weight": float(term["weight"]),
                    "enabled": bool(term["enabled"]),
                }

        if self.return_breakdown:
            loss_breakdown["total_loss"] = float(total_loss.item())
            return {"total_loss": total_loss, "breakdown": loss_breakdown}
        else:
            return total_loss

    def get_enabled_terms(self) -> list[str]:
        """Get list of enabled loss term names."""
        return [term["name"] for term in self.loss_terms if term["enabled"]]

    def get_all_terms(self) -> list[str]:
        """Get list of all loss term names (enabled and disabled)."""
        return [term["name"] for term in self.loss_terms]

    def enable_term(self, term_name: str):
        """Enable a specific loss term by name."""
        for term in self.loss_terms:
            if term["name"] == term_name:
                term["enabled"] = True
                return
        raise ValueError(f"Loss term '{term_name}' not found")

    def disable_term(self, term_name: str):
        """Disable a specific loss term by name."""
        for term in self.loss_terms:
            if term["name"] == term_name:
                term["enabled"] = False
                return
        raise ValueError(f"Loss term '{term_name}' not found")

    def set_term_weight(self, term_name: str, weight: float):
        """Set the weight for a specific loss term by name."""
        for term in self.loss_terms:
            if term["name"] == term_name:
                term["weight"] = weight
                return
        raise ValueError(f"Loss term '{term_name}' not found")

    def get_term_weight(self, term_name: str) -> float:
        """Get the weight for a specific loss term by name."""
        for term in self.loss_terms:
            if term["name"] == term_name:
                return term["weight"]
        raise ValueError(f"Loss term '{term_name}' not found")


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="criterion", name="base_multi_regression_loss", node=MultiRegressionLossConfig
)
