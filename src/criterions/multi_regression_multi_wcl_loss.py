"""Multi-Regression Multi-WCL Loss for combining regression and WCL losses.

This module provides a composite loss implementation that combines
multiple regression losses (L1, MSE, etc.) with multiple weighted
contrastive losses (WCL) in a single, configurable criterion.

Demographic Field Usage:
    WCL terms using demographic data must reference individual field names:
    - age_raw: Age values for age-based WCL
    - gender_raw: Gender values for gender-based WCL
    - height_raw, weight_raw, bmi_raw: Other demographic fields

Configuration:
    This criterion should be configured via YAML files using Hydra's instantiation.
    See src/conf/criterion/multi_wcl_loss.yaml for examples of how to configure
    individual WeightedContrastiveLoss terms with their respective parameters.

    Hydra directly instantiates WeightedContrastiveLoss instances using the
    WeightedContrastiveLossConfig dataclass, which includes all necessary parameters
    (temperature, threshold, embedding_key, weight_key, etc.).
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType
from src.criterions.multi_regression_loss import MultiRegressionLoss
from src.criterions.multi_regression_loss import MultiRegressionLossConfig
from src.criterions.multi_weighted_contrastive_loss import MultiWCLLossConfig
from src.criterions.multi_weighted_contrastive_loss import MultiWeightedContrastiveLoss


@dataclass
class MultiRegressionMultiWCLLossConfig(CriterionBaseConfig):
    """Pure data container for Multi Regression Multi WCL configuration."""

    _target_: str = (
        "src.criterions.multi_regression_multi_wcl_loss.MultiRegressionMultiWCLLoss"
    )
    regression_weight: float = 1.0
    wcl_weight: float = 0.2
    return_breakdown: bool = True
    regression_config: MultiRegressionLossConfig | None = None
    wcl_config: MultiWCLLossConfig | None = None

    def __post_init__(self):
        """Validate that weights are non-negative."""
        if self.regression_weight < 0:
            raise ValueError(
                f"regression_weight must be non-negative, got {self.regression_weight}"
            )
        if self.wcl_weight < 0:
            raise ValueError(f"wcl_weight must be non-negative, got {self.wcl_weight}")

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "regression_weight": self.regression_weight,
            "wcl_weight": self.wcl_weight,
            "return_breakdown": self.return_breakdown,
            "name": self.name,
            "enabled": self.enabled,
            "regression_config": (
                self.regression_config.to_dict() if self.regression_config else None
            ),
            "wcl_config": self.wcl_config.to_dict() if self.wcl_config else None,
        }


class MultiRegressionMultiWCLLoss(BaseCriterion):
    """Multi-Regression Multi-WCL Loss for combining multiple regression and WCL losses.

    This class combines multiple regression losses (L1, MSE, etc.) with multiple
    weighted contrastive losses (WCL) in a single, configurable criterion.

    Args:
        config: MultiRegressionMultiWCLLossConfig containing both loss configurations
    """

    def __init__(
        self,
        regression_weight: float = 1.0,
        wcl_weight: float = 0.2,
        return_breakdown: bool = True,
        regression_config: MultiRegressionLoss | None = None,
        wcl_config: MultiWeightedContrastiveLoss | None = None,
        reduction: ReductionType = ReductionType.MEAN,
        device: torch.device | None = None,
        name: str = "multi_regression_multi_wcl_loss",
        log_loss: bool = False,
        enabled: bool = True,
    ):
        """Initialize Multi Regression Multi WCL Loss.

        Args:
            regression_weight: Weight for regression loss component.
            wcl_weight: Weight for WCL loss component.
            return_breakdown: Whether to return detailed loss breakdown.
            regression_config: Instantiated MultiRegressionLoss object.
            wcl_config: Instantiated MultiWeightedContrastiveLoss object.
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
        self.regression_weight = regression_weight
        self.wcl_weight = wcl_weight
        self.return_breakdown = return_breakdown

        # Store the instantiated loss objects (Hydra will pass these as
        # already-instantiated)
        self.regression_loss = regression_config
        self.wcl_loss = wcl_config

    def compute_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor | dict[str, Any]:
        """Compute multi-regression multi-WCL loss; skips tensor-only validation.

        This override skips _validate_inputs since this criterion
        expects dict inputs (model_outputs and batch) rather than tensor
        arguments.

        Args:
            model_outputs: Dictionary containing model outputs and embeddings
                Keys: 'ecg_embeddings', 'ppg_embeddings', 'text_embeddings',
                predictions, etc.
            batch: Dictionary containing target values and demographic data
                Keys: 'SBP', 'DBP', 'age_raw', 'gender_raw', 'height_raw',
                'weight_raw', 'bmi_raw', etc. Note: Individual demographic
                field names (age_raw, gender_raw, etc.) are the only
                supported format.
            **kwargs: Additional keyword arguments

        Returns:
            If return_breakdown is False: Total loss tensor
            If return_breakdown is True: Dictionary with total loss and
                breakdown
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
    ) -> torch.Tensor | dict[str, Any]:
        """Compute the multi-regression multi-WCL loss.

        Args:
            model_outputs: Dictionary containing model outputs and embeddings
                Keys: 'ecg_embeddings', 'ppg_embeddings', 'text_embeddings',
                predictions, etc.
            batch: Dictionary containing target values and demographic data
                Keys: 'SBP', 'DBP', 'age_raw', 'gender_raw', 'height_raw',
                'weight_raw', 'bmi_raw', etc. Note: Individual demographic
                field names (age_raw, gender_raw, etc.) are the only
                supported format.

        Returns:
            If return_breakdown is False: Total loss tensor
            If return_breakdown is True: Dictionary with total loss and
                breakdown
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

        total_loss = torch.tensor(0.0, device=next(iter(model_outputs.values())).device)
        loss_breakdown = {}

        if self.regression_loss is None:
            regression_loss = torch.tensor(0.0, device=total_loss.device)
            # Skip breakdown for None regression_loss
        else:
            regression_result = self.regression_loss(model_outputs, batch)
            if isinstance(regression_result, dict):
                regression_loss = regression_result["total_loss"]
                if self.return_breakdown:
                    loss_breakdown["regression"] = regression_result["breakdown"]
            elif isinstance(regression_result, tuple):
                regression_loss = regression_result[0]
            else:
                regression_loss = regression_result

        weighted_regression_loss = self.regression_weight * regression_loss
        total_loss = total_loss + weighted_regression_loss

        if self.wcl_loss is not None:
            wcl_result = self.wcl_loss(model_outputs, batch)
            if isinstance(wcl_result, dict):
                wcl_loss = wcl_result["total_loss"]
                if self.return_breakdown:
                    loss_breakdown["wcl"] = wcl_result["breakdown"]
            elif isinstance(wcl_result, tuple):
                wcl_loss = wcl_result[0]
            else:
                wcl_loss = wcl_result

            weighted_wcl_loss = self.wcl_weight * wcl_loss
            total_loss = total_loss + weighted_wcl_loss
        else:
            wcl_loss = torch.tensor(0.0, device=total_loss.device)
            weighted_wcl_loss = torch.tensor(0.0, device=total_loss.device)

        if self.return_breakdown:
            loss_breakdown["total"] = {
                "regression_loss": regression_loss.item(),
                "wcl_loss": wcl_loss.item(),
                "weighted_regression_loss": weighted_regression_loss.item(),
                "weighted_wcl_loss": weighted_wcl_loss.item(),
                "total_loss": total_loss.item(),
            }
            return {"total_loss": total_loss, "breakdown": loss_breakdown}
        else:
            return total_loss

    def get_regression_terms(self) -> list[str]:
        """Get list of regression loss term names."""
        if self.regression_loss is None:
            return []
        return self.regression_loss.get_all_terms()

    def get_wcl_terms(self) -> list[str]:
        """Get list of WCL loss term names."""
        if self.wcl_loss is not None:
            return self.wcl_loss.get_all_terms()
        return []

    def enable_regression_term(self, term_name: str):
        """Enable a specific regression loss term by name."""
        if self.regression_loss is None:
            raise ValueError(
                "Cannot enable regression term: regression_loss is not configured"
            )
        self.regression_loss.enable_term(term_name)

    def disable_regression_term(self, term_name: str):
        """Disable a specific regression loss term by name."""
        if self.regression_loss is None:
            raise ValueError(
                "Cannot disable regression term: regression_loss is not configured"
            )
        self.regression_loss.disable_term(term_name)

    def enable_wcl_term(self, term_name: str):
        """Enable a specific WCL loss term by name."""
        if self.wcl_loss is not None:
            self.wcl_loss.enable_term(term_name)

    def disable_wcl_term(self, term_name: str):
        """Disable a specific WCL loss term by name."""
        if self.wcl_loss is not None:
            self.wcl_loss.disable_term(term_name)

    def set_regression_weight(self, weight: float):
        """Set the weight for regression losses."""
        if weight < 0:
            raise ValueError(f"regression_weight must be non-negative, got {weight}")
        self.regression_weight = weight

    def set_wcl_weight(self, weight: float):
        """Set the weight for WCL losses."""
        if weight < 0:
            raise ValueError(f"wcl_weight must be non-negative, got {weight}")
        self.wcl_weight = weight


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="criterion",
    name="base_multi_regression_multi_wcl_loss",
    node=MultiRegressionMultiWCLLossConfig,
)
