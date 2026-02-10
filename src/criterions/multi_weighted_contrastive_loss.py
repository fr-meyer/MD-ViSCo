"""Multi-Weighted Contrastive Loss criterion for handling multiple WCL targets.

This module provides a configurable multi-WCL implementation that can handle
multiple weighted contrastive loss terms with configurable weights and
enabling/disabling per term. Reuses existing WeightedContrastiveLoss class
for modularity.
"""

# Standard library imports
from dataclasses import dataclass

import torch

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Local imports
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType
from src.criterions.weighted_contrastive_loss import WeightedContrastiveLoss
from src.criterions.weighted_contrastive_loss import WeightedContrastiveLossConfig


@dataclass
class MultiWCLLossConfig(CriterionBaseConfig):
    """Pure data container for Multi Weighted Contrastive Loss configuration.

    Note:
        When configuring loss_terms for demographic-based WCL, use individual
        field names (age_raw, gender_raw, etc.) instead of the legacy
        combined demographics_raw field. See src/conf/criterion/
        multi_wcl_loss.yaml for configuration examples.
    """

    _target_: str = (
        "src.criterions.multi_weighted_contrastive_loss.MultiWeightedContrastiveLoss"
    )
    name: str = "multi_weighted_contrastive_loss"
    reduction: ReductionType = ReductionType.MEAN
    loss_terms: list[WeightedContrastiveLossConfig] = MISSING

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "name": self.name,
            "reduction": self.reduction.value,
            "loss_terms": [
                t.to_dict() if hasattr(t, "to_dict") else {} for t in self.loss_terms
            ],
        }


class MultiWeightedContrastiveLoss(BaseCriterion):
    """Multi-Weighted Contrastive Loss for handling multiple WCL targets.

    This class can handle multiple WeightedContrastiveLoss instances.

        Demographic Field Extraction:
        This class automatically extracts tensors from batch_dict using the
        embedding_key and weight_key configured for each loss term. With the
        new demographic field structure, use individual field names:

        - age_raw: For age-based WCL
        - gender_raw: For gender-based WCL
        - height_raw, weight_raw, bmi_raw: For other demographic-based WCL

        Example configuration:
            >>> age_wcl = WeightedContrastiveLoss(
            ...     embedding_key="text_embeddings",
            ...     weight_key="age_raw",  # Individual field
            ...     temperature=4.0
            ... )
            >>> gender_wcl = WeightedContrastiveLoss(
            ...     embedding_key="text_embeddings",
            ...     weight_key="gender_raw",  # Individual field
            ...     temperature=4.0
            ... )
            >>> multi_wcl = MultiWeightedContrastiveLoss(
            ...     loss_terms=[age_wcl, gender_wcl]
            ... )

        The forward() method receives batch_dict as **kwargs and extracts:
        - embeddings = kwargs[term.embedding_key]
        - weights = kwargs[term.weight_key]
    """

    def __init__(
        self, loss_terms: list[WeightedContrastiveLoss] = MISSING, *args, **kwargs
    ):
        """Initialize Multi Weighted Contrastive Loss.

        Args:
            loss_terms: List of WeightedContrastiveLoss instances.
            *args: Variable positional arguments passed to parent.
            **kwargs: Variable keyword arguments passed to parent.
        """
        super().__init__(*args, **kwargs)

        # Instantiate WeightedContrastiveLoss instances using Hydra
        self.loss_terms = loss_terms

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute the multi-weighted contrastive loss.

        Args:
            *args: Variable length argument list (unused)
            **kwargs: Dictionary containing all embeddings and weights with
                their respective keys. This is typically the batch_dict from
                the data loader, which includes:
                - Embedding tensors (e.g., 'text_embeddings',
                  'ecg_embeddings', 'ppg_embeddings')
                - Weight tensors for WCL (e.g., 'age_raw', 'gender_raw', 'SBP', 'DBP')
                - Other batch data (waveforms, targets, etc.)

                Each loss term extracts its required tensors using:
                - embeddings = kwargs[term.embedding_key]
                - weights = kwargs[term.weight_key]

                For demographic-based WCL, use individual field names:
                - 'age_raw': Age values [B, 1]
                - 'gender_raw': Gender values [B, 1]
                - 'height_raw', 'weight_raw', 'bmi_raw': Other demographics
                  [B, 1]

        Returns:
            torch.Tensor: Total loss summed across all enabled loss terms
        """
        device = None
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break

        if device is None:
            device = torch.device("cpu")

        total_loss = torch.tensor(0.0, device=device)

        for term in self.loss_terms:
            embedding_key = term.embedding_key
            weight_key = term.weight_key

            # Skip this loss term if either key is not found in kwargs
            if embedding_key is None or embedding_key not in kwargs:
                continue
            if weight_key is None or weight_key not in kwargs:
                continue

            embeddings = kwargs[embedding_key]
            weights = kwargs[weight_key]

            # Skip if either tensor is None or empty
            if embeddings is None or weights is None:
                continue

            # Use the WeightedContrastiveLoss directly
            loss = term(embeddings, weights)
            total_loss = total_loss + loss

        return total_loss


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_multi_wcl_loss", node=MultiWCLLossConfig)
