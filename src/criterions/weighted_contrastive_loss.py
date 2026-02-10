"""Weighted Contrastive Loss criterion for learning discriminative embeddings.

This module provides a weighted contrastive loss implementation that considers
both embedding similarities and target value similarities. It helps learn
embeddings that are similar for similar target values while being different
for dissimilar ones.
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
class WeightedContrastiveLossConfig(CriterionBaseConfig):
    """Pure data container for Weighted Contrastive Loss configuration."""

    _target_: str = "src.criterions.weighted_contrastive_loss.WeightedContrastiveLoss"
    name: str = "weighted_contrastive_loss"
    temperature: float = 4.0
    temperature_embeddings: float = 1.0
    temperature_weight: float = 1.0
    threshold: float = 0.0
    scale_factor: float = 1.0
    reduction: ReductionType = ReductionType.MEAN
    embedding_key: str | None = None
    weight_key: str | None = None

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_embeddings <= 0:
            raise ValueError(
                f"temperature_embeddings must be positive, got {
                    self.temperature_embeddings
                }"
            )
        if self.temperature_weight <= 0:
            raise ValueError(
                f"temperature_weight must be positive, got {self.temperature_weight}"
            )
        if self.threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(
                f"reduction must be a ReductionType enum, got {type(self.reduction)}"
            )

    @classmethod
    def for_regression(
        cls, temperature: float = 4.0, scale_factor: float = 1.0
    ) -> "WeightedContrastiveLossConfig":
        """Create a WCL config optimized for regression tasks (e.g., BP prediction)."""
        return cls(
            temperature=temperature,
            temperature_embeddings=temperature,
            temperature_weight=4.0,
            threshold=0.0235,
            scale_factor=scale_factor,
            reduction=ReductionType.MEAN,
        )

    @classmethod
    def for_binary_classification(
        cls, temperature: float = 4.0, scale_factor: float = 1.0
    ) -> "WeightedContrastiveLossConfig":
        """Create a WCL config for binary classification (e.g., gender)."""
        return cls(
            temperature=temperature,
            temperature_embeddings=temperature,
            temperature_weight=1.0,
            threshold=1.0,
            scale_factor=scale_factor,
            reduction=ReductionType.MEAN,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "temperature": self.temperature,
            "temperature_embeddings": self.temperature_embeddings,
            "temperature_weight": self.temperature_weight,
            "threshold": self.threshold,
            "scale_factor": self.scale_factor,
            "reduction": self.reduction.value,
            "device": self.device,
            "name": self.name,
            "log_loss": self.log_loss,
        }

    def __str__(self) -> str:
        """Return string representation of the config."""
        return (
            f"WeightedContrastiveLossConfig("
            f"temperature={self.temperature}, "
            f"temperature_embeddings={self.temperature_embeddings}, "
            f"temperature_weight={self.temperature_weight}, "
            f"threshold={self.threshold}, "
            f"scale_factor={self.scale_factor}, "
            f"reduction='{self.reduction.value}', "
            f"enabled={self.enabled})"
        )


class WeightedContrastiveLoss(BaseCriterion):
    """Weighted Contrastive Loss for learning discriminative embeddings.

    This criterion implements a weighted contrastive loss that considers both embedding
    similarities and target value similarities. It helps learn embeddings that are
    similar for similar target values while being different for dissimilar ones.

    The loss is computed as:
    1. Calculate weight similarity matrix using exponential of negative
       absolute differences
    2. Apply threshold to filter out low similarities
    3. Calculate embedding similarities using dot product
    4. Compute log probabilities of embedding similarities
    5. Weight the log probabilities by the weight similarities
    6. Normalize by the sum of weight similarities
    7. Apply scaling factor to the final loss

    Args:
        temperature (float): Main temperature parameter for embedding similarities.
            Higher values make the distribution more uniform. Default: 4.0
        temperature_embeddings (float, optional): Temperature for embedding
            similarities. If None, uses the main temperature. Default: None
        temperature_weight (float): Temperature parameter for weight similarities.
            Higher values make the distribution more uniform. Default: 1.0
        threshold (float): Minimum similarity threshold for weight matrix.
            Values below this are set to zero. Default: 0.0
        scale_factor (float): Scaling factor applied to the final loss value.
            Useful for balancing different loss components. Default: 1.0
        reduction (str): Reduction method for the loss ('none', 'mean', 'sum').
            Default: 'mean'
        device (torch.device, optional): Device to compute the loss on.
            If None, will use the device of the input tensors.
        name (str, optional): Name of the criterion for logging purposes.
            If None, will use the class name.
        log_loss (bool): Whether to log loss values during computation.
            Default: False

    Demographic Field Names:
        When using WCL with demographic data, use the individual field names:
        - age_raw: Age values (scalar tensor per sample)
        - gender_raw: Gender values (0/1 binary encoding)
        - height_raw: Height values
        - weight_raw: Weight values
        - bmi_raw: BMI values

        These fields are provided by the collate function when
        demographics are available.
        Individual field names are the only supported format for demographic data.

        Example with demographics:
            >>> # Configure WCL for gender-based contrastive learning
            >>> config = WeightedContrastiveLossConfig.for_binary_classification(
            ...     temperature=4.0,
            ...     scale_factor=1e-2
            ... )
            >>> criterion = WeightedContrastiveLoss(
            ...     embedding_key="text_embeddings",
            ...     weight_key="gender_raw",  # Use individual field
            ...     **config.to_dict()
            ... )

    Example:
        >>> # Using individual parameters
        >>> criterion = WeightedContrastiveLoss(
        ...     temperature=4.0,
        ...     temperature_weight=4.0,
        ...     threshold=0.0235,
        ...     scale_factor=1e-3  # Scale down the loss
        ... )
        >>>
        >>> # Using config
        >>> config = WeightedContrastiveLossConfig.for_regression(
        ...     temperature=4.0, scale_factor=1e-3
        ... )
        >>> criterion = WeightedContrastiveLoss(**config.to_dict())
        >>>
        >>> embeddings = torch.randn(32, 256)  # batch_size=32, embedding_dim=256
        >>> weights = torch.randn(32, 1)  # batch_size=32, 1
        >>> loss = criterion(embeddings, weights)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        temperature_embeddings: float = 1.0,
        temperature_weight: float = 1.0,
        threshold: float = 0.0,
        scale_factor: float = 1.0,
        reduction: ReductionType = ReductionType.MEAN,
        device: torch.device | None = None,
        name: str = "weighted_contrastive_loss",
        log_loss: bool = False,
        enabled: bool = True,
        embedding_key: (
            str | None
            # Key to extract embeddings from batch_dict
            # (e.g., "text_embeddings", "ecg_embeddings")
        ) = None,
        weight_key: (
            str | None
            # Key to extract weights from batch_dict (e.g., "age_raw", "gender_raw")
        ) = None,
        *args,
        **kwargs,
    ):
        """Initialize WeightedContrastiveLoss.

        Args:
            embedding_key (str, optional): Key to extract embeddings from
                batch_dict when used with MultiWeightedContrastiveLoss. If
                None, embeddings must be passed directly to forward().
                Example: "text_embeddings", "ecg_embeddings"
            weight_key (str, optional): Key to extract weight values from
                batch_dict when used with MultiWeightedContrastiveLoss. If
                None, weights must be passed directly to forward(). For
                demographics, use individual field names like "age_raw",
                "gender_raw"

        Usage Patterns:
            - Standalone: Pass tensors directly to forward(embeddings, weights)
            - With MultiWeightedContrastiveLoss: Set embedding_key and
              weight_key, tensors are extracted automatically from batch_dict
        """
        # Filter out parameters that are already passed explicitly to avoid duplicates
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("reduction", "device", "name", "log_loss")
        }
        super().__init__(
            *args,
            reduction=reduction,
            device=device,
            name=name,
            log_loss=log_loss,
            **filtered_kwargs,
        )
        self.enabled = enabled

        # Store additional fields specific to this criterion
        self.temperature = temperature
        self.temperature_embeddings = temperature_embeddings
        self.temperature_weight = temperature_weight
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.embedding_key = embedding_key
        self.weight_key = weight_key

    def _compute_loss(
        self, embeddings: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute the weighted contrastive loss.

        Args:
            embeddings (torch.Tensor): Feature embeddings of shape
            (batch_size, embedding_dim)
            weights (torch.Tensor): Target values of shape (batch_size, 1) or
                (batch_size,) used for weighting. Accepts both 1-D [B] and
                2-D [B, 1] tensors for robustness.

        Returns:
            torch.Tensor: Weighted contrastive loss value

        Raises:
            ValueError: If input shapes are invalid
        """
        # NOTE: embedding_key and weight_key are used by
        # MultiWeightedContrastiveLoss to extract tensors from batch_dict
        # before calling this method.
        # Direct extraction here is not needed since tensors are already extracted.
        # These parameters are stored for configuration/logging purposes.

        # Reshape 1-D weights to 2-D for robustness (accepts both [B] and [B, 1])
        if weights.dim() == 1:
            weights = weights.unsqueeze(-1)  # [B] -> [B, 1]

        # Validate input shapes
        if embeddings.dim() != 2:
            raise ValueError(
                f"embeddings must be 2D tensor, got shape {embeddings.shape}"
            )
        if weights.dim() != 2 or weights.shape[-1] != 1:
            raise ValueError(
                f"weights must be 2D tensor with shape (batch_size, 1), got shape {
                    weights.shape
                }"
            )
        if embeddings.size(0) != weights.size(0):
            raise ValueError(
                f"embeddings and weights must have same batch size. "
                f"Got {embeddings.size(0)} and {weights.size(0)}"
            )

        # Shape: (batch_size, batch_size)
        weight_similarity = torch.exp(
            -(torch.abs(weights - weights.T) / self.temperature_weight)
        )

        # Zero out similarities below threshold to focus on more similar pairs
        weight_similarity = torch.where(
            weight_similarity >= self.threshold,
            weight_similarity,
            torch.zeros_like(weight_similarity),
        )

        # Shape: (batch_size, 1)
        weight_similarity_norm = weight_similarity.sum(dim=-1, keepdim=True)

        # Shape: (batch_size, batch_size)
        emb_similarity = (
            torch.matmul(embeddings, embeddings.T) / self.temperature_embeddings
        )

        # Shape: (batch_size, batch_size)
        log_prob = F.log_softmax(emb_similarity, dim=-1)

        # Epsilon avoids division by zero when weight_similarity_norm is zero
        loss = -torch.sum(weight_similarity * log_prob, dim=-1) / (
            weight_similarity_norm.squeeze(-1) + 1e-8
        )

        loss = loss * self.scale_factor

        return loss

    def forward(self, embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass of the weighted contrastive loss.

        This method expects direct tensor inputs. When using with
        MultiWeightedContrastiveLoss, the embedding_key and weight_key
        parameters are used to extract tensors from batch_dict before
        calling this method.

        Args:
            embeddings (torch.Tensor): Feature embeddings of shape
                (batch_size, embedding_dim). Extracted from
                batch_dict[embedding_key] by MultiWeightedContrastiveLoss
            weights (torch.Tensor): Target values of shape (batch_size, 1) or
                (batch_size,) used for weighting. Extracted from
                batch_dict[weight_key] by MultiWeightedContrastiveLoss.
                Accepts both 1-D [B] and 2-D [B, 1] tensors for robustness.
                For demographics, use individual fields like age_raw,
                gender_raw

        Returns:
            torch.Tensor: Weighted contrastive loss value (before reduction)
        """
        if not self.enabled:
            # Return zero loss tensor on the same device as embeddings
            device = (
                embeddings.device
                if isinstance(embeddings, torch.Tensor)
                else torch.device("cpu")
            )
            return torch.tensor(0.0, device=device)
        return self._compute_loss(embeddings, weights)

    def extra_repr(self) -> str:
        """Return string representation of the criterion's parameters."""
        return (
            f"temperature={self.temperature}, "
            f"temperature_embeddings={self.temperature_embeddings}, "
            f"temperature_weight={self.temperature_weight}, "
            f"threshold={self.threshold}, "
            f"scale_factor={self.scale_factor}, "
            f"reduction={self.reduction_enum.value}"
        )


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="criterion",
    name="base_weighted_contrastive_loss",
    node=WeightedContrastiveLossConfig,
)
