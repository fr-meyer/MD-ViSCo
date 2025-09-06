"""
Weighted Contrastive Loss criterion for learning discriminative embeddings.

This module provides a weighted contrastive loss implementation that considers both
embedding similarities and target value similarities. It helps learn embeddings that
are similar for similar target values while being different for dissimilar ones.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
from dataclasses import dataclass
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType


@dataclass
class WeightedContrastiveLossConfig(CriterionBaseConfig):
    """Pure data container for Weighted Contrastive Loss configuration"""
    _target_: str = "src.criterions.weighted_contrastive_loss.WeightedContrastiveLoss"
    name: str = "weighted_contrastive_loss"
    temperature: float = 4.0
    temperature_embeddings: float = 1.0
    temperature_weight: float = 1.0
    threshold: float = 0.0
    scale_factor: float = 1.0
    reduction: ReductionType = ReductionType.MEAN
    embedding_key: Optional[str] = None
    weight_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_embeddings <= 0:
            raise ValueError(f"temperature_embeddings must be positive, got {self.temperature_embeddings}")
        if self.temperature_weight <= 0:
            raise ValueError(f"temperature_weight must be positive, got {self.temperature_weight}")
        if self.threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(f"reduction must be a ReductionType enum, got {type(self.reduction)}")
    
    @classmethod
    def for_regression(cls, temperature: float = 4.0, scale_factor: float = 1.0) -> 'WeightedContrastiveLossConfig':
        """Create a WCL config optimized for regression tasks (e.g., BP prediction)."""
        return cls(
            temperature=temperature,
            temperature_embeddings=temperature,
            temperature_weight=4.0,
            threshold=0.0235,
            scale_factor=scale_factor,
            reduction=ReductionType.MEAN
        )
    
    @classmethod
    def for_binary_classification(cls, temperature: float = 4.0, scale_factor: float = 1.0) -> 'WeightedContrastiveLossConfig':
        """Create a WCL config optimized for binary classification tasks (e.g., gender)."""
        return cls(
            temperature=temperature,
            temperature_embeddings=temperature,
            temperature_weight=1.0,
            threshold=1.0,
            scale_factor=scale_factor,
            reduction=ReductionType.MEAN
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'temperature': self.temperature,
            'temperature_embeddings': self.temperature_embeddings,
            'temperature_weight': self.temperature_weight,
            'threshold': self.threshold,
            'scale_factor': self.scale_factor,
            'reduction': self.reduction.value,
            'device': self.device,
            'name': self.name,
            'log_loss': self.log_loss
        }
    
    def __str__(self) -> str:
        """String representation of the config."""
        return (f"WeightedContrastiveLossConfig("
                f"temperature={self.temperature}, "
                f"temperature_embeddings={self.temperature_embeddings}, "
                f"temperature_weight={self.temperature_weight}, "
                f"threshold={self.threshold}, "
                f"scale_factor={self.scale_factor}, "
                f"reduction='{self.reduction.value}', "
                f"enabled={self.enabled})")


class WeightedContrastiveLoss(BaseCriterion):
    """
    Weighted Contrastive Loss for learning discriminative embeddings.
    
    This criterion implements a weighted contrastive loss that considers both embedding
    similarities and target value similarities. It helps learn embeddings that are
    similar for similar target values while being different for dissimilar ones.
    
    The loss is computed as:
    1. Calculate weight similarity matrix using exponential of negative absolute differences
    2. Apply threshold to filter out low similarities
    3. Calculate embedding similarities using dot product
    4. Compute log probabilities of embedding similarities
    5. Weight the log probabilities by the weight similarities
    6. Normalize by the sum of weight similarities
    7. Apply scaling factor to the final loss
    
    Args:
        temperature (float): Main temperature parameter for embedding similarities.
            Higher values make the distribution more uniform. Default: 4.0
        temperature_embeddings (float, optional): Temperature parameter for embedding similarities.
            If None, uses the main temperature parameter. Default: None
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
        >>> config = WeightedContrastiveLossConfig.for_regression(temperature=4.0, scale_factor=1e-3)
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
        device: Optional[torch.device] = None,
        name: str = "weighted_contrastive_loss",
        log_loss: bool = False,
        enabled: bool = True,
        embedding_key: Optional[str] = None,
        weight_key: Optional[str] = None,
        *args,
        **kwargs
    ):
        # Call parent constructor with explicit parameters
        super().__init__(
            reduction=reduction,
            device=device,
            name=name,
            log_loss=log_loss,
            *args,
            **kwargs
        )
        
        # Store additional fields specific to this criterion
        self.temperature = temperature
        self.temperature_embeddings = temperature_embeddings
        self.temperature_weight = temperature_weight
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.embedding_key = embedding_key
        self.weight_key = weight_key
    
    def _compute_loss(
        self,
        embeddings: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the weighted contrastive loss.
        
        Args:
            embeddings (torch.Tensor): Feature embeddings of shape (batch_size, embedding_dim)
            weights (torch.Tensor): Target values of shape (batch_size, 1) used for weighting
            
        Returns:
            torch.Tensor: Weighted contrastive loss value
            
        Raises:
            ValueError: If input shapes are invalid
        """
        # if self.embedding_key is not None:
        #     embeddings = embeddings[self.embedding_key]
        # if self.weight_key is not None:
        #     weights = weights[self.weight_key]
        # Validate input shapes
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2D tensor, got shape {embeddings.shape}")
        if weights.dim() != 2 or weights.shape[-1] != 1:
            raise ValueError(f"weights must be 2D tensor with shape (batch_size, 1), got shape {weights.shape}")
        if embeddings.size(0) != weights.size(0):
            raise ValueError(f"embeddings and weights must have same batch size. "
                           f"Got {embeddings.size(0)} and {weights.size(0)}")
        
        batch_size = embeddings.size(0)
        
        # Calculate weight similarity matrix using exponential of negative absolute differences
        # Shape: (batch_size, batch_size)
        weight_similarity = torch.exp(-(torch.abs(weights - weights.T) / self.temperature_weight))
        
        # Zero out similarities below threshold to focus on more similar pairs
        weight_similarity = torch.where(
            weight_similarity >= self.threshold,
            weight_similarity,
            torch.zeros_like(weight_similarity)
        )

        # Calculate normalization term for weight similarities
        # Shape: (batch_size, 1)
        weight_similarity_norm = weight_similarity.sum(dim=-1, keepdim=True)

        # Calculate embedding similarities using dot product and temperature scaling
        # Shape: (batch_size, batch_size)
        emb_similarity = torch.matmul(embeddings, embeddings.T) / self.temperature_embeddings

        # Calculate log probabilities of embedding similarities
        # Shape: (batch_size, batch_size)
        log_prob = F.log_softmax(emb_similarity, dim=-1)

        # Compute normalized weighted contrastive loss
        # 1. Multiply log probabilities by weight similarities
        # 2. Sum over the batch dimension
        # 3. Normalize by the sum of weight similarities
        # 4. Add small epsilon to avoid division by zero
        loss = -torch.sum(weight_similarity * log_prob, dim=-1) / (weight_similarity_norm.squeeze(-1) + 1e-8)

        # Apply scaling factor to the loss
        loss = loss * self.scale_factor

        return loss
    
    def forward(
        self,
        embeddings: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the weighted contrastive loss.
        
        Args:
            embeddings (torch.Tensor): Feature embeddings of shape (batch_size, embedding_dim)
            weights (torch.Tensor): Target values of shape (batch_size, 1) used for weighting
            
        Returns:
            torch.Tensor: Weighted contrastive loss value (before reduction)
        """
        return self._compute_loss(embeddings, weights)
    
    def extra_repr(self) -> str:
        """Return string representation of the criterion's parameters."""
        return (f"temperature={self.temperature}, "
                f"temperature_embeddings={self.temperature_embeddings}, "
                f"temperature_weight={self.temperature_weight}, "
                f"threshold={self.threshold}, "
                f"scale_factor={self.scale_factor}, "
                f"reduction={self.reduction.value}")

# Lazy registration
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_weighted_contrastive_loss", node=WeightedContrastiveLossConfig)