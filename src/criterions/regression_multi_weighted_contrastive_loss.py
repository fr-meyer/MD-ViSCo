"""
Regression Multi-Weighted Contrastive Loss criterion for joint representation learning and prediction.

This module provides a composite loss function that combines multiple Weighted Contrastive Loss (WCL)
terms with a single regression loss term. It's designed for tasks requiring both discriminative
representation learning and direct numerical prediction.

Key Features:
- Combines multiple WCL terms (for different embeddings) with a single regression loss
- Supports loss breakdown for detailed monitoring and debugging
- Integrates with Hydra configuration system
- Inherits from BaseCriterion for consistent device handling and reduction methods

Use Cases:
- Blood pressure prediction from ECG/PPG signals
- Multi-modal learning with multiple embedding types
- Tasks requiring both representation learning and direct prediction
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict
from omegaconf import MISSING
from .base_criterion import BaseCriterion, CriterionBaseConfig
from .weighted_contrastive_loss import WeightedContrastiveLoss, WeightedContrastiveLossConfig
from .multi_weighted_contrastive_loss import MultiWeightedContrastiveLoss, MultiWCLLossConfig
from .base_criterion import ReductionType


@dataclass
class RegressionMultiWCLLossConfig(CriterionBaseConfig):
    """
    Configuration for Regression Multi Weighted Contrastive Loss.
    
    This configuration combines multiple Weighted Contrastive Loss (WCL) terms
    with a single regression loss term for joint representation learning and
    direct prediction tasks.
    
    Attributes:
        _target_ (str): Hydra target class path for instantiation
        name (str): Name identifier for this loss configuration
        reduction (ReductionType): How to reduce the loss across batch dimension
        wcl_loss_terms (MultiWCLLossConfig): Configuration for multiple WCL terms
        regression_loss_term (CriterionBaseConfig): Configuration for regression loss
        
    Example:
        >>> config = RegressionMultiWCLLossConfig(
        ...     wcl_loss_terms=MultiWCLLossConfig(
        ...         loss_terms=[
        ...             WeightedContrastiveLossConfig(
        ...                 embedding_key="ecg_embeddings",
        ...                 weight_key="SBP",
        ...                 temperature=4.0
        ...             ),
        ...             WeightedContrastiveLossConfig(
        ...                 embedding_key="ppg_embeddings", 
        ...                 weight_key="DBP",
        ...                 temperature=4.0
        ...             )
        ...         ]
        ...     ),
        ...     regression_loss_term=L1LossConfig()
        ... )
    """
    _target_: str = "src.criterions.regression_multi_weighted_contrastive_loss.RegressionMultiWeightedContrastiveLoss"
    name: str = "regression_multi_weighted_contrastive_loss"
    reduction: ReductionType = ReductionType.MEAN
    wcl_loss_terms: MultiWCLLossConfig = MISSING
    regression_loss_term: CriterionBaseConfig = MISSING


class RegressionMultiWeightedContrastiveLoss(BaseCriterion):
    """
    Regression Multi Weighted Contrastive Loss for joint representation learning and prediction.
    
    This criterion combines multiple Weighted Contrastive Loss (WCL) terms with a single
    regression loss term. It's designed for tasks that require both discriminative
    representation learning (via WCL) and direct numerical prediction (via regression).
    
    The loss is computed as:
    total_loss = wcl_loss + regression_loss
    
    Where:
    - wcl_loss: Sum of multiple WCL terms computed on different embeddings
    - regression_loss: Single regression loss computed on predictions vs targets
    
    This is particularly useful for:
    - Blood pressure prediction from ECG/PPG signals
    - Multi-modal learning with multiple embedding types
    - Tasks requiring both representation learning and direct prediction
    
    Args:
        wcl_loss_terms (MultiWeightedContrastiveLoss): Handles multiple WCL terms
        regression_loss_term (BaseCriterion): Single regression loss (L1, MSE, etc.)
        *args: Additional arguments passed to BaseCriterion
        **kwargs: Additional keyword arguments passed to BaseCriterion
        
    Attributes:
        wcl_loss_terms (MultiWeightedContrastiveLoss): Multi-WCL loss component
        regression_loss_term (BaseCriterion): Regression loss component
        
    Example:
        >>> # Create WCL terms for different embeddings
        >>> wcl_terms = MultiWeightedContrastiveLoss([
        ...     WeightedContrastiveLoss(embedding_key="ecg_embeddings", weight_key="SBP"),
        ...     WeightedContrastiveLoss(embedding_key="ppg_embeddings", weight_key="DBP")
        ... ])
        >>> 
        >>> # Create regression loss
        >>> regression_loss = L1Loss()
        >>> 
        >>> # Combine them
        >>> criterion = RegressionMultiWeightedContrastiveLoss(
        ...     wcl_loss_terms=wcl_terms,
        ...     regression_loss_term=regression_loss
        ... )
        >>> 
        >>> # Forward pass
        >>> loss = criterion(
        ...     input=predictions,  # Model predictions
        ...     target=targets,     # Ground truth targets
        ...     ecg_embeddings=ecg_emb,  # ECG embeddings
        ...     ppg_embeddings=ppg_emb,  # PPG embeddings
        ...     SBP=bp_systolic,    # Systolic BP values
        ...     DBP=bp_diastolic    # Diastolic BP values
        ... )
    """

    def __init__(self, wcl_loss_terms: MultiWeightedContrastiveLoss, regression_loss_term: BaseCriterion, *args, **kwargs):
        """
        Initialize the Regression Multi Weighted Contrastive Loss.
        
        Args:
            wcl_loss_terms (MultiWeightedContrastiveLoss): Multi-WCL loss component
            regression_loss_term (BaseCriterion): Regression loss component
            *args: Additional arguments passed to BaseCriterion
            **kwargs: Additional keyword arguments passed to BaseCriterion
        """
        super().__init__(*args, **kwargs)
        self.wcl_loss_terms = wcl_loss_terms
        self.regression_loss_term = regression_loss_term

    def forward(self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        """
        Forward pass of the Regression Multi Weighted Contrastive Loss.
        
        Computes the combined loss by:
        1. Computing multiple WCL terms using embeddings and target values from kwargs
        2. Computing regression loss using input predictions and target values
        3. Summing both loss components
        
        Args:
            input (torch.Tensor): Model predictions for regression loss computation
                Shape: (batch_size, num_targets) or (batch_size,)
            target (torch.Tensor): Ground truth target values for regression loss
                Shape: (batch_size, num_targets) or (batch_size,)
            *args: Additional positional arguments (unused)
            **kwargs: Dictionary containing:
                - Embedding tensors (e.g., 'ecg_embeddings', 'ppg_embeddings')
                - Target value tensors (e.g., 'SBP', 'DBP') for WCL computation
                - Any other data needed by WCL terms
                
        Returns:
            torch.Tensor: Combined loss value (WCL + regression)
            
        Note:
            The method expects kwargs to contain all embeddings and target values
            needed by the configured WCL terms. Each WCL term will extract its
            required data using its embedding_key and weight_key configuration.
            
        Example:
            >>> loss = criterion(
            ...     input=bp_predictions,      # (batch_size, 2) for SBP, DBP
            ...     target=bp_targets,         # (batch_size, 2) for SBP, DBP  
            ...     ecg_embeddings=ecg_emb,   # (batch_size, embedding_dim)
            ...     ppg_embeddings=ppg_emb,   # (batch_size, embedding_dim)
            ...     SBP=sbp_values,           # (batch_size, 1)
            ...     DBP=dbp_values            # (batch_size, 1)
            ... )
        """
        # Initialize total loss on the same device as input tensors
        # total_loss = torch.tensor(0.0, device=next(iter(kwargs.values())).device)
        total_loss = torch.tensor(0.0, device=input.device)

        loss_breakdown = {}
        
        # Compute WCL loss from multiple terms
        wcl_result = self.wcl_loss_terms(**kwargs)
        if isinstance(wcl_result, dict):
            total_loss += wcl_result['total_loss']
            if hasattr(self, 'return_breakdown') and self.return_breakdown:
                loss_breakdown['wcl'] = wcl_result['breakdown']
        else:
            total_loss += wcl_result
        
        # Compute regression loss
        regression_result = self.regression_loss_term(input, target)
        if isinstance(regression_result, dict):
            total_loss += regression_result['total_loss']
            if hasattr(self, 'return_breakdown') and self.return_breakdown:
                loss_breakdown['regression'] = regression_result['breakdown']
        else:
            total_loss += regression_result

        # Return breakdown if requested, otherwise just the total loss
        if hasattr(self, 'return_breakdown') and self.return_breakdown and loss_breakdown:
            return {
                'total_loss': total_loss,
                'breakdown': loss_breakdown
            }
        else:
            return total_loss


# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_regression_multi_wcl_loss", node=RegressionMultiWCLLossConfig) 