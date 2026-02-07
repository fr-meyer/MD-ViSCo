"""
Multi-Weighted Contrastive Loss criterion for handling multiple WCL targets.

This module provides a configurable multi-WCL implementation that can handle
multiple weighted contrastive loss terms with configurable weights and enabling/disabling per term.
Reuses existing WeightedContrastiveLoss class for modularity.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict
from omegaconf import MISSING
from .base_criterion import BaseCriterion, CriterionBaseConfig
from .weighted_contrastive_loss import WeightedContrastiveLoss, WeightedContrastiveLossConfig
from .base_criterion import ReductionType

@dataclass
class MultiWCLLossConfig(CriterionBaseConfig):
    """Pure data container for Multi Weighted Contrastive Loss configuration"""
    _target_: str = "src.criterions.multi_weighted_contrastive_loss.MultiWeightedContrastiveLoss"
    name: str = "multi_weighted_contrastive_loss"
    reduction: ReductionType = ReductionType.MEAN
    loss_terms: List[WeightedContrastiveLossConfig] = MISSING


class MultiWeightedContrastiveLoss(BaseCriterion):
    """
    Multi-Weighted Contrastive Loss for handling multiple WCL targets.
    
    This class can handle multiple WeightedContrastiveLoss instances.
    """
    
    def __init__(
        self,
        loss_terms: List[WeightedContrastiveLoss] = MISSING,
        *args,
        **kwargs
    ):
        # Call parent constructor - BaseCriterion handles everything
        super().__init__(*args, **kwargs)
        
        # Instantiate WeightedContrastiveLoss instances using Hydra
        self.loss_terms = loss_terms
    
    def forward(
        self,
        *args,
        **kwargs
        # embeddings_list: Dict[str, torch.Tensor], 
        # weights_list: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the multi-weighted contrastive loss.
        
        Args:
            *args: Variable length argument list (unused)
            **kwargs: Dictionary containing all embeddings and weights with their respective keys
            
        Returns:
            Total loss tensor
        """
        # Initialize total_loss - get device from first available tensor in kwargs
        device = None
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        
        if device is None:
            device = torch.device('cpu')
        
        total_loss = torch.tensor(0.0, device=device)
        
        for i, term in enumerate(self.loss_terms):
            # Get the embedding and weight keys from the term's config
            embedding_key = term.embedding_key
            weight_key = term.weight_key
            
            # Skip this loss term if either key is not found in kwargs
            if embedding_key is None or embedding_key not in kwargs:
                continue
            if weight_key is None or weight_key not in kwargs:
                continue
            
            # Get the embeddings and weights from kwargs
            embeddings = kwargs[embedding_key]
            weights = kwargs[weight_key]
            
            # Skip if either tensor is None or empty
            if embeddings is None or weights is None:
                continue
            
            # Use the WeightedContrastiveLoss directly
            loss = term(embeddings, weights)
            total_loss = total_loss + loss

            # try:
                
            # except Exception as e:
            #     # Skip this term if there's an error computing the loss
            #     continue
        
        return total_loss


# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_multi_wcl_loss", node=MultiWCLLossConfig) 