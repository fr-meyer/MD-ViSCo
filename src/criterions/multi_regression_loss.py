"""
Multi-Regression Loss criterion for handling multiple scalar regression targets.

This module provides a configurable multi-regression loss implementation that can handle
multiple regression loss terms with configurable weights and enabling/disabling per term.
Reuses existing criterion classes (L1Loss, MSELoss, etc.) for modularity.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Tuple, Any
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType

@dataclass
class MultiRegressionLossConfig(CriterionBaseConfig):
    """Pure data container for Multi Regression Loss configuration"""
    _target_: str = "src.criterions.multi_regression_loss.MultiRegressionLoss"
    return_breakdown: bool = True
    loss_terms: List[List[Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.loss_terms:
            raise ValueError("loss_terms cannot be empty")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'return_breakdown': self.return_breakdown,
            'name': self.name,
            'enabled': self.enabled,
            'num_terms': len(self.loss_terms)
        }
    
    def get_term_names(self) -> List[str]:
        """Get list of term names."""
        return [getattr(term[0], 'name', f'term_{i}') for i, term in enumerate(self.loss_terms)]

class MultiRegressionLoss(BaseCriterion):
    """
    Multi-Regression Loss for handling multiple scalar regression targets.
    
    This class can handle multiple regression loss terms with configurable weights
    and enabling/disabling per term. Reuses existing criterion classes for modularity.
    
    Args:
        config: MultiRegressionLossConfig containing loss terms and settings
    """
    
    def __init__(
        self,
        return_breakdown: bool = True,
        loss_terms: List[List[Any]] = None,
        reduction: ReductionType = ReductionType.MEAN,
        device: Optional[torch.device] = None,
        name: str = "multi_regression_loss",
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
        self.return_breakdown = return_breakdown
        self.loss_terms = loss_terms or []
    
    def forward(self, model_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, Union[float, bool]]]]]:
        """
        Compute the multi-regression loss.
        
        Args:
            model_outputs: Dictionary containing model predictions
            batch: Dictionary containing target values
            
        Returns:
            If return_breakdown is False: Total loss tensor
            If return_breakdown is True: Dictionary with total loss and individual terms
        """
        if not self.enabled:
            # Get device from model outputs for consistency
            device = next(iter(model_outputs.values())).device if model_outputs else torch.device('cpu')
            if self.return_breakdown:
                return {'total_loss': torch.tensor(0.0, device=device), 'breakdown': {}}
            else:
                return torch.tensor(0.0, device=device)
        
        # Initialize total_loss as a tensor to maintain proper typing
        device = next(iter(model_outputs.values())).device
        total_loss = torch.tensor(0.0, device=device)
        loss_breakdown = {}
        
        for term in self.loss_terms:
            if not term['enabled']:
                continue
                
            # Get prediction and target tensors
            if term['prediction_key'] not in model_outputs:
                raise KeyError(f"Prediction key '{term['prediction_key']}' not found in model_outputs")
            if term['target_key'] not in batch:
                raise KeyError(f"Target key '{term['target_key']}' not found in batch")
                
            prediction = model_outputs[term['prediction_key']]
            target = batch[term['target_key']]
            
            # Use the existing criterion directly
            loss = term['criterion'](prediction, target)
            weighted_loss = term['weight'] * loss
            
            total_loss = total_loss + weighted_loss
            
            # Store breakdown if requested
            if self.return_breakdown:
                loss_breakdown[term['name']] = {
                    'raw_loss': loss.item(),
                    'weighted_loss': weighted_loss.item(),
                    'weight': term['weight'],
                    'enabled': term['enabled']
                }
        
        if self.return_breakdown:
            loss_breakdown['total_loss'] = total_loss.item()
            return {
                'total_loss': total_loss,
                'breakdown': loss_breakdown
            }
        else:
            return total_loss
    
    def get_enabled_terms(self) -> List[str]:
        """Get list of enabled loss term names."""
        return [term['name'] for term in self.loss_terms if term['enabled']]
    
    def get_all_terms(self) -> List[str]:
        """Get list of all loss term names (enabled and disabled)."""
        return [term['name'] for term in self.loss_terms]
    
    def enable_term(self, term_name: str):
        """Enable a specific loss term by name."""
        for term in self.loss_terms:
            if term['name'] == term_name:
                term['enabled'] = True
                return
        raise ValueError(f"Loss term '{term_name}' not found")
    
    def disable_term(self, term_name: str):
        """Disable a specific loss term by name."""
        for term in self.loss_terms:
            if term['name'] == term_name:
                term['enabled'] = False
                return
        raise ValueError(f"Loss term '{term_name}' not found")
    
    def set_term_weight(self, term_name: str, weight: float):
        """Set the weight for a specific loss term by name."""
        for term in self.loss_terms:
            if term['name'] == term_name:
                term['weight'] = weight
                return
        raise ValueError(f"Loss term '{term_name}' not found")
    
    def get_term_weight(self, term_name: str) -> float:
        """Get the weight for a specific loss term by name."""
        for term in self.loss_terms:
            if term['name'] == term_name:
                return term['weight']
        raise ValueError(f"Loss term '{term_name}' not found")

    # ============================================================================
    # FACTORY METHODS FOR AUTO-CONFIGURATION
    # ============================================================================
    
    @classmethod
    def from_model(cls, model, **kwargs) -> 'MultiRegressionLoss':
        """
        Create a MultiRegressionLoss instance automatically configured for the given model.
        
        Args:
            model: Model instance to analyze for configuration
            **kwargs: Additional configuration overrides
            
        Returns:
            MultiRegressionLoss: Configured criterion instance
        """
        from hydra.utils import instantiate
        
        # Create simple regression terms for BP predictions
        l1_criterion = instantiate({"target_": "src.criterions.l1_loss.L1Loss"})
        regression_terms = [
                    (l1_criterion, "sbp_pred", "SBP_n", 1.0),
        (l1_criterion, "dbp_pred", "DBP_n", 1.0),
        ]
        
        return cls(
            loss_terms=regression_terms,
            return_breakdown=True,
            name="scalar_regression_loss",
            **kwargs
        )

# Lazy registration
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_multi_regression_loss", node=MultiRegressionLossConfig)
