"""
Multi-Regression Multi-WCL Loss criterion for combining multiple regression and WCL losses.

This module provides a composite loss implementation that combines multiple regression
losses (L1, MSE, etc.) with multiple weighted contrastive losses (WCL) in a single,
configurable criterion.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, List, Any
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType
from .multi_regression_loss import MultiRegressionLoss, MultiRegressionLossConfig
from .multi_weighted_contrastive_loss import MultiWCLLoss, MultiWCLLossConfig

@dataclass
class MultiRegressionMultiWCLLossConfig(CriterionBaseConfig):
    """Pure data container for Multi Regression Multi WCL configuration"""
    _target_: str = "src.criterions.multi_regression_multi_wcl_loss.MultiRegressionMultiWCLLoss"
    regression_weight: float = 1.0
    wcl_weight: float = 0.2
    return_breakdown: bool = True
    regression_config: Dict[str, Any] = field(default_factory=dict)
    wcl_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.regression_weight < 0:
            raise ValueError(f"regression_weight must be non-negative, got {self.regression_weight}")
        if self.wcl_weight < 0:
            raise ValueError(f"wcl_weight must be non-negative, got {self.wcl_weight}")
        
        # Check for duplicate term names across both configs
        regression_names = set(self.regression_config.get_term_names())
        if self.wcl_config is not None:
            wcl_names = set(self.wcl_config.get_term_names())
            overlapping = regression_names & wcl_names
            if overlapping:
                raise ValueError(f"Overlapping loss term names found: {overlapping}")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'regression_weight': self.regression_weight,
            'wcl_weight': self.wcl_weight,
            'return_breakdown': self.return_breakdown,
            'name': self.name,
            'enabled': self.enabled,
            'regression_config': self.regression_config.to_dict(),
            'wcl_config': self.wcl_config.to_dict() if self.wcl_config else None
        }

class MultiRegressionMultiWCLLoss(BaseCriterion):
    """
    Multi-Regression Multi-WCL Loss for combining multiple regression and WCL losses.
    
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
        regression_config: Dict[str, Any] = None,
        wcl_config: Dict[str, Any] = None,
        reduction: ReductionType = ReductionType.MEAN,
        device: Optional[torch.device] = None,
        name: str = "multi_regression_multi_wcl_loss",
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
        self.regression_weight = regression_weight
        self.wcl_weight = wcl_weight
        self.return_breakdown = return_breakdown
        self.regression_config = regression_config or {}
        self.wcl_config = wcl_config or {}
    
    def forward(self, model_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Compute the multi-regression multi-WCL loss.
        
        Args:
            model_outputs: Dictionary containing model outputs and embeddings
            batch: Dictionary containing target values
            
        Returns:
            If return_breakdown is False: Total loss tensor
            If return_breakdown is True: Dictionary with total loss and breakdown
        """
        if not self.enabled:
            # Get device from model outputs for consistency
            device = next(iter(model_outputs.values())).device if model_outputs else torch.device('cpu')
            if self.return_breakdown:
                return {'total_loss': torch.tensor(0.0, device=device), 'breakdown': {}}
            else:
                return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=next(iter(model_outputs.values())).device)
        loss_breakdown = {}
        
        # Compute regression loss
        regression_result = self.regression_loss(model_outputs, batch)
        if isinstance(regression_result, dict):
            regression_loss = regression_result['total_loss']
            if self.return_breakdown:
                loss_breakdown['regression'] = regression_result['breakdown']
        else:
            regression_loss = regression_result
        
        weighted_regression_loss = self.regression_weight * regression_loss
        total_loss = total_loss + weighted_regression_loss
        
        # Compute WCL loss if enabled
        if self.wcl_loss is not None:
            wcl_result = self.wcl_loss(model_outputs, batch)
            if isinstance(wcl_result, dict):
                wcl_loss = wcl_result['total_loss']
                if self.return_breakdown:
                    loss_breakdown['wcl'] = wcl_result['breakdown']
            else:
                wcl_loss = wcl_result
            
            weighted_wcl_loss = self.wcl_weight * wcl_loss
            total_loss = total_loss + weighted_wcl_loss
        else:
            wcl_loss = torch.tensor(0.0, device=total_loss.device)
            weighted_wcl_loss = torch.tensor(0.0, device=total_loss.device)
        
        # Return result
        if self.return_breakdown:
            loss_breakdown['total'] = {
                'regression_loss': regression_loss.item(),
                'wcl_loss': wcl_loss.item(),
                'weighted_regression_loss': weighted_regression_loss.item(),
                'weighted_wcl_loss': weighted_wcl_loss.item(),
                'total_loss': total_loss.item()
            }
            return {
                'total_loss': total_loss,
                'breakdown': loss_breakdown
            }
        else:
            return total_loss
    
    def get_regression_terms(self) -> List[str]:
        """Get list of regression loss term names."""
        return self.regression_loss.get_all_terms()
    
    def get_wcl_terms(self) -> List[str]:
        """Get list of WCL loss term names."""
        if self.wcl_loss is not None:
            return self.wcl_loss.get_all_terms()
        return []
    
    def enable_regression_term(self, term_name: str):
        """Enable a specific regression loss term by name."""
        self.regression_loss.enable_term(term_name)
    
    def disable_regression_term(self, term_name: str):
        """Disable a specific regression loss term by name."""
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

    # ============================================================================
    # FACTORY METHODS FOR AUTO-CONFIGURATION
    # ============================================================================
    
    @classmethod
    def from_model(cls, model, **kwargs) -> 'MultiRegressionMultiWCLLoss':
        """
        Create a criterion instance automatically configured for the given model.
        
        This factory method analyzes the model and creates the appropriate loss
        configuration automatically. For BP models with WCL, returns MultiRegressionMultiWCLLoss.
        For other models, raises ValueError (use MultiRegressionLoss.from_model instead).
        
        Args:
            model: Model instance to analyze for configuration
            **kwargs: Additional configuration overrides
            
        Returns:
            MultiRegressionMultiWCLLoss: Configured criterion instance
            
        Raises:
            ValueError: If model is not a BP model with WCL enabled
        """
        if not cls._is_bp_model_with_wcl(model):
            raise ValueError("Model is not a BP model with WCL enabled. Use MultiRegressionLoss.from_model() for other models.")
        
        # Create the configuration using the new constructor pattern
        regression_config, wcl_config = cls._create_bp_model_configs(model, **kwargs)
        return cls(
            regression_config=regression_config,
            wcl_config=wcl_config,
            regression_weight=1.0,
            wcl_weight=1.0,
            return_breakdown=True,
            name="bp_composite_loss",
            **kwargs
        )
    
    @classmethod
    def _is_bp_model_with_wcl(cls, model) -> bool:
        """Check if the model is a BPModel instance with WCL enabled."""
        return (hasattr(model, 'wcl') and model.wcl and 
                hasattr(model, 'pi') and hasattr(model, 'wcl_criterion'))
    
    @classmethod
    def _create_bp_model_configs(cls, model, **kwargs) -> tuple:
        """Create BP model regression and WCL configurations."""
        from hydra.utils import instantiate
        from .weighted_contrastive_loss import WeightedContrastiveLossConfig
        from .multi_weighted_contrastive_loss import WCLLossTermConfig, MultiWCLLossConfig
        
        # Create regression loss terms for BP predictions
        l1_criterion = instantiate({"target_": "src.criterions.l1_loss.L1Loss"})
        regression_terms = [
                    (l1_criterion, "ecg_sbp", "SBP_n", 1.0),
        (l1_criterion, "ecg_dbp", "DBP_n", 1.0),
        (l1_criterion, "ppg_sbp", "SBP_n", 1.0),
        (l1_criterion, "ppg_dbp", "DBP_n", 1.0),
        ]
        
        regression_config = MultiRegressionLossConfig(
            loss_terms=regression_terms,
            return_breakdown=True,
            name="bp_regression_loss"
        )
        
        # Create WCL terms (WCL is enabled for BP models)
        wcl_terms = []
        
        # ECG embeddings WCL
        if hasattr(model, 'wcl_criterion'):
            ecg_wcl_config = WeightedContrastiveLossConfig(
                temperature=4.0,
                threshold=0.1 if model.normalized_bp else 0.0,
                name="ecg_wcl"
            )
            wcl_terms.append(WCLLossTermConfig(
                name="ecg_wcl",
                embedding_key="ecg_embeddings",
                target_key="SBP",  # Use raw BP for WCL
                weight=1e-3 if model.normalized_bp else 1.0,
                wcl_config=ecg_wcl_config
            ))
        
        # PPG embeddings WCL
        ppg_wcl_config = WeightedContrastiveLossConfig(
            temperature=4.0,
            threshold=0.1 if model.normalized_bp else 0.0,
            name="ppg_wcl"
        )
        wcl_terms.append(WCLLossTermConfig(
            name="ppg_wcl",
            embedding_key="ppg_embeddings",
            target_key="SBP",  # Use raw BP for WCL
            weight=1e-3 if model.normalized_bp else 1.0,
            wcl_config=ppg_wcl_config
        ))
        
        # Text embeddings WCL if PI is enabled
        if model.pi:
            text_gender_wcl_config = WeightedContrastiveLossConfig.for_binary_classification(
                temperature=4.0
            )
            text_gender_wcl_config.name = "text_gender_wcl"
            wcl_terms.append(WCLLossTermConfig(
                name="text_gender_wcl",
                embedding_key="text_embeddings",
                target_key="gender",
                weight=1e-2 if model.normalized_bp else 1.0,
                wcl_config=text_gender_wcl_config
            ))
            
            if model.wcl_age_threshold is not None:
                text_age_wcl_config = WeightedContrastiveLossConfig.for_regression(
                    temperature=4.0
                )
                text_age_wcl_config.threshold = model.wcl_age_threshold
                text_age_wcl_config.name = "text_age_wcl"
                wcl_terms.append(WCLLossTermConfig(
                    name="text_age_wcl",
                    embedding_key="text_embeddings",
                    target_key="age",
                    weight=1e-2 if model.normalized_bp else 1.0,
                    wcl_config=text_age_wcl_config
                ))
        
        wcl_config = MultiWCLLossConfig(
            loss_terms=wcl_terms,
            return_breakdown=True,
            name="bp_wcl_loss"
        )
        
        return regression_config, wcl_config

# Lazy registration
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_multi_regression_multi_wcl_loss", node=MultiRegressionMultiWCLLossConfig) 