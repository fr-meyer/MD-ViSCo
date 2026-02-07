"""
Base criterion class for PyTorch loss functions.

This module provides a base class that all custom loss functions should inherit from.
It includes common functionality like reduction methods, device handling, and logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Optional, Union, Dict, Any, Tuple
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from omegaconf import MISSING

class ReductionType(Enum):
    """Enum for loss reduction types supported by all criteria."""
    NONE = 'none'
    MEAN = 'mean'
    SUM = 'sum'
    
    @classmethod
    def from_string(cls, value: str) -> 'ReductionType':
        """Convert string to enum value."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid reduction type '{value}'. Must be one of: {[e.value for e in cls]}")
    
    def __str__(self) -> str:
        return self.value

@dataclass
class CriterionBaseConfig:
    """Pure data container for criterion configuration"""
    name: str = MISSING
    _target_: str = MISSING
    enabled: bool = True
    log_loss: bool = False
    device: Optional[str] = None  # String-based for YAML compatibility


class BaseCriterion(_Loss):
    """
    Base class for all criterion (loss) functions in the MD-ViSCo project.
    
    This class provides a common interface and functionality for all loss functions,
    including proper initialization, reduction methods, device handling, and logging.
    
    Attributes:
        reduction (ReductionType): Reduction method for the loss
        device (torch.device): Device to compute the loss on
        name (str): Name of the criterion for logging purposes
        log_loss (bool): Whether to log loss values during computation
        logger (logging.Logger): Logger instance for loss tracking
    """
    
    def __init__(
        self,
        reduction: Union[str, ReductionType] = ReductionType.MEAN,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        log_loss: bool = False,
        **kwargs
    ):
        """
        Initialize the base criterion.
        
        Args:
            reduction (Union[str, ReductionType]): Reduction method for the loss.
                Options: 'none', 'mean', 'sum' or ReductionType enum values. Default: ReductionType.MEAN
            device (torch.device, optional): Device to compute the loss on.
                If None, will use the device of the input tensors.
            name (str, optional): Name of the criterion for logging purposes.
                If None, will use the class name.
            log_loss (bool): Whether to log loss values during computation.
                Default: False
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__()
        
        # Validate and convert reduction method
        if isinstance(reduction, str):
            self.reduction = ReductionType.from_string(reduction)
        elif isinstance(reduction, ReductionType):
            self.reduction = reduction
        else:
            raise ValueError(f"Invalid reduction type. Expected str or ReductionType, got {type(reduction)}")
        
        self.device = device
        self.name = name or self.__class__.__name__
        self.log_loss = log_loss
        
        # Setup logging if enabled
        if self.log_loss:
            self.logger = self._setup_logger()
        else:
            self.logger = None
        
        # Store additional configuration
        self.config = kwargs
        
        # Initialize loss statistics
        self.reset_stats()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for loss tracking."""
        logger = logging.getLogger(f"criterion.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def reset_stats(self):
        """Reset loss statistics."""
        self.loss_history = []
        self.total_loss = 0.0
        self.num_calls = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current loss statistics."""
        if self.num_calls == 0:
            return {
                'total_loss': 0.0,
                'mean_loss': 0.0,
                'num_calls': 0,
                'loss_history': []
            }
        
        return {
            'total_loss': self.total_loss,
            'mean_loss': self.total_loss / self.num_calls,
            'num_calls': self.num_calls,
            'loss_history': self.loss_history.copy()
        }
    
    def _update_stats(self, loss_value: float):
        """Update loss statistics."""
        self.loss_history.append(loss_value)
        self.total_loss += loss_value
        self.num_calls += 1
        
        if self.log_loss and self.logger:
            self.logger.info(f"Loss: {loss_value:.6f}")
    
    def _validate_inputs(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Validate and prepare input tensors.
        
        Args:
            *args: Input tensors
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple[torch.Tensor, ...]: Validated tensors
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If tensors are not on the same device
        """
        tensors = []
        
        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                raise ValueError(f"Input {i} must be a torch.Tensor, got {type(arg)}")
            
            # Move to device if specified
            if self.device is not None and arg.device != self.device:
                arg = arg.to(self.device)
            
            tensors.append(arg)
        
        # Check if all tensors are on the same device
        if len(tensors) > 1:
            devices = {t.device for t in tensors}
            if len(devices) > 1:
                raise RuntimeError(f"All tensors must be on the same device. "
                                 f"Found devices: {devices}")
        
        return tuple(tensors)
    
    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply reduction to the loss tensor.
        
        Args:
            loss (torch.Tensor): Loss tensor to reduce
            
        Returns:
            torch.Tensor: Reduced loss tensor
        """
        if self.reduction == ReductionType.NONE:
            return loss
        elif self.reduction == ReductionType.MEAN:
            return loss.mean()
        elif self.reduction == ReductionType.SUM:
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")
    
    def _check_gradients(self, loss: torch.Tensor) -> None:
        """
        Check for potential gradient issues.
        
        Args:
            loss (torch.Tensor): Loss tensor to check
        """
        if not loss.requires_grad:
            warnings.warn(f"{self.name}: Loss tensor does not require gradients. "
                         f"This might cause issues during backpropagation.")
        
        if torch.isnan(loss).any():
            warnings.warn(f"{self.name}: Loss contains NaN values.")
        
        if torch.isinf(loss).any():
            warnings.warn(f"{self.name}: Loss contains infinite values.")
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the criterion.
        
        This method should be overridden by subclasses to implement
        the actual loss computation.
        
        Args:
            *args: Input tensors
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor: Computed loss
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented by subclasses"
        )
    
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the loss with validation and statistics tracking.
        
        This is the main method that should be called to compute the loss.
        It handles input validation, device management, reduction, and statistics.
        
        Args:
            *args: Input tensors
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Validate inputs
        validated_args = self._validate_inputs(*args)
        
        # Compute the loss using the forward method
        loss = self.forward(*validated_args, **kwargs)
        
        # Apply reduction
        loss = self._apply_reduction(loss)
        
        # Check for potential issues
        self._check_gradients(loss)
        
        # Update statistics
        if self.reduction != ReductionType.NONE:
            loss_value = loss.item()
            self._update_stats(loss_value)
        
        return loss
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        extra_repr_parts = [
            f"reduction={self.reduction.value}",
            f"name={self.name}",
            f"log_loss={self.log_loss}"
        ]
        
        if self.device is not None:
            extra_repr_parts.append(f"device={self.device}")
        
        if self.config:
            config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
            extra_repr_parts.append(f"config={{{config_str}}}")
        
        return ", ".join(extra_repr_parts)
    
    def to(self, device: Union[torch.device, str]) -> 'BaseCriterion':
        """
        Move the criterion to the specified device.
        
        Args:
            device (torch.device or str): Device to move to
            
        Returns:
            BaseCriterion: Self for chaining
        """
        super().to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        return self
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Call the criterion with validation and statistics tracking.
        
        Args:
            *args: Input tensors
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor: Computed loss
        """
        return self.compute_loss(*args, **kwargs)


 