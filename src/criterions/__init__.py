"""
Criterions package for PyTorch loss functions.

This package provides a base criterion class and various loss implementations
for the MD-ViSCo project using Hydra for configuration management.
"""

from __future__ import annotations

from .base_criterion import BaseCriterion, CriterionBaseConfig
from src.utils.module_utils import import_modules as _import_modules

def import_criterions() -> int:
    """
    Import all criterion submodules to trigger ConfigStore registration.
    
    Returns:
        Number of successfully imported criterion modules
    """
    return _import_modules(__package__, module_type="criterion")

__all__ = [
    'BaseCriterion',
    'CriterionBaseConfig',
    'import_criterions',
] 