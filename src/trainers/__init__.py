"""
MD-ViSCo Training Framework

A comprehensive training framework supporting various training stages, variants,
and model-specific implementations.
"""
from __future__ import annotations

from .trainer import BaseTrainer
from src.utils.module_utils import import_modules as _import_modules

def import_trainers() -> int:
    """
    Import all trainer submodules to trigger ConfigStore registration.
    
    Returns:
        Number of successfully imported trainer modules
        
    Call this explicitly from your bootstrap (e.g., train.py) before instantiation,
    to avoid import-time side-effects when used as a library.
    """
    return _import_modules(__package__, module_type="trainer")

__version__ = "1.0.0"
__all__ = ["BaseTrainer", "import_trainers"] 