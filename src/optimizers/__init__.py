"""Optimizers package for PyTorch optimizers.

This package provides a base optimizer class and various optimizer implementations
for the MD-ViSCo project using Hydra for configuration management.
"""

from __future__ import annotations


def import_optimizers() -> int:
    """Import all optimizer submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported optimizer modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    if __package__ is not None:
        return _import_modules(__package__ or "src.optimizers", module_type="optimizer")
    return 0


__all__ = [
    "import_optimizers",
]
