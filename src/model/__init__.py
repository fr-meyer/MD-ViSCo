"""Hydra-based model instantiation system.

This package provides model instantiation through Hydra's ConfigStore. All model
registration and config merging is handled natively by Hydra.

**Best Practice:** Use `hydra.utils.instantiate(config.model)` for all model
instantiation. This ensures native Hydra integration and automatic config validation.
"""

from __future__ import annotations

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def import_models() -> int:
    """Import all model submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported model modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    if __package__ is not None:
        return _import_modules(__package__ or "src.model", module_type="model")
    return 0


__all__ = ["import_models"]
