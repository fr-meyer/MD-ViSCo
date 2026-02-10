"""Preprocessor module for input preprocessing operations.

This module provides infrastructure for transforming raw input data into
model-ready formats, following the same pattern as processors.

Public API:
    - BasePreprocessor: Abstract base class for all preprocessors
    - BasePreprocessorConfig: Base configuration dataclass
    - import_preprocessors: Dynamic import function for preprocessor discovery
"""

from src.preprocessors.base_preprocessor import BasePreprocessor
from src.preprocessors.base_preprocessor import BasePreprocessorConfig


def import_preprocessors() -> int:
    """Import all preprocessor submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported preprocessor modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    if __package__ is not None:
        return _import_modules(
            __package__ or "src.preprocessors", module_type="preprocessor"
        )
    return 0


__all__ = [
    # Base classes
    "BasePreprocessor",
    "BasePreprocessorConfig",
    # Dynamic import functions
    "import_preprocessors",
]
