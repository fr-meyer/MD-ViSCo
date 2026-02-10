"""Preprocessing package for MD-ViSCo datasets.

This package provides dataset preprocessing functionality with Hydra
configuration support. All preprocessor registration and config merging is
handled natively by Hydra.

**Best Practice:** Use `hydra.utils.instantiate(config.preprocessor)` for all
preprocessor instantiation. This ensures native Hydra integration and automatic
config validation.
"""


def import_preprocessors() -> int:
    """Import all preprocessor submodules to trigger ConfigStore registration.

    This function dynamically imports all preprocessor modules from the preprocessors/
    subdirectory, which triggers their ConfigStore registration at module import time.

    Returns:
        Number of successfully imported preprocessor modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    # Use the preprocessors subpackage path
    return _import_modules(
        "src.script.preprocess.preprocessors", module_type="preprocessor"
    )


__all__ = ["import_preprocessors"]
