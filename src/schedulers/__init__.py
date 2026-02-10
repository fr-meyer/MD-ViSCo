"""Schedulers package for PyTorch learning rate schedulers.

This package provides a base scheduler class and various scheduler implementations
for the MD-ViSCo project using Hydra for configuration management.

See Also:
    src.trainers.trainer.BaseTrainer.create_scheduler : Method for creating schedulers
        from Hydra configurations during training.
    src.conf.scheduler : Hydra configuration files for scheduler instantiation.
"""

from __future__ import annotations


def import_schedulers() -> int:
    """Import all scheduler submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported scheduler modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    if __package__ is not None:
        return _import_modules(__package__ or "src.schedulers", module_type="scheduler")
    return 0


__all__ = [
    "import_schedulers",
]
