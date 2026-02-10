"""Criterion infrastructure for PyTorch loss functions.

This package provides a base criterion class and various loss implementations
for the MD-ViSCo project. All criteria are configurable via Hydra and
registered in the ConfigStore for dynamic instantiation.

Classes:
    - BaseCriterion: Abstract base class for all loss modules (in submodules)

Functions:
    - import_criterions: Import all criterion submodules for ConfigStore registration

Examples:
    >>> from src.criterions import import_criterions
    >>> import_criterions()

See Also:
    - src.criterions.base_criterion: Base criterion interface
    - src.conf.criterion: Hydra criterion configs
"""

from __future__ import annotations

from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig


def import_criterions() -> int:
    """Import all criterion submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported criterion modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    pkg: str = __package__ or "src.criterions"
    return _import_modules(pkg, module_type="criterion")


__all__ = [
    "BaseCriterion",
    "CriterionBaseConfig",
    "import_criterions",
]
