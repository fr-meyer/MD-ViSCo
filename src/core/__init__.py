"""
This module contains the core components of the MD-ViSCo framework.
"""


def import_core() -> int:
    """
    Import all core submodules to trigger ConfigStore registration.
    """
    from src.utils.module_utils import import_modules as _import_modules
    return _import_modules(__package__, module_type="core")

def register_core():
    """
    Register all core components to the ConfigStore.
    """
    from src.core.direction import register_directions
    from src.dataset.base_dataset import register_vital
    from src.core.domain import register_direction
    register_directions()
    register_vital()
    register_direction()