# Local imports
from src.utils.module_utils import import_modules as _import_modules

# Import core dataset creation functions
from .core import create_training_datasets, create_test_dataset
from src.utils.dataset_utils import _determine_training_scenario


def import_datasets() -> int:
    """
    Import all dataset submodules to trigger ConfigStore registration.
    
    Returns:
        Number of successfully imported dataset modules
    """
    return _import_modules(__package__, module_type="dataset")


# Export clean API
__all__ = [
    'create_training_datasets',
    'create_test_dataset',
    'import_datasets',
    '_determine_training_scenario'
] 