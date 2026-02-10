"""Dataset infrastructure for medical signal loading and splitting.

This module provides core dataset creation, train/val/test splitting, and
dynamic dataset registration for Hydra. Datasets return raw waveforms with
lazy normalization; padding and trimming are handled in the collate function.

Classes:
    - BaseDataset: Abstract base class for vital-sign datasets with shared memory
    - Sample: Dataclass for vital-specific raw waveform fields per sample
    - VitalsDataset: Channel mapping and direction capability for vitals

Functions:
    - create_training_datasets: Build train/val (and optionally test) datasets
    - create_test_dataset: Build test dataset for a given training scenario
    - import_datasets: Import all dataset submodules for ConfigStore registration
    - _determine_training_scenario: Resolve training scenario from config

Examples:
    >>> from src.dataset import create_training_datasets, create_test_dataset
    >>> train_ds, val_ds, test_ds = create_training_datasets(...)
    >>> test_ds = create_test_dataset(...)

See Also:
    - src.dataset.base_dataset: Base dataset and Sample definitions
    - src.dataset.core: Dataset creation and split logic
    - src.utils.dataset_utils: Single-dataset creation and split helpers
"""

# Local imports
from src.utils.dataset_utils import _determine_training_scenario
from src.utils.module_utils import import_modules as _import_modules

from .core import create_test_dataset
from .core import create_training_datasets


def import_datasets() -> int:
    """Import all dataset submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported dataset modules
    """
    pkg: str = __package__ or "src.dataset"
    return _import_modules(pkg, module_type="dataset")


# Export clean API
__all__ = [
    "create_training_datasets",
    "create_test_dataset",
    "import_datasets",
    "_determine_training_scenario",
]
