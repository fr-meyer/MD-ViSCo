"""Utility infrastructure for the MD-ViSCo project.

This package provides shared utilities for validation, training, preprocessing,
checkpoint I/O, collation, dataset creation, and module discovery. Utilities
are used across trainers, evaluators, processors, and datasets.

Functions:
    - flatten_config: Flatten nested config for serialization
    - is_wandb_serializable: Check if value is WandB-serializable
    - print_model_parameters: Log model parameter counts
    - ensure_dir_exists: Check directory existence
    - min_max_norm_torch: Min-max normalization (torch)
    - check_file_exists: Validate file path
    - calculate_map: Compute mean arterial pressure
    - global_min_max_norm: Global min-max normalization
    - safe_create_dataset: Create dataset with validation

Classes:
    - EarlyStopping: Early stopping helper for training

Examples:
    >>> from src.utils import flatten_config, EarlyStopping
    >>> flat = flatten_config(cfg)

See Also:
    - src.utils.checkpoint_io: Checkpoint save/load
    - src.utils.collate_utils: Batch collation
    - src.utils.dataset_utils: Dataset creation helpers
"""

from .train_utils import EarlyStopping
from .utils_preprocessing import calculate_map
from .utils_preprocessing import check_file_exists
from .utils_preprocessing import ensure_dir_exists
from .utils_preprocessing import global_min_max_norm
from .utils_preprocessing import min_max_norm_torch
from .utils_preprocessing import print_model_parameters
from .utils_preprocessing import safe_create_dataset
from .validation_utils import flatten_config
from .validation_utils import is_wandb_serializable

__all__ = [
    # Validation utilities
    "flatten_config",
    "is_wandb_serializable",
    # Training utilities
    "EarlyStopping",
    # Preprocessing utilities
    "print_model_parameters",
    "ensure_dir_exists",
    "min_max_norm_torch",
    "check_file_exists",
    "calculate_map",
    "global_min_max_norm",
    "safe_create_dataset",
]
