"""
Dataset utilities for data loading and splitting.
"""

import logging
from src.dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# NEW: Core scenario detection and orchestration functions
def _determine_training_scenario(
    is_pretraining: bool,
    is_finetuning: bool
) -> str:
    """Determine training scenario from boolean flags
    
    Args:
        is_pretraining: Whether this is a pretraining scenario
        is_finetuning: Whether this is a finetuning scenario
        
    Returns:
        str: "standard", "pretraining", or "finetuning"
    """
    if is_finetuning:
        return "finetuning"
    elif is_pretraining:
        return "pretraining"
    else:
        return "standard"


def create_single_dataset(
    dataset_config,
    split: str = "train"  # "train" or "test"
) -> BaseDataset:
    """
    Create a single dataset from specified split using Hydra instantiate.
    
    Args:
        dataset_config: Dataset configuration object
        split: "train" or "test" (for logging purposes)
        
    Returns:
        Single dataset from specified config
    """
    from hydra.utils import instantiate
    
    if dataset_config is None:
        raise ValueError(f"{split}_dataset configuration is missing")
    
    dataset = instantiate(dataset_config)
    logger.info(f"[Dataset] Created {split} dataset: {type(dataset).__name__} with {len(dataset)} samples")
    return dataset


def _split_by_sample(dataset, indices, train_ratio: float, val_ratio: float, include_test: bool):
    """Generic sample-level split for any dataset."""
    dataset_size = len(dataset)
    train_len = int(train_ratio * dataset_size)
    val_len = int(val_ratio * dataset_size)

    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]

    if include_test:
        test_idx = indices[train_len + val_len:]
        logger.info(f"[Dataset] Sample split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return (
            type(dataset).create_subset(dataset, train_idx),
            type(dataset).create_subset(dataset, val_idx),
            type(dataset).create_subset(dataset, test_idx)
        )
    else:
        logger.info(f"[Dataset] Sample split: train={len(train_idx)}, val={len(val_idx)}")
        return (
            type(dataset).create_subset(dataset, train_idx),
            type(dataset).create_subset(dataset, val_idx)
        )

def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float, scenario: str, tolerance: float = 1e-6):
    """
    Validate split ratios for specific training scenario.
    
    Args:
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        scenario: Training scenario ("standard", "pretraining", or "finetuning")
        tolerance: Numerical tolerance for floating point comparison
        
    Raises:
        ValueError: If ratios don't sum to 1.0 for the given scenario
    """
    if scenario in ["standard", "pretraining"]:
        total = train_ratio + val_ratio
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"{scenario} split ratios must sum to 1.0, got {total:.4f} "
                f"(train={train_ratio:.4f}, val={val_ratio:.4f})"
            )
    elif scenario == "finetuning":
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"Finetuning split ratios must sum to 1.0, got {total:.4f} "
                f"(train={train_ratio:.4f}, val={val_ratio:.4f}, test={test_ratio:.4f})"
            )
    else:
        raise ValueError(f"Unknown training scenario: {scenario}")