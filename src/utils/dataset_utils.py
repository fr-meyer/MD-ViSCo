"""Dataset utilities for data loading and splitting.

This module provides scenario detection, single-dataset creation via Hydra
instantiate, and sample-level split helpers. Used by :mod:`src.dataset.core`
for create_training_datasets and create_test_dataset.

Functions:
    - _determine_training_scenario: Resolve "standard" / "pretraining" /
      "finetuning" / "few_shot"
    - create_single_dataset: Create a single dataset from Hydra config
    - _split_by_sample: Generic sample-level train/val[/test] split
    - _validate_split_ratios: Validate split ratios for a given scenario

Examples:
    >>> scenario = _determine_training_scenario(
    ...     False, True, False
    ... )  # "finetuning"
    >>> train_ds = create_single_dataset(train_config, split="train")

See Also:
    - src.dataset.core: create_training_datasets, create_test_dataset,
      split_dataset
    - src.dataset.base_dataset: BaseDataset
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
from torch.utils.data import DataLoader

# Local imports
from src.dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def _determine_training_scenario(
    is_pretraining: bool, is_finetuning: bool, is_few_shot: bool = False
) -> str:
    """Determine training scenario from boolean flags.

    Args:
        is_pretraining: Whether this is a pretraining scenario
        is_finetuning: Whether this is a finetuning scenario
        is_few_shot: Whether this is a few-shot learning scenario

    Returns:
        str: "standard", "pretraining", "finetuning", or "few_shot"
    """
    if is_few_shot:
        return "few_shot"
    elif is_finetuning:
        return "finetuning"
    elif is_pretraining:
        return "pretraining"
    else:
        return "standard"


def create_single_dataset(
    dataset_config: Any,
    split: str = "train",  # "train" or "test"
) -> BaseDataset:
    """Create a single dataset from specified split using Hydra instantiate.

    Args:
        dataset_config: Dataset configuration object (Hydra-compatible).
        split: "train" or "test" (for logging purposes only).

    Returns:
        BaseDataset (or subclass) instance from the given config.

    Raises:
        ValueError: If dataset_config is None.

    Example:
        >>> train_ds = create_single_dataset(cfg.train_dataset, split="train")
        >>> len(train_ds)
        1000
    """
    from hydra.utils import instantiate

    if dataset_config is None:
        raise ValueError(f"{split}_dataset configuration is missing")

    dataset = instantiate(dataset_config)
    logger.info(
        f"[Dataset] Created {split} dataset: {type(dataset).__name__} "
        f"with {len(dataset)} samples"
    )
    return dataset


def _split_by_sample(
    dataset: Any,
    indices: Any,
    train_ratio: float,
    val_ratio: float,
    include_test: bool,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Split dataset by sample into train/val/test subsets.

    Splits indices into train/val (and optionally test) according to ratios.
    Uses type(dataset).create_subset to preserve shared memory and attributes.

    Args:
        dataset: BaseDataset (or subclass) to split.
        indices: Shuffled array of indices (e.g. np.arange(len(dataset))).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        include_test: If True, remaining indices form test subset.

    Returns:
        Tuple of (train_subset, val_subset) or (train_subset, val_subset,
        test_subset).
    """
    dataset_size = len(dataset)
    train_len = int(train_ratio * dataset_size)
    val_len = int(val_ratio * dataset_size)

    train_idx = indices[:train_len]
    val_idx = indices[train_len : train_len + val_len]

    if include_test:
        test_idx = indices[train_len + val_len :]
        logger.info(
            f"[Dataset] Sample split: train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )
        return (
            type(dataset).create_subset(dataset, train_idx),
            type(dataset).create_subset(dataset, val_idx),
            type(dataset).create_subset(dataset, test_idx),
        )
    else:
        logger.info(
            f"[Dataset] Sample split: train={len(train_idx)}, val={len(val_idx)}"
        )
        return (
            type(dataset).create_subset(dataset, train_idx),
            type(dataset).create_subset(dataset, val_idx),
        )


def _validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    scenario: str,
    tolerance: float = 1e-6,
) -> None:
    """Validate split ratios for specific training scenario.

    Args:
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        scenario: Training scenario ("standard", "pretraining",
            "finetuning", or "few_shot")
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
    elif scenario in ["finetuning", "few_shot"]:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"{scenario} split ratios must sum to 1.0, got "
                f"{total:.4f} (train={train_ratio:.4f}, "
                f"val={val_ratio:.4f}, test={test_ratio:.4f})"
            )
    else:
        raise ValueError(f"Unknown training scenario: {scenario}")


def get_dataset_attribute(
    dataset: Any, attr_name: str, required: bool = False
) -> Any | None:
    """Extract attributes from datasets, handling DataLoader wrappers.

    This utility function provides a universal way to extract attributes
    (e.g., train_ratio, val_ratio, test_ratio, dataset_name) from datasets,
    Subsets, or DataLoaders. It handles DataLoader unwrapping and uses direct
    attribute access, relying on the fact that `create_subset` in
    `base_dataset.py` explicitly copies attributes to Subset instances.

    This eliminates the need for complex traversal logic since attributes
    are propagated to Subset wrappers at creation time, enabling O(1)
    performance with simple getattr.

    Args:
        dataset: Dataset, Subset, or DataLoader object
        attr_name: Name of the attribute to extract (e.g., 'train_ratio',
            'dataset_name')
        required: If True, raises ValueError when attribute is not found. If
            False, returns None.

    Returns:
        Optional[Any]: Attribute value if found, None otherwise (unless
            required=True)

    Raises:
        ValueError: If required=True and attribute not found on dataset

    Examples:
        >>> # Extract train_ratio (required)
        >>> train_ratio = get_dataset_attribute(
        ...     dataloader, 'train_ratio', required=True
        ... )

        >>> # Extract dataset_name (required)
        >>> dataset_name = get_dataset_attribute(
        ...     dataset, 'dataset_name', required=True
        ... )

        >>> # Extract optional attribute
        >>> val_ratio = get_dataset_attribute(
        ...     dataset, 'val_ratio', required=False
        ... )
    """
    # Handle DataLoader by extracting the dataset attribute
    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset

    # Direct attribute access - relies on create_subset copying attributes to
    # Subset instances
    value = getattr(dataset, attr_name, None)

    # Raise error if required and value is None
    if required and value is None:
        raise ValueError(f"Required attribute '{attr_name}' not found on dataset")

    return value
