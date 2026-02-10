"""Core dataset creation and splitting functions.

This module provides the main entry points for building train/val/test datasets
from Hydra config: create_training_datasets (builds splits for a given training
scenario), create_test_dataset (builds the test set), and split_dataset (unified
sample-, patient-, or NABNet-vanilla splitting). Splitting respects use_patient_split
and use_nabnet_vanilla_split and validates split ratios per scenario.

Functions:
    - create_training_datasets: Create train/val (and optionally test) datasets
    - create_test_dataset: Create test dataset for standard/finetuning/few_shot
    - split_dataset: Split a dataset into train/val (and optionally test) subsets

Examples:
    >>> train_ds, val_ds, test_ds = create_training_datasets(
    ...     train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, ...
    ... )

See Also:
    - src.dataset.base_dataset: BaseDataset, DatasetBaseConfig
    - src.utils.dataset_utils: create_single_dataset, _split_by_sample
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
from torch.utils.data import Subset

# Local imports
from src.dataset.base_dataset import BaseDataset
from src.utils.dataset_utils import _split_by_sample
from src.utils.dataset_utils import _validate_split_ratios
from src.utils.dataset_utils import create_single_dataset

logger = logging.getLogger(__name__)

# Type alias for dataset or subset returned by splitting
DatasetOrSubset = BaseDataset | Subset[BaseDataset]


def create_training_datasets(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    use_patient_split: bool,
    use_nabnet_vanilla_split: bool,
    seed: int,
    training_scenario: str,
    train_dataset_config: Any = None,
    test_dataset_config: Any = None,
) -> tuple[DatasetOrSubset, DatasetOrSubset, DatasetOrSubset | None]:
    """Create train/val/test datasets based on training scenario.

    Args:
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        use_patient_split: Whether to use patient-level splitting
        use_nabnet_vanilla_split: Whether to use NABNet vanilla splitting
        seed: Random seed for reproducibility
        training_scenario: "standard", "pretraining", "finetuning", or "few_shot"
        train_dataset_config: Configuration for training dataset
        test_dataset_config: Configuration for test dataset

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) or
        (train_dataset, val_dataset, None)

    Raises:
        ValueError: If ``training_scenario`` is not one of "standard", "pretraining",
            "finetuning", or "few_shot".
        ValueError: If split ratios are invalid, or if patient-level split
            is requested but the dataset does not support it
            (e.g. ``supports_patient_split`` is False).
        ValueError: If NABNet vanilla split is used with a scenario that requires a test
            split (e.g. finetuning, few_shot).
    """
    logger.info(f"[Dataset] Creating datasets for {training_scenario} scenario")

    train_dataset = create_single_dataset(train_dataset_config, split="train")

    if training_scenario in ["standard", "pretraining"]:
        splits = split_dataset(
            train_dataset,
            train_ratio,
            val_ratio,
            test_ratio,
            use_patient_split,
            use_nabnet_vanilla_split,
            seed,
            training_scenario,
        )
        assert len(splits) == 2
        train_split, val_split = splits[0], splits[1]
        test_dataset = create_single_dataset(test_dataset_config, split="test")
        return train_split, val_split, test_dataset
    elif training_scenario in ["finetuning", "few_shot"]:
        splits = split_dataset(
            train_dataset,
            train_ratio,
            val_ratio,
            test_ratio,
            use_patient_split,
            use_nabnet_vanilla_split,
            seed,
            training_scenario,
        )
        assert len(splits) == 3
        train_split, val_split, test_split = splits[0], splits[1], splits[2]
        return train_split, val_split, test_split
    else:
        raise ValueError(f"Unknown training scenario: {training_scenario}")


def create_test_dataset(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    use_patient_split: bool,
    use_nabnet_vanilla_split: bool,
    seed: int,
    training_scenario: str,
    train_dataset_config: Any = None,
    test_dataset_config: Any = None,
) -> DatasetOrSubset:
    """Create test dataset based on training scenario.

    Args:
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        use_patient_split: Whether to use patient-level splitting
        use_nabnet_vanilla_split: Whether to use NABNet vanilla splitting
        seed: Random seed for reproducibility
        training_scenario: "standard", "pretraining", "finetuning", or "few_shot"
        train_dataset_config: Configuration for training dataset
        test_dataset_config: Configuration for test dataset

    Returns:
        Test dataset (from separate file or extracted from training file)

    Raises:
        ValueError: If split ratios are invalid (when splitting is used for
            finetuning/few_shot), or if patient-level split is requested but the
            dataset does not support it (e.g. ``supports_patient_split`` is False).
        ValueError: If NABNet vanilla split is used with finetuning/few_shot
            (test split not supported).
    """
    logger.info(f"[Dataset] Creating test dataset for {training_scenario} scenario")
    if training_scenario in ["finetuning", "few_shot"]:
        train_dataset = create_single_dataset(train_dataset_config, split="train")

        # Use the main split function to get consistent behavior
        splits = split_dataset(
            train_dataset,
            train_ratio,
            val_ratio,
            test_ratio,
            use_patient_split,
            use_nabnet_vanilla_split,
            seed,
            training_scenario,
        )
        assert len(splits) == 3
        _train_split, _val_split, test_split = splits[0], splits[1], splits[2]
        return test_split
    else:
        # Use separate test file
        return create_single_dataset(test_dataset_config, split="test")


def split_dataset(
    dataset: BaseDataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    use_patient_split: bool,
    use_nabnet_vanilla_split: bool,
    seed: int,
    scenario: str,
) -> (
    tuple[DatasetOrSubset, DatasetOrSubset]
    | tuple[DatasetOrSubset, DatasetOrSubset, DatasetOrSubset]
):
    """Unified split function: chooses sample, patient, or NABNet vanilla split.

    Splits the dataset into train/val (and optionally test) subsets according to
    scenario and config. Validates split ratios via _validate_split_ratios before
    applying the chosen strategy.

    Args:
        dataset: BaseDataset (or subclass) to split.
        train_ratio: Fraction of data for training (0.0--1.0).
        val_ratio: Fraction of data for validation (0.0--1.0).
        test_ratio: Fraction of data for test (0.0--1.0). Must sum to 1.0
            with train/val.
        use_patient_split: If True, use patient-level splitting when supported.
        use_nabnet_vanilla_split: If True, use NABNet fixed 80/20
            train/val split.
        seed: Random seed for reproducibility.
        scenario: One of "standard", "pretraining", "finetuning", "few_shot".
            "few_shot" and "finetuning" may produce a test subset;
            "standard"/"pretraining" use a separate test file and do not
            include test in this split.

    Returns:
        Tuple of (train_subset, val_subset) or (train_subset, val_subset, test_subset)
        as :class:`torch.utils.data.Subset` instances.

    Raises:
        ValueError: If split ratios are invalid, scenario is unknown, or
            patient split is requested but dataset does not support it.
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio, scenario)

    logger.info(f"[Dataset] Splitting dataset with seed {seed}")
    np.random.seed(seed)
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    include_test = scenario in ["finetuning", "few_shot"]

    if scenario == "few_shot":
        # For few-shot, use patient-level splitting to ensure all patients contribute
        if not dataset.supports_patient_split:
            raise ValueError(
                "Few-shot scenario requires patient-level splitting support"
            )
        return _split_samples_by_patient(
            dataset, indices, train_ratio, val_ratio, include_test, seed
        )
    elif use_nabnet_vanilla_split:
        if include_test:
            raise ValueError("NABNet vanilla split does not support test split")
        return _split_nabnet_vanilla(
            dataset, indices, train_ratio, val_ratio, include_test, seed
        )
    elif use_patient_split:
        if not dataset.supports_patient_split:
            raise ValueError("Patient-level split is not supported for this dataset.")
        # Use sample-by-patient split instead of patient-level split
        return _split_samples_by_patient(
            dataset, indices, train_ratio, val_ratio, include_test, seed
        )
    else:
        return _split_by_sample(dataset, indices, train_ratio, val_ratio, include_test)


def _split_samples_by_patient(
    dataset,
    indices,
    train_ratio: float,
    val_ratio: float,
    include_test: bool,
    seed: int | None = None,
):
    """Split samples within each patient according to ratios.

    Each patient contributes samples to all splits.

    Args:
        dataset: Dataset with subject_ids
        indices: Array of sample indices
        train_ratio: Ratio of samples per patient for training
        val_ratio: Ratio of samples per patient for validation
        include_test: Whether to include test split
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subset, val_subset) or (train_subset, val_subset, test_subset)
    """
    if seed is not None:
        logger.info(f"[Dataset] Setting seed to {seed}")
        np.random.seed(seed)

    subject_ids = dataset.data["subject_ids"]
    unique_subjects = np.unique(subject_ids)

    train_indices = []
    val_indices = []
    test_indices = []

    # For each patient, split their samples according to ratios
    for subject in unique_subjects:
        patient_indices = indices[subject_ids[indices] == subject]
        np.random.shuffle(patient_indices)

        n_samples = len(patient_indices)
        train_split = int(train_ratio * n_samples)
        val_split = int((train_ratio + val_ratio) * n_samples)

        # Split patient samples according to ratios
        train_indices.extend(patient_indices[:train_split])
        val_indices.extend(patient_indices[train_split:val_split])

        if include_test:
            test_indices.extend(patient_indices[val_split:])

    if include_test:
        logger.info(
            f"[Dataset] Sample-by-patient split: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )
        return (
            type(dataset).create_subset(dataset, train_indices),
            type(dataset).create_subset(dataset, val_indices),
            type(dataset).create_subset(dataset, test_indices),
        )
    else:
        logger.info(
            f"[Dataset] Sample-by-patient split: train={len(train_indices)}, "
            f"val={len(val_indices)}"
        )
        return (
            type(dataset).create_subset(dataset, train_indices),
            type(dataset).create_subset(dataset, val_indices),
        )


def _split_nabnet_vanilla(
    dataset,
    indices,
    train_ratio: float,
    val_ratio: float,
    include_test: bool,
    seed: int | None = None,
):
    """Split using sklearn's train_test_split for train/val only.

    This mimics the original NABNet vanilla split approach with
    test_size=0.2 and random_state=42.

    Args:
        dataset: Dataset to split
        indices: Array of sample indices
        train_ratio: Ratio of samples for training (ignored, uses fixed 0.8)
        val_ratio: Ratio of samples for validation (ignored, uses fixed 0.2)
        include_test: Whether to include test split (ignored for this function)
        seed: Random seed for reproducibility (ignored for split)

    Returns:
        Tuple of (train_subset, val_subset)
    """
    from sklearn.model_selection import train_test_split

    # Use fixed test_size=0.2 and random_state=42 as in the original
    # NABNet implementation
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    logger.info(
        f"[Dataset] NABNet vanilla split: train={len(train_indices)}, "
        f"val={len(val_indices)}"
    )
    return (
        type(dataset).create_subset(dataset, train_indices),
        type(dataset).create_subset(dataset, val_indices),
    )
