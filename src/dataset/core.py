"""
Core dataset creation and splitting functions.
"""
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from src.utils.dataset_utils import create_single_dataset, _split_by_sample, _validate_split_ratios
from src.dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def create_training_datasets(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    use_patient_split: bool,
    use_nabnet_vanilla_split: bool,
    seed: int,
    training_scenario: str,
    train_dataset_config = None,
    test_dataset_config = None
) -> tuple:
    """
    Create train/val/test datasets based on training scenario.
    
    Args:
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        use_patient_split: Whether to use patient-level splitting
        use_nabnet_vanilla_split: Whether to use NABNet vanilla splitting
        seed: Random seed for reproducibility
        training_scenario: "standard", "pretraining", or "finetuning"
        train_dataset_config: Configuration for training dataset
        test_dataset_config: Configuration for test dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) or (train_dataset, val_dataset, None)
    """
    logger.info(f"[Dataset] Creating datasets for {training_scenario} scenario")

    # Only create train dataset initially
    train_dataset = create_single_dataset(train_dataset_config, split="train")

    if training_scenario in ["standard", "pretraining"]:
        train_split, val_split = split_dataset(
            train_dataset, train_ratio, val_ratio, test_ratio, use_patient_split, use_nabnet_vanilla_split, seed, training_scenario
        )
        # Only create test dataset when needed for standard/pretraining
        test_dataset = create_single_dataset(test_dataset_config, split="test")
        return train_split, val_split, test_dataset
    elif training_scenario == "finetuning":
        train_split, val_split, test_split = split_dataset(
            train_dataset, train_ratio, val_ratio, test_ratio, use_patient_split, use_nabnet_vanilla_split, seed, training_scenario
        )
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
    train_dataset_config = None,
    test_dataset_config = None
) -> BaseDataset:
    """
    Create test dataset based on training scenario.
    
    Args:
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        use_patient_split: Whether to use patient-level splitting
        use_nabnet_vanilla_split: Whether to use NABNet vanilla splitting
        seed: Random seed for reproducibility
        training_scenario: "standard", "pretraining", or "finetuning"
        train_dataset_config: Configuration for training dataset
        test_dataset_config: Configuration for test dataset
        
    Returns:
        Test dataset (from separate file or extracted from training file)
    """
    logger.info(f"[Dataset] Creating test dataset for {training_scenario} scenario")
    if training_scenario == "finetuning":
        # Extract test from training file
        train_dataset = create_single_dataset(train_dataset_config, split="train")
        
        # Use the main split function to get consistent behavior
        train_split, val_split, test_split = split_dataset(
            train_dataset, train_ratio, val_ratio, test_ratio, use_patient_split, use_nabnet_vanilla_split, seed, "finetuning"
        )
        return test_split
    else:
        # Use separate test file
        return create_single_dataset(test_dataset_config, split="test")

def split_dataset(dataset, train_ratio, val_ratio, test_ratio, use_patient_split, use_nabnet_vanilla_split, seed: int, scenario: str):
    """
    Unified split function: chooses sample, patient, or nabnet vanilla split based on config and dataset capability.
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio, scenario)

    logger.info(f"[Dataset] Splitting dataset with seed {seed}")
    np.random.seed(seed)
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    include_test = (scenario == "finetuning")

    if use_nabnet_vanilla_split:
        if include_test:
            raise ValueError("NABNet vanilla split does not support test split")
        return _split_nabnet_vanilla(dataset, indices, train_ratio, val_ratio, include_test, seed)
    elif use_patient_split:
        if not dataset.supports_patient_split:
            raise ValueError("Patient-level split is not supported for this dataset.")
        # Use sample-by-patient split instead of patient-level split
        return _split_samples_by_patient(dataset, indices, train_ratio, val_ratio, include_test, seed)
    else:
        return _split_by_sample(dataset, indices, train_ratio, val_ratio, include_test)

def _split_samples_by_patient(
    dataset, 
    indices, 
    train_ratio: float, 
    val_ratio: float, 
    include_test: bool,
    seed: int = None
):
    """
    Split samples within each patient according to ratios.
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
        patient_indices = indices[subject_ids == subject]
        np.random.shuffle(patient_indices)
        
        n_samples = len(patient_indices)
        train_split = int(train_ratio * n_samples)
        val_split = int((train_ratio + val_ratio) * n_samples)
        
        # Split patient samples according to ratios
        train_indices.extend(patient_indices[:train_split])
        val_indices.extend(patient_indices[train_split:val_split])
        
        if include_test:
            test_indices.extend(patient_indices[val_split:])
    
    # Create subsets and return
    if include_test:
        logger.info(f"[Dataset] Sample-by-patient split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        return (
            type(dataset).create_subset(dataset, train_indices),
            type(dataset).create_subset(dataset, val_indices),
            type(dataset).create_subset(dataset, test_indices)
        )
    else:
        logger.info(f"[Dataset] Sample-by-patient split: train={len(train_indices)}, val={len(val_indices)}")
        return (
            type(dataset).create_subset(dataset, train_indices),
            type(dataset).create_subset(dataset, val_indices)
        )

def _split_nabnet_vanilla(dataset, indices, train_ratio: float, val_ratio: float, include_test: bool, seed: int = None):
    """
    Split using sklearn's train_test_split for train/val only.
    This mimics the original NABNet vanilla split approach with test_size=0.2 and random_state=42.
    
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
    
    # Use fixed test_size=0.2 and random_state=42 as in the original NABNet implementation
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42
    )
    
    logger.info(f"[Dataset] NABNet vanilla split: train={len(train_indices)}, val={len(val_indices)}")
    return (
        type(dataset).create_subset(dataset, train_indices),
        type(dataset).create_subset(dataset, val_indices)
    )

def get_vitals_dataset(ds):
    """Unwrap torch.utils.data.Subset (and similar wrappers) to get base dataset's vitals_dataset."""
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    if not hasattr(ds, "vitals_dataset"):
        raise AttributeError(f"{type(ds).__name__} lacks .vitals_dataset")
    return ds.vitals_dataset