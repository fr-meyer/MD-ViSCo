#!/usr/bin/env python3
"""
Main training script for MD-ViSCo project.

This script serves as the entry point for launching training experiments.
It handles configuration loading, data/model setup, and invokes the appropriate trainer.

Usage (torchrun for everything - even single GPU):
    # Single GPU (use torchrun even for single GPU)
    torchrun --standalone --nproc_per_node=1 src/train.py dataset=pulsedb
    torchrun --standalone --nproc_per_node=1 src/train.py dataset=uci model_type=refinement
    
    # Multi-GPU
    torchrun --standalone --nproc_per_node=2 src/train.py dataset=pulsedb
    
    # CPU-only
    CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=1 src/train.py dataset=pulsedb

Note: The trainer now handles all runtime orchestration (hardware setup, DDP, training loops).
This script is a thin entry point that delegates to the trainer.
"""

# Standard library imports
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Any

# Third-party imports
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Local imports
from .conf.config import Config
from src.dataset import create_training_datasets, _determine_training_scenario, import_datasets
from src.model import import_models
from src.criterions import import_criterions
from src.trainers import import_trainers
from src.core import register_core
from src.core.direction import Directions

# Import modules to trigger ConfigStore registration
import src.utils.checkpoint_manager
import src.loggings.progress_bar

# Trigger ConfigStore registration at module import time
register_core()
# core_imported = import_core()
datasets_imported = import_datasets()
models_imported = import_models()
criterions_imported = import_criterions()
trainers_imported = import_trainers()


def setup_logging(log_level: str, log_file_path: str) -> None:
    """Setup logging configuration with Hydra-compatible DDP support.
    
    Args:
        log_level: Logging level (e.g., 'INFO', 'DEBUG')
        log_file_path: Path to log file
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path)
        ]
    )


def create_directories(dirs: List[str]) -> None:
    """Create required directories.
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)


def setup_debug_environment() -> None:
    """Setup debug environment with limited thread usage.
    
    Configures environment variables and PyTorch settings for debugging.
    """
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    logging.info("Debug mode enabled - limited thread usage")


def validate_config(cfg: Config) -> None:
    """Validate configuration - fail early before any instantiation.
    
    Args:
        cfg: Configuration object to validate
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    # Check if dataset is provided
    if cfg.train_dataset is None:
        raise ValueError("train_dataset configuration is required")
    elif cfg.train_dataset.dataset_name is None:
        raise ValueError("train_dataset.dataset_name is required")

    if cfg.test_dataset is None and not cfg.trainer.is_finetuning:
        raise ValueError("test_dataset configuration is required")
    elif cfg.test_dataset.dataset_name is None:
        raise ValueError("test_dataset.dataset_name is required")
    
    if cfg.trainer.directions is None:
        raise ValueError("directions is required")
    
    # Validate direction mode configuration
    if not hasattr(cfg.trainer, 'direction_mode'):
        raise ValueError("trainer.direction_mode is required")
    
    # Validate direction mode values
    valid_direction_modes = ["single", "multi"]
    if cfg.trainer.direction_mode not in valid_direction_modes:
        raise ValueError(f"Invalid direction_mode: {cfg.trainer.direction_mode}. Must be one of: {valid_direction_modes}")
    
    # Validate multi-directional logic
    if cfg.trainer.direction_mode == "multi":
        if len(cfg.trainer.directions.active_directions) <= 1:
            raise ValueError("direction_mode is 'multi', but only one direction is provided")
    elif cfg.trainer.direction_mode == "single":
        if len(cfg.trainer.directions.active_directions) > 1:
            raise ValueError("direction_mode is 'single', but multiple directions are provided")
    
    # Validate required trainer components
    if cfg.trainer.model is None:
        raise ValueError("trainer.model is required")
    if cfg.trainer.criterion is None:
        raise ValueError("trainer.criterion is required")
    if cfg.trainer.checkpoint_manager is None:
        raise ValueError("trainer.checkpoint_manager is required")
    

    if cfg.trainer.input_preprocessing is None:
        raise ValueError("input_preprocessing is required")
    
    logging.info(f"Configuration validated successfully")


def setup_environment(cfg: Config) -> None:
    """Setup training environment.
    
    Args:
        cfg: Configuration object containing environment settings
    """
    if cfg.trainer.debug:
        setup_debug_environment()
    
    create_directories(['checkpoints', 'logs', 'results'])
    
    if cfg.trainer.hydra_run_dir:
        Path(cfg.trainer.hydra_run_dir).mkdir(parents=True, exist_ok=True)
    
    logging.info("Environment setup completed")


def log_training_configuration(cfg: Config) -> None:
    """Log comprehensive training configuration.
    
    Args:
        cfg: Configuration object to log
    """
    logging.info("=" * 60)
    logging.info("MD-ViSCo Training Configuration")
    logging.info("=" * 60)
    logging.info(f"Working Directory: {os.getcwd()}")
    logging.info(f"Python Path: {sys.path[0]}")
    logging.info(f"Configuration: {OmegaConf.to_yaml(cfg, resolve=True)}")
    
    
    # Training configuration
    trainer_name = cfg.trainer.trainer_name
    logging.info(f"Using {trainer_name} trainer")
    
    # Dataset configuration
    log_dataset_configuration(cfg)


def log_dataset_configuration(cfg: Config) -> None:
    """Log dataset-specific configuration.
    
    Args:
        cfg: Configuration object containing dataset settings
    """
    logging.info("=" * 60)
    logging.info("Dataset Configuration")
    logging.info("=" * 60)
    logging.info(f"Dataset Name: {getattr(cfg.train_dataset, 'dataset_name', 'Unknown')}")
    
    logging.info(f"Input Size: {getattr(cfg.train_dataset, 'input_size', 'Unknown')}")
    dbp_min = getattr(cfg.train_dataset, 'dbp_min', 'Unknown')
    sbp_max = getattr(cfg.train_dataset, 'sbp_max', 'Unknown')
    logging.info(f"BP Range: DBP min={dbp_min}, SBP max={sbp_max}")
    
    logging.info(f"Trainer Name: {cfg.trainer.trainer_name}")
    logging.info(f"Model Name: {cfg.trainer.model.model_name}")
    logging.info(f"Direction: {cfg.trainer.directions.active_directions}")
    logging.info(f"Path Folder: {cfg.trainer.checkpoint_manager.base_dir if hasattr(cfg.trainer, 'checkpoint_manager') and cfg.trainer.checkpoint_manager else 'Not set'}")
    logging.info("=" * 60)


def _check_directions_support(dataset: Any, directions: Directions) -> bool:
    """Check if a dataset supports the given directions.
    
    Args:
        dataset: Dataset object to check
        directions: Directions object to validate against
        
    Returns:
        True if dataset supports the directions, False otherwise
    """
    from torch.utils.data import Subset
    
    # Handle Subset wrappers by accessing the underlying dataset
    if isinstance(dataset, Subset):
        underlying_dataset = dataset.dataset  # PyTorch uses .dataset, not _dataset
        return underlying_dataset.vitals_dataset.supports_directions(directions)
    
    # Direct access for BaseDataset instances
    return dataset.vitals_dataset.supports_directions(directions)

def create_datasets(cfg: Config, directions: Directions) -> Tuple[Any, ...]:
    """Create training datasets.
    
    Args:
        cfg: Configuration object containing dataset settings
        directions: Directions object for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Raises:
        ValueError: If dataset doesn't support the specified directions
    """
    # Extract config values from dataset config
    train_ratio = cfg.train_dataset.train_ratio
    val_ratio = cfg.train_dataset.val_ratio
    test_ratio = cfg.train_dataset.test_ratio
    use_patient_split = cfg.train_dataset.use_patient_split
    use_nabnet_vanilla_split = cfg.train_dataset.use_nabnet_vanilla_split
    seed = cfg.trainer.seed

    training_scenario = _determine_training_scenario(
        is_pretraining=cfg.trainer.is_pretraining,
        is_finetuning=cfg.trainer.is_finetuning
    )
    logging.info(f"Training scenario: {training_scenario}")
    
    dataset_tuple = create_training_datasets(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        use_patient_split=use_patient_split,
        use_nabnet_vanilla_split=use_nabnet_vanilla_split,
        seed=seed,
        training_scenario=training_scenario,
        train_dataset_config=cfg.train_dataset,
        test_dataset_config=cfg.test_dataset
    )
    logging.info(f"Created datasets using Hydra instantiation: {len(dataset_tuple)} datasets")
    
    # Log dataset path
    if dataset_tuple and len(dataset_tuple) > 0:
        train_dataset = dataset_tuple[0]
        logging.info(f"Train Dataset Path: {train_dataset.sample_file}")
        val_dataset = dataset_tuple[1]
        logging.info(f"Val Dataset Path: {val_dataset.sample_file}")
        test_dataset = dataset_tuple[2]
        logging.info(f"Test Dataset Path: {test_dataset.sample_file if test_dataset else 'None'}")

        if not _check_directions_support(train_dataset, directions):
            raise ValueError(f"Train dataset does not support directions: {directions}")
        if val_dataset and not _check_directions_support(val_dataset, directions):
            raise ValueError(f"Val dataset does not support directions: {directions}")
        if test_dataset and not _check_directions_support(test_dataset, directions):
            raise ValueError(f"Test dataset does not support directions: {directions}")

    return dataset_tuple


def initialize_wandb(trainer: Any, cfg: Config) -> None:
    """Initialize WandB logging if available and enabled.
    
    Args:
        trainer: Trainer object containing progress bar with WandB
        cfg: Configuration object for WandB initialization
    """
    rank = int(os.environ.get("RANK", "0"))
    is_rank0 = rank == 0
    
    if (hasattr(trainer, 'progress_bar') and 
        trainer.progress_bar and 
        trainer.progress_bar.wandb and
        trainer.progress_bar.wandb.wandb_enabled):
        trainer.progress_bar.wandb.initialize(is_rank0, cfg)
        logging.info(f"WandB initialized at main level (rank {rank})")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """Main training function - thin entry point that delegates to trainer.
    
    The trainer now owns all runtime orchestration:
    - Hardware setup (seed, threads, device binding via LOCAL_RANK)
    - DDP lifecycle (init, wrap, barriers, cleanup)
    - DataLoader construction with DistributedSampler
    - Optimizer/scheduler creation (post-DDP wrap)
    - Training loops, metric sync, early stopping
    - Checkpointing (rank-0 save + barrier, all-rank load)
    
    Args:
        cfg: Configuration object from Hydra
        
    Raises:
        ValueError: If configuration validation fails
        RuntimeError: If runtime errors occur during training
    """
    try:
        validate_config(cfg)
        # 1. Setup Phase (rank-agnostic)
        setup_logging(cfg.trainer.logging_level, cfg.trainer.log_file_path)
        setup_environment(cfg)
        
        # 2. Log ConfigStore registration status
        logging.info(f"Imported core, {models_imported} models, {criterions_imported} criterions, {trainers_imported} trainers")
        
        # 3. Configuration Logging Phase
        log_training_configuration(cfg)
        
        # 4. Model and Data Setup Phase
        trainer = instantiate(cfg.trainer)  # Hydra handles model instantiation
        
        # 5. Initialize WandB logging if available
        initialize_wandb(trainer, cfg)
        
        # 6. Create datasets
        dataset_tuple = create_datasets(cfg, trainer.directions)
        
        # 7. Training Phase (trainer handles everything: hardware, DDP, training)
        trainer.run_training(dataset_tuple)
        
    except ValueError as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        raise
    except RuntimeError as e:
        logging.error(f"Runtime error during training: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
