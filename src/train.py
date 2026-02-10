#!/usr/bin/env python3
"""Main training script for MD-ViSCo project.

This script serves as the entry point for launching training experiments.
It handles configuration loading, data/model setup, and invokes the appropriate trainer.

Usage (torchrun for everything - even single GPU):
    # Single GPU (use torchrun even for single GPU)
    torchrun --standalone --nproc_per_node=1 src/train.py \\
        train_dataset=train_pulsedb test_dataset=test_pulsedb \\
        trainer=approximation_trainer_mdvisco
    torchrun --standalone --nproc_per_node=1 src/train.py \\
        train_dataset=train_uci test_dataset=test_uci \\
        trainer=refinement_trainer_nabnet

    # Multi-GPU
    torchrun --standalone --nproc_per_node=2 src/train.py \\
        train_dataset=train_pulsedb test_dataset=test_pulsedb \\
        trainer=approximation_trainer_mdvisco

    # CPU-only
    CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=1 \\
        src/train.py train_dataset=train_pulsedb test_dataset=test_pulsedb \\
        trainer=approximation_trainer_mdvisco

Note: The trainer now handles all runtime orchestration (hardware setup, DDP,
training loops).
This script is a thin entry point that delegates to the trainer.
"""

# Standard library imports
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Third-party imports
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Subset

# Local imports
from src.conf.resolvers import register_resolvers
from src.core import register_core
from src.core.direction import Directions
from src.criterions import import_criterions
from src.dataset import _determine_training_scenario
from src.dataset import create_training_datasets
from src.dataset import import_datasets
from src.model import import_models
from src.optimizers import import_optimizers
from src.preprocessors import import_preprocessors
from src.processors import import_extractors
from src.processors import import_processors
from src.schedulers import import_schedulers
from src.trainers import import_trainers
from src.utils.collate_utils import build_vital_channel_mapping
from src.utils.dataset_utils import get_dataset_attribute

from .conf.config import Config

register_resolvers()

register_core()
datasets_imported = import_datasets()
models_imported = import_models()
criterions_imported = import_criterions()
optimizers_imported = import_optimizers()
schedulers_imported = import_schedulers()
trainers_imported = import_trainers()
processors_imported = import_processors()
extractors_imported = import_extractors()
preprocessors_imported = import_preprocessors()


def setup_logging(log_level: str, log_file_path: str) -> None:
    """Set up logging configuration with Hydra-compatible DDP support.

    Args:
        log_level: Logging level (e.g., 'INFO', 'DEBUG')
        log_file_path: Path to log file
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path),
        ],
    )


def create_directories(dirs: list[str]) -> None:
    """Create required directories.

    Args:
        dirs: List of directory paths to create
    """
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)


def setup_debug_environment() -> None:
    """Set up debug environment with limited thread usage.

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
    if cfg.train_dataset is None:
        raise ValueError("train_dataset configuration is required")
    elif cfg.train_dataset.dataset_name is None:
        raise ValueError("train_dataset.dataset_name is required")

    if cfg.test_dataset is None:
        if cfg.trainer is not None and not cfg.trainer.is_finetuning:
            raise ValueError("test_dataset configuration is required")
    elif cfg.test_dataset.dataset_name is None:
        raise ValueError("test_dataset.dataset_name is required")

    if cfg.trainer is not None:
        if cfg.trainer.directions is None:
            raise ValueError("directions is required")

        if not hasattr(cfg.trainer, "direction_mode"):
            raise ValueError("trainer.direction_mode is required")

        valid_direction_modes = ["single", "multi"]
        if cfg.trainer.direction_mode not in valid_direction_modes:
            raise ValueError(
                f"Invalid direction_mode: {
                    cfg.trainer.direction_mode
                }. Must be one of: {valid_direction_modes}"
            )

        if cfg.trainer.direction_mode == "multi":
            if (
                cfg.trainer.directions is not None
                and len(cfg.trainer.directions.active_directions) <= 1
            ):
                raise ValueError(
                    "direction_mode is 'multi', but only one direction is provided"
                )
        elif (
            cfg.trainer.direction_mode == "single"
            and cfg.trainer.directions is not None
            and len(cfg.trainer.directions.active_directions) > 1
        ):
            raise ValueError(
                "direction_mode is 'single', but multiple directions are provided"
            )

        if cfg.trainer.model is None:
            raise ValueError("trainer.model is required")
        if cfg.trainer.criterion is None:
            raise ValueError("trainer.criterion is required")
        if cfg.trainer.optimizer is None:
            raise ValueError("trainer.optimizer is required")

        if not cfg.trainer.checkpoint_managers:
            logging.warning(
                "No checkpoint_managers configured. Checkpoint operations will "
                "be skipped."
            )
        else:
            logging.info(
                f"Checkpoint managers configured: {
                    list(cfg.trainer.checkpoint_managers.keys())
                }"
            )

        if cfg.trainer.scheduler is None:
            logging.info(
                "No scheduler configured - training without learning rate scheduling"
            )
        else:
            logging.info(f"Scheduler configured: {cfg.trainer.scheduler._target_}")

        if cfg.trainer.input_preprocessing is None:
            raise ValueError("input_preprocessing is required")

    logging.info("Configuration validated successfully")


def setup_environment(cfg: Config) -> None:
    """Set up training environment.

    Args:
        cfg: Configuration object containing environment settings
    """
    if cfg.trainer is not None and cfg.trainer.debug:
        setup_debug_environment()

    create_directories(["checkpoints", "logs", "results"])

    if cfg.trainer is not None and cfg.trainer.hydra_run_dir:
        Path(cfg.trainer.hydra_run_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Environment setup completed")


def log_training_configuration(cfg: Config) -> None:
    """Log comprehensive training configuration.

    Args:
        cfg: Configuration object to log
    """
    from src.utils.config_logging import log_config_section

    log_config_section(
        "MD-ViSCo Training Configuration",
        {
            "Working Directory": os.getcwd(),
            "Python Path": sys.path[0],
            "Configuration": OmegaConf.to_yaml(cfg, resolve=True),
            "Trainer": (
                cfg.trainer.trainer_name if cfg.trainer is not None else "Unknown"
            ),
        },
    )

    # Dataset configuration
    log_dataset_configuration(cfg)


def log_dataset_configuration(cfg: Config) -> None:
    """Log dataset-specific configuration.

    Args:
        cfg: Configuration object containing dataset settings
    """
    from src.utils.config_logging import log_config_section

    dbp_min = getattr(cfg.train_dataset, "dbp_min", "Unknown")
    sbp_max = getattr(cfg.train_dataset, "sbp_max", "Unknown")
    checkpoint_managers_info = (
        list(cfg.trainer.checkpoint_managers.keys())
        if cfg.trainer is not None and cfg.trainer.checkpoint_managers
        else "None"
    )

    log_config_section(
        "Dataset Configuration",
        {
            "Dataset Name": getattr(cfg.train_dataset, "dataset_name", "Unknown"),
            "Input Size": getattr(cfg.train_dataset, "input_size", "Unknown"),
            "BP Range": f"DBP min={dbp_min}, SBP max={sbp_max}",
            "Trainer Name": (
                cfg.trainer.trainer_name if cfg.trainer is not None else "Unknown"
            ),
            "Model Name": (
                cfg.trainer.model.model_name
                if cfg.trainer is not None and cfg.trainer.model is not None
                else "Unknown"
            ),
            "Direction": (
                cfg.trainer.directions.active_directions
                if cfg.trainer is not None and cfg.trainer.directions is not None
                else "Unknown"
            ),
            "Checkpoint Managers": checkpoint_managers_info,
            "Optimizer": (
                cfg.trainer.optimizer._target_
                if cfg.trainer is not None and cfg.trainer.optimizer
                else "None"
            ),
            "Scheduler": (
                cfg.trainer.scheduler._target_
                if cfg.trainer is not None and cfg.trainer.scheduler
                else "None"
            ),
        },
    )


def _validate_directions_with_preprocessing(
    input_preprocessing: dict,
    directions: Directions,
    dataset_preprocessing: dict | None = None,
) -> None:
    """Ensure Directions are compatible with preprocessing-defined mapping.

    Args:
        input_preprocessing: Trainer input preprocessing configuration.
        directions: Directions object defining source→target flows.

    Raises:
        ValueError: If any direction references a vital that is not present in the
            preprocessing mapping.
    """
    if input_preprocessing is None:
        raise ValueError("input_preprocessing is required to validate directions.")
    if directions is None or not getattr(directions, "directions", None):
        raise ValueError("Directions are required for training but none were provided.")

    mapping = build_vital_channel_mapping(input_preprocessing)
    available_vitals = set(mapping.keys())

    if dataset_preprocessing:
        dataset_mapping = build_vital_channel_mapping(dataset_preprocessing)
        dataset_vitals = set(dataset_mapping.keys())
        missing_in_dataset = [v for v in available_vitals if v not in dataset_vitals]
        if missing_in_dataset:
            raise ValueError(
                "Trainer input_preprocessing references vitals absent from "
                "dataset input_preprocessing.\n"
                f"Dataset provides: {[v.name for v in dataset_vitals]}\n"
                f"Trainer requires: {[v.name for v in available_vitals]}\n"
                f"Missing in dataset: {[v.name for v in missing_in_dataset]}"
            )

    missing_requirements = []
    for direction in directions.directions:
        # `direction.source` always resolves to List[Vital]
        missing_vitals = [v for v in direction.source if v not in available_vitals]
        if missing_vitals:
            direction_key = (
                direction.key() if hasattr(direction, "key") else str(direction)
            )
            missing_requirements.append(
                f"{direction_key}: missing {[v.name for v in missing_vitals]}"
            )

    if missing_requirements:
        raise ValueError(
            "Directions reference vitals that are not present in "
            "input_preprocessing['source'].\n"
            f"Available vitals: {[v.name for v in available_vitals]}\n"
            f"Issues: {missing_requirements}"
        )


def _get_vitals_dataset_for_direction_check(dataset: Any) -> Any | None:
    """Resolve vitals_dataset from a dataset, unwrapping Subset (or similar) wrappers.

    When datasets are split, train/val (and sometimes test) are returned as
    torch.utils.data.Subset instances. Subset does not expose vitals_dataset;
    the underlying dataset does. This helper mirrors prior _check_directions_support
    logic: if the dataset is a Subset, use dataset.dataset.vitals_dataset;
    otherwise use dataset.vitals_dataset.

    Args:
        dataset: A dataset instance (possibly a Subset of a base dataset).

    Returns:
        The vitals_dataset object for direction checks, or None if not found.
    """
    if dataset is None:
        return None
    if isinstance(dataset, Subset):
        underlying = getattr(dataset, "dataset", None)
        if underlying is not None:
            return getattr(underlying, "vitals_dataset", None)
        return None
    return getattr(dataset, "vitals_dataset", None)


def create_datasets(cfg: Config, directions: Directions) -> tuple[Any, ...]:
    """Create training datasets.

    Args:
        cfg: Configuration object containing dataset settings
        directions: Directions object for validation

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Raises:
        ValueError: If dataset doesn't support the specified directions
    """
    if cfg.train_dataset is None or cfg.trainer is None:
        raise ValueError("train_dataset and trainer configuration are required")
    td, tr = cfg.train_dataset, cfg.trainer
    train_ratio = td.train_ratio
    val_ratio = td.val_ratio
    test_ratio = td.test_ratio
    use_patient_split = td.use_patient_split
    use_nabnet_vanilla_split = td.use_nabnet_vanilla_split
    seed = int(tr.seed) if tr.seed is not None else 42

    training_scenario = _determine_training_scenario(
        is_pretraining=tr.is_pretraining,
        is_finetuning=tr.is_finetuning,
        is_few_shot=tr.is_few_shot,
    )
    logging.info(f"Training scenario: {training_scenario}")

    _validate_directions_with_preprocessing(
        tr.input_preprocessing,
        directions,
        getattr(td, "input_preprocessing", None),
    )

    dataset_tuple = create_training_datasets(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        use_patient_split=use_patient_split,
        use_nabnet_vanilla_split=use_nabnet_vanilla_split,
        seed=seed,
        training_scenario=training_scenario,
        train_dataset_config=td,
        test_dataset_config=cfg.test_dataset,
    )
    logging.info(
        f"Created datasets using Hydra instantiation: {len(dataset_tuple)} datasets"
    )

    # Log dataset path (Subset does not have sample_file; use get_dataset_attribute)
    if dataset_tuple and len(dataset_tuple) > 0:
        train_ds = dataset_tuple[0]
        val_ds = dataset_tuple[1]
        test_ds = dataset_tuple[2] if len(dataset_tuple) > 2 else None
        train_path = get_dataset_attribute(train_ds, "sample_file", required=False)
        val_path = get_dataset_attribute(val_ds, "sample_file", required=False)
        test_path = (
            get_dataset_attribute(test_ds, "sample_file", required=False)
            if test_ds
            else None
        )
        logging.info(f"Train Dataset Path: {train_path}")
        logging.info(f"Val Dataset Path: {val_path}")
        logging.info(f"Test Dataset Path: {test_path if test_path else 'None'}")

    train_dataset = dataset_tuple[0]
    val_dataset = dataset_tuple[1]
    test_dataset = dataset_tuple[2] if len(dataset_tuple) > 2 else None

    train_vitals_dataset = _get_vitals_dataset_for_direction_check(train_dataset)
    if train_vitals_dataset is None:
        raise ValueError("train_dataset does not have vitals_dataset attribute")
    if not train_vitals_dataset.supports_directions(directions):
        raise ValueError(
            f"train_dataset does not support the provided directions: {
                directions.directions if directions is not None else 'Unknown'
            }"
        )

    val_vitals_dataset = _get_vitals_dataset_for_direction_check(val_dataset)
    if val_vitals_dataset is None:
        raise ValueError("val_dataset does not have vitals_dataset attribute")
    if not val_vitals_dataset.supports_directions(directions):
        raise ValueError(
            f"val_dataset does not support the provided directions: {
                directions.directions if directions is not None else 'Unknown'
            }"
        )

    if test_dataset is not None:
        test_vitals_dataset = _get_vitals_dataset_for_direction_check(test_dataset)
        if test_vitals_dataset is None:
            raise ValueError("test_dataset does not have vitals_dataset attribute")
        if not test_vitals_dataset.supports_directions(directions):
            raise ValueError(
                f"test_dataset does not support the provided directions: {
                    directions.directions if directions is not None else 'Unknown'
                }"
            )

    return dataset_tuple


def initialize_wandb(trainer: Any, cfg: Config) -> None:
    """Initialize WandB logging if available and enabled.

    Args:
        trainer: Trainer object containing progress bar with WandB
        cfg: Configuration object for WandB initialization
    """
    from src.loggings.wandb_wrapper import initialize_wandb_if_enabled

    rank = int(os.environ.get("RANK", "0"))
    is_rank0 = rank == 0
    initialize_wandb_if_enabled(trainer, cfg, is_rank0)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """Execute main training workflow.

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
        dataset_layout = getattr(cfg.train_dataset, "input_preprocessing", None)
        if (
            cfg.trainer is not None
            and OmegaConf.is_missing(cfg.trainer, "input_preprocessing")
            and dataset_layout
        ):
            cfg.trainer.input_preprocessing = dataset_layout

        validate_config(cfg)
        # 1. Setup Phase (rank-agnostic)
        setup_logging(
            cfg.trainer.logging_level if cfg.trainer is not None else "INFO",
            cfg.trainer.log_file_path if cfg.trainer is not None else "training.log",
        )
        setup_environment(cfg)

        # 2. Log ConfigStore registration status
        logging.info(
            f"Imported core, {models_imported} models, {
                criterions_imported
            } criterions, {trainers_imported} trainers, {
                processors_imported
            } processors, {extractors_imported} extractors"
        )

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
