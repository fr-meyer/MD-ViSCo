"""MD-ViSCo Trainer with Canonical PyTorch Checkpoint Loading.

Implements the canonical PyTorch checkpoint loading pattern for production DDP training:

**Key Rules:**
1. **Model weights**: Load only on global rank-0 before DDP wrap
2. **Trainer state**: Load on rank-0, broadcast payload only after DDP wrap

**Architecture:**
1. Pick device → 2. Rank-0 preloads checkpoint → 3. Build model → 4. Load weights
   (rank-0 only) → 5. Init DDP → 6. Wrap with DDP (auto-broadcasts) → 7. Create
   optimizer/scheduler → 8. Load trainer states

**Checkpoint Loading Flow (DDP-aware):**
1. Rank 0: load_checkpoint_unified() - Load checkpoint from disk using
   CheckpointIO, store it
2. Broadcast: All ranks receive checkpoint_loaded status
3. Rank 0: prepare_model_weights() - Load model weights BEFORE DDP wrapping
   (uses BaseModel.load_checkpoint())
4. All ranks: DDP wrapping - Weights automatically broadcast to all ranks
5. All ranks: Optimizer/scheduler creation AFTER DDP wrapping
6. Rank 0: load_trainer_states() - Pack trainer state using CheckpointIO,
   broadcast to all ranks
7. All ranks: Apply trainer state from broadcast payload

**Usage:**
- Always call run_training() - handles all scenarios automatically
- For manual loading: load_checkpoint_unified() → prepare_model_weights() →
  load_trainer_states()

**Structure:**
- DirectionMode, TrainerBaseConfig: config and enum for direction handling
- BaseTrainer: abstract trainer; key methods run_training(), load_checkpoint_unified(),
  prepare_model_weights(), load_trainer_states(); hooks for dataset, model, loop
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import random
import sys

# Standard library imports
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

# Third-party imports
from omegaconf import MISSING
from omegaconf import ListConfig
from torch.nn.parallel import (
    DistributedDataParallel as DDP,  # noqa: N817  # conventional alias DDP
)
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

if TYPE_CHECKING:
    from src.core.direction import Directions
    from src.core.direction import DirectionsConfig
    from src.criterions.base_criterion import BaseCriterion
    from src.criterions.base_criterion import CriterionBaseConfig
    from src.loggings.progress_bar import ProgressBar
    from src.loggings.progress_bar import ProgressBarConfig
    from src.preprocessors.base_preprocessor import BasePreprocessor
    from src.preprocessors.base_preprocessor import BasePreprocessorConfig
    from src.processors import OutputProcessor
    from src.utils.checkpoint_io import CheckpointIO
    from src.utils.checkpoint_io import CheckpointIOConfig
    from src.utils.checkpoint_manager import CheckpointManager
    from src.utils.checkpoint_manager import CheckpointManagerConfig
    from src.utils.train_utils import EarlyStopping
    from src.utils.train_utils import EarlyStoppingConfig

# Local imports
from src.core.domain import Vital
from src.loggings.metrics import metrics

# Import optimizer and scheduler packages to register configs
from src.optimizers import import_optimizers
from src.schedulers import import_schedulers
from src.utils.collate_utils import build_vital_channel_mapping
from src.utils.collate_utils import compute_max_source_channels
from src.utils.collate_utils import create_direction_aware_collate_fn
from src.utils.constants import BATCH_KEY_DIRECTION
from src.utils.constants import CHECKPOINT_EARLY_STOPPING_KEY
from src.utils.constants import CHECKPOINT_EARLY_STOPPING_STATE_KEY
from src.utils.constants import CHECKPOINT_MODEL_STATE_KEY
from src.utils.constants import CHECKPOINT_OPTIMIZER_KEY
from src.utils.constants import CHECKPOINT_OPTIMIZER_STATE_KEY
from src.utils.constants import CHECKPOINT_SCHEDULER_KEY
from src.utils.constants import CHECKPOINT_SCHEDULER_STATE_KEY
from src.utils.constants import STAGE_TEST
from src.utils.constants import STAGE_TRAIN
from src.utils.constants import STAGE_VAL
from src.utils.dataset_utils import get_dataset_attribute
from src.utils.utils_preprocessing import print_model_parameters

# Register optimizer and scheduler configs
import_optimizers()
import_schedulers()

logger = logging.getLogger(__name__)


class DirectionMode(Enum):
    """Enumeration for training direction modes."""

    SINGLE = "single"
    MULTI = "multi"

    @classmethod
    def from_string(cls, value: str) -> DirectionMode:
        """Convert string to enum with helpful error message.

        Args:
            value: String to convert ('single' or 'multi').

        Returns:
            DirectionMode: The corresponding enum member.

        Raises:
            ValueError: If value is not a valid direction mode.
        """
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(
                f"Invalid direction_mode '{value}'. Must be one of: {valid_values}"
            ) from None


@dataclass
class TrainerBaseConfig:
    """Configuration for BaseTrainer with Hydra-compatible defaults.

    Attributes:
        checkpoint_mapping: Dictionary mapping component names to checkpoint
            loading configuration. For multi-stage trainers, stage1 and stage2
            should be defined explicitly in YAML configs rather than relying on
            the default mapping.
        max_source_channels: Maximum number of source channels per sample.
            If None, automatically computed from directions. Can be explicitly
            set to override auto-computation (useful for testing or special cases).
    """

    _target_: str = "src.trainers.trainer.BaseTrainer"
    _recursive_: bool = True
    _convert_: str = "all"
    trainer_name: str = MISSING

    model: Any = MISSING
    criterion: CriterionBaseConfig = MISSING
    optimizer: Any = MISSING
    scheduler: Any | None = None
    checkpoint_managers: dict[str, CheckpointManagerConfig] = field(
        default_factory=dict
    )
    checkpoint_mapping: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "model": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_MODEL_STATE_KEY,
                "target_key": "model",
            },
            "optimizer": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_OPTIMIZER_STATE_KEY,
                "target_key": CHECKPOINT_OPTIMIZER_KEY,
            },
            "scheduler": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_SCHEDULER_STATE_KEY,
                "target_key": CHECKPOINT_SCHEDULER_KEY,
            },
            "early_stopping": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_EARLY_STOPPING_STATE_KEY,
                "target_key": CHECKPOINT_EARLY_STOPPING_KEY,
            },
        }
    )
    checkpoint_io: CheckpointIOConfig = MISSING
    progress_bar: ProgressBarConfig = MISSING
    early_stopping: EarlyStoppingConfig | None = None
    directions: DirectionsConfig = MISSING
    processor: Any | None = None
    compute_full_metrics_during_train: bool = False

    # Direction strategy configuration (string in YAML, converted to Enum)
    direction_mode: str = "single"

    max_source_channels: int | None = None

    # Common config parameters
    is_pretraining: bool = False
    is_finetuning: bool = False
    is_few_shot: bool = False
    resume_training: bool = False
    use_wcl: bool = False

    # Checkpoint config parameters
    strict_loading: bool = True
    load_model_weights: bool = False
    load_optimizer: bool = True
    load_scheduler: bool = True

    # Checkpoint path parameters (for CheckpointManager)
    # abs_path moved to CheckpointManagerConfig.base_dir
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    scheduler_patience: int | None = 3
    use_patient_split: bool = False
    use_patient_information: bool = False
    save_checkpoint_frequency: int | None = (
        5  # Persist checkpoint every N epochs; None disables
    )
    overwrite_checkpoint: bool = False
    load_checkpoint_from_epoch: int | None = None

    # Hardware configuration parameters (keep these - they're performance knobs)
    num_threads: int = 2
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    timeout: int = 3600
    seed: int = 42

    # Data preprocessing configuration
    input_preprocessing: dict[str, Any] = MISSING

    # Debug mode settings
    debug: bool = False

    # Hydra run directory for storing logs and results
    hydra_run_dir: str | None = None

    # Logging configuration (moved from common config)
    logging_level: str = "INFO"
    log_file_path: str = "training.log"

    demographics_text_encoder: BasePreprocessorConfig | None = None

    scheduler_mode: str = "min"

    def __post_init__(self) -> None:
        """Validate checkpoint_mapping manager references.

        Mirrors evaluator behavior: if a manager is not found, log a warning
        and skip that component instead of raising. This keeps backward
        compatibility while still surfacing misconfigurations.
        """
        for component, mapping in self.checkpoint_mapping.items():
            manager_name = mapping.get("manager")
            if manager_name and manager_name not in self.checkpoint_managers:
                available = list(self.checkpoint_managers.keys())
                logger.warning(
                    f"checkpoint_mapping['{component}'] references manager '{
                        manager_name
                    }' "
                    f"which is not in checkpoint_managers. Available managers: {
                        available
                    }. "
                    f"This component will be skipped during loading/saving."
                )

        if (
            self.save_checkpoint_frequency is not None
            and self.save_checkpoint_frequency < 1
        ):
            raise ValueError(
                f"save_checkpoint_frequency must be a positive integer (>= 1), "
                f"got {self.save_checkpoint_frequency}"
            )


def unwrap_model(obj: nn.Module) -> nn.Module:
    """Return underlying module if obj is DDP-wrapped; else return obj."""
    return obj.module if hasattr(obj, "module") else obj


def recursively_set_layout(
    obj: nn.Module,
    layout: dict[Vital, int],
    _visited: set[int] | None = None,
) -> bool:
    """Recursively apply layout using m.children(), with DDP unwrap and cycle
    protection.

    - Automatically unwraps DistributedDataParallel wrappers via
      unwrap_model().
    - Iterates through all child modules using m.children() (no
      pattern-specific checks).
    - Uses a visited set to prevent infinite recursion on cyclic graphs.
    - Emits a top-level warning if no set_layout() is found on the module
      or its children.

    Args:
        obj: Module to recursively set layout on
        layout: Vital to channel mapping
        _visited: Set of visited module IDs (for cycle detection)

    Returns:
        bool: True if layout was applied to at least one module, False otherwise
    """
    is_root = _visited is None
    if _visited is None:
        _visited = set()

    m = unwrap_model(obj)
    m_id = id(m)

    if m_id in _visited:
        return False
    _visited.add(m_id)

    layout_applied = False

    if hasattr(m, "set_layout"):
        m.set_layout(layout)
        layout_applied = True

    for child in m.children():
        if recursively_set_layout(child, layout, _visited):
            layout_applied = True

    if is_root and not layout_applied:
        logger.warning(
            "No set_layout() method found on %s or its children", type(m).__name__
        )

    return layout_applied


class BaseTrainer(ABC):
    """Base trainer class designed for torchrun-driven distributed training
    with simplified canonical checkpoint loading.

    This trainer is designed to work with PyTorch's torchrun launcher, which
    automatically handles process spawning, environment variable setup, and
    DDP initialization.

    Key features:
    - Single entry point: run_training() method with proper checkpoint loading order
    - Automatic DDP setup via torchrun environment variables with auto-backend detection
    - Unified broadcast-based checkpoint loading for all scenarios (single-GPU and DDP)
    - Clean separation of concerns between infrastructure and training logic
    - Subclasses implement _execute_training_logic() for specific training
      behavior
    - Optional processor support: Base-level processor parameter with
      subclass-specific validation

    Checkpoint Loading Architecture:
    ===============================

    The trainer implements a clean, logical checkpoint loading approach with
    clear separation of concerns:

    1. **load_checkpoint_unified()**: Only loads checkpoint from disk and
       stores it (rank 0 only)
    2. **prepare_model_weights()**: Loads model weights from stored checkpoint
       (rank 0 only, before DDP wrapping)
    3. **load_trainer_states()**: Loads trainer states from stored checkpoint
       (rank 0 only, then broadcast) AND returns training metadata (epoch,
       best_loss) to ALL ranks

    This approach eliminates mixed concerns and unnecessary branching logic, providing:
    - Clear separation of responsibilities
    - Consistent behavior across single-GPU and DDP
    - Linear, logical flow that's easy to understand and maintain
    - Industry standard pattern that teams converge on
    """

    # Required components (set by Hydra; validated in __init__)
    model: nn.Module
    checkpoint_io: CheckpointIO
    progress_bar: ProgressBar
    directions: Directions
    # Optional components (criterion from Hydra; optimizer/scheduler from create_*)
    criterion: BaseCriterion | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None

    def __init__(
        self,
        trainer_name: str,
        model: nn.Module | None = None,
        criterion: BaseCriterion | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        checkpoint_managers: dict[str, CheckpointManager] | None = None,
        checkpoint_mapping: dict[str, dict[str, str]] | None = None,
        checkpoint_io: CheckpointIO | None = None,
        progress_bar: ProgressBar | None = None,
        early_stopping: EarlyStopping | None = None,
        directions: Directions | None = None,
        is_pretraining: bool = False,
        is_finetuning: bool = False,
        is_few_shot: bool = False,
        resume_training: bool = False,
        strict_loading: bool = True,
        load_model_weights: bool = False,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_early_stopping: bool = True,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        scheduler_patience: int = 10,
        use_patient_split: bool = False,
        use_wcl: bool = False,
        num_threads: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        timeout: int = 3600,
        seed: int = 42,
        direction_mode: str = "auto",
        max_source_channels: int = 2,
        debug: bool = False,
        hydra_run_dir: str | None = None,
        logging_level: str = "INFO",
        log_file_path: str = "training.log",
        input_preprocessing: dict[str, Any] | None = None,
        use_patient_information: bool = False,
        save_checkpoint_frequency: int | None = 5,
        overwrite_checkpoint: bool = False,
        load_checkpoint_from_epoch: int | None = None,
        scheduler_mode: str = "min",
        demographics_text_encoder: BasePreprocessor | None = None,
        processor: OutputProcessor | None = None,
        compute_full_metrics_during_train: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize BaseTrainer with extensive parameter list.

        IMPORTANT CONVENTION - Subclass Implementation Pattern:
        ======================================================
        All subclasses MUST use the *args, **kwargs pattern when calling
        super().__init__():

        Example (CORRECT):
            class MyTrainer(BaseTrainer):
                def __init__(self, my_param: str = None, *args, **kwargs):
                    # Safe pattern
                    super().__init__(*args, **kwargs)
                    self.my_param = my_param

        Example (INCORRECT - DO NOT DO THIS):
            class MyTrainer(BaseTrainer):
                def __init__(self, my_param: str = None, trainer_name: str = None, ...):
                    # Positional args - breaks on signature changes
                    super().__init__(trainer_name, ...)

        Why This Pattern:
        ----------------
        - Hydra's instantiate() uses keyword arguments from YAML configs (safe)
        - The *args, **kwargs pattern makes subclasses resilient to signature changes
        - New parameters can be safely added to BaseTrainer without breaking subclasses
        - All current subclasses follow this pattern

        Parameter Addition Guidelines:
        -----------------------------
        When adding new parameters to this signature:
        1. Add them at the END, before *args, **kwargs
        2. Always provide a default value
        3. Update TrainerBaseConfig dataclass accordingly
        4. Document in the Parameters section below

        Audit Status:
        ------------
        All trainer subclasses have been audited (2024) and confirmed to use the safe
        *args, **kwargs pattern:
        - WaveformReconstructionTrainer
        - ScalarRegressionTrainer
        - ClassificationTrainer
        - GANTrainer

        Args:
            trainer_name: Name of the trainer (e.g., 'approximation', 'refinement')
            model: Model instance (instantiated by Hydra)
            criterion: Loss criterion instance
            optimizer: Optimizer partial function (created by Hydra)
            scheduler: Scheduler partial function (created by Hydra)
            checkpoint_managers: Dictionary of checkpoint managers by role
            checkpoint_io: CheckpointIO instance for loading/saving
            progress_bar: Progress bar instance for logging
            early_stopping: Early stopping instance
            directions: Directions instance defining training directions
            is_pretraining: Whether this is a pretraining run
            is_finetuning: Whether this is a finetuning run
            is_few_shot: Whether this is a few-shot learning run
            resume_training: Whether to resume from checkpoint
            strict_loading: Whether to strictly enforce checkpoint key matching
            load_model_weights: Whether to load model weights from checkpoint
            load_optimizer: Whether to load optimizer state from checkpoint
            load_scheduler: Whether to load scheduler state from checkpoint
            load_early_stopping: Whether to load early stopping state
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            scheduler_patience: Scheduler patience for learning rate reduction
            use_patient_split: Whether to use patient-level data splitting
            use_wcl: Whether to use weighted contrastive learning
            num_threads: Number of CPU threads to use
            num_workers: Number of DataLoader worker processes
            pin_memory: Whether to pin memory for DataLoader
            persistent_workers: Whether to keep worker processes alive
            prefetch_factor: Number of batches to prefetch in workers
            timeout: DataLoader timeout in seconds
            seed: Random seed for reproducibility
            direction_mode: Direction mode ('single' or 'multi')
            max_source_channels: Maximum number of source channels
            debug: Whether to enable debug mode
            hydra_run_dir: Hydra run directory path
            logging_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file_path: Path to log file
            input_preprocessing: Input preprocessing configuration
            use_patient_information: Whether to use patient metadata
            save_checkpoint_frequency: How often to save checkpoints (in epochs)
            overwrite_checkpoint: Whether to overwrite existing checkpoints
            load_checkpoint_from_epoch: Specific epoch to load from (None for latest)
            scheduler_mode: Scheduler mode ('min' or 'max')
            processor: Optional processor for post-processing model outputs.
                If None, subclasses requiring processor must validate and raise
                error. Default: None.
            compute_full_metrics_during_train: When False (default), training
                steps skip processor-driven metrics to reduce overhead. Set
                True to compute full metrics during training.

        Raises:
            ValueError: If configuration validation fails
        """
        super().__init__()

        self.trainer_name = trainer_name
        self.is_pretraining = is_pretraining
        self.is_finetuning = is_finetuning
        self.is_few_shot = is_few_shot
        self.resume_training = resume_training
        self.use_wcl = use_wcl
        self.strict_loading = strict_loading
        self.load_model_weights = load_model_weights
        self.load_optimizer = load_optimizer
        self.load_scheduler = load_scheduler
        self.load_early_stopping = load_early_stopping
        self.direction_mode = DirectionMode.from_string(direction_mode)
        self.max_source_channels = max_source_channels
        self.debug = debug
        self.hydra_run_dir = hydra_run_dir
        self.scheduler_mode = scheduler_mode
        # Optional demographics text encoder
        self.demographics_text_encoder = demographics_text_encoder
        # Optional processor
        self.processor = processor
        if processor is not None:
            logger.info(
                "BaseTrainer initialized with processor: %s", type(processor).__name__
            )
        self.compute_full_metrics_during_train = compute_full_metrics_during_train
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.input_preprocessing = input_preprocessing

        self.logging_level = logging_level
        self.log_file_path = log_file_path

        if save_checkpoint_frequency is not None and save_checkpoint_frequency < 1:
            raise ValueError(
                f"save_checkpoint_frequency must be a positive integer (>= 1), "
                f"got {save_checkpoint_frequency}"
            )

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.use_patient_split = use_patient_split
        self.save_checkpoint_frequency = save_checkpoint_frequency
        self.overwrite_checkpoint = overwrite_checkpoint
        self.load_checkpoint_from_epoch = load_checkpoint_from_epoch

        self.num_threads = num_threads
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.seed = seed

        self.criterion = criterion
        self.checkpoint_managers = checkpoint_managers or {}
        self.checkpoint_mapping = checkpoint_mapping or {
            "model": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_MODEL_STATE_KEY,
                "target_key": "model",
            },
            "optimizer": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_OPTIMIZER_STATE_KEY,
                "target_key": CHECKPOINT_OPTIMIZER_KEY,
            },
            "scheduler": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_SCHEDULER_STATE_KEY,
                "target_key": CHECKPOINT_SCHEDULER_KEY,
            },
            "early_stopping": {
                "manager": "save",
                "checkpoint_key": CHECKPOINT_EARLY_STOPPING_STATE_KEY,
                "target_key": CHECKPOINT_EARLY_STOPPING_KEY,
            },
        }
        if model is None:
            raise ValueError("BaseTrainer requires model to be set")
        if checkpoint_io is None:
            raise ValueError("BaseTrainer requires checkpoint_io to be set")
        if progress_bar is None:
            raise ValueError("BaseTrainer requires progress_bar to be set")
        if directions is None:
            raise ValueError("BaseTrainer requires directions to be set")
        self.model = model
        self.checkpoint_io = checkpoint_io
        self.progress_bar = progress_bar
        self.early_stopping = early_stopping
        self.directions = directions

        self._vital_channel_mapping: dict[Vital, int] | None = None

        # Add missing attributes for CheckpointManager compatibility
        self.use_patient_information = use_patient_information

        # Single source of truth for training state (epoch, best_loss, global_step).
        self.training_metadata = {"epoch": 0, "best_loss": None, "global_step": 0}

        try:
            self._validate_directions_against_preprocessing()
            logger.debug(
                "Direction validation passed: All source vitals present in "
                "input_preprocessing"
            )
        except ValueError as e:
            logger.error("Direction validation failed: %s", e, exc_info=True)
            raise

        self._setup_hardware_config()

        self.set_seed()

        self.metrics = metrics

    def _get_model_name(self) -> str:
        """Get model name from the model object.

        Returns:
            str: Model name (e.g., 'nabnet', 'ppg2abp', 'patchtst', 'mdvisco')
        """
        assert self.model is not None
        return unwrap_model(self.model).model_name

    def _get_model_supports_multi(self) -> bool:
        """Get model multi-directional support from the model object.

        Returns:
            bool: Whether the model supports multi-directional training
        """
        assert self.model is not None
        return unwrap_model(self.model).supports_multi_directional

    def _get_dataset_name(self, dataset: Any) -> str:
        """Extract dataset name from dataset or DataLoader.

        Uses the universal get_dataset_attribute utility which handles DataLoader
        unwrapping and direct attribute access.

        Args:
            dataset: Dataset object or DataLoader for checkpoint path building

        Returns:
            str: Dataset name extracted from the underlying dataset
        """
        result = get_dataset_attribute(dataset, "dataset_name", required=True)
        if result is None:
            raise ValueError("dataset_name is required but was not found on dataset")
        return str(result)

    def get_direction_name(self) -> str:
        """Get the current direction name for single-direction training.

        Returns:
            str: Direction name for single-direction training, or empty string for
                multi-directional
        """
        if len(self.directions.directions) > 1:
            return ""  # Multi-directional training
        elif len(self.directions.directions) == 1:
            key = self.directions.directions[0].key()
            return key if isinstance(key, str) else ""
        else:
            return ""  # No directions configured

    def _setup_hardware_config(self) -> None:
        """Set up hardware configuration - simplified for torchrun."""
        if (
            not hasattr(self, "num_threads")
            or self.num_threads is None
            or self.num_threads <= 0
        ):
            raise ValueError(
                f"num_threads must be explicitly set to a positive integer. "
                f"Current value: {getattr(self, 'num_threads', 'Not set')}"
            )

        if (
            not hasattr(self, "num_workers")
            or self.num_workers is None
            or self.num_workers < 0
        ):
            raise ValueError(
                f"num_workers must be explicitly set to a non-negative integer. "
                f"Current value: {getattr(self, 'num_workers', 'Not set')}"
            )

        logger.info(
            f"[HW] cpu={os.cpu_count()} threads={self.num_threads} "
            f"workers={self.num_workers} cuda={torch.cuda.is_available()} "
            f"gpus={torch.cuda.device_count()}"
        )

    def _validate_preprocessing_config(self, dataset: Any) -> None:
        """Validate that requested preprocessing attributes exist in dataset.

        Supports nested configuration format:
        - Single vital: {"source": {"vital": "ppg", "norm": "minmax_zc"},
          "target": {"vital": "abp", "norm": "global_minmax"}}
        - Multi vital: {"source": [{"vital": "ppg", "norm": "minmax_zc"},
          {"vital": "ecg", "norm": "minmax_zc"}], "target": [...]}

        Args:
            dataset: Dataset to validate preprocessing configuration against

        Raises:
            ValueError: If requested preprocessing attributes are not found or
                configuration is invalid
        """
        if not dataset:
            return

        sample = dataset[0] if len(dataset) > 0 else None
        if not sample:
            return

        vital_requirements = []

        input_preprocessing = self.input_preprocessing
        if input_preprocessing is None:
            return
        for output_key, config_value in input_preprocessing.items():
            if output_key not in ["source", "target"]:
                continue

            # Normalize to list format (following
            # _validate_directions_against_preprocessing pattern)
            configs_to_validate = []
            if isinstance(config_value, dict):
                # Single vital: {"vital": "ppg", "norm": "minmax_zc"}
                configs_to_validate = [config_value]
            elif isinstance(config_value, list):
                # Multi vital: [{"vital": "ppg", ...}, {"vital": "ecg", ...}]
                configs_to_validate = config_value
            else:
                raise ValueError(
                    f"Invalid preprocessing config for '{output_key}': Expected dict "
                    f"or list, got {type(config_value)}"
                )

            for config_dict in configs_to_validate:
                if not isinstance(config_dict, dict):
                    raise ValueError(
                        f"Each preprocessing config must be a dict, got "
                        f"{type(config_dict)} in '{output_key}'"
                    )

                if "vital" not in config_dict:
                    raise ValueError(
                        f"Missing 'vital' key in preprocessing config for "
                        f"'{output_key}': {config_dict}"
                    )
                if "norm" not in config_dict:
                    raise ValueError(
                        f"Missing 'norm' key in preprocessing config for "
                        f"'{output_key}': {config_dict}"
                    )

                vital_str = config_dict["vital"]
                if not isinstance(vital_str, str):
                    raise ValueError(
                        f"Vital must be a string, got {type(vital_str)} for "
                        f"'{output_key}': {vital_str}"
                    )

                try:
                    vital = Vital[vital_str.upper()]
                except KeyError:
                    valid_vitals = [v.value for v in Vital]
                    raise ValueError(
                        f"Invalid vital '{vital_str}' for '{output_key}'. "
                        f"Valid options: {valid_vitals}"
                    ) from None

                # Sampling records to check dataset support; avoids coupling
                # to VitalsDataset.
                num_samples_to_check = min(32, len(dataset))
                vital_found_in_samples = False

                for i in range(num_samples_to_check):
                    check_sample = dataset[i] if i < len(dataset) else None
                    if check_sample is None:
                        continue
                    has_vital_fn = getattr(check_sample, "has_vital", None)
                    if callable(has_vital_fn) and has_vital_fn(vital):
                        vital_found_in_samples = True
                        break

                vital_missing = not vital_found_in_samples

                if vital_missing:
                    vital_requirements.append(
                        f"{output_key} -> {vital.value} (vital: {vital_str})"
                    )

        if vital_requirements:
            available_attrs = [attr for attr in dir(sample) if not attr.startswith("_")]
            raise ValueError(
                f"Requested preprocessing attributes not found in dataset: {
                    vital_requirements
                }\n"
                f"Available attributes: {available_attrs}"
            )

    def _validate_include_list(self, include_list: list[str]) -> None:
        """Validate include list against allowed vocabulary.

        Args:
            include_list: List of raw field names to include in batch

        Raises:
            ValueError: If include list contains invalid keys
        """
        if not include_list:
            return  # Empty list is valid

        # Define allowed vocabulary (raw dataset fields only)
        allowed_keys = {
            "bp_raw",
            "age_raw",
            "gender_raw",
            "height_raw",
            "weight_raw",
            "bmi_raw",
        }

        invalid_keys = [key for key in include_list if key not in allowed_keys]
        if invalid_keys:
            raise ValueError(
                f"Invalid keys in input_preprocessing.include: {invalid_keys}\n"
                f"Allowed keys (raw dataset fields): {sorted(allowed_keys)}\n\n"
                f"Examples:\n"
                f"  include: [bp_raw, age_raw, gender_raw]  # For WCL training\n"
                f"  include: [age_raw, gender_raw, height_raw, weight_raw, bmi_raw]  "
                f"# For demographics\n\n"
                f"Note: Collate function returns raw fields. Models handle "
                f"broadcasting/processing.\n"
            )

        logger.debug("Include list validated: %s", include_list)

    def _validate_directions_against_preprocessing(self) -> None:
        """Validate that all direction source vitals exist in
        input_preprocessing.

        This validation catches configuration errors early by ensuring that
        every vital required by the directions is actually available in the
        input_preprocessing["source"] list.

        Raises:
            ValueError: If any direction's source vital is not in
                input_preprocessing["source"]

        Examples of error messages:
            >>> # Missing ECG in preprocessing
            >>> ValueError: Direction 'PPG+ECG2ABP' requires source vital
            ... 'ECG' which is not in input_preprocessing["source"].
            ... Available vitals: ['PPG']
            ... Fix: Add {"vital": "ecg", "norm": "..."} to
            ... input_preprocessing["source"]

        Note:
            This validation helps prevent silent data corruption where a
            direction expects a vital that isn't actually being extracted from
            the dataset.
        """
        if not self.input_preprocessing or "source" not in self.input_preprocessing:
            logger.warning(
                "Cannot validate directions: input_preprocessing is missing or has no "
                "'source' key. Skipping validation."
            )
            return

        source_config = self.input_preprocessing["source"]

        # Normalize to list format for uniform handling
        if isinstance(source_config, dict):
            # Single source: {"vital": "ppg", "norm": "minmax_zc"}
            source_configs = [source_config]
        elif isinstance(source_config, list):
            # Multi-source: [{"vital": "ppg", ...}, {"vital": "ecg", ...}]
            source_configs = source_config
        else:
            logger.warning(
                f"Cannot validate directions: input_preprocessing['source'] has "
                f"unexpected type {type(source_config)}. Expected dict or list. "
                f"Skipping validation."
            )
            return

        available_vitals = set()
        for config in source_configs:
            if isinstance(config, dict) and "vital" in config:
                available_vitals.add(config["vital"].upper())

        if not available_vitals:
            logger.warning(
                "Cannot validate directions: No vitals found in "
                "input_preprocessing['source']. Skipping validation."
            )
            return

        if not self.directions or not hasattr(self.directions, "directions"):
            logger.warning(
                "Cannot validate directions: self.directions is missing or invalid. "
                "Skipping validation."
            )
            return

        from src.core.domain import Vital

        for direction in self.directions.directions:
            source_vitals = direction.source
            for vital in source_vitals:
                # Vital enum -> string for config comparison
                if isinstance(vital, Vital):
                    vital_name = vital.value.upper()
                else:
                    vital_name = str(vital).upper()

                if vital_name not in available_vitals:
                    direction_key = (
                        direction.key() if hasattr(direction, "key") else str(direction)
                    )
                    raise ValueError(
                        f"Direction '{direction_key}' requires source vital '{
                            vital_name
                        }' "
                        f"which is not in input_preprocessing['source'].\n"
                        f"Available vitals: {sorted(available_vitals)}\n"
                        f"Fix: Add {{'vital': '{vital_name.lower()}', 'norm': '...'}} "
                        f"to input_preprocessing['source']"
                    )

        logger.debug(
            f"Validation passed: All direction source vitals {available_vitals} "
            f"are present in input_preprocessing['source']"
        )

    def _build_vital_channel_mapping(self) -> dict[Vital, int]:
        """Build dynamic vital-to-channel mapping from input_preprocessing
        configuration.

        This method establishes input_preprocessing as the single source of
        truth for channel indices. The order in input_preprocessing['source']
        determines channel positions in batch tensors.

        Returns:
            Dict[Vital, int]: Mapping each vital to its channel index

        Examples:
            >>> # For input_preprocessing["source"] = [{"vital": "ppg", ...},
            ... # {"vital": "ecg", ...}, {"vital": "abp", ...}]
            >>> mapping = trainer._build_vital_channel_mapping()
            >>> print(mapping)
            {Vital.PPG: 0, Vital.ECG: 1, Vital.ABP: 2}

        Note:
            Channel mapping derived from input_preprocessing order (single
            source of truth).
        """
        if self.input_preprocessing is None:
            return {}
        mapping = build_vital_channel_mapping(self.input_preprocessing)
        logger.info("Built vital channel mapping from input_preprocessing: %s", mapping)
        logger.debug(
            "Channel mapping derived from input_preprocessing order "
            "(single source of truth)"
        )

        return mapping

    def _get_distributed_config(self) -> tuple[int, int, int]:
        """Get rank, world_size, local_rank with priority to distributed
        module if initialized.

        Priority order:
        1. If dist.is_initialized(): use dist.get_rank() and dist.get_world_size()
        2. Fallback to torchrun environment variables (RANK, WORLD_SIZE)
        3. Local rank always from torchrun environment (LOCAL_RANK)
        """
        # Try to get rank and world_size from distributed module first
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            if "RANK" not in os.environ:
                raise RuntimeError(
                    "RANK environment variable not found. Are you running "
                    "with torchrun?"
                )
            if "WORLD_SIZE" not in os.environ:
                raise RuntimeError(
                    "WORLD_SIZE environment variable not found. Are you "
                    "running with torchrun?"
                )

            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "LOCAL_RANK environment variable not found. Are you running "
                "with torchrun?"
            )

        local_rank = int(os.environ["LOCAL_RANK"])

        return rank, world_size, local_rank

    def print_memory_stats(self, location: str) -> None:
        """Print GPU memory statistics for current device.

        Args:
            location: String describing where in the code this is called
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self._get_device()) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(self._get_device()) / (1024 * 1024)
            rank, _, _ = self._get_distributed_config()
            logger.info("Rank %s at %s:", rank, location)
            logger.info("Allocated: %.2fMB", allocated)
            logger.info("Cached: %.2fMB", cached)

    def _setup_cpu_environment(self) -> None:
        """Set up CPU and threading environment - respect existing env vars."""
        torch.set_num_threads(self.num_threads)

        # Thread envs (only set if not already set by user)
        for k in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            os.environ.setdefault(k, str(self.num_threads))

    def _get_device(self) -> torch.device:
        """Determine the appropriate device based on distributed config and backend.

        Returns:
            torch.device: The device to use (CPU or specific CUDA device)

        Raises:
            RuntimeError: If there's a CUDA device configuration error
        """
        rank, world_size, local_rank = self._get_distributed_config()

        backend = self._get_appropriate_backend()
        if backend == "gloo":  # CPU backend
            return torch.device("cpu")

        # GPU backend (nccl) - torchrun has already validated LOCAL_RANK
        # Minimal sanity check (fast, local, catches common mislaunches)
        if torch.cuda.current_device() != local_rank:
            logger.error(
                f"CUDA device mismatch: current_device={torch.cuda.current_device()} "
                f"!= LOCAL_RANK={local_rank}"
            )
            raise RuntimeError("CUDA device configuration error")

        return torch.device(f"cuda:{local_rank}")

    def _get_early_stopping_patience(self) -> int:
        """Get early stopping patience value with null safety.

        Returns:
            int: Early stopping patience if enabled, 0 if disabled
        """
        return self.early_stopping.patience if self.early_stopping is not None else 0

    def _set_device(self) -> None:
        """Set the device using _get_device and configure CUDA settings."""
        device = self._get_device()

        if device.type == "cuda":
            torch.cuda.set_device(device)
            logger.info(f"[DEVICE] {device}")
        else:
            # CPU device
            logger.info("[DEVICE] CPU")

        self.device = device

    def set_seed(self) -> None:
        """Set random seeds for reproducibility - modern PyTorch approach."""
        if self.seed is None:
            raise ValueError("Seed is not set")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Policy: disable determinism for performance; fixed seed provides
        # reproducibility of major components (Python, NumPy, PyTorch RNG).
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        logger.info(
            f"[SEED] Set to {self.seed} (non-deterministic mode for performance)"
        )

    def setup_logging(self) -> None:
        """Set up logging configuration using trainer config."""
        logging.basicConfig(
            level=getattr(logging, self.logging_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file_path),
            ],
        )
        logging.getLogger("src.utils.collate_utils").setLevel(logging.WARNING)
        logging.getLogger("src.processors.waveform_processor").setLevel(logging.WARNING)
        logger.info(
            f"Logging configured: level={self.logging_level}, file={self.log_file_path}"
        )

    # Properties that use environment truth, not parsed config
    @property
    def is_rank0(self) -> bool:
        """Check if this is the rank-0 process (master).

        Returns:
            True if single-process or rank 0 in DDP; False otherwise.
        """
        rank, world_size, local_rank = self._get_distributed_config()
        return rank == 0

    def _get_appropriate_backend(self) -> str:
        """Auto-detect and return appropriate backend for current hardware.

        This method automatically selects the optimal backend based on
        available hardware, eliminating the need for manual backend
        configuration and ensuring optimal performance across different
        environments.

        Returns:
            str: 'nccl' for GPU, 'gloo' for CPU
        """
        if torch.cuda.is_available():
            return "nccl"  # GPU backend
        else:
            return "gloo"  # CPU backend

    def init_ddp(self) -> None:
        """Simplified DDP initialization with auto-backend detection - always succeeds.

        This method always initializes DDP regardless of world size, with backend
        automatically selected based on hardware (GPU: nccl, CPU: gloo).
        """
        if dist.is_initialized():
            logger.warning("[DDP] Process group already initialized")
            return

        # Always initialize DDP - backend selection handles hardware differences
        backend = self._get_appropriate_backend()
        logger.info(f"[DDP] Initializing process group: backend={backend}")

        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",  # torchrun sets this
                timeout=datetime.timedelta(minutes=30),
            )
            logger.info("[DDP] Process group initialized successfully")
        except Exception as e:
            logger.error(f"[DDP] Failed to initialize process group: {e}")
            raise

    def cleanup_ddp(self) -> None:
        """Cleanup DDP process group with barrier for clean shutdown - handles
        all scenarios."""
        if not dist.is_available():
            logger.warning("[DDP] Distributed package not available")
            return

        if not dist.is_initialized():
            logger.info("[DDP] No process group to cleanup")
            return

        try:
            logger.info("[DDP] Flushing outstanding collectives with barrier")
            dist.barrier()  # Flush any pending collectives
        except Exception as e:
            logger.warning(f"[DDP] Barrier failed (non-critical): {e}")

        logger.info("[DDP] Destroying process group")
        dist.destroy_process_group()
        logger.info("[DDP] Process group cleanup completed")

    def create_optimizer(self) -> None:
        """Create optimizer - called AFTER DDP wrapping.

        Uses Hydra partial if supplied; otherwise default Adam.
        """
        assert self.model is not None
        if self.optimizer_partial is not None:
            self.optimizer = self.optimizer_partial(params=self.model.parameters())
        else:
            # Default: Adam on model parameters when no optimizer config is supplied
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        logger.info(
            f"Optimizer created: {
                getattr(self.optimizer, 'name', type(self.optimizer).__name__)
            }"
        )

    def create_scheduler(self) -> None:
        """Create scheduler from Hydra partial function - called AFTER optimizer
        creation."""
        if self.scheduler_partial is None:
            self.scheduler = None
            logger.info("No scheduler configured")
            return

        # Hydra partial has mode, patience, etc.; we supply the missing optimizer.
        assert self.optimizer is not None
        self.scheduler = self.scheduler_partial(optimizer=self.optimizer)
        assert self.scheduler is not None
        logger.info(f"Scheduler created: {self.scheduler.name}")

    def sync_metric_for_scheduler(self, metric: float) -> float:
        """All-reduce metric across ranks for scheduler stepping.

        Only performs all-reduce when distributed is available, initialized,
        and world size > 1; otherwise returns the local metric unchanged.
        """
        if not dist.is_available() or not dist.is_initialized():
            return metric
        if dist.get_world_size() <= 1:
            return metric
        metric_t = torch.tensor([metric], device=self.device, dtype=torch.float32)
        dist.all_reduce(metric_t, op=dist.ReduceOp.SUM)
        return (metric_t / dist.get_world_size()).item()

    def step_scheduler(self, metric: float) -> None:
        """Step scheduler with synchronized metric - handles both types of schedulers.

        In distributed mode (available, initialized, world_size > 1), the metric
        is all-reduced before stepping. Otherwise the local metric is used so
        ReduceLROnPlateau and other schedulers step correctly without crashing.
        """
        if self.scheduler is None:
            return  # Early return for None scheduler - no work needed

        synced_metric = self.sync_metric_for_scheduler(metric)

        # Handle both ReduceLROnPlateau and other schedulers
        if (
            self.scheduler is not None
            and hasattr(self.scheduler, "step")
            and callable(self.scheduler.step)
        ):
            if hasattr(
                self.scheduler, "_step_count"
            ):  # StepLR, CosineAnnealingLR, etc.
                self.scheduler.step()
            else:  # ReduceLROnPlateau
                self.scheduler.step(synced_metric)
        else:
            # Handle dictionary of schedulers (for GANs)
            sched_obj = self.scheduler
            if isinstance(sched_obj, dict):
                for _name, sched in sched_obj.items():
                    if hasattr(sched, "step") and callable(sched.step):
                        if hasattr(sched, "_step_count"):
                            sched.step()
                        else:
                            sched.step(synced_metric)

        if self.is_rank0:
            logger.info(f"Scheduler stepped with metric: {synced_metric:.6f}")

    def _worker_init_fn(self, worker_id: int) -> None:
        """Initialize worker with unique seed for reproducibility."""
        # Base seed + rank*1000 + worker_id for unique seeds per worker
        base_seed = self.seed
        rank, _, _ = self._get_distributed_config()
        worker_seed = base_seed + rank * 1000 + worker_id

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        # Align with main set_seed policy: non-deterministic for performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def get_dataloader_settings(self) -> dict[str, Any]:
        """Get DataLoader settings - ensure PyTorch-safe combinations with
        worker initialization."""
        use_cuda = torch.cuda.is_available()
        nw = max(0, int(getattr(self, "num_workers", 0)))

        settings = {
            "num_workers": nw,
            # Only enable pin_memory when using CUDA
            "pin_memory": bool(use_cuda),
            "persistent_workers": bool(nw > 0),
            "timeout": 0 if nw == 0 else 3600,
        }

        # prefetch_factor is only valid when num_workers > 0
        pf = getattr(self, "prefetch_factor", 2)
        if nw > 0 and pf is not None:
            settings["prefetch_factor"] = int(pf)

        # Pin memory device for PyTorch >=2.0 (only when CUDA)
        if use_cuda:
            settings["pin_memory_device"] = "cuda"

        # Add worker initialization function for proper seeding
        if nw > 0:
            settings["worker_init_fn"] = self._worker_init_fn

        return settings

    # ============================================================================
    # DATALOADER BUILDING METHODS - Shared across all trainers
    # ============================================================================

    def _get_collate_fn_kwargs(
        self,
        dataset_norm_params: Any | None,
        include_list: list[str],
        window_size: int | None,
        trim_strategy: str,
        source_channel_mapping: dict[Any, int],
        target_channel_mapping: dict[Any, int],
        encoder_to_pass: Any | None,
    ) -> dict[str, Any]:
        """Build kwargs for create_direction_aware_collate_fn for reuse across
        train/val/test.

        Returns:
            Dict of keyword arguments to pass to create_direction_aware_collate_fn.
        """
        return {
            "input_preprocessing": self.input_preprocessing,
            "directions": self.directions,
            "direction_mode": self.direction_mode,
            "max_source_channels": self.max_source_channels,
            "dataset_norm_params": dataset_norm_params,
            "include_list": include_list,
            "window_size": window_size,
            "trim_strategy": trim_strategy,
            "source_channel_mapping": source_channel_mapping,
            "target_channel_mapping": target_channel_mapping,
            "demographics_text_encoder": encoder_to_pass,
        }

    def _build_dataloaders(
        self, dataset_tuple: tuple[Any, Any, Any]
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build dataloaders with direction-aware collate function and proper
        DDP support.

        This method creates dataloaders that handle both preprocessing and
        direction logic at the collate level, following industry standards
        for optimal performance.

        NEW (Lazy Normalization):
        - Extracts dataset normalization parameters (e.g., sbp_max, dbp_min) from
          train_dataset
        - Passes normalization params to collate function for batch-time normalization
        - Includes raw data (bp_raw, demographics_raw) when use_wcl=True for Weighted
          Contrastive Learning
        """
        train_dataset, val_dataset, test_dataset = dataset_tuple

        self._validate_preprocessing_config(train_dataset)

        loader_settings = self.get_dataloader_settings()

        # Collate uses this for global norm (e.g. ABP/BP); None if no BP data.
        dataset_norm_params = train_dataset.get_normalization_params()

        if dataset_norm_params is not None:
            logger.info(
                f"Using dataset normalization parameters: {dataset_norm_params}"
            )
        else:
            logger.info(
                "No dataset normalization parameters available (dataset may not "
                "have BP data)"
            )

        include_list = (
            self.input_preprocessing.get("include", [])
            if self.input_preprocessing is not None
            else []
        )
        if include_list:
            logger.info(f"Including additional raw data fields: {include_list}")

        self._validate_include_list(include_list)

        # Configure demographics text encoder if demographics are requested and encoder
        # is available
        demo_fields = ["age_raw", "gender_raw", "height_raw", "weight_raw", "bmi_raw"]
        has_demographics = any(field in include_list for field in demo_fields)

        if has_demographics and self.demographics_text_encoder is None:
            detected_fields = [f for f in demo_fields if f in include_list]
            raise ValueError(
                f"Demographics fields detected in include_list ({detected_fields}), "
                f"but demographics_text_encoder is None. Configure "
                f"demographics_text_encoder in trainer config or remove demographics "
                f"fields from input_preprocessing.include."
            )

        if has_demographics and self.demographics_text_encoder is not None:
            self.demographics_text_encoder.configure_from_include_list(include_list)
            detected_fields = [f for f in demo_fields if f in include_list]
            logger.info(
                f"Configured demographics text encoder with fields from include_list: "
                f"{detected_fields}"
            )

        encoder_to_pass = (
            self.demographics_text_encoder
            if has_demographics and self.demographics_text_encoder is not None
            else None
        )

        # Log max_source_channels configuration
        if self.max_source_channels is None:
            logger.info(
                "max_source_channels not specified, computing from directions..."
            )
        else:
            logger.info(
                f"Using explicitly configured max_source_channels: {
                    self.max_source_channels
                }"
            )

        if self.max_source_channels is None:
            self.max_source_channels = compute_max_source_channels(self.directions)
            logger.info(
                f"Auto-computed max_source_channels: {self.max_source_channels}"
            )

        window_size = getattr(train_dataset, "input_size", None)
        trim_strategy = getattr(train_dataset, "trim_strategy", "center")

        if window_size is not None:
            logger.info(
                f"Batch-time padding/trimming enabled: window_size={window_size} "
                f"(from dataset), trim_strategy={trim_strategy}"
            )
        else:
            logger.info(
                "Batch-time padding/trimming disabled: using pre-padded "
                "sequences from dataset"
            )

        # Log NEW mode activation
        logger.info(
            "Using NEW mode: Channel mapping derived from input_preprocessing "
            "(vitals_dataset not passed to collate)"
        )

        assert self.input_preprocessing is not None
        source_config = self.input_preprocessing["source"]
        target_config = self.input_preprocessing["target"]
        source_channel_mapping = build_vital_channel_mapping({"source": source_config})
        target_channel_mapping = build_vital_channel_mapping({"source": target_config})
        logger.info(
            "Computed channel mappings once for reuse across train/val/test "
            "collate functions"
        )

        # Helper function for safe formatting of vital configs
        def format_vital_norm(config: Any) -> str:
            """Format a vital config as 'vital(norm)', handling missing norm keys."""
            if isinstance(config, (list, ListConfig)):
                return (
                    "["
                    + ", ".join(
                        f"{c['vital']}({c.get('norm', 'unknown')})" for c in config
                    )
                    + "]"
                )
            else:
                return f"{config['vital']}({config.get('norm', 'unknown')})"

        # Log source→target mapping summary
        # Format source and target strings with safe norm handling and
        # ListConfig support
        source_str = format_vital_norm(source_config)
        target_str = format_vital_norm(target_config)
        logger.info(
            f"Creating collate functions for train/val/test: {source_str} → {
                target_str
            }"
        )

        collate_kwargs = self._get_collate_fn_kwargs(
            dataset_norm_params=dataset_norm_params,
            include_list=include_list,
            window_size=window_size,
            trim_strategy=trim_strategy,
            source_channel_mapping=source_channel_mapping,
            target_channel_mapping=target_channel_mapping,
            encoder_to_pass=encoder_to_pass,
        )
        collate_fn_train = create_direction_aware_collate_fn(**collate_kwargs)
        collate_fn_val = create_direction_aware_collate_fn(**collate_kwargs)
        collate_fn_test = create_direction_aware_collate_fn(**collate_kwargs)

        # DDP dataloaders with DistributedSampler
        train_sampler = self.create_distributed_sampler(
            train_dataset, shuffle=True, drop_last=True
        )
        val_sampler = self.create_distributed_sampler(val_dataset, shuffle=False)
        test_sampler = self.create_distributed_sampler(test_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn_train,  # Use direction-aware collate
            **loader_settings,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn_val,  # Use direction-aware collate
            **loader_settings,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn_test,  # Use direction-aware collate
            **loader_settings,
        )

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

        return train_loader, val_loader, test_loader

    def create_distributed_sampler(
        self, dataset: Any, shuffle: bool = True, drop_last: bool = False
    ) -> DistributedSampler:
        """Create DistributedSampler with fixed seed for perfect resume reproducibility.

        Args:
            dataset: Dataset to sample from
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch

        Returns:
            DistributedSampler: Configured sampler with fixed seed
        """
        if self.seed is None:
            raise ValueError("Seed is not set")

        sampler: Any | None = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=self.seed,  # Fixed seed for perfect reproducibility
        )
        return sampler

    @abstractmethod
    def on_checkpoint_loaded(self, checkpoint: dict[str, Any]) -> None:
        """Handle checkpoint loaded event.

        Subclasses can override this to implement custom post-load logic
        (e.g., logging, additional state restoration).
        """

    def to_device(self, obj: Any) -> Any:
        """Move object to device robustly, handling nested structures."""
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(x) for x in obj)
        return obj

    def save_checkpoint(
        self,
        epoch: int | None,
        dataset: Any,
        optimizer_state_dict: dict[str, dict] | None = None,
        scheduler_state_dict: dict | None = None,
        early_stopping_state: dict | None = None,
        additional_info: dict | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Save checkpoint with rank-0 write + all-rank sync.

        Requires a 'save' manager to be configured in checkpoint_managers.
        If no 'save' manager exists, checkpoint saving is skipped.

        Model checkpoint structure is defined by model._checkpoint_state_dict()
        method. All models must implement this method to define their
        checkpoint structure.

        Args:
            epoch: Current training epoch (None for best model checkpoints)
            dataset: Dataset or DataLoader for checkpoint path building
            optimizer_state_dict: Dict of optimizer state dicts
            scheduler_state_dict: Scheduler state dict
            early_stopping_state: Early stopping state dict
            additional_info: Additional information to save

        Returns:
            Optional[str]: Path to saved checkpoint (rank>0 returns None)
        """
        checkpoint_path = None

        # Add deprecation warning for model_state_dict parameter
        if CHECKPOINT_MODEL_STATE_KEY in kwargs:
            logger.warning(
                "Passing model_state_dict to save_checkpoint() is deprecated. Models "
                "should implement _checkpoint_state_dict() instead."
            )

        try:
            if self.is_rank0:
                if "save" not in self.checkpoint_managers:
                    logger.info(
                        "No 'save' manager configured in checkpoint_managers. "
                        "Skipping checkpoint save. To enable saving, add a 'save' "
                        "manager to your checkpoint_managers configuration."
                    )
                    pass  # no save manager configured
                else:
                    # Use 'save' manager directly (no fallback)
                    save_manager = self.checkpoint_managers["save"]

                    # Source train_ratio strictly from dataset configuration using
                    # universal utility
                    train_ratio = get_dataset_attribute(
                        dataset, "train_ratio", required=True
                    )

                    full_kwargs = {
                        "model_name": self._get_model_name(),
                        "trainer_name": self.trainer_name,
                        "dataset_name": self._get_dataset_name(dataset),
                        "batch_size": self.batch_size,
                        "num_epochs": self.num_epochs,
                        "learning_rate": self.learning_rate,
                        "scheduler_patience": self.scheduler_patience,
                        "early_stopping_patience": self._get_early_stopping_patience(),
                        "is_finetuning": self.is_finetuning,
                        "use_patient_split": self.use_patient_split,
                        "use_patient_information": self.use_patient_information,
                        "use_wcl": self.use_wcl,
                        "train_ratio": train_ratio,
                    }

                    checkpoint_path = save_manager.build_path(
                        key="save",
                        epoch=epoch,
                        direction=self.get_direction_name(),
                        seed=self.seed,
                        overwrite=self.overwrite_checkpoint,
                        **full_kwargs,
                    )

                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                    # Log train_ratio usage if it's in the path format
                    if (
                        "train_ratio" in save_manager.path_format
                        or "train_ratio" in save_manager.filename_format
                    ):
                        logger.info(
                            f"Using train_ratio={
                                train_ratio
                            } from dataset in checkpoint path"
                        )

                    # Log trainer metadata during save
                    logger.info(f"Saving checkpoint to path: {checkpoint_path}")

                    # Prepare checkpoint data with enhanced metadata
                    checkpoint = {
                        "epoch": self.get_epoch() + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "flags": self.training_metadata.copy(),
                        "best_loss": self.get_best_loss(),
                    }

                    # All models must implement _checkpoint_state_dict()
                    unwrapped_model = unwrap_model(self.model)
                    checkpoint.update(unwrapped_model._checkpoint_state_dict())

                    if optimizer_state_dict is not None:
                        for opt_name, state_dict in optimizer_state_dict.items():
                            checkpoint[f"{opt_name}_state_dict"] = state_dict

                    if scheduler_state_dict:
                        checkpoint[CHECKPOINT_SCHEDULER_STATE_KEY] = (
                            scheduler_state_dict
                        )
                    if early_stopping_state:
                        checkpoint[CHECKPOINT_EARLY_STOPPING_STATE_KEY] = (
                            early_stopping_state
                        )
                    if additional_info:
                        checkpoint["additional_info"] = additional_info

                    # Debug logging
                    logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
                    ext = Path(checkpoint_path).suffix.lower()
                    if getattr(self.checkpoint_io, "safe_only", False) and ext not in (
                        ".safetensors",
                    ):
                        raise ValueError(
                            f"CheckpointIO.safe_only=True requires a "
                            f".safetensors path, got: {ext}. Configure "
                            f"CheckpointManager.file_ext='.safetensors'."
                        )

                    self.checkpoint_io.save(checkpoint_path, checkpoint)
                    logger.info(f"Checkpoint saved successfully: {checkpoint_path}")

            # All ranks must participate in the barrier or we risk a hang
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        except Exception as e:
            logger.error("Checkpoint saving failed: %s", e, exc_info=True)
            raise

        return checkpoint_path  # rank>0 will return None (fine)

    def load_checkpoint_unified(self, dataset: Any) -> bool:
        """Load checkpoint from disk and store it for later use, then
        broadcast status to all ranks.

        This method handles the actual file I/O, stores the checkpoint, and
        ensures all ranks know whether a checkpoint was loaded successfully.

        Args:
            dataset: Dataset for checkpoint path discovery

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
                (ALL ranks get this)
        """
        # Only rank 0 loads checkpoint from disk
        checkpoint_loaded = False
        if self.is_rank0:
            checkpoint = self._load_checkpoint_from_disk(dataset)

            if checkpoint is not None:
                self._stored_checkpoint = checkpoint
                checkpoint_loaded = True
                logger.info("Rank 0: Checkpoint loaded and stored successfully")
            else:
                logger.info("Rank 0: No checkpoint found or loaded")

        # Broadcast checkpoint_loaded status to all ranks
        checkpoint_loaded_tensor = torch.tensor(
            [checkpoint_loaded], dtype=torch.bool, device=self._get_device()
        )
        dist.broadcast(checkpoint_loaded_tensor, src=0)
        checkpoint_loaded = bool(checkpoint_loaded_tensor.item())
        logger.info(
            f"Device {self._get_device()}: Checkpoint loaded status: {
                checkpoint_loaded
            }"
        )

        return checkpoint_loaded

    def _validate_checkpoint_path(self, path: str, component: str) -> None:
        """Validate that a checkpoint path exists on disk.

        Args:
            path: Path to the checkpoint file
            component: Component name (e.g., 'model', 'optimizer') for error messages

        Raises:
            ValueError: If path does not exist
        """
        if not os.path.exists(path):
            raise ValueError(
                f"Expected checkpoint not found at {path} for component '{component}'"
            )

    def _load_checkpoint_from_disk(self, dataset: Any) -> dict[str, Any] | None:
        """Load checkpoint from disk on rank-0 only.

        This method handles the actual file I/O and returns the checkpoint
        dictionary. It's called only on rank-0 and handles checkpoint
        discovery and loading. Uses checkpoint_mapping to iterate over
        components and load from appropriate managers.

        Args:
            dataset: Dataset for checkpoint path discovery

        Returns:
            Optional[Dict[str, Any]]: Dictionary of loaded checkpoints keyed
                by component name (always returns dict, even if only one
                component is loaded). Returns None if no checkpoints were
                loaded.

        Raises:
            ValueError: If a checkpoint path for a component in
                checkpoint_mapping is not found.
        """
        loaded_checkpoints = {}
        direction_name = self.get_direction_name()

        for component, mapping in self.checkpoint_mapping.items():
            manager_name = mapping.get("manager")
            if not manager_name or manager_name not in self.checkpoint_managers:
                logger.debug(
                    f"Skipping component '{component}': manager '{
                        manager_name
                    }' not in "
                    f"checkpoint_managers"
                )
                continue

            manager = self.checkpoint_managers[manager_name]

            # Special-case the 'save' manager (or components mapped to it) to
            # use full kwargs to match the path format used during saving
            # (trainer-aligned checkpoints)
            if manager_name == "save":
                train_ratio = get_dataset_attribute(
                    dataset, "train_ratio", required=True
                )
                full_kwargs = {
                    "model_name": self._get_model_name(),
                    "trainer_name": self.trainer_name,
                    "dataset_name": self._get_dataset_name(dataset),
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "scheduler_patience": self.scheduler_patience,
                    "early_stopping_patience": self._get_early_stopping_patience(),
                    "is_finetuning": self.is_finetuning,
                    "use_patient_split": self.use_patient_split,
                    "use_patient_information": self.use_patient_information,
                    "use_wcl": self.use_wcl,
                    "train_ratio": train_ratio,
                }
                path_kwargs = full_kwargs
            else:
                # For non-save managers (stage1, stage2, auxiliary), use minimal
                # kwargs with formats that require at most epoch, direction,
                # seed, and optionally train_ratio
                minimal_kwargs = {
                    "epoch": self.load_checkpoint_from_epoch,
                    "direction": direction_name,
                    "seed": self.seed,
                }
                if (
                    "train_ratio" in manager.path_format
                    or "train_ratio" in manager.filename_format
                ):
                    # Source train_ratio strictly from dataset configuration using
                    # universal utility
                    train_ratio = get_dataset_attribute(
                        dataset, "train_ratio", required=True
                    )
                    minimal_kwargs["train_ratio"] = train_ratio
                    logger.debug(
                        f"Added train_ratio={train_ratio} to kwargs for component '{
                            component
                        }'"
                    )

                path_kwargs = minimal_kwargs

            try:
                candidate_path = manager.build_path(
                    key=component,
                    epoch=self.load_checkpoint_from_epoch,
                    direction=direction_name,
                    seed=self.seed,
                    **path_kwargs,
                )
            except KeyError as e:
                logger.warning(
                    f"Failed to build path for component '{
                        component
                    }': missing parameter {e}"
                )
                continue

            self._validate_checkpoint_path(candidate_path, component)
            logger.info(
                f"Rank 0: Loading checkpoint for component '{component}' from: {
                    candidate_path
                }"
            )
            io = self.checkpoint_io
            checkpoint = io.load(candidate_path, map_location="cpu")
            loaded_checkpoints[component] = checkpoint
            logger.info(
                f"Rank 0: Checkpoint for component '{component}' loaded successfully"
            )

        if len(loaded_checkpoints) == 0:
            logger.warning(
                "No checkpoints loaded from disk. Available managers: "
                + f"{list(self.checkpoint_managers.keys())}"
            )
            return None

        return loaded_checkpoints

    def prepare_model_weights(self, models: dict[str, nn.Module]) -> None:
        """Public interface: DDP-aware coordination for model weight loading.

        CRITICAL: This method MUST be called BEFORE DDP wrapping. It loads
        model weights on rank 0 only. DDP will automatically broadcast these
        weights to all ranks during wrapping. Do NOT call this after DDP
        wrapping.

        Uses CheckpointIO for unified checkpoint loading and BaseModel.load_checkpoint()
        for model weight loading with automatic DDP prefix stripping.

        Args:
            models: Dict of models to load weights into
        """
        if not hasattr(self, "_stored_checkpoint") or self._stored_checkpoint is None:
            logger.warning("No stored checkpoint available for model weight loading")
            return

        if not self.is_rank0:
            return

        self._apply_model_weights(self._stored_checkpoint, models)
        logger.info("Rank 0: Model weights loaded (will be broadcast by DDP)")

    def _apply_model_weights(
        self,
        checkpoint: dict[str, Any],
        models: dict[str, nn.Module],
    ) -> None:
        """Private implementation: delegates model weight loading to model's
        load_from_checkpoint_dict.

        This is the ONLY place where model weights are actually loaded into models.
        DDP will automatically broadcast these weights to all ranks during wrap.

        Delegates model components to model.load_from_checkpoint_dict() for standardized
        loading. Handles non-model components (optimizer, scheduler, early_stopping)
        separately via mapping.

        Args:
            checkpoint: Dictionary of loaded checkpoints keyed by component name
            models: Dict of models to load weights into
        """
        if not checkpoint:
            return

        # Assume checkpoint is dict of components from _load_checkpoint_from_disk
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dict of component checkpoints")

        # Delegate model components to model (filters 'model', 'stage1', 'stage2' etc.)
        model_components = {
            k: v for k, v in checkpoint.items() if k in ["model", "stage1", "stage2"]
        }
        if model_components:
            load_results = self.model.load_from_checkpoint_dict(
                model_components, self.checkpoint_mapping
            )
            for comp, success in load_results.items():
                if success:
                    logger.info(
                        f"Rank 0: {
                            comp
                        } weights loaded successfully (will be broadcast by DDP)"
                    )
                else:
                    logger.warning(f"Rank 0: Failed to load {comp} weights")

        # Handle non-model components (optimizer, scheduler, early_stopping) via mapping
        non_model_components = {
            k: v
            for k, v in checkpoint.items()
            if k not in ["model", "stage1", "stage2"]
        }
        for component, mapping in self.checkpoint_mapping.items():
            if component in non_model_components:
                checkpoint_key = mapping.get("checkpoint_key")
                if checkpoint_key is None or not isinstance(checkpoint_key, str):
                    continue
                if component == CHECKPOINT_OPTIMIZER_KEY:
                    if hasattr(self, "optimizer") and self.optimizer is not None:
                        state = self.checkpoint_io.extract_optimizer(
                            non_model_components[component], key=checkpoint_key
                        )
                        if self.optimizer is not None and state is not None:
                            self.optimizer.load_state_dict(state)
                            logger.info(
                                f"Rank 0: {component} state loaded successfully"
                            )
                elif component == "scheduler":
                    if hasattr(self, "scheduler") and self.scheduler is not None:
                        state = self.checkpoint_io.extract_scheduler(
                            non_model_components[component], key=checkpoint_key
                        )
                        if state is not None:
                            self.scheduler.load_state_dict(state)
                        logger.info(f"Rank 0: {component} state loaded successfully")
                elif (
                    component == "early_stopping"
                    and hasattr(self, "early_stopping")
                    and self.early_stopping is not None
                ):
                    state = self.checkpoint_io.extract_early_stopping(
                        non_model_components[component], key=checkpoint_key
                    )
                    if state is not None:
                        self.early_stopping.load_state_dict(state)
                    logger.info(f"Rank 0: {component} state loaded successfully")

    def load_trainer_states(self) -> dict[str, Any] | None:
        """Load trainer states from stored checkpoint (rank 0 only, then broadcast).

        CRITICAL: This method MUST be called AFTER DDP wrapping. It loads
        trainer states (optimizer, scheduler, early stopping) on rank 0, then
        broadcasts them to all ranks using
        torch.distributed.broadcast_object_list(). Do NOT call this before
        DDP wrapping.

        Uses CheckpointIO extraction methods for consistent handling of
        optimizer/scheduler/early_stopping states with legacy key support.

        Returns:
            Optional[Dict[str, Any]]: Training metadata (epoch, best_loss) that ALL
                ranks now have access to. Returns None if no checkpoint was available.
        """
        if not hasattr(self, "_stored_checkpoint") or self._stored_checkpoint is None:
            logger.warning("No stored checkpoint available for trainer state loading")
            return None

        # Pack trainer state payload on rank 0
        if self.is_rank0:
            trainer_payload = self._pack_trainer_payload(self._stored_checkpoint)

            # Broadcast payload to all ranks
            obj_list = [trainer_payload]
            dist.broadcast_object_list(obj_list, src=0)
            logger.info("Rank 0: Trainer state payload broadcasted")

            self._load_trainer_states_from_payload(trainer_payload)
        else:
            # Receive payload on other ranks
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            trainer_payload = obj_list[0]
            if trainer_payload is None:
                logger.warning("Received None trainer payload on non-rank-0")
                return None

            self._load_trainer_states_from_payload(trainer_payload)
            logger.info(
                f"Device {self._get_device()}: Trainer state loaded from "
                f"broadcast payload"
            )

        # Clean up stored checkpoint (only exists on rank-0)
        if hasattr(self, "_stored_checkpoint"):
            delattr(self, "_stored_checkpoint")

    def _pack_trainer_payload(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Pack only trainer state into payload (not model weights).

        Uses CheckpointIO extraction methods for consistent handling of trainer states
        with legacy key support. Uses checkpoint_mapping to determine checkpoint keys
        for each component.

        Args:
            checkpoint: Dictionary of loaded checkpoints keyed by component name

        Returns:
            Dict[str, Any]: Packed trainer state payload containing
                optimizer, scheduler, early_stopping states and training
                metadata
        """
        io = self.checkpoint_io
        payload = {}

        # Pack optimizer states (only if load_optimizer is True)
        if self.optimizer and self.load_optimizer:
            optimizer_mapping = self.checkpoint_mapping.get("optimizer", {})
            optimizer_key = optimizer_mapping.get(
                "checkpoint_key", CHECKPOINT_OPTIMIZER_STATE_KEY
            )

            optimizer_state = io.extract_optimizer(checkpoint, key=optimizer_key)
            logger.debug(f"Extracted optimizer state: {optimizer_state is not None}")
            if optimizer_state:
                payload["optimizers"] = {CHECKPOINT_OPTIMIZER_KEY: optimizer_state}
            elif optimizer_key != CHECKPOINT_OPTIMIZER_STATE_KEY:
                logger.warning(
                    f"Optimizer checkpoint key '{
                        optimizer_key
                    }' not found in checkpoint, "
                    f"skipping optimizer state"
                )

        # Pack scheduler state (only if load_scheduler is True)
        if self.scheduler and self.load_scheduler:
            scheduler_mapping = self.checkpoint_mapping.get("scheduler", {})
            scheduler_key = scheduler_mapping.get(
                "checkpoint_key", CHECKPOINT_SCHEDULER_STATE_KEY
            )

            scheduler_state = io.extract_scheduler(checkpoint, key=scheduler_key)
            logger.debug(f"Extracted scheduler state: {scheduler_state is not None}")
            if scheduler_state:
                payload["scheduler"] = scheduler_state
            elif scheduler_key != CHECKPOINT_SCHEDULER_STATE_KEY:
                logger.warning(
                    f"Scheduler checkpoint key '{
                        scheduler_key
                    }' not found in checkpoint, "
                    f"skipping scheduler state"
                )

        # Pack early stopping state
        if self.early_stopping:
            early_stopping_mapping = self.checkpoint_mapping.get("early_stopping", {})
            early_stopping_key = early_stopping_mapping.get(
                "checkpoint_key", CHECKPOINT_EARLY_STOPPING_STATE_KEY
            )

            early_stopping_state = io.extract_early_stopping(
                checkpoint, key=early_stopping_key
            )
            logger.debug(
                f"Extracted early_stopping state: {early_stopping_state is not None}"
            )
            if early_stopping_state:
                payload["early_stopping"] = early_stopping_state
            elif early_stopping_key != CHECKPOINT_EARLY_STOPPING_STATE_KEY:
                logger.warning(
                    f"Early stopping checkpoint key '{
                        early_stopping_key
                    }' not found in "
                    f"checkpoint, skipping early stopping state"
                )

        # Pack simple flags (use tensor broadcast for efficiency)
        payload["flags"] = {
            "epoch": checkpoint.get("epoch", 0),
            "best_loss": checkpoint.get("best_loss"),
            "load_model_weights": self.load_model_weights,
            "load_optimizer": self.load_optimizer,
            "load_scheduler": self.load_scheduler,
        }

        return payload

    def _load_trainer_states_from_payload(self, payload: dict[str, Any]) -> None:
        """Load trainer states from payload on any rank.

        Args:
            payload: Packed trainer state payload containing optimizer, scheduler,
                early_stopping states and training metadata
        """
        if self.optimizer and "optimizers" in payload:
            for opt_name, optimizer_state in payload["optimizers"].items():
                if opt_name == CHECKPOINT_OPTIMIZER_KEY and optimizer_state is not None:
                    self.optimizer.load_state_dict(optimizer_state)
                    logger.info(f"Loaded optimizer state: {opt_name}")

        if self.scheduler and "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
            logger.info("Loaded scheduler state")

        if self.early_stopping and "early_stopping" in payload:
            self.early_stopping.load_state_dict(payload["early_stopping"])
            logger.info("Loaded early stopping state")

        if "flags" in payload:
            self.training_metadata.update(payload["flags"])
            logger.info(f"Loaded training metadata: {self.training_metadata}")

    def get_training_metadata(self) -> dict[str, Any]:
        """Get current training metadata - single source of truth.

        Returns:
            dict: Current training state
        """
        return self.training_metadata

    def update_training_state(self, epoch: int, best_loss: float | None = None) -> bool:
        """Update training state in metadata dict.

        Args:
            epoch: Current epoch number
            best_loss: New best loss value (optional)

        Returns:
            bool: True if best_loss was improved, False otherwise
        """
        self.training_metadata["epoch"] = epoch
        if best_loss is not None:
            # Use set_best_loss to properly check if it's an improvement
            return self.set_best_loss(best_loss)
        return False

    def get_epoch(self) -> int:
        """Return the current epoch."""
        epoch = self.training_metadata.get("epoch")
        return int(epoch) if epoch is not None else 0

    def get_best_loss(self) -> float | None:
        """Return the best loss."""
        return self.training_metadata.get("best_loss")

    def set_best_loss(self, loss: float) -> bool:
        """Set best loss if improved, returns True if improved.

        Args:
            loss: New loss value

        Returns:
            bool: True if this is a new best loss
        """
        current_best = self.training_metadata.get("best_loss")
        if current_best is None or loss < current_best:
            self.training_metadata["best_loss"] = loss
            self.training_metadata["best_loss_epoch"] = self.training_metadata.get(
                "epoch"
            )
            return True
        return False

    def _validate_early_stopping(self) -> bool:
        """Validate early stopping is properly initialized.

        This method checks if the early stopping object exists and is properly
        initialized before attempting to use it. This prevents runtime errors
        when early stopping is not configured.

        Returns:
            bool: True if early stopping is properly initialized, False otherwise
        """
        if not hasattr(self, "early_stopping") or self.early_stopping is None:
            logger.warning(
                "Early stopping not initialized - skipping early stopping checks"
            )
            return False
        return True

    def check_early_stopping_common(self) -> bool:
        """Check early stopping with proper DDP synchronization and error handling.

        This method should be called AFTER updating early stopping state:
        1. First call: self.early_stopping(val_metric)  # Update state
        2. Then call: self.check_early_stopping_common()  # Check state

        Example usage in training loop:
        ```python
        for epoch in range(num_epochs):
            # ... training and validation ...

            # Update early stopping state with validation metric
            if val_metric is not None and self.early_stopping:
                self.early_stopping(val_metric)

            if self.check_early_stopping_common():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        ```

        Returns:
            bool: Whether early stopping should be triggered

        Raises:
            RuntimeError: If early stopping is not properly initialized
        """
        if not self._validate_early_stopping():
            return False

        try:
            should_stop = False
            # Make decision on rank-0
            if self.is_rank0 and self.early_stopping is not None:
                should_stop = bool(self.early_stopping.early_stop)
                if should_stop:
                    logger.info("Early stopping triggered on rank 0")

            # Broadcast decision to all ranks
            should_stop_tensor = torch.tensor(
                [should_stop], device=self._get_device(), dtype=torch.bool
            )
            torch.distributed.broadcast(should_stop_tensor, src=0)

            return bool(should_stop_tensor.item())
        except Exception as e:
            logger.error("Early stopping check failed: %s", e, exc_info=True)
            return False

    # ============================================================================
    # UNIFIED LOGGING METHODS - New unified logging system
    # ============================================================================

    def log_epoch_metrics_unified(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        best_loss: float | None,
        loss_improved: bool | None,
    ) -> None:
        """Enhanced epoch logging with automatic metric organization - rank-0 only.

        Automatically logs all provided metrics with appropriate prefixes:
        - train/metric_name for training metrics
        - val/metric_name for validation metrics
        - test/metric_name for test metrics

        No metric registration required - all computed metrics are logged automatically.
        """
        if not self.is_rank0:
            return  # Only rank-0 logs to W&B/TensorBoard
        if self.progress_bar is None or self.progress_bar.wandb is None:
            return

        if self.progress_bar.wandb._is_ready():
            try:
                log_dict = {}

                # Log ALL train metrics with simple prefixing
                for metric_name, metric_value in train_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(
                        metric_value
                    ):
                        log_dict[f"train/{metric_name}"] = float(metric_value)

                # Log ALL val metrics with simple prefixing
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(
                        metric_value
                    ):
                        log_dict[f"val/{metric_name}"] = float(metric_value)

                # Log ALL test metrics with simple prefixing
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(
                        metric_value
                    ):
                        log_dict[f"test/{metric_name}"] = float(metric_value)

                learning_rate = (
                    float(self.optimizer.param_groups[0]["lr"])
                    if self.optimizer is not None
                    else 0.0
                )
                log_dict.update(
                    {
                        "epoch": epoch,
                        "learning_rate": learning_rate,
                        "best_loss": float(best_loss or 0.0),
                        "improved": (
                            loss_improved if loss_improved is not None else False
                        ),
                    }
                )

                if hasattr(self, "early_stopping") and self.early_stopping:
                    log_dict.update(
                        {
                            "early_stopping/counter": int(
                                getattr(self.early_stopping, "counter", 0)
                            ),
                            "early_stopping/best_loss": float(
                                getattr(self.early_stopping, "best_loss", 0.0)
                            ),
                            "early_stopping/patience": int(
                                getattr(self.early_stopping, "patience", 0)
                            ),
                        }
                    )

                # Log to WandB
                self.progress_bar.wandb.log(log_dict, is_rank0=self.is_rank0)

            except (RuntimeError, KeyError, AttributeError) as e:
                logger.error(
                    "Failed to log epoch metrics to WandB: %s", e, exc_info=True
                )

    def log_step_metrics_unified(self, metrics_dict: dict[str, Any], step: int) -> None:
        """Unified step metrics logging - simplified and focused.

        This method handles ONLY the core metric logging infrastructure:
        - Logs all provided metrics to the metrics system
        - External logging (WandB, CSV) is handled by the progress bar

        DESIGN PATTERN:
        ===============
        This method follows a simplified approach:
        1. All metrics are computed in _step_core before calling this method
        2. No hooks or additional processing needed
        3. Each trainer is responsible for computing its own metrics in _step_core
        4. Progress bar handles all external logging with proper stage prefixes

        Args:
            metrics_dict: Dictionary containing all metrics to log (computed
                in _step_core)
            step: Current global step number
        """
        # Log all metrics to the metrics system
        for name, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and math.isnan(value)
            ):
                self.metrics.log_scalar(name, value)

        # Note: External logging (WandB, CSV) is now handled by the progress bar

    def create_training_progress_bar(
        self, train_loader: DataLoader, epoch: int, master_process: bool = True
    ) -> None:
        """Create unified training progress bar."""
        self.progress_bar.create_bar(
            total=len(train_loader),
            description=f"Train - Epoch {epoch}",
            disable=not master_process,
        )

    def create_validation_progress_bar(
        self, val_loader: DataLoader, epoch: int, master_process: bool = True
    ) -> None:
        """Create unified validation progress bar."""
        self.progress_bar.create_bar(
            total=len(val_loader),
            description=f"Val - Epoch {epoch}",
            disable=not master_process,
        )

    def create_test_progress_bar(
        self, test_loader: DataLoader, epoch: int, master_process: bool = True
    ) -> None:
        """Create unified test progress bar."""
        self.progress_bar.create_bar(
            total=len(test_loader),
            description=f"Test - Epoch {epoch}",
            disable=not master_process,
        )

    def update_progress_bar(
        self,
        metrics_dict: dict[str, Any] | None = None,
        step: int | None = None,
        is_rank0: bool = False,
        stage: str | None = None,
        to_log: bool = False,
    ) -> None:
        """Update progress bar with stage awareness.

        Args:
            metrics_dict: Dictionary of metrics to display and log
            step: Current step number
            is_rank0: Whether this is the rank 0 process
            stage: Stage name for metric prefixing and step management
                ("train", "val", or "test")
        """
        if metrics_dict and stage:
            # Pre-prefix metrics with stage name for proper WandB organization
            prefixed_metrics = {f"{stage}/{k}": v for k, v in metrics_dict.items()}
            self.progress_bar.update(
                metrics_dict=prefixed_metrics,
                step=step,
                is_rank0=is_rank0,
                to_log=to_log,
            )
        else:
            self.progress_bar.update(
                metrics_dict=metrics_dict, step=step, is_rank0=is_rank0, to_log=to_log
            )

    def close_progress_bar(self) -> None:
        """Close progress bar."""
        self.progress_bar.close_progress_bar()

    def close_wandb(self) -> None:
        """Close WandB."""
        self.progress_bar.close_wandb()

    def close_csv(self) -> None:
        """Close CSV."""
        self.progress_bar.close_csv()

    def close_all(self) -> None:
        """Close all loggers."""
        self.progress_bar.close()

    def set_sampler_epoch(self, epoch: int) -> None:
        """Set epoch for DistributedSampler with proper synchronization."""
        if hasattr(self, "train_sampler") and self.train_sampler is not None:
            # Only set epoch for training sampler (not validation)
            self.train_sampler.set_epoch(epoch)
            logger.debug(f"Set training sampler epoch to {epoch}")

        # Note: Do NOT call set_epoch on validation sampler

    # ============================================================================
    # Public API methods kept for backward compatibility
    # ============================================================================

    def cleanup_datasets(self, dataset_tuple: tuple[Any, Any, Any]) -> None:
        """Cleanup datasets using polymorphism."""
        if dataset_tuple:
            train_dataset, val_dataset, test_dataset = dataset_tuple
            for dataset in [train_dataset, val_dataset, test_dataset]:
                if dataset is not None and hasattr(dataset, "cleanup_shared_memory"):
                    try:
                        dataset.cleanup_shared_memory()
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup dataset {type(dataset).__name__}: {e}"
                        )

    def setup_hardware(self) -> torch.device:
        """Set up hardware environment - returns device for convenience."""
        self._setup_hardware_config()
        self._setup_cpu_environment()
        self._set_device()
        self.setup_hydra_environment()

        self.set_seed()

        return self._get_device()

    def cleanup_resources(self, dataset_tuple: tuple[Any, Any, Any]) -> None:
        """Cleanup resources after training."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cleanup DDP if needed
        self.cleanup_ddp()

        # Cleanup datasets
        self.cleanup_datasets(dataset_tuple)

    def setup_hydra_environment(self) -> None:
        """Set up Hydra environment with rank awareness."""
        rank, _, _ = self._get_distributed_config()
        os.environ["HYDRA_OUTPUT_DIR"] = f"./outputs/rank_{rank}"

        if self.is_rank0:
            log_file = f"./logs/training_rank_{rank}.log"
            os.environ["LOG_FILE"] = log_file
        else:
            # Non-rank-0 processes don't write to files
            os.environ["LOG_FILE"] = "/dev/null"

    def log_to_wandb(self, metrics: dict[str, Any]) -> None:
        """Log to WandB only on rank-0."""
        if not self.is_rank0:
            return
        wandb_obj = getattr(self.progress_bar, "wandb", None)
        if wandb_obj is None:
            return
        try:
            wandb_obj.log(metrics)
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")

    def log_split_statistics(
        self,
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
        total_size: int,
        stage_name: str,
        split_type: str = "Random",
        unique_subjects: int | None = None,
    ) -> None:
        """Log dataset split statistics.

        Args:
            train_indices: Training indices
            val_indices: Validation indices
            test_indices: Test indices
            total_size: Total dataset size
            stage_name: Name of the training stage
            split_type: Type of split used
            unique_subjects: Number of unique subjects (for patient-level splits)
        """
        logger.info(f"\n{stage_name} - {split_type} Split Statistics:")
        logger.info(f"Total samples: {total_size}")
        logger.info(
            f"Train samples: {len(train_indices)} ({
                len(train_indices) / total_size * 100:.1f}%)"
        )
        logger.info(
            f"Val samples: {len(val_indices)} "
            f"({len(val_indices) / total_size * 100:.1f}%)"
        )
        logger.info(
            f"Test samples: {len(test_indices)} ({
                len(test_indices) / total_size * 100:.1f}%)"
        )
        if unique_subjects:
            logger.info(f"Unique subjects: {unique_subjects}")

    def _setup_ddp_wrapping(self, model: nn.Module, local_rank: int) -> None:
        """Set up DDP wrapping - can be overridden by subclasses for special
        handling."""
        # Default DDP wrapping
        self.model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=True,
        )

    def _move_model_to_device(self, train_loader: DataLoader | None = None) -> None:
        """Move model to device - can be overridden by subclasses for special handling.

        This method handles the basic case of moving a single model to device.
        Subclasses like GANTrainer can override this to handle multiple models
        (e.g., generator and discriminator).

        Args:
            model: The model to move to device

        Returns:
            The model moved to the appropriate device
        """
        if self.model is not None:
            if self.is_rank0:
                logger.info(f"Moving model to device: {self._get_device()}")
            unwrap_model(self.model).to(self._get_device())

    def _run_epoch(
        self,
        epoch: int,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        master_process: bool,
        stage: str,
        optim: Any | None = None,
    ) -> dict[str, float]:
        """Run epoch for training, validation, or testing.

        This method provides shared epoch-running logic that is common across
        all trainers. Subclasses can override _on_epoch_end() for
        stage-specific behavior (e.g., validation logging).

        Args:
            epoch: Current epoch number
            model: The model to run
            data_loader: DataLoader for the stage
            device: Device to run on
            master_process: Whether this is the master process
            stage: Stage name ("train", "val", or "test")
            optim: Optimizer (required for training, None for validation/test)

        Returns:
            Dict[str, float]: Aggregated metrics for the stage
        """
        if stage not in (STAGE_TRAIN, STAGE_VAL, STAGE_TEST):
            raise ValueError(
                f"Stage must be '{STAGE_TRAIN}', '{STAGE_VAL}', or '{STAGE_TEST}', got {
                    stage
                }"
            )

        self.metrics.reset_meters(stage)

        if self.is_rank0:
            if stage == STAGE_TRAIN:
                self.create_training_progress_bar(data_loader, epoch, master_process)
            elif stage == STAGE_VAL:
                self.create_validation_progress_bar(data_loader, epoch, master_process)
            else:  # test
                self.create_test_progress_bar(data_loader, epoch, master_process)

        if (
            hasattr(self, "_vital_channel_mapping")
            and self._vital_channel_mapping is not None
        ):
            vital_channel_mapping = self._vital_channel_mapping
        else:
            vital_channel_mapping = self._build_vital_channel_mapping()

        if stage == STAGE_TRAIN:
            recursively_set_layout(model, vital_channel_mapping)
            logger.debug(
                "Training phase: Model layout set using input_preprocessing mapping"
            )
            model.train()
            context_manager = torch.enable_grad()
        elif stage == STAGE_VAL:
            recursively_set_layout(model, vital_channel_mapping)
            logger.debug(
                "Validation phase: Model layout set using input_preprocessing mapping"
            )
            model.eval()
            context_manager = torch.no_grad()
        else:  # test
            recursively_set_layout(model, vital_channel_mapping)
            logger.debug(
                "Test phase: Model layout set using input_preprocessing mapping"
            )
            model.eval()
            context_manager = torch.no_grad()

        try:
            with context_manager:
                for step, batch in enumerate(data_loader):
                    # Batch already contains direction metadata from collate
                    # Just move tensors to device (filtering non-tensor values)
                    prepared_batch = {
                        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }

                    # Use prepared_batch with unified structure
                    loss, metrics, outputs = self._step_core(
                        model, prepared_batch, stage=stage
                    )

                    # Stage-specific operations
                    if stage == STAGE_TRAIN:
                        # Training side effects
                        if optim is not None:
                            optim.zero_grad(set_to_none=True)
                        loss.backward()
                        if optim is not None:
                            optim.step()

                    # Use unified metrics logging (every rank)
                    with self.metrics.aggregate(stage):
                        self.log_step_metrics_unified(metrics, step)

                    # rank-0 only
                    if self.is_rank0:
                        current_metrics = self.metrics.get_smoothed_values(stage)
                        to_log = stage == STAGE_TRAIN
                        self.update_progress_bar(
                            current_metrics, step, is_rank0=self.is_rank0, to_log=to_log
                        )

                    # Clear memory (avoid per-batch empty_cache to prevent
                    # fragmentation)
                    del batch, prepared_batch, outputs, loss, metrics

        finally:
            if self.is_rank0:
                self.close_progress_bar()

        # Memory cleanup after epoch (not per-batch to avoid fragmentation)
        torch.cuda.empty_cache()

        # Synchronize metrics across all ranks for correct global averages
        _, world_size, _ = self._get_distributed_config()
        self.metrics.sync_distributed(device, world_size=world_size)

        stage_metrics = self.metrics.get_smoothed_values(stage)

        # Subclass hook for validation logging, metrics, etc.
        self._on_epoch_end(stage, stage_metrics, master_process)

        return stage_metrics

    @abstractmethod
    def _on_epoch_end(
        self,
        stage: str,
        stage_metrics: dict[str, float],
        master_process: bool,
    ) -> None:
        """Handle end-of-epoch event for stage-specific behavior.

        Subclasses can override this method to add stage-specific logging or
        other behavior. Default implementation does nothing.

        Args:
            stage: Stage name ("train", "val", or "test")
            stage_metrics: Aggregated metrics for the stage
            master_process: Whether this is the master process
        """

    @abstractmethod
    def _step_core(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, float], Any]:
        """Pure compute: parse, forward, target, loss, metrics. No optimizer/AMP.

        This is the core training step that varies significantly between trainers.
        Each trainer must implement its own forward pass logic, loss computation,
        and metrics calculation.

        Args:
            model: The model (DDP-wrapped or not)
            batch: Batch dictionary with unified structure
            stage: Training stage ("train", "val", "test")

        Returns:
            Tuple containing:
                - loss: Computed loss (torch.Tensor)
                - metrics: Dictionary of metrics (Dict[str, float])
                - outputs: Model outputs (Any - depends on model type)
        """

    @abstractmethod
    def _extract_target_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Extract target from unified batch structure for loss computation.

        Different trainers work with different data structures and target formats.
        This method ensures each trainer can properly extract its required targets.

        Args:
            batch: Batch dictionary with unified structure

        Returns:
            torch.Tensor: Target tensor for loss computation
        """

    @abstractmethod
    def _get_main_output(self, outputs: Any) -> torch.Tensor:
        """Handle different model output formats for consistent processing.

        Different models return different output structures (tensors, tuples, dicts).
        This method ensures each trainer can properly extract the main output
        for loss computation and metrics.

        Args:
            outputs: Raw model outputs (can be tensor, tuple, dict, etc.)

        Returns:
            torch.Tensor: Main output tensor for loss computation
        """

    @abstractmethod
    def _execute_training_logic(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        """Execute training logic using provided infrastructure from BaseTrainer.

        Args:
            train_loader: Training DataLoader (already configured with proper sampler)
            val_loader: Validation DataLoader (already configured)
            test_loader: Test DataLoader (already configured)
        """

    def predict_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Unified prediction step that performs processor::process(stage='test').

        Returns:
            Dict[str, Any]: Processed outputs from the processor.
        """
        if self.processor is None:
            raise RuntimeError(
                "predict_step requires a processor. Configure trainer.processor "
                "in Hydra config."
            )

        self.model.eval()

        prepared_batch = {
            key: (
                value.to(self.device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for key, value in batch.items()
        }

        with torch.no_grad():
            raw_outputs = self.model(prepared_batch)
            direction_name = self.get_direction_name()
            if direction_name and not prepared_batch.get(BATCH_KEY_DIRECTION):
                prepared_batch[BATCH_KEY_DIRECTION] = direction_name

            processed_outputs = self.processor.process(
                raw_outputs,
                prepared_batch,
                stage=STAGE_TEST,
            )

        return processed_outputs

    def run_training(self, dataset_tuple: tuple[Any, ...]) -> None:
        """Unified training entry point with clean checkpoint loading and clear
        separation of concerns.

        Args:
            dataset_tuple: Tuple of (train_loader, val_loader, test_loader) or
                dataset(s) used to build loaders. Exact type depends on trainer config.

        This method is the single entry point for all training scenarios. It
        implements a clean, logical checkpoint loading approach:

        - Reading torchrun environment variables (WORLD_SIZE, LOCAL_RANK, RANK)
        - Hardware setup and device assignment
        - Checkpoint loading: load from disk → store → load weights → DDP wrap
          → load trainer states
        - Auto-backend detection and DDP initialization
        - Model wrapping with DDP (auto-broadcasts weights from rank-0)
        - Dataloader creation with proper DistributedSampler setup
        - Optimizer and scheduler creation AFTER DDP wrapping
        - Clean trainer state loading via broadcast (works for all scenarios)
        - Delegation to subclasses' _execute_training_logic() for training
          implementation

        Always call this method - it uses the same clean approach for both
        single-GPU and DDP scenarios, providing consistent behavior across all
        hardware configurations.
        """
        try:
            # Always initialize DDP - it creates consistent environment for
            # all scenarios
            self.init_ddp()

            rank, world_size, local_rank = self._get_distributed_config()
            self.setup_hardware()
            self.setup_logging()

            train_loader, val_loader, test_loader = self._build_dataloaders(
                dataset_tuple
            )
            self._move_model_to_device(train_loader=train_loader)

            # === Print model parameters (rank 0 only to avoid duplicate output) ===
            if self.is_rank0:
                logger.info("Model Architecture Summary:")
                print_model_parameters(self.model)

            # === CRITICAL: Load checkpoint BEFORE DDP wrapping ===
            # Clean approach: load checkpoint → store → load weights → DDP wrap → load
            # trainer states
            # TIMING: Load checkpoint and prepare model weights BEFORE DDP wrapping
            # (rank 0 only)
            checkpoint_loaded = False
            if self.resume_training or self.load_model_weights:
                logger.info("Resuming training from checkpoint...")
                # Load from disk and broadcast status so all ranks know resume state
                checkpoint_loaded = self.load_checkpoint_unified(train_loader)

                # Apply weights on rank 0 before DDP so broadcast has correct state
                if checkpoint_loaded and self.load_model_weights:
                    self.prepare_model_weights(models={"model": self.model})

            # TIMING: DDP wrapping broadcasts model weights from rank 0 to all ranks
            # DDP initialization and wrapping (auto-backend detection)
            self._setup_ddp_wrapping(self.model, local_rank)
            self.create_optimizer()
            self.create_scheduler()

            # TIMING: Load trainer states AFTER DDP wrapping and broadcast to all ranks
            # === CRITICAL: Load trainer states AND get training metadata AFTER DDP
            # wrapping ===
            if checkpoint_loaded and self.resume_training:
                # checkpoint_loaded was already broadcast; all ranks load trainer state.
                self.load_trainer_states()

            self._execute_training_logic(train_loader, val_loader, test_loader)

        except Exception:
            logger.error("Error in training", exc_info=True)
            raise
        finally:
            self.cleanup_resources(dataset_tuple)
            logger.info("Training completed or terminated")

        # Final Architecture Summary (Clean Separation of Concerns):
        # 1. Pick device (always)
        # 2. Rank-0 loads checkpoint from disk, stores it, AND broadcasts status to all
        #    ranks (load_checkpoint_unified)
        # 3. Build model → move to device
        # 4. If checkpoint exists: rank-0 loads model weights into unwrapped model
        #    (prepare_model_weights)
        # 5. Always initialize DDP (creates consistent environment for all scenarios)
        # 6. Auto-detect backend (GPU: nccl, CPU: gloo)
        # 7. Wrap with DDP if multi-GPU (auto-broadcasts weights from rank-0)
        # 8. Create optimizer/scheduler/scaler (post-wrap)
        # 9. Trainer state: rank-0 loads from stored checkpoint → broadcast → all ranks
        #    apply (load_trainer_states) AND get training metadata
        # 10. Always cleanup DDP (handles both single-GPU and DDP scenarios)
