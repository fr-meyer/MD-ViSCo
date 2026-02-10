"""Enhanced base evaluator with integrated logging and unified runtime
orchestration.

Defines the common interface and shared functionality for all evaluators,
with full logging integration.

Architecture:
============
The evaluator follows the same clean separation pattern as the trainer:

- test.py: Thin entry point - only configuration validation, logging,
  and Hydra instantiation
- BaseEvaluator: Runtime orchestration - handles all hardware setup,
  environment setup, model setup, dataloader creation, evaluation
  execution, and cleanup

Key Features:
- Unified evaluation entry point: run_evaluation() method handles all orchestration
- Automatic environment setup (hardware, logging, device)
- Integrated checkpoint loading and model setup
- Progress bar and metrics aggregation setup
- Error handling and resource cleanup
- Clean delegation to subclass-specific evaluation logic
"""

# Standard library imports
import copy
import logging
import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import MISSING
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Local imports
from src.core.direction import Directions
from src.core.direction import DirectionsConfig
from src.loggings.metrics import metrics
from src.loggings.progress_bar import ProgressBar
from src.loggings.progress_bar import ProgressBarConfig
from src.model.base_model import BaseModel
from src.model.base_model import BaseModelConfig
from src.processors.metrics_utils import extract_target_from_batch
from src.processors.output_processor import OutputProcessor
from src.utils.checkpoint_io import CheckpointIO
from src.utils.checkpoint_io import CheckpointIOConfig
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.checkpoint_manager import CheckpointManagerConfig
from src.utils.constants import PROCESSOR_KEY_PREDICTIONS
from src.utils.dataset_utils import get_dataset_attribute

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorBaseConfig:
    """Base configuration for evaluators with Hydra compatibility and
    support for both model-based and checkpoint-free evaluation modes.

    Checkpoint Managers:
        Evaluators support multiple checkpoint managers via the
        ``checkpoint_managers`` dict (e.g., ``save``, ``load``,
        ``source``). Typical roles include:

        - ``save``: Primary model checkpoints for evaluation.
        - ``load``: Pretrained weights staged separately from primary saves.
        - ``source``: Auxiliary models required during evaluation
          (e.g., AF classifiers).

        The ``checkpoint_mapping`` field defines how checkpoints are
        loaded for each component (model, stage1, stage2, etc.),
        specifying which manager to use and which keys to extract.

    Non-checkpoint Modes:
        When ``load_model_weights=False`` there are two supported options:

        - **GT-only (model disabled)** — disable the evaluator model
          with the Hydra override ``model@evaluator.model=null`` and
          omit all checkpoint-related fields (``checkpoint_managers``,
          ``checkpoint_io``, ``trainer_name``, ``dataset_name``,
          ``checkpoint_epoch``). This mode is ideal when metrics are
          derived purely from ground-truth signals.

        - **Checkpoint skip (random initialization)** — keep model and
          checkpoint configuration but set ``load_model_weights`` to
          ``False``. The evaluator instantiates the model with randomly
          initialized weights and skips checkpoint discovery/loading
          entirely. Use this strictly for infrastructure validation—not
          for evaluating model quality.

        Preprocessing validation uses trainer-aligned semantics: missing
        configs emit warnings instead of raising errors, enabling a
        frictionless workflow when collate functions are unnecessary.

    Validation Rule:
        When ``load_model_weights=True`` (default), ``model``,
        ``checkpoint_managers`` dict, ``checkpoint_mapping``,
        ``checkpoint_io``, ``trainer_name``, ``dataset_name``, and
        ``checkpoint_epoch`` are required and will be validated in
        ``__init__``. Set ``load_model_weights=False`` to bypass this
        requirement.

    Note:
        Evaluators only support single direction mode. The ``direction_mode``
        field is fixed to "single" and cannot be changed. VitalsDataset is
        extracted from the dataset configuration automatically and does not
        need to be specified in the config. Processor is auto-instantiated by
        Hydra when configured with ``_target_``.

    Example:
        .. code-block:: yaml

            evaluator:
              checkpoint_managers:
                save:
                  base_dir: ./weights/
                af_classifier:
                  base_dir: ./af_weights/

        Accessing managers in subclass code::

            # Access managers via checkpoint_mapping
            # For multi-stage models, mapping defines which manager to
            # use for each component

            evaluator:
              _target_: src.evaluators.feature_extraction.FeatureExtractionEvaluator
              load_model_weights: false
              # model, checkpoint_manager, checkpoint_io omitted
              directions: ppg2ecg
              batch_size: 32
              seed: 42
    """

    _target_: str = "src.evaluators.base_evaluator.BaseEvaluator"
    enabled: bool = True
    sampling_rate: int = 125

    # Core infrastructure components (same as trainer)
    model: BaseModelConfig | None = None
    checkpoint_managers: dict[str, CheckpointManagerConfig] = field(
        default_factory=dict
    )
    checkpoint_mapping: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            # For two/multi-stage: add 'stage1', 'stage2' etc. via config overrides
        }
    )
    checkpoint_io: CheckpointIOConfig | None = None
    checkpoint_manager_name: str | None = None
    progress_bar: ProgressBarConfig = MISSING
    directions: DirectionsConfig = MISSING
    processor: Any | None = None
    direction_mode: str = "single"
    max_source_channels: int | None = None

    # Collate function configuration
    input_preprocessing: dict[str, Any] = MISSING

    demographics_text_encoder: Any | None = None
    """Optional demographics text encoder for generating text embeddings.

    If provided and demographics are in include_list, will generate input_ids
    and attention_mask for models that use text encoder pipeline (e.g., MDViSCo
    BPModel with pi=True). Configured via:
    ``/input_preprocessor@demographics_text_encoder: demographics_text_encoder``
    """

    # Evaluation-specific parameters
    load_model_weights: bool = True
    strict_loading: bool = True
    checkpoint_epoch: int | None = None
    trainer_name: str | None = None
    dataset_name: str | None = None

    # Training parameters (needed for checkpoint path building)
    batch_size: int = MISSING
    num_epochs: int = MISSING  # Checkpoint epoch to load
    learning_rate: float = MISSING
    scheduler_patience: int | None = MISSING
    early_stopping_patience: int = MISSING
    is_pretraining: bool = False
    is_finetuning: bool = False
    is_few_shot: bool = False
    use_patient_split: bool = False
    use_patient_information: bool = True
    seed: int = 42
    use_wcl: bool = False

    # Logging configuration
    log_metrics: bool = True
    save_results: bool = True
    logging_level: str = "INFO"
    log_file_path: str = "logs/test.log"

    # Hardware configuration (for test environment)
    num_threads: int = 2
    num_workers: int = 0
    pin_memory: bool = True
    timeout: int = 3600

    # WandB configuration
    hydra_run_dir: str | None = None


class BaseEvaluator(ABC):
    """Enhanced abstract base class for all evaluators with logging integration.

    Model-dependent methods (require a loaded model):
        - ``_predict_batch``
        - ``_load_model_weights_from_checkpoint``
        - ``_get_model_name``
        - ``get_model``

    Model-agnostic utilities (safe in GT-only mode):
        - ``to_device``
        - ``_move_batch_to_device``
        - ``_process_batch_modern`` (delegates to
          ``metrics_utils.extract_target_from_batch``)
        - Dataloader setup helpers and progress/metrics infrastructure.

    GT-Only Evaluation Mode:
        Evaluators can run without a model when
        ``load_model_weights=False``. In this mode:
        - Set ``model=None``, ``checkpoint_managers={}``,
          ``checkpoint_io=None`` in config
        - Pre-flight validation (lines 244-261) will skip model requirement checks
        - ``run_evaluation()`` will skip model setup (lines 315-319)
        - Subclasses must implement GT-only logic in
          ``_execute_evaluation_logic()`` by branching on
          ``self.model is None``
        - Model-dependent methods (``_predict_batch``, etc.) will
          raise ``RuntimeError`` if called

        See ``FeatureExtractionEvaluator`` for a reference
        implementation of GT-only evaluation.

    Evaluators that operate without inference must avoid the
    model-dependent helpers or provide alternative implementations
    tailored for GT-only workflows.

    Multi-Manager Support:
        Evaluators can be configured with multiple checkpoint managers
        (e.g., auxiliary AF classifiers). Access named managers via
        ``self.get_checkpoint_manager('manager_name')`` or directly via
        ``self.checkpoint_managers['manager_name']``. The default manager
        (typically 'save') is available via ``self.get_checkpoint_manager()``.

    Checkpoint Epoch Behavior:
        When ``load_model_weights=True``, the ``checkpoint_epoch``
        parameter controls which checkpoint to load. Both an explicit
        ``checkpoint_epoch=None`` and an omitted ``checkpoint_epoch``
        (using the default value) are treated as "best checkpoint" mode,
        which loads the suffix-free checkpoint filename (typically the
        best-performing checkpoint saved during training).

        For code paths that go through ``src.test.main``,
        ``validate_test_config`` requires ``checkpoint_epoch`` to be
        explicitly provided in the configuration. For direct
        instantiation of ``BaseEvaluator`` or its subclasses (bypassing
        ``test.py``), omitting ``checkpoint_epoch`` or setting it to
        ``None`` will result in best-checkpoint loading.

        To load a specific epoch checkpoint, provide an integer value:
        ``checkpoint_epoch=100``.
    """

    def __init__(
        self,
        # Core parameters
        enabled: bool = True,
        sampling_rate: int = 125,
        # Core infrastructure components
        model: BaseModel | None = None,
        *args: Any,
        checkpoint_managers: dict[str, CheckpointManager] | None = None,
        checkpoint_mapping: dict[str, dict[str, str]] | None = None,
        checkpoint_io: CheckpointIO | None = None,
        progress_bar: ProgressBar | None = None,
        directions: Directions | None = None,
        processor: OutputProcessor | None = None,
        # Remove vitals_dataset from constructor - will be extracted from dataset
        direction_mode: str = "single",
        max_source_channels: int | None = None,
        # Collate function configuration
        input_preprocessing: dict[str, Any] | None = None,
        demographics_text_encoder: Any | None = None,
        # MANDATORY checkpoint loading parameters
        load_model_weights: bool = True,
        strict_loading: bool = True,  # Whether to strictly enforce
        # that keys in state_dict match the model
        trainer_name: str | None = None,
        dataset_name: str | None = None,
        # Training parameters (needed for checkpoint path building)
        batch_size: int = MISSING,  # Required
        num_epochs: int = MISSING,  # Required - checkpoint epoch to load
        learning_rate: float = MISSING,  # Required
        scheduler_patience: int | None = MISSING,
        early_stopping_patience: int = MISSING,  # Required
        is_pretraining: bool = False,
        is_finetuning: bool = False,
        use_patient_split: bool = False,
        use_patient_information: bool = True,
        seed: int = 42,
        use_wcl: bool = False,
        checkpoint_epoch: int | None = None,
        # Logging parameters
        log_metrics: bool = True,
        save_results: bool = True,
        logging_level: str = "INFO",
        log_file_path: str = "logs/test.log",
        # Hardware parameters
        num_threads: int = 2,
        num_workers: int = 0,
        pin_memory: bool = True,
        timeout: int = 3600,
        hydra_run_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the base evaluator with runtime infrastructure.

        Args:
            checkpoint_managers: Dict of checkpoint managers for
                multi-model workflows (e.g., ``{'save': manager,
                'stage1': manager1, 'stage2': manager2}``).
            checkpoint_mapping: Dict mapping component names to their
                checkpoint loading configuration. Each entry specifies
                which manager to use and which keys to extract. Example:
                ``{'model': {'manager': 'save', 'checkpoint_key':
                'model_state_dict'}, ...}``
            input_preprocessing: Optional input preprocessing config.
                Can be ``None`` when running in GT-only mode (no
                dataloader), but should provide ``'source'`` and
                ``'target'`` sections when dataloaders are required.

        Other parameters (batch_size, num_epochs, learning_rate,
        logging_level, etc.) are passed through for checkpoint path
        building and runtime config; see EvaluatorBaseConfig and the
        method signature for the full list.
        """
        # Direct assignment from individual parameters (Hydra-provided)
        self.enabled = enabled
        self.sampling_rate = sampling_rate

        # Core infrastructure components
        self.model = model
        self.checkpoint_managers = {}
        # Do not inject a hard-coded default mapping here; require explicit config
        # via Hydra/YAML/CLI so evaluators can freely choose which manager to use.
        self.checkpoint_mapping = checkpoint_mapping or {}
        self.checkpoint_io = checkpoint_io
        self.progress_bar = progress_bar
        self.directions = directions
        self.processor = processor
        # Remove vitals_dataset assignment - will be extracted from dataset when needed
        self.direction_mode = direction_mode
        self.max_source_channels = max_source_channels

        # Collate function configuration
        self.input_preprocessing = self._normalize_input_preprocessing(
            input_preprocessing
        )
        self.demographics_text_encoder = demographics_text_encoder
        self._validate_directions_against_preprocessing()

        # MANDATORY checkpoint loading parameters
        self.load_model_weights = load_model_weights
        self.strict_loading = strict_loading
        self.checkpoint_epoch = checkpoint_epoch
        self.trainer_name = trainer_name
        self.dataset_name = dataset_name

        # Training parameters (needed for checkpoint path building)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.early_stopping_patience = early_stopping_patience
        self.is_pretraining = is_pretraining
        self.is_finetuning = is_finetuning
        self.use_patient_split = use_patient_split
        self.use_patient_information = use_patient_information
        self.seed = seed
        self.use_wcl = use_wcl

        # Evaluators default to requiring a non-None test_loader unless they
        # explicitly opt in to supporting None. Subclasses that operate without
        # a DataLoader (e.g., pure logging/aggregation evaluators) should set
        # this flag to True in their own __init__ after calling super().__init__.
        self._supports_none_test_loader = False

        # Logging parameters
        self.log_metrics = log_metrics
        self.save_results = save_results
        self.logging_level = logging_level
        self.log_file_path = log_file_path

        # Hardware parameters
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.hydra_run_dir = hydra_run_dir

        self.checkpoint_managers = (
            dict(checkpoint_managers) if checkpoint_managers else {}
        )

        self._validate_checkpoint_mapping()

        if self.load_model_weights:
            missing_components = []
            if self.model is None:
                missing_components.append("model")
            if self.checkpoint_io is None:
                missing_components.append("checkpoint_io")
            if missing_components:
                missing_str = ", ".join(missing_components)
                raise ValueError(
                    f"When load_model_weights=True, model and checkpoint_io "
                    f"must be provided. To run without checkpoints, set "
                    f"load_model_weights=False and either disable the model "
                    f"(GT-only) or acknowledge you are running with random "
                    f"initialization (checkpoint skip). Missing: {missing_str}"
                )
            if not self.checkpoint_managers:
                raise ValueError(
                    "When load_model_weights=True, at least one checkpoint "
                    "manager must be provided via checkpoint_managers dict. "
                    "To bypass checkpoints, set load_model_weights=False."
                )
            if not self.checkpoint_mapping:
                raise ValueError(
                    "When load_model_weights=True, checkpoint_mapping must "
                    "be provided. It defines which managers to use for "
                    "loading each component (model, stage1, stage2, etc.)."
                )

            # Path format may reference trainer_name or dataset_name
            requires_trainer_aligned = False
            for _manager_name, manager in self.checkpoint_managers.items():
                format_str = manager.path_format + manager.filename_format
                if any(
                    key in format_str
                    for key in [
                        "trainer_name",
                        "dataset_name",
                        "model_name",
                        "batch_size",
                        "num_epochs",
                        "learning_rate",
                        "scheduler_patience",
                        "early_stopping_patience",
                        "is_finetuning",
                        "use_patient_split",
                        "use_patient_information",
                        "use_wcl",
                    ]
                ):
                    requires_trainer_aligned = True
                    break

            # Only require trainer_name/dataset_name when at least one manager uses
            # trainer-aligned formats
            if requires_trainer_aligned:
                missing_identifiers = []
                if self.trainer_name is None:
                    missing_identifiers.append("trainer_name")
                if self.dataset_name is None:
                    missing_identifiers.append("dataset_name")
                if missing_identifiers:
                    raise ValueError(
                        "When load_model_weights=True and at least one "
                        "checkpoint manager uses trainer-aligned path "
                        "formats (references trainer_name, dataset_name, or "
                        "other training hyperparameters), checkpoint "
                        "identification fields must be provided. Missing: "
                        f"{', '.join(missing_identifiers)}"
                    )
            logger.info(
                "Pre-flight validation passed (%s): %d checkpoint manager(s) available",
                self._evaluation_mode_label(),
                len(self.checkpoint_managers),
            )
        else:
            logger.info(
                "Evaluator configured for %s; skipping checkpoint "
                "validation because load_model_weights=False.",
                self._evaluation_mode_label(),
            )

    def _normalize_input_preprocessing(self, config: Any) -> dict[str, Any]:
        """Normalize Hydra containers to plain dict/list structures
        (trainer-aligned semantics).

        Returns:
            Dict[str, Any]: Normalized preprocessing configuration.
            Returns an empty dict for ``None``/``MISSING`` configs,
            allowing GT-only evaluation mode without dataloaders.
        """
        if config is None or config is MISSING:
            logger.warning(
                "input_preprocessing is None or MISSING. Defaulting to "
                "empty mapping {}. This is acceptable when running "
                "without checkpoint loading (GT-only or checkpoint skip)."
            )
            return {}

        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)

        if isinstance(config, dict):
            return copy.deepcopy(config)

        logger.warning(
            "input_preprocessing must be a dict-like mapping, got %s. "
            "Defaulting to empty mapping {}.",
            type(config),
        )
        return {}

    def _validate_checkpoint_mapping(self) -> None:
        """Validate checkpoint_mapping manager references."""
        if not self.checkpoint_mapping:
            return

        for component, mapping in self.checkpoint_mapping.items():
            manager_name = mapping.get("manager")
            if manager_name and manager_name not in self.checkpoint_managers:
                available = list(self.checkpoint_managers.keys())
                logger.warning(
                    f"checkpoint_mapping['{component}'] references manager '{
                        manager_name
                    }' "
                    f"which is not in checkpoint_managers. Available "
                    f"managers: {available}. "
                    f"This component will be skipped during loading."
                )

    def _validate_directions_against_preprocessing(self) -> None:
        """Validate that all direction source vitals exist in
        input_preprocessing (trainer-aligned semantics).

        Warns and returns early for missing/invalid configs. Only
        raises ``ValueError`` when a required vital is actually missing.
        """
        if not self.input_preprocessing or "source" not in self.input_preprocessing:
            logger.warning(
                "Cannot validate directions: input_preprocessing is "
                "missing or has no 'source' key. Skipping validation."
            )
            return

        source_config = self.input_preprocessing.get("source")

        if isinstance(source_config, dict):
            source_configs = [source_config]
        elif isinstance(source_config, list):
            source_configs = source_config
        else:
            logger.warning(
                "Cannot validate directions: input_preprocessing['source'] "
                "has unexpected type %s. Expected dict or list. "
                "Skipping validation.",
                type(source_config),
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
                "Cannot validate directions: self.directions is "
                "missing or invalid. Skipping validation."
            )
            return

        from src.core.domain import Vital

        for direction in self.directions.directions:
            for vital in direction.source:
                vital_name = (
                    vital.value.upper()
                    if isinstance(vital, Vital)
                    else str(vital).upper()
                )

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
            "Validation passed: All direction source vitals %s are present in "
            "input_preprocessing['source']",
            available_vitals,
        )

    def is_main_process(self) -> bool:
        """Check if current process is the main process (rank 0).

        This method provides DDP-aware process checking for
        evaluators. Returns True for single-process mode or rank 0
        in DDP mode.
        """
        try:
            import torch.distributed as dist

            if not dist.is_initialized():
                return True  # Single process mode
            return dist.get_rank() == 0
        except ImportError:
            # PyTorch distributed not available
            return True

    def run_evaluation(self, test_dataset) -> tuple[dict[str, Any], DataLoader | None]:
        """Run evaluation with unified infrastructure and CSV run management.

        This method handles all common evaluation concerns:
        - Environment setup (hardware, logging, device)
        - Model setup and device placement
        - Progress bar initialization
        - Metrics aggregation setup
        - CSV run management
        - Error handling and cleanup
        - Delegation to subclass-specific evaluation logic
        Subclasses should call :meth:`_predict_batch` for inference
        to ensure processor-based
        post-processing is applied consistently with the trainer.

        Args:
            test_dataset: Test dataset (will be converted to DataLoader internally)

        Returns:
            Tuple[Dict[str, Any], DataLoader | None]: Evaluation results and test_loader

        Raises:
            RuntimeError: If evaluation fails
        """
        import time

        evaluation_start_time = time.time()
        test_loader = None

        try:
            # 1. Environment setup (similar to trainer)
            self._setup_evaluation_environment()

            # 2. Create test dataloader FIRST (like trainer builds dataloaders first)
            should_create_loader = True
            if not self.load_model_weights:
                preprocessing = self.input_preprocessing or {}
                has_source = (
                    isinstance(preprocessing, dict) and "source" in preprocessing
                )
                has_target = (
                    isinstance(preprocessing, dict) and "target" in preprocessing
                )
                if not has_source or not has_target:
                    raise ValueError(
                        "GT-only path (load_model_weights=False) "
                        "requires input_preprocessing with 'source' "
                        "and 'target' sections to construct a "
                        "dataloader. "
                        f"Got source={has_source}, target={has_target}. "
                        "Configure input_preprocessing properly or "
                        "use model-based evaluation."
                    )
            if should_create_loader:
                test_loader = self._setup_dataloader(test_dataset)
                logger.info("Created test dataloader with %d batches", len(test_loader))

            # 3. Model setup (now with access to test_loader for checkpoint loading)
            if self.model is None:
                logger.info("GT-only (model disabled): model setup skipped")
                model = None
            else:
                if test_loader is None:
                    raise ValueError(
                        "test_loader is required for model checkpoint loading"
                    )
                model = self._setup_model_for_evaluation(test_loader)

            # 4. Progress bar setup
            if test_loader is not None:
                self._setup_progress_bar(
                    total=len(test_loader),
                    description=self._get_evaluation_description(),
                )
            else:
                logger.debug("Progress bar setup skipped: no dataloader available.")

            # 5. Start new CSV run
            if self.progress_bar and self.progress_bar.csv:
                self.progress_bar.csv.start_new_run("evaluation")

            # 6. Metrics aggregation setup
            with metrics.aggregate(self._get_metrics_name()) as aggregator:
                # 7. Delegate to subclass-specific evaluation logic
                if test_loader is None and not self._supports_none_test_loader:
                    raise ValueError(
                        "test_loader is None. Evaluators must either "
                        "construct a dataloader or explicitly support "
                        "None by setting "
                        "_supports_none_test_loader = True."
                    )
                assert test_loader is not None  # narrowed by check above
                results = self._execute_evaluation_logic(model, test_loader, aggregator)

            # 8. Post-processing and finalization
            self._finalize_evaluation_results(results, test_loader)

            # Log evaluation duration
            duration = time.time() - evaluation_start_time
            logger.info("Evaluation completed in %.2f seconds", duration)

            return results, test_loader

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during evaluation: {e}")
            self._handle_oom_error()
            raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_evaluation_resources()

    @abstractmethod
    def _execute_evaluation_logic(
        self,
        model: nn.Module | None,
        test_loader: DataLoader,
        aggregator: Any,
    ) -> dict[str, Any]:
        """Execute evaluation logic without infrastructure concerns.

        This method contains only the core evaluation logic specific
        to each evaluator type.
        All infrastructure concerns (setup, cleanup, error handling) are handled by the
        unified entry point.

        Args:
            model: Model ready for evaluation
            test_loader: DataLoader containing test data
            aggregator: Metrics aggregator for logging

        Returns:
            Dict[str, Any]: Evaluation results
        """

    @abstractmethod
    def print_results(
        self,
        results: dict[str, Any],
        test_loader: DataLoader | None = None,
    ) -> None:
        """Print formatted evaluation results.

        Args:
            results: Dictionary containing evaluation results
            test_loader: Optional DataLoader for additional context
        """

    def _setup_model_for_evaluation(self, test_loader: DataLoader) -> nn.Module | None:
        """Set up model for evaluation with checkpoint loading if required.

        Args:
            test_loader: Test loader needed for checkpoint path building.

        Returns:
            Optional[nn.Module]: Model ready for evaluation, or None
                if model is disabled.

        Raises:
            ValueError: If checkpoint loading fails or required checkpoints are missing.
            RuntimeError: If model initialization or device placement fails.
        """
        if self.model is None:
            logger.info("GT-only (model disabled): skipping model setup")
            return None

        if self.load_model_weights:
            self._load_model_checkpoint(test_loader)
        else:
            logger.warning(
                "Checkpoint skip: load_model_weights=False; continuing with randomly "
                "initialized parameters."
            )

        self._finalize_model_device()
        self._prepare_processor()

        return self.model

    def _evaluation_mode_label(self) -> str:
        """Human-readable description of the evaluator's checkpoint mode."""
        if self.load_model_weights:
            return "model-based evaluation (checkpoint restored)"
        if self.model is None:
            return "GT-only (model disabled)"
        return "Checkpoint skip (random initialization)"

    def _load_model_checkpoint(self, test_loader: DataLoader) -> None:
        """Load checkpoint and apply weights when a model is available.

        Args:
            test_loader: Test loader needed for checkpoint path building.

        Raises:
            FileNotFoundError: If checkpoint file is missing or path is invalid.
            RuntimeError: If checkpoint loading or weight application fails.
        """
        if self.model is None:
            return None
        assert self.checkpoint_io is not None, "checkpoint_io required for loading"

        checkpoint = self._load_checkpoint_from_disk(test_loader)
        # _load_checkpoint_from_disk() raises ValueError for genuine
        # failures (no checkpoints loaded, missing required checkpoints
        # for two-stage models). It only returns None for intentional
        # skips (non-main ranks in DDP), in which case we return
        # early without loading.
        if checkpoint is None:
            return None
        assert self.model is not None, "model required for weight loading"

        self._load_model_weights_from_checkpoint(checkpoint)
        logger.info("Model weights loaded successfully from checkpoint")

    def _finalize_model_device(self) -> None:
        """Finalize model device placement and evaluation mode."""
        if self.model is None:
            return
        model = self.model
        model.eval()
        model.to(self.device)
        logger.info("Model moved to device: %s", self.device)

    def _prepare_processor(self) -> OutputProcessor:
        """Ensure a processor is configured for evaluation and log its usage.

        Returns:
            OutputProcessor: The configured processor instance

        Raises:
            RuntimeError: If no processor is configured
        """
        if self.processor is None:
            logger.error(
                "No processor configured; set evaluator.processor in "
                "Hydra defaults or pass "
                "an OutputProcessor instance"
            )
            raise RuntimeError(
                "Processor is required for evaluation. Configure "
                "evaluator.processor via "
                "Hydra defaults or provide a pre-built OutputProcessor."
            )

        logger.info("Using processor instance: %s", type(self.processor).__name__)
        return self.processor

    def _resolve_checkpoint_path(
        self,
        component: str,
        manager_name: str,
        manager: Any,
        test_loader: DataLoader,
    ) -> str | None:
        """Build checkpoint path for a component using manager and path kwargs.

        Args:
            component: Component name (e.g., 'model', 'stage1')
            manager_name: Name of the checkpoint manager
            manager: CheckpointManager instance
            test_loader: Test loader for dataset attributes (e.g., train_ratio)

        Returns:
            Resolved path string, or None if path building fails (e.g., KeyError).
        """
        format_str = manager.path_format + manager.filename_format
        requires_full_kwargs = manager_name == "save" or any(
            key in format_str
            for key in [
                "trainer_name",
                "dataset_name",
                "model_name",
                "batch_size",
                "num_epochs",
                "learning_rate",
                "scheduler_patience",
                "early_stopping_patience",
                "is_finetuning",
                "use_patient_split",
                "use_patient_information",
                "use_wcl",
            ]
        )
        if requires_full_kwargs:
            full_kwargs = {
                "model_name": self._get_model_name(),
                "trainer_name": self.trainer_name,
                "dataset_name": self.dataset_name,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "scheduler_patience": self.scheduler_patience,
                "early_stopping_patience": self.early_stopping_patience,
                "is_finetuning": self.is_finetuning,
                "use_patient_split": self.use_patient_split,
                "use_patient_information": self.use_patient_information,
                "use_wcl": self.use_wcl,
            }
            if "train_ratio" in format_str:
                full_kwargs["train_ratio"] = get_dataset_attribute(
                    test_loader, "train_ratio", required=True
                )
            path_kwargs = full_kwargs
        else:
            path_kwargs = {}
            if "train_ratio" in format_str:
                path_kwargs["train_ratio"] = get_dataset_attribute(
                    test_loader, "train_ratio", required=True
                )
        try:
            return manager.build_path(
                key=component,
                epoch=self.checkpoint_epoch,
                direction=self._get_direction_name(),
                seed=self.seed,
                **path_kwargs,
            )
        except KeyError as e:
            logger.warning(
                f"Failed to build path for component '{component}': missing parameter {
                    e
                }"
            )
            return None

    def _load_checkpoint_at_path(
        self, path: str, component: str
    ) -> dict[str, Any] | None:
        """Load checkpoint from a resolved path.

        Args:
            path: Path to checkpoint file
            component: Component name for logging

        Returns:
            Loaded checkpoint dict, or None on failure.
        """
        if self.checkpoint_io is None:
            raise ValueError("checkpoint_io required for loading")
        try:
            logger.info(f"Loading checkpoint for component '{component}' from: {path}")
            checkpoint = self.checkpoint_io.load(path, map_location="cpu")
            logger.info(f"Checkpoint for component '{component}' loaded successfully")
            return checkpoint
        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint for component '{component}' from {path}: {
                    e
                }"
            )
            return None

    def _load_checkpoint_from_disk(
        self, test_loader: DataLoader
    ) -> dict[str, Any] | None:
        """Gather checkpoints dict from disk using checkpoint_mapping.

        Iterates over checkpoint_mapping to load each component from
        appropriate managers. Supports single/two/multi-stage via
        mapping (e.g., just 'model' for single-stage).

        Args:
            test_loader: Test loader needed for dataset extraction

        Returns:
            Optional[Dict[str, Any]]: Dictionary of loaded checkpoints
                keyed by component name.
                Returns None if not main process or if load_model_weights is False.

        Raises:
            ValueError: If no checkpoints were loaded or if two-stage model is missing
                required checkpoints
        """
        if not self.is_main_process():
            return None  # Only rank 0 loads checkpoint

        if not self.load_model_weights:
            logger.info("load_model_weights=False, skipping checkpoint loading")
            return None

        loaded_checkpoints = {}

        # Evaluator loads model/stage components only
        for component, mapping in self.checkpoint_mapping.items():
            if component not in [
                "model",
                "stage1",
                "stage2",
            ]:  # Evaluator: model components only
                continue

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
            candidate_path = self._resolve_checkpoint_path(
                component, manager_name, manager, test_loader
            )
            if candidate_path is None:
                continue

            if os.path.exists(candidate_path):
                checkpoint = self._load_checkpoint_at_path(candidate_path, component)
                if checkpoint is not None:
                    loaded_checkpoints[component] = checkpoint
            else:
                logger.warning(
                    f"Checkpoint not found for component '{
                        component
                    }' at expected path: "
                    f"{candidate_path}. Skipping this component and "
                    "continuing with remaining "
                    f"checkpoints."
                )

        if len(loaded_checkpoints) == 0:
            configured_components = list(self.checkpoint_mapping.keys())
            available_managers = list(self.checkpoint_managers.keys())
            raise ValueError(
                f"No checkpoints were loaded from disk. At least one "
                "checkpoint must be "
                f"available.\n"
                f"For two-stage models (TwoStageCascadeModel, TwoStageScalingModel): "
                f"ensure that stage1 and stage2 checkpoints exist.\n"
                f"For single-stage models: ensure that the model checkpoint exists.\n"
                f"Configured components in checkpoint_mapping: {
                    configured_components
                }\n"
                f"Available checkpoint managers: {available_managers}"
            )

        # Two-stage enforcement is driven by explicit checkpoint_mapping
        # entries, not by the model class alone. This allows cascade-style models
        # (e.g., PPG2ABPCascade) to be evaluated from a single joint checkpoint
        # when only a 'model' component is mapped.
        is_two_stage_model = (
            "stage1" in self.checkpoint_mapping and "stage2" in self.checkpoint_mapping
        )

        if is_two_stage_model:
            missing_stages = []
            if "stage1" not in loaded_checkpoints:
                missing_stages.append("stage1")
            if "stage2" not in loaded_checkpoints:
                missing_stages.append("stage2")

            if missing_stages:
                missing_str = " and ".join(missing_stages)
                raise ValueError(
                    f"Two-stage model detected but required checkpoint(s) missing: "
                    f"{missing_str}. Both 'stage1' and 'stage2' "
                    "checkpoints are required for "
                    f"two-stage model evaluation. Loaded checkpoints: "
                    f"{list(loaded_checkpoints.keys())}"
                )

        return loaded_checkpoints

    def _load_model_weights_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Load model weights from checkpoint using model.load_from_checkpoint_dict().

        This method delegates to model.load_from_checkpoint_dict() which provides:
        - CheckpointIO integration for state extraction and legacy key mapping
        - Automatic handling of legacy GAN keys ('G_model_state_dict', 'G_state_dict')
        - Automatic DDP 'module.' prefix stripping
        - Uniform logging of missing/unexpected keys
        - Strict mode enforcement via strict_loading flag
        - Support for single-stage and multi-stage models via checkpoint_mapping

        Args:
            checkpoint: Dictionary of component checkpoints loaded from disk
                (e.g., {'model': ckpt_data} for single-stage or
                {'stage1': ckpt1, 'stage2': ckpt2}
                for multi-stage)

        Raises:
            RuntimeError: If model does not provide load_from_checkpoint_dict() method
        """
        self._ensure_model_available("checkpoint loading")

        # Ensure model provides load_from_checkpoint_dict() method
        if not hasattr(self.model, "load_from_checkpoint_dict"):
            raise RuntimeError(
                f"Model {type(self.model).__name__} does not provide "
                f"load_from_checkpoint_dict() method. Expected BaseModel or "
                f"TwoStageCascadeModel subclass."
            )

        # Delegate to model's load_from_checkpoint_dict() with checkpoint_mapping and
        # strict_loading (self.model already asserted non-None in caller)
        model = self.model
        assert model is not None
        load_results = model.load_from_checkpoint_dict(
            checkpoint, self.checkpoint_mapping, strict=self.strict_loading
        )

        # Log success/failure per component
        for component, success in load_results.items():
            if success:
                logger.info(
                    f"Component '{component}' weights loaded successfully "
                    f"(strict={self.strict_loading})"
                )
            else:
                logger.warning(f"Failed to load component '{component}' weights")

        if not any(load_results.values()):
            raise RuntimeError(
                "No model components were loaded successfully from checkpoint"
            )

    def _ensure_model_available(self, context: str) -> None:
        """Ensure the evaluator has a model before performing model-dependent work."""
        if self.model is None:
            raise RuntimeError(
                f"Model is required for {
                    context
                }. This evaluator cannot run in GT-only mode "
                f"for this operation."
            )

    def _get_model_for_inference(self) -> nn.Module:
        """Return model for inference; raises if model is None (e.g. GT-only)."""
        if self.model is None:
            raise RuntimeError("Model is required for inference")
        return self.model

    def _get_model_name(self) -> str:
        """Return model name from the model object.

        In GT-only mode (when model is None), returns "ground_truth"
        instead of raising an error. This allows CSV export and
        result saving to work correctly in GT-only evaluation mode.
        """
        # Handle GT-only mode: return "ground_truth" when no model is available
        if self.model is None:
            return "ground_truth"

        return self.model.model_name

    def _get_dataset_name(self, test_loader: DataLoader) -> str:
        """Return dataset name from the test loader.

        Uses the universal get_dataset_attribute utility which
        handles DataLoader unwrapping and
        direct attribute access.
        """
        if test_loader is None:
            raise ValueError("No test loader available")

        result = get_dataset_attribute(test_loader, "dataset_name", required=True)
        if result is None:
            raise ValueError("dataset_name not found on test loader")
        return str(result)

    def _get_main_output(self, outputs: Any) -> torch.Tensor:
        """Extract main output from v3 canonical format.

        Handles v3 model output formats:
        - Dictionary with 'predictions' key (standard v3 format)
        - Direct tensor output
        - Lists/tuples (returns first element)

        Args:
            outputs: Model output (dict, tensor, list, or tuple)

        Returns:
            torch.Tensor: Main output tensor

        Raises:
            ValueError: If dict outputs missing 'predictions' key or no tensor found
        """
        if isinstance(outputs, dict):
            if PROCESSOR_KEY_PREDICTIONS in outputs:
                predictions = outputs[PROCESSOR_KEY_PREDICTIONS]
                # Handle deep supervision case where predictions is a list/tuple
                if isinstance(predictions, (list, tuple)):
                    return predictions[0]
                return predictions
            else:
                raise ValueError(
                    f"Dict outputs must include '{
                        PROCESSOR_KEY_PREDICTIONS
                    }' key. Got keys: "
                    f"{list(outputs.keys())}"
                )
        elif isinstance(outputs, (list, tuple)):
            return outputs[0]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError(
                f"Unexpected output format: {type(outputs)}. Expected dict with "
                f"'{PROCESSOR_KEY_PREDICTIONS}', tensor, list, or tuple."
            )

    def _prefix_keys(self, metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
        """Add consistent prefixes and filter invalid values."""
        prefixed = {}
        for k, v in metrics.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                prefixed[f"{prefix}/{k}"] = v
        return prefixed

    def _log_metrics(
        self,
        metrics_dict: dict[str, Any],
        step: int | None = None,
        prefix: str = "eval",
    ) -> None:
        """Log metrics with key management using the progress bar."""
        if self.progress_bar is None:  # DDP safety
            return

        clean_metrics = self._prefix_keys(metrics_dict, prefix)
        # Use proper parameters to match trainer pattern
        self.progress_bar.update(
            metrics_dict=clean_metrics,
            step=step,
            is_rank0=True,  # Evaluators run single-process
            to_log=False,  # Don't log every batch during evaluation
        )

    def update_progress_bar(
        self,
        metrics_dict: dict[str, Any] | None = None,
        step: int | None = None,
        is_rank0: bool = True,
        to_log: bool = False,
    ) -> None:
        """Update progress bar with proper parameters (aligned with trainer).

        Args:
            metrics_dict: Dictionary of metrics to display
            step: Current step number
            is_rank0: Whether this is the main process (evaluators run single-process)
            to_log: Whether to log to WandB (False for evaluation)
        """
        if self.progress_bar is None:
            return

        # Use same pattern as trainer
        self.progress_bar.update(
            metrics_dict=metrics_dict, step=step, is_rank0=is_rank0, to_log=to_log
        )

    def _format_batch_metrics_for_progress(
        self, batch_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Format batch metrics for progress bar display (simplified version).

        Args:
            batch_metrics: Raw batch metrics from evaluation

        Returns:
            dict: Simplified metrics for progress bar display
        """
        progress_metrics = {}

        # Extract mean values for progress bar display
        if isinstance(batch_metrics, dict):
            # First, try to use summary metrics if available
            # (NEW - for waveform evaluator)
            if "batch_mae" in batch_metrics:
                progress_metrics["mae"] = batch_metrics["batch_mae"]
            if "batch_corr" in batch_metrics:
                progress_metrics["corr"] = batch_metrics["batch_corr"]

            # Handle blood pressure specific summary metrics (NEW)
            if "batch_sbp_mae" in batch_metrics:
                progress_metrics["sbp_mae"] = batch_metrics["batch_sbp_mae"]
            if "batch_dbp_mae" in batch_metrics:
                progress_metrics["dbp_mae"] = batch_metrics["batch_dbp_mae"]
            if "batch_map_mae" in batch_metrics:
                progress_metrics["map_mae"] = batch_metrics["batch_map_mae"]
            if "batch_waveform_mae" in batch_metrics:
                progress_metrics["waveform_mae"] = batch_metrics["batch_waveform_mae"]
            if "batch_waveform_corr" in batch_metrics:
                progress_metrics["waveform_corr"] = batch_metrics["batch_waveform_corr"]

            # Handle feature extraction specific metrics (NEW)
            if "batch_features_extracted" in batch_metrics:
                progress_metrics["features"] = batch_metrics["batch_features_extracted"]

            # If no summary metrics found, fall back to aggregation (existing behavior)
            if not progress_metrics:
                mae_values = [
                    v
                    for k, v in batch_metrics.items()
                    if isinstance(v, (int, float)) and "mae" in k.lower()
                ]
                corr_values = [
                    v
                    for k, v in batch_metrics.items()
                    if isinstance(v, (int, float)) and "corr" in k.lower()
                ]
                loss_values = [
                    v
                    for k, v in batch_metrics.items()
                    if isinstance(v, (int, float)) and "loss" in k.lower()
                ]

                if mae_values:
                    progress_metrics["mae"] = np.mean(mae_values)
                if corr_values:
                    progress_metrics["corr"] = np.mean(corr_values)
                if loss_values:
                    progress_metrics["loss"] = np.mean(loss_values)

        return progress_metrics

    def _setup_progress_bar(self, total: int, description: str) -> None:
        """Set up the progress bar if available."""
        if self.progress_bar:
            self.progress_bar.create_bar(total=total, description=description)

    def _log_final_results(
        self, results: dict[str, Any], prefix: str = "final"
    ) -> None:
        """Log final aggregated results to WandB through the progress bar."""
        if self.progress_bar is None:
            logger.warning(
                "Progress bar not available - cannot log final results to WandB"
            )
            return

        if self.progress_bar.wandb is None:
            logger.warning(
                "WandB wrapper not available - cannot log final results to WandB"
            )
            return

        final_metrics = {}
        for key, value in results.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                final_metrics[f"{prefix}/{key}"] = value

        if final_metrics:
            try:
                # Use ProgressBar's integrated WandB wrapper with proper parameters
                self.progress_bar.wandb.log(
                    final_metrics, is_rank0=self.is_main_process()
                )
                logger.info(
                    f"Logged {len(final_metrics)} final metrics to WandB with prefix '{
                        prefix
                    }'"
                )
            except Exception as e:
                logger.error(f"Failed to log final results to WandB: {e}")
        else:
            logger.warning("No valid metrics to log to WandB")

    # -------------------------------------------------------------------------
    # Shared CSV export helpers (used by waveform, blood_pressure, af_classification)
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_sample_key_from_metric(
        metric_key: str, prefix: str
    ) -> tuple[int, int] | None:
        """Parse sample indices from metric key like
        'prefix/sample_batch_idx_sample_idx'.

        Args:
            metric_key: Full metric key (e.g. 'mae/sample_0_1', 'sbp_mae/sample_2_3').
            prefix: Expected prefix before '/sample_' (e.g. 'mae', 'sbp_mae').

        Returns:
            Tuple[int, int] for (batch_idx, sample_idx), or None if parse fails.
        """
        expected = f"{prefix}/sample_"
        if not metric_key.startswith(expected):
            return None
        suffix = metric_key[len(expected) :]
        try:
            parts = suffix.split("_")
            if len(parts) >= 2:
                return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            # Fallback when version string cannot be parsed
            pass
        return None

    @staticmethod
    def _validate_csv_value(
        value: Any, default: float = 0.0, name: str = "value"
    ) -> float:
        """Validate and convert value for CSV export.

        Returns default on invalid.
        """
        if not isinstance(value, (int, float)) or (
            isinstance(value, float) and np.isnan(value)
        ):
            logger.warning(
                "Invalid %s for CSV: %s (using default %.2f)", name, value, default
            )
            return default
        return float(value)

    def _get_csv_base_metadata(
        self,
        test_loader: DataLoader,
        direction: str,
        seed: int,
        unit: str = "normalized",
    ) -> dict[str, Any]:
        """Return base metadata dict for CSV rows.

        The metadata includes dataset, method, direction, seed, and unit.
        """
        return {
            "dataset": self._get_dataset_name(test_loader),
            "method": self._get_model_name(),
            "direction": direction,
            "seed": seed,
            "unit": unit,
        }

    def _prepare_csv_data(
        self,
        results: dict[str, Any],
        test_loader: DataLoader,
        direction: str,
        seed: int,
    ) -> pd.DataFrame:
        """Prepare data for CSV export.

        Subclasses are expected to override this with task-specific schemas.
        """
        # Default implementation - subclasses should override
        data = {
            "dataset": [self._get_dataset_name(test_loader)],
            "method": [self._get_model_name()],
            "direction": [direction],
            "seed": [seed],
            "timestamp": [pd.Timestamp.now()],
        }

        for key, value in results.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                data[key] = [value]

        return pd.DataFrame(data)

    def save_results_to_csv(
        self,
        results: dict[str, Any],
        test_loader: DataLoader | None,
        direction: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Save evaluation results to CSV using the unified CSV helper.

        Args:
            results: Dictionary containing evaluation results
            test_loader: DataLoader containing test data
            direction: Direction string (e.g., 'PPG2ECG'). If None,
                uses configured direction
            seed: Seed value for filename generation. If None, uses configured seed
        """
        if test_loader is None:
            logger.warning("save_results_to_csv skipped: no dataloader provided.")
            return

        if direction is None:
            direction = self._get_direction_name()
        if seed is None:
            seed = self._get_seed()

        self._save_results_to_csv(results, test_loader, direction, seed)

    def get_model(self) -> BaseModel:
        """Return the model instance.

        Returns:
            BaseModel: The evaluator's model. Must have been set up (e.g. after
                _setup_model_for_evaluation or run_evaluation).

        Raises:
            RuntimeError: If model is not available (e.g. not yet loaded).
        """
        self._ensure_model_available("get_model accessor")
        assert self.model is not None  # guaranteed by _ensure_model_available
        return self.model

    def get_checkpoint_manager(self, name: str | None = None) -> CheckpointManager:
        """Return a checkpoint manager from the evaluator.

        Mirrors the usage described in ``EvaluatorBaseConfig`` and the project README.

        Args:
            name: Optional manager key from ``checkpoint_managers``.
                When ``None`` (default),
                defaults to 'save' for primary model checkpoints.

        Returns:
            CheckpointManager: The requested checkpoint manager instance.

        Raises:
            KeyError: If a named manager is requested but not configured.
            ValueError: If no managers are configured.

        Examples:
            ``manager = self.get_checkpoint_manager()``  # Default 'save' manager
            ``af_manager = self.get_checkpoint_manager("af_classifier")``
            # Auxiliary manager
        """
        if name is None:
            name = "save"  # Default to 'save' for primary model checkpoints

        if not self.checkpoint_managers:
            raise ValueError(
                "No checkpoint managers configured. Ensure checkpoint_managers dict is "
                "populated in config."
            )

        if name not in self.checkpoint_managers:
            raise KeyError(
                f"Checkpoint manager '{name}' not found. Available managers: "
                f"{list(self.checkpoint_managers.keys())}"
            )

        return self.checkpoint_managers[name]

    def get_progress_bar(self) -> ProgressBar | None:
        """Return the progress bar instance.

        Returns:
            ProgressBar if configured; None otherwise.
        """
        return self.progress_bar

    def get_directions(self) -> Directions | None:
        """Return the directions instance.

        Returns:
            Directions if configured; None otherwise.
        """
        return self.directions

    def _create_collate_function(
        self, dataset: Any | None
    ) -> Callable[..., Any] | None:
        """Create a direction-aware collate function similar to the trainer."""
        if dataset is None:
            logger.warning("No dataset provided, collate function will be None")
            return None

        try:
            from src.trainers.trainer import DirectionMode
            from src.utils.collate_utils import create_direction_aware_collate_fn

            # Log max_source_channels configuration
            if self.max_source_channels is None:
                logger.info(
                    "max_source_channels not specified, will be "
                    "auto-computed from directions"
                )
            else:
                logger.info(
                    f"Using explicitly configured max_source_channels: {
                        self.max_source_channels
                    }"
                )

            # Collate uses this for global norm (e.g. ABP/BP); None if no BP data.
            dataset_norm_params = dataset.get_normalization_params()

            if dataset_norm_params is not None:
                logger.info(
                    f"Using dataset normalization parameters: {dataset_norm_params}"
                )
            else:
                logger.info(
                    "No dataset normalization parameters available "
                    "(dataset may not have BP data)"
                )

            window_size = dataset.input_size if hasattr(dataset, "input_size") else None
            trim_strategy = (
                dataset.trim_strategy if hasattr(dataset, "trim_strategy") else "center"
            )

            if window_size is not None:
                logger.info(
                    f"Batch-time padding/trimming enabled: window_size={window_size} "
                    f"(from dataset), trim_strategy={trim_strategy}"
                )
            else:
                logger.info(
                    "Batch-time padding/trimming disabled: using "
                    "pre-padded sequences from dataset"
                )

            include_list = self.input_preprocessing.get("include", [])

            # Configure demographics text encoder if demographics are
            # requested and encoder is
            # available (Following trainer pattern exactly - trainer.py:1155-1161)
            demo_fields = [
                "age_raw",
                "gender_raw",
                "height_raw",
                "weight_raw",
                "bmi_raw",
            ]
            has_demographics = any(field in include_list for field in demo_fields)
            if (
                has_demographics
                and self.demographics_text_encoder is not None
                and hasattr(
                    self.demographics_text_encoder, "configure_from_include_list"
                )
            ):
                self.demographics_text_encoder.configure_from_include_list(include_list)
                logger.info(
                    "Configured demographics text encoder with fields from include_list"
                )
            encoder_to_pass = (
                self.demographics_text_encoder
                if has_demographics and self.demographics_text_encoder is not None
                else None
            )

            return create_direction_aware_collate_fn(
                input_preprocessing=self.input_preprocessing,
                directions=self.directions,
                direction_mode=DirectionMode.SINGLE.value,  # Evaluators always single
                max_source_channels=self.max_source_channels,
                dataset_norm_params=dataset_norm_params,
                window_size=window_size,
                trim_strategy=trim_strategy,
                include_list=include_list,
                demographics_text_encoder=encoder_to_pass,
            )
        except ImportError as e:
            logger.error(f"Failed to import collate utilities: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create collate function: {e}")
            return None

    def _process_batch_modern(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract only the target and let the model handle input extraction.

        This method follows the same pattern as trainers:
        - Only extract target tensor for loss/metric computation
        - Let model handle input processing via model(batch)
        - Eliminates redundant extract_input calls

        Args:
            batch: Unified batch dictionary from collate function

        Returns:
            torch.Tensor: Target tensor [B, 1, T] for loss computation
        """
        y_target = extract_target_from_batch(batch)  # [B, 1, T]
        return y_target

    def _setup_dataloader(self, dataset: Any) -> DataLoader:
        """Set up a dataloader with a collate function similar to the trainer."""
        collate_fn = self._create_collate_function(dataset)

        if collate_fn is None:
            raise ValueError(
                "Collate function is required but not available. "
                "Check dataset and direction "
                "configuration."
            )

        # Use centralized settings like trainer
        loader_settings = self.get_dataloader_settings()

        return DataLoader(
            dataset,
            batch_size=self.batch_size,  # Direct access, no fallback
            shuffle=False,
            collate_fn=collate_fn,
            **loader_settings,  # Centralized settings
        )

    def get_dataloader_settings(self) -> dict[str, Any]:
        """Return DataLoader settings with PyTorch-safe worker initialization.

        This method mirrors the trainer's approach for consistent
        configuration handling.
        """
        use_cuda = torch.cuda.is_available()
        nw = max(0, int(self.num_workers))

        settings = {
            "num_workers": nw,
            "pin_memory": bool(use_cuda),
            "persistent_workers": bool(nw > 0),
            "timeout": 0 if nw == 0 else 3600,
        }

        # prefetch_factor is only valid when num_workers > 0
        if nw > 0:
            settings["prefetch_factor"] = 2

        # Pin memory device for PyTorch >=2.0 (only when CUDA)
        if use_cuda:
            settings["pin_memory_device"] = "cuda"

        return settings

    def _get_source_vital_from_direction(self) -> Any:
        """Return the source vital from the directions configuration.

        Note: Assumes single-source directions and returns the first source vital
        from the canonical `source` property.
        """
        if self.directions is None:
            raise ValueError("directions must be configured")
        # Guard against empty directions list
        if not self.directions.directions:
            raise ValueError(
                "Evaluator assumes single-source directions and found "
                "no directions configured"
            )

        # Guard against empty source list
        if not self.directions.directions[0].source:
            raise ValueError(
                "Evaluator assumes single-source directions and found "
                "no source in the first "
                "direction"
            )

        return self.directions.directions[0].source[0]

    def _get_target_vital_from_direction(self) -> Any:
        """Return the target vital from the directions configuration.

        Uses the canonical `target` property.
        """
        if self.directions is None:
            raise ValueError("directions must be configured")
        return self.directions.directions[0].target

    def _get_direction_name(self) -> str:
        """Return direction name for checkpoint paths and logging.

        Returns the direction string in the format expected by checkpoint manager
        and other components that need direction information.

        Returns:
            str: Direction string (e.g., 'PPG2ECG', 'ECG2PPG')
        """
        if self.directions is None:
            raise ValueError("directions must be configured")
        return self.directions.directions[0].key()

    def _get_trainer_name(self) -> str:
        """Return trainer name from configuration.

        Raises:
            ValueError: If trainer_name is missing when load_model_weights=True.
            RuntimeError: If accessed in GT-only mode where
                trainer_name is not provided.
        """
        if self.trainer_name is not None:
            return self.trainer_name
        if self.load_model_weights:
            raise ValueError(
                "trainer_name must be provided when load_model_weights=True."
            )
        raise RuntimeError(
            "trainer_name is undefined in GT-only mode (load_model_weights=False)."
        )

    def _get_checkpoint_epoch(self) -> int | None:
        """Return checkpoint epoch from configuration.

        Returns:
            Optional[int]: Checkpoint epoch number, or None to load the best checkpoint
                (suffix-free).

        Raises:
            RuntimeError: If accessed in GT-only mode where
                checkpoint_epoch is not required.
        """
        if self.load_model_weights:
            return self.checkpoint_epoch
        raise RuntimeError(
            "checkpoint_epoch is undefined in GT-only mode (load_model_weights=False)."
        )

    def _get_batch_size(self) -> int:
        """Return batch size from configuration."""
        return self.batch_size

    def _get_num_epochs(self) -> int:
        """Return number of epochs from configuration."""
        return self.num_epochs

    def _get_learning_rate(self) -> float:
        """Return learning rate from configuration."""
        return self.learning_rate

    def _get_scheduler_patience(self) -> int | None:
        """Return scheduler patience from configuration."""
        return self.scheduler_patience

    def _get_early_stopping_patience(self) -> int:
        """Return early stopping patience from configuration."""
        return self.early_stopping_patience

    def _get_is_pretraining(self) -> bool:
        """Return pretraining flag from configuration."""
        return self.is_pretraining

    def _get_is_finetuning(self) -> bool:
        """Return finetuning flag from configuration."""
        return self.is_finetuning

    def _get_use_patient_split(self) -> bool:
        """Return patient split flag from configuration."""
        return self.use_patient_split

    def _get_use_patient_information(self) -> bool:
        """Return patient information flag from configuration."""
        return self.use_patient_information

    def _get_seed(self) -> int:
        """Return seed from configuration."""
        return self.seed

    def _get_use_wcl(self) -> bool:
        """Return WCL flag from configuration."""
        return self.use_wcl

    def cleanup(self) -> None:
        """Clean up logging resources."""
        if self.progress_bar:
            self.progress_bar.close()

    # ============================================================================
    # UNIFIED EVALUATION INFRASTRUCTURE METHODS
    # ============================================================================

    def _setup_evaluation_environment(self) -> None:
        """Set up the evaluation environment similar to trainer setup."""
        # Directory creation
        self._create_directories()

        # Hardware setup
        self._setup_hardware_environment()

        # Logging setup
        self._setup_logging()

        # Device setup
        self._setup_device()

    def _create_directories(self) -> None:
        """Create required directories for evaluation."""
        from pathlib import Path

        directories = ["logs", "results"]
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)

        if self.hydra_run_dir:
            Path(self.hydra_run_dir).mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(__name__)
        logger.info("Evaluation directories created successfully")

    def _setup_hardware_environment(self) -> None:
        """Set up the hardware environment for evaluation."""
        import os

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

        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"[EVAL HW] cpu={os.cpu_count()} threads={self.num_threads} "
            f"workers={self.num_workers} cuda={torch.cuda.is_available()} "
            f"gpus={torch.cuda.device_count()}"
        )

    def _setup_logging(self) -> None:
        """Set up logging for evaluation."""
        import sys

        # Resolve logging level name to numeric value without relying on getattr.
        level_name = self.logging_level.upper()
        resolved_level = logging.getLevelName(level_name)
        if not isinstance(resolved_level, int):
            resolved_level = logging.INFO

        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file_path),
            ],
        )

        logger.info(
            f"Evaluation logging configured: level={self.logging_level}, file={
                self.log_file_path
            }"
        )

    def _setup_device(self) -> None:
        """Set up the device for evaluation."""
        import os

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            logger.info(
                f"Using device: {self.device} (torchrun LOCAL_RANK={local_rank})"
            )
        else:
            self.device = torch.device("cpu")
            logger.info(f"Using device: {self.device} (CPU-only)")

    def to_device(self, obj: Any) -> Any:
        """Move an object to the evaluation device, handling nested structures.

        Args:
            obj: Object to move to device (tensor, numpy array,
                dict, list, tuple, or other)

        Returns:
            Any: Object moved to device (same type as input)
        """
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(x) for x in obj)
        return obj

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move all tensor-like objects in the batch to the active evaluation device.

        Delegates to :meth:`to_device` to keep a single entry point
        for device transfers, mirroring the trainer interface while
        allowing future hooks for additional preprocessing.
        """
        return self.to_device(batch)

    def _predict_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run model inference with post-processing for a single batch.

        Mirrors :meth:`BaseTrainer.predict_step` to keep trainer and
        evaluator inference aligned. The method moves the batch to the
        configured device, executes the model forward pass under
        ``torch.no_grad()``, collects processing metadata, and routes
        outputs through the
        configured processor.

        Args:
            batch: Batch dictionary produced by the evaluator's dataloader.

        Returns:
            Dict[str, Any]: Post-processed outputs ready for metrics computation.

        Raises:
            RuntimeError: If no processor has been configured for this evaluator.
        """
        if self.processor is None:
            raise RuntimeError(
                "_predict_batch requires a processor. Configure "
                "evaluator.processor in Hydra "
                "config."
            )

        model = self._get_model_for_inference()
        model.eval()
        prepared_batch = self._move_batch_to_device(batch)

        with torch.no_grad():
            raw_outputs = model(prepared_batch)

        direction_name = self._get_direction_name()
        if direction_name and not prepared_batch.get("direction"):
            prepared_batch["direction"] = direction_name

        return (
            self.processor.process(raw_outputs, prepared_batch, stage="test")
            if self.processor is not None
            else raw_outputs
        )

    # Deferred: Unify inference logic with BaseTrainer.predict_step via shared helper
    # module (low priority; evaluator and trainer contexts differ).

    def _get_evaluation_description(self) -> str:
        """Return evaluation description for the progress bar."""
        return f"Evaluating {self.__class__.__name__.replace('Evaluator', '')}"

    def _get_metrics_name(self) -> str:
        """Return metrics name for aggregation."""
        return self.__class__.__name__.lower().replace("evaluator", "")

    def _finalize_evaluation_results(
        self, results: dict[str, Any], test_loader: DataLoader | None
    ) -> None:
        """Finalize evaluation results and optionally save them to CSV.

        Note: Individual evaluators should handle their own specific WandB logging
        in their print_results() method implementation.
        """
        # Note: Individual evaluators handle their own specific
        # logging in print_results()

        if test_loader is None:
            if self.save_results:
                logger.warning(
                    "No dataloader available; skipping CSV export "
                    "for GT-only evaluation."
                )
            return

        if self.save_results:
            self._save_results_to_csv(
                results, test_loader, self._get_direction_name(), self._get_seed()
            )

    def _save_results_to_csv(
        self,
        results: dict[str, Any],
        test_loader: DataLoader,
        direction: str,
        seed: int,
    ) -> None:
        """Save results to CSV using the unified CSV system."""
        # Single check: progress bar has CSV functionality
        if not (self.progress_bar and self.progress_bar.csv is not None):
            return

        # Let evaluator prepare the DataFrame
        df = self._prepare_csv_data(results, test_loader, direction, seed)

        dataset = self._get_dataset_name(test_loader)
        method = self._get_model_name()
        result_type = self.__class__.__name__.lower().replace("evaluator", "")

        # Generate filepath using unified CSV configuration
        filepath = self.progress_bar.csv.generate_filepath(
            dataset=dataset,
            method=method,
            direction=direction,
            seed=seed,
            result_type=result_type,
        )
        self.progress_bar.csv.save_dataframe(
            df, filepath, file_key="evaluation", is_rank0=self.is_main_process()
        )

    def _cleanup_evaluation_resources(self) -> None:
        """Clean up evaluation resources."""
        import logging

        logger = logging.getLogger(__name__)

        # Finish CSV run
        if (
            hasattr(self, "progress_bar")
            and self.progress_bar
            and self.progress_bar.csv
        ):
            self.progress_bar.csv.finish_run("evaluation")

        # Close progress bar
        if hasattr(self, "progress_bar") and self.progress_bar:
            self.progress_bar.close_progress_bar()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Evaluation resources cleaned up")

    def _handle_oom_error(self) -> None:
        """Handle CUDA out of memory error."""
        import logging

        logger = logging.getLogger(__name__)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.warning("CUDA cache cleared due to OOM error")
