"""Feature extraction evaluator with integrated logging.

Extracts features from model-generated waveforms and saves them to HDF5 format
with full metadata traceability for alignment with ground truth features.

The evaluator is a key component of the three-stage feature extraction workflow:
1. Extract GT features using the evaluator in GT-only mode
2. Extract model features using this evaluator (FeatureExtractionEvaluator)
3. Compare features using src/script/features/feature_analysis.py

The evaluator automatically detects the target signal type from the direction
configuration (e.g., PPG2ECG → extract ECG features, ECG2PPG → extract PPG features)
and extracts the appropriate physiological features from model outputs.

This implementation is now powered by the processor/extractor stack introduced in
Phase 18. Each batch is post-processed through a Hydra-configured
``WaveformOutputProcessor`` that trims padding, applies normalization logic, and
delegates feature extraction to either ``ECGFeatureExtractor`` or
``PPGFeatureExtractor``. The resulting feature dictionaries are accumulated and
persisted in the legacy HDF5 schema so downstream analysis remains unchanged.

Usage Examples:

    Example 1: PPG2ECG Feature Extraction
    ```bash
    # Extract ECG features from PPG2ECG model outputs
    python -m src.test \
        evaluator=feature_extraction_evaluator \
        test_dataset=test_uci \
        evaluator.model=mdvisco_approximation_uci \
        evaluator.directions=ppg2ecg \
        evaluator.checkpoint_epoch=100 \
        evaluator.seed=42 \
        evaluator.output_dir=results/features
    ```

    Example 2: ECG2PPG Feature Extraction
    ```bash
    # Extract PPG features from ECG2PPG model outputs
    python -m src.test \
        evaluator=feature_extraction_evaluator \
        test_dataset=test_pulsedb \
        evaluator.model=wavenet_approximation_pulsedb \
        evaluator.directions=ecg2ppg \
        evaluator.checkpoint_epoch=100 \
        evaluator.seed=42
    ```

    Example 3: Custom Configuration
    ```bash
    # With custom sampling rate and non-strict mode
    python -m src.test \
        evaluator=feature_extraction_evaluator \
        test_dataset=test_uci \
        evaluator.model=nabnet_approximation_uci \
        evaluator.directions=ppg2ecg \
        evaluator.sampling_rate=250 \
        evaluator.strict_mode=false \
        evaluator.normalize_signals=true
    ```

    Example 4: GT-Only Feature Extraction (No Model)
    ```bash
    # Extract features from ground truth waveforms without model inference
    python -m src.test \
        evaluator=feature_extraction_evaluator \
        test_dataset=test_uci \
        evaluator.load_model_weights=false \
        evaluator.directions=ppg2ecg \
        evaluator.seed=42 \
        evaluator.output_dir=results/features_gt \
        +evaluator.model=null \
        +evaluator.checkpoint_managers=null \
        +evaluator.checkpoint_io=null
    ```

    Note: In GT-only mode, omit or set to null: ``model``,
        ``checkpoint_managers``, ``checkpoint_io``, ``checkpoint_epoch``,
        ``trainer_name``. The evaluator will extract features directly
        from ground truth waveforms in the test dataset.

Output HDF5 Structure:
    The evaluator produces HDF5 files with the following structure:

    ```
    ├── metadata/
    │   ├── sampling_rate: int
    │   ├── normalization_method: str
    │   ├── feature_extraction_version: str
    │   ├── feature_extraction_config: JSON string
    │   ├── sample_ids: JSON array
    │   ├── subject_ids: JSON array (optional)
    │   ├── sample_indices: JSON array (optional, for alignment tracking)
    │   ├── num_samples: int
    │   └── model_info/
    │       ├── model_name: str
    │       ├── direction: str
    │       └── seed: int
    ├── sample_0/
    │   ├── sample_id: str
    │   └── ecg_features/ (or ppg_features/)
    │       ├── peak_locations: nested dict (ECG only)
    │       ├── qt_intervals: array (ECG only)
    │       ├── mean_ecg_quality: float (ECG only)
    │       ├── Asp_deltaT: array (PPG only)
    │       └── IPR: array (PPG only)
    ├── sample_1/
    │   └── ...
    └── sample_N/
        └── ...
    ```

Integration Workflow:
    The complete feature extraction workflow involves:

    1. Extract GT features:
       python -m src.test evaluator=feature_extraction_evaluator test_dataset=test_uci
           evaluator.model=null

    2. Extract model features:
       python -m src.test evaluator=feature_extraction_evaluator ...

    3. Compare features using src/script/features/feature_analysis.py:

       python -m src.script.features.feature_analysis \
         feature_analysis.gt_features_file=results/features/PulseDB/ground_truth/seed_42
             /features_PPG2ECG.h5 \
         feature_analysis.model_name=mdvisco \
         feature_analysis.seed=42 \
         feature_analysis.direction=PPG2ECG \
         feature_analysis.dataset_name=PulseDB \
         feature_analysis.features_base_path=results/features

Important Notes:
    - Target signal type is automatically determined from direction
      (PPG2ECG/ABP2ECG → ECG, ECG2PPG/ABP2PPG → PPG)
    - Sample alignment is ensured by storing sample_indices and subject_ids
    - Output file naming: {dataset}_{model}_{direction}_{seed}_features.h5
    - Compatible with load_features_only() and verify_sample_alignment()
      from feature_io.py
    - Metadata includes model_info for traceability and reproducibility
"""

import logging
import os

# Standard library imports
from collections.abc import Sized
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing import cast

# Third-party imports
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Local imports
from src.evaluators.base_evaluator import BaseEvaluator
from src.evaluators.base_evaluator import EvaluatorBaseConfig
from src.evaluators.feature_io import ExtractionConfig
from src.evaluators.feature_io import save_features_only
from src.loggings.metrics import metrics

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionConfig(EvaluatorBaseConfig):
    """Configuration for feature extraction evaluator.

    This configuration extends EvaluatorBaseConfig with feature-specific parameters.
    All base evaluator parameters (checkpoint loading, logging, hardware, etc.) are
    inherited and can be overridden via Hydra.

    Feature-Specific Parameters:
        sampling_rate (int): Sampling rate for feature extraction in Hz. Default: 125
        output_dir (str): Directory where extracted features will be saved.
                         Default: "results/features"
        strict_mode (bool): Whether to raise exceptions on feature extraction failures.
                           If False, failed samples are skipped. Default: True
        normalize_signals (bool): Whether to normalize signals before feature
            extraction.
                                 Default: True

    Inherited Parameters (from EvaluatorBaseConfig):
        Model configuration:
            - model: Model configuration dict

        Checkpoint:
            - checkpoint_managers: Checkpoint manager configurations
            - load_model_weights: Whether to load model weights
            - checkpoint_epoch: Epoch to load checkpoint from
            - trainer_name: Name of trainer for checkpoint path
            - dataset_name: Name of dataset for checkpoint path

        Logging:
            - log_file_path: Path to log file
            - log_metrics: Whether to log metrics
            - save_results: Whether to save results
            - logging_level: Logging level (INFO, DEBUG, etc.)

        Hardware:
            - device: Device to use (cuda, cpu)
            - num_threads: Number of CPU threads
            - num_workers: Number of data loader workers
            - pin_memory: Whether to pin memory in data loader
            - timeout: Data loader timeout

        Training:
            - batch_size: Batch size for evaluation
            - seed: Random seed for reproducibility
            - use_patient_split: Whether to use patient-level splitting
            - use_patient_information: Whether to use patient information

        Progress:
            - progress_bar: Whether to show progress bar

        Directions:
            - directions: Direction configuration (ppg2ecg, ecg2ppg, etc.)
            - direction_mode: Direction mode configuration

    Example Configuration Override:
        In your Hydra config file or command line:

        ```yaml
        evaluator:
          _target_: src.evaluators.feature_extraction.FeatureExtractionEvaluator
          sampling_rate: 250
          output_dir: custom/output/path
          strict_mode: false
          checkpoint_epoch: 150
          seed: 123
        ```

        Or via command line:
        ```bash
        python -m src.test evaluator=feature_extraction_evaluator \
            evaluator.sampling_rate=250 \
            evaluator.strict_mode=false \
            evaluator.checkpoint_epoch=150
        ```

    GT-Only Mode Configuration:
        For extracting features from ground truth waveforms without model inference:

        ```yaml
        evaluator:
          _target_: src.evaluators.feature_extraction.FeatureExtractionEvaluator
          load_model_weights: false
          # Omit model, checkpoint_managers, checkpoint_io
          directions: ppg2ecg
          sampling_rate: 125
          output_dir: results/features_gt
          batch_size: 32
          seed: 42
          # Other base parameters still required
          num_epochs: 100  # Not used but required by config
          learning_rate: 0.001  # Not used but required by config
          scheduler_patience: 10
          early_stopping_patience: 20
        ```

        Or via command line:
        ```bash
        python -m src.test evaluator=feature_extraction_evaluator \
            test_dataset=test_uci \
            evaluator.load_model_weights=false \
            evaluator.directions=ppg2ecg \
            +evaluator.model=null \
            +evaluator.checkpoint_managers=null
        ```

        In GT-only mode:
            - Features are extracted from ``batch["target"]`` instead of model outputs
            - ``origin`` metadata is set to "ground_truth" instead of "model_generated"
            - ``model_name`` in HDF5 metadata is set to "ground_truth"
            - Checkpoint loading is skipped entirely
    """

    _target_: str = "src.evaluators.feature_extraction.FeatureExtractionEvaluator"

    # Feature-specific parameters only
    sampling_rate: int = 125
    output_dir: str = "results/features"
    strict_mode: bool = True
    normalize_signals: bool = True

    # Feature-specific overrides for defaults
    log_file_path: str = "logs/feature_extraction_test.log"

    # Collate function configuration
    input_preprocessing: dict[str, Any] = MISSING


class FeatureExtractionEvaluator(BaseEvaluator):
    """Enhanced evaluator for feature extraction from model-generated waveforms.

    This evaluator loads a trained model, runs inference on the test dataset,
    extracts physiological features from model outputs, and saves them to HDF5
    format with full metadata traceability for alignment with ground truth features.

    Key Features:
        - Automatic target signal type detection from direction configuration
        - Batch-wise feature extraction with progress tracking
        - Sample alignment via sample_indices and subject_ids storage
        - Model metadata storage for traceability (model_name, direction, seed)
        - Compatible with feature_io.py (I/O utilities) and feature_analysis.py
            (comparison)
        - Supports both ECG and PPG feature extraction
        - Configurable strict mode for error handling
        - Evaluator-local inference via _predict_batch() and configured processor (no
            trainer dependency)
        - **GT-Only Mode Support**: Can extract features from ground truth
            waveforms without a model by setting
            ``load_model_weights=False`` and omitting model/checkpoint
            components. This replaces the deprecated script-based GT
            extraction workflow.

    Workflow:
        1. Load model checkpoint from specified epoch
        2. Determine target signal type from direction (PPG2ECG → ECG, ECG2PPG → PPG)
        3. Iterate through test dataset batches
        4. Run model inference using _predict_batch() to generate waveforms
        5. Extract features via processor.process() and configured extractor
        6. Accumulate features with proper sample indexing
        7. Save features to HDF5 with model metadata

    GT-Only Workflow (when model is None):
        1. Skip model checkpoint loading
        2. Determine target signal type from direction (same as model mode)
        3. Iterate through test dataset batches
        4. Extract ground truth waveforms using ``_process_batch_modern()``
        5. Invoke ``processor.process()`` on GT waveforms to extract features
        6. Accumulate features with proper sample indexing
        7. Save features to HDF5 with ``model_name="ground_truth"`` and
            ``origin="ground_truth"``

    Extracted Features:
        ECG (for PPG2ECG, ABP2ECG):
            - peak_locations: R, P, T, Q, S wave locations (sample indices)
            - qt_intervals: QT interval durations (seconds)
            - mean_ecg_quality: Average ECG signal quality score (0-1)

        PPG (for ECG2PPG, ABP2PPG):
            - Asp_deltaT: Systolic peak timing (seconds from onset)
            - IPR: Inflection point ratio (amplitude ratio)

    Usage:
        Via Hydra configuration:
        ```bash
        python -m src.test evaluator=feature_extraction_evaluator test_dataset=test_uci
            \
            evaluator.model=mdvisco_approximation_uci evaluator.directions=ppg2ecg \
            evaluator.checkpoint_epoch=100 evaluator.seed=42
        ```

        Programmatic usage:
        ```python
        evaluator = FeatureExtractionEvaluator(
            sampling_rate=125,
            output_dir="results/features",
            strict_mode=True,
            model=model_config,
            checkpoint_managers={"save": checkpoint_manager},
            directions=directions_config,
            seed=42
        )
        results, test_loader = evaluator.run_evaluation(test_dataset)
        ```

    GT-only usage:
        ```bash
        python -m src.test evaluator=feature_extraction_evaluator test_dataset=test_uci
            \
            evaluator.load_model_weights=false evaluator.directions=ppg2ecg \
            evaluator.seed=42 +evaluator.model=null
        ```

    Output:
        HDF5 file:
            {output_dir}/{DatasetName}/{MappedModelName}/seed_{seed}/features_{DIRECTION}.h5

        File structure:
            - metadata/: Configuration and model info
            - sample_N/: Features for each sample
            - Alignment info: sample_indices, subject_ids

    See Also:
        - For GT-only feature extraction, set `evaluator.model=null` in the command.
        - src/script/features/feature_analysis.py: Feature comparison and analysis
    """

    def __init__(
        self,
        # ONLY feature-specific parameters
        sampling_rate: int = 125,
        output_dir: str = "results/features",
        strict_mode: bool = True,
        normalize_signals: bool = True,
        # All other parameters passed through
        *args,
        **kwargs,
    ):
        """Initialize feature extraction evaluator.

        Args:
            sampling_rate: Sampling rate in Hz used for feature extraction and
                stored in the output metadata.
            output_dir: Base directory where HDF5 feature files will be written.
            strict_mode: Whether to raise exceptions on feature extraction
                failures; when False, failing batches are skipped with warnings.
            normalize_signals: Whether to normalize signals prior to feature
                extraction; this flag is propagated to the HDF5 metadata.
            *args: Additional positional arguments forwarded to ``BaseEvaluator``.
            **kwargs: Additional keyword arguments forwarded to ``BaseEvaluator``.

        Notes:
            - The evaluator can operate in both model-based and GT-only modes,
              controlled via the ``load_model_weights`` and ``model`` settings
              on ``EvaluatorBaseConfig``.
            - A processor with an attached feature extractor (for example
              ``waveform_processor_ecg_features`` or
              ``waveform_processor_ppg_features``) must be configured for
              processor-based extraction to succeed.
        """
        super().__init__(*args, sampling_rate=sampling_rate, **kwargs)
        self.output_dir = output_dir
        self.strict_mode = strict_mode
        self.normalize_signals = normalize_signals
        self.extracted_features: dict[str, Any] = {}
        self.sample_counter = 0
        self.sample_indices: list[int] = []
        self.subject_ids: list[str] = []

        processor = self.processor
        if processor is None:
            logger.warning(
                "FeatureExtractionEvaluator initialised without a "
                "processor. Configure `processor` with an extractor "
                "(e.g., `processor=waveform_processor_ecg_features` "
                "for ECG features) before running evaluation; feature "
                "extraction cannot proceed without it."
            )
        else:
            processor_name = processor.__class__.__name__
            extractor = processor.extractor if hasattr(processor, "extractor") else None
            extractor_name = (
                extractor.__class__.__name__ if extractor is not None else "None"
            )
            logger.info(
                "FeatureExtractionEvaluator using processor=%s, extractor=%s",
                processor_name,
                extractor_name,
            )
            if extractor is None:
                logger.warning(
                    "Processor '%s' does not expose an extractor. "
                    "Configure a feature extractor on the processor "
                    "(e.g., waveform_processor_ecg_features) to enable "
                    "feature extraction; otherwise evaluation will fail "
                    "when no features are emitted.",
                    processor_name,
                )

    def _execute_evaluation_logic(
        self, model, test_loader: DataLoader, aggregator
    ) -> dict[str, Any]:
        """Execute feature extraction evaluation logic.

        This method orchestrates the complete feature extraction workflow:
        1. Determine target signal type from direction configuration
        2. Create output directory
        3. Iterate through test batches:
           - Move batch to device
           - Collect sample indices and subject IDs for alignment
           - Run model inference
           - Extract features from model outputs
           - Accumulate features with proper indexing
           - Update progress bar and metrics
        4. Save accumulated features to HDF5 with metadata

        Args:
            model: Trained model for inference (torch.nn.Module) or None for GT-only
                mode.
                   When None, features are extracted from ground truth waveforms instead
                       of model outputs.
            test_loader: DataLoader for test dataset
            aggregator: Metrics aggregator (not used in feature extraction,
                       kept for interface compatibility with base evaluator)

        Returns:
            Dict[str, Any]: Evaluation results containing:
                - total_samples (int): Number of samples processed
                - output_file (str): Path to saved HDF5 file
                - target_signal_type (str): Type of signal extracted ('ecg' or 'ppg')
                - features_extracted (str): Summary of extracted feature types

        Raises:
            RuntimeError: If feature extraction fails and strict_mode=True
            FileNotFoundError: If checkpoint file not found during model loading
            ValueError: If direction configuration is invalid or cannot be parsed

        Notes:
            - Target signal type is automatically determined from direction:
              * PPG2ECG, ABP2ECG → 'ecg'
              * ECG2PPG, ABP2PPG → 'ppg'
            - Sample alignment is ensured via sample_indices storage which maps
              features back to original dataset indices
            - Model metadata (model_name, direction, seed) is stored in HDF5
              for traceability and reproducibility
            - Progress bar shows real-time extraction progress with batch metrics
            - CUDA cache is cleared periodically (every 10 batches) to prevent OOM
            - In non-strict mode, failed batches are logged but don't halt processing
            - **GT-Only Mode**: When ``model is None``, the evaluator
                extracts features from ground truth waveforms
                (``batch["target"]``) instead of running inference. This
                is useful for baseline feature extraction or when
                comparing GT features extracted via the evaluator vs
                the standalone script.
            - The branching logic handles both modes transparently, ensuring
              consistent feature extraction and HDF5 output format regardless of mode.
            - Processor-based extraction relies on the configured extractor. If no
              per-sample feature payloads are emitted, the evaluator raises a
              RuntimeError instructing the user to configure an appropriate processor.
        """
        # Determine target signal type from direction configuration
        target_vital = self._get_target_vital_from_direction()
        target_signal_type = target_vital.value.lower()  # API expects lowercase string
        has_model = self.model is not None
        extraction_mode = "model_generated" if has_model else "ground_truth"

        if self.processor is None and not has_model:
            raise RuntimeError(
                "GT-only feature extraction requires a configured processor with an "
                "extractor."
            )

        logger.info(
            "Starting feature extraction for target signal type: %s (mode: %s)",
            target_signal_type,
            extraction_mode,
        )

        # Reset accumulators to avoid leaking state between runs
        self.extracted_features = {}
        self.sample_counter = 0
        self.sample_indices = []
        self.subject_ids = []

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_metrics: dict[str, Any] = {}
                y_target = None
                processor_output: dict[str, Any] | None = None
                waveform_for_features = None
                batch_size = (
                    test_loader.batch_size if test_loader.batch_size is not None else 1
                )
                dataset_len = (
                    len(test_loader.dataset)
                    if isinstance(test_loader.dataset, Sized)
                    else 0
                )
                try:
                    y_target = self._process_batch_modern(batch)

                    waveform_for_features = y_target

                    if has_model:
                        outputs = self._predict_batch(batch)
                        if not isinstance(outputs, dict):
                            raise ValueError(
                                f"_predict_batch() returned {type(outputs)}, "
                                "expected dict"
                            )
                        processor_output = outputs
                        if "waveform" not in processor_output:
                            raise ValueError(
                                "_predict_batch() output missing "
                                "'waveform' key. "
                                "Available keys: "
                                f"{list(processor_output.keys())}"
                            )
                        waveform_for_features = processor_output["waveform"]
                    elif self.processor is not None:
                        batch["extract_scalars"] = True
                        direction_hint = self._get_direction_name()
                        if direction_hint and not batch.get("direction"):
                            batch["direction"] = direction_hint
                        model_output = {
                            "predictions": waveform_for_features,
                            "extras": {},
                        }
                        processor_output = self.processor.process(
                            model_output, batch, stage="test"
                        )
                        if (
                            isinstance(processor_output, dict)
                            and "waveform" in processor_output
                        ):
                            waveform_for_features = processor_output["waveform"]
                    else:
                        logger.debug(
                            "Processor unavailable; GT-only mode "
                            "falling back to legacy helper."
                        )

                    if (
                        hasattr(waveform_for_features, "dim")
                        and waveform_for_features.dim() == 3
                    ):
                        num_channels = waveform_for_features.size(1)
                        shape = tuple(waveform_for_features.shape)
                        if num_channels > 1:
                            message = (
                                "Multi-channel waveform detected before "
                                "feature extraction: batch_idx="
                                f"{batch_idx}, shape={shape}, "
                                f"num_channels={num_channels}, "
                                f"origin={extraction_mode}. Feature "
                                "extraction will use channel 0 only. "
                                "Consider pre-filtering to single channel "
                                "upstream."
                            )
                            if self.strict_mode:
                                logger.warning(message)
                            else:
                                logger.debug(message)
                        else:
                            logger.debug(
                                "Feature extraction input validated: "
                                f"batch_idx={batch_idx}, shape={shape}, "
                                f"origin={extraction_mode}"
                            )
                    else:
                        if hasattr(waveform_for_features, "shape"):
                            shape = waveform_for_features.shape
                        else:
                            shape = "unknown"
                        logger.debug(
                            "Feature extraction input validated: "
                            f"batch_idx={batch_idx}, shape={shape}, "
                            f"origin={extraction_mode}"
                        )

                    batch_features: dict[str, Any] = {}
                    used_processor = False
                    if processor_output is not None and self.processor is not None:
                        batch_features = self._extract_features_from_processor_output(
                            processor_output,
                            target_signal_type=target_signal_type,
                        )
                        used_processor = bool(batch_features)

                    if not batch_features:
                        reason = (
                            "processor output missing feature keys"
                            if self.processor is not None
                            else "processor not configured"
                        )
                        direction_hint = self._get_direction_name()
                        message = (
                            f"Batch {batch_idx}: {reason}. Configure "
                            "`processor` with a feature extractor so "
                            "that processor.process() emits per-sample "
                            "feature payloads (e.g., "
                            "`processor=waveform_processor_ecg_features` "
                            f"for direction {direction_hint})."
                        )
                        logger.error(message)
                        raise RuntimeError(message)

                    used_processor = True
                    logger.debug(
                        "Batch %s: extracted feature keys via processor pipeline: %s",
                        batch_idx,
                        sorted(self._summarize_feature_keys(batch_features)),
                    )

                    numeric_keys = sorted(
                        [key for key in batch_features if key.isdigit()],
                        key=int,
                    )
                    if not numeric_keys:
                        logger.warning(
                            "Batch %s: no per-sample feature entries "
                            "found after extraction.",
                            batch_idx,
                        )
                        continue

                    # Extract batch indices for alignment (will be used per-sample)
                    batch_indices = self._extract_batch_indices(
                        batch=batch,
                        batch_idx=batch_idx,
                        batch_size=batch_size,
                        dataset_len=dataset_len,
                    )

                    # Extract subject IDs if available
                    batch_subject_ids = None
                    if "subject_id" in batch:
                        batch_subject_ids = (
                            batch["subject_id"].cpu().tolist()
                            if torch.is_tensor(batch["subject_id"])
                            else batch["subject_id"]
                        )

                    for local_key in numeric_keys:
                        sample_payload = batch_features.get(local_key)
                        if sample_payload is None:
                            logger.debug(
                                "Batch %s: skipping empty feature payload for "
                                "local index %s.",
                                batch_idx,
                                local_key,
                            )
                            continue
                        self.extracted_features[str(self.sample_counter)] = (
                            sample_payload
                        )
                        self.sample_counter += 1

                        # Append index and subject_id for extracted sample
                        local_idx = int(local_key)
                        if local_idx < len(batch_indices):
                            self.sample_indices.append(batch_indices[local_idx])
                        if batch_subject_ids is not None and local_idx < len(
                            batch_subject_ids
                        ):
                            self.subject_ids.append(batch_subject_ids[local_idx])

                    batch_metrics = {
                        "samples_processed": len(numeric_keys),
                        "total_samples": self.sample_counter,
                        "batch_features_extracted": len(numeric_keys),
                        "processor_path": 1 if used_processor else 0,
                    }

                    if self.progress_bar and self.is_main_process():
                        current_metrics = self._format_batch_metrics_for_progress(
                            batch_metrics
                        )
                        self.update_progress_bar(
                            metrics_dict=current_metrics,
                            step=batch_idx,
                            is_rank0=True,
                            to_log=False,
                        )

                    for key, value in batch_metrics.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            metrics.log_scalar(
                                key, value, sample_size=len(numeric_keys)
                            )

                except Exception as exc:
                    if self.strict_mode:
                        logger.error(
                            "Feature extraction failed for batch %s: %s",
                            batch_idx,
                            exc,
                        )
                        raise
                    logger.warning(
                        "Feature extraction failed for batch %s "
                        "(strict_mode=False): %s",
                        batch_idx,
                        exc,
                    )
                    continue
                finally:
                    # Clean up temporary tensors to avoid GPU memory growth
                    del y_target, processor_output, waveform_for_features
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

        # After all batches, save features to HDF5
        dataset_name = self._get_dataset_name(test_loader)
        model_name = self._get_model_name() if has_model else "ground_truth"
        direction = self._get_direction_name()
        seed = self._get_seed()
        output_path = self._create_output_path(
            dataset_name, model_name, direction, seed
        )
        extraction_config = ExtractionConfig(
            sampling_rate=self.sampling_rate,
            real_time=True,
            strict_mode=self.strict_mode,
            device=str(self.device),
            normalize_signals=self.normalize_signals,
            origin=extraction_mode,
        )

        processor_snapshot = self._get_processor_config_snapshot()
        if processor_snapshot:
            extraction_config.processor_class = processor_snapshot.get(
                "processor_class"
            )
            extraction_config.processor_config = processor_snapshot.get(
                "processor_config"
            )
            extraction_config.extractor_class = processor_snapshot.get(
                "extractor_class"
            )
            extraction_config.extractor_config = processor_snapshot.get(
                "extractor_config"
            )
            logger.debug(
                "Captured processor metadata for HDF5: processor=%s, extractor=%s",
                processor_snapshot.get("processor_class"),
                processor_snapshot.get("extractor_class"),
            )

        # Prepare model info for metadata
        model_info = {"model_name": model_name, "direction": direction, "seed": seed}
        self.extracted_features["sample_indices"] = self.sample_indices
        if self.subject_ids:
            self.extracted_features["subject_ids"] = self.subject_ids

        logger.info(f"Saving extracted features to: {output_path}")
        save_features_only(
            features=self.extracted_features,
            output_path=output_path,
            config=extraction_config,
            model_info=model_info,
        )

        # Prepare summary results
        feature_summary = self._get_feature_summary()
        results = {
            "total_samples": self.sample_counter,
            "output_file": output_path,
            "target_signal_type": target_signal_type,
            "features_extracted": feature_summary,
        }

        return results

    def _extract_batch_indices(
        self,
        batch: dict,
        batch_idx: int,
        batch_size: int | None,
        dataset_len: int,
    ) -> list[int]:
        """Extract or generate batch indices for alignment.

        Args:
            batch: Batch dictionary that may contain 'index' or 'sample_index'
            batch_idx: Current batch index in iteration
            batch_size: Size of each batch (None treated as 1)
            dataset_len: Total length of dataset

        Returns:
            List of sample indices for this batch
        """
        import torch

        # Priority 1: Use batch['index'] if available
        if "index" in batch or "sample_index" in batch:
            key = "index" if "index" in batch else "sample_index"
            indices = (
                batch[key].cpu().tolist() if torch.is_tensor(batch[key]) else batch[key]
            )
            return list(indices)

        # Priority 2: Generate from batch_idx
        if batch_size is None:
            batch_size = 1
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, dataset_len)
        return list(range(start_idx, end_idx))

    def _create_output_path(
        self, dataset_name: str, model_name: str, direction: str, seed: int
    ) -> str:
        """Build output file path with nested directory structure.

        Args:
            dataset_name: Name of the dataset (e.g., 'uci', 'pulsedb')
            model_name: Name of the model (e.g., 'mdvisco', 'nabnet')
            direction: Direction string (e.g., 'PPG2ECG')
            seed: Seed value

        Returns:
            str: Full path to output file in nested structure:
                 {output_dir}/{dataset_name}/{model_name}/seed_{seed}/features_{DIRECTION}.h5
        """
        # Build nested path using names directly
        output_path = os.path.join(
            self.output_dir,
            dataset_name,
            model_name,
            f"seed_{seed}",
            f"features_{direction.upper()}.h5",
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        return output_path

    def _extract_features_from_processor_output(
        self,
        processor_output: dict[str, Any],
        target_signal_type: str,
    ) -> dict[str, Any]:
        """Convert processor output into the legacy feature extraction schema."""
        if not isinstance(processor_output, dict):
            logger.warning(
                "Processor output type %s is not a dictionary; "
                "skipping processor-based feature extraction.",
                type(processor_output).__name__,
            )
            return {}

        waveform = processor_output.get("waveform")
        batch_size = 0
        if torch.is_tensor(waveform):
            batch_size = waveform.size(0)
        elif isinstance(waveform, np.ndarray):
            batch_size = waveform.shape[0]

        signal_type = target_signal_type.lower()
        if signal_type == "ecg":
            group_key = "ecg_features"
            expected_keys = {
                "peak_locations": processor_output.get("peak_locations"),
                "qt_intervals": processor_output.get("qt_intervals"),
                "mean_ecg_quality": processor_output.get("mean_ecg_quality"),
            }
        elif signal_type == "ppg":
            group_key = "ppg_features"
            expected_keys = {
                "Asp_deltaT": processor_output.get("Asp_deltaT"),
                "IPR": processor_output.get("IPR"),
            }
        else:
            logger.warning(
                "Unsupported target signal type '%s' for "
                "processor-based feature extraction.",
                target_signal_type,
            )
            return {}

        candidate_lengths = []
        for value in expected_keys.values():
            length = self._get_feature_length(value)
            if length:
                candidate_lengths.append(length)

        if batch_size == 0 and candidate_lengths:
            unique_lengths = set(candidate_lengths)
            if len(unique_lengths) > 1:
                logger.warning(
                    "Processor output feature lengths are "
                    "inconsistent: %s. Using minimum length.",
                    unique_lengths,
                )
            batch_size = min(unique_lengths)

        if batch_size == 0:
            logger.warning(
                "Unable to determine batch size from processor output; skipping "
                "processor-based features."
            )
            return {}

        batch_features: dict[str, dict[str, Any]] = {
            str(_idx): {group_key: {}} for _idx in range(batch_size)
        }
        has_features = False

        missing_keys: list[str] = []
        for feature_name, raw_value in expected_keys.items():
            if raw_value is None:
                missing_keys.append(feature_name)
                continue

            per_sample_values = self._split_feature_values(raw_value, batch_size)
            if per_sample_values is None:
                missing_keys.append(feature_name)
                continue

            for idx, sample_value in enumerate(per_sample_values):
                if sample_value is None:
                    continue
                feature_payload = batch_features[str(idx)][group_key]
                feature_payload[feature_name] = sample_value
                has_features = True

        for _idx, payload in batch_features.items():
            feature_payload = payload[group_key]
            if not feature_payload:
                payload[group_key] = None

        if not has_features:
            return {}

        if missing_keys:
            logger.debug(
                "Processor output missing expected keys for %s extraction: %s",
                signal_type,
                missing_keys,
            )

        return batch_features

    def _split_feature_values(self, value: Any, batch_size: int) -> list[Any] | None:
        """Split processor feature value into per-sample entries."""
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray):
            length = value.shape[0]
            if length < batch_size:
                logger.warning(
                    "Processor feature length %s shorter than batch size %s.",
                    length,
                    batch_size,
                )
            per_sample = []
            for _idx in range(min(batch_size, length)):
                per_sample.append(np.asarray(value[_idx]))
            if length < batch_size:
                per_sample.extend([None] * (batch_size - length))
            return per_sample

        if isinstance(value, (list, tuple)):
            length = len(value)
            if length < batch_size:
                logger.warning(
                    "Processor feature sequence length %s shorter than batch size %s.",
                    length,
                    batch_size,
                )
            per_sample = []
            for _idx in range(min(batch_size, length)):
                per_sample.append(deepcopy(value[_idx]))
            if length < batch_size:
                per_sample.extend([None] * (batch_size - length))
            return per_sample

        if isinstance(value, (int, float, str)):
            return [value] * batch_size

        if isinstance(value, dict):
            numeric_items = []
            for key, item in value.items():
                if isinstance(key, int):
                    numeric_items.append((key, item))
                elif isinstance(key, str) and key.isdigit():
                    numeric_items.append((int(key), item))
                else:
                    logger.warning(
                        "Encountered dict-valued processor feature with "
                        "non-indexed keys %s; unable to split per sample.",
                        list(value.keys()),
                    )
                    return None

            if not numeric_items:
                logger.warning(
                    "Encountered dict-valued processor feature with "
                    "keys %s; unable to split per sample.",
                    list(value.keys()),
                )
                return None

            per_sample: list[Any] = [None] * batch_size
            for idx, item in sorted(numeric_items, key=lambda pair: pair[0]):
                if idx < 0:
                    logger.debug(
                        "Skipping negative index %s in dict-valued processor feature.",
                        idx,
                    )
                    continue
                if idx >= batch_size:
                    logger.warning(
                        "Processor feature dict index %s exceeds batch "
                        "size %s; ignoring entry.",
                        idx,
                        batch_size,
                    )
                    continue
                per_sample[idx] = deepcopy(item)
            return per_sample

        logger.warning(
            "Unsupported processor feature type '%s'; skipping.",
            type(value).__name__,
        )
        return None

    @staticmethod
    def _get_feature_length(value: Any) -> int:
        """Infer batch dimension length from processor feature output."""
        if value is None:
            return 0
        if torch.is_tensor(value):
            return value.size(0)
        if isinstance(value, np.ndarray):
            return value.shape[0]
        if isinstance(value, (list, tuple)):
            return len(value)
        return 0

    @staticmethod
    def _summarize_feature_keys(batch_features: dict[str, Any]) -> list[str]:
        """Collect unique feature keys from per-sample payloads."""
        summary: set[str] = set()
        for sample_payload in batch_features.values():
            if not isinstance(sample_payload, dict):
                continue
            for group_payload in sample_payload.values():
                if isinstance(group_payload, dict):
                    summary.update(group_payload.keys())
        return list(summary)

    def _get_processor_config_snapshot(self) -> dict[str, Any]:
        """Serialize processor and extractor configuration for metadata storage."""
        processor = self.processor
        if processor is None:
            return {}

        snapshot: dict[str, Any] = {
            "processor_class": processor.__class__.__name__,
            "processor_config": None,
            "extractor_class": None,
            "extractor_config": None,
        }

        processor_config = self._serialize_object_attributes(
            processor,
            exclude={"extractor"},
        )
        if processor_config:
            snapshot["processor_config"] = processor_config

        if hasattr(processor, "extractor") and processor.extractor is not None:
            extractor = processor.extractor
            snapshot["extractor_class"] = extractor.__class__.__name__
            extractor_config = self._serialize_object_attributes(extractor)
            if extractor_config:
                snapshot["extractor_config"] = extractor_config

        return snapshot

    def _serialize_object_attributes(
        self,
        obj: Any,
        exclude: set | None = None,
    ) -> dict[str, Any] | None:
        """Extract simple, serializable attributes from an object."""
        if exclude is None:
            exclude = set()
        serializable: dict[str, Any] = {}

        for attr, value in vars(obj).items():
            if attr.startswith("_") or attr in exclude:
                continue
            coerced = self._coerce_to_serializable(value)
            if coerced is not None:
                serializable[attr] = coerced

        if not serializable:
            return None

        cfg = OmegaConf.create(serializable)
        result = OmegaConf.to_container(cfg, resolve=True)
        # Type narrowing: ensure we return dict[str, Any] | None
        if isinstance(result, dict):
            return cast("dict[str, Any]", result)
        return None

    def _coerce_to_serializable(self, value: Any) -> Any | None:
        """Convert complex attribute values to JSON-serializable structures."""
        simple_types = (int, float, bool, str)
        if value is None or isinstance(value, simple_types):
            return value

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (list, tuple)):
            converted = [self._coerce_to_serializable(item) for item in value]
            if any(item is not None for item in converted):
                return converted
            return None

        if isinstance(value, dict):
            converted_dict = {}
            for key, item in value.items():
                coerced = self._coerce_to_serializable(item)
                if coerced is not None:
                    converted_dict[key] = coerced
            return converted_dict if converted_dict else None

        if isinstance(value, (set,)):
            converted = [self._coerce_to_serializable(item) for item in value]
            return converted if any(item is not None for item in converted) else None

        return None

    def _get_feature_summary(self) -> str:
        """Generate summary of extracted features for logging.

        Returns:
            str: Summary string of feature types
        """
        if not self.extracted_features:
            return "No features extracted"
        sample_keys = [int(k) for k in self.extracted_features if k.isdigit()]
        if not sample_keys:
            return "No features extracted"
        first_sample = self.extracted_features[str(min(sample_keys))]
        feature_types = []

        if "ecg_features" in first_sample and first_sample["ecg_features"] is not None:
            ecg_features = first_sample["ecg_features"]
            ecg_keys = list(ecg_features.keys())
            feature_types.append(f"ECG features: {', '.join(ecg_keys)}")

        if "ppg_features" in first_sample and first_sample["ppg_features"] is not None:
            ppg_features = first_sample["ppg_features"]
            ppg_keys = list(ppg_features.keys())
            feature_types.append(f"PPG features: {', '.join(ppg_keys)}")

        return "; ".join(feature_types) if feature_types else "No features available"

    def print_results(
        self, results: dict[str, Any], test_loader: DataLoader | None = None
    ) -> None:
        """Print feature extraction results."""
        logger.info("\nFeature Extraction Results:")
        logger.info("=" * 50)

        logger.info(f"Total samples processed: {results.get('total_samples', 0)}")
        logger.info(
            f"Target signal type: {results.get('target_signal_type', 'unknown')}"
        )
        logger.info(f"Output file: {results.get('output_file', 'unknown')}")

        logger.info("\nExtracted Features:")
        logger.info("-" * 30)
        logger.info(f"{results.get('features_extracted', 'No features')}")

        logger.info("=" * 50)


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_feature_extraction", node=FeatureExtractionConfig, group="evaluator"
)
