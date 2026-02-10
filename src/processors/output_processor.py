"""Core processor infrastructure for post-processing model outputs.

Foundational classes for separating model forward pass logic from
post-processing operations. Establishes a clean interface that subsequent
phases can build upon without modification.

Classes:
    - ProcessingMetadata: Dataclass encapsulating metadata for post-processing
    - OutputProcessor: Abstract base class defining processor contract
"""

from __future__ import annotations

import logging
import numbers

# Standard library imports
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetadata:
    """Metadata container for post-processing operations.

    This dataclass encapsulates all metadata needed to perform post-processing
    on model outputs, including padding information, extraction flags, and
    normalization parameters. It follows the OmegaConf-compatible dataclass
    pattern used throughout the MD-ViSCo codebase.

    The metadata flows through the pipeline as follows:
        Dataset → Collate Function → Batch Dict → ProcessingMetadata → Processor

    Attributes:
        padding_length: Padding applied to each end of waveform sequences.
            For example, if a 1250-length sequence is padded to 1280, this value
            would be 15. Used to trim predictions back to original length.
            Default: 0 (no padding).

        original_length: Original sequence length before padding was applied.
            If provided, can be used for validation or alternative trimming
            strategies. Default: None.

        extract_scalars: Whether to extract scalar values (e.g., SBP/DBP) from
            waveform predictions. When True, processors will invoke scalar
            extraction logic. Default: False.

        vital_sign_type: Type of vital sign being processed (e.g., 'ABP', 'ECG',
            'PPG'). Legacy hint retained for backward compatibility; extractor
            selection now happens via Hydra config. Default: None.

        is_normalized: Whether the data has been normalized. If True and
            normalization_params are provided, processors can apply inverse
            transforms. Default: True.

        normalization_params: Parameters for inverse normalization transform.
            Typically contains keys like 'mean', 'std', 'min', 'max' depending
            on normalization strategy. Default: None.

    Examples:
        >>> # Create metadata manually
        >>> metadata = ProcessingMetadata(
        ...     padding_length=15,
        ...     extract_scalars=True,
        ...     vital_sign_type='ABP'
        ... )

        >>> # Extract from batch dict (typical usage in trainers)
        >>> batch = {
        ...     'x': torch.randn(32, 1, 1280),
        ...     'y_abp': torch.randn(32, 1, 1250),
        ...     'padding_length': 15,
        ...     'vital_sign_type': 'ABP'
        ... }
        >>> metadata = ProcessingMetadata.from_batch(batch)
        >>> print(metadata.padding_length)  # 15

    Note:
        This dataclass is designed to be extensible. Future phases may add
        additional fields without breaking existing code, as long as defaults
        are provided.
    """

    padding_length: int = 0
    original_length: int | None = None
    extract_scalars: bool = False
    vital_sign_type: str | None = None
    is_normalized: bool = True
    normalization_params: dict[str, float] | None = None
    direction: str | None = None
    batch: dict[str, Any] | None = None

    @classmethod
    def from_batch(cls, batch: dict[str, Any]) -> ProcessingMetadata:
        """Extract metadata from batch dictionary.

        This class method safely extracts metadata fields from the batch dict
        produced by the collate function. It handles missing keys gracefully
        by using default values.

        The batch dict structure follows the unified format defined in
        src/model/base_model.py and src/trainers/refinement_trainer.py,
        with keys like 'x', 'y_abp', 'y_bp', 'bp_raw', etc.

        Args:
            batch (Dict[str, Any]): Batch dictionary from collate function.
                May contain metadata keys like:
                - 'padding_length': int
                - 'original_length': int
                - 'extract_scalars': bool
                - 'vital_sign_type': str
                - 'is_normalized': bool
                - 'normalization_params': Dict[str, float]

        Returns:
            ProcessingMetadata: Metadata instance with extracted values.

        Examples:
            >>> batch = {
            ...     'x': torch.randn(32, 1, 1280),
            ...     'padding_length': 15,
            ...     'extract_scalars': True
            ... }
            >>> metadata = ProcessingMetadata.from_batch(batch)
            >>> metadata.padding_length
            15
            >>> metadata.extract_scalars
            True

        Note:
            Missing keys are handled gracefully using dataclass defaults.
            Debug logging is emitted for diagnostic purposes.
        """
        # Extract metadata fields with defaults
        padding_length_raw = batch.get("padding_length", 0)
        original_length_raw = batch.get("original_length")
        extract_scalars_raw = batch.get("extract_scalars", False)
        vital_sign_type = batch.get("vital_sign_type")
        is_normalized_raw = batch.get("is_normalized", True)
        normalization_params_raw = batch.get("normalization_params")

        # Validate and coerce padding_length to int, clamp to minimum 0
        try:
            padding_length = int(padding_length_raw)
            if padding_length < 0:
                logger.warning(
                    f"padding_length={padding_length_raw} is negative, clamping to 0"
                )
                padding_length = 0
        except (TypeError, ValueError):
            logger.warning(
                f"padding_length={padding_length_raw} is not a valid int, "
                "defaulting to 0"
            )
            padding_length = 0

        # Validate original_length as int if provided
        original_length = None
        if original_length_raw is not None:
            try:
                original_length = int(original_length_raw)
            except (TypeError, ValueError):
                logger.warning(
                    f"original_length={original_length_raw} is not a valid "
                    "int, setting to None"
                )
                original_length = None

        # Normalize extract_scalars to boolean
        extract_scalars = cls._coerce_bool(
            extract_scalars_raw, "extract_scalars", False
        )

        # Normalize is_normalized to boolean
        is_normalized = cls._coerce_bool(is_normalized_raw, "is_normalized", True)

        # Extract direction information when available (coerce to str | None)
        direction_raw = batch.get("direction") if batch is not None else None
        direction: str | None = None
        if direction_raw is not None:
            if hasattr(direction_raw, "key") and callable(direction_raw.key):
                direction = str(direction_raw.key())
            else:
                direction = str(direction_raw)
        else:
            for key_name in ("direction_key", "direction_name"):
                if (
                    batch is not None
                    and key_name in batch
                    and batch[key_name] is not None
                ):
                    direction = str(batch[key_name])
                    break

        # Shallow-validate normalization_params as dict with numeric values
        normalization_params = None
        if normalization_params_raw is not None:
            if not isinstance(normalization_params_raw, dict):
                logger.warning(
                    f"normalization_params={normalization_params_raw} is not "
                    "a dict, setting to None"
                )
            else:
                invalid_keys = [
                    k
                    for k, v in normalization_params_raw.items()
                    if not isinstance(v, numbers.Real)
                ]
                if invalid_keys:
                    logger.warning(
                        "normalization_params contains non-numeric values "
                        f"for keys {invalid_keys}"
                    )
                normalization_params = normalization_params_raw

        logger.debug(
            f"Extracted metadata from batch: padding_length={padding_length}, "
            f"extract_scalars={extract_scalars}, vital_sign_type={vital_sign_type}"
        )

        return cls(
            padding_length=padding_length,
            original_length=original_length,
            extract_scalars=extract_scalars,
            vital_sign_type=vital_sign_type,
            is_normalized=is_normalized,
            normalization_params=normalization_params,
            direction=direction,
            batch=batch,
        )

    @staticmethod
    def _coerce_bool(value: Any, field_name: str, default: bool) -> bool:
        """Coerce a value to boolean, handling strings and numbers.

        Args:
            value: Value to coerce
            field_name: Name of the field (for logging)
            default: Default value if coercion fails

        Returns:
            bool: Coerced boolean value
        """
        if isinstance(value, bool):
            return value

        # Handle string representations
        if isinstance(value, str):
            lower_val = value.lower().strip()
            if lower_val in ("true", "1", "yes", "on"):
                if lower_val != "true":
                    logger.warning(
                        f"{field_name}='{value}' coerced to True from string"
                    )
                return True
            elif lower_val in ("false", "0", "no", "off", ""):
                if lower_val != "false":
                    logger.warning(
                        f"{field_name}='{value}' coerced to False from string"
                    )
                return False
            else:
                logger.warning(
                    f"{field_name}='{value}' is not a valid bool string, "
                    f"defaulting to {default}"
                )
                return default

        # Handle numeric representations (0 = False, non-zero = True)
        if isinstance(value, (int, float)):
            result = bool(value)
            logger.warning(
                f"{field_name}={value} coerced to {result} from numeric value"
            )
            return result

        # Default for unhandled types
        logger.warning(
            f"{field_name}={value} (type {type(value).__name__}) is not "
            f"coercible to bool, defaulting to {default}"
        )
        return default


class OutputProcessor(ABC):
    """Abstract base class for model output processors.

    The processor layer receives canonical model outputs of the form::

        {"predictions": <tensor>, "extras": {...}}

    and transforms them into stage-aware dictionaries that downstream trainers,
    evaluators, and criteria can consume uniformly. Processors expose a single
    `process()` handles both lightweight parsing and stage-aware enrichment
    in one place.

    **Unified Processing Contract**

    Processors produce dictionaries with three canonical keys:

    - ``"predictions"``: Tensor (or tuple for deep supervision) used for
      loss computation.
    - ``"metrics"``: Nested dictionary containing any metrics computed
      during processing. Empty during training unless explicit
      configuration requests otherwise.
    - ``"extras"``: Passthrough payload that merges the model's ``extras`` with any
      processor-generated artefacts (e.g., extracted scalars, reconstructed signals).

    **Stage Awareness**

    The `stage` argument enables processors to run different code paths without
    leaking control logic into trainers:

    - ``stage="train"``: Minimal work for throughput—no trimming, no metric computation,
      optional extractor invocations only when `compute_full_metrics_during_train=True`.
    - ``stage="val"`` / ``"test"``: Full pipeline—padding trim, scalar extraction,
      denormalization, and optional metric computation controlled by
      ``compute_metrics`` and ``enabled_directions`` flags.

    **Passthrough Key Pattern**

    Models frequently emit auxiliary artefacts alongside predictions
    (reconstructed waveforms, attention maps, embeddings). Processors must copy any
    unrecognized keys from ``model_output["extras"]`` into the returned ``"extras"``
    dictionary to preserve composability across cascade models and evaluator tooling.

    **Benefits**

    1. Single responsibility: models predict, processors contextualize.
    2. Uniform data flow: trainers/evaluators always receive the same schema.
    3. Centralized metric logic: processors decide when and how to compute metrics.
    4. Extensibility: new stage-specific behaviors can be added without touching
       trainer code.

    Examples:
        >>> processor = WaveformOutputProcessor()
        >>> model_output = model(batch)
        >>> processed = processor.process(model_output, batch, stage="val")
        >>> loss = criterion(processed["predictions"], target)
        >>> log_metrics(processed["metrics"])

    Note:
        This unified contract is intentionally a breaking change. All subclasses
        implement only `process()`—there are no auxiliary parsing or post-processing
        entrypoints.

    Attributes:
        extractor: Optional extractor instance for scalar/feature extraction.
            Subclasses like WaveformOutputProcessor set this attribute.
    """

    extractor: Any | None = None  # Optional extractor for scalar/feature extraction

    @abstractmethod
    def process(
        self, model_output: dict[str, Any], batch: dict[str, Any], stage: str
    ) -> dict[str, Any]:
        """Process model outputs for a specific stage of the pipeline.

        Args:
            model_output: Canonical dictionary emitted by models. Must contain
                at least ``"predictions"`` (Tensor or Tuple of tensors) and may
                optionally include ``"extras"`` for auxiliary artefacts.
            batch: Original batch dictionary produced by the collate function.
                It must be the same mapping that was provided to the model so
                processors can inspect padding length, normalization parameters,
                and other metadata needed for trimming or metric computation.
            stage: Execution stage identifier. Expected values:
                - ``"train"`` for training/evaluation steps that only need tensors.
                - ``"val"`` for validation loops requiring trimmed outputs + metrics.
                - ``"test"`` for inference/evaluation flows identical to validation.

        Returns:
            Dict[str, Any]: Dictionary with canonical keys::

                {
                    "predictions": <tensor or tuple>,
                    "metrics": {<metric_groups>},
                    "extras": {**model_output.get("extras", {}), ...}
                }

            Processors may add additional convenience keys (e.g., ``"waveform"``,
            ``"probabilities"``) but must always include the canonical schema.

        Stage-specific expectations:
            - Training: processors SHOULD avoid expensive computation unless
              ``compute_full_metrics_during_train`` is set in their config.
            - Validation/Test: processors SHOULD trim padding, compute metrics
              when ``compute_metrics`` is True, and honor ``enabled_directions``.

        Raises:
            ValueError: If ``model_output`` is missing required keys or contains
                unsupported structures.
            TypeError: If ``batch`` is not a dictionary.
            RuntimeError: If processor-specific invariants are violated.
        """
        raise NotImplementedError
