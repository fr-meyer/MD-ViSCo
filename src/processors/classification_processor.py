"""Classification output processor for classification tasks (binary and multi-class).

This module provides :class:`ClassificationOutputProcessor`, which consumes the
canonical ``model_output`` dictionary emitted by classification models::

    {"predictions": <logits_tensor>, "extras": {...}}

All classifier architectures—standalone or cascaded—must follow this schema. Raw
logits go into ``model_output["predictions"]`` and any auxiliary payload (e.g.,
reconstructed waveforms, diagnostics, metadata) flows through
``model_output["extras"]``.

Required processor outputs:
    - ``logits`` (Tensor[B, num_classes]): Raw logits from the classifier head.
    - ``probabilities`` (Tensor[B, num_classes]): Probabilities after activation.
    - ``predictions`` (Tensor[B]): Class predictions (alias configurable via config).

Optional passthrough keys (carried via ``model_output["extras"]``):
    - ``reconstructed_waveform`` (Tensor[B, C, T]): Waveform reconstructions produced
      by cascade models.
    - ``waveform_metrics`` (Dict[str, Tensor]): Reconstruction quality metrics such as
      MAE and Pearson correlation.
    - Any additional diagnostic or auxiliary payloads specific to the model.

ClassificationOutputProcessor preserves optional keys while enforcing
classification-specific post-processing so evaluators can rely on a single schema.

Examples:
    >>> # Classifier-only pipeline
    >>> model_output = {"predictions": torch.randn(8, 2), "extras": {}}
    >>> processed = processor.process(model_output, batch, stage="test")
    >>> sorted(processed.keys())
    ['class_predictions', 'extras', 'logits', 'metrics', 'predictions', 'probabilities']

    >>> # Cascade pipeline with waveform reconstruction stored in extras
    >>> model_output = {
    ...     "predictions": torch.randn(4, 2),
    ...     "extras": {
    ...         "reconstructed_waveform": torch.randn(4, 1, 1250),
    ...         "waveform_metrics": {
    ...             "mae": torch.rand(4), "correlation": torch.rand(4)
    ...         },
    ...     },
    ... }
    >>> processed = processor.process(model_output, batch, stage="test")
    >>> "reconstructed_waveform" in processed["extras"]
    True

See Also:
    - ``WaveformOutputProcessor`` for waveform-only regression pipelines.
    - ``AFClassificationEvaluator`` for unified consumer expectations.
    - ``TwoStageCascadeModel`` for cascade architectures that emit mixed outputs.
"""

import logging

# Standard library imports
from dataclasses import dataclass
from typing import Any

# Third-party imports
import torch
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional
from hydra.core.config_store import ConfigStore
from torch import Tensor

# Local imports
from src.processors.output_processor import OutputProcessor

logger = logging.getLogger(__name__)


@dataclass
class ClassificationProcessorConfig:
    """Configuration for ClassificationOutputProcessor.

    This config controls how classification outputs are processed, including the number
    of classes, threshold for binary classification, and activation function selection.

    Attributes:
        _target_: Full path to the processor class for Hydra instantiation
        num_classes: Number of classes (2 for binary, >2 for multi-class)
        threshold: Threshold for binary classification predictions
            (only used when num_classes=2). Default 0.5. For imbalanced
            datasets, adjust to trade precision/recall.
        apply_softmax: Whether to use softmax (True) or sigmoid (False).
            For binary with single output [B,1], False uses sigmoid.
            For binary with two outputs [B,2] or multi-class, True uses softmax.
        output_key_logits: Key name for logits in output dict
        output_key_probabilities: Key name for probabilities in output dict
        output_key_predictions: Key name for predictions in output dict
        validate_probability_constraints: Enable debug validations for
            probability ranges and row sums

    Examples:
        >>> # Binary AF classification (single output + sigmoid)
        >>> config = ClassificationProcessorConfig(
        ...     num_classes=2,
        ...     threshold=0.5,
        ...     apply_softmax=False
        ... )

        >>> # Multi-class disease classification
        >>> config = ClassificationProcessorConfig(
        ...     num_classes=5,
        ...     apply_softmax=True
        ... )

        >>> # Binary with custom threshold for imbalanced data
        >>> config = ClassificationProcessorConfig(
        ...     num_classes=2,
        ...     threshold=0.3,  # Lower threshold for rare positive class
        ...     apply_softmax=False
        ... )
    """

    _target_: str = (
        "src.processors.classification_processor.ClassificationOutputProcessor"
    )
    num_classes: int = 2
    threshold: float = 0.5
    apply_softmax: bool = True
    output_key_logits: str = "logits"
    output_key_probabilities: str = "probabilities"
    output_key_predictions: str = "predictions"
    enabled_directions: list[str] | None = None
    validate_probability_constraints: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")


class ClassificationOutputProcessor(OutputProcessor):
    """ClassificationOutputProcessor handles output parsing and post-processing
    for classification tasks.

    Models must emit canonical dictionaries containing ``"predictions"``
    (logits tensor) and an optional ``"extras"`` mapping. This processor
    applies the appropriate activation (sigmoid for binary, softmax for
    multi-class), produces class predictions, and merges any extra payloads
    back into the returned ``"extras"`` dictionary.

    Usage:
        All stages call ``processor.process(model_output, batch, stage)``
        once per batch. The trainer uses ``processed["predictions"]`` for
        loss computation and the derived probabilities/predictions for metrics.

    Unified Contract Summary:
        - Required model input: ``{"predictions": logits_tensor, "extras": {...}}``
        - Derived outputs: probabilities (Tensor[B, num_classes]),
          class predictions (Tensor[B])
        - Optional passthrough: reconstructed_waveform, waveform_metrics,
          custom diagnostics

    Guaranteed Output Contract:
        - outputs['class_predictions']: Tensor[B] with dtype long,
          labels in [0, num_classes-1]
        - outputs['probabilities']: Tensor[B, num_classes] with dtype float,
          valid distribution
        - outputs['predictions']: Tensor[B, num_classes] logits kept for
          trainer loss computation
        - Binary AF (num_classes=2): probabilities always have shape [B, 2]
          even for [B, 1] logits

    Shapes and dtypes are validated/coerced during processing so downstream consumers
    can rely on this contract without fallback logic.

    Examples:
        >>> config = ClassificationProcessorConfig(num_classes=2, threshold=0.5)
        >>> processor = ClassificationOutputProcessor(config)
        >>> model_output = {"predictions": torch.randn(8, 1), "extras": {}}
        >>> processed = processor.process(model_output, batch, stage="train")
        >>> logits = processed["predictions"]
        >>> probs = processed["probabilities"]

        >>> # Cascade classifier emitting waveform artefacts via extras
        >>> cascade_output = {
        ...     "predictions": torch.randn(4, 2),
        ...     "extras": {"reconstructed_waveform": torch.randn(4, 1, 1250)},
        ... }
        >>> processed = processor.process(cascade_output, batch, stage="test")
        >>> "reconstructed_waveform" in processed["extras"]
        True

    See Also:
        - OutputProcessor: Abstract base class
        - ProcessingMetadata: Metadata dataclass (minimal usage for classification)
        - WaveformOutputProcessor: Processor for waveform regression tasks
        - ScalarOutputProcessor: Processor for scalar regression tasks
    """

    def __init__(
        self,
        num_classes: int = 2,
        threshold: float = 0.5,
        apply_softmax: bool = True,
        output_key_logits: str = "logits",
        output_key_probabilities: str = "probabilities",
        output_key_predictions: str = "predictions",
        enabled_directions: list[str] | None = None,
        validate_probability_constraints: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize ClassificationOutputProcessor.

        Args:
            num_classes: Number of classes (2 for binary, >2 for multi-class)
            threshold: Threshold for binary classification predictions
            apply_softmax: Whether to use softmax activation (True) or sigmoid (False)
            output_key_logits: Key name for logits in output dict
            output_key_probabilities: Key name for probabilities in output dict
            output_key_predictions: Key name for predictions in output dict
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.apply_softmax = apply_softmax
        self.output_key_logits = output_key_logits
        self.output_key_probabilities = output_key_probabilities
        self.output_key_predictions = output_key_predictions
        self.enabled_directions = (
            list(enabled_directions) if enabled_directions is not None else None
        )
        self.validate_probability_constraints = validate_probability_constraints

        # Warn about invalid threshold values
        if not (0.0 <= self.threshold <= 1.0):
            logger.warning(
                f"Threshold {self.threshold} is outside [0, 1] range. "
                f"This may lead to unexpected behavior."
            )

        # Determine activation type for logging
        if self.num_classes == 2:
            activation_type = "sigmoid (binary)"
        else:
            activation_type = "softmax (multi-class)"

        logger.info(
            f"Initialized ClassificationOutputProcessor: "
            f"num_classes={self.num_classes}, "
            f"threshold={self.threshold}, "
            f"activation={activation_type}"
        )
        self.compute_full_metrics_during_train: bool = False

    def process(
        self, model_output: dict[str, Any], batch: dict[str, Any], stage: str
    ) -> dict[str, Any]:
        """Unified processing for classification logits."""
        if not isinstance(model_output, dict):
            raise TypeError(
                "ClassificationOutputProcessor expects `model_output` to be a "
                "dict with 'predictions' and optional 'extras' keys."
            )
        if not isinstance(batch, dict):
            raise TypeError(
                "ClassificationOutputProcessor expects `batch` to be the "
                "original batch dict."
            )
        if model_output.get("predictions") is None:
            raise ValueError(
                "model_output must include a 'predictions' tensor for "
                "classification processing."
            )

        logits = self._extract_logits(model_output.get("predictions"))
        model_extras = dict(model_output.get("extras") or {})

        provided_probabilities = self._extract_probabilities(model_extras)
        if provided_probabilities is not None:
            probabilities = provided_probabilities
        else:
            probabilities = self._apply_activation(logits)

        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(1)
        if probabilities.ndim != 2:
            raise ValueError(
                "Expected probabilities tensor to be 2D after normalization, "
                f"got shape {tuple(probabilities.shape)}"
            )

        if self.num_classes == 2 and logits.shape[1] == 1:
            positive_column = (
                probabilities
                if probabilities.shape[1] == 1
                else probabilities[:, -1:].contiguous()
            )
            probabilities_for_output = torch.cat(
                [1 - positive_column, positive_column], dim=1
            )
            probabilities_for_threshold = positive_column
        else:
            probabilities_for_output = probabilities
            probabilities_for_threshold = probabilities

        if self.num_classes == 2 and logits.shape[1] == 1:
            predictions = self._apply_threshold(probabilities_for_threshold)
        else:
            predictions = torch.argmax(probabilities_for_threshold, dim=1)

        if predictions.ndim > 1:
            predictions = predictions.squeeze(1)

        if predictions.ndim != 1:
            raise ValueError(
                f"[stage={
                    stage
                }] class_predictions must be 1D tensor with shape [B], got shape {
                    tuple(predictions.shape)
                }"
            )

        if predictions.dtype != torch.long:
            predictions = predictions.long()

        batch_size = predictions.shape[0]

        if probabilities_for_output.ndim != 2:
            raise ValueError(
                f"[stage={
                    stage
                }] probabilities must be 2D tensor with shape [B, C], got shape {
                    tuple(probabilities_for_output.shape)
                }"
            )

        if probabilities_for_output.shape[1] != self.num_classes:
            raise ValueError(
                f"[stage={stage}] probabilities must have {
                    self.num_classes
                } classes in dimension 1, "
                f"got {probabilities_for_output.shape[1]}"
            )

        if probabilities_for_output.shape[0] != batch_size:
            raise ValueError(
                f"[stage={stage}] probabilities batch size {
                    probabilities_for_output.shape[0]
                } "
                f"does not match class_predictions batch size {batch_size}"
            )

        if not probabilities_for_output.is_floating_point():
            probabilities_for_output = probabilities_for_output.float()

        if self.num_classes == 2 and probabilities_for_output.shape[1] != 2:
            raise ValueError(
                f"[stage={stage}] Binary classification requires probabilities "
                f"with 2 columns, got {probabilities_for_output.shape[1]}"
            )
        elif self.num_classes == 2:
            logger.debug(
                f"[stage={stage}] Binary classification probabilities validated: shape={
                    tuple(probabilities_for_output.shape)
                }"
            )

        if self.validate_probability_constraints:
            probs_in_range = torch.all(
                (probabilities_for_output >= 0) & (probabilities_for_output <= 1)
            )
            if not probs_in_range:
                logger.warning(
                    f"[stage={stage}] Probabilities contain values outside [0, 1]."
                )

            row_sums = probabilities_for_output.sum(dim=1)
            ones = torch.ones_like(row_sums)
            if not torch.allclose(row_sums, ones, atol=1e-5):
                logger.warning(
                    f"[stage={stage}] Probability rows do not sum to 1.0 "
                    "within atol=1e-5."
                )

        logger.debug(
            f"[stage={stage}] Classification outputs validated: "
            f"class_predictions shape={tuple(predictions.shape)} dtype={
                predictions.dtype
            }, "
            f"probabilities shape={tuple(probabilities_for_output.shape)} dtype={
                probabilities_for_output.dtype
            }"
        )

        extras = dict(model_extras)
        class_predictions_key = self.output_key_predictions
        if class_predictions_key == "predictions":
            class_predictions_key = "class_predictions"

        results = {
            "predictions": logits,  # logits in 'predictions' for trainer loss
            self.output_key_logits: logits,
            "probabilities": probabilities_for_output,
            "class_predictions": predictions,
            "metrics": {},
            "extras": extras,
        }

        if self.output_key_probabilities != "probabilities":
            results[self.output_key_probabilities] = probabilities_for_output

        results[class_predictions_key] = predictions
        if class_predictions_key != "class_predictions":
            results["class_predictions"] = predictions
        extras.setdefault("probabilities", probabilities_for_output)
        extras.setdefault(self.output_key_probabilities, probabilities_for_output)
        extras.setdefault("class_predictions", predictions)
        extras.setdefault(class_predictions_key, predictions)

        return results

    def _extract_logits(self, predictions: Any) -> Tensor:
        logits = None

        if isinstance(predictions, Tensor):
            logits = predictions
        elif isinstance(predictions, (tuple, list)):
            if not predictions:
                raise ValueError(
                    "Received empty prediction sequence for classification processor."
                )
            candidate = predictions[0]
            if isinstance(candidate, Tensor):
                logits = candidate
        else:
            raise TypeError(
                f"Unsupported predictions type '{
                    type(predictions).__name__
                }' for classification processor."
            )

        if logits is None:
            raise ValueError(
                "Classification processor expected tensor logits in 'predictions'."
            )

        if logits.ndim == 1:
            if self.num_classes == 2:
                logits = logits.unsqueeze(1)
            else:
                raise ValueError(
                    "1D tensor [B] is ambiguous for multi-class classification "
                    f"with num_classes={self.num_classes}."
                )
        elif logits.ndim == 2:
            batch_size, num_logits = logits.shape
            if self.num_classes == 2:
                if num_logits not in [1, 2]:
                    logger.warning(
                        f"Binary classification expects [B,1] or [B,2], got [{
                            batch_size
                        }, {num_logits}]."
                    )
            else:
                if num_logits != self.num_classes:
                    raise ValueError(
                        f"Multi-class classification expects {
                            self.num_classes
                        } logits, got {num_logits}."
                    )
                if num_logits == 1:
                    raise ValueError(
                        "Multi-class classification cannot operate with a "
                        f"single logit (received {num_logits})."
                    )
        else:
            raise ValueError(
                f"Invalid logits tensor dimensions: {
                    logits.ndim
                }. Expected 1D or 2D tensor."
            )

        return logits

    def _extract_probabilities(self, extras: dict[str, Any]) -> Tensor | None:
        for candidate_key in (
            self.output_key_probabilities,
            "probabilities",
        ):
            if candidate_key in extras and isinstance(extras[candidate_key], Tensor):
                return extras[candidate_key]
        return None

    def _apply_activation(self, logits: Tensor) -> Tensor:
        """Apply appropriate activation function based on num_classes, logits shape, and
        apply_softmax config.

        Activation selection:
            - Binary with single output [B, 1] and apply_softmax=False: sigmoid
            - Binary with single output [B, 1] and apply_softmax=True: softmax
            - Binary with two outputs [B, 2]: softmax (apply_softmax ignored)
            - Multi-class [B, C]: softmax (apply_softmax ignored)

        Args:
            logits: Raw logits tensor [B, C]

        Returns:
            Probabilities tensor [B, C] after activation

        Examples:
            >>> # Binary with single output and sigmoid: sigmoid([B, 1]) → [B, 1]
            >>> config = ClassificationProcessorConfig(apply_softmax=False)
            >>> processor = ClassificationOutputProcessor(config)
            >>> logits = torch.tensor([[0.5], [-0.3], [1.2]])
            >>> probs = processor._apply_activation(logits)
            >>> print(probs)  # [[0.62], [0.43], [0.77]]

            >>> # Binary single output + softmax: softmax([B, 1]) → [B, 1]
            >>> # (all ones)
            >>> config = ClassificationProcessorConfig(apply_softmax=True)
            >>> processor = ClassificationOutputProcessor(config)
            >>> probs = processor._apply_activation(logits)
            >>> print(probs)  # [[1.0], [1.0], [1.0]]

            >>> # Binary with two outputs: softmax([B, 2]) → [B, 2]
            >>> logits = torch.tensor([[0.5, -0.5], [-0.3, 0.3]])
            >>> probs = processor._apply_activation(logits)
            >>> print(probs)  # [[0.73, 0.27], [0.35, 0.65]]

            >>> # Multi-class: softmax([B, C]) → [B, C]
            >>> logits = torch.randn(32, 5)
            >>> probs = processor._apply_activation(logits)
            >>> print(probs.shape)  # [32, 5]
        """
        # For binary classification with a single logit, honor apply_softmax:
        # False -> sigmoid; True -> softmax (degenerate: single column → all 1.0)
        if self.num_classes == 2 and logits.shape[1] == 1:
            if self.apply_softmax:
                return F.softmax(logits, dim=1)
            return torch.sigmoid(logits)
        # Otherwise (binary with two logits or multi-class), use softmax along class dim
        return F.softmax(logits, dim=1)

    def _apply_threshold(self, probabilities: Tensor) -> Tensor:
        """Apply threshold to probabilities for binary classification.

        Only used for binary classification with single output [B, 1].
        For threshold=0.5, probabilities > 0.5 are predicted as class 1,
        otherwise class 0.

        Args:
            probabilities: Probabilities tensor [B, 1] or [B, 2]

        Returns:
            Predictions tensor [B] with class labels (0 or 1)

        Examples:
            >>> # Single output binary classification
            >>> probs = torch.tensor([[0.3], [0.7], [0.6]])
            >>> config = ClassificationProcessorConfig(threshold=0.5)
            >>> processor = ClassificationOutputProcessor(config)
            >>> preds = processor._apply_threshold(probs)
            >>> print(preds)  # [0, 1, 1]

            >>> # Custom threshold for imbalanced data
            >>> config = ClassificationProcessorConfig(threshold=0.3)
            >>> processor = ClassificationOutputProcessor(config)
            >>> preds = processor._apply_threshold(probs)
            >>> print(preds)  # [0, 1, 1]  (0.3 is still < 0.3)
            >>> # Actually for probs [[0.3]], it would be [1] since 0.3 is not > 0.3

        Note:
            Only used for binary classification with single output [B, 1].
            For [B, 2] shape, argmax is used instead (handled within process()).
        """
        if probabilities.shape[1] == 2:
            # Binary with two outputs: use argmax instead
            return torch.argmax(probabilities, dim=1)

        # Binary with single output: apply threshold
        predictions = (probabilities > self.threshold).long()

        # Squeeze to [B] shape
        if predictions.ndim > 1:
            predictions = predictions.squeeze(1)

        return predictions


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="processor",
    name="base_classification_processor",
    node=ClassificationProcessorConfig,
)

# Usage in config:
# processor:
#   _target_: src.processors.classification_processor.ClassificationOutputProcessor
#   config:
#     num_classes: 2
#     threshold: 0.5
