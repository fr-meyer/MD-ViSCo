"""AF Classification Evaluator.

This evaluator applies a single AF classifier to processor outputs and reports atrial
    fibrillation
classification metrics.
"""

import logging

# Standard library imports
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from typing import cast

# Third-party imports
import numpy as np
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Local imports
from src.evaluators.base_evaluator import BaseEvaluator
from src.evaluators.base_evaluator import EvaluatorBaseConfig
from src.processors.metrics_utils import is_direction_enabled
from src.processors.output_processor import ProcessingMetadata
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.checkpoint_manager import CheckpointManagerConfig

logger = logging.getLogger(__name__)


@dataclass
class AFClassificationConfig(EvaluatorBaseConfig):
    """Configuration for AF classification evaluator operating in classification-only
    mode."""

    _target_: str = "src.evaluators.af_classification.AFClassificationEvaluator"

    # Checkpoint managers dictionary
    checkpoint_managers: dict[str, CheckpointManagerConfig] = (
        MISSING  # Dictionary of checkpoint manager configs (instantiated by Hydra)
    )

    # Logging parameters
    log_detailed_results: bool = True
    log_classification_metrics: bool = True

    # Collate function configuration
    input_preprocessing: dict[str, Any] = MISSING

    # Results storage
    save_results: bool = True

    # Log file path
    log_file_path: str = "logs/af_classification_test.log"


class AFClassificationEvaluator(BaseEvaluator):
    """Evaluator for atrial fibrillation classification using a single AF classifier.

    The evaluator consumes classification outputs from the processor and computes
        standard
    classification metrics (accuracy, precision, recall, F1, specificity, ROC-AUC).
        Relies on
    ClassificationOutputProcessor contract guaranteeing class_predictions [B] and
        probabilities [B,
    2] outputs. Stores and exports only positive-class probabilities (probabilities[:,
        1]) as 1D
    scores. No backward compatibility with legacy logits-only outputs.
    """

    def __init__(
        self,
        checkpoint_managers: dict[str, CheckpointManager],
        log_detailed_results: bool = True,
        log_classification_metrics: bool = True,
        save_results: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize AF classification evaluator.

        Args:
            checkpoint_managers: Mapping of checkpoint manager names to instances used
                to restore the AF classification model. Passed through to
                ``BaseEvaluator`` for checkpoint discovery and loading.
            log_detailed_results: Whether to store and log per-sample predictions,
                labels, and probabilities for CSV export and downstream analysis.
            log_classification_metrics: Whether to compute and log aggregate
                classification metrics such as accuracy, precision, recall, F1,
                specificity, and AUC.
            save_results: Whether to enable CSV saving and persistence of
                evaluation outputs through the base evaluator.
            *args: Additional positional arguments forwarded to ``BaseEvaluator``.
            **kwargs: Additional keyword arguments forwarded to ``BaseEvaluator``.

        Notes:
            - The evaluator assumes that the configured ``processor`` implements the
              ClassificationOutputProcessor contract and provides ``class_predictions``
              and ``probabilities`` in its outputs.
            - The underlying `BaseEvaluator` handles hardware setup, checkpoint
              loading, dataloader creation, and logging; this class focuses purely on
              AF-specific evaluation logic.
        """
        super().__init__(*args, checkpoint_managers=checkpoint_managers, **kwargs)

        # Other parameters
        self.log_detailed_results = log_detailed_results
        self.log_classification_metrics = log_classification_metrics
        self.save_results = save_results

    def _resolve_enabled_directions(self) -> Iterable[str] | None:
        """Return the configured list of enabled directions, if any."""
        enabled = None
        if self.processor is not None:
            enabled = getattr(self.processor, "enabled_directions", None)

        if enabled is None and hasattr(self.input_preprocessing, "get"):
            enabled = self.input_preprocessing.get("enabled_directions")

        if enabled is None:
            return None

        if isinstance(enabled, str):
            return [enabled]

        if isinstance(enabled, Iterable) and not isinstance(
            enabled, (bytes, bytearray)
        ):
            return [str(item) for item in enabled]

        try:
            return [str(item) for item in enabled]
        except TypeError:
            return [str(enabled)]

    def _resolve_batch_direction(self, batch: dict[str, Any]) -> str | None:
        """Extract direction key from batch metadata."""
        direction = None
        if isinstance(batch, dict):
            raw_direction = batch.get("direction")
            if raw_direction is None:
                for key in ("direction_key", "direction_name"):
                    candidate = batch.get(key)
                    if candidate is not None:
                        raw_direction = candidate
                        break

            if raw_direction is not None:
                if hasattr(raw_direction, "key") and callable(raw_direction.key):
                    direction = raw_direction.key()
                else:
                    direction = str(raw_direction)

        if direction is None and isinstance(batch, dict):
            try:
                metadata = ProcessingMetadata.from_batch(batch)
            except Exception:  # pragma: no cover - defensive fallback
                metadata = None
            if metadata and metadata.direction:
                direction = metadata.direction

        if direction is None:
            try:
                direction = self._get_direction_name()
            except (
                Exception
            ):  # pragma: no cover - evaluator may not have directions configured
                direction = None
        return None if direction is None else str(direction)

    def _should_process_direction(self, batch: dict[str, Any]) -> bool:
        """Return True when the current batch direction is enabled."""
        enabled = self._resolve_enabled_directions()
        if not enabled:
            return True
        direction = self._resolve_batch_direction(batch)
        return is_direction_enabled(direction, enabled)

    def _setup_model_for_evaluation(self, test_loader):
        """Load the unified classification model via the base evaluator."""
        return super()._setup_model_for_evaluation(test_loader)

    def _execute_evaluation_logic(
        self, model, test_loader: DataLoader, aggregator
    ) -> dict[str, Any]:
        """Execute AF classification evaluation using the unified model outputs.

        Enforces strict ClassificationOutputProcessor contract: outputs must contain
        'class_predictions' (Tensor[B], long) and 'probabilities' (Tensor[B, 2], float).
        Fails fast with ValueError/TypeError if contract is violated.

        Raises:
            TypeError: If processor outputs are not a dict or tensor types are invalid.
            ValueError: If required keys are missing, shapes/dtypes are wrong, or batch
                size mismatches.
            KeyError: If processor outputs lack 'class_predictions' or 'probabilities'.
        """
        logger.info("Starting AF classification evaluation...")

        results: dict[str, Any] = {
            "predictions": [],
            "probabilities": [],
            "af_labels": [],
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                class_predictions = None
                probabilities = None
                try:
                    batch = self.to_device(batch)
                    self._process_batch_modern(batch)
                    af_labels = batch["class_labels"]

                    if not self._should_process_direction(batch):
                        continue

                    outputs = self._predict_batch(batch)
                    if not isinstance(outputs, dict):
                        raise TypeError(
                            f"Classification processor outputs must be a dict, got {
                                type(outputs).__name__
                            }"
                        )

                    # Enforce processor contract: fail fast if keys missing or wrong
                    # types/shapes/dtypes
                    try:
                        class_predictions = outputs["class_predictions"]
                        probabilities = outputs["probabilities"]
                    except KeyError as exc:
                        missing_key = exc.args[0]
                        raise ValueError(
                            "ClassificationOutputProcessor contract violation: "
                            f"missing required key '{missing_key}' in outputs. "
                            "Processor must provide both 'class_predictions' and "
                            "'probabilities'."
                        ) from exc

                    if not isinstance(class_predictions, torch.Tensor):
                        raise TypeError(
                            f"Expected class_predictions to be a torch.Tensor, got "
                            f"{type(class_predictions)}. Processor contract requires "
                            "tensor outputs."
                        )
                    if not isinstance(probabilities, torch.Tensor):
                        raise TypeError(
                            f"Expected probabilities to be a torch.Tensor, got "
                            f"{type(probabilities)}. Processor contract requires "
                            "tensor outputs."
                        )

                    if class_predictions.ndim != 1:
                        raise ValueError(
                            f"class_predictions must have shape [B], got "
                            f"{tuple(class_predictions.shape)}. Processor contract "
                            "violation."
                        )
                    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
                        raise ValueError(
                            "Binary AF classification requires probabilities with "
                            f"shape [B, 2], got {tuple(probabilities.shape)}. "
                            "Processor contract violation."
                        )
                    if class_predictions.shape[0] != probabilities.shape[0]:
                        raise ValueError(
                            "Batch size mismatch: class_predictions has "
                            f"{class_predictions.shape[0]} samples but probabilities "
                            f"has {probabilities.shape[0]}."
                        )

                    if class_predictions.dtype not in (
                        torch.long,
                        torch.int,
                        torch.int32,
                        torch.int64,
                    ):
                        raise ValueError(
                            f"class_predictions must have integer/long dtype, got "
                            f"{class_predictions.dtype}. Processor contract violation."
                        )
                    if not probabilities.is_floating_point():
                        raise ValueError(
                            f"probabilities must have float dtype, got "
                            f"{probabilities.dtype}. Processor contract violation."
                        )

                    self._store_batch_results(
                        results,
                        af_labels,
                        class_predictions,
                        probabilities,
                    )

                except Exception as e:
                    logger.error("Error in batch %d: %s", batch_idx, e, exc_info=True)
                    logger.error(
                        "Batch %d details - predictions: %s, probabilities: %s",
                        batch_idx,
                        (
                            class_predictions.shape
                            if torch.is_tensor(class_predictions)
                            else "N/A"
                        ),
                        (
                            probabilities.shape
                            if torch.is_tensor(probabilities)
                            else "N/A"
                        ),
                    )
                    import traceback

                    logger.error(
                        "Batch %d traceback: %s", batch_idx, traceback.format_exc()
                    )
                    raise  # Re-raise to be caught by base evaluator

                if self.progress_bar and self.is_main_process():
                    if torch.is_tensor(af_labels):
                        af_labels = af_labels.view(-1)
                    else:
                        af_labels = torch.as_tensor(af_labels, device=self.device).view(
                            -1
                        )

                    class_predictions_for_metrics = class_predictions.view(-1)
                    if class_predictions_for_metrics.device != af_labels.device:
                        class_predictions_for_metrics = (
                            class_predictions_for_metrics.to(af_labels.device)
                        )

                    current_metrics = {
                        "af_acc": class_predictions_for_metrics.eq(af_labels)
                        .float()
                        .mean()
                        .item()
                    }
                    self.update_progress_bar(
                        metrics_dict=current_metrics,
                        step=batch_idx,
                        is_rank0=True,
                        to_log=False,
                    )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        final_metrics = self._calculate_final_metrics(results)

        return final_metrics

    def _store_batch_results(
        self,
        results,
        af_labels,
        class_predictions,
        probabilities,
    ):
        """Store results for a single batch.

        Expects validated tensors from processor contract. Stores positive-class
        probabilities (probabilities[:, 1]) as 1D scalar scores. No fallback logic.
        """
        try:
            batch_size = class_predictions.size(0)
            labels_size = (
                af_labels.size(0) if torch.is_tensor(af_labels) else len(af_labels)
            )

            if labels_size != batch_size:
                raise ValueError(
                    f"Mismatch between af_labels size ({labels_size}) and batch_size "
                    f"({batch_size}) in AF classification results storage."
                )

            # Extract positive-class probability ([:, 1]) on GPU, then move to CPU once
            predictions_cpu = class_predictions.detach().cpu()
            probabilities_pos_cpu = probabilities[:, 1].detach().cpu()
            af_labels_cpu = (
                af_labels.detach().cpu()
                if torch.is_tensor(af_labels)
                else torch.as_tensor(af_labels).cpu()
            )

            for i in range(batch_size):
                results["af_labels"].append(af_labels_cpu[i].item())
                results["predictions"].append(predictions_cpu[i].item())
                results["probabilities"].append(probabilities_pos_cpu[i].item())

        except Exception as e:
            logger.error("Error storing batch results: %s", e, exc_info=True)
            logger.error(
                "Input shapes - predictions: %s, probabilities: %s",
                (
                    class_predictions.shape
                    if torch.is_tensor(class_predictions)
                    else "N/A"
                ),
                probabilities.shape if torch.is_tensor(probabilities) else "N/A",
            )
            import traceback

            logger.error("Store batch results traceback: %s", traceback.format_exc())
            raise

    def _calculate_final_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Calculate final evaluation metrics for AF classification.

        Expects 1D positive-class probability scores (no dual-column format).
        """
        if not results["predictions"]:
            logger.debug(
                "No AF classification predictions were generated; returning "
                "default metrics."
            )
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "specificity": 0.0,
                "auc": 0.5,
                "predictions": [],
                "af_labels": [],
                "probabilities": [],
            }

        af_labels = np.array(results["af_labels"])
        predictions = np.array(results["predictions"])
        probabilities = np.array(results["probabilities"])

        if af_labels.size > 0:
            accuracy = accuracy_score(af_labels, predictions)
            _zd = cast("Any", 0)
            precision = precision_score(
                af_labels, predictions, average="binary", pos_label=1, zero_division=_zd
            )
            recall_val = recall_score(
                af_labels, predictions, average="binary", pos_label=1, zero_division=_zd
            )
            f1_val = f1_score(
                af_labels, predictions, average="binary", pos_label=1, zero_division=_zd
            )
            specificity = recall_score(
                af_labels, predictions, average="binary", pos_label=0, zero_division=_zd
            )
        else:
            accuracy = precision = recall_val = f1_val = specificity = 0.0

        if (
            af_labels.size > 0
            and probabilities.size > 0
            and np.unique(af_labels).size == 2
        ):
            auc_val = roc_auc_score(af_labels, probabilities)
        else:
            auc_val = 0.5

        final_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall_val,
            "f1": f1_val,
            "specificity": specificity,
            "auc": auc_val,
            "predictions": predictions.tolist(),
            "af_labels": af_labels.tolist(),
            "probabilities": probabilities.tolist(),
        }

        return final_metrics

    def print_results(
        self, results: dict[str, Any], test_loader: DataLoader | None = None
    ) -> None:
        """Print AF classification evaluation results."""
        logger.info("\nAF Classification Evaluation Results:")
        logger.info("=" * 60)

        logger.info("\nAF Classification Metrics:")
        logger.info("-" * 30)
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1']:.4f}")
        logger.info(f"Specificity: {results['specificity']:.4f}")

        logger.info("\nROC-AUC:")
        logger.info("-" * 20)
        logger.info(f"AUC: {results['auc']:.4f}")

        self._log_final_results(results, prefix="af_classification")

    def _prepare_csv_data(
        self, results: dict[str, Any], test_loader, direction: str, seed: int
    ) -> pd.DataFrame:
        """Prepare AF classification data for CSV export with single positive-class
        probability column.

        No backward compatibility with dual-column format.
        """
        meta = self._get_csv_base_metadata(
            test_loader, direction, seed, unit="normalized"
        )
        columns = [
            "Dataset",
            "Method",
            "Direction",
            "Seed",
            "Unit",
            "sample_index",
            "af_label",
            "prediction",
            "positive_class_probability",
        ]
        data: dict[str, Any] = {col: [] for col in columns}
        af_labels = results["af_labels"]
        predictions = results["predictions"]
        probabilities = results["probabilities"]
        sample_count = len(predictions)

        for i in range(sample_count):
            data["Dataset"].append(meta["dataset"])
            data["Method"].append(meta["method"])
            data["Direction"].append(meta["direction"])
            data["Seed"].append(meta["seed"])
            data["Unit"].append(meta["unit"])
            data["sample_index"].append(i)
            data["af_label"].append(af_labels[i])
            data["prediction"].append(predictions[i])
            data["positive_class_probability"].append(
                self._validate_csv_value(
                    probabilities[i], 0.0, f"probability sample {i}"
                )
            )
        return pd.DataFrame(data)


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_af_classification", node=AFClassificationConfig, group="evaluator")
