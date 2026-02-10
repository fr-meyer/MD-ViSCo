"""Enhanced waveform reconstruction evaluator with integrated logging.

Evaluates waveform reconstruction quality using MAE and Pearson correlation
metrics with full logging system integration.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from typing import Any
from typing import cast

# Third-party imports
import numpy as np
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader

# Local imports
from src.evaluators.base_evaluator import BaseEvaluator
from src.evaluators.base_evaluator import EvaluatorBaseConfig
from src.loggings.metrics import metrics
from src.utils.constants import BATCH_KEY_DIRECTION
from src.utils.constants import BATCH_KEY_INPUT
from src.utils.constants import PROCESSOR_KEY_METRICS
from src.utils.constants import PROCESSOR_KEY_WAVEFORM
from src.utils.validation_utils import validate_tensor_shapes_match

logger = logging.getLogger(__name__)


@dataclass
class WaveformReconstructionConfig(EvaluatorBaseConfig):
    """Configuration for waveform reconstruction evaluator.

    This configuration adds only waveform-specific parameters on top of
    ``EvaluatorBaseConfig`` while keeping the base interface intact.
    """

    _target_: str = (
        "src.evaluators.waveform_reconstruction.WaveformReconstructionEvaluator"
    )

    # Waveform-specific parameters only
    correlation_threshold: float = 0.0
    mae_threshold: float = 0.0
    fail_on_missing_metrics: bool = False

    # Waveform-specific overrides for defaults
    log_file_path: str = "logs/waveform_test.log"

    # Collate function configuration
    input_preprocessing: dict[str, Any] = MISSING


class WaveformReconstructionEvaluator(BaseEvaluator):
    """Enhanced evaluator for waveform reconstruction quality.

    **Model Requirement**: This evaluator requires a trained model for waveform
        generation
    and cannot run in GT-only mode. It relies on ``_predict_batch()`` for inference,
        which
    needs both the model and the configured processor to produce reconstructed
        waveforms.

    Uses _predict_batch() to get processed waveforms from WaveformOutputProcessor.
    Expects outputs with 'waveform' key containing trimmed waveform [B,C,T].
    Evaluator performs inference locally through its configured processor, so no trainer
        dependency.
    Processor handles padding removal automatically. For GT-only evaluation workflows,
        use
    ``FeatureExtractionEvaluator`` which is being extended for that scenario.
    """

    def __init__(
        self,
        # ONLY waveform-specific parameters
        correlation_threshold: float = 0.0,
        mae_threshold: float = 0.0,
        fail_on_missing_metrics: bool = False,
        # All other parameters passed through
        *args,
        **kwargs,
    ):
        """Initialize waveform reconstruction evaluator.

        Args:
            correlation_threshold: Optional lower bound on acceptable Pearson
                correlation values used for domain-specific reporting or gating.
            mae_threshold: Optional upper bound on acceptable MAE values used
                for clinical-style checks.
            fail_on_missing_metrics: Whether to raise when processor waveform
                metrics are missing; when False, such batches are skipped.
            *args: Additional positional arguments forwarded to ``BaseEvaluator``.
            **kwargs: Additional keyword arguments forwarded to ``BaseEvaluator``.

        Notes:
            - This evaluator always requires a trained model and a waveform
              processor; GT-only mode is not supported here.
            - All infrastructure (device, checkpoints, dataloaders, logging)
              is managed by ``BaseEvaluator``.
        """
        self.correlation_threshold = correlation_threshold
        self.mae_threshold = mae_threshold
        self.fail_on_missing_metrics = fail_on_missing_metrics

        super().__init__(*args, **kwargs)

    def _execute_evaluation_logic(
        self, model, test_loader: DataLoader, aggregator
    ) -> dict[str, Any]:
        """Execute waveform reconstruction evaluation logic - modern approach only.

        Args:
            model: Trained model for waveform reconstruction. Required and must not be
                None.
            test_loader: Evaluation DataLoader.
            aggregator: Metrics aggregator context.

        Calls _predict_batch() for inference with padding trimming.
        Processor handles padding removal automatically. If ``model`` is None,
            ``_predict_batch()``
        will raise a RuntimeError via the BaseEvaluator guard.

        Raises:
            RuntimeError: If model is None (via BaseEvaluator guard).
            KeyError: If processor outputs lack required 'waveform' key.
            TypeError: If waveform output is not a tensor.
            ValueError: If waveform tensor shape is not [B, C, T].
        """
        self._ensure_model_available("waveform reconstruction evaluation")
        skipped_batches = 0
        total_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                total_batches += 1
                # Run inference using _predict_batch() for processed
                # outputs (handles device transfer)
                outputs = self._predict_batch(batch)
                if PROCESSOR_KEY_WAVEFORM not in outputs:
                    raise KeyError(
                        f"Processor outputs missing "
                        f"'{PROCESSOR_KEY_WAVEFORM}' key. "
                        f"Got: {outputs.keys()}"
                    )

                # Extract waveform from processor outputs
                output = outputs[PROCESSOR_KEY_WAVEFORM]
                if not torch.is_tensor(output):
                    raise TypeError(f"Expected tensor for waveform, got {type(output)}")
                if output.dim() != 3:
                    raise ValueError(
                        f"Expected 3D waveform [B,C,T], got shape {output.shape}"
                    )

                waveform_metrics = None
                if isinstance(outputs.get(PROCESSOR_KEY_METRICS), dict):
                    waveform_metrics = outputs[PROCESSOR_KEY_METRICS].get(
                        PROCESSOR_KEY_WAVEFORM
                    )

                if waveform_metrics is None:
                    direction_hint = batch.get(BATCH_KEY_DIRECTION)
                    if (
                        direction_hint is None
                        and hasattr(self, "directions")
                        and self.directions is not None
                    ):
                        try:
                            direction_hint = self._get_direction_name()
                        except Exception:  # pragma: no cover - defensive fallback
                            direction_hint = None
                    if self.fail_on_missing_metrics:
                        raise ValueError(
                            "Processor waveform metrics required. Enable "
                            "compute_metrics=true in waveform processor config."
                        )
                    logger.warning(
                        "Skipping waveform evaluation batch %s due to missing "
                        "precomputed metrics (direction=%s). "
                        "Enable compute_metrics=true in waveform processor config.",
                        batch_idx,
                        direction_hint,
                    )
                    skipped_batches += 1
                    del outputs, output
                    continue

                # Extract target AFTER confirming metrics exist
                y_target = self._process_batch_modern(batch)

                batch_metrics = self._format_reconstruction_metrics_from_payload(
                    output,
                    y_target,
                    batch_idx,
                    precomputed_metrics=waveform_metrics,
                )

                # Note: Removed _log_metrics call to prevent duplicate
                # progress bar updates
                if self.progress_bar and self.is_main_process():
                    current_metrics = self._format_batch_metrics_for_progress(
                        batch_metrics
                    )
                    self.update_progress_bar(
                        metrics_dict=current_metrics,
                        step=batch_idx,
                        is_rank0=True,  # Evaluators run single-process
                        to_log=False,  # Don't log every batch during evaluation
                    )

                # Use batch size from batch for correct aggregation
                batch_size = batch[BATCH_KEY_INPUT].size(0)
                for key, value in batch_metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics.log_scalar(key, value, sample_size=batch_size)

                # Clean up
                del y_target, outputs, output, batch_metrics

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if skipped_batches > 0:
            logger.warning(
                "Evaluation incomplete: skipped %d/%d batches due to "
                "missing metrics. Ensure processor.compute_metrics=true "
                "in config.",
                skipped_batches,
                total_batches,
            )

        final_metrics = aggregator.get_smoothed_values()
        return final_metrics

    def _format_reconstruction_metrics_from_payload(
        self,
        output,
        y_target,
        batch_idx,
        precomputed_metrics: dict[str, torch.Tensor] | None = None,
    ):
        """Format processor-provided reconstruction metrics for aggregation."""
        validate_tensor_shapes_match(
            output,
            y_target,
            tensor1_name="output",
            tensor2_name="y_target",
            error_context=(
                "This indicates an upstream processor issue. The evaluator "
                "relies on the processor to provide shape-compatible "
                "tensors."
            ),
        )

        formatted = {}

        mae_per_sample: torch.Tensor
        correlations: torch.Tensor

        if not isinstance(precomputed_metrics, dict):
            raise ValueError(
                "Processor metrics required. Enable compute_metrics=true "
                "in processor config."
            )

        mae_tensor = precomputed_metrics.get("mae")
        corr_tensor = precomputed_metrics.get("correlation")

        if mae_tensor is None or corr_tensor is None:
            raise ValueError(
                "Processor metrics incomplete. Expected 'mae' and "
                "'correlation'. Enable compute_metrics=true in processor "
                "config."
            )

        if not torch.is_tensor(mae_tensor) or not torch.is_tensor(corr_tensor):
            raise TypeError(
                "Processor metrics must provide torch.Tensor values for "
                "'mae' and 'correlation'."
            )

        mae_per_sample = mae_tensor.detach().flatten()
        correlations = corr_tensor.detach().flatten()

        if mae_per_sample.dim() > 1:
            mae_per_sample = mae_per_sample.flatten()
        if correlations.dim() > 1:
            correlations = correlations.flatten()

        for i, (mae, corr) in enumerate(zip(mae_per_sample, correlations, strict=True)):
            sample_id = f"sample_{batch_idx}_{i}"
            formatted[f"mae/{sample_id}"] = mae.item()
            formatted[f"corr/{sample_id}"] = corr.item()

        formatted["batch_mae"] = mae_per_sample.mean().item()
        formatted["batch_corr"] = correlations.mean().item()

        return formatted

    def print_results(
        self, results: dict[str, Any], test_loader: DataLoader | None = None
    ) -> None:
        """Print waveform reconstruction results."""
        mean_mae = std_mae = mean_corr = std_corr = 0.0

        logger.info("\nWaveform Reconstruction Results:")
        logger.info("=" * 50)

        # Group metrics by type
        mae_metrics = {k: v for k, v in results.items() if k.startswith("mae/")}
        corr_metrics = {k: v for k, v in results.items() if k.startswith("corr/")}

        if mae_metrics:
            logger.info("\nMAE Results:")
            logger.info("-" * 30)
            mean_mae = np.mean(list(mae_metrics.values()))
            std_mae = np.std(list(mae_metrics.values()))
            logger.info("Mean MAE: %.4f ± %.4f", mean_mae, std_mae)

        if corr_metrics:
            logger.info("\nCorrelation Results:")
            logger.info("-" * 30)
            mean_corr = np.mean(list(corr_metrics.values()))
            std_corr = np.std(list(corr_metrics.values()))
            logger.info("Mean Correlation: %.4f ± %.4f", mean_corr, std_corr)

        # Log final results to WandB
        final_results = {
            "mae_mean": mean_mae,
            "mae_std": std_mae,
            "corr_mean": mean_corr,
            "corr_std": std_corr,
        }
        self._log_final_results(final_results, prefix="waveform")

    def _prepare_csv_data(
        self, results: dict[str, Any], test_loader, direction: str, seed: int
    ) -> pd.DataFrame:
        """Prepare waveform reconstruction data for CSV export - per-sample metrics.

        Creates a DataFrame with one row per sample containing individual MAE and
        correlation values for detailed analysis.

        Args:
            results: Dictionary containing per-sample metrics with keys like
                    'mae/sample_{batch_idx}_{sample_idx}' and
                        'corr/sample_{batch_idx}_{sample_idx}'
            test_loader: DataLoader containing test data
            direction: Direction string (e.g., 'PPG2ECG')
            seed: Seed value for filename generation

        Returns:
            pd.DataFrame: DataFrame with columns [Dataset, Method, Direction, Seed,
                         Unit, sample_index, mae_value, corr_value]
        """
        columns = [
            "Dataset",
            "Method",
            "Direction",
            "Seed",
            "Unit",
            "sample_index",
            "mae_value",
            "corr_value",
        ]
        data: dict[str, Any] = {col: [] for col in columns}
        meta = self._get_csv_base_metadata(
            test_loader, direction, seed, unit="normalized"
        )
        mae_metrics = {k: v for k, v in results.items() if k.startswith("mae/")}
        corr_metrics = {k: v for k, v in results.items() if k.startswith("corr/")}

        sample_data = []
        for mae_key, mae_value in mae_metrics.items():
            parsed = self._parse_sample_key_from_metric(mae_key, "mae")
            if parsed is not None:
                batch_idx, sample_idx = parsed
                sample_data.append((batch_idx, sample_idx, mae_value))

        if not sample_data:
            logger.warning("No sample data found in results")
            return pd.DataFrame(columns=cast("Any", columns))

        mae_count = len(mae_metrics)
        corr_count = len(corr_metrics)
        if mae_count != corr_count:
            logger.warning(
                "MAE samples (%d) != Correlation samples (%d)", mae_count, corr_count
            )

        sample_data.sort(key=lambda x: (x[0], x[1]))

        for global_idx, (batch_idx, sample_idx, mae_value) in enumerate(sample_data):
            corr_key = f"corr/sample_{batch_idx}_{sample_idx}"
            corr_value = corr_metrics.get(corr_key, 0.0)
            mae_val = self._validate_csv_value(
                mae_value, 0.0, f"mae sample {global_idx}"
            )
            corr_val = self._validate_csv_value(
                corr_value, 0.0, f"corr sample {global_idx}"
            )

            data["Dataset"].append(meta["dataset"])
            data["Method"].append(meta["method"])
            data["Direction"].append(meta["direction"])
            data["Seed"].append(meta["seed"])
            data["Unit"].append(meta["unit"])
            data["sample_index"].append(global_idx)
            data["mae_value"].append(mae_val)
            data["corr_value"].append(corr_val)

        return pd.DataFrame(data)


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_waveform_reconstruction",
    node=WaveformReconstructionConfig,
    group="evaluator",
)
