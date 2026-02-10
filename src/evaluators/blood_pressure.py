"""Simplified blood pressure evaluator without legacy compatibility.

Evaluates blood pressure estimation accuracy using SBP/DBP/MAP metrics and
BHS standards with full logging system integration.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from dataclasses import field
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
from src.loggings.meters import AverageMeter

logger = logging.getLogger(__name__)


@dataclass
class BloodPressureConfig(EvaluatorBaseConfig):
    """Configuration for blood pressure evaluator - only BP-specific parameters."""

    _target_: str = "src.evaluators.blood_pressure.BloodPressureEvaluator"

    # BP-specific parameters only
    global_min: float = 0.0
    global_max: float = 200.0
    bhs_thresholds: list = field(default_factory=lambda: [5, 10, 15])

    # Denormalization configuration
    denormalize: bool = (
        True  # Whether to denormalize predictions (waveform and BP values)
    )
    fail_on_missing_metrics: bool = False

    # BP-specific overrides for defaults
    log_file_path: str = "logs/bp_test.log"

    # Collate function configuration
    input_preprocessing: dict[str, Any] = MISSING


class BloodPressureEvaluator(BaseEvaluator):
    """Evaluator for blood pressure prediction tasks.

    Uses the configured ScalarOutputProcessor via _predict_batch() for inference.
    Padding, trimming, and BP extraction are handled by the processor.

    Notes:
        Expects processor output keys 'sbp', 'dbp', and 'waveform'. Legacy keys
        (y_pred_sbp, y_pred_dbp) are not used.
    """

    def __init__(
        self,
        # ONLY BP-specific parameters
        global_min: float,
        global_max: float,
        bhs_thresholds: list,
        denormalize: bool,
        fail_on_missing_metrics: bool = False,
        # All other parameters passed through
        *args,
        **kwargs,
    ):
        """Initialize blood pressure evaluator with BP-specific configuration.

        Args:
            global_min: Global minimum blood pressure value used for clipping
                and denormalization sanity checks.
            global_max: Global maximum blood pressure value used for clipping
                and denormalization sanity checks.
            bhs_thresholds: List of absolute error thresholds (in mmHg) used to
                compute BHS-style percentage metrics for SBP/DBP/MAP.
            denormalize: Whether to expect and report metrics in physical units
                (mmHg). When False, metrics are treated as normalized values.
            fail_on_missing_metrics: Whether to raise an error when the
                processor does not provide precomputed BP metrics. When False,
                batches without metrics are skipped with a warning.
            *args: Additional positional arguments forwarded to ``BaseEvaluator``.
            **kwargs: Additional keyword arguments forwarded to ``BaseEvaluator``.

        Notes:
            - The evaluator expects the configured scalar processor to emit a
              ``metrics['bp']`` payload containing SBP/DBP/MAP errors and
              optional waveform metrics.
            - Core infrastructure such as checkpoint loading, dataloader
              construction, and logging is delegated to ``BaseEvaluator``.
        """
        self.global_min = global_min
        self.global_max = global_max
        self.bhs_thresholds = bhs_thresholds
        self.denormalize = denormalize
        self.fail_on_missing_metrics = fail_on_missing_metrics
        self._bp_metrics_unit: str | None = None

        super().__init__(*args, **kwargs)

    def _execute_evaluation_logic(
        self, model, test_loader: DataLoader, aggregator
    ) -> dict[str, Any]:
        """Execute blood pressure evaluation logic - modern approach only.

        Uses _predict_batch() to get processed outputs from ScalarOutputProcessor.
        Processor handles padding trimming and BP extraction automatically.

        Raises:
            ValueError: If batch lacks required 'bp_raw' key.
            KeyError: If processor outputs lack required keys ('sbp', 'dbp').
        """
        skipped_batches = 0
        total_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                total_batches += 1
                batch = self.to_device(batch)

                # Extract BP targets from batch metadata (now on correct device)
                bp_raw = batch.get("bp_raw")
                if bp_raw is None:
                    raise ValueError(
                        "BloodPressureEvaluator requires 'bp_raw' in the "
                        "incoming batch. Ensure the input preprocessing "
                        "config enables and propagates 'bp_raw'."
                    )

                outputs = self._predict_batch(batch)
                if "sbp" not in outputs or "dbp" not in outputs:
                    raise KeyError(
                        f"Processor outputs missing required keys. "
                        f"Expected 'sbp', 'dbp', got: {outputs.keys()}"
                    )

                bp_metrics_payload = None
                if isinstance(outputs.get("metrics"), dict):
                    bp_metrics_payload = outputs["metrics"].get("bp")

                if bp_metrics_payload is None:
                    direction_hint = batch.get("direction")
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
                            "Processor BP metrics required. Enable "
                            "compute_metrics=true in scalar_processor_bp "
                            "config."
                        )
                    logger.warning(
                        "BP metrics: missing from processor outputs; "
                        "skipping batch %s (direction=%s). Enable "
                        "compute_metrics=true in scalar_processor_bp "
                        "config.",
                        batch_idx,
                        direction_hint,
                    )
                    skipped_batches += 1
                    del outputs, bp_raw
                    continue

                batch_metrics = self._format_bp_metrics_from_payload(
                    precomputed_metrics=bp_metrics_payload,
                    bp_raw=bp_raw,
                    sbp_pred=outputs["sbp"],
                    dbp_pred=outputs["dbp"],
                    batch_idx=batch_idx,
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
                batch_size = batch["x"].size(0)
                self._log_batch_metrics(aggregator, batch_metrics, batch_size)

                # Clean up
                del outputs, bp_raw

        if skipped_batches > 0:
            logger.warning(
                "Blood pressure evaluation incomplete: skipped %d/%d "
                "batches due to missing metrics. Ensure "
                "processor.compute_metrics=true in config.",
                skipped_batches,
                total_batches,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        final_metrics = aggregator.get_smoothed_values()
        unit = getattr(self, "_bp_metrics_unit", None)
        if unit is not None:
            final_metrics["unit"] = unit
        return final_metrics

    def _format_bp_metrics_from_payload(
        self,
        *,
        precomputed_metrics: dict[str, Any] | None,
        bp_raw: torch.Tensor,
        sbp_pred: torch.Tensor,
        dbp_pred: torch.Tensor,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Validate and format processor-supplied BP metrics for aggregation.

        Raises:
            ValueError: If precomputed_metrics is not a dict or BP
                metrics are required but missing.
            TypeError: If bp_raw is not a tensor.
        """
        if not isinstance(precomputed_metrics, dict):
            raise ValueError(
                "Processor BP metrics required. Enable compute_metrics=true in "
                "scalar_processor_bp config."
            )

        units = precomputed_metrics.get("units", "normalized")
        logger.debug("BP metrics units: %s", units)
        expected_units = "mmHg" if self.denormalize else "normalized"
        if units != expected_units:
            logger.warning(
                "Evaluator expects denormalized metrics (%s) but "
                "processor provided %s values. Verify "
                "processor.denormalize configuration.",
                "mmHg" if self.denormalize else "normalized",
                units,
            )
        unit = units
        self._bp_metrics_unit = unit

        if not torch.is_tensor(bp_raw):
            raise TypeError(
                f"BloodPressureEvaluator expects 'bp_raw' tensor, got {type(bp_raw)}."
            )

        if not torch.is_tensor(sbp_pred) or not torch.is_tensor(dbp_pred):
            raise TypeError("SBP/DBP predictions must be torch.Tensor values.")

        required_keys = ["sbp_mae", "dbp_mae", "map_mae", "sbp_me", "dbp_me", "map_me"]
        missing = [key for key in required_keys if key not in precomputed_metrics]
        if missing:
            raise ValueError(
                "Processor BP metrics incomplete. Missing keys: "
                f"{', '.join(sorted(missing))}. "
                "Enable compute_metrics=true in scalar_processor_bp config."
            )

        map_pred = self._calculate_map(sbp_pred, dbp_pred)

        if bp_raw.dim() < 2 or bp_raw.size(1) < 3:
            raise ValueError(
                "bp_raw tensor must contain SBP/DBP/MAP columns in that order."
            )

        sbp_true = bp_raw[:, 0]  # Column 0: SBP (systolic)
        dbp_true = bp_raw[:, 1]  # Column 1: DBP (diastolic)
        map_true = bp_raw[:, 2]  # Column 2: MAP (mean arterial)

        tensor_keys = required_keys + ["waveform_mae", "waveform_me", "waveform_corr"]
        error_metrics: dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            tensor = precomputed_metrics.get(key)
            if tensor is None:
                if key in required_keys:
                    raise ValueError(
                        "Processor BP metrics missing required tensor "
                        f"for key '{key}'. Enable compute_metrics=true "
                        "in scalar_processor_bp config."
                    )
                device = sbp_pred.device
                error_metrics[key] = torch.empty(0, device=device)
                continue

            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"Processor BP metric '{key}' must be a torch.Tensor, "
                    f"got {type(tensor)}."
                )

            error_metrics[key] = tensor.detach().reshape(-1)

        sbp_pred_values = sbp_pred.reshape(-1)
        dbp_pred_values = dbp_pred.reshape(-1)
        map_pred_values = map_pred.reshape(-1)
        sbp_true_values = sbp_true.reshape(-1)
        dbp_true_values = dbp_true.reshape(-1)
        map_true_values = map_true.reshape(-1)

        total_samples = error_metrics["sbp_mae"].size(0)

        waveform_mae_tensor = error_metrics.get("waveform_mae")
        waveform_me_tensor = error_metrics.get("waveform_me")
        waveform_corr_tensor = error_metrics.get("waveform_corr")

        formatted: dict[str, Any] = {"unit": unit}
        for i in range(total_samples):
            sample_id = f"sample_{batch_idx}_{i}"

            formatted[f"sbp_mae/{sample_id}"] = error_metrics["sbp_mae"][i].item()
            formatted[f"dbp_mae/{sample_id}"] = error_metrics["dbp_mae"][i].item()
            formatted[f"map_mae/{sample_id}"] = error_metrics["map_mae"][i].item()
            formatted[f"sbp_me/{sample_id}"] = error_metrics["sbp_me"][i].item()
            formatted[f"dbp_me/{sample_id}"] = error_metrics["dbp_me"][i].item()
            formatted[f"map_me/{sample_id}"] = error_metrics["map_me"][i].item()

            if waveform_mae_tensor is not None and waveform_mae_tensor.numel() > i:
                formatted[f"waveform_mae/{sample_id}"] = waveform_mae_tensor[i].item()
            if waveform_me_tensor is not None and waveform_me_tensor.numel() > i:
                formatted[f"waveform_me/{sample_id}"] = waveform_me_tensor[i].item()
            if waveform_corr_tensor is not None and waveform_corr_tensor.numel() > i:
                formatted[f"waveform_corr/{sample_id}"] = waveform_corr_tensor[i].item()

            formatted[f"sbp_gt/{sample_id}"] = sbp_true_values[i].item()
            formatted[f"dbp_gt/{sample_id}"] = dbp_true_values[i].item()
            formatted[f"map_gt/{sample_id}"] = map_true_values[i].item()
            formatted[f"sbp_pred/{sample_id}"] = sbp_pred_values[i].item()
            formatted[f"dbp_pred/{sample_id}"] = dbp_pred_values[i].item()
            formatted[f"map_pred/{sample_id}"] = map_pred_values[i].item()

        formatted["batch_sbp_mae"] = error_metrics["sbp_mae"].mean().item()
        formatted["batch_dbp_mae"] = error_metrics["dbp_mae"].mean().item()
        formatted["batch_map_mae"] = error_metrics["map_mae"].mean().item()
        formatted["batch_waveform_mae"] = (
            waveform_mae_tensor.mean().item()
            if waveform_mae_tensor is not None and waveform_mae_tensor.numel() > 0
            else 0.0
        )
        formatted["batch_waveform_corr"] = (
            waveform_corr_tensor.mean().item()
            if waveform_corr_tensor is not None and waveform_corr_tensor.numel() > 0
            else 0.0
        )

        def _safe_to_float(value, default=0.0):
            """Convert value to float; on failure return default.

            Args:
                value: int, float, tensor, or value convertible to float.
                default: Value returned on conversion failure.

            Returns:
                float: Converted value or default on failure.
            """
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                if torch.is_tensor(value):
                    return float(value.detach().cpu().item())
                return float(value)
            except (TypeError, ValueError, AttributeError):
                logger.debug(
                    "Failed to convert value to float: %s, using default %s",
                    value,
                    default,
                )
                return default

        aami_summary = (
            precomputed_metrics.get("aami")
            if isinstance(precomputed_metrics, dict)
            else {}
        )
        if not isinstance(aami_summary, dict):
            aami_summary = {}

        for bp_type in ["sbp", "dbp", "map"]:
            type_summary = (
                aami_summary.get(bp_type) if isinstance(aami_summary, dict) else None
            )
            if not isinstance(type_summary, dict):
                type_summary = {}
            mean_value = _safe_to_float(type_summary.get("mean", 0.0))
            std_value = _safe_to_float(type_summary.get("std", 0.0))
            formatted[f"aami_mean/{bp_type}"] = mean_value
            formatted[f"aami_std/{bp_type}"] = std_value

        return formatted

    def _log_batch_metrics(self, aggregator, batch_metrics, sample_size):
        """Log scalar batch metrics to the aggregator for final smoothed values."""
        for key, value in batch_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if key not in aggregator:
                    aggregator.add_meter(key, AverageMeter())
                aggregator[key].update(value, sample_size)

    def _prepare_final_results_for_logging(
        self, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare final results for logging using the results dictionary."""
        unit = results.get("unit", "mmHg" if self.denormalize else "normalized")
        # Extract metrics from results dictionary
        waveform_mae_metrics = {
            k: v for k, v in results.items() if k.startswith("waveform_mae/")
        }
        waveform_corr_metrics = {
            k: v for k, v in results.items() if k.startswith("waveform_corr/")
        }
        sbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("sbp_mae/")}
        dbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("dbp_mae/")}
        map_mae_metrics = {k: v for k, v in results.items() if k.startswith("map_mae/")}

        waveform_mae_values = (
            list(waveform_mae_metrics.values()) if waveform_mae_metrics else [0.0]
        )
        waveform_corr_values = (
            list(waveform_corr_metrics.values()) if waveform_corr_metrics else [0.0]
        )
        sbp_mae_values = list(sbp_mae_metrics.values()) if sbp_mae_metrics else [0.0]
        dbp_mae_values = list(dbp_mae_metrics.values()) if dbp_mae_metrics else [0.0]
        map_mae_values = list(map_mae_metrics.values()) if map_mae_metrics else [0.0]

        aami_results = {}
        for bp_type in ["sbp", "dbp", "map"]:
            aami_results[f"{bp_type}_aami_mean"] = results.get(
                f"aami_mean/{bp_type}", 0.0
            )
            aami_results[f"{bp_type}_aami_std"] = results.get(
                f"aami_std/{bp_type}", 0.0
            )

        return {
            "unit": unit,
            # Waveform metrics
            "waveform_mae_mean": np.mean(waveform_mae_values),
            "waveform_mae_std": np.std(waveform_mae_values),
            "waveform_corr_mean": np.mean(waveform_corr_values),
            "waveform_corr_std": np.std(waveform_corr_values),
            # BP metrics
            "sbp_mae_mean": np.mean(sbp_mae_values),
            "sbp_mae_std": np.std(sbp_mae_values),
            "dbp_mae_mean": np.mean(dbp_mae_values),
            "dbp_mae_std": np.std(dbp_mae_values),
            "map_mae_mean": np.mean(map_mae_values),
            "map_mae_std": np.std(map_mae_values),
            # AAMI metrics (mean/std summaries)
            **aami_results,
            # Clinical standards (BHS)
            **self._prepare_clinical_standards_for_logging(results),
        }

    def _prepare_clinical_standards_for_logging(
        self, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare clinical standards metrics for logging from the results dict.

        Returns a dictionary suitable for logging clinical standard metrics.

        """
        clinical_metrics = {}

        # Extract MAE metrics
        sbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("sbp_mae/")}
        dbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("dbp_mae/")}
        map_mae_metrics = {k: v for k, v in results.items() if k.startswith("map_mae/")}

        total_samples = len(sbp_mae_metrics)

        for bp_type in ["sbp", "dbp", "map"]:
            if bp_type == "sbp":
                mae_metrics = sbp_mae_metrics
            elif bp_type == "dbp":
                mae_metrics = dbp_mae_metrics
            else:  # map
                mae_metrics = map_mae_metrics

            for threshold in self.bhs_thresholds:
                if mae_metrics and total_samples > 0:
                    # Count samples within threshold
                    mae_values = list(mae_metrics.values())
                    count = sum(1 for mae in mae_values if mae <= threshold)
                    percentage = (count / total_samples) * 100
                else:
                    percentage = 0.0
                clinical_metrics[f"{bp_type}_bhs_{threshold}mmHg_pct"] = percentage

        return clinical_metrics

    def print_results(
        self, results: dict[str, Any], test_loader: DataLoader | None = None
    ) -> None:
        """Print comprehensive blood pressure results using the results dictionary."""
        print("\nBlood Pressure Results:")
        print("=" * 50)
        unit = results.get("unit", "mmHg" if self.denormalize else "normalized")
        print(f"Unit: {unit}")

        # Print waveform reconstruction results
        self._print_waveform_results(results)

        # Print BP values results
        self._print_bp_results(results)

        # Print clinical standards
        self._print_clinical_standards(results)

        # Log final results to WandB (using base evaluator method)
        final_results = self._prepare_final_results_for_logging(results)
        self._log_final_results(final_results, prefix="blood_pressure")

        # Specialized WandB logging (only if test_loader is available)
        if test_loader is not None:
            self._log_to_wandb(test_loader, results)

            # Note: CSV saving is handled automatically by base evaluator
            # in _finalize_evaluation_results. The _prepare_csv_data
            # method will be called by base evaluator for detailed CSV
            # export.

    def _print_waveform_results(self, results: dict[str, Any]):
        """Print waveform reconstruction results using the results dictionary."""
        print("\nABP Waveform Reconstruction:")
        print("-" * 30)

        # Extract waveform metrics from results dictionary
        waveform_mae_metrics = {
            k: v for k, v in results.items() if k.startswith("waveform_mae/")
        }
        waveform_me_metrics = {
            k: v for k, v in results.items() if k.startswith("waveform_me/")
        }
        waveform_corr_metrics = {
            k: v for k, v in results.items() if k.startswith("waveform_corr/")
        }

        if waveform_mae_metrics:
            waveform_mae_values = list(waveform_mae_metrics.values())
            waveform_mae_mean = np.mean(waveform_mae_values)
            waveform_mae_std = np.std(waveform_mae_values)
        else:
            waveform_mae_mean = waveform_mae_std = 0.0

        if waveform_me_metrics:
            waveform_me_values = list(waveform_me_metrics.values())
            waveform_me_mean = np.mean(waveform_me_values)
            waveform_me_std = np.std(waveform_me_values)
        else:
            waveform_me_mean = waveform_me_std = 0.0

        if waveform_corr_metrics:
            waveform_corr_values = list(waveform_corr_metrics.values())
            waveform_corr_mean = np.mean(waveform_corr_values)
            waveform_corr_std = np.std(waveform_corr_values)
        else:
            waveform_corr_mean = waveform_corr_std = 0.0

        print(f"MAE: {waveform_mae_mean:.2f} ± {waveform_mae_std:.2f}")
        print(f"ME: {waveform_me_mean:.2f} ± {waveform_me_std:.2f}")
        print(
            f"Pearson Correlation: {waveform_corr_mean:.3f} ± {waveform_corr_std:.3f}"
        )

    def _print_bp_results(self, results: dict[str, Any]):
        """Print BP values results using the results dictionary."""
        print("\nBP Values Results:")
        print("-" * 30)

        for bp_type in ["sbp", "dbp", "map"]:
            # Extract metrics from results dictionary
            mae_metrics = {
                k: v for k, v in results.items() if k.startswith(f"{bp_type}_mae/")
            }
            me_metrics = {
                k: v for k, v in results.items() if k.startswith(f"{bp_type}_me/")
            }

            if mae_metrics:
                mae_values = list(mae_metrics.values())
                mae_mean = np.mean(mae_values)
                mae_std = np.std(mae_values)
            else:
                mae_mean = mae_std = 0.0

            if me_metrics:
                me_values = list(me_metrics.values())
                me_mean = np.mean(me_values)
                me_std = np.std(me_values)
            else:
                me_mean = me_std = 0.0

            aami_mean = results.get(f"aami_mean/{bp_type}", 0.0)
            aami_std = results.get(f"aami_std/{bp_type}", 0.0)

            print(f"{bp_type.upper()} MAE: {mae_mean:.2f} ± {mae_std:.2f}")
            print(f"{bp_type.upper()} ME: {me_mean:.2f} ± {me_std:.2f}")
            print(f"{bp_type.upper()} AAMI Mean: {aami_mean:.2f} ± {aami_std:.2f}")

    def _print_clinical_standards(self, results: dict[str, Any]):
        """Print clinical standards using the results dictionary."""
        print("\nBHS Standards:")
        print("-" * 30)

        # Extract MAE metrics to calculate BHS standards
        sbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("sbp_mae/")}
        dbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("dbp_mae/")}
        map_mae_metrics = {k: v for k, v in results.items() if k.startswith("map_mae/")}

        total_samples = len(sbp_mae_metrics)
        bhs_thresholds = self.bhs_thresholds

        for bp_type in ["sbp", "dbp", "map"]:
            print(f"\n{bp_type.upper()}:")

            if bp_type == "sbp":
                mae_metrics = sbp_mae_metrics
            elif bp_type == "dbp":
                mae_metrics = dbp_mae_metrics
            else:  # map
                mae_metrics = map_mae_metrics

            for threshold in bhs_thresholds:
                if mae_metrics:
                    # Count samples within threshold
                    mae_values = list(mae_metrics.values())
                    count = sum(1 for mae in mae_values if mae <= threshold)
                    percentage = (count / total_samples) * 100
                else:
                    count = 0
                    percentage = 0.0
                print(f"≤{threshold}mmHg: {percentage:.1f}%")

    def _prepare_csv_data(
        self, results: dict[str, Any], test_loader, direction: str, seed: int
    ) -> pd.DataFrame:
        """Prepare comprehensive BP data for CSV export.

        The export contains per-sample metrics following the waveform pattern.

        Raises:
            KeyError: If required metric keys are missing from results.
            TypeError: If result values have unexpected types during CSV construction.
        """
        unit = results.get("unit", "mmHg" if self.denormalize else "normalized")
        meta = self._get_csv_base_metadata(test_loader, direction, seed, unit=unit)
        columns = [
            "Dataset",
            "Method",
            "Direction",
            "Seed",
            "Unit",
            "sbp_gt",
            "sbp_pred",
            "sbp_mae",
            "sbp_me",
            "dbp_gt",
            "dbp_pred",
            "dbp_mae",
            "dbp_me",
            "map_gt",
            "map_pred",
            "map_mae",
            "map_me",
            "waveform_mae",
            "waveform_me",
            "waveform_corr",
            "sample_index",
        ]
        data: dict[str, Any] = {col: [] for col in columns}

        metric_prefixes = [
            "sbp_mae",
            "dbp_mae",
            "map_mae",
            "sbp_me",
            "dbp_me",
            "map_me",
            "waveform_mae",
            "waveform_me",
            "waveform_corr",
            "sbp_gt",
            "dbp_gt",
            "map_gt",
            "sbp_pred",
            "dbp_pred",
            "map_pred",
        ]
        metrics_by_prefix = {
            p: {k: v for k, v in results.items() if k.startswith(f"{p}/")}
            for p in metric_prefixes
        }

        sample_data = []
        for sbp_key, _sbp_mae_value in metrics_by_prefix["sbp_mae"].items():
            parsed = self._parse_sample_key_from_metric(sbp_key, "sbp_mae")
            if parsed is not None:
                batch_idx, sample_idx = parsed
                sample_data.append((batch_idx, sample_idx))

        if not sample_data:
            logger.warning("No sample data found in results")
            return pd.DataFrame(columns=cast("Any", columns))

        sample_data.sort(key=lambda x: (x[0], x[1]))

        for global_idx, (batch_idx, sample_idx) in enumerate(sample_data):
            sample_id = f"sample_{batch_idx}_{sample_idx}"
            row = {}
            for prefix in metric_prefixes:
                key = f"{prefix}/{sample_id}"
                val = metrics_by_prefix[prefix].get(key, 0.0)
                row[prefix] = self._validate_csv_value(
                    val, 0.0, f"{prefix} sample {global_idx}"
                )

            data["Dataset"].append(meta["dataset"])
            data["Method"].append(meta["method"])
            data["Direction"].append(meta["direction"])
            data["Seed"].append(meta["seed"])
            data["Unit"].append(meta["unit"])
            data["sample_index"].append(global_idx)
            data["sbp_gt"].append(row["sbp_gt"])
            data["sbp_pred"].append(row["sbp_pred"])
            data["sbp_mae"].append(row["sbp_mae"])
            data["sbp_me"].append(row["sbp_me"])
            data["dbp_gt"].append(row["dbp_gt"])
            data["dbp_pred"].append(row["dbp_pred"])
            data["dbp_mae"].append(row["dbp_mae"])
            data["dbp_me"].append(row["dbp_me"])
            data["map_gt"].append(row["map_gt"])
            data["map_pred"].append(row["map_pred"])
            data["map_mae"].append(row["map_mae"])
            data["map_me"].append(row["map_me"])
            data["waveform_mae"].append(row["waveform_mae"])
            data["waveform_me"].append(row["waveform_me"])
            data["waveform_corr"].append(row["waveform_corr"])

        return pd.DataFrame(data)

    # Helper methods

    def _calculate_map(self, sbp, dbp):
        """Calculate MAP from SBP and DBP."""
        return dbp + (sbp - dbp) / 3

    def _log_to_wandb(self, test_loader, results: dict[str, Any]):
        """Log results to WandB using results dictionary."""
        if not hasattr(self, "progress_bar") or self.progress_bar is None:
            return

        dataset = self._get_dataset_name(test_loader)
        method = self._get_model_name()
        unit = results.get("unit", "mmHg" if self.denormalize else "normalized")

        direction = self._get_direction_name()

        # Extract metrics from results dictionary
        sbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("sbp_mae/")}
        dbp_mae_metrics = {k: v for k, v in results.items() if k.startswith("dbp_mae/")}
        map_mae_metrics = {k: v for k, v in results.items() if k.startswith("map_mae/")}
        sbp_me_metrics = {k: v for k, v in results.items() if k.startswith("sbp_me/")}
        dbp_me_metrics = {k: v for k, v in results.items() if k.startswith("dbp_me/")}
        map_me_metrics = {k: v for k, v in results.items() if k.startswith("map_me/")}
        waveform_mae_metrics = {
            k: v for k, v in results.items() if k.startswith("waveform_mae/")
        }
        waveform_corr_metrics = {
            k: v for k, v in results.items() if k.startswith("waveform_corr/")
        }

        total_samples = len(sbp_mae_metrics)
        bhs_thresholds = self.bhs_thresholds

        # Log BHS standards
        for bp_type in ["sbp", "dbp", "map"]:
            if bp_type == "sbp":
                mae_metrics = sbp_mae_metrics
            elif bp_type == "dbp":
                mae_metrics = dbp_mae_metrics
            else:  # map
                mae_metrics = map_mae_metrics

            for threshold in bhs_thresholds:
                if mae_metrics and total_samples > 0:
                    # Count samples within threshold
                    mae_values = list(mae_metrics.values())
                    count = sum(1 for mae in mae_values if mae <= threshold)
                    percentage = (count / total_samples) * 100
                else:
                    percentage = 0.0

                self._log_domain_result(
                    task="BHS",
                    dataset=dataset,
                    method=method,
                    direction=direction,
                    value=percentage,
                    seed=self._get_seed(),
                    Threshold=f"≤{threshold}mmHg",
                    Measure=bp_type.upper(),
                    Unit=unit,
                )

        # Log AAMI standards
        for bp_type in ["sbp", "dbp", "map"]:
            if bp_type == "sbp":
                me_metrics = sbp_me_metrics
            elif bp_type == "dbp":
                me_metrics = dbp_me_metrics
            else:  # map
                me_metrics = map_me_metrics

            if me_metrics:
                me_values = list(me_metrics.values())
                me_mean = np.mean(me_values)
                me_std = np.std(me_values)
            else:
                me_mean = me_std = 0.0

            self._log_domain_result(
                task="AAMI",
                dataset=dataset,
                method=method,
                direction=direction,
                value=float(me_mean),
                seed=self._get_seed(),
                Measure=bp_type.upper(),
                Std=float(me_std),
                Unit=unit,
            )

        # Log clinical tasks
        if waveform_mae_metrics:
            waveform_mae_values = list(waveform_mae_metrics.values())
            waveform_mae_mean = np.mean(waveform_mae_values)
            waveform_mae_std = np.std(waveform_mae_values)
        else:
            waveform_mae_mean = waveform_mae_std = 0.0

        if waveform_corr_metrics:
            waveform_corr_values = list(waveform_corr_metrics.values())
            waveform_corr_mean = np.mean(waveform_corr_values)
            waveform_corr_std = np.std(waveform_corr_values)
        else:
            waveform_corr_mean = waveform_corr_std = 0.0

        # Log MAE
        self._log_domain_result(
            task="Clinical_Tasks",
            dataset=dataset,
            method=method,
            direction=direction,
            value=float(waveform_mae_mean),
            seed=self._get_seed(),
            Metric="MAE",
            Std=float(waveform_mae_std),
            Unit=unit,
        )

        # Log Pearson Correlation
        self._log_domain_result(
            task="Clinical_Tasks",
            dataset=dataset,
            method=method,
            direction=direction,
            value=float(waveform_corr_mean),
            seed=self._get_seed(),
            Metric="PC",  # Pearson Correlation
            Std=float(waveform_corr_std),
            Unit=unit,
        )

    # Logging helpers
    def _log_domain_result(
        self,
        task: str,
        dataset: str,
        method: str,
        direction: str,
        value: float,
        seed: int,
        **kwargs,
    ):
        """Log domain-specific results to WandB.

        Args:
            task: Task name (BHS, AAMI, Clinical_Tasks)
            dataset: Dataset name
            method: Model method name
            direction: Direction (e.g., 'ppg2abp', 'ecg2abp')
            value: Primary metric value
            seed: Random seed for reproducibility
            **kwargs: Additional task-specific parameters
        """
        if not hasattr(self, "progress_bar") or self.progress_bar is None:
            return
        wandb_ref = getattr(self.progress_bar, "wandb", None)
        if wandb_ref is None:
            return

        wandb_ref.log_domain_metric(
            task=task,
            Dataset=dataset,
            Method=method,
            Direction=direction,
            Value=value,
            seed=seed,
            **kwargs,
        )


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_blood_pressure", node=BloodPressureConfig, group="evaluator")
