"""Utility functions for processor-level metric computations.

This module centralizes all metric calculations used across waveform and scalar
processors. Processors invoke these helpers from their unified ``process()``
method, enabling trainers and evaluators to receive stage-aware metrics inside
the canonical processor output payload. Functions are intentionally defensive:
they validate inputs, perform shape checks, and fail gracefully by logging
warnings rather than propagating exceptions.

Functions:
    - is_direction_enabled: Return True when metrics for a direction should be computed
    - extract_target_from_batch: Extract target tensor from unified batch structure
    - compute_waveform_metrics: Compute MAE and Pearson correlation for waveforms
    - compute_bp_metrics: Compute BP metrics (SBP/DBP/MAP, BHS, AAMI) in mmHg
      or normalized
    - calculate_map: Compute Mean Arterial Pressure from SBP/DBP
    - calculate_pearson_correlation: Compute Pearson correlation per sample

Examples:
    >>> metrics = compute_waveform_metrics(processed_outputs, batch, padding_length=15)
    >>> bp_metrics = compute_bp_metrics(processed_outputs, batch, denormalize=True, ...)

See Also:
    - src.processors.waveform_processor: WaveformOutputProcessor
    - src.processors.scalar_processor: ScalarOutputProcessor
"""

from __future__ import annotations

# Standard library imports
import logging
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# Third-party imports
import torch
from torch import Tensor

# Local imports
from src.utils.bp import get_dbp
from src.utils.bp import get_map
from src.utils.bp import get_sbp
from src.utils.utils_preprocessing import global_min_max_norm

logger = logging.getLogger(__name__)


def _normalize_direction(direction: str | None) -> str | None:
    """Normalize direction key to lowercase stripped string for comparison."""
    if direction is None:
        return None
    return direction.strip().lower()


def is_direction_enabled(
    direction: str | None,
    enabled_directions: Iterable[str] | None,
) -> bool:
    """Return True when metrics for a direction should be computed.

    Args:
        direction: The direction key associated with the current sample/pipeline.
        enabled_directions: Optional iterable of direction keys that are allowed.

    Returns:
        bool: ``True`` if the provided direction is enabled or no filter is set.
    """
    if not enabled_directions:
        return True

    normalized_direction = _normalize_direction(direction)
    normalized_filters: set[str] = {
        d
        for d in (_normalize_direction(value) for value in enabled_directions)
        if d is not None
    }

    if not normalized_filters:
        return True

    # If direction metadata is missing, keep metrics enabled by default.
    if normalized_direction is None:
        return True

    return normalized_direction in normalized_filters


def extract_target_from_batch(
    batch: dict[str, Any],
    target_key: str = "x",
    index_key: str = "tgt_idxs",
) -> Tensor:
    """Extract the target tensor from a unified batch structure.

    Selection logic mirrors :meth:`WaveformOutputProcessor._extract_target_waveform`:
    when ``tgt_idxs`` is present, the per-sample channel index is used; when absent,
    the target is used as-is only if it has a single channel (otherwise selection
    is ambiguous and a ``KeyError`` is raised).

    Args:
        batch: Batch dictionary produced by the collate function.
        target_key: Key pointing to the multi-channel waveform tensor (e.g. ``"y"``
            for ground-truth target, ``"x"`` for input).
        index_key: Key pointing to the per-sample channel indices (``"tgt_idxs``).
            Optional: when missing, target must have exactly one channel.

    Returns:
        Tensor shaped ``[B, 1, T]`` containing the selected target channel.

    Raises:
        ValueError: If batch is None or waveform tensor has invalid shape (not 2D/3D).
        KeyError: If target_key is missing from batch or index_key is missing when
            target has multiple channels.
        TypeError: If waveform at target_key is not a tensor.
    """
    if batch is None:
        raise ValueError("Batch is required to extract targets for metric computation.")

    if target_key not in batch:
        raise KeyError(f"Batch missing '{target_key}' required for target extraction.")

    waveform = batch[target_key]
    tgt_idxs = batch.get(index_key) if isinstance(batch, dict) else None

    if not torch.is_tensor(waveform):
        raise TypeError(f"Expected tensor for '{target_key}', got {type(waveform)!r}")
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)
    if waveform.dim() != 3:
        raise ValueError(
            f"Waveform tensor must be 2D [B, T] or 3D [B, C, T]; received shape {
                tuple(waveform.shape)
            }."
        )

    if tgt_idxs is None or not torch.is_tensor(tgt_idxs):
        if waveform.size(1) == 1:
            return waveform
        raise KeyError(
            f"Batch missing '{index_key}' required for target extraction when target "
            f"has multiple channels (got {waveform.size(1)} channels)."
        )

    batch_size = waveform.size(0)
    batch_arange = torch.arange(batch_size, device=waveform.device)
    target = waveform[batch_arange, tgt_idxs].unsqueeze(1)
    return target


def compute_waveform_metrics(
    processed_outputs: dict[str, Tensor],
    batch: dict[str, Any],
    *,
    direction: str | None = None,
    device: torch.device | None = None,
    enabled_directions: Iterable[str] | None = None,
    padding_length: int = 0,
) -> dict[str, Tensor] | None:
    """Compute waveform reconstruction metrics from processor outputs.

    Called by :class:`WaveformOutputProcessor.process()` and related processors to
    attach MAE and Pearson correlation metrics for validation and test stages. On
    failure, logs a warning and returns ``None`` so processors can omit metrics
    without crashing the pipeline.

    Args:
        processed_outputs: Dict containing "waveform" key with predicted tensor
            [B, C, T]. Multi-channel waveforms are reduced using tgt_idxs when present.
        batch: Collated batch with "y" (target waveform) and optionally "tgt_idxs".
        direction: Optional direction key for filtering (metrics only
            computed when enabled).
        device: Reserved for future use; ignored.
        enabled_directions: Optional iterable of direction keys for which
            to compute metrics.
        padding_length: Number of samples to trim from each end of
            target/prediction before metrics.

    Returns:
        Dict with keys "mae" and "correlation" (tensors per sample), or None if
        direction is disabled or computation fails (e.g. shape mismatch, missing keys).
    """
    del device  # Reserved for future use

    if not is_direction_enabled(direction, enabled_directions):
        return None

    try:
        waveform = processed_outputs.get("waveform")
        if waveform is None:
            raise KeyError(
                "Processed outputs missing 'waveform' entry required for metrics."
            )

        tgt_idxs = batch.get("tgt_idxs") if isinstance(batch, dict) else None
        if (
            tgt_idxs is not None
            and torch.is_tensor(tgt_idxs)
            and waveform.dim() >= 3
            and waveform.size(1) > 1
        ):
            if tgt_idxs.device != waveform.device:
                tgt_idxs = tgt_idxs.to(waveform.device)
            batch_size = waveform.size(0)
            index_range = torch.arange(batch_size, device=waveform.device)
            waveform = waveform[index_range, tgt_idxs].unsqueeze(1)

        target = extract_target_from_batch(batch, target_key="y")
        if padding_length:
            if target.size(-1) <= padding_length * 2:
                raise ValueError(
                    f"Padding length {padding_length} too large for target "
                    f"sequence of length {target.size(-1)}"
                )
            target = target[..., padding_length:-padding_length]

        _validate_pair_shapes(waveform, target, "waveform", "target")

        mae = torch.nn.functional.l1_loss(waveform, target, reduction="none").mean(
            dim=-1
        )
        if mae.dim() == 2:
            mae = mae.mean(dim=1)
        elif mae.dim() > 2:
            mae = mae.view(mae.size(0), -1).mean(dim=1)

        mae = mae.squeeze(-1) if mae.dim() > 1 else mae

        correlation = calculate_pearson_correlation(waveform, target)

        return {
            "mae": mae,
            "correlation": correlation,
        }
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.warning("Waveform metrics computation failed: %s", exc, exc_info=True)
        return None


def compute_bp_metrics(
    processed_outputs: dict[str, Tensor],
    batch: dict[str, Any],
    *,
    direction: str | None = None,
    device: torch.device | None = None,
    denormalize: bool = False,
    global_min: float | None = None,
    global_max: float | None = None,
    bhs_thresholds: Iterable[float] | None = None,
    enabled_directions: Iterable[str] | None = None,
    padding_length: int = 0,
) -> dict[str, Any] | None:
    """Compute blood pressure metrics from processor outputs and batch metadata.

    Args:
        processed_outputs: Dictionary containing BP scalars (and optionally waveform).
        batch: Collated batch with target information (requires ``bp_raw``).
            Note: ``bp_raw`` is always in raw mmHg units (never normalized).
        direction: Optional direction filter metadata.
        denormalize: Whether to denormalize predictions back to mmHg.
            - True: Predictions assumed normalized [0,1], denormalized to mmHg
            - False: Predictions assumed already in raw mmHg
            Ground truth (bp_raw) is always raw mmHg regardless of this flag.
        global_min: Minimum value for denormalization (required when denormalize=True).
            Typically the dataset's DBP minimum (e.g., UCI: 50.0, PulseDB: 2.34).
        global_max: Maximum value for denormalization (required when denormalize=True).
            Typically the dataset's SBP maximum (e.g., UCI: 189.98, PulseDB: 286.58).
        bhs_thresholds: Optional override for BHS thresholds.
        enabled_directions: Optional set of directions to enable metrics.
        padding_length: Amount of symmetrical padding trimmed from predictions.

    Returns:
        Dictionary with BP metrics in mmHg (if denormalize=True) or normalized units,
        or None if computation fails or direction is disabled.

    Raises:
        ValueError: If denormalize=True but global_min or global_max is None.
    """
    del device  # Reserved for future use

    if not is_direction_enabled(direction, enabled_directions):
        return None

    try:
        if denormalize and (global_min is None or global_max is None):
            raise ValueError(
                "BP metrics computation requires global_min and global_max "
                "when denormalize=True. Set processor.global_min and "
                "processor.global_max in your config. "
                "Dataset-specific values: "
                "UCI (global_min=50.0, global_max=189.98), "
                "PulseDB (global_min=2.34, global_max=286.58)."
            )

        waveform = processed_outputs.get("waveform")
        sbp_pred = processed_outputs.get("sbp")
        dbp_pred = processed_outputs.get("dbp")

        if sbp_pred is None or dbp_pred is None:
            raise KeyError(
                "Processed outputs must include 'sbp' and 'dbp' for BP metrics."
            )

        if "bp_raw" not in batch:
            raise KeyError("Batch missing 'bp_raw' required for BP metric computation.")
        target_waveform: Tensor | None = None
        waveform_eval: Tensor | None = None

        if waveform is not None:
            target_waveform = extract_target_from_batch(batch, target_key="y")
            if padding_length:
                if target_waveform.size(-1) <= padding_length * 2:
                    raise ValueError(
                        f"Padding length {padding_length} too large for "
                        f"target sequence of length {target_waveform.size(-1)}"
                    )
                target_waveform = target_waveform[..., padding_length:-padding_length]
            _validate_pair_shapes(waveform, target_waveform, "waveform", "target")
            waveform_eval = _maybe_denormalize_tensor(
                waveform, denormalize, global_min, global_max
            )
            target_waveform = _maybe_denormalize_tensor(
                target_waveform, denormalize, global_min, global_max
            )

        # Extract ground truth BP scalars from batch
        # Note: bp_raw is ALWAYS in raw mmHg units (never normalized)
        bp_raw = batch["bp_raw"]

        # Denormalize predictions based on the denormalize flag
        # - denormalize=True: predictions are [0,1], convert to mmHg
        # - denormalize=False: predictions already in mmHg, use as-is
        sbp_eval = _maybe_denormalize_tensor(
            sbp_pred, denormalize, global_min, global_max
        )
        dbp_eval = _maybe_denormalize_tensor(
            dbp_pred, denormalize, global_min, global_max
        )

        map_pred = calculate_map(sbp_eval, dbp_eval)

        # bp_raw is always in raw mmHg units - never denormalize it
        sbp_true = get_sbp(bp_raw)
        dbp_true = get_dbp(bp_raw)
        map_true = get_map(bp_raw)

        base_metrics = _calculate_bp_error_metrics(
            sbp_eval,
            dbp_eval,
            map_pred,
            sbp_true,
            dbp_true,
            map_true,
            waveform_eval,
            target_waveform,
        )
        metrics: dict[str, Any] = dict(base_metrics)
        metrics["bhs"] = _calculate_bhs(base_metrics, bhs_thresholds)
        metrics["aami"] = _calculate_aami(base_metrics)
        metrics["units"] = "mmHg" if denormalize else "normalized"

        return metrics
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.warning("BP metrics computation failed: %s", exc, exc_info=True)
        return None


def calculate_map(sbp: Tensor, dbp: Tensor) -> Tensor:
    """Calculate Mean Arterial Pressure (MAP) from systolic and diastolic values.

    Uses the formula MAP = DBP + (SBP - DBP) / 3.

    Args:
        sbp: Systolic blood pressure tensor (any shape).
        dbp: Diastolic blood pressure tensor (broadcast-compatible with sbp).

    Returns:
        MAP tensor with shape broadcast from sbp and dbp.
    """
    sbp_aligned, dbp_aligned = torch.broadcast_tensors(sbp, dbp)
    return dbp_aligned + (sbp_aligned - dbp_aligned) / 3.0


def calculate_pearson_correlation(pred: Tensor, target: Tensor) -> Tensor:
    """Compute Pearson correlation coefficient per sample.

    Args:
        pred: Predicted tensor [B, ...]; flattened per sample for correlation.
        target: Target tensor same shape as pred.

    Returns:
        Tensor of shape [B] with Pearson correlation per sample. Zero
        where denominator is zero.
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)

    numerator = torch.sum((pred_flat - pred_mean) * (target_flat - target_mean), dim=1)
    denominator = torch.sqrt(
        torch.sum((pred_flat - pred_mean) ** 2, dim=1)
        * torch.sum((target_flat - target_mean) ** 2, dim=1)
    )

    return torch.where(
        denominator > 0, numerator / denominator, torch.zeros_like(numerator)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_pair_shapes(
    pred: Tensor, target: Tensor, pred_name: str, target_name: str
) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch between {pred_name} {tuple(pred.shape)} and {target_name} {
                tuple(target.shape)
            }"
        )


def _calculate_bp_error_metrics(
    sbp_pred: Tensor,
    dbp_pred: Tensor,
    map_pred: Tensor,
    sbp_true: Tensor,
    dbp_true: Tensor,
    map_true: Tensor,
    waveform_pred: Tensor | None,
    waveform_true: Tensor | None,
) -> dict[str, Tensor]:
    sbp_pred, dbp_pred, map_pred = [
        _squeeze_last_dim(t) for t in (sbp_pred, dbp_pred, map_pred)
    ]
    sbp_true, dbp_true, map_true = [
        _squeeze_last_dim(t) for t in (sbp_true, dbp_true, map_true)
    ]

    sbp_mae = torch.abs(sbp_pred - sbp_true)
    dbp_mae = torch.abs(dbp_pred - dbp_true)
    map_mae = torch.abs(map_pred - map_true)

    sbp_me = sbp_pred - sbp_true
    dbp_me = dbp_pred - dbp_true
    map_me = map_pred - map_true
    metrics: dict[str, Any] = {
        "sbp_mae": sbp_mae,
        "dbp_mae": dbp_mae,
        "map_mae": map_mae,
        "sbp_me": sbp_me,
        "dbp_me": dbp_me,
        "map_me": map_me,
    }

    if waveform_pred is not None and waveform_true is not None:
        waveform_mae = torch.nn.functional.l1_loss(
            waveform_pred, waveform_true, reduction="none"
        ).mean(dim=-1)
        if waveform_mae.dim() == 2:
            waveform_mae = waveform_mae.mean(dim=1)
        elif waveform_mae.dim() > 2:
            waveform_mae = waveform_mae.view(waveform_mae.size(0), -1).mean(dim=1)

        waveform_mae = (
            waveform_mae.squeeze(-1) if waveform_mae.dim() > 1 else waveform_mae
        )

        waveform_me = (waveform_pred - waveform_true).mean(dim=-1)
        if waveform_me.dim() == 2:
            waveform_me = waveform_me.mean(dim=1)
        elif waveform_me.dim() > 2:
            waveform_me = waveform_me.view(waveform_me.size(0), -1).mean(dim=1)

        waveform_me = waveform_me.squeeze(-1) if waveform_me.dim() > 1 else waveform_me

        waveform_corr = calculate_pearson_correlation(waveform_pred, waveform_true)

        metrics.update(
            {
                "waveform_mae": waveform_mae,
                "waveform_me": waveform_me,
                "waveform_corr": waveform_corr,
            }
        )
    else:
        empty = torch.empty(0, device=sbp_mae.device, dtype=sbp_mae.dtype)
        metrics.update(
            {
                "waveform_mae": empty,
                "waveform_me": empty,
                "waveform_corr": empty,
            }
        )

    return metrics


def _calculate_bhs(
    metrics: dict[str, Tensor], thresholds: Iterable[float] | None
) -> dict[str, dict[float, int]]:
    default_thresholds = [5, 10, 15]
    thresholds_list = list(thresholds) if thresholds is not None else default_thresholds

    bhs_results: dict[str, dict[float, int]] = {}
    for bp_type in ("sbp", "dbp", "map"):
        metric_tensor = metrics.get(f"{bp_type}_mae")
        if metric_tensor is None:
            continue

        bhs_results[bp_type] = {}
        for threshold in thresholds_list:
            count = int((metric_tensor <= threshold).sum().item())
            bhs_results[bp_type][float(threshold)] = count

    return bhs_results


def _calculate_aami(metrics: dict[str, Tensor]) -> dict[str, dict[str, float]]:
    aami_results: dict[str, dict[str, float]] = {}
    for bp_type in ("sbp", "dbp", "map"):
        metric_tensor = metrics.get(f"{bp_type}_me")
        if metric_tensor is None:
            continue

        mean_val = (
            float(metric_tensor.mean().item()) if metric_tensor.numel() > 0 else 0.0
        )
        std_val = (
            float(metric_tensor.std(unbiased=False).item())
            if metric_tensor.numel() > 1
            else 0.0
        )

        aami_results[bp_type] = {
            "mean": mean_val,
            "std": std_val,
        }

    return aami_results


def _maybe_denormalize_tensor(
    tensor: Tensor,
    denormalize: bool,
    global_min: float | None,
    global_max: float | None,
) -> Tensor:
    if not denormalize:
        return tensor

    if global_min is None or global_max is None:
        raise ValueError(
            "Global min and max must be provided when denormalization is enabled."
        )

    original_device = tensor.device
    tensor_np = tensor.detach().cpu().numpy()
    tensor_np = global_min_max_norm(
        tensor_np,
        global_min_max={"min": global_min, "max": global_max},
        unnorm=True,
    )
    return torch.from_numpy(tensor_np).to(original_device)


def _squeeze_last_dim(tensor: Tensor) -> Tensor:
    return tensor.squeeze(-1) if tensor.dim() > 1 and tensor.size(-1) == 1 else tensor
