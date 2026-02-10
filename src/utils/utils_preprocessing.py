"""Preprocessing utilities for normalization and dataset creation.

This module provides min-max normalization (numpy and torch), zero-centering,
global min-max normalization, MAP calculation, and safe dataset creation
helpers. Used by preprocessors, collate functions, and metric computation.

Functions:
    - min_max_norm: Min-max scaling to [0,1] (numpy)
    - zero_centered: Center data by subtracting mean (numpy)
    - min_max_norm_torch: Min-max scaling for tensors
    - check_file_exists: Validate file path
    - calculate_map: Compute mean arterial pressure
    - global_min_max_norm: Global min-max normalize/denormalize
    - safe_create_dataset: Create dataset with validation
    - print_model_parameters: Log model parameter counts
    - ensure_dir_exists: Check directory existence

Examples:
    >>> x_norm = min_max_norm(x, axis=-1)
    >>> tensor_norm = global_min_max_norm(
    ...     arr, global_min_max={"min": 0, "max": 1}
    ... )

See Also:
    - src.utils.collate_utils: normalize_waveform_batch_torch
    - src.processors.metrics_utils: global_min_max_norm for denormalization
"""

# Standard library imports
import logging
import os
from typing import Any

# Third-party library imports
import numpy as np
import pandas as pd
import torch

# Local imports

logger = logging.getLogger(__name__)


def min_max_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize data using min-max scaling to [0,1] range.

    Args:
        x: Input data array.
        axis: Axis along which to compute min/max. Default is -1 (last axis).

    Returns:
        Normalized data array in [0,1] range.

    Note:
        For flat inputs (where min == max), returns zeros to avoid division
        by zero. This ensures finite outputs for constant arrays.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy array")

    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)

    # Handle zero-range (flat inputs) to avoid division by zero
    range_vals = x_max - x_min
    # Replace zero ranges with 1.0 to avoid division by zero
    # This results in zeros for flat inputs, which is a safe default
    range_vals = np.where(range_vals == 0, 1.0, range_vals)

    # Normalize to [0,1] range
    x_scaled = (x - x_min) / range_vals

    return x_scaled


def zero_centered(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Center data around zero by subtracting mean.

    Args:
        x: Input data array.
        axis: Axis along which to compute mean. Default is -1.

    Returns:
        Zero-centered data array.

    Raises:
        TypeError: If x is not a numpy array
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy array")

    mean = np.mean(x, axis=axis, keepdims=True)
    x_scaled = x - mean

    return x_scaled


def min_max_norm_torch(
    x: torch.Tensor,
    feature_range: dict[str, float] | None = None,
) -> torch.Tensor:
    """Normalize tensor data using min-max scaling.

    Args:
        x: Input tensor with shape [B, C, T] or similar.
        feature_range: Target range for normalization with keys "min" and "max".
            If None, defaults to [0, 1] range.

    Returns:
        Normalized tensor in specified range.

    Note:
        For flat inputs (where min == max), returns zeros to avoid division
        by zero. This ensures finite outputs for constant tensors.
    """
    if feature_range is None:
        feature_range = {"max": 1, "min": 0}

    x_min = torch.min(x, 2)[0].unsqueeze(-1)  # [B, C, 1]
    x_max = torch.max(x, 2)[0].unsqueeze(-1)  # [B, C, 1]

    # Avoid division by zero for flat inputs
    range_vals = x_max - x_min  # [B, C, 1]
    range_vals = torch.where(range_vals < 1e-8, torch.ones_like(range_vals), range_vals)

    # Normalize to [0,1] range first
    x_std = (x - x_min) / range_vals

    # Scale to target feature_range if specified
    target_min = feature_range.get("min", 0)
    target_max = feature_range.get("max", 1)
    if target_min != 0 or target_max != 1:
        x_std = x_std * (target_max - target_min) + target_min
    return x_std


def global_min_max_norm(
    x: np.ndarray,
    global_min_max: dict[str, float],
    unnorm: bool = False,
) -> np.ndarray:
    """Normalize/unnormalize values using global min-max values to [0,1] range.

    Args:
        x: Input array of values to normalize/unnormalize.
        global_min_max: Dictionary with global min/max values. Required keys:
            "min" (global minimum), "max" (global maximum).
        unnorm: If True, unnormalize the data back to original range. If False,
            normalize to [0,1] range.

    Returns:
        Normalized/unnormalized data array.

    Raises:
        ValueError: If global_min_max is missing required keys or if
            global_max == global_min
        TypeError: If x is not a numpy array
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy.ndarray")
    required_keys = ["min", "max"]
    if not all(key in global_min_max for key in required_keys):
        raise ValueError("global_min_max must contain 'min' and 'max' keys")

    global_min = global_min_max["min"]
    global_max = global_min_max["max"]

    if global_max == global_min:
        raise ValueError(
            f"Global min and max are equal ({global_min}), cannot normalize. "
            f"This indicates a constant global range which would cause "
            f"division by zero."
        )

    if unnorm:
        x_scaled = x * (global_max - global_min) + global_min
    else:
        x_scaled = (x - global_min) / (global_max - global_min)

    return x_scaled


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count the total, trainable and non-trainable parameters of a PyTorch model.

    Args:
        model: PyTorch model to analyze.

    Returns:
        Dictionary containing total, trainable and non-trainable parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def print_model_parameters(model: torch.nn.Module) -> None:
    """Print a formatted summary of model parameters.

    Args:
        model: PyTorch model to analyze.
    """
    params = count_model_parameters(model)

    logger.info("=" * 50)
    logger.info("Total parameters: %s", f"{params['total']:,}")
    logger.info("Trainable parameters: %s", f"{params['trainable']:,}")
    logger.info("Non-trainable parameters: %s", f"{params['non_trainable']:,}")
    logger.info("=" * 50)


def normalize_signals(signals: np.ndarray) -> tuple:
    """Normalize signals using different methods.

    Args:
        signals: Input signals array with shape (Channel Number, Waveform length).

    Returns:
        Tuple of (min-max normalized signals, zero-centered signals).
    """
    # Copy so caller's array is not modified
    signals_min_max = signals.copy()
    signals_min_max = min_max_norm(signals_min_max, axis=1)
    signals_zc = zero_centered(signals_min_max.copy(), axis=1)

    return signals_min_max, signals_zc


def calculate_map(
    sbp: float | np.ndarray | torch.Tensor,
    dbp: float | np.ndarray | torch.Tensor,
) -> float | np.ndarray | torch.Tensor:
    """Calculate Mean Arterial Pressure (MAP) from SBP and DBP.

    Args:
        sbp: Systolic Blood Pressure.
        dbp: Diastolic Blood Pressure.

    Returns:
        Calculated MAP values (same type as inputs).
    """
    return (sbp + (2 * dbp)) / 3


def ensure_dir_exists(directory: str) -> None:
    """Create directory if it doesn't exist.

    Args:
        directory (str): Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info("Created directory: %s", directory)


def check_file_exists(filepath: str, overwrite: bool = False) -> bool:
    """Check if a file exists and handle it according to overwrite parameter.

    Args:
        filepath (str): Path to the file to check
        overwrite (bool): If True, allow overwriting existing file. If False,
            raise error if file exists.

    Returns:
        bool: True if file exists and overwrite is True, False if file
            doesn't exist

    Raises:
        FileExistsError: If file already exists and overwrite is False
    """
    if os.path.exists(filepath):
        if not overwrite:
            raise FileExistsError(
                f"File {filepath} already exists. Set overwrite=True to overwrite it."
            )
        else:
            logger.warning("Overwriting existing file %s", filepath)
            return True
    return False


def safe_create_dataset(group: Any, key: str, value: Any) -> None:
    """Create an HDF5 dataset or group under `group` with type-aware conversion.

    Handles None (→ np.nan), dicts (subgroups), pandas DataFrames, lists/tuples
    (→ arrays), and numpy arrays. Object arrays are converted to float or string
    where possible; conversion failures are logged and re-raised.

    Args:
        group: HDF5 Group to attach the new dataset or group to.
        key: Name of the dataset or group to create.
        value: Value to store; may be dict, DataFrame, list, tuple, ndarray, or
            scalar. None is converted to np.nan.

    Note:
        Writes into `group` in place.

    Raises:
        Exception: Re-raised after logging if creation or conversion fails.
    """
    try:
        # Handle None values by converting to np.nan
        if value is None:
            value = np.nan

        # Handle nested dictionaries by creating subgroups
        if isinstance(value, dict):
            sub_group = group.create_group(key)
            for sub_key, sub_value in value.items():
                safe_create_dataset(sub_group, sub_key, sub_value)
            return

        if isinstance(value, pd.DataFrame):
            df_group = group.create_group(key)
            df_group.attrs["columns"] = value.columns.tolist()
            if not isinstance(value.index, pd.RangeIndex):
                df_group.attrs["index"] = value.index.tolist()
            for col in value.columns:
                col_data = value[col].to_numpy()
                if col_data.dtype == object:
                    col_data = np.where(pd.isna(col_data), np.nan, col_data).astype(
                        float
                    )
                df_group.create_dataset(col, data=col_data)
            return

        if isinstance(value, (list, tuple)):
            value = np.array(value)
            if value.dtype == object:
                value = np.where(pd.isna(value), np.nan, value).astype(float)

        if isinstance(value, np.ndarray) and value.dtype == object:
            logger.warning("Converting object array for %s", key)
            if all(isinstance(x, str) for x in value):
                value = np.array(value, dtype="S")
            else:
                # Try to convert to float with NA handling
                try:
                    value = np.where(pd.isna(value), np.nan, value).astype(float)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Conversion failed for {key}: {e}")
                    for i, x in enumerate(value):
                        safe_create_dataset(group, f"{key}_{i}", x)
                    return

        group.create_dataset(key, data=value)

    except Exception as e:
        extra = ""
        if isinstance(value, np.ndarray):
            extra = f" (shape={value.shape}, dtype={value.dtype})"
        logger.error(
            "Error saving %s: type=%s%s, error=%s",
            key,
            type(value).__name__,
            extra,
            e,
            exc_info=True,
        )
        raise
