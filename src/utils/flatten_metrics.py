"""Metric aggregation utilities for MD-ViSCo.

This module provides helpers for flattening nested metric dictionaries
into scalar summaries for logging and checkpointing.

Usage:
    >>> from src.utils.flatten_metrics import flatten_nested_metrics
    >>> metrics = {'waveform': {'mae': torch.tensor(0.5), 'mse': 0.3}}
    >>> flatten_nested_metrics(metrics)
    {'waveform_mae': 0.5, 'waveform_mse': 0.3}

See Also:
    - src.trainers.waveform_reconstruction_trainer: Uses flatten_nested_metrics
    - src.utils.constants: METRIC_KEY_* constants
"""

# Standard library imports
from typing import Any

# Third-party imports
import torch


def flatten_nested_metrics(
    metrics: dict[str, Any],
    prefix: str = "",
    separator: str = "_",
) -> dict[str, float]:
    """Recursively flatten nested metric dictionaries into scalar summaries.

    Args:
        metrics: Nested dictionary of metrics (may contain tensors, dicts,
            scalars)
        prefix: Prefix for flattened keys
        separator: Separator between nested keys

    Returns:
        Flattened dictionary with scalar values

    Note:
        Keys whose value is ``None`` are omitted and do not appear in the
        output.

    Examples:
        >>> metrics = {'waveform': {'mae': torch.tensor(0.5), 'mse': 0.3}}
        >>> flatten_nested_metrics(metrics)
        {'waveform_mae': 0.5, 'waveform_mse': 0.3}
    """
    flattened: dict[str, float] = {}

    for key, value in metrics.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(
                flatten_nested_metrics(value, prefix=full_key, separator=separator)
            )
        elif torch.is_tensor(value):
            flattened[full_key] = float(value.mean().detach())
        elif isinstance(value, (float, int)):
            flattened[full_key] = float(value)
        elif value is None:
            continue

    return flattened
