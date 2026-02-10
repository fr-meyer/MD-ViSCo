"""Utility helpers for working with blood pressure tensors.

These helpers centralize our assumptions about `bp_raw` tensor layout so
that call sites do not need to repeat column indexing logic. The canonical
layout is ``[B, 3]`` with columns ordered as SBP, DBP, MAP. Some datasets
omit the MAP column; in that case `get_map` falls back to the standard MAP
formula.
"""

from __future__ import annotations

# Third-party imports
import torch
from torch import Tensor


def _validate_bp_tensor(bp_raw: Tensor, *, min_columns: int = 2) -> Tensor:
    """Ensure ``bp_raw`` is a tensor with the expected column count.

    Args:
        bp_raw: Tensor to validate (expected shape [B, C] with C >= min_columns).
        min_columns: Minimum number of columns required. Default: 2.

    Returns:
        The same tensor (for chaining).

    Raises:
        TypeError: If bp_raw is not a tensor.
        ValueError: If bp_raw is not at least 2D or has fewer than min_columns
            columns.
    """
    if not torch.is_tensor(bp_raw):
        raise TypeError(f"Expected tensor for 'bp_raw', got {type(bp_raw)!r}")
    if bp_raw.dim() < 2:
        raise ValueError(
            f"'bp_raw' tensor must be at least 2D with shape [B, C]; "
            f"received {tuple(bp_raw.shape)}."
        )
    if bp_raw.size(1) < min_columns:
        raise ValueError(
            f"'bp_raw' tensor must have at least {min_columns} columns; "
            f"received {bp_raw.size(1)}."
        )
    return bp_raw


def get_sbp(bp_raw: Tensor, *, keepdim: bool = False) -> Tensor:
    """Return systolic blood pressure column from ``bp_raw``.

    Args:
        bp_raw: Tensor of shape [B, C] with SBP in column 0.
        keepdim: If True, output shape [B, 1]; otherwise [B].

    Returns:
        Tensor of systolic values, shape [B, 1] if keepdim else [B].
    """
    tensor = _validate_bp_tensor(bp_raw, min_columns=1)
    sbp = tensor[:, 0]
    return sbp.unsqueeze(-1) if keepdim else sbp


def get_dbp(bp_raw: Tensor, *, keepdim: bool = False) -> Tensor:
    """Return diastolic blood pressure column from ``bp_raw``.

    Args:
        bp_raw: Tensor of shape [B, C] with DBP in column 1.
        keepdim: If True, output shape [B, 1]; otherwise [B].

    Returns:
        Tensor of diastolic values, shape [B, 1] if keepdim else [B].
    """
    tensor = _validate_bp_tensor(bp_raw, min_columns=2)
    dbp = tensor[:, 1]
    return dbp.unsqueeze(-1) if keepdim else dbp


def get_map(bp_raw: Tensor, *, keepdim: bool = False) -> Tensor:
    """Return mean arterial pressure from ``bp_raw`` (computed if missing).

    Args:
        bp_raw: Tensor of shape [B, C] with SBP in column 0, DBP in column 1;
            optional MAP in column 2.
        keepdim: If True, output shape [B, 1]; otherwise [B].

    Returns:
        Tensor of MAP values, shape [B, 1] if keepdim else [B]. Computed from
        SBP/DBP when MAP column is missing.
    """
    tensor = _validate_bp_tensor(bp_raw, min_columns=2)
    if tensor.size(1) > 2:
        map_values = tensor[:, 2]
    else:
        sbp = get_sbp(tensor)
        dbp = get_dbp(tensor)
        map_values = dbp + (sbp - dbp) / 3.0
    return map_values.unsqueeze(-1) if keepdim else map_values
