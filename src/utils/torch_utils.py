"""PyTorch utility functions for common operations on torch modules.

This module provides shared utilities for PyTorch models, including
unwrapping DDP (DistributedDataParallel) or DataParallel wrapped modules
and moving batches to device. Used by trainers and evaluators when loading
or saving model state.

Functions:
    - unwrap_module: Unwrap DDP or DataParallel wrapper from a PyTorch module
    - move_batch_to_device: Move all tensors in batch dict to device

Examples:
    >>> model = torch.nn.DataParallel(MyModel())
    >>> unwrapped = unwrap_module(model)
    >>> batch = move_batch_to_device(batch, device)

See Also:
    - src.trainers.trainer: Model loading/saving
    - torch.nn.parallel.DistributedDataParallel: DDP wrapper
    - torch.nn.parallel.DataParallel: DataParallel wrapper
"""

# Standard library imports
from typing import Any
from typing import cast

# Third-party imports
import torch
import torch.nn as nn


def unwrap_module(m: nn.Module) -> nn.Module:
    """Unwrap DDP or DataParallel wrapper from a PyTorch module.

    When using PyTorch DDP (DistributedDataParallel) or DataParallel, the
    actual model is stored in a .module attribute. This function unwraps
    the model if it is wrapped, otherwise returns the module unchanged.

    Args:
        m: PyTorch module (potentially DDP- or DataParallel-wrapped).

    Returns:
        Unwrapped PyTorch module (the actual model if wrapped, else the
        module itself). Always returns an ``nn.Module`` instance for
        type-checker friendliness.

    Example:
        >>> model = torch.nn.DataParallel(MyModel())
        >>> unwrapped = unwrap_module(model)
        >>> # unwrapped is the underlying MyModel(); also works with DDP
    """
    module = m.module if hasattr(m, "module") else m
    return cast("nn.Module", module)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    non_blocking: bool = True,
) -> dict[str, Any]:
    """Move all tensors in batch dictionary to specified device.

    Args:
        batch: Dictionary potentially containing tensors
        device: Target device
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Batch dictionary with tensors moved to device
    """
    moved_batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved_batch[key] = value
    return moved_batch
