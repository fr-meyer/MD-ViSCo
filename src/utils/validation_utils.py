"""Validation utilities for configuration and data processing.

This module provides helpers for flattening configs (e.g. for WandB),
checking serializability, and validating tensor shapes. Used by trainers,
evaluators, and logging wrappers.

Functions:
    - flatten_config: Flatten nested config keeping only wandb-serializable
      types
    - is_wandb_serializable: Check if a value can be safely passed to wandb
      config
    - validate_tensor_shapes_match: Validate that two tensors have identical
      shapes
    - validate_tensor_dimension: Validate tensor has expected number of
      dimensions

Examples:
    >>> flat = flatten_config(cfg, sep='_')
    >>> is_wandb_serializable(0.5)  # True
    >>> validate_tensor_shapes_match(
    ...     pred, target, tensor1_name="pred", tensor2_name="target"
    ... )

See Also:
    - src.loggings.wandb_wrapper: WandB integration
    - src.trainers.waveform_reconstruction_trainer: Uses
      validate_tensor_shapes_match
"""

from typing import Any

import torch


def flatten_config(cfg: Any, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten config object, keeping only wandb-serializable types.

    This function is used by the modern WandBWrapper system for config
    flattening.

    Args:
        cfg: Configuration object to flatten (can be dict, object with
            __dict__, or OmegaConf)
        parent_key: Parent key for nested items (used in recursion)
        sep: Separator for nested keys

    Returns:
        Flattened dictionary with only wandb-serializable values
    """
    items = {}

    # Handle different config types - prioritize OmegaConf over __dict__
    if hasattr(cfg, "_content"):  # OmegaConf object
        try:
            from omegaconf import OmegaConf

            cfg_dict = OmegaConf.to_container(cfg, resolve=True)

            if isinstance(cfg_dict, dict):
                cfg_items = cfg_dict.items()
            else:
                return {}
        except Exception:
            return {}
    elif isinstance(cfg, dict):
        cfg_items = cfg.items()
    elif hasattr(cfg, "__dict__"):
        cfg_items = vars(cfg).items()
    else:
        return {}

    for k, v in cfg_items:
        # String keys for nested dict / serialization
        k_str = str(k)
        new_key = f"{parent_key}{sep}{k_str}" if parent_key else k_str

        # Skip private attributes and methods
        if k_str.startswith("_"):
            continue

        # Handle nested objects recursively
        if hasattr(v, "__dict__") or (isinstance(v, dict) and v):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            # Only keep wandb-serializable types
            if is_wandb_serializable(v):
                items[new_key] = v
            # Skip non-serializable types silently

    return items


def is_wandb_serializable(value: Any) -> bool:
    """Check if a value can be safely passed to wandb config.

    Args:
        value: Value to check for wandb serializability

    Returns:
        True if the value is wandb-serializable, False otherwise
    """
    # wandb supports these types natively
    serializable_types = (
        int,
        float,
        str,
        bool,
        type(None),  # Basic types
        list,
        tuple,  # Sequences (if they contain serializable items)
        dict,  # Dictionaries (if they contain serializable items)
    )

    if isinstance(value, serializable_types):
        # For sequences, check if all items are serializable
        if isinstance(value, (list, tuple)):
            return all(is_wandb_serializable(item) for item in value)
        # For dicts, check if all values are serializable
        elif isinstance(value, dict):
            return all(is_wandb_serializable(v) for v in value.values())
        else:
            return True

    return False


def validate_tensor_shapes_match(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    tensor1_name: str = "tensor1",
    tensor2_name: str = "tensor2",
    error_context: str = "",
) -> None:
    """Validate that two tensors have identical shapes.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        tensor1_name: Name for tensor1 in error messages
        tensor2_name: Name for tensor2 in error messages
        error_context: Additional context for error message

    Raises:
        ValueError: If shapes don't match
    """
    if tensor1.shape != tensor2.shape:
        context_msg = f" {error_context}" if error_context else ""
        raise ValueError(
            f"{tensor1_name} and {tensor2_name} must have identical "
            f"shapes.{context_msg} "
            f"Got {tensor1_name} {tuple(tensor1.shape)} vs "
            f"{tensor2_name} {tuple(tensor2.shape)}."
        )


def validate_tensor_dimension(
    tensor: torch.Tensor,
    expected_dim: int,
    tensor_name: str = "tensor",
    error_context: str = "",
) -> None:
    """Validate tensor has expected number of dimensions.

    Args:
        tensor: Tensor to check
        expected_dim: Expected number of dimensions
        tensor_name: Name for tensor in error messages
        error_context: Additional context for error message

    Raises:
        ValueError: If tensor dimension doesn't match
    """
    if tensor.dim() != expected_dim:
        context_msg = f" {error_context}" if error_context else ""
        raise ValueError(
            f"{tensor_name} must be {expected_dim}D.{context_msg} "
            f"Got shape {tensor.shape} ({tensor.dim()}D)."
        )
