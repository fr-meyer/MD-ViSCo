"""Waveform padding and trimming utilities for medical signal processing.

This module provides comprehensive utilities for resizing waveforms using
padding and trimming strategies. It supports PyTorch tensors for efficient
batch-time processing of variable-length sequences.

Key Features:
- Automatic resizing with padding or trimming as needed
- Multiple trimming strategies (center, start, end, random)
- Batch-aware processing for collate functions
- Memory-efficient tensor operations
- Comprehensive dimension support (1D, 2D, 3D)

Usage:
    # PyTorch tensors
    resized = resize_waveform_tensor(
        waveform, target_length=1280, trim_strategy='center'
    )

    # Batch processing
    batch_tensor = batch_resize_waveforms(waveforms, target_length=1280)

Padding-Aware Processing:
    The ``trim_target_padding`` helper underpins the padding-aware waveform
    pipeline that pairs ``WaveformOutputProcessor`` with
    ``WaveformReconstructionTrainer``. During validation and test stages the
    processor trims predictions and emits ``padding_metadata`` (including
    ``padding_length``). Trainers pass that metadata into
    ``trim_target_padding`` to apply the exact same symmetric trimming to
    target tensors before computing loss. This keeps training (padded tensors)
    and evaluation (trimmed tensors) consistent while preventing dimension
    mismatches when metrics and losses focus on unpadded regions.
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# TENSOR VERSIONS (for collate function and GPU processing)
# ============================================================================


def resize_waveform_tensor(
    waveform: torch.Tensor, target_length: int, trim_strategy: str = "center"
) -> torch.Tensor:
    """Resize waveform tensor to target length via padding or trimming.

    This function provides transparent preprocessing for tensors:
    - If waveform is shorter than target_length: pad with zeros (symmetric)
    - If waveform is longer than target_length: trim using specified strategy
    - If waveform equals target_length: return unchanged

    Args:
        waveform: Input waveform tensor of any length
        target_length: Desired output length
        trim_strategy: Strategy for trimming ('center', 'start', 'end',
            'random')

    Returns:
        Waveform tensor of exactly target_length

    Examples:
        >>> waveform = torch.tensor([1, 2, 3, 4, 5])  # Length 5
        >>> resized = resize_waveform_tensor(waveform, 10, 'center')
        >>> print(resized.shape)  # torch.Size([10])
        >>> print(resized)  # tensor([0, 0, 1, 2, 3, 4, 5, 0, 0, 0])
    """
    current_length = waveform.shape[-1]

    if current_length == target_length:
        return waveform  # Perfect match, no processing needed

    elif current_length < target_length:
        # PAD: Use symmetric padding
        return pad_waveform_tensor_symmetric(waveform, target_length)

    else:  # current_length > target_length
        # TRIM: Use trimming logic
        return trim_waveform_tensor(waveform, target_length, trim_strategy)


def pad_waveform_tensor_symmetric(
    waveform: torch.Tensor, target_length: int
) -> torch.Tensor:
    """Zero pad tensor waveform to target length with symmetric padding.

    For odd differences, extra padding goes to the right (future direction).
    This ensures consistent behavior and maintains temporal causality.

    Args:
        waveform: Input waveform tensor
        target_length: Desired output length

    Returns:
        Padded waveform tensor of exactly target_length

    Examples:
        >>> waveform = torch.tensor([1, 2, 3])  # Length 3
        >>> padded = pad_waveform_tensor_symmetric(waveform, 7)
        >>> print(padded)  # tensor([0, 0, 1, 2, 3, 0, 0])
    """
    current_length = waveform.shape[-1]
    if current_length >= target_length:
        return waveform

    total_pad = target_length - current_length
    pad_left = total_pad // 2
    # Handles odd differences by putting extra on right
    pad_right = total_pad - pad_left

    return torch.nn.functional.pad(
        waveform, (pad_left, pad_right), mode="constant", value=0
    )


def trim_waveform_tensor(
    waveform: torch.Tensor, target_length: int, strategy: str = "center"
) -> torch.Tensor:
    """Trim tensor waveform to target length using specified strategy.

    Args:
        waveform: Input waveform tensor
        target_length: Desired output length
        strategy: Trimming strategy
            - 'center': Extract center portion (default)
            - 'start': Extract from beginning
            - 'end': Extract from end
            - 'random': Extract random segment (for augmentation)

    Returns:
        Trimmed waveform tensor of exactly target_length

    Examples:
        >>> waveform = torch.tensor(
        ...     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ... )  # Length 10
        >>> trimmed = trim_waveform_tensor(waveform, 5, 'center')
        >>> print(trimmed)  # tensor([3, 4, 5, 6, 7])
    """
    current_length = waveform.shape[-1]

    if strategy == "center":
        start = (current_length - target_length) // 2
        end = start + target_length
    elif strategy == "start":
        start = 0
        end = target_length
    elif strategy == "end":
        start = current_length - target_length
        end = current_length
    elif strategy == "random":
        start = int(torch.randint(0, current_length - target_length + 1, (1,)).item())
        end = start + target_length
    else:
        raise ValueError(
            f"Unknown trimming strategy: {strategy}. "
            f"Valid options: 'center', 'start', 'end', 'random'"
        )

    # Handle different dimensions
    if waveform.ndim == 3:
        return waveform[:, :, start:end]
    elif waveform.ndim == 2:
        return waveform[:, start:end]
    else:
        return waveform[start:end]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compute_padding_length(original_length: int, target_length: int) -> int:
    """Compute padding applied to each end of a waveform.

    This utility calculates the symmetric padding applied to each side
    when resizing a waveform from original_length to target_length.

    Args:
        original_length: Original sequence length
        target_length: Target length after padding

    Returns:
        Padding applied to each end (int)

    Examples:
        >>> compute_padding_length(1000, 1280)  # 140 padding on each side
        140

        >>> compute_padding_length(1000, 1000)  # No padding needed
        0

        >>> compute_padding_length(1000, 999)   # No padding (trimming instead)
        0
    """
    if original_length >= target_length:
        return 0
    return max(0, (target_length - original_length) // 2)


def batch_resize_waveforms(
    waveforms: list[torch.Tensor],
    target_length: int,
    trim_strategy: str = "center",
) -> torch.Tensor:
    """Apply padding/trimming to list of waveforms before stacking.

    This function efficiently processes a batch of variable-length waveforms
    by resizing each to the target length, then stacking into a uniform tensor.

    Args:
        waveforms: List of variable-length waveform tensors
        target_length: Target length for all waveforms
        trim_strategy: Trimming strategy for sequences longer than
            target_length

    Returns:
        Stacked tensor [B, C, T] where T = target_length

    Examples:
        >>> waveforms = [
        ...     torch.randn(1, 1000),  # Length 1000
        ...     torch.randn(1, 1200),  # Length 1200
        ...     torch.randn(1, 800)    # Length 800
        ... ]
        >>> batch_tensor = batch_resize_waveforms(waveforms, 1280, 'center')
        >>> print(batch_tensor.shape)  # torch.Size([3, 1, 1280])
    """
    if not waveforms:
        raise ValueError("Cannot resize empty batch of waveforms")

    # Apply padding/trimming to each waveform
    resized = []
    for wf in waveforms:
        resized_wf = resize_waveform_tensor(wf, target_length, trim_strategy)
        resized.append(resized_wf)

    # Stack into uniform batch
    return torch.stack(resized)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================


def extract_padding_length(
    padding_metadata: dict[str, Any] | None,
    default: int = 0,
) -> int:
    """Safely extract padding length from metadata dictionary.

    Args:
        padding_metadata: Metadata dict from processor outputs
        default: Default value if metadata is None or missing key

    Returns:
        Padding length as integer
    """
    if not isinstance(padding_metadata, dict):
        return default
    return int(padding_metadata.get("padding_length", default) or default)


def compute_trimmed_length(
    original_length: int,
    padding_length: int,
) -> int:
    """Compute expected length after symmetric padding removal.

    Args:
        original_length: Original tensor length
        padding_length: Padding applied to each side

    Returns:
        Expected trimmed length

    Raises:
        ValueError: If trimmed length would be non-positive
    """
    trimmed_length = original_length - (2 * padding_length)
    if trimmed_length <= 0:
        raise ValueError(
            f"Invalid padding_length={padding_length} for "
            f"original_length={original_length}. "
            f"Trimmed length would be {trimmed_length}."
        )
    return trimmed_length


def validate_trim_strategy(strategy: str) -> None:
    """Validate trimming strategy parameter.

    Args:
        strategy: Trimming strategy to validate

    Raises:
        ValueError: If strategy is not valid
    """
    valid_strategies = ["center", "start", "end", "random"]
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid trim_strategy '{strategy}'. Valid options: {valid_strategies}"
        )


def trim_target_padding(target: torch.Tensor, padding_length: int) -> torch.Tensor:
    """Remove symmetric padding from the temporal dimension of a target tensor.

    Purpose:
        This helper participates in the padding-aware waveform pipeline. The
        ``WaveformOutputProcessor`` trims predictions (during validation /
        test) and emits ``padding_metadata`` describing how much padding was
        stripped. Trainers, such as ``WaveformReconstructionTrainer``, invoke
        ``trim_target_padding`` with that metadata to trim targets before
        computing validation / test losses so predictions and targets remain
        shape-aligned.

    Typical usage:
        - Called by ``WaveformReconstructionTrainer._step_core`` whenever
          ``stage != "train"`` and ``padding_metadata["padding_length"] >
          0``.
        - Removes ``padding_length`` samples from both the start and end of
          the temporal dimension (``T``) so targets match the already-trimmed
          predictions.
        - Ensures loss computation and downstream metrics operate on the same
          unpadded region.

    Relationship to processor:
        - Mirrors the trimming logic inside
          ``WaveformOutputProcessor._trim_padding`` so both predictions and
          targets undergo identical symmetric cropping using shared metadata.
        - ``padding_metadata`` is emitted precisely so trainers do not
          recompute padding.

    Args:
        target: Target tensor shaped ``[B, C, T]`` (or compatible
            broadcastable shape).
        padding_length: Number of samples trimmed from both the start and end
            of ``T``.

    Returns:
        torch.Tensor: Target tensor with symmetric padding removed.

    Raises:
        ValueError: If ``padding_length`` is negative, removes the entire
            sequence, or if the target tensor is ``None`` / has invalid
            dimensionality.
    """
    if target is None:
        raise ValueError("target tensor cannot be None when trimming padding.")

    if not torch.is_tensor(target):
        raise TypeError("target must be a torch.Tensor.")

    if target.ndim < 1:
        raise ValueError(
            f"target tensor must have at least one dimension, got shape "
            f"{tuple(target.shape)}"
        )

    if padding_length < 0:
        raise ValueError(f"padding_length must be non-negative, got {padding_length}")

    if padding_length == 0:
        return target

    total_trim = padding_length * 2
    sequence_length = target.shape[-1]
    if total_trim >= sequence_length:
        raise ValueError(
            "Cannot trim padding: requested removal of "
            f"{total_trim} samples from a sequence of length "
            f"{sequence_length}."
        )

    return target[..., padding_length:-padding_length]
