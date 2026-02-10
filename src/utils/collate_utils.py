"""Collation utilities for batching samples and resizing waveforms.

This module provides collate functions and batch-level normalization for
medical signal datasets. It handles padding/trimming to a target length,
waveform normalization (raw, minmax, global_minmax, etc.), and assembly
of batch dictionaries compatible with processors and trainers.

Functions:
    - normalize_waveform_batch_torch: Batch-optimized waveform normalization
    - collate_fn: Main collate function for DataLoader (configurable)
    - (Additional helpers for padding, trimming, and batch assembly)

Examples:
    >>> from src.utils.collate_utils import collate_fn, normalize_waveform_batch_torch
    >>> loader = DataLoader(dataset, collate_fn=collate_fn, ...)

See Also:
    - src.dataset.base_dataset: Sample, BaseDataset
    - src.utils.waveform_utils: batch_resize_waveforms, compute_padding_length
"""

from __future__ import annotations

import logging

# Standard library imports
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.dataset.base_dataset import Sample
    from src.preprocessors.base_preprocessor import BasePreprocessor

# Third-party imports
import torch
from omegaconf import DictConfig
from omegaconf import ListConfig

from src.core.domain import Vital
from src.utils.waveform_utils import batch_resize_waveforms
from src.utils.waveform_utils import compute_padding_length
from src.utils.waveform_utils import validate_trim_strategy

logger = logging.getLogger(__name__)


def normalize_waveform_batch_torch(
    waveforms: torch.Tensor,
    norm_type: str,
    global_params: dict[str, float] | None = None,
) -> torch.Tensor:
    """Batch-optimized normalization for waveform tensors using pure PyTorch operations.

    This function applies normalization to batches of waveforms efficiently using
    vectorized tensor operations. It translates the logic from
    `src/utils/utils_preprocessing.py` (min_max_norm, zero_centered,
    global_min_max_norm) to batch tensor operations.

    Args:
        waveforms: Input waveform tensor with shape [B, C, T]
            - B: Batch size
            - C: Number of channels (vital signs)
            - T: Time steps (waveform length)
        norm_type: Normalization type to apply
            - "raw": No normalization (identity operation)
            - "minmax": Local min-max normalization to [0, 1] per sample/channel
            - "minmax_zc": Min-max to [0, 1] then zero-center (subtract mean)
            - "global_minmax": Global min-max using dataset-wide constants
            - "local_minmax": Alias for "minmax" (per-channel local normalization)
        global_params: Dictionary with global normalization constants (required for
            "global_minmax")
            - Must contain keys: "min" (global minimum), "max" (global maximum)
            - Example: {"min": 2.34, "max": 286.58} for BP data

    Returns:
        Normalized waveform tensor with same shape [B, C, T]

    Raises:
        ValueError: If norm_type is unknown or global_params is missing for
            "global_minmax"

    Examples:
        >>> # Local min-max normalization
        >>> waveforms = torch.randn(32, 1, 1280)  # [B=32, C=1, T=1280]
        >>> normalized = normalize_waveform_batch_torch(waveforms, "minmax")
        >>> assert normalized.min() >= 0.0 and normalized.max() <= 1.0

        >>> # Min-max with zero-centering
        >>> normalized_zc = normalize_waveform_batch_torch(waveforms, "minmax_zc")
        >>> assert normalized_zc.mean().abs() < 0.1  # Approximately zero-centered

        >>> # Global normalization for ABP
        >>> abp_waveforms = torch.randn(32, 1, 1280)
        >>> global_params = {"min": 2.34, "max": 286.58}
        >>> normalized_global = normalize_waveform_batch_torch(
        ...     abp_waveforms, "global_minmax", global_params
        ... )

    Note:
        - This function uses pure PyTorch operations for GPU compatibility
        - Batch operations are ~10-20x faster than per-sample numpy operations
        - Edge cases (division by zero) are handled by replacing zero ranges with 1.0
        - The "minmax_zc" normalization is a two-step process: min-max then zero-center
          (replicates the logic from utils_preprocessing.py lines 159-162)
    """
    if norm_type == "raw":
        return waveforms

    elif norm_type == "minmax" or norm_type == "local_minmax":
        min_vals = waveforms.min(dim=-1, keepdim=True)[0]
        max_vals = waveforms.max(dim=-1, keepdim=True)[0]
        ranges = max_vals - min_vals
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
        return (waveforms - min_vals) / ranges

    elif norm_type == "minmax_zc":
        min_vals = waveforms.min(dim=-1, keepdim=True)[0]
        max_vals = waveforms.max(dim=-1, keepdim=True)[0]
        ranges = max_vals - min_vals
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
        normalized = (waveforms - min_vals) / ranges
        mean_vals = normalized.mean(dim=-1, keepdim=True)
        return normalized - mean_vals

    elif norm_type == "global_minmax":
        if global_params is None:
            raise ValueError(
                "global_minmax normalization requires global_params dict with "
                "'min' and 'max' keys"
            )

        if "min" not in global_params or "max" not in global_params:
            raise ValueError(
                f"global_params must contain 'min' and 'max' keys, got: "
                f"{list(global_params.keys())}"
            )

        global_min = global_params["min"]
        global_max = global_params["max"]
        if global_max == global_min:
            raise ValueError(
                f"Global min and max are equal ({
                    global_min
                }), cannot normalize waveforms. "
                f"This indicates a constant global range which would cause "
                f"division by zero."
            )
        return (waveforms - global_min) / (global_max - global_min)

    else:
        raise ValueError(
            f"Unknown normalization type: '{norm_type}'. "
            f"Valid options: 'raw', 'minmax', 'minmax_zc', 'global_minmax', "
            f"'local_minmax'"
        )


# ============================================================================
# Channel Mapping Functions
# ============================================================================


def build_vital_channel_mapping(
    input_preprocessing: dict[str, Any],
) -> dict[Vital, int]:
    """Build dynamic channel mapping from input_preprocessing order.

    This function extracts the channel indices from the `input_preprocessing["source"]`
    and `input_preprocessing["target"]` configuration, making it the single source of
    truth for channel ordering.

    The order in the source list IS the channel order for input channels:
    - First source vital in list → channel index 0
    - Second source vital in list → channel index 1
    - And so on...
    - Target vital → next available index (unless already in sources)

    Args:
        input_preprocessing: Preprocessing configuration dict with "source" and
            optionally "target" keys. Single source format:
                {"source": {"vital": "ppg", "norm": "minmax_zc"},
                 "target": {"vital": "abp", "norm": "global_minmax"}}
            Multi-source format:
                {"source": [{"vital": "ppg", "norm": "minmax_zc"},
                            {"vital": "ecg", "norm": "minmax_zc"}],
                 "target": {"vital": "abp", "norm": "global_minmax"}}

    Returns:
        Dict[Vital, int]: Mapping from Vital enum to channel index
            Example: {Vital.PPG: 0, Vital.ECG: 1, Vital.ABP: 2}

    Raises:
        ValueError: If input_preprocessing format is invalid or contains duplicate
            vitals
        KeyError: If vital name is invalid/unknown

    Examples:
        >>> # Single source
        >>> config = {"source": {"vital": "ppg", "norm": "minmax_zc"},
        ...           "target": {"vital": "abp", "norm": "global_minmax"}}
        >>> mapping = build_vital_channel_mapping(config)
        >>> assert mapping == {Vital.PPG: 0, Vital.ABP: 1}

        >>> # Multi-source (order matters!)
        >>> config = {"source": [
        ...     {"vital": "ppg", "norm": "minmax_zc"},
        ...     {"vital": "ecg", "norm": "minmax_zc"}
        ... ], "target": {"vital": "abp", "norm": "global_minmax"}}
        >>> mapping = build_vital_channel_mapping(config)
        >>> assert mapping == {Vital.PPG: 0, Vital.ECG: 1, Vital.ABP: 2}

    Note:
        This function is case-insensitive for vital names ("ppg", "PPG", "Ppg"
        all work). Duplicate vitals in the source list will raise a ValueError
        to prevent ambiguous mappings. If target vital is already in sources,
        it will not be added again.
    """
    if "source" not in input_preprocessing:
        raise ValueError(
            f"input_preprocessing must contain 'source' key. Got: "
            f"{list(input_preprocessing.keys())}"
        )

    source_config = input_preprocessing["source"]
    dict_like_types = (dict, DictConfig)
    list_like_types = (list, ListConfig)

    if isinstance(source_config, dict_like_types):
        source_configs = [source_config]
    elif isinstance(source_config, list_like_types):
        source_configs = source_config
    else:
        raise ValueError(
            f"input_preprocessing['source'] must be dict or list, "
            f"got {type(source_config)}"
        )

    vital_channel_mapping: dict[Vital, int] = {}
    for idx, config in enumerate(source_configs):
        if not isinstance(config, dict_like_types) or "vital" not in config:
            raise ValueError(
                f"Each source config must be a dict with 'vital' key. Got: {config}"
            )

        vital_name = config["vital"]
        try:
            vital_enum = Vital[vital_name.upper()]
        except KeyError:
            valid_vitals = [v.name for v in Vital]
            raise ValueError(
                f"Invalid vital name '{vital_name}'. Valid options: {valid_vitals}"
            ) from None

        # Prevent ambiguous channel mappings
        if vital_enum in vital_channel_mapping:
            previous_idx = vital_channel_mapping[vital_enum]
            raise ValueError(
                f"Duplicate vital '{vital_name}' found in "
                f"input_preprocessing['source'] at positions {previous_idx} and "
                f"{idx}. Each vital must appear only once."
            )

        vital_channel_mapping[vital_enum] = idx

    if "target" in input_preprocessing:
        target_config = input_preprocessing["target"]
        if isinstance(target_config, dict_like_types):
            target_configs = [target_config]
        elif isinstance(target_config, list_like_types):
            target_configs = target_config
        else:
            raise ValueError(
                f"input_preprocessing['target'] must be dict or list, "
                f"got {type(target_config)}"
            )
        for target_cfg in target_configs:
            if not isinstance(target_cfg, dict_like_types) or "vital" not in target_cfg:
                raise ValueError(
                    f"Each target config must be a dict with 'vital' key. "
                    f"Got: {target_cfg}"
                )

            target_name = target_cfg["vital"]
            try:
                target_enum = Vital[target_name.upper()]
                if target_enum not in vital_channel_mapping:
                    next_idx = len(vital_channel_mapping)
                    vital_channel_mapping[target_enum] = next_idx
            except KeyError:
                valid_vitals = [v.name for v in Vital]
                raise ValueError(
                    f"Invalid target vital name '{target_name}'. "
                    f"Valid options: {valid_vitals}"
                ) from None

    logger.debug(
        f"Built vital channel mapping from input_preprocessing: {vital_channel_mapping}"
    )
    return vital_channel_mapping


def get_channel_index(vital: Vital, vital_channel_mapping: dict[Vital, int]) -> int:
    """Safely lookup channel index for a given vital.

    This is the recommended accessor for channel indices to ensure consistent
    error handling and helpful error messages.

    Args:
        vital: The vital sign to lookup
        vital_channel_mapping: Mapping from Vital to channel index

    Returns:
        int: Channel index for the vital

    Raises:
        KeyError: If vital not found in mapping (with helpful error message)

    Examples:
        >>> mapping = {Vital.PPG: 0, Vital.ECG: 1}
        >>> get_channel_index(Vital.PPG, mapping)
        0
        >>> get_channel_index(Vital.ABP, mapping)  # doctest: +SKIP
        KeyError: Vital.ABP not found in channel mapping. Available vitals:
            [Vital.PPG, Vital.ECG]
    """
    if vital not in vital_channel_mapping:
        available = list(vital_channel_mapping.keys())
        raise KeyError(
            f"{vital} not found in channel mapping. Available vitals: {available}"
        )
    return vital_channel_mapping[vital]


def normalize_bp_batch_torch(
    bp: torch.Tensor, norm_type: str, global_params: dict[str, float] | None = None
) -> torch.Tensor:
    """Batch-optimized normalization for blood pressure tensors.

    This function applies normalization to batches of BP values (SBP, DBP, MAP)
    efficiently using vectorized tensor operations.

    Args:
        bp: Input BP tensor with shape [B, 3]
            - B: Batch size
            - 3: BP components (SBP, DBP, MAP)
        norm_type: Normalization type to apply
            - "raw": No normalization (identity operation)
            - "minmax": Local min-max normalization to [0, 1] per sample
            - "minmax_zc": Min-max to [0, 1] then zero-center (subtract mean) per sample
            - "global_minmax": Global min-max using dataset-wide BP constants
        global_params: Dictionary with global BP normalization constants (required for
            "global_minmax")
            - Must contain keys: "sbp_max" (max SBP), "dbp_min" (min DBP)
            - Example: {"sbp_max": 286.58, "dbp_min": 2.34}

    Returns:
        Normalized BP tensor with same shape [B, 3]

    Raises:
        ValueError: If norm_type is unknown or global_params is missing for
            "global_minmax"

    Examples:
        >>> # Local min-max normalization per sample
        >>> bp = torch.tensor([[120.0, 80.0, 93.3], [140.0, 90.0, 106.7]])  # [B=2, 3]
        >>> normalized = normalize_bp_batch_torch(bp, "minmax")
        >>> assert normalized.min() >= 0.0 and normalized.max() <= 1.0

        >>> # Min-max with zero-centering per sample
        >>> normalized_zc = normalize_bp_batch_torch(bp, "minmax_zc")
        >>> assert normalized_zc.mean(dim=1).abs().max() < 0.1  # Approximately
        ... # zero-centered per sample

        >>> # Global normalization using dataset constants
        >>> global_params = {"sbp_max": 286.58, "dbp_min": 2.34}
        >>> normalized_global = normalize_bp_batch_torch(
        ...     bp, "global_minmax", global_params
        ... )

    Note:
        - BP values are typically in range [2.34, 286.58] mmHg (from PulseDB
          dataset)
        - Global normalization uses the same min/max for all three components
          (SBP, DBP, MAP)
        - This ensures consistent scaling across all BP measurements
        - Per-sample normalization preserves individual sample characteristics
    """
    if norm_type == "raw":
        return bp

    elif norm_type == "minmax":
        min_vals = bp.min(dim=-1, keepdim=True)[0]
        max_vals = bp.max(dim=-1, keepdim=True)[0]
        ranges = max_vals - min_vals
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
        return (bp - min_vals) / ranges

    elif norm_type == "minmax_zc":
        min_vals = bp.min(dim=-1, keepdim=True)[0]
        max_vals = bp.max(dim=-1, keepdim=True)[0]
        ranges = max_vals - min_vals
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
        normalized = (bp - min_vals) / ranges
        mean_vals = normalized.mean(dim=-1, keepdim=True)
        return normalized - mean_vals

    elif norm_type == "global_minmax":
        if global_params is None:
            raise ValueError(
                "global_minmax normalization requires global_params dict with "
                "'sbp_max' and 'dbp_min' keys"
            )

        if "sbp_max" not in global_params or "dbp_min" not in global_params:
            raise ValueError(
                f"global_params must contain 'sbp_max' and 'dbp_min' keys, got: "
                f"{list(global_params.keys())}"
            )

        global_min = global_params["dbp_min"]
        global_max = global_params["sbp_max"]
        if global_max == global_min:
            raise ValueError(
                f"Global min and max are equal ({global_min}), cannot normalize BP. "
                f"This indicates a constant global range which would cause "
                f"division by zero."
            )
        return (bp - global_min) / (global_max - global_min)

    else:
        raise ValueError(
            f"Unknown normalization type: '{norm_type}'. "
            f"Valid options: 'raw', 'minmax', 'minmax_zc', 'global_minmax'"
        )


def compute_max_source_channels(directions) -> int:
    """Compute maximum number of source channels from direction definitions.

    This function automatically determines the maximum number of source vitals
    across all directions, eliminating the need for manual configuration.

    Uses the canonical `direction.source` property which always returns List[Vital].

    Args:
        directions: Collection of Direction objects (Directions instance or list).
            If empty/None, returns safe default of 1 to allow inference/trainer setups
            without directions.

    Returns:
        Maximum number of source vitals across all directions, or 1 if directions
        is empty/None (safe default for inference scenarios)

    Raises:
        ValueError: If directions contains invalid Direction objects or no valid
            source vitals found (only when directions is non-empty)

    Examples:
        Single source: [Direction(source=PPG, target=ABP)] → returns 1
        Multi-source: [Direction(source=[PPG, ECG], target=ABP)] → returns 2
        Mixed: [Direction(source=PPG, target=ABP),
                 Direction(source=[PPG, ECG], target=ABP)]
            → returns 2
        Empty/None: [] or None → returns 1 (safe default)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Handle empty/None directions with safe default for inference scenarios
    if not directions:
        logger.debug(
            "directions is empty/None, returning safe default max_source_channels=1 "
            "for inference/trainer setups without directions"
        )
        return 1

    # Handle non-sized iterables (e.g., generators, custom iterators)
    try:
        logger.debug(f"Computing max_source_channels from {len(directions)} directions")
    except TypeError:
        logger.debug("Computing max_source_channels from directions iterable")

    max_sources = 0

    for i, direction in enumerate(directions):
        try:
            sv = direction.source
        except AttributeError:
            raise ValueError(
                "Direction object missing canonical 'source' property. "
                "Use Direction.source instead of legacy 'src' or 'source_vitals'."
            ) from None
        num_sources = len(sv)

        if num_sources == 0:
            logger.warning(
                f"Direction {i}: Direction object missing source property, skipping"
            )
            continue

        logger.debug(f"Direction {i}: {num_sources} source(s)")
        max_sources = max(max_sources, num_sources)

    # Post-validation
    if max_sources == 0:
        raise ValueError(
            "No valid source vitals found in directions. Check direction definitions."
        )

    # INFO-level logging removed to avoid duplicate logs; trainer is
    # responsible for INFO logs
    return max_sources


# ============================================================================
# Format Detection and Lazy Normalization Helpers
# ============================================================================


def _build_valid_sample_mask(
    batch: list[Sample], vital_configs: list[dict[str, str]]
) -> list[bool]:
    """Build mask of samples that have all required vitals.

    Args:
        batch: List of Sample objects
        vital_configs: List of vital configurations to check

    Returns:
        List[bool]: Mask indicating which samples have all required vitals
    """
    valid_mask = []

    for sample in batch:
        has_all_vitals = True
        for config in vital_configs:
            vital_str = config["vital"].upper()
            try:
                vital = Vital[vital_str]
                if not sample.has_vital(vital):
                    has_all_vitals = False
                    break
            except KeyError:
                has_all_vitals = False
                break

        valid_mask.append(has_all_vitals)

    return valid_mask


def _extract_and_normalize_vital_batch(
    batch: list[Sample],
    vital_config: dict[str, str],
    dataset_norm_params: dict[str, float] | None = None,
    role: str = "source",
    target_length: int | None = None,
    trim_strategy: str = "center",
    valid_sample_mask: list[bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract raw vital data from batch and apply lazy normalization with
    optional padding/trimming.

    Preserves batch size by extracting all samples and using zero/NaN placeholders for
    invalid samples, returning a mask indicating valid samples.

    Args:
        batch: List of Sample objects
        vital_config: Dict with "vital" and "norm" keys (e.g., {"vital": "ppg",
            "norm": "minmax_zc"})
        dataset_norm_params: Optional dict with normalization constants from
            dataset.get_normalization_params()
        role: Role description for logging ("source" or "target")
        target_length: Optional target length for padding/trimming (None = no resizing)
        trim_strategy: Strategy for trimming sequences longer than target_length
        valid_sample_mask: Optional pre-computed mask indicating valid samples.
            If None, builds mask based on sample.has_vital(vital).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Normalized vital batch with shape [B, C, T] for waveforms or [B, 3] for BP
            - Valid sample mask [B] indicating which samples have valid vital data

    Raises:
        ValueError: If vital is invalid, normalization fails, or no valid samples remain

    Examples:
        >>> # Extract and normalize PPG waveforms
        >>> batch = [sample1, sample2, sample3]  # List of Sample objects
        >>> config = {"vital": "ppg", "norm": "minmax_zc"}
        >>> normalized_ppg, mask = _extract_and_normalize_vital_batch(batch, config)
        >>> print(normalized_ppg.shape)  # [3, 1, 1280]
        >>> print(mask.shape)  # [3]

        >>> # Extract and normalize BP with global parameters
        >>> config = {"vital": "bp", "norm": "global_minmax"}
        >>> norm_params = {"sbp_max": 286.58, "dbp_min": 2.34}
        >>> normalized_bp, mask = _extract_and_normalize_vital_batch(
        ...     batch, config, norm_params
        ... )
        >>> print(normalized_bp.shape)  # [3, 3] (SBP, DBP, MAP)
    """
    if not batch:
        raise ValueError(f"Cannot extract {role} vital from empty batch")

    if "vital" not in vital_config or not vital_config["vital"]:
        raise ValueError(
            f"Missing 'vital' key in {role} vital configuration: {vital_config}"
        )
    if "norm" not in vital_config or not vital_config["norm"]:
        raise ValueError(
            f"Missing 'norm' key in {role} vital configuration: {vital_config}"
        )

    vital_str = vital_config["vital"].upper()
    norm_type = vital_config["norm"]

    try:
        vital = Vital[vital_str]
    except KeyError:
        raise ValueError(
            f"Invalid vital '{vital_str}'. Valid options: {[v.value for v in Vital]}"
        ) from None

    if valid_sample_mask is None:
        valid_sample_mask = []
        for sample in batch:
            valid_sample_mask.append(sample.has_vital(vital))

    valid_mask_tensor = torch.tensor(valid_sample_mask, dtype=torch.bool)

    # Determine reference shape from first valid sample (needed for placeholders)
    reference_shape = None
    valid_count = 0
    for sample, is_valid in zip(batch, valid_sample_mask, strict=True):
        if is_valid:
            raw_data = sample.get_vital(vital)
            if raw_data is not None:
                reference_shape = raw_data.shape
                valid_count += 1
                break  # Found reference shape, can proceed

    if valid_count == 0:
        raise ValueError(
            f"No valid samples with {vital.value} vital found in batch of {
                len(batch)
            } samples"
        )

    if vital == Vital.BP:
        placeholder_shape = (3,)
    else:
        # For waveforms, use reference shape if available
        if reference_shape is None:
            raise ValueError(
                f"Cannot determine waveform shape: no valid {vital.value} samples found"
            )
        placeholder_shape = reference_shape

    all_waveforms = []
    for _, (sample, is_valid) in enumerate(zip(batch, valid_sample_mask, strict=True)):
        if is_valid:
            raw_data = sample.get_vital(vital)
            if raw_data is not None:
                all_waveforms.append(raw_data)
            else:
                logger.warning(
                    f"Sample {sample.sample_index} has None {
                        vital.value
                    } data, but was "
                    f"marked as valid, using zero placeholder"
                )
                # Use zero placeholder for None data even if marked valid
                placeholder = torch.zeros(placeholder_shape, dtype=torch.float32)
                all_waveforms.append(placeholder)
        else:
            logger.debug(
                f"Sample {sample.sample_index} missing {
                    vital.value
                } vital, using zero placeholder"
            )
            placeholder = torch.zeros(placeholder_shape, dtype=torch.float32)
            all_waveforms.append(placeholder)

    # Apply padding/trimming and stack raw waveforms (now full batch size)
    if target_length is not None and vital != Vital.BP:
        raw_batch = batch_resize_waveforms(all_waveforms, target_length, trim_strategy)
    else:
        # No padding/trimming needed (BP data or target_length=None)
        raw_batch = torch.stack(
            all_waveforms
        )  # [B, C, T] for waveforms or [B, 3] for BP

    # Apply normalization based on vital type
    if vital == Vital.BP:
        # BP data: use BP-specific normalization
        normalized_batch = normalize_bp_batch_torch(
            raw_batch, norm_type, dataset_norm_params
        )
    else:
        # Waveform data: use waveform normalization
        # Special handling for ABP global_minmax: map dataset keys to waveform keys
        if (
            vital == Vital.ABP
            and norm_type == "global_minmax"
            and dataset_norm_params is not None
        ):
            # Map dataset normalization params to waveform normalization keys
            waveform_norm_params = {
                "min": dataset_norm_params.get("dbp_min"),
                "max": dataset_norm_params.get("sbp_max"),
            }

            if (
                waveform_norm_params["min"] is None
                or waveform_norm_params["max"] is None
            ):
                raise ValueError(
                    f"ABP global_minmax normalization requires 'dbp_min' and "
                    f"'sbp_max' in dataset_norm_params. Got: "
                    f"{
                        list(dataset_norm_params.keys())
                        if dataset_norm_params
                        else 'None'
                    }"
                )

            normalized_batch = normalize_waveform_batch_torch(
                raw_batch,
                norm_type,
                cast("dict[str, float]", waveform_norm_params),
            )
        else:
            # Standard waveform normalization
            normalized_batch = normalize_waveform_batch_torch(
                raw_batch, norm_type, dataset_norm_params
            )

    return normalized_batch, valid_mask_tensor


def _extract_and_normalize_multi_source_batch(
    batch: list[Sample],
    source_configs: list[dict[str, str]],
    dataset_norm_params: dict[str, float] | None = None,
    target_length: int | None = None,
    trim_strategy: str = "center",
    valid_sample_mask: list[bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and normalize multiple source vitals, then concatenate with
    optional padding/trimming.

    Preserves batch size and returns a combined mask indicating samples valid
    for all sources.

    Args:
        batch: List of Sample objects
        source_configs: List of dicts, each with "vital" and "norm" keys
            Example: [{"vital": "ppg", "norm": "minmax_zc"},
                       {"vital": "ecg", "norm": "minmax_zc"}]
        dataset_norm_params: Optional dict with normalization constants
        target_length: Optional target length for padding/trimming (None = no resizing)
        trim_strategy: Strategy for trimming sequences longer than target_length
        valid_sample_mask: Optional pre-computed mask indicating valid samples.
            If None, each source vital builds its own mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Concatenated normalized vitals [B, C1+C2+..., T]
            - Combined valid sample mask [B] (True if sample is valid for ALL sources)

    Examples:
        >>> # Extract PPG + ECG
        >>> configs = [
        ...     {"vital": "ppg", "norm": "minmax_zc"},
        ...     {"vital": "ecg", "norm": "minmax_zc"}
        ... ]
        >>> multi_source, mask = _extract_and_normalize_multi_source_batch(
        ...     batch, configs
        ... )
        >>> print(multi_source.shape)  # [32, 2, 1280] - 2 channels (PPG + ECG)
        >>> print(mask.shape)  # [32]
    """
    if not source_configs:
        raise ValueError("source_configs cannot be empty")

    normalized_sources = []
    source_masks = []
    for i, config in enumerate(source_configs):
        normalized, mask = _extract_and_normalize_vital_batch(
            batch,
            config,
            dataset_norm_params,
            role=f"source_{i}",
            target_length=target_length,
            trim_strategy=trim_strategy,
            valid_sample_mask=valid_sample_mask,
        )
        normalized_sources.append(normalized)
        source_masks.append(mask)

    concatenated = torch.cat(normalized_sources, dim=1)

    # Combine masks: sample is valid only if valid for ALL sources
    combined_mask = torch.stack(source_masks, dim=0).all(dim=0)  # [B]

    logger.info(
        f"Concatenated {len(source_configs)} source vitals: "
        f"{[c['vital'] for c in source_configs]} -> shape {concatenated.shape}"
    )

    return concatenated, combined_mask


def _extract_demographic_fields_from_batch(
    batch: list[Sample], force_keys: bool = False
) -> dict[str, torch.Tensor]:
    """Extract individual demographic fields from batch.

    This helper function centralizes the logic for extracting demographic fields from
    Sample objects. It extracts individual fields only (no broadcasting).

    Args:
        batch: List of Sample objects
        force_keys: If True, always return demographic keys with NaN tensors even if no
            demographics exist. This ensures stable batch schema.

    Returns:
        Dict[str, torch.Tensor]:
            - "age_raw": [B, 1] tensor with age values (NaN if missing)
            - "gender_raw": [B, 1] tensor with gender values (NaN if missing)
            - "height_raw": [B, 1] tensor with height values (NaN if missing)
            - "weight_raw": [B, 1] tensor with weight values (NaN if missing)
            - "bmi_raw": [B, 1] tensor with BMI values (NaN if missing)

    Examples:
        >>> # Extract individual fields
        >>> demo_dict = _extract_demographic_fields_from_batch(batch)
        >>> print(demo_dict.keys())
        dict_keys(['age_raw', 'gender_raw', 'height_raw', 'weight_raw', 'bmi_raw'])

    Note:
        - Missing individual fields are filled with NaN placeholders
        - All tensors are converted to float32 for consistency
        - Models handle their own broadcasting and feature engineering
    """
    has_any_demographics = any(
        s.age_raw is not None
        or s.gender_raw is not None
        or s.height_raw is not None
        or s.weight_raw is not None
        or s.bmi_raw is not None
        for s in batch
    )

    if not has_any_demographics and not force_keys:
        return {}

    # Detect device and dtype from first available demographic tensor
    reference_tensor = None
    for s in batch:
        for field in [s.age_raw, s.gender_raw, s.height_raw, s.weight_raw, s.bmi_raw]:
            if field is not None:
                reference_tensor = field
                break
        if reference_tensor is not None:
            break

    if reference_tensor is not None:
        device = reference_tensor.device
        dtype = reference_tensor.dtype
    else:
        # Fallback when no demographics exist (force_keys=True case)
        device = torch.device("cpu")
        dtype = torch.float32

    def make_nan_placeholder():
        """Create NaN placeholder with consistent device/dtype."""
        return torch.tensor([float("nan")], device=device, dtype=dtype)

    # Extract individual demographic fields with NaN handling and atleast_1d protection
    age_raw = torch.stack(
        [
            torch.atleast_1d(
                s.age_raw if s.age_raw is not None else make_nan_placeholder()
            )
            for s in batch
        ]
    ).to(torch.float32)  # [B, 1]

    gender_raw = torch.stack(
        [
            torch.atleast_1d(
                s.gender_raw if s.gender_raw is not None else make_nan_placeholder()
            )
            for s in batch
        ]
    ).to(torch.float32)  # [B, 1]

    height_raw = torch.stack(
        [
            torch.atleast_1d(
                s.height_raw if s.height_raw is not None else make_nan_placeholder()
            )
            for s in batch
        ]
    ).to(torch.float32)  # [B, 1]

    weight_raw = torch.stack(
        [
            torch.atleast_1d(
                s.weight_raw if s.weight_raw is not None else make_nan_placeholder()
            )
            for s in batch
        ]
    ).to(torch.float32)  # [B, 1]

    bmi_raw = torch.stack(
        [
            torch.atleast_1d(
                s.bmi_raw if s.bmi_raw is not None else make_nan_placeholder()
            )
            for s in batch
        ]
    ).to(torch.float32)  # [B, 1]

    result = {
        "age_raw": age_raw,
        "gender_raw": gender_raw,
        "height_raw": height_raw,
        "weight_raw": weight_raw,
        "bmi_raw": bmi_raw,
    }

    return result


# ============================================================================
# Direction-Aware Collate Functions
# ============================================================================


def _populate_source_channels_collate(
    direction,  # Direction type from src.core.domain
    vital_channel_mapping: dict[Vital, int],
    src_idxs: torch.Tensor,
    src_mask: torch.Tensor,
    sample_idx: int,
    max_source_channels: int,
) -> None:
    """Shared logic for populating source channel indices and masks.

    This function centralizes the source channel handling logic that was previously
    duplicated in the trainer. It's designed to work efficiently in the collate context.
    Mappings must be produced by `build_vital_channel_mapping()`.

    Args:
        direction: Direction object containing source and target information
        vital_channel_mapping: Dict mapping Vital enum to channel indices
        src_idxs: Tensor to populate with source channel indices
            [B, max_source_channels]
        src_mask: Tensor to populate with source channel masks
            [B, max_source_channels]
        sample_idx: Sample index in batch (0 for single-direction, i for
            multi-directional)
        max_source_channels: Maximum number of source channels

    Note:
        The vital_channel_mapping comes from build_vital_channel_mapping()
        Uses the canonical direction.source property which always returns List[Vital].
    """
    # Use canonical source property (always returns List[Vital])
    for j, vital in enumerate(direction.source):
        if j < max_source_channels:
            src_chan = get_channel_index(vital, vital_channel_mapping)
            src_idxs[sample_idx, j] = src_chan
            src_mask[sample_idx, j] = True


def _add_single_direction_metadata(
    direction,  # Direction type from src.core.domain
    source_channel_mapping: dict[Vital, int],
    target_channel_mapping: dict[Vital, int],
    batch_dict: dict[str, Any],
    max_source_channels: int,
) -> dict[str, Any]:
    """Add single-direction metadata to batch.

    Args:
        direction: Single direction for the entire batch
        source_channel_mapping: Mapping for sources
        target_channel_mapping: Mapping for targets
        batch_dict: Existing batch with preprocessing data
        max_source_channels: Maximum number of source channels

    Returns:
        Updated batch_dict with direction metadata
    """
    waveform = batch_dict["x"]
    batch_size = waveform.shape[0]

    src_idxs = torch.zeros(1, max_source_channels, dtype=torch.long)
    src_mask = torch.zeros(1, max_source_channels, dtype=torch.bool)

    _populate_source_channels_collate(
        direction, source_channel_mapping, src_idxs, src_mask, 0, max_source_channels
    )

    tgt_chan = get_channel_index(direction.target, target_channel_mapping)

    return {
        "src_idxs": src_idxs.repeat(
            batch_size, 1
        ).contiguous(),  # [B, max_source_channels] - No aliasing risk
        "src_mask": src_mask.repeat(
            batch_size, 1
        ).contiguous(),  # [B, max_source_channels] - No aliasing risk
        "tgt_idxs": torch.full(
            (batch_size,), tgt_chan, dtype=torch.long
        ),  # [B] - Fresh tensor, no aliasing
        "direction": direction,
    }


def _add_multi_direction_metadata(
    directions,  # Directions type from src.core.direction
    source_channel_mapping: dict[Vital, int],
    target_channel_mapping: dict[Vital, int],
    batch_dict: dict[str, Any],
    max_source_channels: int,
) -> dict[str, Any]:
    """Add multi-directional metadata to batch.

    Args:
        directions: Collection of available directions
        source_channel_mapping: Mapping for sources
        target_channel_mapping: Mapping for targets
        batch_dict: Existing batch with preprocessing data
        max_source_channels: Maximum number of source channels

    Returns:
        Updated batch_dict with multi-directional metadata
    """
    waveform = batch_dict["x"]
    batch_size = waveform.shape[0]

    # Randomly select directions for each sample
    available_directions = list(directions)
    selected_directions = [
        available_directions[
            int(torch.randint(0, len(available_directions), (1,)).item())
        ]
        for _ in range(batch_size)
    ]

    src_idxs = torch.zeros(batch_size, max_source_channels, dtype=torch.long)
    src_mask = torch.zeros(batch_size, max_source_channels, dtype=torch.bool)
    tgt_idxs = torch.zeros(batch_size, dtype=torch.long)

    for i, direction in enumerate(selected_directions):
        _populate_source_channels_collate(
            direction,
            source_channel_mapping,
            src_idxs,
            src_mask,
            i,
            max_source_channels,
        )
        tgt_idxs[i] = get_channel_index(direction.target, target_channel_mapping)

    target_channels = [
        get_channel_index(d.target, target_channel_mapping) for d in selected_directions
    ]
    domain_shift_target = torch.nn.functional.one_hot(
        torch.tensor(target_channels), len(target_channel_mapping)
    ).to(torch.float32)

    return {
        "src_idxs": src_idxs,  # [B, max_source_channels]
        "src_mask": src_mask,  # [B, max_source_channels]
        "tgt_idxs": tgt_idxs,  # [B]
        "domain_shift_target": domain_shift_target,
        "directions": selected_directions,
        "mixed_batch": True,
    }


def create_direction_aware_collate_fn(
    input_preprocessing: dict[str, Any],
    directions: Any,
    direction_mode: str | None = None,
    max_source_channels: int | None = None,
    dataset_norm_params: dict[str, float] | None = None,
    include_list: list[str] | None = None,
    window_size: int | None = None,
    trim_strategy: str = "center",
    source_channel_mapping: dict[Vital, int] | None = None,
    target_channel_mapping: dict[Vital, int] | None = None,
    demographics_text_encoder: BasePreprocessor | None = None,
) -> Callable:
    """Create direction-aware collate function with lazy normalization.

    The returned collate function supports:
    - Single-source inputs (e.g., PPG → ABP)
    - Multi-source inputs (e.g., PPG+ECG → ABP)
    - Single-target outputs
    - Multi-target outputs
    - Lazy normalization at batch time
    - Optional WCL raw data inclusion
    - Optional demographics inclusion for models like PatchTST

    Args:
        input_preprocessing: Dict with source and target configuration:
            Single source, single target:
                {"source": {"vital": "ppg", "norm": "minmax_zc"},
                 "target": {"vital": "abp", "norm": "global_minmax"}}

            Multi-source, single target:
                {"source": [{"vital": "ppg", "norm": "minmax_zc"},
                            {"vital": "ecg", "norm": "minmax_zc"}],
                 "target": {"vital": "abp", "norm": "global_minmax"}}

            Multi-target:
                {"source": {"vital": "ppg", "norm": "minmax_zc"},
                 "target": [{"vital": "abp", "norm": "global_minmax"},
                            {"vital": "ecg", "norm": "minmax_zc"}]}

        directions: Collection of available directions
        direction_mode: Training mode (single or multi-directional)
        max_source_channels: Maximum number of source channels. If None (default),
            automatically computed from directions. Can be explicitly set to override
            auto-computation.
        dataset_norm_params: Dict with normalization constants
            (e.g., {"sbp_max": 286.58, "dbp_min": 2.34})
        include_list: List of raw field names to include in batch. Available keys:
            - 'bp_raw': Raw blood pressure values [B, 3]
            - 'age_raw', 'gender_raw', 'height_raw', 'weight_raw', 'bmi_raw': Individual
              demographics [B, 1]
            Default: None (no additional data)

            Note: Collate function returns raw fields as-is. Models handle their own
            broadcasting and feature engineering (industry standard pattern).
        window_size: Target window size for padding/trimming. If None, uses raw sequence
            lengths (variable-length batches). If provided, all sequences are resized to
            this length.
        trim_strategy: Strategy for trimming sequences longer than window_size. Options:
            'center' (default), 'start', 'end', 'random'.
        source_channel_mapping: Optional pre-computed mapping from source
            Vital enums to channel indices. If None (default), will be computed
            from input_preprocessing. Primarily for performance optimization when
            creating multiple collate functions with identical preprocessing
            configurations.
        target_channel_mapping: Optional pre-computed mapping from target
            Vital enums to channel indices. If None (default), will be computed
            from input_preprocessing. Primarily for performance optimization when
            creating multiple collate functions with identical preprocessing
            configurations.
        demographics_text_encoder: Optional demographics text encoder for
            generating text tokens from demographics fields. If provided, will
            generate batch_dict['text'] with input_ids and attention_mask after
            stacking demographics.

    Returns:
        Collate function for DataLoader. The returned collate function produces
        batch_dict with:
            - "x": Normalized source waveform(s) [B, C_sources, T] (resized to
              target_length if window_size provided)
            - "y": Normalized target waveform(s) [B, C_targets, T] - multi-channel
              when target is list (resized to target_length if window_size
              provided)
            - If include_list specified: Raw fields as-is (e.g., "bp_raw" [B, 3],
              "age_raw" [B, 1], etc.)
            - Direction metadata: "src_idxs", "src_mask", "tgt_idxs", etc.
            - Processor metadata: "padding_length" (computed from actual padding
              applied at batch time)

    Raises:
        ValueError: If input_preprocessing format is invalid

    Examples:
        Note: max_source_channels is auto-computed from directions and can be omitted

        >>> # Single source, single target PPG → ABP
        >>> collate_fn = create_direction_aware_collate_fn(
            ...     input_preprocessing={
            ...         "source": {"vital": "ppg", "norm": "minmax_zc"},
            ...         "target": {"vital": "abp", "norm": "global_minmax"}
            ...     },
            ...     directions=directions,
            ...     direction_mode="single",
            ...     dataset_norm_params={"sbp_max": 286.58, "dbp_min": 2.34}
            ... )

        >>> # Multi-source, single target PPG+ECG → ABP
        >>> collate_fn = create_direction_aware_collate_fn(
        ...     input_preprocessing={
        ...         "source": [
        ...             {"vital": "ppg", "norm": "minmax_zc"},
        ...             {"vital": "ecg", "norm": "minmax_zc"}
        ...         ],
        ...         "target": {"vital": "abp", "norm": "global_minmax"}
        ...     },
        ...     directions=directions,
        ...     direction_mode="single"
        ... )

        >>> # With window size for padding/trimming
        >>> collate_fn = create_direction_aware_collate_fn(
        ...     input_preprocessing={
        ...         "source": {"vital": "ppg", "norm": "minmax_zc"},
        ...         "target": {"vital": "abp", "norm": "global_minmax"}
        ...     },
        ...     directions=directions,
        ...     direction_mode="single",
        ...     window_size=1280,
        ...     trim_strategy='center'
        ... )

        >>> # Variable-length batches (no padding/trimming)
        >>> collate_fn = create_direction_aware_collate_fn(
        ...     input_preprocessing={
        ...         "source": {"vital": "ppg", "norm": "minmax_zc"},
        ...         "target": {"vital": "abp", "norm": "global_minmax"}
        ...     },
        ...     directions=directions,
        ...     direction_mode="single",
        ...     window_size=None  # Keep raw lengths, no padding/trimming
        ... )
    """
    if "source" not in input_preprocessing or "target" not in input_preprocessing:
        raise ValueError(
            f"Invalid input_preprocessing format. Expected NEW format with "
            f"'source' and 'target' keys.\n"
            f"Got: {input_preprocessing}\n\n"
            f"NEW format examples:\n"
            f"  Single source: {{'source': {{'vital': 'ppg', 'norm': 'minmax_zc'}}, "
            f"'target': {{'vital': 'abp', 'norm': 'global_minmax'}}}}\n"
            f"  Multi-source: {{'source': [{{'vital': 'ppg', 'norm': 'minmax_zc'}}, "
            f"{{'vital': 'ecg', 'norm': 'minmax_zc'}}], 'target': {{'vital': 'abp', "
            f"'norm': 'global_minmax'}}}}\n\n"
            f"Note: OLD format is no longer supported. Please migrate to NEW format."
        )

    source_config = input_preprocessing["source"]
    target_config = input_preprocessing["target"]

    is_multi_source = isinstance(source_config, list)
    is_multi_target = isinstance(target_config, list)

    include_list = include_list or []

    if include_list:
        logger.debug(
            "Collate returns raw fields; models handle broadcasting/processing"
        )

        allowed_keys = {
            "bp_raw",
            "age_raw",
            "gender_raw",
            "height_raw",
            "weight_raw",
            "bmi_raw",
        }
        invalid_keys = [key for key in include_list if key not in allowed_keys]
        if invalid_keys:
            raise ValueError(
                f"Invalid include_list keys: {invalid_keys}. "
                f"Allowed keys: {sorted(allowed_keys)}"
            )

    direction_mode = getattr(direction_mode, "value", direction_mode)
    mode_str = str(direction_mode).lower() if direction_mode is not None else None

    if max_source_channels is None:
        max_source_channels = compute_max_source_channels(directions)
    else:
        logger.debug(
            f"Using explicitly provided max_source_channels: {max_source_channels}"
        )

    if max_source_channels <= 0:
        raise ValueError(
            f"max_source_channels must be positive, got {max_source_channels}"
        )

    # ============================================================================
    # BUILD SEPARATE CHANNEL MAPPINGS FOR SOURCES AND TARGETS
    # ============================================================================
    if source_channel_mapping is not None:
        _source_channel_mapping = source_channel_mapping
    else:
        _source_channel_mapping = build_vital_channel_mapping({"source": source_config})

    if target_channel_mapping is not None:
        _target_channel_mapping = target_channel_mapping
    else:
        _target_channel_mapping = build_vital_channel_mapping({"source": target_config})

    logger.debug(f"Source channel mapping: {_source_channel_mapping}")
    logger.debug(f"Target channel mapping: {_target_channel_mapping}")

    if window_size is not None and window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    validate_trim_strategy(trim_strategy)

    def collate_fn(batch: list[Sample]) -> dict[str, Any]:
        """Collate function with lazy normalization (NEW format only)."""
        if not batch:
            raise ValueError("Cannot collate empty batch")

        batch_size = len(batch)

        # ============================================================================
        # BATCH-TIME PADDING/TRIMMING (NEW)
        # ============================================================================
        target_length = None
        if window_size is not None:
            target_length = window_size
            logger.debug(f"Using window_size: {target_length}")
        else:
            # No padding/trimming, use raw lengths (variable-length batch)
            logger.debug("Variable-length batch: no padding/trimming applied")

        # Build consistent mask for all required vitals to ensure batch alignment
        all_vital_configs = []
        if is_multi_source:
            all_vital_configs.extend(source_config)
        else:
            all_vital_configs.append(source_config)

        if is_multi_target:
            all_vital_configs.extend(target_config)
        else:
            all_vital_configs.append(target_config)

        valid_sample_mask = _build_valid_sample_mask(batch, all_vital_configs)

        # Fail fast if any samples are missing required vitals to preserve batch
        # size and alignment
        num_valid_samples = sum(valid_sample_mask)
        if num_valid_samples < batch_size:
            # Identify which samples are missing required vitals
            missing_samples = [
                (i, sample.sample_index if hasattr(sample, "sample_index") else i)
                for i, sample in enumerate(batch)
                if not valid_sample_mask[i]
            ]
            raise ValueError(
                f"Batch contains {
                    batch_size - num_valid_samples
                } sample(s) missing required vitals. "
                f"All samples must have all required vitals to preserve batch "
                f"size and alignment. "
                f"Missing samples (indices): {[idx for _, idx in missing_samples]}. "
                f"Required vitals: {all_vital_configs}"
            )
        elif num_valid_samples == 0:
            raise ValueError(
                f"No valid samples found in batch of {batch_size} samples. "
                f"All samples are missing required vitals: {all_vital_configs}"
            )

        if is_multi_source:
            # Multi-source: extract each vital and concatenate
            x, source_mask = _extract_and_normalize_multi_source_batch(
                batch,
                source_config,
                dataset_norm_params,
                target_length,
                trim_strategy,
                valid_sample_mask,
            )
        else:
            # Single source: extract one vital
            x, source_mask = _extract_and_normalize_vital_batch(
                batch,
                source_config,
                dataset_norm_params,
                "source",
                target_length,
                trim_strategy,
                valid_sample_mask,
            )

        if is_multi_target:
            # Multi-target: extract each target vital and concatenate
            y, target_mask = _extract_and_normalize_multi_source_batch(
                batch,
                target_config,
                dataset_norm_params,
                target_length,
                trim_strategy,
                valid_sample_mask,
            )
        else:
            # Single target: extract one target vital
            y, target_mask = _extract_and_normalize_vital_batch(
                batch,
                target_config,
                dataset_norm_params,
                "target",
                target_length,
                trim_strategy,
                valid_sample_mask,
            )

        valid_mask = source_mask & target_mask  # [B]

        batch_dict: dict[str, Any] = {
            "x": x,  # [B, C_sources, T] - normalized source(s)
            # [B, C_targets, T] - normalized target(s) (multi-channel when
            # target is list)
            "y": y,
            # [B] - mask indicating valid samples (preserves batch size)
            "valid_mask": valid_mask,
        }

        if include_list:
            for field_name in include_list:
                if field_name == "bp_raw":
                    has_any_bp = any(s.bp_raw is not None for s in batch)
                    if has_any_bp:
                        reference_bp = next(
                            (s.bp_raw for s in batch if s.bp_raw is not None), None
                        )
                        if reference_bp is None:
                            device = torch.device("cpu")
                            dtype = torch.float32
                        else:
                            device = reference_bp.device
                            dtype = reference_bp.dtype
                        nan_placeholder = torch.full(
                            (3,), float("nan"), device=device, dtype=dtype
                        )
                        bp_raw_list = [
                            torch.atleast_1d(
                                s.bp_raw if s.bp_raw is not None else nan_placeholder
                            )
                            for s in batch
                        ]
                        batch_dict["bp_raw"] = torch.stack(bp_raw_list).to(
                            torch.float32
                        )
                    else:
                        # Entire batch lacks bp_raw, provide NaN placeholder [B, 3]
                        batch_dict["bp_raw"] = torch.full(
                            (batch_size, 3), float("nan"), dtype=torch.float32
                        )

                elif field_name in [
                    "age_raw",
                    "gender_raw",
                    "height_raw",
                    "weight_raw",
                    "bmi_raw",
                ]:
                    has_field = any(
                        getattr(s, field_name, None) is not None for s in batch
                    )
                    if has_field:
                        reference = next(
                            (
                                getattr(s, field_name)
                                for s in batch
                                if getattr(s, field_name, None) is not None
                            ),
                            None,
                        )
                        if reference is None:
                            device = torch.device("cpu")
                            dtype = torch.float32
                        else:
                            device = reference.device
                            dtype = reference.dtype
                        nan_placeholder = torch.tensor(
                            [float("nan")], device=device, dtype=dtype
                        )

                        field_list = [
                            torch.atleast_1d(
                                getattr(s, field_name)
                                if getattr(s, field_name, None) is not None
                                else nan_placeholder
                            )
                            for s in batch
                        ]
                        batch_dict[field_name] = torch.stack(field_list).to(
                            torch.float32
                        )
                    else:
                        # Entire batch lacks this field, provide NaN placeholder [B, 1]
                        batch_dict[field_name] = torch.full(
                            (batch_size, 1), float("nan"), dtype=torch.float32
                        )

        if demographics_text_encoder is not None:
            demo_fields = [
                "age_raw",
                "gender_raw",
                "height_raw",
                "weight_raw",
                "bmi_raw",
            ]
            batch_demographics = {
                f: batch_dict[f] for f in demo_fields if f in batch_dict
            }
            if batch_demographics:
                tokens = demographics_text_encoder.encode_batch(
                    cast("dict[str, torch.Tensor]", batch_demographics)
                )
                batch_dict["text"] = {
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                }
                logger.debug(
                    f"Generated text tokens from demographics: input_ids shape "
                    f"{tokens['input_ids'].shape}"
                )

        if "class_labels" in input_preprocessing:
            field_name = input_preprocessing["class_labels"]

            missing_samples = []
            for i, sample in enumerate(batch):
                if (
                    not hasattr(sample, field_name)
                    or getattr(sample, field_name, None) is None
                ):
                    missing_samples.append(i)

            if missing_samples:
                raise ValueError(
                    f"Not all samples have '{field_name}' field. "
                    f"Missing samples: {missing_samples}. "
                    f"This field is required by input_preprocessing['class_labels']."
                )

            labels_list = []
            for sample in batch:
                label = getattr(sample, field_name)
                # Handle both scalar and 1D tensor labels
                if isinstance(label, torch.Tensor):
                    if label.dim() == 0:
                        labels_list.append(label.unsqueeze(0))
                    else:
                        labels_list.append(label)
                else:
                    labels_list.append(
                        torch.tensor(label, dtype=torch.long).unsqueeze(0)
                        if not isinstance(label, torch.Tensor)
                        else label
                    )

            batch_dict["class_labels"] = torch.stack(labels_list).squeeze(
                -1
            )  # [B] or [B, 1]
            logger.debug(
                f"Populated class_labels from field '{field_name}': shape "
                f"{batch_dict['class_labels'].shape}"
            )

        if mode_str == "single":
            if not directions or len(directions) == 0:
                raise ValueError(
                    "direction_mode='single' requires at least one direction. "
                    "Provided directions collection is empty."
                )
            direction = directions[0]
            metadata = _add_single_direction_metadata(
                direction,
                _source_channel_mapping,
                _target_channel_mapping,
                batch_dict,
                max_source_channels,
            )
            batch_dict.update(metadata)
        elif mode_str is not None:
            if not directions or len(directions) == 0:
                raise ValueError(
                    f"direction_mode='{mode_str}' requires at least one direction. "
                    "Provided directions collection is empty."
                )
            metadata = _add_multi_direction_metadata(
                directions,
                _source_channel_mapping,
                _target_channel_mapping,
                batch_dict,
                max_source_channels,
            )
            batch_dict.update(metadata)
        # If mode_str is None, skip direction metadata (for inference
        # scenarios without directions)

        # ============================================================================
        # PROCESSOR METADATA EXTRACTION (Phase 9)
        # ============================================================================
        # Extract metadata from samples for OutputProcessor post-processing.
        # These fields flow to ProcessingMetadata.from_batch() in
        # trainer.predict_step().
        # Fields:
        # - padding_length: Max padding across batch (for trimming)
        # - vital_sign_type: Target vital type (for extractor selection)
        # - extract_scalars: Whether to extract scalars (for BP extraction)
        # ============================================================================

        if target_length is not None:
            original_lengths = []
            for sample in batch:
                for vital_field in ["ecg_raw", "ppg_raw", "abp_raw", "imp_raw"]:
                    waveform = getattr(sample, vital_field, None)
                    if waveform is not None:
                        original_lengths.append(waveform.shape[-1])
                        break

            if original_lengths:
                max_original_length = max(original_lengths)
                padding_length = compute_padding_length(
                    max_original_length, target_length
                )
                logger.debug(
                    f"Computed padding_length: {padding_length} "
                    f"(original={max_original_length}, target={target_length})"
                )
            else:
                padding_length = 0
        else:
            padding_lengths = [getattr(s, "padding_length", 0) for s in batch]
            padding_length = max(padding_lengths, default=0)

            if len(set(padding_lengths)) > 1:
                logger.warning(
                    f"Non-uniform padding_length detected in batch: {padding_lengths}. "
                    f"Using max={
                        padding_length
                    }. Consider ensuring uniform padding per batch."
                )

        vital_sign_type = next(
            (
                getattr(s, "vital_sign_type", None)
                for s in batch
                if getattr(s, "vital_sign_type", None) is not None
            ),
            None,
        )

        extract_scalars = any(getattr(s, "extract_scalars", False) for s in batch)

        batch_dict["padding_length"] = padding_length
        batch_dict["vital_sign_type"] = vital_sign_type
        batch_dict["extract_scalars"] = extract_scalars

        return batch_dict

    return collate_fn
