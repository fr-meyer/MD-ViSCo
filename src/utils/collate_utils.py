# Standard library imports
from typing import Dict, Any, List, Optional, Callable
import logging

# Third-party imports
import torch

# Local imports
from src.dataset.base_dataset import Sample

logger = logging.getLogger(__name__)


def stack_optional_tensor_field(batch: List[Sample], field_name: str) -> Optional[torch.Tensor]:
    """Helper to stack optional tensor fields from a batch of samples
    
    Args:
        batch: List of Sample objects
        field_name: Name of the field to stack
        
    Returns:
        Stacked tensor if all samples have the field, None otherwise
    """
    if all(getattr(s, field_name) is not None for s in batch):
        # Data is already tensors, just stack (keep on CPU) with dtype safety
        return torch.stack([
            getattr(s, field_name) for s in batch
        ]).to(torch.float32)  # Ensure float32 dtype
    return None


def sample_collate_fn(batch: List[Sample]) -> Dict[str, Any]:
    """Custom collate function for Sample objects with shared memory tensors
    
    Following industry standard: keep tensors on CPU during collation.
    Device transfer happens in training loop with pin_memory optimization.
    
    Args:
        batch: List of Sample objects from dataloader
        
    Returns:
        dict: Standardized batch dict with CPU tensors
    """
    # Stack tensors on CPU (let pin_memory work) with dtype safety
    abp_global_minmax = torch.stack([
        s.abp_global_minmax for s in batch
    ]).to(torch.float32)  # Ensure float32 dtype
    
    bp_raw = torch.stack([
        s.bp_raw for s in batch
    ]).to(torch.float32)  # Ensure float32 dtype
    
    waveform_minmax_zc = torch.stack([
        s.waveform_minmax_zc for s in batch
    ]).to(torch.float32)  # Ensure float32 dtype
    
    bp_global_minmax = torch.stack([
        s.bp_global_minmax for s in batch
    ]).to(torch.float32)  # Ensure float32 dtype
    
    # Handle optional tensor fields (keep on CPU)
    abp_raw = stack_optional_tensor_field(batch, "abp_raw")
    demographics_raw = stack_optional_tensor_field(batch, "demographics_raw")
    demographics_split = stack_optional_tensor_field(batch, "demographics_split")
    text_encoded = stack_optional_tensor_field(batch, "text_encoded")
    
    # Build standardized batch dict (all tensors on CPU)
    batch_dict: Dict[str, Any] = {
        "waveform": waveform_minmax_zc,
        "bp_raw": bp_raw,
    }
    
    # Add optional fields if available
    if bp_global_minmax is not None:
        batch_dict["bp"] = bp_global_minmax
    if abp_global_minmax is not None:
        batch_dict["abp"] = abp_global_minmax
    if text_encoded is not None:
        batch_dict["text"] = text_encoded
    if demographics_split is not None:
        batch_dict["demographics"] = demographics_split
    elif demographics_raw is not None:
        batch_dict["demographics"] = demographics_raw
    if abp_raw is not None:
        batch_dict["abp_raw"] = abp_raw
        
    return batch_dict


def create_collate_fn():
    """Factory function to create a collate function
    
    Note: Device parameter is ignored as per industry standard.
    Device transfer happens in training loop with pin_memory optimization.
    
    Args:
        device: Ignored - kept for API compatibility
        
    Returns:
        Callable: Collate function that keeps tensors on CPU
    """
    def collate_fn(batch: List[Sample]) -> Dict[str, Any]:
        return sample_collate_fn(batch)
    return collate_fn


def create_configurable_collate_fn(input_preprocessing: Dict[str, str]):
    """Create a collate function that respects ONLY the user-specified preprocessing configuration
    
    This function creates a collate function that will include ONLY the fields specified
    in input_preprocessing. No hardcoded fields are added - users have full control
    over what gets included in the batch.
    
    Note: Preprocessing configuration validation happens at the config level before
    this function is called. This function assumes valid configuration and focuses
    purely on data batching.
    
    Args:
        input_preprocessing: Dict mapping output keys to Sample attribute names
        Example: {"waveform": "waveform_minmax_zc", "abp": "abp_global_minmax"}
    
    Returns:
        Collate function that can be passed to DataLoader
    """
    
    def collate_fn(batch: List[Sample]) -> Dict[str, Any]:
        """Configurable collate function that respects ONLY input_preprocessing config"""
        if not batch:
            return {}
        
        batch_dict = {}
        
        # Handle ONLY the user-specified preprocessing configuration
        # No validation needed - config already validated at startup
        for output_key, sample_attr in input_preprocessing.items():
            tensor_list = [getattr(s, sample_attr) for s in batch]
            batch_dict[output_key] = torch.stack(tensor_list).to(torch.float32)  # Ensure float32 dtype
        
        return batch_dict
    
    return collate_fn


def validate_preprocessing_config(input_preprocessing: Dict[str, str], sample: Sample) -> bool:
    """Validate that requested preprocessing attributes exist in dataset
    
    Args:
        input_preprocessing: Dict mapping output keys to Sample attribute names
        sample: Sample object to validate against
        
    Returns:
        bool: True if all attributes exist, False otherwise
    """
    if not sample:
        return False
    
    missing_attrs = []
    for output_key, sample_attr in input_preprocessing.items():
        if not hasattr(sample, sample_attr):
            missing_attrs.append(f"{output_key} -> {sample_attr}")
    
    if missing_attrs:
        available_attrs = [attr for attr in dir(sample) if not attr.startswith('_')]
        logger.warning(
            f"Requested preprocessing attributes not found: {missing_attrs}\n"
            f"Available attributes: {available_attrs}"
        )
        return False
    
    return True


# ============================================================================
# Direction-Aware Collate Functions
# ============================================================================

def _populate_source_channels_collate(
    direction,  # Direction type from src.core.domain
    vitals_dataset,  # VitalsDataset type from src.dataload.base_dataset
    src_idxs: torch.Tensor,
    src_mask: torch.Tensor,
    sample_idx: int,
    S_max: int
) -> None:
    """Shared logic for populating source channel indices and masks
    
    This function centralizes the source channel handling logic that was previously
    duplicated in the trainer. It's designed to work efficiently in the collate context.
    
    Args:
        direction: Direction object containing source and target information
        vitals_dataset: Dataset for channel mapping
        src_idxs: Tensor to populate with source channel indices [B, S_max]
        src_mask: Tensor to populate with source channel masks [B, S_max]
        sample_idx: Sample index in batch (0 for single-direction, i for multi-directional)
        S_max: Maximum number of source channels
    """
    if hasattr(direction, 'src'):
        src = direction.src
        
        if hasattr(src, 'name'):  # Single Vital (e.g., Vital.PPG)
            src_chan = vitals_dataset.chan(src)
            src_idxs[sample_idx, 0] = src_chan
            src_mask[sample_idx, 0] = True
        
        elif isinstance(src, (list, tuple)):
            # Multiple vital sources (e.g., [Vital.PPG, Vital.ECG])
            for j, vital in enumerate(src):
                if j < S_max:
                    src_chan = vitals_dataset.chan(vital)
                    src_idxs[sample_idx, j] = src_chan
                    src_mask[sample_idx, j] = True
        
        elif hasattr(direction, 'source_vitals'):
            # Multi-variate sources with source_vitals attribute
            for j, vital in enumerate(direction.source_vitals):
                if j < S_max:
                    src_chan = vitals_dataset.chan(vital)
                    src_idxs[sample_idx, j] = src_chan
                    src_mask[sample_idx, j] = True


def _add_single_direction_metadata(
    direction,  # Direction type from src.core.domain
    vitals_dataset,  # VitalsDataset type from src.dataload.base_dataset
    batch_dict: Dict[str, Any],
    S_max: int
) -> Dict[str, Any]:
    """Add single-direction metadata to batch
    
    Args:
        direction: Single direction for the entire batch
        vitals_dataset: Dataset for channel mapping
        batch_dict: Existing batch with preprocessing data
        S_max: Maximum number of source channels
        
    Returns:
        Updated batch_dict with direction metadata
    """
    waveform = batch_dict["x"]
    batch_size = waveform.shape[0]
    
    # Initialize tensors
    src_idxs = torch.zeros(1, S_max, dtype=torch.long)
    src_mask = torch.zeros(1, S_max, dtype=torch.bool)
    
    # Populate source channels using shared logic
    _populate_source_channels_collate(direction, vitals_dataset, src_idxs, src_mask, 0, S_max)
    
    # Get target channel
    tgt_chan = vitals_dataset.chan(direction.tgt)
    
    return {
        "src_idxs": src_idxs.repeat(batch_size, 1).contiguous(),  # [B, S_max] - No aliasing risk
        "src_mask": src_mask.repeat(batch_size, 1).contiguous(),  # [B, S_max] - No aliasing risk
        "tgt_idxs": torch.full((batch_size,), tgt_chan, dtype=torch.long),  # [B] - Fresh tensor, no aliasing
        "direction": direction
    }


def _add_multi_direction_metadata(
    directions,  # Directions type from src.core.direction
    vitals_dataset,  # VitalsDataset type from src.dataload.base_dataset
    batch_dict: Dict[str, Any],
    S_max: int
) -> Dict[str, Any]:
    """Add multi-directional metadata to batch
    
    Args:
        directions: Collection of available directions
        vitals_dataset: Dataset for channel mapping
        batch_dict: Existing batch with preprocessing data
        S_max: Maximum number of source channels
        
    Returns:
        Updated batch_dict with multi-directional metadata
    """
    waveform = batch_dict["x"]
    batch_size = waveform.shape[0]
    
    # Randomly select directions for each sample
    available_directions = list(directions)
    selected_directions = [available_directions[torch.randint(0, len(available_directions), (1,)).item()] for _ in range(batch_size)]
    
    # Initialize tensors
    src_idxs = torch.zeros(batch_size, S_max, dtype=torch.long)
    src_mask = torch.zeros(batch_size, S_max, dtype=torch.bool)
    tgt_idxs = torch.zeros(batch_size, dtype=torch.long)
    
    # Populate for each sample
    for i, direction in enumerate(selected_directions):
        _populate_source_channels_collate(direction, vitals_dataset, src_idxs, src_mask, i, S_max)
        tgt_idxs[i] = vitals_dataset.chan(direction.tgt)
    
    # Create domain shift targets
    target_channels = [vitals_dataset.chan(d.tgt) for d in selected_directions]
    domain_shift_target = torch.nn.functional.one_hot(
        torch.tensor(target_channels), len(vitals_dataset.vitals())
    ).to(torch.float32)
    
    return {
        "src_idxs": src_idxs,      # [B, S_max]
        "src_mask": src_mask,      # [B, S_max] 
        "tgt_idxs": tgt_idxs,     # [B]
        "domain_shift_target": domain_shift_target,
        "directions": selected_directions,
        "mixed_batch": True
    }


def create_direction_aware_collate_fn(
    input_preprocessing: Dict[str, str],
    directions,  # Directions type from src.core.direction
    vitals_dataset,  # VitalsDataset type from src.dataload.base_dataset
    direction_mode,  # DirectionMode enum from src.trainers.trainer
    max_source_channels: int = 2
) -> Callable:
    """Create a direction-aware collate function using composition
    
    This function follows the industry-standard composed factory pattern:
    1. Creates the base preprocessing collate function ONCE
    2. Returns a composed function that applies both preprocessing and direction logic
    3. Ensures optimal performance with no nested function creation
    
    Args:
        input_preprocessing: Dict mapping output keys to Sample attribute names
        directions: Collection of available directions for training
        vitals_dataset: Dataset for channel mapping
        direction_mode: Training mode (single or multi-directional)
        max_source_channels: Maximum number of source channels per sample
        
    Returns:
        Collate function that handles both preprocessing and direction metadata
        
    Example:
        ```python
        # Create once during dataloader setup
        collate_fn = create_direction_aware_collate_fn(
            input_preprocessing={"waveform": "waveform_minmax_zc"},
            directions=directions,
            vitals_dataset=vitals_dataset,
            direction_mode=DirectionMode.SINGLE
        )
        
        # Use in DataLoader
        train_loader = DataLoader(
            dataset,
            collate_fn=collate_fn,  # Reused for all batches
            ...
        )
        ```
    """
    # Create the base preprocessing collate function ONCE
    base_collate_fn = create_configurable_collate_fn(input_preprocessing)
    
    def collate_fn(batch: List[Sample]) -> Dict[str, Any]:
        """Composed collate function that handles preprocessing + direction logic
        
        This function is created once and reused for all batches, ensuring
        optimal performance while maintaining clean separation of concerns.
        """
        # 1. Apply base preprocessing (using pre-created function)
        batch_dict = base_collate_fn(batch)
        
        # 2. Add direction-specific metadata based on mode
        if direction_mode.value == "single":  # Access enum value
            batch_dict.update(_add_single_direction_metadata(
                directions[0], vitals_dataset, batch_dict, max_source_channels
            ))
        else:
            batch_dict.update(_add_multi_direction_metadata(
                directions, vitals_dataset, batch_dict, max_source_channels
            ))
        
        return batch_dict
    
    return collate_fn 