import torch
from typing import Optional, Dict, Any

def load_state_dict_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cpu',
    key: str = 'model_state_dict',
    strict: bool = True
) -> Dict[str, Any]:
    """
    Loads only the model weights from a checkpoint file.

    Args:
        model: The model instance to load weights into.
        checkpoint_path: Path to the checkpoint file.
        device: Device to map the checkpoint to.
        key: The key in the checkpoint dict for the model state dict.
        strict: Whether to strictly enforce that the keys in state_dict match the model.

    Returns:
        checkpoint: The full checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint[key]
    # Remove 'module.' prefix if present (for DDP checkpoints)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=strict)
    return checkpoint


def load_checkpoint_components(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    key_map: Optional[Dict[str, str]] = None,
    device: str = 'cpu',
    strict: bool = True
) -> Dict[str, Any]:
    """
    Loads model, optimizer, and scheduler state from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model instance (optional).
        optimizer: Optimizer instance (optional).
        scheduler: Scheduler instance (optional).
        key_map: Dict mapping component names to checkpoint keys.
        device: Device to map the checkpoint to.
        strict: Whether to strictly enforce that the keys in state_dict match the model.

    Returns:
        checkpoint: The full checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    key_map = key_map or {
        'model': 'model_state_dict',
        'optimizer': 'optim_state_dict',
        'scheduler': 'scheduler_state_dict',
    }

    if model and key_map.get('model') in checkpoint:
        state_dict = checkpoint[key_map['model']]
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=strict)

    if optimizer and key_map.get('optimizer') in checkpoint:
        optimizer.load_state_dict(checkpoint[key_map['optimizer']])

    if scheduler and key_map.get('scheduler') in checkpoint:
        scheduler.load_state_dict(checkpoint[key_map['scheduler']])

    return checkpoint 