"""
Validation utilities for configuration and data processing.
"""

import warnings
from typing import Any, Dict


def flatten_config(cfg: Any, parent_key: str = '', sep: str = '_') -> dict:
    """
    Flatten config object, keeping only wandb-serializable types.
    
    This function is used by the modern WandBWrapper system for config flattening.
    
    Args:
        cfg: Configuration object to flatten (can be dict, object with __dict__, or OmegaConf)
        parent_key: Parent key for nested items (used in recursion)
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary with only wandb-serializable values
    """
    # Add debug logging
    # logger.debug(f"flatten_config called with type: {type(cfg)}")
    # logger.debug(f"Config has __dict__: {hasattr(cfg, '__dict__')}")
    # logger.debug(f"Config is dict: {isinstance(cfg, dict)}")
    
    items = {}
    
    # Handle different config types - prioritize OmegaConf over __dict__
    if hasattr(cfg, '_content'):  # OmegaConf object
        # logger.debug("Using OmegaConf approach")
        try:
            from omegaconf import OmegaConf
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            # logger.debug(f"OmegaConf conversion result type: {type(cfg_dict)}")
            # logger.debug(f"OmegaConf conversion result: {cfg_dict}")
            
            if isinstance(cfg_dict, dict):
                cfg_items = cfg_dict.items()
                # logger.debug("Using OmegaConf dict approach")
            else:
                # logger.warning(f"OmegaConf conversion returned non-dict: {type(cfg_dict)}")
                return {}
        except Exception as e:
            # logger.error(f"OmegaConf conversion failed: {e}")
            return {}
    elif isinstance(cfg, dict):
        cfg_items = cfg.items()
        # logger.debug("Using dict approach")
    elif hasattr(cfg, '__dict__'):
        cfg_items = vars(cfg).items()
        # logger.debug("Using __dict__ approach")
    else:
        # logger.warning(f"Unknown config type: {type(cfg)}")
        return {}
    
    for k, v in cfg_items:
        # Convert key to string for safe handling
        k_str = str(k)
        new_key = f"{parent_key}{sep}{k_str}" if parent_key else k_str
        
        # Skip private attributes and methods
        if k_str.startswith('_'):
            continue
            
        # Handle nested objects recursively
        if hasattr(v, '__dict__') or (isinstance(v, dict) and v):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            # Only keep wandb-serializable types
            if is_wandb_serializable(v):
                items[new_key] = v
            # Skip non-serializable types silently
    
    return items

def is_wandb_serializable(value: Any) -> bool:
    """
    Check if a value can be safely passed to wandb config.
    
    Args:
        value: Value to check for wandb serializability
        
    Returns:
        True if the value is wandb-serializable, False otherwise
    """
    # wandb supports these types natively
    serializable_types = (
        int, float, str, bool, type(None),  # Basic types
        list, tuple,  # Sequences (if they contain serializable items)
        dict  # Dictionaries (if they contain serializable items)
    )
    
    # Check basic type
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