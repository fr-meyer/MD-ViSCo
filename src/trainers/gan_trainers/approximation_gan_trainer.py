# Standard library imports
import logging
from typing import Tuple, Any, Dict, Optional, List

# Third-party imports
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

# Local imports
from .gan_trainer import GANTrainer, GANTrainerConfig
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ApproximationGANTrainerConfig(GANTrainerConfig):
    """Configuration for ApproximationGANTrainer with Hydra-compatible defaults"""
    _target_: str = "src.trainers.gan_trainers.approximation_gan_trainer.ApproximationGANTrainer"

    
    # Force trainer_name to be approximation
    trainer_name: str = "App"
    
    # GAN training parameters specific to approximation stage
    n_critic: int = 5  # Standard critic iterations for approximation
    

    input_preprocessing: Dict[str, str] = field(default_factory=lambda: {
        "x": "waveform_minmax_zc",  # Default for approximation
        "y": "waveform_minmax_zc",  # Default for approximation
    })

class ApproximationGANTrainer(GANTrainer):
    """GAN trainer specifically for the approximation stage
    
    This trainer overrides the base GANTrainer to provide what's specifically needed for the approximation stage.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.is_rank0:
            logger.info("ApproximationGANTrainer initialized")
    
    def _extract_target_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract target from unified batch structure for GAN training."""
        # Extract target using the same logic as ApproximationTrainer
        waveform = batch["x"]
        tgt_idxs = batch["tgt_idxs"]
        batch_arange = torch.arange(waveform.size(0), device=waveform.device)
        y_target = waveform[batch_arange, tgt_idxs].unsqueeze(1)  # [B, 1, T]
        
        return {"y": y_target}


cs = ConfigStore.instance()
cs.store(name="base_gan_approximation", node=ApproximationGANTrainerConfig, group="trainer")
 