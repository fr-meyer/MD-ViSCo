# Standard library imports
import logging
from typing import Dict, Optional, List, Tuple, Any

# Third-party imports
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

# Local imports
from .gan_trainer import GANTrainer, GANTrainerConfig
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RefinementGANTrainerConfig(GANTrainerConfig):
    """Configuration for RefinementGANTrainer with Hydra-compatible defaults"""
    _target_: str = "src.trainers.gan_trainers.refinement_gan_trainer.RefinementGANTrainer"
    
    # Force trainer_name to be refinement
    trainer_name: str = "Ref"
    
    # GAN training parameters specific to refinement stage
    n_critic: int = 5  # Standard critic iterations for refinement

    input_preprocessing: Dict[str, str] = field(default_factory=lambda: {
        "x": "waveform_minmax_zc",  # Default for refinement
        "y": "abp_global_minmax",  # Default for refinement
        "y_bp": "bp_global_minmax",  # Default for refinement
    })

class RefinementGANTrainer(GANTrainer):
    """GAN trainer specifically for the refinement stage
    
    This trainer overrides only the essential methods needed for refinement stage.
    All other functionality (loss computation, BP metrics, training loop) is inherited 
    from the parent GANTrainer class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        if self.is_rank0:
            logger.info("RefinementGANTrainer initialized")

    def _extract_target_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract target from unified batch structure for loss computation.
        
        Args:
            batch: Batch dictionary with unified structure
            outputs: Model outputs (for compatibility with parent class)
            
        Returns:
            Dict[str, torch.Tensor]: Target dictionary for loss computation
        """
        target = {}

        target["y"] = batch["y"]

        if "y_bp" in batch:
            target["y_sbp"] = batch["y_bp"][:, 0]
            target["y_dbp"] = batch["y_bp"][:, 1]

        return target


cs = ConfigStore.instance()
cs.store(name="base_gan_refinement", node=RefinementGANTrainerConfig, group="trainer")