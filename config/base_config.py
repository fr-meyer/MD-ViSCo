# Standard library imports
import random
from dataclasses import dataclass

# Third-party imports
import numpy as np
import torch

@dataclass
class BaseModelConfig:
    """Base configuration for all models"""
    # Paths
    data_path: str = './datasets/'
    abs_path: str = './weights/'
    
    path_folder: str = None
    
    # Training Parameters
    batch_size: int = 256
    test_batch_size: int = batch_size
    num_epochs: int = 100
    learning_rate: float = 1e-3
    scheduler_patience: int = 3 # for learning rate scheduler
    resume_training: bool = False
    early_stopping_patience: int = 5 # for early stopping
    checkpoint_name: str = None
    checkpoint_epoch: int = None
    model_type: str = None

    calculate_bpm: bool = False
    extract_waveform_features: bool = False
    # Add seed parameter to the config
    seed: int = 42  # Default seed value

    is_pretraining = False
    is_finetuning = False
    
    def set_seed(self, seed: int = None):
        """Set random seeds for reproducibility across all relevant libraries
        
        Args:
            seed (int, optional): Seed value to use. If None, uses the config's seed value.
                                Defaults to None.
        """
        if seed is not None:
            self.seed = seed
            
        # Set seeds for Python's random module
        random.seed(self.seed)
        
        # Set seeds for NumPy
        np.random.seed(self.seed)
        
        # Set seeds for PyTorch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU
        
        # Set deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"Random seeds set to: {self.seed}")# Add seed parameter to the config

    def __post_init__(self):
        pass