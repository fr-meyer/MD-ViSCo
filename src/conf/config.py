# Standard library imports
from dataclasses import dataclass
from typing import Optional

# Local imports
from src.dataset.base_dataset import DatasetBaseConfig
from src.trainers.trainer import TrainerBaseConfig


@dataclass
class Config:
    """Main configuration class for MD-ViSCo training pipeline.
    
    This class orchestrates core configuration components through composition:
    - train_dataset: Training dataset configuration (split ratios, paths, strategies)
    - test_dataset: Test dataset configuration (split ratios, paths, strategies)
    - trainer: Trainer configuration (training parameters, hardware, logging, seed)
    
    Additional configurations (model, criterion, features) are handled through
    Hydra's composition system via the YAML configuration files.
    """
    
    # Core configuration components
    train_dataset: Optional[DatasetBaseConfig] = None
    test_dataset: Optional[DatasetBaseConfig] = None
    trainer: Optional[TrainerBaseConfig] = None
    
