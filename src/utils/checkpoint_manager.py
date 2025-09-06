import os
import glob
import logging
from typing import Dict, Any, Optional
import torch
from dataclasses import dataclass, field
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CheckpointManagerConfig:
    """Configuration for CheckpointManager with Hydra-compatible defaults"""
    _target_: str = "src.utils.checkpoint_manager.CheckpointManager"
    base_dir: str = "./weights/"
    file_ext: str = ".pt"
    path_format: str = "{trainer_name}Model/{dataset_name}/{trainer_name}Model_{model_name}_BS_{batch_size}_E_{num_epochs}_LR_{learning_rate}{optional_params}_P_{scheduler_patience}_ES_{early_stopping_patience}{extra_flags}"
    filename_format: str = "{direction}checkpoint_S_{seed}{epoch_suffix}{file_ext}"


class CheckpointManager:
    """Manages checkpoint file paths, directories, and naming conventions
    
    This class focuses solely on path management and does not handle
    checkpoint loading/saving logic or overwrite policies, which belong in the trainer.
    
    Key Features:
    - All parameters are required (no optional parameters)
    - Configurable path and filename formats (set at initialization, cannot be overridden)
    - Support for trainer-specific path logic
    - Future extensibility through **kwargs
    - No business logic validation (focuses on path construction only)
    - No overwrite handling (trainer responsibility)
    - Configuration attributes (base_dir, file_ext, path_format, filename_format) are always used
    
    Usage:
        # All parameters are now required
        path = manager.build_path(
            model_name="my_model",
            trainer_name="approximation", 
            dataset_name="my_dataset",
            epoch=100,
            batch_size=32,
            num_epochs=100,
            learning_rate=0.001,
            scheduler_patience=10,
            early_stopping_patience=20,
            is_finetuning=False,
            use_patient_split=False,
            use_patient_information=True,
            seed=42,
            direction="source2target",
            use_wcl=False
        )
    """
    
    def __init__(self, 
                 base_dir: str = './weights/',
                 file_ext: str = ".pt",
                 path_format: str = "{trainer_name}Model/{dataset_name}/{trainer_name}Model_{model_name}_BS_{batch_size}_E_{num_epochs}_LR_{learning_rate}{optional_params}_P_{scheduler_patience}_ES_{early_stopping_patience}{extra_flags}",
                 filename_format: str = "{direction}checkpoint_S_{seed}{epoch_suffix}{file_ext}"):
        """Initialize with path management parameters only
        
        Args:
            base_dir: Base directory for checkpoints
            file_ext: File extension for checkpoint files
            path_format: Format string for subfolder path construction
            filename_format: Format string for filename construction
        """
        # Store configuration as attributes
        self.base_dir = base_dir
        self.file_ext = file_ext
        self.path_format = path_format
        self.filename_format = filename_format
        
        # Validate configuration
        self._validate_config()
        logger.debug(f"Initialized CheckpointManager: {self}")
    
    def __repr__(self) -> str:
        """String representation for logging and debugging"""
        return (f"<CheckpointManager(base_dir='{self.base_dir}', "
                f"file_ext='{self.file_ext}')>")
    
    def _validate_config(self) -> None:
        """Validate configuration settings
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.base_dir:
            raise ValueError("base_dir cannot be empty")
        if not self.file_ext:
            raise ValueError("file_ext cannot be empty")

    
    def build_path(self, 
                   # Core path parameters
                   model_name: str,
                   trainer_name: str,
                   dataset_name: str,
                   epoch: int,
                   
                   # Training parameters (explicit)
                   batch_size: int,
                   num_epochs: int,
                   learning_rate: float,
                   scheduler_patience: int,
                   early_stopping_patience: int,
                   
                   # Trainer-specific parameters
                   is_finetuning: bool,
                   use_patient_split: bool,
                   use_patient_information: bool,
                   
                   # Filename parameters
                   seed: int,
                   direction: str,
                   
                   # Additional parameters
                   use_wcl: bool,
                   
                   # Future extensions
                   **kwargs) -> str:
        """Build checkpoint path with parameter collection (no business logic validation)
        
        Args:
            model_name: Model name (required)
            trainer_name: Trainer name (required)
            dataset_name: Dataset name (required)
            epoch: Epoch number (required)
            batch_size: Training batch size (required)
            num_epochs: Number of epochs (required)
            learning_rate: Learning rate (required)
            scheduler_patience: Scheduler patience (required)
            early_stopping_patience: Early stopping patience (required)
            is_finetuning: Whether in finetuning mode (required)
            use_patient_split: Whether using patient split (required)
            use_patient_information: Whether using patient information (required)
            seed: Seed for reproducibility (required)
            direction: Direction for multi-directional training (required)
            use_wcl: Whether to use WCL for BP models (required)
            **kwargs: Additional parameters for future extensions
            
        Returns:
            str: Full checkpoint path
            
        Raises:
            ValueError: If required path parameters are missing
        """
        # Extract and validate all parameters
        params = self._extract_all_parameters(
            model_name=model_name,
            trainer_name=trainer_name,
            dataset_name=dataset_name,
            epoch=epoch,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            scheduler_patience=scheduler_patience,
            early_stopping_patience=early_stopping_patience,
            is_finetuning=is_finetuning,
            use_patient_split=use_patient_split,
            use_patient_information=use_patient_information,
            seed=seed,
            direction=direction,
            use_wcl=use_wcl,
            **kwargs
        )
        
        subfolder = self._build_subfolder(params)
        filename = self._build_filename(params)
        
        full_path = os.path.join(params['base_dir'], subfolder, filename)
        
        # Create directory structure (always)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        logger.debug(f"Built checkpoint path: {full_path}")
        return full_path
    
    def find_checkpoint(self, 
                        # Core path parameters
                        model_name: str,
                        trainer_name: str,
                        dataset_name: str,
                        epoch: int,
                        
                        # Training parameters (explicit)
                        batch_size: int,
                        num_epochs: int,
                        learning_rate: float,
                        scheduler_patience: int,
                        early_stopping_patience: int,
                        
                        # Trainer-specific parameters
                        is_finetuning: bool,
                        use_patient_split: bool,
                        use_patient_information: bool,
                        
                        # Filename parameters
                        seed: int,
                        direction: str,
                        
                        # Additional parameters
                        use_wcl: bool,
                        
                        # Future extensions
                        **kwargs) -> Optional[str]:
        """Find checkpoint using parameter collection (no business logic validation)
        
        Args:
            model_name: Model name (required)
            trainer_name: Trainer name (required)
            dataset_name: Dataset name (required)
            epoch: Epoch number (required)
            batch_size: Training batch size (required)
            num_epochs: Number of epochs (required)
            learning_rate: Learning rate (required)
            scheduler_patience: Scheduler patience (required)
            early_stopping_patience: Early stopping patience (required)
            is_finetuning: Whether in finetuning mode (required)
            use_patient_split: Whether using patient split (required)
            use_patient_information: Whether using patient information (required)
            seed: Seed for reproducibility (required)
            direction: Direction for multi-directional training (required)
            use_wcl: Whether to use WCL for BP models (required)
            **kwargs: Additional parameters for future extensions
            
        Returns:
            Optional[str]: Checkpoint path if found, None otherwise
            
        Raises:
            ValueError: If required path parameters are missing
        """
        try:
            new_path = self.build_path(
                model_name=model_name,
                trainer_name=trainer_name,
                dataset_name=dataset_name,
                epoch=epoch,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                scheduler_patience=scheduler_patience,
                early_stopping_patience=early_stopping_patience,
                is_finetuning=is_finetuning,
                use_patient_split=use_patient_split,
                use_patient_information=use_patient_information,
                seed=seed,
                direction=direction,
                use_wcl=use_wcl,
                **kwargs
            )
            return new_path if os.path.exists(new_path) else None
        except ValueError as e:
            logger.debug(f"Failed to build path: {e}")
            
        logger.warning(f"No checkpoint found for: {dataset_name}/{trainer_name}/{model_name}")
        return None
    
    def _extract_all_parameters(self, 
                               model_name: str,
                               trainer_name: str,
                               dataset_name: str,
                               epoch: int,
                               batch_size: int,
                               num_epochs: int,
                               learning_rate: float,
                               scheduler_patience: int,
                               early_stopping_patience: int,
                               is_finetuning: bool,
                               use_patient_split: bool,
                               use_patient_information: bool,
                               
                               # Filename parameters
                               seed: int,
                               direction: str,
                               
                               # Additional parameters
                               use_wcl: bool,
                               **kwargs) -> Dict[str, Any]:
        """Extract parameters for path construction without business logic validation"""
    
        
        # Build complete parameter dictionary without validation
        params = {
            'base_dir': self.base_dir,
            'model_name': model_name,
            'trainer_name': trainer_name,
            'dataset_name': dataset_name,
            'epoch': epoch,
            'file_ext': self.file_ext,
            'is_finetuning': is_finetuning,
            'use_patient_split': use_patient_split,
            'use_patient_information': use_patient_information,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'scheduler_patience': scheduler_patience,
            'early_stopping_patience': early_stopping_patience,
            'seed': seed,
            'direction': direction,
            'use_wcl': use_wcl,
            **kwargs
        }
        
        return params

    
    def _build_subfolder(self, params: Dict[str, Any]) -> str:
        """Build subfolder using configurable format
        
        Args:
            params: Complete parameter dictionary from _extract_all_parameters
            
        Returns:
            str: Subfolder path with finetuning/patient split suffixes
            
        Raises:
            KeyError: If path_format references a parameter that doesn't exist in params
        """
        # Always use path_format - it controls what gets included
        format_params = params.copy()

        optional_params = ""
        extra_flags = ""

        if format_params['model_name'] == 'BPModel':
            optional_params += f"_WCL_{format_params['use_wcl']}"
            extra_flags += f"_BP_NORM"
        if format_params['model_name'] in ['BPModel', 'PatchTST']:
            optional_params += f"_PI_{format_params['use_patient_information']}"
        
        if format_params['is_finetuning']:
            extra_flags += f"_finetuning"
        
        if format_params['use_patient_split']:
            extra_flags += f"_Patient_Split"

        format_params['optional_params'] = optional_params
        format_params['extra_flags'] = extra_flags
        
        try:
            subfolder = self.path_format.format(**format_params)
        except KeyError as e:
            raise KeyError(f"Missing parameter '{e}' for path format. Available: {list(format_params.keys())}") from e

        return subfolder
    
    def _build_filename(self, params: Dict[str, Any]) -> str:
        """Build filename using configurable format
        
        Args:
            params: Complete parameter dictionary from _extract_all_parameters
            
        Returns:
            str: Filename
            
        Raises:
            KeyError: If filename_format references a parameter that doesn't exist in params
        """
        format_params = params.copy()

        if params.get('direction') is not None:
            format_params['direction'] += "_" if params['direction'] != "" else ""

        # Add epoch suffix if provided
        epoch_suffix = ""
        if params.get('epoch') is not None:
            epoch_suffix = f"_epoch_{params['epoch']}"
        
        format_params['epoch_suffix'] = epoch_suffix
        format_params['file_ext'] = params['file_ext']
        
        try:
            return self.filename_format.format(**format_params)
        except KeyError as e:
            raise KeyError(f"Missing parameter '{e}' for filename format. Available: {list(format_params.keys())}") from e
    


cs = ConfigStore.instance()
cs.store(name="base_checkpoint_manager", node=CheckpointManagerConfig, group="checkpoint_manager")