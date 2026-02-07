from typing import Dict, Any, Optional
from tqdm import tqdm
from .wandb_wrapper import WandBWrapperConfig, WandBWrapper
from .csv_wrapper import CSVWrapperConfig, CSVWrapper
from dataclasses import dataclass, field
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressBarConfig:
    """Configuration for ProgressBar with Hydra-compatible defaults"""
    _target_: str = "src.loggings.progress_bar.ProgressBar"
    log_interval: int = 1
    wandb_log_interval: int = 1
    csv_log_interval: int = 1
    wandb_wrapper: Optional[WandBWrapperConfig] = None
    csv_wrapper: Optional[CSVWrapperConfig] = None
    # Remove: rank: int = 0

class ProgressBar:
    """Enhanced progress bar with integrated WandB wrapper and simple step management
    
    Step Management:
    - Training: Logs every batch with step=self._step (monotonically increasing)
    - Validation: Handled at epoch level by trainer (no step collision)
    
    Features:
    - Real-time metrics display in progress bar postfix
    - Proper interval control for different logging systems
    - Clean separation of concerns between step management and logging
    - Complete CSV logging for both training and validation
    - Step collision prevention for validation metrics
    """
    
    def __init__(self, log_interval: int = 1, wandb_log_interval: int = 1,
                 csv_log_interval: int = 1, wandb_wrapper: WandBWrapper = None, 
                 csv_wrapper: CSVWrapper = None):
        """Initialize without rank - rank will be determined at runtime
        
        Args:
            log_interval: Interval for progress bar updates
            wandb_log_interval: Interval for WandB logging
            csv_log_interval: Interval for CSV logging
            wandb_wrapper: Optional pre-configured WandBWrapper instance
            csv_wrapper: Optional pre-configured CSVWrapper instance
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.log_interval = log_interval
        self.wandb_log_interval = wandb_log_interval
        self.csv_log_interval = csv_log_interval
        self._current_bar = None
        self._step = 0  # Simple step counter
        
        # Validate configuration
        self._validate_config()
        
        # Use provided wrappers
        self.wandb = wandb_wrapper
        self.csv = csv_wrapper
        
        logger.debug(f"Initialized ProgressBar: {self}")
    
    def __repr__(self) -> str:
        """String representation for logging and debugging"""
        return (f"<ProgressBar(log_interval={self.log_interval}, "
                f"wandb_log_interval={self.wandb_log_interval}, "
                f"csv_log_interval={self.csv_log_interval})>")
    
    def _validate_config(self) -> None:
        """Validate configuration settings
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.wandb_log_interval <= 0:
            raise ValueError("wandb_log_interval must be positive")
        if self.csv_log_interval <= 0:
            raise ValueError("csv_log_interval must be positive")
    

    
    def create_bar(self, total, description="", disable=False, **kwargs):
        """Create tqdm progress bar"""
        self._current_bar = tqdm(
            total=total,
            desc=description,
            disable=disable,
            ncols=125,
            **kwargs
        )
    
    def update(self, n=1, metrics_dict=None, step=None, is_rank0: bool = False, to_log: bool = False):
        """Update progress bar with simple step management"""
        if self._current_bar is None:
            return
        
        # Update step counter
        if step is not None:
            self._step = step
        else:
            self._step += n
        
        self._current_bar.update(n)
        
        # Update metrics display and logging
        if metrics_dict and self._step % self.log_interval == 0:
            self._update_metrics(metrics_dict, is_rank0, to_log)
    
    def _update_metrics(self, metrics_dict, is_rank0: bool = False, to_log: bool = False):
        """Update metrics display, WandB, and CSV logging with interval control"""
        # Progress bar update (always)
        if hasattr(self._current_bar, 'set_postfix'):
            postfix = {}
            for key, value in metrics_dict.items():
                if isinstance(value, float):
                    postfix[key] = f"{value:.4f}"
                else:
                    postfix[key] = str(value)
            self._current_bar.set_postfix(**postfix)
        
        # WandB logging (only on rank 0, at interval)
        if self._step % self.wandb_log_interval == 0 and self.wandb is not None and to_log:
            self.wandb.log(metrics_dict, step=None, is_rank0=is_rank0)
        
        # CSV logging (only on rank 0, at interval)
        if self._step % self.csv_log_interval == 0 and self.csv is not None and to_log:
            self.csv.log_metrics(metrics_dict, step=self._step, is_rank0=is_rank0)
    
    def close(self):
        """Close progress bar, WandB, and CSV"""
        if self._current_bar and hasattr(self._current_bar, 'close'):
            self._current_bar.close()
        if self.wandb is not None:
            self.wandb.finish()
        if self.csv is not None:
            self.csv.finish()
    
    def close_wandb(self):
        """Close WandB"""
        if self.wandb is not None:
            self.wandb.finish()
    
    def close_progress_bar(self):
        """Close progress bar"""
        if self._current_bar and hasattr(self._current_bar, 'close'):
            self._current_bar.close()
    
    def close_csv(self):
        """Close CSV"""
        if self.csv is not None:
            self.csv.finish()

    @classmethod
    def from_config(cls, config: DictConfig) -> 'ProgressBar':
        """Create ProgressBar from main config with error handling
        
        Args:
            config: Main configuration object (DictConfig for better Hydra integration)
            
        Returns:
            ProgressBar: Initialized progress bar
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # First try to use dedicated progress_bar config
            if hasattr(config, 'progress_bar') and config.progress_bar is not None:
                progress_config = config.progress_bar
                return cls(
                    log_interval=progress_config.log_interval,
                    wandb_log_interval=progress_config.wandb_log_interval,
                    csv_log_interval=progress_config.csv_log_interval,
                    wandb_wrapper=progress_config.wandb_wrapper if hasattr(progress_config, 'wandb_wrapper') else None,
                    csv_wrapper=progress_config.csv_wrapper if hasattr(progress_config, 'csv_wrapper') else None
                )
            else:
                # Fallback to common config for backward compatibility
                return cls(
                    log_interval=1,  # Default values
                    wandb_log_interval=1,
                    csv_log_interval=100,
                    wandb_wrapper=None,
                    csv_wrapper=None
                )
        except AttributeError:
            raise ValueError("Config must have either 'progress_bar' or fallback to default values")

cs = ConfigStore.instance()
cs.store(name="base_progress_bar", node=ProgressBarConfig, group="progress_bar")