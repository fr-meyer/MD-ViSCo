# src/loggings/wandb_wrapper.py
import logging
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logging.warning("wandb not found, pip install wandb")

@dataclass
class WandBWrapperConfig:
    """Configuration for WandBWrapper with Hydra-compatible defaults"""
    _target_: str = "src.loggings.wandb_wrapper.WandBWrapper"
    project_name: str = MISSING
    run_name: Optional[str] = None
    wandb_enabled: bool = False
    entity: str = MISSING
    # Remove: config: Any = MISSING

class WandBWrapper:
    """Enhanced WandB wrapper with DDP support, explicit config control, and robust initialization"""
    
    def __init__(self, run_name: str = None,
                 project_name: str = None, wandb_enabled: bool = True,
                 entity: str = "single_waveform"):
        """Initialize without rank - will be set later via initialize method
        
        Args:
            run_name: Optional run name for WandB
            project_name: Project name (from config or environment)
            wandb_enabled: Whether to enable WandB logging
            entity: WandB entity name
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.run_name = run_name
        self.project_name = project_name or os.environ.get('WANDB_PROJECT')
        self.wandb_enabled = wandb_enabled
        self.entity = entity
        self._initialized = False
        
        # Validate configuration
        self._validate_config()
        
        logger.debug(f"Created WandBWrapper: {self}")
    
    def __repr__(self) -> str:
        """String representation for logging and debugging"""
        return (f"<WandBWrapper(project_name='{self.project_name}', "
                f"wandb_enabled={self.wandb_enabled}, "
                f"initialized={self._initialized})>")
    
    def _validate_config(self) -> None:
        """Validate configuration settings
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.wandb_enabled, bool):
            raise ValueError("wandb_enabled must be a boolean")
        if self.entity and not isinstance(self.entity, str):
            raise ValueError("entity must be a string")
    
    def initialize(self, is_rank0: bool, config: Any):
        """Initialize WandB with rank boolean and config
        
        Args:
            is_rank0: Whether this is the rank-0 process
            config: Configuration object from Hydra (required)
        """
        if not self._should_initialize(is_rank0):
            return
            
        self._initialize(is_rank0, config)
    
    def _should_initialize(self, is_rank0: bool) -> bool:
        """Check if WandB should be initialized based on rank boolean"""
        return (
            wandb is not None and 
            is_rank0 and  # ✅ Only rank 0
            self.wandb_enabled and
            self.project_name is not None and
            not self._initialized
        )
    
    def _initialize(self, is_rank0: bool, config: Any):
        """Initialize WandB with rank-specific settings"""
        try:
            # Create config for WandB initialization
            from src.utils.validation_utils import flatten_config
            wandb_config = flatten_config(config)
            
            wandb.init(
                entity=self.entity,
                project=self.project_name,
                name=self.run_name,
                config=wandb_config,
                # return_previous=False
            )
            self._initialized = True
            
            logger.info(f"WandB initialized on rank 0")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
    
    def log(self, metrics: Dict[str, Any], step: int = None, is_rank0: bool = False):
        """Log metrics only on rank 0"""
        if not self._initialized or not is_rank0:
            return
            
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to WandB: {e}")
    
    def _is_ready(self) -> bool:
        """Check if WandB is ready for logging"""
        return wandb is not None and wandb.run is not None and self._initialized
    
    def finish(self):
        """Finish WandB run and clean up environment"""
        if self._is_ready():
            wandb.finish()
            self._initialized = False
    
    @contextmanager
    def run(self):
        """Context manager for WandB run"""
        try:
            yield self
        finally:
            self.finish()
    
    def log_status_if_master(self, message: str, is_rank0: bool = False):
        """Log status message only if this is the master process"""
        if is_rank0:
            logger.info(message)
    
    def url(self) -> str | None:
        """Get WandB run URL if available"""
        if self._is_ready():
            return wandb.run.get_url()
        return None
    
    def log_status_with_url(self, message: str, is_rank0: bool = False):
        """Log status message with WandB URL if available"""
        if is_rank0:
            url = self.url()
            if url:
                logger.info(f"{message} - WandB URL: {url}")
            else:
                logger.info(message)
    
    def log_domain_metric(self, task: str, is_rank0: bool = False, **kwargs):
        """Log domain-specific metrics with task prefix"""
        if not kwargs:
            return
            
        # Create task-specific metric names
        task_metrics = {f"{task}/{k}": v for k, v in kwargs.items()}
        self.log(task_metrics, is_rank0=is_rank0)
    
    def watch(self, model, is_rank0: bool = False, log="gradients", log_freq=100):
        """Watch model for gradients and parameters only on rank 0"""
        if not self._initialized or not is_rank0:
            return
            
        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.warning(f"Failed to watch model in WandB: {e}")

cs = ConfigStore.instance()
cs.store(group="wandb_wrapper", name="base_wandb_wrapper", node=WandBWrapperConfig)