# Standard library imports
import logging
import os
import random
import numpy as np
import datetime
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Any, Optional, List, Dict, Union
from src.utils.collate_utils import create_direction_aware_collate_fn
        

"""
MD-ViSCo Trainer with Canonical PyTorch Checkpoint Loading
=========================================================

Implements the canonical PyTorch checkpoint loading pattern for production DDP training:

**Key Rules:**
1. **Model weights**: Load only on global rank-0 before DDP wrap
2. **Trainer state**: Load on rank-0, broadcast payload only after DDP wrap

**Architecture:**
1. Pick device → 2. Rank-0 preloads checkpoint → 3. Build model → 4. Load weights (rank-0 only) 
→ 5. Init DDP → 6. Wrap with DDP (auto-broadcasts) → 7. Create optimizer/scheduler → 8. Load trainer states

**Usage:**
- Always call run_training() - handles all scenarios automatically
- For manual loading: load_checkpoint_unified() → prepare_model_weights() → load_trainer_states()
"""

# Third-party imports
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
import torch.nn as nn
from enum import Enum
import math

# Local imports
# from src.dataload.base_dataset import Sample
# from src.utils.checkpoint_utils import load_checkpoint_components
from src.utils.checkpoint_manager import CheckpointManager, CheckpointManagerConfig
from src.utils.collate_utils import sample_collate_fn
from src.loggings.metrics import metrics
from src.criterions.base_criterion import CriterionBaseConfig, BaseCriterion
from src.loggings.progress_bar import ProgressBarConfig, ProgressBar
from src.utils.train_utils import EarlyStopping, EarlyStoppingConfig
from src.core.direction import Directions, DirectionsConfig

logger = logging.getLogger(__name__)

class DirectionMode(Enum):
    SINGLE = "single"
    MULTI = "multi"
    
    @classmethod
    def from_string(cls, value: str) -> 'DirectionMode':
        """Convert string to enum with helpful error message"""
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"Invalid direction_mode '{value}'. Must be one of: {valid_values}")

@dataclass
class TrainerBaseConfig:
    """Configuration for BaseTrainer with Hydra-compatible defaults"""
    _target_: str = "src.trainers.trainer.BaseTrainer"
    _recursive_: bool = True
    trainer_name: str = MISSING  # 'approximation' or 'refinement' - determines training pipeline

    model: Any = MISSING
    criterion: CriterionBaseConfig = MISSING
    checkpoint_manager: CheckpointManagerConfig = MISSING
    progress_bar: ProgressBarConfig = MISSING
    early_stopping: EarlyStoppingConfig = MISSING
    directions: DirectionsConfig = MISSING
    
    # Direction strategy configuration (string in YAML, converted to Enum)
    direction_mode: str = "single"  # Default to single-direction for safety
    
    # Direction-specific parameters
    max_source_channels: int = MISSING  # Maximum number of source channels per sample
    
    # Common config parameters
    is_pretraining: bool = False
    is_finetuning: bool = False
    resume_training: bool = False
    
    # Checkpoint config parameters
    strict_loading: bool = True
    load_model_weights: bool = False
    # load_optimizer: bool = True
    # load_scheduler: bool = True
    # load_early_stopping: bool = True
    
    # Checkpoint path parameters (for CheckpointManager)
    # abs_path moved to CheckpointManagerConfig.base_dir
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    scheduler_patience: int = 3
    early_stopping_patience: int = 5
    use_patient_split: bool = False
    use_patient_information: bool = False
    use_wcl: bool = False
    save_checkpoint_frequency: int = 5 # Save checkpoint every 5 epochs
    overwrite_checkpoint: bool = False
    load_checkpoint_from_epoch: Optional[int] = None
    
    # Hardware configuration parameters (keep these - they're performance knobs)
    num_threads: int = 2
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    timeout: int = 3600
    seed: int = 42
    
    # Training pipeline configuration
    bp_norm: bool = False  # Whether to use BP normalization in data preprocessing
    
    # Data preprocessing configuration
    input_preprocessing: Dict[str, str] = MISSING
    
    # Debug mode settings
    debug: bool = False
    
    # Hydra run directory for storing logs and results
    hydra_run_dir: Optional[str] = None
    
    # Logging configuration (moved from common config)
    logging_level: str = "INFO"
    log_file_path: str = "training.log"

class BaseTrainer(ABC):
    """Base trainer class designed for torchrun-driven distributed training with simplified canonical checkpoint loading
    
    This trainer is designed to work with PyTorch's torchrun launcher, which automatically
    handles process spawning, environment variable setup, and DDP initialization.
    
    Key features:
    - Single entry point: run_training() method with proper checkpoint loading order
    - Automatic DDP setup via torchrun environment variables with auto-backend detection
    - Unified broadcast-based checkpoint loading for all scenarios (single-GPU and DDP)
    - Clean separation of concerns between infrastructure and training logic
    - Subclasses implement _execute_training_logic() for specific training behavior
    
    Checkpoint Loading Architecture:
    ===============================
    
    The trainer implements a clean, logical checkpoint loading approach with clear separation of concerns:
    
    1. **load_checkpoint_unified()**: Only loads checkpoint from disk and stores it (rank 0 only)
    2. **prepare_model_weights()**: Loads model weights from stored checkpoint (rank 0 only, before DDP wrapping)
    3. **load_trainer_states()**: Loads trainer states from stored checkpoint (rank 0 only, then broadcast) AND returns training metadata (epoch, best_loss) to ALL ranks
    
    This approach eliminates mixed concerns and unnecessary branching logic, providing:
    - Clear separation of responsibilities
    - Consistent behavior across single-GPU and DDP
    - Linear, logical flow that's easy to understand and maintain
    - Industry standard pattern that teams converge on
    """
    def __init__(self, trainer_name: str, model: nn.Module = None, criterion: BaseCriterion = None, 
                 checkpoint_manager: CheckpointManager = None, progress_bar: ProgressBar = None,
                 early_stopping: EarlyStopping = None, directions: Directions = None,
                 is_pretraining: bool = False, is_finetuning: bool = False, resume_training: bool = False,
                 strict_loading: bool = True, load_optimizer: bool = True,
                 load_scheduler: bool = True, load_early_stopping: bool = True,
                 batch_size: int = 32, num_epochs: int = 100,
                 learning_rate: float = 0.001, scheduler_patience: int = 10,
                 early_stopping_patience: int = 20, use_patient_split: bool = False,
                 num_threads: int = 0, num_workers: int = 0, pin_memory: bool = True,
                 persistent_workers: bool = True, prefetch_factor: int = 2, timeout: int = 3600,
                 seed: int = 42, load_model_weights: bool = False, direction_mode: str = "auto",
                 max_source_channels: int = 2, bp_norm: bool = False, debug: bool = False, 
                 hydra_run_dir: Optional[str] = None, logging_level: str = "INFO", 
                 log_file_path: str = "training.log", input_preprocessing: Dict[str, str] = None, 
                 use_patient_information: bool = False, use_wcl: bool = False, 
                 save_checkpoint_frequency: int = 5, overwrite_checkpoint: bool = False,
                 load_checkpoint_from_epoch: Optional[int] = None,
                 *args, **kwargs):
        
        super().__init__()
        
        self.trainer_name = trainer_name
        # Store individual config parameters
        self.is_pretraining = is_pretraining
        self.is_finetuning = is_finetuning
        self.resume_training = resume_training
        self.strict_loading = strict_loading
        self.load_optimizer = load_optimizer
        self.load_scheduler = load_scheduler
        self.load_early_stopping = load_early_stopping
        self.load_model_weights = load_model_weights
        self.direction_mode = DirectionMode.from_string(direction_mode)
        self.max_source_channels = max_source_channels
        self.bp_norm = bp_norm
        self.debug = debug
        self.hydra_run_dir = hydra_run_dir
        
        # Store preprocessing configuration
        self.input_preprocessing = input_preprocessing
        
        # Store logging configuration
        self.logging_level = logging_level
        self.log_file_path = log_file_path
        
        # Store checkpoint path parameters
        # abs_path moved to checkpoint_manager.base_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.early_stopping_patience = early_stopping_patience
        self.use_patient_split = use_patient_split
        self.save_checkpoint_frequency = save_checkpoint_frequency
        self.overwrite_checkpoint = overwrite_checkpoint
        self.load_checkpoint_from_epoch = load_checkpoint_from_epoch

        # Store hardware configuration parameters
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.seed = seed
        
        # Store other components
        self.model = model
        self.criterion = criterion
        self.checkpoint_manager = checkpoint_manager
        self.progress_bar = progress_bar
        self.early_stopping = early_stopping
        self.directions = directions
        
        # Add missing attributes for CheckpointManager compatibility
        self.use_patient_information = use_patient_information
        self.use_wcl = use_wcl
        
        # NEW: Initialize training state tracking - single source of truth
        self.training_metadata = {
            'epoch': 0,
            'best_loss': None,
            'global_step': 0
        }
        
        # Initialize hardware configuration
        self._setup_hardware_config()
        
        # Set seed for reproducibility
        self.set_seed()
        
        # Initialize metrics
        self.metrics = metrics
        
        # REMOVED: Redundant validation - already handled at config level
    

    def _get_model_name(self) -> str:
        """Get model name from the model object
        
        Returns:
            str: Model name (e.g., 'nabnet', 'ppg2abp', 'patchtst', 'mdvisco')
        """
        return self.unwrap(self.model).model_name
    
    def _get_model_supports_multi(self) -> bool:
        """Get model multi-directional support from the model object
        
        Returns:
            bool: Whether the model supports multi-directional training
        """
        return self.unwrap(self.model).supports_multi_directional
    
    def _get_dataset_name(self, dataset) -> str:
        """Extract dataset name from dataset or DataLoader
        
        Args:
            dataset: Dataset object or DataLoader for checkpoint path building
            
        Returns:
            str: Dataset name extracted from the underlying dataset
        """
        ds = dataset
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        if not hasattr(ds, "dataset_name"):
            raise AttributeError(f"{type(ds).__name__} lacks .dataset_name")
        # Handle case where dataset might be wrapped (e.g., Subset)
        return ds.dataset_name

    def get_direction_name(self) -> str:
        """Get the current direction name for single-direction training
        
        Returns:
            str: Direction name for single-direction training, or empty string for multi-directional
        """
        if len(self.directions.directions) > 1:
            return ""  # Multi-directional training
        elif len(self.directions.directions) == 1:
            return self.directions.directions[0].key()  # Single direction
        else:
            return ""  # No directions configured

    def _setup_hardware_config(self):
        """Setup hardware configuration - simplified for torchrun"""
        
        # Check if num_threads is properly configured
        if not hasattr(self, "num_threads") or self.num_threads is None or self.num_threads <= 0:
            raise ValueError(
                f"num_threads must be explicitly set to a positive integer. "
                f"Current value: {getattr(self, 'num_threads', 'Not set')}"
            )
        
        # Check if num_workers is properly configured
        if not hasattr(self, "num_workers") or self.num_workers is None or self.num_workers < 0:
            raise ValueError(
                f"num_workers must be explicitly set to a non-negative integer. "
                f"Current value: {getattr(self, 'num_workers', 'Not set')}"
            )
        
        logger.info(
            f"[HW] cpu={os.cpu_count()} threads={self.num_threads} "
            f"workers={self.num_workers} cuda={torch.cuda.is_available()} "
            f"gpus={torch.cuda.device_count()}"
        )

    def _validate_preprocessing_config(self, dataset):
        """Validate that requested preprocessing attributes exist in dataset
        
        Args:
            dataset: Dataset to validate preprocessing configuration against
            
        Raises:
            ValueError: If requested preprocessing attributes are not found
        """
        if not dataset:
            return
        
        # Get a sample to check attributes
        sample = dataset[0] if len(dataset) > 0 else None
        if not sample:
            return
        
        missing_attrs = []
        for output_key, sample_attr in self.input_preprocessing.items():
            if not hasattr(sample, sample_attr):
                missing_attrs.append(f"{output_key} -> {sample_attr}")
        
        if missing_attrs:
            available_attrs = [attr for attr in dir(sample) if not attr.startswith('_')]
            raise ValueError(
                f"Requested preprocessing attributes not found in dataset: {missing_attrs}\n"
                f"Available attributes: {available_attrs}"
            )

    
    def _get_distributed_config(self) -> tuple[int, int, int]:
        """Get rank, world_size, local_rank with priority to distributed module if initialized
        
        Priority order:
        1. If dist.is_initialized(): use dist.get_rank() and dist.get_world_size()
        2. Fallback to torchrun environment variables (RANK, WORLD_SIZE)
        3. Local rank always from torchrun environment (LOCAL_RANK)
        """
        # Try to get rank and world_size from distributed module first
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            # Check if torchrun environment variables exist
            if "RANK" not in os.environ:
                raise RuntimeError("RANK environment variable not found. Are you running with torchrun?")
            if "WORLD_SIZE" not in os.environ:
                raise RuntimeError("WORLD_SIZE environment variable not found. Are you running with torchrun?")
            
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        
        # Check if LOCAL_RANK exists
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("LOCAL_RANK environment variable not found. Are you running with torchrun?")
        
        local_rank = int(os.environ["LOCAL_RANK"])
        
        return rank, world_size, local_rank

    def print_memory_stats(self, location: str):
        """Print GPU memory statistics for current device
        
        Args:
            location: String describing where in the code this is called
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self._get_device()) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(self._get_device()) / (1024 * 1024)
            rank, _, _ = self._get_distributed_config()
            logger.info(f"Rank {rank} at {location}:")
            logger.info(f"Allocated: {allocated:.2f}MB")
            logger.info(f"Cached: {cached:.2f}MB")

    def _setup_cpu_environment(self):
        """Setup CPU and threading environment - respect existing env vars"""
        # Set PyTorch threads
        torch.set_num_threads(self.num_threads)
        
        # Thread envs (only set if not already set by user)
        for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            os.environ.setdefault(k, str(self.num_threads))

    def _get_device(self) -> torch.device:
        """Determine the appropriate device based on distributed config and backend
        
        Returns:
            torch.device: The device to use (CPU or specific CUDA device)
            
        Raises:
            RuntimeError: If there's a CUDA device configuration error
        """
        rank, world_size, local_rank = self._get_distributed_config()
        
        # Use _get_appropriate_backend to determine device type
        backend = self._get_appropriate_backend()
        if backend == "gloo":  # CPU backend
            return torch.device("cpu")
        
        # GPU backend (nccl) - torchrun has already validated LOCAL_RANK
        # Minimal sanity check (fast, local, catches common mislaunches)
        if torch.cuda.current_device() != local_rank:
            logger.error(f"CUDA device mismatch: current_device={torch.cuda.current_device()} != LOCAL_RANK={local_rank}")
            raise RuntimeError("CUDA device configuration error")

        return torch.device(f"cuda:{local_rank}")

    def _set_device(self):
        """Set the device using _get_device and configure CUDA settings"""
        device = self._get_device()
        
        if device.type == "cuda":
            # Set CUDA device and configure
            torch.cuda.set_device(device)
            logger.info(f"[DEVICE] {device}")
        else:
            # CPU device
            logger.info("[DEVICE] CPU")
        
        self.device = device

    def set_seed(self):
        """Set random seeds for reproducibility - modern PyTorch approach"""
        if self.seed is None:
            raise ValueError("Seed is not set")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Strong determinism (may disable some ops / slow down)
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"[SEED] Set to {self.seed}")
    
    def setup_logging(self):
        """Setup logging configuration using trainer config"""
        logging.basicConfig(
            level=getattr(logging, self.logging_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file_path)
            ]
        )
        logger.info(f"Logging configured: level={self.logging_level}, file={self.log_file_path}")
    
    # Properties that use environment truth, not parsed config
    @property
    def is_rank0(self) -> bool:
        """Check if this is the rank-0 process (master)"""
        rank, world_size, local_rank = self._get_distributed_config()
        return rank == 0

    def _get_appropriate_backend(self):
        """Auto-detect and return appropriate backend for current hardware
        
        This method automatically selects the optimal backend based on available hardware,
        eliminating the need for manual backend configuration and ensuring optimal
        performance across different environments.
        
        Returns:
            str: 'nccl' for GPU, 'gloo' for CPU
        """
        if torch.cuda.is_available():
            return "nccl"  # GPU backend
        else:
            return "gloo"  # CPU backend

    def init_ddp(self):
        """Simplified DDP initialization with auto-backend detection - always succeeds
        
        This method always initializes DDP regardless of world size, with backend
        automatically selected based on hardware (GPU: nccl, CPU: gloo).
        """
        if dist.is_initialized():
            logger.warning("[DDP] Process group already initialized")
            return
            
        # Always initialize DDP - backend selection handles hardware differences
        backend = self._get_appropriate_backend()
        logger.info(f"[DDP] Initializing process group: backend={backend}")
        
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",  # torchrun sets this
                timeout=datetime.timedelta(minutes=30),
            )
            logger.info("[DDP] Process group initialized successfully")
        except Exception as e:
            logger.error(f"[DDP] Failed to initialize process group: {e}")
            raise

    def cleanup_ddp(self):
        """Cleanup DDP process group with barrier for clean shutdown - handles all scenarios"""
        if not dist.is_available():
            logger.warning("[DDP] Distributed package not available")
            return
            
        if not dist.is_initialized():
            logger.info("[DDP] No process group to cleanup")
            return
            
        try:
            logger.info("[DDP] Flushing outstanding collectives with barrier")
            dist.barrier()  # Flush any pending collectives
        except Exception as e:
            logger.warning(f"[DDP] Barrier failed (non-critical): {e}")
        
        logger.info("[DDP] Destroying process group")
        dist.destroy_process_group()
        logger.info("[DDP] Process group cleanup completed")

    def create_optimizer(self):
        """Create optimizer - called AFTER DDP wrapping, works on every rank"""
        # Model is wrapped, use .module to access original model
        self.optimizer = torch.optim.Adam(self.model.module.parameters(), lr=self.learning_rate)
        
    def create_scheduler(self):
        """Create scheduler for the optimizer - works on every rank"""
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=self.scheduler_patience
        )

    def sync_metric_for_scheduler(self, metric: float) -> float:
        """All-reduce metric across ranks for scheduler stepping"""
        # All-reduce the metric to ensure all ranks have the same value
        metric_t = torch.tensor([metric], device=self.device, dtype=torch.float32)
        torch.distributed.all_reduce(metric_t, op=torch.distributed.ReduceOp.SUM)
        synced_metric = (metric_t / torch.distributed.get_world_size()).item()
        
        return synced_metric

    def unwrap(self, model):
        """Helper method to unwrap DDP-wrapped models for state_dict access"""
        if hasattr(model, 'module'):
            return model.module
        return model

    def step_scheduler(self, metric: float):
        """Step scheduler with synchronized metric - handles both types of schedulers"""
        synced_metric = self.sync_metric_for_scheduler(metric)
        
        # Handle both ReduceLROnPlateau and other schedulers
        if hasattr(self.scheduler, 'step') and callable(self.scheduler.step):
            if hasattr(self.scheduler, '_step_count'):  # StepLR, CosineAnnealingLR, etc.
                self.scheduler.step()
            else:  # ReduceLROnPlateau
                self.scheduler.step(synced_metric)
        else:
            # Handle dictionary of schedulers (for GANs)
            if isinstance(self.scheduler, dict):
                for name, sched in self.scheduler.items():
                    if hasattr(sched, 'step') and callable(sched.step):
                        if hasattr(sched, '_step_count'):
                            sched.step()
                        else:
                            sched.step(synced_metric)
        
        if self.is_rank0:
            logger.info(f"Scheduler stepped with metric: {synced_metric:.6f}")

    def _worker_init_fn(self, worker_id: int):
        """Initialize worker with unique seed for reproducibility"""
        # Base seed + rank*1000 + worker_id for unique seeds per worker
        base_seed = self.seed
        rank, _, _ = self._get_distributed_config()
        worker_seed = base_seed + rank * 1000 + worker_id
        
        # Set seeds for this worker
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        # Set deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_dataloader_settings(self) -> dict:
        """Get DataLoader settings - ensure PyTorch-safe combinations with worker initialization"""
        use_cuda = torch.cuda.is_available()
        nw = max(0, int(getattr(self, "num_workers", 0)))
        
        settings = {
            "num_workers": nw,
            # Only enable pin_memory when using CUDA
            "pin_memory": bool(use_cuda),
            "persistent_workers": bool(nw > 0),
            "timeout": 0 if nw == 0 else 3600,
        }
        
        # prefetch_factor is only valid when num_workers > 0
        pf = getattr(self, "prefetch_factor", 2)
        if nw > 0 and pf is not None:
            settings["prefetch_factor"] = int(pf)
        
        # Pin memory device for PyTorch >=2.0 (only when CUDA)
        if use_cuda:
            settings["pin_memory_device"] = "cuda"
        
        # Add worker initialization function for proper seeding
        if nw > 0:
            settings["worker_init_fn"] = self._worker_init_fn
        
        return settings
    
    # ============================================================================
    # DATALOADER BUILDING METHODS - Shared across all trainers
    # ============================================================================
    
    def _get_vitals_dataset(self, dataset):
        """Safely extract vitals_dataset from any dataset, handling Subset wrappers."""
        from torch.utils.data import Subset
        
        if isinstance(dataset, Subset):
            return dataset.dataset.vitals_dataset
        return dataset.vitals_dataset

    def _build_dataloaders(self, dataset_tuple):
        """Build dataloaders with direction-aware collate function and proper DDP support
        
        This method creates dataloaders that handle both preprocessing and direction
        logic at the collate level, following industry standards for optimal performance.
        """
        train_dataset, val_dataset, test_dataset = dataset_tuple
        
        # Validate preprocessing configuration against dataset
        self._validate_preprocessing_config(train_dataset)
        
        # Get centralized loader settings
        loader_settings = self.get_dataloader_settings()
        
        # Create direction-aware collate function ONCE
        collate_fn_train = create_direction_aware_collate_fn(
            input_preprocessing=self.input_preprocessing,
            directions=self.directions,
            vitals_dataset=self._get_vitals_dataset(train_dataset),
            direction_mode=self.direction_mode,
            max_source_channels=self.max_source_channels
        )

        collate_fn_val = create_direction_aware_collate_fn(
            input_preprocessing=self.input_preprocessing,
            directions=self.directions,
            vitals_dataset=self._get_vitals_dataset(val_dataset),
            direction_mode=self.direction_mode,
            max_source_channels=self.max_source_channels
        )

        collate_fn_test = create_direction_aware_collate_fn(
            input_preprocessing=self.input_preprocessing,
            directions=self.directions,
            vitals_dataset=self._get_vitals_dataset(test_dataset),
            direction_mode=self.direction_mode,
            max_source_channels=self.max_source_channels
        )
        
        # DDP dataloaders with DistributedSampler
        train_sampler = self.create_distributed_sampler(
            train_dataset, shuffle=True, drop_last=True
        )
        val_sampler = self.create_distributed_sampler(
            val_dataset, shuffle=False
        )
        test_sampler = self.create_distributed_sampler(
            test_dataset, shuffle=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn_train,  # Use direction-aware collate
            **loader_settings
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn_val,  # Use direction-aware collate
            **loader_settings
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn_test,  # Use direction-aware collate
            **loader_settings
        )
        
        # Store samplers for epoch setting
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        
        return train_loader, val_loader, test_loader

    def create_distributed_sampler(self, dataset, shuffle: bool = True, drop_last: bool = False) -> DistributedSampler:
        """Create DistributedSampler with fixed seed for perfect resume reproducibility
        
        Args:
            dataset: Dataset to sample from
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            DistributedSampler: Configured sampler with fixed seed
        """
        if self.seed is None:
            raise ValueError("Seed is not set")
        
        sampler = DistributedSampler(
            dataset, 
            shuffle=shuffle, 
            drop_last=drop_last,
            seed=self.seed  # Fixed seed for perfect reproducibility
        )
        return sampler
    
    # REMOVED: Redundant validation method - validation handled at config level
    


    # ============================================================================
    # COMMON CHECKPOINT LOADING METHODS - Shared across all trainers
    # ============================================================================
    
    def _strip_module_prefix(self, state_dict):
        """Strip 'module.' prefix from state dict keys (handles DDP checkpoints)
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            dict: State dictionary with 'module.' prefix removed
        """
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    def _normalize_state_dict_keys(self, state_dict, additional_prefixes=None):
        """Normalize state dict keys by removing common prefixes
        
        Args:
            state_dict: Model state dictionary
            additional_prefixes: List of additional prefixes to remove (e.g., ['backbone.', 'encoder.'])
            
        Returns:
            dict: State dictionary with prefixes removed
        """
        prefixes = ['module.']  # Default DDP prefix
        if additional_prefixes:
            prefixes.extend(additional_prefixes)
        
        result = state_dict
        for prefix in prefixes:
            result = {k.replace(prefix, ''): v for k, v in result.items()}
        return result

    def _log_state_dict_warnings(self, missing_keys, unexpected_keys, strict_loading):
        """Log warnings about state dict key mismatches
        
        Args:
            missing_keys: Keys missing from model
            unexpected_keys: Keys not expected by model
            strict_loading: Whether strict loading was used
        """
        if missing_keys:
            logger.warning(f"Warning: Missing keys: {len(missing_keys)} keys")
            logger.warning(f"First few missing keys: {missing_keys[:5]}")
            if strict_loading:
                logger.warning("Note: Strict loading was enabled - consider setting checkpoint_strict_loading=False")
        
        if unexpected_keys:
            logger.warning(f"Warning: Unexpected keys: {len(unexpected_keys)} keys")
            logger.warning(f"First few unexpected keys: {unexpected_keys[:5]}")

    def on_checkpoint_loaded(self, checkpoint: dict):
        """
        Optional hook called after checkpoint is loaded. Subclasses can override this
        to implement custom post-load logic (e.g., logging, additional state restoration).
        """
        pass

    def to_device(self, obj):
        """Move object to device robustly, handling nested structures"""
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(x) for x in obj)
        return obj



    def save_checkpoint(self, epoch: int, dataset, 
                       model_state_dict: Dict[str, Dict] = None, optimizer_state_dict: Dict[str, Dict] = None,
                       scheduler_state_dict: Dict = None, early_stopping_state: Dict = None, 
                       additional_info: Dict = None, **kwargs):
        """Save checkpoint with rank-0 write + all-rank sync.
        
        Args:
            epoch: Current training epoch (None for best model checkpoints)
            dataset: Dataset or DataLoader for checkpoint path building (typically train_loader)
            model_state_dict: Dict of model state dicts to save, e.g., {'model': state_dict} or {'G': state_dict, 'D': state_dict}
            optimizer_state_dict: Dict of optimizer state dicts to save, e.g., {'optimizer': state_dict} or {'G': state_dict, 'D': state_dict}
            scheduler_state_dict: Scheduler state dict to save (optional)
            early_stopping_state: Early stopping state dict to save (optional)
            additional_info: Additional information to save in checkpoint (optional)
            **kwargs: Additional keyword arguments (passed to additional_info)
            
        Returns:
            str: Path to saved checkpoint (rank>0 returns None)
        """
        checkpoint_path = None
        
        try:
            # Save on rank-0 only
            if self.is_rank0:
                # Build checkpoint path using CheckpointManager
                checkpoint_path = self.checkpoint_manager.build_path(
                    model_name=self._get_model_name(),
                    trainer_name=self.trainer_name,
                    dataset_name=self._get_dataset_name(dataset),
                    epoch=epoch,
                    batch_size=self.batch_size,
                    num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    scheduler_patience=self.scheduler_patience,
                    early_stopping_patience=self.early_stopping_patience,
                    is_finetuning=self.is_finetuning,
                    use_patient_split=self.use_patient_split,
                    use_patient_information=self.use_patient_information,
                    seed=self.seed,
                    direction=self.get_direction_name(),
                    use_wcl=self.use_wcl
                )
                
                # Check if checkpoint already exists and handle according to overwrite_checkpoint setting
                from src.utils.utils_preprocessing import check_file_exists
                
                try:
                    check_file_exists(checkpoint_path, overwrite=self.overwrite_checkpoint)
                except FileExistsError:
                    logger.warning(f"Checkpoint already exists at {checkpoint_path}. Skipping save due to overwrite_checkpoint={self.overwrite_checkpoint}")
                    return None  # Return None to indicate no checkpoint was saved
                
                # Log trainer metadata during save
                logger.info(f"Saving checkpoint to path: {checkpoint_path}")
                
                # Prepare checkpoint data with enhanced metadata
                checkpoint = {
                    'epoch': self.get_epoch()+1,
                    'timestamp': datetime.datetime.now().isoformat(),
                    # NEW: Include all training metadata
                    'flags': self.training_metadata.copy(),
                    'best_loss': self.get_best_loss()
                }
                
                # Add all model state dicts
                if model_state_dict is not None:
                    for model_name, state_dict in model_state_dict.items():
                        checkpoint[f'{model_name}_state_dict'] = state_dict
                
                # Add all optimizer state dicts
                if optimizer_state_dict is not None:
                    for opt_name, state_dict in optimizer_state_dict.items():
                        checkpoint[f'{opt_name}_state_dict'] = state_dict
                
                # Add optional components
                if scheduler_state_dict:
                    checkpoint['scheduler_state_dict'] = scheduler_state_dict
                if early_stopping_state:
                    checkpoint['early_stopping_state'] = early_stopping_state
                if additional_info:
                    checkpoint.update(additional_info)
                
                # Save checkpoint
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved successfully: {checkpoint_path}")
            
            # All ranks must participate in the barrier or we risk a hang
            torch.distributed.barrier()
                
        except Exception as e:
            logger.error(f"Checkpoint saving failed: {e}")
            raise
        
        return checkpoint_path  # rank>0 will return None (fine)
    

    
    def load_checkpoint_unified(self, dataset):
        """Load checkpoint from disk and store it for later use, then broadcast status to all ranks
        
        This method handles the actual file I/O, stores the checkpoint, and ensures all ranks
        know whether a checkpoint was loaded successfully.
        
        Args:
            dataset: Dataset for checkpoint path discovery
            **kwargs: Additional arguments for checkpoint discovery (unused, kept for compatibility)
            
        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise (ALL ranks get this)
        """
        # Only rank 0 loads checkpoint from disk
        checkpoint_loaded = False
        if self.is_rank0:
            # Load checkpoint from disk using class attributes
            checkpoint = self._load_checkpoint_from_disk(dataset)
            
            # Store checkpoint for later usage
            self._stored_checkpoint = checkpoint
            
            checkpoint_loaded = True
            logger.info("Rank 0: Checkpoint loaded and stored successfully")
            
        
        # Broadcast checkpoint_loaded status to all ranks
        checkpoint_loaded_tensor = torch.tensor([checkpoint_loaded], dtype=torch.bool, device=self._get_device())
        dist.broadcast(checkpoint_loaded_tensor, src=0)
        checkpoint_loaded = checkpoint_loaded_tensor.item()
        logger.info(f"Device {self._get_device()}: Checkpoint loaded status: {checkpoint_loaded}")
        
        return checkpoint_loaded
    
    def _load_checkpoint_from_disk(self, dataset):
        """Load checkpoint from disk on rank-0 only
        
        This method handles the actual file I/O and returns the checkpoint dictionary.
        It's called only on rank-0 and handles checkpoint discovery and loading.
        """
        # Find checkpoint path using class attributes
        checkpoint_path = self.checkpoint_manager.find_checkpoint(
            model_name=self._get_model_name(),
            trainer_name=self.trainer_name,
            dataset_name=self._get_dataset_name(dataset),
            epoch=self.load_checkpoint_from_epoch,  # Default epoch for checkpoint discovery
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            scheduler_patience=self.scheduler_patience,
            early_stopping_patience=self.early_stopping_patience,
            is_finetuning=self.is_finetuning,
            use_patient_split=self.use_patient_split,
            use_patient_information=self.use_patient_information,
            seed=self.seed,
            direction=self.get_direction_name(),
            use_wcl=self.use_wcl
        )
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found")
        
        logger.info(f"Rank 0: Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint on CPU to avoid device issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        logger.info(f"Rank 0: Checkpoint loaded successfully")
        return checkpoint

    def prepare_model_weights(self, models):
        """Public interface: DDP-aware coordination for model weight loading
        
        This method coordinates the loading of model weights from stored checkpoint on rank 0 only,
        before DDP wrapping. DDP will automatically broadcast these weights to all ranks during wrapping.
        
        Args:
            models: Dict of models to load weights into
        """
        if not hasattr(self, '_stored_checkpoint') or self._stored_checkpoint is None:
            logger.warning("No stored checkpoint available for model weight loading")
            return
            
        if not self.is_rank0:
            return
            
        # Load model weights on rank 0 only
        self._apply_model_weights(self._stored_checkpoint, models)
        logger.info("Rank 0: Model weights loaded (will be broadcast by DDP)")
    
    def _apply_model_weights(self, checkpoint, models):
        """Private implementation: applies checkpoint state dict to models
        
        This is the ONLY place where model weights are actually loaded into models.
        DDP will automatically broadcast these weights to all ranks during wrap.
        """
        if not checkpoint:
            return
        
        for model_name, model in models.items():
            checkpoint_key = f'{model_name}_state_dict'
            if checkpoint_key in checkpoint:
                state_dict = checkpoint[checkpoint_key]
                new_state_dict = self._strip_module_prefix(state_dict)
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=self.strict_loading)
                
                # Log warnings about keys
                self._log_state_dict_warnings(missing_keys, unexpected_keys, self.strict_loading)
                logger.info(f"Rank 0: {model_name} weights loaded (will be broadcast by DDP)")
            else:
                logger.warning(f"Rank 0: No {checkpoint_key} found in checkpoint")

    def load_trainer_states(self):
        """Load trainer states from stored checkpoint (rank 0 only, then broadcast)
        
        This method loads trainer states from the stored checkpoint on rank 0,
        then broadcasts them to all other ranks. Call this AFTER DDP wrapping.
        
        Returns:
            dict: Training metadata (epoch, best_loss) that ALL ranks now have access to
        """
        if not hasattr(self, '_stored_checkpoint') or self._stored_checkpoint is None:
            logger.warning("No stored checkpoint available for trainer state loading")
            return None
                
        # Pack trainer state payload on rank 0
        if self.is_rank0:
            trainer_payload = self._pack_trainer_payload(
                self._stored_checkpoint
            )
            
            # Broadcast payload to all ranks
            obj_list = [trainer_payload]
            dist.broadcast_object_list(obj_list, src=0)
            logger.info("Rank 0: Trainer state payload broadcasted")
            
            # Load on rank 0
            self._load_trainer_states_from_payload(trainer_payload)
        else:
            # Receive payload on other ranks
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            trainer_payload = obj_list[0]
            
            # Load from payload
            self._load_trainer_states_from_payload(trainer_payload)
            logger.info(f"Device {self._get_device()}: Trainer state loaded from broadcast payload")
        
        # Clean up stored checkpoint
        delattr(self, '_stored_checkpoint')

    def _pack_trainer_payload(self, checkpoint):
        """Pack only trainer state into payload (not model weights)"""
        payload = {}
        
        # Pack optimizer states
        if self.optimizer:
            payload['optimizers'] = {'optimizer': checkpoint.get('optimizer_state_dict')}
        
        # Pack scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            payload['scheduler'] = checkpoint['scheduler_state_dict']
        
        # Pack early stopping state
        if self.early_stopping and 'early_stopping_state' in checkpoint:
            payload['early_stopping'] = checkpoint['early_stopping_state']
        
        # Pack simple flags (use tensor broadcast for efficiency)
        payload['flags'] = {
            'epoch': checkpoint.get('epoch', 0),
            'best_loss': checkpoint.get('best_loss', None)
        }
        
        return payload

    def _load_trainer_states_from_payload(self, payload):
        """Load trainer states from payload on any rank"""
        # Load optimizer states
        if self.optimizer and 'optimizers' in payload:
            for opt_name, optimizer_state in payload['optimizers'].items():
                if opt_name == 'optimizer' and optimizer_state is not None:
                    self.optimizer.load_state_dict(optimizer_state)
                    logger.info(f"Loaded optimizer state: {opt_name}")
        
        # Load scheduler state
        if self.scheduler and 'scheduler' in payload:
            self.scheduler.load_state_dict(payload['scheduler'])
            logger.info("Loaded scheduler state")
        
        # Load early stopping state
        if self.early_stopping and 'early_stopping' in payload:
            self.early_stopping.load_state_dict(payload['early_stopping'])
            logger.info("Loaded early stopping state")
        
        # NEW: Store training metadata flags
        if 'flags' in payload:
            # Load all training state into metadata dict
            self.training_metadata.update(payload['flags'])
            logger.info(f"Loaded training metadata: {self.training_metadata}")

    def get_training_metadata(self):
        """Get current training metadata - single source of truth
        
        Returns:
            dict: Current training state
        """
        return self.training_metadata

    def update_training_state(self, epoch: int, best_loss: float = None):
        """Update training state in metadata dict
        
        Args:
            epoch: Current epoch number
            best_loss: New best loss value (optional)
            
        Returns:
            bool: True if best_loss was improved, False otherwise
        """
        self.training_metadata['epoch'] = epoch
        if best_loss is not None:
            # Use set_best_loss to properly check if it's an improvement
            return self.set_best_loss(best_loss)
        return False

    def get_epoch(self) -> int:
        """Get current epoch from metadata"""
        return self.training_metadata.get('epoch')

    def get_best_loss(self) -> Optional[float]:
        """Get best loss from metadata"""
        return self.training_metadata.get('best_loss')

    def set_best_loss(self, loss: float) -> bool:
        """Set best loss if improved, returns True if improved
        
        Args:
            loss: New loss value
        
        Returns:
            bool: True if this is a new best loss
        """
        current_best = self.training_metadata.get('best_loss')
        if current_best is None or loss < current_best:
            self.training_metadata['best_loss'] = loss
            self.training_metadata['best_loss_epoch'] = self.training_metadata.get('epoch')
            return True
        return False


    def _validate_early_stopping(self):
        """Validate early stopping is properly initialized
        
        This method checks if the early stopping object exists and is properly
        initialized before attempting to use it. This prevents runtime errors
        when early stopping is not configured.
        
        Returns:
            bool: True if early stopping is properly initialized, False otherwise
        """
        if not hasattr(self, 'early_stopping') or self.early_stopping is None:
            logger.warning("Early stopping not initialized - skipping early stopping checks")
            return False
        return True

    def check_early_stopping_common(self):
        """Check early stopping with proper DDP synchronization and error handling
        
        This method should be called AFTER updating early stopping state:
        1. First call: self.early_stopping(val_metric)  # Update state
        2. Then call: self.check_early_stopping_common()  # Check state
        
        Example usage in training loop:
        ```python
        for epoch in range(num_epochs):
            # ... training and validation ...
            
            # Update early stopping state with validation metric
            if val_metric is not None and self.early_stopping:
                self.early_stopping(val_metric)
            
            # Check if early stopping should trigger
            if self.check_early_stopping_common():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        ```
        
        Returns:
            bool: Whether early stopping should be triggered
            
        Raises:
            RuntimeError: If early stopping is not properly initialized
        """
        if not self._validate_early_stopping():
            return False
        
        try:
            should_stop = False
            # Make decision on rank-0
            if self.is_rank0:
                should_stop = self.early_stopping.early_stop
                if should_stop:
                    logger.info("Early stopping triggered on rank 0")
            
            # Broadcast decision to all ranks
            should_stop_tensor = torch.tensor([should_stop], device=self._get_device(), dtype=torch.bool)
            torch.distributed.broadcast(should_stop_tensor, src=0)
            
            return should_stop_tensor.item()
        except Exception as e:
            logger.error(f"Early stopping check failed: {e}")
            return False

    # ============================================================================
    # UNIFIED LOGGING METHODS - New unified logging system
    # ============================================================================
    
    def log_epoch_metrics_unified(self, epoch, train_metrics, val_metrics, test_metrics, best_loss, loss_improved):
        """Enhanced epoch logging with automatic metric organization - rank-0 only
        
        Automatically logs all provided metrics with appropriate prefixes:
        - train/metric_name for training metrics
        - val/metric_name for validation metrics  
        - test/metric_name for test metrics
        
        No metric registration required - all computed metrics are logged automatically.
        """
        if not self.is_rank0:
            return  # Only rank-0 logs to W&B/TensorBoard
        
        if self.progress_bar.wandb._is_ready():
            try:
                log_dict = {}
                
                # Log ALL train metrics with simple prefixing
                for metric_name, metric_value in train_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                        log_dict[f"train/{metric_name}"] = float(metric_value)
                
                # Log ALL val metrics with simple prefixing  
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                        log_dict[f"val/{metric_name}"] = float(metric_value)
                
                # Log ALL test metrics with simple prefixing
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                        log_dict[f"test/{metric_name}"] = float(metric_value)
                
                # Add standard metrics
                log_dict.update({
                    "epoch": epoch,
                    "learning_rate": float(self.optimizer.param_groups[0]['lr']),
                    "best_loss": float(best_loss or 0.0),
                    "improved": loss_improved if loss_improved is not None else False,
                })
                
                # Add early stopping metrics if available
                if hasattr(self, 'early_stopping') and self.early_stopping:
                    log_dict.update({
                        "early_stopping/counter": int(getattr(self.early_stopping, 'counter', 0)),
                        "early_stopping/best_loss": float(getattr(self.early_stopping, 'best_loss', 0.0)),
                        "early_stopping/patience": int(getattr(self.early_stopping, 'patience', 0))
                })
                
                # Log to WandB
                self.progress_bar.wandb.log(log_dict, is_rank0=self.is_rank0)
                
            except Exception as e:
                print(f"Failed to log to wandb: {e}")
    
    def log_step_metrics_unified(self, metrics_dict: Dict[str, Any], step: int):
        """Unified step metrics logging - simplified and focused
        
        This method handles ONLY the core metric logging infrastructure:
        - Logs all provided metrics to the metrics system
        - External logging (WandB, CSV) is handled by the progress bar
        
        DESIGN PATTERN:
        ===============
        This method follows a simplified approach:
        1. All metrics are computed in _step_core before calling this method
        2. No hooks or additional processing needed
        3. Each trainer is responsible for computing its own metrics in _step_core
        4. Progress bar handles all external logging with proper stage prefixes
        
        Args:
            metrics_dict: Dictionary containing all metrics to log (computed in _step_core)
            step: Current global step number
        """
        # Log all metrics to the metrics system
        for name, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                self.metrics.log_scalar(name, value)
        
        # Note: External logging (WandB, CSV) is now handled by the progress bar


    
    def create_training_progress_bar(self, train_loader, epoch, master_process=True):
        """Create unified training progress bar"""
        self.progress_bar.create_bar(
            total=len(train_loader),
            description=f"Train - Epoch {epoch}",
            disable=not master_process
        )
    
    def create_validation_progress_bar(self, val_loader, epoch, master_process=True):
        """Create unified validation progress bar"""
        self.progress_bar.create_bar(
            total=len(val_loader),
            description=f"Val - Epoch {epoch}",
            disable=not master_process
        )
    
    def create_test_progress_bar(self, test_loader, epoch, master_process=True):
        """Create unified test progress bar"""
        self.progress_bar.create_bar(
            total=len(test_loader),
            description=f"Test - Epoch {epoch}",
            disable=not master_process
        )
    
    def update_progress_bar(self, metrics_dict=None, step=None, is_rank0: bool = False, stage: str = None, to_log: bool = False):
        """Update progress bar with stage awareness
    
        Args:
            metrics_dict: Dictionary of metrics to display and log
            step: Current step number  
            is_rank0: Whether this is the rank 0 process
            stage: Stage name for metric prefixing and step management ("train", "val", or "test")
        """
        if metrics_dict and stage:
            # Pre-prefix metrics with stage name for proper WandB organization
            prefixed_metrics = {f"{stage}/{k}": v for k, v in metrics_dict.items()}
            self.progress_bar.update(metrics_dict=prefixed_metrics, step=step, is_rank0=is_rank0, to_log=to_log)
        else:
            self.progress_bar.update(metrics_dict=metrics_dict, step=step, is_rank0=is_rank0, to_log=to_log)
    
    def close_progress_bar(self):
        """Close progress bar"""
        self.progress_bar.close_progress_bar()
    
    def close_wandb(self):
        """Close WandB"""
        self.progress_bar.close_wandb()
    
    def close_csv(self):
        """Close CSV"""
        self.progress_bar.close_csv()
    
    def close_all(self):
        """Close all loggers"""
        self.progress_bar.close()
    
    def set_sampler_epoch(self, epoch: int):
        """Set epoch for DistributedSampler with proper synchronization"""
        if hasattr(self, 'train_sampler') and self.train_sampler is not None:
            # Only set epoch for training sampler (not validation)
            self.train_sampler.set_epoch(epoch)
            logger.debug(f"Set training sampler epoch to {epoch}")
        
        # Note: Do NOT call set_epoch on validation sampler

    # ============================================================================
    # EXISTING METHODS (keeping for backward compatibility)
    # ============================================================================
        
    def cleanup_datasets(self, dataset_tuple: Tuple[Any, Any, Any]):
        """Cleanup datasets using polymorphism"""
        if dataset_tuple:
            train_dataset, val_dataset, test_dataset = dataset_tuple
            for dataset in [train_dataset, val_dataset, test_dataset]:
                if dataset is not None and hasattr(dataset, 'cleanup_shared_memory'):
                    try:
                        dataset.cleanup_shared_memory()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup dataset {type(dataset).__name__}: {e}")
    
    def setup_hardware(self):
        """Setup hardware environment - returns device for convenience"""
        self._setup_hardware_config()
        self._setup_cpu_environment()
        self._set_device()
        
        # Setup Hydra environment with rank awareness
        self.setup_hydra_environment()
        
        # Set seed for reproducibility
        self.set_seed()
        
        return self._get_device()
    
    def cleanup_resources(self, dataset_tuple: Tuple[Any, Any, Any]):
        """Cleanup resources after training"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Cleanup DDP if needed
        self.cleanup_ddp()

        # Cleanup datasets
        self.cleanup_datasets(dataset_tuple)
        
    def setup_hydra_environment(self):
        """Setup Hydra environment with rank awareness"""
        rank, _, _ = self._get_distributed_config()
        # Set Hydra output directory with rank suffix
        os.environ["HYDRA_OUTPUT_DIR"] = f"./outputs/rank_{rank}"
            
        # Set logging file with rank suffix (or gate to rank-0 only)
        if self.is_rank0:
            log_file = f"./logs/training_rank_{rank}.log"
            os.environ["LOG_FILE"] = log_file
        else:
            # Non-rank-0 processes don't write to files
            os.environ["LOG_FILE"] = "/dev/null"

    def log_to_wandb(self, metrics: dict):
        """Log to WandB only on rank-0"""
        if not self.is_rank0:
            return
        
        try:
            self.progress_bar.wandb.log(metrics)
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")

    def log_split_statistics(self, train_indices: List[int], val_indices: List[int], 
                           test_indices: List[int], total_size: int, stage_name: str, 
                           split_type: str = "Random", unique_subjects: Optional[int] = None):
        """Log dataset split statistics
        
        Args:
            train_indices: Training indices
            val_indices: Validation indices
            test_indices: Test indices
            total_size: Total dataset size
            stage_name: Name of the training stage
            split_type: Type of split used
            unique_subjects: Number of unique subjects (for patient-level splits)
        """
        logger.info(f"\n{stage_name} - {split_type} Split Statistics:")
        logger.info(f"Total samples: {total_size}")
        logger.info(f"Train samples: {len(train_indices)} ({len(train_indices)/total_size*100:.1f}%)")
        logger.info(f"Val samples: {len(val_indices)} ({len(val_indices)/total_size*100:.1f}%)")
        logger.info(f"Test samples: {len(test_indices)} ({len(test_indices)/total_size*100:.1f}%)")
        if unique_subjects:
            logger.info(f"Unique subjects: {unique_subjects}")
    
    
    def _setup_ddp_wrapping(self, model, local_rank: int):
        """Setup DDP wrapping - can be overridden by subclasses for special handling"""
        # Default DDP wrapping
        self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=True)

    def _move_model_to_device(self):
        """Move model to device - can be overridden by subclasses for special handling
        
        This method handles the basic case of moving a single model to device.
        Subclasses like GANTrainer can override this to handle multiple models
        (e.g., generator and discriminator).
        
        Args:
            model: The model to move to device
            
        Returns:
            The model moved to the appropriate device
        """
        if self.model is not None:
            if self.is_rank0:
                logger.info(f"Moving model to device: {self._get_device()}")
            # Use the existing unwrap method
            self.unwrap(self.model).to(self._get_device())
    
    @abstractmethod
    def _step_core(self, model, batch, *, stage: str):
        """Pure compute: parse, forward, target, loss, metrics. No optimizer/AMP.
        
        This is the core training step that varies significantly between trainers.
        Each trainer must implement its own forward pass logic, loss computation,
        and metrics calculation.
        
        Args:
            model: The model (DDP-wrapped or not)
            batch: Batch dictionary with unified structure
            stage: Training stage ("train", "val", "test")
            
        Returns:
            loss: Computed loss (torch.Tensor)
            metrics: Dictionary of metrics (Dict[str, float])
            outputs: Model outputs (Any - depends on model type)
        """
        pass

    @abstractmethod
    def _extract_target_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract target from unified batch structure for loss computation.
        
        Different trainers work with different data structures and target formats.
        This method ensures each trainer can properly extract its required targets.
        
        Args:
            batch: Batch dictionary with unified structure
            
        Returns:
            torch.Tensor: Target tensor for loss computation
        """
        pass

    @abstractmethod
    def _get_main_output(self, outputs) -> torch.Tensor:
        """Handle different model output formats for consistent processing.
        
        Different models return different output structures (tensors, tuples, dicts).
        This method ensures each trainer can properly extract the main output
        for loss computation and metrics.
        
        Args:
            outputs: Raw model outputs (can be tensor, tuple, dict, etc.)
            
        Returns:
            torch.Tensor: Main output tensor for loss computation
        """
        pass

    @abstractmethod
    def _execute_training_logic(
        self,
        train_loader,
        val_loader,
        test_loader,
    ):
        """Execute training logic using provided infrastructure from BaseTrainer
        
        Args:
            train_loader: Training DataLoader (already configured with proper sampler)
            val_loader: Validation DataLoader (already configured)
            test_loader: Test DataLoader (already configured)
        """
        pass

    def run_training(self, dataset_tuple):
        """Unified training entry point with clean checkpoint loading and clear separation of concerns
        
        This method is the single entry point for all training scenarios. It implements
        a clean, logical checkpoint loading approach:
        
        - Reading torchrun environment variables (WORLD_SIZE, LOCAL_RANK, RANK)
        - Hardware setup and device assignment
        - Checkpoint loading: load from disk → store → load weights → DDP wrap → load trainer states
        - Auto-backend detection and DDP initialization
        - Model wrapping with DDP (auto-broadcasts weights from rank-0)
        - Dataloader creation with proper DistributedSampler setup
        - Optimizer and scheduler creation AFTER DDP wrapping
        - Clean trainer state loading via broadcast (works for all scenarios)
        - Delegation to subclasses' _execute_training_logic() for training implementation
        
        Always call this method - it uses the same clean approach for both single-GPU and DDP
        scenarios, providing consistent behavior across all hardware configurations.
        """
        try:
            # Always initialize DDP - it creates consistent environment for all scenarios
            self.init_ddp()

            # Get environment truth from torchrun
            rank, world_size, local_rank = self._get_distributed_config()
            
            # Setup hardware (seed, threads, device)
            self.setup_hardware()
            
            # Setup logging using trainer config
            self.setup_logging()

            # Build dataloaders
            train_loader, val_loader, test_loader = self._build_dataloaders(dataset_tuple)
            
            # Move model to device BEFORE potential DDP wrapping
            self._move_model_to_device()
            
            # === CRITICAL: Load checkpoint BEFORE DDP wrapping ===
            # Clean approach: load checkpoint → store → load weights → DDP wrap → load trainer states
            checkpoint_loaded = False
            if self.resume_training or self.load_model_weights:
                logger.info("Resuming training from checkpoint...")
                # Load checkpoint from disk, store it, AND broadcast status to all ranks
                checkpoint_loaded = self.load_checkpoint_unified(train_loader)
                
                # Load model weights (rank 0 only, before DDP wrapping)
                if checkpoint_loaded and self.load_model_weights:
                    self.prepare_model_weights(
                        models={'model': self.model}
                    )
            
            # DDP initialization and wrapping (auto-backend detection)
            self._setup_ddp_wrapping(self.model, local_rank)
            
            # # === CRITICAL: Watch model after DDP wrapping ===
            # if hasattr(self, 'progress_bar') and self.progress_bar.wandb:
            #     self.progress_bar.wandb.watch(self.model, is_rank0=self.is_rank0)
            
            # Create optimizer/scheduler AFTER DDP wrapping
            self.create_optimizer()
            self.create_scheduler()
            
            # === CRITICAL: Load trainer states AND get training metadata AFTER DDP wrapping ===
            if checkpoint_loaded and self.resume_training:
                # Now ALL ranks get the training metadata (checkpoint_loaded was already broadcast!)
                self.load_trainer_states()
            
            # Execute training logic
            self._execute_training_logic(
                train_loader, val_loader, test_loader
            )
                
        except Exception as e:
            logger.error("Error in training", exc_info=True)
            raise
        finally:
            # Final cleanup
            self.cleanup_resources(dataset_tuple)
            logger.info("Training completed or terminated")
            
        # Final Architecture Summary (Clean Separation of Concerns):
        # 1. Pick device (always)
        # 2. Rank-0 loads checkpoint from disk, stores it, AND broadcasts status to all ranks (load_checkpoint_unified)
        # 3. Build model → move to device
        # 4. If checkpoint exists: rank-0 loads model weights into unwrapped model (prepare_model_weights)
        # 5. Always initialize DDP (creates consistent environment for all scenarios)
        # 6. Auto-detect backend (GPU: nccl, CPU: gloo)
        # 7. Wrap with DDP if multi-GPU (auto-broadcasts weights from rank-0)
        # 8. Create optimizer/scheduler/scaler (post-wrap)
        # 9. Trainer state: rank-0 loads from stored checkpoint → broadcast → all ranks apply (load_trainer_states) AND get training metadata
        # 10. Always cleanup DDP (handles both single-GPU and DDP scenarios)

# cs = ConfigStore.instance()
# cs.store(name="base_trainer", node=TrainerBaseConfig, group="trainer")