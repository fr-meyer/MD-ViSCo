# Standard library imports
import resource
import atexit
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, Any
import os # Added for os.path.join

# Third-party imports
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from src.core.domain import Vital, Direction
from src.core.direction import Directions

# Setup logger for this module
logger = logging.getLogger(__name__)

@dataclass
class Sample:
    """Standardized sample format for all datasets"""
    # Core waveform data (shared memory tensors, not numpy arrays)
    waveform_raw: torch.Tensor
    waveforms_raw_abp_global_minmax: torch.Tensor
    abp_global_minmax: torch.Tensor
    bp_raw: torch.Tensor
    waveform_minmax_zc: torch.Tensor
    waveform_local_minmax: torch.Tensor
    bp_global_minmax: torch.Tensor
    abp_raw: torch.Tensor  # Required - both datasets have this
    sample_index: int      # Required - always available
    
    # Optional fields (shared memory tensors or None)
    subject_id: Optional[str] = None
    demographics_raw: Optional[torch.Tensor] = None
    demographics_split: Optional[torch.Tensor] = None
    text_encoded: Optional[torch.Tensor] = None
    
    def has_demographics(self) -> bool:
        """Check if demographic information is available"""
        return self.demographics_raw is not None or self.demographics_split is not None
    
    def has_text_encoding(self) -> bool:
        """Check if text encoding is available"""
        return self.text_encoded is not None

@dataclass
class VitalsDatasetConfig:
    """Configuration for VitalsDataset."""
    channels: Dict[str, int] = MISSING  # Fixed: removed union typing
    _target_: str = "src.dataset.base_dataset.VitalsDataset"

class VitalsDataset:
    def __init__(self, channels: Dict[str, int]):
        self._map = {}
        for k, v in channels.items():
            try:
                vital_key = Vital[k.upper()]
                self._map[vital_key] = int(v)
            except KeyError:
                raise ValueError(f"Invalid vital name: {k}. Valid options: {[v.name for v in Vital]}")
        
        inv = {v: k for k, v in self._map.items()}
        if len(inv) != len(self._map):
            raise ValueError("Channel indices must be unique.")
        self._inv = inv

    def vitals(self) -> Iterable[Vital]: return self._map.keys()

    def has(self, v: Vital) -> bool: return v in self._map

    def chan(self, v: Vital) -> int: return self._map[v]

    def chans_for(self, d: Direction) -> Tuple[int,int]:
        if not (self.has(d.src) and self.has(d.tgt)):
            have = ", ".join(v.name for v in self._map)
            raise ValueError(f"{d.key()} not available. Available vitals: {have}")
        return self.chan(d.src), self.chan(d.tgt)

    def supports_directions(self, directions: Directions) -> bool:
        """Return True if all directions can be executed with available vitals."""
        for direction in directions.directions:
            if not (self.has(direction.src) and self.has(direction.tgt)):
                return False
        return True

def register_vital():
    """
    Register the Vital class to the ConfigStore.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_vitals_dataset", node=VitalsDatasetConfig, group="vitals_dataset")


@dataclass
class DatasetBaseConfig:
    """Base configuration class for dataset configurations"""
    _target_: str = MISSING
    dataset_name: str = MISSING
    
    # Common dataset attributes
    dataset_path: str = 'datasets/'
    seed: int = 42
    
    # File path structure (common across datasets)
    dataset_folder: str = ''  # e.g., 'PulseDB/', 'UCI/'
    
    # File naming - NEW APPROACH
    file_name: str = MISSING

    # Vitals
    vitals_dataset: VitalsDatasetConfig = MISSING
    
    # NEW: Dataset splitting configuration
    # These ratios determine how the dataset is split into train/validation/test sets
    # All ratios must be between 0.0 and 1.0, and must sum to 1.0
    train_ratio: float = MISSING  # Ratio of data for training (default: 80%)
    val_ratio: float = MISSING    # Ratio of data for validation (default: 20%)
    test_ratio: float = MISSING    # Ratio of data for testing (default: 0% - no test set)
    
    # NEW: Dataset splitting strategy flags
    # These flags control how the dataset splitting is performed
    use_patient_split: bool = False  # Whether to use patient-level splitting
    use_nabnet_vanilla_split: bool = False  # Whether to use NABNet vanilla splitting
    
    # Blood pressure normalization constants (to be overridden by subclasses)
    sbp_max: Optional[float] = None
    dbp_min: Optional[float] = None
    
    # Channel labels (to be overridden by subclasses)
    ecg_label: Optional[int] = 0
    ppg_label: Optional[int] = 1
    abp_label: Optional[int] = 2
    
    # Channel names mapping (set in __post_init__)
    channel_names: Optional[Dict] = None
    
    # Input size for models
    input_size: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization validation and channel_names setup"""
        # Validate required attributes
        if self.sbp_max is None or self.dbp_min is None:
            raise ValueError("sbp_max and dbp_min must be set in subclasses")
        
        if self.ecg_label is None or self.ppg_label is None or self.abp_label is None:
            raise ValueError("Channel labels must be set in subclasses")
        
        # NEW: Validate file_name is set
        if not self.file_name:
            raise ValueError("file_name must be set in configuration")
        
        # NEW: Validate split ratios
        for name, value in {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
        
        # Validate split ratios sum appropriately
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f} (train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, test={self.test_ratio:.3f})")
        
        # NEW: Validate splitting strategy configuration
        if self.use_patient_split and self.use_nabnet_vanilla_split:
            raise ValueError("Cannot use both patient_split and nabnet_vanilla_split simultaneously")
        
        # Set channel_names if not already set
        if self.channel_names is None:
            self.channel_names = {
                self.ecg_label: "ECG",
                self.ppg_label: "PPG",
                self.abp_label: "ABP"
            }
    
    def get_split_info(self) -> Dict[str, float]:
        """Get information about dataset splitting configuration
        
        Returns:
            Dict containing train_ratio, val_ratio, and test_ratio
        """
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio
        }
    
    def get_splitting_strategy(self) -> Dict[str, bool]:
        """Get information about dataset splitting strategy
        
        Returns:
            Dict containing use_patient_split and use_nabnet_vanilla_split flags
        """
        return {
            "use_patient_split": self.use_patient_split,
            "use_nabnet_vanilla_split": self.use_nabnet_vanilla_split
        }
    
    def get_complete_split_config(self) -> Dict[str, Any]:
        """Get complete dataset splitting configuration
        
        Returns:
            Dict containing both split ratios and splitting strategy
        """
        return {
            **self.get_split_info(),
            **self.get_splitting_strategy()
        }
    
    def validate_split_ratios(self) -> bool:
        """Validate that split ratios are properly configured
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If split ratios are invalid
        """
        # Check individual ratio bounds
        for name, value in {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
        
        # Check sum equals 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        
        return True

    def parse_direction(self, direction: str, refinement_only: bool = False) -> tuple[str, str]:
        """Parse direction string into source and target channels
        
        Args:
            direction: Direction string in format "SOURCE2TARGET" (e.g. "PPG2ABP")
            refinement_only: If True, only ABP can be target
            
        Returns:
            tuple: (source_channel, target_channel)
            
        Raises:
            ValueError: If direction format is invalid
        """
        if self.channel_names is None:
            raise ValueError("channel_names must be set to parse direction")
            
        try:
            source, target = direction.split('2')
            valid_channels = set(self.channel_names.values())
            
            if refinement_only:
                valid_sources = valid_channels - {'ABP'}
                if source not in valid_sources:
                    raise ValueError(f"Invalid source channel '{source}'. Valid sources: {valid_sources}")
                if target != 'ABP':
                    raise ValueError(f"Invalid target channel '{target}'. Only 'ABP' allowed for refinement")
            else:
                if source not in valid_channels:
                    raise ValueError(f"Invalid source channel '{source}'. Valid channels: {valid_channels}")
                if target not in valid_channels:
                    raise ValueError(f"Invalid target channel '{target}'. Valid channels: {valid_channels}")
                if source == target:
                    raise ValueError(f"Source and target channels cannot be the same: {direction}")
            
            return source, target
        except ValueError as e:
            raise ValueError(f"Invalid direction format {direction}. Must be SOURCE2TARGET (e.g. PPG2ABP). {str(e)}")


class BaseDataset(Dataset, ABC):
    """Base class for medical signal datasets with shared memory support"""
    _shared_data = {}  # Class variable to store shared data across instances

    def __init__(self, dataset_name: str, dataset_path: str = 'datasets/', dataset_folder: str = '', file_name: str = '',
                 input_size: int = None, vitals_dataset: VitalsDataset = MISSING, *args, **kwargs):
        # Store individual config parameters
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_folder = dataset_folder
        self.file_name = file_name
        self.input_size = input_size
        self.vitals_dataset = vitals_dataset

        # Store additional kwargs as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # NEW: Use file_name directly instead of split-based approach
        self.sample_file = self.get_sample_file()
        self.sample_length = input_size       # Must be provided by config
        # ... any other shared logic ...

    def get_sample_file(self) -> str:
        """Get the sample file path using the file_name from config"""
        return os.path.join(
            self.dataset_path, 
            self.dataset_folder, 
            self.file_name
        )

    @property
    def vitals_dataset(self):
        """Property that automatically handles Subset wrappers to access vitals_dataset
        
        This property ensures that even when the dataset is wrapped in a PyTorch Subset
        (e.g., during train/val split), we can still access the underlying vitals_dataset.
        
        Returns:
            VitalsDataset: The vitals_dataset from the underlying dataset
        """
        # If this is a Subset wrapper, access the underlying dataset
        if hasattr(self, 'dataset') and hasattr(self.dataset, 'vitals_dataset'):
            return self.dataset.vitals_dataset
        
        # Direct access to vitals_dataset
        return self._vitals_dataset
    
    @vitals_dataset.setter
    def vitals_dataset(self, value):
        """Setter for vitals_dataset"""
        self._vitals_dataset = value

    @classmethod
    def load_shared_data(cls, sample_file, extract_fn):
        """Generic shared memory loader.
        Args:
            sample_file: Path to the HDF5 file.
            extract_fn: Function that takes an h5py.File and returns a dict of arrays.
        """
        if sample_file not in cls._shared_data:
            logger.info(f"Loading dataset {sample_file}")
            with h5py.File(sample_file, "r") as f:
                cls._shared_data[sample_file] = extract_fn(f)

    @classmethod
    def create_subset(cls, base_dataset, indices):
        """Create a subset while maintaining shared memory and attributes."""
        subset = Subset(base_dataset, indices)
        # Copy over relevant attributes
        for attr in ['sample_length', 'sample_file', 'data', 'tokenizer']:
            if hasattr(base_dataset, attr):
                setattr(subset, attr, getattr(base_dataset, attr))
        return subset

    @staticmethod
    def to_tensor(array, dtype=torch.float):
        """Convert numpy array to torch tensor with correct dtype."""
        if isinstance(array, torch.Tensor):
            return array.type(dtype)
        return torch.tensor(array, dtype=dtype)
    
    @classmethod
    def _increase_file_limit(cls):
        """Increase the file descriptor limit"""
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            logger.info(f"Increased file descriptor limit to {hard}")
        except Exception as e:
            logger.warning(f"Could not increase file descriptor limit: {e}")
    
    @staticmethod
    def _pad_waveform(waveform: np.ndarray, target_length: int) -> np.ndarray:
        """Zero pad waveform to target length (centered)"""
        current_length = waveform.shape[-1]
        if current_length >= target_length:
            return waveform
            
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        
        # Adjust padding based on input dimensions
        ndim = waveform.ndim
        if ndim == 3:
            pad_width = [(0, 0), (0, 0), (pad_left, pad_right)]
        elif ndim == 2:
            pad_width = [(0, 0), (pad_left, pad_right)]
        else:
            pad_width = [(pad_left, pad_right)]
        
        return np.pad(waveform, pad_width, mode='constant', constant_values=0)
    
    @staticmethod
    def _pad_waveform_tensor(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """Zero pad tensor waveform to target length (centered)"""
        current_length = waveform.shape[-1]
        if current_length >= target_length:
            return waveform
            
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        
        # Adjust padding based on input dimensions
        ndim = waveform.ndim
        if ndim == 3:
            return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0)
        elif ndim == 2:
            return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0)
        else:
            return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0)
    
    @classmethod
    def cleanup_shared_memory(cls):
        """Clean up shared memory (call this at the end of training)"""
        try:
            # Delete all tensors from shared memory
            for key in list(cls._shared_data.keys()):
                data = cls._shared_data[key]
                if isinstance(data, dict):
                    for data_key, value in list(data.items()):
                        if torch.is_tensor(value):
                            try:
                                del value
                            except Exception as e:
                                logger.warning(f"Error cleaning up tensor {data_key}: {e}")
                del cls._shared_data[key]
            
            # Force CUDA cache cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Shared memory cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_sample(self, idx: int = 0) -> Sample:
        """
        Debug method to get and log sample information for a given index.
        Logs class name, index, waveform_raw, bp_raw, and optional fields.
        Args:
            idx: Sample index to retrieve
        Returns:
            Sample data
        """
        sample = self[idx]
        logger.info(f"[{self.__class__.__name__}] Sample[{idx}]:")
        logger.info(f"  waveform_raw: {sample.waveform_raw.shape} {sample.waveform_raw.dtype}")
        logger.info(f"  bp_raw: {sample.bp_raw.shape} {sample.bp_raw.dtype}")
        if sample.has_demographics():
            logger.info(f"  demographics: available")
        if sample.has_text_encoding():
            logger.info(f"  text_encoding: available")
        return sample
    
    def __del__(self):
        """Clean up shared memory when dataset is deleted"""
        if hasattr(self, 'data'):
            self.cleanup_shared_memory()
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _load_shared_data(self, *args, **kwargs):
        """Load data into shared memory - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index - must be implemented by subclasses"""
        pass

    @property
    def supports_patient_split(self) -> bool:
        """Whether this dataset supports patient-level splitting (default: False)."""
        return False

    @staticmethod
    def safe_load(dataset):
        """
        Safely load a dataset from HDF5, handling both scalars and arrays.
        Returns shared memory tensors for DDP compatibility with consistent float32 dtype.
        
        Args:
            dataset: h5py.Dataset object
            
        Returns:
            torch.Tensor or None
            
        Note:
            All tensors are converted to float32 dtype to ensure compatibility with
            PyTorch model parameters and prevent dtype mismatch errors.
        """
        try:
            logger.info(f"Loading dataset {getattr(dataset, 'name', str(dataset))}")
            if dataset.shape == ():
                # Handle scalar values
                return torch.tensor(dataset[()], dtype=torch.float32).share_memory_()
            
            # Handle array values - convert to float32 for consistency
            return torch.from_numpy(dataset[:]).to(torch.float32).share_memory_()
        except Exception as e:
            logger.warning(f"Error loading dataset {getattr(dataset, 'name', str(dataset))}: {e}")
            return None
