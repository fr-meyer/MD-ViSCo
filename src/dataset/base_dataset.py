"""Base dataset architecture for medical signal datasets with shared memory
and lazy normalization.

This module defines the core dataset abstractions used across MD-ViSCo:
Sample (vital-specific raw waveform fields), VitalsDataset (channel mapping
and direction capability), DatasetBaseConfig (Hydra-compatible config with
split ratios and trimming), and BaseDataset (abstract dataset with shared
HDF5 memory and padding/trimming contract). Datasets return raw waveforms;
normalization and padding are applied at batch time in the collate function.

Classes:
    - NormalizationEnum: Normalization strategies (RAW, MINMAX, GLOBAL_MINMAX, etc.)
    - Sample: Dataclass for per-sample raw waveforms and demographics
    - VitalsDatasetConfig: Hydra config for VitalsDataset
    - VitalsDataset: Channel mapping and direction support for vitals
    - DatasetBaseConfig: Base Hydra config for all dataset implementations
    - BaseDataset: Abstract base class for vital-sign datasets

Examples:
    >>> from src.dataset.base_dataset import BaseDataset, Sample, VitalsDataset
    >>> sample = Sample(sample_index=0, ecg_raw=ecg_tensor, ppg_raw=ppg_tensor)

See Also:
    - src.dataset.core: create_training_datasets, create_test_dataset, split_dataset
    - src.utils.collate_utils: Collate and padding logic
"""

# Standard library imports
import logging
import os
import resource
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Third-party imports
import h5py
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import Dataset
from torch.utils.data import Subset

# Local imports
from src.core.direction import Directions
from src.core.domain import Vital

logger = logging.getLogger(__name__)


class NormalizationEnum(str, Enum):
    """Normalization types for vital signs.

    This enum defines the available normalization strategies for vital signs:
    - RAW: No normalization (raw values)
    - MINMAX: Local min-max normalization to [0, 1]
    - MINMAX_ZC: Min-max normalization with zero-centering to [-1, 1]
    - GLOBAL_MINMAX: Global min-max using dataset-wide constants
    - LOCAL_MINMAX: Local min-max normalization per channel
    """

    RAW = "raw"
    MINMAX = "minmax"
    MINMAX_ZC = "minmax_zc"
    GLOBAL_MINMAX = "global_minmax"
    LOCAL_MINMAX = "local_minmax"


@dataclass
class Sample:
    """Sample with vital-specific raw waveform fields.

    This dataclass stores raw waveforms per vital sign type for lazy normalization.
    Normalization is performed at batch time in the collate function.

    Vital-specific fields:
    - ecg_raw: [1, T] or [C, T] - ECG waveform
    - ppg_raw: [1, T] or [C, T] - PPG waveform
    - abp_raw: [1, T] or [C, T] - ABP waveform
    - imp_raw: [1, T] or [C, T] - IMP/respiratory waveform
    - bp_raw: [3] - Blood pressure (SBP, DBP, MAP)

    Demographic fields (individual format):
    - age_raw: [1] - Age value as scalar tensor
    - gender_raw: [1] - Gender value (0/1 binary encoding)
    - height_raw: [1] - Height value
    - weight_raw: [1] - Weight value
    - bmi_raw: [1] - BMI value

    All demographic fields are optional. Use has_demographics() to check availability.

    Missing vital signs and demographics:
    - Dataset-level: Field is None (not available in dataset)
    - Sample-level: Field is torch.tensor([nan]) (missing for this sample)
    - Partial demographics: Individual fields can be None, use NaN as
      placeholder

    Processor metadata fields (for OutputProcessor post-processing):
    - padding_length: [int] - Padding applied to each end (0 = no padding)
    - vital_sign_type: [str] - Target vital sign type for extractor
      selection ('ABP', 'ECG', etc.)
    - extract_scalars: [bool] - Whether to extract scalars during
      inference (False = waveform only)

    These fields flow through the pipeline: Sample → Collate → Batch →
    ProcessingMetadata → Processor
    """

    # Core identification
    sample_index: int

    # Vital-specific waveform fields
    ecg_raw: torch.Tensor | None = None
    ppg_raw: torch.Tensor | None = None
    abp_raw: torch.Tensor | None = None
    imp_raw: torch.Tensor | None = None

    # Optional BP fields
    bp_raw: torch.Tensor | None = None

    # Optional respiratory rate fields
    rr_labels: torch.Tensor | None = None

    # Optional AF classification fields
    af_labels: torch.Tensor | None = None

    # Optional metadata fields
    subject_id: str | None = None

    # Individual demographic fields
    age_raw: torch.Tensor | None = None
    gender_raw: torch.Tensor | None = None
    height_raw: torch.Tensor | None = None
    weight_raw: torch.Tensor | None = None
    bmi_raw: torch.Tensor | None = None

    # Processor metadata fields
    padding_length: int = 0
    vital_sign_type: str | None = None
    extract_scalars: bool = False

    def has_demographics(self) -> bool:
        """Check if demographic information is available.

        Returns:
            True if any individual demographic field is available
        """
        return (
            self.age_raw is not None
            or self.gender_raw is not None
            or self.height_raw is not None
            or self.weight_raw is not None
            or self.bmi_raw is not None
        )

    def has_abp_data(self) -> bool:
        """Check if ABP/BP data is available.

        Returns:
            True if ABP waveform or BP scalars are available, False otherwise.
        """
        return self.abp_raw is not None or self.bp_raw is not None

    def has_rr_data(self) -> bool:
        """Check if respiratory rate data is available.

        Returns:
            True if respiratory rate labels are available, False otherwise.
        """
        return self.rr_labels is not None

    def has_af_labels(self) -> bool:
        """Check if AF labels are available.

        Returns:
            True if atrial fibrillation labels are available, False otherwise.
        """
        return self.af_labels is not None

    def get_vital(self, vital: Vital) -> torch.Tensor | None:
        """Get raw waveform or scalar data for a specific vital sign.

        Args:
            vital: Vital sign type (ECG, PPG, ABP, IMP, BP)

        Returns:
            Raw tensor for the vital sign, or None if not available
            - For waveforms (ECG, PPG, ABP, IMP): [1, T] or [C, T] tensor
            - For BP: [3] tensor (SBP, DBP, MAP)

        Note:
            This method returns ONLY raw data. Normalization is performed
            at batch time in the collate function (lazy normalization).
        """
        if vital == Vital.ECG:
            return self.ecg_raw
        elif vital == Vital.PPG:
            return self.ppg_raw
        elif vital == Vital.ABP:
            return self.abp_raw
        elif vital == Vital.IMP:
            return self.imp_raw
        elif vital == Vital.BP:
            return self.bp_raw
        return None

    def has_vital(self, vital: Vital) -> bool:
        """Check if vital sign is available and contains valid data.

        Args:
            vital: Vital sign type to check

        Returns:
            True if vital sign is available and not all NaN, False otherwise
        """
        waveform = self.get_vital(vital)
        if waveform is None:
            return False

        # Guard torch.isnan() with floating-point check
        if torch.is_floating_point(waveform):
            return not torch.isnan(waveform).all()
        else:
            # Integer tensors (e.g., BP) are valid if non-empty
            return waveform.numel() > 0


@dataclass
class VitalsDatasetConfig:
    """Configuration for :class:`VitalsDataset`.

    Attributes:
        channels: Mapping from vital name (e.g. "ECG", "PPG", "ABP") to channel index.
        _target_: Full path to VitalsDataset for Hydra instantiation.
    """

    channels: dict[str, int] = MISSING
    _target_: str = "src.dataset.base_dataset.VitalsDataset"


class VitalsDataset:
    """Channel mapping and direction capability for vital signs.

    Maps vital names (ECG, PPG, ABP, IMP, BP) to channel indices and supports
    checking whether a set of directions can be executed with available vitals.

    Attributes:
        _map: Internal mapping from Vital enum to channel index.
        _inv: Inverse mapping from channel index to Vital enum (for uniqueness check).
    """

    def __init__(self, channels: dict[str, int]):
        """Initialize VitalsDataset with channel mapping.

        Args:
            channels: Mapping from vital name (e.g. "ECG", "PPG") to channel index.

        Raises:
            ValueError: If vital name is invalid or channel indices are not unique.
        """
        self._map = {}
        for k, v in channels.items():
            try:
                vital_key = Vital[k.upper()]
                self._map[vital_key] = int(v)
            except KeyError:
                raise ValueError(
                    f"Invalid vital name: {k}. Valid options: {[v.name for v in Vital]}"
                ) from None

        inv = {v: k for k, v in self._map.items()}
        if len(inv) != len(self._map):
            raise ValueError("Channel indices must be unique.")
        self._inv = inv

    def vitals(self) -> Iterable[Vital]:
        """Return iterable of available vital signs."""
        return self._map.keys()

    def has(self, v: Vital) -> bool:
        """Check if vital sign is available in channel mapping."""
        return v in self._map

    def supports_directions(self, directions: Directions) -> bool:
        """Return True if all directions can be executed with available
        vitals."""
        for direction in directions.directions:
            for source_vital in direction.source:
                if not self.has(source_vital):
                    return False
            if not self.has(direction.target):
                return False
        return True


def register_vital():
    """Register VitalsDatasetConfig in Hydra ConfigStore (group "vitals_dataset").

    Side effect only: registers with Hydra ConfigStore. Call before
    Hydra composes configs that reference the vitals_dataset group.
    """
    # Register with Hydra ConfigStore
    cs = ConfigStore.instance()
    cs.store(
        name="base_vitals_dataset",
        node=VitalsDatasetConfig,
        group="vitals_dataset",
    )


@dataclass
class DatasetBaseConfig:
    """Base configuration class for dataset configurations.

    Hydra-compatible config for all dataset implementations. Validates split
    ratios, trimming strategy, and required fields in __post_init__.

    Attributes:
        _target_: Full path to dataset class for Hydra instantiation.
        dataset_name: Human-readable dataset name.
        dataset_path: Root path for dataset files.
        dataset_folder: Subfolder (e.g. 'PulseDB/', 'UCI/').
        file_name: Base file name for HDF5 or data files.
        vitals_dataset: Optional VitalsDataset config for channel mapping.
        input_preprocessing: Optional preprocessing config.
        train_ratio: Fraction of data for training (0.0--1.0).
        val_ratio: Fraction of data for validation (0.0--1.0).
        test_ratio: Fraction of data for test (0.0--1.0). Ratios must sum to 1.0.
        use_patient_split: If True, split by patient; otherwise by sample.
        use_nabnet_vanilla_split: If True, use NABNet fixed 80/20 split.
        sbp_max: Maximum SBP for normalization (subclass override).
        dbp_min: Minimum DBP for normalization (subclass override).
        input_size: Target sequence length for padding/trimming.
        trim_strategy: How to trim when length > input_size
            ('center', 'start', 'end', 'random').
    """

    _target_: str = MISSING
    dataset_name: str = MISSING

    dataset_path: str = "datasets/"
    dataset_folder: str = ""  # e.g., 'PulseDB/', 'UCI/'
    file_name: str = MISSING

    vitals_dataset: VitalsDatasetConfig | None = None
    input_preprocessing: dict[str, Any] | None = None

    train_ratio: float = MISSING
    val_ratio: float = MISSING
    test_ratio: float = MISSING

    use_patient_split: bool = False
    use_nabnet_vanilla_split: bool = False

    sbp_max: float | None = None
    dbp_min: float | None = None

    input_size: int | None = None
    trim_strategy: str = "center"  # 'center', 'start', 'end', 'random'

    def __post_init__(self):
        """Validate post initialization."""
        if self.sbp_max is None or self.dbp_min is None:
            raise ValueError("sbp_max and dbp_min must be set in subclasses")

        if not self.file_name:
            raise ValueError("file_name must be set in configuration")

        for name, value in {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f} "
                f"(train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, "
                f"test={self.test_ratio:.3f})"
            )

        # Only one of patient_split / nabnet_vanilla_split allowed.
        if self.use_patient_split and self.use_nabnet_vanilla_split:
            raise ValueError(
                "Cannot use both patient_split and nabnet_vanilla_split simultaneously"
            )

        if self.input_size is None:
            raise ValueError("input_size must be set in configuration")

        valid_strategies = ["center", "start", "end", "random"]
        if self.trim_strategy not in valid_strategies:
            raise ValueError(
                f"trim_strategy must be one of {valid_strategies}, "
                f"got {self.trim_strategy}"
            )

    def get_split_info(self) -> dict[str, float]:
        """Get information about dataset splitting configuration.

        Returns:
            Dict containing train_ratio, val_ratio, and test_ratio
        """
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }

    def get_splitting_strategy(self) -> dict[str, bool]:
        """Get information about dataset splitting strategy.

        Returns:
            Dict containing use_patient_split and use_nabnet_vanilla_split flags
        """
        return {
            "use_patient_split": self.use_patient_split,
            "use_nabnet_vanilla_split": self.use_nabnet_vanilla_split,
        }

    def get_complete_split_config(self) -> dict[str, Any]:
        """Get complete dataset splitting configuration.

        Returns:
            Dict containing both split ratios and splitting strategy
        """
        return {**self.get_split_info(), **self.get_splitting_strategy()}

    def validate_split_ratios(self) -> bool:
        """Validate that split ratios are properly configured.

        Returns:
            True if validation passes

        Raises:
            ValueError: If split ratios are invalid
        """
        for name, value in {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

        return True


class BaseDataset(Dataset, ABC):
    """Abstract dataset with shared HDF5 memory and padding/trimming contract.

    Datasets return raw waveforms from ``__getitem__`` at their original lengths.
    Padding and trimming happen in the collate function so batches can adapt to
    model requirements while remaining memory-efficient. Subclasses should emit
    any metadata (e.g., ``padding_length``) needed by the collate utilities to
    resize tensors consistently.

    Attributes:
        _shared_data: Class-level dict mapping sample_file paths to extracted
            data (shared across instances for memory efficiency).
        dataset_name: Human-readable dataset name.
        dataset_path: Root path for dataset files.
        dataset_folder: Subfolder (e.g. 'PulseDB/', 'UCI/').
        file_name: Base file name for HDF5 or data files.
        input_size: Target sequence length for padding/trimming.
        sample_file: Full path to the sample file
            (dataset_path/dataset_folder/file_name).
        sample_length: Same as input_size; length of each sample after
            padding/trimming.
        trim_strategy: How to trim when length > input_size
            ('center', 'start', 'end', 'random').
    """

    _shared_data: dict[str, Any] = {}

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str = "datasets/",
        dataset_folder: str = "",
        file_name: str = "",
        input_size: int | None = None,
        vitals_dataset: VitalsDataset | None = None,
        trim_strategy: str = "center",
        *args,
        **kwargs,
    ):
        """Initialize the base dataset with paths, size, vitals mapping,
        and trimming options.

        Args:
            dataset_name: Human-readable name of the dataset.
            dataset_path: Root path for dataset files. Defaults to ``'datasets/'``.
            dataset_folder: Subfolder under dataset_path
                (e.g. ``'PulseDB/'``, ``'UCI/'``). Defaults to ``''``.
            file_name: Base file name for the HDF5 or data file.
            input_size: Target sequence length for padding/trimming.
                Must be provided.
            vitals_dataset: Optional channel mapping and direction
                capability for vitals. Defaults to None.
            trim_strategy: How to trim when length > input_size. One of ``'center'``,
                ``'start'``, ``'end'``, ``'random'``. Defaults to ``'center'``.
            *args: Additional positional arguments (reserved for subclasses).
            **kwargs: Additional keyword arguments; stored as instance
                attributes for subclass use.

        Raises:
            ValueError: If ``input_size`` is None (missing required input_size).
            ValueError: If ``trim_strategy`` is not one of
                ``'center'``, ``'start'``, ``'end'``, ``'random'``.
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_folder = dataset_folder
        self.file_name = file_name
        self.input_size = input_size
        self._vitals_dataset = vitals_dataset
        self.trim_strategy = trim_strategy

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.input_size is None:
            raise ValueError("input_size must be provided to BaseDataset")

        valid_strategies = ["center", "start", "end", "random"]
        if self.trim_strategy not in valid_strategies:
            raise ValueError(
                f"trim_strategy must be one of {valid_strategies}, "
                f"got {self.trim_strategy}"
            )

        self.sample_file = self.get_sample_file()
        self.sample_length = input_size

    def get_sample_file(self) -> str:
        """Get the sample file path using the file_name from config.

        Returns:
            Full path: os.path.join(dataset_path, dataset_folder, file_name).
        """
        return os.path.join(self.dataset_path, self.dataset_folder, self.file_name)

    @property
    def vitals_dataset(self):
        """Dataset-level vital metadata used for direction capability checks.

        The property unwraps Subset wrappers so trainers can determine
        whether a dataset supports requested ``Direction`` configurations.
        Channel indices are sourced from preprocessing metadata rather than
        this object.
        """
        # Unwrap Subset wrappers to access underlying dataset
        inner = getattr(self, "dataset", None)
        if inner is not None and hasattr(inner, "vitals_dataset"):
            return inner.vitals_dataset

        return getattr(self, "_vitals_dataset", None)

    @vitals_dataset.setter
    def vitals_dataset(self, value):
        """Setter used to propagate capability descriptors to wrapped datasets."""
        self._vitals_dataset = value

    @classmethod
    def load_shared_data(cls, sample_file, extract_fn):
        """Load data into shared memory once per file and cache for reuse.

        Loads HDF5 data once per file and caches in cls._shared_data so multiple
        dataset instances share the same in-memory arrays.

        Args:
            sample_file: Path to the HDF5 file.
            extract_fn: Callable that takes an h5py.File and returns a dict of arrays.

        Note:
            Result is stored in cls._shared_data[sample_file].
        """
        if sample_file not in cls._shared_data:
            logger.info(f"Loading dataset {sample_file}")
            with h5py.File(sample_file, "r") as f:
                cls._shared_data[sample_file] = extract_fn(f)

    @classmethod
    def create_subset(cls, base_dataset, indices):
        """Create a subset while maintaining shared memory and attributes.

        Copies sample_length, sample_file, data, input_size, trim_strategy, and
        other attributes from base_dataset onto the Subset so trainers and
        collate functions can access them. Proxies get_normalization_params when
        present on the base dataset.

        Args:
            base_dataset: BaseDataset (or subclass) instance to subset.
            indices: Array or list of indices to include in the subset.

        Returns:
            torch.utils.data.Subset of base_dataset with indices, with
            attributes and get_normalization_params proxy attached.
        """
        subset = Subset(base_dataset, indices)
        simple_attrs = [
            "sample_length",
            "sample_file",
            "data",
            "tokenizer",
            "input_size",
            "trim_strategy",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "dataset_name",
        ]
        for attr in simple_attrs:
            if hasattr(base_dataset, attr):
                setattr(subset, attr, getattr(base_dataset, attr))

        # Proxy get_normalization_params for Subset compatibility
        # (Pyright doesn't recognize dynamic attributes)
        if hasattr(base_dataset, "get_normalization_params") and not hasattr(
            subset, "get_normalization_params"
        ):

            def proxy_get_normalization_params():
                """Proxy that calls base_dataset.get_normalization_params()."""
                return base_dataset.get_normalization_params()

            setattr(  # noqa: B010
                subset,
                "get_normalization_params",
                proxy_get_normalization_params,
            )

        return subset

    @staticmethod
    def to_tensor(array, dtype=torch.float):
        """Convert numpy array to torch tensor with correct dtype.

        Args:
            array: Numpy array or existing tensor to convert.
            dtype: Target dtype (default: torch.float).

        Returns:
            Tensor of the given dtype; unchanged if input is already a tensor.
        """
        if isinstance(array, torch.Tensor):
            return array.type(dtype)
        return torch.tensor(array, dtype=dtype)

    @classmethod
    def _increase_file_limit(cls):
        """Increase the file descriptor limit."""
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            logger.info(f"Increased file descriptor limit to {hard}")
        except Exception as e:
            logger.warning(f"Could not increase file descriptor limit: {e}")

    @classmethod
    def cleanup_shared_memory(cls):
        """Clean up shared memory (call this at the end of training)."""
        try:
            for key in list(cls._shared_data.keys()):
                data = cls._shared_data[key]
                if isinstance(data, dict):
                    for data_key, value in list(data.items()):
                        if torch.is_tensor(value):
                            try:
                                del value
                            except Exception as e:
                                logger.warning(
                                    f"Error cleaning up tensor {data_key}: {e}"
                                )
                del cls._shared_data[key]

            # Force CUDA cache cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Shared memory cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_sample(self, idx: int = 0) -> Sample:
        """Debug method to get and log sample information for a given index.

        Logs class name, index, vital-specific raw data, and optional fields.

        Args:
            idx: Sample index to retrieve

        Returns:
            Sample data
        """
        sample = self[idx]
        logger.info(f"[{self.__class__.__name__}] Sample[{idx}]:")

        for vital_name in ["ecg_raw", "ppg_raw", "abp_raw", "imp_raw"]:
            vital_data = getattr(sample, vital_name)
            if vital_data is not None:
                logger.info(f"  {vital_name}: {vital_data.shape} {vital_data.dtype}")

        if sample.bp_raw is not None:
            logger.info(f"  bp_raw: {sample.bp_raw.shape} {sample.bp_raw.dtype}")

        if sample.has_demographics():
            logger.info("  demographics: available")
        return sample

    def __del__(self):
        """Dataset destructor.

        Note: Shared memory cleanup is NOT performed here to prevent use-after-free
        errors when multiple dataset instances coexist. Use explicit lifecycle hooks
        (e.g., trainer shutdown) or reference counting to clean shared data only when
        no dataset instances remain.
        """
        # Removed cleanup_shared_memory() call to prevent use-after-free when
        # multiple dataset instances share the same class-level _shared_data

    def _load_shared_data_default(self, sample_file):
        """Load shared data using default implementation.

        This helper method provides a standard implementation that calls
        load_shared_data with the sample_file and the subclass's _extract_data method.
        Subclasses can override _load_shared_data if they need custom behavior,
        otherwise they can simply call this method or rely on the default.

        Args:
            sample_file: Path to the HDF5 file containing the dataset
        """
        self.load_shared_data(sample_file, getattr(self, "_extract_data", None))

    def _load_shared_data(self, sample_file):
        """Load data into shared memory.

        Default implementation calls _load_shared_data_default. Subclasses can
        override this method if they need custom behavior.

        Args:
            sample_file: Path to the HDF5 file containing the dataset
        """
        self._load_shared_data_default(sample_file)

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples - must be implemented by subclasses."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index - must be implemented by subclasses."""

    @property
    def supports_patient_split(self) -> bool:
        """Whether this dataset supports patient-level splitting (default: False)."""
        return False

    def get_normalization_params(self) -> dict[str, float] | None:
        """Get dataset-specific normalization parameters.

        This method should be overridden by child classes that have BP data
        to provide global normalization constants (e.g., sbp_max, dbp_min).

        Returns:
            Dict with normalization parameters, or None if not applicable
            Example: {"sbp_max": 286.58, "dbp_min": 2.34}
        """
        return None

    @staticmethod
    def safe_load(dataset, dtype=None):
        """Safely load a dataset from HDF5, handling both scalars and arrays.

        Returns shared memory tensors for DDP compatibility.

        Args:
            dataset: h5py.Dataset object
            dtype: Optional torch dtype to coerce the data to. If None, preserves
                   the source dtype from HDF5. Use float32 for model inputs,
                   int32/int64 for identifiers and labels.

        Returns:
            torch.Tensor or None

        Note:
            - If dtype is None, preserves the source dtype from HDF5
            - If dtype is specified, converts to that dtype
              (e.g., float32 for model inputs)
            - For identifiers and labels, pass appropriate integer dtypes
              (e.g., torch.int64)
        """
        try:
            name = getattr(dataset, "name", str(dataset))
            logger.info(f"Loading dataset {name}")
            if dataset.shape == ():
                tensor = torch.tensor(dataset[()])
                if dtype is not None:
                    tensor = tensor.to(dtype)
                return tensor.share_memory_()

            # Preserve source dtype unless dtype is specified (for array values)
            tensor = torch.from_numpy(dataset[:])
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor.share_memory_()
        except Exception as e:
            logger.warning(
                f"Error loading dataset {getattr(dataset, 'name', str(dataset))}: {e}"
            )
            return None
