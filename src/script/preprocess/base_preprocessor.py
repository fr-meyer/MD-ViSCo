"""Base preprocessor class for MD-ViSCo datasets.

This module provides the base preprocessor class with common functionality
for HDF5 saving, verification, and data processing operations.

REFACTORING NOTE (Lazy Normalization):
The base class now supports saving separate raw waveforms per vital sign
(ecg_raw, ppg_raw, abp_raw) instead of pre-normalized versions.
Normalization is performed at batch time in the collate function.

The base class handles:
- HDF5 file saving with optimal chunking and compression
- Dataset verification for new structure (separate vital sign fields)
- Common normalization workflows (used by collate function, not preprocessing)
- Global min-max normalization (used by collate function, not preprocessing)
- MAP calculation
- Output path construction
- Common statistics calculation utilities
"""

import logging

# Standard library imports
import os
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py

# Third-party imports
import numpy as np
from omegaconf import MISSING

# Local imports
from src.utils.utils_preprocessing import calculate_map
from src.utils.utils_preprocessing import global_min_max_norm
from src.utils.utils_preprocessing import normalize_signals

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Base configuration for all dataset preprocessors."""

    dataset: str = MISSING
    input_file: str = MISSING
    output_file: str = MISSING
    save_files: bool = True
    stats_enabled: bool = True
    dbp_min: float = 40.0
    sbp_max: float = 200.0


class BaseDatasetPreprocessor(ABC):
    """Shared logic for dataset preprocessing (HDF5, verification, normalization).

    This class provides common functionality for all dataset preprocessors including:
    - HDF5 file saving with optimal chunking and compression
    - Dataset verification and statistics processing
    - Common normalization workflows
    - Global min-max normalization
    - MAP calculation
    - Output path construction

    The class is designed to be flexible for different waveform configurations.
    Datasets may have different channel types (ECG, PPG, ABP, IMP) and varying
    availability of vital signs across records.
    """

    def __init__(
        self,
        dataset: str,
        input_file: str,
        output_file: str,
        save_files: bool = True,
        stats_enabled: bool = True,
        dbp_min: float = 40.0,
        sbp_max: float = 200.0,
        **kwargs,
    ):
        """Initialize base preprocessor with individual parameters (Hydra standard).

        Args:
            dataset: Dataset name (e.g., 'PulseDB', 'UCI')
            input_file: Path to input data file
            output_file: Path to output processed file
            save_files: Whether to save processed files to disk
            stats_enabled: Whether to calculate and log statistics
            dbp_min: Minimum diastolic blood pressure for normalization
            sbp_max: Maximum systolic blood pressure for normalization
            **kwargs: Additional parameters for child classes (absorbed but not used)
        """
        self.dataset = dataset
        self.input_file = input_file
        self.output_file = output_file
        self.save_files = save_files
        self.stats_enabled = stats_enabled
        self.dbp_min = dbp_min
        self.sbp_max = sbp_max

    def _save_to_hdf5(
        self,
        output_path: str,
        arrays_info: dict,
        subject_ids_data: np.ndarray | None = None,
        additional_metadata: dict[Any, Any] | None = None,
    ):
        """Save arrays to HDF5 file with proper chunking and optional metadata.

        Args:
            output_path: Path to save the HDF5 file
            arrays_info: Dictionary of numpy arrays to save
            subject_ids_data: Optional subject IDs array (PulseDB only)
            additional_metadata: Optional additional metadata to store in file
                attributes
        """
        if not self.save_files:
            logger.info(f"Skipping file save (save_files=False): {output_path}")
            logger.info(
                f"Would have saved {len(arrays_info)} datasets with shapes: {
                    [(k, v.shape) for k, v in arrays_info.items()]
                }"
            )
            return output_path

        logger.info(f"Saving dataset to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            if subject_ids_data is not None:
                chunks_subject_ids = self._calculate_chunk_size(
                    subject_ids_data.shape, subject_ids_data.dtype
                )
                f.create_dataset(
                    "subject_ids",
                    data=subject_ids_data,
                    chunks=chunks_subject_ids,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            for name, data in arrays_info.items():
                chunks = self._calculate_chunk_size(data.shape, data.dtype)
                logger.debug(
                    f"Creating dataset {name} with shape {data.shape} and chunks {
                        chunks
                    }"
                )
                f.create_dataset(
                    name,
                    data=data,
                    chunks=chunks,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            f.attrs["chunked"] = True
            f.attrs["creation_date"] = datetime.now().isoformat()
            f.attrs["dataset_type"] = self.dataset

            # Chunk info stored for debugging
            chunk_info = {name: dataset.chunks for name, dataset in f.items()}
            f.attrs["chunk_info"] = str(chunk_info)

            if additional_metadata:
                for key, value in additional_metadata.items():
                    f.attrs[key] = value

        logger.info(f"Successfully saved dataset to {output_path}")

    def _calculate_chunk_size(self, shape: tuple, dtype) -> tuple:
        """Calculate optimal chunk size based on data characteristics.

        Computes chunk dimensions based on a target chunk byte size (1-4 MB) to balance
        IO performance and compression. Scales samples-per-chunk with dataset length and
        sequence length. Avoids single-sample chunks when possible.

        Args:
            shape: Shape tuple of the array
            dtype: Data type of the array

        Returns:
            Tuple representing chunk dimensions
        """
        element_size = np.dtype(dtype).itemsize
        # Target chunk size range: 1-4 MB (use 2 MB as middle ground)
        target_chunk_bytes_min = 1 * 1024 * 1024  # 1 MB
        target_chunk_bytes_max = 4 * 1024 * 1024  # 4 MB
        target_chunk_bytes = 2 * 1024 * 1024  # 2 MB default

        # Handle empty or single-sample datasets
        if len(shape) == 0 or shape[0] == 0:
            return shape

        if len(shape) == 3:
            # 3D array: (N, C, T)
            channels = shape[1]
            seq_length = shape[2]
            bytes_per_sample = element_size * channels * seq_length

            # If single sample is larger than max target, use single sample
            # (unavoidable)
            if bytes_per_sample > target_chunk_bytes_max:
                return (1, channels, seq_length)

            samples_per_chunk = target_chunk_bytes // bytes_per_sample

            # Ensure at least 1 sample, but prefer multiple samples when possible
            # For very small datasets, use all samples; for larger datasets, scale
            # appropriately
            if shape[0] <= 10:
                # Small dataset: use all samples in one chunk
                samples_per_chunk = shape[0]
            else:
                # Scale samples-per-chunk with dataset size and sequence length
                # Larger datasets and shorter sequences can have more samples per chunk
                scale_factor = min(
                    shape[0] / 1000, 10.0
                )  # Scale up to 10x for large datasets
                samples_per_chunk = max(1, int(samples_per_chunk * scale_factor))
                samples_per_chunk = min(
                    samples_per_chunk, shape[0]
                )  # Don't exceed dataset size

            # Ensure we have at least a few samples per chunk when possible
            if (
                samples_per_chunk == 1
                and shape[0] > 1
                and bytes_per_sample < target_chunk_bytes_min
            ):
                # Try to get at least 2-4 samples if sample size allows
                samples_per_chunk = min(
                    4, shape[0], target_chunk_bytes_min // bytes_per_sample
                )
                samples_per_chunk = max(1, samples_per_chunk)

            return (samples_per_chunk, channels, seq_length)

        elif len(shape) == 2:
            # 2D array: (N, F)
            features = shape[1]
            bytes_per_sample = element_size * features

            # If single sample is larger than max target, use single sample
            # (unavoidable)
            if bytes_per_sample > target_chunk_bytes_max:
                return (1, features)

            samples_per_chunk = target_chunk_bytes // bytes_per_sample

            # Scale with dataset length
            if shape[0] <= 10:
                samples_per_chunk = shape[0]
            else:
                scale_factor = min(shape[0] / 1000, 10.0)
                samples_per_chunk = max(1, int(samples_per_chunk * scale_factor))
                samples_per_chunk = min(samples_per_chunk, shape[0])

            # Ensure multiple samples when possible
            if (
                samples_per_chunk == 1
                and shape[0] > 1
                and bytes_per_sample < target_chunk_bytes_min
            ):
                samples_per_chunk = min(
                    4, shape[0], target_chunk_bytes_min // bytes_per_sample
                )
                samples_per_chunk = max(1, samples_per_chunk)

            return (samples_per_chunk, features)

        else:
            # 1D or other: use target size based on first dimension
            bytes_per_sample = (
                element_size * np.prod(shape[1:]) if len(shape) > 1 else element_size
            )

            # If single sample is larger than max target, use single sample
            # (unavoidable)
            if bytes_per_sample > target_chunk_bytes_max:
                return (1,) + shape[1:]

            samples_per_chunk = target_chunk_bytes // bytes_per_sample

            # Scale with dataset length
            if shape[0] <= 10:
                samples_per_chunk = shape[0]
            else:
                scale_factor = min(shape[0] / 1000, 10.0)
                samples_per_chunk = max(1, int(samples_per_chunk * scale_factor))
                samples_per_chunk = min(samples_per_chunk, shape[0])

            # Ensure multiple samples when possible
            if (
                samples_per_chunk == 1
                and shape[0] > 1
                and bytes_per_sample < target_chunk_bytes_min
            ):
                samples_per_chunk = min(
                    4, shape[0], target_chunk_bytes_min // bytes_per_sample
                )
                samples_per_chunk = max(1, samples_per_chunk)

            return (samples_per_chunk,) + shape[1:]

    def _verify_saved_dataset(self, output_file: str):
        """Verify the saved dataset with optimized chunk reading.

        Works for both PulseDB and UCI datasets since they share the same core
        structure. Both dataset types save datasets directly in the file root,
        not in groups.
        """
        if not self.save_files:
            logger.info(
                f"Skipping dataset verification (save_files=False): {output_file}"
            )
            return

        logger.info(f"Verifying dataset: {output_file}")
        try:
            with h5py.File(output_file, "r") as f:
                required_datasets = ["ecg_raw", "ppg_raw", "abp_raw", "bp_raw"]
                missing_datasets = [ds for ds in required_datasets if ds not in f]
                if missing_datasets:
                    logger.error(f"Missing required datasets: {missing_datasets}")
                    logger.error(f"Available datasets: {list(f.keys())}")
                    raise KeyError(f"Missing required datasets: {missing_datasets}")

                # NEW STRUCTURE: Separate fields per vital sign
                # - ecg_raw: [N, 1, T] - ECG waveform
                # - ppg_raw: [N, 1, T] - PPG waveform
                # - abp_raw: [N, 1, T] - ABP waveform
                # - bp_raw: [N, 3] - Blood pressure (SBP, DBP, MAP)
                # Optional: age, gender, height, weight, bmi, demographics
                # (PulseDB only), subject_ids (PulseDB only)

                # Verify individual demographic fields are present
                try:
                    individual_demo_fields = [
                        "age",
                        "gender",
                        "height",
                        "weight",
                        "bmi",
                    ]
                    has_individual_fields = all(
                        field in f for field in individual_demo_fields
                    )
                    has_legacy_field = "demographics" in f

                    if has_individual_fields:
                        logger.debug(
                            "Individual demographic fields detected (age, gender, "
                            "height, weight, bmi)"
                        )
                    elif has_legacy_field:
                        logger.warning(
                            "LEGACY FORMAT DETECTED: This HDF5 file uses the old "
                            "combined 'demographics' field. Please regenerate with "
                            "the latest preprocessor for individual demographic "
                            "fields."
                        )
                except Exception as e:
                    logger.debug(f"Demographic field verification skipped: {e}")

                # Use ecg_raw to get dimensions (all vital signs have same shape)
                ecg_raw_ref = f["ecg_raw"]
                if not isinstance(ecg_raw_ref, h5py.Dataset):
                    raise ValueError(
                        f"ecg_raw in {output_file} is not a Dataset (no shape)"
                    )
                ecg_ds: h5py.Dataset = ecg_raw_ref
                n_samples, n_channels, seq_length = ecg_ds.shape

                # Guard for empty datasets
                if n_samples == 0:
                    raise ValueError(f"Dataset is empty (n_samples=0) in {output_file}")

                # Note: n_channels will be 1 for each vital sign now
                # Ensure sample_size is at least 1, even for small datasets
                sample_size = min(200, max(1, n_samples // 10))
                if n_samples == 1:
                    sample_indices = np.array([0], dtype=int)
                else:
                    sample_indices = np.linspace(
                        0, n_samples - 1, sample_size, dtype=int
                    )

                logger.debug("Dataset Structure:")
                for name, dataset in f.items():
                    chunk_mb = (
                        np.prod(dataset.chunks) * dataset.dtype.itemsize / (1024**2)
                        if dataset.chunks
                        else 0
                    )
                    logger.debug(
                        f"{name}: Shape={dataset.shape}, Chunks={dataset.chunks}, "
                        f"Chunk size={chunk_mb:.2f}MB, Compression={
                            dataset.compression
                        }, "
                        f"Compression ratio={dataset.nbytes / dataset.size:.2f}x"
                    )

                stats = self._process_chunks_statistics(f, sample_indices)
                self._print_verification_results(stats, "dataset")

        except KeyError as e:
            logger.error(f"Missing required dataset: {e}")
            # Don't try to access f.keys() here as the file might be closed
            raise
        except Exception as e:
            logger.error(f"Error loading {output_file}: {str(e)}")
            raise

    def _process_chunks_statistics(self, file_group, sample_indices):
        """Process dataset statistics using chunked reading.

        Note: NEW STRUCTURE with separate vital sign fields
        Required datasets: ecg_raw, ppg_raw, abp_raw, bp_raw
        Optional datasets: subject_ids (PulseDB only), demographics/individual
        demographic fields (PulseDB only)
        """
        # Type narrowing for HDF5 Dataset access
        ecg_raw_item = file_group["ecg_raw"]
        if not isinstance(ecg_raw_item, h5py.Dataset):
            raise ValueError("Expected 'ecg_raw' to be a Dataset")
        ecg_raw = ecg_raw_item
        n_samples = ecg_raw.shape[0]

        # Handle empty datasets explicitly
        if n_samples == 0:
            raise ValueError(
                "Cannot process statistics for empty dataset (n_samples=0)"
            )

        # Ensure we have at least one sample to process
        if len(sample_indices) == 0:
            sample_indices = np.array([0], dtype=int)

        bp_raw_item = file_group["bp_raw"]
        abp_raw_item = file_group["abp_raw"]
        if not isinstance(bp_raw_item, h5py.Dataset):
            raise ValueError("Expected 'bp_raw' to be a Dataset")
        if not isinstance(abp_raw_item, h5py.Dataset):
            raise ValueError("Expected 'abp_raw' to be a Dataset")
        bp_raw_ds = bp_raw_item
        abp_raw_ds = abp_raw_item

        stats = {
            "n_samples": n_samples,
            "n_channels": 3,  # ECG, PPG, ABP (stored separately)
            "seq_length": ecg_raw.shape[2],
            "file_size_mb": os.path.getsize(file_group.file.filename) / (1024**2),
        }
        bp_stats = {"raw": {"min": float("inf"), "max": float("-inf")}}
        abp_stats = {"raw": {"min": float("inf"), "max": float("-inf")}}
        chunk_size = bp_raw_ds.chunks[0] if bp_raw_ds.chunks else len(sample_indices)
        for i in range(0, len(sample_indices), chunk_size):
            batch_indices = sample_indices[i : i + chunk_size]
            bp_raw = bp_raw_ds[batch_indices]
            abp_raw = abp_raw_ds[batch_indices]

            bp_stats["raw"]["min"] = min(bp_stats["raw"]["min"], np.min(bp_raw))
            bp_stats["raw"]["max"] = max(bp_stats["raw"]["max"], np.max(bp_raw))
            abp_stats["raw"]["min"] = min(abp_stats["raw"]["min"], np.min(abp_raw))
            abp_stats["raw"]["max"] = max(abp_stats["raw"]["max"], np.max(abp_raw))

        stats["bp_ranges"] = {"raw": (bp_stats["raw"]["min"], bp_stats["raw"]["max"])}
        stats["abp_ranges"] = {
            "raw": (abp_stats["raw"]["min"], abp_stats["raw"]["max"])
        }
        return stats

    def _print_verification_results(self, stats, dataset_name):
        """Print verification results in a formatted way."""
        logger.info(f"=== Verification Results for {dataset_name} ===")
        logger.info(
            f"Dataset Dimensions: Samples={stats['n_samples']}, "
            f"Channels={stats['n_channels']}, Sequence Length={stats['seq_length']}"
        )
        logger.info(f"File Size: {stats['file_size_mb']:.2f} MB")

        logger.info("Blood Pressure Ranges:")
        logger.info(
            f"  Raw: Min={stats['bp_ranges']['raw'][0]:.2f}, Max={
                stats['bp_ranges']['raw'][1]:.2f}"
        )

        logger.info("ABP Ranges:")
        logger.info(
            f"  Raw: Min={stats['abp_ranges']['raw'][0]:.2f}, Max={
                stats['abp_ranges']['raw'][1]:.2f}"
        )

    def _normalize_waveforms(
        self, waveforms: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize waveforms using local min-max normalization and zero-centering.

        Args:
            waveforms: Input waveforms array with shape (channels, sequence_length)

        Returns:
            Tuple of (min-max normalized waveforms, zero-centered waveforms)
        """
        return normalize_signals(waveforms)

    def _apply_global_normalization(
        self,
        data: np.ndarray,
        global_min: float | None = None,
        global_max: float | None = None,
    ) -> np.ndarray:
        """Apply global min-max normalization to data.

        Args:
            data: Input data array
            global_min: Global minimum value (uses config dbp_min if not provided)
            global_max: Global maximum value (uses config sbp_max if not provided)

        Returns:
            Globally normalized data array
        """
        if global_min is None:
            global_min = self.dbp_min
        if global_max is None:
            global_max = self.sbp_max

        return global_min_max_norm(
            data, global_min_max={"min": global_min, "max": global_max}
        )

    def _calculate_map(self, sbp: float, dbp: float) -> float:
        """Calculate Mean Arterial Pressure (MAP) from systolic and diastolic
        blood pressure.

        Args:
            sbp: Systolic Blood Pressure
            dbp: Diastolic Blood Pressure

        Returns:
            Calculated MAP value
        """
        result = calculate_map(sbp, dbp)
        return float(result) if not isinstance(result, float) else result

    def _build_output_path(
        self,
        output_dir: str,
        dataset_name: str,
        split_name: str = "",
        file_stem: str = "",
    ) -> str:
        """Construct standardized output file paths.

        Args:
            output_dir: Base output directory
            dataset_name: Name of the dataset
            split_name: Optional split name (e.g., 'train', 'test')
            file_stem: Optional file stem for additional identification

        Returns:
            Constructed output file path
        """
        path_parts = [output_dir, dataset_name]
        if split_name:
            path_parts.append(split_name)
        if file_stem:
            path_parts.append(file_stem)

        return os.path.join(*path_parts) + ".h5"

    def _calculate_measurements_per_subject(self, subject_ids: np.ndarray) -> dict:
        """Calculate measurements-per-subject statistics.

        This is a common operation across all datasets that track subject IDs.
        Returns statistics about how many measurements each subject has.

        Args:
            subject_ids: Array of subject identifiers (can be strings or integers)

        Returns:
            Dictionary containing:
            - min: Minimum measurements per subject
            - max: Maximum measurements per subject
            - mean: Average measurements per subject
            - median: Median measurements per subject
            - std: Standard deviation of measurements per subject
            - total_subjects: Total number of unique subjects
            - total_measurements: Total number of measurements
        """
        try:
            subject_array = np.asarray(subject_ids)
        except Exception:
            subject_array = np.array(subject_ids)

        unique_subjects, counts = np.unique(subject_array, return_counts=True)

        return {
            "min": int(np.min(counts)) if len(counts) > 0 else 0,
            "max": int(np.max(counts)) if len(counts) > 0 else 0,
            "mean": float(np.mean(counts)) if len(counts) > 0 else 0.0,
            "median": float(np.median(counts)) if len(counts) > 0 else 0.0,
            "std": float(np.std(counts)) if len(counts) > 0 else 0.0,
            "total_subjects": int(len(unique_subjects)),
            "total_measurements": int(len(subject_array)),
        }

    def _calculate_descriptive_stats(self, data: np.ndarray) -> dict:
        """Calculate standard descriptive statistics for a data array.

        This is a common operation for analyzing distributions of continuous variables
        (e.g., blood pressure, heart rate, respiratory rate).

        Args:
            data: 1D numpy array of numerical data

        Returns:
            Dictionary containing:
            - min: Minimum value
            - max: Maximum value
            - mean: Mean value
            - median: Median value
            - std: Standard deviation
            - q25: 25th percentile
            - q75: 75th percentile
            - iqr: Interquartile range (q75 - q25)
        """
        data_array = np.asarray(data).astype(float)

        return {
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "std": float(np.std(data_array)),
            "q25": float(np.percentile(data_array, 25)),
            "q75": float(np.percentile(data_array, 75)),
            "iqr": float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
        }

    def _build_subject_groups(self, subject_ids: np.ndarray) -> dict[str, list[int]]:
        """Build a mapping from subject IDs to their measurement indices.

        This is useful for subject-level analysis where you need to group
        measurements by subject (e.g., checking if ANY measurement meets a criterion).

        Args:
            subject_ids: Array of subject identifiers

        Returns:
            Dictionary mapping subject_id (as string) to list of measurement indices
        """
        try:
            subject_array = np.array([str(s) for s in np.ravel(subject_ids)])
        except Exception:
            subject_array = np.array([str(s) for s in subject_ids])

        subject_to_indices: dict[str, list[int]] = {}
        for idx, sid in enumerate(subject_array):
            subject_to_indices.setdefault(sid, []).append(idx)

        return subject_to_indices

    def _log_measurements_per_subject(
        self, stats: dict, context_name: str = "Dataset"
    ) -> None:
        """Log measurements-per-subject statistics in a formatted manner.

        Args:
            stats: Statistics dictionary from _calculate_measurements_per_subject()
            context_name: Context for logging (e.g., "Dataset", "Train Split",
                "Test Split")
        """
        logger.info(
            f"{context_name} - Measurements per Subject: Min={stats['min']}, Max={
                stats['max']
            }, "
            f"Mean={stats['mean']:.1f}±{stats['std']:.1f}, Median={stats['median']:.1f}"
        )
        logger.info(
            f"{context_name} - Total Subjects: {stats['total_subjects']}, "
            f"Total Measurements: {stats['total_measurements']}"
        )

    def _log_descriptive_stats(
        self, stats: dict, variable_name: str, unit: str = ""
    ) -> None:
        """Log descriptive statistics in a formatted manner.

        Args:
            stats: Statistics dictionary from _calculate_descriptive_stats()
            variable_name: Name of the variable (e.g., "SBP", "DBP", "Heart Rate")
            unit: Optional unit string (e.g., "mmHg", "bpm", "years")
        """
        unit_str = f" {unit}" if unit else ""
        logger.info(
            f"{variable_name}: {stats['min']:.1f}-{stats['max']:.1f}{unit_str} "
            f"(Mean={stats['mean']:.1f}±{stats['std']:.1f}, Median={
                stats['median']:.1f}, "
            f"IQR={stats['q25']:.1f}-{stats['q75']:.1f})"
        )

    @abstractmethod
    def preprocess_dataset(
        self, input_file: str, output_file: str
    ) -> None | list[str] | str:
        """Abstract method to be implemented by child classes.

        Args:
            input_file: Path to input data file
            output_file: Path to output processed file

        Returns:
            None, list of output file paths, or single output file path depending
            on implementation
        """
