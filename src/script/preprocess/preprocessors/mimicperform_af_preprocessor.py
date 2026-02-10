"""MimicPERFormAF dataset preprocessor.

This module provides preprocessing functionality for the MimicPERFormAF dataset,
which contains ECG, PPG, and potentially IMP (impedance) waveforms with AF
classification labels.

The MimicPERFormAF dataset includes:
- ECG waveforms
- PPG (photoplethysmography) waveforms
- IMP (impedance/respiratory) waveforms (when available)
- ABP (arterial blood pressure) waveforms (when available)
- AF classification labels (binary: AF vs non-AF)
- Patient-level splitting to prevent data leakage
- Merging of AF/non-AF datasets

The preprocessor handles the most complex preprocessing workflow including:
- AF classification task with binary labels
- Patient-level splitting to prevent data leakage
- Segment processing with NaN filtering
- Train/test splits at patient or sample level
- Merging of AF and non-AF datasets
- Comprehensive AF-specific statistics

Note: Some records may not have all waveform types available - this is a key
consideration for this dataset and should be clearly documented.

Note: imp_label and abp_label were removed from the config; only ecg_label and
ppg_label are used in main waveforms (IMP/ABP not in main waveform set).
"""

# Standard library imports
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import h5py

# Third-party imports
import numpy as np
from hydra.core.config_store import ConfigStore
from scipy.io import loadmat as scipy_loadmat
from tqdm import tqdm

# Local imports
from src.script.preprocess.base_preprocessor import BaseDatasetPreprocessor
from src.script.preprocess.base_preprocessor import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class MimicPERFormAFPreprocessingConfig(PreprocessingConfig):
    """Configuration for MimicPERFormAF dataset preprocessing."""

    _target_: str = (
        "src.script.preprocess.preprocessors.mimicperform_af_preprocessor."
        "MimicPERFormAFDatasetPreprocessor"
    )
    dataset: str = "MimicPERFormAF"

    # Signal labels (ECG, PPG only in main waveforms)
    ecg_label: int = 0
    ppg_label: int = 1

    # Sampling rate (all signals are at 125 Hz)
    sampling_rate: int = 125

    # Signal length (all signals are 150,001 samples - ~20 minutes)
    signal_length: int = 150001

    # Task type
    task_type: str = "classification"

    # Signal availability handling
    handle_missing_imp: bool = True  # Enable IMP availability handling
    create_imp_subsets: bool = True  # Separate arrays for IMP-available vs missing
    create_abp_subsets: bool = True  # Separate arrays for ABP-available vs missing

    # Segment size configuration
    segment_size: int = 1024  # Process only one segment size per run
    process_all_sizes: bool = False  # Legacy mode: process all sizes in one file

    # Train/Test split configuration
    create_train_test_splits: bool = True  # Enable/disable train/test splits
    split_at_patient_level: bool = (
        True  # Split by patients (80% train / 20% test) vs by samples per patient
    )
    train_ratio: float = 0.8  # Ratio for splitting (patients or samples per patient)
    test_ratio: float = 0.2  # Ratio for splitting (patients or samples per patient)

    # Merge configuration
    merge_splits: bool = True  # Enable/disable merging AF and non-AF files
    merge_filename_template: str = (
        # Template for size-specific merged filename
        "MimicPERFormAF_{segment_size}_merged.h5"
    )
    preserve_individual_files: bool = True  # Keep separate files when merging


class MimicPERFormAFDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor for MimicPERFormAF dataset.

    The MimicPERFormAF dataset is the most complex preprocessor with AF classification
    labels, patient-level splitting to prevent data leakage, and merging of AF/non-AF
    datasets. This dataset contains ECG, PPG, and potentially IMP (impedance) waveforms,
    with AF labels. Some records may not have all waveform types available.

    This preprocessor handles the specific characteristics of MimicPERFormAF data
    including:
    - AF classification task with binary labels
    - Patient-level splitting to prevent data leakage
    - Segment processing with NaN filtering
    - Train/test splits at patient or sample level
    - Merging of AF and non-AF datasets
    - Comprehensive AF-specific statistics

    This class follows Hydra instantiation standards with individual parameters
    and direct attribute access (self.parameter_name) throughout the code.
    """

    def __init__(
        self,
        segment_size: int = 1024,
        create_train_test_splits: bool = True,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        split_at_patient_level: bool = True,
        merge_splits: bool = True,
        preserve_individual_files: bool = False,
        merge_filename_template: str = "MimicPERFormAF_{segment_size}_merged.h5",
        ecg_label: int = 0,
        ppg_label: int = 1,
        imp_label: int = 2,
        **kwargs,
    ):
        """Initialize MimicPERFormAF preprocessor with dataset-specific parameters.

        Args:
            segment_size: Size of fixed segments for processing (default: 1024,
                matches config)
            create_train_test_splits: Whether to create train/test splits
            train_ratio: Proportion of data for training set
            test_ratio: Proportion of data for test set
            split_at_patient_level: Whether to split at patient level (vs sample level)
            merge_splits: Whether to merge AF and non-AF datasets (default: True,
                matches config)
            preserve_individual_files: Whether to keep individual files after merging
            merge_filename_template: Template for merged filename with segment size
                placeholder
            ecg_label: Channel index for ECG signal
            ppg_label: Channel index for PPG signal
            imp_label: Channel index for impedance signal
            **kwargs: Base preprocessor parameters (dataset, input_file,
                output_file, etc.)
        """
        super().__init__(**kwargs)
        self.segment_size = segment_size
        self.create_train_test_splits = create_train_test_splits
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.split_at_patient_level = split_at_patient_level
        self.merge_splits = merge_splits
        self.preserve_individual_files = preserve_individual_files
        self.merge_filename_template = merge_filename_template
        self.ecg_label = ecg_label
        self.ppg_label = ppg_label
        self.imp_label = imp_label

    def preprocess_dataset(self, input_file: str, output_file: str) -> list[str]:
        """Process single segment size per run with train/test splits at subject
        level."""
        segment_size = self.segment_size
        logger.info(
            f"Preprocessing MimicPERFormAF dataset: {input_file} "
            f"(segment size: {segment_size})"
        )

        splits = self._load_af_splits(input_file)

        logger.info("Dataset Statistics:")
        for split_name, split_data in splits.items():
            if split_data["waveforms"]:  # Only print if data exists
                logger.info(f"{split_name} samples: {len(split_data['waveforms'])}")

        output_files = []
        for split_name, split_data in splits.items():
            if not split_data["waveforms"]:  # Skip empty splits
                continue

            logger.info(
                f"Processing {split_name} split for segment size {segment_size}..."
            )
            processed_data = self._process_segment_size(split_data, segment_size)

            if self.create_train_test_splits:
                logger.info(
                    f"Creating train/test splits for {split_name} (train: "
                    f"{self.train_ratio:.1%}, test: {self.test_ratio:.1%})"
                )
                train_data, test_data = self._create_subject_level_splits(
                    processed_data
                )

                train_output_path = self._get_output_filename(
                    f"{split_name}_train", segment_size, output_file
                )
                self._save_with_standard_keys(
                    train_data, train_output_path, segment_size
                )
                output_files.append(train_output_path)

                test_output_path = self._get_output_filename(
                    f"{split_name}_test", segment_size, output_file
                )
                self._save_with_standard_keys(test_data, test_output_path, segment_size)
                output_files.append(test_output_path)

                self._log_split_statistics(
                    train_data, test_data, split_name, segment_size
                )
            else:
                output_path = self._get_output_filename(
                    split_name, segment_size, output_file
                )
                self._save_with_standard_keys(processed_data, output_path, segment_size)
                output_files.append(output_path)

                self._log_dataset_statistics(processed_data, split_name, segment_size)

        # Merge into single files when requested (reduces number of open handles)
        if self.merge_splits and self.save_files:
            logger.info("Creating merged datasets...")
            try:
                if self.create_train_test_splits:
                    merged_train_file = self._create_merged_file(
                        [f for f in output_files if "train" in f and "merged" not in f],
                        segment_size,
                        output_file,
                        split_type="train",
                    )
                    merged_test_file = self._create_merged_file(
                        [f for f in output_files if "test" in f and "merged" not in f],
                        segment_size,
                        output_file,
                        split_type="test",
                    )
                    output_files.extend([merged_train_file, merged_test_file])

                    self._verify_merged_dataset(merged_train_file)
                    self._verify_merged_dataset(merged_test_file)
                else:
                    merged_file = self._create_merged_file(
                        output_files, segment_size, output_file
                    )
                    output_files.append(merged_file)
                    self._verify_merged_dataset(merged_file)

                # Optionally remove individual files
                if not self.preserve_individual_files:
                    logger.info("Removing individual split files as requested...")
                    individual_files = [f for f in output_files if "merged" not in f]
                    for file_path in individual_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Removed individual file: {file_path}")
                    output_files = [f for f in output_files if "merged" in f]

            except Exception as e:
                logger.error(f"Failed to create merged files: {e}")
                logger.warning("Continuing with individual files only")
        elif self.merge_splits and not self.save_files:
            logger.info("Skipping merge operations (save_files=False)")

        logger.info(f"Preprocessing complete! Output files: {output_files}")
        return output_files

    def _load_af_splits(self, input_file: str) -> dict:
        """Load AF and non-AF splits from .mat files."""
        splits: dict[str, Any] = {
            "af": {
                "waveforms": [],
                "af_labels": [],
                "subject_ids": [],
                "record_ids": [],
                "file_ids": [],
                "abp_available": [],
                "imp_available": [],
            },
            "non_af": {
                "waveforms": [],
                "af_labels": [],
                "subject_ids": [],
                "record_ids": [],
                "file_ids": [],
                "abp_available": [],
                "imp_available": [],
            },
        }

        base_dir = os.path.dirname(input_file)

        if "af_data.mat" in input_file:
            af_file = input_file
            non_af_file = os.path.join(base_dir, "mimic_perform_non_af_data.mat")
            logger.info(f"Loading AF data from {af_file}...")
            af_data = self._build_mimicperforaf_dataset(af_file, is_af=True)
            splits["af"] = af_data

            logger.info(f"Loading non-AF data from {non_af_file}...")
            non_af_data = self._build_mimicperforaf_dataset(non_af_file, is_af=False)
            splits["non_af"] = non_af_data
        else:
            logger.info(f"Loading non-AF data from {input_file}...")
            non_af_data = self._build_mimicperforaf_dataset(input_file, is_af=False)
            splits["non_af"] = non_af_data

        return splits

    def _process_segment_size(self, split_data: dict, segment_size: int) -> dict:
        """Process data for a specific segment size (RAW only for lazy
        normalization)."""
        processed_data: dict[str, Any] = {
            "waveforms_raw": [],  # Temporarily keep stacked ECG+PPG; split at save time
            "af_labels": [],
            "subject_ids": [],
            "record_ids": [],
            "file_ids": [],
            "abp_available": [],
            "imp_available": [],
        }

        n_samples = len(split_data["waveforms"])
        with tqdm(
            total=n_samples, desc=f"Processing {segment_size}-sample segments"
        ) as pbar:
            for i in range(n_samples):
                current_waveforms = split_data["waveforms"][
                    i
                ]  # Always 2-channel: [ECG, PPG]
                current_af_label = split_data["af_labels"][i]
                current_subject_id = split_data["subject_ids"][i]
                current_record_id = split_data["record_ids"][i]
                current_file_id = split_data["file_ids"][i]
                current_abp_available = split_data["abp_available"][i]
                current_imp_available = split_data["imp_available"][i]

                # Segment size fixed by config; one size per run for consistent batches
                segments = self._create_segments_for_size(
                    current_waveforms, segment_size
                )

                if i < 3:
                    logger.info(
                        f"Sample {i}: Created {len(segments)} segments of size "
                        f"{segment_size} from waveform of length "
                        f"{current_waveforms.shape[1]}"
                    )

                for segment_idx, segment_waveform in enumerate(segments):
                    if self._has_nan_values(segment_waveform):
                        logger.warning(
                            f"Skipping {segment_size}-sample segment {segment_idx} "
                            f"from sample {i} due to NaN values"
                        )
                        continue

                    processed_data["waveforms_raw"].append(segment_waveform)
                    processed_data["af_labels"].append(current_af_label)
                    processed_data["subject_ids"].append(current_subject_id)
                    processed_data["record_ids"].append(current_record_id)
                    processed_data["file_ids"].append(current_file_id)
                    processed_data["abp_available"].append(current_abp_available)
                    processed_data["imp_available"].append(current_imp_available)

                pbar.update(1)

        # Log final statistics
        total_segments = len(processed_data["waveforms_raw"])
        logger.info(
            f"Created {total_segments} {segment_size}-sample segments from {n_samples} "
            f"original samples"
        )
        logger.info(
            f"Average segments per original sample: {total_segments / n_samples:.1f}"
        )

        return processed_data

    def _create_segments_for_size(
        self, waveform: np.ndarray, segment_size: int
    ) -> list[np.ndarray]:
        """Create segments for a specific size only."""
        segments = []
        waveform_length = waveform.shape[1]
        num_segments = waveform_length // segment_size

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            if end_idx > waveform_length:
                break
            segments.append(waveform[:, start_idx:end_idx])

        return segments

    def _create_subject_level_splits(self, processed_data: dict) -> tuple[dict, dict]:
        """Create train/test splits at subject level.

        Two modes:
        1. Patient-level split (split_at_patient_level=True): 80% of subjects go
            to train,
           20% to test
        2. Sample-level split (split_at_patient_level=False): For each subject, 80% of
           samples go to train, 20% to test

        This ensures no data leakage between train and test sets.

        Args:
            processed_data: Dictionary containing all processed data

        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        subject_ids = processed_data["subject_ids"]
        unique_subjects = sorted(set(subject_ids))

        logger.info(
            f"Creating subject-level splits for {len(unique_subjects)} unique subjects"
        )
        logger.info(f"Total samples before splitting: {len(subject_ids)}")

        train_data: dict[str, Any] = {key: [] for key in processed_data}
        test_data: dict[str, Any] = {key: [] for key in processed_data}

        if self.split_at_patient_level:
            logger.info(
                f"Using patient-level splitting: {self.train_ratio:.1%} patients "
                f"to train, {self.test_ratio:.1%} to test"
            )

            n_subjects = len(unique_subjects)
            train_subject_count = int(n_subjects * self.train_ratio)

            train_subjects = unique_subjects[:train_subject_count]
            test_subjects = unique_subjects[train_subject_count:]

            logger.info(
                f"Train subjects: {len(train_subjects)}, "
                f"Test subjects: {len(test_subjects)}"
            )

            for i, subject_id in enumerate(subject_ids):
                if subject_id in train_subjects:
                    for key in processed_data:
                        train_data[key].append(processed_data[key][i])
                else:
                    for key in processed_data:
                        test_data[key].append(processed_data[key][i])

            train_subject_count = len(set(train_data["subject_ids"]))
            test_subject_count = len(set(test_data["subject_ids"]))
            logger.info(
                f"Patient-level split: {len(train_subjects)} subjects "
                f"({train_subject_count} actual) → train, {len(test_subjects)} "
                f"subjects ({test_subject_count} actual) → test"
            )

        else:
            logger.info(
                f"Using sample-level splitting: {self.train_ratio:.1%} samples "
                f"per subject to train, {self.test_ratio:.1%} to test"
            )

            for subject_id in unique_subjects:
                subject_indices = [
                    i for i, sid in enumerate(subject_ids) if sid == subject_id
                ]
                n_samples = len(subject_indices)

                train_count = int(n_samples * self.train_ratio)
                test_count = n_samples - train_count

                train_indices = subject_indices[:train_count]
                test_indices = subject_indices[train_count:]

                for key in processed_data:
                    if isinstance(processed_data[key], list):
                        train_data[key].extend(
                            [processed_data[key][i] for i in train_indices]
                        )
                        test_data[key].extend(
                            [processed_data[key][i] for i in test_indices]
                        )
                    else:
                        train_data[key].extend(processed_data[key][train_indices])
                        test_data[key].extend(processed_data[key][test_indices])

                logger.debug(
                    f"Subject {subject_id}: {train_count} train, {test_count} "
                    f"test samples"
                )

        # Log final statistics
        train_total = len(train_data["subject_ids"])
        test_total = len(test_data["subject_ids"])
        total_samples = train_total + test_total
        train_pct = train_total / total_samples * 100
        test_pct = test_total / total_samples * 100
        logger.info(
            f"Final split: {train_total} train samples ({train_pct:.1f}%), "
            f"{test_total} test samples ({test_pct:.1f}%)"
        )

        return train_data, test_data

    def _calculate_mimicperforaf_statistics(self, data: dict, split_name: str) -> dict:
        """Calculate comprehensive statistics for MimicPERFormAF dataset split.

        Args:
            data: Dictionary containing processed data for a split
            split_name: Name of the split (Train/Test)

        Returns:
            Dictionary with comprehensive statistics
        """
        subject_ids = data["subject_ids"]
        af_labels = data["af_labels"]

        # NumPy for downstream indexing and stats
        subject_ids_array = np.array(subject_ids)
        af_labels_array = np.array(af_labels)

        measurements_per_subject = self._calculate_measurements_per_subject(
            subject_ids_array
        )

        total_measurements = measurements_per_subject["total_measurements"]
        total_subjects = measurements_per_subject["total_subjects"]

        unique_subjects = np.unique(subject_ids_array)

        af_measurements = np.sum(af_labels_array == 1)
        non_af_measurements = np.sum(af_labels_array == 0)

        af_patients = 0
        non_af_patients = 0

        for subject_id in unique_subjects:
            subject_mask = subject_ids_array == subject_id
            subject_af_labels = af_labels_array[subject_mask]

            if np.any(subject_af_labels == 1):
                af_patients += 1
            else:
                non_af_patients += 1

        # AF distribution statistics
        af_distribution = {
            "af_measurements": af_measurements,
            "non_af_measurements": non_af_measurements,
            "af_patients": af_patients,
            "non_af_patients": non_af_patients,
            "af_measurement_percentage": (
                (af_measurements / total_measurements * 100)
                if total_measurements > 0
                else 0
            ),
            "af_patient_percentage": (
                (af_patients / total_subjects * 100) if total_subjects > 0 else 0
            ),
        }

        return {
            "split_name": split_name,
            "measurements_per_subject": measurements_per_subject,
            "af_distribution": af_distribution,
            "total_subjects": total_subjects,
            "total_measurements": total_measurements,
            "af_patients": af_patients,
            "non_af_patients": non_af_patients,
        }

    def _log_mimicperforaf_statistics(self, stats: dict, split_name: str) -> None:
        """Log MimicPERFormAF statistics in a formatted, clinical manner.

        Args:
            stats: Statistics dictionary from _calculate_mimicperforaf_statistics
            split_name: Name of the split (Train/Test)
        """
        logger.info(f"=== {split_name} Split Statistics ===")

        self._log_measurements_per_subject(
            stats["measurements_per_subject"], split_name
        )

        af_dist = stats["af_distribution"]
        logger.info("=== Atrial Fibrillation Statistics ===")
        logger.info(
            f"AF Measurements: {af_dist['af_measurements']} "
            f"({af_dist['af_measurement_percentage']:.1f}%)"
        )
        non_af_pct = 100 - af_dist["af_measurement_percentage"]
        logger.info(
            f"Non-AF Measurements: {af_dist['non_af_measurements']} ({non_af_pct:.1f}%)"
        )
        logger.info(
            f"Patients with AF: {af_dist['af_patients']} "
            f"({af_dist['af_patient_percentage']:.1f}%)"
        )
        non_af_patient_pct = 100 - af_dist["af_patient_percentage"]
        logger.info(
            f"Patients without AF: {af_dist['non_af_patients']} "
            f"({non_af_patient_pct:.1f}%)"
        )

    def _log_dataset_statistics(self, data: dict, split_name: str, segment_size: int):
        """Log comprehensive dataset statistics for a single split (when train/test
        splits are disabled)."""
        stats = self._calculate_mimicperforaf_statistics(data, split_name)

        logger.info(
            f"=== {split_name.upper()} Dataset Statistics ({segment_size} samples) ==="
        )
        self._log_mimicperforaf_statistics(stats, split_name)

    def _log_split_statistics(
        self, train_data: dict, test_data: dict, split_name: str, segment_size: int
    ):
        """Log detailed statistics about train/test splits including patient-level
        and AF-specific statistics."""
        train_stats = self._calculate_mimicperforaf_statistics(train_data, "Train")
        test_stats = self._calculate_mimicperforaf_statistics(test_data, "Test")

        logger.info(
            f"=== {split_name.upper()} Split Statistics ({segment_size} samples) ==="
        )

        self._log_mimicperforaf_statistics(train_stats, "Train")
        self._log_mimicperforaf_statistics(test_stats, "Test")

        logger.info("=== Split Summary ===")
        logger.info(
            f"Total patients: Train={train_stats['total_subjects']}, "
            f"Test={test_stats['total_subjects']}"
        )
        logger.info(
            f"Total measurements: Train={train_stats['total_measurements']}, "
            f"Test={test_stats['total_measurements']}"
        )
        logger.info(
            f"AF patients: Train={train_stats['af_patients']}, "
            f"Test={test_stats['af_patients']}"
        )
        logger.info(
            f"Non-AF patients: Train={train_stats['non_af_patients']}, "
            f"Test={test_stats['non_af_patients']}"
        )

    def _save_with_standard_keys(self, data: dict, output_path: str, segment_size: int):
        """Save data with standard key names (NEW format: vital-specific raw arrays)."""
        if not self.save_files:
            logger.info(
                f"Skipping standard keys save (save_files=False): {output_path}"
            )
            logger.info(
                f"Would have saved {len(data['waveforms_raw'])} segments (NEW format)"
            )
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            # Split stacked waveforms into vital-specific raw arrays
            waveforms_array = np.array(data["waveforms_raw"])  # [N, 2, T]
            ecg_raw = waveforms_array[:, 0:1, :]
            ppg_raw = waveforms_array[:, 1 : 1 + 1, :]

            arrays_info = {
                "ecg_raw": ecg_raw,
                "ppg_raw": ppg_raw,
                "af_labels": np.array(data["af_labels"]),
                "abp_available": np.array(data["abp_available"]),
                "imp_available": np.array(data["imp_available"]),
            }

            for key, array in arrays_info.items():
                f.create_dataset(key, data=array)

            self._create_utf8_dataset(f, "subject_ids", data["subject_ids"])
            self._create_utf8_dataset(f, "record_ids", data["record_ids"])
            self._create_utf8_dataset(f, "file_ids", data["file_ids"])

            f.attrs["dataset_name"] = "MimicPERFormAF"
            f.attrs["segment_size"] = segment_size
            f.attrs["total_segments"] = len(data["waveforms_raw"])

        logger.info(
            f"Saved {len(data['waveforms_raw'])} {segment_size}-sample segments "
            f"(NEW format: ecg_raw, ppg_raw) to: {output_path}"
        )

    def _get_output_filename(
        self, split_name: str, segment_size: int, output_file: str
    ) -> str:
        """Generate size-specific filename with train/test split support."""
        if split_name == "merged":
            return os.path.join(
                output_file, self.dataset, f"MimicPERFormAF_{segment_size}_merged.h5"
            )
        elif "merged_train" in split_name:
            return os.path.join(
                output_file,
                self.dataset,
                f"MimicPERFormAF_{segment_size}_merged_train.h5",
            )
        elif "merged_test" in split_name:
            return os.path.join(
                output_file,
                self.dataset,
                f"MimicPERFormAF_{segment_size}_merged_test.h5",
            )
        else:
            return os.path.join(
                output_file,
                self.dataset,
                f"MimicPERFormAF_{segment_size}_{split_name}.h5",
            )

    def _create_merged_file(
        self,
        output_files: list[str],
        segment_size: int,
        output_file: str,
        split_type: str | None = None,
    ) -> str:
        """Create merged file from individual split files with train/test support."""
        if split_type:
            merged_filename = f"MimicPERFormAF_{segment_size}_merged_{split_type}.h5"
        else:
            merged_filename = self.merge_filename_template.format(
                segment_size=segment_size
            )

        merged_path = os.path.join(output_file, self.dataset, merged_filename)

        # Use the existing merge logic but with size-specific filename
        merged_file = self._merge_af_files(
            output_files, output_file, split_type=split_type
        )

        # Rename to size-specific filename
        if os.path.exists(merged_file):
            os.rename(merged_file, merged_path)
            logger.info(f"Renamed merged file to: {merged_path}")

        return merged_path

    def _create_utf8_dataset(self, h5file, name, strings, **kwargs):
        """Create UTF-8 string dataset in HDF5 file.

        Args:
            h5file: HDF5 file object
            name: Dataset name
            strings: List of strings to save
            **kwargs: Additional arguments for create_dataset

        Returns:
            Created dataset
        """
        dt = h5py.string_dtype(encoding="utf-8")  # modern, replaces special_dtype
        data = np.asarray(strings, dtype=dt)
        return h5file.create_dataset(name, data=data, **kwargs)

    def _has_nan_values(self, waveform: np.ndarray) -> bool:
        """Check if waveform contains any NaN values.

        Args:
            waveform: Input waveform of shape (channels, length)

        Returns:
            True if any NaN values are found, False otherwise
        """
        return bool(np.any(np.isnan(waveform)))

    def _build_mimicperforaf_dataset(self, path: str, is_af: bool = True) -> dict:
        """Build MimicPERFormAF dataset from .mat file."""
        data = scipy_loadmat(path)
        data_array = data["data"][0]  # Remove the extra dimension

        waveforms = []
        af_labels = []
        subject_ids = []
        record_ids = []
        file_ids = []
        abp_available = []
        imp_available = []

        logger.info(f"Processing {len(data_array)} samples from {path}")

        for i in range(len(data_array)):
            ecg_data = data_array[i]["ekg"]["v"][0][0].T  # ECG signal
            ppg_data = data_array[i]["ppg"]["v"][0][0].T  # PPG signal

            imp_available_flag = False
            try:
                _ = data_array[i]["imp"]["v"][0][0].T
                imp_available_flag = True
            except (KeyError, IndexError, AttributeError):
                imp_available_flag = False

            waveform = np.vstack([ecg_data, ppg_data])
            waveforms.append(waveform)

            fix_data = data_array[i]["fix"][0][0]
            af_status = fix_data["af_status"][0][0]
            subj_id = fix_data["subj_id"][0]
            rec_id = fix_data["rec_id"][0]
            file_id = fix_data["files"][0][0][0]

            abp_available_flag = False
            try:
                data_array[i]["abp"]["v"][0][0]
                abp_available_flag = True
            except (KeyError, IndexError, AttributeError):
                abp_available_flag = False

            af_labels.append(int(af_status))
            subject_ids.append(str(subj_id))
            record_ids.append(str(rec_id))
            file_ids.append(str(file_id))
            abp_available.append(abp_available_flag)
            imp_available.append(imp_available_flag)

        return {
            "waveforms": waveforms,  # Always 2-channel: [ECG, PPG]
            "af_labels": af_labels,
            "subject_ids": subject_ids,
            "record_ids": record_ids,
            "file_ids": file_ids,
            "abp_available": abp_available,
            "imp_available": imp_available,  # NEW
        }

    def _merge_af_files(
        self, split_files: list[str], output_dir: str, split_type: str | None = None
    ) -> str:
        """Merge AF and non-AF files into a single HDF5 file with train/test support.

        Args:
            split_files: List of split file paths [af_file, non_af_file]
            output_dir: Output directory for merged file
            split_type: Optional split type ('train' or 'test') for filename

        Returns:
            Path to merged file
        """
        split_suffix = f"_{split_type}" if split_type else ""
        logger.info(f"Merging AF and non-AF datasets{split_suffix}...")

        af_file = None
        non_af_file = None

        for file_path in split_files:
            if (
                "af_" in file_path
                and "non_af" not in file_path
                and file_path.endswith(".h5")
            ):
                af_file = file_path
            elif "non_af_" in file_path and file_path.endswith(".h5"):
                non_af_file = file_path

        if not af_file or not non_af_file:
            raise ValueError(
                f"Could not find both AF and non-AF files in: {split_files}"
            )

        logger.info(f"Loading AF data from: {af_file}")
        logger.info(f"Loading non-AF data from: {non_af_file}")

        af_data = {}
        with h5py.File(af_file, "r") as f:
            for key in f:
                item = f[key]
                if key is not None and str(key).startswith(
                    ("subject_ids", "record_ids", "file_ids")
                ):
                    if isinstance(item, h5py.Dataset):
                        if item.dtype.kind == "O":
                            af_data[key] = np.array(item.asstr()[:], dtype=str)
                        else:
                            af_data[key] = item[:]
                    else:
                        af_data[key] = item
                else:
                    af_data[key] = item[:] if isinstance(item, h5py.Dataset) else item
            af_segment_size = f.attrs.get("segment_size", 1024)

        non_af_data = {}
        with h5py.File(non_af_file, "r") as f:
            for key in f:
                item = f[key]
                if key is not None and str(key).startswith(
                    ("subject_ids", "record_ids", "file_ids")
                ):
                    if isinstance(item, h5py.Dataset):
                        if item.dtype.kind == "O":
                            non_af_data[key] = np.array(item.asstr()[:], dtype=str)
                        else:
                            non_af_data[key] = item[:]
                    else:
                        non_af_data[key] = item
                else:
                    non_af_data[key] = (
                        item[:] if isinstance(item, h5py.Dataset) else item
                    )
            non_af_segment_size = f.attrs.get("segment_size", 1024)

        # Verify both files have the same segment size
        if af_segment_size != non_af_segment_size:
            raise ValueError(
                f"Segment size mismatch: AF file has {af_segment_size}, "
                f"non-AF file has {non_af_segment_size}"
            )

        segment_size = af_segment_size
        logger.info(f"Merging datasets with segment size: {segment_size}")

        af_count = len(af_data["af_labels"])
        non_af_count = len(non_af_data["af_labels"])
        total_samples = af_count + non_af_count

        logger.info(
            f"Merging {af_count} AF and {non_af_count} non-AF samples "
            f"(total: {total_samples})"
        )

        # Merge datasets using standard key names
        merged_data = {}
        for key in af_data:
            if key in non_af_data:
                merged_data[key] = np.concatenate(
                    [af_data[key], non_af_data[key]], axis=0
                )
                logger.debug(
                    f"Merged {key}: {af_data[key].shape} + {non_af_data[key].shape} = "
                    f"{merged_data[key].shape}"
                )

        # AF vs non-AF labels required for downstream split-aware evaluation
        split_info = np.concatenate(
            [
                np.full(af_count, "af", dtype="S10"),
                np.full(non_af_count, "non_af", dtype="S10"),
            ],
            axis=0,
        )
        merged_data["split_info"] = split_info

        if split_type:
            merged_file_path = os.path.join(
                output_dir, f"MimicPERFormAF_merged_{split_type}.h5"
            )
        else:
            merged_file_path = os.path.join(output_dir, "MimicPERFormAF_merged.h5")

        logger.info(f"Saving merged dataset to: {merged_file_path}")
        with h5py.File(merged_file_path, "w") as f:
            numerical_keys = [
                k
                for k in merged_data
                if not k.startswith(("subject_ids", "record_ids", "file_ids"))
                and k != "split_info"
            ]
            for key in numerical_keys:
                logger.debug(
                    f"Saving numerical dataset {key} with shape "
                    f"{merged_data[key].shape}"
                )
                f.create_dataset(key, data=merged_data[key])

            string_keys = [
                k
                for k in merged_data
                if k.startswith(("subject_ids", "record_ids", "file_ids"))
            ]
            for key in string_keys:
                logger.debug(
                    f"Saving string dataset {key} with shape {merged_data[key].shape}"
                )
                self._create_utf8_dataset(f, key, merged_data[key])

            if "split_info" in merged_data:
                logger.debug(
                    f"Saving split_info with shape {merged_data['split_info'].shape}"
                )
                self._create_utf8_dataset(f, "split_info", merged_data["split_info"])

            f.attrs["dataset_name"] = "MimicPERFormAF"
            f.attrs["segment_size"] = segment_size
            f.attrs["total_samples"] = total_samples
            f.attrs["af_samples"] = af_count
            f.attrs["non_af_samples"] = non_af_count
            f.attrs["task_type"] = "classification"
            f.attrs["base_channels"] = 2  # ECG + PPG
            if split_type:
                f.attrs["split_type"] = split_type

        logger.info("Merged dataset saved successfully!")
        logger.info(
            f"Total samples: {total_samples} ({af_count} AF + {non_af_count} non-AF)"
        )
        logger.info(f"Merged file saved to: {merged_file_path}")

        return merged_file_path

    def _verify_merged_dataset(self, merged_file: str):
        """Verify merged dataset integrity."""
        logger.info(f"Verifying merged dataset: {merged_file}")
        try:
            with h5py.File(merged_file, "r") as f:
                required_datasets = [
                    "ecg_raw",
                    "ppg_raw",
                    "af_labels",
                    "split_info",
                    "subject_ids",
                ]
                missing_datasets = [ds for ds in required_datasets if ds not in f]
                if missing_datasets:
                    logger.error(
                        f"Missing required datasets in merged file: {missing_datasets}"
                    )
                    raise KeyError(f"Missing required datasets: {missing_datasets}")

                ecg_raw = f["ecg_raw"]
                ppg_raw = f["ppg_raw"]
                af_labels = f["af_labels"]
                split_info = f["split_info"]
                if not isinstance(ecg_raw, h5py.Dataset):
                    raise ValueError("ecg_raw is not an HDF5 Dataset")
                if not isinstance(ppg_raw, h5py.Dataset):
                    raise ValueError("ppg_raw is not an HDF5 Dataset")

                n_samples, n_channels, seq_length = ecg_raw.shape
                logger.info(
                    f"Merged dataset dimensions: {n_samples} samples, "
                    f"{n_channels} channels, {seq_length} sequence length for ECG"
                )
                n_samples, n_channels, seq_length = ppg_raw.shape
                logger.info(
                    f"Merged dataset dimensions: {n_samples} samples, "
                    f"{n_channels} channels, {seq_length} sequence length for PPG"
                )

                # Verify AF label distribution (narrow to Dataset for shape/len)
                if not isinstance(af_labels, h5py.Dataset):
                    raise ValueError("af_labels is not an HDF5 Dataset")
                af_labels_ds: h5py.Dataset = af_labels
                af_data = af_labels_ds[:]
                af_count = int(np.sum(af_data == 1))
                non_af_count = int(np.sum(af_data == 0))
                total_count = af_labels_ds.shape[0]

                af_pct = af_count / total_count * 100
                non_af_pct = non_af_count / total_count * 100
                logger.info(
                    f"AF label distribution: {af_count} AF ({af_pct:.1f}%), "
                    f"{non_af_count} non-AF ({non_af_pct:.1f}%)"
                )

                # Verify split_info array
                af_split_count = np.sum(split_info == b"af")
                non_af_split_count = np.sum(split_info == b"non_af")

                logger.info(
                    f"Split info distribution: {af_split_count} 'af', "
                    f"{non_af_split_count} 'non_af'"
                )

                # Verify consistency
                if af_count != af_split_count or non_af_count != non_af_split_count:
                    logger.error(
                        f"Inconsistent split information: AF labels={af_count}, "
                        f"AF splits={af_split_count}"
                    )
                    raise ValueError("Inconsistent split information")

                if "merge_info" in f.attrs:
                    merge_info = f.attrs["merge_info"]
                    logger.info(f"Merge metadata: {merge_info}")

                logger.info("Merged dataset verification completed successfully")

        except Exception as e:
            logger.error(f"Error verifying merged dataset: {e}")
            raise


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_mimicperformaf_preprocessing",
    group="preprocessor",
    node=MimicPERFormAFPreprocessingConfig,
)
