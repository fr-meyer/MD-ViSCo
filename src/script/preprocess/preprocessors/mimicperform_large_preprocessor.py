"""MimicPERFormLarge dataset preprocessor.

This module provides preprocessing functionality for the MimicPERFormLarge dataset,
which contains ECG, PPG, and IMP (impedance/respiratory) waveforms with rich metadata.

The MimicPERFormLarge dataset includes:
- ECG waveforms
- PPG (photoplethysmography) waveforms
- IMP (impedance/respiratory) waveforms
- Respiratory rate (RR) labels
- Rich metadata including subject IDs, window IDs, RR intervals, record numbers,
    start/end samples
- Temporal windowing information

The preprocessor handles normalization workflows and comprehensive statistics
tailored to MimicPERFormLarge's characteristics including measurements per subject
and RR interval analysis. This dataset has different characteristics than PulseDB/UCI,
including temporal windowing information and respiratory rate labels instead
of blood pressure.

Note:
    Saved-dataset verification (``_verify_saved_dataset``) is deferred for
    MimicPERFormLarge until the output schema is final. PulseDB and UCI
    preprocessors call it after saving; this preprocessor will do the same
    once the schema is stable.
"""

# Standard library imports
import logging
import os
from dataclasses import dataclass
from typing import Any

# Third-party imports
import numpy as np
from hydra.core.config_store import ConfigStore
from scipy.io import loadmat
from tqdm import tqdm

# Local imports
from src.script.preprocess.base_preprocessor import BaseDatasetPreprocessor
from src.script.preprocess.base_preprocessor import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class MimicPERFormLargePreprocessingConfig(PreprocessingConfig):
    """Configuration for MimicPERFormLarge dataset preprocessing."""

    _target_: str = (
        "src.script.preprocess.preprocessors.mimicperform_large_preprocessor."
        "MimicPERFormLargeDatasetPreprocessor"
    )
    dataset: str = "MimicPERFormLarge"

    # Signal labels (ECG, PPG, RESP/IMP)
    ecg_label: int = 0
    ppg_label: int = 1
    resp_label: int = 2  # Respiratory signal (impedance)

    # Sampling rate (all signals are at 125 Hz)
    sampling_rate: int = 125

    # Signal length (all signals are 4001 samples)
    signal_length: int = 4001


class MimicPERFormLargeDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor for MimicPERFormLarge dataset.

    The MimicPERFormLarge dataset contains ECG, PPG, and IMP (impedance/respiratory)
    waveforms with rich metadata including subject IDs, window IDs, RR intervals,
    record numbers, start/end samples, and temporal windowing information.

    This preprocessor handles the specific characteristics of MimicPERFormLarge
    data including:
    - Measurements per subject statistics
    - RR interval analysis
    - Rich metadata extraction and processing
    - Temporal windowing information

    This class follows Hydra instantiation standards with individual parameters
    and direct attribute access (self.parameter_name) throughout the code.

    Note:
        Verification of saved HDF5 output (``_verify_saved_dataset``) is
        deferred until the output schema is stable; it will be enabled in
        a future update.
    """

    def __init__(
        self,
        ecg_label: int = 0,
        ppg_label: int = 1,
        resp_label: int = 2,
        sampling_rate: int = 125,
        signal_length: int = 4001,
        **kwargs,
    ):
        """Initialize MimicPERFormLarge preprocessor with dataset-specific parameters.

        Args:
            ecg_label: Channel index for ECG signal
            ppg_label: Channel index for PPG signal
            resp_label: Channel index for respiratory/impedance signal
            sampling_rate: Sampling rate of signals in Hz
            signal_length: Length of signal sequences in samples
            **kwargs: Base preprocessor parameters (dataset, input_file,
                output_file, etc.)
        """
        super().__init__(**kwargs)
        self.ecg_label = ecg_label
        self.ppg_label = ppg_label
        self.resp_label = resp_label
        self.sampling_rate = sampling_rate
        self.signal_length = signal_length

    def preprocess_dataset(self, input_file: str, output_file: str) -> list[str]:
        """Preprocess MimicPERFormLarge dataset and save to HDF5 files.

        Args:
            input_file: Path to input .mat file
            output_file: Base path for output HDF5 files

        Returns:
            List of output file paths for train and test splits
        """
        logger.info(f"Preprocessing MimicPERFormLarge dataset: {input_file}")

        splits: dict[str, Any] = {
            "train": {
                "waveforms": [],
                "rr_labels": [],
                "subject_ids": [],
                "window_ids": [],
            },
            "test": {
                "waveforms": [],
                "rr_labels": [],
                "subject_ids": [],
                "window_ids": [],
            },
        }

        base_dir = os.path.dirname(input_file)

        train_file = os.path.join(base_dir, "mimic_perform_large_train_a_data.mat")
        logger.info(f"Loading training data from {train_file}...")
        train_data = self._build_mimicperforlarge_dataset(train_file)
        splits["train"] = train_data

        test_file = os.path.join(base_dir, "mimic_perform_large_test_a_data.mat")
        logger.info(f"Loading test data from {test_file}...")
        test_data = self._build_mimicperforlarge_dataset(test_file)
        splits["test"] = test_data

        logger.info("Dataset Statistics:")
        for split_name, split_data in splits.items():
            logger.info(f"{split_name} samples: {len(split_data['waveforms'])}")
            # Detailed subject/measurement statistics
            if split_data["subject_ids"]:
                mps = self._calculate_measurement_stats(split_data["subject_ids"])
                self._log_measurement_stats(mps, split_name)
            else:
                logger.info(f"=== {split_name.upper()} Split Statistics ===")
                logger.info("No samples in this split.")

        # Process each split and save to separate files
        output_files = []
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split...")
            n_samples = len(split_data["waveforms"])

            # Per NEW format: raw per-vital lists for lazy normalization at collate
            ecg_raw = []
            ppg_raw = []
            imp_raw = []
            rr_labels = []
            subject_ids = []
            subject_nos = []
            window_ids = []

            # Rich metadata required for dataset verification and downstream indexing
            subj_ids = []
            rec_ids = []
            file_ids = []
            groups = []
            rec_nos = []
            start_samps = []
            end_samps = []

            with tqdm(total=n_samples, desc=f"Processing {split_name} samples") as pbar:
                for i in range(n_samples):
                    current_waveforms = split_data["waveforms"][i]
                    current_rr = split_data["rr_labels"][i]
                    current_subject_id = split_data["subject_ids"][i]
                    current_subject_no = split_data["subject_nos"][i]
                    current_window_id = split_data["window_ids"][i]

                    current_subj_id = split_data["subj_ids"][i]
                    current_rec_id = split_data["rec_ids"][i]
                    current_file_id = split_data["file_ids"][i]
                    current_group = split_data["groups"][i]
                    current_rec_no = split_data["rec_nos"][i]
                    current_start_samp = split_data["start_samps"][i]
                    current_end_samp = split_data["end_samps"][i]

                    # [ECG, PPG, RESP/IMP]; RAW for lazy norm at collate
                    ecg_raw.append(
                        current_waveforms[self.ecg_label : self.ecg_label + 1]
                    )
                    ppg_raw.append(
                        current_waveforms[self.ppg_label : self.ppg_label + 1]
                    )
                    imp_raw.append(
                        current_waveforms[self.resp_label : self.resp_label + 1]
                    )
                    rr_labels.append(current_rr)
                    subject_ids.append(current_subject_id)
                    subject_nos.append(current_subject_no)
                    window_ids.append(current_window_id)

                    subj_ids.append(current_subj_id)
                    rec_ids.append(current_rec_id)
                    file_ids.append(current_file_id)
                    groups.append(current_group)
                    rec_nos.append(current_rec_no)
                    start_samps.append(current_start_samp)
                    end_samps.append(current_end_samp)

                    pbar.update(1)

            # NEW format: vital-specific keys for HDF5 (see base_dataset.Sample)
            arrays_info = {
                "ecg_raw": np.array(ecg_raw),
                "ppg_raw": np.array(ppg_raw),
                "imp_raw": np.array(imp_raw),
                "rr_labels": np.array(rr_labels),
                "subject_ids": np.array(subject_ids),
                "subject_nos": np.array(subject_nos),
                "window_ids": np.array(window_ids),
                "subj_ids": np.array(subj_ids),
                "rec_ids": np.array(rec_ids),
                # 'file_ids': np.array(file_ids),
                # 'groups': np.array(groups),
                "rec_nos": np.array(rec_nos),
                "start_samps": np.array(start_samps),
                "end_samps": np.array(end_samps),
            }

            split_output_path = self._build_output_path(
                output_file, "MimicPERFormLarge", split_name=split_name
            )
            additional_metadata = {
                "dataset_name": "MimicPERFormLarge",
                "split": split_name,
                "sampling_rate": self.sampling_rate,
                "signal_length": self.signal_length,
            }
            self._save_to_hdf5(
                split_output_path,
                arrays_info,
                subject_ids_data=None,
                additional_metadata=additional_metadata,
            )

            # Verification deferred until output schema is final (see module docstring).
            if self.save_files:
                logger.info(
                    f"Saved dataset verification deferred for {split_name} "
                    "(schema not yet final)."
                )
            else:
                logger.info(
                    f"Skipping dataset verification for {split_name} (save_files=False)"
                )

            output_files.append(split_output_path)

        logger.info(f"Preprocessing complete! Output files: {output_files}")
        return output_files

    def _calculate_measurement_stats(self, subject_ids) -> dict:
        """Compute measurements-per-subject stats for a split.

        This is a thin wrapper around the base class utility method.

        Args:
            subject_ids: Array of subject identifiers

        Returns:
            Dictionary with measurements-per-subject statistics
        """
        return self._calculate_measurements_per_subject(subject_ids)

    def _log_measurement_stats(self, stats: dict, split_name: str) -> None:
        """Log measurements-per-subject stats for a split.

        This is a thin wrapper around the base class utility method with
        split-specific context.

        Args:
            stats: Statistics dictionary from _calculate_measurement_stats()
            split_name: Name of the split (e.g., 'train', 'test')
        """
        context_name = f"{split_name.upper()} Split"
        self._log_measurements_per_subject(stats, context_name)

    def _build_mimicperforlarge_dataset(self, path: str) -> dict:
        """Build MimicPERFormLarge dataset from .mat file with metadata extraction."""
        data = loadmat(path)
        data_array = data["data"][0]  # Remove the extra dimension

        waveforms = []
        rr_labels = []
        subject_ids = []
        subject_nos = []  # Numeric subject numbers
        window_ids = []

        # Rich metadata fields
        subj_ids = []  # True subject IDs (e.g., '3000086')
        rec_ids = []  # Record IDs
        file_ids = []  # File identifiers
        groups = []  # Group assignments
        rec_nos = []  # Record numbers
        start_samps = []  # Start sample indices
        end_samps = []  # End sample indices

        logger.info(f"Processing {len(data_array)} samples from {path}")

        for i in range(len(data_array)):
            # Extract signals from the correct fields
            ecg_data = data_array[i]["ekg"]["v"][0][0].T  # ECG signal
            ppg_data = data_array[i]["ppg"]["v"][0][0].T  # PPG signal
            imp_data = data_array[i]["imp"]["v"][0][
                0
            ].T  # Impedance (respiratory) signal

            # Stack signals: [ECG, PPG, RESP]
            waveform = np.vstack([ecg_data, ppg_data, imp_data])
            waveforms.append(waveform)

            # Extract basic labels and metadata
            rr_value = data_array[i]["rr"][0][0]  # Respiratory rate
            rr_value = float(rr_value)  # Ensure rr_value is always float type
            subject_no = data_array[i]["subj_no"][0][0]  # Subject number (numeric)
            window_id = data_array[i]["win_no"][0][0]  # Window number

            # Extract rich metadata from the 'fix' field which contains both
            # mimic_details and mimic_perform_details
            try:
                # The 'fix' field contains both mimic_perform_details and mimic_details
                fix_data = data_array[i]["fix"][0][0]

                if fix_data is None or fix_data.size == 0:
                    logger.warning(f"fix field is None or empty for sample {i}")
                    raise ValueError("fix field is None or empty")

                # Extract mimic_perform_details
                try:
                    mimic_perform = fix_data["mimic_perform_details"][0][0]
                    if mimic_perform is not None and mimic_perform.size > 0:
                        # Handle both string and array data types
                        rec_no_data = mimic_perform["rec_no"][0][0]
                        start_samp_data = mimic_perform["start_samp"][0][0]
                        end_samp_data = mimic_perform["end_samp"][0][0]

                        rec_no = (
                            rec_no_data.item()
                            if hasattr(rec_no_data, "item")
                            else int(rec_no_data)
                        )
                        start_samp = (
                            start_samp_data.item()
                            if hasattr(start_samp_data, "item")
                            else int(start_samp_data)
                        )
                        end_samp = (
                            end_samp_data.item()
                            if hasattr(end_samp_data, "item")
                            else int(end_samp_data)
                        )
                    else:
                        raise ValueError("mimic_perform_details is empty")
                except (KeyError, ValueError, AttributeError, IndexError) as e:
                    logger.warning(
                        f"Could not access mimic_perform_details for sample {i}: {e}"
                    )
                    rec_no = 0
                    start_samp = 0
                    end_samp = 0

                # Extract mimic_details
                try:
                    mimic_details = fix_data["mimic_details"][0][0]

                    if mimic_details is not None and mimic_details.size > 0:
                        # Handle both string and array data types
                        subj_id_data = mimic_details["subj_id"][0]
                        rec_id_data = mimic_details["rec_id"][0]
                        file_id_data = mimic_details["file"][0]
                        group_data = mimic_details["group"][0]

                        subj_id_raw = (
                            subj_id_data.item()
                            if hasattr(subj_id_data, "item")
                            else str(subj_id_data)
                        )
                        rec_id_raw = (
                            rec_id_data.item()
                            if hasattr(rec_id_data, "item")
                            else str(rec_id_data)
                        )
                        file_id = (
                            file_id_data.item()
                            if hasattr(file_id_data, "item")
                            else str(file_id_data)
                        )
                        group = (
                            group_data.item()
                            if hasattr(group_data, "item")
                            else str(group_data)
                        )

                        # HDF5 may store IDs as string or numeric
                        try:
                            subj_id = int(subj_id_raw)
                        except (ValueError, TypeError):
                            subj_id = -1  # Use numeric sentinel if conversion fails
                        try:
                            rec_id = int(rec_id_raw)
                        except (ValueError, TypeError):
                            rec_id = -1  # Use numeric sentinel if conversion fails
                    else:
                        raise ValueError("mimic_details is empty")
                except (KeyError, ValueError, AttributeError, IndexError) as e:
                    logger.warning(
                        f"Could not access mimic_details for sample {i}: {e}"
                    )
                    subj_id = -1  # Use numeric sentinel for missing IDs
                    rec_id = -1  # Use numeric sentinel for missing IDs
                    file_id = f"unknown_{i}"
                    group = "unknown"

            except (KeyError, ValueError, AttributeError, IndexError) as e:
                logger.warning(f"Could not access fix field for sample {i}: {e}")
                subj_id = -1
                rec_id = -1  # Use numeric sentinel for missing IDs
                file_id = f"unknown_{i}"
                group = "unknown"
                rec_no = 0
                start_samp = 0
                end_samp = 0

            # Store data
            rr_labels.append(rr_value)
            # subj_id and rec_id are now always int (numeric IDs or -1 sentinel)
            subject_ids.append(subj_id)
            subject_nos.append(int(subject_no))  # Keep numeric subject number
            window_ids.append(int(window_id))

            # Store rich metadata (subj_id and rec_id are now always int)
            subj_ids.append(subj_id)
            rec_ids.append(rec_id)
            file_ids.append(file_id)
            groups.append(group)
            rec_nos.append(rec_no)
            start_samps.append(start_samp)
            end_samps.append(end_samp)

        return {
            "waveforms": waveforms,
            "rr_labels": rr_labels,
            "subject_ids": subject_ids,
            "subject_nos": subject_nos,  # Numeric subject numbers
            "window_ids": window_ids,
            # Additional rich metadata
            "subj_ids": subj_ids,
            "rec_ids": rec_ids,
            "file_ids": file_ids,
            "groups": groups,
            "rec_nos": rec_nos,
            "start_samps": start_samps,
            "end_samps": end_samps,
        }


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_mimicperformlarge_preprocessing",
    group="preprocessor",
    node=MimicPERFormLargePreprocessingConfig,
)
