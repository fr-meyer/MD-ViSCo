"""UCI dataset preprocessor.

This module provides preprocessing functionality for the UCI dataset,
which contains ECG, PPG, and ABP waveforms with blood pressure labels.

The UCI dataset includes:
- ECG waveforms
- PPG (photoplethysmography) waveforms
- ABP (arterial blood pressure) waveforms
- Blood pressure labels (SBP, DBP)
- Split into 4 parts (1-3 for training, 4 for testing)

Raw waveforms (ecg_raw, ppg_raw, abp_raw) are stored; normalization is applied
at batch time in the collate function.
"""

# Standard library imports
import logging
import os
from dataclasses import dataclass
from typing import Any

# Third-party imports
import h5py
import numpy as np
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

# Local imports
from src.script.preprocess.base_preprocessor import BaseDatasetPreprocessor
from src.script.preprocess.base_preprocessor import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class UCIPreprocessingConfig(PreprocessingConfig):
    """Configuration for UCI dataset preprocessing."""

    _target_: str = (
        "src.script.preprocess.preprocessors.uci_preprocessor.UCIDatasetPreprocessor"
    )
    dataset: str = "UCI"

    # UCI-specific defaults/overrides
    sbp_max: float = 189.98421357007769
    dbp_min: float = 50


class UCIDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor for UCI dataset.

    The UCI dataset contains ECG, PPG, and ABP waveforms with blood pressure
    labels. The dataset is split into 4 parts (1-3 for training, 4 for testing).
    This preprocessor handles the specific characteristics of UCI data including:
    - Clinical BP range analysis based on AHA guidelines
    - Data quality metrics
    - Split-specific processing

    This class follows Hydra instantiation standards with individual parameters
    and direct attribute access (self.parameter_name) throughout the code.
    """

    def __init__(self, **kwargs):
        """Initialize UCI preprocessor.

        UCI preprocessor uses default base parameters and doesn't require
        additional dataset-specific parameters beyond those in the config.

        Args:
            **kwargs: Base preprocessor parameters (dataset, input_file, output_file,
                     save_files, stats_enabled, dbp_min, sbp_max)
        """
        super().__init__(**kwargs)

    def _calculate_bp_statistics(
        self, sbp_data: np.ndarray, dbp_data: np.ndarray, split_name: str
    ) -> dict[str, Any]:
        """Calculate comprehensive blood pressure statistics for a dataset split.

        Args:
            sbp_data: Systolic blood pressure array
            dbp_data: Diastolic blood pressure array
            split_name: Name of the dataset split (train/test)

        Returns:
            Dictionary containing all calculated statistics
        """
        sbp_stats = self._calculate_descriptive_stats(sbp_data)
        dbp_stats = self._calculate_descriptive_stats(dbp_data)

        # Clinical range classifications (based on AHA guidelines)
        # Normal: SBP <120, DBP <80
        # Prehypertension: SBP 120-139, DBP 80-89
        # Hypertension: SBP ≥140, DBP ≥90

        normal_sbp = np.sum(sbp_data < 120)
        prehypertension_sbp = np.sum((sbp_data >= 120) & (sbp_data < 140))
        hypertension_sbp = np.sum(sbp_data >= 140)
        total_samples = len(sbp_data)

        normal_dbp = np.sum(dbp_data < 80)
        prehypertension_dbp = np.sum((dbp_data >= 80) & (dbp_data < 90))
        hypertension_dbp = np.sum(dbp_data >= 90)

        if total_samples == 0:
            clinical_stats = {
                "normal_sbp_pct": 0.0,
                "prehypertension_sbp_pct": 0.0,
                "hypertension_sbp_pct": 0.0,
                "normal_dbp_pct": 0.0,
                "prehypertension_dbp_pct": 0.0,
                "hypertension_dbp_pct": 0.0,
                "total_samples": 0,
            }
        else:
            clinical_stats = {
                "normal_sbp_pct": (normal_sbp / total_samples) * 100,
                "prehypertension_sbp_pct": (prehypertension_sbp / total_samples) * 100,
                "hypertension_sbp_pct": (hypertension_sbp / total_samples) * 100,
                "normal_dbp_pct": (normal_dbp / total_samples) * 100,
                "prehypertension_dbp_pct": (prehypertension_dbp / total_samples) * 100,
                "hypertension_dbp_pct": (hypertension_dbp / total_samples) * 100,
                "total_samples": total_samples,
            }

        quality_stats = {
            "sbp_nan_count": int(np.sum(np.isnan(sbp_data))),
            "dbp_nan_count": int(np.sum(np.isnan(dbp_data))),
            "sbp_negative_count": int(np.sum(sbp_data < 0)),
            "dbp_negative_count": int(np.sum(dbp_data < 0)),
        }

        return {
            "sbp": sbp_stats,
            "dbp": dbp_stats,
            "clinical": clinical_stats,
            "quality": quality_stats,
        }

    def _log_bp_statistics(self, stats: dict[str, Any], split_name: str) -> None:
        """Log blood pressure statistics in a formatted, clinical manner.

        Args:
            stats: Statistics dictionary from _calculate_bp_statistics
            split_name: Name of the dataset split
        """
        logger.info(f"=== {split_name.upper()} Blood Pressure Statistics ===")

        self._log_descriptive_stats(stats["sbp"], "SBP", "mmHg")
        self._log_descriptive_stats(stats["dbp"], "DBP", "mmHg")

        clinical = stats["clinical"]
        logger.info(
            f"Clinical Ranges (SBP): Normal={clinical['normal_sbp_pct']:.1f}%, "
            f"Prehypertension={clinical['prehypertension_sbp_pct']:.1f}%, "
            f"Hypertension={clinical['hypertension_sbp_pct']:.1f}%"
        )
        logger.info(
            f"Clinical Ranges (DBP): Normal={clinical['normal_dbp_pct']:.1f}%, "
            f"Prehypertension={clinical['prehypertension_dbp_pct']:.1f}%, "
            f"Hypertension={clinical['hypertension_dbp_pct']:.1f}%"
        )

        quality = stats["quality"]
        if quality["sbp_nan_count"] > 0 or quality["dbp_nan_count"] > 0:
            logger.warning(
                f"Data Quality: SBP NaN count={quality['sbp_nan_count']}, "
                f"DBP NaN count={quality['dbp_nan_count']}"
            )

        if quality["sbp_negative_count"] > 0 or quality["dbp_negative_count"] > 0:
            logger.warning(
                f"Data Quality: SBP negative count={quality['sbp_negative_count']}, "
                f"DBP negative count={quality['dbp_negative_count']}"
            )

        logger.info(
            f"Normalization Range: DBP_min={self.dbp_min}, SBP_max={self.sbp_max}"
        )

    def preprocess_dataset(self, input_file: str, output_file: str) -> list[str]:
        """Preprocess UCI dataset and save to HDF5 files.

        Args:
            input_file: Path to input .h5 file (used to determine base directory)
            output_file: Base path for output HDF5 files

        Returns:
            List of output file paths for train and test splits
        """
        logger.info(f"Preprocessing UCI dataset: {input_file}")

        splits: dict[str, Any] = {
            "train": {"X": [], "ABP_GRND": [], "SBP": [], "DBP": []},
            "test": {"X": [], "SBP": [], "DBP": [], "ABP_GRND": []},
        }

        base_dir = os.path.dirname(input_file)

        for part_num in range(1, 4):
            part_file = os.path.join(
                base_dir, f"UCI_Dataset_Part_{part_num}_Preprocessed.h5"
            )
            logger.info(f"Loading training part {part_num} from {part_file}...")
            with h5py.File(part_file, "r") as f:
                ppg_data = np.array(f["PPG"])
                ecg_data = np.array(f["ECG"])
                abp_data = np.array(f["ABP_GRND"])[:, np.newaxis]
                abp_norm_data = np.array(f["ABP_RNorm"])

                x_data = np.stack((ecg_data, ppg_data, abp_norm_data), axis=1)

                sbp_data = np.array(f["SBP"])
                dbp_data = np.array(f["DBP"])

                splits["train"]["X"].append(x_data)
                splits["train"]["ABP_GRND"].append(abp_data)
                splits["train"]["SBP"].append(sbp_data)
                splits["train"]["DBP"].append(dbp_data)

        test_file = os.path.join(base_dir, "UCI_Dataset_Part_4_Preprocessed.h5")
        logger.info(f"Loading test data from {test_file}...")
        with h5py.File(test_file, "r") as f:
            ppg_data = np.array(f["PPG"])
            ecg_data = np.array(f["ECG"])
            abp_data = np.array(f["ABP_GRND"])[:, np.newaxis]
            abp_norm_data = np.array(f["ABP_RNorm"])

            x_data = np.stack((ecg_data, ppg_data, abp_norm_data), axis=1)
            sbp_data = np.array(f["SBP"])
            dbp_data = np.array(f["DBP"])

            splits["test"]["X"] = x_data
            splits["test"]["SBP"] = sbp_data
            splits["test"]["DBP"] = dbp_data
            splits["test"]["ABP_GRND"] = abp_data

        for key in ["X", "ABP_GRND", "SBP", "DBP"]:
            splits["train"][key] = np.concatenate(splits["train"][key], axis=0)

        logger.info("Dataset Statistics:")
        for split_name, split_data in splits.items():
            logger.info(f"{split_name} samples: {len(split_data['X'])}")

            sbp_data = split_data["SBP"]
            dbp_data = split_data["DBP"]

            bp_stats = self._calculate_bp_statistics(sbp_data, dbp_data, split_name)
            self._log_bp_statistics(bp_stats, split_name)

        output_files = []
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split...")
            n_samples = len(split_data["X"])

            ecg_raw = []
            ppg_raw = []
            abp_raw = []
            bp_raw = []

            with tqdm(total=n_samples, desc=f"Processing {split_name} samples") as pbar:
                for i in range(n_samples):
                    current_waveforms = split_data["X"][
                        i
                    ]  # [3, T] - ECG, PPG, ABP_norm (UCI channel order)
                    ecg_raw_sample = current_waveforms[0:1]  # [1, T] - ECG
                    ppg_raw_sample = current_waveforms[1:2]  # [1, T] - PPG
                    # For ABP, use ground truth from ABP_GRND (not the normalized
                    # version in channel 2)
                    abp_grnd = split_data["ABP_GRND"][i]  # [1, T] - ABP ground truth

                    sbp = split_data["SBP"][i]
                    dbp = split_data["DBP"][i]
                    map_value = self._calculate_map(sbp, dbp)

                    ecg_raw.append(ecg_raw_sample)
                    ppg_raw.append(ppg_raw_sample)
                    abp_raw.append(abp_grnd)
                    bp_raw.append([sbp, dbp, map_value])

                    pbar.update(1)

            arrays_info = {
                "ecg_raw": np.array(ecg_raw),  # [N, 1, T]
                "ppg_raw": np.array(ppg_raw),  # [N, 1, T]
                "abp_raw": np.array(abp_raw),  # [N, 1, T]
                "bp_raw": np.array(bp_raw),  # [N, 3] - SBP, DBP, MAP
            }

            split_output_path = self._build_output_path(
                output_file, "UCI", split_name=split_name
            )
            additional_metadata = {"dataset_name": "UCI", "split": split_name}
            self._save_to_hdf5(
                split_output_path,
                arrays_info,
                subject_ids_data=None,
                additional_metadata=additional_metadata,
            )

            if self.save_files:
                logger.info(f"Verifying saved dataset for {split_name}...")
                self._verify_saved_dataset(split_output_path)
            else:
                logger.info(
                    f"Skipping dataset verification for {split_name} (save_files=False)"
                )

            output_files.append(split_output_path)

        logger.info(f"Preprocessing complete! Output files: {output_files}")
        return output_files


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_uci_preprocessing", group="preprocessor", node=UCIPreprocessingConfig
)
