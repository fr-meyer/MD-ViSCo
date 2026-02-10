"""PulseDB dataset preprocessor.

This module provides preprocessing functionality for the PulseDB dataset,
which contains ECG (lead II), PPG, and ABP waveforms with blood pressure
labels and demographic information.

The PulseDB dataset includes:
- ECG (lead II) waveforms
- PPG (photoplethysmography) waveforms
- ABP (arterial blood pressure) waveforms
- Blood pressure labels (SBP, DBP)
- Demographic information (saved as separate fields: age, gender, height, weight, bmi)
- Subject IDs for tracking individual patients

REFACTORING NOTE (Lazy Normalization):
This preprocessor now saves ONLY raw waveforms per vital sign (ecg_raw, ppg_raw,
abp_raw).
Normalization is performed at batch time in the collate function (lazy normalization).
This reduces storage by ~75% and enables flexible normalization strategies.

Demographics Storage:
Demographics are saved as 5 separate HDF5 datasets: age, gender, height, weight, bmi.
These individual fields are loaded by the dataset and used throughout the pipeline.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np
from hydra.core.config_store import ConfigStore
from mat73 import loadmat
from tqdm import tqdm

# Local imports
from src.script.preprocess.base_preprocessor import BaseDatasetPreprocessor
from src.script.preprocess.base_preprocessor import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class PulseDBPreprocessingConfig(PreprocessingConfig):
    """Configuration for PulseDB dataset preprocessing."""

    _target_: str = (
        "src.script.preprocess.preprocessors.pulsedb_preprocessor."
        "PulseDBDatasetPreprocessor"
    )
    dataset: str = "PulseDB"

    # PulseDB-specific defaults/overrides
    dbp_min: float = 2.341260731456743
    sbp_max: float = 286.58240014784946

    # Signal labels
    ecg_label: int = 0
    ppg_label: int = 1
    abp_label: int = 2

    # Statistics controls
    stats_enabled: bool = True


class PulseDBDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor for PulseDB dataset.

    The PulseDB dataset contains ECG (lead II), PPG, and ABP waveforms with
    blood pressure labels and demographic information. This preprocessor
    handles the specific characteristics of PulseDB data including:
    - Subject-level statistics and demographics
    - Blood pressure threshold analysis
    - Clinical categorization of BP measurements

    This class follows Hydra instantiation standards with individual parameters
    and direct attribute access (self.parameter_name) throughout the code.
    """

    def __init__(
        self, ecg_label: int = 0, ppg_label: int = 1, abp_label: int = 2, **kwargs
    ):
        """Initialize PulseDB preprocessor with dataset-specific parameters.

        Args:
            ecg_label: Channel index for ECG signal
            ppg_label: Channel index for PPG signal
            abp_label: Channel index for ABP signal
            **kwargs: Base preprocessor parameters (dataset, input_file,
                output_file, etc.)
        """
        super().__init__(**kwargs)
        self.ecg_label = ecg_label
        self.ppg_label = ppg_label
        self.abp_label = abp_label

    def preprocess_dataset(self, input_file: str, output_file: str) -> str:
        """Preprocess PulseDB dataset and save to HDF5 file.

        Args:
            input_file: Path to input .mat file
            output_file: Base path for output HDF5 file

        Returns:
            Output file path
        """
        logger.info(f"Preprocessing PulseDB dataset: {input_file}")
        complete_dataset = self._build_pulsedb_dataset(input_file)

        if getattr(self, "stats_enabled", True):
            logger.info("Calculating dataset statistics...")
            stats = self._calculate_pulsedb_statistics(complete_dataset)
            self._log_pulsedb_statistics(stats)
        waveforms = complete_dataset["waveforms"]
        sbp_dbp = complete_dataset["bp_labels"]
        demographics = complete_dataset["demographics"]
        subject_ids = complete_dataset["subject_ids"]

        n_samples = len(waveforms)
        logger.info(f"Total samples: {n_samples}")

        ecg_raw_list = []
        ppg_raw_list = []
        abp_raw_list = []
        bp_raw_list = []
        demographics_list = []
        subject_ids_list = []

        with tqdm(total=n_samples, desc="Processing samples") as pbar:
            for i in range(n_samples):
                current_waveforms = waveforms[i]
                current_demographics = demographics[i]
                current_subject_id = subject_ids[i]

                ecg_raw = current_waveforms[
                    self.ecg_label : self.ecg_label + 1
                ]  # [1, T]
                ppg_raw = current_waveforms[
                    self.ppg_label : self.ppg_label + 1
                ]  # [1, T]
                abp_raw = current_waveforms[
                    self.abp_label : self.abp_label + 1
                ]  # [1, T]

                sbp, dbp = sbp_dbp[i]
                map_value = self._calculate_map(sbp, dbp)

                ecg_raw_list.append(ecg_raw)
                ppg_raw_list.append(ppg_raw)
                abp_raw_list.append(abp_raw)
                bp_raw_list.append([sbp, dbp, map_value])
                demographics_list.append(current_demographics)
                subject_ids_list.append(current_subject_id)

                pbar.update(1)

        ecg_raw = np.array(ecg_raw_list)  # [N, 1, T]
        ppg_raw = np.array(ppg_raw_list)  # [N, 1, T]
        abp_raw = np.array(abp_raw_list)  # [N, 1, T]
        bp_raw = np.array(bp_raw_list)
        demographics_array = np.array(demographics_list)
        subject_ids_array = np.array(subject_ids_list)

        age_array = demographics_array[:, 0]
        gender_array = demographics_array[:, 1]
        height_array = demographics_array[:, 2]
        weight_array = demographics_array[:, 3]
        bmi_array = demographics_array[:, 4]

        # Cast to explicit numeric dtypes and ensure 1D shape for HDF5 compatibility
        age_array = age_array.astype(np.float32).reshape(-1)
        gender_array = gender_array.astype(np.int8).reshape(-1)
        height_array = height_array.astype(np.float32).reshape(-1)
        weight_array = weight_array.astype(np.float32).reshape(-1)
        bmi_array = bmi_array.astype(np.float32).reshape(-1)

        arrays_info = {
            # Separate raw waveforms per vital sign
            "ecg_raw": ecg_raw,  # [N, 1, T]
            "ppg_raw": ppg_raw,  # [N, 1, T]
            "abp_raw": abp_raw,  # [N, 1, T]
            # Blood pressure data
            "bp_raw": bp_raw,  # [N, 3] - SBP, DBP, MAP
            # Individual demographic fields
            "age": age_array,  # [N] - Individual age values
            "gender": gender_array,  # [N] - Individual gender values (0/1 encoding)
            "height": height_array,  # [N] - Individual height values
            "weight": weight_array,  # [N] - Individual weight values
            "bmi": bmi_array,  # [N] - Individual BMI values
        }

        # Handle subject_ids separately with proper string encoding
        subject_ids_data = np.array(
            [str(s).encode("ascii") for s in subject_ids_array], dtype="S9"
        )

        file_name = Path(input_file).stem
        output_path = self._build_output_path(
            output_file, "PulseDB", file_stem=file_name
        )
        additional_metadata = {"dataset_name": "PulseDB"}
        self._save_to_hdf5(
            output_path, arrays_info, subject_ids_data, additional_metadata
        )

        logger.info(
            "Demographics saved as 5 individual fields: age, gender, height, "
            "weight, bmi"
        )
        logger.info(f"  - age: {age_array.shape}")
        logger.info(f"  - gender: {gender_array.shape}")
        logger.info(f"  - height: {height_array.shape}")
        logger.info(f"  - weight: {weight_array.shape}")
        logger.info(f"  - bmi: {bmi_array.shape}")

        if self.save_files:
            logger.info("Verifying saved dataset...")
            self._verify_saved_dataset(output_path)
        else:
            logger.info("Skipping dataset verification (save_files=False)")

        return output_path

    def _build_pulsedb_dataset(self, path: str, field_name: str = "Subset") -> dict:
        """Load PulseDB dataset from .mat file."""
        data = loadmat(path)
        waveforms = data[field_name]["Signals"]
        sbp_labels = data[field_name]["SBP"]
        dbp_labels = data[field_name]["DBP"]
        sbp_dbp_labels = np.stack((sbp_labels, dbp_labels), axis=1)
        age = data[field_name]["Age"]
        gender_array = data[field_name]["Gender"]
        gender = np.array([1 if g[0] == "M" else 0 for g in gender_array])
        height = data[field_name]["Height"]
        weight = data[field_name]["Weight"]
        bmi = data[field_name]["BMI"]
        subject_ids = data[field_name]["Subject"]
        demographics = np.stack((age, gender, height, weight, bmi), axis=1)
        return {
            "waveforms": waveforms,
            "bp_labels": sbp_dbp_labels,
            "demographics": demographics,
            "subject_ids": subject_ids,
        }

    def _calculate_pulsedb_statistics(self, complete_dataset: dict) -> dict:
        """Calculate comprehensive statistics for PulseDB dataset.

        Returns a dictionary with sample-level BP ranges, subject-level
        thresholds/categories, measurements-per-subject, and age statistics.
        """
        sbp_dbp = complete_dataset["bp_labels"]
        demographics = complete_dataset["demographics"]
        subject_ids_raw = complete_dataset["subject_ids"]

        # Normalize subject ids to flat string array for grouping
        try:
            subject_ids = np.array([str(s) for s in np.ravel(subject_ids_raw)])
        except Exception:
            subject_ids = np.array([str(s) for s in subject_ids_raw])

        sbp_data = np.asarray(sbp_dbp)[:, 0].astype(float)
        dbp_data = np.asarray(sbp_dbp)[:, 1].astype(float)

        measurements_per_subject = self._calculate_measurements_per_subject(subject_ids)

        bp_range_stats = {
            "sbp": self._calculate_descriptive_stats(sbp_data),
            "dbp": self._calculate_descriptive_stats(dbp_data),
        }

        subject_to_indices = self._build_subject_groups(subject_ids)

        subjects_sbp_high = sum(
            1 for idxs in subject_to_indices.values() if np.any(sbp_data[idxs] > 180.0)
        )
        subjects_sbp_low = sum(
            1 for idxs in subject_to_indices.values() if np.any(sbp_data[idxs] < 100.0)
        )
        subjects_dbp_high = sum(
            1 for idxs in subject_to_indices.values() if np.any(dbp_data[idxs] > 100.0)
        )
        subjects_dbp_low = sum(
            1 for idxs in subject_to_indices.values() if np.any(dbp_data[idxs] < 60.0)
        )

        bp_distribution_stats = {
            "sbp_high_count": subjects_sbp_high,
            "sbp_low_count": subjects_sbp_low,
            "dbp_high_count": subjects_dbp_high,
            "dbp_low_count": subjects_dbp_low,
            "total_subjects": int(len(subject_to_indices)),
        }

        subjects_sbp_lt_90 = sum(
            1 for idxs in subject_to_indices.values() if np.any(sbp_data[idxs] < 90.0)
        )
        subjects_sbp_90_129 = sum(
            1
            for idxs in subject_to_indices.values()
            if np.any((sbp_data[idxs] >= 90.0) & (sbp_data[idxs] <= 129.0))
        )
        subjects_sbp_130_160 = sum(
            1
            for idxs in subject_to_indices.values()
            if np.any((sbp_data[idxs] >= 130.0) & (sbp_data[idxs] <= 160.0))
        )
        subjects_sbp_161_180 = sum(
            1
            for idxs in subject_to_indices.values()
            if np.any((sbp_data[idxs] >= 161.0) & (sbp_data[idxs] <= 180.0))
        )
        subjects_sbp_gt_180 = sum(
            1 for idxs in subject_to_indices.values() if np.any(sbp_data[idxs] > 180.0)
        )

        sbp_categories = {
            "sbp_lt_90": subjects_sbp_lt_90,
            "sbp_90_129": subjects_sbp_90_129,
            "sbp_130_160": subjects_sbp_130_160,
            "sbp_161_180": subjects_sbp_161_180,
            "sbp_gt_180": subjects_sbp_gt_180,
        }

        subjects_dbp_lt_60 = sum(
            1 for idxs in subject_to_indices.values() if np.any(dbp_data[idxs] < 60.0)
        )
        subjects_dbp_60_79 = sum(
            1
            for idxs in subject_to_indices.values()
            if np.any((dbp_data[idxs] >= 60.0) & (dbp_data[idxs] <= 79.0))
        )
        subjects_dbp_80_100 = sum(
            1
            for idxs in subject_to_indices.values()
            if np.any((dbp_data[idxs] >= 80.0) & (dbp_data[idxs] <= 100.0))
        )
        subjects_dbp_101_110 = sum(
            1
            for idxs in subject_to_indices.values()
            if np.any((dbp_data[idxs] >= 101.0) & (dbp_data[idxs] <= 110.0))
        )
        subjects_dbp_gt_110 = sum(
            1 for idxs in subject_to_indices.values() if np.any(dbp_data[idxs] > 110.0)
        )

        dbp_categories = {
            "dbp_lt_60": subjects_dbp_lt_60,
            "dbp_60_79": subjects_dbp_60_79,
            "dbp_80_100": subjects_dbp_80_100,
            "dbp_101_110": subjects_dbp_101_110,
            "dbp_gt_110": subjects_dbp_gt_110,
        }

        age_col = np.asarray(demographics)[:, 0].astype(float)
        subj_ages = np.array(
            [float(age_col[idxs[0]]) for idxs in subject_to_indices.values()]
        )
        age_stats = self._calculate_descriptive_stats(subj_ages)

        return {
            "measurements_per_subject": measurements_per_subject,
            "bp_range": bp_range_stats,
            "bp_distribution": bp_distribution_stats,
            "sbp_categories": sbp_categories,
            "dbp_categories": dbp_categories,
            "age_stats": age_stats,
        }

    def _log_pulsedb_statistics(self, stats: dict) -> None:
        """Log PulseDB statistics in a formatted, clinical manner."""
        logger.info("=== PulseDB Dataset Statistics ===")

        self._log_measurements_per_subject(stats["measurements_per_subject"], "PulseDB")

        self._log_descriptive_stats(stats["bp_range"]["sbp"], "SBP Range", "mmHg")
        self._log_descriptive_stats(stats["bp_range"]["dbp"], "DBP Range", "mmHg")

        total_subjects = stats["bp_distribution"]["total_subjects"]
        bp_dist = stats["bp_distribution"]
        logger.info("Subject-level BP Thresholds (ANY measurement):")
        logger.info(
            f"  SBP > 180 mmHg: {bp_dist['sbp_high_count']} ({
                (bp_dist['sbp_high_count'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  SBP < 100 mmHg: {bp_dist['sbp_low_count']} ({
                (bp_dist['sbp_low_count'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  DBP > 100 mmHg: {bp_dist['dbp_high_count']} ({
                (bp_dist['dbp_high_count'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  DBP < 60 mmHg: {bp_dist['dbp_low_count']} ({
                (bp_dist['dbp_low_count'] / total_subjects * 100):.1f}%)"
        )

        sbp_cat = stats["sbp_categories"]
        logger.info("SBP Categories (subjects with ANY measurement in category):")
        logger.info(
            f"  <90 mmHg: {sbp_cat['sbp_lt_90']} ({
                (sbp_cat['sbp_lt_90'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  90–129 mmHg: {sbp_cat['sbp_90_129']} ({
                (sbp_cat['sbp_90_129'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  130–160 mmHg: {sbp_cat['sbp_130_160']} ({
                (sbp_cat['sbp_130_160'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  161–180 mmHg: {sbp_cat['sbp_161_180']} ({
                (sbp_cat['sbp_161_180'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  >180 mmHg: {sbp_cat['sbp_gt_180']} ({
                (sbp_cat['sbp_gt_180'] / total_subjects * 100):.1f}%)"
        )

        dbp_cat = stats["dbp_categories"]
        logger.info("DBP Categories (subjects with ANY measurement in category):")
        logger.info(
            f"  <60 mmHg: {dbp_cat['dbp_lt_60']} ({
                (dbp_cat['dbp_lt_60'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  60–79 mmHg: {dbp_cat['dbp_60_79']} ({
                (dbp_cat['dbp_60_79'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  80–100 mmHg: {dbp_cat['dbp_80_100']} ({
                (dbp_cat['dbp_80_100'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  101–110 mmHg: {dbp_cat['dbp_101_110']} ({
                (dbp_cat['dbp_101_110'] / total_subjects * 100):.1f}%)"
        )
        logger.info(
            f"  >110 mmHg: {dbp_cat['dbp_gt_110']} ({
                (dbp_cat['dbp_gt_110'] / total_subjects * 100):.1f}%)"
        )

        self._log_descriptive_stats(stats["age_stats"], "Age Range", "years")


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_pulsedb_preprocessing",
    group="preprocessor",
    node=PulseDBPreprocessingConfig,
)
