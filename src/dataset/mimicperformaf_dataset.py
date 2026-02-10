"""MIMIC PERform AF dataset loader for atrial fibrillation classification.

This module loads the MIMIC PERform AF dataset (preprocessed in MD-ViSCo format)
for binary AF classification from ECG/PPG. Supports multiple segment sizes
(1024, 1250) and patient-level splitting.

Classes:
    - MimicPERFormAFConfig: Hydra config for MimicPERFormAF dataset
    - MimicPERFormAFDataset: BaseDataset implementation for AF classification

See Also:
    - src.dataset.base_dataset: BaseDataset, Sample, DatasetBaseConfig
    - src.dataset.mimicPerformLarge_dataset: MIMIC PERform Large for RR prediction
"""

import logging

# Standard library imports
from dataclasses import dataclass

import numpy as np
import torch

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.dataset.base_dataset import BaseDataset
from src.dataset.base_dataset import DatasetBaseConfig
from src.dataset.base_dataset import Sample

logger = logging.getLogger(__name__)


@dataclass
class MimicPERFormAFConfig(DatasetBaseConfig):
    """Configuration for :class:`MimicPERFormAFDataset`.

    Attributes:
        _target_: Full path to MimicPERFormAFDataset for Hydra instantiation.
        dataset_name: Fixed to "MimicPERFormAF".
        input_size: Segment length (default 1024; 1250 also supported).
        sbp_max: Unused for AF; set to satisfy base config validation.
        dbp_min: Unused for AF; set to satisfy base config validation.
        dataset_folder: Subfolder for MIMIC PERform AF preprocessed files.
        task_type: Fixed to "classification" for AF task.
    """

    _target_: str = "src.dataset.mimicperformaf_dataset.MimicPERFormAFDataset"
    dataset_name: str = "MimicPERFormAF"

    # Based on preprocessing: supports multiple segment sizes (1024, 1250)
    # Default to 1024 for compatibility with existing models
    input_size: int | None = 1024

    # No BP normalization needed for AF classification, but base config
    # enforces these fields
    # Hydra schema requires these; not used by MimicPERFormAF
    sbp_max: float | None = 189.98421357007769
    dbp_min: float | None = 50.0

    # File path structure
    dataset_folder: str = "MIMIC_PERform/AF/mdvisco_processed/"

    # AF-specific configuration
    task_type: str = "classification"  # AF classification task

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()


class MimicPERFormAFDataset(BaseDataset):
    """
    Dataset for MimicPERFormAF - Atrial Fibrillation classification from
    ECG/PPG signals.

    This dataset supports:
    - Binary AF classification (AF vs non-AF)
    - Multiple segment sizes (1024, 1250 samples) - ALL loaded and available
    - ECG and PPG signals (2 channels)
    - Signal availability tracking (ABP, IMP)
    - Patient-level splitting support

    The trainer can select which segment size to use via
    input_preprocessing configuration.
    """

    @staticmethod
    def _extract_data(f):
        """Load data from MimicPERFormAF preprocessed files (NEW format only)."""
        # NEW FORMAT ONLY: Separate vital sign fields (ECG, PPG only - no ABP/BP)
        data = {
            "ecg_raw": BaseDataset.safe_load(
                f["ecg_raw"], dtype=torch.float32
            ),  # [N, 1, T]
            "ppg_raw": BaseDataset.safe_load(
                f["ppg_raw"], dtype=torch.float32
            ),  # [N, 1, T]
            "af_labels": BaseDataset.safe_load(f["af_labels"], dtype=torch.int64),
            "abp_available": BaseDataset.safe_load(
                f["abp_available"], dtype=torch.int64
            ),
            "imp_available": BaseDataset.safe_load(
                f["imp_available"], dtype=torch.int64
            ),
        }

        # Store original lengths for padding calculation
        ecg_raw = data["ecg_raw"]
        ppg_raw = data["ppg_raw"]
        if ecg_raw is None or ppg_raw is None:
            raise ValueError("ecg_raw and ppg_raw must be loaded")
        assert ecg_raw is not None and ppg_raw is not None  # narrow for type checker
        af_labels = data["af_labels"]
        if af_labels is None:
            raise ValueError("af_labels must be loaded")
        data["ecg_original_lengths"] = torch.tensor(
            [ecg_raw.shape[-1]] * len(af_labels)
        )
        data["ppg_original_lengths"] = torch.tensor(
            [ppg_raw.shape[-1]] * len(af_labels)
        )

        data["subject_ids"] = np.array(
            [
                sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)
                for sid in f["subject_ids"][:]
            ],
            dtype=str,
        )

        data["record_ids"] = np.array(
            [
                rid.decode("utf-8") if isinstance(rid, bytes) else str(rid)
                for rid in f["record_ids"][:]
            ],
            dtype=str,
        )

        data["file_ids"] = np.array(
            [
                fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                for fid in f["file_ids"][:]
            ],
            dtype=str,
        )

        # Store size information
        data["size"] = len(data["subject_ids"])

        if "segment_size" in f.attrs:
            data["segment_size"] = f.attrs["segment_size"]
        else:
            # Infer from waveform shape
            waveform_shape = ecg_raw.shape
            if len(waveform_shape) >= 2:
                data["segment_size"] = waveform_shape[-1]
            else:
                data["segment_size"] = 1024  # Default fallback

        logger.info(
            f"Loaded dataset with segment size: {data['segment_size']}, total samples: {
                data['size']
            }"
        )

        return data

    def __init__(self, task_type: str = "classification", *args, **kwargs):
        """Initialize MimicPERFormAF dataset.

        Args:
            task_type: Type of task (default "classification").
            *args: Additional positional arguments passed to BaseDataset.
            **kwargs: Additional keyword arguments passed to BaseDataset.
        """
        super().__init__(*args, **kwargs)

        self.task_type = task_type

        self.load_shared_data(self.sample_file, self._extract_data)
        self.data = self._shared_data[self.sample_file]

        logger.info("MimicPerformAF loaded with NEW format (vital-specific fields)")

        self.segment_size = self.data.get("segment_size", 1024)

    def __len__(self):
        """Get dataset length."""
        return self.data["size"]

    def __getitem__(self, idx: int) -> Sample:
        """Return a Sample with raw sequences (no preprocessing).

        Waveforms are returned at their original lengths without padding/trimming.
        The collate function handles:
        - Padding/trimming to uniform length
        - Normalization (via input_preprocessing config)

        This enables batch-optimal padding and flexible window sizes.
        """
        # NEW ARCHITECTURE: Return raw sequences without preprocessing
        # Padding/trimming handled at batch time in collate function

        # Padding/trimming applied at collate; dataset returns raw
        ecg_raw = self.data["ecg_raw"][idx]  # [1, T_original]
        ppg_raw = self.data["ppg_raw"][idx]  # [1, T_original]

        # No preprocessing applied, padding_length is 0 (padding happens in collate)
        padding_length = 0

        return Sample(
            ecg_raw=ecg_raw,
            ppg_raw=ppg_raw,
            abp_raw=None,
            imp_raw=None,
            bp_raw=None,
            sample_index=idx,
            subject_id=f"{self.data['subject_ids'][idx]}_{
                self.data['record_ids'][idx]
            }",
            af_labels=self.data["af_labels"][idx],
            padding_length=padding_length,
            vital_sign_type="ECG",  # AF classification uses ECG waveforms
            extract_scalars=False,  # AF classification doesn't need scalar extraction
        )

    def get_normalization_params(self) -> None:
        """Return None; MimicPerformAF has no BP normalization parameters."""
        return None

    def get_af_label_distribution(self) -> dict[str, int | float]:
        """Return the distribution of AF labels (counts and percentage).

        Returns:
            Dict with keys af_count, non_af_count, total_count, af_percentage.
        """
        af_labels = self.data["af_labels"]
        af_count = int(torch.sum(af_labels == 1).item())
        non_af_count = int(torch.sum(af_labels == 0).item())
        total_count = len(af_labels)

        # Guard against division by zero
        if total_count == 0:
            af_percentage = 0.0
            non_af_percentage = 0.0
        else:
            af_percentage = (af_count / total_count) * 100
            non_af_percentage = (non_af_count / total_count) * 100

        return {
            "af_count": af_count,
            "non_af_count": non_af_count,
            "total_count": total_count,
            "af_percentage": af_percentage,
            "non_af_percentage": non_af_percentage,
        }

    def get_signal_availability_stats(self) -> dict[str, dict[str, int | float]]:
        """Get statistics about signal availability."""
        abp_available = self.data["abp_available"]
        imp_available = self.data["imp_available"]

        abp_count = int(torch.sum(abp_available).item())
        imp_count = int(torch.sum(imp_available).item())
        total_count = len(abp_available)

        # Guard against division by zero
        if total_count == 0:
            abp_percentage = 0.0
            imp_percentage = 0.0
        else:
            abp_percentage = (abp_count / total_count) * 100
            imp_percentage = (imp_count / total_count) * 100

        return {
            "abp_availability": {
                "available": abp_count,
                "unavailable": total_count - abp_count,
                "percentage": abp_percentage,
            },
            "imp_availability": {
                "available": imp_count,
                "unavailable": total_count - imp_count,
                "percentage": imp_percentage,
            },
        }

    @property
    def supports_patient_split(self) -> bool:
        """Check if MimicPERFormAF supports patient-level splitting."""
        return True

    def get_patient_ids(self) -> list[str]:
        """Get unique patient IDs for patient-level splitting."""
        subject_ids = self.data["subject_ids"]
        return list(set(subject_ids))


# Register with Hydra ConfigStore
if __name__ != "__main__":
    cs = ConfigStore.instance()
    cs.store(
        name="base_mimicperformaf", node=MimicPERFormAFConfig, group="train_dataset"
    )
    cs.store(
        name="base_mimicperformaf", node=MimicPERFormAFConfig, group="test_dataset"
    )
