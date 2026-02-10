"""PulseDB dataset loader for cuff-less blood pressure estimation.

This module loads the PulseDB dataset (preprocessed in MD-ViSCo format) for
cuff-less blood pressure estimation. Data includes ECG, PPG, ABP waveforms,
BP scalars (SBP/DBP/MAP), and optional demographics (age, gender, height,
weight, BMI). Shared memory and lazy normalization follow BaseDataset contract.

Classes:
    - PulseDBConfig: Hydra config for PulseDB dataset
    - PulseDBDataset: BaseDataset implementation for PulseDB

See Also:
    - src.dataset.base_dataset: BaseDataset, Sample, DatasetBaseConfig
    - src.dataset.uci_dataset: UCIDataset for UCI BP estimation
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

# Demographics Storage:
# Demographics are stored as 5 separate fields in HDF5: age, gender, height, weight, bmi
# All demographic fields are loaded individually and populated in the Sample object.
# See src/dataset/base_dataset.py Sample dataclass for field definitions.


@dataclass
class PulseDBConfig(DatasetBaseConfig):
    """Configuration for :class:`PulseDBDataset`.

    Attributes:
        _target_: Full path to PulseDBDataset for Hydra instantiation.
        dataset_name: Fixed to "PulseDB".
        input_size: Sequence length (default 1280).
        sbp_max: Maximum SBP for normalization (PulseDB range).
        dbp_min: Minimum DBP for normalization (PulseDB range).
        dataset_folder: Subfolder for PulseDB preprocessed files.
    """

    _target_: str = "src.dataset.pulsedb_dataset.PulseDBDataset"
    dataset_name: str = "PulseDB"

    input_size: int | None = 1280

    # PulseDB BP value ranges
    sbp_max: float | None = 286.58240014784946
    dbp_min: float | None = 2.341260731456743

    # File path structure (inherited from base)
    dataset_folder: str = "PulseDB/mdvisco_processed/PulseDB"

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()


class PulseDBDataset(BaseDataset):
    """BaseDataset for PulseDB cuff-less blood pressure estimation.

    Loads ECG, PPG, ABP waveforms, BP scalars (SBP/DBP/MAP), and optional
    demographics from preprocessed HDF5. Shared memory and lazy
    normalization follow BaseDataset contract.
    """

    @staticmethod
    def _extract_data(f):
        shared_data = {
            "ecg_raw": BaseDataset.safe_load(
                f["ecg_raw"], dtype=torch.float32
            ),  # [N, 1, T]
            "ppg_raw": BaseDataset.safe_load(
                f["ppg_raw"], dtype=torch.float32
            ),  # [N, 1, T]
            "abp_raw": BaseDataset.safe_load(
                f["abp_raw"], dtype=torch.float32
            ),  # [N, 1, T]
            "bp_raw": BaseDataset.safe_load(f["bp_raw"], dtype=torch.float32),  # [N, 3]
            "subject_ids": np.array(
                [
                    (sid.decode("utf-8") if isinstance(sid, bytes) else str(sid))
                    .removeprefix("['")
                    .removesuffix("']")
                    for sid in f["subject_ids"][:]
                ],
                dtype=str,
            ),
        }

        # Store original lengths for padding calculation
        # Assumption: All samples have uniform length within the dataset.
        # Using constant shape[-1] for all samples. If variable lengths exist,
        # replace with per-sample lengths stored during preprocessing.
        shared_data["ecg_original_lengths"] = torch.tensor(
            [shared_data["ecg_raw"].shape[-1]] * len(shared_data["bp_raw"])
        )
        shared_data["ppg_original_lengths"] = torch.tensor(
            [shared_data["ppg_raw"].shape[-1]] * len(shared_data["bp_raw"])
        )
        shared_data["abp_original_lengths"] = torch.tensor(
            [shared_data["abp_raw"].shape[-1]] * len(shared_data["bp_raw"])
        )

        # Verify individual demographic fields exist (error if old format)
        required_demo_fields = ["age", "gender", "height", "weight", "bmi"]
        missing = [k for k in required_demo_fields if k not in f]
        if missing:
            raise ValueError(
                f"Missing demographic datasets {missing}. "
                f"This HDF5 file appears to be in an old format. "
                f"Please regenerate using the latest preprocessor."
            )

        # float32 for model inputs
        shared_data["age"] = BaseDataset.safe_load(f["age"], dtype=torch.float32)  # [N]
        shared_data["gender"] = BaseDataset.safe_load(
            f["gender"], dtype=torch.float32
        )  # [N]
        shared_data["height"] = BaseDataset.safe_load(
            f["height"], dtype=torch.float32
        )  # [N]
        shared_data["weight"] = BaseDataset.safe_load(
            f["weight"], dtype=torch.float32
        )  # [N]
        shared_data["bmi"] = BaseDataset.safe_load(f["bmi"], dtype=torch.float32)  # [N]

        shared_data["size"] = len(f["bp_raw"])
        return shared_data

    def __init__(self, *args, **kwargs):
        """Initialize PulseDB dataset.

        Args:
            *args: Positional arguments passed to BaseDataset.
            **kwargs: Keyword arguments passed to BaseDataset.
        """
        super().__init__(*args, **kwargs)

        # Use new base method for shared memory
        self.load_shared_data(self.sample_file, self._extract_data)
        self.data = self._shared_data[self.sample_file]

        # Store BP normalization constants (from config)
        self.sbp_max = kwargs.get("sbp_max", 286.58240014784946)
        self.dbp_min = kwargs.get("dbp_min", 2.341260731456743)

        # Verify initialization (explicit check; assertions disabled in optimized runs)
        actual_size = len(self.data["bp_raw"])
        expected_size = self.data["size"]
        if actual_size != expected_size:
            raise ValueError(
                f"Data size mismatch in PulseDB dataset: "
                f"expected {expected_size} samples but found {actual_size} in bp_raw"
            )

    def __len__(self):
        """Get dataset length."""
        return self.data["size"]

    def __getitem__(self, idx: int) -> Sample:
        """Get item with shared memory tensors - returns Sample with raw sequences.

        This method returns raw waveforms without preprocessing.
        Padding/trimming is performed at batch time in the collate function
        for better memory efficiency and flexibility.

        Waveforms are variable-length and will be resized to uniform
        length during batching.
        Original lengths are tracked in *_original_lengths tensors for validation.
        """
        # NEW ARCHITECTURE: Return raw sequences without preprocessing
        # Padding/trimming is now handled at batch time in collate function
        # This enables:
        # - Batch-optimal padding (pad to max in batch, not global max)
        # - Memory savings (store raw sequences, pad only when needed)
        # - Flexibility (change window size without reprocessing datasets)

        # Padding/trimming applied at collate; dataset returns raw
        ecg_raw = self.data["ecg_raw"][idx]  # [1, T_original]
        ppg_raw = self.data["ppg_raw"][idx]  # [1, T_original]
        abp_raw = self.data["abp_raw"][idx]  # [1, T_original]

        # No preprocessing applied, padding_length is 0 (padding happens in collate)
        padding_length = 0

        bp_tensor = self.data["bp_raw"][idx]  # already tensor

        # Extract individual demographic fields
        # Collate expects [1] shape for scalar fields
        age_value = self.data["age"][idx]
        gender_value = self.data["gender"][idx]
        height_value = self.data["height"][idx]
        weight_value = self.data["weight"][idx]
        bmi_value = self.data["bmi"][idx]

        # Normalize to [1] for both 0D and 1D so collate sees uniform shape
        age_tensor = torch.atleast_1d(
            age_value.clone()
            if torch.is_tensor(age_value)
            else torch.tensor(age_value, dtype=torch.float32)
        )
        gender_tensor = torch.atleast_1d(
            gender_value.clone()
            if torch.is_tensor(gender_value)
            else torch.tensor(gender_value, dtype=torch.float32)
        )
        height_tensor = torch.atleast_1d(
            height_value.clone()
            if torch.is_tensor(height_value)
            else torch.tensor(height_value, dtype=torch.float32)
        )
        weight_tensor = torch.atleast_1d(
            weight_value.clone()
            if torch.is_tensor(weight_value)
            else torch.tensor(weight_value, dtype=torch.float32)
        )
        bmi_tensor = torch.atleast_1d(
            bmi_value.clone()
            if torch.is_tensor(bmi_value)
            else torch.tensor(bmi_value, dtype=torch.float32)
        )

        return Sample(
            # Core identification
            sample_index=idx,
            # Vital-specific raw data
            ecg_raw=ecg_raw,  # [1, T]
            ppg_raw=ppg_raw,  # [1, T]
            abp_raw=abp_raw,  # [1, T]
            imp_raw=None,  # Not available in PulseDB
            # BP data
            bp_raw=bp_tensor,
            # Metadata
            subject_id=self.data["subject_ids"][idx],
            # Individual demographic fields
            age_raw=age_tensor,
            gender_raw=gender_tensor,
            height_raw=height_tensor,
            weight_raw=weight_tensor,
            bmi_raw=bmi_tensor,
            padding_length=padding_length,
            vital_sign_type="ABP",  # PulseDB dataset is for ABP prediction
            extract_scalars=True,  # Always extract SBP/DBP for BP tasks
        )

    def get_normalization_params(self) -> dict[str, float]:
        """Get PulseDB-specific BP normalization parameters.

        Returns:
            Dict with sbp_max and dbp_min for global normalization
        """
        return {"sbp_max": self.sbp_max, "dbp_min": self.dbp_min}

    @property
    def supports_patient_split(self) -> bool:
        """Check if dataset supports patient-level splitting."""
        return True


# Register with Hydra ConfigStore
if __name__ != "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_pulsedb", node=PulseDBConfig, group="train_dataset")
    cs.store(name="base_pulsedb", node=PulseDBConfig, group="test_dataset")
