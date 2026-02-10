"""MIMIC PERform Large dataset loader for respiratory rate prediction.

This module loads the MIMIC PERform Large dataset (preprocessed in MD-ViSCo
format) for respiratory rate (RR) prediction from ECG, PPG, and IMP. Signals
are length 4001 at 125 Hz; target vital type defaults to IMP.

Classes:
    - MimicPERFormLargeConfig: Hydra config for MimicPERFormLarge dataset
    - MimicPERFormLargeDataset: BaseDataset implementation for RR prediction

See Also:
    - src.dataset.base_dataset: BaseDataset, Sample, DatasetBaseConfig
    - src.dataset.mimicPerformAF_dataset: MIMIC PERform AF for AF classification
"""

# Standard library imports
from dataclasses import dataclass

import torch

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.dataset.base_dataset import BaseDataset
from src.dataset.base_dataset import DatasetBaseConfig
from src.dataset.base_dataset import Sample


@dataclass
class MimicPERFormLargeConfig(DatasetBaseConfig):
    """Configuration for :class:`MimicPERFormLargeDataset`.

    Attributes:
        _target_: Full path to MimicPERFormLargeDataset for Hydra instantiation.
        dataset_name: Fixed to "MimicPERFormLarge".
        input_size: Sequence length (default 4001 at 125 Hz).
        sbp_max: Unused for RR; set to satisfy base config validation.
        dbp_min: Unused for RR; set to satisfy base config validation.
        dataset_folder: Subfolder for MIMIC PERform Large preprocessed files.
        target_vital_type: Target vital for extractor (default 'IMP').
        extract_scalars_default: Default False for RR prediction.
    """

    _target_: str = "src.dataset.mimicperformlarge_dataset.MimicPERFormLargeDataset"
    dataset_name: str = "MimicPERFormLarge"

    # Based on preprocessing: signals are length 4001 at 125 Hz
    input_size: int | None = 4001

    # No BP normalization needed here, but base config enforces these fields
    # Hydra schema requires these; not used by MimicPERFormLarge
    sbp_max: float | None = 189.98421357007769
    dbp_min: float | None = 50.0

    # File path structure
    dataset_folder: str = "MIMIC_PERform/Large/mdvisco_processed/"

    target_vital_type: str = "IMP"  # Default target vital type for RR prediction
    extract_scalars_default: bool = (
        False  # RR prediction doesn't need scalar extraction
    )


class MimicPERFormLargeDataset(BaseDataset):
    """BaseDataset for MIMIC PERform Large respiratory rate (RR) prediction.

    Loads ECG, PPG, IMP waveforms and RR labels from preprocessed HDF5.
    Supports patient-level splitting and shared memory. Target vital type
    defaults to IMP for RR prediction.
    """

    @staticmethod
    def _extract_data(f):
        """Load data from MimicPERFormLarge preprocessed files (NEW format only)."""
        data = {
            "ecg_raw": BaseDataset.safe_load(f["ecg_raw"], dtype=torch.float32),
            "ppg_raw": BaseDataset.safe_load(f["ppg_raw"], dtype=torch.float32),
            "imp_raw": BaseDataset.safe_load(f["imp_raw"], dtype=torch.float32),
            # Labels and IDs - preserve integer types
            "rr_labels": BaseDataset.safe_load(f["rr_labels"], dtype=torch.int64),
            "subject_ids": BaseDataset.safe_load(f["subject_ids"], dtype=torch.int64),
            "subject_nos": BaseDataset.safe_load(f["subject_nos"], dtype=torch.int64),
            "window_ids": BaseDataset.safe_load(f["window_ids"], dtype=torch.int64),
            # Rich metadata (optional) - preserve integer types
            "subj_ids": BaseDataset.safe_load(f["subj_ids"], dtype=torch.int64),
            "rec_ids": BaseDataset.safe_load(f["rec_ids"], dtype=torch.int64),
            "rec_nos": BaseDataset.safe_load(f["rec_nos"], dtype=torch.int64),
            "start_samps": BaseDataset.safe_load(f["start_samps"], dtype=torch.int64),
            "end_samps": BaseDataset.safe_load(f["end_samps"], dtype=torch.int64),
        }
        # Derive dataset size from waveforms
        ecg_raw = data["ecg_raw"]
        ppg_raw = data["ppg_raw"]
        imp_raw = data["imp_raw"]
        if ecg_raw is None or ppg_raw is None or imp_raw is None:
            raise ValueError("ecg_raw, ppg_raw, and imp_raw must be loaded")
        data["size"] = ecg_raw.shape[0]

        # Store original lengths for padding calculation
        data["ecg_original_lengths"] = torch.tensor([ecg_raw.shape[-1]] * data["size"])
        data["ppg_original_lengths"] = torch.tensor([ppg_raw.shape[-1]] * data["size"])
        data["imp_original_lengths"] = torch.tensor([imp_raw.shape[-1]] * data["size"])

        return data

    def __init__(
        self,
        target_vital_type: str = "IMP",
        extract_scalars_default: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize MimicPERFormLarge dataset.

        Args:
            target_vital_type: Target vital sign type (default "IMP").
            extract_scalars_default: Whether to extract scalars by default
                (default False).
            *args: Additional positional arguments passed to BaseDataset.
            **kwargs: Additional keyword arguments passed to BaseDataset.
        """
        super().__init__(*args, **kwargs)
        self.load_shared_data(self.sample_file, self._extract_data)
        self.data = self._shared_data[self.sample_file]

        # Store processor metadata configuration
        self.target_vital_type = target_vital_type
        self.extract_scalars_default = extract_scalars_default

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.data["size"]

    def __getitem__(self, idx: int) -> Sample:
        """Return a Sample with raw sequences (no preprocessing).

        Waveforms are returned at their original lengths. Padding/trimming
        is performed at batch time in the collate function.
        """
        ecg_raw = self.data["ecg_raw"][idx]
        ppg_raw = self.data["ppg_raw"][idx]
        imp_raw = self.data["imp_raw"][idx]
        padding_length = 0

        rr_labels = self.data["rr_labels"][idx]

        # Subject ID (convert numeric tensor to string)
        sid_tensor = self.data["subject_ids"][idx]
        sid_item = (
            int(sid_tensor.item())
            if sid_tensor.numel() == 1
            else int(sid_tensor.flatten()[0].item())
        )
        subject_id = str(sid_item)

        return Sample(
            ecg_raw=ecg_raw,
            ppg_raw=ppg_raw,
            imp_raw=imp_raw,
            abp_raw=None,
            bp_raw=None,
            sample_index=idx,
            subject_id=subject_id,
            rr_labels=rr_labels,
            padding_length=padding_length,
            vital_sign_type=self.target_vital_type,  # Configurable target vital type
            # RR prediction doesn't need scalar extraction
            extract_scalars=self.extract_scalars_default,
        )

    def get_normalization_params(self) -> None:
        """Return normalization parameters for MimicPerformLarge."""
        return None

    @property
    def supports_patient_split(self) -> bool:
        """Check if MimicPERFormLarge supports patient-level splitting."""
        return True


# Register with Hydra ConfigStore
if __name__ != "__main__":
    cs = ConfigStore.instance()
    cs.store(
        name="base_mimicperformlarge",
        node=MimicPERFormLargeConfig,
        group="train_dataset",
    )
    cs.store(
        name="base_mimicperformlarge",
        node=MimicPERFormLargeConfig,
        group="test_dataset",
    )
