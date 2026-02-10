"""UCI Dataset Loader.

This module loads the UCI Cuff-Less Blood Pressure Estimation dataset.

Dataset Information:
- Dataset Source: https://archive.ics.uci.edu/dataset/340
- Preprocessing: Based on NABNet repository (https://github.com/Sakib1263/NABNet)
- Paper: "NABNet: A Nested Attention-guided BiConvLSTM network for a robust "
  "prediction of Blood Pressure components from reconstructed Arterial Blood "
  "Pressure waveforms using PPG and ECG signals"
  https://linkinghub.elsevier.com/retrieve/pii/S1746809422007017
- License: CC BY 4.0 (dataset)

Note: This implementation loads preprocessed data in MD-ViSCo format.
See preprocessing documentation for data preparation steps.
"""

import logging

# Standard library imports
from dataclasses import dataclass

import torch

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.dataset.base_dataset import BaseDataset
from src.dataset.base_dataset import DatasetBaseConfig
from src.dataset.base_dataset import Sample

logger = logging.getLogger(__name__)


@dataclass
class UCIConfig(DatasetBaseConfig):
    """Configuration for UCI dataset."""

    _target_: str = "src.dataset.uci_dataset.UCIDataset"
    dataset_name: str = "UCI"

    input_size: int | None = 1024

    # UCI BP value ranges
    sbp_max: float | None = 189.98421357007769
    dbp_min: float | None = 50.0

    # File path structure (inherited from base)
    dataset_folder: str = "UCI_Dataset_Preprocessed/mdvisco_processed/UCI"

    # File naming patterns

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()


class UCIDataset(BaseDataset):
    """BaseDataset for UCI cuff-less blood pressure estimation.

    Loads ECG, PPG, ABP waveforms and BP scalars (SBP/DBP/MAP) from
    preprocessed HDF5. Shared memory and lazy normalization follow
    BaseDataset contract.
    """

    @staticmethod
    def _extract_data(f):
        # Only support new format with separate vital sign fields
        data = {
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
        }

        # Store original lengths for padding calculation
        # Assumption: All samples have uniform length within the dataset.
        ecg_raw = data["ecg_raw"]
        ppg_raw = data["ppg_raw"]
        abp_raw = data["abp_raw"]
        bp_raw = data["bp_raw"]
        if ecg_raw is None or ppg_raw is None or abp_raw is None or bp_raw is None:
            raise ValueError(
                "UCI dataset failed to load one or more required tensors "
                "(ecg_raw, ppg_raw, abp_raw, bp_raw) from HDF5."
            )
        data["ecg_original_lengths"] = torch.tensor([ecg_raw.shape[-1]] * len(bp_raw))
        data["ppg_original_lengths"] = torch.tensor([ppg_raw.shape[-1]] * len(bp_raw))
        data["abp_original_lengths"] = torch.tensor([abp_raw.shape[-1]] * len(bp_raw))

        data["size"] = len(bp_raw)

        return data

    def __init__(self, *args, **kwargs):
        """Initialize UCI dataset.

        Args:
            *args: Positional arguments passed to BaseDataset.
            **kwargs: Keyword arguments passed to BaseDataset.
        """
        super().__init__(*args, **kwargs)
        self.load_shared_data(self.sample_file, self._extract_data)
        self.data = self._shared_data[self.sample_file]

        # Log successful loading
        logger.info("UCI loaded with new format (vital-specific fields)")

        # Store BP normalization constants (from config)
        self.sbp_max = kwargs.get("sbp_max", 189.98421357007769)
        self.dbp_min = kwargs.get("dbp_min", 50.0)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.data["size"]

    def __getitem__(self, idx: int) -> Sample:
        """Get item with shared memory tensors - returns Sample with raw sequences.

        This method returns raw waveforms without preprocessing.
        Padding/trimming is performed at batch time in the collate function
        for better memory efficiency and flexibility.

        Waveforms are variable-length and will be resized to uniform length
        during batching.
        """
        # NEW ARCHITECTURE: Return raw sequences without preprocessing
        # Padding/trimming handled at batch time in collate function

        # Padding/trimming applied at collate; dataset returns raw
        ecg_raw = self.data["ecg_raw"][idx]  # [1, T_original]
        ppg_raw = self.data["ppg_raw"][idx]  # [1, T_original]
        abp_raw = self.data["abp_raw"][idx]  # [1, T_original]

        # No preprocessing applied, padding_length is 0 (padding happens in collate)
        padding_length = 0

        return Sample(
            # Vital-specific fields
            ecg_raw=ecg_raw,  # [1, T]
            ppg_raw=ppg_raw,  # [1, T]
            abp_raw=abp_raw,  # [1, T]
            imp_raw=None,  # Not available in UCI
            # BP data
            bp_raw=self.data["bp_raw"][idx],
            # Metadata
            sample_index=idx,
            padding_length=padding_length,
            vital_sign_type="ABP",  # UCI dataset is for ABP prediction
            extract_scalars=True,  # Always extract SBP/DBP for BP tasks
        )

    def get_normalization_params(self) -> dict[str, float]:
        """Get UCI-specific BP normalization parameters.

        Returns:
            Dict with sbp_max and dbp_min for global normalization
        """
        return {"sbp_max": self.sbp_max, "dbp_min": self.dbp_min}


# Register with Hydra ConfigStore
if __name__ != "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_uci", node=UCIConfig, group="train_dataset")
    cs.store(name="base_uci", node=UCIConfig, group="test_dataset")
