# Standard library imports
from dataclasses import dataclass

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from .base_dataset import BaseDataset, DatasetBaseConfig, Sample



@dataclass
class UCIConfig(DatasetBaseConfig):
    """Configuration for UCI dataset"""
    _target_: str = "src.dataset.uci_dataset.UCIDataset"
    dataset_name: str = "UCI"

    input_size: int = 1024
    
    # UCI BP value ranges
    sbp_max: float = 189.98421357007769
    dbp_min: float = 50
    
    # File path structure (inherited from base)
    dataset_folder: str = 'UCI/'
    
    # File naming patterns
    
    
    def __post_init__(self):
        super().__post_init__()

class UCIDataset(BaseDataset):
    
    @staticmethod
    def _extract_data(f):
        return {
            'waveforms_raw': BaseDataset.safe_load(f['waveforms_raw']),
            'waveforms_raw_abp_global_minmax': BaseDataset.safe_load(f['waveforms_raw_abp_global_minmax']),
            'waveforms_local_minmax': BaseDataset.safe_load(f['waveforms_local_minmax']),
            'waveforms_minmax_zc': BaseDataset.safe_load(f['waveforms_minmax_zc']),
            'bp_raw': BaseDataset.safe_load(f['bp_raw']),
            'bp_global_minmax': BaseDataset.safe_load(f['bp_global_minmax']),
            'abp_global_minmax': BaseDataset.safe_load(f['abp_global_minmax']),
            'abp_raw': BaseDataset.safe_load(f['abp_raw']),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load data into shared memory using the new base method
        self.load_shared_data(self.sample_file, self._extract_data)
        self.data = self._shared_data[self.sample_file]

    def __len__(self):
        return len(self.data['waveforms_raw'])

    def _load_shared_data(self, sample_file):
        self.load_shared_data(sample_file, self._extract_data)

    def __getitem__(self, idx: int) -> Sample:
        """Get item with shared memory tensors - returns Sample"""
        # Data is already shared memory tensors from safe_load
        return Sample(
            waveform_raw=self.data['waveforms_raw'][idx],  # already tensor
            waveforms_raw_abp_global_minmax=self.data['waveforms_raw_abp_global_minmax'][idx],  # already tensor
            abp_global_minmax=self.data['abp_global_minmax'][idx],  # already tensor
            bp_raw=self.data['bp_raw'][idx],  # already tensor
            waveform_minmax_zc=self.data['waveforms_minmax_zc'][idx],  # already tensor
            waveform_local_minmax=self.data['waveforms_local_minmax'][idx],  # already tensor
            bp_global_minmax=self.data['bp_global_minmax'][idx],  # already tensor
            abp_raw=self.data['abp_raw'][idx],  # already tensor
            sample_index=idx,
        )

# Register Config class with Hydra after classes are defined
if __name__ != "__main__":
    # Register with Hydra
    cs = ConfigStore.instance()
    cs.store(name="base_uci", node=UCIConfig, group="train_dataset")
    cs.store(name="base_uci", node=UCIConfig, group="test_dataset")