# Standard library imports
from dataclasses import dataclass

# Third-party library imports
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from transformers import AutoTokenizer

# Local imports
from .base_dataset import BaseDataset, DatasetBaseConfig, Sample

@dataclass
class PulseDBConfig(DatasetBaseConfig):
    """Configuration for PulseDB dataset"""
    _target_: str = "src.dataset.pulsedb_dataset.PulseDBDataset"
    dataset_name: str = "PulseDB"
    
    input_size: int = 1280
    
    # PulseDB BP value ranges
    sbp_max: float = 286.58240014784946
    dbp_min: float = 2.341260731456743
    
    # File path structure (inherited from base)
    dataset_folder: str = 'PulseDB/'
    
    # Tokenizer
    tokenizer_name: str = "distilbert-base-uncased"

    def __post_init__(self):
        super().__post_init__()
        
    



class PulseDBDataset(BaseDataset):
    @staticmethod
    def _extract_data(f):
        shared_data = {
            'waveforms_raw': BaseDataset.safe_load(f['waveforms_raw']),
            'waveforms_raw_abp_global_minmax': BaseDataset.safe_load(f['waveforms_raw_abp_global_minmax']),
            'waveforms_local_minmax': BaseDataset.safe_load(f['waveforms_local_minmax']),
            'waveforms_minmax_zc': BaseDataset.safe_load(f['waveforms_minmax_zc']),
            'abp_global_minmax': BaseDataset.safe_load(f['abp_global_minmax']),
            'abp_raw': BaseDataset.safe_load(f['abp_raw']),
            'bp_raw': BaseDataset.safe_load(f['bp_raw']),
            'demographics': BaseDataset.safe_load(f['demographics']),
            'bp_global_minmax': BaseDataset.safe_load(f['bp_global_minmax']),
            'subject_ids': np.array([
                sid.decode('utf-8').strip("['").strip("']") if isinstance(sid, bytes)
                else str(sid).strip("['").strip("']")
                for sid in f['subject_ids'][:]
            ], dtype=str),
        }
        shared_data['size'] = len(f['bp_raw'])
        return shared_data

    def __init__(self, tokenizer_name: str = "distilbert-base-uncased", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Use new base method for shared memory
        self.load_shared_data(self.sample_file, self._extract_data)
        self.data = self._shared_data[self.sample_file]
        # Verify initialization
        assert len(self.data['bp_raw']) == self.data['size'], "Data size mismatch"

    def __len__(self):
        """Get dataset length"""
        return self.data['size']

    def _load_shared_data(self, sample_file):
        self.load_shared_data(sample_file, self._extract_data)

    def __getitem__(self, idx: int) -> Sample:
        """Get item with shared memory tensors - returns Sample"""
        # Get waveforms from shared memory (already tensors)
        waveforms = self._pad_waveform_tensor(
            self.data['waveforms_raw'][idx],  # already tensor
            self.sample_length
        )
        waveforms_local_minmax = self._pad_waveform_tensor(
            self.data['waveforms_local_minmax'][idx],  # already tensor
            self.sample_length
        )
        waveforms_minmax_zc = self._pad_waveform_tensor(
            self.data['waveforms_minmax_zc'][idx],  # already tensor
            self.sample_length
        )
        abp_global_minmax = self._pad_waveform_tensor(
            self.data['abp_global_minmax'][idx],  # already tensor
            self.sample_length
        )
        waveforms_raw_abp_global_minmax = self._pad_waveform_tensor(
            self.data['waveforms_raw_abp_global_minmax'][idx],  # already tensor
            self.sample_length
        )
        
        # Get data from shared memory (tensors)
        bp_tensor = self.data['bp_raw'][idx]  # already tensor
        demographics = self.data['demographics'][idx]  # already tensor
        
        # Get ABP raw data
        abp_raw = self.data['abp_raw'][idx]  # already tensor
        
        # Create text description (convert tensor to numpy for processing)
        demographics_np = demographics.cpu().numpy() if torch.is_tensor(demographics) else demographics
        text = self._create_text_description(demographics_np)
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt'
        )
        
        return Sample(
            waveform_raw=waveforms,  # tensor
            waveforms_raw_abp_global_minmax=waveforms_raw_abp_global_minmax,  # tensor (padded)
            abp_global_minmax=abp_global_minmax,  # tensor
            bp_raw=bp_tensor,  # tensor
            waveform_minmax_zc=waveforms_minmax_zc,  # tensor
            waveform_local_minmax=waveforms_local_minmax,  # tensor
            bp_global_minmax=self.data['bp_global_minmax'][idx],  # tensor
            abp_raw=abp_raw,  # tensor
            sample_index=idx,
            subject_id=self.data['subject_ids'][idx],
            demographics_raw=demographics,  # tensor
            demographics_split=demographics,  # tensor (already contains age, gender, height, weight, bmi)
            text_encoded=torch.cat([encoded_text['input_ids'], encoded_text['attention_mask']]),  # tensor
        )

    def _create_text_description(self, demographics):
        """Create text description from patient demographic data"""
        age, gender, height, weight, bmi = demographics

        text = f"Patient Age: {age:.1f} {'year' if age <= 1 else 'years'} / "
        text += f"Patient Gender: {'Male' if gender == 1 else 'Female'}"
        
        if not np.isnan(height):
            text += f" / Patient Height: {height:.1f} cm"
        
        if not np.isnan(weight):
            text += f" / Patient Weight: {weight:.1f} kg"
        
        if not np.isnan(bmi):
            text += f" / Patient BMI: {bmi:.1f} kg/m2"
            
        return text

    @property
    def supports_patient_split(self) -> bool:
        return True

# Register Config class with Hydra after classes are defined
if __name__ != "__main__":
    # Register with Hydra
    cs = ConfigStore.instance()
    cs.store(name="base_pulsedb", node=PulseDBConfig, group="train_dataset")
    cs.store(name="base_pulsedb", node=PulseDBConfig, group="test_dataset")
