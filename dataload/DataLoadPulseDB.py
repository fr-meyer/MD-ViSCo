# Standard library imports
import resource
import atexit

# Third-party library imports
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import DistilBertTokenizer

class PulseDBDataset(Dataset):
    _shared_data = {}  # Class variable to store shared data across instances
    
    @classmethod
    def _increase_file_limit(cls):
        """Increase the file descriptor limit"""
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            # Try to set to the maximum allowed
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"Increased file descriptor limit to {hard}")
        except Exception as e:
            print(f"Warning: Could not increase file descriptor limit: {e}")
    
    @classmethod
    def initialize_shared_memory(cls, sample_file):
        """Initialize shared memory for dataset"""
        try:
            # Increase file descriptor limit
            cls._increase_file_limit()
            
            if sample_file not in cls._shared_data:
                print(f"Loading data from {sample_file} into shared memory...")
                
                with h5py.File(f"{sample_file}.h5", 'r') as f:
                    # Helper function to safely load data
                    def safe_load(dataset):
                        try:
                            if dataset.shape == ():  # Scalar dataset
                                return torch.tensor(dataset[()]).share_memory_()
                            return torch.from_numpy(dataset[:]).share_memory_()
                        except Exception as e:
                            print(f"Warning: Error loading dataset {dataset.name}: {e}")
                            return None
                    
                    # Load and share everything including large arrays
                    shared_data = {
                        # Large arrays
                        'waveforms_raw': safe_load(f['waveforms_raw']),
                        'waveforms_local_minmax': safe_load(f['waveforms_local_minmax']),
                        'waveforms_minmax_zc': safe_load(f['waveforms_minmax_zc']),
                        'abp_global_minmax': safe_load(f['abp_global_minmax']),
                        
                        # Small arrays
                        'bp_raw': safe_load(f['bp_raw']),
                        'demographics': safe_load(f['demographics']),
                        'bp_global_minmax': safe_load(f['bp_global_minmax']),
                        
                        # Subject IDs (strings don't need sharing)
                        'subject_ids': np.array([
                            sid.decode('utf-8').strip("['").strip("]'") if isinstance(sid, bytes) 
                            else str(sid).strip("['").strip("]'") 
                            for sid in f['subject_ids'][:]
                        ], dtype=str),
                    }
                    
                    # Add rates if available (from shared memory)
                    if 'heart_rate' in f:
                        shared_data['heart_rate'] = safe_load(f['heart_rate'])
                    if 'pulse_rate' in f:
                        shared_data['pulse_rate'] = safe_load(f['pulse_rate'])
                    
                    # Store dataset size
                    shared_data['size'] = len(f['bp_raw'])
                    
                    # Store in class variable
                    cls._shared_data[sample_file] = shared_data
                    
                    print(f"Finished loading into shared memory")
                    total_gb = sum(arr.element_size() * arr.nelement() / (1024**3) 
                                 for arr in shared_data.values() 
                                 if torch.is_tensor(arr))
                    print(f"Total shared memory used: {total_gb:.2f} GB")
                    
                    # Register cleanup function
                    atexit.register(cls.cleanup_shared_memory)
                    
        except Exception as e:
            print(f"Error in initialize_shared_memory: {str(e)}")
            # Clean up any partially loaded data
            if sample_file in cls._shared_data:
                cls.cleanup_shared_memory()
            raise

    def __init__(self, sample_file, sample_length=1280):
        """Initialize dataset with shared memory support"""
        self.sample_length = sample_length
        self.sample_file = sample_file
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Initialize shared memory if not already done
        self.initialize_shared_memory(sample_file)
        
        # Get reference to shared data
        self.data = self._shared_data[sample_file]
        
        # Verify initialization
        assert len(self.data['bp_raw']) == self.data['size'], "Data size mismatch"

    def __len__(self):
        """Get dataset length"""
        return self.data['size']

    def __getitem__(self, idx):
        """Get item with all data from shared memory"""
        # Get waveforms from shared memory
        waveforms = self._pad_waveform(
            self.data['waveforms_raw'][idx].numpy(),  # Convert to numpy for padding
            self.sample_length
        )
        waveforms_local_minmax = self._pad_waveform(
            self.data['waveforms_local_minmax'][idx].numpy(),
            self.sample_length
        )
        waveforms_minmax_zc = self._pad_waveform(
            self.data['waveforms_minmax_zc'][idx].numpy(),
            self.sample_length
        )
        abp_global_minmax = self._pad_waveform(
            self.data['abp_global_minmax'][idx].numpy(),
            self.sample_length
        )
        
        # Get data from shared memory (no copying needed)
        sbp, dbp, map_value = self.data['bp_raw'][idx]
        demographics = self.data['demographics'][idx]
        age, gender, height, weight, bmi = demographics[0], demographics[1], demographics[2], demographics[3], demographics[4]
        
        # Get rates if available (from shared memory)
        rates = torch.zeros(2, dtype=torch.float32)
        if 'heart_rate' in self.data and 'pulse_rate' in self.data:
            rates = torch.stack([
                self.data['heart_rate'][idx][0],
                self.data['pulse_rate'][idx][0]
            ])
        
        # Create text description
        text = self._create_text_description(
            demographics.cpu().numpy(),
            torch.tensor([sbp, dbp, map_value]).numpy(),
            rates
        )
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt'
        )
        
        # Helper function to safely load HDF5 dataset
        def safe_load_dataset(dataset):
            if dataset.shape == ():  # Scalar dataset
                return torch.tensor(dataset[()])
            return torch.from_numpy(dataset[:])
        
        # Load ECG features on-demand
        ecg_features = torch.zeros(1)  # Default empty tensor
        ppg_features = torch.zeros(1)  # Default empty tensor
        
        # Open HDF5 file for ECG and PPG features
        with h5py.File(f"{self.sample_file}.h5", 'r') as h5_file:
            if 'ecg_features' in h5_file:
                try:
                    sample_group = h5_file['ecg_features'][f'sample_{idx}']
                    ecg_feat = {}
                    
                    # Load peak locations
                    if 'peak_locations' in sample_group:
                        peak_locs = {}
                        for wave_type in sample_group['peak_locations']:
                            wave_group = sample_group['peak_locations'][wave_type]
                            wave_data = {}
                            for key in wave_group:
                                if isinstance(wave_group[key], h5py.Group):
                                    sub_data = {}
                                    for sub_key in wave_group[key]:
                                        sub_data[sub_key] = safe_load_dataset(wave_group[key][sub_key])
                                    wave_data[key] = sub_data
                                else:
                                    wave_data[key] = safe_load_dataset(wave_group[key])
                            peak_locs[wave_type] = wave_data
                        ecg_feat['peak_locations'] = peak_locs
                    
                    # Load other ECG features
                    for key in ['qt_intervals', 'mean_ecg_quality']:
                        if key in sample_group:
                            ecg_feat[key] = safe_load_dataset(sample_group[key])
                    
                    ecg_features = ecg_feat
                except Exception as e:
                    print(f"Warning: Error loading ECG features for sample {idx}: {e}")
            
            # Load PPG features on-demand
            if 'ppg_features' in h5_file:
                try:
                    sample_group = h5_file['ppg_features'][f'sample_{idx}']
                    ppg_feat = {}
                    
                    # Load PPG features
                    for key in ['Asp_deltaT', 'IPR']:
                        if key in sample_group:
                            ppg_feat[key] = safe_load_dataset(sample_group[key])
                    
                    ppg_features = ppg_feat
                except Exception as e:
                    print(f"Warning: Error loading PPG features for sample {idx}: {e}")
        
        return (
            self.data['subject_ids'][idx],
            idx,
            torch.from_numpy(waveforms).float(),
            torch.tensor([sbp, dbp, map_value]).float(),
            demographics.float(),
            torch.tensor([age, gender, height, weight, bmi]).float(),
            torch.cat([encoded_text['input_ids'], encoded_text['attention_mask']]),
            torch.from_numpy(waveforms_local_minmax).float(),
            torch.from_numpy(waveforms_minmax_zc).float(),
            rates,
            self.data['bp_global_minmax'][idx].float(),
            torch.from_numpy(abp_global_minmax).float(),
            ecg_features,
            ppg_features
        )

    def _create_text_description(self, demographics, bp_values, rates=None):
        """Create text description from patient data"""
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

    @classmethod
    def create_subset(cls, dataset, indices):
        """Create a subset while maintaining shared memory access"""
        subset = Subset(dataset, indices)
        # Share necessary attributes and methods
        subset.sample_length = dataset.sample_length
        subset.data = dataset.data  # Share the same memory
        subset.tokenizer = dataset.tokenizer
        subset.sample_file = dataset.sample_file
        subset._pad_waveform = dataset._pad_waveform
        subset._create_text_description = dataset._create_text_description
        return subset

    @classmethod
    def cleanup_shared_memory(cls):
        """Clean up shared memory (call this at the end of training)"""
        try:
            # Delete all tensors from shared memory
            for sample_file in list(cls._shared_data.keys()):
                for key, value in list(cls._shared_data[sample_file].items()):
                    if torch.is_tensor(value):
                        try:
                            del value
                        except Exception as e:
                            print(f"Warning: Error cleaning up tensor {key}: {e}")
                del cls._shared_data[sample_file]
            
            # Force CUDA cache cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Shared memory cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    @staticmethod
    def _pad_waveform(waveform, target_length):
        """Zero pad waveform to target length (centered)"""
        current_length = waveform.shape[-1]
        if current_length >= target_length:
            return waveform
            
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        
        # Adjust padding based on input dimensions
        ndim = waveform.ndim
        if ndim == 3:
            pad_width = [(0, 0), (0, 0), (pad_left, pad_right)]
        elif ndim == 2:
            pad_width = [(0, 0), (pad_left, pad_right)]
        else:
            pad_width = [(pad_left, pad_right)]
        
        return np.pad(waveform, pad_width, mode='constant', constant_values=0)

    def __del__(self):
        """Clean up shared memory when dataset is deleted"""
        if hasattr(self, 'data'):
            self.cleanup_shared_memory()
