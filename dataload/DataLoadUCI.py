# Standard library imports

# Third-party imports
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

class RecordSamplesDatasetUC(Dataset):
    _shared_data = {}  # Class variable to store shared data across instances
    
    @classmethod
    def load_shared_data(cls, sample_file, dataset):
        """Load preprocessed data into shared memory if not already loaded"""
        key = (sample_file, dataset)
        if key not in cls._shared_data:
            with h5py.File(sample_file, "r") as f:
                data = {}
                split_name = 'train' if dataset == 0 else 'val' if dataset == 1 else 'test'
                
                # Load preprocessed data for the specific split
                data['waveforms_raw'] = np.array(f[split_name]['waveforms_raw'])
                data['waveforms_local_minmax'] = np.array(f[split_name]['waveforms_local_minmax'])
                data['waveforms_minmax_zc'] = np.array(f[split_name]['waveforms_minmax_zc'])
                data['bp_raw'] = np.array(f[split_name]['bp_raw'])
                data['bp_global_minmax'] = np.array(f[split_name]['bp_global_minmax'])
                data['abp_global_minmax'] = np.array(f[split_name]['abp_global_minmax'])
                data['abp_raw'] = np.array(f[split_name]['abp_raw'])
                
                # Load additional test data if available
                if dataset == 2:
                    try:
                        data['heart_rate'] = np.array(f[split_name]['heart_rate'])
                        data['pulse_rate'] = np.array(f[split_name]['pulse_rate'])
                    except:
                        print("Warning: Rate data not found in test set")
                
                cls._shared_data[key] = data

    @classmethod
    def create_subset(cls, base_dataset, indices):
        """Create a subset of the dataset using the given indices
        
        Args:
            base_dataset (RecordSamplesDatasetUC): The base dataset to create a subset from
            indices (np.ndarray or list): Array of indices to include in the subset
            
        Returns:
            Subset: A PyTorch Subset containing only the specified indices
        """
        # Convert indices to list if it's a numpy array
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        elif isinstance(indices, int):
            indices = [indices]
        
        # Create a new instance of the dataset class
        subset = cls(base_dataset.sample_file, base_dataset.dataset, 
                    sample_freq=base_dataset.sample_freq, 
                    sample_length=base_dataset.sample_length)
        
        # Create a Subset with the new dataset instance
        subset = Subset(subset, indices)
        
        return subset

    def __init__(self, sample_file, dataset, sample_freq=125, sample_length=1024):
        self.sample_length = sample_length
        self.sample_freq = sample_freq
        self.dataset = dataset
        self.sample_file = sample_file
        
        # Load data into shared memory
        self.load_shared_data(sample_file, dataset)
        self.data = self._shared_data[(sample_file, dataset)]
        
        # Store split name for feature loading
        self.split_name = 'train' if dataset == 0 else 'val' if dataset == 1 else 'test'

    def __len__(self):
        return len(self.data['waveforms_raw'])

    def __getitem__(self, idx):
        # Helper function to safely load HDF5 dataset
        def safe_load_dataset(dataset):
            if dataset.shape == ():  # Scalar dataset
                return torch.tensor(dataset[()])
            return torch.from_numpy(dataset[:])

        # Load features on-demand
        ecg_features = torch.zeros(1)
        ppg_features = torch.zeros(1)
        
        try:
            with h5py.File(self.sample_file, 'r') as f:
                sample_key = f'sample_{idx}'
                
                # Load ECG features
                if self.split_name in f and 'ecg_features' in f[self.split_name]:
                    if sample_key in f[self.split_name]['ecg_features']:
                        sample_group = f[self.split_name]['ecg_features'][sample_key]
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
                
                # Load PPG features
                if self.split_name in f and 'ppg_features' in f[self.split_name]:
                    if sample_key in f[self.split_name]['ppg_features']:
                        sample_group = f[self.split_name]['ppg_features'][sample_key]
                        ppg_feat = {}
                        
                        # Load specific PPG features
                        for key in ['Asp_deltaT', 'IPR']:
                            if key in sample_group:
                                ppg_feat[key] = safe_load_dataset(sample_group[key])
                        
                        ppg_features = ppg_feat
        except Exception as e:
            print(f"Warning: Error loading features for sample {idx}: {e}")

        if self.dataset in [0, 1]:  # Train and Val sets
            return (
                torch.tensor(self.data['waveforms_raw'][idx]),
                torch.tensor(self.data['abp_global_minmax'][idx]),
                torch.tensor(self.data['bp_raw'][idx]).squeeze(),
                torch.tensor(self.data['waveforms_minmax_zc'][idx]),
                torch.tensor(self.data['waveforms_local_minmax'][idx]),
                torch.tensor(self.data['bp_global_minmax'][idx]).squeeze(),
                ecg_features,
                ppg_features,
                torch.tensor(self.data['abp_raw'][idx])
            )
        else:  # Test set
            # Create default tensors for heart_rate and pulse_rate if not available
            heart_rate = torch.tensor(self.data['heart_rate'][idx]) if 'heart_rate' in self.data else torch.zeros(1)
            pulse_rate = torch.tensor(self.data['pulse_rate'][idx]) if 'pulse_rate' in self.data else torch.zeros(1)
            
            return (
                torch.tensor(self.data['waveforms_raw'][idx]),
                torch.tensor(self.data['abp_global_minmax'][idx]),
                torch.tensor(self.data['bp_raw'][idx]).squeeze(),
                torch.tensor(self.data['waveforms_minmax_zc'][idx]),
                torch.tensor(self.data['waveforms_local_minmax'][idx]),
                torch.tensor(self.data['bp_global_minmax'][idx]).squeeze(),
                heart_rate,
                pulse_rate,
                ecg_features,
                ppg_features,
                torch.tensor(self.data['abp_raw'][idx])
            )

    @classmethod
    def cleanup_shared_memory(cls):
        """Clean up shared memory (call this at the end of training)"""
        try:
            # Delete all tensors from shared memory
            for sample_file in cls._shared_data:
                for key, value in cls._shared_data[sample_file].items():
                    if torch.is_tensor(value):
                        del value
            
            # Clear the dictionary
            cls._shared_data.clear()
            
            # Force CUDA cache cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Shared memory cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


