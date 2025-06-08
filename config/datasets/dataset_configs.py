# Standard library imports
from dataclasses import dataclass
from datetime import datetime
import os

# Third-party imports
import h5py
import numpy as np
from mat73 import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports
from config.base_config import BaseModelConfig
from dataload.DataLoadPulseDB import PulseDBDataset
from dataload.DataLoadUCI import RecordSamplesDatasetUC
from utils.ecg_features import (
    calculate_qt_intervals,
    extract_ecg_features,
    get_peak_locations
)
from utils.ppg_features import extract_ppg_features
from utils.utils_preprocessing import (
    Global_Min_Max_Norm,
    normalize_signals,
    calculate_BPM,
    calculate_MAP,
)

@dataclass
class UCIBaseConfig(BaseModelConfig):
    """Base configuration for UCI dataset"""
    input_size: int = 1024
    num_channels: int = 2
    
    # Domain labels for UCI dataset
    ecg_label: int = 1
    ppg_label: int = 0
    abp_label: int = 2

    channel_names = {
        ecg_label: "ECG",
        ppg_label: "PPG",
        abp_label: "ABP"
    }

    # UCI BP value ranges
    uci_sbp_max: float = 189.98421357007769
    uci_dbp_min: float = 50

    dataset_path: str = 'UCI/'
    preprocessed_path: str = 'Preprocessed/'

    sample_file = 'UCI_Dataset_Preprocessed'

    def __post_init__(self):
        super().__post_init__()
        self.sbp_max = self.uci_sbp_max
        self.dbp_min = self.uci_dbp_min
    
    def get_sample_file(self):
        """Get the appropriate sample file based on number of channels"""
        channel_files = {
            2: 'UCI_Dataset_Preprocessed.h5'
        }
        filename = channel_files.get(self.num_channels)
        if filename:
            return os.path.join(self.data_path, self.dataset_path, self.preprocessed_path, filename)
        raise ValueError(f"Invalid number of channels: {self.num_channels}")
    
    def create_ddp_dataset(self):
        """Create train/val dataloaders with DDP samplers"""
        print(f"Creating DDP dataset with seed: {self.seed}")
        np.random.seed(self.seed)  # Original Seed - Keep consistent shuffling

        if self.is_pretraining or self.model_type == 'approximation':
            # Create datasets using UCI custom dataloader
            sample_file = self.get_sample_file()
            train_dataset = RecordSamplesDatasetUC(sample_file, dataset=0)
            val_dataset = RecordSamplesDatasetUC(sample_file, dataset=1)
            test_dataset = self.create_test_dataset()

            print(f"Total training samples: {len(train_dataset)}")
            print(f"Total validation samples: {len(val_dataset)}")
            print(f"Total test samples: {len(test_dataset)}")
                
            return (train_dataset, val_dataset, test_dataset)
        elif self.is_finetuning or self.model_type == 'refinement':
            # For finetuning/refinement, load test set and split it
            sample_file = self.get_sample_file()
            base_dataset = RecordSamplesDatasetUC(sample_file, dataset=2)  # Load test set
            dataset_size = len(base_dataset)
            
            # Split ratios for finetuning
            train_ratio, val_ratio, test_ratio = 0.81, 0.09, 0.10
            
            # Calculate split sizes
            train_size = int(train_ratio * dataset_size)
            val_size = int(val_ratio * dataset_size)
            test_size = dataset_size - train_size - val_size
            
            # Create random indices and shuffle
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            # Split indices
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create datasets
            train_dataset = RecordSamplesDatasetUC.create_subset(base_dataset, train_indices)
            val_dataset = RecordSamplesDatasetUC.create_subset(base_dataset, val_indices)
            test_dataset = RecordSamplesDatasetUC.create_subset(base_dataset, test_indices)
            
            # Print split statistics
            print(f"\nFinetuning/Refinement Split Statistics:")
            print(f"Seed: {np.random.get_state()[1][0]}")
            print(f"Total samples: {dataset_size}")
            print(f"Train samples: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
            print(f"Val samples: {len(val_indices)} ({len(val_indices)/dataset_size*100:.1f}%)")
            print(f"Test samples: {len(test_indices)} ({len(test_indices)/dataset_size*100:.1f}%)")
            
            return (train_dataset, val_dataset, test_dataset)
    
    def create_test_dataset(self, is_finetuning=False):
        """Create test dataset
        
        Args:
            is_finetuning (bool): If True, returns the 10% test portion from the training file,
                                 otherwise returns the separate test file dataset
        """
        print(f"Creating test dataset with seed: {self.seed}")
        np.random.seed(self.seed)  # Original Seed - Keep consistent shuffling

        if self.is_pretraining or self.model_type == 'approximation':
            # For pretraining/approximation, return the regular test set
            sample_file = self.get_sample_file()
            test_dataset = RecordSamplesDatasetUC(sample_file, dataset=2)
            return test_dataset
        elif self.is_finetuning or self.model_type == 'refinement':
            # For finetuning/refinement, load test set and extract the test portion
            sample_file = self.get_sample_file()
            base_dataset = RecordSamplesDatasetUC(sample_file, dataset=2)  # Load test set
            dataset_size = len(base_dataset)
            
            # Split ratios for finetuning
            train_ratio, val_ratio = 0.81, 0.09  # test_ratio = 0.10
            
            # Create random indices and shuffle
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            # Calculate split point for test portion
            test_start = int((train_ratio + val_ratio) * dataset_size)
            test_indices = indices[test_start:]
            
            # Create test dataset
            test_dataset = RecordSamplesDatasetUC.create_subset(base_dataset, test_indices)
            
            # Print test set statistics
            print(f"\nFinetuning/Refinement Test Set Statistics:")
            print(f"Seed: {np.random.get_state()[1][0]}")
            print(f"Total test samples: {len(test_indices)}")
            print(f"Percentage of original data: {len(test_indices)/dataset_size*100:.1f}%")
            return test_dataset
        else:
            # Default case for UCI dataset - use dataset=2 for test set
            sample_file = self.get_sample_file()
            test_dataset = RecordSamplesDatasetUC(sample_file, dataset=2)
            return test_dataset

    def preprocess_dataset(self, input_file, output_file, calculate_bpm=False):
        """Load and preprocess the UCI dataset
        
        Args:
            input_file (str): Base path to input .h5 file (without Part_x suffix)
            output_file (str): Path to output .h5 file
            calculate_bpm (bool): Whether to calculate BPM for test set
        """
        print("\nLoading UCI dataset...")
        
        # Load data from .h5 files
        splits = {
            'train': {
                'X': [],
                'ABP_GRND': [],
                'SBP': [],
                'DBP': []
            },
            'val': {
                'X': [],
                'ABP_GRND': [],
                'SBP': [],
                'DBP': []
            },
            'test': {
                'X': [],
                'SBP': [],
                'DBP': [],
                'ABP_GRND': []
            }
        }

        # Get the base directory from input_file
        base_dir = os.path.dirname(input_file)

        # Load training data from parts 1-3 
        for part_num in range(1, 4):
            part_file = os.path.join(base_dir, f"UCI_Dataset_Part_{part_num}_Preprocessed.h5")
            print(f"\nLoading training part {part_num} from {part_file}...")
            with h5py.File(part_file, "r") as f:
                # Load PPG and ECG data
                ppg_data = np.array(f.get('PPG'))
                ecg_data = np.array(f.get('ECG'))

                # Load ABP data
                abp_data = np.array(f.get('ABP_GRND'))[:, np.newaxis]
                abp_norm_data = np.array(f.get('ABP_RNorm'))
                
                # Stack channels for X
                x_data = np.stack((ppg_data, ecg_data, abp_norm_data), axis=1)
                
                # Load BP values
                sbp_data = np.array(f.get('SBP'))
                dbp_data = np.array(f.get('DBP'))
                
                # Append to splits
                splits['train']['X'].append(x_data)
                splits['train']['ABP_GRND'].append(abp_data)
                splits['train']['SBP'].append(sbp_data)
                splits['train']['DBP'].append(dbp_data)

        # Load test data from part 4
        test_file = os.path.join(base_dir, "UCI_Dataset_Part_4_Preprocessed.h5")
        print(f"\nLoading test data from {test_file}...")
        with h5py.File(test_file, "r") as f:
            # Load PPG and ECG data
            ppg_data = np.array(f.get('PPG'))
            ecg_data = np.array(f.get('ECG'))

            # Load ABP data
            abp_data = np.array(f.get('ABP_GRND'))[:, np.newaxis]
            abp_norm_data = np.array(f.get('ABP_RNorm'))
            
            # Stack channels for X
            x_data = np.stack((ppg_data, ecg_data, abp_norm_data), axis=1)
            
            # Load BP values
            sbp_data = np.array(f.get('SBP'))
            dbp_data = np.array(f.get('DBP'))
            
            # Append to splits
            splits['test']['X'] = x_data
            splits['test']['SBP'] = sbp_data
            splits['test']['DBP'] = dbp_data
            splits['test']['ABP_GRND'] = abp_data

        # Concatenate training data
        for key in ['X', 'ABP_GRND', 'SBP', 'DBP']:
            splits['train'][key] = np.concatenate(splits['train'][key], axis=0)

        # Split training data into train and validation sets
        print("\nSplitting training data into train and validation sets...")
        X_train, X_val, Y_train, Y_val, SBP_train, SBP_val, DBP_train, DBP_val = train_test_split(
            splits['train']['X'],
            splits['train']['ABP_GRND'],
            splits['train']['SBP'],
            splits['train']['DBP'],
            test_size=0.2,
            random_state=42
        )

        # Update splits with train/val data
        splits['train']['X'] = X_train
        splits['train']['ABP_GRND'] = Y_train
        splits['train']['SBP'] = SBP_train
        splits['train']['DBP'] = DBP_train

        splits['val']['X'] = X_val
        splits['val']['ABP_GRND'] = Y_val
        splits['val']['SBP'] = SBP_val
        splits['val']['DBP'] = DBP_val

        # Print dataset statistics
        print("\nDataset Statistics:")
        for split_name, split_data in splits.items():
            print(f"{split_name} samples: {len(split_data['X'])}")

        # Initialize arrays for saving
        arrays_info = {}
        
        # Process each split
        for split_name, split_data in splits.items():
            print(f"\nProcessing {split_name} split...")
            n_samples = len(split_data['X'])
            
            # Initialize arrays for current split
            waveforms_raw = []
            waveforms_local_minmax = []
            waveforms_minmax_zc = []
            bp_raw = []
            bp_global_minmax = []
            abp_global_minmax = []
            abp_raw = []
            
            if calculate_bpm and split_name == 'test':
                heart_rate = []
                pulse_rate = []
                failed_bpm_count = 0
            
            # Process samples
            with tqdm(total=n_samples, desc=f"Processing {split_name} samples") as pbar:
                for i in range(n_samples):
                    # Get current waveforms
                    current_waveforms = split_data['X'][i]
                    
                    # Calculate BPM for test set
                    if calculate_bpm and split_name == 'test':
                        bpm_result = calculate_BPM(current_waveforms,
                                                 ii_label=self.ecg_label,
                                                 ppg_label=self.ppg_label)
                        if bpm_result is None:
                            failed_bpm_count += 1
                            pbar.update(1)
                            continue
                        hr, pr = bpm_result
                        heart_rate.append([hr])
                        pulse_rate.append([pr])
                    
                    # Normalize waveforms
                    current_minmax, current_minmax_zc = normalize_signals(current_waveforms)

                    # Calculate BP values
                    sbp = split_data['SBP'][i]
                    dbp = split_data['DBP'][i]
                    map_value = calculate_MAP(sbp, dbp)
                    
                    # Get ABP ground truth
                    abp_grnd = split_data['ABP_GRND'][i]
                    
                    # Apply global min-max normalization
                    Global_Min_Max = {"min": self.dbp_min, "max": self.sbp_max}
                    global_min_max_abp = Global_Min_Max_Norm(abp_grnd, Global_Min_Max=Global_Min_Max)
                    global_min_max_sbp = Global_Min_Max_Norm(sbp, Global_Min_Max=Global_Min_Max)
                    global_min_max_dbp = Global_Min_Max_Norm(dbp, Global_Min_Max=Global_Min_Max)
                    global_min_max_map = Global_Min_Max_Norm(map_value, Global_Min_Max=Global_Min_Max)
                    
                    # Store processed data
                    waveforms_raw.append(current_waveforms)
                    waveforms_local_minmax.append(current_minmax)
                    waveforms_minmax_zc.append(current_minmax_zc)
                    bp_raw.append([sbp, dbp, map_value])
                    bp_global_minmax.append([global_min_max_sbp, global_min_max_dbp, global_min_max_map])
                    abp_global_minmax.append(global_min_max_abp)
                    abp_raw.append(abp_grnd)
                    
                    pbar.update(1)
            
            # Convert lists to arrays
            arrays_info[split_name] = {
                'waveforms_raw': np.array(waveforms_raw),
                'waveforms_local_minmax': np.array(waveforms_local_minmax),
                'waveforms_minmax_zc': np.array(waveforms_minmax_zc),
                'bp_raw': np.array(bp_raw),
                'bp_global_minmax': np.array(bp_global_minmax),
                'abp_global_minmax': np.array(abp_global_minmax),
                'abp_raw': np.array(abp_raw)
            }
            
            if calculate_bpm and split_name == 'test':
                arrays_info[split_name].update({
                    'heart_rate': np.array(heart_rate),
                    'pulse_rate': np.array(pulse_rate)
                })
                print(f"\nBPM calculation failed for {failed_bpm_count} samples out of {n_samples}")
                print(f"Success rate: {((n_samples - failed_bpm_count) / n_samples) * 100:.2f}%")

        # Save processed data
        output_file = f"{output_file}.h5"
        print("\nSaving processed data...")
        with h5py.File(output_file, "w") as f:
            for split_name, split_arrays in arrays_info.items():
                # Create a group for each split
                split_group = f.create_group(split_name)
                
                for name, data in split_arrays.items():
                    chunks = self._calculate_chunk_size(data.shape, data.dtype)
                    print(f"Creating dataset {split_name}/{name} with shape {data.shape} and chunks {chunks}")
                    split_group.create_dataset(
                        name,
                        data=data,
                        chunks=chunks,
                        compression='gzip',
                        compression_opts=4,
                        shuffle=True
                    )
            
            # Add metadata
            f.attrs['chunked'] = True
            f.attrs['creation_date'] = np.string_(datetime.now().isoformat())
            chunk_info = {f"{split_name}/{name}": dataset.chunks 
                         for split_name in f.keys() 
                         for name, dataset in f[split_name].items()}
            f.attrs['chunk_info'] = str(chunk_info)

        # Verify the saved dataset
        print("\nVerifying saved dataset...")
        dataset_stats = self.verify_saved_dataset(output_file)
        return dataset_stats

    def verify_saved_dataset(self, output_file):
        """Verify the saved dataset with optimized chunk reading"""
        print(f"\nVerifying dataset: {output_file}")
        try:
            with h5py.File(output_file, 'r') as f:
                # Get splits from root level
                splits = list(f.keys())
                
                # Process each split
                stats = {}
                for split_name in splits:
                    print(f"\nVerifying {split_name} split...")
                    split_group = f[split_name]
                    
                    # Get basic info without loading data
                    waveforms = split_group['waveforms_raw']
                    n_samples, n_channels, seq_length = waveforms.shape
                    
                    # Sample size based on dataset size
                    sample_size = min(200, n_samples // 10)
                    sample_indices = np.linspace(0, n_samples-1, sample_size, dtype=int)
                    
                    # Print dataset structure and chunk info
                    print(f"\nDataset Structure for {split_name}:")
                    for name, dataset in split_group.items():
                        chunk_mb = np.prod(dataset.chunks) * dataset.dtype.itemsize / (1024**2) if dataset.chunks else 0
                        print(f"\n{name}:")
                        print(f"  Shape: {dataset.shape}")
                        print(f"  Chunks: {dataset.chunks}")
                        print(f"  Chunk size: {chunk_mb:.2f} MB")
                        print(f"  Compression: {dataset.compression}")
                        print(f"  Compression ratio: {dataset.nbytes / dataset.size:.2f}x")
                    
                    # Process statistics in chunks
                    split_stats = self._process_chunks_statistics(split_group, sample_indices)
                    stats[split_name] = split_stats
                    
                    # Print verification results
                    self._print_verification_results(split_stats, split_name)
                
                return stats
        except Exception as e:
            print(f"Error loading {output_file}: {str(e)}")
            return None

    def _process_chunks_statistics(self, split_group, sample_indices):
        """Process dataset statistics using chunked reading"""
        stats = {
            'n_samples': split_group['waveforms_raw'].shape[0],
            'n_channels': split_group['waveforms_raw'].shape[1],
            'seq_length': split_group['waveforms_raw'].shape[2],
            'file_size_mb': os.path.getsize(split_group.file.filename) / (1024**2)
        }
        
        # Initialize min/max trackers
        bp_stats = {
            'raw': {'min': float('inf'), 'max': float('-inf')},
            'norm': {'min': float('inf'), 'max': float('-inf')}
        }
        
        abp_stats = {
            'raw': {'min': float('inf'), 'max': float('-inf')},
            'norm': {'min': float('inf'), 'max': float('-inf')}
        }
        
        # Process in chunks based on dataset chunk size
        chunk_size = split_group['bp_raw'].chunks[0]
        for i in range(0, len(sample_indices), chunk_size):
            batch_indices = sample_indices[i:i + chunk_size]
            
            # Read BP data chunks
            bp_raw = split_group['bp_raw'][batch_indices]
            bp_norm = split_group['bp_global_minmax'][batch_indices]
            
            # Read ABP data chunks
            abp_raw = split_group['abp_raw'][batch_indices]
            abp_norm = split_group['abp_global_minmax'][batch_indices]
            
            # Update BP statistics
            bp_stats['raw']['min'] = min(bp_stats['raw']['min'], np.min(bp_raw))
            bp_stats['raw']['max'] = max(bp_stats['raw']['max'], np.max(bp_raw))
            bp_stats['norm']['min'] = min(bp_stats['norm']['min'], np.min(bp_norm))
            bp_stats['norm']['max'] = max(bp_stats['norm']['max'], np.max(bp_norm))
            
            # Update ABP statistics
            abp_stats['raw']['min'] = min(abp_stats['raw']['min'], np.min(abp_raw))
            abp_stats['raw']['max'] = max(abp_stats['raw']['max'], np.max(abp_raw))
            abp_stats['norm']['min'] = min(abp_stats['norm']['min'], np.min(abp_norm))
            abp_stats['norm']['max'] = max(abp_stats['norm']['max'], np.max(abp_norm))
        
        stats['bp_ranges'] = {
            'raw': (bp_stats['raw']['min'], bp_stats['raw']['max']),
            'norm': (bp_stats['norm']['min'], bp_stats['norm']['max'])
        }
        
        stats['abp_ranges'] = {
            'raw': (abp_stats['raw']['min'], abp_stats['raw']['max']),
            'norm': (abp_stats['norm']['min'], abp_stats['norm']['max'])
        }
        
        # Process heart rate and pulse rate if available
        if 'heart_rate' in split_group and 'pulse_rate' in split_group:
            hr_chunk = split_group['heart_rate'][sample_indices]
            pr_chunk = split_group['pulse_rate'][sample_indices]
            stats['rate_ranges'] = {
                'heart_rate': (np.min(hr_chunk), np.max(hr_chunk)),
                'pulse_rate': (np.min(pr_chunk), np.max(pr_chunk))
            }
        
        return stats

    def _print_verification_results(self, stats, split_name):
        """Print verification results in a formatted way"""
        print(f"\n=== Verification Results for {split_name} ===")
        print(f"\nDataset Dimensions:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Channels: {stats['n_channels']}")
        print(f"  Sequence Length: {stats['seq_length']}")
        print(f"\nFile Size: {stats['file_size_mb']:.2f} MB")
        
        print("\nBlood Pressure Ranges:")
        print("  Raw:")
        print(f"    Min: {stats['bp_ranges']['raw'][0]:.2f}")
        print(f"    Max: {stats['bp_ranges']['raw'][1]:.2f}")
        print("  Normalized:")
        print(f"    Min: {stats['bp_ranges']['norm'][0]:.2f}")
        print(f"    Max: {stats['bp_ranges']['norm'][1]:.2f}")
        
        print("\nABP Ranges:")
        print("  Raw:")
        print(f"    Min: {stats['abp_ranges']['raw'][0]:.2f}")
        print(f"    Max: {stats['abp_ranges']['raw'][1]:.2f}")
        print("  Normalized:")
        print(f"    Min: {stats['abp_ranges']['norm'][0]:.2f}")
        print(f"    Max: {stats['abp_ranges']['norm'][1]:.2f}")
        
        if 'rate_ranges' in stats:
            print("\nRate Ranges:")
            print(f"  Heart Rate: {stats['rate_ranges']['heart_rate'][0]:.1f} - "
                  f"{stats['rate_ranges']['heart_rate'][1]:.1f} BPM")
            print(f"  Pulse Rate: {stats['rate_ranges']['pulse_rate'][0]:.1f} - "
                  f"{stats['rate_ranges']['pulse_rate'][1]:.1f} BPM")

    def _calculate_chunk_size(self, shape, dtype):
        """Calculate optimal chunk size based on data characteristics"""
        element_size = np.dtype(dtype).itemsize  # float64 = 8 bytes
        
        # Case 1: Waveform data (samples, channels, length) - (N, 3, 1024)
        if len(shape) == 3:
            if shape[1] == 3 and shape[2] == 1024:  # Waveform specific
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, 3, 1024)
            elif shape[1] == 1 and shape[2] == 1024:  # ABP waveform
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, 1, 1024)
        
        # Case 2: 2D arrays (bp_raw, bp_global_minmax)
        elif len(shape) == 2:
            if shape[1] <= 3:  # For bp_raw and bp_global_minmax (3 cols)
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, shape[1])
            else:
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, shape[1])
        
        # Case 3: 1D arrays (heart_rate, pulse_rate)
        else:
            samples_per_chunk = min(1, shape[0])
            return (samples_per_chunk,)

    def extract_waveform_features_method(self, sampling_rate: int = 125):
        """Load test set and extract features from PPG and ECG waveforms for UCI dataset
        
        Args:
            sampling_rate (int): Sampling rate of the signals in Hz. Default is 125.
            
        Returns:
            dict: Dictionary containing:
                - ecg_features: List of dictionaries containing ECG features for each sample
                - ppg_features: List of dictionaries containing PPG features for each sample
                - waveforms_raw: Raw waveform data
                - waveforms_local_minmax: Locally min-max normalized waveforms
                - waveforms_minmax_zc: Zero-centered min-max normalized waveforms
                - bp_raw: Raw blood pressure values (SBP, DBP, MAP)
                - bp_global_minmax: Globally min-max normalized BP values
                - abp_global_minmax: Globally min-max normalized ABP values
                - abp_raw: Raw ABP values
                - heart_rate: Heart rate values (if available)
                - pulse_rate: Pulse rate values (if available)
        """
        # Load test dataset
        test_dataset = self.create_test_dataset()
        
        if test_dataset is None:
            return None
            
        # Initialize lists to store features
        ecg_features = []
        ppg_features = []
        
        # Get the first sample to determine the number of channels
        first_sample = test_dataset[0]
        waveforms = first_sample[0]  # Get waveforms from first sample
        actual_num_channels = waveforms.shape[0]  # Get actual number of channels from data
        
        # Initialize arrays for storing data with actual number of channels
        n_samples = len(test_dataset)
        waveforms_raw = np.zeros((n_samples, actual_num_channels, self.input_size))
        waveforms_local_minmax = np.zeros((n_samples, actual_num_channels, self.input_size))
        waveforms_minmax_zc = np.zeros((n_samples, actual_num_channels, self.input_size))
        abp_global_minmax = np.zeros((n_samples, 1, self.input_size))
        abp_raw = np.zeros((n_samples, 1, self.input_size))  # Add abp_raw array
        bp_raw = np.zeros((n_samples, 3))  # SBP, DBP, MAP
        bp_global_minmax = np.zeros((n_samples, 3))  # normalized SBP, DBP, MAP
        heart_rate = np.zeros((n_samples, 1))
        pulse_rate = np.zeros((n_samples, 1))
        
        # Process each sample with tqdm progress bar
        for idx in tqdm(range(len(test_dataset)), desc="Extracting waveform features"):
            try:
                # Get sample data from UCI dataset
                sample_data = test_dataset[idx]
                
                # Unpack the values
                waveforms = sample_data[0]
                abp = sample_data[1]
                bp_values = sample_data[2]
                waveforms_minmax_zc_data = sample_data[3]
                waveforms_local_minmax_data = sample_data[4]
                bp_global_minmax_data = sample_data[5]
                hr = sample_data[6]
                pr = sample_data[7]
                ecg_feat = sample_data[8]
                ppg_feat = sample_data[9]
                abp_raw_data = sample_data[10]  # Get abp_raw data
                
                # Convert tensors to numpy arrays and store
                waveforms_raw[idx] = waveforms.numpy()  # Store all channels
                waveforms_local_minmax[idx] = waveforms_local_minmax_data.numpy()
                waveforms_minmax_zc[idx] = waveforms_minmax_zc_data.numpy()
                abp_global_minmax[idx] = abp.numpy()
                abp_raw[idx] = abp_raw_data.numpy()  # Store abp_raw data
                bp_raw[idx] = bp_values.numpy()
                bp_global_minmax[idx] = bp_global_minmax_data.numpy()
                
                # Store rates if available
                if hr is not None:
                    heart_rate[idx] = hr.numpy()
                if pr is not None:
                    pulse_rate[idx] = pr.numpy()
                
                # Extract ECG and PPG waveforms
                # Note: UCI dataset has different channel ordering (PPG=0, ECG=1)
                ecg_signal = waveforms[self.ecg_label].numpy()
                ppg_signal = waveforms[self.ppg_label].numpy()
                
                try:
                    # Extract PPG features
                    ppg_results = extract_ppg_features(ppg_signal, sampling_rate)
                    
                    # Extract ECG features
                    ecg_signals, ecg_info = extract_ecg_features(ecg_signal, sampling_rate)
                    peak_locations = get_peak_locations(ecg_signals)
                    
                    # Calculate ECG intervals and morphology
                    qt_intervals = calculate_qt_intervals(peak_locations, sampling_rate)
                    
                    # Convert peak locations to numpy arrays
                    peak_locations_np = {}
                    for wave_type, wave_data in peak_locations.items():
                        peak_locations_np[wave_type] = {}
                        for key, value in wave_data.items():
                            if isinstance(value, dict):
                                peak_locations_np[wave_type][key] = {}
                                for sub_key, sub_value in value.items():
                                    peak_locations_np[wave_type][key][sub_key] = np.array(sub_value)
                            else:
                                peak_locations_np[wave_type][key] = np.array(value)
                    
                    # Store features - only store the durations array from qt_intervals
                    ecg_features.append({
                        'peak_locations': peak_locations_np,
                        'qt_intervals': np.array(qt_intervals['durations']),  # Store only durations array
                        'mean_ecg_quality': np.array(np.mean(ecg_signals['ECG_Quality']))
                    })
                    
                    ppg_features.append({
                        'Asp_deltaT': np.array(ppg_results['Asp_deltaT']),
                        'IPR': np.array(ppg_results['IPR'])
                    })
                except Exception as e:
                    print(f"Error processing features for sample {idx}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Create dictionary with all data and features
        data_dict = {
            'waveforms_raw': waveforms_raw,
            'waveforms_local_minmax': waveforms_local_minmax,
            'waveforms_minmax_zc': waveforms_minmax_zc,
            'abp_global_minmax': abp_global_minmax,
            'abp_raw': abp_raw,  # Add abp_raw to data dictionary
            'bp_raw': bp_raw,
            'bp_global_minmax': bp_global_minmax,
            'heart_rate': heart_rate,
            'pulse_rate': pulse_rate
        }
        
        # Add extracted features
        features_dict = {
            'ecg_features': ecg_features,
            'ppg_features': ppg_features
        }
        
        # Combine data and features
        return {**data_dict, **features_dict}

@dataclass
class PulseDBBaseConfig(BaseModelConfig):
    """Base configuration for PulseDB dataset"""
    input_size: int = 1280

    # Dataset specific parameters
    train_sample_file: str = 'Train_Subset_PulseDB'
    test_sample_file: str = 'CalFree_Test_Subset_PulseDB'
    
    # Split configuration
    use_patient_split: bool = False  # Whether to use patient-level splitting for train/val
    is_finetuning: bool = False
    is_pretraining: bool = False

    dataset_path: str = 'PulseDB/'
    preprocessed_path: str = 'Preprocessed/'
    
    # BP normalization constants for each dataset obtained from training set value
    # PulseDB
    pulsedb_sbp_max: float = 286.58240014784946  # Global max from SBP
    pulsedb_dbp_min: float = 2.341260731456743   # Global min from DBP

    
    # Domain labels for PulseDB dataset
    ecg_label: int = 0
    ppg_label: int = 1
    abp_label: int = 2

    channel_names = {
        ecg_label: "ECG",
        ppg_label: "PPG",
        abp_label: "ABP"
        }

    def __post_init__(self):
        super().__post_init__()
        self.sbp_max = self.pulsedb_sbp_max
        self.dbp_min = self.pulsedb_dbp_min

    @staticmethod
    def build_dataset(path, field_name='Subset'):
        """
        Load and process dataset fields from a .mat file
        
        Args:
            path (str): Path to the .mat file
            field_name (str): Name of the field to load from the .mat file
        
        Returns:
            tuple: Contains processed arrays for complete dataset, MIMIC-III subset, and VitalDB subset
        """
        data = loadmat(path)
        # Load waveforms
        waveforms = data[field_name]['Signals']
        
        # Load blood pressure labels
        sbp_labels = data[field_name]['SBP']
        dbp_labels = data[field_name]['DBP']
        sbp_dbp_labels = np.stack((sbp_labels, dbp_labels), axis=1)
        
        # Load and process demographics
        age = data[field_name]['Age']
        
        # Convert gender from string array to numeric
        gender_array = data[field_name]['Gender']
        gender = np.array([1 if g[0] == 'M' else 0 for g in gender_array])
        
        height = data[field_name]['Height']
        weight = data[field_name]['Weight'] 
        bmi = data[field_name]['BMI']
        
        # Load subject IDs
        subject_ids = data[field_name]['Subject']

        # Create complete dataset (original data)
        demographics = np.stack((age, gender, height, weight, bmi), axis=1)

        # Return the complete dataset with subject IDs included
        complete_dataset = {
            'waveforms': waveforms,
            'bp_labels': sbp_dbp_labels,
            'demographics': demographics,
            'subject_ids': subject_ids
        }
        
        return complete_dataset

    def _calculate_chunk_size(self, shape, dtype):
        """Calculate optimal chunk size based on data characteristics"""
        element_size = np.dtype(dtype).itemsize # float64 = 8 bytes
        
        # Case 1: Waveform data (samples, channels, length) - (902160, 3, 1250)
        if len(shape) == 3:
            if shape[1] == 3 and shape[2] == 1250:  # Waveform specific
                # Keep all channels (3) in one chunk, optimize samples vs length
                # Aim for ~2MB chunks for waveforms
                # Calculate memory per chunk
                # Current: 50 * 3 * 1250 * 8 bytes = 1.5MB per chunk
                # Aim for ~8-10MB chunks for better I/O performance
                samples_per_chunk = min(1, shape[0])  # 50 samples per chunk
                return (samples_per_chunk, 3, 1250)
            elif shape[1] == 1 and shape[2] == 1250:  # ABP waveform
                # For single channel data
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, 1, 1250)
        
        # Case 2: 2D arrays with few columns (bp_raw, demographics)
        elif len(shape) == 2:
            if shape[1] <= 5:  # For bp_raw (3 cols) and demographics (5 cols)
                # Use larger chunks since these are small arrays
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, shape[1])
            else:
                # Generic 2D data
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, shape[1])
        
        # Case 3: 1D arrays or small data (heart_rate, pulse_rate, subject_ids)
        else:
            # For very small arrays, keep more samples per chunk
            samples_per_chunk = min(1, shape[0])
            return (samples_per_chunk,)

    def preprocess_dataset(self, input_file, output_file, calculate_bpm=False):
        """Load and preprocess the dataset with chunking"""
        # Load data using Build_Dataset
        complete_dataset = self.build_dataset(input_file)
        waveforms = complete_dataset['waveforms']
        sbp_dbp = complete_dataset['bp_labels']
        demographics = complete_dataset['demographics']
        subject_ids = complete_dataset['subject_ids']

        n_samples = len(waveforms)
        print("\nPreprocessing complete dataset...")
        
        # Initialize arrays
        waveforms_raw_list = []
        waveforms_local_minmax_list = []
        waveforms_minmax_zc_list = []
        bp_raw_list = []
        demographics_list = []
        subject_ids_list = []
        if calculate_bpm:
            hr_list = []
            pr_list = []
            failed_bpm_count = 0

        # Process all samples once
        with tqdm(total=n_samples, desc="Processing samples") as pbar:
            for i in range(n_samples):
                current_waveforms = waveforms[i]
                current_demographics = demographics[i]
                current_subject_id = subject_ids[i]
                
                if calculate_bpm:
                    bpm_result = calculate_BPM(current_waveforms,
                                             ii_label=self.ecg_label,
                                             ppg_label=self.ppg_label)
                    if bpm_result is None:
                        failed_bpm_count += 1
                        pbar.update(1)
                        continue
                    hr, pr = bpm_result
                    hr_list.append([hr])
                    pr_list.append([pr])
                
                current_minmax, current_minmax_zc = normalize_signals(current_waveforms)
                sbp, dbp = sbp_dbp[i]
                map_value = calculate_MAP(sbp, dbp)
                
                waveforms_raw_list.append(current_waveforms)
                waveforms_local_minmax_list.append(current_minmax)
                waveforms_minmax_zc_list.append(current_minmax_zc)
                bp_raw_list.append([sbp, dbp, map_value])
                demographics_list.append(current_demographics)
                subject_ids_list.append(current_subject_id)
                
                pbar.update(1)

        # Convert lists to arrays
        waveforms_raw = np.array(waveforms_raw_list)
        waveforms_local_minmax = np.array(waveforms_local_minmax_list)
        waveforms_minmax_zc = np.array(waveforms_minmax_zc_list)
        bp_raw = np.array(bp_raw_list)
        demographics_array = np.array(demographics_list)
        subject_ids_array = np.array(subject_ids_list)
        if calculate_bpm:
            heart_rate = np.array(hr_list)
            pulse_rate = np.array(pr_list)

        # Define base dataset configuration
        dataset_configs = {
            'pulsedb': {
                'indices': np.arange(len(waveforms_raw)),
                'global_minmax': {
                    "min": self.pulsedb_dbp_min,
                    "max": self.pulsedb_sbp_max
                },
                'filename': f"{output_file}_PulseDB.h5"
            }
        }

        # Before saving datasets, print information about arrays
        for dataset_name, config in dataset_configs.items():
            indices = config['indices']
            global_minmax = config['global_minmax']
            output_path = config['filename']
                        
            # Prepare arrays with their actual indices
            arrays_info = {
                'waveforms_raw': waveforms_raw[indices],
                'waveforms_local_minmax': waveforms_local_minmax[indices],
                'waveforms_minmax_zc': waveforms_minmax_zc[indices],
                'bp_raw': bp_raw[indices],
                'bp_global_minmax': Global_Min_Max_Norm(bp_raw[indices].reshape(-1, 3), 
                                                     Global_Min_Max=global_minmax),
                'demographics': demographics_array[indices],
                'abp_global_minmax': Global_Min_Max_Norm(waveforms_raw[indices, self.abp_label:self.abp_label+1].reshape(len(indices), 1, -1),
                                                      Global_Min_Max=global_minmax)
            }

            if calculate_bpm:
                arrays_info.update({
                    'heart_rate': heart_rate[indices],
                    'pulse_rate': pulse_rate[indices]
                })

            # Handle subject_ids separately with proper string encoding
            subject_ids_data = subject_ids_array[indices]
            # Convert Unicode strings to ASCII bytes with fixed length
            subject_ids_data = np.array([str(s).encode('ascii') for s in subject_ids_data], dtype='S9')
            
            # Calculate chunk size for subject IDs
            chunks_subject_ids = self._calculate_chunk_size(subject_ids_data.shape, subject_ids_data.dtype)
            
            with h5py.File(output_path, "w") as f:
                f.create_dataset('subject_ids', 
                               data=subject_ids_data,
                               chunks=chunks_subject_ids,
                               compression='gzip',
                               compression_opts=4,
                               shuffle=True)

                # Save all other numerical arrays with appropriate chunking
                for name, data in arrays_info.items():
                    chunks = self._calculate_chunk_size(data.shape, data.dtype)
                    print(f"Creating dataset {name} with shape {data.shape} and chunks {chunks}")
                    f.create_dataset(
                        name,
                        data=data,
                        chunks=chunks,
                        compression='gzip',
                        compression_opts=4,
                        shuffle=True
                    )

                # Add metadata
                f.attrs['chunked'] = True
                f.attrs['creation_date'] = np.string_(datetime.now().isoformat())
                chunk_info = {name: dataset.chunks for name, dataset in f.items()}
                f.attrs['chunk_info'] = str(chunk_info)

        if calculate_bpm:
            print(f"\nBPM calculation failed for {failed_bpm_count} samples out of {n_samples}")
            print(f"Success rate: {((n_samples - failed_bpm_count) / n_samples) * 100:.2f}%")
        
        # Verify the saved datasets
        print("\nVerifying saved datasets...")
        dataset_stats = self.verify_saved_dataset(output_file)
        return dataset_stats

    def verify_saved_dataset(self, output_file_base):
        """Verify the saved datasets with optimized chunk reading"""
        dataset_stats = {}
        dataset_types = ['PulseDB']
        
        for dataset_type in dataset_types:
            file_path = f"{output_file_base}_{dataset_type}.h5"
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"\nVerifying {dataset_type} dataset...")
            try:
                with h5py.File(file_path, 'r') as f:
                    # Get basic info without loading data
                    waveforms = f['waveforms_raw']
                    n_samples, n_channels, seq_length = waveforms.shape
                    
                    # Sample size based on dataset size
                    sample_size = min(200, n_samples // 10)
                    sample_indices = np.linspace(0, n_samples-1, sample_size, dtype=int)
                    
                    # Print dataset structure and chunk info
                    print("\nDataset Structure:")
                    for name, dataset in f.items():
                        chunk_mb = np.prod(dataset.chunks) * dataset.dtype.itemsize / (1024**2) if dataset.chunks else 0
                        print(f"\n{name}:")
                        print(f"  Shape: {dataset.shape}")
                        print(f"  Chunks: {dataset.chunks}")
                        print(f"  Chunk size: {chunk_mb:.2f} MB")
                        print(f"  Compression: {dataset.compression}")
                        print(f"  Compression ratio: {dataset.nbytes / dataset.size:.2f}x")
                    
                    # Process statistics in chunks
                    stats = self._process_chunks_statistics(f, sample_indices)
                    
                    # Print verification results
                    self._print_verification_results(stats, dataset_type)
                    
                    dataset_stats[dataset_type] = stats
                    
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        return dataset_stats

    def _process_chunks_statistics(self, f, sample_indices):
        """Process dataset statistics using chunked reading"""
        stats = {
            'n_samples': f['waveforms_raw'].shape[0],
            'n_channels': f['waveforms_raw'].shape[1],
            'seq_length': f['waveforms_raw'].shape[2],
            'file_size_mb': os.path.getsize(f.filename) / (1024**2)
        }
        
        # Initialize min/max trackers
        bp_stats = {
            'raw': {'min': float('inf'), 'max': float('-inf')},
            'norm': {'min': float('inf'), 'max': float('-inf')}
        }
        
        # Process in chunks based on dataset chunk size
        chunk_size = f['bp_raw'].chunks[0]
        for i in range(0, len(sample_indices), chunk_size):
            batch_indices = sample_indices[i:i + chunk_size]
            
            # Read BP data chunks
            bp_raw = f['bp_raw'][batch_indices]
            bp_norm = f['bp_global_minmax'][batch_indices]
            
            # Update BP statistics
            bp_stats['raw']['min'] = min(bp_stats['raw']['min'], np.min(bp_raw))
            bp_stats['raw']['max'] = max(bp_stats['raw']['max'], np.max(bp_raw))
            bp_stats['norm']['min'] = min(bp_stats['norm']['min'], np.min(bp_norm))
            bp_stats['norm']['max'] = max(bp_stats['norm']['max'], np.max(bp_norm))
        
        stats['bp_ranges'] = {
            'raw': (bp_stats['raw']['min'], bp_stats['raw']['max']),
            'norm': (bp_stats['norm']['min'], bp_stats['norm']['max'])
        }
        
        # Process heart rate and pulse rate if available
        if 'heart_rate' in f and 'pulse_rate' in f:
            hr_chunk = f['heart_rate'][sample_indices]
            pr_chunk = f['pulse_rate'][sample_indices]
            stats['rate_ranges'] = {
                'heart_rate': (np.min(hr_chunk), np.max(hr_chunk)),
                'pulse_rate': (np.min(pr_chunk), np.max(pr_chunk))
            }
        
        # Count unique subjects efficiently
        unique_subjects = set()
        for i in range(0, len(sample_indices), chunk_size):
            batch_indices = sample_indices[i:i + chunk_size]
            subjects = f['subject_ids'][batch_indices]
            unique_subjects.update([s[0].decode('utf-8') if isinstance(s, np.ndarray) 
                                  else s.decode('utf-8') for s in subjects])
        stats['n_unique_subjects'] = len(unique_subjects)
        
        return stats

    def _print_verification_results(self, stats, dataset_type):
        """Print verification results in a formatted way"""
        print(f"\n=== Verification Results for {dataset_type} ===")
        print(f"\nDataset Dimensions:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Channels: {stats['n_channels']}")
        print(f"  Sequence Length: {stats['seq_length']}")
        print(f"  Unique Subjects: {stats['n_unique_subjects']}")
        print(f"\nFile Size: {stats['file_size_mb']:.2f} MB")
        
        print("\nBlood Pressure Ranges:")
        print("  Raw:")
        print(f"    Min: {stats['bp_ranges']['raw'][0]:.2f}")
        print(f"    Max: {stats['bp_ranges']['raw'][1]:.2f}")
        print("  Normalized:")
        print(f"    Min: {stats['bp_ranges']['norm'][0]:.2f}")
        print(f"    Max: {stats['bp_ranges']['norm'][1]:.2f}")
        
        if 'rate_ranges' in stats:
            print("\nRate Ranges:")
            print(f"  Heart Rate: {stats['rate_ranges']['heart_rate'][0]:.1f} - "
                  f"{stats['rate_ranges']['heart_rate'][1]:.1f} BPM")
            print(f"  Pulse Rate: {stats['rate_ranges']['pulse_rate'][0]:.1f} - "
                  f"{stats['rate_ranges']['pulse_rate'][1]:.1f} BPM")

    def create_ddp_dataset(self):
        """Create train and validation datasets efficiently
        
        Args:
            is_finetuning (bool): If True, splits data at patient level using ratios [0.81, 0.09, 0.1]
                                 Only train (81%) and val (9%) are returned, test (10%) is kept for later
        """
        print(f"Creating DDP dataset with seed: {self.seed}")
        np.random.seed(self.seed)  # Original Seed - Keep consistent shuffling
        if self.is_pretraining:
            base_dataset = PulseDBDataset(
                sample_file=os.path.join(self.data_path, self.dataset_path,
                                            self.preprocessed_path, self.train_sample_file),
                sample_length=self.input_size
            )
            dataset_size = len(base_dataset)

            test_dataset = PulseDBDataset(
                sample_file=os.path.join(self.data_path, self.dataset_path,
                                            self.preprocessed_path, self.test_sample_file),
                sample_length=self.input_size
            )

            if self.use_patient_split:
                # Group indices by patient ID for patient-level splitting
                indices = np.arange(dataset_size)
                subject_ids = base_dataset.data['subject_ids']
                unique_subjects = np.unique(subject_ids)
                train_indices = []
                val_indices = []
                
                # For each patient, split their samples according to 80/20 ratio
                for subject in unique_subjects:
                    patient_indices = indices[subject_ids == subject]
                    np.random.shuffle(patient_indices)
                    
                    # Calculate split point
                    n_samples = len(patient_indices)
                    train_split = int(0.8 * n_samples)
                    
                    # Split patient samples
                    train_indices.extend(patient_indices[:train_split])
                    val_indices.extend(patient_indices[train_split:])
                
                # Create train and validation datasets
                train_dataset = PulseDBDataset.create_subset(base_dataset, train_indices)
                val_dataset = PulseDBDataset.create_subset(base_dataset, val_indices)
                
                # Print split statistics
                print(f"\nPretraining - Patient-Level Split Statistics:")
                print(f"Seed: {np.random.get_state()[1][0]}")
                print(f"Total samples: {dataset_size}")
                print(f"Train samples: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
                print(f"Val samples: {len(val_indices)} ({len(val_indices)/dataset_size*100:.1f}%)")
                print(f"Test samples: {len(test_dataset)} ({len(test_dataset)/dataset_size*100:.1f}%)")
                print(f"Unique subjects: {len(unique_subjects)}")
            else:
                train_size = int(0.8 * dataset_size)
                validation_size = int(dataset_size-train_size)

                indices = np.arange(dataset_size)
                # Original random split across all samples (80/20)
                np.random.shuffle(indices)

                train_dataset = PulseDBDataset.create_subset(base_dataset, indices[:train_size])
                val_dataset = PulseDBDataset.create_subset(base_dataset, indices[train_size:])

                # Print split statistics
                print(f"\nPretraining - Random Split Statistics:")
                print(f"Seed: {np.random.get_state()[1][0]}")
                print(f"Total samples: {dataset_size}")
                print(f"Train samples: {len(train_dataset)} ({len(train_dataset)/dataset_size*100:.1f}%)")
                print(f"Val samples: {len(val_dataset)} ({len(val_dataset)/dataset_size*100:.1f}%)")
                print(f"Test samples: {len(test_dataset)} ({len(test_dataset)/dataset_size*100:.1f}%)")
        elif self.is_finetuning or self.model_type == 'refinement':
            # For regular refinement model or our model finetuning, we should load the test set (CalFree)
            base_dataset = PulseDBDataset(
                sample_file=os.path.join(self.data_path, self.dataset_path,
                                            self.preprocessed_path, self.test_sample_file),
                sample_length=self.input_size
            )
            dataset_size = len(base_dataset)
            indices = np.arange(dataset_size)
            # Group indices by patient ID
            subject_ids = base_dataset.data['subject_ids']
            unique_subjects = np.unique(subject_ids)
            train_indices = []
            val_indices = []
            test_indices = []  # Will be kept for later evaluation
            
            # Split ratios for finetuning
            train_ratio, val_ratio, test_ratio = 0.81, 0.09, 0.10
            
            # For each patient, split their samples according to ratios
            for subject in unique_subjects:
                patient_indices = indices[subject_ids == subject]
                np.random.shuffle(patient_indices)
                
                # Calculate split points
                n_samples = len(patient_indices)
                train_split = int(train_ratio * n_samples)
                val_split = int((train_ratio + val_ratio) * n_samples)
                
                # Split patient samples according to ratios
                train_indices.extend(patient_indices[:train_split])
                val_indices.extend(patient_indices[train_split:val_split])
                test_indices.extend(patient_indices[val_split:])  # Kept for later
            
            # Create train and validation datasets (test portion is not used here)
            train_dataset = PulseDBDataset.create_subset(base_dataset, train_indices)
            val_dataset = PulseDBDataset.create_subset(base_dataset, val_indices)
            test_dataset = PulseDBDataset.create_subset(base_dataset, test_indices)
            
            # Print split statistics
            print(f"\n {'Finetuning' if self.is_finetuning else 'Refinement'} - Patient-Level Split Statistics:")
            print(f"Seed: {np.random.get_state()[1][0]}")
            print(f"Total samples: {dataset_size}")
            print(f"Train samples: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
            print(f"Val samples: {len(val_indices)} ({len(val_indices)/dataset_size*100:.1f}%)")
            print(f"Test samples (reserved): {len(test_indices)} ({len(test_indices)/dataset_size*100:.1f}%)")
            print(f"Unique subjects: {len(unique_subjects)}")
        else:
            base_dataset = PulseDBDataset(
                sample_file=os.path.join(self.data_path, self.dataset_path,
                                            self.preprocessed_path, self.train_sample_file),
                sample_length=self.input_size
            )
            dataset_size = len(base_dataset)

            test_dataset = PulseDBDataset(
                sample_file=os.path.join(self.data_path, self.dataset_path,
                                            self.preprocessed_path, self.test_sample_file),
                sample_length=self.input_size
            )

            if self.use_patient_split:
                # Group indices by patient ID for patient-level splitting
                indices = np.arange(dataset_size)
                subject_ids = base_dataset.data['subject_ids']
                unique_subjects = np.unique(subject_ids)
                train_indices = []
                val_indices = []
                
                # For each patient, split their samples according to 80/20 ratio
                for subject in unique_subjects:
                    patient_indices = indices[subject_ids == subject]
                    np.random.shuffle(patient_indices)
                    
                    # Calculate split point
                    n_samples = len(patient_indices)
                    train_split = int(0.8 * n_samples)
                    
                    # Split patient samples
                    train_indices.extend(patient_indices[:train_split])
                    val_indices.extend(patient_indices[train_split:])
                
                # Create train and validation datasets
                train_dataset = PulseDBDataset.create_subset(base_dataset, train_indices)
                val_dataset = PulseDBDataset.create_subset(base_dataset, val_indices)
                
                # Print split statistics
                print(f"\nApproximation - Patient-Level Split Statistics:")
                print(f"Seed: {np.random.get_state()[1][0]}")
                print(f"Total samples: {dataset_size}")
                print(f"Train samples: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
                print(f"Val samples: {len(val_indices)} ({len(val_indices)/dataset_size*100:.1f}%)")
                print(f"Test samples: {len(test_dataset)} ({len(test_dataset)/dataset_size*100:.1f}%)")
                print(f"Unique subjects: {len(unique_subjects)}")
            else:
                train_size = int(0.8 * dataset_size)
                validation_size = int(dataset_size-train_size)

                indices = np.arange(dataset_size)
                # Original random split across all samples (80/20)
                np.random.shuffle(indices)

                train_dataset = PulseDBDataset.create_subset(base_dataset, indices[:train_size])
                val_dataset = PulseDBDataset.create_subset(base_dataset, indices[train_size:])

                # Print split statistics
                print(f"\nApproximation - Random Split Statistics:")
                print(f"Seed: {np.random.get_state()[1][0]}")
                print(f"Total samples: {dataset_size}")
                print(f"Train samples: {len(train_dataset)} ({len(train_dataset)/dataset_size*100:.1f}%)")
                print(f"Val samples: {len(val_dataset)} ({len(val_dataset)/dataset_size*100:.1f}%)")
                print(f"Test samples: {len(test_dataset)} ({len(test_dataset)/dataset_size*100:.1f}%)")
        return (train_dataset, val_dataset, test_dataset)

    def create_test_dataset(self):
        """Create test dataset
        
        Args:
            is_finetuning (bool): If True, returns the 10% test portion from the training file,
                                 otherwise returns the separate test file dataset
        """
        print(f"Creating test dataset with seed: {self.seed}")
        np.random.seed(self.seed)  # Original Seed - Keep consistent shuffling
        if (self.is_finetuning or self.model_type == 'refinement') and (not self.is_pretraining):
            # Load the training file and extract the 10% test portion
            base_dataset = PulseDBDataset(
                sample_file=os.path.join(self.data_path, self.dataset_path,
                                       self.preprocessed_path, self.test_sample_file),
                sample_length=self.input_size
            )
            
            # Get all indices and subject IDs
            indices = np.arange(len(base_dataset))
            subject_ids = base_dataset.data['subject_ids']
            unique_subjects = np.unique(subject_ids)
            test_indices = []
            
            # Split ratios for finetuning
            train_ratio, val_ratio = 0.81, 0.09  # test_ratio = 0.10
            
            # For each patient, get their test portion (last 10%)
            for subject in unique_subjects:
                patient_indices = indices[subject_ids == subject]
                np.random.shuffle(patient_indices)
                
                # Calculate split point for test portion
                val_split = int((train_ratio + val_ratio) * len(patient_indices))
                
                # Get test indices (last 10%)
                test_indices.extend(patient_indices[val_split:])
            
            # Create and return test dataset
            test_dataset = PulseDBDataset.create_subset(base_dataset, test_indices)
            
            # Print test set statistics
            print(f"\nFinetuning Test Set Statistics:")
            print(f"Seed: {np.random.get_state()[1][0]}")
            print(f"Total test samples: {len(test_indices)}")
            print(f"Percentage of original data: {len(test_indices)/len(base_dataset)*100:.1f}%")
            print(f"Number of test subjects: {len(unique_subjects)}")
            
            return test_dataset
        else:
            # Regular test dataset from separate test file
            base_dataset = PulseDBDataset(
                sample_file=os.path.join(self.data_path, self.dataset_path,
                                       self.preprocessed_path, self.test_sample_file),
                sample_length=self.input_size
            )
            return base_dataset    
    
    def extract_waveform_features_method(self, sampling_rate: int = 125):
        """Load test set and extract features from PPG and ECG waveforms
        
        Args:
            sampling_rate (int): Sampling rate of the signals in Hz. Default is 1000.
            
        Returns:
            dict: Dictionary containing:
                - ecg_features: List of dictionaries containing ECG features for each sample
                - ppg_features: List of dictionaries containing PPG features for each sample
                - subject_ids: List of subject IDs
                - sample_indices: List of sample indices
                - waveforms: List of waveform arrays
                - bp_values: List of blood pressure values
                - demographics: List of demographic data
                - demographics_explicit: List of explicit demographic data
                - text_encoding: List of text encodings
                - waveforms_local_minmax: List of locally min-max normalized waveforms
                - waveforms_minmax_zc: List of zero-centered min-max normalized waveforms
                - rates: List of heart/pulse rates
                - bp_global_minmax: List of globally min-max normalized BP values
                - abp_global_minmax: List of globally min-max normalized ABP values
        """
        # Load test dataset
        test_dataset = self.create_test_dataset()
        print(f"\nLoaded test dataset with {len(test_dataset)} samples")
        
        # Initialize lists to store features and data
        ecg_features = []
        ppg_features = []
        subject_ids = []
        sample_indices = []
        
        # Initialize arrays for storing data in DataLoadPulseDB format
        n_samples = len(test_dataset)
        waveforms_raw = np.zeros((n_samples, 3, self.input_size))
        waveforms_local_minmax = np.zeros((n_samples, 3, self.input_size))
        waveforms_minmax_zc = np.zeros((n_samples, 3, self.input_size))
        abp_global_minmax = np.zeros((n_samples, 1, self.input_size))
        bp_raw = np.zeros((n_samples, 3))  # SBP, DBP, MAP
        demographics = np.zeros((n_samples, 5))  # age, gender, height, weight, bmi
        bp_global_minmax = np.zeros((n_samples, 3))  # normalized SBP, DBP, MAP
        heart_rate = np.zeros((n_samples, 1))
        pulse_rate = np.zeros((n_samples, 1))
        
        # Process each sample with tqdm progress bar
        for idx in tqdm(range(len(test_dataset)), desc="Extracting waveform features"):
            # Get sample data
            subject_id, sample_idx, waveforms, bp_values, demographics_data, demographics_explicit, \
            text_encoding, waveforms_local_minmax_data, waveforms_minmax_zc_data, rates, \
            bp_global_minmax_data, abp_global_minmax_data, ecg_features_data, ppg_features_data = test_dataset[idx]
            
            # Store data in DataLoadPulseDB format
            subject_ids.append(subject_id)
            sample_indices.append(sample_idx)
            
            # Convert tensors to numpy arrays and store
            waveforms_raw[idx] = waveforms.numpy()
            waveforms_local_minmax[idx] = waveforms_local_minmax_data.numpy()
            waveforms_minmax_zc[idx] = waveforms_minmax_zc_data.numpy()
            abp_global_minmax[idx] = abp_global_minmax_data.numpy()
            bp_raw[idx] = bp_values.numpy()
            demographics[idx] = demographics_data.numpy()
            bp_global_minmax[idx] = bp_global_minmax_data.numpy()
            
            # Store rates if available
            if rates is not None:
                heart_rate[idx] = rates[0].numpy()
                pulse_rate[idx] = rates[1].numpy()
            
            # Extract ECG and PPG waveforms
            ecg_signal = waveforms[self.ecg_label,15:-15].numpy()
            ppg_signal = waveforms[self.ppg_label,15:-15].numpy()
            
            try:
                # Extract PPG features
                ppg_results = extract_ppg_features(ppg_signal, sampling_rate)

                # # Extract ECG features
                ecg_signals, ecg_info = extract_ecg_features(ecg_signal, sampling_rate)
                peak_locations = get_peak_locations(ecg_signals)
                
                # # Calculate ECG intervals and morphology
                qt_intervals = calculate_qt_intervals(peak_locations, sampling_rate)
                
                # # Store features
                ecg_features.append({
                    'peak_locations': peak_locations,
                    'qt_intervals': qt_intervals['durations'],  # Store only durations
                    'mean_ecg_quality': np.mean(ecg_signals['ECG_Quality'])
                })

                # Store specific PPG features
                ppg_features.append({
                    'Asp_deltaT': ppg_results['Asp_deltaT'],
                    'IPR': ppg_results['IPR']
                })
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        print(f"\nSuccessfully processed {len(ecg_features)} samples")
        
        # Create dictionary in DataLoadPulseDB format
        data_dict = {
            'waveforms_raw': waveforms_raw,
            'waveforms_local_minmax': waveforms_local_minmax,
            'waveforms_minmax_zc': waveforms_minmax_zc,
            'abp_global_minmax': abp_global_minmax,
            'bp_raw': bp_raw,
            'demographics': demographics,
            'bp_global_minmax': bp_global_minmax,
            'subject_ids': np.array(subject_ids, dtype=str),
            'heart_rate': heart_rate,
            'pulse_rate': pulse_rate,
            'size': n_samples
        }
        
        # Add extracted features
        features_dict = {
            'ecg_features': ecg_features,
            'ppg_features': ppg_features,
            'sample_indices': sample_indices
        }
        
        return {**data_dict, **features_dict}

