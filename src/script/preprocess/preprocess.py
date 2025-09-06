"""
Independent preprocessing script for PulseDB and UCI datasets using Hydra.
This script can be run standalone to preprocess data files and save them in HDF5 format.

Features:
- Type-safe configuration using OmegaConf.structured()
- Automatic validation of configuration parameters
- Support for command-line overrides with type checking
- Input/output path validation
- Strict configuration validation to prevent typos
- Structured logging with Hydra integration

Usage:
    python -m src.script.preprocess.preprocess dataset=pulsedb input_file=path/to/data.mat output_file=output/path
    python -m src.script.preprocess.preprocess dataset=uci input_file=path/to/data.h5 output_file=output/path sbp_max=200.0
"""

# Standard library imports
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

# Third-party imports
import numpy as np
import h5py
from mat73 import loadmat
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, MISSING, OmegaConf
from hydra.core.config_store import ConfigStore

# Local imports
from src.utils.utils_preprocessing import (
    normalize_signals, calculate_MAP, 
    Global_Min_Max_Norm
)

# Setup logging
import logging
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    # Required fields for Hydra (MISSING ensures they must be provided)
    dataset: str = MISSING  # 'pulsedb' or 'uci'
    input_file: str = MISSING
    output_file: str = MISSING

    # Signal labels
    ecg_label: int = 0
    ppg_label: int = 1
    abp_label: int = 2

@dataclass
class UCIPreprocessingConfig(PreprocessingConfig):
    # UCI-specific defaults/overrides
    sbp_max: float = 189.98421357007769
    dbp_min: float = 50
    # Add or override more UCI-specific fields as needed

@dataclass
class PulseDBPreprocessingConfig(PreprocessingConfig):
    # PulseDB-specific defaults/overrides
    dbp_min: float = 2.341260731456743
    sbp_max: float = 286.58240014784946

# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="preprocessing_config", node=PreprocessingConfig)

class BaseDatasetPreprocessor:
    """Base class for dataset preprocessing, containing shared logic."""
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def _save_to_hdf5(self, output_path: str, arrays_info: Dict, subject_ids_data: np.ndarray = None, additional_metadata: Dict = None):
        """Save arrays to HDF5 file with proper chunking and optional metadata
        
        Args:
            output_path: Path to save the HDF5 file
            arrays_info: Dictionary of numpy arrays to save
            subject_ids_data: Optional subject IDs array (PulseDB only)
            additional_metadata: Optional additional metadata to store in file attributes
        """
        logger.info(f"Saving dataset to {output_path}")
        with h5py.File(output_path, "w") as f:
            # Save subject IDs separately if provided
            if subject_ids_data is not None:
                chunks_subject_ids = self._calculate_chunk_size(subject_ids_data.shape, subject_ids_data.dtype)
                f.create_dataset('subject_ids', 
                               data=subject_ids_data,
                               chunks=chunks_subject_ids,
                               compression='gzip',
                               compression_opts=4,
                               shuffle=True)
            # Save all other numerical arrays
            for name, data in arrays_info.items():
                chunks = self._calculate_chunk_size(data.shape, data.dtype)
                logger.debug(f"Creating dataset {name} with shape {data.shape} and chunks {chunks}")
                f.create_dataset(
                    name,
                    data=data,
                    chunks=chunks,
                    compression='gzip',
                    compression_opts=4,
                    shuffle=True
                )
            # Add standard metadata
            f.attrs['chunked'] = True
            f.attrs['creation_date'] = np.string_(datetime.now().isoformat())
            chunk_info = {name: dataset.chunks for name, dataset in f.items()}
            f.attrs['chunk_info'] = str(chunk_info)
            
            # Add dataset-specific metadata if provided
            if additional_metadata:
                for key, value in additional_metadata.items():
                    f.attrs[key] = value
                    
        logger.info(f"Successfully saved dataset to {output_path}")

    def _calculate_chunk_size(self, shape: Tuple, dtype) -> Tuple:
        """Calculate optimal chunk size based on data characteristics"""
        element_size = np.dtype(dtype).itemsize
        if len(shape) == 3:
            if shape[1] == 3:
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, 3, shape[2])
            elif shape[1] == 1:
                samples_per_chunk = min(1, shape[0])
                return (samples_per_chunk, 1, shape[2])
        elif len(shape) == 2:
            samples_per_chunk = min(1, shape[0])
            return (samples_per_chunk, shape[1])
        else:
            samples_per_chunk = min(1, shape[0])
            return (samples_per_chunk,)

    def _verify_saved_dataset(self, output_file: str):
        """Verify the saved dataset with optimized chunk reading
        
        Works for both PulseDB and UCI datasets since they share the same core structure.
        Both dataset types save datasets directly in the file root, not in groups.
        """
        logger.info(f"Verifying dataset: {output_file}")
        try:
            with h5py.File(output_file, 'r') as f:
                # Check if required datasets exist
                required_datasets = ['waveforms_raw', 'bp_raw', 'bp_global_minmax', 'abp_raw', 'abp_global_minmax']
                missing_datasets = [ds for ds in required_datasets if ds not in f]
                if missing_datasets:
                    logger.error(f"Missing required datasets: {missing_datasets}")
                    logger.error(f"Available datasets: {list(f.keys())}")
                    raise KeyError(f"Missing required datasets: {missing_datasets}")
                
                # Both PulseDB and UCI use flat structure with same core datasets
                waveforms = f['waveforms_raw']
                n_samples, n_channels, seq_length = waveforms.shape
                sample_size = min(200, n_samples // 10)
                sample_indices = np.linspace(0, n_samples-1, sample_size, dtype=int)
                
                logger.debug("Dataset Structure:")
                for name, dataset in f.items():
                    chunk_mb = np.prod(dataset.chunks) * dataset.dtype.itemsize / (1024**2) if dataset.chunks else 0
                    logger.debug(f"{name}: Shape={dataset.shape}, Chunks={dataset.chunks}, "
                               f"Chunk size={chunk_mb:.2f}MB, Compression={dataset.compression}, "
                               f"Compression ratio={dataset.nbytes / dataset.size:.2f}x")
                
                stats = self._process_chunks_statistics(f, sample_indices)
                self._print_verification_results(stats, "dataset")
                
        except KeyError as e:
            logger.error(f"Missing required dataset: {e}")
            # Don't try to access f.keys() here as the file might be closed
            raise
        except Exception as e:
            logger.error(f"Error loading {output_file}: {str(e)}")
            raise

    def _process_chunks_statistics(self, file_group, sample_indices):
        """Process dataset statistics using chunked reading
        
        Note: Both PulseDB and UCI use flat HDF5 structure with same core datasets
        Required datasets: waveforms_raw, bp_raw, bp_global_minmax, abp_raw, abp_global_minmax
        Optional datasets: subject_ids (PulseDB only), demographics (PulseDB only)
        """
        stats = {
            'n_samples': file_group['waveforms_raw'].shape[0],
            'n_channels': file_group['waveforms_raw'].shape[1],
            'seq_length': file_group['waveforms_raw'].shape[2],
            'file_size_mb': os.path.getsize(file_group.file.filename) / (1024**2)
        }
        bp_stats = {
            'raw': {'min': float('inf'), 'max': float('-inf')},
            'norm': {'min': float('inf'), 'max': float('-inf')}
        }
        abp_stats = {
            'raw': {'min': float('inf'), 'max': float('-inf')},
            'norm': {'min': float('inf'), 'max': float('-inf')}
        }
        chunk_size = file_group['bp_raw'].chunks[0]
        for i in range(0, len(sample_indices), chunk_size):
            batch_indices = sample_indices[i:i + chunk_size]
            bp_raw = file_group['bp_raw'][batch_indices]
            bp_norm = file_group['bp_global_minmax'][batch_indices]
            abp_raw = file_group['abp_raw'][batch_indices]
            abp_norm = file_group['abp_global_minmax'][batch_indices]
            bp_stats['raw']['min'] = min(bp_stats['raw']['min'], np.min(bp_raw))
            bp_stats['raw']['max'] = max(bp_stats['raw']['max'], np.max(bp_raw))
            bp_stats['norm']['min'] = min(bp_stats['norm']['min'], np.min(bp_norm))
            bp_stats['norm']['max'] = max(bp_stats['norm']['max'], np.max(bp_norm))
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
        return stats

    def _print_verification_results(self, stats, dataset_name):
        """Print verification results in a formatted way"""
        logger.info(f"=== Verification Results for {dataset_name} ===")
        logger.info(f"Dataset Dimensions: Samples={stats['n_samples']}, "
                   f"Channels={stats['n_channels']}, Sequence Length={stats['seq_length']}")
        logger.info(f"File Size: {stats['file_size_mb']:.2f} MB")
        
        logger.info("Blood Pressure Ranges:")
        logger.info(f"  Raw: Min={stats['bp_ranges']['raw'][0]:.2f}, Max={stats['bp_ranges']['raw'][1]:.2f}")
        logger.info(f"  Normalized: Min={stats['bp_ranges']['norm'][0]:.2f}, Max={stats['bp_ranges']['norm'][1]:.2f}")
        
        logger.info("ABP Ranges:")
        logger.info(f"  Raw: Min={stats['abp_ranges']['raw'][0]:.2f}, Max={stats['abp_ranges']['raw'][1]:.2f}")
        logger.info(f"  Normalized: Min={stats['abp_ranges']['norm'][0]:.2f}, Max={stats['abp_ranges']['norm'][1]:.2f}")

class PulseDBDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor for PulseDB dataset."""
    def preprocess_dataset(self, input_file: str, output_file: str):
        logger.info(f"Preprocessing PulseDB dataset: {input_file}")
        complete_dataset = self._build_pulsedb_dataset(input_file)
        waveforms = complete_dataset['waveforms']
        sbp_dbp = complete_dataset['bp_labels']
        demographics = complete_dataset['demographics']
        subject_ids = complete_dataset['subject_ids']

        n_samples = len(waveforms)
        logger.info(f"Total samples: {n_samples}")
        
        # Initialize arrays
        waveforms_raw_list = []
        waveforms_raw_abp_global_minmax_list = []
        waveforms_local_minmax_list = []
        waveforms_minmax_zc_list = []
        bp_raw_list = []
        demographics_list = []
        subject_ids_list = []
        abp_raw_list = []

        # Process all samples
        with tqdm(total=n_samples, desc="Processing samples") as pbar:
            for i in range(n_samples):
                current_waveforms = waveforms[i]
                current_demographics = demographics[i]
                current_subject_id = subject_ids[i]
                
                current_minmax, current_minmax_zc = normalize_signals(current_waveforms)
                
                abp_raw = current_waveforms[self.config.abp_label:self.config.abp_label+1]
                sbp, dbp = sbp_dbp[i]
                map_value = calculate_MAP(sbp, dbp)
                
                waveforms_raw = np.concatenate((current_waveforms[:self.config.ppg_label+1], current_minmax[self.config.abp_label:self.config.abp_label+1]), axis=0)

                abp_global_minmax = Global_Min_Max_Norm(abp_raw,
                                                  Global_Min_Max={"min": self.config.dbp_min, "max": self.config.sbp_max})

                waveforms_raw_abp_global_minmax = waveforms_raw.copy()
                waveforms_raw_abp_global_minmax[2] = abp_global_minmax.squeeze()

                waveforms_raw_list.append(waveforms_raw)
                waveforms_raw_abp_global_minmax_list.append(waveforms_raw_abp_global_minmax)
                waveforms_local_minmax_list.append(current_minmax)
                waveforms_minmax_zc_list.append(current_minmax_zc)
                abp_raw_list.append(abp_raw)
                bp_raw_list.append([sbp, dbp, map_value])
                demographics_list.append(current_demographics)
                subject_ids_list.append(current_subject_id)
                
                pbar.update(1)

        # Convert lists to arrays
        waveforms_raw = np.array(waveforms_raw_list)
        waveforms_raw_abp_global_minmax = np.array(waveforms_raw_abp_global_minmax_list)
        waveforms_local_minmax = np.array(waveforms_local_minmax_list)
        waveforms_minmax_zc = np.array(waveforms_minmax_zc_list)
        abp_raw = np.array(abp_raw_list)
        bp_raw = np.array(bp_raw_list)
        demographics_array = np.array(demographics_list)
        subject_ids_array = np.array(subject_ids_list)

        # Prepare arrays for saving
        arrays_info = {
            'waveforms_raw': waveforms_raw,
            'waveforms_raw_abp_global_minmax': waveforms_raw_abp_global_minmax,
            'waveforms_local_minmax': waveforms_local_minmax,
            'waveforms_minmax_zc': waveforms_minmax_zc,
            'abp_raw': abp_raw,
            'bp_raw': bp_raw,
            'bp_global_minmax': Global_Min_Max_Norm(bp_raw.reshape(-1, 3), 
                                                 Global_Min_Max={"min": self.config.dbp_min, "max": self.config.sbp_max}),
            'demographics': demographics_array,
            'abp_global_minmax': Global_Min_Max_Norm(abp_raw.reshape(len(abp_raw), 1, -1),
                                                  Global_Min_Max={"min": self.config.dbp_min, "max": self.config.sbp_max}),
        }

        # Handle subject_ids separately with proper string encoding
        subject_ids_data = np.array([str(s).encode('ascii') for s in subject_ids_array], dtype='S9')
        
        # Save to HDF5 file
        file_name = Path(input_file).stem
        output_path = os.path.join(output_file, f"PulseDB_{file_name}.h5")
        additional_metadata = {'dataset_name': 'PulseDB'}
        self._save_to_hdf5(output_path, arrays_info, subject_ids_data, additional_metadata)
        
        # Verify the saved dataset
        logger.info("Verifying saved dataset...")
        self._verify_saved_dataset(output_path)
        
        return output_path
    def _build_pulsedb_dataset(self, path: str, field_name: str = 'Subset') -> Dict:
        data = loadmat(path)
        waveforms = data[field_name]['Signals']
        sbp_labels = data[field_name]['SBP']
        dbp_labels = data[field_name]['DBP']
        sbp_dbp_labels = np.stack((sbp_labels, dbp_labels), axis=1)
        age = data[field_name]['Age']
        gender_array = data[field_name]['Gender']
        gender = np.array([1 if g[0] == 'M' else 0 for g in gender_array])
        height = data[field_name]['Height']
        weight = data[field_name]['Weight'] 
        bmi = data[field_name]['BMI']
        subject_ids = data[field_name]['Subject']
        demographics = np.stack((age, gender, height, weight, bmi), axis=1)
        return {
            'waveforms': waveforms,
            'bp_labels': sbp_dbp_labels,
            'demographics': demographics,
            'subject_ids': subject_ids
        }

class UCIDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor for UCI dataset."""
    def preprocess_dataset(self, input_file: str, output_file: str):
        logger.info(f"Preprocessing UCI dataset: {input_file}")
        
        # Load data from .h5 files
        splits = {
            'train': {'X': [], 'ABP_GRND': [], 'SBP': [], 'DBP': []},
            'test': {'X': [], 'SBP': [], 'DBP': [], 'ABP_GRND': []}
        }

        # Get the base directory from input_file
        base_dir = os.path.dirname(input_file)

        # Load training data from parts 1-3 
        for part_num in range(1, 4):
            part_file = os.path.join(base_dir, f"UCI_Dataset_Part_{part_num}_Preprocessed.h5")
            logger.info(f"Loading training part {part_num} from {part_file}...")
            with h5py.File(part_file, "r") as f:
                # Load PPG and ECG data
                ppg_data = np.array(f.get('PPG'))
                ecg_data = np.array(f.get('ECG'))
                abp_data = np.array(f.get('ABP_GRND'))[:, np.newaxis]
                abp_norm_data = np.array(f.get('ABP_RNorm'))
                
                # Stack channels for X
                x_data = np.stack((ecg_data, ppg_data, abp_norm_data), axis=1)
                
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
        logger.info(f"Loading test data from {test_file}...")
        with h5py.File(test_file, "r") as f:
            ppg_data = np.array(f.get('PPG'))
            ecg_data = np.array(f.get('ECG'))
            abp_data = np.array(f.get('ABP_GRND'))[:, np.newaxis]
            abp_norm_data = np.array(f.get('ABP_RNorm'))
            
            x_data = np.stack((ecg_data, ppg_data, abp_norm_data), axis=1)
            sbp_data = np.array(f.get('SBP'))
            dbp_data = np.array(f.get('DBP'))
            
            splits['test']['X'] = x_data
            splits['test']['SBP'] = sbp_data
            splits['test']['DBP'] = dbp_data
            splits['test']['ABP_GRND'] = abp_data

        # Concatenate training data
        for key in ['X', 'ABP_GRND', 'SBP', 'DBP']:
            splits['train'][key] = np.concatenate(splits['train'][key], axis=0)

        # Print dataset statistics
        logger.info("Dataset Statistics:")
        for split_name, split_data in splits.items():
            logger.info(f"{split_name} samples: {len(split_data['X'])}")

        # Process each split and save to separate files
        output_files = []
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split...")
            n_samples = len(split_data['X'])
            
            # Initialize arrays for current split
            waveforms_raw = []
            waveforms_raw_abp_global_minmax = []
            waveforms_local_minmax = []
            waveforms_minmax_zc = []
            bp_raw = []
            bp_global_minmax = []
            abp_global_minmax = []
            abp_raw = []
            
            # Process samples
            with tqdm(total=n_samples, desc=f"Processing {split_name} samples") as pbar:
                for i in range(n_samples):
                    current_waveforms = split_data['X'][i]
                    
                    # Normalize waveforms
                    current_minmax, current_minmax_zc = normalize_signals(current_waveforms)

                    # Calculate BP values
                    sbp = split_data['SBP'][i]
                    dbp = split_data['DBP'][i]
                    map_value = calculate_MAP(sbp, dbp)
                    
                    # Get ABP ground truth
                    abp_grnd = split_data['ABP_GRND'][i]
                    
                    # Apply global min-max normalization
                    Global_Min_Max = {"min": self.config.dbp_min, "max": self.config.sbp_max}
                    global_min_max_abp = Global_Min_Max_Norm(abp_grnd, Global_Min_Max=Global_Min_Max)
                    global_min_max_sbp = Global_Min_Max_Norm(sbp, Global_Min_Max=Global_Min_Max)
                    global_min_max_dbp = Global_Min_Max_Norm(dbp, Global_Min_Max=Global_Min_Max)
                    global_min_max_map = Global_Min_Max_Norm(map_value, Global_Min_Max=Global_Min_Max)

                    waveforms_raw_global_minmax_abp = current_waveforms.copy()
                    waveforms_raw_global_minmax_abp[2] = global_min_max_abp
                    
                    # Store processed data
                    waveforms_raw.append(current_waveforms)
                    waveforms_raw_abp_global_minmax.append(waveforms_raw_global_minmax_abp)
                    waveforms_local_minmax.append(current_minmax)
                    waveforms_minmax_zc.append(current_minmax_zc)
                    bp_raw.append([sbp, dbp, map_value])
                    bp_global_minmax.append([global_min_max_sbp, global_min_max_dbp, global_min_max_map])
                    abp_global_minmax.append(global_min_max_abp)
                    abp_raw.append(abp_grnd)
                    
                    pbar.update(1)
            
            # Convert lists to arrays
            arrays_info = {
                'waveforms_raw': np.array(waveforms_raw),
                'waveforms_raw_abp_global_minmax': np.array(waveforms_raw_abp_global_minmax),
                'waveforms_local_minmax': np.array(waveforms_local_minmax),
                'waveforms_minmax_zc': np.array(waveforms_minmax_zc),
                'bp_raw': np.array(bp_raw),
                'bp_global_minmax': np.array(bp_global_minmax),
                'abp_global_minmax': np.array(abp_global_minmax),
                'abp_raw': np.array(abp_raw)
            }

            # Save each split to a separate file
            split_output_path = os.path.join(output_file, f"UCI_{split_name}.h5")
            additional_metadata = {'dataset_name': 'UCI', 'split': split_name}
            self._save_to_hdf5(split_output_path, arrays_info, 
                               subject_ids_data=None, 
                               additional_metadata=additional_metadata)
            logger.info(f"Verifying saved dataset for {split_name}...")
            self._verify_saved_dataset(split_output_path)
            output_files.append(split_output_path)
        
        logger.info(f"Preprocessing complete! Output files: {output_files}")
        return output_files

def validate_paths(input_file: str, output_file: str):
    """✅ Validate input/output paths before processing"""
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

@hydra.main(version_base=None, config_name="preprocessing_config")
def main(cfg: DictConfig):
    """Main preprocessing function with Hydra configuration and enhanced validation"""
    
    logger.info("=" * 60)
    logger.info("MD-ViSCo Preprocessing Configuration")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cfg.dataset}")
    logger.info(f"Input file: {cfg.input_file}")
    logger.info(f"Output file: {cfg.output_file}")
    
    # ✅ SAFE: Use OmegaConf.structured() for type-safe config conversion
    if cfg.dataset == "pulsedb":
        # Convert config to dict and create structured config
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        structured_cfg = OmegaConf.structured(PulseDBPreprocessingConfig(**cfg_dict))
        # ✅ STRICT: Prevent accidental use of undeclared fields
        OmegaConf.set_struct(structured_cfg, True)
        preprocessor = PulseDBDatasetPreprocessor(structured_cfg)
    elif cfg.dataset == "uci":
        # Convert config to dict and create structured config
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        structured_cfg = OmegaConf.structured(UCIPreprocessingConfig(**cfg_dict))
        # ✅ STRICT: Prevent accidental use of undeclared fields
        OmegaConf.set_struct(structured_cfg, True)
        preprocessor = UCIDatasetPreprocessor(structured_cfg)
    else:
        logger.error(f"Unknown dataset name: {cfg.dataset}. Must be 'pulsedb' or 'uci'")
        raise ValueError(f"Unknown dataset name: {cfg.dataset}. Must be 'pulsedb' or 'uci'")
    
    # ✅ VALIDATE: Check input/output paths before processing
    validate_paths(cfg.input_file, cfg.output_file)
    
    # Preprocess the dataset
    output_file = preprocessor.preprocess_dataset(cfg.input_file, cfg.output_file)
    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()