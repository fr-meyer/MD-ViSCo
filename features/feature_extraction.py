# Standard library imports
import os
import argparse
from datetime import datetime

# Third-party imports
import numpy as np
import h5py

# Local imports
from config.datasets.dataset_configs import PulseDBBaseConfig, UCIBaseConfig
from utils.utils_preprocessing import safe_create_dataset

# Set environment variable
os.environ['OMP_NUM_THREADS'] = '2'

def save_features_to_h5(features, output_path, dataset_type='pulsedb'):
    """Save extracted features to HDF5 file in format compatible with DataLoadPulseDB or DataLoad_UCI
    
    Args:
        features (dict): Dictionary containing extracted features and data
        output_path (str): Path to save the HDF5 file
        dataset_type (str): Type of dataset ('pulsedb' or 'uci')
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nSaving features to {output_path}")
    with h5py.File(output_path, 'w') as f:
        if dataset_type == 'pulsedb':
            # Save large arrays
            safe_create_dataset(f, 'waveforms_raw', features['waveforms_raw'])
            safe_create_dataset(f, 'waveforms_local_minmax', features['waveforms_local_minmax'])
            safe_create_dataset(f, 'waveforms_minmax_zc', features['waveforms_minmax_zc'])
            safe_create_dataset(f, 'abp_global_minmax', features['abp_global_minmax'])
            
            # Save small arrays
            safe_create_dataset(f, 'bp_raw', features['bp_raw'])
            safe_create_dataset(f, 'demographics', features['demographics'])
            safe_create_dataset(f, 'bp_global_minmax', features['bp_global_minmax'])
            
            # Save subject IDs as fixed-length ASCII strings
            subject_ids = np.array([str(sid).encode('ascii') for sid in features['subject_ids']], dtype='S9')
            safe_create_dataset(f, 'subject_ids', subject_ids)
            
            # Save optional arrays if available
            if 'heart_rate' in features:
                safe_create_dataset(f, 'heart_rate', features['heart_rate'])
            if 'pulse_rate' in features:
                safe_create_dataset(f, 'pulse_rate', features['pulse_rate'])
            
            # Save extracted features in separate groups
            if 'ecg_features' in features:
                ecg_group = f.create_group('ecg_features')
                for i, ecg_feat in enumerate(features['ecg_features']):
                    sample_group = ecg_group.create_group(f'sample_{i}')
                    safe_create_dataset(sample_group, 'peak_locations', ecg_feat['peak_locations'])
                    safe_create_dataset(sample_group, 'qt_intervals', ecg_feat['qt_intervals'])
                    safe_create_dataset(sample_group, 'mean_ecg_quality', ecg_feat['mean_ecg_quality'])
            
            if 'ppg_features' in features:
                ppg_group = f.create_group('ppg_features')
                for i, ppg_feat in enumerate(features['ppg_features']):
                    sample_group = ppg_group.create_group(f'sample_{i}')
                    safe_create_dataset(sample_group, 'Asp_deltaT', ppg_feat['Asp_deltaT'])
                    safe_create_dataset(sample_group, 'IPR', ppg_feat['IPR'])
            
            # Add metadata
            f.attrs['creation_date'] = np.string_(datetime.now().isoformat())
            f.attrs['size'] = len(features['subject_ids'])
            
        else:  # uci dataset
            # Create test group
            test_group = f.create_group('test')
            
            # Save data in test group
            safe_create_dataset(test_group, 'waveforms_raw', features['waveforms_raw'])
            safe_create_dataset(test_group, 'waveforms_local_minmax', features['waveforms_local_minmax'])
            safe_create_dataset(test_group, 'waveforms_minmax_zc', features['waveforms_minmax_zc'])
            safe_create_dataset(test_group, 'bp_raw', features['bp_raw'])
            safe_create_dataset(test_group, 'bp_global_minmax', features['bp_global_minmax'])
            safe_create_dataset(test_group, 'abp_global_minmax', features['abp_global_minmax'])
            safe_create_dataset(test_group, 'abp_raw', features['abp_raw'])
            
            # Save rate data if available
            if 'heart_rate' in features:
                safe_create_dataset(test_group, 'heart_rate', features['heart_rate'])
            if 'pulse_rate' in features:
                safe_create_dataset(test_group, 'pulse_rate', features['pulse_rate'])
            
            # Save ECG features if available
            if 'ecg_features' in features:
                ecg_group = test_group.create_group('ecg_features')
                for i, ecg_feat in enumerate(features['ecg_features']):
                    sample_group = ecg_group.create_group(f'sample_{i}')
                    safe_create_dataset(sample_group, 'peak_locations', ecg_feat['peak_locations'])
                    safe_create_dataset(sample_group, 'qt_intervals', ecg_feat['qt_intervals'])
                    safe_create_dataset(sample_group, 'mean_ecg_quality', ecg_feat['mean_ecg_quality'])
            
            # Save PPG features if available
            if 'ppg_features' in features:
                ppg_group = test_group.create_group('ppg_features')
                for i, ppg_feat in enumerate(features['ppg_features']):
                    sample_group = ppg_group.create_group(f'sample_{i}')
                    safe_create_dataset(sample_group, 'Asp_deltaT', ppg_feat['Asp_deltaT'])
                    safe_create_dataset(sample_group, 'IPR', ppg_feat['IPR'])
    
    print("Features saved successfully!")

def main():
    """Main function to extract features from test set waveforms"""
    parser = argparse.ArgumentParser(description='Extract features from test set waveforms')
    parser.add_argument('--output_dir', type=str, default='./datasets',
                      help='Directory to save extracted features')
    parser.add_argument('--sampling_rate', type=int, default=125,
                      help='Sampling rate of the signals in Hz')
    parser.add_argument('--dataset', type=str, choices=['pulsedb', 'uci'], default='pulsedb',
                      help='Dataset to use for feature extraction (pulsedb or uci)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config instance based on dataset choice
    if args.dataset == 'pulsedb':
        config = PulseDBBaseConfig()
        sampling_rate = args.sampling_rate
        # Create output filename using test_sample_file
        output_file = os.path.join(args.output_dir, config.dataset_path, 'Features', f"{config.test_sample_file}_Features.h5")
    else:  # uci
        config = UCIBaseConfig()
        sampling_rate = args.sampling_rate
        # Create output filename using UCI format
        output_file = os.path.join(args.output_dir, config.dataset_path, 'Features', f"{config.sample_file}_Features.h5")
    
    # Extract features
    print(f"\nExtracting features from {args.dataset.upper()} test set...")
    features = config.extract_waveform_features_method(sampling_rate=sampling_rate)
    
    # Save features
    save_features_to_h5(features, output_file, dataset_type=args.dataset)
    
    # Print summary
    print("\nFeature Extraction Summary:")
    print(f"Features saved to: {output_file}")
    print(f"Total samples processed: {len(features['ppg_features'])}")
    
    if len(features['ppg_features']) > 0:
        print("\nSample PPG Features:")
        sample_ppg = features['ppg_features'][0]
        print(f"- Number of features: {len(sample_ppg)}")
        print(f"- Features available: {list(sample_ppg.keys())}")

    if len(features['ecg_features']) > 0:
        print("\nSample ECG Features:")
        sample_ecg = features['ecg_features'][0]
        print(f"- Number of QT intervals: {len(sample_ecg['qt_intervals'])}")

if __name__ == "__main__":
    main()
