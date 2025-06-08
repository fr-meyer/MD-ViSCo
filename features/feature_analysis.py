# Standard library imports
import os
import argparse
import csv

# Third-party imports
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Local imports
from dataload.DataLoadPulseDB import PulseDBDataset
from dataload.DataLoadUCI import RecordSamplesDatasetUC

# Set environment variable
os.environ['OMP_NUM_THREADS'] = '2'

def calculate_qtc_bazett(qt, rr):
    """
    Calculate QTc using Bazett's formula.
    
    Parameters:
        qt (float or array): QT interval(s) in seconds
        rr (float or array): RR interval(s) in seconds

    Returns:
        float or array: QTc interval(s) in seconds
    """
    return qt / np.sqrt(rr)

def compute_bpm(r_peaks, sampling_rate=125):
    """
    Calculate BPM from R-peak locations
    
    Args:
        r_peaks (np.ndarray): Array of sample indices of R-peaks
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        tuple: (average_bpm, instantaneous_bpm, rr_intervals)
    """
    if len(r_peaks) < 2:
        return None, None, None
        
    rr_intervals = np.diff(r_peaks) / sampling_rate  # seconds
    bpm_per_beat = 60 / rr_intervals  # instantaneous BPM
    bpm_avg = np.mean(bpm_per_beat)  # average BPM over signal
    return bpm_avg, bpm_per_beat, rr_intervals

def load_dataset(sample_file: str, dataset_type: str = 'PulseDB', dataset: int = 2, sample_length: int = 1280):
    """
    Load ground truth dataset using existing dataset loaders
    
    Args:
        sample_file (str): Path to the HDF5 file containing features (without .h5 extension for PulseDB)
        dataset_type (str): Type of dataset ('PulseDB' or 'UCI')
        dataset (int): Dataset split (0: train, 1: val, 2: test)
        sample_length (int): Length of each sample
    
    Returns:
        Dataset: Loaded dataset instance
    """
    if dataset_type.lower() == 'pulsedb':
        # For PulseDB, remove .h5 extension if present
        if sample_file.endswith('.h5'):
            sample_file = sample_file[:-3]
            
        return PulseDBDataset(
            sample_file=sample_file,
            sample_length=sample_length
        )
    else:  # UCI dataset
        return RecordSamplesDatasetUC(
            sample_file=sample_file,
            dataset=dataset,
            sample_freq=125, 
            sample_length=sample_length
        )

def construct_reconstructed_path(base_path: str, dataset_type: str, model_name: str, seed: int, direction: str) -> str:
    """
    Construct the path to reconstructed features file
    
    Args:
        base_path (str): Base path to features directory
        dataset_type (str): Type of dataset ('PulseDB' or 'UCI')
        model_name (str): Name of the model (e.g., 'mdvisco')
        seed (int): Random seed number
        direction (str): Direction of reconstruction (e.g., 'PPG2ECG')
    
    Returns:
        str: Full path to the reconstructed features file
    """
    # Map model name using METHOD_MAP
    mapped_model_name = METHOD_MAP.get(model_name.lower(), model_name)
    
    # Try uppercase first
    upper_path = os.path.join(
        base_path,
        dataset_type,
        mapped_model_name,
        f"seed_{seed}",
        f"features_{direction.upper()}.h5"
    )
    
    # If uppercase path doesn't exist, try lowercase
    if not os.path.exists(upper_path):
        lower_path = os.path.join(
            base_path,
            dataset_type,
            mapped_model_name,
            f"seed_{seed}",
            f"features_{direction.lower()}.h5"
        )
        if os.path.exists(lower_path):
            return lower_path
        # If neither exists, return the uppercase path (will be handled by the caller)
    
    return upper_path

# Define mapping dictionaries
METHOD_MAP = {
    'patchtst': 'PatchTST',
    'p2ewgan': 'P2E-WGAN',
    'P2EWGAN': 'P2E-WGAN',
    'ppg2abp': 'PPG2ABP',
    'nabnet': 'NABNet',
    # 'mdvisco': 'our',
    'mdvisco': 'MD-ViSCo',
}

COLUMN_MAP = {
    'Dataset': 'dataset',
    'Method': 'method',
    'Direction': 'direction',
    'Seed': 'seed',
    'sample_index': 'sample_idx',
}

Dataset_MAP = {
    'pulsedb': 'PulseDB',    
    'uci': 'UCI',
}

def save_sample_features_to_csv(ground_truth_dataset, reconstructed_file: str, direction: str, dataset_type: str, output_csv: str, seed: int, model_name: str):
    """
    Save individual sample features to a CSV file
    
    Args:
        ground_truth_dataset: Dataset containing ground truth features
        reconstructed_file (str): Path to the HDF5 file containing reconstructed features
        direction (str): Direction of reconstruction ('PPG2ECG', 'ECG2PPG', 'ABP2PPG', 'ECG2PPG')
        dataset_type (str): Type of dataset ('PulseDB' or 'UCI')
        output_csv (str): Path to save the CSV file
        seed (int): Random seed number used for the model
        model_name (str): Name of the model
    """
    # Determine if we're analyzing ECG or PPG features based on the target signal
    is_ecg = direction.endswith('ECG')  # True for PPG2ECG, False for ECG2PPG, ABP2PPG
    
    # Open the reconstructed features file
    with h5py.File(reconstructed_file, 'r') as f:
        # Prepare CSV headers
        headers = ['sample_idx', 'dataset', 'method', 'direction', 'seed', 'unit']
        if is_ecg:
            headers.extend([
                'qt_intervals_gt', 'qt_intervals_rec', 'qt_intervals_mae',
                'bpm_avg_gt', 'bpm_avg_rec', 'bpm_avg_mae',
                'qtc_gt', 'qtc_rec', 'qtc_mae'
            ])
        else:
            headers.extend([
                'Asp_deltaT_gt', 'Asp_deltaT_rec', 'Asp_deltaT_mae',
                'IPR_gt', 'IPR_rec', 'IPR_mae'
            ])
        
        # Open CSV file for writing
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            # Iterate through ground truth dataset
            for i in tqdm(range(len(ground_truth_dataset)), desc=f"Saving {direction} features"):
                sample_key = f'sample_{i}'
                if sample_key not in f:
                    continue
                
                # Access the appropriate feature group based on target signal
                feature_group = 'ecg_features' if is_ecg else 'ppg_features'
                try:
                    reconstructed_sample = f[sample_key][feature_group]
                except KeyError:
                    reconstructed_sample = {}
                
                ground_truth_sample = ground_truth_dataset[i]
                
                # Skip if ground truth sample is None
                if ground_truth_sample is None:
                    continue
                
                # Get ground truth features from the correct index
                if dataset_type == 'PulseDB':
                    gt_features = ground_truth_sample[12] if is_ecg else ground_truth_sample[13]
                else:  # UCI dataset
                    gt_features = ground_truth_sample[8] if is_ecg else ground_truth_sample[9]
                
                # Skip if ground truth features are None or empty
                if gt_features is None or (isinstance(gt_features, torch.Tensor) and gt_features.numel() == 0):
                    continue
                
                # Apply mappings
                mapped_dataset = Dataset_MAP.get(dataset_type.lower(), dataset_type)
                mapped_method = METHOD_MAP.get(model_name.lower(), model_name)
                
                # Prepare row data with mapped values
                row_data = [i, mapped_dataset, mapped_method, direction, seed, 's']
                
                if is_ecg:
                    # QT intervals
                    qt_gt = None
                    if isinstance(gt_features, dict) and 'qt_intervals' in gt_features and len(gt_features['qt_intervals']) > 0:
                        qt_gt = gt_features['qt_intervals'].mean().item()
                    qt_rec = np.mean(reconstructed_sample['qt_intervals'][:]) if 'qt_intervals' in reconstructed_sample else None
                    qt_mae = abs(qt_gt - qt_rec) if qt_gt is not None and qt_rec is not None else None
                    
                    # BPM
                    bpm_gt = None
                    bpm_rec = None
                    bpm_mae = None
                    gt_rr_intervals = None
                    rec_rr_intervals = None
                    
                    # Calculate BPM for ground truth
                    if isinstance(gt_features, dict) and 'peak_locations' in gt_features and 'r_wave' in gt_features['peak_locations'] and 'indices' in gt_features['peak_locations']['r_wave']:
                        gt_r_peaks = gt_features['peak_locations']['r_wave']['indices']
                        bpm_gt, _, gt_rr_intervals = compute_bpm(gt_r_peaks)
                    
                    # Calculate BPM for reconstructed
                    if 'peak_locations' in reconstructed_sample and 'r_wave' in reconstructed_sample['peak_locations'] and 'indices' in reconstructed_sample['peak_locations']['r_wave']:
                        rec_r_peaks = reconstructed_sample['peak_locations']['r_wave']['indices'][:]
                        bpm_rec, _, rec_rr_intervals = compute_bpm(rec_r_peaks)
                    
                    # Calculate BPM MAE if both values are available
                    if bpm_gt is not None and bpm_rec is not None:
                        bpm_mae = abs(bpm_gt - bpm_rec)
                    
                    # QTc
                    qtc_gt = None
                    qtc_rec = None
                    qtc_mae = None
                    
                    if qt_gt is not None and gt_rr_intervals is not None:
                        gt_rr = np.mean(gt_rr_intervals)  # Use mean RR interval
                        qtc_gt = calculate_qtc_bazett(qt_gt, gt_rr)
                        
                        if qt_rec is not None and rec_rr_intervals is not None:
                            rec_rr = np.mean(rec_rr_intervals)  # Use mean RR interval
                            qtc_rec = calculate_qtc_bazett(qt_rec, rec_rr)
                            
                            if qtc_gt is not None and qtc_rec is not None:
                                qtc_mae = abs(qtc_gt - qtc_rec)
                    
                    row_data.extend([
                        qt_gt, qt_rec, qt_mae,
                        bpm_gt, bpm_rec, bpm_mae,
                        qtc_gt, qtc_rec, qtc_mae
                    ])
                else:
                    # PPG features
                    ppg_features = ['Asp_deltaT', 'IPR']
                    ppg_data = []
                    
                    for feature in ppg_features:
                        # Ground truth
                        gt_val = None
                        if isinstance(gt_features, dict) and feature in gt_features:
                            gt_val = gt_features[feature].item() if isinstance(gt_features[feature], torch.Tensor) else gt_features[feature]
                        # Reconstructed
                        rec_val = reconstructed_sample[feature][()][0] if feature in reconstructed_sample else None
                        # MAE
                        mae_val = abs(gt_val - rec_val) if gt_val is not None and rec_val is not None else None
                        
                        ppg_data.extend([gt_val, rec_val, mae_val])
                    
                    row_data.extend(ppg_data)
                
                writer.writerow(row_data)

def load_saved_features(csv_path: str):
    """
    Load and verify the saved features from CSV file
    
    Args:
        csv_path (str): Path to the CSV file containing saved features
    
    Returns:
        dict: Dictionary containing the loaded features and metadata
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Print basic information
        print("\nLoaded CSV file information:")
        print(f"Number of samples: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Print summary statistics for each feature
        print("\nFeature statistics:")
        for col in df.columns:
            if col not in ['sample_idx', 'dataset', 'method', 'direction', 'seed', 'unit']:
                print(f"\n{col}:")
                print(f"  Mean: {df[col].mean():.4f}")
                print(f"  Std:  {df[col].std():.4f}")
                print(f"  Min:  {df[col].min():.4f}")
                print(f"  Max:  {df[col].max():.4f}")
        
        return df
    
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None

def main():
    """Main function to load and test dataset"""
    parser = argparse.ArgumentParser(description='Load and test PulseDB or UCI dataset')
    parser.add_argument('--dataset_type', type=str, choices=['PulseDB', 'UCI'], required=True,
                      help='Type of dataset to load (PulseDB or UCI)')
    parser.add_argument('--sample_file', type=str, required=True,
                      help='Path to the ground truth dataset file (without .h5 extension for PulseDB)')
    parser.add_argument('--features_base_path', type=str, default='./results/features_extraction',
                      help='Base path to features directory')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model (e.g., MD-ViSCo)')
    parser.add_argument('--seed', type=int, required=True,
                      help='Random seed number')
    parser.add_argument('--direction', type=str, required=True, choices=['PPG2ECG', 'ECG2PPG', 'ABP2PPG', 'ABP2ECG'],
                       help='Direction of reconstruction')
    parser.add_argument('--dataset', type=int, default=2, choices=[0, 1, 2],
                      help='Dataset split (0: train, 1: val, 2: test)')
    parser.add_argument('--sample_length', type=int, default=1280,
                      help='Length of each sample')
    parser.add_argument('--project_name', type=str, default=None,
                      help='Name of the wandb project')
    parser.add_argument('--output_csv', type=str, default='./results/features_results',
                      help='Base directory to save the CSV file with individual sample features')
    
    args = parser.parse_args()
    
    # Construct path to reconstructed features
    reconstructed_file = construct_reconstructed_path(
        args.features_base_path,
        args.dataset_type,
        args.model_name,
        args.seed,
        args.direction
    )
    
    # Generate CSV filename based on parameters
    csv_filename = f"{args.model_name}_{args.direction}_{args.dataset_type}_seed{args.seed}.csv"
    output_csv_path = os.path.join(args.output_csv, csv_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_csv, exist_ok=True)
    
    # Check if files exist
    file_to_check = args.sample_file if args.dataset_type == 'UCI' else (args.sample_file if args.sample_file.endswith('.h5') else args.sample_file + '.h5')
    if not os.path.exists(file_to_check):
        print(f"Error: Ground truth file {file_to_check} does not exist")
        return
    
    if not os.path.exists(reconstructed_file):
        print(f"Error: Reconstructed features file {reconstructed_file} does not exist")
        return
    
    # Load datasets
    try:
        # Load ground truth dataset
        ground_truth_dataset = load_dataset(
            sample_file=args.sample_file,
            dataset_type=args.dataset_type,
            dataset=args.dataset,
            sample_length=args.sample_length
        )
        
        print(f"\nSuccessfully loaded {args.dataset_type} dataset")
        print(f"Ground truth dataset size: {len(ground_truth_dataset)}")
        
        # Save individual sample features to CSV
        save_sample_features_to_csv(
            ground_truth_dataset,
            reconstructed_file,
            args.direction,
            args.dataset_type,
            output_csv_path,
            args.seed,  # Pass the seed from arguments
            args.model_name  # Pass model_name directly
        )
        print(f"\nSaved individual sample features to {output_csv_path}")
        
        # Load and verify the saved CSV file
        print("\nVerifying saved features...")
        loaded_features = load_saved_features(output_csv_path)
        
        if loaded_features is not None:
            print("\nSuccessfully verified saved features")
        
        # Print metadata from reconstructed features if available
        with h5py.File(reconstructed_file, 'r') as f:
            if 'direction' in f.attrs:
                print(f"Reconstruction direction: {f.attrs['direction']}")
            if 'seed' in f.attrs:
                print(f"Model seed: {f.attrs['seed']}")
            if 'model_type' in f.attrs:
                print(f"Model type: {f.attrs['model_type']}")
        
        # Load and print first sample from ground truth
        sample = ground_truth_dataset[0]
        if args.dataset_type == 'PulseDB':
            print("\nPulseDB sample structure:")
            print(f"Subject ID: {sample[0]}")
            print(f"Sample index: {sample[1]}")
            print(f"Waveforms shape: {sample[2].shape}")
            print(f"BP values: {sample[3]}")
            print(f"Demographics: {sample[4]}")
        else:  # UCI
            print("\nUCI sample structure:")
            print(f"Waveforms raw shape: {sample[0].shape}")
            print(f"ABP global minmax shape: {sample[1].shape}")
            print(f"BP raw shape: {sample[2].shape}")
            print(f"Waveforms minmax zc shape: {sample[3].shape}")
            print(f"Waveforms local minmax shape: {sample[4].shape}")
            print(f"BP global minmax shape: {sample[5].shape}")
            if len(sample) > 7:  # Test set has additional heart rate and pulse rate
                print(f"Heart rate: {sample[6]}")
                print(f"Pulse rate: {sample[7]}")
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()
