# Standard library imports
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Third-party library imports
import h5py
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import torch

# Local imports
from utils.ecg_features import (
    calculate_qt_intervals,
    extract_ecg_features,
    get_peak_locations
)
from utils.ppg_features import extract_ppg_features
from utils.utils_preprocessing import (
    Global_Min_Max_Norm,
    check_file_exists,
    safe_create_dataset,
    calculate_MAP
)
from utils.wandb_utils import (
    log_aami_style_result,
    log_approx_result,
    log_bhs_style_result,
    log_clinical_tasks_result
)

# APPROXIMATION MODEL TESTING UTILS
def calculate_reconstruction_losses(
    recons: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_fn: torch.nn.Module
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[float]]]:
    """Calculate reconstruction losses for all signal direction pairs.
    
    Args:
        recons: Dictionary of reconstructions (on device_output)
        targets: Dictionary of target tensors (on device_output)
        loss_fn: Loss function module
        
    Returns:
        Tuple containing:
        - Dictionary of loss tensors (on device_output)
        - Dictionary of batch loss values for tracking
    """
    # Calculate losses - both recons and targets are on device_output
    losses = {}
    batch_losses = {}
    
    # Process only the pairs that exist in both recons and targets
    for src in recons.keys():
        # Extract target key from source (e.g., 'ecg2ppg' -> 'ppg')
        tgt = src.split('2')[1]
        
        if tgt in targets:
            # Calculate mean loss across dimensions except batch
            losses[src] = torch.mean(loss_fn(recons[src], targets[tgt]), dim=2)
            # Convert to numpy array for tracking
            batch_losses[f'{src}_loss'] = losses[src].cpu().detach().numpy()
        
    return losses, batch_losses

def print_reconstruction_results(best_loss: Dict[str, List[float]], args: dict) -> None:
    """Print formatted reconstruction results with mean and standard deviation and log to wandb.
    
    Args:
        best_loss: Dictionary containing lists of loss values
        dataset: Name of the dataset (default: "PulseDB")
        method: Name of the method used (default: "Ours")
        seed: Random seed used (default: 1)
    """
    # Group metrics by target domain
    domains = {'ECG': [], 'PPG': [], 'ABP': []}
    for metric in best_loss:
        # Only process metrics that end with '_loss' and have values
        if metric.endswith('_loss') and len(best_loss[metric]) > 0:
            mean = np.mean(best_loss[metric])
            std = np.std(best_loss[metric])
            
            # Extract target domain from metric name
            if '2' in metric:  # Ensure it's a reconstruction metric
                source, target = metric.split('2')
                target = target.split('_')[0].upper()
                if target in domains:
                    domains[target].append((metric, mean, std))
                    
                    # Log to wandb using log_approx_result
                    direction = f"{source.upper()}→{target}"
                    log_approx_result(
                        dataset=args.dataset,
                        method=args.model_name,
                        direction=direction,
                        metric="MAE",
                        value=mean,
                        std=std,
                        seed=args.seed,
                        args=args
                    )
    print("\nReconstruction Results - Mean ± Std:")
    print("="*50)
    # Print results grouped by domain
    for domain, results in domains.items():
        if results:  # Only print if there are results for this domain
            print(f"\n{domain} Reconstructions:")
            print("-"*30)
            for metric, mean, std in results:
                # Clean up metric name for display
                display_name = metric.replace('_loss', '').upper()
                print(f"{display_name}: {mean:.3f} ± {std:.3f}")


def calculate_pearson_correlations(
    recons: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    best_loss: Dict[str, List[float]]
) -> Dict[str, float]:
    """Calculate Pearson correlations between reconstructed waveforms and ground truth.
    
    Args:
        recons: Dictionary of reconstruction tensors
        targets: Dictionary of target tensors 
        best_loss: Dictionary to store correlation values
        
    Returns:
        Dictionary containing mean correlation values for each direction
    """
    mean_correlations = {}
    
    # Process only the pairs that exist in both recons and targets
    for src in recons.keys():
        # Extract target key from source (e.g., 'ecg2ppg' -> 'ppg')
        tgt = src.split('2')[1]
        
        if tgt in targets:
            # Convert source key to uppercase format for correlation key
            pair_name = src.lower()
            corr_key = f'pC_{pair_name}_loss'
            
            if corr_key not in best_loss:
                best_loss[corr_key] = []
                
            # Get reconstructed and target signals
            recon_signal = recons[src].squeeze()  # Shape: [batch_size, signal_length]
            target_signal = targets[tgt].squeeze()  # Shape: [batch_size, signal_length]
            
            # Calculate means
            recon_mean = recon_signal.mean(dim=1, keepdim=True)
            target_mean = target_signal.mean(dim=1, keepdim=True)
            
            # Calculate correlation using vectorized operations
            numerator = torch.sum((recon_signal - recon_mean) * (target_signal - target_mean), dim=1)
            denominator = torch.sqrt(
                torch.sum((recon_signal - recon_mean)**2, dim=1) * 
                torch.sum((target_signal - target_mean)**2, dim=1)
            )
            
            # Handle potential division by zero
            correlations = torch.where(
                denominator > 0,
                numerator / denominator,
                torch.zeros_like(numerator)
            )
            
            best_loss[corr_key].extend(correlations.cpu().detach().numpy())
            
            # Calculate mean correlation for this pair
            mean_correlations[pair_name] = np.mean(best_loss[corr_key])
    
    return mean_correlations

def print_correlation_results(best_loss: Dict[str, List[float]], args: dict) -> None:
    """Print formatted correlation results with mean and standard deviation and log to wandb.
    
    Args:
        best_loss: Dictionary containing correlation values
        args: Arguments dictionary containing wandb configuration
    """
    # Group metrics by target domain
    domains = {'ECG': [], 'PPG': [], 'ABP': []}
    for metric in best_loss:
        if metric.startswith('pC_') and len(best_loss[metric]) > 0:
            mean = np.mean(best_loss[metric])
            std = np.std(best_loss[metric])
            
            # Convert metric to lowercase for consistent comparison
            metric_lower = metric.lower()
            if 'ecg_loss' in metric_lower:
                domains['ECG'].append((metric, mean, std))
            elif 'ppg_loss' in metric_lower:
                domains['PPG'].append((metric, mean, std))
            elif 'abp_loss' in metric_lower:
                domains['ABP'].append((metric, mean, std))
            
            # Log to wandb using log_approx_result
            # Extract direction from metric (e.g., 'pC_ppg2ecg_loss' -> 'PPG→ECG')
            direction = metric.replace('pC_', '').replace('_loss', '').upper()
            if '2' in direction:
                source, target = direction.split('2')
                direction = f"{source}→{target}"
            
            log_approx_result(
                dataset=args.dataset,
                method=args.model_name,
                direction=direction,
                metric="PC",  # Pearson Correlation
                value=mean,
                std=std,
                seed=args.seed,
                args=args
            )
    print("\nPearson Correlation Results - Mean ± Std:")
    print("="*50)
    # Print results grouped by domain
    for domain, results in domains.items():
        if results:  # Only print domains that have results
            print(f"\n{domain} Correlations:")
            print("-"*30)
            for metric, mean, std in results:
                # Clean up metric name for display
                display_name = metric.replace('pC_', '').replace('_loss', '').upper()
                print(f"{display_name}: {mean:.3f} ± {std:.3f}")


def calculate_bpm_metrics(
    recons: Dict[str, torch.Tensor],
    target_bpm: Dict[str, torch.Tensor],  # Ground truth BPM values
    bpm_metrics: Dict[str, List[float]],
    sampling_rate: int = 125
) -> Dict[str, float]:
    """Calculate BPM/Heart Rate metrics for reconstructed signals.
    
    Args:
        recons: Dictionary of reconstruction tensors
        target_bpm: Dictionary containing ground truth BPM values for each domain
        bpm_metrics: Dictionary to store all BPM errors across batches
        sampling_rate: Sampling rate of the signals in Hz
        
    Returns:
        Dictionary containing current batch's mean BPM errors for progress bar display
    """
    valid_targets = {'ecg', 'ppg'}  # Only process these target domains
    current_batch_means = {}
    
    # Initialize success and total counts in bpm_metrics if they don't exist
    for src in recons.keys():
        tgt = src.split('2')[1]
        if tgt in valid_targets:
            if f'success_count_{src}' not in bpm_metrics:
                bpm_metrics[f'success_count_{src}'] = 0
            if f'total_count_{src}' not in bpm_metrics:
                bpm_metrics[f'total_count_{src}'] = 0
            # Initialize lists for storing rates and indices
            if f'recon_rates_{src}' not in bpm_metrics:
                bpm_metrics[f'recon_rates_{src}'] = []
            if f'target_rates_{src}' not in bpm_metrics:
                bpm_metrics[f'target_rates_{src}'] = []
            if f'sample_indices_{src}' not in bpm_metrics:
                bpm_metrics[f'sample_indices_{src}'] = []
    
    for src in recons.keys():
        # Extract target domain from source key (e.g., 'ppg2ecg' -> 'ecg')
        tgt = src.split('2')[1]
        
        # Skip if target is not ECG or PPG or if we don't have ground truth BPM
        if tgt not in valid_targets or tgt not in target_bpm:
            continue
            
        # Initialize metric keys for both normal and filtered errors
        normal_metric_key = f'bpm_{src}'
        filtered_metric_key = f'bpm_filtered_{src}'
        
        if normal_metric_key not in bpm_metrics:
            bpm_metrics[normal_metric_key] = []
        if filtered_metric_key not in bpm_metrics:
            bpm_metrics[filtered_metric_key] = []
            
        batch_errors = []  # Store current batch errors for progress bar
        filtered_batch_errors = []  # Store filtered batch errors
        
        recon_signal = recons[src]
        ground_truth_bpm = target_bpm[tgt]
        
        for i in range(recon_signal.size(0)):
            bpm_metrics[f'total_count_{src}'] += 1  # Increment total count for this direction
            current_index = bpm_metrics[f'total_count_{src}'] - 1  # Get current sample index
            bpm_metrics[f'sample_indices_{src}'].append(current_index)
            
            try:
                # Calculate BPM from reconstructed signal
                if tgt == 'ecg':
                    signals_recon, _ = nk.ecg_process(
                        recon_signal[i].cpu().numpy().flatten(),
                        sampling_rate=sampling_rate
                    )
                    rate_key = 'ECG_Rate'
                elif tgt == 'ppg':  # ppg
                    signals_recon, _ = nk.ppg_process(
                        recon_signal[i].cpu().numpy().flatten(),
                        sampling_rate=sampling_rate
                    )
                    rate_key = 'PPG_Rate'
                
                # Calculate mean rate from reconstruction
                recon_rate = np.mean(signals_recon[rate_key])
                
                # Get ground truth BPM
                target_rate = ground_truth_bpm[i].item()
                
                # Store both rates
                bpm_metrics[f'recon_rates_{src}'].append(recon_rate)
                bpm_metrics[f'target_rates_{src}'].append(target_rate)
                
                # Calculate absolute error for both normal and filtered cases
                rate_error = abs(recon_rate - target_rate)
                bpm_metrics[normal_metric_key].append(rate_error)
                bpm_metrics[filtered_metric_key].append(rate_error)
                batch_errors.append(rate_error)
                filtered_batch_errors.append(rate_error)

                bpm_metrics[f'success_count_{src}'] += 1  # Increment success count for this direction
                
            except Exception as e:
                # If BPM extraction fails, only add to normal error calculation
                print(f"Warning: Failed to calculate BPM for {src} sample {i}: {str(e)}")
                target_rate = ground_truth_bpm[i].item()
                # Store rates even for failed extractions
                bpm_metrics[f'recon_rates_{src}'].append(0)  # Use 0 for failed extraction
                bpm_metrics[f'target_rates_{src}'].append(target_rate)
                
                rate_error = abs(0 - target_rate)  # Use 0 as failed extraction value
                bpm_metrics[normal_metric_key].append(rate_error)
                batch_errors.append(rate_error)
        
        # Calculate mean of errors for current batch (for progress bar)
        current_batch_means[src] = np.mean(batch_errors) if batch_errors else 0
        current_batch_means[f'filtered_{src}'] = np.mean(filtered_batch_errors) if filtered_batch_errors else 0
    
    # Calculate and store success rates for each direction
    for src in recons.keys():
        tgt = src.split('2')[1]
        if tgt in valid_targets and bpm_metrics[f'total_count_{src}'] > 0:
            success_rate = (bpm_metrics[f'success_count_{src}'] / bpm_metrics[f'total_count_{src}']) * 100
            bpm_metrics[f'success_rate_{src}'] = success_rate

    return current_batch_means

def print_bpm_results(best_loss: Dict[str, List[float]], args: dict) -> None:
    """Print formatted BPM/Heart Rate results with mean and standard deviation and log to wandb.
    
    Args:
        best_loss: Dictionary containing BPM error values
        args: Arguments dictionary containing wandb configuration
    """
    # Group metrics by direction
    directions = {}
    for metric in best_loss:
        if metric.startswith('bpm_') and len(best_loss[metric]) > 0:
            # Extract direction from metric (e.g., 'bpm_ppg2ecg' -> 'ppg2ecg')
            direction = metric.replace('bpm_', '').replace('filtered_', '')
            if direction not in directions:
                directions[direction] = []
            directions[direction].append((metric, best_loss[metric]))
    
    print("\nBPM/Heart Rate Results - Mean ± Std:")
    print("="*50)
    
    # Print results grouped by direction
    for direction, metrics in directions.items():
        print(f"\n{direction.upper()} Results:")
        print("-"*30)
        
        # Get success rate for this direction
        success_rate = best_loss.get(f'success_rate_{direction}', None)
        
        # Print normal BPM errors
        print("\nNormal BPM Errors (including failed extractions):")
        for metric, values in metrics:
            if not metric.startswith('bpm_filtered_'):
                mean = np.mean(values)
                std = np.std(values)
                print(f"MAE: {mean:.2f} ± {std:.2f}")
        
        # Print filtered BPM errors
        print("\nFiltered BPM Errors (only successful extractions):")
        for metric, values in metrics:
            if metric.startswith('bpm_filtered_'):
                mean = np.mean(values)
                std = np.std(values)
                print(f"MAE: {mean:.2f} ± {std:.2f}")
        
        # Print success rate if available
        if success_rate is not None:
            print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        # Log to wandb
        if '2' in direction:
            source, target = direction.split('2')
            direction_display = f"{source.upper()}→{target.upper()}"
            
            # Log normal BPM errors
            for metric, values in metrics:
                if not metric.startswith('bpm_filtered_'):
                    mean = np.mean(values)
                    std = np.std(values)
                    log_clinical_tasks_result(
                        dataset=args.dataset,
                        method=args.model_name,
                        direction=direction_display,
                        metric="MAE",
                        value=mean,
                        std=std,
                        seed=args.seed,
                        args=args,
                        filtered=False
                    )
            
            # Log filtered BPM errors
            for metric, values in metrics:
                if metric.startswith('bpm_filtered_'):
                    mean = np.mean(values)
                    std = np.std(values)
                    log_clinical_tasks_result(
                        dataset=args.dataset,
                        method=args.model_name,
                        direction=direction_display,
                        metric="MAE",
                        value=mean,
                        std=std,
                        seed=args.seed,
                        args=args,
                        filtered=True
                    )
            
            # Log success rate
            if success_rate is not None:
                log_clinical_tasks_result(
                    dataset=args.dataset,
                    method=args.model_name,
                    direction=direction_display,
                    metric="Success Rate",
                    value=success_rate,
                    std=0.0,  # No std for success rate
                    seed=args.seed,
                    args=args,
                    filtered=False
                )
            
            if args is not None and args.project_name is not None:
                save_bpm_results_to_csv(
                    bpm_metrics=best_loss,
                    dataset=args.dataset,
                    method=args.model_name if args.model_name is not None else "Ours",
                    direction=direction_display,
                    seed=args.seed,
                    output_path='./results/bpm_results'
                )

def extract_waveform_features(ecg_signal: np.ndarray, ppg_signal: np.ndarray, sampling_rate: int = 125) -> dict:
    """Extract features from ECG and PPG waveforms in PulseDB format.
    
    Args:
        ecg_signal (np.ndarray): ECG signal array
        ppg_signal (np.ndarray): PPG signal array
        sampling_rate (int): Sampling rate of the signals in Hz (default: 125)
        
    Returns:
        dict: Dictionary containing features in PulseDB format:
            - ecg_features: Dictionary containing:
                - peak_locations: Dictionary of peak locations
                - qt_intervals: Array of QT interval durations
                - mean_ecg_quality: Mean ECG signal quality
            - ppg_features: Dictionary containing:
                - Asp_deltaT: Mean Asp/deltaT ratio
                - IPR: Mean IPR ratio
    """
    features = {}
    
    # Process ECG signal if it's not empty
    if ecg_signal is not None and len(ecg_signal) > 0:
        try:
            # Extract basic ECG features
            ecg_signals, ecg_info = extract_ecg_features(ecg_signal, sampling_rate)
            peak_locations = get_peak_locations(ecg_signals)
            
            # Calculate ECG intervals and morphology
            qt_intervals = calculate_qt_intervals(peak_locations, sampling_rate)
            
            # Store ECG features in PulseDB format
            features['ecg_features'] = {
                'peak_locations': peak_locations,
                'qt_intervals': qt_intervals['durations'],  # Store only durations
                'mean_ecg_quality': np.mean(ecg_signals['ECG_Quality'])
            }
        except Exception as e:
            print(f"Warning: Failed to extract ECG features: {str(e)}")
            features['ecg_features'] = None
    else:
        features['ecg_features'] = None
    
    # Process PPG signal if it's not empty
    if ppg_signal is not None and len(ppg_signal) > 0:
        try:
            # Extract PPG features
            ppg_results = extract_ppg_features(ppg_signal, sampling_rate)
            
            features['ppg_features'] = {
                'Asp_deltaT': ppg_results['Asp_deltaT'],
                'IPR': ppg_results['IPR']
            }

        except Exception as e:
            print(f"Warning: Failed to extract PPG features: {str(e)}")
            features['ppg_features'] = None
    else:
        features['ppg_features'] = None
    
    return features

def save_waveform_features(features_dict: dict, dataset: str, model_name: str, seed: int, direction: str, output_dir: str = '.', overwrite: bool = False) -> None:
    """Save extracted waveform features to a file for multiple samples.
    
    Args:
        features_dict (dict): Dictionary containing features for multiple samples:
            {
                sample_idx: {
                    'ecg_features': {
                        'peak_locations': {...},
                        'qt_intervals': [...],
                        'mean_ecg_quality': float
                    },
                    'ppg_features': {
                        'Asp_deltaT': float,
                        'IPR': float
                    }
                },
                ...
            }
        dataset (str): Name of the dataset
        model_name (str): Name of the model used
        seed (int): Random seed number
        direction (str): Direction of the transformation (e.g., 'PPG2ECG')
        output_dir (str): Base directory for saving features (default: '.')
        overwrite (bool): If True, overwrite existing file. If False, raise error if file exists.
        
    Raises:
        FileExistsError: If file already exists and overwrite is False
        ValueError: If PPG features structure is invalid
    """
    # Create features directory if it doesn't exist
    features_dir = os.path.join('./results/features_extraction', dataset, model_name, f'seed_{seed}')
    os.makedirs(features_dir, exist_ok=True)
    
    # Create filename without timestamp
    filename = f'features_{direction}.h5'
    filepath = os.path.join(features_dir, filename)
    
    # Check if file exists using the new method
    check_file_exists(filepath, overwrite)
    
    with h5py.File(filepath, 'w') as f:
        # Save metadata
        f.attrs['dataset'] = dataset
        f.attrs['model_name'] = model_name
        f.attrs['seed'] = seed
        f.attrs['direction'] = direction
        f.attrs['num_samples'] = len(features_dict)
        
        # Create groups for each sample
        for sample_idx, features in features_dict.items():
            print(f"\nProcessing sample {sample_idx}")
            sample_group = f.create_group(f'sample_{sample_idx}')
            
            # Save ECG features if they exist
            if features.get('ecg_features') is not None:
                print(f"  Saving ECG features for sample {sample_idx}")
                ecg_group = sample_group.create_group('ecg_features')
                for key, value in features['ecg_features'].items():
                    print(f"    Processing ECG key: {key}")
                    safe_create_dataset(ecg_group, key, value)
            
            # Save PPG features if they exist
            if features.get('ppg_features') is not None:
                print(f"  Saving PPG features for sample {sample_idx}")
                ppg_group = sample_group.create_group('ppg_features')
                
                # Validate PPG features structure
                expected_ppg_keys = {'Asp_deltaT', 'IPR'}
                actual_ppg_keys = set(features['ppg_features'].keys())
                
                if not actual_ppg_keys.issubset(expected_ppg_keys):
                    raise ValueError(f"Invalid PPG features structure. Expected keys: {expected_ppg_keys}, got: {actual_ppg_keys}")
                
                # Save each PPG biomarker
                for key in expected_ppg_keys:
                    if key in features['ppg_features']:
                        print(f"    Processing PPG key: {key}")
                        value = features['ppg_features'][key]
                        # Convert scalar values to numpy array with shape (1,)
                        if np.isscalar(value):
                            value = np.array([value])
                        safe_create_dataset(ppg_group, key, value)
        print(f"Closing file in {filepath}")
    
    print(f"Saved features for {len(features_dict)} samples to {filepath}")

def save_bpm_results_to_csv(bpm_metrics: Dict[str, List[float]], dataset: str, method: str, direction: str, seed: int, output_path: str = None) -> None:
    """Save BPM results to a CSV file in a standardized format.
    
    Args:
        bpm_metrics (Dict[str, List[float]]): Dictionary containing BPM metrics and values
        dataset (str): Name of the dataset (e.g., 'PulseDB', 'UCI')
        method (str): Name of the method used (e.g., 'NABNet', 'P2EWGAN', 'PatchTST', 'our')
        direction (str): Direction of transformation (e.g., 'PPG→ECG', 'ECG→PPG')
        seed (int): Random seed used
        output_path (str, optional): Path to save the CSV file. If None, saves to current directory.
    """
    # Create DataFrame with required columns
    data = {
        'Dataset': [],
        'Method': [],
        'Direction': [],
        'Seed': [],
        'Unit': [],
        'bpm_gt': [],
        'bpm_rec': [],
        'bpm_mae': [],
        'sample_index': []
    }
    
    # Get the prefix from direction (e.g., 'PPG→ECG' -> 'PPG2ECG')
    prefix = direction.replace('→', '2')
    
    # Get the lists of values from bpm_metrics
    target_rates = bpm_metrics.get(f'target_rates_{prefix}', [])
    recon_rates = bpm_metrics.get(f'recon_rates_{prefix}', [])
    bpm_mae = bpm_metrics.get(f'bpm_{prefix}', [])
    sample_indices = bpm_metrics.get(f'sample_indices_{prefix}', [])
    
    # Ensure all lists have the same length
    min_length = min(len(target_rates), len(recon_rates), len(bpm_mae), len(sample_indices))
    
    # Fill the data dictionary
    for i in range(min_length):
        data['Dataset'].append(dataset)
        data['Method'].append(method)
        data['Direction'].append(direction)
        data['Seed'].append(seed)
        data['Unit'].append('BPM')
        data['bpm_gt'].append(target_rates[i])
        data['bpm_rec'].append(recon_rates[i])
        data['bpm_mae'].append(bpm_mae[i])
        data['sample_index'].append(sample_indices[i])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set default output path if not provided
    if output_path is None:
        # Create filename in the format: Dataset_Method_Direction_Seed_bpm_results.csv
        output_path = os.path.join("./results/bpm_results", f"{dataset}_{method}_{direction}_{seed}_bpm_results.csv")
    else:
        output_path = os.path.join(output_path, "bpm_results", f"{dataset}_{method}_{direction}_{seed}_bpm_results.csv")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved BPM results to {output_path}")

# REFINEMENT MODEL TESTING UTILS
def extract_bp_values(waveform, dim=2):
    """Extract SBP and DBP values from waveform.
    
    Args:
        waveform (torch.Tensor): Input waveform tensor
        dim (int): Dimension along which to extract min/max values (default: 2 for time dimension)
        
    Returns:
        tuple: (sbp, dbp) tensors
    """
    # Get maximum (SBP) and minimum (DBP) values along the specified dimension
    sbp_values = torch.max(waveform, dim=dim)[0]
    dbp_values = torch.min(waveform, dim=dim)[0]
    
    # Ensure proper shape - typically we want [batch_size, 1]
    if len(sbp_values.shape) == 1:
        sbp_values = sbp_values.unsqueeze(1)
        dbp_values = dbp_values.unsqueeze(1)
    
    return sbp_values, dbp_values

def calculate_bp_metrics(predictions, waveform, sbp_true, dbp_true, abp_gt, prefix, best_loss,
                        global_min=2.341260731456743, global_max=286.58240014784946, normalize=False):
    """Calculate BP metrics for both waveform and BP predictions.
    
    Args:
        predictions: Tuple of (sbp_pred, dbp_pred) or None if only waveform prediction
        waveform: Predicted waveform tensor (normalized between 0 and 1, or global min-max)
        sbp_true: True SBP values
        dbp_true: True DBP values
        abp_gt: Ground truth ABP waveform
        prefix: Metric prefix (e.g., 'PPG2ABP')
        best_loss: Dictionary to store metrics
        global_min: Global minimum value for normalization
        global_max: Global maximum value for normalization
        normalize: Whether to normalize predictions
    
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Initialize required keys in best_loss if they don't exist
    if f'{prefix}_loss' not in best_loss:
        best_loss[f'{prefix}_loss'] = []
    if f'{prefix}_ME_loss' not in best_loss:
        best_loss[f'{prefix}_ME_loss'] = []
    if f'{prefix}_corr' not in best_loss:
        best_loss[f'{prefix}_corr'] = []
    if f'{prefix}_SBP_loss' not in best_loss:
        best_loss[f'{prefix}_SBP_loss'] = []
    if f'{prefix}_DBP_loss' not in best_loss:
        best_loss[f'{prefix}_DBP_loss'] = []
    if f'{prefix}_MAP_loss' not in best_loss:
        best_loss[f'{prefix}_MAP_loss'] = []
    if f'{prefix}_SBP_ME_loss' not in best_loss:
        best_loss[f'{prefix}_SBP_ME_loss'] = []
    if f'{prefix}_DBP_ME_loss' not in best_loss:
        best_loss[f'{prefix}_DBP_ME_loss'] = []
    if f'{prefix}_MAP_ME_loss' not in best_loss:
        best_loss[f'{prefix}_MAP_ME_loss'] = []
    
    # Initialize storage for true and predicted BP values
    if f'{prefix}_SBP_true' not in best_loss:
        best_loss[f'{prefix}_SBP_true'] = []
    if f'{prefix}_DBP_true' not in best_loss:
        best_loss[f'{prefix}_DBP_true'] = []
    if f'{prefix}_MAP_true' not in best_loss:
        best_loss[f'{prefix}_MAP_true'] = []
    if f'{prefix}_SBP_pred' not in best_loss:
        best_loss[f'{prefix}_SBP_pred'] = []
    if f'{prefix}_DBP_pred' not in best_loss:
        best_loss[f'{prefix}_DBP_pred'] = []
    if f'{prefix}_MAP_pred' not in best_loss:
        best_loss[f'{prefix}_MAP_pred'] = []
    
    # Initialize BHS standard keys
    for threshold in [5, 10, 15]:
        if f'{prefix}_SBP_BHS_{threshold}' not in best_loss:
            best_loss[f'{prefix}_SBP_BHS_{threshold}'] = 0
        if f'{prefix}_DBP_BHS_{threshold}' not in best_loss:
            best_loss[f'{prefix}_DBP_BHS_{threshold}'] = 0
        if f'{prefix}_MAP_BHS_{threshold}' not in best_loss:
            best_loss[f'{prefix}_MAP_BHS_{threshold}'] = 0
    
    # If BP predictions are available, reconstruct full waveform
    if predictions is not None and predictions[0] is not None:
        sbp_pred, dbp_pred = predictions
        
        # Denormalize BP predictions if needed
        if normalize:
            Global_Min_Max = {"min": global_min, "max": global_max}
            
            # Convert predictions to numpy for Global_Min_Max_Norm
            sbp_pred_np = sbp_pred.cpu().numpy()
            dbp_pred_np = dbp_pred.cpu().numpy()
            
            # Denormalize predictions and ground truth
            sbp_pred_np = Global_Min_Max_Norm(sbp_pred_np, Global_Min_Max=Global_Min_Max, unnorm=True)
            dbp_pred_np = Global_Min_Max_Norm(dbp_pred_np, Global_Min_Max=Global_Min_Max, unnorm=True)
            
            # Convert back to torch tensors
            sbp_pred = torch.from_numpy(sbp_pred_np).to(sbp_pred.device)
            dbp_pred = torch.from_numpy(dbp_pred_np).to(dbp_pred.device)
        
        # Reconstruct full waveform using linear transformation
        # waveform is normalized between 0 and 1, we need to scale it to BP range
        waveform_reconstructed = waveform * (sbp_pred - dbp_pred).unsqueeze(-1) + dbp_pred.unsqueeze(-1)
    else:
        # If no predictions, use waveform directly and denormalize if needed
        if normalize:
            Global_Min_Max = {"min": global_min, "max": global_max}
            waveform_np = waveform.cpu().numpy()
            waveform_np = Global_Min_Max_Norm(waveform_np, Global_Min_Max=Global_Min_Max, unnorm=True)
            waveform_reconstructed = torch.from_numpy(waveform_np).to(waveform.device)
        else:
            waveform_reconstructed = waveform
    
    # Calculate waveform MAE and ME
    waveform_mae_loss = torch.nn.functional.l1_loss(waveform_reconstructed, abp_gt, reduction='none')
    waveform_me_loss = torch.sub(waveform_reconstructed, abp_gt)  # ME = pred - target
    
    # Calculate mean values
    waveform_mae = waveform_mae_loss.mean().item()
    waveform_me = waveform_me_loss.mean().item()
    
    # Calculate Pearson correlation between reconstructed and ground truth waveforms
    # Reshape tensors to [batch_size, signal_length]
    recon_flat = waveform_reconstructed.reshape(waveform_reconstructed.size(0), -1)
    gt_flat = abp_gt.reshape(abp_gt.size(0), -1)
    
    # Calculate means along signal dimension
    recon_mean = recon_flat.mean(dim=1, keepdim=True)
    gt_mean = gt_flat.mean(dim=1, keepdim=True)
    
    # Calculate correlation using vectorized operations
    numerator = torch.sum((recon_flat - recon_mean) * (gt_flat - gt_mean), dim=1)
    denominator = torch.sqrt(
        torch.sum((recon_flat - recon_mean)**2, dim=1) * 
        torch.sum((gt_flat - gt_mean)**2, dim=1)
    )
    
    # Handle potential division by zero
    correlations = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator)
    )
    
    # Calculate mean correlation
    waveform_corr = correlations.mean().item()
    
    # Store metrics
    metrics['waveform_mae'] = waveform_mae
    metrics['waveform_me'] = waveform_me
    metrics['waveform_corr'] = waveform_corr
    
    # Store losses for later statistics
    best_loss[f'{prefix}_loss'].extend(waveform_mae_loss.mean(dim=-1).cpu().numpy().tolist())
    best_loss[f'{prefix}_ME_loss'].extend(waveform_me_loss.mean(dim=-1).cpu().numpy().tolist())
    best_loss[f'{prefix}_corr'].extend(correlations.cpu().numpy().tolist())
    
    # Extract BP values from reconstructed waveform
    sbp_from_wave, dbp_from_wave = extract_bp_values(waveform_reconstructed)
    map_from_wave = calculate_MAP(sbp_from_wave, dbp_from_wave)

    map_true = calculate_MAP(sbp_true, dbp_true)
    
    # Store true and predicted BP values
    best_loss[f'{prefix}_SBP_true'].extend(sbp_true.cpu().numpy().tolist())
    best_loss[f'{prefix}_DBP_true'].extend(dbp_true.cpu().numpy().tolist())
    best_loss[f'{prefix}_MAP_true'].extend(map_true.cpu().numpy().tolist())
    best_loss[f'{prefix}_SBP_pred'].extend(sbp_from_wave.cpu().numpy().tolist())
    best_loss[f'{prefix}_DBP_pred'].extend(dbp_from_wave.cpu().numpy().tolist())
    best_loss[f'{prefix}_MAP_pred'].extend(map_from_wave.cpu().numpy().tolist())
    
    # Calculate BP MAE
    sbp_mae = torch.abs(sbp_from_wave - sbp_true)
    dbp_mae = torch.abs(dbp_from_wave - dbp_true)
    map_mae = torch.abs(map_from_wave - map_true)
    
    # Calculate BP ME
    sbp_me = torch.sub(sbp_from_wave, sbp_true)  # ME = pred - target
    dbp_me = torch.sub(dbp_from_wave, dbp_true)
    map_me = torch.sub(map_from_wave, map_true)
    
    # Store BP MAE metrics
    best_loss[f'{prefix}_SBP_loss'].extend(sbp_mae.cpu().numpy().tolist())
    best_loss[f'{prefix}_DBP_loss'].extend(dbp_mae.cpu().numpy().tolist())
    best_loss[f'{prefix}_MAP_loss'].extend(map_mae.cpu().numpy().tolist())
    
    # Store BP ME metrics
    best_loss[f'{prefix}_SBP_ME_loss'].extend(sbp_me.cpu().numpy().tolist())
    best_loss[f'{prefix}_DBP_ME_loss'].extend(dbp_me.cpu().numpy().tolist())
    best_loss[f'{prefix}_MAP_ME_loss'].extend(map_me.cpu().numpy().tolist())
    
    # Calculate BHS standard metrics
    for threshold in [5, 10, 15]:
        sbp_within = (sbp_mae <= threshold).sum().item()
        dbp_within = (dbp_mae <= threshold).sum().item()
        map_within = (map_mae <= threshold).sum().item()
        
        best_loss[f'{prefix}_SBP_BHS_{threshold}'] += sbp_within
        best_loss[f'{prefix}_DBP_BHS_{threshold}'] += dbp_within
        best_loss[f'{prefix}_MAP_BHS_{threshold}'] += map_within
    
    # Store mean errors
    metrics.update({
        'sbp_mae': sbp_mae.mean().item(),
        'dbp_mae': dbp_mae.mean().item(),
        'map_mae': map_mae.mean().item(),
        'sbp_me': sbp_me.mean().item(),
        'dbp_me': dbp_me.mean().item(),
        'map_me': map_me.mean().item()
    })
    
    return metrics

def print_abp_evaluation_results(best_loss, direction=None, args=None):
    """Print formatted evaluation results for ABP waveform reconstruction and BP values.
    
    Args:
        best_loss (dict): Dictionary containing evaluation metrics
        direction (str, optional): Specific direction to evaluate (e.g., 'PPG2ABP')
        args (dict, optional): Arguments dictionary containing wandb configuration
    """
    if direction is None:
        for prefix in ['PPG2ABP', 'ECG2ABP']:
            print(f"\n{prefix} Results:")
            print("-" * 50)

            # Print ABP Waveform Reconstruction Results
            print(f"ABP Waveform Reconstruction:")
            print(f"MAE: {np.mean(best_loss[f'{prefix}_loss']):.2f} ± {np.std(best_loss[f'{prefix}_loss']):.2f}")
            print(f"ME: {np.mean(best_loss[f'{prefix}_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_ME_loss']):.2f}")
            print(f"Pearson Correlation: {np.mean(best_loss[f'{prefix}_corr']):.3f} ± {np.std(best_loss[f'{prefix}_corr']):.3f}")
            print("-" * 50)

            # Print BP Values MAE Results
            print(f"BP Values MAE Results:")
            print(f"SBP: {np.mean(best_loss[f'{prefix}_SBP_loss']):.2f} ± {np.std(best_loss[f'{prefix}_SBP_loss']):.2f}")
            print(f"DBP: {np.mean(best_loss[f'{prefix}_DBP_loss']):.2f} ± {np.std(best_loss[f'{prefix}_DBP_loss']):.2f}")
            print(f"MAP: {np.mean(best_loss[f'{prefix}_MAP_loss']):.2f} ± {np.std(best_loss[f'{prefix}_MAP_loss']):.2f}")

            # Print BP Values ME Results
            print(f"\nBP Values ME Results:")
            print(f"SBP: {np.mean(best_loss[f'{prefix}_SBP_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_SBP_ME_loss']):.2f}")
            print(f"DBP: {np.mean(best_loss[f'{prefix}_DBP_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_DBP_ME_loss']):.2f}")
            print(f"MAP: {np.mean(best_loss[f'{prefix}_MAP_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_MAP_ME_loss']):.2f}")

            # Print BHS Standards
            print(f"\nBHS Standards:")
            total_samples = best_loss['total_Sample']

            for bp_type in ['SBP', 'DBP', 'MAP']:
                print(f"\n{bp_type}:")
                for threshold in [5, 10, 15]:
                    count = best_loss[f'{prefix}_{bp_type}_BHS_{threshold}']
                    percentage = (count / total_samples) * 100
                    print(f"≤{threshold}mmHg: {percentage:.1f}%")
                    
                    # Log BHS results to wandb if args is provided
                    if args is not None:
                        log_bhs_style_result(
                            dataset=args.dataset,
                            threshold=f"≤{threshold}mmHg",
                            direction=prefix,
                            measure=bp_type,
                            method=args.model_name,
                            value=percentage,
                            seed=args.seed,
                            args=args
                        )
            
            # Log AAMI results to wandb if args is provided
            if args is not None:
                for bp_type in ['SBP', 'DBP', 'MAP']:
                    log_aami_style_result(
                        dataset=args.dataset,
                        method=args.model_name,
                        direction=prefix,
                        measure=bp_type,
                        value=np.mean(best_loss[f'{prefix}_{bp_type}_ME_loss']),
                        std=np.std(best_loss[f'{prefix}_{bp_type}_ME_loss']),
                        seed=args.seed,
                        args=args
                    )
            
            # Log ABP Waveform Reconstruction metrics to wandb if args is provided
            if args is not None:
                # Log MAE
                log_clinical_tasks_result(
                    dataset=args.dataset,
                    method=args.model_name,
                    direction=prefix,
                    metric="MAE",
                    value=np.mean(best_loss[f'{prefix}_loss']),
                    std=np.std(best_loss[f'{prefix}_loss']),
                    seed=args.seed,
                    args=args
                )
                # Log Pearson Correlation
                log_clinical_tasks_result(
                    dataset=args.dataset,
                    method=args.model_name,
                    direction=prefix,
                    metric="PC",  # Pearson Correlation
                    value=np.mean(best_loss[f'{prefix}_corr']),
                    std=np.std(best_loss[f'{prefix}_corr']),
                    seed=args.seed,
                    args=args
                )
            
            # Save BP results to CSV if args is provided
            if args is not None and args.project_name is not None:
                # Convert prefix to direction format (e.g., 'PPG2ABP' -> 'PPG→ABP')
                direction_display = prefix
                save_bp_results_to_csv(
                    best_loss=best_loss,
                    dataset=args.dataset,
                    method=args.model_name,
                    direction=direction_display,
                    seed=args.seed
                )
            
    else:
        prefix = direction

        print(f"\n{prefix} Results:")
        print("-" * 50)

        # Print ABP Waveform Reconstruction Results
        print(f"ABP Waveform Reconstruction:")
        print(f"MAE: {np.mean(best_loss[f'{prefix}_loss']):.2f} ± {np.std(best_loss[f'{prefix}_loss']):.2f}")
        print(f"ME: {np.mean(best_loss[f'{prefix}_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_ME_loss']):.2f}")
        print(f"Pearson Correlation: {np.mean(best_loss[f'{prefix}_corr']):.3f} ± {np.std(best_loss[f'{prefix}_corr']):.3f}")
        print("-" * 50)

        # Print BP Values MAE Results
        print(f"BP Values MAE Results:")
        print(f"SBP: {np.mean(best_loss[f'{prefix}_SBP_loss']):.2f} ± {np.std(best_loss[f'{prefix}_SBP_loss']):.2f}")
        print(f"DBP: {np.mean(best_loss[f'{prefix}_DBP_loss']):.2f} ± {np.std(best_loss[f'{prefix}_DBP_loss']):.2f}")
        print(f"MAP: {np.mean(best_loss[f'{prefix}_MAP_loss']):.2f} ± {np.std(best_loss[f'{prefix}_MAP_loss']):.2f}")

        # Print BP Values ME Results
        print(f"\nBP Values ME Results:")
        print(f"SBP: {np.mean(best_loss[f'{prefix}_SBP_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_SBP_ME_loss']):.2f}")
        print(f"DBP: {np.mean(best_loss[f'{prefix}_DBP_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_DBP_ME_loss']):.2f}")
        print(f"MAP: {np.mean(best_loss[f'{prefix}_MAP_ME_loss']):.2f} ± {np.std(best_loss[f'{prefix}_MAP_ME_loss']):.2f}")

        # Print BHS Standards
        print(f"\nBHS Standards:")
        total_samples = best_loss['total_Sample']

        for bp_type in ['SBP', 'DBP', 'MAP']:
            print(f"\n{bp_type}:")
            for threshold in [5, 10, 15]:
                count = best_loss[f'{prefix}_{bp_type}_BHS_{threshold}']
                percentage = (count / total_samples) * 100
                print(f"≤{threshold}mmHg: {percentage:.1f}%")
                
                # Log BHS results to wandb if args is provided
                if args is not None:
                    log_bhs_style_result(
                        dataset=args.dataset,
                        threshold=f"≤{threshold}mmHg",
                        direction=prefix,
                        measure=bp_type,
                        method=args.model_name,
                        value=percentage,
                        seed=args.seed,
                        args=args
                    )
        
        # Log AAMI results to wandb if args is provided
        if args is not None:
            for bp_type in ['SBP', 'DBP', 'MAP']:
                log_aami_style_result(
                    dataset=args.dataset,
                    method=args.model_name,
                    direction=prefix,
                    measure=bp_type,
                    value=np.mean(best_loss[f'{prefix}_{bp_type}_ME_loss']),
                    std=np.std(best_loss[f'{prefix}_{bp_type}_ME_loss']),
                    seed=args.seed,
                    args=args
                )

        # Log ABP Waveform Reconstruction metrics to wandb if args is provided
        if args is not None:
            # Log MAE
            log_clinical_tasks_result(
                dataset=args.dataset,
                method=args.model_name,
                direction=prefix,
                metric="MAE",
                value=np.mean(best_loss[f'{prefix}_loss']),
                std=np.std(best_loss[f'{prefix}_loss']),
                seed=args.seed,
                args=args
            )
            # Log Pearson Correlation
            log_clinical_tasks_result(
                dataset=args.dataset,
                method=args.model_name,
                direction=prefix,
                metric="PC",  # Pearson Correlation
                value=np.mean(best_loss[f'{prefix}_corr']),
                std=np.std(best_loss[f'{prefix}_corr']),
                seed=args.seed,
                args=args
            )
        
        # Save BP results to CSV if args is provided
        if args is not None and args.project_name is not None:
            # Convert prefix to direction format (e.g., 'PPG2ABP' -> 'PPG→ABP')
            direction_display = prefix
            save_bp_results_to_csv(
                best_loss=best_loss,
                dataset=args.dataset,
                method=args.model_name if args.model_name is not None else "Ours",
                direction=direction_display,
                seed=args.seed
            )

def save_bp_results_to_csv(best_loss: Dict[str, List[float]], dataset: str, method: str, direction: str, seed: int, output_path: str = None) -> None:
    """Save BP results to a CSV file in a standardized format.
    
    Args:
        best_loss (Dict[str, List[float]]): Dictionary containing BP metrics and values
        dataset (str): Name of the dataset (e.g., 'PulseDB', 'UCI')
        method (str): Name of the method used (e.g., 'NABNet', 'P2EWGAN', 'PatchTST', 'our')
        direction (str): Direction of transformation (e.g., 'PPG→ABP', 'ECG→ABP')
        seed (int): Random seed used
        output_path (str, optional): Path to save the CSV file. If None, saves to current directory.
    """
    # Create DataFrame with required columns
    data = {
        'Dataset': [],
        'Method': [],
        'Direction': [],
        'Seed': [],
        'Unit': [],
        'sbp_gt': [],
        'sbp_pred': [],
        'sbp_mae': [],
        'dbp_gt': [],
        'dbp_pred': [],
        'dbp_mae': [],
        'map_gt': [],
        'map_pred': [],
        'map_mae': [],
        'sample_index': []
    }
    
    # Get the prefix from direction (e.g., 'PPG→ABP' -> 'PPG2ABP')
    prefix = direction.replace('→', '2')
    
    # Get the lists of values from best_loss
    sbp_true = best_loss.get(f'{prefix}_SBP_true', [])
    sbp_pred = best_loss.get(f'{prefix}_SBP_pred', [])
    sbp_mae = best_loss.get(f'{prefix}_SBP_loss', [])
    dbp_true = best_loss.get(f'{prefix}_DBP_true', [])
    dbp_pred = best_loss.get(f'{prefix}_DBP_pred', [])
    dbp_mae = best_loss.get(f'{prefix}_DBP_loss', [])
    map_true = best_loss.get(f'{prefix}_MAP_true', [])
    map_pred = best_loss.get(f'{prefix}_MAP_pred', [])
    map_mae = best_loss.get(f'{prefix}_MAP_loss', [])
    
    # Ensure all lists have the same length
    min_length = min(len(sbp_true), len(sbp_pred), len(sbp_mae),
                    len(dbp_true), len(dbp_pred), len(dbp_mae),
                    len(map_true), len(map_pred), len(map_mae))
    
    # Fill the data dictionary
    for i in range(min_length):
        data['Dataset'].append(dataset)
        data['Method'].append(method)
        data['Direction'].append(direction)
        data['Seed'].append(seed)
        data['Unit'].append('mmHg')
        data['sbp_gt'].append(sbp_true[i])
        data['sbp_pred'].append(sbp_pred[i])
        data['sbp_mae'].append(sbp_mae[i])
        data['dbp_gt'].append(dbp_true[i])
        data['dbp_pred'].append(dbp_pred[i])
        data['dbp_mae'].append(dbp_mae[i])
        data['map_gt'].append(map_true[i])
        data['map_pred'].append(map_pred[i])
        data['map_mae'].append(map_mae[i])
        data['sample_index'].append(i)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set default output path if not provided
    if output_path is None:
        # Create filename in the format: Dataset_Method_Direction_Seed_sbp_dbp_map_results.csv
        output_path = os.path.join("./results/sbp_dbp_map_results", f"{dataset}_{method}_{direction}_{seed}_sbp_dbp_map_results.csv")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved BP results to {output_path}")