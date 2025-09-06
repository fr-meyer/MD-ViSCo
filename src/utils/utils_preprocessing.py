# Standard library imports
import os

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk

# Local imports

def Min_Max_Norm(X, unnorm=False, axis=-1):
    """Normalize/unnormalize data using min-max scaling.
    
    Args:
        X (np.ndarray): Input data array
        unnorm (bool): If True, unnormalize the data. If False, normalize it
        axis (int): Axis along which to compute min/max. Default is -1 (last axis)
        
    Returns:
        np.ndarray: Normalized/unnormalized data array
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
        
    X_min = np.min(X, axis=axis, keepdims=True)
    X_max = np.max(X, axis=axis, keepdims=True)
    
    if unnorm:
        # Unnormalize: scale back to original range
        X_scaled = X * (X_max - X_min) + X_min
    else:
        # Normalize to [0,1] range
        X_scaled = (X - X_min) / (X_max - X_min)
    
    return X_scaled

def Zero_Centered(X, unnorm=False, axis=-1):
    """Center data around zero by subtracting/adding mean.
    
    Args:
        X (np.ndarray): Input data array
        unnorm (bool): If True, restore original mean. If False, center around zero
        axis (int): Axis along which to compute mean. Default is -1
        
    Returns:
        np.ndarray: Zero-centered/uncentered data array
        
    Raises:
        TypeError: If X is not a numpy array
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    
    # Compute mean along specified axis
    mean = np.mean(X, axis=axis, keepdims=True)
    
    if unnorm:
        X_scaled = X + mean  # Restore original mean
    else:
        X_scaled = X - mean  # Center around zero
        
    return X_scaled

def Min_Max_Norm_Torch(X, Feature_Range={"max": 1, "min": 0}):
    """Normalize tensor data using min-max scaling.
    
    Args:
        X (torch.Tensor): Input tensor
        Feature_Range (dict): Target range for normalization
        
    Returns:
        torch.Tensor: Normalized tensor in specified range
    """
    # Scale to [0,1] range first
    X_std = torch.div(
        torch.sub(X, torch.min(X, 2)[0].unsqueeze(1)),
        torch.sub(torch.max(X, 2)[0], torch.min(X, 2)[0]).unsqueeze(1)
    )
    return X_std

def Global_Min_Max_Norm(X, Global_Min_Max, unnorm=False):
    """Normalize/unnormalize values using global min-max values to [0,1] range.
    
    Args:
        X (np.ndarray): Input array of values to normalize/unnormalize
        Global_Min_Max (dict): Dictionary with global min/max values
            Required keys: "min" (global minimum), "max" (global maximum)
        unnorm (bool): If True, unnormalize the data back to original range. 
                      If False, normalize to [0,1] range
        
    Returns:
        np.ndarray: Normalized/unnormalized data array
        
    Raises:
        ValueError: If Global_Min_Max is missing required keys
        TypeError: If X is not a numpy array
    """
    # Input validation
    required_keys = ["min", "max"]
    if not all(key in Global_Min_Max for key in required_keys):
        raise ValueError("Global_Min_Max must contain 'min' and 'max' keys")
        
    if unnorm:
        # Scale from [0,1] back to original range
        X_scaled = X * (Global_Min_Max["max"] - Global_Min_Max["min"]) + Global_Min_Max["min"]
    else:
        # Normalize to [0,1] using global values
        X_scaled = (X - Global_Min_Max["min"]) / (Global_Min_Max["max"] - Global_Min_Max["min"])
    
    return X_scaled

def count_model_parameters(model):
    """Count the total, trainable and non-trainable parameters of a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model to analyze
        
    Returns:
        dict: Dictionary containing total, trainable and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }

def print_model_parameters(model):
    """Print a formatted summary of model parameters.
    
    Args:
        model (torch.nn.Module): PyTorch model to analyze
    """
    params = count_model_parameters(model)
    
    print('='*50)
    print(f'Total parameters: {params["total"]:,}')
    print(f'Trainable parameters: {params["trainable"]:,}')
    print(f'Non-trainable parameters: {params["non_trainable"]:,}')
    print('='*50)

def normalize_signals(signals):
    """Normalize signals using different methods
    
    Args:
        signals (np.ndarray): Input signals array with shape (Channel Number, Waveform length)
        
    Returns:
        tuple: (min-max normalized signals, zero-centered signals)
    """
    # Create copies to avoid modifying original data
    signals_min_max = signals.copy()
    
    # Min-max normalization for all channels
    signals_min_max = Min_Max_Norm(signals_min_max, axis=1)
    
    # Zero-centering for all signals after local min-max normalization
    signals_zc = Zero_Centered(signals_min_max.copy(), axis=1)

    return signals_min_max, signals_zc

def calculate_BPM(signals, ii_label, ppg_label):
    """Calculate heart rate from ECG and pulse rate from PPG using correct channel indices
    
    Args:
        signals (np.ndarray): Input signals array with ECG and PPG channels
        ii_label (int): Index of the ECG (lead II) channel
        ppg_label (int): Index of the PPG channel
        
    Returns:
        tuple: (heart_rate, pulse_rate) or None if either calculation fails
    """
    try:
        signals_ecg, info_ecg = nk.ecg_process(signals[ii_label].flatten(), sampling_rate=125)
        hr = np.mean(signals_ecg['ECG_Rate'])
    except:
        return None  # Return None if HR calculation fails

    try:
        signals_ppg, info_ppg = nk.ppg_process(signals[ppg_label].flatten(), sampling_rate=125)
        pr = np.mean(signals_ppg['PPG_Rate'])
    except:
        return None  # Return None if PR calculation fails
    
    # Only return values if both calculations succeeded
    if hr is not None and pr is not None:
        return hr, pr
    return None

def calculate_MAP(sbp, dbp):
    """Calculate Mean Arterial Pressure (MAP) from Systolic and Diastolic Blood Pressure.
    
    Args:
        sbp (Union[float, np.ndarray, torch.Tensor]): Systolic Blood Pressure
        dbp (Union[float, np.ndarray, torch.Tensor]): Diastolic Blood Pressure
        
    Returns:
        Union[float, np.ndarray, torch.Tensor]: Calculated MAP values
    """
    return (sbp + (2 * dbp)) / 3

def isExist_dir(directory):
    """Create directory if it doesn't exist.
    
    Args:
        directory (str): Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def check_file_exists(filepath: str, overwrite: bool = False) -> bool:
    """Check if a file exists and handle it according to overwrite parameter.
    
    Args:
        filepath (str): Path to the file to check
        overwrite (bool): If True, allow overwriting existing file. If False, raise error if file exists.
        
    Returns:
        bool: True if file exists and overwrite is True, False if file doesn't exist
        
    Raises:
        FileExistsError: If file already exists and overwrite is False
    """
    if os.path.exists(filepath):
        if not overwrite:
            raise FileExistsError(f"File {filepath} already exists. Set overwrite=True to overwrite it.")
        else:
            print(f"Warning: Overwriting existing file {filepath}")
            return True
    return False

def safe_create_dataset(group, key, value):
    """Helper function to safely create datasets with proper type conversion"""
    try:
        # Handle None values by converting to np.nan
        if value is None:
            value = np.nan
        
        # Handle nested dictionaries by creating subgroups
        if isinstance(value, dict):
            sub_group = group.create_group(key)
            for sub_key, sub_value in value.items():
                safe_create_dataset(sub_group, sub_key, sub_value)
            return
        
        # Handle pandas DataFrame
        if isinstance(value, pd.DataFrame):
            # Create a subgroup for the DataFrame
            df_group = group.create_group(key)
            # Save column names as attributes
            df_group.attrs['columns'] = value.columns.tolist()
            # Save index as attributes if it's not the default
            if not isinstance(value.index, pd.RangeIndex):
                df_group.attrs['index'] = value.index.tolist()
            # Convert each column to numpy array with NaN handling
            for col in value.columns:
                col_data = value[col].to_numpy()
                if col_data.dtype == object:
                    col_data = np.where(pd.isna(col_data), np.nan, col_data).astype(float)
                df_group.create_dataset(col, data=col_data)
            return
        
        # Convert lists/tuples to numpy arrays
        if isinstance(value, (list, tuple)):
            value = np.array(value)
            # Convert NA to np.nan and cast to float if possible
            if value.dtype == object:
                value = np.where(pd.isna(value), np.nan, value).astype(float)
        
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                print(f"Warning: Converting object array for {key}")
                # Try to convert object array to string array
                if all(isinstance(x, str) for x in value):
                    value = np.array(value, dtype='S')
                else:
                    # Try to convert to float with NA handling
                    try:
                        value = np.where(pd.isna(value), np.nan, value).astype(float)
                    except:
                        # If conversion fails, save each element separately
                        for i, x in enumerate(value):
                            safe_create_dataset(group, f"{key}_{i}", x)
                        return
        
        # Create the dataset
        group.create_dataset(key, data=value)
        
    except Exception as e:
        print(f"Error saving {key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
        print(f"  Value: {value}")
        print(f"  Error: {str(e)}")
        raise
