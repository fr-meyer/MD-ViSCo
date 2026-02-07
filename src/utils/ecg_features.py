# Standard library imports

# Third-party imports
import numpy as np
import pandas as pd
import neurokit2 as nk

def extract_ecg_features(ecg_signal: np.ndarray, sampling_rate: int = 1000) -> tuple:
    """
    Extract peaks and offsets from an ECG waveform using NeuroKit2.
    
    Args:
        ecg_signal (np.ndarray): The raw ECG signal
        sampling_rate (int): The sampling rate of the signal in Hz. Default is 1000.
        
    Returns:
        tuple: A tuple containing:
            - signals (pd.DataFrame): DataFrame containing processed ECG signals and features
            - info (dict): Dictionary containing information about the detected peaks
            
    Raises:
        ValueError: If the input signal is empty or invalid
        RuntimeError: If there's an error processing the ECG signal
    """
    try:
        # Validate input signal
        if ecg_signal is None or len(ecg_signal) == 0:
            raise ValueError("Input ECG signal is empty")
        
        if not isinstance(ecg_signal, np.ndarray):
            raise ValueError("Input ECG signal must be a numpy array")
        
        if np.all(np.isnan(ecg_signal)):
            raise ValueError("Input ECG signal contains only NaN values")
        
        # Process the ECG signal
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        
        # Validate the processed signals
        if signals is None or len(signals) == 0:
            raise ValueError("No valid ECG features could be extracted")
        
        return signals, info
        
    except Exception as e:
        raise RuntimeError(f"Error in extract_ecg_features: {str(e)}")

def get_peak_locations(signals: pd.DataFrame) -> dict:
    """
    Extract peak locations and their corresponding signal values from the processed ECG signals.
    
    Args:
        signals (pd.DataFrame): DataFrame containing processed ECG signals
        
    Returns:
        dict: Dictionary containing dictionaries of peak information for each wave type.
              For R, P, and T waves, each contains:
              - indices: Array of indices where peaks occur
              - values: Array of corresponding signal values
              - onsets: Array of onset indices
              - offsets: Array of offset indices
              For Q and S waves, each contains:
              - indices: Array of indices where peaks occur
              - values: Array of corresponding signal values
              
    Raises:
        ValueError: If the input signals DataFrame is invalid
        RuntimeError: If there's an error processing the peak locations
    """
    try:
        if signals is None or not isinstance(signals, pd.DataFrame):
            raise ValueError("Invalid signals DataFrame")
            
        peak_locations = {}
        
        # Process waves with full features (R, P, T)
        for wave in ['R', 'P', 'T']:
            try:
                # Get peak indices
                peak_indices = np.where(signals[f'ECG_{wave}_Peaks'] == 1)[0]
                onset_indices = np.where(signals[f'ECG_{wave}_Onsets'] == 1)[0]
                offset_indices = np.where(signals[f'ECG_{wave}_Offsets'] == 1)[0]
                
                # Get corresponding signal values, handling empty arrays
                peak_values = signals['ECG_Clean'].iloc[peak_indices].values if len(peak_indices) > 0 else np.array([])
                onset_values = signals['ECG_Clean'].iloc[onset_indices].values if len(onset_indices) > 0 else np.array([])
                offset_values = signals['ECG_Clean'].iloc[offset_indices].values if len(offset_indices) > 0 else np.array([])
                
                # Store peak information
                peak_locations[f'{wave.lower()}_wave'] = {
                    'indices': peak_indices,
                    'values': peak_values,
                    'onsets': {
                        'indices': onset_indices,
                        'values': onset_values
                    },
                    'offsets': {
                        'indices': offset_indices,
                        'values': offset_values
                    }
                }
            except KeyError as e:
                raise ValueError(f"Missing required column in signals DataFrame: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Error processing {wave}-wave: {str(e)}")
        
        # Process waves with only peaks (Q, S)
        for wave in ['Q', 'S']:
            try:
                # Get peak indices
                peak_indices = np.where(signals[f'ECG_{wave}_Peaks'] == 1)[0]
                
                # Get corresponding signal values, handling empty arrays
                peak_values = signals['ECG_Clean'].iloc[peak_indices].values if len(peak_indices) > 0 else np.array([])
                
                # Store peak information
                peak_locations[f'{wave.lower()}_wave'] = {
                    'indices': peak_indices,
                    'values': peak_values
                }
            except KeyError as e:
                raise ValueError(f"Missing required column in signals DataFrame: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Error processing {wave}-wave: {str(e)}")
        
        return peak_locations
        
    except Exception as e:
        raise RuntimeError(f"Error in get_peak_locations: {str(e)}")

def calculate_qt_intervals(peak_locations: dict, sampling_rate: int = 1000, max_qt: float = 0.6) -> dict:
    """
    Calculate QT intervals from QRS onset to T-wave offset using R-peak-aligned delineation.
    
    The QT interval is measured from the onset of the QRS complex to the end of the T wave.
    Normal QT interval varies with heart rate, so rate-corrected QT (QTc) is often used.
    Normal QTc is < 0.44s (440ms) for men and < 0.46s (460ms) for women.
    
    This function uses R-peaks as anchors to align QRS onsets and T-wave offsets,
    ensuring each measurement corresponds to the same heartbeat.
    
    Args:
        peak_locations (dict): Dictionary containing peak locations from get_peak_locations()
        sampling_rate (int): The sampling rate of the signal in Hz. Default is 1000.
        max_qt (float): Maximum allowed QT interval in seconds. Default is 0.6s (600ms).
        
    Returns:
        dict: Dictionary containing:
            - durations: Array of QT durations in seconds
            - qtc_durations: Array of rate-corrected QT durations in seconds
            - indices: Dictionary containing:
                - r_peaks: Array of R-peak indices
                - qrs_onsets: Array of QRS onset indices
                - t_offsets: Array of T-wave offset indices
            - clinical_interpretation: Dictionary containing:
                - is_normal: Array of booleans indicating if each QTc is within normal range
                - is_prolonged: Array of booleans indicating if each QTc is prolonged
                - is_short: Array of booleans indicating if each QTc is short
            - summary: Dictionary containing:
                - mean_qt: Mean QT interval in seconds
                - mean_qtc: Mean QTc interval in seconds
                - std_qt: Standard deviation of QT intervals
                - min_qt: Minimum QT interval in seconds
                - max_qt: Maximum QT interval in seconds
                - abnormal_beats_percentage: Percentage of beats with abnormal QTc
    """
    try:
        if not isinstance(peak_locations, dict):
            raise ValueError("Invalid peak_locations dictionary")
            
        # Get all relevant indices
        try:
            r_peaks = peak_locations['r_wave']['indices']
            qrs_onsets = peak_locations['r_wave']['onsets']['indices']
            t_offsets = peak_locations['t_wave']['offsets']['indices']
        except KeyError as e:
            raise ValueError(f"Missing required key in peak_locations: {str(e)}")
        
        qt_durations = []
        valid_r_peaks = []
        valid_qrs_onsets = []
        valid_t_offsets = []
        
        # Use R-peaks as anchors to align QRS onsets and T-wave offsets
        n = len(r_peaks)
        for i in range(n):
            try:
                # Get corresponding QRS onset and T offset for this R-peak
                onset = qrs_onsets[i] if i < len(qrs_onsets) else None
                offset = t_offsets[i] if i < len(t_offsets) else None
                
                # Only process if we have both onset and offset
                if onset is not None and offset is not None:
                    duration = (offset - onset) / sampling_rate
                    
                    # Only include if duration is within physiological range
                    if 0.2 <= duration <= max_qt:  # physiological plausible range (200-600ms)
                        qt_durations.append(duration)
                        valid_r_peaks.append(r_peaks[i])
                        valid_qrs_onsets.append(onset)
                        valid_t_offsets.append(offset)
            except Exception as e:
                print(f"Warning: Error processing beat at R-peak {r_peaks[i]}: {str(e)}")
                continue
        
        if not qt_durations:
            return {
                'durations': np.array([]),
                'qtc_durations': np.array([]),
                'indices': {
                    'r_peaks': np.array([]),
                    'qrs_onsets': np.array([]),
                    't_offsets': np.array([])
                },
                'clinical_interpretation': {
                    'is_normal': np.array([]),
                    'is_prolonged': np.array([]),
                    'is_short': np.array([])
                },
                'summary': {
                    'mean_qt': 0.0,
                    'mean_qtc': 0.0,
                    'std_qt': 0.0,
                    'min_qt': 0.0,
                    'max_qt': 0.0,
                    'abnormal_beats_percentage': 0.0
                }
            }
        
        try:
            # Convert lists to numpy arrays
            qt_durations = np.array(qt_durations)
            
            # Calculate RR intervals for rate correction
            rr_intervals = np.diff(valid_r_peaks) / sampling_rate
            if len(rr_intervals) > 0:
                rr_intervals = np.append(rr_intervals, rr_intervals[-1])  # Use last RR for last beat
                # Calculate QTc using Bazett's formula: QTc = QT / sqrt(RR)
                qtc_durations = qt_durations / np.sqrt(rr_intervals)
            else:
                qtc_durations = np.array([])
            
            # Clinical interpretation
            is_normal = np.array([]) if len(qtc_durations) == 0 else (qtc_durations >= 0.3) & (qtc_durations <= 0.44)  # Normal range: 300-440ms
            is_prolonged = np.array([]) if len(qtc_durations) == 0 else qtc_durations > 0.44  # Prolonged: >440ms
            is_short = np.array([]) if len(qtc_durations) == 0 else qtc_durations < 0.3  # Short: <300ms
            
            # Calculate summary statistics
            mean_qt = np.mean(qt_durations) if len(qt_durations) > 0 else 0.0
            mean_qtc = np.mean(qtc_durations) if len(qtc_durations) > 0 else 0.0
            std_qt = np.std(qt_durations) if len(qt_durations) > 0 else 0.0
            min_qt = np.min(qt_durations) if len(qt_durations) > 0 else 0.0
            max_qt = np.max(qt_durations) if len(qt_durations) > 0 else 0.0
            abnormal_beats_percentage = (np.sum(~is_normal) / len(qtc_durations)) * 100 if len(qtc_durations) > 0 else 0.0
            
            return {
                'durations': qt_durations,
                'qtc_durations': qtc_durations,
                'indices': {
                    'r_peaks': np.array(valid_r_peaks),
                    'qrs_onsets': np.array(valid_qrs_onsets),
                    't_offsets': np.array(valid_t_offsets)
                },
                'clinical_interpretation': {
                    'is_normal': is_normal,
                    'is_prolonged': is_prolonged,
                    'is_short': is_short
                },
                'summary': {
                    'mean_qt': mean_qt,
                    'mean_qtc': mean_qtc,
                    'std_qt': std_qt,
                    'min_qt': min_qt,
                    'max_qt': max_qt,
                    'abnormal_beats_percentage': abnormal_beats_percentage
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error in QT interval calculations: {str(e)}")
            
    except Exception as e:
        raise RuntimeError(f"Error in calculate_qt_intervals: {str(e)}")
