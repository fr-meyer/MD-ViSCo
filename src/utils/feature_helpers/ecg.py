"""ECG feature extraction utilities used by ECGFeatureExtractor.

This module serves as the canonical home for ECG feature helpers using
NeuroKit2. Used by extractors and evaluators for ECG-derived features.

Functions:
    - extract_ecg_features: Extract peaks and offsets from ECG using NeuroKit2

Examples:
    >>> signals, info = extract_ecg_features(ecg_signal, sampling_rate=1000)

See Also:
    - src.processors.extractors: ECGFeatureExtractor
    - neurokit2: ECG analysis library
"""

from __future__ import annotations

import logging
from typing import Any

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def extract_ecg_features(
    ecg_signal: np.ndarray,
    sampling_rate: int = 1000,
    fl_hz: float | None = None,
    fh_hz: float | None = None,
    strict_mode: bool = False,
) -> tuple:
    """Extract peaks and offsets from an ECG waveform using NeuroKit2.

    Args:
        ecg_signal: Input ECG signal as numpy array
        sampling_rate: Sampling rate in Hz
        fl_hz: Low-frequency cutoff for band-pass filtering (Hz). If provided,
            applies filtering before peak detection.
        fh_hz: High-frequency cutoff for band-pass filtering (Hz). If provided,
            applies filtering before peak detection.
        strict_mode: If True, raises exceptions on failure instead of returning
            fallback data. If False, returns zeroed peaks DataFrame on error.

    Returns:
        Tuple of (signals DataFrame, info dict). Info dict contains "status"
        key which is "success" on normal completion or "fallback" if fallback
        was used (only when strict_mode=False).

    Raises:
        ValueError: If input validation fails or strict_mode=True and
            extraction fails
        RuntimeError: If strict_mode=True and extraction fails
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        raise ValueError("Input ECG signal is empty")

    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("Input ECG signal must be a numpy array")

    if np.all(np.isnan(ecg_signal)):
        raise ValueError("Input ECG signal contains only NaN values")

    try:
        # Apply filtering if frequency cutoffs are provided
        processed_signal = ecg_signal.copy()
        if fl_hz is not None or fh_hz is not None:
            # Apply band-pass filtering using scipy.signal
            # If only one cutoff is provided, use reasonable defaults for
            # the other
            lowcut = fl_hz if fl_hz is not None else 0.5
            highcut = fh_hz if fh_hz is not None else None

            # Validate cutoff frequencies
            nyquist = sampling_rate / 2.0
            if lowcut >= nyquist:
                raise ValueError(
                    f"Low cutoff frequency {lowcut} Hz must be less than "
                    f"Nyquist frequency {nyquist} Hz"
                )
            if highcut is not None and highcut >= nyquist:
                raise ValueError(
                    f"High cutoff frequency {highcut} Hz must be less than "
                    f"Nyquist frequency {nyquist} Hz"
                )
            if highcut is not None and highcut <= lowcut:
                raise ValueError(
                    f"High cutoff frequency {highcut} Hz must be greater "
                    f"than low cutoff frequency {lowcut} Hz"
                )

            if highcut is not None:
                # Band-pass filter
                sos = scipy_signal.butter(
                    N=4,  # 4th order Butterworth filter
                    Wn=[lowcut, highcut],
                    btype="band",
                    fs=sampling_rate,
                    output="sos",
                )
                processed_signal = scipy_signal.sosfiltfilt(sos, processed_signal)
            else:
                # High-pass filter only
                sos = scipy_signal.butter(
                    N=4,
                    Wn=lowcut,
                    btype="high",
                    fs=sampling_rate,
                    output="sos",
                )
                processed_signal = scipy_signal.sosfiltfilt(sos, processed_signal)

        signals, info = nk.ecg_process(processed_signal, sampling_rate=sampling_rate)

        if signals is None or len(signals) == 0:
            raise ValueError("No valid ECG features could be extracted")

        # Mark successful extraction
        if "status" not in info:
            info["status"] = "success"
        return signals, info

    except Exception as e:
        if strict_mode:
            # In strict mode, propagate the error
            raise RuntimeError(f"ECG feature extraction failed: {str(e)}") from e

        # Fallback mode: return zeroed peaks DataFrame
        logger.warning("ECG feature extraction fallback engaged due to error: %s", e)
        length = int(len(ecg_signal))
        zeros = np.zeros(length, dtype=int)
        nan_series = np.full(length, np.nan, dtype=float)

        data = {
            "ECG_Clean": pd.Series(ecg_signal),
            "ECG_Quality": pd.Series(nan_series),
        }
        for wave in ["R", "P", "T"]:
            data[f"ECG_{wave}_Peaks"] = pd.Series(zeros)
            data[f"ECG_{wave}_Onsets"] = pd.Series(zeros)
            data[f"ECG_{wave}_Offsets"] = pd.Series(zeros)
        for wave in ["Q", "S"]:
            data[f"ECG_{wave}_Peaks"] = pd.Series(zeros)

        fallback_df = pd.DataFrame(data)
        return fallback_df, {"status": "fallback", "error": str(e)}


def get_peak_locations(signals: pd.DataFrame) -> dict:
    """Extract peak locations and signal values from an ECG signals DataFrame.

    Expects columns produced by NeuroKit2-style ECG processing (e.g.
    ``ECG_R_Peaks``, ``ECG_P_Onsets``, ``ECG_T_Offsets``, etc.). For each
    wave type (R, P, T, Q, S), returns indices and optionally
    values/onsets/offsets as applicable.

    Args:
        signals: DataFrame from ECG processing (e.g.
            :func:`extract_ecg_features`). Must contain ``ECG_Clean`` and
            wave-specific columns such as ``ECG_R_Peaks``, ``ECG_R_Onsets``,
            ``ECG_R_Offsets``, and similarly for P, T, Q, S waves.

    Returns:
        Dict mapping wave keys (e.g. ``"r_wave"``, ``"p_wave"``,
        ``"t_wave"``, ``"q_wave"``, ``"s_wave"``) to sub-dicts. For R, P, T:
        each sub-dict has ``"indices"`` (peak sample indices), ``"values"``
        (signal values at peaks), ``"onsets"`` and ``"offsets"`` (each with
        ``"indices"`` and ``"values"``). For Q, S: sub-dict has only
        ``"indices"`` and ``"values"``. Missing columns result in empty
        arrays for that wave.

    Raises:
        ValueError: If ``signals`` is None or is not a pandas DataFrame.
        RuntimeError: If any unexpected error occurs during extraction (e.g.
            invalid structure or indexing failure); re-raised from internal
            exceptions.
    """
    try:
        if signals is None or not isinstance(signals, pd.DataFrame):
            raise ValueError("Invalid signals DataFrame")

        peak_locations: dict[str, dict[str, Any]] = {}

        for wave in ["R", "P", "T"]:
            try:
                peak_indices = np.where(signals[f"ECG_{wave}_Peaks"] == 1)[0]
                onset_indices = np.where(signals[f"ECG_{wave}_Onsets"] == 1)[0]
                offset_indices = np.where(signals[f"ECG_{wave}_Offsets"] == 1)[0]

                peak_values = (
                    signals["ECG_Clean"].iloc[peak_indices].values
                    if len(peak_indices) > 0
                    else np.array([])
                )
                onset_values = (
                    signals["ECG_Clean"].iloc[onset_indices].values
                    if len(onset_indices) > 0
                    else np.array([])
                )
                offset_values = (
                    signals["ECG_Clean"].iloc[offset_indices].values
                    if len(offset_indices) > 0
                    else np.array([])
                )

                peak_locations[f"{wave.lower()}_wave"] = {
                    "indices": peak_indices,
                    "values": peak_values,
                    "onsets": {
                        "indices": onset_indices,
                        "values": onset_values,
                    },
                    "offsets": {
                        "indices": offset_indices,
                        "values": offset_values,
                    },
                }
            except KeyError:
                peak_locations[f"{wave.lower()}_wave"] = {
                    "indices": np.array([]),
                    "values": np.array([]),
                    "onsets": {
                        "indices": np.array([]),
                        "values": np.array([]),
                    },
                    "offsets": {
                        "indices": np.array([]),
                        "values": np.array([]),
                    },
                }

        for wave in ["Q", "S"]:
            try:
                peak_indices = np.where(signals[f"ECG_{wave}_Peaks"] == 1)[0]
                peak_values = (
                    signals["ECG_Clean"].iloc[peak_indices].values
                    if len(peak_indices) > 0
                    else np.array([])
                )
                peak_locations[f"{wave.lower()}_wave"] = {
                    "indices": peak_indices,
                    "values": peak_values,
                }
            except KeyError:
                peak_locations[f"{wave.lower()}_wave"] = {
                    "indices": np.array([]),
                    "values": np.array([]),
                }

        return peak_locations

    except Exception as e:
        raise RuntimeError(f"Error in get_peak_locations: {str(e)}") from e


def calculate_qt_intervals(
    peak_locations: dict,
    sampling_rate: int = 1000,
    max_qt: float | None = None,
) -> dict[str, Any]:
    """Calculate QT intervals and stats from R-wave and T-wave peak/offset data.

    Uses R-wave peak indices and T-wave offset indices to compute QT duration
    per beat (in seconds), then returns min/max/mean/median/std and the raw
    durations array. Optionally filters out QT values above ``max_qt``
    seconds.

    Args:
        peak_locations: Dict from :func:`get_peak_locations` (or compatible
            structure). Must contain ``"r_wave"`` and ``"t_wave"`` entries,
            each with ``"indices"`` and (for ``t_wave``) ``"offsets"`` with
            ``"indices"``.
        sampling_rate: Sampling rate in Hz used to convert sample counts to
            seconds. Defaults to 1000.
        max_qt: Optional maximum QT duration in seconds; intervals above this
            are excluded from statistics. Defaults to None (no cap).

    Returns:
        Dict with keys: ``"min"``, ``"max"``, ``"mean"``, ``"median"``,
        ``"std"`` (floats; may be NaN if no valid intervals), and
        ``"durations"`` (1D numpy array of QT durations in seconds).

    Raises:
        RuntimeError: If any error occurs during conversion or computation
            (e.g. invalid ``peak_locations`` structure or indexing);
            re-raised from internal exceptions.
    """
    try:
        peak_locations_converted = {}
        for wave in ["r_wave", "p_wave", "t_wave"]:
            wave_data = peak_locations.get(wave, {})
            indices = wave_data.get("indices", np.array([]))
            values = wave_data.get("values", np.array([]))
            onsets = wave_data.get("onsets", {}).get("indices", np.array([]))
            offsets = wave_data.get("offsets", {}).get("indices", np.array([]))

            peak_locations_converted[wave] = {
                "indices": np.array(indices, dtype=np.int64),
                "values": np.array(values, dtype=np.float64),
                "onsets": np.array(onsets, dtype=np.int64),
                "offsets": np.array(offsets, dtype=np.int64),
            }

        qt_durations = []
        for r_idx, t_offset in zip(
            peak_locations_converted["r_wave"]["indices"],
            peak_locations_converted["t_wave"]["offsets"],
            strict=False,
        ):
            qt_samples = t_offset - r_idx
            qt_seconds = qt_samples / sampling_rate
            if qt_seconds > 0 and (max_qt is None or qt_seconds <= max_qt):
                qt_durations.append(qt_seconds)

        qt_durations_array = np.array(qt_durations, dtype=np.float64)
        if qt_durations_array.size > 0:
            qt_stats = {
                "min": float(np.min(qt_durations_array)),
                "max": float(np.max(qt_durations_array)),
                "mean": float(np.mean(qt_durations_array)),
                "median": float(np.median(qt_durations_array)),
                "std": float(np.std(qt_durations_array)),
                "durations": qt_durations_array,
            }
        else:
            qt_stats = {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "durations": np.array([], dtype=np.float64),
            }

        return qt_stats

    except Exception as e:
        raise RuntimeError(f"Error in calculate_qt_intervals: {str(e)}") from e


__all__ = [
    "extract_ecg_features",
    "get_peak_locations",
    "calculate_qt_intervals",
]
