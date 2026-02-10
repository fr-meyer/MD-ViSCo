"""PPG feature extraction utilities used by PPGFeatureExtractor.

This module hosts the canonical PPG biomarker extraction implementation
using pyPPG. Used by extractors and evaluators for PPG-derived features.

Functions:
    - extract_ppg_features: Extract biomarkers from a PPG signal using pyPPG

Examples:
    >>> result = extract_ppg_features(ppg_signal, fs=100)
    >>> asp_deltat = result["Asp_deltaT"]
    >>> ipr = result["IPR"]

See Also:
    - src.processors.extractors: PPGFeatureExtractor
    - pyPPG: PPG analysis library

Note:
    Fiducial detection and signal processing may fail on poor-quality segments;
    such failures are caught and skipped to allow batch processing.
    Naming: Symbols such as fL, fH, N, and biomarker names (e.g. Asp, deltaT)
    match pyPPG/algorithm conventions; noqa N802/N803/N806/N815 used for those.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np
import pandas as pd
import pyPPG.preproc as preproc
from dotmap import DotMap

if TYPE_CHECKING:
    from typing import Protocol

    from numpy.typing import NDArray

    class PPGProtocol(Protocol):
        """Protocol for pyPPG.PPG class."""

        ppg: Any
        fs: float

    class PPGLike(Protocol):
        """Protocol for PPG-like objects with signal derivatives."""

        ppg: Any
        fs: float
        vpg: Any
        apg: Any
        jpg: Any
        correction: Any

    class SignalDotMap(Protocol):
        """Protocol for DotMap-like signal objects with .v and .fs."""

        v: NDArray[np.floating[Any]]
        fs: float

        def __len__(self) -> int:
            """Return the number of samples in the signal."""
            ...

        def get_row(self, i: int) -> Any:
            """Return the i-th row of fiducial data."""
            ...

    class FiducialsProtocol(Protocol):
        """Protocol for pyPPG.Fiducials class."""

        sp: Any
        on: Any
        off: Any

        def get_row(self, i: int) -> Any:
            """Return the i-th row of fiducial data."""
            ...


from pyPPG import Fiducials
from pyPPG.pack_ppg._ErrorHandler import WrongParameter
from scipy import signal
from scipy.signal import detrend
from scipy.signal import filtfilt
from scipy.signal import find_peaks
from scipy.signal import firls
from scipy.signal import firwin
from scipy.signal import kaiserord
from scipy.signal import lfilter
from scipy.signal import periodogram
from scipy.signal import resample

logger = logging.getLogger(__name__)


def extract_ppg_features(
    ppg_signal: np.ndarray | Any,
    fs: float,
    start_sig: int = 0,
    end_sig: int = -1,
    fL: float = 0.5000001,  # noqa: N803
    fH: float = 12.0,  # noqa: N803
    order: int = 4,
    sm_wins: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Extract biomarkers from a PPG signal using pyPPG.

    Args:
        ppg_signal: Raw PPG signal
        fs: Sampling frequency in Hz
        start_sig: Start sample for analysis (default: 0)
        end_sig: End sample for analysis (default: -1 for end of signal)
        fL: Lower cutoff frequency for band-pass filter
        fH: Upper cutoff frequency for band-pass filter
        order: Butterworth filter order
        sm_wins: Smoothing windows for PPG derivatives

    Returns:
        dict: Dictionary containing:
            - Asp_deltaT: Mean value of Asp/deltaT (Stiffness index) biomarker,
              or None/NaN if unavailable
            - IPR: Mean value of IPR (Instantaneous pulse rate) biomarker, or
              None/NaN if unavailable
    """
    ppg_signal = np.array(ppg_signal[:], dtype=np.float64)

    signal_dict = {
        "v": ppg_signal,
        "fs": float(fs),
        "start_sig": int(start_sig),
        "end_sig": int(end_sig),
        "filtering": True,
        "fL": float(fL),  # Lower cutoff frequency (Hz)
        "fH": float(fH),  # Upper cutoff frequency (Hz)
        "order": int(order),  # Filter order
        "sm_wins": (
            copy.deepcopy(sm_wins)
            if sm_wins is not None
            else {
                "ppg": 50,  # window for PPG signal
                "vpg": 10,  # window for PPG' signal
                "apg": 10,  # window for PPG" signal
                "jpg": 10,  # window for PPG'" signal
            }
        ),
        "name": "ppg_signal",  # Adding a name as required by the library
        "correction": pd.DataFrame(),
    }

    corr_on = ["on", "dn", "dp", "v", "w", "f"]
    signal_dict["correction"].loc[0, corr_on] = True

    signal = DotMap(signal_dict)
    s = PPG(signal)  # pyPPG API compatible

    prep = preproc.Preprocess(
        fL=s.fL,
        fH=s.fH,  # type: ignore[arg-type]  # pyPPG Preprocess expects float
        order=int(s.order),
        sm_wins=s.sm_wins,
    )

    s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(cast("Any", s))

    # Ensure all signals are numpy arrays with proper dtype
    s.ppg = np.array(s.ppg, dtype=np.float64)
    s.vpg = np.array(s.vpg, dtype=np.float64)
    s.apg = np.array(s.apg, dtype=np.float64)
    s.jpg = np.array(s.jpg, dtype=np.float64)

    try:
        fpex = FpCollection(s=s)
        fiducials = fpex.get_fiducials(s=s)
        fp = Fiducials(fp=fiducials)
        df_pw_sig, df_biomarkers_sig, biomarkers_lst_sig = get_sig_ratios(s, fp)

        # Safety check: ensure DataFrame is not empty
        if df_biomarkers_sig.empty or len(df_biomarkers_sig) == 0:
            return {"Asp_deltaT": np.nan, "IPR": np.nan}

        # Filter out NaN values before computing mean
        asp_deltat_values = (
            df_biomarkers_sig["Asp/deltaT"].dropna()
            if "Asp/deltaT" in df_biomarkers_sig.columns
            else []
        )
        ipr_values = (
            df_biomarkers_sig["IPR"].dropna()
            if "IPR" in df_biomarkers_sig.columns
            else []
        )

        return {
            "Asp_deltaT": (
                np.mean(asp_deltat_values) if len(asp_deltat_values) > 0 else np.nan
            ),
            "IPR": np.mean(ipr_values) if len(ipr_values) > 0 else np.nan,
        }
    except Exception as e:
        logger.error("Error in fiducial point extraction: %s", e, exc_info=True)
        return {"Asp_deltaT": np.nan, "IPR": np.nan}


def _check_shape_(signal, fs):
    """Check signal shape and length.

    Args:
        signal: Input signal to validate
        fs: Sampling frequency in Hz

    Raises:
        ValueError: If signal length is insufficient or signal is empty
        TypeError: If signal shape is invalid
    """
    if len(signal) < fs * 8:
        raise ValueError(
            f"Signal must be at least eight seconds long. "
            f"Got {len(signal)} samples at {fs} Hz "
            f"(duration: {len(signal) / fs:.2f} seconds)"
        )
    signal = np.array(signal)
    if len(signal.shape) == 0:
        raise ValueError("Signal must not be empty")
    if len(signal.shape) >= 3:
        raise TypeError(
            f"Signal can be 1-dimensional array or a matrix with different "
            f"signal in every column. Got signal with {len(signal.shape)} "
            f"dimensions (shape: {signal.shape})"
        )


class PPG:
    """PPG signal container with preprocessing and validation."""

    ppg: NDArray[np.float64]
    vpg: NDArray[np.float64]
    apg: NDArray[np.float64]
    jpg: NDArray[np.float64]
    fs: float
    fL: float  # noqa: N815
    fH: float  # noqa: N815
    order: int
    sm_wins: Any
    correction: Any

    def __init__(self, s=None, check_ppg_len: bool = True) -> None:
        r"""Initialize PPG signal with preprocessing and validation.

        Args:
            s: Dictionary of the PPG signal (DotMap or dict). Expected keys
                include: start_sig, end_sig, v (raw PPG values), fs (sampling
                frequency in Hz), name, ppg/vpg/apg/jpg (filtered signals),
                filtering, fL/fH (cutoff Hz), order, sm_wins (smoothing windows
                per signal), correction (DataFrame for fiducial point flags).
                Default None creates an empty mapping.
            check_ppg_len: If True, validate PPG length and sampling frequency.
        """
        # pyPPG API expects DotMap for settings
        if s is None:
            s = DotMap({})
        elif not isinstance(s, DotMap):
            s = DotMap(s)

        missing_keys = []
        if "fs" not in s:
            missing_keys.append("fs")
        if "v" not in s:
            missing_keys.append("v")

        if missing_keys:
            raise ValueError(
                f"PPG constructor requires a DotMap with required keys 'fs' (sampling "
                f"frequency) and 'v' (PPG signal values). Missing keys: "
                f"{', '.join(missing_keys)}."
            )

        if s.fs <= 0:
            raise WrongParameter("Sampling frequency should be strictly positive")

        if check_ppg_len:
            _check_shape_(s.v, s.fs)

        s.check_ppg_len = check_ppg_len

        try:
            _ = s.start_sig > 0
        except AttributeError:
            # pyPPG may omit or use different attribute names for signal bounds.
            s.start_sig = 0

        try:
            _ = s.end_sig > -1
        except AttributeError:
            # pyPPG may omit or use different attribute names for signal bounds.
            s.end_sig = -1

        # Initialise the correction for fiducial points
        if len(s.correction) < 1:
            corr_on = ["on", "dn", "dp", "v", "w", "f"]
            corr_off = ["dn"]
            correction = pd.DataFrame()
            correction.loc[0, corr_on] = True
            correction.loc[0, corr_off] = False
            s.correction = correction

        keys = s.keys()
        keys_list = list(keys)
        for i in keys_list:
            setattr(self, i, s[i])

    def get_s(self) -> pd.DataFrame:
        """Retrieve the PPG signal as a DataFrame of attributes.

        Returns:
            pd.DataFrame: Mapping of attribute names to values (PPG signal data).
        """
        keys = self.__dict__.keys()
        keys_list = list(keys)
        s = {}
        for i in keys_list:
            s[i] = getattr(self, i, None)

        return pd.DataFrame(s)


class FpCollection:
    """Collection of fiducial points for PPG signal analysis."""

    ppg: NDArray[np.float64]
    vpg: NDArray[np.float64]
    apg: NDArray[np.float64]
    jpg: NDArray[np.float64]
    fs: float

    def __init__(self, s: PPGLike | Any) -> None:
        """Initialize fiducial point collection from a PPG signal object.

        Args:
            s: PPG signal object (pyPPG.PPG compatible) whose attributes are
                copied to this collection.
        """
        keys = s.__dict__.keys()
        keys_list = list(keys)
        for i in keys_list:
            setattr(self, i, getattr(s, i))

    def get_fiducials(self, s: PPGLike | Any) -> pd.DataFrame:
        """Calculate the PPG fiducial points.

        Detects: original signal (systolic peak, onset, dicrotic notch,
        diastolic peak); 1st derivative (u, v); 2nd derivative (a, b, c, d, e).

        Args:
            s: PPG signal object (pyPPG.PPG compatible) used for correction
                settings.

        Returns:
            pd.DataFrame: Fiducial points with columns for each detected point
                type (on, sp, dn, dp, off, etc.).
        """
        # "ABD" refers the original Aboy peak detector, and "PPGdet" refers
        # the improved version.
        peak_detector = "PPGdet"

        # Extract Fiducial Points
        ppg_fp = pd.DataFrame()
        peaks, onsets = self.get_peak_onset(peak_detector)
        dicroticnotch = self.get_dicrotic_notch(peaks, onsets)

        vpg_fp = self.get_vpg_fiducials(onsets)
        apg_fp = self.get_apg_fiducials(onsets, peaks)
        jpg_fp = self.get_jpg_fiducials(onsets, apg_fp)

        diastolicpeak = self.get_diastolic_peak(onsets, dicroticnotch, apg_fp.e)

        # Merge Fiducial Points
        keys = ("on", "sp", "dn", "dp")
        dummy = np.empty(len(peaks))
        dummy.fill(np.NaN)
        n = 0
        for temp_val in (onsets, peaks, dicroticnotch, diastolicpeak):
            ppg_fp[keys[n]] = dummy
            ppg_fp.loc[0 : len(temp_val) - 1, keys[n]] = temp_val
            n = n + 1

        fiducials = pd.DataFrame()
        for temp_sig in (ppg_fp, vpg_fp, apg_fp, jpg_fp):
            for key in list(temp_sig.keys()):
                fiducials[key] = dummy
                temp_val = temp_sig[key].values
                fiducials.loc[0 : len(temp_val) - 1, key] = temp_val

        # Correct Fiducial Points
        fiducials = self.correct_fiducials(fiducials, s.correction)

        fiducials = fiducials.astype("Int64")

        # Extract pulse offsets
        offsets = copy.deepcopy(fiducials.on[1:])
        offsets.index = offsets.index - 1

        fiducials = fiducials.drop(len(fiducials) - 1)
        fiducials = fiducials.rename_axis("Index of pulse")

        # Add pulse offsets
        fiducials.insert(4, "off", offsets)

        return fiducials

    # PPG beat detector
    def get_peak_onset(
        self, peak_detector: str = "PPGdet"
    ) -> tuple[NDArray[Any] | list[Any], NDArray[Any] | list[Any]]:
        """Detect beats in a photoplethysmogram (PPG) signal.

        Uses the improved 'Automatic Beat Detection' of Aboy M et al.

        Args:
            peak_detector: Type of peak detector (e.g., 'PPGdet', 'ABD').

        Returns:
            tuple: (peaks, onsets) where peaks and onsets are 1-d arrays of
                indices of detected systolic peaks and pulse onsets.

        Reference
        ---------
        Aboy M et al., An automatic beat detection algorithm for pressure signals.
        IEEE Trans Biomed Eng 2005; 52: 1662 - 70. <https://doi.org/10.1109/TBME.2005.855725>

        Author:
        Marton A. Goda – Faculty of Biomedical Engineering,
        Technion – Israel Institute of Technology, Haifa, Israel (August 2022)

        Original Matlab implementation:
        Peter H. Charlton – King's College London (August 2017) – University
        of Cambridge (February 2022)
        <https://github.com/peterhcharlton/ppg-beats>

        Changes from Charlton's implementation:
            1) Detect Maxima:
                *  Systolic peak-to-peak distance is predicted by the heart
                   rate estimate over the preceding 10 sec window.
                *  The peak location is estimated by distances and prominences
                   of the previous peaks.
            2) Find Onsets:
                *  The onset is a local minimum, which is always calculated
                   from the peak that follows it within a given time window
            3) Tidy of Peaks and Onsets:
                *  There is a one-to-one correspondence between onsets and peaks
                *  There are only onset and peak pairs
                *  The distance between the onset and peak pairs can't be
                   smaller than 30 ms
            4) Correct Peaks and Onsets:
                * The Peaks must be the highest amplitude between two
                  consecutive pulse onsets, if not, then these are corrected
                * After the correction of Peaks, the Onsets are recalculated
        """
        # inputs
        x = copy.deepcopy(self.ppg)  # signal
        fso = self.fs

        fs = 75
        x = resample(x, int(len(self.ppg) * (fs / fso)))
        up = self.get_beat_detection_params()  # settings
        win_sec = 5
        w = fs * win_sec  # window length(number of samples)
        win_starts = np.array(list(range(0, len(x), round(0.8 * w))))
        win_starts = win_starts[0 : min(np.where([win_starts >= len(x) - w])[1])]
        win_starts = np.insert(win_starts, len(win_starts), len(x) + 1 - w)

        # before pre-processing
        hr_win = 0  # the estimated systolic peak-to-peak distance, initially it is 0
        hr_win_v = []
        x_sig = np.asarray(x, dtype=np.float64)
        px = self.detect_maxima(x_sig, 0, hr_win, peak_detector)  # detect all maxima
        if len(px) == 0:
            peaks = []
            onsets = []
            return peaks, onsets

        # detect peaks in windows
        all_p4 = []
        all_hr = np.empty(len(win_starts) - 1)
        all_hr[:] = np.NaN
        hr_past = 0  # the actual heart rate
        hrvi = 0  # heart rate variability index

        for win_no in range(0, len(win_starts) - 1):
            curr_els = np.arange(
                win_starts[win_no], win_starts[win_no] + w, dtype=np.intp
            )
            curr_x = x[curr_els]

            y1 = self.def_bandpass(
                curr_x, fs, 0.9 * up.fl_hz, 3 * up.fh_hz
            )  # Filter no.1
            hr = self.estimate_HR(
                y1, fs, up, hr_past
            )  # Estimate HR from weakly filtered signal
            hr_past = hr
            all_hr[win_no] = hr

            if (peak_detector == "PPGdet") and (hr > 40):
                if win_no == 0:
                    p1 = self.detect_maxima(y1, 0, hr_win, peak_detector)
                    tr = np.percentile(np.diff(p1), 50)
                    pks_diff = np.diff(p1)
                    pks_diff = pks_diff[pks_diff >= tr]
                    hrvi = np.std(pks_diff) / np.mean(pks_diff) * 5

                hr_win = fs / ((1 + hrvi) * 3)
                hr_win_v.append(hr_win)
            else:
                hr_win = 0

            y2 = self.def_bandpass(
                curr_x, fs, 0.9 * up.fl_hz, 2.5 * hr / 60
            )  # Filter no. 2
            y2_deriv = self.estimate_deriv(
                y2
            )  # Estimate derivative from highly filtered signal
            p2 = self.detect_maxima(
                y2_deriv, up.deriv_threshold, hr_win, peak_detector
            )  # Detect maxima in derivative
            y3 = self.def_bandpass(curr_x, fs, 0.9 * up.fl_hz, 10 * hr / 60)
            p3 = self.detect_maxima(
                y3, 50, hr_win, peak_detector
            )  # Detect maxima in moderately filtered signal
            p4 = self.find_pulse_peaks(p2, p3)
            p4 = np.unique(p4)

            if peak_detector == "PPGdet" and len(p4) > round(win_sec / 2):
                pks_diff = np.diff(p4)
                tr = np.percentile(pks_diff, 30)
                pks_diff = pks_diff[pks_diff >= tr]

                med_hr = np.median(all_hr[np.where(all_hr > 0)])
                if (med_hr * 0.5 < np.mean(pks_diff)) and (
                    med_hr * 1.5 < np.mean(pks_diff)
                ):
                    hrvi = np.std(pks_diff) / np.mean(pks_diff) * 10

            all_p4 = np.concatenate((all_p4, win_starts[win_no] + p4), axis=None)

        all_p4 = np.asarray(all_p4).astype(int)
        all_p4 = np.unique(all_p4)

        peaks, fn = self.correct_IBI(
            np.asarray(all_p4, dtype=np.float64),
            np.asarray(px, dtype=np.float64),
            float(np.median(all_hr)),
            fs,
            up,
        )

        peaks = (all_p4 / fs * fso).astype(int)
        onsets, peaks = self.find_onsets(
            self.ppg, fso, up, peaks, float(60 / np.median(all_hr) * fs)
        )

        # Correct Peaks
        for i in range(0, len(peaks) - 1):
            max_loc = np.argmax(self.ppg[onsets[i] : onsets[i + 1]]) + onsets[i]
            if peaks[i] != max_loc:
                peaks[i] = max_loc

        # Correct onsets
        onsets, peaks = self.find_onsets(
            self.ppg, fso, up, peaks, float(60 / np.median(all_hr) * fs)
        )

        temp_i = np.where(np.diff(onsets) == 0)[0]
        if len(temp_i) > 0:
            peaks = np.delete(peaks, temp_i)
            onsets = np.delete(onsets, temp_i)

        temp_i = np.where((peaks - onsets) < fso / 30)[0]
        if len(temp_i) > 0:
            peaks = np.delete(peaks, temp_i)
            onsets = np.delete(onsets, temp_i)

        return peaks, onsets

    # Maximum detector
    def detect_maxima(
        self,
        sig: NDArray[np.floating[Any]],
        percentile: int,
        hr_win: int | float | Any,
        peak_detector: str,
    ):
        """Detect all peaks in the raw and filtered signal.

        Args:
            sig: Signal array with shape (N,) where N is the length.
            percentile: Rank filter percentile for peak detection.
            hr_win: Window for adaptive heart rate estimate.
            peak_detector: Type of peak detector (e.g., 'PPGdet', 'ABD').

        Returns:
            np.ndarray: Indices of maximum peaks, 1-d array.

        Notes:
            Implementation follows Table VI pseudocode from the PPG beat detection
            literature (Aboy et al., IEEE Trans Biomed Eng 2005).
        """
        tr = np.percentile(sig, percentile)
        max_pks = np.array([])

        if peak_detector == "ABD":
            s1, s2, s3 = sig[2:], sig[1:-1], sig[0:-2]
            m = 1 + np.array(np.where((s1 < s2) & (s3 < s2)))
            max_pks = m[sig[m] > tr]

        if peak_detector == "PPGdet":
            s1, s2, s3 = sig[2:], sig[1:-1], sig[0:-2]

            max_loc = []
            min_loc = []
            max_pks = []
            intensity_v = []
            if hr_win == 0:
                m = 1 + np.array(np.where((s1 < s2) & (s3 < s2)))
                max_pks = m[sig[m] > tr]
            else:
                max_loc = find_peaks(sig, distance=hr_win)[0]
                min_loc = find_peaks(-sig, distance=hr_win)[0]

                if len(max_loc) > 0 and len(min_loc) > 0:
                    for i in range(0, len(max_loc)):
                        values = abs(max_loc[i] - min_loc)
                        min_v = min(values)
                        min_i = np.where(min_v == values)[0][0]
                        intensity_v.append(sig[max_loc[i]] - sig[min_loc[min_i]])

                    tr2 = np.mean(intensity_v) * 0.25
                    peaks_result = find_peaks(
                        sig + min(sig), prominence=tr2, distance=hr_win
                    )[0]
                    max_pks = peaks_result if len(peaks_result) > 0 else np.array([])
                else:
                    max_pks = np.array([])

        return max_pks

    # Bandpass filtering
    def def_bandpass(
        self,
        sig: NDArray[np.floating[Any]],
        fs: int | float,
        lower_cutoff: float,
        upper_cutoff: float,
    ):
        """Apply bandpass filter to the signal.

        Args:
            sig: Signal array with shape (N,) where N is the length.
            fs: Sampling frequency in Hz.
            lower_cutoff: Lower cutoff frequency in Hz.
            upper_cutoff: Upper cutoff frequency in Hz.

        Returns:
            np.ndarray: Bandpass-filtered signal, 1-d array.
        """
        # Filter characteristics: Eliminate VLFs (below resp freqs): For 4bpm cutoff
        up = DotMap()
        up.paramSet.elim_vlf.Fpass = 1.3 * lower_cutoff  # in Hz
        up.paramSet.elim_vlf.Fstop = 0.8 * lower_cutoff  # in Hz
        up.paramSet.elim_vlf.Dpass = 0.05
        up.paramSet.elim_vlf.Dstop = 0.01

        # Filter characteristics: Eliminate VHFs (above frequency content of signals)
        up.paramSet.elim_vhf.Fpass = 1.2 * upper_cutoff  # in Hz
        up.paramSet.elim_vhf.Fstop = 0.8 * upper_cutoff  # in Hz
        up.paramSet.elim_vhf.Dpass = 0.05
        up.paramSet.elim_vhf.Dstop = 0.03

        # perform BPF
        s = DotMap()
        s.v = sig
        s.fs = fs

        b, a = cast(
            "tuple[Any, Any]",
            signal.iirfilter(
                5,
                [2 * np.pi * lower_cutoff, 2 * np.pi * upper_cutoff],
                rs=60,
                btype="band",
                analog=True,
                ftype="cheby2",
            ),
        )

        b_digital, a_digital = signal.bilinear(b, a, fs=fs)
        bpf_sig = filtfilt(b_digital, a_digital, s.v)

        return bpf_sig

    # Filter the high frequency components
    def elim_vlfs(
        self,
        s: SignalDotMap,
        up: DotMap,
        lower_cutoff: float,
    ):
        """Filter the high frequency components.

        Args:
            s: Signal container (DotMap with .v and .fs).
            up: Algorithm parameters (DotMap).
            lower_cutoff: Lower cutoff frequency in Hz.

        Returns:
            DotMap: Filtered signal with .v and .fs (high-frequency filtered).
        """
        # Filter pre-processed signal to remove frequencies below resp
        # Adapted from RRest

        # Eliminate nans
        s.v[np.isnan(s.v)] = np.mean(s.v[~np.isnan(s.v)])

        # Make filter
        fc = lower_cutoff
        ripple = -20 * np.log10(up.paramSet.elim_vlf.Dstop)
        width = abs(up.paramSet.elim_vlf.Fpass - up.paramSet.elim_vlf.Fstop) / (
            s.fs / 2
        )
        [N, beta] = kaiserord(ripple, width)  # noqa: N806
        if len(s) < N * 3:
            N = round(N / 3)  # noqa: N806
        b = firwin(N, fc * 2 / s.fs, window=cast("Any", ("kaiser", beta)), scale=True)
        AMfilter = b  # noqa: N806

        s_filt = DotMap()
        try:
            s_filt.v = filtfilt(AMfilter, 1, s.v)
            s_filt.v = s.v - s_filt.v
        except Exception:
            s_filt.v = s.v

        s_filt.fs = s.fs

        return s_filt

    # Filter the low frequency components
    def elim_vhfs(
        self,
        s: SignalDotMap,
        up: DotMap,
        upper_cutoff: float,
    ):
        """Filter the low frequency components (eliminate VHFs).

        Args:
            s: Signal container (DotMap with .v and .fs).
            up: Algorithm parameters (DotMap).
            upper_cutoff: Upper cutoff frequency in Hz.

        Returns:
            DotMap: Filtered signal with .v and .fs (VHFs removed).
        """
        # Filter signal to remove VHFs
        # Adapted from RRest
        s_filt = DotMap()

        # Eliminate nans
        s.v[np.isnan(s.v)] = np.mean(s.v[~np.isnan(s.v)])

        if (up.paramSet.elim_vhf.Fpass / (s.fs / 2)) >= 1:
            s_filt.v = s.v
            s_filt.fs = s.fs
            return s_filt

        fc = upper_cutoff
        ripple = -20 * np.log10(up.paramSet.elim_vhf.Dstop)
        width = abs(up.paramSet.elim_vhf.Fpass - up.paramSet.elim_vhf.Fstop) / (
            s.fs / 2
        )
        [N, beta] = kaiserord(ripple, width)  # noqa: N806
        if len(s) < N * 3:
            N = round(N / 3)  # noqa: N806
        b = firwin(N, fc * 2 / s.fs, window=cast("Any", ("kaiser", beta)), scale=True)
        AMfilter = b  # noqa: N806

        # Remove VHFs
        s_dt = detrend(s.v)
        s_filt.v = filtfilt(AMfilter, 1, s_dt)

        return s_filt

    # Heart Rate estimation
    def estimate_HR(  # noqa: N802
        self, sig: NDArray[np.floating[Any]], fs: int, up: DotMap, hr_past: int
    ):
        """Estimate heart rate from signal in the given time window.

        Args:
            sig: Signal array with shape (N,) where N is the length.
            fs: Sampling frequency in Hz.
            up: Algorithm parameters (DotMap).
            hr_past: Previous heart rate estimate (bpm) for the window.

        Returns:
            int: Estimated heart rate in bpm.
        """
        # Estimate PSD
        blackman_window = np.blackman(len(sig))
        f, pxx = periodogram(sig, fs, window=cast("Any", blackman_window))
        ph = pxx
        fh = f

        # Extract HR
        if (hr_past / 60 < up.fl_hz) | (hr_past / 60 > up.fh_hz):
            rel_els = np.where((fh >= up.fl_hz) & (fh <= up.fh_hz))
        else:
            rel_els = np.where((fh >= hr_past / 60 * 0.5) & (fh <= hr_past / 60 * 1.4))

        rel_p = ph[rel_els]
        rel_f = fh[rel_els]
        max_el = np.where(rel_p == max(rel_p))
        hr = rel_f[max_el] * 60
        hr = int(hr[0])

        return hr

    # Estimate derivative from highly filtered signal
    def estimate_deriv(self, sig: NDArray[np.floating[Any]]):
        """Estimate derivative from highly filtered signal.

        Uses Savitzky-Golay smoothing and differentiation.

        Args:
            sig: Signal array with shape (N,) where N is the length.

        Returns:
            np.ndarray: First derivative, 1-d array.
        """
        # Savitzky Golay
        deriv_no = 1
        win_size = 5
        deriv = self.savitzky_golay(sig, deriv_no, win_size)

        return deriv

    def savitzky_golay(
        self, sig: NDArray[np.floating[Any]], deriv_no: int, win_size: int
    ):
        """Compute Savitzky-Golay derivative of the signal.

        Args:
            sig: Signal array with shape (N,) where N is the length.
            deriv_no: Derivative order (0=smoothing, 1–4 derivatives).
            win_size: Window size (5, 7, or 9).

        Returns:
            np.ndarray: Savitzky-Golay derivative, 1-d array.
        """
        if deriv_no == 0:
            # smoothing
            if win_size == 5:
                coeffs = [-3, 12, 17, 12, -3]
                norm_factor = 35
            elif win_size == 7:
                coeffs = [-2, 3, 6, 7, 6, 3, -2]
                norm_factor = 21
            elif win_size == 9:
                coeffs = [-21, 14, 39, 54, 59, 54, 39, 14, -21]
                norm_factor = 231
            else:
                raise ValueError("Unsupported window size for smoothing")
        elif deriv_no == 1:
            # first derivative
            if win_size == 5:
                coeffs = range(-2, 3)
                norm_factor = 10
            elif win_size == 7:
                coeffs = range(-3, 4)
                norm_factor = 28
            elif win_size == 9:
                coeffs = range(-4, 5)
                norm_factor = 60
            else:
                raise ValueError("Unsupported window size for first derivative")
        elif deriv_no == 2:
            # second derivative
            if win_size == 5:
                coeffs = [2, -1, -2, -1, 2]
                norm_factor = 7
            elif win_size == 7:
                coeffs = [5, 0, -3, -4, -3, 0, 5]
                norm_factor = 42
            elif win_size == 9:
                coeffs = [28, 7, -8, -17, -20, -17, -8, 7, 28]
                norm_factor = 462
            else:
                raise ValueError("Unsupported window size for second derivative")
        elif deriv_no == 3:
            # third derivative
            if win_size == 5:
                coeffs = [-1, 2, 0, -2, 1]
                norm_factor = 2
            elif win_size == 7:
                coeffs = [-1, 1, 1, 0, -1, -1, 1]
                norm_factor = 6
            elif win_size == 9:
                coeffs = [-14, 7, 13, 9, 0, -9, -13, -7, 14]
                norm_factor = 198
            else:
                raise ValueError("Unsupported window size for third derivative")
        elif deriv_no == 4:
            # fourth derivative
            if win_size == 7:
                coeffs = [3, -7, 1, 6, 1, -7, 3]
                norm_factor = 11
            elif win_size == 9:
                coeffs = [14, -21, -11, 9, 18, 9, -11, -21, 14]
                norm_factor = 143
            else:
                raise ValueError("Unsupported window size for fourth derivative")
        else:
            raise ValueError("Unsupported derivative order")

        if deriv_no % 2 == 1:
            coeffs = -np.array(coeffs)

        A = [1, 0]  # noqa: N806
        filtered_sig = lfilter(coeffs, A, sig)
        s = len(sig)
        half_win_size = np.floor(win_size * 0.5)
        zero_pad = filtered_sig[win_size] * np.ones(int(half_win_size))
        sig_in = filtered_sig[win_size - 1 : s]
        sig_end = filtered_sig[s - 1] * np.ones(int(half_win_size))
        deriv = [*zero_pad, *sig_in, *sig_end]
        deriv = deriv / np.array(norm_factor)

        return deriv

    # Pulse detection
    def find_pulse_peaks(
        self,
        p2: NDArray[np.floating[Any]] | NDArray[np.signedinteger[Any]],
        p3: NDArray[np.floating[Any]] | NDArray[np.signedinteger[Any]],
    ):
        """Detect pulse peaks from 1st- and 2nd-derivative peaks.

        Args:
            p2: Peaks of the 1st derivative, 1-d array.
            p3: Peaks of the 2nd derivative, 1-d array.

        Returns:
            np.ndarray: Pulse peak indices, 1-d array.
        """
        p4 = np.empty(len(p2))
        p4[:] = np.NaN
        for k in range(0, len(p2)):
            rel_el = np.where(p3 > p2[k])
            if np.any(rel_el) and ~np.isnan(rel_el[0][0]):
                p4[k] = p3[rel_el[0][0]]

        p4 = p4[np.where(~np.isnan(p4))]
        p4 = p4.astype(int)
        return p4

    # Correct peaks' location error
    def correct_IBI(  # noqa: N802
        self,
        p: NDArray[np.floating[Any]],
        m: NDArray[np.floating[Any]],
        hr: float,
        fs: int,
        up: DotMap,
    ) -> tuple[list[Any], NDArray[np.bool_]]:
        """Correct peak location (interbeat interval) errors.

        Args:
            p: Systolic peak indices, 1-d array.
            m: All maxima of the PPG signal, 1-d array.
            hr: Heart rate in bpm.
            fs: Sampling frequency in Hz.
            up: Algorithm parameters (DotMap).

        Returns:
            tuple: (corrected_peaks, fn) where fn marks false negatives.
        """
        # Correct peaks' location error due to pre-processing
        pc = np.empty(len(p))
        pc[:] = np.NaN
        pc1 = []
        for k in range(0, len(p)):
            temp_pk = abs(m - p[k])
            rel_el = np.where(temp_pk == min(temp_pk))
            pc1 += list(m[rel_el])

        # Correct false positives
        # identify FPs
        d = np.diff(pc1) / fs  # interbeat intervals in secs
        fp = self.find_reduced_IBIs(d, hr, up)

        # remove FPs
        pc = np.array(pc1)[fp]

        # Correct false negatives
        d = np.diff(pc) / fs  # interbeat intervals in secs
        fn = self.find_prolonged_IBIs(d, hr, up)

        pc = pc1

        return pc, fn

    def find_reduced_IBIs(  # noqa: N802
        self,
        ibis: NDArray[np.floating[Any]],
        med_hr: float,
        up: DotMap,  # noqa: N803
    ):
        """Find reduced interbeat intervals (false positives).

        Args:
            ibis: Interbeat intervals in seconds, 1-d array.
            med_hr: Median heart rate in bpm.
            up: Algorithm parameters (DotMap).

        Returns:
            list: Indices of false positive (reduced IBI) beats.
        """
        IBI_thresh = up.lower_hr_thresh_prop * 60 / med_hr  # noqa: N806
        fp = ibis < IBI_thresh
        fp = [*np.where(fp == 0)[0].astype(int)]
        return fp

    def find_prolonged_IBIs(  # noqa: N802
        self,
        ibis: NDArray[np.floating[Any]],
        med_hr: float,
        up: DotMap,  # noqa: N803
    ):
        """Find prolonged interbeat intervals (false negatives).

        Args:
            ibis: Interbeat intervals in seconds, 1-d array.
            med_hr: Median heart rate in bpm.
            up: Algorithm parameters (DotMap).

        Returns:
            np.ndarray: Boolean mask of false negative (prolonged IBI) beats.
        """
        IBI_thresh = up.upper_hr_thresh_prop * 60 / med_hr  # noqa: N806
        fn = ibis > IBI_thresh

        return fn

    def get_beat_detection_params(self):
        """Return filter and beat-detection parameters for the algorithm.

        Returns:
            DotMap: Algorithm parameters (HR bounds, thresholds, window size).
        """
        # plausible HR limits
        up = DotMap()
        up.fl = 30  # lower bound for HR
        up.fh = 200  # upper bound for HR
        up.fl_hz = up.fl / 60
        up.fh_hz = up.fh / 60

        # Thresholds
        up.deriv_threshold = 75  # originally 90
        up.upper_hr_thresh_prop = 2.25  # originally 1.75
        up.lower_hr_thresh_prop = 0.5  # originally 0.75

        # Other parameters
        up.win_size = 10  # in secs

        return up

    # Find PPG onsets
    def find_onsets(
        self,
        sig: NDArray[np.floating[Any]],
        fs: int | float,
        up: DotMap,
        peaks: NDArray[np.floating[Any]],
        med_hr: float,
    ):
        """Find pulse onsets of the PPG signal.

        Args:
            sig: Signal array with shape (N,) where N is the length.
            fs: Sampling frequency in Hz.
            up: Algorithm parameters (DotMap).
            peaks: Systolic peak indices, 1-d array.
            med_hr: Median heart rate (used for search distance).

        Returns:
            tuple: (onsets, peaks) as 1-d arrays of indices.
        """
        Y1 = self.def_bandpass(sig, fs, 0.9 * up.fl_hz, 3 * up.fh_hz)  # noqa: N806
        peaks_result = find_peaks(-Y1, distance=med_hr * 0.3)[0]
        if len(peaks_result) > 0:
            temp_oi0 = peaks_result
        else:
            temp_oi0 = np.array([])
            return np.array([]), np.array([])

        # Ensure peaks is not empty before accessing peaks[0]
        if len(peaks) == 0:
            return np.array([]), np.array([])

        null_indexes = np.where(temp_oi0 < peaks[0])
        if len(null_indexes[0]) != 0:
            if len(null_indexes[0]) == 1:
                onsets = [null_indexes[0][0]]
            else:
                onsets = [null_indexes[0][-1]]
        else:
            onsets = [peaks[0] - round(fs / 50)]

        i = 1
        while i < len(peaks):
            min_SUT = fs * 0.12  # minimum Systolic Upslope Time 120 ms  # noqa: N806
            min_DT = fs * 0.3  # minimum Diastolic Time 300 ms  # noqa: N806

            before_peak = temp_oi0 < peaks[i]
            after_last_onset = temp_oi0 > onsets[i - 1]
            SUT_time = peaks[i] - temp_oi0 > min_SUT  # noqa: N806
            DT_time = temp_oi0 - peaks[i - 1] > min_DT  # noqa: N806
            temp_oi1 = temp_oi0[
                np.where(before_peak * after_last_onset * SUT_time * DT_time)
            ]
            if len(temp_oi1) > 0:
                if len(temp_oi1) == 1:
                    onsets.append(temp_oi1[0])
                else:
                    onsets.append(temp_oi1[-1])
                i = i + 1
            else:
                peaks = np.delete(peaks, i)

        return onsets, peaks

    # Detect dicrotic notch
    def get_dicrotic_notch(
        self,
        peaks: NDArray[Any] | list[Any],
        onsets: list[Any] | NDArray[Any],
    ) -> Any:
        """Estimate dicrotic notch locations between systolic and diastolic peaks.

        Args:
            peaks: Systolic peak indices, 1-d array or list.
            onsets: Onset indices, list or 1-d array.

        Returns:
            list: Dicrotic notch sample indices.
        """
        dxx = np.diff(np.diff(self.ppg))
        fs = self.fs

        Fn = fs / 2  # Nyquist Frequency  # noqa: N806
        FcU = 20  # Cut off Frequency: 20 Hz  # noqa: N806
        FcD = FcU + 5  # Transition Frequency: 5 Hz  # noqa: N806

        n = 21  # Filter order
        f = [0, (FcU / Fn), (FcD / Fn), 1]  # Frequency band edges
        a = [1, 1, 0, 0]  # Amplitudes
        b = firls(n, f, a)

        lp_ppg = filtfilt(b, 1, dxx)

        def t_wmax(i, peaks, onsets):
            """Time from onset to systolic peak for beat i (seconds).

            Uses heuristic for first 3 beats; otherwise mean of last 3 beat durations.
            """
            if i < 3:
                HR = np.mean(np.diff(peaks)) / fs  # noqa: N806
                t_wmax = -0.1 * HR + 0.45
            else:
                t_wmax = np.mean(peaks[i - 3 : i] - onsets[i - 3 : i]) / fs
            return t_wmax

        dic_not = []
        for i in range(0, len(onsets) - 1):
            nth_beat = lp_ppg[onsets[i] : onsets[i + 1]]

            i_Pmax = peaks[i] - onsets[i]  # noqa: N806
            t_Pmax = (peaks[i] - onsets[i]) / fs  # noqa: N806
            t = np.linspace(0, len(nth_beat) - 1, len(nth_beat)) / fs
            T_beat = (len(nth_beat) - 1) / fs  # noqa: N806
            tau = (t - t_Pmax) / (T_beat - t_Pmax)
            tau[0:i_Pmax] = 0
            beta = 5

            t_w = t_wmax(i, peaks, onsets) if len(peaks) > 1 else np.NaN

            tau_wmax = (t_w - t_Pmax) / (T_beat - t_Pmax) if t_w != T_beat else 0.9

            alfa = (beta * tau_wmax - 2 * tau_wmax + 1) / (1 - tau_wmax)
            if (alfa > 4.5) or (alfa < 1.5):
                HR = np.mean(np.diff(peaks)) / fs  # noqa: N806
                t_w = -0.1 * HR + 0.45
                tau_wmax = (t_w - t_Pmax) / (T_beat - t_Pmax)
                alfa = (beta * tau_wmax - 2 * tau_wmax + 1) / (1 - tau_wmax)

            if alfa > 1:
                w = tau ** (alfa - 1) * (1 - tau) ** (beta - 1)
            else:
                w = tau * (1 - tau) ** (beta - 1)

            pp = w * nth_beat
            pp = pp[np.where(~np.isnan(pp))]
            max_pp_v = np.max(pp)
            max_pp_i = np.where(pp == max_pp_v)[0][0]
            shift = int(self.fs * 0.026)
            dic_not.append(max_pp_i + onsets[i] + shift)

        return dic_not

    # Detect diastolic peak
    def get_diastolic_peak(
        self,
        onsets: list[Any] | NDArray[Any],
        dicroticnotch: list[Any] | NDArray[Any],
        e_point: pd.Series,
    ) -> Any:
        """Estimate diastolic peak locations from onsets, dicrotic notches, e-points.

        Args:
            onsets: Onset indices, list or 1-d array.
            dicroticnotch: Dicrotic notch indices, list or 1-d array.
            e_point: E-point indices (pd.Series).

        Returns:
            np.ndarray: Diastolic peak sample indices, 1-d array.
        """
        nan_v = np.empty(len(dicroticnotch))
        nan_v[:] = np.NaN
        diastolicpeak = nan_v

        for i in range(0, len(dicroticnotch)):
            max_locs = np.array([])
            start_segment = 0
            try:
                len_segments = np.diff(onsets) * 0.80
                end_segment = int(onsets[i] + len_segments[i])
                try:
                    start_segment = int(dicroticnotch[i])
                    temp_segment = self.ppg[start_segment:end_segment]
                    max_locs, _ = find_peaks(temp_segment)

                    if len(max_locs) == 0:
                        start_segment = int(e_point[i])
                        temp_segment = self.vpg[start_segment:end_segment]
                        max_locs, _ = find_peaks(temp_segment)

                except Exception as e:
                    logger.debug("Skip segment on fiducial detection failure: %s", e)
                    pass

                if len(max_locs) > 0:
                    max_dn = max_locs[0] + start_segment
                    diastolicpeak[i] = max_dn
            except Exception as e:
                logger.debug("Skip beat on fiducial detection failure: %s", e)
                pass

        return diastolicpeak

    def get_vpg_fiducials(self, onsets: list[Any] | NDArray[Any]) -> pd.DataFrame:
        """Calculate first-derivative fiducials (u, v, w) from the PPG' signal.

        Args:
            onsets: Onset indices, list or 1-d array.

        Returns:
            pd.DataFrame: Columns u (max between onset and systolic peak), v (min
                between u and diastolic peak), w (first max after dicrotic notch).
        """
        dx = self.vpg

        nan_v = np.empty(len(onsets) - 1)
        nan_v[:] = np.NaN
        u, v, w = copy.deepcopy(nan_v), copy.deepcopy(nan_v), copy.deepcopy(nan_v)

        for i in range(0, len(onsets) - 1):
            try:
                segment = dx[onsets[i] : onsets[i + 1]]

                # u fiducial point
                max_loc = np.argmax(segment) + onsets[i]
                u[i] = max_loc

                # v fiducial point
                upper_bound_coeff = 0.66
                v_upper_bound = (
                    (onsets[i + 1] - onsets[i]) * upper_bound_coeff + onsets[i]
                ).astype(int)
                min_loc = np.argmin(dx[int(u[i]) : v_upper_bound]) + u[i] - 1
                v[i] = min_loc

                # w fiducial point
                temp_segment = dx[int(v[i]) : onsets[i + 1]]
                max_locs, _ = find_peaks(temp_segment)
                if len(max_locs) > 0:
                    max_w = max_locs[0] + v[i] - 1
                    w[i] = max_w

            except Exception as e:
                logger.debug("Skip beat on fiducial detection failure: %s", e)
                pass

        vpg_fp = pd.DataFrame({"u": [], "v": [], "w": []})
        vpg_fp.u, vpg_fp.v, vpg_fp.w = u, v, w
        return vpg_fp

    def get_apg_fiducials(
        self,
        onsets: list[Any] | NDArray[Any],
        peaks: NDArray[Any] | list[Any] | None,
    ) -> pd.DataFrame:
        """Calculate second-derivative fiducials (a, b, c, d, e, f) from PPG".

        Args:
            onsets: Onset indices, list or 1-d array.
            peaks: Systolic peak indices, 1-d array or list; optional.

        Returns:
            pd.DataFrame: Columns a, b, c, d, e, f (fiducial indices on PPG").
        """
        sig = self.ppg
        ddx = self.apg
        dddx = self.jpg

        nan_v = np.empty(len(onsets) - 1)
        nan_v[:] = np.NaN
        a, b, c, d, e, f = (
            copy.deepcopy(nan_v),
            copy.deepcopy(nan_v),
            copy.deepcopy(nan_v),
            copy.deepcopy(nan_v),
            copy.deepcopy(nan_v),
            copy.deepcopy(nan_v),
        )
        for i in range(0, len(onsets) - 1):
            try:
                # a fiducial point
                temp_pk = np.argmax(sig[onsets[i] : onsets[i + 1]]) + onsets[i] - 1
                temp_segment = ddx[onsets[i] : temp_pk]
                max_locs, _ = find_peaks(temp_segment)
                try:
                    max_loc = max_locs[np.argmax(temp_segment[max_locs])]
                except Exception:
                    max_loc = temp_segment.argmax()

                max_a = max_loc + onsets[i] - 1
                a[i] = max_a

                # b fiducial point
                temp_segment = ddx[int(a[i]) : onsets[i + 1]]
                min_locs, _ = find_peaks(-temp_segment)
                if len(min_locs) > 0:
                    min_b = min_locs[0] + a[i] - 1
                    b[i] = min_b

                # e fiducial point
                e_lower_bound = peaks[i] if peaks is not None else onsets[i]
                upper_bound_coeff = 0.6
                e_upper_bound = (
                    (onsets[i + 1] - onsets[i]) * upper_bound_coeff + onsets[i]
                ).astype(int)
                temp_segment = ddx[int(e_lower_bound) : int(e_upper_bound)]
                max_locs, _ = find_peaks(temp_segment)
                if max_locs.size == 0:
                    if peaks is not None:
                        temp_segment = ddx[int(peaks[i]) : onsets[i + 1]]
                    else:
                        temp_segment = ddx[int(onsets[i]) : onsets[i + 1]]
                    max_locs, _ = find_peaks(temp_segment)

                if len(max_locs) > 0:
                    max_loc = max_locs[np.argmax(temp_segment[max_locs])]
                else:
                    max_loc = temp_segment.argmax()

                max_e = max_loc + e_lower_bound - 1
                e[i] = max_e

                # c fiducial point
                temp_segment = ddx[int(b[i]) : int(e[i])]
                max_locs, _ = find_peaks(temp_segment)
                if max_locs.size > 0:
                    max_loc = max_locs[0]
                else:
                    temp_segment = dddx[int(b[i]) : int(e[i])]
                    min_locs, _ = find_peaks(-temp_segment)

                    if min_locs.size > 0:
                        max_loc = min_locs[np.argmin(temp_segment[min_locs])]
                    else:
                        max_locs, _ = find_peaks(temp_segment)
                        if len(max_locs) > 0:
                            max_loc = max_locs[0]
                        else:
                            max_loc = temp_segment.argmax()

                max_c = max_loc + b[i] - 1
                c[i] = max_c

                # d fiducial point
                temp_segment = ddx[int(c[i]) : int(e[i])]
                min_locs, _ = find_peaks(-temp_segment)
                if min_locs.size > 0:
                    min_loc = min_locs[np.argmin(temp_segment[min_locs])]
                    min_d = min_loc + c[i] - 1
                else:
                    min_d = max_c

                d[i] = min_d

                # f fiducial point
                temp_segment = ddx[int(e[i]) : onsets[i + 1]]
                min_locs, _ = find_peaks(-temp_segment)
                if (min_locs.size > 0) and (min_locs[0] < len(sig) * 0.8):
                    min_loc = min_locs[0]
                else:
                    min_loc = 0

                min_f = min_loc + e[i] - 1
                f[i] = min_f
            except Exception as exc:
                logger.debug("Skip beat on fiducial detection failure: %s", exc)
                pass

        apg_fp = pd.DataFrame({"a": [], "b": [], "c": [], "d": [], "e": [], "f": []})
        apg_fp.a, apg_fp.b, apg_fp.c, apg_fp.d, apg_fp.e, apg_fp.f = a, b, c, d, e, f
        return apg_fp

    def get_jpg_fiducials(
        self, onsets: list[Any] | NDArray[Any], apg_fp: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate third-derivative fiducials (p1, p2) from PPG'".

        Args:
            onsets: Onset indices, list or 1-d array.
            apg_fp: Second-derivative fiducials (DataFrame with a, b, c, d, e, f).

        Returns:
            pd.DataFrame: Columns p1 (first max after b), p2 (last min before d).
        """
        dddx = self.jpg

        nan_v = np.empty(len(onsets) - 1)
        nan_v[:] = np.NaN
        p1, p2 = copy.deepcopy(nan_v), copy.deepcopy(nan_v)

        for i in range(0, len(onsets) - 1):
            try:
                # p1 fiducial point
                ref_b = apg_fp.b[
                    np.squeeze(
                        np.where(
                            np.logical_and(
                                apg_fp.b > onsets[i], apg_fp.b < onsets[i + 1]
                            )
                        )
                    )
                ]
                if ref_b.size == 0:
                    ref_b = onsets[i]

                temp_segment = dddx[int(ref_b) : onsets[i + 1]]
                max_locs, _ = find_peaks(temp_segment)
                max_loc = max_locs[0] if len(max_locs) > 0 else temp_segment.argmax()

                max_p1 = max_loc + ref_b - 1
                p1[i] = max_p1

                # p2 fiducial point
                ref_start = p1[i]
                ref_c = apg_fp.c[
                    np.squeeze(
                        np.where(
                            np.logical_and(
                                apg_fp.c > onsets[i], apg_fp.c < onsets[i + 1]
                            )
                        )
                    )
                ]
                ref_d = apg_fp.d[
                    np.squeeze(
                        np.where(
                            np.logical_and(
                                apg_fp.d > onsets[i], apg_fp.d < onsets[i + 1]
                            )
                        )
                    )
                ]

                ref_end = 0
                min_ind = 0
                if ref_d > ref_c:
                    ref_end = ref_d
                    min_ind = -1
                elif ref_c.size > 0:
                    ref_start = ref_c
                    ref_end = onsets[i + 1]
                    min_ind = 0
                elif apg_fp.e.size > 0:
                    ref_end = onsets[i + 1]
                    min_ind = 0

                temp_segment = dddx[int(ref_start) : int(ref_end)]
                min_locs, _ = find_peaks(-temp_segment)
                if min_locs.size > 0:
                    min_p2 = min_locs[min_ind] + ref_start - 1
                else:
                    min_p2 = ref_end

                p2[i] = min_p2

            except Exception as e:
                logger.debug("Skip beat on fiducial detection failure: %s", e)
                pass

        jpg_fp = pd.DataFrame({"p1": [], "p2": []})
        jpg_fp.p1, jpg_fp.p2 = p1, p2

        return jpg_fp

    def correct_fiducials(
        self, fiducials: pd.DataFrame | None = None, correction: Any = None
    ) -> pd.DataFrame:
        """Correct fiducial point locations using correction flags.

        Args:
            fiducials: DataFrame of fiducial point names to sample indices.
            correction: DataFrame of fiducial names to bool (enable correction).

        Returns:
            pd.DataFrame: Corrected fiducials (same structure as input).
        """
        if fiducials is None:
            fiducials = pd.DataFrame()
        if correction is None:
            correction = pd.DataFrame()
        for i in range(0, len(fiducials.on)):
            # Correct onset
            if correction.on[0]:
                try:
                    win_onr = self.fs * 0.005
                    win_onl = win_onr if fiducials.on[i] > win_onr else fiducials.on[i]

                    min_loc = (
                        np.argmin(
                            self.ppg[
                                fiducials.on[i] - win_onl : fiducials.on[i] + win_onr
                            ]
                        )
                        + fiducials.on[i]
                    )
                    if fiducials.on[i] != min_loc:
                        if fiducials.a[i] > self.fs * 0.075:
                            win_a = int(self.fs * 0.075)
                        else:
                            win_a = int(fiducials.a[i])

                        fiducials.loc[i, "on"] = (
                            np.argmax(
                                self.jpg[
                                    int(fiducials.a[i]) - win_a : int(fiducials.a[i])
                                ]
                            )
                            + int(fiducials.a[i])
                            - win_a
                        )
                except Exception as e:
                    logger.debug("Skip onset correction for this beat on error: %s", e)
                    pass

            # Correct dicrotic notch
            if correction.dn[0]:
                try:
                    temp_segment = self.ppg[int(fiducials.sp[i]) : int(fiducials.dp[i])]
                    peaks_result = find_peaks(-temp_segment)[0]
                    if len(peaks_result) > 0:
                        min_dn = peaks_result + fiducials.sp[i]
                        if len(min_dn) > 0:
                            diff_dn = abs(min_dn[0] - fiducials.dp[i])
                            if diff_dn > round(self.fs / 100):
                                fiducials.loc[i, "dn"] = min_dn[0]
                                try:
                                    strt_dn = int(fiducials.sp[i])
                                    stp_dn = int(fiducials.f[i])
                                    peaks_result2 = find_peaks(
                                        -self.ppg[strt_dn:stp_dn]
                                    )[0]
                                    if len(peaks_result2) > 0:
                                        fiducials.loc[i, "dn"] = (
                                            peaks_result2[-1] + strt_dn
                                        )
                                        if fiducials.loc[i, "dn"] > min_dn[0]:
                                            fiducials.loc[i, "dn"] = min_dn[0]
                                except Exception:
                                    strt_dn = fiducials.e[i]
                                    stp_dn = fiducials.f[i]
                                    fiducials.loc[i, "dn"] = (
                                        np.argmin(self.jpg[strt_dn:stp_dn]) + strt_dn
                                    )
                                    if fiducials.loc[i, "dn"] > min_dn[0]:
                                        fiducials.loc[i, "dn"] = min_dn[0]
                except Exception as e:
                    logger.debug(
                        "Skip dicrotic notch correction for this beat on error: %s",
                        e,
                    )
                    pass

            # Correct w-point
            if correction.w[0]:
                if fiducials.w[i] > fiducials.f[i]:
                    fiducials.loc[i, "w"] = fiducials.f[i]

                if fiducials.w[i] < fiducials.e[i]:
                    with contextlib.suppress(Exception):
                        fiducials.loc[i, "w"] = [
                            np.argmax(
                                self.vpg[int(fiducials.e[i]) : int(fiducials.f[i])]
                            )
                            + fiducials.e[i]
                        ]

            # Correct v-point and w-point
            if correction.v[0] and correction.w[0] and fiducials.v[i] > fiducials.e[i]:
                try:
                    fiducials.loc[i, "v"] = [
                        np.argmin(self.vpg[int(fiducials.u[i]) : int(fiducials.e[i])])
                        + fiducials.u[i]
                    ]
                    peaks_result = find_peaks(
                        self.vpg[int(fiducials.v[i]) : int(fiducials.f[i])]
                    )[0]
                    if len(peaks_result) > 0:
                        fiducials.loc[i, "w"] = [peaks_result[0] + fiducials.v[i]]
                except Exception as e:
                    logger.debug(
                        "Skip v-point/w-point correction for this beat on error: %s",
                        e,
                    )
                    pass

            # Correct f-point
            if correction.f[0]:
                try:
                    temp_end = int(np.diff(fiducials.on[i : i + 2]) * 0.8)
                    temp_segment = self.apg[
                        int(fiducials.e[i]) : int(fiducials.on[i] + temp_end)
                    ]
                    min_f = np.argmin(temp_segment) + fiducials.e[i]

                    if fiducials.w[i] > fiducials.f[i]:
                        fiducials.loc[i, "f"] = min_f
                except Exception as e:
                    logger.debug(
                        "Skip f-point correction for this beat on error: %s", e
                    )
                    pass

        # Correct diastolic peak
        if correction.dp[0]:
            with contextlib.suppress(Exception):
                fiducials.dp = self.get_diastolic_peak(
                    fiducials.on, fiducials.dn, fiducials.e
                )

        return fiducials


class BmExtractor:
    """Biomarker extractor for PPG biomarkers from signal segments and fiducials.

    Computes requested biomarkers for a single beat using segments and fiducials.
    """

    def __init__(
        self,
        data,
        peak_value,
        peak_time,
        next_peak_value,
        next_peak_time,
        onsets_values,
        onsets_times,
        fs,
        biomarkers_lst,
        fiducials,
    ):
        """Initialize biomarker extractor for a single beat.

        Args:
            data: DotMap with signal segments (ppg, vpg, apg, jpg).
            peak_value: Amplitude of the current peak.
            peak_time: Time of the current peak.
            next_peak_value: Amplitude of the next peak.
            next_peak_time: Time of the next peak.
            onsets_values: Array of onset amplitudes.
            onsets_times: Array of onset times.
            fs: Sampling frequency in Hz.
            biomarkers_lst: List of biomarker names to compute.
            fiducials: DataFrame of fiducial points for this beat.
        """
        self.data = data
        self.peak_value = peak_value
        self.peak_time = peak_time
        self.next_peak_value = next_peak_value
        self.next_peak_time = next_peak_time
        self.onsets_values = onsets_values
        self.onsets_times = onsets_times
        self.fs = fs
        self.biomarkers_lst = biomarkers_lst
        self.fiducials = fiducials

    def map_func(self):
        """Map biomarker names to their calculation functions.

        Returns:
            dict: Mapping of biomarker name to callable.
        """
        return {
            "IPR": self.get_ipr,
            "Asp/deltaT": self.get_ratio_Asp_deltaT,
        }

    def get_biomarker_extract_func(self):
        """Compute all requested biomarkers for the current beat.

        Returns:
            list: Biomarker values in the order of biomarkers_lst.
        """
        func_map = self.map_func()
        biomarkers_vec = []

        for bm_name in self.biomarkers_lst:
            if bm_name in func_map:
                try:
                    value = func_map[bm_name]()
                    biomarkers_vec.append(value)
                except Exception:
                    biomarkers_vec.append(np.nan)
            else:
                biomarkers_vec.append(np.nan)

        return biomarkers_vec

    def get_ipr(self):
        """Calculate Instantaneous Pulse Rate (IPR).

        IPR = 60 / Tpi, where Tpi is the time between consecutive peaks.

        Returns:
            float: IPR in bpm, or NaN if calculation fails.
        """
        try:
            # Tpi is the time interval between current and next peak
            Tpi = self.next_peak_time - self.peak_time  # noqa: N806
            if Tpi > 0:
                IPR = 60.0 / Tpi  # noqa: N806
                return IPR
            else:
                return np.nan
        except Exception:
            return np.nan

    def get_ratio_Asp_deltaT(self):  # noqa: N802
        """Calculate Stiffness index: Asp/deltaT.

        Ratio of systolic peak amplitude vs. the time delay.

        Returns:
            float: Asp/deltaT value, or NaN if calculation fails.
        """
        try:
            # Asp: Amplitude of systolic peak (difference between peak and onset)
            Asp = self.peak_value - self.onsets_values[0]  # noqa: N806

            # deltaT: Time delay from onset to peak
            deltaT = self.peak_time - self.onsets_times[0]  # noqa: N806

            if deltaT > 0 and Asp != 0:
                ratio = Asp / deltaT
                return ratio
            else:
                return np.nan
        except Exception:
            return np.nan


def get_biomarkers(
    s: PPGLike,
    fp: FiducialsProtocol | Any,
    biomarkers_lst: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract biomarkers with safe peak handling.

    Only processes beats with valid next peaks to prevent NaN values.

    Args:
        s: PPG signal object (with .fs, .ppg, .vpg, .apg, .jpg).
        fp: Fiducials object with .sp, .on, .off and get_row(i).
        biomarkers_lst: List of biomarker names to extract.

    Returns:
        tuple: (df, df_biomarkers) where df has onset/offset/peak columns and
            df_biomarkers has one column per biomarker.
    """
    fs = s.fs
    ppg = s.ppg
    data = DotMap()

    df = pd.DataFrame(columns=["onset", "offset", "peak"])  # type: ignore[arg-type]  # pandas columns type
    df_biomarkers = pd.DataFrame(columns=biomarkers_lst)
    peaks = fp.sp.values
    onsets = fp.on.values
    offsets = fp.off.values

    for i in range(len(onsets)):
        onset = onsets[i]
        offset = offsets[i]
        data.ppg = ppg[int(onset) : int(offset)]
        data.vpg = s.vpg[int(onset) : int(offset)]
        data.apg = s.apg[int(onset) : int(offset)]
        data.jpg = s.jpg[int(onset) : int(offset)]
        peak = peaks[(peaks > onset) * (peaks < offset)]

        if len(peak) != 1:
            continue
        peak = peak[0]

        temp_fiducials = fp.get_row(i)
        peak_value = ppg[peak]
        peak_time = peak / fs
        onset_value = ppg[onset]
        onset_time = onset / fs

        if (peak_value - onset_value) == 0:
            continue

        offset_value = ppg[offset]
        offset_time = offset / fs

        idx_array = np.where(peaks == peak)
        idx = idx_array[0][0]
        onsets_values = np.array([onset_value, offset_value])
        onsets_times = np.array([onset_time, offset_time])

        # CRITICAL: Only process if next peak exists
        if (idx + 1) < len(peaks):
            try:
                next_peak_idx = int(peaks[idx + 1])
                next_peak_value = ppg[next_peak_idx]
                next_peak_time = next_peak_idx / fs

                nan_fidu = temp_fiducials.columns[np.where(temp_fiducials.isna())[1]]
                temp_fiducials[nan_fidu] = np.nan

                biomarkers_extractor = BmExtractor(
                    data,
                    peak_value,
                    peak_time,
                    next_peak_value,
                    next_peak_time,
                    onsets_values,
                    onsets_times,
                    fs,
                    biomarkers_lst,
                    temp_fiducials,
                )
                biomarkers_vec = biomarkers_extractor.get_biomarker_extract_func()
                lst = list(biomarkers_vec)
                df_biomarkers.loc[i] = lst
                df.loc[i] = {"onset": onset, "offset": offset, "peak": peak}
            except Exception as e:
                logger.debug("Skip beat on biomarker extraction failure: %s", e)
                pass
        # else: Skip beats without next peak (prevents NaN)

    return df, df_biomarkers


def get_sig_ratios(
    s: PPGLike,
    fp: FiducialsProtocol | Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return signal-ratio biomarkers (e.g., IPR, Asp/deltaT).

    Args:
        s: PPG signal object (pyPPG.PPG compatible).
        fp: Fiducial points object (pyPPG.Fiducials compatible).

    Returns:
        tuple: (df_pw, df_biomarkers, biomarkers_lst) where df_pw has onsets/
            offsets/peaks, df_biomarkers has biomarker values, and
            biomarkers_lst is the definition table (name, definition, unit).
    """
    biomarkers_lst = [
        ["IPR", "Instantaneous pulse rate, 60 / Tpi", "[bpm]"],
        [
            "Asp/deltaT",
            "Stiffness index, the ratio of the systolic peak amplitude vs. "
            "the time delay",
            "[nu]",
        ],
    ]

    header = ["name", "definition", "unit"]
    df_biomarkers_lst = pd.DataFrame(biomarkers_lst)
    df_biomarkers_lst.columns = header
    biomarkers_lst = df_biomarkers_lst

    df_pw, df_biomarkers = get_biomarkers(s, fp, biomarkers_lst.name)

    return df_pw, df_biomarkers, biomarkers_lst


__all__: list[str] = ["extract_ppg_features", "get_sig_ratios", "PPG", "FpCollection"]
