"""Scalar extraction infrastructure for waveform post-processing.

This module provides abstract and concrete extractors for deriving structured
measurements from continuous waveform predictions. It is consumed by
`WaveformOutputProcessor` to expose metadata-driven extraction of blood
pressure, heart rate, and feature dictionaries produced by modality-specific
pipelines. Output patterns: (1) scalar dictionaries with keys mapping to
tensors shaped `[B, 1]` (e.g., `BPExtractor`); (2) feature dictionaries with
nested dicts/arrays/tensors for richer feature sets (e.g., future
`ECGFeatureExtractor`, `PPGFeatureExtractor`). To add new extractors, subclass
`ScalarExtractor`, implement `extract()`, and register a dataclass config with
Hydra's ConfigStore.

Examples:
    Direct instantiation for scalar extraction::

        extractor = BPExtractor(min_name='dbp', max_name='sbp')
        abp_waveform = torch.randn(32, 1, 1250)
        bp_values = extractor.extract(abp_waveform)
        bp_values.keys()  # dict_keys(['sbp', 'dbp'])

    Hydra instantiation (recommended)::

        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            "_target_": "src.processors.extractors.BPExtractor",
            "min_name": "dbp",
            "max_name": "sbp"
        })
        extractor = instantiate(config)
        bp_values = extractor.extract(abp_waveform)

    Processor YAML config::

        processor:
          _target_: src.processors.waveform_processor.WaveformOutputProcessor
          extractor:
            _target_: src.processors.extractors.BPExtractor
            min_name: "dbp"
            max_name: "sbp"
          extract_scalars: true

See Also:
    - src.processors.output_processor: Base processor infrastructure.
    - src.processors.waveform_processor: Waveform processor using extractors.
"""

import logging

# Standard library imports
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any

# Third-party imports
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ScalarExtractorConfig:
    """Base configuration for scalar extractors."""

    _target_: str = MISSING


@dataclass
class MinMaxExtractorConfig(ScalarExtractorConfig):
    """Configuration for MinMaxExtractor."""

    _target_: str = "src.processors.extractors.MinMaxExtractor"
    min_name: str = "min"
    max_name: str = "max"


@dataclass
class BPExtractorConfig(ScalarExtractorConfig):
    """Configuration for BPExtractor."""

    _target_: str = "src.processors.extractors.BPExtractor"
    min_name: str = "dbp"
    max_name: str = "sbp"


@dataclass
class HRExtractorConfig(ScalarExtractorConfig):
    """Configuration for `ECGFeatureExtractor` (heart-rate and ECG feature extraction).

    Attributes:
        sampling_rate: Sampling cadence of the ECG in Hz. NeuroKit2 routines
            expect high-resolution (≥250 Hz) signals; defaults to 1000 Hz to
            match `extract_ecg_features()` in `src/utils/feature_helpers/ecg.py`.
        max_qt: Physiological ceiling for QT intervals in seconds. Values above
            this threshold (default 0.6 s) are discarded to guard against noise
            and to reflect arrhythmia screening heuristics.
        fl_hz: Low-frequency cutoff (Hz) for band-pass filtering prior to peak
            delineation. Eliminates baseline wander such as respiration drift.
        fh_hz: High-frequency cutoff (Hz) for band-pass filtering. Suppresses
            high-frequency artifacts while preserving the QRS complex; defaults
            to 3 Hz for morphology-focused pipelines.
        hr_past: Optional prior heart-rate estimate (BPM) to stabilize detection
            in streaming settings. Reserved for future adaptive implementations.
        strict_mode: Whether feature extraction should raise exceptions on
            failure (`True`) or return empty/NaN-filled arrays (`False`).

    Clinical context:
        - QT interval assessment is critical for arrhythmia risk stratification.
        - Filtering choices determine fidelity of R/P/T delineation.
        - Configuration details are documented in the ECGFeatureExtractor
          class docstring.
    """

    _target_: str = "src.processors.extractors.ECGFeatureExtractor"
    sampling_rate: int = 125
    max_qt: float = 0.6
    fl_hz: float = 0.5
    fh_hz: float = 3.0
    hr_past: int | None = None
    strict_mode: bool = False


@dataclass
class PPGExtractorConfig(ScalarExtractorConfig):
    """Configuration for `PPGFeatureExtractor`.

    Attributes:
        sampling_rate: Sampling cadence of the photoplethysmogram in Hz. pyPPG
            defaults to 125 Hz, representative of wearables and bedside monitors.
        f_l: Low-frequency cutoff (Hz) for Butterworth band-pass filtering. Values
            near 0.5 Hz suppress motion-induced baseline drift.
        f_h: High-frequency cutoff (Hz) for Butterworth filtering. Default 12 Hz
            preserves systolic upstroke dynamics while attenuating sensor noise.
        order: Filter order controlling roll-off steepness (Butterworth). An
            order of 4 balances attenuation with phase characteristics.
        sm_wins: Smoothing window sizes (samples) applied to the original PPG
            and its derivatives (VPG/APG/JPG). Defaults mirror pyPPG empirical
            settings: 50-sample PPG smoothing and 10-sample derivative windows.
        strict_mode: Whether to raise extraction exceptions or tolerate partial
            outputs when fiducial detection fails.

    Clinical context:
        - VPG/APG/JPG derivatives support detection of systolic peaks and notch
          dynamics for arterial stiffness assessment.
        - Asp_deltaT (onset-to-peak timing) and IPR (inflection point ratio) are
          documented in the PPGFeatureExtractor class docstring.

    Note:
        JPG (jerk photoplethysmogram) denotes the third derivative of the PPG
        signal, adopting standard physics nomenclature (position, velocity,
        acceleration, jerk). The implementation is mirrored in
        `src/utils/ppg_features.py`.
    """

    _target_: str = "src.processors.extractors.PPGFeatureExtractor"
    sampling_rate: float = 125.0
    f_l: float = 0.5000001
    f_h: float = 12.0
    order: int = 4
    sm_wins: dict[str, int] = field(
        default_factory=lambda: {"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10}
    )
    strict_mode: bool = True


class ScalarExtractor(ABC):
    """Abstract base class for extracting structured values from waveform predictions.

    The contract intentionally supports two families of extractors:

    1. Scalar extractors (current production path) return dictionaries of tensors
       shaped `[B, 1]`, e.g., `BPExtractor` for systolic/diastolic pressure.
    2. Feature extractors (in design) return dictionaries that may include nested
       dicts, NumPy arrays, or tensors that summarise fiducial measurements for
       ECG/PPG modalities. These outputs are documented in their respective
       class docstrings.

    Subclasses must implement :meth:`extract` with deterministic logic and should
    document their schemas in the class docstring.

    Examples:
        # Blood pressure extraction from ABP (scalar dictionary)
        >>> bp_extractor = BPExtractor()
        >>> abp_waveform = torch.randn(32, 1, 1250)
        >>> bp_values = bp_extractor.extract(abp_waveform)
        >>> bp_values['sbp'].shape
        torch.Size([32, 1])

        # ECG feature extraction (future implementation)
        >>> ecg_extractor = ECGFeatureExtractor()
        >>> ecg_waveform = torch.randn(4, 1, 5000)
        >>> ecg_features = ecg_extractor.extract(ecg_waveform)
        >>> sorted(ecg_features.keys())
        ['mean_ecg_quality', 'peak_locations', 'qt_intervals']

        # PPG feature extraction (future implementation)
        >>> ppg_extractor = PPGFeatureExtractor()
        >>> ppg_waveform = torch.randn(8, 1, 1250)
        >>> ppg_features = ppg_extractor.extract(ppg_waveform)
        >>> sorted(ppg_features.keys())
        ['Asp_deltaT', 'IPR']

    Notes:
        - Extraction should be deterministic and reproducible for testing.
        - Consider computational efficiency; extraction runs during inference.
        - Called by `WaveformOutputProcessor.process()` when
          `metadata.extract_scalars` is true.
        - For scalar extractors, ensure tensors are `[B, 1]`. Feature extractors
          may emit richer structures but must document their contract.
    """

    @abstractmethod
    def extract(self, waveform: Tensor) -> dict[str, Any]:
        """Extract structured values from a waveform tensor.

        This method performs the core extraction logic to derive scalar or
        feature-level summaries from continuous waveform predictions. The specific
        values returned depend on the subclass implementation (e.g., SBP/DBP for
        arterial blood pressure, QT intervals for ECG).

        Args:
            waveform (Tensor): Input waveform tensor with shape [B, C, T] where:
                - B = batch size
                - C = number of channels (typically 1 for single-lead signals)
                - T = time steps (sequence length)

        Returns:
            Dict[str, Union[Tensor, np.ndarray, Dict[str, Any]]]: Dictionary mapping
            descriptive names to extracted values.
                - Scalar extractors return tensors shaped `[B, 1]`.
                - Feature extractors may return NumPy arrays (e.g., QT intervals),
                  nested dictionaries (e.g., peak locations), or tensors.
                - Downstream processors must handle both patterns; see class
                  docstrings for details.

        Raises:
            ValueError: If waveform shape is invalid (not 3D, empty batch, etc.)
            RuntimeError: If extraction algorithm fails (e.g., no peaks detected)

        Examples:
            # Blood pressure extraction from ABP
            >>> bp_extractor = BPExtractor()
            >>> abp_waveform = torch.randn(32, 1, 1250)
            >>> bp_values = bp_extractor.extract(abp_waveform)
            >>> bp_values.keys()
            dict_keys(['sbp', 'dbp'])
            >>> bp_values['sbp'].shape, bp_values['dbp'].shape
            (torch.Size([32, 1]), torch.Size([32, 1]))

            # ECG feature extraction (future implementation)
            >>> ecg_extractor = ECGFeatureExtractor()
            >>> ecg_features = ecg_extractor.extract(torch.randn(2, 1, 5000))
            >>> sorted(ecg_features.keys())
            ['mean_ecg_quality', 'peak_locations', 'qt_intervals']
        """


class MinMaxExtractor(ScalarExtractor):
    """Generic extractor that computes minimum and maximum values from waveforms.

    This extractor performs dimension reduction along the time axis to extract
    the minimum and maximum values from waveform tensors. It serves as a base
    class for extractors that need min/max logic (e.g., blood pressure, amplitude
    features).

    The output key names can be customized via constructor parameters, allowing
    subclasses to provide domain-specific naming (e.g., 'sbp'/'dbp' for blood
    pressure instead of generic 'max'/'min').

    Shape transformations: [B, C, T] → [B, C] (via min/max) → [B, 1] (first channel)

    Examples:
        # Generic min/max extraction with default names
        >>> extractor = MinMaxExtractor()
        >>> waveform = torch.randn(32, 1, 1250)
        >>> values = extractor.extract(waveform)
        >>> values.keys()
        dict_keys(['max', 'min'])
        >>> values['max'].shape
        torch.Size([32, 1])

        # Custom key names
        >>> extractor = MinMaxExtractor(min_name='min_val', max_name='max_val')
        >>> waveform = torch.randn(32, 1, 1250)
        >>> values = extractor.extract(waveform)
        >>> values.keys()
        dict_keys(['max_val', 'min_val'])

    Attributes:
        min_name (str): Key name for minimum value in output dictionary
        max_name (str): Key name for maximum value in output dictionary
        logger (logging.Logger): Logger instance for debugging and warnings
    """

    def __init__(self, min_name: str = "min", max_name: str = "max"):
        """Initialize MinMaxExtractor with customizable output key names.

        Args:
            min_name (str): Key name for minimum value in output dictionary.
            max_name (str): Key name for maximum value in output dictionary.
        """
        self.min_name = min_name
        self.max_name = max_name
        logger.debug(
            f"Initialized MinMaxExtractor with min_name='{min_name}', "
            f"max_name='{max_name}'"
        )

    def extract(self, waveform: Tensor) -> dict[str, Tensor]:
        """Extract minimum and maximum values from waveform along time dimension.

        This method computes the min and max values along the time dimension (dim=2)
        of the input waveform, then normalizes the output shape to [B, 1] for
        consistent downstream processing.

        The extraction follows these steps:
        1. Validate input shape is 3D [B, C, T]
        2. Compute max along time dimension: [B, C, T] → [B, C]
        3. Compute min along time dimension: [B, C, T] → [B, C]
        4. Normalize to [B, 1] by taking first channel only
        5. Return dictionary with customizable key names

        Multi-channel behavior:
            For multi-channel inputs (C>1), the extractor uses the first channel only
            with a warning, ensuring [B, 1] output shape for downstream compatibility.

        Args:
            waveform (Tensor): Input waveform tensor with shape [B, C, T] where:
                - B = batch size (must be > 0)
                - C = number of channels (typically 1)
                - T = time steps (must be > 0)

        Returns:
            Dict[str, Tensor]: Dictionary with min/max values
                - Keys: self.max_name and self.min_name
                - Values: Tensors with shape [B, 1] (guaranteed)

        Raises:
            ValueError: If waveform is not 3D tensor
            ValueError: If waveform has zero batch size
            ValueError: If waveform has zero time steps
            ValueError: If waveform has zero channels

        Input Validation:
            This method performs strict validation on the input waveform:
            - Must be 3D tensor with shape [B, C, T]
            - Batch size (B) must be > 0
            - Time steps (T) must be > 0
            - Number of channels (C) must be > 0
            Each validation failure raises a ValueError with descriptive message.

        Examples:
            # Single-channel waveform
            >>> extractor = MinMaxExtractor()
            >>> waveform = torch.randn(32, 1, 1250)
            >>> values = extractor.extract(waveform)
            >>> values['max'].shape, values['min'].shape
            (torch.Size([32, 1]), torch.Size([32, 1]))

            # Multi-channel waveform (takes first channel with warning)
            >>> waveform = torch.randn(32, 3, 1250)
            >>> values = extractor.extract(waveform)
            >>> values['max'].shape
            torch.Size([32, 1])

        Notes:
            - Multi-channel inputs log a warning and use only the first channel
            - Always returns [B, 1] shape for consistent downstream processing
            - Used by WaveformOutputProcessor.process()
            - Produces identical results to the legacy `extract_bp_values`
              helper for single-channel inputs
        """
        # Step 1: Validate input shape
        if waveform.ndim != 3:
            raise ValueError(
                f"Expected 3D waveform tensor with shape [B, C, T], "
                f"but got {waveform.ndim}D tensor with shape {waveform.shape}"
            )

        batch_size, num_channels, time_steps = waveform.shape

        if batch_size == 0:
            raise ValueError(
                f"Waveform batch size must be > 0, but got batch_size={batch_size}"
            )

        if time_steps == 0:
            raise ValueError(
                f"Waveform time steps must be > 0, but got time_steps={time_steps}"
            )

        if num_channels == 0:
            raise ValueError(
                f"Waveform must have at least one channel, but got "
                f"num_channels={num_channels}"
            )

        logger.debug(f"Extracting min/max from waveform with shape {waveform.shape}")

        # Step 2: Extract max and min values along time dimension (dim=2)
        max_val = torch.max(waveform, dim=2)[0]  # [B, C, T] → [B, C]
        min_val = torch.min(waveform, dim=2)[0]  # [B, C, T] → [B, C]

        # Step 3: Normalize to [B, 1] shape by taking first channel
        if max_val.shape[1] > 1:
            logger.warning(
                f"Multi-channel waveform detected with {num_channels} channels. "
                f"Using first channel only for extraction."
            )
        max_val = max_val[:, 0:1]  # [B, C] → [B, 1]
        min_val = min_val[:, 0:1]  # [B, C] → [B, 1]

        # Step 4: Build output dictionary with customizable key names
        result = {self.max_name: max_val, self.min_name: min_val}  # [B, 1]  # [B, 1]

        logger.debug(
            f"Extracted {self.max_name} shape: {max_val.shape}, "
            f"{self.min_name} shape: {min_val.shape}"
        )

        return result


class BPExtractor(MinMaxExtractor):
    """Extractor for SBP and DBP from ABP waveforms.

    This extractor extends MinMaxExtractor with blood pressure-specific
    naming conventions. It extracts systolic blood pressure (SBP) as the
    maximum pressure value and diastolic blood pressure (DBP) as the minimum
    pressure value from arterial blood pressure (ABP) waveforms.

        Input/Output Normalization Contract:
        BPExtractor expects its input waveform tensor to be in the same
        normalized space as the model outputs (e.g., [0,1] when using
        `global_minmax` normalization). The scalar outputs (sbp, dbp) will be
        in that same normalized scale. Downstream processors with
        denormalize=True will convert these scalars back to mmHg using
        global_min/global_max. Therefore, BPExtractor should not perform any
        internal un-normalization.

    The extraction produces identical results to the deprecated
    `extract_bp_values` helper function from bp_utils.py for single-channel
    inputs only, ensuring backward compatibility with existing code.

    Medical Context:
        - SBP (Systolic Blood Pressure): Peak pressure during cardiac contraction
        - DBP (Diastolic Blood Pressure): Trough pressure during cardiac relaxation
        - Measured in mmHg (millimeters of mercury) after denormalization
        - Typically SBP ranges 90-140 mmHg, DBP ranges 60-90 mmHg

    Examples:
        # Extract blood pressure from normalized ABP waveform
        >>> extractor = BPExtractor()
        >>> abp_waveform = torch.randn(32, 1, 1250)  # Normalized [0,1] waveform
        >>> bp_values = extractor.extract(abp_waveform)
        >>> bp_values.keys()
        dict_keys(['sbp', 'dbp'])
        >>> bp_values['sbp'].shape
        torch.Size([32, 1])
        # bp_values['sbp'] and bp_values['dbp'] are in normalized scale [0,1]
        # Downstream processors with denormalize=True convert to mmHg using
        # global_min/global_max

        # Example with normalized waveform values
        >>> abp_waveform = torch.tensor(
        ...     [[[0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4]]]
        ... )  # [0,1] normalized
        >>> bp_values = extractor.extract(abp_waveform)
        >>> bp_values['sbp'].item()  # Maximum value in normalized scale
        0.6
        >>> bp_values['dbp'].item()  # Minimum value in normalized scale
        0.2
        # These normalized values will be denormalized to mmHg by downstream processors

    Notes:
        - Extends MinMaxExtractor with min_name='dbp', max_name='sbp'
        - Returns dict with 'sbp' and 'dbp' keys, both shaped [B, 1] unconditionally
        - Multi-channel ABP inputs use first channel only with warning
        - Replaces the legacy helper with identical behavior for
          single-channel inputs only
        - No need to override extract() - inherits from MinMaxExtractor
        - Input waveforms must be in the same normalized space as model
          outputs (e.g., [0,1])
        - Output scalars are in normalized scale and require downstream
          denormalization for mmHg
        - BPExtractor does not perform input validation or range checking;
          it assumes correct configuration
    """

    def __init__(self, min_name: str = "dbp", max_name: str = "sbp"):
        """Initialize BPExtractor with blood pressure-specific key names.

        Args:
            min_name (str): Key name for minimum value in output dictionary.
            max_name (str): Key name for maximum value in output dictionary.

        The constructor configures the parent MinMaxExtractor to use:
        - max_name='sbp' (systolic is the maximum pressure)
        - min_name='dbp' (diastolic is the minimum pressure)
        """
        super().__init__(min_name=min_name, max_name=max_name)
        logger.info("Initialized BPExtractor for SBP/DBP extraction")


class ECGFeatureExtractor(ScalarExtractor):
    """Feature extractor for ECG waveforms leveraging legacy NeuroKit2 utilities.

    This extractor enforces single-channel input, converts tensors to NumPy,
    and iterates per sample to compute fiducial markers. Output schema is
    documented in the class docstring and extract() method.
    """

    def __init__(
        self,
        sampling_rate: int = 1000,
        max_qt: float = 0.6,
        fl_hz: float = 0.5,
        fh_hz: float = 3.0,
        hr_past: int | None = None,
        strict_mode: bool = True,
    ):
        """Initialize ECG feature extractor parameters.

        Args:
            sampling_rate (int): ECG sampling rate in Hz.
            max_qt (float): Maximum QT interval to retain in seconds.
            fl_hz (float): Low cutoff frequency for preprocessing filter.
            fh_hz (float): High cutoff frequency for preprocessing filter.
            hr_past (Optional[int]): Reserved for future temporal smoothing.
            strict_mode (bool): Raise on errors if True, otherwise warn.
        """
        self.sampling_rate = sampling_rate
        self.max_qt = max_qt
        self.fl_hz = fl_hz
        self.fh_hz = fh_hz
        self.hr_past = hr_past
        self.strict_mode = strict_mode
        logger.debug(
            "Initialized ECGFeatureExtractor with sampling_rate=%s, max_qt=%s, "
            "fl_hz=%s, fh_hz=%s, hr_past=%s, strict_mode=%s",
            sampling_rate,
            max_qt,
            fl_hz,
            fh_hz,
            hr_past,
            strict_mode,
        )

    def extract(self, waveform: Tensor) -> dict[str, Any]:
        """Extract peak locations, QT intervals, and signal quality scores.

        Args:
            waveform (Tensor): ECG waveform tensor of shape [B, C, T].

        Returns:
            Dict[str, Union[np.ndarray, Dict[str, Any]]]: Feature dictionary.
        """
        if waveform.ndim != 3:
            raise ValueError(
                "ECGFeatureExtractor expects 3D waveform tensor [B, C, T], "
                f"received {waveform.ndim}D tensor with shape {waveform.shape}"
            )

        batch_size, num_channels, time_steps = waveform.shape
        if batch_size == 0:
            raise ValueError("Waveform batch must be non-empty for ECG extraction.")
        if time_steps == 0:
            logger.warning(
                "Waveform contains zero time steps; skipping ECG feature extraction."
            )
            return {}
        if num_channels != 1:
            message = (
                "ECGFeatureExtractor requires single-channel input; "
                f"received {num_channels} channels."
            )
            if self.strict_mode:
                raise ValueError(message)
            logger.warning("%s Using first channel only.", message)
            waveform = waveform[:, 0:1, :]

        batch_np = waveform.detach().cpu().numpy()

        results_peak_locations: list[dict[str, Any]] = []
        results_qt_intervals: list[np.ndarray] = []
        results_quality: list[float] = []

        from src.utils.feature_helpers.ecg import calculate_qt_intervals
        from src.utils.feature_helpers.ecg import extract_ecg_features
        from src.utils.feature_helpers.ecg import get_peak_locations

        for index in range(batch_size):
            sample = batch_np[index, 0]
            try:
                signals_df, info = extract_ecg_features(
                    sample,
                    sampling_rate=self.sampling_rate,
                    fl_hz=self.fl_hz,
                    fh_hz=self.fh_hz,
                    strict_mode=self.strict_mode,
                )

                # Check if fallback occurred (only when strict_mode=False in
                # extract_ecg_features)
                if info.get("status") == "fallback":
                    if self.strict_mode:
                        error_msg = info.get("error", "Unknown error")
                        raise RuntimeError(
                            f"ECG feature extraction fallback occurred for "
                            f"sample {index}: {error_msg}. Set strict_mode=False "
                            f"to allow fallback behavior."
                        )
                    logger.warning(
                        "ECG feature extraction fallback occurred for sample %s: %s",
                        index,
                        info.get("error", "Unknown error"),
                    )

                peak_locations = get_peak_locations(signals_df)
                qt_metrics = calculate_qt_intervals(
                    peak_locations,
                    sampling_rate=self.sampling_rate,
                    max_qt=self.max_qt,
                )

                converted_peaks = self._convert_peak_locations_to_numpy(peak_locations)
                qt_array = np.asarray(
                    qt_metrics.get("durations", np.array([])),
                    dtype=np.float64,
                )

                quality_value = np.nan
                if isinstance(signals_df, dict):
                    logger.debug(
                        "Unexpected dict returned for ECG signals at index %s; "
                        "skipping quality computation.",
                        index,
                    )
                elif "ECG_Quality" in signals_df:
                    quality_series = signals_df["ECG_Quality"].to_numpy(
                        dtype=np.float64
                    )
                    if quality_series.size:
                        valid_quality = quality_series[~np.isnan(quality_series)]
                        if valid_quality.size:
                            quality_value = float(valid_quality.mean())

                results_peak_locations.append(converted_peaks)
                results_qt_intervals.append(qt_array)
                results_quality.append(float(quality_value))
            except Exception as err:  # pragma: no cover - logging aid
                if self.strict_mode:
                    raise
                logger.warning(
                    "ECG feature extraction failed for sample %s: %s",
                    index,
                    err,
                )
                results_peak_locations.append({})
                results_qt_intervals.append(np.array([], dtype=np.float64))
                results_quality.append(float(np.nan))

        return {
            "peak_locations": results_peak_locations,
            "qt_intervals": results_qt_intervals,
            "mean_ecg_quality": np.asarray(results_quality, dtype=np.float64),
        }

    @staticmethod
    def _convert_peak_locations_to_numpy(peaks: dict[str, Any]) -> dict[str, Any]:
        """Recursively convert peak location structures to NumPy arrays.

        Args:
            peaks (Dict[str, Any]): Nested peak location mapping.

        Returns:
            Dict[str, Any]: Peak locations with NumPy arrays for numerical values.
        """

        def _to_array(value: Any) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            if value is None:
                return np.array([], dtype=np.float64)
            return np.asarray(value, dtype=np.float64)

        preferred_order = ["r_wave", "p_wave", "t_wave", "q_wave", "s_wave"]
        ordered_keys = [name for name in preferred_order if name in peaks]
        ordered_keys.extend([name for name in peaks if name not in ordered_keys])

        converted: dict[str, Any] = {}
        for wave_name in ordered_keys:
            wave_content = peaks[wave_name]
            if isinstance(wave_content, dict):
                converted_wave: dict[str, Any] = {}
                for key, val in wave_content.items():
                    if isinstance(val, dict):
                        preferred_suborder = ["indices", "values"]
                        ordered_subkeys = [
                            sub_key for sub_key in preferred_suborder if sub_key in val
                        ]
                        ordered_subkeys.extend(
                            [
                                sub_key
                                for sub_key in val
                                if sub_key not in ordered_subkeys
                            ]
                        )
                        converted_wave[key] = {
                            sub_key: _to_array(val[sub_key])
                            for sub_key in ordered_subkeys
                        }
                    else:
                        converted_wave[key] = _to_array(val)
                converted[wave_name] = converted_wave
            else:
                converted[wave_name] = _to_array(wave_content)
        return converted


class PPGFeatureExtractor(ScalarExtractor):
    """Feature extractor for PPG waveforms leveraging pyPPG-based utilities.

    Produces Asp_deltaT and IPR metrics per batch sample and manages strict
    versus tolerant error-handling consistent with historical feature
    extraction scripts.
    """

    def __init__(
        self,
        sampling_rate: float = 125.0,
        f_l: float = 0.5000001,
        f_h: float = 12.0,
        order: int = 4,
        sm_wins: dict[str, int] | None = None,
        strict_mode: bool = True,
    ):
        """Initialize PPGFeatureExtractor parameters.

        Args:
            sampling_rate (float): PPG sampling rate in Hz.
            f_l (float): Low cutoff frequency for band-pass filter.
            f_h (float): High cutoff frequency for band-pass filter.
            order (int): Butterworth filter order.
            sm_wins (Optional[Dict[str, int]]): Smoothing window sizes.
            strict_mode (bool): Raise on errors if True, otherwise warn.
        """
        self.sampling_rate = sampling_rate
        self.f_l = f_l
        self.f_h = f_h
        self.order = order
        self.sm_wins = sm_wins or {
            "ppg": 50,
            "vpg": 10,
            "apg": 10,
            "jpg": 10,
        }
        self.strict_mode = strict_mode
        logger.debug(
            "Initialized PPGFeatureExtractor with sampling_rate=%s, f_l=%s, "
            "f_h=%s, order=%s, sm_wins=%s, strict_mode=%s",
            sampling_rate,
            f_l,
            f_h,
            order,
            self.sm_wins,
            strict_mode,
        )

    def extract(self, waveform: Tensor) -> dict[str, Any]:
        """Extract arterial stiffness metrics from PPG waveform batch.

        Args:
            waveform (Tensor): PPG waveform tensor of shape [B, C, T].

        Returns:
            Dict[str, np.ndarray]: Dictionary containing Asp_deltaT and IPR arrays.
        """
        if waveform.ndim != 3:
            raise ValueError(
                "PPGFeatureExtractor expects 3D waveform tensor [B, C, T], "
                f"received {waveform.ndim}D tensor with shape {waveform.shape}"
            )

        batch_size, num_channels, time_steps = waveform.shape
        if batch_size == 0:
            raise ValueError("Waveform batch must be non-empty for PPG extraction.")
        if time_steps == 0:
            logger.warning(
                "Waveform contains zero time steps; skipping PPG feature extraction."
            )
            return {}
        if num_channels != 1:
            message = (
                "PPGFeatureExtractor requires single-channel input; "
                f"received {num_channels} channels."
            )
            if self.strict_mode:
                raise ValueError(message)
            logger.warning("%s Using first channel only.", message)
            waveform = waveform[:, 0:1, :]

        signals = waveform.detach().cpu().numpy()

        asp_values: list[float] = []
        ipr_values: list[float] = []

        from src.utils.feature_helpers.ppg import extract_ppg_features

        for index in range(batch_size):
            sample = signals[index, 0]
            try:
                features = extract_ppg_features(
                    sample,
                    fs=self.sampling_rate,
                    fL=self.f_l,
                    fH=self.f_h,
                    order=self.order,
                    sm_wins=self.sm_wins,
                )
                # Handle None from extract_ppg_features (extraction can fail)
                asp_raw = features.get("Asp_deltaT", np.nan)
                ipr_raw = features.get("IPR", np.nan)
                asp = float(asp_raw if asp_raw is not None else np.nan)
                ipr = float(ipr_raw if ipr_raw is not None else np.nan)

                # In strict mode, fail immediately on first failure
                if np.isnan(asp) and np.isnan(ipr):
                    if self.strict_mode:
                        raise ValueError(
                            "PPG feature extraction returned NaN for "
                            f"Asp_deltaT and IPR for sample {index} "
                            f"(batch size: {batch_size}). "
                            "Set strict_mode=False to allow NaN for "
                            "failed extractions."
                        )
                    else:
                        logger.warning(
                            "PPG feature extraction returned NaN for "
                            "Asp_deltaT and IPR for sample %s "
                            "(strict_mode=False, continuing)",
                            index,
                        )

                asp_values.append(asp)
                ipr_values.append(ipr)
            except Exception as err:  # pragma: no cover - logging aid
                if self.strict_mode:
                    # In strict mode, re-raise immediately
                    raise
                logger.warning(
                    "PPG feature extraction failed for sample %s: %s",
                    index,
                    err,
                )
                asp_values.append(np.nan)
                ipr_values.append(np.nan)

        return {
            "Asp_deltaT": np.asarray(asp_values, dtype=np.float64),
            "IPR": np.asarray(ipr_values, dtype=np.float64),
        }


# Register with Hydra ConfigStore
cs = ConfigStore.instance()

cs.store(group="extractor", name="base_scalar_extractor", node=ScalarExtractorConfig)
cs.store(group="extractor", name="base_minmax_extractor", node=MinMaxExtractorConfig)
cs.store(group="extractor", name="base_bp_extractor", node=BPExtractorConfig)
cs.store(group="extractor", name="base_ecg_feature_extractor", node=HRExtractorConfig)
cs.store(group="extractor", name="base_ppg_feature_extractor", node=PPGExtractorConfig)
