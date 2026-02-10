"""Feature Analysis Script for MD-ViSCo.

This script compares physiological features extracted from ground truth waveforms
with features extracted from model-generated/reconstructed waveforms. It supports
ECG features (QT intervals, BPM, QTc) and PPG features (Asp_deltaT, IPR).

Features:
- Hydra-based configuration management
- Automatic sample alignment verification
- Sampling rate validation with warnings
- Comprehensive error handling and logging
- CSV output with detailed comparison metrics
- Support for multiple directions: PPG2ECG, ECG2PPG, ABP2PPG, ABP2ECG
- Support for multiple datasets: PulseDB, UCI

Configuration:
    All parameters can be configured via Hydra config files or command-line overrides.
    See src/conf/feature_analysis_config.yaml for configuration and parameter
    definitions.

Usage Examples:

    Basic usage (PPG2ECG direction):
        python -m src.script.features.feature_analysis \
          feature_analysis.gt_features_file=\
results/features/PulseDB/ground_truth/seed_42/features_PPG2ECG.h5 \
          feature_analysis.model_name=mdvisco \
          feature_analysis.seed=42 \
          feature_analysis.direction=PPG2ECG \
          feature_analysis.dataset_name=PulseDB

    With custom output path:
        python -m src.script.features.feature_analysis \
          feature_analysis.gt_features_file=path/to/gt_features.h5 \
          feature_analysis.model_name=mdvisco \
          feature_analysis.seed=42 \
          feature_analysis.direction=PPG2ECG \
          feature_analysis.dataset_name=PulseDB \
          feature_analysis.output_csv=./custom/results/path

    ECG2PPG direction (PPG features):
        python -m src.script.features.feature_analysis \
          feature_analysis.gt_features_file=\
results/features/UCI/ground_truth/seed_42/features_ECG2PPG.h5 \
          feature_analysis.model_name=nabnet \
          feature_analysis.seed=42 \
          feature_analysis.direction=ECG2PPG \
          feature_analysis.dataset_name=UCI

    ABP-based directions:
        python -m src.script.features.feature_analysis \
          feature_analysis.gt_features_file=\
results/features/PulseDB/ground_truth/seed_42/features_ABP2PPG.h5 \
          feature_analysis.model_name=mdvisco \
          feature_analysis.seed=42 \
          feature_analysis.direction=ABP2PPG \
          feature_analysis.dataset_name=PulseDB

Output:
    CSV file with columns:
    - Metadata: sample_id, dataset, method, direction, seed
    - For ECG directions (PPG2ECG, ABP2ECG):
        * qt_intervals_gt, qt_intervals_rec, qt_intervals_mae,
          qt_intervals_unit (seconds)
        * bpm_avg_gt, bpm_avg_rec, bpm_avg_mae, bpm_avg_unit (bpm)
        * qtc_gt, qtc_rec, qtc_mae, qtc_unit (seconds)
    - For PPG directions (ECG2PPG, ABP2PPG):
        * Asp_deltaT_gt, Asp_deltaT_rec, Asp_deltaT_mae, Asp_deltaT_unit (seconds)
        * IPR_gt, IPR_rec, IPR_mae, IPR_unit (unitless)

Required Parameters:
    - gt_features_file: Path to ground truth features HDF5 file
    - model_name: Model identifier (mdvisco, patchtst, nabnet, etc.)
    - seed: Random seed integer
    - direction: Reconstruction direction (PPG2ECG, ECG2PPG, ABP2PPG, ABP2ECG)
    - dataset_name: Dataset identifier (PulseDB, UCI)

Optional Parameters:
    - features_base_path: Base path for features (default: results/features)
    - output_csv: Output directory for CSV results (default: results/features_results)

For more details, see src/conf/feature_analysis_config.yaml.
"""

import csv
import logging

# Standard library imports
import os
from typing import Any

import hydra

# Third-party imports
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

# Local imports
from src.evaluators.feature_io import load_features_only
from src.evaluators.feature_io import verify_sample_alignment

# Limit OpenMP threads for reproducible feature analysis
os.environ["OMP_NUM_THREADS"] = "2"

logger = logging.getLogger(__name__)


def calculate_qtc_bazett(
    qt: float | np.ndarray, rr: float | np.ndarray
) -> float | np.ndarray:
    """Calculate QTc using Bazett's formula.

    Args:
        qt: QT interval(s) in seconds.
        rr: RR interval(s) in seconds.

    Returns:
        QTc interval(s) in seconds (same shape as inputs).
    """
    return qt / np.sqrt(rr)


def compute_bpm(
    r_peaks: np.ndarray, sampling_rate: int = 125
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """Calculate BPM from R-peak locations.

    Args:
        r_peaks: Array of sample indices of R-peaks.
        sampling_rate: Sampling rate in Hz.

    Returns:
        (average_bpm, instantaneous_bpm, rr_intervals), or (None, None, None)
        if fewer than two peaks.
    """
    if len(r_peaks) < 2:
        return None, None, None

    rr_intervals = np.diff(r_peaks) / sampling_rate
    bpm_per_beat = 60 / rr_intervals
    bpm_avg = float(np.mean(bpm_per_beat))
    return bpm_avg, bpm_per_beat, rr_intervals


def construct_reconstructed_path(
    base_path: str, dataset_name: str, model_name: str, seed: int, direction: str
) -> str:
    """Construct the path to reconstructed features file.

    Args:
        base_path: Base path to features directory.
        dataset_name: Name of dataset ('PulseDB' or 'UCI').
        model_name: Name of the model (e.g., 'mdvisco').
        seed: Random seed number.
        direction: Direction of reconstruction (e.g., 'PPG2ECG').

    Returns:
        Full path to the reconstructed features file.
    """
    upper_path = os.path.join(
        base_path,
        dataset_name,
        model_name,
        f"seed_{seed}",
        f"features_{direction.upper()}.h5",
    )

    if not os.path.exists(upper_path):
        lower_path = os.path.join(
            base_path,
            dataset_name,
            model_name,
            f"seed_{seed}",
            f"features_{direction.lower()}.h5",
        )
        if os.path.exists(lower_path):
            return lower_path
        # If neither exists, return the uppercase path (will be handled by the caller)

    return upper_path


def extract_ecg_metrics(
    ecg_features: dict[str, Any], sampling_rate: int = 125
) -> dict[str, Any]:
    """Extract ECG metrics from features dictionary.

    Args:
        ecg_features: Dictionary containing ECG features
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with extracted metrics
    """
    metrics: dict[str, Any] = {"qt_intervals": None, "bpm_avg": None, "qtc": None}

    if "qt_intervals" in ecg_features and len(ecg_features["qt_intervals"]) > 0:
        metrics["qt_intervals"] = np.mean(ecg_features["qt_intervals"])

    if "peak_locations" in ecg_features and "r_wave" in ecg_features["peak_locations"]:
        r_wave_data = ecg_features["peak_locations"]["r_wave"]
        if "indices" in r_wave_data:
            r_peaks = r_wave_data["indices"]
            bpm_avg, _, rr_intervals = compute_bpm(r_peaks, sampling_rate)
            metrics["bpm_avg"] = bpm_avg

            if metrics["qt_intervals"] is not None and rr_intervals is not None:
                rr_mean = float(np.mean(rr_intervals))
                metrics["qtc"] = calculate_qtc_bazett(metrics["qt_intervals"], rr_mean)

    return metrics


def extract_ppg_metrics(ppg_features: dict[str, Any]) -> dict[str, Any]:
    """Extract PPG metrics from features dictionary.

    Args:
        ppg_features: Dictionary containing PPG features

    Returns:
        Dictionary with extracted metrics
    """
    metrics: dict[str, Any] = {"Asp_deltaT": None, "IPR": None}

    for key in ["Asp_deltaT", "IPR"]:
        if key in ppg_features:
            value = ppg_features[key]
            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                metrics[key] = np.mean(value)
            else:
                metrics[key] = value

    return metrics


def compare_features(
    gt_features: dict[str, Any],
    model_features: dict[str, Any],
    direction: str,
    gt_sampling_rate: int = 125,
    model_sampling_rate: int = 125,
) -> dict[str, Any]:
    """Compare ground truth and model-generated features.

    Args:
        gt_features: Ground truth features dictionary
        model_features: Model-generated features dictionary
        direction: Reconstruction direction
        gt_sampling_rate: Ground truth sampling rate in Hz
        model_sampling_rate: Model sampling rate in Hz

    Returns:
        Dictionary with comparison results
    """
    is_ecg = direction.endswith("ECG")

    comparison = {"sample_id": gt_features.get("sample_id"), "direction": direction}

    if is_ecg:
        gt_ecg = extract_ecg_metrics(
            gt_features.get("ecg_features", {}), gt_sampling_rate
        )
        model_ecg = extract_ecg_metrics(
            model_features.get("ecg_features", {}), model_sampling_rate
        )

        for metric in ["qt_intervals", "bpm_avg", "qtc"]:
            gt_val = gt_ecg.get(metric)
            model_val = model_ecg.get(metric)

            comparison[f"{metric}_gt"] = gt_val
            comparison[f"{metric}_rec"] = model_val
            comparison[f"{metric}_mae"] = (
                abs(gt_val - model_val)
                if gt_val is not None and model_val is not None
                else None
            )
    else:
        gt_ppg = extract_ppg_metrics(gt_features.get("ppg_features", {}))
        model_ppg = extract_ppg_metrics(model_features.get("ppg_features", {}))

        for metric in ["Asp_deltaT", "IPR"]:
            gt_val = gt_ppg.get(metric)
            model_val = model_ppg.get(metric)

            comparison[f"{metric}_gt"] = gt_val
            comparison[f"{metric}_rec"] = model_val
            comparison[f"{metric}_mae"] = (
                abs(gt_val - model_val)
                if gt_val is not None and model_val is not None
                else None
            )

    return comparison


def generate_csv_headers(direction: str) -> list[str]:
    """Generate CSV headers based on the reconstruction direction.

    Args:
        direction: Reconstruction direction (e.g., 'PPG2ECG', 'ECG2PPG')

    Returns:
        List of CSV headers
    """
    headers = ["sample_id", "dataset", "method", "direction", "seed"]

    is_ecg = direction.endswith("ECG")

    if is_ecg:
        ecg_metrics = ["qt_intervals", "bpm_avg", "qtc"]
        for metric in ecg_metrics:
            headers.extend(
                [f"{metric}_gt", f"{metric}_rec", f"{metric}_mae", f"{metric}_unit"]
            )
    else:
        ppg_metrics = ["Asp_deltaT", "IPR"]
        for metric in ppg_metrics:
            headers.extend(
                [f"{metric}_gt", f"{metric}_rec", f"{metric}_mae", f"{metric}_unit"]
            )

    return headers


def save_comparison_results(
    comparisons: list[dict[str, Any]],
    output_csv: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    direction: str,
) -> None:
    """Save comparison results to CSV file.

    Args:
        comparisons: List of comparison dictionaries
        output_csv: Path to save the CSV file
        dataset_name: Name of dataset
        model_name: Name of the model
        seed: Random seed number
        direction: Reconstruction direction
    """
    if not comparisons:
        logger.warning("Warning: No comparisons to save")
        return

    headers = generate_csv_headers(direction)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        is_ecg = direction.endswith("ECG")
        if is_ecg:
            metric_units = {"qt_intervals": "s", "bpm_avg": "bpm", "qtc": "s"}
            metrics = ["qt_intervals", "bpm_avg", "qtc"]
        else:
            metric_units = {"Asp_deltaT": "s", "IPR": "unitless"}
            metrics = ["Asp_deltaT", "IPR"]

        for comparison in comparisons:
            row_data = [
                comparison.get("sample_id"),
                dataset_name,
                model_name,
                direction,
                seed,
            ]

            for metric in metrics:
                row_data.extend(
                    [
                        comparison.get(f"{metric}_gt"),
                        comparison.get(f"{metric}_rec"),
                        comparison.get(f"{metric}_mae"),
                        metric_units[metric],
                    ]
                )

            writer.writerow(row_data)


def load_saved_features(csv_path: str) -> pd.DataFrame | None:
    """Load and verify the saved features from CSV file.

    Args:
        csv_path: Path to the CSV file containing saved features.

    Returns:
        DataFrame containing the loaded features, or None if file not found
        or on read error.
    """
    if not os.path.exists(csv_path):
        logger.error(f"Error: CSV file not found at {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)

        logger.info("\nLoaded CSV file information:")
        logger.info(f"Number of samples: {len(df)}")
        logger.info(f"Columns: {', '.join(df.columns)}")

        logger.info("\nFeature statistics:")
        # Exclude metadata columns and unit columns from statistics
        excluded_cols = ["sample_id", "dataset", "method", "direction", "seed"]
        excluded_cols.extend([col for col in df.columns if col.endswith("_unit")])

        for col in df.columns:
            if col not in excluded_cols:
                col_data = df[col].dropna()  # Drop NaN values before computing stats
                if len(col_data) > 0:
                    logger.info(f"\n{col}:")
                    logger.info(f"  Mean: {col_data.mean():.4f}")
                    logger.info(f"  Std:  {col_data.std():.4f}")
                    logger.info(f"  Min:  {col_data.min():.4f}")
                    logger.info(f"  Max:  {col_data.max():.4f}")
                else:
                    logger.warning(f"{col}: All values are NaN")

        return df

    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return None


@hydra.main(
    version_base=None, config_path="../../conf", config_name="feature_analysis_config"
)
def main(cfg: DictConfig) -> None:
    """Analyze and compare features from ground truth and model-generated signals.

    This function uses Hydra for configuration management, allowing flexible parameter
    overrides via command line or config files.

    Args:
        cfg: Hydra configuration object containing all parameters

    Raises:
        FileNotFoundError: If required input files are not found
        ValueError: If configuration validation fails
    """
    gt_features_file = cfg.feature_analysis.gt_features_file
    features_base_path = cfg.feature_analysis.features_base_path
    model_name = cfg.feature_analysis.model_name
    seed = cfg.feature_analysis.seed
    direction = cfg.feature_analysis.direction
    dataset_name = cfg.feature_analysis.dataset_name
    output_csv_dir = cfg.feature_analysis.output_csv

    # Normalize all user-provided paths to absolute paths relative to original
    # working directory. This ensures paths work correctly even when Hydra changes
    # the run directory
    gt_features_file = to_absolute_path(gt_features_file)
    features_base_path = to_absolute_path(features_base_path)
    output_csv_dir = to_absolute_path(output_csv_dir)

    logger.info("=" * 60)
    logger.info("Feature Analysis Configuration")
    logger.info("=" * 60)
    logger.info(f"Ground Truth File: {gt_features_file}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Direction: {direction}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Features Base Path: {features_base_path}")
    logger.info(f"Output CSV Directory: {output_csv_dir}")

    valid_directions = ["PPG2ECG", "ECG2PPG", "ABP2PPG", "ABP2ECG"]
    if direction not in valid_directions:
        logger.error(
            f"Invalid direction: {direction}. Must be one of {valid_directions}"
        )
        raise ValueError(f"Invalid direction: {direction}")

    valid_datasets = ["PulseDB", "UCI"]
    if dataset_name not in valid_datasets:
        logger.error(
            f"Invalid dataset_name: {dataset_name}. Must be one of {valid_datasets}"
        )
        raise ValueError(f"Invalid dataset_name: {dataset_name}")

    model_features_file = construct_reconstructed_path(
        features_base_path, dataset_name, model_name, seed, direction
    )

    csv_filename = f"{model_name}_{direction}_{dataset_name}_seed{seed}.csv"
    output_csv_path = os.path.join(output_csv_dir, csv_filename)

    os.makedirs(output_csv_dir, exist_ok=True)

    if not os.path.exists(gt_features_file):
        logger.error(f"Ground truth features file {gt_features_file} does not exist")
        raise FileNotFoundError(
            f"Ground truth features file {gt_features_file} does not exist"
        )

    if not os.path.exists(model_features_file):
        logger.error(f"Model features file {model_features_file} does not exist")
        raise FileNotFoundError(
            f"Model features file {model_features_file} does not exist"
        )

    try:
        logger.info(f"\nLoading ground truth features from {gt_features_file}")
        gt_data = load_features_only(gt_features_file)

        logger.info(f"Loading model features from {model_features_file}")
        model_data = load_features_only(model_features_file)

        logger.info("\nVerifying sample alignment...")
        if not verify_sample_alignment(gt_features_file, model_features_file):
            logger.error("Sample alignment verification failed!")
            raise ValueError("Sample alignment verification failed!")

        gt_sampling_rate = gt_data["metadata"]["sampling_rate"]
        model_sampling_rate = model_data["metadata"].get(
            "sampling_rate", gt_sampling_rate
        )
        if gt_sampling_rate != model_sampling_rate:
            logger.warning(
                f"Sampling rate mismatch: GT={gt_sampling_rate} Hz, Model={
                    model_sampling_rate
                } Hz. "
                f"Using separate rates for GT and model features."
            )
        logger.info(f"Ground truth sampling rate: {gt_sampling_rate} Hz")
        logger.info(f"Model sampling rate: {model_sampling_rate} Hz")

        logger.info(f"\nComparing features for {len(gt_data['features'])} samples...")
        comparisons = []

        for gt_feat, model_feat in tqdm(
            zip(gt_data["features"], model_data["features"], strict=True),
            total=len(gt_data["features"]),
            desc="Comparing features",
        ):
            comparison = compare_features(
                gt_feat, model_feat, direction, gt_sampling_rate, model_sampling_rate
            )
            comparisons.append(comparison)

        logger.info(f"\nSaving comparison results to {output_csv_path}")
        save_comparison_results(
            comparisons, output_csv_path, dataset_name, model_name, seed, direction
        )

        logger.info("\nVerifying saved results...")
        loaded_results = load_saved_features(output_csv_path)

        if loaded_results is not None:
            logger.info("\nSuccessfully verified saved results")

            logger.info("\nComparison Summary:")
            logger.info(f"Total samples compared: {len(comparisons)}")

            mae_columns = [
                col for col in loaded_results.columns if col.endswith("_mae")
            ]
            for col in mae_columns:
                mae_values = loaded_results[col].dropna()
                if len(mae_values) > 0:
                    logger.info(
                        f"{col}: {mae_values.mean():.4f} ± {mae_values.std():.4f}"
                    )

        logger.info("\nMetadata Information:")
        logger.info(
            f"Ground Truth - Sampling Rate: {gt_data['metadata']['sampling_rate']} Hz"
        )
        logger.info(
            f"Ground Truth - Normalization: {
                gt_data['metadata']['normalization_method']
            }"
        )
        logger.info(
            f"Ground Truth - Version: {
                gt_data['metadata']['feature_extraction_version']
            }"
        )

        if model_data["metadata"].get("model_info"):
            model_info = model_data["metadata"]["model_info"]
            logger.info(f"Model - Name: {model_info['model_name']}")
            logger.info(f"Model - Direction: {model_info['direction']}")
            logger.info(f"Model - Seed: {model_info['seed']}")

        logger.info("\n" + "=" * 60)
        logger.info("Feature analysis completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during feature analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
