"""Shared data structures and I/O utilities for feature extraction workflows.

The `ExtractionConfig` dataclass and I/O functions (`save_features_only`,
`load_features_only`, `verify_sample_alignment`) are shared utilities used by
both the evaluator and any future feature extraction workflows. They handle
HDF5 persistence for extracted features with full metadata traceability.
"""

import json
import logging
import os

# Standard library imports
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast

# Third-party imports
import h5py

# Local imports
from src.utils.utils_preprocessing import safe_create_dataset

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction.

    This dataclass is shared between the evaluator and standalone utilities.
    """

    sampling_rate: int = 125
    real_time: bool = False
    strict_mode: bool = True
    device: str = "cpu"
    normalize_signals: bool = True
    trim_edges: bool = False  # For PulseDB signals
    trim_samples: int = 15
    origin: Literal["ground_truth", "model_generated"] = "ground_truth"

    # Optional processor / extractor metadata populated by evaluators.
    processor_class: str | None = None
    processor_config: dict[str, Any] | None = None
    extractor_class: str | None = None
    extractor_config: dict[str, Any] | None = None


def save_features_only(
    features: dict[str, Any],
    output_path: str,
    config: ExtractionConfig,
    model_info: dict | None = None,
):
    """Save only extracted features with minimal required metadata.

    When invoked by the processor-based evaluator, additional metadata fields
    (``processor_class``, ``processor_config``, ``extractor_class``,
    ``extractor_config``) are embedded for reproducibility.

    Args:
        features: Dictionary containing extracted features.
        output_path: Path to save the HDF5 file.
        config: Feature extraction configuration (augmented with processor
            metadata when available).
        model_info: Optional model information for generated features.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Saving features to %s", output_path)
    with h5py.File(output_path, "w") as f:
        # 1. IMPERATIVE: Store minimal metadata
        metadata = f.create_group("metadata")
        metadata.attrs["sampling_rate"] = config.sampling_rate
        metadata.attrs["normalization_method"] = (
            "min_max" if config.normalize_signals else "none"
        )
        metadata.attrs["feature_extraction_version"] = "2.0"
        config_dict = {
            "sampling_rate": config.sampling_rate,
            "real_time": config.real_time,
            "strict_mode": config.strict_mode,
            "device": config.device,
            "normalize_signals": config.normalize_signals,
            "trim_edges": config.trim_edges,
            "trim_samples": config.trim_samples,
            "origin": config.origin,
        }
        metadata.attrs["feature_extraction_config"] = json.dumps(config_dict)

        processor_class = config.processor_class
        if processor_class:
            metadata.attrs["processor_class"] = processor_class

        processor_config = config.processor_config
        if processor_config:
            try:
                metadata.attrs["processor_config"] = json.dumps(processor_config)
            except (TypeError, ValueError):
                metadata.attrs["processor_config"] = json.dumps(
                    {"raw": str(processor_config)}
                )

        extractor_class = config.extractor_class
        if extractor_class:
            metadata.attrs["extractor_class"] = extractor_class

        extractor_config = config.extractor_config
        if extractor_config:
            try:
                metadata.attrs["extractor_config"] = json.dumps(extractor_config)
            except (TypeError, ValueError):
                metadata.attrs["extractor_config"] = json.dumps(
                    {"raw": str(extractor_config)}
                )

        # 2. IMPERATIVE: Store sample_ids for alignment
        # Always use sequential sample_{i} format for sample_ids
        numeric_keys = sorted(
            [str(k) for k in features if str(k).isdigit()],
            key=int,
        )
        num_samples = len(numeric_keys)
        sample_ids = [f"sample_{i}" for i in range(num_samples)]

        metadata.attrs["sample_ids"] = json.dumps(sample_ids)
        metadata.attrs["num_samples"] = len(sample_ids)
        if "subject_ids" in features:
            metadata.attrs["subject_ids"] = json.dumps(
                [str(sid) for sid in features["subject_ids"]]
            )
        if "sample_indices" in features:
            metadata.attrs["sample_indices"] = json.dumps(
                [int(i) for i in features["sample_indices"]]
            )

        # 3. IMPERATIVE: Add model_info if provided (for generated features)
        if model_info:
            model_group = metadata.create_group("model_info")
            model_group.attrs["model_name"] = model_info.get("model_name", "unknown")
            model_group.attrs["direction"] = model_info.get("direction", "unknown")
            model_group.attrs["seed"] = model_info.get("seed", -1)

        for i, (sample_id, feature_key) in enumerate(
            zip(sample_ids, numeric_keys, strict=True)
        ):
            sample_group = f.create_group(f"sample_{i}")
            sample_group.attrs["sample_id"] = sample_id

            sample_features = features.get(feature_key, {})

            if (
                "ecg_features" in sample_features
                and sample_features["ecg_features"] is not None
            ):
                ecg_group = sample_group.create_group("ecg_features")
                ecg_feat = sample_features["ecg_features"]
                if "peak_locations" in ecg_feat:
                    safe_create_dataset(
                        ecg_group, "peak_locations", ecg_feat["peak_locations"]
                    )
                if "qt_intervals" in ecg_feat:
                    safe_create_dataset(
                        ecg_group, "qt_intervals", ecg_feat["qt_intervals"]
                    )
                if "mean_ecg_quality" in ecg_feat:
                    safe_create_dataset(
                        ecg_group, "mean_ecg_quality", ecg_feat["mean_ecg_quality"]
                    )

            if (
                "ppg_features" in sample_features
                and sample_features["ppg_features"] is not None
            ):
                ppg_group = sample_group.create_group("ppg_features")
                ppg_feat = sample_features["ppg_features"]
                if "Asp_deltaT" in ppg_feat:
                    safe_create_dataset(ppg_group, "Asp_deltaT", ppg_feat["Asp_deltaT"])
                if "IPR" in ppg_feat:
                    safe_create_dataset(ppg_group, "IPR", ppg_feat["IPR"])

    logger.info("Features saved successfully to %s", output_path)


def load_features_only(file_path: str) -> dict[str, Any]:
    """Load a features-only file with minimal metadata.

    When available, processor and extractor configuration snapshots are
    included under ``metadata['processor_info']``.

    Args:
        file_path: Path to the features-only HDF5 file.

    Returns:
        Dictionary containing features and metadata.
    """

    def _decode_if_bytes(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def _load_hdf5_group(group: h5py.Group) -> dict[str, Any]:
        group_data: dict[str, Any] = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                group_data[key] = _load_hdf5_group(item)
            elif isinstance(item, h5py.Dataset):
                try:
                    value = item[()] if item.shape == () else item[:]
                except TypeError:
                    # Fallback for scalar-like datasets without shape attribute
                    value = item[()]
                group_data[key] = _decode_if_bytes(value)
        return group_data

    with h5py.File(file_path, "r") as f:
        metadata = f["metadata"]

        # Safely load JSON attributes
        config_dict = {}
        sample_ids = []

        try:
            config_str = metadata.attrs.get("feature_extraction_config", "{}")
            if isinstance(config_str, bytes):
                config_str = config_str.decode("utf-8")
            config_dict = json.loads(config_str)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Could not load config from %s", file_path)

        try:
            sample_ids_str = metadata.attrs.get("sample_ids", "[]")
            if isinstance(sample_ids_str, bytes):
                sample_ids_str = sample_ids_str.decode("utf-8")
            sample_ids = json.loads(sample_ids_str)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Could not load sample_ids from %s", file_path)

            # Generate sample IDs if not available
            def _starts_sample(key: Any) -> bool:
                return getattr(key, "startswith", lambda _: False)("sample_")

            sample_keys = [k for k in f if k is not None and _starts_sample(k)]
            sample_ids = [f"sample_{i}" for i in range(len(sample_keys))]

        subject_ids = []
        try:
            subject_ids_str = metadata.attrs.get("subject_ids", "[]")
            if isinstance(subject_ids_str, bytes):
                subject_ids_str = subject_ids_str.decode("utf-8")
            subject_ids = json.loads(subject_ids_str)
        except (json.JSONDecodeError, AttributeError):
            pass  # subject_ids are optional

        sample_indices = []
        try:
            sample_indices_str = metadata.attrs.get("sample_indices", "[]")
            if isinstance(sample_indices_str, bytes):
                sample_indices_str = sample_indices_str.decode("utf-8")
            sample_indices = json.loads(sample_indices_str)
        except (json.JSONDecodeError, AttributeError):
            pass  # sample_indices are optional

        processor_info = None
        try:
            processor_class_attr = metadata.attrs.get("processor_class", None)
            extractor_class_attr = metadata.attrs.get("extractor_class", None)
            processor_config_attr = metadata.attrs.get("processor_config", None)
            extractor_config_attr = metadata.attrs.get("extractor_config", None)

            processor_class = (
                _decode_if_bytes(processor_class_attr)
                if processor_class_attr is not None
                else None
            )
            extractor_class = (
                _decode_if_bytes(extractor_class_attr)
                if extractor_class_attr is not None
                else None
            )

            processor_config = None
            if processor_config_attr is not None:
                processor_config_str = _decode_if_bytes(processor_config_attr)
                if isinstance(processor_config_str, str):
                    processor_config = json.loads(processor_config_str)

            extractor_config = None
            if extractor_config_attr is not None:
                extractor_config_str = _decode_if_bytes(extractor_config_attr)
                if isinstance(extractor_config_str, str):
                    extractor_config = json.loads(extractor_config_str)

            if any(
                value is not None
                for value in (
                    processor_class,
                    extractor_class,
                    processor_config,
                    extractor_config,
                )
            ):
                processor_info = {
                    "processor_class": processor_class,
                    "processor_config": processor_config,
                    "extractor_class": extractor_class,
                    "extractor_config": extractor_config,
                }
        except Exception as exc:  # pragma: no cover - defensive parsing
            logger.error("Could not load processor metadata: %s", exc, exc_info=True)
            processor_info = None

        # metadata may be Group/Dataset/Datatype
        model_info = None
        if isinstance(metadata, h5py.Group) and "model_info" in metadata:
            try:
                model_info_item = metadata["model_info"]
                if isinstance(model_info_item, h5py.Group):
                    model_group = model_info_item
                    model_info = {
                        "model_name": model_group.attrs.get("model_name", "unknown"),
                        "direction": model_group.attrs.get("direction", "unknown"),
                        "seed": model_group.attrs.get("seed", -1),
                    }
            except Exception as e:
                logger.error("Could not load model info: %s", e, exc_info=True)

        features = []
        for i, sample_id in enumerate(sample_ids):
            sample_key = f"sample_{i}"
            if sample_key in f:
                try:
                    sample_data_item = f[sample_key]
                    if not isinstance(sample_data_item, h5py.Group):
                        logger.warning(
                            f"Expected '{sample_key}' to be a Group, skipping"
                        )
                        features.append({"sample_id": sample_id})
                        continue
                    sample_data = sample_data_item
                    sample_features: dict[str, Any] = {"sample_id": sample_id}

                    if "ecg_features" in sample_data:
                        ecg_item = sample_data["ecg_features"]
                        if isinstance(ecg_item, h5py.Group):
                            ecg_group = ecg_item
                            sample_features["ecg_features"] = cast(
                                "Any", _load_hdf5_group(ecg_group)
                            )

                    if "ppg_features" in sample_data:
                        ppg_item = sample_data["ppg_features"]
                        if isinstance(ppg_item, h5py.Group):
                            ppg_group = ppg_item
                            sample_features["ppg_features"] = cast(
                                "Any", _load_hdf5_group(ppg_group)
                            )

                    features.append(sample_features)
                except Exception as e:
                    logger.error(
                        "Error loading sample %s: %s", sample_key, e, exc_info=True
                    )
                    # Add empty sample to maintain alignment
                    features.append({"sample_id": sample_id})

        return {
            "features": features,
            "metadata": {
                "sampling_rate": metadata.attrs.get("sampling_rate", 125),
                "normalization_method": metadata.attrs.get(
                    "normalization_method", "unknown"
                ),
                "feature_extraction_version": metadata.attrs.get(
                    "feature_extraction_version", "unknown"
                ),
                "config": config_dict,
                "sample_ids": sample_ids,
                "subject_ids": subject_ids,
                "sample_indices": sample_indices,
                "model_info": model_info,
                "processor_info": processor_info,
            },
        }


def verify_sample_alignment(gt_file: str, model_file: str) -> bool:
    """Verify sample alignment between ground truth and model-generated features.

    Args:
        gt_file: Path to ground truth features file.
        model_file: Path to model-generated features file.

    Returns:
        True if samples are properly aligned.
    """
    gt_data = load_features_only(gt_file)
    model_data = load_features_only(model_file)

    gt_sample_ids = gt_data["metadata"]["sample_ids"]
    model_sample_ids = model_data["metadata"]["sample_ids"]

    if len(gt_sample_ids) != len(model_sample_ids):
        logger.warning(
            "Sample count mismatch: %s vs %s",
            len(gt_sample_ids),
            len(model_sample_ids),
        )
        return False

    for i, (gt_id, model_id) in enumerate(
        zip(gt_sample_ids, model_sample_ids, strict=True)
    ):
        if gt_id != model_id:
            logger.warning(
                "Sample ID mismatch at index %s: %s vs %s", i, gt_id, model_id
            )
            return False

    logger.info("Sample alignment verified successfully!")
    return True
