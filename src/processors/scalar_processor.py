"""Processor for scalar regression model outputs.

This module provides the ScalarOutputProcessor for handling outputs from models
that perform scalar regression tasks, such as direct prediction of Systolic
and Diastolic Blood Pressure (SBP/DBP) without waveform reconstruction.

Classes:
    - ScalarProcessorConfig: Dataclass for configuring the scalar processor.
    - ScalarOutputProcessor: Processor for scalar regression outputs.

Note:
    Example config usage (output_keys, return_combined) is documented in the
    inline comment block at the end of this module.
"""

from __future__ import annotations

import logging

# Standard library imports
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

# Third-party imports
import torch
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional
from hydra.core.config_store import ConfigStore

# Local imports
from src.processors.metrics_utils import compute_bp_metrics
from src.processors.output_processor import OutputProcessor
from src.processors.output_processor import ProcessingMetadata
from src.utils.bp import get_dbp
from src.utils.bp import get_sbp

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ScalarProcessorConfig:
    """Configuration for ScalarOutputProcessor.

    This dataclass defines how the ScalarOutputProcessor should parse and
    format model outputs for scalar regression tasks.

    The processor expects model outputs as tensors [B, C] where columns map
    to semantic names via output_keys. Column i corresponds to output_keys[i].

    Attributes:
        _target_: Full path to the processor class for Hydra instantiation
        output_keys (List[str]): Column-order mapping from tensor columns to names.
            output_keys[i] corresponds to column i in model output tensor [B, C].
            Example: ['sbp', 'dbp'] means column 0=sbp, column 1=dbp.
            Default: ['predictions'].

        return_combined (bool): If True, the post-processed output dict will
            include the combined scalar tensor in addition to individual scalars.
            Default: True.

        combined_key (str): The key name for the combined prediction tensor in the
            post-processed output dict.
            Default: 'predictions'.
    """

    _target_: str = "src.processors.scalar_processor.ScalarOutputProcessor"
    output_keys: list[str] = field(default_factory=lambda: ["predictions"])
    return_combined: bool = True
    combined_key: str = "predictions"
    compute_metrics: bool = False
    enabled_directions: list[str] | None = None
    denormalize: bool = False
    global_min: float | None = None
    global_max: float | None = None


class ScalarOutputProcessor(OutputProcessor):
    """Unified processor for scalar regression outputs.

    Models emit canonical dictionaries (``{"predictions": tensor, "extras": {...}}``)
    and must provide the original batch dict so metadata can be extracted for
    denormalization or metric computation. This processor normalizes prediction
    tensors, splits them into named columns via ``output_keys``, and optionally
    computes BP metrics during validation/test stages. Trainers keep lightweight
    MAE/MSE logging, while this processor focuses on schema management and
    richer BP analytics.

    Stage-aware behavior:
        - ``stage="train"``: Minimal processing for fast loss computation while
          still attaching basic MAE/MSE metrics under ``metrics.basic`` so
          trainers can log them.
        - ``stage="val"/"test"``: Full processing, including metrics and
          optional denormalization when configured.

    **BP Normalization Behavior:**
        The processor handles BP predictions that may be in normalized [0,1]
        space or raw mmHg space, depending on model configuration. Ground truth
        BP values from ``bp_raw`` in the batch are always in raw mmHg units
        (never normalized).

        - When ``denormalize=True``: Model predictions are assumed to be
          normalized [0,1] and are denormalized to mmHg using ``global_min``
          (DBP minimum) and ``global_max`` (SBP maximum) before computing
          metrics. Both basic metrics (MAE/MSE) and BP-specific metrics use
          denormalized predictions compared against raw ``bp_raw`` ground truth.
        - When ``denormalize=False``: Model predictions are assumed to already
          be in raw mmHg scale, matching the ``bp_raw`` ground truth scale. No
          denormalization is performed.

        The ``global_min`` and ``global_max`` parameters are required when
        ``denormalize=True`` and specify the dataset's BP value ranges used for
        denormalization:
        - UCI dataset: ``global_min=50.0`` (DBP), ``global_max=189.98`` (SBP)
        - PulseDB dataset: ``global_min=2.34`` (DBP), ``global_max=286.58`` (SBP)

        For implementation details, see ``compute_bp_metrics()`` in
        ``file:src/processors/metrics_utils.py``.

    Returned schema:
        {
            "predictions": <combined tensor>,
            "<output_key>": <[B,1] tensor>,  # per scalar
            "metrics": {"basic": {...}, "bp": {...}} | {"basic": {...}},
            "extras": {**model_extras, **scalar_aliases}
        }
    """

    def __init__(
        self,
        output_keys: list[str] | None = None,
        return_combined: bool = True,
        combined_key: str = "predictions",
        compute_metrics: bool = False,
        enabled_directions: list[str] | None = None,
        denormalize: bool = False,
        global_min: float | None = None,
        global_max: float | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the ScalarOutputProcessor.

        Args:
            output_keys: Column-order mapping from tensor columns to names.
                output_keys[i] corresponds to column i in model output [B, C].
            return_combined: If True, the post-processed output dict will
                include the combined scalar tensor.
            combined_key: The key name for the combined prediction tensor.
            compute_metrics: If True, enables BP metrics during val/test stages.
                When enabled, the processor calls ``compute_bp_metrics()`` from
                ``file:src/processors/metrics_utils.py`` for BP-specific
                metrics (MAE, RMSE, BHS grades) vs ``bp_raw`` ground truth.
            enabled_directions: Optional list of direction keys to filter
                metrics. When provided, BP metrics only when active direction
                matches one of the enabled keys (case-insensitive).
            denormalize: If True, predictions denormalized from [0,1] to raw
                mmHg before metrics. When False, predictions assumed already
                in raw mmHg matching ``bp_raw``. Required when model outputs
                normalized predictions needing conversion to mmHg.
            global_min: Dataset DBP minimum for denormalization. Required when
                ``denormalize=True``. Converts normalized DBP from [0,1] to
                mmHg: ``dbp_mmhg = norm * (global_max - global_min) + global_min``.
                See ``compute_bp_metrics()`` in metrics_utils.py.
            global_max: Dataset SBP maximum for denormalization. Required when
                ``denormalize=True``. Converts normalized SBP from [0,1] to
                mmHg. See ``compute_bp_metrics()`` in metrics_utils.py.
        """
        super().__init__(*args, **kwargs)
        if output_keys is None:
            output_keys = ["predictions"]
        self.output_keys = output_keys

        # Validation
        if not output_keys:
            raise ValueError("output_keys cannot be empty")
        if len(set(output_keys)) != len(output_keys):
            raise ValueError(f"output_keys must be unique: {output_keys}")

        self.return_combined = return_combined
        self.combined_key = combined_key
        self.compute_metrics = compute_metrics
        self.enabled_directions = enabled_directions
        self.denormalize = denormalize
        self.global_min = global_min
        self.global_max = global_max
        if self.denormalize and (self.global_min is None or self.global_max is None):
            raise ValueError(
                "ScalarOutputProcessor requires global_min and global_max when "
                "denormalize=True.\n"
                "These bounds are used to denormalize predictions from [0,1] to "
                "mmHg scale for BP metrics.\n\n"
                "When denormalize=False, global_min and global_max are not "
                "required, so users can either provide bounds in their processor "
                "YAML or disable denormalization for metrics.\n\n"
                "Dataset-specific bounds:\n"
                "  - UCI: global_min=50.0, global_max=189.98\n"
                "  - PulseDB: global_min=2.34, global_max=286.58\n\n"
                "The consolidated processor configs (waveform_processor_ref_test "
                "and scalar_processor) automatically use the correct bounds via "
                "Hydra variable interpolation:\n"
                "  processor:\n"
                "    global_min: ${train_dataset.dbp_min}\n"
                "    global_max: ${train_dataset.sbp_max}\n\n"
                "See: src/conf/processor/waveform_processor_ref_test.yaml or "
                "src/conf/processor/scalar_processor.yaml"
            )
        logger.info(
            "Initialized ScalarOutputProcessor with column-order mapping: %s",
            self.output_keys,
        )
        self.compute_full_metrics_during_train: bool = False

    def process(
        self, model_output: dict[str, Any], batch: dict[str, Any], stage: str
    ) -> dict[str, Any]:
        """Unified processing entrypoint for scalar predictions."""
        if not isinstance(model_output, dict):
            raise TypeError(
                "ScalarOutputProcessor expects `model_output` to be a dict with "
                "'predictions' and optional 'extras' keys."
            )
        if not isinstance(batch, dict):
            raise TypeError(
                "ScalarOutputProcessor requires the original batch dict for processing."
            )
        if model_output.get("predictions") is None:
            raise ValueError(
                "model_output must include a 'predictions' tensor for "
                "scalar processing."
            )

        stage = (stage or "train").lower()
        metadata = ProcessingMetadata.from_batch(batch)
        tensor = self._coerce_scalar_tensor(model_output.get("predictions"))
        extras = dict(model_output.get("extras") or {})

        results: dict[str, Any] = {
            "predictions": tensor,
            "metrics": {},
            "extras": extras,
        }

        scalar_map = self._materialize_scalars(tensor)
        results.update(scalar_map)
        results["extras"].update(scalar_map)

        if self.return_combined:
            results[self.combined_key] = tensor
            results["extras"][self.combined_key] = tensor

        basic_metrics = self._compute_basic_scalar_metrics(tensor, metadata)
        if basic_metrics:
            results["metrics"]["basic"] = basic_metrics

        run_full_pipeline = stage != "train" or batch.get(
            "_compute_full_metrics_during_train"
        )
        should_compute_metrics = (
            self.compute_metrics and metadata.batch is not None and run_full_pipeline
        )

        if should_compute_metrics:
            try:
                batch_for_metrics = metadata.batch
                if batch_for_metrics is None:
                    raise ValueError("batch required for BP metrics")
                metrics = compute_bp_metrics(
                    scalar_map,
                    batch_for_metrics,
                    denormalize=self.denormalize,
                    global_min=self.global_min,
                    global_max=self.global_max,
                    direction=getattr(metadata, "direction", None),
                    enabled_directions=self.enabled_directions,
                )
                if metrics is not None:
                    results["metrics"]["bp"] = metrics
            except Exception as exc:  # pragma: no cover - defensive catch
                logger.warning("BP metrics attachment failed: %s", exc, exc_info=True)

        if not results["metrics"]:
            results.pop("metrics")

        return results

    def _coerce_scalar_tensor(
        self, predictions: Tensor | list[Any] | tuple[Any, ...] | None
    ) -> Tensor:
        if predictions is None:
            raise ValueError(
                "Scalar processor expected 'predictions' key in model_output."
            )

        tensor: Tensor | None = None
        if isinstance(predictions, torch.Tensor):
            tensor = predictions
        elif isinstance(predictions, (list, tuple)):
            if not predictions:
                raise ValueError(
                    "Received empty prediction sequence for scalar processor."
                )
            candidate = predictions[0]
            tensor = candidate if isinstance(candidate, torch.Tensor) else None
        else:
            raise TypeError(
                f"Unsupported predictions type '{
                    type(predictions).__name__
                }' for scalar processor."
            )

        if tensor is None:
            raise ValueError("Scalar processor could not locate tensor predictions.")

        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 3:
            if tensor.shape[2] == 1:
                tensor = tensor.squeeze(-1)
            else:
                raise ValueError(
                    f"Expected scalar output, but got waveform-like shape {
                        tuple(tensor.shape)
                    }."
                )
        elif tensor.dim() > 3:
            raise ValueError(
                f"Invalid tensor shape for scalar output: {tuple(tensor.shape)}"
            )

        column_mismatch = tensor.shape[1] != len(self.output_keys)
        allow_flexible_predictions = (
            len(self.output_keys) == 1 and self.output_keys[0] == "predictions"
        )
        if column_mismatch and not allow_flexible_predictions:
            raise ValueError(
                f"Tensor columns ({tensor.shape[1]}) don't match output_keys length ({
                    len(self.output_keys)
                }). "
                f"Ensure model output columns match: {self.output_keys}"
            )

        return tensor

    def _split_scalars(self, tensor: Tensor, keys: list[str]) -> dict[str, Tensor]:
        """Split a multi-column scalar tensor into individual named scalars.

        For a tensor of shape [B, C] with keys ['k1', 'k2', ..., 'kC'], this returns
        {'k1': [B,1], 'k2': [B,1], ...}.

        Args:
            tensor (Tensor): A tensor of shape [B, C].
            keys (List[str]): A list of C keys for the output dictionary.

        Returns:
            Dict[str, Tensor]: A dictionary mapping keys to scalar tensors.

        Raises:
            ValueError: If the number of keys does not match the number of columns.
        """
        num_columns = tensor.shape[1]
        if len(keys) != num_columns:
            raise ValueError(
                f"Number of output_keys ({len(keys)}) does not match number of "
                f"tensor columns ({num_columns})."
            )

        # Split tensor along the channel dimension
        split_tensors = tensor.split(1, dim=1)  # List of [B, 1] tensors

        return dict(zip(keys, split_tensors, strict=True))

    def _materialize_scalars(self, tensor: Tensor) -> dict[str, Tensor]:
        allow_flexible_predictions = (
            len(self.output_keys) == 1 and self.output_keys[0] == "predictions"
        )

        if tensor.shape[1] == 1 or (
            allow_flexible_predictions and len(self.output_keys) == 1
        ):
            key = self.output_keys[0] if self.output_keys else self.combined_key
            return {key: tensor}

        return self._split_scalars(tensor, self.output_keys)

    def get_column_mapping(self) -> dict[str, int]:
        """Return mapping from output_keys to column indices.

        Returns:
            Dict[str, int]: Mapping where output_keys[i] -> column index i.
                Example: {'sbp': 0, 'dbp': 1} for output_keys=['sbp', 'dbp'].
        """
        return {key: idx for idx, key in enumerate(self.output_keys)}

    def _compute_basic_scalar_metrics(
        self,
        predictions: Tensor,
        metadata: ProcessingMetadata,
    ) -> dict[str, Tensor]:
        target = self._extract_target_scalars(
            metadata.batch, predictions.device, predictions.dtype, predictions.shape[1]
        )
        if target is None:
            return {}

        # Denormalize predictions to match bp_raw scale (mmHg) for metrics
        # Note: bp_raw is always raw mmHg (see _extract_target_scalars)
        if self.denormalize:
            # Denormalize from normalized space to mmHg using linear rescaling
            from src.processors.metrics_utils import _maybe_denormalize_tensor

            pred_eval = _maybe_denormalize_tensor(
                predictions,
                denormalize=True,
                global_min=self.global_min,
                global_max=self.global_max,
            )
            # Ensure device and dtype match predictions
            pred_eval = pred_eval.to(device=predictions.device, dtype=predictions.dtype)
        else:
            pred_eval = predictions

        return {
            "mae": F.l1_loss(pred_eval, target),
            "mse": F.mse_loss(pred_eval, target),
        }

    def _extract_target_scalars(
        self,
        batch: dict[str, Any] | None,
        device: torch.device,
        dtype: torch.dtype,
        expected_columns: int,
    ) -> Tensor | None:
        """Extract target scalars from raw BP ground truth.

        Always extracts SBP and DBP from raw bp_raw [B,3] tensor to ensure
        basic metrics (MAE/MSE) are computed in the same scale as BP-specific
        metrics (both use raw mmHg values from bp_raw). This ensures consistency
        with compute_bp_metrics in metrics_utils.py which also uses bp_raw as
        ground truth.

        Args:
            batch: Batch dict with bp_raw [B,3] tensor; columns [SBP, DBP, MAP]
            device: Target device for output tensor
            dtype: Target dtype for output tensor
            expected_columns: Expected number of columns (typically 2 for SBP/DBP)

        Returns:
            Target tensor [B, expected_columns] with SBP (column 0) and DBP (column 1),
            or None if bp_raw is not available
        """
        if not isinstance(batch, dict):
            return None

        # Always use raw bp_raw for ground truth to ensure consistent metric scales
        # bp_raw is [B, 3] with columns [SBP, DBP, MAP]
        bp_raw = batch.get("bp_raw")
        if not torch.is_tensor(bp_raw):
            return None

        # Extract SBP (column 0) and DBP (column 1) from bp_raw
        sbp = get_sbp(bp_raw)  # [B]
        dbp = get_dbp(bp_raw)  # [B]
        target = torch.stack([sbp, dbp], dim=1)  # [B, 2]

        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if target.dim() == 3 and target.size(-1) == 1:
            target = target.squeeze(-1)

        if target.dim() != 2:
            return None

        if target.size(1) != expected_columns:
            if target.size(1) > expected_columns:
                target = target[:, :expected_columns]
            else:
                return None

        return target.to(device=device, dtype=dtype)


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(group="processor", name="base_scalar_processor", node=ScalarProcessorConfig)

# Usage in config:
# processor:
#   _target_: src.processors.scalar_processor.ScalarOutputProcessor
#   config:
#     output_keys: ['sbp', 'dbp']  # Splits 'predictions' [B,2] (col0=SBP, col1=DBP)
#     # Legacy y_pred_sbp/y_pred_dbp removed; use 'predictions' from MDViSCo/BPModel
#     return_combined: true
