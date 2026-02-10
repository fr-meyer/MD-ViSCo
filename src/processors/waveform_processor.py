"""Waveform output processor for regression and reconstruction tasks.

The WaveformOutputProcessor normalises waveform predictions and, when configured with
``compute_metrics=True``, attaches waveform quality metrics under
``results["metrics"]["waveform"]`` using keys such as ``mae`` and ``correlation``.
Metrics are optional and only computed when explicitly enabled, ensuring evaluators
can rely on a canonical location without imposing overhead on models that do not
request them.
"""

import logging
from collections.abc import Iterable

# Standard library imports
from dataclasses import dataclass
from typing import Any

# Third-party imports
import torch
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional
from hydra.core.config_store import ConfigStore
from torch import Tensor

# Local imports
from src.processors.metrics_utils import compute_bp_metrics
from src.processors.metrics_utils import compute_waveform_metrics
from src.processors.output_processor import OutputProcessor
from src.processors.output_processor import ProcessingMetadata

logger = logging.getLogger(__name__)


@dataclass
class WaveformProcessorConfig:
    """Configuration options for :class:`WaveformOutputProcessor`.

    Attributes:
        _target_: Full path to the processor class for Hydra instantiation
        extractor: Optional extractor configuration for scalar extraction. When
            provided, scalars are extracted using the configured extractor; when
            omitted, no scalar extraction is performed.
        trim_padding: If ``True``, waveform padding is removed whenever metadata
            specifies a positive ``padding_length``.
        output_key: Key name used for the waveform entry in the post-processed
            dictionary returned by :meth:`process`.
        select_channel_from_batch: When ``True`` and the batch metadata provides
            ``tgt_idxs``, multi-channel waveforms are reduced to the selected channel
            during post-processing. When ``False`` (default), the waveform retains all
            channels even if ``tgt_idxs`` is present.
        enabled_directions: Optional list of direction keys allowed for metrics
            computation. When provided, waveform metrics are only computed when
            the active direction matches one of the enabled keys (case-insensitive).
        denormalize: If ``True``, BP metrics are converted back to mmHg scale using
            ``global_min`` and ``global_max``. When ``False``, metrics remain in the
            model's normalized scale.
        global_min: Minimum training value used for denormalization. Required when
            ``denormalize`` is ``True``.
        global_max: Maximum training value used for denormalization. Required when
            ``denormalize`` is ``True``.
        bhs_thresholds: Optional iterable of floats overriding default BHS grade
            thresholds. Defaults to ``[5, 10, 15]`` when not provided.
        ignore_extractor_errors: When ``True``, extractor exceptions are logged and
            suppressed regardless of extractor strictness. When ``False`` (default),
            extractors that declare ``strict_mode=True`` cause ``process`` to reraises.
    """

    _target_: str = "src.processors.waveform_processor.WaveformOutputProcessor"
    extractor: Any | None = None  # Instantiated ScalarExtractor instance or config
    trim_padding: bool = True
    output_key: str = "waveform"
    compute_metrics: bool = False
    enabled_directions: list[str] | None = None
    select_channel_from_batch: bool = False
    denormalize: bool = False
    global_min: float | None = None
    global_max: float | None = None
    bhs_thresholds: list[float] | None = None
    ignore_extractor_errors: bool = False


class WaveformOutputProcessor(OutputProcessor):
    """Stage-aware processor for waveform regression and reconstruction outputs.

    Models emit canonical dictionaries (``{"predictions": tensor, "extras": {...}}``)
    and must provide the original batch dict so padding / metadata can be derived.
    This processor is responsible for transforming those outputs into the unified
    schema expected by trainers, evaluators, and criteria:

        {
            "predictions_raw": <waveform tensor>,
            "predictions": <stage-aware tensor>,
            self.output_key: <stage-aware tensor>,
            "padding_metadata": {"padding_length": int,
                "original_length": Optional[int]},
            "metrics": {"basic": {...}, "waveform": {...}, "bp": {...}},
            "extras": {**model_extras, **extractor_outputs}
        }

    Dual tensor contract:
        - ``predictions_raw`` always mirrors the model's direct output (padded) so
          trainers can compute losses on the full receptive field during training or
          for diagnostics.
        - ``predictions`` represents the stage-appropriate view. It is identical to
          ``predictions_raw`` during training, but becomes trimmed during validation
          / test when ``trim_padding`` is enabled and the batch carries padding
          metadata.
        - ``self.output_key`` (default: ``"waveform"``) acts as a compatibility alias
          that points to ``predictions`` so existing evaluators continue to function.

    Padding metadata contract:
        - ``padding_metadata`` always includes ``padding_length`` (number of samples
          trimmed from both the start and end) and optionally ``original_length`` when
          the batch exposes it. This metadata is emitted so downstream components can
          trim targets or perform post-processing without recomputing padding
          parameters. Trainers must rely on this metadata instead of inferring padding
          from tensor shapes.

    Stage-aware behavior:
        - ``stage="train"``: Minimal processing. Padding is untouched,
          ``predictions`` == ``predictions_raw``, and basic MAE / MSE metrics are
          computed on the padded view.
        - ``stage="val"`` / ``"test"``: Full processing. Padding is trimmed (when
          enabled), optional scalar extraction runs, waveform / BP metrics operate on
          trimmed tensors, and the trainer is expected to symmetrically trim targets
          before computing loss.
        - Trainers should prefer ``predictions_raw`` when they need the padded tensor
          (e.g., for training loss), and ``predictions`` when they need the evaluation
          scope (e.g., validation loss or metric calculation).

    Trainer expectations:
        - Training losses typically use ``predictions_raw`` together with untrimmed
          targets so gradients cover the full receptive field the model produced.
        - Validation / test losses use ``predictions`` together with targets trimmed
          according to ``padding_metadata["padding_length"]``. This keeps loss curves
          aligned with reported metrics and avoids dimension mismatch errors.

    **BP Normalization for Extracted Scalars:**
        When an extractor is configured (e.g., BP extractor), it extracts SBP/DBP
        scalars from normalized waveform predictions. The extractor outputs normalized
        scalars in [0,1] space derived from normalized waveform predictions.

        Ground truth BP values from ``bp_raw`` in the batch are always in raw mmHg
        units (never normalized), ensuring consistent metric computation regardless of
        normalization settings.

        - When ``denormalize=True``: Extracted scalars are denormalized from [0,1] to
          raw mmHg scale using ``global_min`` (DBP minimum) and ``global_max`` (SBP
          maximum) before computing BP metrics. This ensures extracted scalars match
          the ``bp_raw`` ground truth scale for accurate metric computation.
        - When ``denormalize=False``: Extracted scalars remain in normalized [0,1]
          space. Note that this may result in metrics comparing normalized predictions
          against raw ground truth, which may not be meaningful for BP-specific
          metrics.

        The ``global_min`` and ``global_max`` parameters are required when
        ``denormalize=True`` and specify the dataset's BP value ranges used for
        denormalization:
        - UCI dataset: ``global_min=50.0`` (DBP), ``global_max=189.98`` (SBP)
        - PulseDB dataset: ``global_min=2.34`` (DBP), ``global_max=286.58`` (SBP)

        For implementation details, see ``compute_bp_metrics()`` in
        ``file:src/processors/metrics_utils.py``.

    The processor also enforces the passthrough pattern by copying any keys found in
    ``model_output["extras"]`` into the returned ``"extras"`` dictionary so that cascade
    models can expose auxiliary artefacts (e.g., reconstructed ECG segments) without
    bespoke evaluator code.
    """

    def __init__(
        self,
        extractor: Any | None = None,
        trim_padding: bool = True,
        output_key: str = "waveform",
        compute_metrics: bool = False,
        enabled_directions: list[str] | None = None,
        select_channel_from_batch: bool = False,
        denormalize: bool = False,
        global_min: float | None = None,
        global_max: float | None = None,
        bhs_thresholds: Iterable[float] | None = None,
        ignore_extractor_errors: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the WaveformOutputProcessor.

        Args:
            extractor: Optional scalar extractor instance for extracting BP scalars
                (SBP/DBP) from waveform predictions. When provided, the extractor runs
                during validation/test stages to extract scalars from normalized
                waveform predictions. The extractor outputs normalized scalars [0,1]
                which are then optionally denormalized based on the ``denormalize``
                flag.
            trim_padding: If True, removes symmetrical padding from waveform
                predictions during validation/test stages when ``padding_length`` is
                specified in batch metadata. Padding is preserved during training to
                maintain full receptive field for loss computation. Default: True.
            output_key: Key name used for the waveform entry in the post-processed
                output dictionary. This key acts as a compatibility alias pointing to
                the stage-aware ``predictions`` tensor. Default: "waveform".
            compute_metrics: If True, enables waveform and BP metrics computation during
                validation/test stages. When enabled, the processor calls
                ``compute_waveform_metrics()`` and ``compute_bp_metrics()`` from
                ``file:src/processors/metrics_utils.py`` to compute quality metrics
                comparing predictions against ground truth.
            enabled_directions: Optional list of direction keys to filter metrics
                computation. When provided, waveform and BP metrics are only computed
                when the active direction matches one of the enabled keys
                (case-insensitive). If None, metrics are computed for all directions.
            select_channel_from_batch: If True, selects a specific channel from
                multi-channel waveforms using ``tgt_idxs`` from batch metadata during
                post-processing. When False (default), the waveform retains all
                channels even if ``tgt_idxs`` is present.
            denormalize: If True, extracted SBP/DBP scalars are denormalized from
                normalized [0,1] space to raw mmHg scale before computing BP metrics.
                When False, extracted scalars remain in normalized space. Required when
                extractor outputs normalized scalars that need conversion to mmHg for
                meaningful BP metric computation. See ``compute_bp_metrics()`` in
                ``file:src/processors/metrics_utils.py`` for implementation details.
            global_min: Dataset DBP (Diastolic Blood Pressure) minimum value used for
                denormalization of extracted scalars. Required when
                ``denormalize=True``. Used to convert normalized DBP scalars from [0,1]
                to mmHg using linear rescaling. See ``compute_bp_metrics()`` in
                ``file:src/processors/metrics_utils.py`` for implementation.
            global_max: Dataset SBP (Systolic Blood Pressure) maximum value used for
                denormalization of extracted scalars. Required when
                ``denormalize=True``. Used to convert normalized SBP scalars from [0,1]
                to mmHg. See ``compute_bp_metrics()`` in
                ``file:src/processors/metrics_utils.py`` for implementation details.
            bhs_thresholds: Optional iterable of floats overriding default BHS
                (British Hypertension Society) grade thresholds for BP accuracy
                assessment. Defaults to ``[5, 10, 15]`` mmHg when not provided. These
                thresholds define the error ranges for BHS Grade A, B, and C
                classifications.
            ignore_extractor_errors: If True, exceptions raised by the extractor
                during scalar extraction are logged as warnings and suppressed,
                allowing processing to continue. When False (default), extractors that
                declare ``strict_mode=True`` cause ``process()`` to re-raise
                exceptions, halting processing.
        """
        if args:
            raise TypeError(
                f"WaveformOutputProcessor received unexpected positional arguments: {
                    args
                }. "
                "Only keyword arguments defined in the signature are supported."
            )

        if "vital_sign_type" in kwargs:
            raise TypeError(
                "WaveformOutputProcessor no longer accepts 'vital_sign_type'. "
                "Use ProcessingMetadata.vital_sign_type in the batch metadata instead."
            )

        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(
                f"WaveformOutputProcessor received unexpected keyword arguments: {
                    unexpected
                }"
            )

        super().__init__()

        self.extractor = extractor

        self.trim_padding = trim_padding
        self.output_key = output_key
        self.compute_metrics = compute_metrics
        self.enabled_directions = enabled_directions
        self.select_channel_from_batch = select_channel_from_batch
        self.denormalize = denormalize
        self.global_min = global_min
        self.global_max = global_max
        self.bhs_thresholds = bhs_thresholds
        self.ignore_extractor_errors = ignore_extractor_errors
        self.compute_full_metrics_during_train: bool = False

        if self.denormalize and (self.global_min is None or self.global_max is None):
            raise ValueError(
                "WaveformOutputProcessor requires global_min and global_max when "
                "denormalize=True.\n"
                "These bounds are used to denormalize extracted SBP/DBP scalars from "
                "normalized waveform predictions [0,1] to mmHg scale for BP "
                "metrics.\n\n"
                "When denormalize=False, global_min and global_max are not required, "
                "so users can either provide bounds in their processor YAML or disable "
                "denormalization for metrics.\n\n"
                "Dataset-specific bounds:\n"
                "  - UCI: global_min=50.0, global_max=189.98\n"
                "  - PulseDB: global_min=2.34, global_max=286.58\n\n"
                "The consolidated processor configs (waveform_processor_ref_test and "
                "scalar_processor) automatically use the correct bounds via Hydra "
                "variable interpolation:\n"
                "  processor:\n"
                "    global_min: ${train_dataset.dbp_min}\n"
                "    global_max: ${train_dataset.sbp_max}\n\n"
                "See: src/conf/processor/waveform_processor_ref_test.yaml or "
                "src/conf/processor/scalar_processor.yaml"
            )

        # Note: The global_min/global_max parameters are used to denormalize extracted
        # SBP/DBP scalars (not the waveform itself) for BP-specific metrics. The BP
        # extractor outputs normalized scalars [0,1] from normalized waveform
        # predictions, which are then denormalized by compute_bp_metrics in
        # metrics_utils.py using these bounds.
        # These bounds should match the dataset's BP value ranges:
        #   - UCI: 50.0-189.98 mmHg (from uci_dataset.py lines 24-25)
        #   - PulseDB: 2.34-286.58 mmHg (from pulsedb_dataset.py lines 28-29)

        logger.info(
            "Initialised WaveformOutputProcessor with extractor: %s", self.extractor
        )

    def process(
        self, model_output: dict[str, Any], batch: dict[str, Any], stage: str
    ) -> dict[str, Any]:
        """Unified processing entrypoint for waveform predictions.

        This method implements the dual tensor contract described at the class level:
        - ``predictions_raw``: Always contains the padded model output, mirroring the
          model's direct output. This tensor preserves the full receptive field and is
          used by trainers for training-time loss computation.
        - ``predictions``: Stage-aware tensor that may be trimmed during
          validation/test when ``trim_padding`` is enabled and ``padding_length`` is
          set in the batch metadata. During training, ``predictions`` is identical to
          ``predictions_raw``. During validation/test, padding is removed from both
          ends of the temporal dimension if ``trim_padding=True`` and
          ``padding_length > 0``.

        The method also handles optional scalar extraction, waveform metrics
        computation, and BP metrics computation based on configuration and stage.

        Args:
            model_output: Dictionary with 'predictions' key containing the model's
                output waveform tensor, and optional 'extras' key for auxiliary
                outputs.
            batch: Original batch dictionary containing metadata for processing (e.g.,
                padding_length, original_length, target waveforms).
            stage: Processing stage ('train', 'val', or 'test'). Determines whether
                full processing pipeline runs and whether padding is trimmed.

        Returns:
            Dictionary with keys:
                - "predictions_raw": Padded waveform tensor (always matches model
                  output)
                - "predictions": Stage-aware waveform tensor (padded for train,
                  trimmed for val/test when trim_padding and padding_length are set)
                - self.output_key: Alias pointing to "predictions" for compatibility
                - "padding_metadata": Dict with padding_length and optional
                  original_length
                - "metrics": Optional dict with basic, waveform, and/or BP metrics
                - "extras": Merged dict containing model extras and extractor outputs
        """
        if not isinstance(model_output, dict):
            raise TypeError(
                "WaveformOutputProcessor expects `model_output` to be a dict with "
                "'predictions' and optional 'extras' keys."
            )
        if not isinstance(batch, dict):
            raise TypeError(
                "WaveformOutputProcessor requires the original batch dict for "
                "processing."
            )
        if model_output.get("predictions") is None:
            raise ValueError(
                "model_output must include a 'predictions' tensor for waveform "
                "processing."
            )

        stage = (stage or "train").lower()
        metadata = ProcessingMetadata.from_batch(batch)
        model_extras = model_output.get("extras") or {}
        waveform = self._coerce_waveform_tensor(model_output.get("predictions"))

        full_processing_requested = stage != "train"
        allow_train_full = bool(batch.get("_compute_full_metrics_during_train"))
        run_full_pipeline = full_processing_requested or allow_train_full

        # Optional channel selection prior to trimming
        if (
            self.select_channel_from_batch
            and isinstance(metadata.batch, dict)
            and "tgt_idxs" in metadata.batch
            and waveform.dim() >= 3
            and waveform.size(1) > 1
        ):
            tgt_idxs = metadata.batch["tgt_idxs"]
            if torch.is_tensor(tgt_idxs):
                if tgt_idxs.device != waveform.device:
                    tgt_idxs = tgt_idxs.to(waveform.device)
                batch_indices = torch.arange(waveform.size(0), device=waveform.device)
                waveform = waveform[batch_indices, tgt_idxs].unsqueeze(1)

        # Store the original padded tensor as predictions_raw. This tensor always
        # reflects the model's direct output and is consumed by trainers for
        # training-time loss so gradients cover the full receptive field. Validation /
        # test use trimmed predictions.
        predictions_raw = waveform
        trimmed_waveform = waveform
        padding_length = metadata.padding_length or 0
        if run_full_pipeline and self.trim_padding and padding_length > 0:
            trimmed_waveform = self._trim_padding(trimmed_waveform, padding_length)

        padding_metadata: dict[str, Any] = {
            "padding_length": padding_length,
        }
        original_length = getattr(metadata, "original_length", None)
        if original_length is not None:
            padding_metadata["original_length"] = original_length

        # Stage-specific view of the waveform tensor:
        #   - train  -> padded (predictions_raw) to match training-time metrics / losses
        #   - val/test -> trimmed so evaluation metrics focus on the unpadded scope
        predictions_for_stage = (
            trimmed_waveform if stage != "train" else predictions_raw
        )

        results: dict[str, Any] = {
            "predictions_raw": predictions_raw,
            "predictions": predictions_for_stage,
            self.output_key: predictions_for_stage,
            "padding_metadata": padding_metadata,
            "metrics": {},
            "extras": dict(model_extras),
        }
        results["extras"][self.output_key] = predictions_for_stage
        results["extras"]["padding_metadata"] = padding_metadata

        target_waveform = self._extract_target_waveform(
            metadata.batch, waveform.device, waveform.dtype
        )
        basic_metrics = self._compute_basic_waveform_metrics(
            waveform=waveform,
            trimmed_waveform=trimmed_waveform,
            target_waveform=target_waveform,
            padding_length=padding_length,
            full_processing_requested=full_processing_requested,
        )
        if basic_metrics:
            results["metrics"]["basic"] = basic_metrics

        padding_for_metrics = (
            padding_length if (run_full_pipeline and self.trim_padding) else 0
        )

        extractor_requested = metadata.extract_scalars is not False
        should_extract_scalars = (
            self.extractor is not None
            and extractor_requested
            and (run_full_pipeline or stage != "train")
        )

        if should_extract_scalars:
            time_steps = trimmed_waveform.shape[-1]
            num_channels = (
                trimmed_waveform.shape[1] if trimmed_waveform.ndim >= 2 else 1
            )

            if time_steps == 0 or num_channels == 0:
                logger.warning(
                    "Skipping scalar extraction due to empty waveform dimensions "
                    "(shape=%s, time_steps=%d, num_channels=%d).",
                    trimmed_waveform.shape,
                    time_steps,
                    num_channels,
                )
            else:
                try:
                    extractor = self.extractor
                    if extractor is None:
                        raise RuntimeError("extractor required for scalar extraction")
                    scalars = extractor.extract(trimmed_waveform)
                    for key, value in scalars.items():
                        results[key] = value
                        results["extras"][key] = value
                except Exception as exc:
                    if self.ignore_extractor_errors:
                        logger.warning(
                            "Failed to extract scalars from extractor %s: %s. "
                            "Skipping scalar outputs.",
                            type(self.extractor).__name__,
                            exc,
                        )
                    else:
                        raise

        should_compute_metrics = (
            self.compute_metrics and metadata.batch is not None and run_full_pipeline
        )

        if should_compute_metrics:
            try:
                batch_for_metrics = metadata.batch
                if batch_for_metrics is None:
                    raise ValueError("batch required for waveform metrics")
                metrics = compute_waveform_metrics(
                    {"waveform": trimmed_waveform},
                    batch_for_metrics,
                    direction=getattr(metadata, "direction", None),
                    enabled_directions=self.enabled_directions,
                    padding_length=padding_for_metrics,
                )
                if metrics is not None:
                    results["metrics"]["waveform"] = metrics
            except Exception as exc:  # pragma: no cover - defensive catch
                logger.warning(
                    "Waveform metrics attachment failed: %s", exc, exc_info=True
                )

        if should_compute_metrics and "sbp" in results and "dbp" in results:
            if not isinstance(metadata.batch, dict) or "bp_raw" not in metadata.batch:
                logger.debug(
                    "BP metrics: computation skipped because 'bp_raw' ground truth "
                    "is missing."
                )
            else:
                try:
                    bp_metrics = compute_bp_metrics(
                        {"sbp": results["sbp"], "dbp": results["dbp"]},
                        metadata.batch,
                        direction=getattr(metadata, "direction", None),
                        denormalize=self.denormalize,
                        global_min=self.global_min,
                        global_max=self.global_max,
                        bhs_thresholds=self.bhs_thresholds,
                        enabled_directions=self.enabled_directions,
                        padding_length=padding_for_metrics,
                    )
                    if bp_metrics is not None:
                        results["metrics"]["bp"] = bp_metrics
                except Exception as exc:  # pragma: no cover - defensive catch
                    logger.warning(
                        "BP metrics: computation failed - %s", exc, exc_info=True
                    )

        if not results["metrics"]:
            results.pop("metrics")

        return results

    def _coerce_waveform_tensor(
        self, predictions: Tensor | tuple[Any, ...] | list[Any] | None
    ) -> Tensor:
        if predictions is None:
            raise ValueError(
                "Waveform processor expected 'predictions' key in model_output."
            )

        waveform: Tensor | None = None
        if torch.is_tensor(predictions):
            waveform = predictions
        elif isinstance(predictions, (list, tuple)):
            if not predictions:
                raise ValueError(
                    "Received empty prediction sequence for waveform processor."
                )
            candidate = predictions[0]
            if not torch.is_tensor(candidate):
                raise ValueError(
                    "First element of prediction sequence must be a tensor."
                )
            waveform = candidate
        else:
            raise TypeError(
                f"Unsupported predictions type '{
                    type(predictions).__name__
                }' for waveform "
                "processor."
            )

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.ndim != 3:
            raise ValueError(
                "Waveform tensor must have 2 or 3 dimensions, "
                f"but has shape {tuple(waveform.shape)}."
            )

        return waveform

    def _extract_target_waveform(
        self,
        batch: dict[str, Any] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if not isinstance(batch, dict):
            return None
        targets = batch.get("y")
        if not torch.is_tensor(targets):
            return None
        target_tensor = targets.to(device=device, dtype=dtype)
        if target_tensor.dim() == 2:
            target_tensor = target_tensor.unsqueeze(1)
        if target_tensor.dim() != 3:
            return None

        tgt_idxs = batch.get("tgt_idxs")
        if tgt_idxs is None:
            if target_tensor.size(1) == 1:
                return target_tensor
            return None

        if not torch.is_tensor(tgt_idxs):
            return None
        tgt_idxs = tgt_idxs.to(device)
        batch_indices = torch.arange(target_tensor.size(0), device=device)
        return target_tensor[batch_indices, tgt_idxs].unsqueeze(1)

    def _compute_basic_waveform_metrics(
        self,
        waveform: Tensor,
        trimmed_waveform: Tensor,
        target_waveform: Tensor | None,
        padding_length: int,
        full_processing_requested: bool,
    ) -> dict[str, Tensor]:
        if target_waveform is None:
            return {}

        basic_metrics: dict[str, Tensor] = {
            "mae": F.l1_loss(waveform, target_waveform),
            "mse": F.mse_loss(waveform, target_waveform),
        }

        if full_processing_requested:
            trimmed_target = target_waveform
            if padding_length > 0 and self.trim_padding:
                trimmed_target = self._trim_padding(trimmed_target, padding_length)
            basic_metrics["mae_trimmed"] = F.l1_loss(trimmed_waveform, trimmed_target)
            basic_metrics["mse_trimmed"] = F.mse_loss(trimmed_waveform, trimmed_target)

        return basic_metrics

    def _is_scalar_output(self, value: Any, batch_size: int) -> bool:
        """Determine whether an extractor output represents a scalar tensor.

        Args:
            value (Any): Extractor output value.
            batch_size (int): Expected batch dimension for scalar tensors.

        Returns:
            bool: True when value is a `[batch_size, 1]` tensor.
        """
        if torch.is_tensor(value) and value.ndim == 2:
            return value.shape[0] == batch_size and value.shape[1] == 1
        return False

    def _trim_padding(self, waveform: Tensor, padding_length: int) -> Tensor:
        """Remove symmetrical padding from the temporal dimension of a waveform.

            Args:
                waveform: Waveform tensor shaped ``[batch, channels, time]``.
                padding_length: Number of samples to remove from both start and end of
                    the temporal axis.

            Returns:
                torch.Tensor: Waveform tensor with padding removed.

        Raises:
            ValueError: If the requested padding would remove the entire sequence.
        """
        if padding_length <= 0:
            return waveform

        total_trim = padding_length * 2
        sequence_length = waveform.shape[-1]
        if total_trim >= sequence_length:
            raise ValueError(
                f"Cannot trim padding: requested removal of {total_trim} samples "
                f"from a sequence of length {sequence_length}."
            )

        return waveform[..., padding_length:-padding_length]


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="processor", name="base_waveform_processor", node=WaveformProcessorConfig
)
