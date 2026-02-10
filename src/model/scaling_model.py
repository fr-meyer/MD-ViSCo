"""Two-Stage Scaling Model for BP Waveform Reconstruction.

This module provides a lightweight orchestrator interface that wraps two BaseModel
instances for BP waveform reconstruction with automatic scaling. Unlike
    TwoStageCascadeModel
which simply passes outputs between stages, this model unscales the normalized waveform
using SBP/DBP predictions from stage2.

Usage Example:
    # Via Hydra config (recommended):
    model:
      _target_: src.model.scaling_model.TwoStageScalingModel
      stage1_model:
        _target_: src.model.mdvisco.MDViSCo
        # ... stage1 config (approximation model)
      stage2_model:
        _target_: src.model.nabnet.ShallowUNetBP
        # ... stage2 config (refinement/BP prediction model)

    # Via Python (for advanced usage):
    stage1 = MDViSCo(...)
    stage2 = ShallowUNetBP(...)
    scaling_model = TwoStageScalingModel(stage1, stage2)

    # Cascade stage1 checkpoint; loaded separately from stage2
    scaling_model.load_checkpoint("cascade_checkpoint.pth")

    # Use in evaluator:
    from src.trainers.trainer import recursively_set_layout
    from src.core.domain import Vital
    layout = {Vital.PPG: 0, Vital.ECG: 1, Vital.ABP: 2}  # Dict[Vital, int] mapping
    recursively_set_layout(scaling_model, layout)
    scaling_model.eval()
    scaling_model.to(device)
    outputs = scaling_model(batch_dict)

Scaling Logic:
    The model performs a three-step process:
    1. Stage1 produces a normalized BP waveform (typically [0,1] range)
    2. Stage2 predicts SBP (systolic) and DBP (diastolic) blood pressure values
    3. The normalized waveform is unscaled using Global_Min_Max_Norm with:
       - SBP as the maximum value (peak pressure)
       - DBP as the minimum value (baseline pressure)

    This converts the normalized waveform back to physiologically meaningful mmHg
        values.

Checkpoint Format Compatibility:
    - Nested format (from cascade trainer): {'stage1_model': {...}, 'stage2_model':
        {...}}
    - DDP format: Handles 'module.' prefix removal

Key Features:
    - load_checkpoint(): Convenient loading from path or dict, handles trainer format
    - load_state_dict(): Low-level loading with format detection and IncompatibleKeys
        return
    - auto_eval_in_forward: Optional flag to control eval() behavior in forward pass
    - Robust stage1 output extraction: Supports 'y_pred', 'y_pred_waveform', tensor
        outputs
    - BP-based unscaling: Converts normalized waveforms to physiological scale
"""

import logging

# Standard library imports
from dataclasses import dataclass
from typing import Any

import torch

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.model.base_model import BaseModel
from src.model.two_stage_model import TwoStageModel
from src.model.two_stage_model import TwoStageModelConfig
from src.utils.checkpoint_io import CheckpointIO

logger = logging.getLogger(__name__)


@dataclass
class TwoStageScalingModelConfig(TwoStageModelConfig):
    """Configuration for TwoStageScalingModel.

    This config inherits from TwoStageModelConfig and is designed for evaluator usage
    where both stage1 and stage2 models are instantiated by Hydra. The scaling model
    orchestrates the two stages and performs BP-based waveform unscaling.

    Workflow:
        1. Stage1 (approximation) produces normalized BP waveform
        2. Stage2 (refinement) predicts SBP and DBP values
        3. Waveform is unscaled using SBP as max and DBP as min

    Attributes:
        _target_: Hydra target path to TwoStageScalingModel class
        model_name: Identifier for the scaling model (overrides default)
        stage1_model: Inherited from TwoStageModelConfig - stage1 configuration
        stage2_model: Inherited from TwoStageModelConfig - stage2 configuration
        checkpoint_io: Inherited from BaseModelConfig - checkpoint I/O configuration
        return_dict: Controls dict output format. The scaling model now requires dict
            outputs,
            so this flag is retained for Hydra compatibility but defaults to True.
    """

    _target_: str = "src.model.scaling_model.TwoStageScalingModel"
    model_name: str = "TwoStageScaling"
    return_dict: bool = True


class TwoStageScalingModel(TwoStageModel):
    """Two-stage scaling model orchestrator for BP waveform reconstruction.

    This class inherits from TwoStageModel and implements the scaling forward
    pass with BP-based waveform unscaling. Checkpoint loading uses CheckpointIO
    for unified format handling, security features, and compatibility with
    legacy CascadeTwoStageTrainer checkpoints.

    Scaling flow:
        stage1(batch) → normalized waveform
        stage2(batch with stage1 output) → {'predictions': [B,2] SBP/DBP}
        unscale waveform using SBP as max and DBP as min
        return {y_pred_waveform, y_pred_sbp, y_pred_dbp}

    The key difference from TwoStageCascadeModel is the BP-based unscaling logic
    that converts normalized waveforms back to physiologically meaningful mmHg values.

    Checkpoint Format Compatibility:
        The model supports multiple checkpoint formats via CheckpointIO:

        1. Nested cascade format (legacy CascadeTwoStageTrainer):
           {'model_state_dict': {'stage1_model': {...}, 'stage2_model': {...}}, ...}
           OR flat nested: {'stage1_model': {...}, 'stage2_model': {...}}

        2. Trainer checkpoint format:
           {'model_state_dict': <nested>, 'optimizer_state_dict': {...}, ...}

        3. DDP format: Automatically removes 'module.' prefix
        4. SafeTensors format: Supported via CheckpointIO (.safetensors files)

        The nested format matches the historical
        CascadeTwoStageTrainer.save_checkpoint() structure, ensuring
        bidirectional compatibility between trainer and orchestrator.

    Architecture Design Rationale:
        This section explains the fundamental design decisions behind the multi-stage
        orchestrator architecture and why it differs from traditional single-model
            patterns.

        1. Why Checkpoint Loading is Centralized in Orchestrators:
           Multi-stage models handle fundamentally different checkpoint formats than
           individual BaseModel instances. While BaseModel.load_checkpoint() expects
           single-model format {'model_state_dict': {...}}, orchestrators must handle
           nested format {'model_state_dict': {'stage1_model': {...}, 'stage2_model':
               {...}}}.

           Orchestrators must unwrap nested structures, coordinate loading across
           sub-models, handle format variants (trainer format, flat nested, DDP), and
           leverage CheckpointIO infrastructure. Without this orchestration, users would
           need to manually deserialize and split checkpoints before loading into
           individual models.

        2. Why Composition Over Inheritance (Liskov Substitution Principle):
           BaseModel is designed for individual trainable models with single-model
           checkpoint format, data processing methods (extract_input), and training-
           specific lifecycle hooks. Multi-stage models are composition-oriented
           orchestrators that wrap already-initialized BaseModel instances, delegate
           lifecycle calls to sub-models, and handle checkpoint coordination rather
           than data processing.

           Inheritance would violate Liskov Substitution Principle because checkpoint
           loading methods expect incompatible formats (single vs nested), and
           orchestrators don't own training semantics beyond coordination. Orchestrators
           use nn.ModuleDict for proper PyTorch registration and delegate common hooks
           (eval, train, to, set_layout) to sub-models.

        3. Checkpoint Format Compatibility (Legacy CascadeTwoStageTrainer):
           The nested format matches the historical
               CascadeTwoStageTrainer.save_checkpoint()
           structure, providing bidirectional compatibility: legacy trainers saved the
           nested format and modern orchestrators can still load it. This ensures
               seamless
           workflow continuity without manual checkpoint manipulation.
           `CheckpointIO.extract_multi_model_state()` provides optimized extraction for
           this specific format.

        4. Industry Standards Alignment:
           This pattern aligns with established industry standards:
           - PyTorch Lightning: LightningModule for individual models,
             separate orchestrators for ensembles
           - Hugging Face: Pipeline classes compose multiple models without
             inheriting from base model classes
           - TensorFlow: tf.keras.Model for individual models, separate
             coordinator classes for multi-model pipelines

           Benefits include single responsibility (each class handles
           appropriate checkpoint format), type safety (clear contracts),
           maintainability (format changes only affect orchestrator), and
           reusability (individual models remain independent).

        Note: TwoStageScalingModel shares identical checkpoint loading
        architecture with TwoStageCascadeModel because both face the same
        multi-model checkpoint format challenges. The only difference between
        the two orchestrators is the forward pass logic:
        - TwoStageCascadeModel: Simple pass-through (stage1 → stage2)
        - TwoStageScalingModel: BP-based waveform unscaling (stage1 →
          normalized waveform → stage2 → SBP/DBP → unscale)

        This reinforces that the checkpoint orchestration pattern is reusable across
        different multi-stage model types.

    Key Features:
        - Inherits nested/per-stage checkpoint loading from TwoStageModel
        - load_checkpoint(): Supports nested, separate, and mixed loading via
            CheckpointIO
        - load_state_dict(): Low-level loading with format detection and
            IncompatibleKeys return
        - auto_eval_in_forward: Optional flag to control eval() behavior in forward pass
        - Robust stage1 output extraction: Supports 'predictions',
          'y_pred_waveform', tensor outputs
        - BP-based unscaling: Converts normalized waveforms to physiological scale

    Args:
        stage1_model: Stage 1 model instance (approximation/normalized waveform
            generation)
        stage2_model: Stage 2 model instance (refinement/BP prediction)
        model_name: Optional name for the scaling model (default: "TwoStageScaling")
        auto_eval_in_forward: If True (default), automatically sets stage1 to eval()
            mode inside forward() while leaving stage2 under external control. This
                supports
            the pattern where stage1 is a frozen feature extractor and stage2 is
                trainable.
            Set to False if you want to control both stages externally.
        checkpoint_io: Optional CheckpointIO instance for checkpoint loading. Must be
            provided via Hydra configuration or explicitly in the constructor. If None,
            a RuntimeError will be raised when load_checkpoint() is called.

    See Also:
        TwoStageModel (two_stage_model.py): Checkpoint loading, delegation (eval,
        train, to, set_layout), and model_name/stage1/stage2 properties.

    Raises:
        TypeError: If stage1_model or stage2_model are not BaseModel instances
    """

    def __init__(
        self,
        stage1_model: BaseModel,
        stage2_model: BaseModel,
        model_name: str = "TwoStageScaling",
        auto_eval_in_forward: bool = True,
        return_dict: bool = True,
        checkpoint_io: CheckpointIO | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize TwoStageScalingModel with stage models and configuration.

        Args:
            stage1_model: Stage 1 model instance (approximation/normalized waveform
                generation)
            stage2_model: Stage 2 model instance (refinement/BP prediction)
            model_name: Optional name for the scaling model (default: "TwoStageScaling")
            auto_eval_in_forward: If True (default), automatically sets stage1 to eval()
                mode inside forward() while leaving stage2 under external control. This
                    supports
                the pattern where stage1 is a frozen feature extractor and stage2 is
                    trainable.
                Set to False if you want to control both stages externally.
            return_dict: Controls dict output format. The scaling model now requires
                dict outputs,
                so this flag is retained for Hydra compatibility but defaults to True.
            checkpoint_io: Optional CheckpointIO instance for checkpoint loading. Must
                be
                provided via Hydra configuration or explicitly in the constructor. If
                    None,
                a RuntimeError will be raised when load_checkpoint() is called.
        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("auto_eval_in_forward", "model_name", "checkpoint_io")
        }
        super().__init__(
            stage1_model,
            stage2_model,
            *args,
            auto_eval_in_forward=auto_eval_in_forward,
            model_name=model_name,
            checkpoint_io=checkpoint_io,
            **kwargs,
        )
        if not return_dict:
            raise ValueError(
                "TwoStageScalingModel now always returns dict outputs. "
                "Set return_dict=True (default) to continue."
            )
        self.return_dict = True

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Execute scaling forward pass with BP-based waveform unscaling.

        Scaling Flow:
            1. Run stage1 model on input batch to generate normalized waveform
            2. Extract normalized waveform prediction from stage1 outputs
            3. Create stage2 batch by replacing 'x' with stage1 waveform
            4. Run stage2 model on modified batch to predict SBP and DBP
            5. Unscale normalized waveform using SBP as max and DBP as min
            6. Return dict with unscaled waveform and BP predictions

        Args:
            batch_dict: Input batch dictionary containing at least 'x' key
                Expected format: {'x': input_tensor, 'y': target, ...}

        Returns:
            Dict[str, torch.Tensor]: Dictionary following the canonical model schema:
                - "predictions": Unscaled waveform tensor of shape [B, 1, T] containing
                    the
                  BP waveform reconstructed from the normalized stage1 output and
                      unscaled
                  using SBP/DBP predictions from stage2.
                - "extras": Dictionary containing:
                    - "y_pred_sbp": SBP (systolic blood pressure) predictions of shape
                        [B, 1]
                    - "y_pred_dbp": DBP (diastolic blood pressure) predictions of shape
                        [B, 1]
                    - "stage1_predictions": Original normalized waveform predictions
                        from stage1
                    - Additional keys from stage1 and stage2 extras dictionaries

            Stage2 contract: expects {"predictions": [B,2]} with column0=SBP and
                column1=DBP;
            these columns are split to drive unscaling.

        Note:
            If auto_eval_in_forward is True (default), stage1 is forced to eval mode
                while
            stage2 mode is controlled externally (via model.train()/model.eval()). This
                supports
            frozen feature extractor + trainable head patterns. For full external
                control of
            both stages, set auto_eval_in_forward=False.
            Deep supervision outputs (list/tuple) are handled by taking first element.

            The unscaling logic uses Global_Min_Max_Norm(unnorm=True) with per-sample
            SBP/DBP values to convert normalized [0,1] waveforms to physiological mmHg
                scale.

            When stage1 outputs [B,1,T] waveform, src_idxs/src_mask are normalized to
            ensure stage2's BaseModel.extract_input() correctly handles single-channel
                input.
        """
        if self.auto_eval_in_forward:
            # Force stage1 to eval (e.g., frozen feature extractor)
            # Leave stage2 mode under external control (trainer: .train() / .eval())
            self.models["stage1"].eval()

        # Stage 1: Generate normalized waveform approximation
        stage1_outputs = self.models["stage1"](batch_dict)
        stage1_predictions = stage1_outputs["predictions"]
        stage1_extras = stage1_outputs.get("extras", {})

        # Extract stage1 prediction (handle deep supervision and format variations)
        # Use default waveform-oriented behavior (preserve_2d_features=False)
        stage1_waveform_normalized = self._extract_stage1_output(stage1_predictions)

        # Stage 2: Create batch with stage1 output as input
        stage2_batch = batch_dict.copy()
        stage2_batch["x"] = stage1_waveform_normalized

        # Guard stage2 input selection when stage1 outputs [B,1,T]
        if (
            stage1_waveform_normalized.dim() == 3
            and stage1_waveform_normalized.size(1) == 1
        ):
            b = stage1_waveform_normalized.size(0)
            if "src_idxs" in stage2_batch:
                stage2_batch["src_idxs"] = torch.zeros(
                    (b, stage2_batch["src_idxs"].size(1)),
                    dtype=stage2_batch["src_idxs"].dtype,
                    device=stage1_waveform_normalized.device,
                )
            if "src_mask" in stage2_batch:
                stage2_batch["src_mask"] = torch.ones_like(
                    stage2_batch["src_mask"], dtype=torch.bool
                )

        stage2_outputs = self.models["stage2"](stage2_batch)
        stage2_predictions = stage2_outputs["predictions"]
        if stage2_predictions.size(1) < 2:
            raise ValueError(
                "TwoStageScalingModel requires stage2 predictions with at least two "
                "columns for SBP/DBP. "
                f"Received shape: {stage2_predictions.shape}."
            )
        stage2_extras = stage2_outputs.get("extras", {})

        y_pred_sbp = stage2_predictions[:, 0:1]
        y_pred_dbp = stage2_predictions[:, 1:2]

        y_pred_waveform_unscaled = self._unscale_waveform(
            waveform_normalized=stage1_waveform_normalized,
            sbp=y_pred_sbp,
            dbp=y_pred_dbp,
        )

        return {
            "predictions": y_pred_waveform_unscaled,
            "extras": {
                "y_pred_sbp": y_pred_sbp,
                "y_pred_dbp": y_pred_dbp,
                "stage1_predictions": stage1_predictions,
                **stage1_extras,
                **stage2_extras,
            },
        }

    def get_output_components(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Split orchestrator dict outputs into waveform, SBP, and DBP components.

        Args:
            outputs: Dict emitted by ``forward`` containing waveform/SBP/DBP entries.

        Returns:
            Dictionary with keys:
                - y_pred_waveform: Waveform component [B, 1, T]
                - y_pred_sbp: SBP component broadcast to [B, 1, T]
                - y_pred_dbp: DBP component broadcast to [B, 1, T]
        """
        if not isinstance(outputs, dict):
            raise TypeError(
                "TwoStageScalingModel.get_output_components expects canonical "
                "dict outputs."
            )

        waveform = outputs["predictions"]
        extras_raw = outputs.get("extras", {})
        extras = extras_raw if isinstance(extras_raw, dict) else {}

        try:
            sbp = extras["y_pred_sbp"]
            dbp = extras["y_pred_dbp"]
        except KeyError as exc:
            raise KeyError(
                "Canonical outputs must include 'y_pred_sbp' and 'y_pred_dbp' "
                "in extras."
            ) from exc

        time_dim = waveform.shape[-1]

        def _expand_bp(component: torch.Tensor) -> torch.Tensor:
            if component.dim() == 0:
                raise ValueError(
                    "BP component must have batch dimension; received scalar value."
                )
            batch = component.shape[0]
            component = component.reshape(batch, -1).unsqueeze(1)
            if component.size(-1) == 1 and time_dim > 1:
                component = component.expand(-1, -1, time_dim)
            elif component.size(-1) != time_dim:
                raise ValueError(
                    f"BP component must align with waveform length. Expected "
                    f"last dim {time_dim}, got {component.size(-1)}."
                )
            return component

        return {
            "y_pred_waveform": waveform,
            "y_pred_sbp": _expand_bp(sbp),
            "y_pred_dbp": _expand_bp(dbp),
        }

    def _unscale_waveform(
        self, waveform_normalized: torch.Tensor, sbp: torch.Tensor, dbp: torch.Tensor
    ) -> torch.Tensor:
        """Unscale normalized waveform using SBP and DBP predictions.

        This method converts normalized waveforms (typically [0,1] range) back to
        physiologically meaningful mmHg values by treating SBP as the maximum value
        and DBP as the minimum value for each sample.

        The unscaling is performed on-device using pure PyTorch operations with
        automatic guards against misordered SBP/DBP predictions. For each sample i:
            waveform_unscaled[i] = waveform_normalized[i] * (max - min) + min

        Args:
            waveform_normalized: Normalized waveform from stage1, shape [B, 1, T]
            sbp: SBP predictions from stage2, shape [B] or [B, 1]
            dbp: DBP predictions from stage2, shape [B] or [B, 1]

        Returns:
            Unscaled waveform tensor in mmHg, shape [B, 1, T]

        Note:
            The function guards against misordered predictions by computing:
            - sbp_e = max(sbp, dbp) as the effective maximum
            - dbp_e = min(sbp, dbp) as the effective minimum

            This ensures valid min-max scaling even if the model predicts SBP < DBP.
            All operations are performed on-device without CPU/NumPy conversion.
        """
        sbp_e = torch.maximum(sbp.view(-1, 1, 1), dbp.view(-1, 1, 1))
        dbp_e = torch.minimum(sbp.view(-1, 1, 1), dbp.view(-1, 1, 1))
        waveform_unscaled = waveform_normalized * (sbp_e - dbp_e) + dbp_e

        return waveform_unscaled


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="model", name="base_two_stage_scaling_model", node=TwoStageScalingModelConfig
)
