"""Two-Stage Cascade Model for Evaluator Usage.

This module provides a lightweight orchestrator interface that wraps two BaseModel
instances for cascade inference. Unlike cascade_two_stage_trainer.py which handles
training logic, this module provides a single model interface for evaluators.

Usage Example:
    # Via Hydra config (recommended):
    model:
      _target_: src.model.cascade_model.TwoStageCascadeModel
      stage1_model:
        _target_: src.model.mdvisco.MDViSCo
        # ... stage1 config
      stage2_model:
        _target_: src.model.nabnet.ShallowUNetBP
        # ... stage2 config

    # Via Python (for advanced usage):
    stage1 = MDViSCo(...)
    stage2 = ShallowUNetBP(...)
    cascade = TwoStageCascadeModel(stage1, stage2)

    # Recommended: Single checkpoint file (simple cascade logic)
    cascade.load_checkpoint("cascade_checkpoint.pth")

    # Special case: Separate checkpoints (only for advanced scenarios)
    # Not recommended - use single checkpoint file for production
    cascade.load_checkpoint(stage1_checkpoint="stage1.pth",
        stage2_checkpoint="stage2.pth")

    # Use in evaluator:
    from src.trainers.trainer import recursively_set_layout
    from src.core.domain import Vital
    layout = {Vital.PPG: 0, Vital.ECG: 1, Vital.ABP: 2}  # Dict[Vital, int] mapping
    recursively_set_layout(cascade, layout)
    cascade.eval()
    cascade.to(device)
    outputs = cascade(batch_dict)

Feature Extraction Mode:
    When stage1 is configured for feature extraction (e.g., ShallowUNet with
        feature_extraction_only=True in YAML),
    the cascade receives features instead of waveforms from stage1.

    Stage1 Contract:
        - If stage1.feature_extraction_only=True (YAML-configured): returns 2D features
            [B, F]
        - Otherwise: returns 3D waveforms [B, 1, T]

    Stage2 Contract:
        - If stage2 is BaseModel: receives batch_dict with 'x' key containing stage1
            output
        - If stage2 is nn.Module: receives raw tensor directly from stage1

    Tensor Shape Flow:
        - Feature extraction: Stage1 [B, F] → Stage2 [B, F] (preserved 2D)
        - Waveform mode: Stage1 [B, 1, T] → Stage2 [B, 1, T] (preserved 3D)

Checkpoint Format Compatibility:
    - Nested format (from cascade trainer): {'stage1_model': {...}, 'stage2_model':
        {...}}
    - DDP format: Handles 'module.' prefix removal
    - Flat format: Requires explicit target parameter ('stage1' or 'stage2')
    - Single checkpoint file approach: stage1 output → stage2 input, period
    - Component → manager → checkpoint_key → target_key mapping supported

Key Features:
    - load_checkpoint(): Convenient loading from path or dict, handles trainer format
    - load_state_dict(): Low-level loading with format detection and IncompatibleKeys
        return
    - auto_eval_in_forward: Optional flag to control eval() behavior in forward pass
    - Robust stage1 output extraction: Supports 'y_pred', 'y_pred_waveform', tensor
        outputs
    - Instance-configured feature extraction: Stage1 mode determined by YAML
        configuration
"""

import logging

# Standard library imports
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.model.base_model import BaseModel
from src.model.two_stage_model import TwoStageModel
from src.model.two_stage_model import TwoStageModelConfig
from src.utils.checkpoint_io import CheckpointIO

logger = logging.getLogger(__name__)


@dataclass
class TwoStageCascadeModelConfig(TwoStageModelConfig):
    """Configuration for TwoStageCascadeModel.

    This config inherits from TwoStageModelConfig and is designed for evaluator usage
    where both stage1 and stage2 models are instantiated by Hydra. For training,
    pair the cascade model with model-specific trainers (e.g.,
        `trainer=refinement_trainer_nabnet` for NABNet cascade).

    Inherits CheckpointIOConfig from BaseModelConfig; defines structure for separate
        Hydra instantiation and passing to __init__.

    Attributes:
        _target_: Hydra target path to TwoStageCascadeModel class
        model_name: Identifier for the cascade model (overrides default)
        stage1_model: Inherited from TwoStageModelConfig - stage1 configuration
        stage2_model: Inherited from TwoStageModelConfig - stage2 configuration
        checkpoint_io: Inherited from BaseModelConfig - checkpoint I/O configuration
    """

    _target_: str = "src.model.cascade_model.TwoStageCascadeModel"
    model_name: str = "TwoStageCascade"


class TwoStageCascadeModel(TwoStageModel):
    """Two-Stage Cascade Model Orchestrator.

    This class inherits from TwoStageModel and implements the cascade forward pass:
    stage1(batch) → extract output → replace batch['x'] → stage2(batch).
    Checkpoint loading uses CheckpointIO for unified format handling, security features,
    and compatibility with legacy CascadeTwoStageTrainer checkpoints.

    The class provides evaluator-compatible methods while properly managing both
    underlying models through the inherited TwoStageModel infrastructure.

    CheckpointIO accepted as separate instantiated param in __init__ (Hydra-instantiated
    from inherited config); supports standardized loading via
    `load_from_checkpoint_dict`. Sub-models receive passed checkpoint_io.

    Checkpoint Format Compatibility:
        The model supports multiple checkpoint formats via CheckpointIO:

        1. Nested cascade format (legacy CascadeTwoStageTrainer):
           {'model_state_dict': {'stage1_model': {...}, 'stage2_model': {...}}, ...}
           OR flat nested: {'stage1_model': {...}, 'stage2_model': {...}}

        2. Trainer checkpoint format:
           {'model_state_dict': <nested>, 'optimizer_state_dict': {...}, ...}

        3. DDP format: Automatically removes 'module.' prefix

        4. Flat format: Requires explicit target parameter ('stage1' or 'stage2')

        5. SafeTensors format: Supported via CheckpointIO (.safetensors files)

        The nested format matches the historical
            CascadeTwoStageTrainer.save_checkpoint() structure,
        ensuring bidirectional compatibility between trainer and orchestrator.

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
           nested format and modern orchestrators can still load it. This ensures a
           smooth path for older checkpoints without manual manipulation.
           `CheckpointIO.extract_multi_model_state()` provides optimized extraction for
           this specific format.

        4. Industry Standards Alignment:
           This pattern aligns with established industry standards:
           - PyTorch Lightning: LightningModule for individual models, separate
               orchestrators for ensembles
           - Hugging Face: Pipeline classes compose multiple models without inheriting
               from base model classes
           - TensorFlow: tf.keras.Model for individual models, separate coordinator
               classes for multi-model pipelines

           Benefits include single responsibility (each class handles appropriate
               checkpoint format),
           type safety (clear contracts), maintainability (format changes only affect
               orchestrator),
           and reusability (individual models remain independent).

    Key Features:
        - Inherits flexible checkpoint loading from TwoStageModel (nested, separate,
            mixed, partial)
        - load_checkpoint(): Supports 4 loading scenarios via CheckpointIO with security
            features
        - load_state_dict(): Low-level loading with format detection and
            IncompatibleKeys return
        - auto_eval_in_forward: Optional flag to control eval() behavior in forward pass
        - Robust stage1 output extraction: Supports 'y_pred', 'y_pred_waveform', tensor
            outputs

    Args:
        stage1_model: Stage 1 model instance (approximation/waveform generation)
        stage2_model: Stage 2 model instance (refinement/BP extraction) - can be
            BaseModel or nn.Module
        model_name: Optional name for the cascade model (default: "TwoStageCascade")
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
        TypeError: If stage1_model is not a BaseModel instance or stage2_model is not
            BaseModel/nn.Module
    """

    def __init__(
        self,
        stage1_model: BaseModel,
        stage2_model: BaseModel | nn.Module,
        model_name: str = "TwoStageCascade",
        auto_eval_in_forward: bool = True,
        checkpoint_io: CheckpointIO | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize TwoStageCascadeModel with stage models and configuration.

        Args:
            stage1_model: Stage 1 model instance (approximation/waveform generation)
            stage2_model: Stage 2 model instance (refinement/BP extraction) - can be
                BaseModel or nn.Module
            model_name: Optional name for the cascade model (default: "TwoStageCascade")
            auto_eval_in_forward: If True (default), automatically sets stage1 to eval()
                mode inside forward() while leaving stage2 under external control. This
                    supports
                the pattern where stage1 is a frozen feature extractor and stage2 is
                    trainable.
                Set to False if you want to control both stages externally.
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

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Execute cascade forward pass.

        Cascade Flow:
            1. Run stage1 model on input batch to generate waveform or features
            2. Extract prediction from stage1 outputs (handles features vs waveforms)
            3. Pass stage1 output to stage2 (handles BaseModel vs nn.Module stage2)
            4. Return stage2 outputs as final cascade result

        Feature Extraction Mode:
            When stage1 is configured with feature_extraction_only=True (YAML),
            it returns features instead of waveforms. This enables:
            - Stage1 returns 2D features [B, F] instead of 3D waveforms [B, 1, T]
            - Stage2 receives appropriate input format based on its type
            - Preserves 2D feature tensors without unnecessary dimension conversion

        Args:
            batch_dict: Input batch dictionary containing at least 'x' key
                Expected format: {'x': input_tensor, 'y': target, ...}

        Returns:
            Dictionary with stage2 outputs (final cascade predictions)
                Typical format: {'predictions': [B,2] SBP/DBP}
                or {'sbp': sbp_pred, 'dbp': dbp_pred} for MultiMLPRegressor

        Note:
            If auto_eval_in_forward is True (default), stage1 is forced to eval mode
                while
            stage2 mode is controlled externally (via model.train()/model.eval()). This
                supports
            frozen feature extractor + trainable head patterns. For full external
                control of
            both stages, set auto_eval_in_forward=False.
            Deep supervision outputs (list/tuple) are handled by taking first element.

            Stage1 output format detection:
            - 2D tensors with small second dimension (≤2048) are treated as features
            - 2D tensors with large second dimension (>2048) are treated as waveforms
            - 3D tensors are passed through unchanged

            Stage2 input handling:
            - BaseModel stage2: receives batch_dict with 'x' containing stage1 output
            - nn.Module stage2: receives raw tensor directly from stage1
        """
        # Conditionally set eval mode based on auto_eval_in_forward flag
        if self.auto_eval_in_forward:
            # Force stage1 to eval (e.g., frozen feature extractor)
            # Leave stage2 mode under external control (trainer: .train() / .eval())
            self.models["stage1"].eval()

        # Stage 1: Waveform approximation or features (instance-configured mode)
        stage1_outputs = self.models["stage1"](batch_dict)
        stage1_predictions = stage1_outputs["predictions"]
        stage1_extras = stage1_outputs.get("extras", {})

        # Extract stage1 prediction (handle deep supervision)
        # Use preserve_2d_features=True to retain feature tensors for cascade models
        stage1_prediction = self._extract_stage1_output(
            stage1_predictions, preserve_2d_features=True
        )

        # Stage 2: Handle different stage2 input formats
        if isinstance(self.models["stage2"], BaseModel):
            # Stage2 is BaseModel - expects batch_dict
            stage2_batch = batch_dict.copy()
            stage2_batch["x"] = stage1_prediction

            # Guard stage2 input selection when stage1 outputs [B,1,T]
            if stage1_prediction.dim() == 3 and stage1_prediction.size(1) == 1:
                b = stage1_prediction.size(0)
                if "src_idxs" in stage2_batch:
                    stage2_batch["src_idxs"] = torch.zeros(
                        (b, stage2_batch["src_idxs"].size(1)),
                        dtype=stage2_batch["src_idxs"].dtype,
                        device=stage1_prediction.device,
                    )
                if "src_mask" in stage2_batch:
                    stage2_batch["src_mask"] = torch.ones_like(
                        stage2_batch["src_mask"], dtype=torch.bool
                    )

            # Run stage2 refinement
            stage2_outputs = self.models["stage2"](stage2_batch)
        else:
            # Stage2 is raw nn.Module - expects tensor input directly
            stage2_outputs = self.models["stage2"](stage1_prediction)

        stage2_predictions = stage2_outputs["predictions"]
        stage2_extras = stage2_outputs.get("extras", {})
        sbp_pred = stage2_predictions[:, 0:1]
        dbp_pred = stage2_predictions[:, 1:2]

        return {
            "predictions": stage2_predictions,
            "extras": {
                "y_pred_sbp": sbp_pred,
                "y_pred_dbp": dbp_pred,
                "stage1_predictions": stage1_predictions,
                "stage2_predictions": stage2_predictions,
                **stage1_extras,
                **stage2_extras,
            },
        }


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="model", name="base_two_stage_cascade_model", node=TwoStageCascadeModelConfig
)
