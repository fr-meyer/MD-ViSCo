"""Two-Stage Model Base Class.

This module provides the base class for all two-stage orchestrator models that wrap
two BaseModel instances in a composition pattern. The base class handles shared
orchestration infrastructure including ModuleDict storage, delegation methods,
properties, and checkpoint loading while leaving forward pass implementation
to concrete subclasses.

Design Rationale:
    The two-stage architecture uses composition over inheritance to wrap two
    independent BaseModel instances. This design allows:
    - Independent checkpoint loading for each stage
    - Flexible stage combinations (any BaseModel can be stage1/stage2)
    - Preserved checkpoint format compatibility with trainers
    - Clean separation between orchestration and stage-specific logic

Subclasses:
    - TwoStageCascadeModel: Simple pass-through between stages
    - TwoStageScalingModel: BP-based waveform unscaling between stages

Checkpoint Loading:
    The base class supports the canonical nested cascade schema and explicit
    per-stage checkpoints:

    - Nested format: Load both stages from a single checkpoint containing
      {'stage1_model': {...}, 'stage2_model': {...}}.
    - Separate checkpoints: Load stages from individual files via the
      stage-specific keyword arguments.
    - Mixed loading: Combine separate and nested checkpoints by overriding one
      stage from a dedicated artifact while loading the other from the nested
      checkpoint.
    - Evaluator loading: load_from_checkpoint_dict() supports loading from
      multiple checkpoint managers via checkpoint_mapping (used by evaluators).

    All loading uses CheckpointIO for unified format support (.pt, .pth, .safetensors)
    with security features.

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

Key Features:
    - ModuleDict storage for proper PyTorch registration
    - Delegation methods (eval, train, to, set_layout) for both stages
    - Convenient properties (stage1, stage2, model_name)
    - Abstract forward method requiring subclass implementation
    - Flexible checkpoint loading with 4 loading scenarios
    - CheckpointIO integration with security features
    - Backward compatibility with existing cascade trainer checkpoints
"""

import logging
from collections.abc import Mapping

# Standard library imports
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Third-party imports
from omegaconf import MISSING
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys

# Local imports
from src.core.domain import Vital
from src.model.base_model import BaseModel
from src.model.base_model import BaseModelConfig
from src.utils.constants import CHECKPOINT_MODEL_STATE_KEY

logger = logging.getLogger(__name__)


@dataclass
class TwoStageModelConfig(BaseModelConfig):
    """Base configuration for all two-stage models.

    This config provides the common structure for two-stage orchestrator models.
    Concrete subclasses should inherit from this and add their specific fields.

    Attributes:
        stage1_model: Configuration for stage1 (approximation) model
        stage2_model: Configuration for stage2 (refinement) model
        model_name: Inherited from BaseModelConfig - model identifier
        checkpoint_io: Inherited from BaseModelConfig - checkpoint I/O configuration
    """

    _target_: str = "src.model.two_stage_model.TwoStageModel"
    model_name: str = MISSING
    stage1_model: BaseModelConfig = MISSING
    stage2_model: BaseModelConfig = MISSING
    auto_eval_in_forward: bool = (
        True  # Forces stage1 to eval; stage2 controlled externally
    )


class TwoStageModel(BaseModel):
    """Orchestrator for all two-stage models (cascade or scaling).

    This class provides shared orchestration infrastructure for models that wrap
    two BaseModel instances. It handles ModuleDict storage, delegation methods,
    properties, and flexible checkpoint loading while leaving forward pass
    implementation to subclasses.

    The class uses composition to wrap two independent BaseModel instances,
    allowing flexible stage combinations and preserved checkpoint compatibility.

    Checkpoint Loading Methods:
        - load_checkpoint(): Supports nested cascade checkpoints alongside explicit
            per-stage checkpoints.
        - load_state_dict(): Low-level loading for nested cascade dictionaries with DDP
            prefix handling.
        - load_from_checkpoint_dict(): Evaluator-compatible loading from
            checkpoint_mapping (handles 'model', 'stage1', 'stage2').
        - _remove_ddp_prefix(): Helper for DDP checkpoint compatibility.

    Args:
        stage1_model: First stage BaseModel instance
        stage2_model: Second stage BaseModel instance
        auto_eval_in_forward: If True, forces stage1 to eval mode in forward pass while
            leaving stage2 under external control. Supports frozen feature extractor
                patterns.
        *args: Additional positional arguments passed to BaseModel
        **kwargs: Additional keyword arguments including model_name and checkpoint_io
            passed to BaseModel

    Note: model_name and checkpoint_io are inherited from BaseModel and should be passed
    via **kwargs. Hydra configs must explicitly specify model_name (e.g., model_name:
        TwoStageCascade).
    """

    def __init__(
        self,
        stage1_model: BaseModel,
        stage2_model: BaseModel | nn.Module,
        auto_eval_in_forward: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize TwoStageModel with stage models and configuration.

        Args:
            stage1_model: First stage BaseModel instance
            stage2_model: Second stage BaseModel instance
            auto_eval_in_forward: If True, forces stage1 to eval mode in forward pass
                while
                leaving stage2 under external control. Supports frozen feature extractor
                    patterns.
        """
        super().__init__(*args, **kwargs)

        if not isinstance(stage1_model, BaseModel):
            raise TypeError(
                f"stage1_model must be a BaseModel instance, got {type(stage1_model)}"
            )
        # Allow nn.Module for stage2 to support MultiMLPRegressor
        if not isinstance(stage2_model, BaseModel) and not isinstance(
            stage2_model, nn.Module
        ):
            raise TypeError(
                f"stage2_model must be a BaseModel or nn.Module instance, "
                f"got {type(stage2_model)}"
            )

        # ModuleDict required so PyTorch registers stage1/stage2 as submodules
        self.models = nn.ModuleDict({"stage1": stage1_model, "stage2": stage2_model})

        self.auto_eval_in_forward = auto_eval_in_forward

        logger.info(
            f"Initialized {self._model_name} with "
            f"stage1={type(stage1_model).__name__}, "
            f"stage2={type(stage2_model).__name__}"
        )

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Execute two-stage forward pass.

        Must be implemented by subclasses. Each subclass implements its
        specific two-stage logic (cascade vs scaling).

        Args:
            batch_dict: Input batch dictionary containing at least 'x' key

        Returns:
            Dict following canonical schema:
                {'predictions': <tensor or tuple>, 'extras': {...}}

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "forward() must be implemented by subclasses. "
            "Use TwoStageCascadeModel or TwoStageScalingModel."
        )

    # Delegation methods for proper two-stage behavior

    def eval(self) -> "TwoStageModel":
        """Set both stage models to evaluation mode.

        Returns:
            Self for method chaining
        """
        return super().eval()

    def train(self, mode: bool = True) -> "TwoStageModel":
        """Set both stage models to training mode.

        Args:
            mode: If True, set to training mode; if False, set to eval mode

        Returns:
            Self for method chaining
        """
        return super().train(mode)

    def to(self, *args, **kwargs) -> "TwoStageModel":
        """Move both stage models to specified device/dtype/memory_format.

        Args:
            *args: Positional arguments (device, dtype, etc.)
            **kwargs: Keyword arguments (memory_format, etc.)

        Returns:
            Self for method chaining
        """
        self.models["stage1"].to(*args, **kwargs)
        self.models["stage2"].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def set_layout(self, layout: dict[Vital, int]) -> None:
        """Set vital-to-channel layout for this module only.

        The trainer's recursive walker (recursively_set_layout) handles DDP
        unwrapping and propagation to child modules (stage1 and stage2). Each
        module manages only its own cached layout state.

        Args:
            layout: Mapping produced by ``build_vital_channel_mapping()`` assigning
                channel indices to each ``Vital`` enum value.
        """
        super().set_layout(layout)

    def _extract_stage1_output(
        self,
        stage1_outputs: torch.Tensor | list | tuple,
        preserve_2d_features: bool = False,
    ) -> torch.Tensor:
        """Extract prediction tensor from stage1 outputs.

        This shared method handles deep-supervision outputs (lists/tuples) and
        dimension conversion for both feature tensors and waveform tensors.

        Args:
            stage1_outputs: Stage1 model predictions (Tensor, List, or Tuple)
            preserve_2d_features: If True, preserves 2D feature tensors [B, F] without
                unsqueezing. If False (default), unsqueezes any 2D tensor to 3D format
                [B, 1, T] for waveform-oriented models (e.g., scaling models).

        Returns:
            Tensor containing stage1 prediction with appropriate dimensions:
            - [B, F] if preserve_2d_features=True and input is 2D
            - [B, 1, T] if preserve_2d_features=False and input is 2D
            - [B, 1, T] if input is already 3D (unchanged)
        """
        stage1_tensor = stage1_outputs

        if isinstance(stage1_tensor, (list, tuple)):
            stage1_tensor = stage1_tensor[0]

        if stage1_tensor is None:
            raise ValueError(
                f"{self.__class__.__name__} expected stage1 predictions, got None."
            )

        # Unsqueeze 2D to [B, 1, T] only when not preserving 2D features (e.g. cascade)
        if stage1_tensor.dim() == 2 and not preserve_2d_features:
            stage1_tensor = stage1_tensor.unsqueeze(1)

        return stage1_tensor

    # Properties for convenient access

    @property
    def stage1(self) -> BaseModel:
        """Get stage1 model for direct access."""
        m = self.models["stage1"]
        assert isinstance(m, BaseModel), "stage1 must be a BaseModel"
        return m

    @property
    def stage2(self) -> BaseModel:
        """Get stage2 model for direct access."""
        m = self.models["stage2"]
        assert isinstance(m, BaseModel), "stage2 must be a BaseModel"
        return m

    # Checkpoint loading methods

    def _remove_ddp_prefix(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Remove 'module.' prefix from DDP checkpoints.

        Args:
            state_dict: State dictionary possibly containing DDP prefix

        Returns:
            State dictionary with 'module.' prefix removed
        """
        if any(key.startswith("module.") for key in state_dict):
            logger.debug("Removing 'module.' prefix from DDP checkpoint")
            return {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> IncompatibleKeys:
        """Load a two-stage checkpoint from the canonical nested schema.

        Args:
            state_dict: Cascade checkpoint dictionary (nested or wrapped in trainer
                format).
            strict: Whether to strictly enforce key matching when delegating to stages.

        Returns:
            ``IncompatibleKeys`` containing aggregated missing/unexpected keys across
                both stages.

        Raises:
            ValueError: If the provided state dict does not contain nested
                ``stage1``/``stage2`` payloads.
        """
        # Unwrap trainer-format dicts with top-level 'model_state_dict'
        if "model_state_dict" in state_dict and isinstance(
            state_dict["model_state_dict"], dict
        ):
            logger.debug(
                "Unwrapping trainer-format checkpoint with top-level 'model_state_dict'"
            )
            state_dict = state_dict["model_state_dict"]

        missing_keys_all: list[str] = []
        unexpected_keys_all: list[str] = []

        nested_keys: tuple[str, str] | None = None
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected state_dict to be dict, got {type(state_dict)}")

        if {"stage1_model", "stage2_model"}.issubset(state_dict.keys()):
            nested_keys = ("stage1_model", "stage2_model")
            logger.info(
                "Loading cascade checkpoint with nested format (stage*_model keys)"
            )
        elif {"stage1", "stage2"}.issubset(state_dict.keys()):
            nested_keys = ("stage1", "stage2")
            logger.info("Loading cascade checkpoint with nested format (stage* keys)")

        if nested_keys is None:
            raise ValueError(
                "Unsupported checkpoint format for TwoStageModel. Expected nested keys "
                "containing both stages (e.g. stage1_model, stage2_model)."
            )

        stage1_state_raw = state_dict.get(nested_keys[0])
        stage2_state_raw = state_dict.get(nested_keys[1])

        if stage1_state_raw is None:
            raise ValueError(f"Missing required key '{nested_keys[0]}' in checkpoint")
        if stage2_state_raw is None:
            raise ValueError(f"Missing required key '{nested_keys[1]}' in checkpoint")

        if not isinstance(stage1_state_raw, dict):
            raise TypeError(
                f"Expected {nested_keys[0]} to be dict, got {type(stage1_state_raw)}"
            )
        if not isinstance(stage2_state_raw, dict):
            raise TypeError(
                f"Expected {nested_keys[1]} to be dict, got {type(stage2_state_raw)}"
            )

        stage1_state_dict = self._remove_ddp_prefix(stage1_state_raw)
        stage2_state_dict = self._remove_ddp_prefix(stage2_state_raw)

        incompat1 = self.stage1.load_state_dict(
            stage1_state_dict, strict=strict, assign=assign
        )
        missing_keys, unexpected_keys = (
            incompat1.missing_keys,
            incompat1.unexpected_keys,
        )
        if missing_keys:
            logger.debug(
                f"Stage1 missing keys: {[f'stage1.{k}' for k in missing_keys]}"
            )
        if unexpected_keys:
            logger.debug(
                f"Stage1 unexpected keys: {[f'stage1.{k}' for k in unexpected_keys]}"
            )
        missing_keys_all.extend(missing_keys)
        unexpected_keys_all.extend(unexpected_keys)

        incompat2 = self.stage2.load_state_dict(
            stage2_state_dict, strict=strict, assign=assign
        )
        missing_keys, unexpected_keys = (
            incompat2.missing_keys,
            incompat2.unexpected_keys,
        )
        if missing_keys:
            logger.debug(
                f"Stage2 missing keys: {[f'stage2.{k}' for k in missing_keys]}"
            )
        if unexpected_keys:
            logger.debug(
                f"Stage2 unexpected keys: {[f'stage2.{k}' for k in unexpected_keys]}"
            )
        missing_keys_all.extend(missing_keys)
        unexpected_keys_all.extend(unexpected_keys)

        return IncompatibleKeys(missing_keys_all, unexpected_keys_all)

    def load_checkpoint(
        self,
        checkpoint: str | dict[str, Any] | None = None,
        key: str = CHECKPOINT_MODEL_STATE_KEY,
        strict: bool = True,
        *,
        stage1_checkpoint: str | dict[str, Any] | None = None,
        stage2_checkpoint: str | dict[str, Any] | None = None,
    ) -> IncompatibleKeys:
        """Load checkpoint artifacts for two-stage models.

        This method prioritizes explicit per-stage checkpoints over nested checkpoints
        so callers can override one stage while keeping the other intact. Remaining
        stages fall back to the canonical nested cascade schema.

        Args:
            checkpoint: Optional nested checkpoint (path or dict) containing both
                stages.
            strict: Whether to strictly enforce key matching (default: True).
            stage1_checkpoint: Optional separate checkpoint for stage1 (path or dict).
            stage2_checkpoint: Optional separate checkpoint for stage2 (path or dict).

        Returns:
            ``IncompatibleKeys`` aggregating missing/unexpected keys across stages.

        Raises:
            ValueError: If no checkpoint parameters provided.
            RuntimeError: If ``checkpoint_io`` is None.
            FileNotFoundError: If a checkpoint path does not exist.
        """
        if (
            checkpoint is None
            and stage1_checkpoint is None
            and stage2_checkpoint is None
        ):
            raise ValueError(
                "No checkpoint specified. Provide either 'checkpoint', "
                "'stage1_checkpoint', or 'stage2_checkpoint'"
            )

        if self.checkpoint_io is None:
            raise RuntimeError(
                "checkpoint_io is required but was not provided. "
                "Provide checkpoint_io via Hydra config or constructor."
            )

        io = self.checkpoint_io
        missing_keys_all = []
        unexpected_keys_all = []
        stage1_loaded = False
        stage2_loaded = False

        # Priority 1: Load separate checkpoints (highest priority)
        # Enables flexible loading of individual stages for fine-tuning or evaluation
        if stage1_checkpoint is not None:
            _src = "path" if isinstance(stage1_checkpoint, str) else "dict"
            logger.info(f"Loading stage1 from separate checkpoint: {_src}")

            if isinstance(stage1_checkpoint, str):
                stage1_checkpoint = io.load(stage1_checkpoint, map_location="cpu")

            try:
                stage1_state = io.extract_model_state(stage1_checkpoint)
            except KeyError:
                stage1_state = stage1_checkpoint

            if stage1_state is None:
                raise ValueError("Failed to extract stage1 state dict from checkpoint")

            if not isinstance(stage1_state, dict):
                raise TypeError(
                    f"Expected stage1_state to be dict, got {type(stage1_state)}"
                )

            stage1_state = self._remove_ddp_prefix(stage1_state)

            if "stage1_model" in stage1_state or "stage2_model" in stage1_state:
                raise ValueError(
                    "Nested cascade in stage1_checkpoint. Use 'checkpoint' parameter "
                    "instead: model.load_checkpoint(checkpoint='path.pth')"
                )

            missing_keys, unexpected_keys = self.stage1.load_state_dict(
                stage1_state, strict=strict
            )
            if missing_keys:
                logger.debug(
                    f"Stage1 missing keys: {[f'stage1.{k}' for k in missing_keys]}"
                )
            if unexpected_keys:
                logger.debug(
                    f"Stage1 unexpected keys: "
                    f"{[f'stage1.{k}' for k in unexpected_keys]}"
                )
            missing_keys_all.extend(missing_keys)
            unexpected_keys_all.extend(unexpected_keys)
            stage1_loaded = True

        if stage2_checkpoint is not None:
            logger.info(
                f"Loading stage2 from separate checkpoint: "
                f"{stage2_checkpoint if isinstance(stage2_checkpoint, str) else 'dict'}"
            )

            if isinstance(stage2_checkpoint, str):
                stage2_checkpoint = io.load(stage2_checkpoint, map_location="cpu")

            try:
                stage2_state = io.extract_model_state(stage2_checkpoint)
            except KeyError:
                stage2_state = stage2_checkpoint

            if stage2_state is None:
                raise ValueError("Failed to extract stage2 state dict from checkpoint")

            if not isinstance(stage2_state, dict):
                raise TypeError(
                    f"Expected stage2_state to be dict, got {type(stage2_state)}"
                )

            stage2_state = self._remove_ddp_prefix(stage2_state)

            if "stage1_model" in stage2_state or "stage2_model" in stage2_state:
                raise ValueError(
                    "Nested cascade in stage2_checkpoint. Use 'checkpoint' parameter "
                    "instead: model.load_checkpoint(checkpoint='path.pth')"
                )

            missing_keys, unexpected_keys = self.stage2.load_state_dict(
                stage2_state, strict=strict
            )
            if missing_keys:
                logger.debug(
                    f"Stage2 missing keys: {[f'stage2.{k}' for k in missing_keys]}"
                )
            if unexpected_keys:
                keys_str = [f"stage2.{k}" for k in unexpected_keys]
                logger.debug(f"Stage2 unexpected keys: {keys_str}")
            missing_keys_all.extend(missing_keys)
            unexpected_keys_all.extend(unexpected_keys)
            stage2_loaded = True

        # Priority 2: Load remaining stages from nested checkpoint
        # Fallback when separate checkpoints unavailable (backward compatibility)
        if checkpoint is not None and (not stage1_loaded or not stage2_loaded):
            logger.info("Loading remaining stages from nested checkpoint")

            if isinstance(checkpoint, str):
                logger.info(f"Loading checkpoint from: {checkpoint}")
                checkpoint = io.load(checkpoint, map_location="cpu")

            states = io.extract_multi_model_state(
                checkpoint, {"stage1": "stage1_model", "stage2": "stage2_model"}
            )

            if states.get("stage1") is not None and states.get("stage2") is not None:
                logger.info(
                    "Loading cascade checkpoint using extract_multi_model_state()"
                )

                if not stage1_loaded:
                    stage1_state_raw = states.get("stage1")
                    if stage1_state_raw is None:
                        raise ValueError("Missing 'stage1' in extracted states")
                    if not isinstance(stage1_state_raw, dict):
                        raise TypeError(
                            f"Expected stage1 state dict, got {type(stage1_state_raw)}"
                        )
                    stage1_state_dict = self._remove_ddp_prefix(stage1_state_raw)
                    missing_keys, unexpected_keys = self.stage1.load_state_dict(
                        stage1_state_dict, strict=strict
                    )
                    if missing_keys:
                        logger.debug(
                            f"Stage1 missing keys: "
                            f"{[f'stage1.{k}' for k in missing_keys]}"
                        )
                    if unexpected_keys:
                        logger.debug(
                            f"Stage1 unexpected keys: "
                            f"{[f'stage1.{k}' for k in unexpected_keys]}"
                        )
                    missing_keys_all.extend(missing_keys)
                    unexpected_keys_all.extend(unexpected_keys)

                if not stage2_loaded:
                    stage2_state_raw = states.get("stage2")
                    if stage2_state_raw is None:
                        raise ValueError("Missing 'stage2' in extracted states")
                    if not isinstance(stage2_state_raw, dict):
                        raise TypeError(
                            f"Expected stage2 state dict, got {type(stage2_state_raw)}"
                        )
                    stage2_state_dict = self._remove_ddp_prefix(stage2_state_raw)
                    missing_keys, unexpected_keys = self.stage2.load_state_dict(
                        stage2_state_dict, strict=strict
                    )
                    if missing_keys:
                        logger.debug(
                            f"Stage2 missing keys: "
                            f"{[f'stage2.{k}' for k in missing_keys]}"
                        )
                    if unexpected_keys:
                        logger.debug(
                            f"Stage2 unexpected keys: "
                            f"{[f'stage2.{k}' for k in unexpected_keys]}"
                        )
                    missing_keys_all.extend(missing_keys)
                    unexpected_keys_all.extend(unexpected_keys)
            else:
                # Fall back to extract_model_state() + load_state_dict()
                try:
                    state_dict = io.extract_model_state(checkpoint, key=key)
                except KeyError:
                    state_dict = checkpoint

                if (
                    state_dict is None
                    and isinstance(checkpoint, dict)
                    and key in checkpoint
                ):
                    state_dict = checkpoint[key]
                if state_dict is None:
                    state_dict = checkpoint

                if state_dict is None:
                    raise ValueError("Failed to extract state dict from checkpoint")

                # Delegate to load_state_dict (which handles trainer-format wrappers)
                return self.load_state_dict(state_dict, strict=strict)

        logger.info("Two-stage model checkpoint loading completed")
        return IncompatibleKeys(missing_keys_all, unexpected_keys_all)

    def load_from_checkpoint_dict(
        self,
        checkpoints_dict: dict[str, Any],
        mapping: dict[str, dict[str, str]],
        strict: bool = True,
    ) -> dict[str, bool]:
        """Load from gathered checkpoints dict using mapping (two-stage: handles
            'model', 'stage1',
        'stage2').

        This method is shared by all TwoStageModel subclasses (TwoStageCascadeModel,
            TwoStageScalingModel)
        since they all use the same self.models structure for stage1 and stage2.

        Args:
            checkpoints_dict: Dict of raw checkpoints keyed by component.
            mapping: checkpoint_mapping dict specifying keys/targets.
            strict: Whether to strictly enforce that keys in state_dict match the model.
                Default: True.

        Returns:
            Dict of load success per component.
        """
        results = {}

        if self.checkpoint_io is None:
            raise ValueError("checkpoint_io must be provided")

        for component, map_info in mapping.items():
            if component not in ["model", "stage1", "stage2"]:
                results[component] = False
                logger.debug(
                    f"Skipping non-model component '{component}' in TwoStageModel"
                )
                continue

            ckpt = checkpoints_dict.get(component)
            if ckpt is None:
                results[component] = False
                logger.warning(f"No checkpoint for '{component}'")
                continue

            key = map_info.get("checkpoint_key", "model_state_dict")
            state_dict = self.checkpoint_io.extract_model_state(ckpt, key=key)

            if state_dict is None:
                results[component] = False
                logger.warning(f"No {key} in checkpoint for '{component}'")
                continue

            if not isinstance(state_dict, dict):
                results[component] = False
                logger.warning(
                    f"Invalid state_dict type for '{component}': {type(state_dict)}"
                )
                continue

            target = map_info.get("target") or map_info.get("target_key")
            try:
                if target in ["stage1", "stage2"]:
                    sub_model = self.models[target]  # e.g., self.models['stage1']
                    sub_model.load_state_dict(state_dict, strict=strict)
                    results[component] = True
                    logger.info(
                        f"TwoStageModel sub-module '{target}' ('{component}') "
                        f"loaded successfully"
                    )
                else:
                    # Joint 'model' checkpoint
                    self.load_state_dict(state_dict, strict=strict)
                    results[component] = True
                    logger.info(
                        f"TwoStageModel ('{component}') weights loaded successfully"
                    )
            except Exception as e:
                results[component] = False
                logger.error(f"Failed to load '{component}': {e}")

        return results

    def _checkpoint_state_dict(self) -> dict[str, Any]:
        """Return checkpoint state dict structure for two-stage models.

        Returns nested state dicts for stage1 and stage2 with consistent keys
        that align with trainer's nested format and
            CheckpointIO.extract_multi_model_state()
        expectations.

        Returns:
            Dict[str, Any]: Dictionary with nested structure containing:
                - 'model_state_dict': Nested dict with 'stage1_model' and 'stage2_model'
                    keys
                - 'stage1_model_state_dict': Direct stage1 state dict (for
                    compatibility)
                - 'stage2_model_state_dict': Direct stage2 state dict (for
                    compatibility)
        """
        stage1_state = self.models["stage1"].state_dict()
        stage2_state = self.models["stage2"].state_dict()
        return {
            "model_state_dict": {
                "stage1_model": stage1_state,
                "stage2_model": stage2_state,
            },
            "stage1_model_state_dict": stage1_state,
            "stage2_model_state_dict": stage2_state,
        }
