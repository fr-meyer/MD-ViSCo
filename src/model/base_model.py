"""Base model classes and configurations for all model types.

This module defines the minimal shared interface for single-stage and
multi-stage models. Main public names: ``BaseModel`` (abstract base for all
models; provides checkpoint loading and batch input extraction) and
``BaseModelConfig`` (Hydra config for model instantiation and checkpoint_io).
"""

from __future__ import annotations

# Standard library imports
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import torch
import torch.nn as nn

# Third-party imports
from omegaconf import MISSING

# Local imports
from src.core.domain import Vital
from src.utils.checkpoint_io import CheckpointIO
from src.utils.checkpoint_io import CheckpointIOConfig
from src.utils.checkpoint_io import remove_ddp_prefix_from_state_dict
from src.utils.constants import BATCH_KEY_INPUT
from src.utils.constants import CHECKPOINT_MODEL_STATE_KEY

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig:
    """Base config for all models (single and multi-stage); minimal shared interface.

    Includes CheckpointIOConfig for checkpoint_io in YAML; instantiated separately
    via Hydra and passed to __init__. Mirrors Trainer pattern.

    Attributes:
        _target_: Hydra instantiation target (e.g., 'src.model.nabnet.NABNet').
        model_name: Model identifier string (e.g., 'nabnet', 'mdvisco').
        checkpoint_io: Checkpoint I/O configuration for loading/saving model state.
    """

    _target_: str = MISSING
    model_name: str = MISSING
    checkpoint_io: CheckpointIOConfig = MISSING


class BaseModel(nn.Module):
    """Minimal shared interface for all models (single and multi-stage).

    CheckpointIO accepted as separate instantiated param in __init__ (Hydra-instantiated
        from config); supports standardized loading via `load_from_checkpoint_dict`.
    """

    def __init__(
        self,
        model_name: str = MISSING,
        checkpoint_io: CheckpointIO | None = None,
        supports_multi_directional: bool = False,
        input_length: int | None = None,
        dictionary_key: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize base model with configuration.

        Args:
            model_name: Name identifier for the model.
            checkpoint_io: Checkpoint I/O handler instance. If None, creates a default
                instance.
            supports_multi_directional: Whether the model supports multi-directional
                processing. Default: False.
            input_length: Input sequence length used by the model. Default: None.
            dictionary_key: Dictionary key for batch extraction (backward
                compatibility). Default: None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._model_name = model_name
        if checkpoint_io is None:
            checkpoint_io = CheckpointIO()
        self.checkpoint_io = checkpoint_io
        self._supports_multi_directional = supports_multi_directional
        self._input_length = input_length
        self._dictionary_key = dictionary_key

    @property
    def model_name(self) -> str:
        """Get the model name, handling DDP wrapping automatically."""
        return self._model_name

    @property
    def supports_multi_directional(self) -> bool:
        """Whether the model supports multi-directional processing.

        Default False; overridden by SingleStageModel.
        """
        return getattr(self, "_supports_multi_directional", False)

    @property
    def input_length(self) -> int | None:
        """Input sequence length used by the model.

        Default None; overridden by SingleStageModel.
        """
        return getattr(self, "_input_length", None)

    def extract_input(
        self, batch_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor | None, ...]:
        """Extract input from batch - can return Tensor, dict, or tuple.

        Default: return batch_dict['x']; overridden by SingleStageModel and others.
        """
        if BATCH_KEY_INPUT not in batch_dict:
            raise ValueError(
                f"extract_input expected key '{BATCH_KEY_INPUT}' in batch dict"
            )
        return batch_dict[BATCH_KEY_INPUT]

    def set_layout(self, layout: dict[Vital, int]) -> None:
        """Cache vital-to-channel mapping for efficient lookup.

        Args:
            layout: Mapping from `Vital` enum to channel indices.

        Raises:
            TypeError: If the provided layout is not a ``Dict[Vital, int]``.
        """
        if not isinstance(layout, dict):
            raise TypeError(
                "BaseModel.set_layout() expects a Dict[Vital, int] produced by "
                "build_vital_channel_mapping(); received "
                f"{type(layout).__name__}."
            )

        for key, value in layout.items():
            if not isinstance(key, Vital):
                raise TypeError(
                    f"Expected Vital enum keys, got {type(key)} for key {key!r}."
                )
            if not isinstance(value, int):
                raise TypeError(
                    f"Expected int channel indices for {key.name}, got {type(value)}."
                )

        self._vital_channel_mapping = layout
        display_dict = {v.name: idx for v, idx in layout.items()}
        logger.info(f"Model layout set: {display_dict}")

    @property
    def vital_channel_mapping(self) -> dict[Vital, int]:
        """Get the cached vital-to-channel mapping.

        Returns:
            Dict[Vital, int]: Mapping of vitals to channel indices, or empty dict if
                set_layout() not called.
        """
        return getattr(self, "_vital_channel_mapping", {})

    def _checkpoint_state_dict(self) -> dict[str, Any]:
        """Return checkpoint state dict structure for this model.

        This method defines how the model's state should be saved in checkpoints.
        All models must implement this method to define their checkpoint structure.

        Default implementation returns a single model state dict with standard key.
        Subclasses should override this for multi-model scenarios (cascade, GAN, etc.).

        Returns:
            Dict[str, Any]: Dictionary mapping checkpoint keys to state dicts.
                Examples:
                    - Single model (default):
                      ``{'model_state_dict': self.state_dict()}``
                    - Two-stage cascade:
                      ``{'stage1model_state_dict': ..., 'stage2model_state_dict': ...}``
                    - GAN:
                      ``{'generator_state_dict': ..., 'discriminator_state_dict': ...}``

        Note:
            This method is called by BaseTrainer.save_checkpoint() to get the
            model's checkpoint structure. It is the ONLY way to define how a
            model's state is saved in checkpoints.
        """
        # Default implementation: single model with standard key
        return {CHECKPOINT_MODEL_STATE_KEY: self.state_dict()}

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load a state dictionary with unified handling for DDP prefixes and logging.

        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce key matching. Default: True.
            assign: Whether to use assign() instead of load_state_dict().
                Default: False.

        Returns:
            IncompatibleKeys object containing missing and unexpected keys.

        Raises:
            RuntimeError: If strict=True and there are missing/unexpected keys.
        """
        state_dict = remove_ddp_prefix_from_state_dict(state_dict)

        incompat = super().load_state_dict(state_dict, strict=False, assign=assign)

        if incompat.missing_keys:
            logger.warning(
                f"Missing keys when loading checkpoint: {incompat.missing_keys}"
            )
        if incompat.unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading checkpoint: {incompat.unexpected_keys}"
            )

        if strict and (incompat.missing_keys or incompat.unexpected_keys):
            raise RuntimeError("Strict loading failed due to missing/unexpected keys.")

        if not incompat.missing_keys and not incompat.unexpected_keys:
            logger.info("Checkpoint loaded successfully")

        return incompat

    def load_checkpoint(
        self,
        checkpoint: str | dict[str, Any],
        key: str = CHECKPOINT_MODEL_STATE_KEY,
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Unified checkpoint loading for all models.

        Handles loading from file paths or in-memory dictionaries and delegates to
        load_state_dict() after extracting the appropriate state dict via CheckpointIO.

        Args:
            checkpoint: File path (str) or checkpoint dictionary.
            key: Key to extract from checkpoint. Default: CHECKPOINT_MODEL_STATE_KEY.
            strict: Whether to strictly enforce key matching. Default: True.

        Returns:
            IncompatibleKeys object containing missing and unexpected keys.

        Note:
            Assumes checkpoint_io is provided. For multi-stage models, use
            `load_from_checkpoint_dict` instead.
        """
        if isinstance(checkpoint, str):
            logger.info(f"Loading checkpoint from: {checkpoint}")
            checkpoint = self.checkpoint_io.load(checkpoint, map_location="cpu")
        else:
            logger.debug("Using provided checkpoint dictionary")

        extracted = self.checkpoint_io.extract_model_state(checkpoint, key=key)
        if extracted is None and key in checkpoint:
            state_dict = checkpoint[key]
            logger.debug(f"Extracted state dict with direct key: {key}")
        elif extracted is None:
            state_dict = checkpoint
            logger.debug("Key not found, assuming checkpoint is already a state dict")
        else:
            state_dict = extracted
            logger.debug(f"Extracted state dict with key: {key}")

        return self.load_state_dict(state_dict, strict=strict)

    def load_from_checkpoint_dict(
        self,
        checkpoints_dict: dict[str, Any],
        mapping: dict[str, dict[str, str]],
        strict: bool = True,
    ) -> dict[str, bool]:
        """Load from gathered checkpoints dict (single-stage: handles 'model' only).

        Args:
            checkpoints_dict: Dict of raw checkpoints keyed by component (e.g.,
                {'model': ckpt_data}).
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
            if component != "model":  # Single-stage: only 'model'
                results[component] = False
                logger.debug(f"Skipping non-model component '{component}' in BaseModel")
                continue

            ckpt = checkpoints_dict.get(component)
            if ckpt is None:
                results[component] = False
                logger.warning(f"No checkpoint for '{component}'")
                continue

            key = map_info.get("checkpoint_key", CHECKPOINT_MODEL_STATE_KEY)
            state_dict = self.checkpoint_io.extract_model_state(ckpt, key=key)

            if state_dict is None:
                results[component] = False
                logger.warning(f"No {key} in checkpoint for '{component}'")
                continue

            try:
                self.load_state_dict(state_dict, strict=strict)
                results[component] = True
                logger.info(f"BaseModel ('{component}') weights loaded successfully")
            except Exception as e:
                results[component] = False
                logger.error(f"Failed to load '{component}': {e}")

        return results
