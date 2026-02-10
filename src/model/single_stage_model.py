"""Single-stage model base class for MD-ViSCo.

This module implements the SingleStageModel base class for all single-stage
models in the MD-ViSCo project. SingleStageModel serves as an intermediate
class between BaseModel and concrete single-stage models, providing
single-stage specific functionality while inheriting the minimal shared
interface from BaseModel.

Inheritance Hierarchy:
    BaseModel (minimal shared interface)
    ├── SingleStageModel (single-stage specific logic)
    │   ├── NABNet, ShallowUNet, ShallowUNetBP
    │   ├── UNet_SwinUnet, BPModel (MDViSCo)
    │   ├── PatchTST
    │   ├── WaveNetModel
    │   ├── UNetDS64, MultiResUNet1D (PPG2ABP)
    │   ├── GeneratorUNet, Discriminator, P2EWGAN
    │   └── AFClassifier
    └── TwoStageModel (two-stage specific logic)

This separation of concerns was introduced in Phase 3 refactoring to create
clean inheritance hierarchies: BaseModel (shared), SingleStageModel
(single-stage), TwoStageModel (two-stage).
"""

from __future__ import annotations

import logging

# Standard library imports
from dataclasses import dataclass
from typing import Any

import torch

# Third-party imports
from omegaconf import MISSING

# Local imports
from src.model.base_model import BaseModel
from src.model.base_model import BaseModelConfig
from src.utils.constants import CHECKPOINT_MODEL_STATE_KEY

logger = logging.getLogger(__name__)


@dataclass
class SingleStageModelConfig(BaseModelConfig):
    """Configuration for all single-stage models (models that process input in one
        pass).

    This configuration class extends BaseModelConfig with single-stage specific fields
    that are common to all single-stage models in the MD-ViSCo project.

    Inherited from BaseModelConfig:
        _target_: str = MISSING
        model_name: str = MISSING
        checkpoint_io: CheckpointIOConfig = MISSING

    Single-stage specific fields:
        supports_multi_directional: Whether model supports multi-directional processing
        input_length: Common input length parameter used by all models

    Note: input_length is marked Optional for schema flexibility
    but is REQUIRED for Hydra instantiation. The __init__ signature enforces this.
    For stricter compile-time validation, consider changing input_length to use
    omegaconf.MISSING instead of Optional[int].
    """

    supports_multi_directional: bool = False
    input_length: int | None = None


class SingleStageModel(BaseModel):
    """Single-stage models that process input in one pass.

    This class contains single-stage specific functionality including:
    - Input extraction from NEW format batch structure
    - Checkpoint loading with CheckpointIO integration
    - State dict loading with DDP prefix stripping
    - Single-stage specific properties and configuration

    All single-stage models inherit from this class to get common functionality
    while maintaining clean separation from two-stage models.
    """

    def __init__(
        self,
        supports_multi_directional: bool = False,
        input_length: int = MISSING,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize SingleStageModel with single-stage specific parameters.

        Args:
            supports_multi_directional: Whether model supports multi-directional
                processing
            input_length: Common input length parameter used by all models (required
                despite Optional type in config)
            *args, **kwargs: Additional arguments including model_name and checkpoint_io
                passed to BaseModel

        Note: input_length is a required parameter despite being marked
        as Optional in SingleStageModelConfig for schema flexibility. Hydra will enforce
        this requirement at instantiation time.

        Note: model_name and checkpoint_io are inherited from BaseModel and should be
            passed
        via **kwargs (typically handled automatically by Hydra instantiation).
        """
        super().__init__(*args, **kwargs)
        self._supports_multi_directional = supports_multi_directional
        self._input_length = input_length

    @property
    def supports_multi_directional(self) -> bool:
        """Get multi-directional support flag, handling DDP wrapping automatically."""
        return self._supports_multi_directional

    @property
    def input_length(self) -> int:
        """Get input length parameter, handling DDP wrapping automatically."""
        return self._input_length

    def extract_input(
        self, batch_dict: dict[str, torch.Tensor]
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor | None]
    ):
        """Extract and prepare input for models using NEW format batch structure.

        This method validates and extracts source waveforms from the batch_dict created
        by the collate function using the NEW format batch structure.

        Common functionality for all models that need source extraction.

        Args:
            batch_dict: NEW format batch dict from collate function with the following
                structure:

                **Required fields** (always present):
                - "x" (torch.Tensor): Normalized source waveforms [B, C, T]
                    - B: Batch size
                    - C: Number of channels in input_preprocessing['source']
                    - T: Time steps (waveform length)
                - "src_idxs" (torch.LongTensor): Source channel indices [B, S_max]
                    - Channel indices from input_preprocessing order, indicating which
                        channels are active sources for each sample
                    - S_max is auto-computed from directions in collate function
                - "src_mask" (torch.BoolTensor): Source channel mask [B, S_max]
                    - True for active sources, False for inactive
                - "tgt_idxs" (torch.LongTensor): Target channel indices [B]
                    - Indicates target channel for each sample

                **Optional fields** (when use_wcl=True):
                - "bp_raw" (torch.Tensor): Raw blood pressure values [B, 3] (SBP, DBP,
                    MAP)
                - "age_raw", "gender_raw", etc.: Individual demographic features [B, 1]
                    each


        Returns:
            torch.Tensor: Prepared input tensor [B, S_max, T] with inactive sources
                zeroed
                - B: Batch size
                - S_max: Maximum number of source channels (auto-computed from
                    directions)
                - T: Time steps (waveform length)

        Raises:
            ValueError: If required fields are missing from batch_dict

        Examples:
            >>> # NEW format batch structure
            >>> batch_dict = {
            ...     "x": torch.randn(32, 1, 1280),  # [B, C, T] - only source vital
            ...     "src_idxs": torch.tensor([[0], [1], ...]),  # [B, S_max] -
                input_preprocessing channel indices
            ...     "src_mask": torch.tensor([[True], [True], ...]),  # [B, S_max]
            ...     "tgt_idxs": torch.tensor([2, 2, ...]),  # [B]
            ... }
            >>> x = model.extract_input(batch_dict)
            >>> print(x.shape)  # [32, 1, 1280]

        Note:
            - This method is called by all model forward() methods
            - Normalization happens in collate function, transparent to models
            - Models can override this method for custom input extraction (e.g.,
                MDViSCo, WaveNetModel)
            - S_max is automatically computed from directions in the collate function
            - src_idxs contain channel indices derived from input_preprocessing order,
                not from VitalsDataset.
              The collate function builds these indices using
                  build_vital_channel_mapping().
        """
        # Validate required fields
        required_fields = ["x", "src_idxs", "src_mask"]
        missing_fields = [field for field in required_fields if field not in batch_dict]
        if missing_fields:
            raise ValueError(
                f"Missing required fields: {missing_fields}. "
                f"Expected unified batch structure."
            )

        # Extract waveform source (lazy normalization at collate time)
        # The "x" field contains waveforms normalized at batch time by collate.
        # This enables flexible normalization strategies without reprocessing datasets.
        waveform = batch_dict["x"]
        device = waveform.device
        b, c, t = waveform.shape

        # ---- indices/masks (types & device) ----
        src_idxs = batch_dict["src_idxs"].to(
            device=device, dtype=torch.long
        )  # [B, S_max]
        src_mask = batch_dict["src_mask"].to(
            device=device, dtype=torch.bool
        )  # [B, S_max]

        s_max = src_idxs.size(1)

        # ---- source extraction (fixed [B, S_max, T]) ----
        # Use src_idxs directly; indices from input_preprocessing
        idxs_3d = src_idxs.unsqueeze(-1).expand(b, s_max, t)  # [B, S_max, T]
        x_gathered = waveform.gather(dim=1, index=idxs_3d)  # [B, S_max, T]
        x_real = x_gathered * src_mask.unsqueeze(-1)  # [B, S_max, T]

        return x_real

    def _checkpoint_state_dict(self) -> dict[str, Any]:
        """Return checkpoint state dict for single-stage model.

        This method defines how the model's state should be saved in checkpoints.
        Single-stage models save their state under the standard 'model_state_dict' key.

        Returns:
            Dict[str, Any]: Dictionary mapping checkpoint keys to state dicts

        Example:
            {
                'model_state_dict': {
                    'layer1.weight': tensor(...),
                    'layer1.bias': tensor(...),
                    # ... all model parameters
                }
            }

        Note:
            This method is called by BaseTrainer.save_checkpoint() to get the
            model's checkpoint structure. It is the ONLY way to define how a
            single-stage model's state is saved in checkpoints.
        """
        return {CHECKPOINT_MODEL_STATE_KEY: self.state_dict()}
