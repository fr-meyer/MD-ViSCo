"""PatchTST Implementation.

This module implements PatchTST (Patch Time Series Transformer) for time series
forecasting and blood pressure estimation.

References:
- Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
  https://arxiv.org/abs/2211.14730
- Paper: "End-To-End Personalized Cuff-Less Blood Pressure Monitoring Using ECG and PPG
    Signals"
  https://ieeexplore.ieee.org/document/10445970/
- Original Implementation: https://github.com/yuqinie98/PatchTST
- Documentation: https://huggingface.co/docs/transformers/main/en/model_doc/patchtst
- License: Apache 2.0

Note: This implementation is adapted from the original codebase for use in
the MD-ViSCo framework.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

import torch
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from transformers import PatchTSTConfig
from transformers import PatchTSTForRegression

# Local imports
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig

logger = logging.getLogger(__name__)


@dataclass
class PatchTSTModelConfig(SingleStageModelConfig):
    """Configuration for PatchTST architecture parameters.

    Args:
        patch_len (int): Length of each patch
        stride (int): Stride between patches
        d_model (int): Dimension of model
        num_encoder_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        dropout (float): Attention dropout rate
        fc_dropout (float): Dropout rate for fully connected layers
        head_dropout (float): Dropout rate for prediction head
        in_channels (int): Number of input channels
        num_targets (int): Number of target values to predict
        use_cls_token (bool): Whether to use CLS token for regression
        use_demographics (bool): Whether to fuse demographic information
    """

    _target_: str = "src.model.patchtst.PatchTST"
    supports_multi_directional: bool = False  # PatchTST only supports single-direction
    model_name: str = "PatchTST"

    # Model Architecture configuration
    patch_len: int = 16  # Length of each patch
    stride: int = 8  # Stride between patches
    d_model: int = (
        128  # Dimension of model (paper: Nie et al. 2022, Section A.1.4 page 14)
    )
    num_encoder_layers: int = (
        3  # Number of encoder layers (paper: Section A.1.4 page 14)
    )
    num_heads: int = 16  # Number of attention heads (paper: Section A.1.4 page 14)
    dropout: float = 0.2  # Attention dropout rate (paper: Section A.1.4 page 14)
    fc_dropout: float = 0.1  # Dropout rate for fully connected layers
    head_dropout: float = 0.1  # Dropout rate for prediction head
    in_channels: int = 1  # Number of input channels
    num_targets: int = MISSING  # Number of target values to predict
    use_cls_token: bool = False  # Whether to use CLS token for regression
    use_demographics: bool = MISSING  # Whether to fuse demographic information
    # NOTE: use_demographics controls architecture (demographic inputs)
    # Collate function provides raw demographics via input_preprocessing.include:
    #   include: [age_raw, gender_raw, height_raw, weight_raw, bmi_raw]
    # Model handles stacking, broadcasting, and concatenation internally.

    # Demographic encoder configuration
    use_demographic_encoder: bool = (
        False  # Whether to use learned embeddings for demographics
    )
    demographic_embedding_dim: int = (
        32  # Dimension of encoded demographics when encoder is enabled
    )
    num_demographic_channels: int = (
        5  # Total number of demographic channels to allocate
    )
    demographic_channel_map: dict[str, int] = field(
        default_factory=lambda: {
            "age_raw": 0,
            "gender_raw": 1,
            "height_raw": 2,
            "weight_raw": 3,
            "bmi_raw": 4,
        }
    )  # Fixed mapping: demographic field name → channel index
    active_demographics: list[str] = field(
        default_factory=lambda: [
            "age_raw",
            "gender_raw",
            "height_raw",
            "weight_raw",
            "bmi_raw",
        ]
    )  # Which demographics to use from the data
    demographic_mask_policy: str = (
        "all"  # 'all' or 'any' - how to reduce demographic mask validity
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if not 0 <= self.fc_dropout <= 1:
            raise ValueError("fc_dropout must be between 0 and 1")
        if not 0 <= self.head_dropout <= 1:
            raise ValueError("head_dropout must be between 0 and 1")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.num_targets <= 0:
            raise ValueError("num_targets must be positive")
        if self.input_length is not None and self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if self.demographic_mask_policy not in ["all", "any"]:
            raise ValueError("demographic_mask_policy must be 'all' or 'any'")


class DemographicEncoder(nn.Module):
    """Encoder for demographic features with learned embeddings.

    This module processes raw demographic values and produces learned embeddings.
    Handles categorical (gender) and continuous (age, height, weight, BMI) features.

    Args:
        num_demographics: Number of input demographic features
        embedding_dim: Output dimension of encoded demographics
        categorical_indices: List of indices for categorical features (e.g., [1] for
            gender at index 1)
        categorical_cardinalities: List of cardinalities for categorical features (e.g.,
            [2] for binary gender)
    """

    def __init__(
        self,
        num_demographics: int = 5,
        embedding_dim: int = 32,
        categorical_indices: list[int] | None = None,
        categorical_cardinalities: list[int] | None = None,
    ) -> None:
        """Initialize DemographicEncoder with configuration parameters.

        Args:
            num_demographics: Number of input demographic features
            embedding_dim: Output dimension of encoded demographics
            categorical_indices: List of indices for categorical features (e.g., [1] for
                gender at index 1)
            categorical_cardinalities: List of cardinalities for categorical features
                (e.g., [2] for binary gender)
        """
        super().__init__()

        self.num_demographics = num_demographics
        self.embedding_dim = embedding_dim

        # Default: gender (index 1) is categorical with 2 values (0/1)
        self.categorical_indices = (
            categorical_indices if categorical_indices is not None else [1]
        )
        self.categorical_cardinalities = (
            categorical_cardinalities if categorical_cardinalities is not None else [2]
        )

        self.continuous_indices = [
            i for i in range(num_demographics) if i not in self.categorical_indices
        ]

        # Learned embeddings for categorical features
        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    cardinality + 1, embedding_dim // len(self.categorical_indices)
                )  # +1 for NaN/unknown
                for cardinality in self.categorical_cardinalities
            ]
        )

        # Linear projection for continuous features
        num_continuous = len(self.continuous_indices)
        if num_continuous > 0:
            self.continuous_projection = nn.Linear(num_continuous, embedding_dim // 2)

        # Learned default embedding for NaN values in continuous features
        self.nan_embedding = nn.Parameter(torch.randn(embedding_dim // 2))

        total_intermediate_dim = sum(
            [
                embedding_dim // len(self.categorical_indices)
                for _ in self.categorical_indices
            ]
        ) + (embedding_dim // 2 if num_continuous > 0 else 0)
        self.final_projection = nn.Linear(total_intermediate_dim, embedding_dim)

    def forward(self, demographics: torch.Tensor) -> torch.Tensor:
        """Encode demographic features.

        Args:
            demographics: [B, N] tensor with raw demographic values

        Returns:
            encoded: [B, D_demo] tensor with encoded demographics
        """
        b = demographics.shape[0]
        embeddings = []

        # Process categorical features
        for i, (cat_idx, embedding_layer) in enumerate(
            zip(self.categorical_indices, self.categorical_embeddings, strict=True)
        ):
            cat_values = demographics[:, cat_idx]

            # Handle NaN and out-of-range: map to unknown index (last index)
            is_nan = torch.isnan(cat_values)
            in_range = (cat_values >= 0) & (
                cat_values < self.categorical_cardinalities[i]
            )
            valid_range = in_range & ~is_nan

            cat_values_int = torch.where(
                valid_range,
                cat_values.long(),
                torch.tensor(
                    self.categorical_cardinalities[i], device=demographics.device
                ),
            )

            cat_embed = embedding_layer(cat_values_int)
            embeddings.append(cat_embed)

        # Process continuous features
        if len(self.continuous_indices) > 0:
            continuous_values = demographics[
                :, self.continuous_indices
            ]  # [B, num_continuous]

            nan_mask = torch.isnan(continuous_values)  # [B, num_continuous]

            # Replace NaN with 0 for projection
            continuous_values_clean = torch.where(
                nan_mask, torch.zeros_like(continuous_values), continuous_values
            )

            continuous_embed = self.continuous_projection(
                continuous_values_clean
            )  # [B, embed_dim/2]

            # For samples with any NaN, blend in the learned NaN embedding
            has_nan = nan_mask.any(dim=1, keepdim=True)  # [B, 1]
            nan_embed_broadcast = self.nan_embedding.unsqueeze(0).expand(
                b, -1
            )  # [B, embed_dim/2]

            continuous_embed = torch.where(
                has_nan,
                (continuous_embed + nan_embed_broadcast)
                / 2,  # Blend with NaN embedding
                continuous_embed,
            )

            embeddings.append(continuous_embed)

        combined = torch.cat(embeddings, dim=1)  # [B, total_intermediate_dim]
        encoded = self.final_projection(combined)  # [B, embedding_dim]

        return encoded


class PatchTST(SingleStageModel):
    """Unified PatchTST model for time series regression.

    This implementation is based on the paper "A Time Series is Worth 64 Words:
        Long-term Forecasting with Transformers"
    and uses Hugging Face's transformers library. It supports both approximation and
        refinement use cases
    through configuration parameters.

    Output contract is task-aware:
        - Approximation mode (`num_targets > 2`): returns {'predictions': [B,
            num_targets]}
        - Refinement mode (`num_targets == 2`): returns {'predictions': [B, 2 SBP/DBP]}

    Args:
        patch_len (int): Length of each patch extracted from the input sequence
        stride (int): Step size between consecutive patches
        d_model (int): Dimensionality of the patch embeddings
        num_encoder_layers (int): Number of transformer encoder layers
        num_heads (int): Number of self-attention heads
        dropout (float): Dropout rate applied in attention layers
        fc_dropout (float): Dropout rate for the feedforward layers
        head_dropout (float): Dropout rate used in the output prediction head
        in_channels (int): Number of input waveform channels
        num_targets (int): Number of target values to predict
        use_cls_token (bool): Whether to use CLS token for regression
        use_demographics (bool): Whether to fuse demographic information with waveform
            data
    """

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 16,
        dropout: float = 0.2,
        fc_dropout: float = 0.1,
        head_dropout: float = 0.1,
        in_channels: int = 1,
        num_targets: int = 1,
        use_cls_token: bool = False,
        use_demographics: bool = False,
        use_demographic_encoder: bool = False,
        demographic_embedding_dim: int = 32,
        num_demographic_channels: int = 5,
        demographic_channel_map: dict[str, int] | None = None,
        active_demographics: list[str] | None = None,
        demographic_mask_policy: str = "all",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize PatchTST with configuration parameters.

        Args:
            patch_len: Length of each patch extracted from the input sequence
            stride: Step size between consecutive patches
            d_model: Dimensionality of the patch embeddings
            num_encoder_layers: Number of transformer encoder layers
            num_heads: Number of self-attention heads
            dropout: Dropout rate applied in attention layers
            fc_dropout: Dropout rate for the feedforward layers
            head_dropout: Dropout rate used in the output prediction head
            in_channels: Number of input waveform channels
            num_targets: Number of target values to predict
            use_cls_token: Whether to use CLS token for regression
            use_demographics: Whether to fuse demographic information with waveform data
            use_demographic_encoder: Whether to use learned embeddings for demographics
            demographic_embedding_dim: Dimension of encoded demographics when encoder is
                enabled
            num_demographic_channels: Total number of demographic channels to allocate
            demographic_channel_map: Fixed mapping: demographic field name → channel
                index
            active_demographics: Which demographics to use from the data
            demographic_mask_policy: 'all' or 'any' - how to reduce demographic mask
                validity
        """
        super().__init__(*args, **kwargs)

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.in_channels = in_channels
        self.num_targets = num_targets
        self.use_cls_token = use_cls_token
        self.use_demographics = use_demographics

        # Demographic encoder parameters
        self.use_demographic_encoder = use_demographic_encoder
        self.demographic_embedding_dim = demographic_embedding_dim
        self.num_demographic_channels = num_demographic_channels
        self.demographic_channel_map = demographic_channel_map or {
            "age_raw": 0,
            "gender_raw": 1,
            "height_raw": 2,
            "weight_raw": 3,
            "bmi_raw": 4,
        }
        self.active_demographics = active_demographics or [
            "age_raw",
            "gender_raw",
            "height_raw",
            "weight_raw",
            "bmi_raw",
        ]
        self.demographic_mask_policy = demographic_mask_policy

        # Validate demographic channel configuration
        expected_channels = len(self.active_demographics)
        if self.num_demographic_channels != expected_channels:
            msg = (
                f"num_demographic_channels ({self.num_demographic_channels}) "
                f"does not match active_demographics length ({expected_channels}). "
                f"Auto-correcting to {expected_channels}."
            )
            logger.warning(msg)
            self.num_demographic_channels = expected_channels

        # Parameter validation
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if not 0 <= fc_dropout <= 1:
            raise ValueError("fc_dropout must be between 0 and 1")
        if not 0 <= head_dropout <= 1:
            raise ValueError("head_dropout must be between 0 and 1")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_targets <= 0:
            raise ValueError("num_targets must be positive")
        if self.input_length is not None and self.input_length <= 0:
            raise ValueError("input_length must be positive")

        if self.use_demographics and self.demographic_channel_map:
            max_channel_idx = max(self.demographic_channel_map.values())
            if max_channel_idx >= self.num_demographic_channels:
                raise ValueError(
                    f"demographic_channel_map contains index {max_channel_idx} but "
                    f"num_demographic_channels is {self.num_demographic_channels}. "
                    f"All indices must be in range [0, "
                    f"{self.num_demographic_channels - 1}]"
                )

        self.demographic_encoder = None
        if self.use_demographics and self.use_demographic_encoder:
            self.demographic_encoder = DemographicEncoder(
                num_demographics=self.num_demographic_channels,
                embedding_dim=self.demographic_embedding_dim,
                categorical_indices=[1],  # gender at index 1
                categorical_cardinalities=[2],  # binary gender (0/1)
            )

        # PatchTST expects [B, T, C]; C = sum of source vitals' channels
        # This depends on whether we use demographics and whether we use the encoder
        if self.use_demographics:
            if self.use_demographic_encoder:
                # Waveform channels + encoded demographic dimension
                num_input_channels = in_channels + self.demographic_embedding_dim
            else:
                # Waveform channels + raw demographic channels
                num_input_channels = in_channels + self.num_demographic_channels
        else:
            # Only waveform channels
            num_input_channels = in_channels

        self.patchtst_config = PatchTSTConfig(
            num_input_channels=num_input_channels,
            num_targets=num_targets,
            context_length=self.input_length,
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            num_hidden_layers=num_encoder_layers,
            num_attention_heads=num_heads,
            attention_dropout=dropout,
            ff_dropout=fc_dropout,
            head_dropout=head_dropout,
            # Additional configuration options from the paper
            share_embedding=True,  # Share embedding across channels
            channel_attention=False,  # No channel attention
            norm_type="batchnorm",  # Use batch normalization
            activation_function="gelu",  # GELU activation
            pre_norm=True,  # Apply normalization before attention
            positional_encoding_type="sincos",  # Sinusoidal positional encoding
            use_cls_token=use_cls_token,
        )

        self.model = PatchTSTForRegression(self.patchtst_config)

        # Determine task mode from configuration
        # Stage-1 approximation models predict sequences (num_targets > 2)
        # Stage-2 refinement models predict SBP/DBP scalars (num_targets == 2)
        if self.num_targets == 2:
            self.task_mode: Literal["refinement", "approximation"] = "refinement"
        else:
            self.task_mode = "approximation"

    def extract_input(
        self, batch_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract and prepare input for PatchTST model from unified batch structure.

        This method handles input processing for both approximation and refinement
            modes:
        - Approximation mode: Channel selection using src_idxs and src_mask
        - Refinement mode: Waveform + demographics fusion with channel-wise
            concatenation

        Demographics Handling (Refinement Mode - Channel-wise Integration):
            When use_demographics=True, demographics are concatenated as additional
                channels.

            Required fields in batch_dict (raw demographic fields):
            - age_raw, gender_raw, height_raw, weight_raw, bmi_raw: [B, 1] each

            The model handles:
            1. Extract raw demographic fields from batch_dict
            2. Stack individual fields [B, 1] + [B, 1] + ... → [B, N]
            3. Broadcast to match waveform length [B, N] → [B, N, T]
            4. Concatenate with waveforms [B, C, T] + [B, N, T] → [B, C+N, T]

            where N = num_demographic_channels (typically 5 for age, gender, height,
                weight, BMI)

            If use_demographic_encoder=True, raw demographics are passed through a
                learned encoder
            before broadcasting, resulting in shape [B, C+D_demo, T] where D_demo is the
                embedding dimension.

            Configure demographics via input_preprocessing.include:
            include: [age_raw, gender_raw, height_raw, weight_raw, bmi_raw]

        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs.
                        When use_demographics=True, must contain raw demographic fields:
                        - age_raw, gender_raw, height_raw, weight_raw, bmi_raw: [B, 1]
                            each

        Returns:
            x: Tensor [B, T, C+N] ready for PatchTST (or [B, T, C+D_demo] with encoder)
            past_observed_mask: BoolTensor [B, T, C+N] or None for missing value
                handling
        """
        if self.use_demographics:
            # Refinement mode: Waveform + demographics fusion (channel-wise)

            # Extract waveform using parent method
            x_waveform = super().extract_input(batch_dict)
            if isinstance(x_waveform, dict):
                x_waveform = x_waveform["x"]
            elif isinstance(x_waveform, tuple):
                x_waveform = x_waveform[0]
            if not isinstance(x_waveform, torch.Tensor):
                raise TypeError("PatchTST extract_input expected Tensor from parent")
            # [B, C, T] where C = num waveform channels
            b, c, t = x_waveform.shape

            # Model handles demographic processing
            demographics = self._prepare_demographics(batch_dict, target_length=t)

            # Verify shape
            if demographics.size(1) != self.num_demographic_channels:
                raise ValueError(
                    f"Expected {self.num_demographic_channels} demographic "
                    f"channels but got {demographics.size(1)}. Check "
                    f"num_demographic_channels configuration."
                )

            # Ensure matching device and dtype
            demographics = demographics.to(
                device=x_waveform.device, dtype=x_waveform.dtype
            )

            # Derive demographics validity mask BEFORE any NaN replacement.
            # Primary (production) path: infer from raw values using torch.isfinite().
            # Secondary (test) path: explicit mask via batch_dict['demographics_mask'].
            if (
                "demographics_mask" in batch_dict
                and batch_dict["demographics_mask"] is not None
            ):
                explicit_mask = batch_dict["demographics_mask"]
                # Move to device; require strict boolean dtype (no implicit casting)
                explicit_mask = explicit_mask.to(device=x_waveform.device)
                if explicit_mask.dtype != torch.bool:
                    raise ValueError(
                        "demographics_mask must be a boolean tensor of shape "
                        "[B,N], [B,N,1], or [B,N,T]"
                    )
                # Accept [B, N], [B, N, 1], or [B, N, T] and expand to [B, N, T]
                if explicit_mask.dim() == 2:
                    # [B, N] -> [B, N, T]
                    demo_mask_raw = explicit_mask.unsqueeze(2).expand(-1, -1, t)
                elif explicit_mask.dim() == 3 and explicit_mask.size(2) == 1:
                    # [B, N, 1] -> [B, N, T]
                    demo_mask_raw = explicit_mask.expand(-1, -1, t)
                elif explicit_mask.dim() == 3 and explicit_mask.size(2) == t:
                    # [B, N, T]
                    demo_mask_raw = explicit_mask
                else:
                    raise ValueError(
                        f"demographics_mask has invalid shape "
                        f"{tuple(explicit_mask.shape)}; "
                        f"expected [B,N], [B,N,1], or [B,N,T] with T={t}"
                    )
                if demo_mask_raw.shape != demographics.shape:
                    raise ValueError(
                        f"Shape mismatch: demographics shape "
                        f"{tuple(demographics.shape)} vs "
                        f"demographics_mask shape {tuple(demo_mask_raw.shape)}"
                    )
            else:
                # Derive mask from data validity before any NaN replacement
                demo_mask_raw = torch.isfinite(demographics)

            # If using demographic encoder, process demographics
            if self.use_demographic_encoder and self.demographic_encoder is not None:
                # Take first timestep (demographics are constant across time)
                demographics_raw = demographics[:, :, 0]  # [B, N]
                demo_mask_raw_t0 = demo_mask_raw[:, :, 0]  # [B, N]

                # Pass through encoder
                demographics_encoded = self.demographic_encoder(
                    demographics_raw
                )  # [B, D_demo]

                # Broadcast encoded demographics across time
                demographics_to_concat = demographics_encoded.unsqueeze(2).expand(
                    -1, -1, t
                )  # [B, D_demo, T]

                # Reduce raw demographic mask according to policy, then broadcast across
                # embedding dim and time.
                if self.demographic_mask_policy == "all":
                    # Conservative: require all demographics valid per sample
                    demo_mask_reduced = demo_mask_raw_t0.all(
                        dim=1, keepdim=True
                    )  # [B, 1]
                elif self.demographic_mask_policy == "any":
                    # Permissive: any demographic valid per sample
                    demo_mask_reduced = demo_mask_raw_t0.any(
                        dim=1, keepdim=True
                    )  # [B, 1]
                else:
                    raise ValueError("demographic_mask_policy must be 'all' or 'any'")

                # Broadcast [B,1] -> [B,D_demo] -> [B,D_demo,T]
                demo_mask_to_concat = demo_mask_reduced.expand(
                    -1, self.demographic_embedding_dim
                )
                demo_mask_to_concat = demo_mask_to_concat.unsqueeze(2).expand(-1, -1, t)
            else:
                demographics_to_concat = demographics  # [B, N, T]
                demo_mask_to_concat = demo_mask_raw

            # Replace NaN with 0 in demographics AFTER mask is finalized
            demographics_to_concat = torch.nan_to_num(demographics_to_concat, nan=0.0)

            waveform_mask = torch.ones(
                (b, c, t), dtype=torch.bool, device=x_waveform.device
            )

            # Concatenate along channel dimension
            x = torch.cat(
                [x_waveform, demographics_to_concat], dim=1
            )  # [B, C+N, T] or [B, C+D_demo, T]
            mask = torch.cat(
                [waveform_mask, demo_mask_to_concat], dim=1
            )  # [B, C+N, T] or [B, C+D_demo, T]

            # Transpose to PatchTST format [B, T, C+N]
            x = x.transpose(1, 2)  # [B, T, C+N]
            mask = mask.transpose(1, 2)  # [B, T, C+N]

            return x, mask
        else:
            # Approximation mode: Use unified batch structure only
            # Parent class validates required fields and extracts input
            x = super().extract_input(batch_dict)
            if isinstance(x, dict):
                x = x["x"]
            elif isinstance(x, tuple):
                x = x[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError("PatchTST extract_input expected Tensor from parent")

            # PatchTST expects [B, T, C]; transpose if currently [B, C, T]
            if x.dim() == 3:
                x = x.transpose(1, 2)  # [B, T, S_max]

            return x, None  # No mask needed for PatchTST

    def _prepare_demographics(
        self, batch_dict: dict[str, torch.Tensor], target_length: int
    ) -> torch.Tensor:
        """Extract and prepare demographics from batch_dict.

        Args:
            batch_dict: Batch dictionary with raw demographic fields
            target_length: Target time dimension for broadcasting

        Returns:
            Broadcasted demographics tensor [B, N, T]
        """
        demo_list = []
        for field_name in self.active_demographics:
            if field_name in batch_dict:
                demo_list.append(batch_dict[field_name])  # [B, 1]

        if not demo_list:
            raise ValueError(
                f"Demographics requested but not found in batch. "
                f"Add to config: input_preprocessing.include: "
                f"{self.active_demographics}"
            )

        # Stack: [B, 1] + [B, 1] + ... → [B, N]
        demographics = torch.cat(demo_list, dim=1)

        # Broadcast: [B, N] → [B, N, T]
        demographics = demographics.unsqueeze(2).expand(-1, -1, target_length)

        return demographics

    def forward(
        self,
        batch_dict: dict[str, torch.Tensor],
        target_values: torch.Tensor | None = None,
        past_observed_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass of the unified PatchTST model.

        Args:
            batch_dict: Dictionary following the unified batch structure with at least
                `x`, `src_idxs`, and `src_mask`. When demographic fusion is enabled,
                the batch must also provide the configured demographic fields.
            target_values: Ground truth values [B, num_targets]
            past_observed_mask: Optional externally-supplied mask [B, L, C]

        Returns:
            Dict[str, torch.Tensor]: Output format depends on task mode:
                - Approximation: canonical schema with regression outputs
                - Refinement: canonical schema with SBP/DBP concatenated
        """
        generated_mask = None

        # Handle dict inputs
        if isinstance(batch_dict, dict):
            x, generated_mask = self.extract_input(batch_dict)
        else:
            x = batch_dict
            if x.size(1) != self.in_channels:
                raise ValueError(
                    f"Expected input shape [B, {self.in_channels}, T] but got "
                    f"{tuple(x.shape)}. Ensure input is in [B, C, T] format where C "
                    "matches model's in_channels."
                )
            x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]

        past_observed_mask = (
            past_observed_mask if past_observed_mask is not None else generated_mask
        )

        output = self.model(
            past_values=x,
            target_values=target_values,
            past_observed_mask=past_observed_mask,
        )
        regression_outputs = output.regression_outputs
        if self.task_mode == "refinement":
            if regression_outputs.shape[-1] < 2:
                raise ValueError(
                    "PatchTST refinement mode expects at least two targets for SBP/DBP "
                    f"prediction. Got shape {tuple(regression_outputs.shape)}."
                )
            sbp = regression_outputs[:, 0:1]
            dbp = regression_outputs[:, 1:2]
            return {
                "predictions": torch.cat([sbp, dbp], dim=1),
                "extras": {},
            }
        else:
            return {
                "predictions": regression_outputs,
                "extras": {},
            }


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_patchtst", group="model", node=PatchTSTModelConfig)
