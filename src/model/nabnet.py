"""NABNet Implementation.

This module implements NABNet (Nested Attention-guided BiConvLSTM Network)
for vital sign waveform conversion.

References:
- Paper: "NABNet: A Nested Attention-guided BiConvLSTM network for robust prediction
  of Blood Pressure from reconstructed ABP using PPG and ECG"
  https://linkinghub.elsevier.com/retrieve/pii/S1746809422007017
- Original Implementation: https://github.com/Sakib1263/NABNet
- License: MIT

Note: This implementation is adapted from the original codebase for use in
the MD-ViSCo framework.
"""

# Standard library imports
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Local imports
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig

# ============================================================================
# MLP REGRESSOR CONFIGURATION
# ============================================================================


@dataclass
class MLPRegressorConfig(SingleStageModelConfig):
    """Configuration for single MLP regressor head."""

    _target_: str = "src.model.nabnet.MLPRegressor"
    name: str = MISSING
    hidden_size: int = 100
    activation: str = "relu"
    alpha: float = 0.0001

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.input_length is None or self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.activation not in ["relu", "tanh"]:
            raise ValueError("activation must be 'relu' or 'tanh'")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")


@dataclass
class MultiMLPRegressorConfig(SingleStageModelConfig):
    """Configuration for multi-head MLP regressor."""

    _target_: str = "src.model.nabnet.MultiMLPRegressor"
    mlp_heads: list[MLPRegressorConfig] = MISSING

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.mlp_heads:
            raise ValueError("mlp_heads cannot be empty")

        names = [head.name for head in self.mlp_heads]
        if len(names) != len(set(names)):
            raise ValueError("All MLP head names must be unique")


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


@dataclass
class NABNetModelConfig(SingleStageModelConfig):
    """Configuration for NABNet architecture parameters.

    Inherits single-stage specific fields from SingleStageModelConfig.
    """

    _target_: str = "src.model.nabnet.NABNet"
    model_name: str = "NabNet"

    # Model Architecture configuration
    model_depth: int = 5  # Number of Level in the CNN Model
    # Width of initial layer (paper Section 3.4: best at 128)
    model_width: int = 128
    D_S: int = 1  # Deep Supervision enabled
    A_E: int = 0  # AutoEncoder Mode disabled
    A_G: int = 1  # Guided Attention enabled
    LSTM: int = 1  # LSTM enabled
    feature_number: int = 1024  # For AutoEncoder mode
    is_transconv: bool = True
    kernel_size: int = 3
    in_channels: int = 1
    out_channels: int = 1

    # Additional model parameters
    problem_type: str = "Regression"
    attention_type: str = "lstm"  # 'standard' or 'lstm' (paper-aligned default: lstm)

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.model_depth <= 0:
            raise ValueError("model_depth must be positive")
        if self.model_width <= 0:
            raise ValueError("model_width must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.feature_number <= 0:
            raise ValueError("feature_number must be positive")
        if self.attention_type not in ["standard", "lstm"]:
            raise ValueError("attention_type must be 'standard' or 'lstm'")
        if self.input_length is None or self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive")


@dataclass
class ShallowUNetModelConfig(SingleStageModelConfig):
    """Configuration for ShallowUNet refinement model architecture parameters.

    Inherits single-stage specific fields from SingleStageModelConfig.
    """

    _target_: str = "src.model.nabnet.ShallowUNet"
    model_name: str = "ShallowUNet"

    # Model Architecture configuration
    model_depth: int = 1  # Depth of the ShallowUNet (number of downsampling layers)
    # Width of the input layer (paper: Mahmud et al. 2022, Section IV.A: 128)
    model_width: int = 128
    feature_number: int = 1024  # Feature vector size for autoencoder bottleneck
    in_channels: int = 1  # Number of input channels (typically 1 for PPG, ECG, etc.)
    output_nums: int = 1  # Number of output channels (typically 1 for ABP)
    kernel_size: int = 3  # Kernel size for convolutional layers
    problem_type: str = (
        "Regression"  # Type of problem (e.g., 'Regression' for waveform output)
    )

    # Architecture feature flags
    deep_supervision: int = 0  # Whether to use deep supervision (0: no, 1: yes)
    autoencoder: int = 1  # Whether autoencoder architecture is enabled (1: yes, 0: no)
    guided_attention: int = 0  # Whether attention modules are included (1: yes, 0: no)
    use_transconv: bool = (
        True  # Whether to use transposed convolutions (True) or upsampling (False)
    )
    use_lstm: int = (
        0  # Whether to include LSTM layers in the architecture (1: yes, 0: no)
    )

    # Additional parameters
    alpha: float = 1.0  # General alpha parameter for UNet
    feature_extraction_only: bool = (
        False  # Whether to only extract features (for cascade stage1)
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.model_depth <= 0:
            raise ValueError("model_depth must be positive")
        if self.model_width <= 0:
            raise ValueError("model_width must be positive")
        if self.feature_number <= 0:
            raise ValueError("feature_number must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.problem_type not in ["Regression", "Classification"]:
            raise ValueError("problem_type must be 'Regression' or 'Classification'")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.input_length is None or self.input_length <= 0:
            raise ValueError("input_length must be positive")


# ============================================================================
# COMMON UTILITY FUNCTIONS AND BLOCKS
# ============================================================================


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: str = "same",
) -> nn.Sequential:
    """Unified 1D Convolutional Block for both NABNet and ShallowUNet.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Padding mode; "same" uses kernel_size // 2.

    Returns:
        nn.Sequential: Conv1d -> BatchNorm1d -> ReLU.
    """
    padding_val = kernel_size // 2 if padding == "same" else padding
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_val),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


def trans_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Unified 1D Transposed Convolutional Block for both NABNet and ShallowUNet.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        nn.Sequential: ConvTranspose1d(2,2,0) -> BatchNorm1d -> ReLU.
    """
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        ),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


class Lambda(nn.Module):
    """Lambda layer for custom functions."""

    def __init__(self, func: Callable[..., Any]) -> None:
        """Initialize Lambda layer with a custom function.

        Args:
            func: Custom function to apply in forward pass
        """
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply custom function to input.

        Args:
            x: Input tensor

        Returns:
            Output tensor after applying custom function
        """
        return self.func(x)


def _extract_2d_features(
    batch_or_tensor: dict[str, torch.Tensor] | torch.Tensor,
    class_name: str = "Class",
) -> torch.Tensor:
    """Extract 2D features [B, F] from either dict or tensor input.

    Shared helper for MLPRegressor and MultiMLPRegressor extraction logic.
    Accepts:
        - Dict with key 'x' containing features tensor [B, F]
        - Direct tensor [B, F] (backward compatibility)

    Raises:
        ValueError: If dict missing 'x' or tensor is not 2D
        TypeError: If extracted value is not a torch.Tensor
    """
    if isinstance(batch_or_tensor, dict):
        if "x" not in batch_or_tensor:
            raise ValueError(
                f"{class_name}.extract_input expected key 'x' in batch dict"
            )
        features = batch_or_tensor["x"]
    else:
        features = batch_or_tensor

    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Expected features to be torch.Tensor, got {type(features)!r}")
    if features.dim() != 2:
        raise ValueError(
            f"{class_name} requires 2D features [B, F], got shape "
            f"{tuple(features.shape)}"
        )
    return features


class FeatureExtractionBlock(nn.Module):
    """Unified Feature Extraction Block for both NABNet and ShallowUNet.

    This class replaces the previous Feature_Extraction_Block and
        ShallowFeatureExtractionBlock
    with a single implementation that can handle both use cases:

    - NABNet: Uses reshape_dims parameter for explicit reshaping (batch, channels,
        length)
    - ShallowUNet: Uses model_width parameter for automatic reshaping (batch,
        model_width, -1)

    Args:
        input_size (int): Total number of input features (channels * length)
        feature_number (int): Number of features to extract in the bottleneck
        reshape_dims (tuple, optional): Explicit reshape dimensions for NABNet
        model_width (int, optional): Model width for ShallowUNet style reshaping
    """

    def __init__(
        self,
        input_size: int,
        feature_number: int,
        reshape_dims: tuple[int, ...] | None = None,
        model_width: int | None = None,
    ) -> None:
        """Initialize FeatureExtractionBlock with configuration parameters.

        Args:
            input_size: Total number of input features (channels * length)
            feature_number: Number of features to extract in the bottleneck
            reshape_dims: Explicit reshape dimensions for NABNet
            model_width: Model width for ShallowUNet style reshaping
        """
        super().__init__()
        self.input_size = input_size
        self.feature_number = feature_number
        self.reshape_dims = reshape_dims
        self.model_width = model_width

        # Core network components
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, feature_number)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feature_number, input_size)

    def forward(self, x: torch.Tensor, feature_only: bool = False) -> torch.Tensor:
        """Forward pass for feature extraction.

        Args:
            x (torch.Tensor): Input tensor
            feature_only (bool): If True, return only the extracted features
                                If False, return reconstructed tensor

        Returns:
            torch.Tensor: Either features or reconstructed tensor based on feature_only
        """
        original_shape = x.shape
        batch_size = x.size(0)

        x = self.flatten(x)

        if x.size(1) != self.input_size:
            raise ValueError(
                f"Input size mismatch. Got tensor with {x.size(1)} features, "
                f"expected {self.input_size} features. Input shape was {original_shape}"
            )

        x = self.relu(self.fc1(x))

        if feature_only:
            return x

        x = self.fc2(x)

        if self.reshape_dims is not None:
            return x.view(batch_size, *self.reshape_dims)
        elif self.model_width is not None:
            return x.view(batch_size, self.model_width, -1)
        else:
            return x.view(original_shape)


# ============================================================================
# APPROXIMATION MODEL: NABNet (Neural Approximation Blood pressure Network)
# ============================================================================


class AttentionBlock(nn.Module):
    """Attention Block for NABNet."""

    def __init__(self, channels: int, multiplier: int) -> None:
        """Initialize AttentionBlock with configuration parameters.

        Args:
            channels: Number of input/output channels
            multiplier: Multiplier for intermediate channel dimensions
        """
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels * multiplier, 1, stride=1)
        self.bn1 = nn.BatchNorm1d(channels * multiplier)
        self.conv2 = nn.Conv1d(channels, channels * multiplier, 1, stride=1)
        self.bn2 = nn.BatchNorm1d(channels * multiplier)
        self.conv3 = nn.Conv1d(channels * multiplier, 1, 1)
        self.bn3 = nn.BatchNorm1d(1)

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to skip connection using gating signal.

        Args:
            skip: Skip connection tensor
            gating: Gating signal tensor

        Returns:
            Attention-weighted skip connection tensor
        """
        # Ensure same spatial dimensions
        if skip.size(2) != gating.size(2):
            gating = F.interpolate(
                gating, size=skip.size(2), mode="linear", align_corners=False
            )

        x1 = self.bn1(self.conv1(skip))
        x2 = self.bn2(self.conv2(gating))
        x = F.relu(x1 + x2)
        x = self.bn3(self.conv3(x))
        x = torch.sigmoid(x)
        return skip * x


class AttentionLSTMBlock(nn.Module):
    """Attention LSTM Block for NABNet."""

    def __init__(self, channels: int, lstm_multiplier: float) -> None:
        """Initialize AttentionLSTMBlock with configuration parameters.

        Args:
            channels: Number of input/output channels
            lstm_multiplier: Multiplier for LSTM hidden dimensions
        """
        super().__init__()
        self.channels = channels
        self.lstm_hidden = int(channels * lstm_multiplier)

        self.lstm_skip = nn.LSTM(
            channels, self.lstm_hidden, bidirectional=True, batch_first=True
        )
        self.lstm_up = nn.LSTM(
            channels, self.lstm_hidden, bidirectional=True, batch_first=True
        )

        # Adjust attention to match LSTM output dimensions
        self.attention = nn.MultiheadAttention(self.lstm_hidden * 2, 1)

        # Adjust output projection to match input channels
        self.out_proj = nn.Sequential(
            nn.Linear(self.lstm_hidden * 4, channels), nn.ReLU()
        )

    def forward(self, skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Apply LSTM-based attention mechanism to skip connection using upsampled
            features.

        Args:
            skip: Skip connection tensor
            up: Upsampled features tensor

        Returns:
            Attention-weighted skip connection tensor
        """
        # Transpose for LSTM (batch, channels, length) -> (batch, length, channels)
        skip = skip.transpose(1, 2)
        up = up.transpose(1, 2)

        skip_feat, _ = self.lstm_skip(skip)  # Output: [batch, seq_len, lstm_hidden*2]
        up_feat, _ = self.lstm_up(up)  # Output: [batch, seq_len, lstm_hidden*2]

        attn_out, _ = self.attention(skip_feat, up_feat, up_feat)

        combined = torch.cat(
            [attn_out, skip_feat], dim=-1
        )  # [batch, seq_len, lstm_hidden*4]

        # Project back to original channel dimension
        output = self.out_proj(combined)  # [batch, seq_len, channels]

        # Transpose back to channel-first format
        output = output.transpose(1, 2)  # [batch, channels, seq_len]

        return output


class NABNet(SingleStageModel):
    """NABNet: Neural Approximation Blood pressure Network (Approximation Model)."""

    def __init__(
        self,
        model_depth: int = 5,
        in_channels: int = 1,
        model_width: int = 128,
        kernel_size: int = 3,
        problem_type: str = "Regression",
        out_channels: int = 1,
        d_s: int = 1,
        a_e: int = 0,
        a_g: int = 1,
        lstm: int = 1,
        feature_number: int = 1024,
        is_transconv: bool = True,
        attention_type: str = "lstm",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize NABNet with configuration parameters.

        Args:
            model_depth: Number of levels in the CNN model
            in_channels: Number of input channels
            model_width: Width of the initial layer
            kernel_size: Size of convolution kernels
            problem_type: Type of problem (Regression or Classification)
            out_channels: Number of output channels
            d_s: Deep supervision enabled flag
            a_e: AutoEncoder mode flag
            a_g: Guided attention enabled flag
            lstm: LSTM enabled flag
            feature_number: Feature vector size for autoencoder bottleneck
            is_transconv: Whether to use transposed convolutions
            attention_type: Type of attention ('standard' or 'lstm')
        """
        # NABNet only supports single-directional training
        super().__init__(*args, **kwargs)

        # Parameter validation
        if model_depth <= 0:
            raise ValueError("model_depth must be positive")
        if model_width <= 0:
            raise ValueError("model_width must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if feature_number <= 0:
            raise ValueError("feature_number must be positive")
        if attention_type not in ["standard", "lstm"]:
            raise ValueError("attention_type must be 'standard' or 'lstm'")
        if problem_type not in ["Regression", "Classification"]:
            raise ValueError("problem_type must be 'Regression' or 'Classification'")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        self.model_depth = model_depth
        self.model_width = model_width
        self.D_S = d_s
        self.A_E = a_e
        self.A_G = a_g
        self.LSTM = lstm
        self.feature_number = feature_number
        self.is_transconv = is_transconv
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.problem_type = problem_type
        self.attention_type = attention_type

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        in_channels = self.in_channels

        for i in range(self.model_depth):
            out_channels = self.model_width * (2**i)
            self.encoder_blocks.append(
                nn.Sequential(
                    conv_block(in_channels, out_channels, self.kernel_size),
                    conv_block(out_channels, out_channels, self.kernel_size),
                )
            )
            in_channels = out_channels

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bridge
        bridge_channels = self.model_width * (2**self.model_depth)
        self.bridge = nn.Sequential(
            conv_block(
                self.model_width * (2 ** (self.model_depth - 1)),
                bridge_channels,
                self.kernel_size,
            ),
            conv_block(bridge_channels, bridge_channels, self.kernel_size),
        )

        # Decoder path with transposed convolution
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.model_depth):
            in_channels = self.model_width * (2 ** (self.model_depth - i))
            out_channels = self.model_width * (2 ** (self.model_depth - i - 1))
            self.decoder_blocks.append(trans_conv_block(in_channels, out_channels))

        # Attention blocks if enabled - now supports different attention types
        if self.A_G:
            self.attention_blocks = nn.ModuleList()
            # Reverse order matches decoder path for skip connections
            for i in range(self.model_depth - 1, -1, -1):
                channels = self.model_width * (2**i)
                if self.attention_type == "lstm":
                    self.attention_blocks.append(AttentionLSTMBlock(channels, 1))
                else:  # 'standard' attention
                    self.attention_blocks.append(AttentionBlock(channels, 2))
        else:
            self.attention_blocks = None

        # Final convolution
        self.final_conv = nn.Conv1d(self.model_width, self.out_channels, 1)
        self.final_activation = (
            nn.Identity() if self.problem_type == "Regression" else nn.Sigmoid()
        )

        self.feature_extraction: FeatureExtractionBlock | None = None
        if self.A_E:
            bottleneck_channels = self.model_width * (2 ** (self.model_depth - 1))
            bottleneck_length = self.input_length // (2**self.model_depth)
            input_size = bottleneck_channels * bottleneck_length

            self.feature_extraction = FeatureExtractionBlock(
                input_size=input_size,
                feature_number=self.feature_number,
                reshape_dims=(bottleneck_channels, bottleneck_length),
            )

            def init_weights(m):
                """Kaiming init for Linear layers; zero bias."""
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            self.feature_extraction.apply(init_weights)
        else:
            self.feature_extraction = None

        self.decoder_conv_blocks = nn.ModuleList()
        for i in range(self.model_depth):
            curr_width = self.model_width * (2 ** (self.model_depth - i - 1))
            in_channels = curr_width * 2  # *2 for concatenation

            self.decoder_conv_blocks.append(
                nn.Sequential(
                    conv_block(in_channels, curr_width, self.kernel_size),
                    conv_block(curr_width, curr_width, self.kernel_size),
                )
            )

        if self.D_S:
            self.ds_convs = nn.ModuleList(
                [
                    nn.Conv1d(self.model_width * (2**i), self.out_channels, 1)
                    for i in range(self.model_depth)
                ][::-1]
            )  # Reverse order to match decoder path
        else:
            self.ds_convs = None

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and prepare input for NABNet from unified batch structure.

        This method handles the unified input processing for NABNet, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, in_channels, L] for NABNet

        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs

        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, in_channels,
                signal_length)
        """
        x = super().extract_input(batch_dict)
        if isinstance(x, dict):
            x = x["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("NABNet extract_input expected Tensor from parent")

        if x.dim() != 3:
            raise ValueError(
                f"NABNet expects input tensor of shape [B, C, L]; received {x.shape}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"NABNet expects input tensor with {self.in_channels} channels, "
                f"but received {x.size(1)} channels. "
                f"Ensure input_preprocessing['source'] defines exactly "
                f"{self.in_channels} vital(s)."
            )
        return x

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Forward pass through NABNet architecture.

        Args:
            batch_dict: Unified batch dictionary with input tensors.

        Returns:
            Dictionary containing model predictions.
        """
        if not isinstance(batch_dict, dict):
            raise TypeError(
                "NABNet.forward expects a unified batch dictionary produced by the "
                f"collate pipeline; received {type(batch_dict).__name__}."
            )
        x = self.extract_input(batch_dict)

        encoder_outputs = []

        # Encoding path
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        # AutoEncoder feature extraction if enabled
        if self.A_E and self.feature_extraction is not None:
            x = self.feature_extraction(x)

        # Bridge
        x = self.bridge(x)

        # Decoding path
        encoder_outputs.reverse()
        decoder_outputs = []

        for i in range(self.model_depth):
            # Upsampling
            x = self.decoder_blocks[i](x)
            enc_feat = encoder_outputs[i]

            if self.A_G and self.attention_blocks is not None:
                enc_feat = self.attention_blocks[i](enc_feat, x)

            # Concatenate and apply convolutions
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder_conv_blocks[i](x)

            if self.D_S and self.ds_convs is not None:
                decoder_outputs.append(self.ds_convs[i](x))

        # Output
        x = self.final_conv(x)
        x = self.final_activation(x)

        if self.D_S and self.ds_convs is not None:
            decoder_outputs.append(x)
            predictions = tuple(
                decoder_outputs[::-1]
            )  # Reverse order to match TF implementation
        else:
            predictions = x

        return {
            "predictions": predictions,
            "extras": {},
        }


# ============================================================================
# REFINEMENT MODEL: ShallowUNet (Refinement Model)
# ============================================================================


class MLPRegressor(SingleStageModel):
    """PyTorch implementation of MLPRegressor for ShallowUNet."""

    def __init__(
        self,
        name: str,
        hidden_size: int = 100,
        activation: str = "relu",
        alpha: float = 0.0001,
        supports_multi_directional: bool = False,
        input_length: int = MISSING,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize MLPRegressor with configuration parameters.

        Args:
            name: Name identifier for this regressor head
            input_length: Number of input features
            hidden_size: Size of hidden layer
            activation: Activation function name ('relu' or 'tanh')
            alpha: L2 regularization strength
            supports_multi_directional: Whether model supports multi-directional
                processing
        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("supports_multi_directional", "input_length")
        }
        super().__init__(supports_multi_directional, input_length, *args, **kwargs)

        self.name = name
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.layers = nn.Sequential(
            nn.Linear(self.input_length, hidden_size),
            self.activation,
            nn.Linear(hidden_size, 1),
        )

        # L2 regularization strength
        self.alpha = alpha

    def extract_input(
        self, batch_dict: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Extract 2D features [B, F] from either dict or tensor input.

        Accepts:
            - Dict with key 'x' containing features tensor [B, F]
            - Direct tensor [B, F] (backward compatibility)

        Raises:
            ValueError: If dict missing 'x' or tensor is not 2D
            TypeError: If extracted value is not a torch.Tensor
        """
        return _extract_2d_features(batch_dict, class_name="MLPRegressor")

    def forward(
        self, x: dict[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        """Forward pass accepting either batch dict {'x': [B, F]} or tensor [B, F]."""
        features = self.extract_input(x)
        return {
            "predictions": self.layers(features),
            "extras": {},
        }


class MultiMLPRegressor(SingleStageModel):
    """Multi-head MLP regressor using ModuleDict for named heads.

    Returns canonical schema with concatenated predictions tensor [B, 2] where the first
        column is
    SBP and the second column is DBP. Internally manages multiple named heads (sbp, dbp)
        but
    outputs a unified tensor for consistency with the standard model interface.
    """

    def __init__(
        self,
        mlp_heads: list[MLPRegressor],
        supports_multi_directional: bool = False,
        input_length: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize MultiMLPRegressor with configuration parameters.

        Args:
            mlp_heads: List of MLPRegressor head instances
            supports_multi_directional: Whether model supports multi-directional
                processing
            input_length: Input sequence length
        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("supports_multi_directional", "input_length")
        }
        super().__init__(supports_multi_directional, input_length, *args, **kwargs)

        self.heads = nn.ModuleDict({head.name: head for head in mlp_heads})
        self.head_names = [head.name for head in mlp_heads]

    def extract_input(
        self, batch_dict: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Extract 2D features [B, F] from either dict or tensor input.

        Accepts:
            - Dict with key 'x' containing features tensor [B, F]
            - Direct tensor [B, F] (backward compatibility)

        Raises:
            ValueError: If dict missing 'x' or tensor is not 2D
            TypeError: If extracted value is not a torch.Tensor
        """
        return _extract_2d_features(batch_dict, class_name="MultiMLPRegressor")

    def forward(
        self, x: dict[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        """Run input through each head and return canonical schema outputs."""
        features = self.extract_input(x)
        predictions_list = []
        for head_name in self.head_names:
            head_output = self.heads[head_name](features)
            predictions_list.append(head_output["predictions"])

        predictions = torch.cat(predictions_list, dim=1)
        return {
            "predictions": predictions,
            "extras": {},
        }

    def __getitem__(self, name: str) -> MLPRegressor:
        """Access individual heads by name."""
        return cast("MLPRegressor", self.heads[name])


class ShallowAttentionBlock(nn.Module):
    """Attention Block for ShallowUNet."""

    def __init__(
        self, in_channels: int, num_filters: int, is_transconv: bool = True
    ) -> None:
        """Initialize ShallowAttentionBlock with configuration parameters.

        Args:
            in_channels: Number of input channels
            num_filters: Number of filters for attention computation
            is_transconv: Whether transposed convolutions are used for upsampling
        """
        super().__init__()
        # First conv for skip connection
        self.conv1x1_1 = nn.Conv1d(in_channels, num_filters, 1, stride=2)
        # Second conv for gating signal - adjust based on upsampling method
        gating_channels = in_channels if is_transconv else in_channels * 2
        self.conv1x1_2 = nn.Conv1d(gating_channels, num_filters, 1, stride=2)
        self.conv_final = nn.Conv1d(num_filters, 1, 1, stride=1)

        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn_final = nn.BatchNorm1d(1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.up_conv = ShallowUpConvBlock()
        self.trans_conv = trans_conv_block(1, 1)

    def forward(
        self, skip_connection: torch.Tensor, gating_signal: torch.Tensor
    ) -> torch.Tensor:
        """Process skip connection with gating signal for attention mechanism.

        Args:
            skip_connection: Skip connection tensor from encoder.
            gating_signal: Gating signal tensor for attention.

        Returns:
            Gated and processed tensor.
        """
        conv1x1_1 = self.conv1x1_1(skip_connection)
        conv1x1_1 = self.bn1(conv1x1_1)

        conv1x1_2 = self.conv1x1_2(gating_signal)
        conv1x1_2 = self.bn2(conv1x1_2)
        conv1_2 = conv1x1_1 + conv1x1_2

        conv1_2 = self.relu(conv1_2)
        conv1_2 = self.conv_final(conv1_2)
        conv1_2 = self.bn_final(conv1_2)

        conv1_2 = self.sigmoid(conv1_2)

        # Upsample for attention gating: enables spatial attention mechanism to
        # modulate skip connections
        resampler1 = self.up_conv(conv1_2)

        resampler2 = self.trans_conv(conv1_2)

        resampler = resampler1 + resampler2

        result = skip_connection * resampler

        return result


class ShallowConcatBlock(nn.Module):
    """Concatenation Block for ShallowUNet."""

    def __init__(self) -> None:
        """Initialize ShallowConcatBlock."""
        super().__init__()

    def forward(self, input1: torch.Tensor, *argv: torch.Tensor) -> torch.Tensor:
        """Concatenate multiple input tensors along channel dimension.

        Args:
            input1: First input tensor.
            *argv: Additional input tensors to concatenate.

        Returns:
            Concatenated tensor.
        """
        cat = input1
        for arg in argv:
            cat = torch.cat([cat, arg], dim=1)
        return cat


class ShallowUpConvBlock(nn.Module):
    """1D UpSampling Block for ShallowUNet."""

    def __init__(self, size: int = 2) -> None:
        """Initialize ShallowUpConvBlock with configuration parameters.

        Args:
            size: Upsampling scale factor
        """
        super().__init__()
        self.size = size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Upsample input tensor using nearest neighbor interpolation.

        Args:
            inputs: Input tensor to upsample.

        Returns:
            Upsampled tensor.
        """
        return F.interpolate(inputs, scale_factor=self.size, mode="nearest")


class ShallowConvLSTM1D(nn.Module):
    """1D Convolutional LSTM for ShallowUNet."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        padding: str = "same",
        go_backwards: bool = True,
    ) -> None:
        """Initialize ShallowConvLSTM1D with configuration parameters.

        Args:
            input_size: Number of input channels
            hidden_size: Number of hidden state channels
            kernel_size: Size of convolution kernels
            padding: Padding mode for convolutions
            go_backwards: Whether to process sequence in reverse order
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.go_backwards = go_backwards

        # Input gate
        self.conv_i = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_i = nn.BatchNorm1d(hidden_size)

        # Forget gate: controls what information to discard from cell state
        # (for LSTM memory management)
        self.conv_f = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_f = nn.BatchNorm1d(hidden_size)

        # Cell gate
        self.conv_c = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_c = nn.BatchNorm1d(hidden_size)

        # Output gate
        self.conv_o = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_o = nn.BatchNorm1d(hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bidirectional convolutional LSTM block.

        Args:
            x: Input tensor with shape (batch, channels, sequence_length).

        Returns:
            Processed tensor after bidirectional LSTM.
        """
        batch_size, channels, seq_len = x.shape

        h_t = torch.zeros(batch_size, self.hidden_size, seq_len).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, seq_len).to(x.device)

        # Reverse sequence if go_backwards
        if self.go_backwards:
            x = torch.flip(x, [2])

        # Gates
        i = self.sigmoid(self.bn_i(self.conv_i(x)))
        f = self.sigmoid(self.bn_f(self.conv_f(x)))
        c = self.tanh(self.bn_c(self.conv_c(x)))
        o = self.sigmoid(self.bn_o(self.conv_o(x)))

        c_t = f * c_t + i * c
        h_t = o * self.tanh(c_t)

        return h_t


class ShallowUNet(SingleStageModel):
    """Core ShallowUNet implementation.

    The canonical constructor argument for input channels is ``in_channels``.
    """

    def __init__(
        self,
        model_depth: int,
        model_width: int,
        kernel_size: int,
        in_channels: int = 1,
        problem_type: str = "Regression",
        output_nums: int = 1,
        deep_supervision: int = 0,
        autoencoder: int = 1,
        guided_attention: int = 0,
        use_lstm: int = 0,
        alpha: float = 1,
        feature_number: int = 1024,
        use_transconv: bool = True,
        feature_extraction_only: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize ShallowUNet with configuration parameters.

        Args:
            model_depth: Depth of the ShallowUNet (number of downsampling layers)
            model_width: Width of the input layer
            kernel_size: Kernel size for convolutional layers
            in_channels: Number of input channels
            problem_type: Type of problem (Regression or Classification)
            output_nums: Number of output channels
            deep_supervision: Whether to use deep supervision
            autoencoder: Whether autoencoder architecture is enabled
            guided_attention: Whether attention modules are included
            use_lstm: Whether to include LSTM layers
            alpha: General alpha parameter for UNet
            feature_number: Feature vector size for autoencoder bottleneck
            use_transconv: Whether to use transposed convolutions
            feature_extraction_only: Whether to only extract features (for cascade
                stage1)
        """
        super().__init__(*args, **kwargs)

        self.model_depth = model_depth
        self.in_channels = in_channels
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = deep_supervision  # Keep internal name for backward compatibility
        self.A_E = autoencoder  # Keep internal name for backward compatibility
        self.A_G = guided_attention  # Keep internal name for backward compatibility
        self.LSTM = use_lstm  # Keep internal name for backward compatibility
        self.alpha = alpha
        self.feature_number = feature_number
        self.is_transconv = (
            use_transconv  # Keep internal name for backward compatibility
        )
        self.feature_extraction_only = feature_extraction_only

        # Input validation
        if (
            self.input_length == 0
            or self.model_depth == 0
            or self.model_width == 0
            or self.in_channels == 0
            or self.kernel_size == 0
        ):
            raise ValueError("Please Check the Values of the Input Parameters!")

        self.encoder_blocks = nn.ModuleList()
        for i in range(1, self.model_depth + 1):
            in_channels = self.in_channels if i == 1 else model_width * (2 ** (i - 2))
            out_channels = model_width * (2 ** (i - 1))
            double_conv = nn.Sequential(
                conv_block(in_channels, out_channels, kernel_size),
                conv_block(out_channels, out_channels, kernel_size),
            )
            self.encoder_blocks.append(double_conv)

        self.pool = nn.MaxPool1d(2)

        # Bottleneck channels should match the last encoder block output
        bottleneck_in_channels = model_width * (2 ** (model_depth - 1))
        bottleneck_out_channels = model_width * (2**model_depth)

        # AutoEncoder feature extraction
        self.feature_extraction: FeatureExtractionBlock | None = None
        if self.A_E:
            # Feature extractor expects fixed input length from encoder output
            input_size = bottleneck_in_channels * (
                self.input_length // (2**model_depth)
            )
            self.feature_extraction = FeatureExtractionBlock(
                input_size=input_size,
                feature_number=feature_number,
                model_width=bottleneck_in_channels,
            )
        else:
            self.feature_extraction = None

        # Bottleneck
        self.bottleneck = nn.Sequential(
            conv_block(bottleneck_in_channels, bottleneck_out_channels, kernel_size),
            conv_block(bottleneck_out_channels, bottleneck_out_channels, kernel_size),
        )

        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        self.ds_convs = nn.ModuleList() if self.D_S else None
        self.attention_blocks = nn.ModuleList() if self.A_G else None
        self.lstm_layers = nn.ModuleList() if self.LSTM else None

        # Utility blocks
        self.concat_block = ShallowConcatBlock()

        for i in range(model_depth):
            # Channel count from encoder output for feature extractor
            in_channels = model_width * (2 ** (model_depth - i))
            out_channels = model_width * (2 ** (model_depth - i - 1))

            # 1. Deep Supervision
            if self.D_S and self.ds_convs is not None:
                self.ds_convs.append(nn.Conv1d(in_channels, 1, 1))

            # 2. Upsampling blocks
            if self.is_transconv:
                self.up_blocks.append(trans_conv_block(in_channels, out_channels))
            else:
                self.up_blocks.append(ShallowUpConvBlock())

            # 3. Attention
            if self.A_G and self.attention_blocks is not None:
                self.attention_blocks.append(
                    ShallowAttentionBlock(
                        out_channels, out_channels, is_transconv=self.is_transconv
                    )
                )

            # 4. LSTM
            if self.LSTM and self.lstm_layers is not None:
                lstm_in_channels = (
                    in_channels + out_channels
                    if not self.is_transconv
                    else out_channels * 2
                )
                self.lstm_layers.append(
                    ShallowConvLSTM1D(
                        input_size=lstm_in_channels,  # From upsampling
                        hidden_size=out_channels,
                        kernel_size=3,
                        padding="same",
                        go_backwards=True,
                    )
                )

            # 5. Decoder Convolutions
            if self.is_transconv:
                decoder_in_channels = (
                    out_channels * 2
                )  # out_channels from TransConv + out_channels from skip
            else:
                decoder_in_channels = (
                    in_channels + out_channels
                )  # in_channels from upsampled + out_channels from skip

            if self.LSTM:
                decoder_in_channels = out_channels  # LSTM output channels

            self.decoder_blocks.append(
                nn.Sequential(
                    conv_block(decoder_in_channels, out_channels, kernel_size),
                    conv_block(out_channels, out_channels, kernel_size),
                )
            )

        # Output layer (moved after decoder part)
        self.final_conv = nn.Conv1d(model_width, output_nums, 1)
        self.final_activation = (
            nn.Softmax(dim=1) if problem_type == "Classification" else nn.Identity()
        )

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and prepare input for ShallowUNet from batch dict.

        This method handles the input processing for ShallowUNet, including:
        - Waveform channel extraction
        - Shape formatting to [B, 1, L] for UNet

        Args:
            batch_dict: Standardized batch dict from DataLoader collate_fn

        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        # Parent class validates required fields and extracts input
        x = super().extract_input(batch_dict)
        if isinstance(x, dict):
            x = x["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("ShallowUNet extract_input expected Tensor from parent")
        return x

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Forward pass of ShallowUNet.

        Feature extraction mode is instance-configured via
            `self.feature_extraction_only` (from YAML).
        When `self.feature_extraction_only=True`, returns 2D features [B, F]. Otherwise
            returns
        waveform predictions [B, 1, T] or a tuple for deep supervision.
        """
        if self.feature_extraction_only and not self.A_E:
            raise ValueError(
                "feature_extraction_only requires ae (AutoEncoder) to be enabled"
            )

        x = self.extract_input(batch_dict)

        skips = []

        # Encoder path
        for _i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # AutoEncoder feature extraction
        if self.A_E and self.feature_extraction is not None:
            x = self.feature_extraction(x, feature_only=self.feature_extraction_only)
            if self.feature_extraction_only:
                return {
                    "predictions": x,
                    "extras": {},
                }

        # Bottleneck
        x = self.bottleneck(x)

        ds_outputs = []

        # Decoder path
        skips = skips[::-1]  # Reverse for easier access

        for i in range(self.model_depth):
            # 1. Deep supervision
            if self.D_S and self.ds_convs is not None:
                ds_outputs.append(self.ds_convs[i](x))

            # 2. Upsampling
            x = self.up_blocks[i](x)

            skip = skips[i]

            # 4. Apply attention if enabled
            if self.A_G and self.attention_blocks is not None:
                skip = self.attention_blocks[i](skip, x)

            # 5. LSTM if enabled
            if self.LSTM and self.lstm_layers is not None:
                combined = self.concat_block(x, skip)
                x = self.lstm_layers[i](combined)
            else:
                x = self.concat_block(x, skip)

            # 6. Convolutions
            x = self.decoder_blocks[i](x)

        # Final output
        x = self.final_conv(x)
        outputs = self.final_activation(x)

        if self.D_S and self.ds_convs is not None:
            ds_outputs.append(outputs)
            predictions = tuple(ds_outputs[::-1])
        else:
            predictions = outputs

        return {
            "predictions": predictions,
            "extras": {},
        }


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_mlp_regressor", group="model", node=MLPRegressorConfig)
cs.store(name="base_multi_mlp_regressor", group="model", node=MultiMLPRegressorConfig)
cs.store(name="base_nabnet", group="model", node=NABNetModelConfig)
cs.store(name="base_shallow_unet", group="model", node=ShallowUNetModelConfig)
