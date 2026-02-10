"""MD-ViSCo Implementation.

This module implements MD-ViSCo (Multi-Directional Vital Sign Waveform Conversion)
for unified vital sign waveform conversion.

References:
- Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
  https://ieeexplore.ieee.org/document/9710580/
- Paper: "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"
  https://link.springer.com/10.1007/978-3-031-25066-8_9
- Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
  https://arxiv.org/abs/2211.14730
- Paper: "End-To-End Personalized Cuff-Less Blood Pressure Monitoring Using ECG
  and PPG Signals"
  https://ieeexplore.ieee.org/document/10445970/
- Code: https://github.com/fr-meyer/MD-ViSCo
- Documentation: https://huggingface.co/docs/transformers/main/en/model_doc/patchtst
- License: MIT

Note: This implementation combines Swin Transformer, Swin-Unet, and PatchTST
architectures for multi-directional vital sign waveform conversion.
"""

# Sections: Imports, Config, Approximation, Refinement

# =====================
# 1. Imports
# =====================

import logging
import math

# Standard library imports
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional
import torch.utils.checkpoint as checkpoint

# Third-party imports
from einops import rearrange
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from transformers import DistilBertConfig
from transformers import DistilBertModel
from transformers import PatchTSMixerConfig
from transformers import PatchTSMixerModel

# Local imports
from src.core.domain import (
    Direction,  # noqa: TC001  # Used at runtime for batch handling
)
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig
from src.utils.constants import BATCH_KEY_INPUT
from src.utils.constants import BATCH_KEY_TARGET_INDICES

logger = logging.getLogger(__name__)

# =====================
# 2. Model Configuration Classes
# =====================


@dataclass
class UNetSwinUnetConfig(SingleStageModelConfig):
    """Configuration for UNet-SwinUnet architecture parameters.

    Core tunable model hyperparameters that define the layout and complexity
    of the approximation network (UNet_SwinUnet), independently of training or dataset.

    Architecture Hyperparameters (Hydra-configurable):
        - embedding_dim_multiplier: Embedding dimension multiplier for Swin Transformer
        - swin_num_heads: Attention heads per layer in Swin Transformer
        - swin_mlp_ratio: MLP expansion ratio in Swin blocks
        - swin_drop_rate: Dropout rate for Swin Transformer
        - swin_attn_drop_rate: Attention dropout rate for Swin Transformer
        - swin_drop_path_rate: Stochastic depth rate for Swin Transformer
        - leaky_relu_negative_slope: Negative slope for LeakyReLU activations

    These parameters control the Swin Transformer approximation architecture and can be
    overridden via Hydra (e.g., model.swin_drop_rate=0.2,
        model.embedding_dim_multiplier=8).
    """

    _target_: str = "src.model.mdvisco.UNetSwinUnet"
    supports_multi_directional: bool = True
    model_name: str = "UNetSwinT"

    # Model Architecture Configuration Attributes
    # These directly influence the model's internal design:

    in_channels: int = 1
    """Number of input channels (e.g., 1 for single waveform input)."""

    out_channels: int = 1
    """Number of output channels (e.g., 1 for target waveform output)."""

    init_features: int = 64
    """Number of filters/features in the first convolutional layer of the U-Net."""

    kernel_size: int = 3
    """Size of convolution kernels used in the model."""

    patch_size: int = 4
    """Patch size for the Swin Transformer block in the bottleneck."""

    depth: int = 1
    """Depth of transformer layers or U-Net encoder/decoder levels."""

    upsample_scale: list[int] = field(default_factory=lambda: [4])
    """Scaling factor(s) used for upsampling layers (e.g., [4] to upscale 4×)."""

    # Additional model parameters
    style_dim: int = 3
    """Dimension of the style vector for AdaIN."""

    embedding_dim_multiplier: int = 4
    """Embedding dimension multiplier for Swin Transformer bottleneck (embed_dim =
        in_chans ×
    multiplier)."""

    swin_num_heads: list[int] = field(default_factory=lambda: [32, 32, 32, 32, 32])
    """Number of attention heads per Swin Transformer layer."""

    swin_mlp_ratio: float = 4.0
    """MLP expansion ratio in Swin Transformer blocks."""

    swin_drop_rate: float = 0.0
    """Dropout rate for Swin Transformer."""

    swin_attn_drop_rate: float = 0.0
    """Attention dropout rate for Swin Transformer."""

    swin_drop_path_rate: float = 0.1
    """Stochastic depth (drop path) rate for Swin Transformer."""

    leaky_relu_negative_slope: float = 0.2
    """Negative slope for LeakyReLU activations throughout the model."""

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if self.init_features <= 0:
            raise ValueError("init_features must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.style_dim <= 0:
            raise ValueError("style_dim must be positive")
        if self.input_length is not None and self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if not self.upsample_scale:
            raise ValueError("upsample_scale cannot be empty")
        if any(scale <= 0 for scale in self.upsample_scale):
            raise ValueError("All values in upsample_scale must be positive")

        if self.embedding_dim_multiplier < 1:
            raise ValueError("embedding_dim_multiplier must be >= 1")
        if not self.swin_num_heads:
            raise ValueError("swin_num_heads cannot be empty")
        if any(heads < 1 for heads in self.swin_num_heads):
            raise ValueError("All values in swin_num_heads must be >= 1")
        if self.swin_mlp_ratio <= 0.0:
            raise ValueError("swin_mlp_ratio must be > 0.0")
        if self.swin_drop_rate < 0.0:
            raise ValueError("swin_drop_rate must be >= 0.0")
        if self.swin_attn_drop_rate < 0.0:
            raise ValueError("swin_attn_drop_rate must be >= 0.0")
        if self.swin_drop_path_rate < 0.0:
            raise ValueError("swin_drop_path_rate must be >= 0.0")
        if self.leaky_relu_negative_slope < 0.0:
            raise ValueError("leaky_relu_negative_slope must be >= 0.0")

        # Validate patch size compatibility
        if self.input_length is not None and self.input_length % self.patch_size != 0:
            raise ValueError(
                f"input_length ({self.input_length}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )


# =====================
# 3. Approximation Model (UNet_SwinUnet and related classes)
# =====================


class UNetSwinUnet(SingleStageModel):
    """A hybrid UNet-Swin Transformer architecture for signal processing with adaptive
        instance
    normalization.

    This model combines the hierarchical feature extraction of UNet with the
    self-attention mechanism of Swin Transformer, enhanced with adaptive instance
    normalization (AdaIN) for style transfer capabilities.

    Architecture:
    - Encoder: Series of convolutional blocks with downsampling
    - Bottleneck: Swin Transformer with AdaIN for feature transformation
    - Decoder: Series of AdaIN residual blocks with upsampling
    - Skip connections between encoder and decoder

    Args:
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        init_features (int): Initial number of features in the first layer. Default: 64
        kernel_size (int): Kernel size for convolutional layers. Default: 3
        style_dim (int): Dimension of the style vector for AdaIN. Default: 3
        upsample_scale (list): List of upsampling scales for the Swin Transformer.
            Default: [4]
        input_length (int): Length of the input signal. Default: 1024
        patch_size (int): Size of patches for the Swin Transformer. Default: 4
        depth (int): Depth of the UNet architecture. Default: 1

    Paper Reference:
        MD-ViSCo paper: IEEE J. Biomed. Health Inform. (2026,
        DOI: 10.1109/JBHI.2025.3639315), Section 6.2.1, page 22
        - Filter channels: 64, kernel size: 3, window size: 4
        - Single-level U-Net with upsample scale [4]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 64,
        kernel_size: int = 3,
        style_dim: int = 3,
        upsample_scale: list[int] | None = None,
        patch_size: int = 4,
        depth: int = 1,
        embedding_dim_multiplier: int = 4,
        swin_num_heads: list[int] | None = None,
        swin_mlp_ratio: float = 4.0,
        swin_drop_rate: float = 0.0,
        swin_attn_drop_rate: float = 0.0,
        swin_drop_path_rate: float = 0.1,
        leaky_relu_negative_slope: float = 0.2,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize UNet_SwinUnet with configuration parameters.

        Args:
            in_channels: Number of input channels. Default: 1
            out_channels: Number of output channels. Default: 1
            init_features: Initial number of features in the first layer. Default: 64
            kernel_size: Kernel size for convolutional layers. Default: 3
            style_dim: Dimension of the style vector for AdaIN. Default: 3
            upsample_scale: List of upsampling scales for the Swin Transformer. Default:
                [4]
            patch_size: Size of patches for the Swin Transformer. Default: 4
            depth: Depth of the UNet architecture. Default: 1
            embedding_dim_multiplier: Embedding dimension multiplier for Swin
                Transformer.
                Default: 4
            swin_num_heads: Number of attention heads per layer in Swin Transformer.
                Default: [32, 32, 32, 32, 32]
            swin_mlp_ratio: MLP expansion ratio in Swin blocks. Default: 4.0
            swin_drop_rate: Dropout rate for Swin Transformer. Default: 0.0
            swin_attn_drop_rate: Attention dropout rate for Swin Transformer. Default:
                0.0
            swin_drop_path_rate: Stochastic depth rate for Swin Transformer. Default:
                0.1
            leaky_relu_negative_slope: Negative slope for LeakyReLU activations.
                Default: 0.2
        """
        super().__init__(*args, **kwargs)
        if upsample_scale is None:
            upsample_scale = [4]
        if swin_num_heads is None:
            swin_num_heads = [32, 32, 32, 32, 32]

        features = init_features
        self.features = init_features
        self.conv_init_features = nn.Conv1d(in_channels, features, 3, 1, 1)

        self.encoder = nn.ModuleList()
        self.depth = depth
        w_size = self.input_length
        for i in range(0, depth):
            self.encoder.append(
                UNetSwinUnet._block(
                    features, features, k=kernel_size, name="enc" + str(i) + "_1"
                )
            )
            self.encoder.append(
                nn.Conv1d(
                    in_channels=features, out_channels=features, kernel_size=2, stride=2
                )
            )
            self.encoder.append(
                UNetSwinUnet._block(
                    features * 2,
                    features * 2,
                    k=kernel_size,
                    name="enc" + str(i) + "_2",
                )
            )

            # Double features and halve spatial dimensions for next layer
            features = features * 2
            w_size = w_size // 2

        # Configure Swin Transformer bottleneck
        in_chans = features
        patch_size = patch_size
        embed_dim = (
            in_chans * embedding_dim_multiplier
        )  # Embedding dimension for transformer

        window_size = patch_size
        num_heads = swin_num_heads  # Number of attention heads for each layer
        num_classes = embed_dim

        self.bottleneck = SwinTransformerSysAdaIn(
            img_size=w_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=upsample_scale,
            depths_decoder=upsample_scale,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=swin_mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=swin_drop_rate,
            attn_drop_rate=swin_attn_drop_rate,
            drop_path_rate=swin_drop_path_rate,
            norm_layer_encoder=nn.InstanceNorm1d,
            norm_layer_decoder=AdaIN,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first",
            style_dim=style_dim,
        )
        w_size = self.input_length

        self.decoder = nn.ModuleList()
        for i in range(0, depth):
            if i == 0:
                # Match Version 1 logic for first decoder block
                if len(upsample_scale) > 1:
                    self.decoder.append(
                        AdainResBlk(
                            dim_in=embed_dim // patch_size,
                            dim_out=features // 2,
                            upsample=True,
                            style_dim=style_dim,
                            upsample_scale=2,
                            k=3,
                            leaky_relu_negative_slope=leaky_relu_negative_slope,
                        )
                    )
                    self.decoder.append(
                        AdainResBlk(
                            dim_in=features,
                            dim_out=features // 2,
                            upsample=False,
                            style_dim=style_dim,
                            upsample_scale=2,
                            k=3,
                            leaky_relu_negative_slope=leaky_relu_negative_slope,
                        )
                    )
                else:
                    self.decoder.append(
                        AdainResBlk(
                            dim_in=embed_dim // patch_size,
                            dim_out=features // 2,
                            upsample=True,
                            style_dim=style_dim,
                            upsample_scale=2,
                            k=3,
                            leaky_relu_negative_slope=leaky_relu_negative_slope,
                        )
                    )
                    self.decoder.append(
                        AdainResBlk(
                            dim_in=features,
                            dim_out=features // 2,
                            upsample=False,
                            style_dim=style_dim,
                            upsample_scale=2,
                            k=3,
                            leaky_relu_negative_slope=leaky_relu_negative_slope,
                        )
                    )
            else:
                self.decoder.append(
                    AdainResBlk(
                        dim_in=features,
                        dim_out=features // 2,
                        upsample=True,
                        style_dim=style_dim,
                        upsample_scale=2,
                        k=3,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
                self.decoder.append(
                    AdainResBlk(
                        dim_in=features,
                        dim_out=features // 2,
                        upsample=False,
                        style_dim=style_dim,
                        upsample_scale=2,
                        k=3,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
            features = features // 2  # Halve features for next layer

        # Final output layers
        self.last = nn.Sequential(
            nn.InstanceNorm1d(features, affine=True),  # Normalize final features
            nn.LeakyReLU(leaky_relu_negative_slope),  # Non-linear activation
            nn.Conv1d(  # Final convolution to output channels
                in_channels=features,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
        )

    def extract_domain_target(
        self, batch_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract domain shift target for MDViSCo.

        MDViSCo-specific functionality for generating one-hot domain vectors.

        Args:
            batch_dict: Unified batch dict with tgt_idxs

        Returns:
            torch.Tensor: One-hot encoded domain vectors [B, D]
        """
        device = batch_dict[BATCH_KEY_INPUT].device
        tgt_idxs = batch_dict[BATCH_KEY_TARGET_INDICES].to(
            device=device, dtype=torch.long
        )  # [B]

        max_idx = tgt_idxs.max().item()
        if max_idx >= self.bottleneck.style_dim:
            raise ValueError(
                f"Target domain index {max_idx} is out of range. "
                f"Indices must be in [0, {self.bottleneck.style_dim - 1}] "
                f"(style_dim={self.bottleneck.style_dim})."
            )

        domain_shift_target = F.one_hot(
            tgt_idxs, num_classes=self.bottleneck.style_dim
        ).to(device=device, dtype=torch.float32)  # [B, D]

        return domain_shift_target

    def extract_input(
        self, batch_dict: dict[str, torch.Tensor]
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor | None]
    ):
        """Extract input for MDViSCo.

        Returns the input tensor extracted from batch_dict. Domain target extraction is
            handled
        separately in forward().
        """
        x = super().extract_input(batch_dict)

        return x

    def forward(
        self,
        batch_dict: dict[str, torch.Tensor],
        s: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the UNet-Swin Transformer network.

        The forward pass consists of three main stages:
        1. Encoder: Progressive downsampling with feature extraction
        2. Bottleneck: Swin Transformer processing with style conditioning
        3. Decoder: Progressive upsampling with skip connections

        Args:
            batch_dict (Dict[str, torch.Tensor]): Input batch dictionary from collate
                function.
                The method calls extract_input(batch_dict) which returns a dict
                    containing:
                - "x": Source waveforms tensor of shape (batch_size, in_channels,
                    input_size)
                - "domain_shift_target": One-hot domain vectors for style conditioning
            s (torch.Tensor, optional): Style vector for AdaIN conditioning. If None,
                uses
                domain_shift_target from extract_input(). Shape: (batch_size, style_dim)

        Returns:
            Dict[str, torch.Tensor]: Canonical schema dictionary containing:
                - "y_pred": Output waveform tensor (compatibility key for
                    trainers/processors)
                - "predictions": Output waveform tensor of shape (batch_size,
                    out_channels,
                  input_size)
                - "extras": Empty dict (for consistency with other models)
        """
        x = self.extract_input(batch_dict)
        if s is None:
            s = self.extract_domain_target(batch_dict)

        enc = self.conv_init_features(x)
        encoder: list[torch.Tensor] = []

        # Encoder path: Progressive downsampling
        for i in range(0, len(self.encoder), 3):
            encoder.insert(0, enc)

            enc = self.encoder[i](enc)

            # Downsampling through two paths:
            # 1. Max pooling
            x_down = F.max_pool1d(enc, 2)
            # 2. Convolutional downsampling
            x_conv_pool = self.encoder[i + 1](enc)
            # Concatenate both downsampled features
            enc = torch.cat([x_down, x_conv_pool], dim=1)

            # Second convolutional block
            enc = self.encoder[i + 2](enc)

        bottleneck = self.bottleneck(enc, s)

        # Decoder path: Progressive upsampling with skip connections
        dec = bottleneck

        for idx, i in enumerate(range(0, len(self.decoder), 2)):
            dec = self.decoder[i](dec, s)

            if idx < len(encoder):
                dec = torch.cat([dec, encoder[idx]], dim=1)

            dec = self.decoder[i + 1](dec, s)
        out = self.last(dec)

        # Final output processing (y_pred for trainer/processor compatibility;
        # predictions/extras for canonical schema)
        return cast(
            "dict[str, torch.Tensor]",
            {
                "y_pred": out,
                "predictions": out,
                "extras": {},
            },
        )

    @staticmethod
    def _block_conv(
        in_channels: int,
        features: int,
        k: int,
        name: str,
        input_size: int,
    ) -> nn.Sequential:
        """Create a basic convolutional block with a single 1D convolution layer.

        This method creates a simple sequential block containing a single convolutional
            layer
        with proper padding to maintain the input size. The block is used as a building
            block
        for more complex architectures.

        Args:
            in_channels (int): Number of input channels
            features (int): Number of output features/channels
            k (int): Kernel size for the convolution
            name (str): Base name for the layer (will be appended with 'conv1')
            input_size (int): Size of the input signal (used for padding calculation)

        Returns:
            nn.Sequential: A sequential block containing the convolutional layer

        Note:
            The padding is set to k//2 to maintain the input size after convolution.
            Bias is disabled in the convolution layer.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",  # Layer name with suffix
                        nn.Conv1d(
                            in_channels=in_channels,  # Input channels
                            out_channels=features,  # Output features
                            kernel_size=k,  # Kernel size
                            padding=k // 2,  # Padding to maintain size
                            bias=False,  # Disable bias
                        ),
                    )
                ]
            )
        )

    @staticmethod
    def _block_ln(
        in_channels: int,
        features: int,
        k: int,
        name: str,
        input_size: int,
    ) -> nn.Sequential:
        """Create a normalization block with LayerNorm and LeakyReLU activation.

        This method creates a sequential block containing a layer normalization layer
        followed by a LeakyReLU activation. The block is used for normalizing and
        activating features in the network.

        Args:
            in_channels (int): Number of input channels (not used in this block)
            features (int): Number of features to normalize
            k (int): Kernel size (not used in this block)
            name (str): Base name for the layers (will be appended with 'norm1' and
                'relu1')
            input_size (int): Size of the input signal (not used in this block)

        Returns:
            nn.Sequential: A sequential block containing:
                - LayerNorm layer for feature normalization
                - LeakyReLU activation for non-linearity

        Note:
            The LeakyReLU is configured with inplace=True for memory efficiency.
            Some parameters (in_channels, k, input_size) are included for interface
            consistency but are not used in this block.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "norm1", nn.LayerNorm(features)),  # Layer normalization
                    (
                        name + "relu1",
                        nn.LeakyReLU(inplace=True),
                    ),  # LeakyReLU activation
                ]
            )
        )

    @staticmethod
    def _block(
        in_channels: int,
        features: int,
        k: int,
        name: str,
    ) -> nn.Sequential:
        """Create a double-convolution block with instance normalization and LeakyReLU
            activation.

        This method creates a sequential block containing two convolutional layers, each
        followed by instance normalization and LeakyReLU activation. This is a common
        building block in the UNet architecture, used for feature extraction and
        transformation.

        The block structure is:
        1. First convolution + normalization + activation
        2. Second convolution + normalization + activation

        Args:
            in_channels (int): Number of input channels for the first convolution
            features (int): Number of output features for both convolutions
            k (int): Kernel size for both convolutions
            name (str): Base name for the layers (will be appended with conv1/2,
                norm1/2, relu1/2)

        Returns:
            nn.Sequential: A sequential block containing:
                - First Conv1d layer with padding to maintain size
                - First InstanceNorm1d layer
                - First LeakyReLU activation
                - Second Conv1d layer
                - Second InstanceNorm1d layer
                - Second LeakyReLU activation

        Note:
            - Both convolutions use padding=k//2 to maintain input size
            - Bias is disabled in both convolutions
            - LeakyReLU is configured with inplace=True for memory efficiency
            - InstanceNorm1d is used for feature normalization
        """
        return nn.Sequential(
            OrderedDict(
                [
                    # First convolution block
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,  # Input channels
                            out_channels=features,  # Output features
                            kernel_size=k,  # Kernel size
                            padding=k // 2,  # Padding to maintain size
                            bias=False,  # Disable bias
                        ),
                    ),
                    (
                        name + "norm1",
                        nn.InstanceNorm1d(num_features=features),
                    ),  # First normalization
                    (name + "relu1", nn.LeakyReLU(inplace=True)),  # First activation
                    # Second convolution block
                    (
                        name + "conv2",
                        nn.Conv1d(
                            # Input features (same as output of first conv)
                            in_channels=features,
                            out_channels=features,  # Output features
                            kernel_size=k,  # Kernel size
                            padding=k // 2,  # Padding to maintain size
                            bias=False,  # Disable bias
                        ),
                    ),
                    (
                        name + "norm2",
                        nn.InstanceNorm1d(num_features=features),
                    ),  # Second normalization
                    (name + "relu2", nn.LeakyReLU(inplace=True)),  # Second activation
                ]
            )
        )


class ResBlk(nn.Module):
    """Residual Block for 1D signal processing with optional normalization and
        downsampling.

    This block implements a residual connection architecture with the following
        features:
    - Optional instance normalization
    - Optional downsampling through dual-path (max pooling + convolution)
    - LeakyReLU activation
    - Residual connection for better gradient flow

    Args:
        dim_in (int): Number of input channels
        dim_out (int): Number of output channels
        actv (nn.Module, optional): Activation function. If None, uses LeakyReLU with
            leaky_relu_negative_slope. Default: None
        normalize (bool): Whether to use instance normalization. Default: False
        downsample (bool): Whether to downsample the input. Default: False
        leaky_relu_negative_slope (float): Negative slope for LeakyReLU activation.
            Default: 0.2
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        actv: nn.Module | None = None,
        normalize: bool = False,
        downsample: bool = False,
        leaky_relu_negative_slope: float = 0.2,
    ) -> None:
        """Initialize ResBlk with configuration parameters.

        Args:
            dim_in: Number of input channels
            dim_out: Number of output channels
            actv: Activation function. If None, uses LeakyReLU with
                leaky_relu_negative_slope. Default: None
            normalize: Whether to use instance normalization. Default: False
            downsample: Whether to downsample the input. Default: False
            leaky_relu_negative_slope: Negative slope for LeakyReLU activation. Default:
                0.2
        """
        super().__init__()
        # Use provided activation or create LeakyReLU with configurable slope
        # (allows flexible activation choice for different architectures)
        self.actv = (
            actv if actv is not None else nn.LeakyReLU(leaky_relu_negative_slope)
        )
        self.normalize = normalize  # Whether to use normalization
        self.downsample = downsample  # Whether to downsample

        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in: int, dim_out: int) -> None:
        """Build the network weights for the residual block.

        This method creates the convolutional layers and normalization layers based on
            the
        configuration (downsampling and normalization flags). The architecture includes:
        - First convolution layer (always present)
        - Second convolution layer (with different input channels based on downsampling)
        - Instance normalization layers (if normalize=True)
        - Downsampling convolution (if downsample=True)

        Args:
            dim_in (int): Number of input channels
            dim_out (int): Number of output channels
        """
        # First convolution layer - always processes input channels
        self.conv1 = nn.Conv1d(dim_in, dim_in, 3, 1, 1)

        # Second convolution layer - input channels depend on downsampling
        if self.downsample:
            # If downsampling, input is concatenated features (2 * dim_in)
            self.conv2 = nn.Conv1d(2 * dim_in, dim_out, 3, 1, 1)
        else:
            # Without downsampling, input is same as first conv output
            self.conv2 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)

        if self.normalize:
            # First normalization layer
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            if self.downsample:
                # Second normalization with doubled channels if downsampling
                self.norm2 = nn.InstanceNorm1d(2 * dim_in, affine=True)
            else:
                # Second normalization with same channels if no downsampling
                self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)

        if self.downsample:
            # Convolution for downsampling in residual path
            self.conv_pool_residual = nn.Conv1d(
                in_channels=dim_in, out_channels=dim_in, kernel_size=2, stride=2
            )

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the residual block's main path.

        This method implements the main processing path of the residual block, which
            includes:
        1. Optional normalization and activation
        2. First convolution
        3. Optional downsampling through dual-path (max pooling + convolution)
        4. Optional normalization and activation
        5. Second convolution

        The downsampling path combines features from both max pooling and convolutional
        downsampling to preserve more information during the downsampling process.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in, length)

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, dim_out, length) or
                         (batch_size, dim_out, length/2) if downsampling is enabled
        """
        # First normalization and activation if enabled
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)

        # First convolution
        x = self.conv1(x)

        # Downsampling through dual-path if enabled
        if self.downsample:
            # Path 1: Max pooling
            x_down = F.max_pool1d(x, 2)
            # Path 2: Convolutional downsampling
            x_conv_pool = self.conv_pool_residual(x)
            # Combine features from both paths
            x = torch.cat([x_down, x_conv_pool], dim=1)

        # Second normalization and activation if enabled
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)

        # Final convolution
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        This method implements the forward pass of the residual block, which processes
        the input through the main residual path. The residual connection is implemented
        in the parent network architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in, length)

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, dim_out, length) or
                         (batch_size, dim_out, length/2) if downsampling is enabled

        Note:
            The actual residual connection (adding the input to the output) is handled
            by the parent network architecture, not within this block.
        """
        x = self._residual(x)
        return x


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN) layer for style transfer.

    This layer implements the AdaIN operation, which adaptively normalizes the input
    features using style information. It consists of:
    1. Instance normalization without learnable parameters
    2. A fully connected layer that generates scaling and shifting parameters from style

    The style information is used to generate adaptive parameters (gamma and beta)
    that modulate the normalized features, allowing for style transfer capabilities.

    Args:
        style_dim (int): Dimension of the style vector
        num_features (int): Number of features to normalize
    """

    def __init__(self, style_dim: int, num_features: int) -> None:
        """Initialize AdaIN with configuration parameters.

        Args:
            style_dim: Dimension of the style vector
            num_features: Number of features to normalize
        """
        super().__init__()
        # Instance normalization without learnable parameters
        self.norm = nn.InstanceNorm1d(num_features, affine=False)

        # Fully connected layer to generate adaptive parameters
        # Outputs 2*num_features parameters (gamma and beta for each feature)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AdaIN layer.

        This method implements the adaptive instance normalization operation:
        1. Generates style parameters (gamma, beta) from the style vector
        2. Normalizes the input features using instance normalization
        3. Modulates the normalized features using the style parameters

        The style modulation is done by:
        - Scaling the normalized features by (1 + gamma)
        - Adding a style-specific bias (beta)

        Args:
            x (torch.Tensor): Input features of shape (batch_size, num_features, length)
            s (torch.Tensor): Style vector of shape (batch_size, style_dim)

        Returns:
            torch.Tensor: Style-modulated features of shape (batch_size, num_features,
                length)
        """
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    """Adaptive Instance Normalization (AdaIN) Residual Block for style transfer.

    This block implements a residual connection architecture with AdaIN for style
        transfer,
    featuring:
    - Optional upsampling through dual-path (nearest neighbor + transposed convolution)
    - AdaIN for style-based feature modulation
    - LeakyReLU activation
    - Residual connection for better gradient flow

    The block can operate in two modes:
    1. Generator mode: Uses AdaIN for style transfer
    2. Discriminator mode: Uses standard instance normalization

    Args:
        dim_in (int): Number of input channels
        dim_out (int): Number of output channels
        k (int): Kernel size for convolutional layers. Default: 3
        style_dim (int): Dimension of the style vector for AdaIN. Default: 64
        w_hpf (int): High-pass filter width. Default: 0
        actv (nn.Module, optional): Activation function. If None, uses LeakyReLU with
            leaky_relu_negative_slope. Default: None
        upsample (bool): Whether to upsample the input. Default: False
        generator (bool): Whether to use AdaIN (True) or standard normalization (False).
            Default: True
        upsample_scale (int): Scale factor for upsampling. Default: 2
        leaky_relu_negative_slope (float): Negative slope for LeakyReLU activation.
            Default: 0.2
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        k: int = 3,
        style_dim: int = 64,
        w_hpf: int = 0,
        actv: nn.Module | None = None,
        upsample: bool = False,
        generator: bool = True,
        upsample_scale: int = 2,
        leaky_relu_negative_slope: float = 0.2,
    ) -> None:
        """Initialize AdainResBlk with configuration parameters.

        Args:
            dim_in: Number of input channels
            dim_out: Number of output channels
            k: Kernel size for convolutional layers. Default: 3
            style_dim: Dimension of the style vector for AdaIN. Default: 64
            w_hpf: High-pass filter width. Default: 0
            actv: Activation function. If None, uses LeakyReLU with
                leaky_relu_negative_slope. Default: None
            upsample: Whether to upsample the input. Default: False
            generator: Whether to use AdaIN (True) or standard normalization (False).
                Default: True
            upsample_scale: Scale factor for upsampling. Default: 2
            leaky_relu_negative_slope: Negative slope for LeakyReLU activation. Default:
                0.2
        """
        super().__init__()
        self.generator = generator  # Whether to use AdaIN or standard normalization
        self.k = k  # Kernel size for convolutions
        self.w_hpf = w_hpf  # High-pass filter width
        # Use provided activation or create LeakyReLU with configurable slope
        # (allows flexible activation choice for different architectures)
        self.actv = (
            actv if actv is not None else nn.LeakyReLU(leaky_relu_negative_slope)
        )
        self.upsample = upsample  # Whether to upsample
        self.upsample_scale = upsample_scale  # Scale factor for upsampling

        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in: int, dim_out: int, style_dim: int = 64) -> None:
        """Build the network weights for the AdaIN residual block.

        This method creates the convolutional layers, normalization layers, and
            upsampling
        components based on the configuration. The architecture includes:
        - First convolution layer (with doubled input channels if upsampling)
        - Second convolution layer
        - AdaIN or InstanceNorm layers based on generator mode
        - Transposed convolution for upsampling (if enabled)

        Args:
            dim_in (int): Number of input channels
            dim_out (int): Number of output channels
            style_dim (int): Dimension of the style vector for AdaIN. Default: 64
        """
        # First convolution layer - input channels depend on upsampling
        if self.upsample:
            # If upsampling, input is concatenated features (2 * dim_in)
            self.conv1 = nn.Conv1d(dim_in * 2, dim_out, self.k, 1, self.k // 2)
        else:
            # Without upsampling, input is same as specified
            self.conv1 = nn.Conv1d(dim_in, dim_out, self.k, 1, self.k // 2)

        # Second convolution layer
        self.conv2 = nn.Conv1d(dim_out, dim_out, self.k, 1, self.k // 2)

        if self.generator:
            # AdaIN layers for style transfer
            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_out)
        else:
            # Standard instance normalization
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_out, affine=True)

        if self.upsample:
            # Transposed convolution for upsampling in residual path
            # Used in combination with nearest neighbor upsampling for better quality
            self.transpose_residual = nn.ConvTranspose1d(
                in_channels=dim_in,
                out_channels=dim_in,
                kernel_size=self.upsample_scale,
                stride=self.upsample_scale,
                padding=0,
                output_padding=0,
                bias=False,
            )

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Process input through the residual block's main path.

        This method implements the main processing path of the AdaIN residual block,
            which
        includes:
        1. First normalization and activation
        2. Optional upsampling through dual-path (nearest neighbor + transposed
            convolution)
        3. First convolution
        4. Second normalization and activation
        5. Final convolution

        The upsampling path combines features from both nearest neighbor interpolation
            and
        transposed convolution to preserve more information during the upsampling
            process.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in, length)
            s (torch.Tensor): Style vector of shape (batch_size, style_dim) for AdaIN

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, dim_out, length) or
                         (batch_size, dim_out, length*upsample_scale) if upsampling is
                             enabled
        """
        # First normalization and activation
        x = self.norm1(x, s) if self.generator else self.norm1(x)
        x = self.actv(x)

        # Upsampling through dual-path if enabled
        if self.upsample:
            # Path 1: Nearest neighbor upsampling
            x_up = F.interpolate(x, scale_factor=self.upsample_scale, mode="nearest")
            # Path 2: Transposed convolution upsampling
            x_trans = self.transpose_residual(x)
            # Combine features from both paths
            x = torch.cat([x_up, x_trans], dim=1)

        # First convolution
        x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x, s) if self.generator else self.norm2(x)
        x = self.actv(x)

        # Final convolution
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AdaIN residual block.

        This method implements the forward pass of the residual block, which processes
        the input through the main residual path. The residual connection is implemented
        in the parent network architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in, length)
            s (torch.Tensor): Style vector of shape (batch_size, style_dim) for AdaIN

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, dim_out, length) or
                         (batch_size, dim_out, length*upsample_scale) if upsampling is
                             enabled

        Note:
            The actual residual connection (adding the input to the output) is handled
            by the parent network architecture, not within this block. This method only
            implements the main processing path through the block.
        """
        out = self._residual(x, s)
        return out


class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP) with optional dropout.

    This module implements a two-layer MLP with the following components:
    1. First fully connected layer with optional hidden dimension
    2. Activation function (default: GELU)
    3. Dropout layer for regularization
    4. Second fully connected layer with optional output dimension

    The network can be configured to have:
    - Same input and output dimensions
    - Different hidden dimension
    - Different output dimension
    - Custom activation function
    - Configurable dropout rate

    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features. If None, uses
            in_features
        out_features (int, optional): Number of output features. If None, uses
            in_features
        act_layer (nn.Module, optional): Activation function. Default: nn.GELU
        drop (float, optional): Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize Mlp with configuration parameters.

        Args:
            in_features: Number of input features
            hidden_features: Number of hidden features. If None, uses in_features.
                Default: None
            out_features: Number of output features. If None, uses in_features. Default:
                None
            act_layer: Activation function to use. Default: nn.GELU
            drop: Dropout probability. Default: 0.0
        """
        super().__init__()
        out_features = (
            out_features or in_features
        )  # Use input dimension if output not specified
        hidden_features = (
            hidden_features or in_features
        )  # Use input dimension if hidden not specified

        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Activation function
        self.act = act_layer()

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Dropout layer for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP network.

        This method implements the forward pass of the MLP, which processes the input
        through the following sequence:
        1. First fully connected layer
        2. Activation function
        3. Dropout regularization
        4. Second fully connected layer
        5. Final dropout regularization

        The dropout layers help prevent overfitting by randomly zeroing some elements
        during training, while having no effect during inference.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition input tensor into non-overlapping windows for window-based attention.

    This function takes a 1D input tensor and divides it into non-overlapping windows
    of specified size. This is a key operation in the Swin Transformer architecture
    that enables local attention computation within windows.

    Args:
        x: Input tensor of shape (B, W, C) where B is batch size, W is sequence
            length (width), and C is number of channels/features.
        window_size: Size of each window. Must divide W evenly.

    Returns:
        Reshaped tensor of shape (num_windows*B, window_size, C) where:
            - num_windows = W // window_size
            - Each window contains window_size consecutive elements
            - Windows are flattened across the batch dimension

    Example:
        If input shape is (4, 100, 64) and window_size=10:
        - Creates 10 windows per sequence (100/10)
        - Output shape will be (40, 10, 64) (4 batches * 10 windows)
    """
    b, w, c = x.shape

    # Reshape input to separate windows:
    # (B, W, C) -> (B, W//window_size, window_size, C)
    # This creates a new dimension for window_size
    x = x.view(b, w // window_size, window_size, c)

    # Permute and reshape to flatten windows across batch dimension:
    # (B, W//window_size, window_size, C) -> (B*W//window_size, window_size, C)
    # This makes it easier to process all windows in parallel
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, c)

    return windows


def window_reverse(windows: torch.Tensor, window_size: int, w: int) -> torch.Tensor:
    """Reverse the window partitioning operation to reconstruct the original input
        tensor.

    This function takes windowed features and reconstructs them back into the original
    sequence format. It is the inverse operation of window_partition and is used after
    processing windows in the Swin Transformer architecture.

    Args:
        windows: Windowed features of shape (num_windows*B, window_size, C) where
            num_windows = w/window_size, B is batch size, and C is number of channels.
        window_size: Size of each window. Must divide w evenly.
        w: Original sequence length (width) of the input tensor.

    Returns:
        Reconstructed tensor of shape (B, w, C) where:
            - B: Batch size
            - w: Original sequence length
            - C: Number of channels/features

    Example:
        If windows shape is (40, 10, 64), window_size=10, and w=100:
        - Input represents 4 batches (40/10 windows)
        - Output shape will be (4, 100, 64)
    """
    # num_windows = w/window_size, so B = windows.shape[0] / num_windows
    b = int(windows.shape[0] / (w / window_size))

    # Reshape windows back to original sequence format:
    # (num_windows*B, window_size, C) -> (B, w//window_size, window_size, C)
    # This separates windows back into their original positions
    x = windows.view(b, w // window_size, window_size, -1)

    # Permute and reshape to reconstruct original sequence:
    # (B, w//window_size, window_size, C) -> (B, w, C)
    # This combines windows back into continuous sequences
    x = x.permute(0, 1, 2, 3).contiguous().view(b, w, -1)

    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position
        bias.

    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5
            if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize the Window-based Multi-head Self-Attention (W-MSA) module.

        This module implements attention computation within local windows, which is a
            key
        component of the Swin Transformer architecture. It includes relative position
            bias
        to capture spatial relationships within windows.

        Args:
            dim (int): Number of input channels/features. This determines the dimension
                      of the query, key, and value vectors.
            window_size (int): Size of the local window for attention computation.
                             All tokens within this window can attend to each other.
            num_heads (int): Number of parallel attention heads. The input dimension
                           must be divisible by this number.
            qkv_bias (bool, optional): If True, adds learnable bias to query, key, and
                                     value projections. Default: True
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5
                                      if set. Default: None
            attn_drop (float, optional): Dropout ratio for attention weights.
                                       Default: 0.0
            proj_drop (float, optional): Dropout ratio for output projection.
                                       Default: 0.0
        """
        super().__init__()

        self.dim = dim  # Input dimension
        self.window_size = window_size  # Size of attention window
        self.num_heads = num_heads  # Number of attention heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5  # Scaling factor for attention scores

        # (2*window_size-1, num_heads); learnable position biases per relative position
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads)
        )

        # Generate relative position indices for all positions in the window
        # This creates a mapping from relative positions to indices in the bias table
        coords_w = torch.arange(self.window_size)  # [0, 1, ..., window_size-1]
        coords = torch.stack(
            torch.meshgrid([coords_w], indexing="ij")
        )  # [1, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [1, window_size]

        # Shape: [window_size, window_size, 1]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # Shift indices to be non-negative
        relative_coords[:, :, 0] += self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [window_size, window_size]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # Output projection
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Window-based Multi-head Self-Attention (W-MSA) module.

        This method implements the attention computation within local windows,
            including:
        1. Projecting input to query, key, and value vectors
        2. Computing attention scores with relative position bias
        3. Applying attention mask if provided
        4. Computing weighted sum of values
        5. Projecting the result back to original dimension

        Args:
            x (torch.Tensor): Input features of shape (num_windows*B, N, C) where:
                - num_windows: Number of windows
                - B: Batch size
                - N: Number of tokens in window (window_size)
                - C: Number of channels/features
            mask (torch.Tensor, optional): Attention mask of shape (num_windows, N, N)
                where N is window_size. Values should be 0 or -inf.
                Default: None

        Returns:
            torch.Tensor: Output features of shape (num_windows*B, N, C)
        """
        b_, n, c = x.shape  # B_ = num_windows*B

        # Project input to query, key, and value vectors
        # Shape: (num_windows*B, N, 3*C) -> (3, num_windows*B,
        # num_heads, N, C//num_heads)
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separate Q, K, V

        q = q * self.scale
        # Shape: (num_windows*B, num_heads, N, N)
        attn = q @ k.transpose(-2, -1)

        # Shape: (W, W, num_heads) -> (num_heads, window_size, window_size)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            n_w = mask.shape[0]  # number of windows
            # Reshape attention scores to separate windows
            # (num_windows*B, num_heads, N, N) -> (B, num_windows, num_heads, N, N)
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            # Reshape back to original form
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # Shape: (num_windows*B, num_heads, N, C//num_heads) -> (num_windows*B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)

        # Project to output dimension and apply dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        """Generate a string representation of the WindowAttention module's key
            parameters.

        This method is used by PyTorch's print() and str() functions to display
        important configuration parameters of the module. It helps in debugging
        and model inspection by showing the module's configuration at a glance.

        Returns:
            str: A formatted string containing the module's key parameters:
                - dim: Input/Output dimension
                - window_size: Size of attention window
                - num_heads: Number of attention heads
        """
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"num_heads={self.num_heads}"
        )


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5
            if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.Module,
        style_dim: int | None = 64,
    ) -> None:
        """Initialize a Swin Transformer Block with optional shifted window attention.

        This block implements a complete transformer block with window-based attention,
        which can operate in either regular or shifted window mode. It includes:
        1. Window-based multi-head self-attention (W-MSA)
        2. Shifted window-based multi-head self-attention (SW-MSA)
        3. Multi-layer perceptron (MLP)
        4. Layer normalization
        5. Residual connections

        Args:
            dim (int): Number of input channels/features
            input_resolution (int): Input sequence length
            num_heads (int): Number of attention heads
            window_size (int, optional): Size of attention window. Default: 7
            shift_size (int, optional): Size of window shift for SW-MSA. Default: 0
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim.
                Default: 4.0
            qkv_bias (bool, optional): If True, add learnable bias to QKV. Default: True
            qk_scale (float, optional): Override default QK scale. Default: None
            drop (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop (float, optional): Dropout rate for attention. Default: 0.0
            drop_path (Union[float, List[float]], optional): Stochastic depth rate.
                Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Adjust window and shift size if input is smaller than window
        if self.input_resolution <= self.window_size:
            # If input is smaller than window, use input size as window
            self.shift_size = 0
            self.window_size = self.input_resolution

        # Validate shift size
        assert 0 <= self.shift_size < self.window_size, (
            "shift_size must in 0-window_size"
        )

        if style_dim is not None:
            # Use AdaIN for style transfer
            self.norm1 = norm_layer(style_dim, dim)
            self.norm2 = norm_layer(style_dim, dim)
        else:
            # Use standard layer normalization
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if isinstance(drop_path, list):
            self.drop_path = (
                DropPath(drop_path[0]) if drop_path[0] > 0.0 else nn.Identity()
            )
        else:
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Generate attention mask for shifted window attention
        if self.shift_size > 0:
            w = self.input_resolution
            img_mask = torch.zeros((1, w, 1))  # [1, W, 1]

            # Define window slices for shifted attention
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            for cnt, w_slice in enumerate(w_slices):
                img_mask[:, w_slice, :] = cnt

            # Partition mask into windows
            mask_windows = window_partition(
                img_mask, self.window_size
            )  # [nW, window_size, 1]
            mask_windows = mask_windows.view(-1, self.window_size)  # [nW, window_size]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # Same-window positions 0, cross-window -inf for attention masking
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None

        # Register attention mask as buffer (not parameter)
        self.register_buffer("attn_mask", attn_mask)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Swin Transformer Block.

        This method implements the complete forward pass of the transformer block,
            including:
        1. Layer normalization
        2. Window-based attention (W-MSA) or shifted window attention (SW-MSA)
        3. Residual connection
        4. MLP processing
        5. Final residual connection

        The block can operate in two modes:
        - Regular mode (shift_size=0): Standard window-based attention
        - Shifted mode (shift_size>0): Attention with shifted windows for cross-window
            connections

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C) where:
                - B: Batch size
                - L: Sequence length (must equal input_resolution)
                - C: Number of channels/features
            s (torch.Tensor, optional): Style vector for AdaIN normalization.
                Required if using AdaIN, shape: (B, style_dim)
                Default: None

        Returns:
            torch.Tensor: Output tensor of same shape as input (B, L, C)

        Note:
            If using AdaIN (style_dim is not None), the style vector s must be provided.
            The input sequence length L must match the input_resolution specified in
                __init__.
        """
        w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == w, "input feature has wrong size"

        shortcut = x

        if s is not None:
            x = self.norm1(x.transpose(1, 2), s)
            x = x.transpose(1, 2)
        else:
            x = self.norm1(x.transpose(1, 2))
            x = x.transpose(1, 2)

        # Reshape for window-based processing
        x = x.view(b, w, c)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x

        # Partition windows for attention computation
        # Shape: (B, W, C) -> (num_windows*B, window_size, C)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size, c)
        # Shape: (num_windows*B, window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows back to sequence
        # Shape: (num_windows*B, window_size, C) -> (B, W, C)
        attn_windows = attn_windows.view(-1, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, w)

        # Reverse cyclic shift if using shifted window attention
        if self.shift_size > 0:
            # Shift the sequence back by shift_size positions
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x

        # Reshape back to original format
        x = x.view(b, w, c)

        # First residual connection
        x = shortcut + self.drop_path(x)

        if s is not None:
            x = x + self.drop_path(
                self.mlp(self.norm2(x.transpose(1, 2), s).transpose(1, 2))
            )
        else:
            x = x + self.drop_path(
                self.mlp(self.norm2(x.transpose(1, 2)).transpose(1, 2))
            )

        return x

    def extra_repr(self) -> str:
        """Return string representation of SwinTransformerBlock configuration.

        Returns:
            String containing key configuration parameters
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class PatchMerging(nn.Module):
    """Patch Merging Layer for Swin Transformer.

    This layer implements a downsampling operation that merges adjacent patches
    to reduce the sequence length while increasing the number of channels.
    It's used in the encoder part of the Swin Transformer to create a hierarchical
    feature representation.

    The merging process:
    1. Takes adjacent pairs of patches
    2. Concatenates their features
    3. Applies normalization
    4. Projects to a new feature dimension

    Args:
        input_resolution (int): Input sequence length
        dim (int): Number of input channels
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(
        self,
        input_resolution: int,
        dim: int,
        norm_layer: type = nn.LayerNorm,
        style_dim: int | None = 64,
    ) -> None:
        """Initialize the Patch Merging layer.

        Args:
            input_resolution (int): Input sequence length
            dim (int): Number of input channels
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN.
                Default: 64
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim

        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        # Uses AdaIN if style_dim is provided, otherwise standard normalization
        norm_style_dim = style_dim if style_dim is not None else 64
        self.norm = norm_layer(norm_style_dim, 2 * dim)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Patch Merging layer.

        This method implements the patch merging operation:
        1. Takes adjacent pairs of patches
        2. Concatenates their features
        3. Applies normalization
        4. Projects to new feature dimension

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C) where:
                - B: Batch size
                - L: Sequence length (must equal input_resolution)
                - C: Number of channels/features
            s (torch.Tensor): Style vector for AdaIN normalization.
                Shape: (B, style_dim)

        Returns:
            torch.Tensor: Output tensor of shape (B, L/2, 2*C) where:
                - L/2: Halved sequence length
                - 2*C: Doubled number of channels

        Note:
            The input sequence length L must be even and match input_resolution.
        """
        w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == w, "input feature has wrong size"
        assert w % 2 == 0, f"x size ({w}) are not even."

        # Reshape input for patch merging
        x = x.view(b, w, c)

        # Extract even and odd indexed patches
        x0 = x[:, 0::2, :]  # [B, W/2, C] - even indices
        x1 = x[:, 1::2, :]  # [B, W/2, C] - odd indices

        # Concatenate patches along feature dimension
        # Shape: [B, W/2, 2*C]
        x = torch.cat([x0, x1], -1)
        x = x.view(b, -1, 2 * c)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        """Generate a string representation of the PatchMerging layer's configuration.

        Returns:
            str: A formatted string containing the layer's key parameters:
                - input_resolution: Input sequence length
                - dim: Number of input channels
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class PatchExpand(nn.Module):
    """Patch Expansion Layer for Swin Transformer.

    This layer implements an upsampling operation that expands patches to increase
    the sequence length while decreasing the number of channels. It's used in the
    decoder part of the Swin Transformer to gradually increase spatial resolution.

    The expansion process:
    1. Projects input features to a lower dimension
    2. Rearranges features to increase sequence length
    3. Applies normalization
    4. Outputs features with doubled sequence length and halved channels

    Args:
        input_resolution (int): Input sequence length
        dim (int): Number of input channels
        dim_scale (int, optional): Scale factor for dimension reduction. Default: 2
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(
        self,
        input_resolution: int,
        dim: int,
        dim_scale: int = 2,
        norm_layer: type = nn.LayerNorm,
        style_dim: int | None = 64,
    ) -> None:
        """Initialize the Patch Expansion layer.

        Args:
            input_resolution (int): Input sequence length
            dim (int): Number of input channels
            dim_scale (int, optional): Scale factor for dimension reduction. Default: 2
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN.
                Default: 64
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim

        self.expand = (
            nn.Linear(dim, dim // dim_scale, bias=False)
            if dim_scale >= 2
            else nn.Identity()
        )
        norm_style_dim = style_dim if style_dim is not None else 64
        self.norm = norm_layer(norm_style_dim, dim // (dim_scale * 2))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Patch Expansion layer.

        This method implements the patch expansion operation:
        1. Projects input features to lower dimension
        2. Rearranges features to increase sequence length
        3. Applies normalization

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C) where:
                - B: Batch size
                - L: Sequence length (must equal input_resolution)
                - C: Number of channels/features
            s (torch.Tensor): Style vector for AdaIN normalization.
                Shape: (B, style_dim)

        Returns:
            torch.Tensor: Output tensor of shape (B, L*2, C/2) where:
                - L*2: Doubled sequence length
                - C/2: Halved number of channels

        Note:
            The input sequence length L must match input_resolution.
        """
        w = self.input_resolution
        x = self.expand(x)
        b, seq_len, c = x.shape
        assert seq_len == w, "input feature has wrong size"

        # Reshape input for patch expansion
        x = x.view(b, w, c)

        # Rearrange features to increase sequence length
        # Shape: [B, W, C] -> [B, W*2, C/2]
        x = rearrange(x, "b w (p1 c)-> b (w p1) c", p1=2, c=c // 2)
        x = x.view(b, -1, c // 2)

        x = self.norm(x.transpose(1, 2), s).transpose(1, 2)
        return x


class FinalPatchExpandX4(nn.Module):
    """Final Patch Expansion Layer for Swin Transformer with 4x upsampling.

    This layer implements the final upsampling operation in the Swin Transformer
        decoder,
    expanding patches by a factor of 4 to achieve the desired output resolution. It's
    specifically designed for the final stage of the decoder to match the input
        resolution.

    The expansion process:
    1. Projects input features to a new dimension
    2. Rearranges features to increase sequence length by 4x
    3. Applies normalization
    4. Outputs features with quadrupled sequence length

    Args:
        input_resolution (int): Input sequence length
        dim (int): Number of input channels
        dim_scale (int, optional): Scale factor for dimension reduction. Default: 4
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(
        self,
        input_resolution: int,
        dim: int,
        dim_scale: int = 4,
        norm_layer: type = nn.Module,
        style_dim: int | None = 64,
    ) -> None:
        """Initialize the Final Patch Expansion layer.

        Args:
            input_resolution (int): Input sequence length
            dim (int): Number of input channels
            dim_scale (int, optional): Scale factor for dimension reduction. Default: 4
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN.
                Default: 64
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim, bias=False)

        self.output_dim = dim // dim_scale
        self.norm = norm_layer(style_dim, self.output_dim)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Final Patch Expansion layer.

        This method implements the final patch expansion operation:
        1. Projects input features to new dimension
        2. Rearranges features to increase sequence length by 4x
        3. Applies normalization

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C) where:
                - B: Batch size
                - L: Sequence length (must equal input_resolution)
                - C: Number of channels/features
            s (torch.Tensor): Style vector for AdaIN normalization.
                Shape: (B, style_dim)

        Returns:
            torch.Tensor: Output tensor of shape (B, L*4, C/4) where:
                - L*4: Quadrupled sequence length
                - C/4: Quartered number of channels

        Note:
            The input sequence length L must match input_resolution.
            This layer is specifically designed for 4x upsampling in the final stage.
        """
        w = self.input_resolution
        x = self.expand(x)
        b, seq_len, c = x.shape
        assert seq_len == w, "input feature has wrong size"

        # Reshape input for patch expansion
        x = x.view(b, w, c)

        # Rearrange features to increase sequence length by 4x
        # Shape: [B, W, C] -> [B, W*4, C/4]
        x = rearrange(
            x, "b w (p1 c)-> b (w p1) c", p1=self.dim_scale, c=c // (self.dim_scale)
        )
        x = x.view(b, self.output_dim, -1)
        x = self.norm(x, s)

        return x


class BasicLayer(nn.Module):
    """Basic Swin Transformer Layer for one stage in the encoder.

    This layer implements a complete stage of the Swin Transformer encoder, consisting
        of:
    1. Multiple Swin Transformer blocks with alternating window attention
    2. Optional patch merging for downsampling
    3. Support for gradient checkpointing to save memory

    The layer processes features through a series of transformer blocks, where each
        block
    alternates between regular and shifted window attention. This allows for both local
    and cross-window feature interactions.

    Args:
        dim (int): Number of input channels
        input_resolution (int): Input sequence length
        depth (int): Number of transformer blocks in this layer
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default:
            4.0
        qkv_bias (bool, optional): If True, add learnable bias to QKV. Default: True
        qk_scale (float, optional): Override default QK scale. Default: None
        drop (float, optional): Dropout rate for MLP. Default: 0.0
        attn_drop (float, optional): Dropout rate for attention. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer. Default: None
        use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default:
            False
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type = nn.Module,
        downsample: type | None = None,
        use_checkpoint: bool = False,
        style_dim: int | None = 64,
    ) -> None:
        """Initialize the Basic Swin Transformer Layer.

        Args:
            dim (int): Number of input channels
            input_resolution (int): Input sequence length
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            window_size (int): Size of attention window
            mlp_ratio (float, optional): MLP expansion ratio. Default: 4.0
            qkv_bias (bool, optional): Whether to use bias in QKV projection. Default:
                True
            qk_scale (float, optional): Scale factor for attention scores. Default: None
            drop (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop (float, optional): Dropout rate for attention. Default: 0.0
            drop_path (Union[float, List[float]], optional): Stochastic depth rate.
                Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            downsample (nn.Module | None, optional): Downsample layer. Default: None
            use_checkpoint (bool, optional): Whether to use gradient checkpointing.
                Default: False
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN.
                Default: 64
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Each block alternates between regular and shifted window attention
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    # Alternate between regular and shifted window attention
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    style_dim=style_dim,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, style_dim=style_dim
            )
        else:
            self.downsample = None

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the Basic Swin Transformer Layer.

        This method processes input features through:
        1. A series of transformer blocks with alternating window attention
        2. Optional patch merging for downsampling

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C) where:
                - B: Batch size
                - L: Sequence length
                - C: Number of channels
            s (torch.Tensor, optional): Style vector for AdaIN normalization.
                Shape: (B, style_dim)
                Required if using AdaIN, otherwise ignored.

        Returns:
            torch.Tensor: Output tensor of shape:
                - (B, L, C) if no downsampling
                - (B, L/2, 2*C) if using patch merging
        """
        # Process through transformer blocks
        for blk in self.blocks:
            x = cast(
                "torch.Tensor",
                checkpoint.checkpoint(blk, x, s) if self.use_checkpoint else blk(x, s),
            )
        if self.downsample is not None:
            x = self.downsample(x, s)

        return x

    def extra_repr(self) -> str:
        """Generate a string representation of the layer's configuration.

        Returns:
            str: A formatted string containing the layer's key parameters:
                - dim: Number of channels
                - input_resolution: Input sequence length
                - depth: Number of transformer blocks
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"depth={self.depth}"
        )


class BasicLayerUp(nn.Module):
    """Basic Swin Transformer Layer for one stage in the decoder.

    This layer implements a complete stage of the Swin Transformer decoder, consisting
        of:
    1. Multiple Swin Transformer blocks with alternating window attention
    2. Optional patch expansion for upsampling
    3. Support for gradient checkpointing to save memory

    The layer processes features through a series of transformer blocks, where each
        block
    alternates between regular and shifted window attention. This allows for both local
    and cross-window feature interactions. After the transformer blocks, features can be
    upsampled using patch expansion.

    Args:
        dim (int): Number of input channels
        input_resolution (int): Input sequence length
        depth (int): Number of transformer blocks in this layer
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default:
            4.0
        qkv_bias (bool, optional): If True, add learnable bias to QKV. Default: True
        qk_scale (float, optional): Override default QK scale. Default: None
        drop (float, optional): Dropout rate for MLP. Default: 0.0
        attn_drop (float, optional): Dropout rate for attention. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
        upsample (nn.Module | None, optional): Upsample layer. Default: None
        use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default:
            False
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        dim_scale (int, optional): Scale factor for dimension adjustment. Default: 2
    """

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type = nn.Module,
        upsample: bool | None = None,
        use_checkpoint: bool = False,
        style_dim: int | None = 64,
        dim_scale: int = 2,
    ) -> None:
        """Initialize the Basic Swin Transformer Layer for decoder.

        Args:
            dim (int): Number of input channels
            input_resolution (int): Input sequence length
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            window_size (int): Size of attention window
            mlp_ratio (float, optional): MLP expansion ratio. Default: 4.0
            qkv_bias (bool, optional): Whether to use bias in QKV projection. Default:
                True
            qk_scale (float, optional): Scale factor for attention scores. Default: None
            drop (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop (float, optional): Dropout rate for attention. Default: 0.0
            drop_path (Union[float, List[float]], optional): Stochastic depth rate.
                Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            upsample (nn.Module | None, optional): Upsample layer. Default: None
            use_checkpoint (bool, optional): Whether to use gradient checkpointing.
                Default: False
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
            dim_scale (int, optional): Scale factor for dimension adjustment. Default: 2
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Each block alternates between regular and shifted window attention
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    # Alternate between regular and shifted window attention
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    style_dim=style_dim,
                )
                for i in range(depth)
            ]
        )
        if upsample is not None:
            self.upsample = PatchExpand(
                input_resolution,
                dim=dim,
                dim_scale=dim_scale,
                norm_layer=norm_layer,
                style_dim=style_dim,
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Basic Swin Transformer Layer in decoder.

        This method processes input features through:
        1. A series of transformer blocks with alternating window attention
        2. Optional patch expansion for upsampling

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C) where:
                - B: Batch size
                - L: Sequence length
                - C: Number of channels
            s (torch.Tensor): Style vector for AdaIN normalization.
                Shape: (B, style_dim)
                Required for AdaIN normalization.

        Returns:
            torch.Tensor: Output tensor of shape:
                - (B, L, C) if no upsampling
                - (B, L*2, C/2) if using patch expansion
        """
        # Process through transformer blocks
        for blk in self.blocks:
            x = cast(
                "torch.Tensor",
                checkpoint.checkpoint(blk, x, s) if self.use_checkpoint else blk(x, s),
            )
        if self.upsample is not None:
            x = self.upsample(x, s)

        return x


class PatchEmbed(nn.Module):
    """Patch Embedding Layer for Swin Transformer.

    This layer implements the initial embedding of input signals into patches, which is
    the first step in the Swin Transformer architecture. It divides the input signal
        into
    non-overlapping patches and projects them into a higher-dimensional embedding space.

    The embedding process:
    1. Divides input signal into patches of specified size
    2. Projects patches into embedding space using convolution
    3. Optionally applies normalization to the embeddings

    Args:
        img_size (int, optional): Input signal length. Default: 224
        patch_size (int, optional): Size of each patch. Default: 4
        in_chans (int, optional): Number of input channels. Default: 3
        embed_dim (int, optional): Dimension of patch embeddings. Default: 96
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type | None = None,
        style_dim: int = 64,
    ) -> None:
        """Initialize the Patch Embedding layer.

        Args:
            img_size (int, optional): Input signal length. Default: 224
            patch_size (int, optional): Size of each patch. Default: 4
            in_chans (int, optional): Number of input channels. Default: 3
            embed_dim (int, optional): Dimension of patch embeddings. Default: 96
            norm_layer (Optional[type], optional): Normalization layer. Default: None
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()

        img_size = img_size
        patch_size = patch_size
        patch_resolution = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(
            in_chans,  # Input channels
            embed_dim,  # Output embedding dimension
            kernel_size=patch_size,  # Patch size
            stride=patch_size,  # Non-overlapping patches
            padding=0,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Patch Embedding layer.

        This method implements the patch embedding process:
        1. Validates input dimensions
        2. Projects input into patch embeddings
        3. Optionally applies normalization
        4. Transposes to sequence format if normalized

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, W) where:
                - B: Batch size
                - C: Number of input channels
                - W: Signal length (must equal img_size)

        Returns:
            torch.Tensor: Output tensor of shape:
                - (B, W/patch_size, embed_dim) if normalized
                - (B, embed_dim, W/patch_size) if not normalized

        Note:
            The input signal length W must match img_size.
            The output sequence length is W/patch_size.
        """
        b, c, w = x.shape
        assert self.img_size == w, (
            f"Input signal length ({w}) doesn't match model ({self.img_size})."
        )

        # Shape: (B, C, W) -> (B, embed_dim, W/patch_size)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
            # Transpose to sequence format
            # Shape: (B, embed_dim, W/patch_size) -> (B, W/patch_size, embed_dim)
            x = x.transpose(1, 2)

        return x


class SwinTransformerSysAdaIn(nn.Module):
    """Swin Transformer with Adaptive Instance Normalization (AdaIN) for style transfer.

    This is a PyTorch implementation of the Swin Transformer architecture with added
    style transfer capabilities through AdaIN. The model combines the hierarchical
    feature extraction of Swin Transformer with style-based feature modulation.

    Architecture:
    1. Patch Embedding: Divides input into patches and projects to embedding space
    2. Encoder: Series of Swin Transformer layers with downsampling
    3. Bottleneck: Final Swin Transformer layer for feature transformation
    4. Decoder: Series of Swin Transformer layers with upsampling and skip connections
    5. Final Upsampling: Expands features to match input resolution

    Args:
        img_size (int, optional): Input signal length. Default: 224
        patch_size (int, optional): Size of patches. Default: 4
        in_chans (int, optional): Number of input channels. Default: 3
        num_classes (int, optional): Number of output classes. Default: 1000
        embed_dim (int, optional): Initial embedding dimension. Default: 96
        depths (list[int], optional): Number of transformer blocks in each encoder
            layer.
            Default: [2, 2, 2, 2]
        depths_decoder (list[int], optional): Number of transformer blocks in each
            decoder
            layer. Default: [1, 2, 2, 2]
        num_heads (list[int], optional): Number of attention heads in each layer.
            Default: [3, 6, 12, 24]
        window_size (int, optional): Size of attention window. Default: 7
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default:
            4.0
        qkv_bias (bool, optional): Whether to use bias in QKV projection. Default: True
        qk_scale (float, optional): Scale factor for attention scores. Default: None
        drop_rate (float, optional): Dropout rate for MLP. Default: 0.0
        attn_drop_rate (float, optional): Dropout rate for attention. Default: 0.0
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1
        norm_layer_encoder (nn.Module, optional): Normalization layer for encoder.
            Default: nn.Module
        norm_layer_decoder (nn.Module, optional): Normalization layer for decoder.
            Default: nn.Module
        ape (bool, optional): Whether to use absolute position embedding. Default: False
        patch_norm (bool, optional): Whether to normalize after patch embedding.
            Default: True
        use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default:
            False
        final_upsample (str, optional): Final upsampling strategy. Default:
            "expand_first"
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: list[int] | None = None,
        depths_decoder: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer_encoder: type = nn.Module,
        norm_layer_decoder: type = nn.Module,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        final_upsample: str = "expand_first",
        style_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        """Initialize the Swin Transformer with AdaIN.

        Args:
            img_size (int, optional): Input signal length. Default: 224
            patch_size (int, optional): Size of patches. Default: 4
            in_chans (int, optional): Number of input channels. Default: 3
            num_classes (int, optional): Number of output classes. Default: 1000
            embed_dim (int, optional): Initial embedding dimension. Default: 96
            depths (list[int], optional): Number of transformer blocks in each encoder
                layer. Default: [2, 2, 2, 2]
            depths_decoder (list[int], optional): Number of transformer blocks in each
                decoder layer. Default: [1, 2, 2, 2]
            num_heads (list[int], optional): Number of attention heads in each layer.
                Default: [3, 6, 12, 24]
            window_size (int, optional): Size of attention window. Default: 7
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim.
                Default: 4.0
            qkv_bias (bool, optional): Whether to use bias in QKV projection. Default:
                True
            qk_scale (float, optional): Scale factor for attention scores. Default: None
            drop_rate (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop_rate (float, optional): Dropout rate for attention. Default: 0.0
            drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1
            norm_layer_encoder (nn.Module, optional): Normalization layer for encoder.
                Default: nn.Module
            norm_layer_decoder (nn.Module, optional): Normalization layer for decoder.
                Default: nn.Module
            ape (bool, optional): Whether to use absolute position embedding. Default:
                False
            patch_norm (bool, optional): Whether to normalize after patch embedding.
                Default: True
            use_checkpoint (bool, optional): Whether to use gradient checkpointing.
                Default: False
            final_upsample (str, optional): Final upsampling strategy. Default:
                "expand_first"
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()

        # Handle mutable default arguments
        depths = (
            [2, 2, 2, 2] if depths is None else list(depths)
        )  # Copy to avoid mutation
        if depths_decoder is None:
            depths_decoder = [1, 2, 2, 2]
        else:
            depths_decoder = list(depths_decoder)
        num_heads = [3, 6, 12, 24] if num_heads is None else list(num_heads)

        # Log configuration for debugging
        logger.debug(
            "SwinTransformerSys expand initial - depths: %s, depths_decoder: %s, "
            "drop_path_rate: %s, num_classes: %s",
            depths,
            depths_decoder,
            drop_path_rate,
            num_classes,
        )

        self.style_dim = style_dim
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_layers_decoder = len(depths_decoder)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer_encoder if self.patch_norm else None,
            style_dim=style_dim,
        )
        num_patches = self.patch_embed.num_patches
        patch_resolution = self.patch_embed.patch_resolution
        self.patch_resolution = patch_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Generate stochastic depth rates
        dpr: list[float] = [
            float(x.item()) for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path_slice = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])]
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=patch_resolution // (2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=(
                    drop_path_slice if len(drop_path_slice) > 1 else drop_path_slice[0]
                ),
                norm_layer=norm_layer_encoder,  # norm_layer is a class, not an instance
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                style_dim=None,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers_decoder):
            drop_path_slice = dpr[
                sum(depths[: (self.num_layers_decoder - 1 - i_layer)]) : sum(
                    depths[: (self.num_layers_decoder - 1 - i_layer) + 1]
                )
            ]
            if i_layer == 0:
                # First decoder layer with special configuration
                layer_up = BasicLayerUp(
                    dim=int(embed_dim * 2 ** (self.num_layers_decoder - 1 - i_layer)),
                    input_resolution=(
                        patch_resolution
                        // (2 ** (self.num_layers_decoder - 1 - i_layer))
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers_decoder - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_slice
                        if len(drop_path_slice) > 1
                        else drop_path_slice[0]
                    ),
                    norm_layer=norm_layer_decoder,  # class, not instance
                    upsample=(
                        True if (i_layer < self.num_layers_decoder - 1) else None
                    ),
                    use_checkpoint=use_checkpoint,
                    style_dim=style_dim,
                    dim_scale=1,
                )
            else:
                # Standard decoder layers
                layer_up = BasicLayerUp(
                    dim=(
                        2
                        * int(embed_dim * 2 ** (self.num_layers_decoder - 1 - i_layer))
                    ),
                    input_resolution=(
                        patch_resolution
                        // (2 ** (self.num_layers_decoder - 1 - i_layer))
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers_decoder - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_slice
                        if len(drop_path_slice) > 1
                        else drop_path_slice[0]
                    ),
                    norm_layer=norm_layer_decoder,
                    upsample=True if (i_layer < self.num_layers_decoder - 1) else None,
                    use_checkpoint=use_checkpoint,
                    style_dim=style_dim,
                )
            self.layers_up.append(layer_up)
        self.norm = norm_layer_encoder(self.num_features)
        if len(self.layers_up) > 1:
            self.norm_up = norm_layer_decoder(self.style_dim, self.embed_dim * 2)
        else:
            self.norm_up = norm_layer_decoder(self.style_dim, self.embed_dim)
        if self.final_upsample == "expand_first":
            logger.debug("Final upsample: expand_first")
            if len(self.layers_up) > 1:
                self.up = FinalPatchExpandX4(
                    input_resolution=img_size // patch_size,
                    dim_scale=patch_size,
                    dim=embed_dim * 2,
                    norm_layer=AdaIN,
                    style_dim=style_dim,
                )
            else:
                self.up = FinalPatchExpandX4(
                    input_resolution=img_size // patch_size,
                    dim_scale=patch_size,
                    dim=embed_dim,
                    norm_layer=AdaIN,
                    style_dim=style_dim,
                )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize the weights of the model.

        This method initializes the weights of linear layers and normalization layers
        using appropriate initialization strategies.

        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore()
    def no_weight_decay(self) -> set[str]:
        """Specify parameters that should not have weight decay applied.

        Returns:
            Set[str]: Set of parameter names that should not have weight decay
        """
        return {"absolute_pos_embed"}

    @torch.jit.ignore()
    def no_weight_decay_keywords(self) -> set[str]:
        """Specify parameter keywords that should not have weight decay applied.

        Returns:
            Set[str]: Set of parameter keywords that should not have weight decay
        """
        return {"relative_position_bias_table"}

    def forward_features(
        self,
        x: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the encoder and bottleneck.

        This method processes input features through:
        1. Patch embedding
        2. Position embedding (if enabled)
        3. Encoder layers with downsampling
        4. Final normalization

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, W)
            s (torch.Tensor, optional): Style vector for AdaIN. Default: None

        Return:
            tuple: (x, x_downsample) where:
                - x: Bottleneck features
                - x_downsample: List of features from each encoder layer for skip
                    connections
        """
        # Initial patch embedding
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample: list[torch.Tensor] = []

        # Process through encoder layers
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                x_downsample.insert(0, x)
            x = layer(x)
            idx += 1

        # Final normalization
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        return x, x_downsample

    def forward_up_features(
        self,
        x: torch.Tensor,
        x_downsample: list[torch.Tensor],
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the decoder with skip connections.

        This method processes features through:
        1. Decoder layers with upsampling
        2. Skip connections from encoder
        3. Final normalization

        Args:
            x (torch.Tensor): Bottleneck features
            x_downsample (list): List of features from encoder layers
            s (torch.Tensor): Style vector for AdaIN

        Returns:
            torch.Tensor: Decoded features
        """
        # Process through decoder layers
        for inx, layer_up in enumerate(self.layers_up):
            x = layer_up(x, s)
            if inx != len(self.layers_up) - 1:
                x = torch.cat([x, x_downsample[inx]], -1)

        # Final normalization
        x = self.norm_up(x.transpose(1, 2), s).transpose(1, 2)

        return x

    def up_x4(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Perform final upsampling to match input resolution.

        This method performs the final upsampling operation to match the input
            resolution,
        using the specified upsampling strategy.

        Args:
            x (torch.Tensor): Features from decoder
            s (torch.Tensor): Style vector for AdaIN

        Returns:
            torch.Tensor: Upsampled features
        """
        # Validate input dimensions
        w = self.patch_resolution
        b, seq_len, c = x.shape
        assert seq_len == w, "input features has wrong size"
        if self.final_upsample == "expand_first":
            x = self.up(x, s)

        return x

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Complete forward pass through the network.

        This method implements the full forward pass through:
        1. Encoder and bottleneck
        2. Decoder with skip connections
        3. Final upsampling

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, W)
            s (torch.Tensor): Style vector for AdaIN

        Returns:
            torch.Tensor: Output tensor of shape (B, C, W)
        """
        # Process through encoder and bottleneck
        x, x_downsample = self.forward_features(x)

        # Process through decoder with skip connections
        x = self.forward_up_features(x, x_downsample, s)

        # Final upsampling
        x = self.up_x4(x, s)

        return x


# =====================
# 4. Refinement Model (BPModel and related classes)
# =====================


class MlpBP(nn.Module):
    """Multi-layer perceptron (MLP) for blood pressure prediction.

    This class implements a two-layer MLP with dropout and activation functions,
    specifically designed for blood pressure prediction tasks. The network can be
    configured with different input, hidden, and output dimensions.

    Architecture:
        Input -> Dropout -> Linear1 -> Activation -> Dropout -> Linear2 -> Output

    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features
        out_features (int): Number of output features
        act_layer (nn.Module, optional): Activation function to use. Defaults to nn.GELU
        drop (float, optional): Dropout probability. Defaults to 0.0

    Example:
        >>> model = MlpBP(in_features=512, hidden_features=256, out_features=1)
        >>> x = torch.randn(32, 512)  # batch_size=32, features=512
        >>> output = model(x)  # shape: (32, 1)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize MlpBP with configuration parameters.

        Args:
            in_features: Number of input features
            hidden_features: Number of hidden features
            out_features: Number of output features
            act_layer: Activation function to use. Default: nn.GELU
            drop: Dropout probability. Default: 0.0
        """
        super().__init__()

        self.in_features = in_features

        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Activation function (default: GELU)
        self.act = act_layer()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout layer for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class WaveformEncoder(nn.Module):
    """Encoder for processing waveform data (ECG/PPG signals).

    This class implements a waveform encoder using the PatchTSMixer architecture,
    which is specifically designed for processing time series data like ECG and PPG
        signals.
    The encoder can be configured with different context lengths and model dimensions.

    Architecture:
        Uses PatchTSMixer model with configurable:
        - Context length (sequence length)
        - Number of input channels
        - Model dimension
        - Number of layers
        - Expansion factor
        - Patch length and stride

    Args:
        model_name (str, optional): Name of the model. Defaults to None
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True
        trainable (bool, optional): Whether to make model parameters trainable. Defaults
            to True
        context_length (int, optional): Length of input sequences. Defaults to 1280
        num_input_channels (int, optional): Number of input channels. Defaults to 1
        d_model (int, optional): Dimension of the model. Defaults to 64
        num_layers (int, optional): Number of transformer layers. Defaults to 15
        expansion_factor (int, optional): Expansion factor for feed-forward networks.
            Defaults to 5
        patch_length (int, optional): Length of each patch for PatchTSMixer. Defaults
            to 4 (paper Section 6.2.2)
        patch_stride (int, optional): Stride between consecutive patches. Defaults to 4
            (paper-aligned)

    Example:
        >>> encoder = WaveformEncoder(context_length=1024, d_model=64)
        >>> x = torch.randn(32, 1, 1024)  # batch_size=32, channels=1, seq_len=1024
        >>> output = encoder(x)  # Returns PatchTSMixer output
    """

    def __init__(
        self,
        model_name: str | None = None,
        pretrained: bool = True,
        trainable: bool = True,
        context_length: int = 1280,
        num_input_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 15,
        expansion_factor: int = 5,
        patch_length: int = 4,
        patch_stride: int = 4,
    ) -> None:
        """Initialize WaveformEncoder with configuration parameters.

        Args:
            model_name: Optional model name for pretrained weights. Default: None
            pretrained: Whether to use pretrained weights. Default: True
            trainable: Whether model parameters are trainable. Default: True
            context_length: Length of input sequences. Default: 1280
            num_input_channels: Number of input channels. Default: 1
            d_model: Model dimension. Default: 64
            num_layers: Number of transformer layers. Default: 15
            expansion_factor: Expansion factor for feed-forward networks. Default: 5
            patch_length: Length of each patch. Default: 4
            patch_stride: Stride between consecutive patches. Default: 4
        """
        super().__init__()
        self.model = PatchTSMixerModel(
            PatchTSMixerConfig(
                context_length=context_length,  # Length of input sequences
                num_input_channels=num_input_channels,  # Number of input channels
                d_model=d_model,  # Model dimension
                num_layers=num_layers,  # Number of transformer layers
                expansion_factor=expansion_factor,  # FFN expansion
                patch_length=patch_length,  # Length of each patch
                patch_stride=patch_stride,  # Stride between consecutive patches
            )
        )

        # Configure parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the waveform encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_channels,
                context_length) [B, C, T]

        Returns:
            torch.Tensor: Output from the PatchTSMixer model

        Note:
            PatchTSMixer expects [B, T, C] format (time-first), so input is transposed
            from [B, C, T] to [B, T, C]
        """
        logger.debug(f"[WaveformEncoder.forward] Input x shape: {x.shape}")
        logger.debug(
            f"[WaveformEncoder.forward] config: context_length="
            f"{self.model.config.context_length}, num_input_channels="
            f"{self.model.config.num_input_channels}"
        )

        # Validate input shape matches model configuration
        expected_seq_len = self.model.config.context_length
        expected_channels = self.model.config.num_input_channels
        actual_seq_len = x.shape[-1] if x.dim() >= 2 else 1
        actual_channels = x.shape[1] if x.dim() >= 2 else x.shape[0]

        # Enforce sequence length mismatch with ValueError
        if actual_seq_len != expected_seq_len:
            raise ValueError(
                f"[WaveformEncoder.forward] Sequence length mismatch: "
                f"Expected {expected_seq_len}, got {actual_seq_len}. "
                f"Input shape: {x.shape}, context_length: {expected_seq_len}"
            )
        # Channel mismatch: keep as warning (can also enforce if desired)
        if actual_channels != expected_channels and x.dim() >= 2:
            logger.warning(
                f"[WaveformEncoder.forward] Channel mismatch: Expected "
                f"{expected_channels}, got {actual_channels}. "
                f"Full input shape: {x.shape}"
            )

        # PatchTSMixer expects [B, T, C] format (time-first), but we receive [B, C, T]
        # (channel-first). Transpose from [B, C, T] to [B, T, C]
        x_transposed = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        logger.debug(
            f"[WaveformEncoder.forward] Transposed x shape: {x_transposed.shape}"
        )

        return self.model(x_transposed).last_hidden_state


class TextEncoder(nn.Module):
    """Encoder for processing text data using DistilBERT.

    This class implements a text encoder using the DistilBERT architecture,
    which is a distilled version of BERT that maintains most of the performance
    while being more efficient. The encoder processes text input and extracts
    contextual embeddings.

    Architecture:
        Uses DistilBERT model with configurable:
        - Model name/version
        - Pretrained weights
        - Parameter trainability

    Args:
        model_name (str, optional): Name of the pretrained model to use.
            Defaults to "distilbert-base-uncased"
        pretrained (bool, optional): Whether to use pretrained weights.
            Defaults to True
        trainable (bool, optional): Whether to make model parameters trainable.
            Defaults to True

    Example:
        >>> encoder = TextEncoder(pretrained=True)
        >>> input_ids = torch.randint(0, 1000, (32, 128))  # batch_size=32, seq_len=128
        >>> attention_mask = torch.ones(32, 128)
        >>> output = encoder(input_ids, attention_mask)  # shape: (32, 768)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pretrained: bool = True,
        trainable: bool = True,
    ) -> None:
        """Initialize TextEncoder with configuration parameters.

        Args:
            model_name: Name of the pretrained DistilBERT model. Default:
                "distilbert-base-uncased"
            pretrained: Whether to use pretrained weights. Default: True
            trainable: Whether model parameters are trainable. Default: True
        """
        super().__init__()
        if pretrained:
            self.model = cast(
                "DistilBertModel", DistilBertModel.from_pretrained(model_name)
            )
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        # Index of the target token to extract from the output
        # Using [CLS] token (index 0) as the default representation
        self.target_token_idx = 0

        # Configure parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the text encoder.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size,
                sequence_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size,
                sequence_length)
                where 1 indicates tokens to attend to and 0 indicates padding tokens

        Returns:
            torch.Tensor: Extracted embeddings from the target token position
                of shape (batch_size, hidden_size)
        """
        # Process input through DistilBERT model
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token embeddings: aggregates sequence information for
        # regression tasks
        return output.last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    """Projection head for transforming embeddings to a common space.

    This class implements a projection head that transforms input embeddings into
    a common representation space. It uses a combination of linear layers,
    activation functions, dropout, and layer normalization to create robust
    embeddings suitable for downstream tasks.

    Architecture:
        Input -> Linear -> GELU -> Linear -> Dropout -> Residual -> LayerNorm -> Output

    The architecture includes:
        - Initial linear projection
        - GELU activation
        - Second linear transformation
        - Dropout for regularization
        - Residual connection
        - Layer normalization

    Args:
        embedding_dim (int): Dimension of input embeddings
        projection_dim (int, optional): Dimension of projected embeddings.
            Defaults to 256
        dropout (float, optional): Dropout probability for regularization.
            Defaults to 0.1

    Example:
        >>> projection = ProjectionHead(embedding_dim=768, projection_dim=256)
        >>> x = torch.randn(32, 768)  # batch_size=32, embedding_dim=768
        >>> output = projection(x)  # shape: (32, 256)
    """

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ProjectionHead with configuration parameters.

        Args:
            embedding_dim: Input embedding dimension
            projection_dim: Output projection dimension. Default: 256
            dropout: Dropout probability. Default: 0.1
        """
        super().__init__()
        # Initial linear projection layer
        self.projection = nn.Linear(embedding_dim, projection_dim)
        # GELU activation function
        self.gelu = nn.GELU()
        # Second linear transformation
        self.fc = nn.Linear(projection_dim, projection_dim)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection head.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, embedding_dim)

        Returns:
            torch.Tensor: Projected embeddings of shape (batch_size, projection_dim)
                with residual connection and layer normalization applied
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


# =====================
# VitalEncoder Component for BPModel
# =====================


@dataclass
class VitalEncoderConfig:
    """Configuration for a single vital encoder used by `BPModel`.

    This dataclass defines the parameters for constructing a `VitalEncoder` that
    encapsulates waveform encoding, projection, optional fusion, and BP heads
    for a single vital sign (e.g., PPG, ECG).

    Note: `input_length` maps to `context_length` in PatchTSMixerConfig.
    `image_embedding` is calculated dynamically from patch geometry.
    """

    _target_: str = "src.model.mdvisco.VitalEncoder"
    vital: str = MISSING
    input_length: int = MISSING  # Maps to context_length in PatchTSMixerConfig
    projection_dim: int = 512
    dropout: float = 0.1
    d_model: int = 64
    pi: bool = True
    # PatchTSMixer hyperparameters (mapped to PatchTSMixerConfig)
    num_input_channels: int = 1  # Maps to PatchTSMixerConfig.num_input_channels
    num_layers: int = 15  # Maps to PatchTSMixerConfig.num_layers
    expansion_factor: int = 5  # Maps to PatchTSMixerConfig.expansion_factor
    # Maps to PatchTSMixerConfig.patch_length (paper Section 6.2.2: patch length 4)
    patch_length: int = 4
    patch_stride: int = (
        4  # Maps to PatchTSMixerConfig.patch_stride (paper-aligned; no overlap)
    )


class VitalEncoder(nn.Module):
    """Per-vital processing pipeline for BP estimation.

    Encapsulates:
    - WaveformEncoder for raw waveform -> features
    - ProjectionHead to map features to a common embedding
    - Fusion encoder for PI mode or BP-only encoder for non-PI
    - SBP/DBP heads for final prediction
    """

    @staticmethod
    def calculate_image_embedding(
        context_length: int,
        d_model: int,
        patch_length: int,
        patch_stride: int,
        num_input_channels: int = 1,
    ) -> int:
        """Calculate image_embedding size from patch geometry.

        Computes the number of patches that can be extracted from a sequence of
        given length using the patch geometry parameters, then multiplies by d_model
        and num_input_channels to get the total embedding dimension.

        Formula: num_patches = floor((context_length - patch_length) / patch_stride) + 1
                 image_embedding = num_patches * d_model * num_input_channels

        Args:
            context_length: Length of the input sequence (maps to PatchTSMixer
                context_length)
            d_model: Model dimension (embedding dimension per patch)
            patch_length: Length of each patch for PatchTSMixer
            patch_stride: Stride between consecutive patches
            num_input_channels: Number of input channels (defaults to 1 for
                single-channel case)

        Returns:
            int: Calculated image_embedding size (num_patches * d_model *
                num_input_channels)

        Raises:
            ValueError: If context_length < patch_length (invalid geometry)
            ValueError: If calculated num_patches <= 0 (no valid patches)
        """
        # Validate patch geometry
        if context_length < patch_length:
            logger.warning(
                f"Invalid patch geometry: context_length ({context_length}) < "
                f"patch_length ({patch_length}). Cannot extract any patches."
            )
            raise ValueError(
                f"context_length ({context_length}) must be >= patch_length "
                f"({patch_length})"
            )

        num_patches = math.floor((context_length - patch_length) / patch_stride) + 1

        # Validate that we have at least one patch
        if num_patches <= 0:
            logger.warning(
                f"Invalid patch geometry: calculated num_patches ({num_patches}) <= 0. "
                f"context_length={context_length}, patch_length={patch_length}, "
                f"patch_stride={patch_stride}"
            )
            raise ValueError(
                f"Calculated num_patches ({num_patches}) must be > 0. "
                f"Check patch geometry parameters."
            )

        return num_patches * d_model * num_input_channels

    def __init__(
        self,
        vital: str,
        input_length: int,
        projection_dim: int = 512,
        dropout: float = 0.1,
        d_model: int = 64,
        pi: bool = True,
        num_input_channels: int = 1,
        num_layers: int = 15,
        expansion_factor: int = 5,
        patch_length: int = 4,
        patch_stride: int = 4,
    ) -> None:
        """Initialize VitalEncoder.

        Args:
            vital: Name of the vital sign (e.g., 'ppg', 'ecg')
            input_length: Length of input waveform sequences (maps to context_length)
            projection_dim: Dimension of the projection head output
            dropout: Dropout probability
            d_model: Model dimension (embedding dimension per patch)
            pi: Whether to use patient information fusion mode
            num_input_channels: Number of input channels for primary waveform encoder
            num_layers: Number of transformer layers in PatchTSMixer
            expansion_factor: Feed-forward expansion factor in PatchTSMixer
            patch_length: Length of each patch for PatchTSMixer. Default: 4 (paper
                Section 6.2.2)
            patch_stride: Stride between consecutive patches. Default: 4 (paper-aligned)

        Note:
            image_embedding is calculated dynamically from patch geometry using
            the formula: num_patches = floor((input_length - patch_length) /
                patch_stride) + 1
            then image_embedding = num_patches * d_model
        """
        super().__init__()

        if not isinstance(vital, str) or not vital:
            raise ValueError("vital must be a non-empty string")
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_input_channels <= 0:
            raise ValueError("num_input_channels must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        if patch_length <= 0:
            raise ValueError("patch_length must be positive")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")

        self.vital = vital
        self.pi = pi

        # Derive embedding size from patch geometry so encoder config stays consistent
        image_embedding = self.calculate_image_embedding(
            context_length=input_length,
            d_model=d_model,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )
        logger.debug(
            f"VitalEncoder({vital}): Calculated image_embedding={image_embedding} "
            f"from input_length={input_length}, d_model={d_model}, "
            f"patch_length={patch_length}, patch_stride={patch_stride}"
        )

        # Waveform encoder for raw vital waveform [B, 1, T]
        self.waveform_encoder = WaveformEncoder(
            context_length=input_length,
            num_input_channels=num_input_channels,
            d_model=d_model,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )

        # Project flattened features -> [B, projection_dim]
        self.projection = ProjectionHead(
            embedding_dim=image_embedding,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        # Encoders for fusion and BP-only paths on [B, projection_dim, C]
        # text_encoder: 2 ch (concat embeddings), bp_encoder: 1 ch (vital only)
        self.text_encoder = WaveformEncoder(
            context_length=projection_dim,
            num_input_channels=2,
            d_model=d_model,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )
        self.bp_encoder = WaveformEncoder(
            context_length=projection_dim,
            num_input_channels=1,
            d_model=d_model,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )

        # Match encoder channel count used in forward() for PI vs non-PI paths
        if self.pi:
            # PI mode: text_encoder uses 2 channels (vital + text embeddings)
            image_embedding_text = self.calculate_image_embedding(
                context_length=projection_dim,
                d_model=d_model,
                patch_length=patch_length,
                patch_stride=patch_stride,
                num_input_channels=2,  # text_encoder: two channels
            )
            in_features = image_embedding_text
            logger.debug(
                f"VitalEncoder({vital}): PI mode - in_features={in_features} "
                "for text_encoder (2 channels)"
            )
        else:
            # Non-PI mode: bp_encoder uses 1 channel (vital only)
            image_embedding_bp = self.calculate_image_embedding(
                context_length=projection_dim,
                d_model=d_model,
                patch_length=patch_length,
                patch_stride=patch_stride,
                num_input_channels=1,
            )
            in_features = image_embedding_bp
            logger.debug(
                f"VitalEncoder({vital}): Non-PI mode - calculated "
                f"in_features={in_features} "
                f"for bp_encoder (1 channel)"
            )

        # BP heads
        self.sbp_head = MlpBP(
            in_features=in_features,
            hidden_features=in_features // 2,
            out_features=1,
            drop=0.2,
        )
        self.dbp_head = MlpBP(
            in_features=in_features,
            hidden_features=in_features // 2,
            out_features=1,
            drop=0.2,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward vital pipeline.

        Args:
            waveform: [B, 1, T] vital waveform
            text_embedding: [B, 1, projection_dim] or None

        Returns:
            Tuple of (sbp [B,1], dbp [B,1], embedding [B, projection_dim, 1])
        """
        logger.debug(
            f"[VitalEncoder.forward] vital={self.vital}, "
            f"waveform shape: {waveform.shape}"
        )
        if text_embedding is not None:
            logger.debug(
                f"[VitalEncoder.forward] text_embedding: {text_embedding.shape}"
            )

        # Waveform -> features
        features = self.waveform_encoder(waveform)
        # Flatten and project -> [B, projection_dim]
        embedding_vec = self.projection(features.view(features.shape[0], -1))
        # Prepare two views: encoder-ready and artifact-preserving
        embedding_for_encoder = embedding_vec.unsqueeze(1)  # [B, 1, projection_dim]
        embedding_for_artifact = embedding_vec.unsqueeze(-1)  # [B, projection_dim, 1]

        # Explicit check for incompatible combination: pi=True but text_embedding=None
        if self.pi and text_embedding is None:
            raise ValueError(
                "text_embedding must be provided when pi is enabled. "
                "Either pass a valid text embedding tensor or set "
                "pi=False in the corresponding VitalEncoderConfig."
            )

        if self.pi and text_embedding is not None:
            # Concatenate along encoder channel dimension -> [B, 2, projection_dim]
            combined = torch.cat((embedding_for_encoder, text_embedding), dim=1)
            final = self.text_encoder(combined).view(combined.shape[0], -1)
        else:
            final = self.bp_encoder(embedding_for_encoder).view(
                embedding_for_encoder.shape[0], -1
            )

        # Runtime assertion to verify final tensor matches expected in_features
        actual_features = final.numel() // final.shape[0] if final.shape[0] > 0 else 0
        assert actual_features == self.sbp_head.in_features, (
            f"Shape mismatch: final tensor has {actual_features} features per sample, "
            f"but BP heads expect {self.sbp_head.in_features}. "
            f"final.shape={final.shape}"
        )

        sbp = self.sbp_head(final)
        dbp = self.dbp_head(final)
        return sbp, dbp, embedding_for_artifact


@dataclass
class TextEncoderPipelineConfig:
    """Configuration for the optional text encoder pipeline used for PI fusion."""

    _target_: str = "src.model.mdvisco.TextEncoderPipeline"
    text_embedding: int = 768
    projection_dim: int = 512
    dropout: float = 0.1


class TextEncoderPipeline(nn.Module):
    """Encapsulates text encoding and projection for patient information processing."""

    def __init__(
        self,
        text_embedding: int = 768,
        projection_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize TextEncoderPipeline with configuration parameters.

        Args:
            text_embedding: Dimension of text embeddings. Default: 768
            projection_dim: Dimension of projected embeddings. Default: 512
            dropout: Dropout probability. Default: 0.1
        """
        super().__init__()
        if text_embedding <= 0:
            raise ValueError("text_embedding must be positive")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding,
            projection_dim=projection_dim,
            dropout=dropout,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode text input and project to embedding space.

        Args:
            input_ids: Token IDs from text tokenizer.
            attention_mask: Attention mask for tokenized input.

        Returns:
            Projected text embeddings.
        """
        features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(features).unsqueeze(-1)


# =====================
# End of new components
# =====================


@dataclass
class BPModelConfig(SingleStageModelConfig):
    """Blood Pressure Model Configuration - Pre-configured component architecture.

    Vital signs are now pre-configured via `vital_encoders` instead of being created
        lazily at
    runtime. This replaces the old dynamic initialization pattern for better DDP
        compatibility.

    All hyperparameters (text_embedding, projection_dim, dropout, d_model) are now
        configured
    entirely through `vital_encoders` and `text_encoder_pipeline` components, with no
        model-level
    hyperparameters for these fields.
    """

    _target_: str = "src.model.mdvisco.BPModel"
    supports_multi_directional: bool = False
    model_name: str = "BPModel"

    # Pre-instantiated components (Hydra-configured)
    vital_encoders: dict[str, VitalEncoderConfig] = MISSING
    """Dictionary of VitalEncoder instances keyed by vital name (e.g., 'ppg', 'ecg').

    Pre-instantiated by Hydra from YAML config.
    """

    text_encoder_pipeline: TextEncoderPipelineConfig | None = None
    """Optional single TextEncoderPipeline config instance.

    If None, pi=False; if provided, pi=True. Pre-instantiated by Hydra from YAML config
        as a single
    nn.Module.
    """

    wcl_age_threshold: float | None = 0.0235
    wcl: bool = True
    normalized_bp: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters after dataclass initialization."""
        if self.input_length is None or self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if self.wcl_age_threshold is not None and self.wcl_age_threshold < 0:
            raise ValueError("wcl_age_threshold must be non-negative if provided")
        if self.vital_encoders is MISSING or not isinstance(self.vital_encoders, dict):
            raise ValueError("vital_encoders must be provided as a dict of components")
            # text_encoder_pipeline validation removed - Hydra will handle instantiation
            # The actual validation happens in BPModel.__init__ where it must be an
            # nn.Module or None


class BPModel(SingleStageModel):
    """Unified blood pressure estimation model.

    This class implements a comprehensive blood pressure estimation model that can
    process both waveform data (ECG/PPG) and patient information to predict blood
    pressure values. The model can operate in two modes:
    1. With patient information (PI): Uses both waveform and text data
    2. Without patient information: Uses only waveform data

    Architecture Components:
        - Waveform Encoders: Process ECG and PPG signals
        - Text Encoder: Process patient information (when PI is enabled)
        - Projection Heads: Transform embeddings to common space
        - BP Prediction Heads: Predict SBP and DBP values
        - Loss Functions: L1 loss and weighted contrastive loss (WCL)

    Output Contract:
        - Forward returns a dictionary with keys 'predictions' [B,2 SBP/DBP],
          'y_pred_sbp', and 'y_pred_dbp'
        - Optional keys include '{vital}_embeddings', '{vital}_sbp', '{vital}_dbp', and
          'text_embeddings'

    The model uses a combination of:
        - PatchTSMixer for waveform processing
        - DistilBERT for text processing
        - Multi-layer perceptrons for BP prediction
        - Weighted contrastive learning for better representations

    Note:
        The input length is inherited from SingleStageModel via `input_length`.

    Args:
        vital_encoders: Dictionary of pre-instantiated VitalEncoder modules keyed by
            vital
            name
        text_encoder_pipeline: Optional pre-instantiated TextEncoderPipeline module. If
            None, pi=False; if provided, pi=True
        temperature (float, optional): Temperature parameter for contrastive loss.
            Defaults to 4.0
        wcl_age_threshold (float, optional): Threshold for age in weighted contrastive
            loss.
            Defaults to 0.0235
        wcl (bool, optional): Whether to use weighted contrastive loss.
            Defaults to True
        normalized_bp (bool, optional): Whether BP values are normalized.
            Defaults to True

    Note:
        Hyperparameters such as text_embedding, projection_dim, dropout, and d_model are
            now
        configured per-vital through `vital_encoders` and in `text_encoder_pipeline`,
            rather than
        as model-level parameters.
    """

    def __init__(
        self,
        vital_encoders: dict[str, Any],
        text_encoder_pipeline: Any | None = None,
        temperature: float = 4.0,
        wcl_age_threshold: float | None = 0.0235,
        wcl: bool = True,
        normalized_bp: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize BPModel with configuration parameters.

        Args:
            vital_encoders: Dictionary of VitalEncoder instances keyed by vital name
            text_encoder_pipeline: Optional TextEncoderPipeline instance for patient
                information
            temperature: Temperature parameter for contrastive learning. Default: 4.0
            wcl_age_threshold: Age threshold for weighted contrastive loss. Default:
                0.0235
            wcl: Whether to use weighted contrastive loss. Default: True
            normalized_bp: Whether BP values are normalized. Default: True
        """
        super().__init__(*args, **kwargs)

        # Validation
        if not isinstance(vital_encoders, dict) or not vital_encoders:
            raise ValueError("vital_encoders must be a non-empty dict of modules")
        for k, v in vital_encoders.items():
            if not isinstance(v, nn.Module):
                raise TypeError(f"vital_encoders['{k}'] must be an nn.Module instance")
        if text_encoder_pipeline is not None and not isinstance(
            text_encoder_pipeline, nn.Module
        ):
            raise TypeError("text_encoder_pipeline must be an nn.Module or None")
        if wcl_age_threshold is not None and wcl_age_threshold < 0:
            raise ValueError("wcl_age_threshold must be non-negative if provided")

        # pi is derived from presence of text encoder pipeline
        self.pi = text_encoder_pipeline is not None

        self.temperature = temperature
        self.wcl_age_threshold = wcl_age_threshold
        self.wcl = wcl
        self.normalized_bp = normalized_bp

        self.vital_encoders = nn.ModuleDict(vital_encoders)
        self.text_encoder_pipeline = text_encoder_pipeline

        # Ensure VitalEncoder.pi matches pipeline setting
        for name, enc in self.vital_encoders.items():
            if hasattr(enc, "pi") and enc.pi != self.pi:
                logger.warning(
                    f"VitalEncoder '{name}' pi={enc.pi} differs from BPModel "
                    f"pi={self.pi}. Overriding to model setting."
                )
                enc.pi = self.pi  # type: ignore[assignment]  # dynamic attribute set by model

        # L1 Loss
        self.l1 = torch.nn.L1Loss()

        logger.info(
            f"Initialized BPModel with {len(vital_encoders)} vital encoders: "
            f"{list(vital_encoders.keys())}, pi={self.pi}"
        )

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Extract and prepare input for BPModel from unified batch structure.

        This method extracts source vitals from the batch input key and splits them
        by channel based on the Direction object.

        Args:
            batch_dict: Unified batch dict with:
                - input key (BATCH_KEY_INPUT): Source waveforms [B, C, T] where C =
                    number
                  of source vitals
                - "direction": Direction object (single mode) or "directions" (multi
                    mode)
                - "text": Text encoding (if available)

        Returns:
            dict: Prepared input with keys for each vital sign (e.g., "ppg", "ecg")
                  plus optional "input_ids", "attention_mask"
        """
        # Extract source waveforms [B, C, T]
        x = batch_dict[BATCH_KEY_INPUT]
        b, c, t = x.shape

        direction: Direction | None = cast(
            "Direction | None", batch_dict.get("direction")
        )
        if direction is None:
            # Try multi-directional batch
            directions = batch_dict.get("directions")
            if directions is not None and isinstance(directions, list):
                direction = cast("Direction", directions[0])  # Use first direction

        if direction is None:
            raise ValueError(
                "No direction in batch_dict. Expected 'direction' or 'directions'. "
                "Configure collate with direction metadata."
            )

        vitals = direction.source

        # Verify vitals have corresponding pre-initialized encoders
        for vital in vitals:
            vital_key = vital.value.lower()
            if vital_key not in self.vital_encoders:
                raise ValueError(
                    f"Vital '{vital_key}' not in vital_encoders. "
                    f"Available: {list(self.vital_encoders.keys())}. "
                    "Add vital under vital_encoders in YAML."
                )

            # Use BaseModel.extract_input to get gathered sources with metadata-driven
            # extraction
        # This is more robust than manual channel splitting as it uses src_idxs/src_mask
        x_gathered = super().extract_input(batch_dict)  # [B, S_max, T]
        assert isinstance(x_gathered, torch.Tensor)
        b, s_max, t = x_gathered.shape

        batch_input: torch.Tensor | str = batch_dict.get(BATCH_KEY_INPUT, "NOT_FOUND")
        shape_str: tuple[int, ...] | str = (
            batch_input.shape if isinstance(batch_input, torch.Tensor) else "NOT_FOUND"
        )
        logger.debug(
            f"[BPModel.extract_input] batch_dict[BATCH_KEY_INPUT] shape: {shape_str}"
        )
        logger.debug(
            f"[BPModel.extract_input] x_gathered shape: {x_gathered.shape} "
            f"(B={b}, S_max={s_max}, T={t})"
        )
        src_mask = batch_dict["src_mask"]  # [B, S_max]

        result = {}
        for idx, vital in enumerate(vitals):
            vital_key = vital.value.lower()
            # Extract slice where this vital is active
            if idx < s_max:
                assert isinstance(x_gathered, torch.Tensor)
                vital_slice = x_gathered[:, idx : idx + 1, :]  # [B, 1, T]
                logger.debug(
                    f"[BPModel.extract_input] vital_key={vital_key}, idx={idx}, "
                    f"vital_slice shape: {vital_slice.shape}"
                )
                # Only keep if mask indicates it's active
                if src_mask[:, idx].any():
                    result[vital_key] = vital_slice  # [B, 1, T]
                    logger.debug(
                        f"[BPModel.extract_input] Added {vital_key} to result with "
                        f"shape: {result[vital_key].shape}"
                    )
            else:
                raise ValueError(
                    f"Direction specifies {len(vitals)} vitals but batch only has "
                    f"{s_max} source slots"
                )

        # Handle text encoding if available
        if "text" in batch_dict:
            text_dict = cast("dict[str, torch.Tensor]", batch_dict["text"])
            result["input_ids"] = text_dict["input_ids"]
            result["attention_mask"] = text_dict["attention_mask"]

        return result

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass with pre-instantiated vital encoders.

        Args:
            batch: Dictionary containing source vitals and optional text data

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'y_pred_sbp': Aggregated SBP predictions [B, 1]
                - 'y_pred_dbp': Aggregated DBP predictions [B, 1]
                - '{vital}_sbp' / '{vital}_dbp': Per-vital predictions (if available)
                - '{vital}_embeddings': Embeddings [B, projection_dim] for each vital
                - 'text_embeddings': Text embeddings [B, projection_dim] (if PI enabled)
        """
        # Extract input
        x = self.extract_input(batch)

        vital_embeddings: dict[str, torch.Tensor] = {}
        sbp_predictions: list[torch.Tensor] = []
        dbp_predictions: list[torch.Tensor] = []
        result: dict[str, torch.Tensor] = {}

        # Process text once for all vitals (if available)
        text_emb = None
        text_for_encoder = None
        if (
            self.text_encoder_pipeline is not None
            and isinstance(x, dict)
            and "input_ids" in x
            and "attention_mask" in x
        ):
            text_emb = self.text_encoder_pipeline(
                x["input_ids"], x["attention_mask"]
            )  # [B, projection_dim, 1]
            text_for_encoder = text_emb.transpose(1, 2)  # [B, 1, projection_dim]

        # Process each vital through its pre-instantiated encoder
        if not isinstance(x, dict):
            raise TypeError(f"Expected extract_input to return dict, got {type(x)}")
        for vital_key in self.vital_encoders:
            if vital_key not in x:
                logger.warning(f"Vital {vital_key} not found in input, skipping")
                continue

            encoder = self.vital_encoders[vital_key]
            logger.debug(
                f"[BPModel.forward] Processing vital_key={vital_key}, x[{vital_key}] "
                f"shape: {x[vital_key].shape}"
            )
            if text_for_encoder is not None:
                logger.debug(
                    f"[BPModel.forward] text_for_encoder shape: "
                    f"{text_for_encoder.shape}"
                )
            sbp, dbp, embedding = encoder(
                x[vital_key], text_for_encoder if text_emb is not None else None
            )

            vital_embeddings[vital_key] = embedding
            sbp_predictions.append(sbp)
            dbp_predictions.append(dbp)
            result[f"{vital_key}_sbp"] = sbp
            result[f"{vital_key}_dbp"] = dbp

        if not sbp_predictions:
            raise ValueError(
                "No valid vitals found in input. Ensure batch contains at least one "
                "vital from vital_encoders."
            )

        # Aggregate predictions (average across all vitals)
        y_sbp = torch.stack(sbp_predictions).mean(dim=0)
        y_dbp = torch.stack(dbp_predictions).mean(dim=0)

        # Ensure [B, 1] layout (upstream encoders may squeeze).
        if y_sbp.dim() == 1:
            y_sbp = y_sbp.unsqueeze(-1)
        if y_dbp.dim() == 1:
            y_dbp = y_dbp.unsqueeze(-1)
        if y_sbp.shape[-1] != 1 or y_dbp.shape[-1] != 1:
            raise ValueError(
                "BPModel.forward expected aggregated SBP/DBP tensors with "
                "trailing dimension 1. Got shapes "
                f"sbp={tuple(y_sbp.shape)}, dbp={tuple(y_dbp.shape)}"
            )

        output_dict: dict[str, Any] = {
            "y_pred_sbp": y_sbp,
            "y_pred_dbp": y_dbp,
        }
        output_dict.update(result)
        output_dict["vital_embeddings"] = vital_embeddings
        output_dict["text_embeddings"] = text_emb
        output_dict["per_vital_bp"] = result

        # Reshape embeddings for WCL: [B, projection_dim, 1] -> [B, projection_dim]
        for vital_key, embedding in vital_embeddings.items():
            output_dict[f"{vital_key}_embeddings"] = embedding.squeeze(-1)

        if text_emb is not None:
            output_dict["text_embeddings"] = text_emb.squeeze(-1)

        # Concat SBP/DBP for processor (col0=SBP, col1=DBP;
        # output_keys: ['sbp','dbp'])
        output_dict["predictions"] = torch.cat(
            [y_sbp, y_dbp],
            dim=1,
            # [B, 2] canonical; processors split via output_keys=['sbp','dbp']
        )

        return output_dict

    def get_component_dict(self, outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Return structured BP components from model outputs.

        Args:
            outputs: Dict with 'y_pred_sbp' / 'y_pred_dbp' keys.

        Returns:
            Dictionary with keys:
                - y_pred_sbp: Aggregated SBP predictions [B, 1]
                - y_pred_dbp: Aggregated DBP predictions [B, 1]
                - predictions: Concatenated BP predictions [B, 2] (col0=SBP, col1=DBP)
                - {vital_key}_sbp / {vital_key}_dbp: Per-vital predictions (if
                    available)
                - {vital_key}_embeddings: Embeddings for each vital (if available)
                - text_embeddings: Text embeddings (if available)
        """
        if not isinstance(outputs, dict):
            raise TypeError(
                "BPModel.get_component_dict() expects dict outputs from forward(); "
                f"received {type(outputs).__name__}."
            )

        missing = [k for k in ("y_pred_sbp", "y_pred_dbp") if k not in outputs]
        if missing:
            raise KeyError(
                f"Expected keys 'y_pred_sbp' and 'y_pred_dbp' in outputs. "
                f"Missing: {missing}"
            )
        if "predictions" not in outputs:
            logger.info("No 'predictions'; ensure forward consistency")
        results = dict(outputs)

        # Canonical BP outputs must remain [B, 1]
        for key, value in (
            ("y_pred_sbp", results["y_pred_sbp"]),
            ("y_pred_dbp", results["y_pred_dbp"]),
        ):
            if value.dim() == 1:
                results[key] = value.unsqueeze(-1)
            elif value.shape[-1] != 1:
                raise ValueError(
                    f"{key} must have trailing dimension 1. Got shape "
                    f"{tuple(value.shape)}."
                )

        per_vital_bp = outputs.get("per_vital_bp")
        if per_vital_bp and isinstance(per_vital_bp, dict):
            for key, value in per_vital_bp.items():
                results.setdefault(key, value)

        vital_embeddings = outputs.get("vital_embeddings")
        if vital_embeddings and isinstance(vital_embeddings, dict):
            for vital_key, embedding in vital_embeddings.items():
                results.setdefault(f"{vital_key}_embeddings", embedding.squeeze(-1))

        if "text_embeddings" in outputs and outputs["text_embeddings"] is not None:
            results.setdefault(
                "text_embeddings", outputs["text_embeddings"].squeeze(-1)
            )

        return results

    def get_sbp_dbp_from_ppg(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get blood pressure predictions from PPG signal.

        This method processes Photoplethysmogram (PPG) signals to predict both systolic
            (SBP)
        and diastolic (DBP) blood pressure values. It returns per-vital PPG predictions.

        Args:
            batch (dict): Dictionary containing:
                - "ppg" (torch.Tensor): PPG signal data of shape (batch_size, channels,
                  sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if
                    using PI

        Returns:
            tuple: (SBP_predictions, DBP_predictions)
                - SBP_predictions (torch.Tensor): Predicted systolic blood pressure
                    values for PPG
                - DBP_predictions (torch.Tensor): Predicted diastolic blood pressure
                    values for PPG

        Example:
            >>> batch = {
            ...     "ppg": torch.randn(32, 1, 1024),  # batch_size=32, channels=1,
                seq_len=1024
            ...     "input_ids": torch.randint(0, 1000, (32, 128)),  # if using PI
            ...     "attention_mask": torch.ones(32, 128)  # if using PI
            ... }
            >>> sbp, dbp = model.get_sbp_dbp_from_ppg(batch)
        """
        outputs = self.forward(batch)
        per_vital_bp = cast("dict[str, torch.Tensor]", outputs["per_vital_bp"])
        return (per_vital_bp["ppg_sbp"], per_vital_bp["ppg_dbp"])

    def get_sbp_dbp_from_ecg(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get blood pressure predictions from ECG signal.

        This method processes ECG signals to predict both systolic (SBP) and diastolic
            (DBP)
        blood pressure values. It returns per-vital ECG predictions.

        Args:
            batch (dict): Dictionary containing:
                - "ecg" (torch.Tensor): ECG signal data of shape (batch_size, channels,
                  sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if
                    using PI

        Returns:
            tuple: (SBP_predictions, DBP_predictions)
                - SBP_predictions (torch.Tensor): Predicted systolic blood pressure
                    values for ECG
                - DBP_predictions (torch.Tensor): Predicted diastolic blood pressure
                    values for ECG

        Example:
            >>> batch = {
            ...     "ecg": torch.randn(32, 1, 1024),  # batch_size=32, channels=1,
                seq_len=1024
            ...     "input_ids": torch.randint(0, 1000, (32, 128)),  # if using PI
            ...     "attention_mask": torch.ones(32, 128)  # if using PI
            ... }
            >>> sbp, dbp = model.get_sbp_dbp_from_ecg(batch)
        """
        outputs = self.forward(batch)
        per_vital_bp_ecg = cast("dict[str, torch.Tensor]", outputs["per_vital_bp"])
        return (per_vital_bp_ecg["ecg_sbp"], per_vital_bp_ecg["ecg_dbp"])


if __name__ != "__main__":
    # Register with Hydra ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="base_unet_swin_unet", group="model", node=UNetSwinUnetConfig)
    cs.store(name="base_bp_model", group="model", node=BPModelConfig)
