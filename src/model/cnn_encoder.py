"""CNN encoder from UNet_SwinUnet (MDViSCo) for use in the AF classifier.

Used by the AF classifier. Registered with Hydra ConfigStore.
"""

# Standard library imports
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig


class ConvBlock(nn.Module):
    """CNN block extracted from UNet_SwinUnet._block.

    Double-convolution block with instance normalization and LeakyReLU
    activation, directly extracted from the UNet_SwinUnet architecture in
    MDViSCo.
    """

    def __init__(
        self,
        in_channels: int,
        features: int,
        kernel_size: int,
        name: str = "",
    ):
        """Initialize ConvBlock with configuration parameters.

        Args:
            in_channels: Number of input channels
            features: Number of output features
            kernel_size: Kernel size for convolutions
            name: Base name for the layers (for debugging)
        """
        super().__init__()

        self.block = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm1d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm1d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        """Forward pass through the convolutional block.

        Args:
            x: Input tensor

        Returns:
            Output tensor after applying convolutional block
        """
        return self.block(x)


@dataclass
class CNNEncoderConfig(SingleStageModelConfig):
    """Configuration for CNNEncoder as a SingleStageModel."""

    _target_: str = "src.model.cnn_encoder.CNNEncoder"
    model_name: str = "CNNEncoder"

    # Architecture parameters
    in_channels: int = 1
    init_features: int = 32
    kernel_size: int = 3
    depth: int = 3


class CNNEncoder(SingleStageModel):
    """CNN Encoder extracted from UNet_SwinUnet architecture.

    Implements the CNN encoder path from UNet_SwinUnet, including initial
    convolution, progressive downsampling with dual-path (MaxPool + Conv),
    and hierarchical feature extraction with increasing channels.

    Architecture:
        Input -> Initial Conv -> [ConvBlock -> DualPathDownsample -> ConvBlock] * depth
    """

    def __init__(
        self, in_channels=1, init_features=32, kernel_size=3, depth=3, *args, **kwargs
    ):
        """Initialize CNNEncoder with configuration parameters.

        Args:
            in_channels: Number of input channels. Default: 1
            init_features: Initial number of features. Default: 32
            kernel_size: Kernel size for convolutions. Default: 3
            depth: Number of encoder levels. Default: 3
        """
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.init_features = init_features
        self.kernel_size = kernel_size
        self.depth = depth

        # Initial convolution (extracted from conv_init_features)
        self.conv_init = nn.Conv1d(in_channels, init_features, 3, 1, 1)

        # Build encoder blocks (extracted from UNet_SwinUnet encoder construction)
        self.encoder = nn.ModuleList()
        features = init_features

        for i in range(depth):
            # First conv block
            self.encoder.append(ConvBlock(features, features, kernel_size, f"enc{i}_1"))
            # Downsampling convolution
            self.encoder.append(
                nn.Conv1d(
                    in_channels=features, out_channels=features, kernel_size=2, stride=2
                )
            )
            # Second conv block with doubled features
            self.encoder.append(
                ConvBlock(features * 2, features * 2, kernel_size, f"enc{i}_2")
            )
            features = features * 2

        self.final_features = features
        self.final_length = None  # Will be calculated during forward pass

    def forward(self, batch_dict):
        """Forward pass through the CNN encoder.

        Accepts either a NEW-format batch dict or a prepared tensor.

        Args:
            batch_dict: Dict with NEW batch structure (uses extract_input) or
                torch.Tensor of shape [B, C, T].

        Returns:
            Dict[str, torch.Tensor]: Dictionary following the canonical model schema:
                - "predictions": Flattened encoded features of shape [B, F] where
                  F = final_features * (T // 2**depth)
                - "extras": Empty dictionary for auxiliary outputs
        """
        # Support dict-or-tensor input
        if isinstance(batch_dict, dict):
            x = self.extract_input(batch_dict)  # [B, C, T]
        else:
            x = batch_dict  # already-prepared tensor [B, C, T]

        # Initial convolution
        x = self.conv_init(x)

        # Encoder path with dual-path downsampling (from UNet_SwinUnet forward)
        for i in range(0, len(self.encoder), 3):
            x = self.encoder[i](x)

            # Dual-path downsampling: MaxPool + Conv (from UNet_SwinUnet forward logic)
            x_down = F.max_pool1d(x, 2)
            x_conv_pool = self.encoder[i + 1](x)
            x = torch.cat([x_down, x_conv_pool], dim=1)

            # Second convolutional block
            x = self.encoder[i + 2](x)

        # Compute final dimensions locally
        final_length = x.shape[-1]
        # Flatten to [B, F]
        features = x.view(x.size(0), -1)
        return {
            "predictions": features,
            "extras": {"final_length": final_length},
        }

    def get_output_size(self, input_length):
        """Calculate the output size for a given input length.

        Args:
            input_length (int): Length of input sequence

        Returns:
            int: Total number of features after flattening
        """
        final_length = input_length // (2**self.depth)
        return self.final_features * final_length


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_cnn_encoder", node=CNNEncoderConfig, group="model")
