"""WaveNet Implementation.

This module implements WaveNet architecture for waveform generation.

References:
- Paper: "WaveNet: A Generative Model for Raw Audio"
  https://arxiv.org/abs/1609.03499
- Original Implementation: https://github.com/vincentherrmann/pytorch-wavenet
- License: MIT

Note: This implementation is adapted from the original codebase for use in
the MD-ViSCo framework.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


@dataclass
class WaveNetModelConfig(SingleStageModelConfig):
    """Configuration for WaveNet architecture parameters.

    The model follows vanilla WaveNet format with input/output shape [B, C, L] (batch,
    channels, length).

    Args:
        layers (int): Number of layers in each block
        blocks (int): Number of wavenet blocks of this model
        dilation_channels (int): Number of channels for the dilated convolution
        residual_channels (int): Number of channels for the residual connection
        skip_channels (int): Number of channels for the skip connections
        end_channels (int): Number of channels for the end convolution
        classes (int): Number of possible values each sample can have
        output_length (int): Number of samples that are generated for each input
        kernel_size (int): Size of the dilation kernel
        bias (bool): Whether to use bias in convolutions

    Note:
        Default architecture parameters (layers=10, blocks=4, dilation_channels=32,
        residual_channels=32, skip_channels=256, end_channels=256, kernel_size=2,
        bias=False) match the reference implementation.

    References:
        - GitHub pytorch-wavenet (vincentherrmann/pytorch-wavenet, wavenet_model.py)
        - WaveNet paper (van den Oord et al., arXiv:1609.03499, 2016, Sections 2.1-2.4)
        - Note: classes=256 and output_length=32 are reference defaults;
          configs override to 1/1 for regression tasks.
    """

    _target_: str = "src.model.wavenet.WaveNetModel"
    supports_multi_directional: bool = False
    model_name: str = "WaveNet"

    layers: int = 10
    blocks: int = 4
    dilation_channels: int = 32
    residual_channels: int = 32
    skip_channels: int = 256
    end_channels: int = 256
    classes: int = 256
    output_length: int = 32
    kernel_size: int = 2
    bias: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.blocks <= 0:
            raise ValueError("blocks must be positive")
        if self.dilation_channels <= 0:
            raise ValueError("dilation_channels must be positive")
        if self.residual_channels <= 0:
            raise ValueError("residual_channels must be positive")
        if self.skip_channels <= 0:
            raise ValueError("skip_channels must be positive")
        if self.end_channels <= 0:
            raise ValueError("end_channels must be positive")
        if self.classes <= 0:
            raise ValueError("classes must be positive")
        if self.output_length is None:
            raise ValueError("output_length must be set")
        if self.output_length <= 0:
            raise ValueError("output_length must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.input_length is None:
            raise ValueError("input_length must be set")
        if self.input_length <= 0:
            raise ValueError("input_length must be positive")


# ============================================================================
# WAVENET MODEL IMPLEMENTATION
# ============================================================================


class WaveNetModel(SingleStageModel):
    """WaveNet model for waveform generation.

    Implements the WaveNet architecture with dilated convolutions, residual
    connections, and skip connections for autoregressive waveform generation.

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` where N is batch size, C_in is
          input channels, L_in is input sequence length
        - Output: :math:`(N, T, C)` where T is output sequence length, C is
          number of classes

    Note:
        L_in should be at least the length of the receptive field for proper
        operation.
    """

    def __init__(
        self,
        layers: int = 10,
        blocks: int = 4,
        dilation_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 256,
        classes: int = 256,
        output_length: int = 32,
        kernel_size: int = 2,
        bias: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize WaveNetModel with configuration parameters.

        Args:
            layers: Number of layers in each block
            blocks: Number of wavenet blocks
            dilation_channels: Number of channels for the dilated convolution
            residual_channels: Number of channels for the residual connection
            skip_channels: Number of channels for the skip connections
            end_channels: Number of channels for the end convolution
            classes: Number of possible values each sample can have
            output_length: Number of samples that are generated for each input
            kernel_size: Size of the dilation kernel
            bias: Whether to use bias in convolutions
        """
        super().__init__(*args, **kwargs)

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size

        receptive_field = 1
        self.dilations = []
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 conv for channel expansion: prepares input for dilated convolutions
        self.start_conv = nn.Conv1d(
            in_channels=self.classes,
            out_channels=residual_channels,
            kernel_size=1,
            bias=bias,
        )

        for _b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _i in range(layers):
                self.dilations.append(new_dilation)

                # Dilated convolutions with causal padding (manual left padding)
                self.filter_convs.append(
                    nn.Conv1d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=kernel_size,
                        dilation=new_dilation,
                        padding=0,
                        bias=bias,
                    )
                )

                self.gate_convs.append(
                    nn.Conv1d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=kernel_size,
                        dilation=new_dilation,
                        padding=0,
                        bias=bias,
                    )
                )

                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Conv1d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=1,
                        bias=bias,
                    )
                )

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv1d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=1,
                        bias=bias,
                    )
                )

                receptive_field += additional_scope
                additional_scope *= 2
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=1,
            bias=True,
        )

        self.end_conv_2 = nn.Conv1d(
            in_channels=end_channels, out_channels=classes, kernel_size=1, bias=True
        )

        self.output_length = output_length
        self.receptive_field = receptive_field

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and prepare input for WaveNet from unified batch structure.

        This method handles the unified input processing for WaveNet, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, C, L] for WaveNet

        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs

        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, channels,
                signal_length)
        """
        # Use parent class implementation for unified batch structure
        x = super().extract_input(batch_dict)
        if isinstance(x, dict):
            x = x["x"]
        elif isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            raise TypeError("WaveNet extract_input expected Tensor from parent")
        return x

    def wavenet(self, input: torch.Tensor) -> torch.Tensor:
        """Execute WaveNet forward pass through all dilated convolution layers.

        Args:
            input: Input tensor of shape [B, C, T] where C is classes.

        Returns:
            Output tensor of shape [B, C, T] where C is classes.
        """
        x = self.start_conv(input)
        skip = None

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # Store residual connection (before dilated convs)
            residual = x

            # Apply causal padding for dilated convolution
            padding_amount = (self.kernel_size - 1) * self.dilations[i]
            x_p = F.pad(x, (padding_amount, 0))  # (left, right) on last dimension

            f = torch.tanh(self.filter_convs[i](x_p))
            g = torch.sigmoid(self.gate_convs[i](x_p))
            x = f * g

            # parametrized skip connection
            s = self.skip_convs[i](x)  # [B, C_skip, T]
            skip = s if skip is None else skip + s

            x = self.residual_convs[i](x)
            # Add residual connection: enables gradient flow in deep networks
            x = x + residual

        if skip is None:
            raise RuntimeError("WaveNet skip connection was never set")
        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Forward pass of the WaveNet model.

        Args:
            batch_dict: Unified batch dict from DataLoader collate_fn

        Returns:
            Dict[str, torch.Tensor]: Dictionary following the canonical model schema:
                - "predictions": Output waveform tensor of shape [B, T, C] where B is
                    batch size,
                  T is the output sequence length, and C is the number of classes
                - "extras": Empty dictionary for auxiliary outputs
        """
        # Handle both dict inputs and pre-processed tensor inputs
        x = self.extract_input(batch_dict)

        x = self.wavenet(x)

        # Transpose to [B, T, C] format expected by downstream processors
        out_len = self.output_length
        x = x[:, :, -out_len:]
        x = x.transpose(1, 2).contiguous()

        return {
            "predictions": x,
            "extras": {},
        }


# ============================================================================
# HYDRA CONFIGURATION REGISTRATION
# ============================================================================

# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_wavenet", group="model", node=WaveNetModelConfig)
