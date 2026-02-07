# Standard library imports
import os
import os.path
import time
import math
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from torch.nn import Parameter
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

# Local imports
from .base_model import BaseModel, BaseModelConfig
from src.utils.bp_utils import extract_bp_values


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class WaveNetModelConfig(BaseModelConfig):
    """Configuration class for WaveNet model architecture parameters.
    
    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the WaveNet model, independently of training or dataset.
    
    The model follows vanilla WaveNet format with input/output shape [B, C, L] (batch, channels, length).
    
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
    """
    _target_: str = "src.model.WaveNet.WaveNetModel"
    supports_multi_directional: bool = False  # WaveNet only supports single-direction
    model_name: str = "WaveNet"
    
    # Model Architecture configuration
    layers: int = 10  # Number of layers in each block
    blocks: int = 4   # Number of wavenet blocks
    dilation_channels: int = 32  # Number of channels for the dilated convolution
    residual_channels: int = 32  # Number of channels for the residual connection
    skip_channels: int = 256     # Number of channels for the skip connections
    end_channels: int = 256      # Number of channels for the end convolution
    classes: int = 256           # Number of possible values each sample can have
    output_length: int = 32      # Number of samples that are generated for each input
    kernel_size: int = 2         # Size of the dilation kernel
    bias: bool = False           # Whether to use bias in convolutions
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
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
        if self.output_length <= 0:
            raise ValueError("output_length must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.input_length <= 0:
            raise ValueError("input_length must be positive")


# ============================================================================
# WAVENET MODEL IMPLEMENTATION
# ============================================================================

class WaveNetModel(BaseModel):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 output_length=32,
                 kernel_size=2,
                 bias=False,
                 *args, **kwargs):

        super(WaveNetModel, self).__init__(*args, **kwargs)

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append(new_dilation)

                # dilated convolutions with proper dilation and causal padding
                # Use padding_mode='replicate' and manual left padding for causal behavior
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   dilation=new_dilation,
                                                   padding=0,  # No automatic padding
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 dilation=new_dilation,
                                                 padding=0,  # No automatic padding
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def extract_input(self, batch_dict: Dict[str, torch.Tensor], source_channel: int = 0, target_channel: int = 0) -> torch.Tensor:
        """Extract and prepare input for WaveNet from unified batch structure.
        
        This method handles the unified input processing for WaveNet, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, C, L] for WaveNet
        
        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs
            source_channel: Legacy parameter (ignored in unified mode)
            target_channel: Legacy parameter (ignored in unified mode)
        
        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, channels, signal_length)
        """
        # Use parent class implementation for unified batch structure
        x = super().extract_input(batch_dict)
        
        # WaveNet expects [B, C, L] format - ensure proper channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        elif x.dim() == 3 and x.size(1) == 1:
            pass  # Already correct format
        else:
            # If multiple channels, take the first one for now
            x = x[:, 0:1, :] if x.size(1) > 1 else x
            
        return x.transpose(1, 2).contiguous()

    def wavenet(self, input):
        """Simplified WaveNet forward pass without dilation function."""
        x = self.start_conv(input)
        skip = None

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # Store residual connection (before dilated convs)
            residual = x

            # Apply causal padding for dilated convolution
            padding_amount = (self.kernel_size - 1) * self.dilations[i]
            x_p = F.pad(x, (padding_amount, 0))   # (left, right) on last dimension
            
            # dilated convolution
            f = torch.tanh(self.filter_convs[i](x_p))
            g = torch.sigmoid(self.gate_convs[i](x_p))
            x = f * g

            # parametrized skip connection
            s = self.skip_convs[i](x)              # [B, C_skip, T]
            skip = s if skip is None else skip + s

            x = self.residual_convs[i](x)
            # Add residual connection - now dimensions should match
            x = x + residual

        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x



    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        """Forward pass of the WaveNet model.
        
        Args:
            batch_dict: Unified batch dict from DataLoader collate_fn
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model outputs
        """
        # Handle both dict inputs and pre-processed tensor inputs
        x = self.extract_input(batch_dict)

        x = self.wavenet(x)

        # reshape output
        [n, c, l] = x.size()
        # l = self.output_length
        # x = x[:, :, -l:]
        x = x.transpose(1, 2).contiguous()

        # Handle BP extraction if needed
        if self._dictionary_key == "y_pred_waveform":
            # x = x.view(n * l, c)
            if x.shape[2] == 1280:  # Length is now dim 1
                sbp, dbp = extract_bp_values(x[:, :, 15:-15])
            else:
                sbp, dbp = extract_bp_values(x)
            
            return {"y_pred_waveform": x, "y_pred_sbp": sbp, "y_pred_dbp": dbp}
        
        return {"y_pred": x}


# ============================================================================
# HYDRA CONFIGURATION REGISTRATION
# ============================================================================

# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_wavenet", group="model", node=WaveNetModelConfig)