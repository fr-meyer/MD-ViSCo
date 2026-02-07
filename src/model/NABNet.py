# Standard library imports
from dataclasses import dataclass
from typing import Dict

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from hydra.core.config_store import ConfigStore
from .base_model import BaseModel, BaseModelConfig
from src.utils.bp_utils import extract_bp_values

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class NABNetModelConfig(BaseModelConfig):
    """NABNet Model Configuration - Centralized model architecture parameters
    
    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the NABNet model, independently of training or dataset.
    """
    _target_: str = "src.model.NABNet.NABNet"
    supports_multi_directional: bool = False  # NABNet only supports single-direction
    model_name: str = "NabNet"
    
    # Model Architecture configuration
    model_depth: int = 5  # Number of Level in the CNN Model
    model_width: int = 48  # Width of the Initial Layer
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
    problem_type: str = 'Regression'
    attention_type: str = 'lstm'  # 'standard' or 'lstm'
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        if self.model_depth <= 0:
            raise ValueError("model_depth must be positive")
        if self.model_width <= 0:
            raise ValueError("model_width must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.feature_number <= 0:
            raise ValueError("feature_number must be positive")
        if self.attention_type not in ['standard', 'lstm']:
            raise ValueError("attention_type must be 'standard' or 'lstm'")
        if self.input_length <= 0:  # UPDATED
            raise ValueError("input_length must be positive")


@dataclass
class ShallowUNetModelConfig(BaseModelConfig):
    """ShallowUNet Model Configuration - Centralized model architecture parameters
    
    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the ShallowUNet refinement model, independently of 
    training or dataset.
    """
    _target_: str = "src.model.NABNet.ShallowUNet"
    supports_multi_directional: bool = False  # ShallowUNet only supports single-direction
    model_name: str = "ShallowUNet"
    
    # Model Architecture configuration
    model_depth: int = 1  # Depth of the ShallowUNet (number of downsampling layers)
    model_width: int = 84  # Width or number of filters at the base layer of the UNet
    feature_number: int = 1024  # Feature vector size for autoencoder bottleneck
    hidden_size: int = 512  # Hidden state size for MLP layers in the head
    num_channel: int = 1  # Number of input channels (typically 1 for PPG, ECG, etc.)
    output_nums: int = 1  # Number of output channels (typically 1 for ABP)
    kernel_size: int = 3  # Kernel size for convolutional layers
    problem_type: str = 'Regression'  # Type of problem (e.g., 'Regression' for waveform output)
    
    # Architecture feature flags
    deep_supervision: int = 0  # Whether to use deep supervision (0: no, 1: yes)
    autoencoder: int = 1  # Whether autoencoder architecture is enabled (1: yes, 0: no)
    guided_attention: int = 0  # Whether attention modules are included (1: yes, 0: no)
    use_transconv: bool = True  # Whether to use transposed convolutions (True) or upsampling (False)
    use_lstm: int = 0  # Whether to include LSTM layers in the architecture (1: yes, 0: no)
    
    # MLP-specific parameters for BP prediction
    mlp_activation: str = 'relu'  # Activation function for MLPs ('relu' or 'tanh')
    mlp_alpha: float = 0.0001  # Alpha parameter for MLPs (L2 regularization)
    
    # Additional parameters
    alpha: float = 1.0  # General alpha parameter for UNet
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        if self.model_depth <= 0:
            raise ValueError("model_depth must be positive")
        if self.model_width <= 0:
            raise ValueError("model_width must be positive")
        if self.feature_number <= 0:
            raise ValueError("feature_number must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.problem_type not in ['Regression', 'Classification']:
            raise ValueError("problem_type must be 'Regression' or 'Classification'")
        if self.mlp_activation not in ['relu', 'tanh']:
            raise ValueError("mlp_activation must be 'relu' or 'tanh'")
        if self.mlp_alpha < 0:
            raise ValueError("mlp_alpha must be non-negative")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.input_length <= 0:  # UPDATED
            raise ValueError("input_length must be positive")

# ============================================================================
# COMMON UTILITY FUNCTIONS AND BLOCKS
# ============================================================================

def ConvBlock(in_channels, out_channels, kernel_size, padding='same'):
    """Unified 1D Convolutional Block for both NABNet and ShallowUNet"""
    if padding == 'same':
        padding = kernel_size // 2
        
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )

def TransConvBlock(in_channels, out_channels):
    """Unified 1D Transposed Convolutional Block for both NABNet and ShallowUNet"""
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )

class Lambda(nn.Module):
    """Lambda layer for custom functions"""
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

class FeatureExtractionBlock(nn.Module):
    """
    Unified Feature Extraction Block for both NABNet and ShallowUNet
    
    This class replaces the previous Feature_Extraction_Block and ShallowFeatureExtractionBlock
    with a single implementation that can handle both use cases:
    
    - NABNet: Uses reshape_dims parameter for explicit reshaping (batch, channels, length)
    - ShallowUNet: Uses model_width parameter for automatic reshaping (batch, model_width, -1)
    
    Args:
        input_size (int): Total number of input features (channels * length)
        feature_number (int): Number of features to extract in the bottleneck
        reshape_dims (tuple, optional): Explicit reshape dimensions for NABNet
        model_width (int, optional): Model width for ShallowUNet style reshaping
    """
    def __init__(self, input_size, feature_number, reshape_dims=None, model_width=None):
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
        

        
    def forward(self, x, feature_only=False):
        """
        Forward pass for feature extraction
        
        Args:
            x (torch.Tensor): Input tensor
            feature_only (bool): If True, return only the extracted features
                                If False, return reconstructed tensor
        
        Returns:
            torch.Tensor: Either features or reconstructed tensor based on feature_only
        """
        original_shape = x.shape
        batch_size = x.size(0)
        
        # Flatten the input
        x = self.flatten(x)
        
        # Validate input size
        if x.size(1) != self.input_size:
            raise ValueError(
                f"Input size mismatch. Got tensor with {x.size(1)} features, "
                f"expected {self.input_size} features. Input shape was {original_shape}"
            )
        
        # First linear layer + ReLU
        x = self.relu(self.fc1(x))
        
        # Return features directly if feature_only is True
        if feature_only:
            return x
            
        # Otherwise continue with reconstruction
        x = self.fc2(x)
        
        # Reshape based on configuration
        if self.reshape_dims is not None:
            # Use explicit reshape dimensions (for NABNet)
            return x.view(batch_size, *self.reshape_dims)
        elif self.model_width is not None:
            # Use model_width for ShallowUNet style reshaping
            return x.view(batch_size, self.model_width, -1)
        else:
            # Fallback to original shape
            return x.view(original_shape)

# ============================================================================
# APPROXIMATION MODEL: NABNet (Neural Approximation Blood pressure Network)
# ============================================================================


class Attention_Block(nn.Module):
    """Attention Block for NABNet"""
    def __init__(self, channels, multiplier):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels * multiplier, 1, stride=1)
        self.bn1 = nn.BatchNorm1d(channels * multiplier)
        self.conv2 = nn.Conv1d(channels, channels * multiplier, 1, stride=1)
        self.bn2 = nn.BatchNorm1d(channels * multiplier)
        self.conv3 = nn.Conv1d(channels * multiplier, 1, 1)
        self.bn3 = nn.BatchNorm1d(1)
        
    def forward(self, skip, gating):
        # Ensure same spatial dimensions
        if skip.size(2) != gating.size(2):
            gating = F.interpolate(gating, size=skip.size(2), mode='linear', align_corners=False)
            
        x1 = self.bn1(self.conv1(skip))
        x2 = self.bn2(self.conv2(gating))
        x = F.relu(x1 + x2)
        x = self.bn3(self.conv3(x))
        x = torch.sigmoid(x)
        return skip * x

class Attention_LSTM_Block(nn.Module):
    """Attention LSTM Block for NABNet"""
    def __init__(self, channels, multiplier, lstm_multiplier):
        super().__init__()
        self.channels = channels
        self.lstm_hidden = int(channels * lstm_multiplier)
        
        self.lstm_skip = nn.LSTM(channels, self.lstm_hidden, 
                                bidirectional=True, batch_first=True)
        self.lstm_up = nn.LSTM(channels, self.lstm_hidden, 
                              bidirectional=True, batch_first=True)
        
        # Adjust attention to match LSTM output dimensions
        self.attention = nn.MultiheadAttention(self.lstm_hidden * 2, 1)
        
        # Adjust output projection to match input channels
        self.out_proj = nn.Sequential(
            nn.Linear(self.lstm_hidden * 4, channels),
            nn.ReLU()
        )

    def forward(self, skip, up):
        batch_size = skip.size(0)
        seq_len = skip.size(2)
        
        # Transpose for LSTM (batch, channels, length) -> (batch, length, channels)
        skip = skip.transpose(1, 2)
        up = up.transpose(1, 2)
        
        # Process through LSTMs
        skip_feat, _ = self.lstm_skip(skip)  # Output: [batch, seq_len, lstm_hidden*2]
        up_feat, _ = self.lstm_up(up)        # Output: [batch, seq_len, lstm_hidden*2]
        
        # Apply attention
        attn_out, _ = self.attention(skip_feat, up_feat, up_feat)
        
        # Combine features
        combined = torch.cat([attn_out, skip_feat], dim=-1)  # [batch, seq_len, lstm_hidden*4]
        
        # Project back to original channel dimension
        output = self.out_proj(combined)  # [batch, seq_len, channels]
        
        # Transpose back to channel-first format
        output = output.transpose(1, 2)  # [batch, channels, seq_len]
        
        return output


class NABNet(BaseModel):
    """NABNet: Neural Approximation Blood pressure Network (Approximation Model)"""
    def __init__(self, model_depth=5, num_channel=1, model_width=48, 
                 kernel_size=3, problem_type='Regression', output_nums=1, 
                 ds=1, ae=0, ag=1, lstm=1, feature_number=1024, 
                 is_transconv=True, attention_type='standard', 
                 *args, **kwargs):
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
        if attention_type not in ['standard', 'lstm']:
            raise ValueError("attention_type must be 'standard' or 'lstm'")
        if problem_type not in ['Regression', 'Classification']:
            raise ValueError("problem_type must be 'Regression' or 'Classification'")
        
        # Store parameters
        self.model_depth = model_depth
        self.model_width = model_width
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.LSTM = lstm
        self.feature_number = feature_number
        self.is_transconv = is_transconv
        self.kernel_size = kernel_size
        self.num_channel = num_channel
        self.output_nums = output_nums
        self.problem_type = problem_type
        self.attention_type = attention_type
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        in_channels = self.num_channel
        
        for i in range(self.model_depth):
            out_channels = self.model_width * (2 ** i)
            self.encoder_blocks.append(
                nn.Sequential(
                    ConvBlock(in_channels, out_channels, self.kernel_size),
                    ConvBlock(out_channels, out_channels, self.kernel_size)
                )
            )
            in_channels = out_channels
            
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bridge
        bridge_channels = self.model_width * (2 ** self.model_depth)
        self.bridge = nn.Sequential(
            ConvBlock(self.model_width * (2 ** (self.model_depth - 1)), bridge_channels, self.kernel_size),
            ConvBlock(bridge_channels, bridge_channels, self.kernel_size)
        )
        
        # Decoder path with transposed convolution
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.model_depth):
            in_channels = self.model_width * (2 ** (self.model_depth - i))
            out_channels = self.model_width * (2 ** (self.model_depth - i - 1))
            self.decoder_blocks.append(TransConvBlock(in_channels, out_channels))
            
        # Attention blocks if enabled - now supports different attention types
        if self.A_G:
            self.attention_blocks = nn.ModuleList()
            # Initialize attention blocks in reverse order to match decoder path
            for i in range(self.model_depth - 1, -1, -1):
                channels = self.model_width * (2 ** i)
                if self.attention_type == 'lstm':
                    self.attention_blocks.append(Attention_LSTM_Block(channels, 1, 1))
                else:  # 'standard' attention
                    self.attention_blocks.append(Attention_Block(channels, 2))
        else:
            self.attention_blocks = None
                
        # Final convolution
        self.final_conv = nn.Conv1d(self.model_width, self.output_nums, 1)
        self.final_activation = nn.Identity() if self.problem_type == 'Regression' else nn.Sigmoid()
        
        # Initialize feature extraction if autoencoder mode is enabled
        if self.A_E:
            bottleneck_channels = self.model_width * (2 ** (self.model_depth - 1))
            bottleneck_length = self.input_length // (2 ** self.model_depth)  # UPDATED
            input_size = bottleneck_channels * bottleneck_length
            
            self.feature_extraction = FeatureExtractionBlock(
                input_size=input_size,
                feature_number=self.feature_number,
                reshape_dims=(bottleneck_channels, bottleneck_length)
            )
            
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            self.feature_extraction.apply(init_weights)
        else:
            self.feature_extraction = None
        
        # Initialize decoder conv blocks
        self.decoder_conv_blocks = nn.ModuleList()
        for i in range(self.model_depth):
            curr_width = self.model_width * (2 ** (self.model_depth - i - 1))
            in_channels = curr_width * 2  # *2 for concatenation
            
            self.decoder_conv_blocks.append(
                nn.Sequential(
                    ConvBlock(in_channels, curr_width, self.kernel_size),
                    ConvBlock(curr_width, curr_width, self.kernel_size)
                )
            )
        
        # Initialize deep supervision convs
        if self.D_S:
            self.ds_convs = nn.ModuleList([
                nn.Conv1d(self.model_width * (2 ** i), self.output_nums, 1)
                for i in range(self.model_depth)
            ][::-1])  # Reverse order to match decoder path
        else:
            self.ds_convs = None

    def extract_input(self, batch_dict: Dict[str, torch.Tensor], source_channel: int = 0, target_channel: int = 0) -> torch.Tensor:
        """Extract and prepare input for NABNet from unified batch structure.
        
        This method handles the unified input processing for NABNet, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, 1, L] for NABNet
        
        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs
            source_channel: Legacy parameter (ignored in unified mode)
            target_channel: Legacy parameter (ignored in unified mode)
        
        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        # Use parent class implementation for unified batch structure
        x = super().extract_input(batch_dict)
        
        # NABNet expects [B, 1, L] format
        if x.dim() == 3 and x.size(1) == 1:
            return x  # Already [B, 1, L]
        else:
            return x.unsqueeze(1)  # Add channel dimension

    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        # Handle both dict inputs and pre-processed tensor inputs
        if isinstance(batch_dict, dict):
            x = self.extract_input(batch_dict)
        
        # Store encoder outputs for skip connections
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
            return {"y_pred": decoder_outputs[::-1]}  # Reverse order to match TF implementation
        return {"y_pred": x}


# ============================================================================
# REFINEMENT MODEL: ShallowUNet (Refinement Model)
# ============================================================================

class ShallowMLPRegressor(nn.Module):
    """PyTorch implementation of MLPRegressor for ShallowUNet"""
    def __init__(self, input_size, hidden_size=100, activation='relu', alpha=0.0001):
        super(ShallowMLPRegressor, self).__init__()
        
        # Define activation function
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        
        # Define network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activation,
            nn.Linear(hidden_size, 1)
        )
        
        # L2 regularization strength
        self.alpha = alpha
        
    def forward(self, x):
        return self.layers(x)


class ShallowAttentionBlock(nn.Module):
    """Attention Block for ShallowUNet"""
    def __init__(self, in_channels, num_filters, is_transconv=True):
        super(ShallowAttentionBlock, self).__init__()
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
        self.trans_conv = TransConvBlock(1, 1)
        
    def forward(self, skip_connection, gating_signal):
        # Process skip connection
        conv1x1_1 = self.conv1x1_1(skip_connection)
        conv1x1_1 = self.bn1(conv1x1_1)
        
        # Process gating signal
        conv1x1_2 = self.conv1x1_2(gating_signal)
        conv1x1_2 = self.bn2(conv1x1_2)
        
        # Add the features
        conv1_2 = conv1x1_1 + conv1x1_2
        
        conv1_2 = self.relu(conv1_2)
        conv1_2 = self.conv_final(conv1_2)
        conv1_2 = self.bn_final(conv1_2)
        
        conv1_2 = self.sigmoid(conv1_2)
        
        # Upsample back to original size
        resampler1 = self.up_conv(conv1_2)
        
        resampler2 = self.trans_conv(conv1_2)
        
        resampler = resampler1 + resampler2
        
        result = skip_connection * resampler
        
        return result

class ShallowConcatBlock(nn.Module):
    """Concatenation Block for ShallowUNet"""
    def __init__(self):
        super(ShallowConcatBlock, self).__init__()
        
    def forward(self, input1, *argv):
        cat = input1
        for arg in argv:
            cat = torch.cat([cat, arg], dim=1)
        return cat

class ShallowUpConvBlock(nn.Module):
    """1D UpSampling Block for ShallowUNet"""
    def __init__(self, size=2):
        super(ShallowUpConvBlock, self).__init__()
        self.size = size
        
    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.size, mode='nearest')

class ShallowConvLSTM1D(nn.Module):
    """1D Convolutional LSTM for ShallowUNet"""
    def __init__(self, input_size, hidden_size, kernel_size=3, padding='same', go_backwards=True):
        super(ShallowConvLSTM1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.go_backwards = go_backwards
        
        # Input gate
        self.conv_i = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn_i = nn.BatchNorm1d(hidden_size)
        
        # Forget gate
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
        
    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        
        # Initialize hidden state and cell state
        h_t = torch.zeros(batch_size, self.hidden_size, seq_len).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, seq_len).to(x.device)
        
        # Reverse sequence if go_backwards
        if self.go_backwards:
            x = torch.flip(x, [2])
            
        # Gates
        i = self.sigmoid(self.bn_i(self.conv_i(x)))  # Input gate
        f = self.sigmoid(self.bn_f(self.conv_f(x)))  # Forget gate
        c = self.tanh(self.bn_c(self.conv_c(x)))     # Cell gate
        o = self.sigmoid(self.bn_o(self.conv_o(x)))  # Output gate
        
        # Update cell state
        c_t = f * c_t + i * c
        # Update hidden state
        h_t = o * self.tanh(c_t)
        
        return h_t

class ShallowUNet(BaseModel):
    """Core ShallowUNet implementation"""
    def __init__(self, model_depth, num_channel, model_width, kernel_size, 
                 problem_type='Regression', output_nums=1, deep_supervision=0, autoencoder=1, guided_attention=0, use_lstm=0,
                 alpha=1, feature_number=1024, use_transconv=True, *args, **kwargs):
        super(ShallowUNet, self).__init__(*args, **kwargs)
        
        # Store other configuration parameters
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = deep_supervision  # Keep internal name for backward compatibility
        self.A_E = autoencoder       # Keep internal name for backward compatibility
        self.A_G = guided_attention  # Keep internal name for backward compatibility
        self.LSTM = use_lstm         # Keep internal name for backward compatibility
        self.alpha = alpha
        self.feature_number = feature_number
        self.is_transconv = use_transconv  # Keep internal name for backward compatibility
        
        # Input validation
        if self.input_length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        # Initialize encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(1, self.model_depth + 1):
            in_channels = num_channel if i == 1 else model_width * (2 ** (i-2))
            out_channels = model_width * (2 ** (i-1))
            double_conv = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size),
                ConvBlock(out_channels, out_channels, kernel_size)
            )
            self.encoder_blocks.append(double_conv)
            
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck channels should match the last encoder block output
        bottleneck_in_channels = model_width * (2 ** (model_depth-1))
        bottleneck_out_channels = model_width * (2 ** model_depth)
        
        # AutoEncoder feature extraction
        if self.A_E:
            # Calculate the correct input size for the feature extraction
            input_size = bottleneck_in_channels * (self.input_length // (2 ** model_depth))
            self.feature_extraction = FeatureExtractionBlock(
                input_size=input_size,
                feature_number=feature_number,
                model_width=bottleneck_in_channels
            )
        else:
            self.feature_extraction = None
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_in_channels, bottleneck_out_channels, kernel_size),
            ConvBlock(bottleneck_out_channels, bottleneck_out_channels, kernel_size)
        )
        
        # Initialize decoder components
        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        # Initialize conditional components only if flags are enabled
        self.ds_convs = nn.ModuleList() if self.D_S else None
        self.attention_blocks = nn.ModuleList() if self.A_G else None
        self.lstm_layers = nn.ModuleList() if self.LSTM else None
        
        # Utility blocks
        self.concat_block = ShallowConcatBlock()
            
        for i in range(model_depth):
            # Calculate channels
            in_channels = model_width * (2 ** (model_depth-i))
            out_channels = model_width * (2 ** (model_depth-i-1))
            
            # 1. Deep Supervision
            if self.D_S and self.ds_convs is not None:
                self.ds_convs.append(nn.Conv1d(in_channels, 1, 1))
                
            # 2. Upsampling blocks
            if self.is_transconv:
                self.up_blocks.append(TransConvBlock(in_channels, out_channels))
            else:
                self.up_blocks.append(ShallowUpConvBlock())
            
            # 3. Attention
            if self.A_G and self.attention_blocks is not None:
                self.attention_blocks.append(
                    ShallowAttentionBlock(out_channels, out_channels, is_transconv=self.is_transconv)
                )
            
            # 4. LSTM
            if self.LSTM and self.lstm_layers is not None:
                lstm_in_channels = in_channels + out_channels if not self.is_transconv else out_channels * 2
                self.lstm_layers.append(
                    ShallowConvLSTM1D(
                        input_size=lstm_in_channels,  # Adjusted based on upsampling method
                        hidden_size=out_channels,
                        kernel_size=3,
                        padding='same',
                        go_backwards=True
                    )
                )
            
            # 5. Decoder Convolutions
            if self.is_transconv:
                decoder_in_channels = out_channels * 2  # out_channels from TransConv + out_channels from skip
            else:
                decoder_in_channels = in_channels + out_channels  # in_channels from upsampled + out_channels from skip

            if self.LSTM:
                decoder_in_channels = out_channels  # LSTM output channels

            self.decoder_blocks.append(nn.Sequential(
                ConvBlock(decoder_in_channels, out_channels, kernel_size),
                ConvBlock(out_channels, out_channels, kernel_size)
            ))

        # Output layer (moved after decoder part)
        self.final_conv = nn.Conv1d(model_width, output_nums, 1)
        self.final_activation = nn.Softmax(dim=1) if problem_type == 'Classification' else nn.Identity()

    def extract_input(self, batch_dict: Dict[str, torch.Tensor], source_channel: int = 0) -> torch.Tensor:
        """
        Extract and prepare input for ShallowUNet from batch dict.
        
        This method handles the input processing for ShallowUNet, including:
        - Waveform channel extraction
        - Shape formatting to [B, 1, L] for UNet
        
        Args:
            batch_dict: Standardized batch dict from DataLoader collate_fn
            source_channel: Source channel index to extract
        
        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        # waveform = batch_dict["waveform"]
        # x = waveform[:, source_channel:source_channel+1, :]
        # Parent class validates required fields and extracts input
        x = super().extract_input(batch_dict)
        return x

    def forward(self, batch_dict: Dict[str, torch.Tensor], feature_extraction_only=False):        
        # Validate feature extraction settings
        if feature_extraction_only and not self.A_E:
            raise ValueError("feature_extraction_only requires ae (AutoEncoder) to be enabled")
        
        x = self.extract_input(batch_dict)
        
        # Store skip connections
        skips = []
        
        # Encoder path
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skips.append(x)
            x = self.pool(x)
            
        # AutoEncoder feature extraction
        if self.A_E and self.feature_extraction is not None:
            x = self.feature_extraction(x, feature_only=feature_extraction_only)
            # Return early if only feature extraction is needed
            if feature_extraction_only:
                return x
                
        # Bottleneck
        x = self.bottleneck(x)
        
        # Store deep supervision outputs
        ds_outputs = []
        
        # Decoder path
        skips = skips[::-1]  # Reverse for easier access
        
        for i in range(self.model_depth):
            # 1. Deep supervision
            if self.D_S and self.ds_convs is not None:
                ds_outputs.append(self.ds_convs[i](x))
                
            # 2. Upsampling
            x = self.up_blocks[i](x)
            
            # 3. Get skip connection
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
            return ds_outputs[::-1]
        
        if outputs.shape[-1] == 1280:
            sbp, dbp = extract_bp_values(outputs[:, :, 15:-15])
        else:
            sbp, dbp = extract_bp_values(outputs)
            
        return {"y_pred_waveform": outputs, "y_pred_sbp": sbp, "y_pred_dbp": dbp}


class ShallowUNetBP(BaseModel):
    """
    ShallowUNet model combining UNet for waveform prediction and MLPs for BP prediction
    """
    
    def __init__(self, feature_number=1024, hidden_size=100, 
                 model_depth=1, num_channel=1,
                 model_width=128, kernel_size=3, problem_type='Regression', 
                 output_nums=1, ds=0, ae=1, ag=0, is_transconv=False,
                 mlp_activation='relu', mlp_alpha=0.0001,
                 feature_extraction_only=False, alpha=1, lstm=0, 
                 *args, **kwargs):
        """Initialize ShallowUNet
        
        Args:
            feature_number (int): Number of features to extract
            hidden_size (int): Hidden layer size for MLPs
            model_depth (int): Depth of UNet model (default=1 for ShallowUNet)
            num_channel (int): Number of input channels
            model_width (int): Width of UNet model
            kernel_size (int): Kernel size for convolutions
            problem_type (str): Type of problem ('Regression' or 'Classification')
            output_nums (int): Number of output channels
            ds (int): Deep supervision flag
            ae (int): AutoEncoder mode flag
            ag (int): Attention guided flag
            is_transconv (bool): Use transposed convolutions flag
            mlp_activation (str): Activation function for MLPs
            mlp_alpha (float): Alpha parameter for MLPs
            feature_extraction_only (bool): Whether to only extract features
            alpha (float): Alpha parameter for UNet
            lstm (int): LSTM flag for UNet
        """
        # input_length is passed through *args or **kwargs from parent
        super().__init__(*args, **kwargs)
        
        # Store configuration parameters as attributes
        self.feature_number = feature_number
        self.hidden_size = hidden_size
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.ds = ds
        self.ae = ae
        self.ag = ag
        self.is_transconv = is_transconv
        self.mlp_activation = mlp_activation
        self.mlp_alpha = mlp_alpha
        self.feature_extraction_only = feature_extraction_only
        self.alpha = alpha
        self.lstm = lstm
        
        # Initialize ShallowUNet (UNet with depth=1)
        self.unet = ShallowUNet(
            input_length=self.input_length,  # UPDATED
            model_depth=self.model_depth,
            num_channel=self.num_channel,
            model_width=self.model_width,
            kernel_size=self.kernel_size,
            problem_type=self.problem_type,
            output_nums=self.output_nums,
            ds=self.ds,
            ae=1,  # Always enable AutoEncoder mode for feature extraction
            ag=self.ag,
            is_transconv=self.is_transconv,
            feature_number=self.feature_number,
            alpha=self.alpha,
            lstm=self.lstm
        )
        
        # Shared MLP configuration to avoid code duplication
        mlp_config = {
            'input_size': self.feature_number,
            'hidden_size': self.hidden_size,
            'activation': self.mlp_activation,
            'alpha': self.mlp_alpha
        }
        
        # Initialize MLPs for SBP and DBP prediction (separate instances with identical architecture)
        self.mlp_sbp = ShallowMLPRegressor(**mlp_config)
        self.mlp_dbp = ShallowMLPRegressor(**mlp_config)
        
    def extract_input(self, batch_dict: Dict[str, torch.Tensor], source_channel: int = 0) -> torch.Tensor:
        """
        Extract and prepare input for ShallowUNet from batch dict.
        
        This method handles the input processing for ShallowUNet, including:
        - Waveform channel extraction
        - Shape formatting to [B, 1, L] for UNet
        
        Args:
            batch_dict: Standardized batch dict from DataLoader collate_fn
            source_channel: Source channel index to extract
        
        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        waveform = batch_dict["waveform"]
        x = waveform[:, source_channel:source_channel+1, :]
        return x

    def forward(self, batch_dict: Dict[str, torch.Tensor], source_channel: int = 0, return_abp=False):
        """
        Forward pass of the ShallowUNet model.
        
        Args:
            batch_dict (dict): Standardized batch dict from DataLoader collate_fn
            source_channel (int): Source channel index to extract
            return_abp (bool): Whether to return waveform prediction instead of BP values
        
        Returns:
            torch.Tensor: Waveform prediction if return_abp=True
            tuple: (sbp, dbp) if return_abp=False
        """
        x = self.extract_input(batch_dict, source_channel)
        if return_abp:
            waveform = self.predict_waveform(x)
            return waveform
        sbp, dbp = self.predict_bp(x)
        return sbp, dbp

    def predict_waveform(self, x):
        """Predict waveform from input signal"""
        waveform = self.unet(x, feature_extraction_only=False)
        return waveform
    
    def extract_features(self, x):
        """Extract features from input signal"""
        with torch.no_grad():
            features = self.unet(x, feature_extraction_only=True)
        return features
    
    def predict_bp(self, x):
        """Predict BP values from input signal"""
        with torch.no_grad():
            features = self.extract_features(x)
        sbp = self.mlp_sbp(features.detach())
        dbp = self.mlp_dbp(features.detach())
        return sbp, dbp

cs = ConfigStore.instance()
cs.store(name="base_nabnet", group="model", node=NABNetModelConfig)
cs.store(name="base_shallow_unet", group="model", node=ShallowUNetModelConfig)