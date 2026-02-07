# MDViSCo: Unified Approximation and Refinement Models
# This file merges the Approximation (UNet-Swin Transformer) and Refinement (BP Estimation) models for MDViSCo.
#
# Sections:
#   1. Imports (deduplicated)
#   2. Model Configuration Classes
#   3. Approximation Model (UNet_SwinUnet and related classes)
#   4. Refinement Model (BPModel and related classes)

# =====================
# 1. Imports
# =====================

# Standard library imports
from collections import OrderedDict
from typing import Optional, Union, List, Dict
from dataclasses import dataclass, field

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from hydra.core.config_store import ConfigStore
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    PatchTSMixerConfig,
    PatchTSMixerModel
)

# Local imports
from src.criterions.weighted_contrastive_loss import WeightedContrastiveLoss, WeightedContrastiveLossConfig 
from omegaconf import MISSING
from .base_model import BaseModel, BaseModelConfig

# =====================
# 2. Model Configuration Classes
# =====================

@dataclass
class UNet_SwinUnetConfig(BaseModelConfig):
    """UNet-SwinUnet Model Configuration - Centralized model architecture parameters
    
    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the UNet_SwinUnet model, independently of training or dataset.
    
    These are the core tunable model hyperparameters that define the layout and complexity
    of the approximation network (UNet_SwinUnet).
    """
    _target_: str = "src.model.MDViSCo.UNet_SwinUnet"
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
    
    upsample_scale: List[int] = field(default_factory=lambda: [4])
    """Scaling factor(s) used for upsampling layers (e.g., [4] to upscale 4×)."""
    
    # Additional model parameters
    style_dim: int = 3
    """Dimension of the style vector for AdaIN."""
    
    # REMOVED: input_size field - now inherited from BaseModelConfig
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
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
        if self.input_length <= 0:  # UPDATED
            raise ValueError("input_length must be positive")
        if not self.upsample_scale:
            raise ValueError("upsample_scale cannot be empty")
        if any(scale <= 0 for scale in self.upsample_scale):
            raise ValueError("All values in upsample_scale must be positive")

@dataclass
class BPModelConfig(BaseModelConfig):
    """Blood Pressure Model Configuration - Centralized model architecture parameters
    
    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the BPModel (refinement model), independently of training or dataset.
    
    These are the core tunable model hyperparameters that define the layout and complexity
    of the blood pressure estimation network.
    """
    _target_: str = "src.model.MDViSCo.BPModel"
    supports_multi_directional: bool = False  # BPModel only supports single-direction (refinement)
    model_name: str = "BPModel"
    
    # Model Architecture Configuration Attributes
    # These directly influence the model's internal design:
    
    pi: bool = True
    """Whether to use patient information (affects presence of text encoder or fusion block)."""
    
    image_embedding: Optional[int] = None
    """Size of the image (waveform) feature vector. If None, computed from d_model and w_length."""
    
    text_embedding: int = 768
    """Size of the text (demographic) feature vector (e.g., 768 for DistilBERT)."""
    
    wcl_age_threshold: Optional[float] = 0.2231
    """Threshold for soft similarity calculation in WCL (indirectly architecture-affecting)."""
    
    wcl: bool = True
    """Whether to use weighted contrastive loss (affects architecture via loss path)."""
    
    w_length: int = 1024
    """Length of waveform input."""
    
    normalized_bp: bool = True
    """Whether BP values are normalized."""
    
    projection_dim: int = 512
    """Output dimension after projecting image and text embeddings."""
    
    dropout: float = 0.1
    """Dropout probability used in projection layers."""
    
    d_model: int = 64
    """Hidden dimension used for transformer or attention layers—usually impacts fusion model size."""
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        if self.text_embedding <= 0:
            raise ValueError("text_embedding must be positive")
        if self.w_length <= 0:
            raise ValueError("w_length must be positive")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.wcl_age_threshold is not None and self.wcl_age_threshold < 0:
            raise ValueError("wcl_age_threshold must be non-negative if provided")

# =====================
# 3. Approximation Model (UNet_SwinUnet and related classes)
# =====================

# --- Begin Approximation Model code ---

class UNet_SwinUnet(BaseModel):
    """
    A hybrid UNet-Swin Transformer architecture for signal processing with adaptive instance normalization.
    
    This model combines the hierarchical feature extraction of UNet with the self-attention mechanism
    of Swin Transformer, enhanced with adaptive instance normalization (AdaIN) for style transfer capabilities.
    
    Architecture:
    - Encoder: Series of convolutional blocks with downsampling
    - Bottleneck: Swin Transformer with AdaIN for feature transformation
    - Decoder: Series of AdaIN residual blocks with upsampling
    - Skip connections between encoder and decoder
    
    Args:
        in_channels (int): Number of input channels. Default: 3
        out_channels (int): Number of output channels. Default: 1
        init_features (int): Initial number of features in the first layer. Default: 32
        kernel_size (int): Kernel size for convolutional layers. Default: 5
        style_dim (int): Dimension of the style vector for AdaIN. Default: 64
        upsample_scale (list): List of upsampling scales for the Swin Transformer. Default: [2, 2, 2]
        input_length (int): Length of the input signal. Default: 1024
        patch_size (int): Size of patches for the Swin Transformer. Default: 5
        depth (int): Depth of the UNet architecture. Default: 5
    """
    def __init__(self, in_channels=3, out_channels=1, init_features=32, kernel_size=5, style_dim=64, upsample_scale=[2, 2, 2], patch_size=5, depth=5, *args, **kwargs):
        super(UNet_SwinUnet, self).__init__(*args, **kwargs)
        
        # Initialize base features and initial convolution
        features = init_features
        self.features = init_features
        self.conv_init_features = nn.Conv1d(in_channels, features, 3, 1, 1)

        # Build encoder with downsampling blocks
        self.encoder = nn.ModuleList()
        self.depth = depth
        w_size = self.input_length  # UPDATED: use self.input_length
        for i in range(0,depth):
            # First convolutional block
            self.encoder.append(UNet_SwinUnet._block(features, features, k=kernel_size, name="enc"+str(i)+"_1"))
            # Downsampling convolution
            self.encoder.append(nn.Conv1d(in_channels=features, out_channels=features, kernel_size=2,
                      stride=2))
            # Second convolutional block with doubled features
            self.encoder.append(UNet_SwinUnet._block(features*2, features*2, k=kernel_size, name="enc" + str(i) + "_2"))

            # Double features and halve spatial dimensions for next layer
            features = features * 2
            w_size = w_size // 2

        # Configure Swin Transformer bottleneck
        in_chans = features
        patch_size = patch_size
        embed_dim = (in_chans * 4)  # Embedding dimension for transformer

        window_size = patch_size
        num_heads = [32, 32, 32, 32, 32]  # Number of attention heads for each layer
        num_classes = embed_dim

        # Initialize Swin Transformer with AdaIN
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
            mlp_ratio=4., 
            qkv_bias=True, 
            qk_scale=None,
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.1, 
            norm_layer_encoder=nn.InstanceNorm1d,
            norm_layer_decoder=AdaIN, 
            ape=False, 
            patch_norm=True,
            use_checkpoint=False, 
            final_upsample="expand_first",
            style_dim=style_dim
        )
        w_size = self.input_length  # UPDATED: use self.input_length

        # Build decoder with AdaIN residual blocks
        self.decoder = nn.ModuleList()
        for i in range(0, depth):
            if i == 0:
                # Match Version 1 logic for first decoder block
                if len(upsample_scale) > 1:
                    self.decoder.append(
                        AdainResBlk(dim_in=in_chans * 2, dim_out=features // 2, upsample=True, style_dim=style_dim,
                                    upsample_scale=2, k=3))
                    self.decoder.append(
                        AdainResBlk(dim_in=features, dim_out=features // 2, upsample=False, style_dim=style_dim,
                                    upsample_scale=2, k=3))
                else:
                    self.decoder.append(
                        AdainResBlk(dim_in=in_chans, dim_out=features // 2, upsample=True, style_dim=style_dim,
                                    upsample_scale=2, k=3))
                    self.decoder.append(
                        AdainResBlk(dim_in=features, dim_out=features // 2, upsample=False, style_dim=style_dim,
                                    upsample_scale=2, k=3))
            else:
                self.decoder.append(
                    AdainResBlk(dim_in=features, dim_out=features // 2, upsample=True, style_dim=style_dim,
                                upsample_scale=2, k=3))
                self.decoder.append(
                    AdainResBlk(dim_in=features, dim_out=features // 2, upsample=False, style_dim=style_dim,
                                upsample_scale=2, k=3))
            features = features // 2  # Halve features for next layer

        # Final output layers
        self.last = nn.Sequential(
            nn.InstanceNorm1d(features, affine=True),  # Normalize final features
            nn.LeakyReLU(0.2),  # Non-linear activation
            nn.Conv1d(  # Final convolution to output channels
                in_channels=features, 
                out_channels=out_channels, 
                kernel_size=1, 
                padding=0, 
                bias=False
            )
        )

    def extract_domain_target(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract domain shift target for MDViSCo.
        
        MDViSCo-specific functionality for generating one-hot domain vectors.
        
        Args:
            batch_dict: Unified batch dict with tgt_idxs
            
        Returns:
            torch.Tensor: One-hot encoded domain vectors [B, D]
        """
        device = batch_dict["x"].device
        tgt_idxs = batch_dict["tgt_idxs"].to(device=device, dtype=torch.long)  # [B]
        
        # Get number of domains from model config
        num_domains = int(batch_dict["x"].shape[1])
        
        # Generate one-hot encoding
        domain_shift_target = F.one_hot(tgt_idxs, num_classes=num_domains).to(
            device=device, dtype=torch.float32
        )  # [B, D]
        
        return domain_shift_target

    def extract_input(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract input and domain target for MDViSCo.
        
        Combines common input extraction with MDViSCo-specific domain extraction.
        """
        # Get common input extraction from parent class
        x = super().extract_input(batch_dict)
        
        # Add MDViSCo-specific domain extraction
        domain_shift_target = self.extract_domain_target(batch_dict)
        
        return {
            "x": x,
            "domain_shift_target": domain_shift_target
        }

    def forward(self, batch_dict, s=None):
        """
        Forward pass through the UNet-Swin Transformer network.
        
        The forward pass consists of three main stages:
        1. Encoder: Progressive downsampling with feature extraction
        2. Bottleneck: Swin Transformer processing with style conditioning
        3. Decoder: Progressive upsampling with skip connections
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_size)
            s (torch.Tensor, optional): Style vector for AdaIN conditioning. Shape: (batch_size, style_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, input_size)
        """
        batch_extract = self.extract_input(batch_dict)
        # Check if input is a dict (from extract_input)
        if isinstance(batch_extract, dict):
            # Extract components from dict
            x = batch_extract["x"]
            # Use domain_shift_target as style vector
            s = batch_extract["domain_shift_target"]
        
        # Initial feature extraction
        enc = self.conv_init_features(x)
        encoder = []  # Store encoder features for skip connections
        
        # Encoder path: Progressive downsampling
        for i in range(0, len(self.encoder), 3):
            # Store current features for skip connection
            encoder.insert(0, enc)
            
            # First convolutional block
            enc = self.encoder[i](enc)

            # Downsampling through two paths:
            # 1. Max pooling
            x_down = F.max_pool1d(enc, 2)
            # 2. Convolutional downsampling
            x_conv_pool = self.encoder[i+1](enc)
            # Concatenate both downsampled features
            enc = torch.cat([x_down, x_conv_pool], dim=1)

            # Second convolutional block
            enc = self.encoder[i+2](enc)

        # Bottleneck: Swin Transformer processing with style conditioning
        bottleneck = self.bottleneck(enc, s)

        # Decoder path: Progressive upsampling with skip connections
        dec = bottleneck
        index = 0

        for i in range(0, len(self.decoder), 2):
            # First AdaIN residual block with upsampling
            dec = self.decoder[i](dec, s)
            
            # Add skip connection if available
            if index < len(encoder):
                dec = torch.cat([dec, encoder[index]], dim=1)
                
            # Second AdaIN residual block
            dec = self.decoder[i+1](dec, s)
            index += 1

        # Final output processing
        return {"y_pred": self.last(dec)}

    @staticmethod
    def load_checkpoint(model, checkpoint_path):
        """
        Load model weights from a checkpoint file, handling DataParallel prefix properly.
        
        This method handles the common issue where model weights saved with DataParallel
        have a 'module.' prefix that needs to be removed when loading into a non-parallel model.
        
        Args:
            model (nn.Module): The model instance to load weights into
            checkpoint_path (str): Path to the checkpoint file containing saved weights
            
        Returns:
            nn.Module: The model with loaded weights
            
        Note:
            The checkpoint file should contain a dictionary with 'model_G_state_dict' key
            containing the model's state dictionary.
        """
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_G_state_dict']
        
        # Process state dict to handle DataParallel prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present (added by DataParallel)
            if k.startswith('module.'):
                name = k[7:]  # remove 'module.' prefix
            else:
                name = k
            new_state_dict[name] = v
            
        # Load the processed state dict into the model
        model.load_state_dict(new_state_dict)
        return model
    
    @staticmethod
    def _block_Conv(in_channels, features, k, name, input_size):
        """
        Create a basic convolutional block with a single 1D convolution layer.
        
        This method creates a simple sequential block containing a single convolutional layer
        with proper padding to maintain the input size. The block is used as a building block
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
                            out_channels=features,   # Output features
                            kernel_size=k,           # Kernel size
                            padding=k // 2,          # Padding to maintain size
                            bias=False,              # Disable bias
                        ),
                    )
                ]
            )
        )
                
    @staticmethod
    def _block_LN(in_channels, features, k, name, input_size):
        """
        Create a normalization block with LayerNorm and LeakyReLU activation.
        
        This method creates a sequential block containing a layer normalization layer
        followed by a LeakyReLU activation. The block is used for normalizing and
        activating features in the network.
        
        Args:
            in_channels (int): Number of input channels (not used in this block)
            features (int): Number of features to normalize
            k (int): Kernel size (not used in this block)
            name (str): Base name for the layers (will be appended with 'norm1' and 'relu1')
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
                    (name + "relu1", nn.LeakyReLU(inplace=True)),  # LeakyReLU activation
                ]
            )
        )
        
    @staticmethod
    def _block(in_channels, features, k, name):
        """
        Create a double-convolution block with instance normalization and LeakyReLU activation.
        
        This method creates a sequential block containing two convolutional layers, each followed by
        instance normalization and LeakyReLU activation. This is a common building block in the
        UNet architecture, used for feature extraction and transformation.
        
        The block structure is:
        1. First convolution + normalization + activation
        2. Second convolution + normalization + activation
        
        Args:
            in_channels (int): Number of input channels for the first convolution
            features (int): Number of output features for both convolutions
            k (int): Kernel size for both convolutions
            name (str): Base name for the layers (will be appended with conv1/2, norm1/2, relu1/2)
            
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
                            out_channels=features,   # Output features
                            kernel_size=k,           # Kernel size
                            padding=k//2,            # Padding to maintain size
                            bias=False,              # Disable bias
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm1d(num_features=features)),  # First normalization
                    (name + "relu1", nn.LeakyReLU(inplace=True)),               # First activation
                    
                    # Second convolution block
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,    # Input features (same as output of first conv)
                            out_channels=features,   # Output features
                            kernel_size=k,           # Kernel size
                            padding=k//2,            # Padding to maintain size
                            bias=False,              # Disable bias
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm1d(num_features=features)),  # Second normalization
                    (name + "relu2", nn.LeakyReLU(inplace=True))                # Second activation
                ]
            )
        )

class ResBlk(nn.Module):
    """
    Residual Block for 1D signal processing with optional normalization and downsampling.
    
    This block implements a residual connection architecture with the following features:
    - Optional instance normalization
    - Optional downsampling through dual-path (max pooling + convolution)
    - LeakyReLU activation
    - Residual connection for better gradient flow
    
    Args:
        dim_in (int): Number of input channels
        dim_out (int): Number of output channels
        actv (nn.Module): Activation function. Default: LeakyReLU(0.2)
        normalize (bool): Whether to use instance normalization. Default: False
        downsample (bool): Whether to downsample the input. Default: False
    """
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        # Store configuration parameters
        self.actv = actv                # Activation function
        self.normalize = normalize      # Whether to use normalization
        self.downsample = downsample    # Whether to downsample
        
        # Build the network weights
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        """
        Build the network weights for the residual block.
        
        This method creates the convolutional layers and normalization layers based on the
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
            
        # Add normalization layers if enabled
        if self.normalize:
            # First normalization layer
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            if self.downsample:
                # Second normalization with doubled channels if downsampling
                self.norm2 = nn.InstanceNorm1d(2 * dim_in, affine=True)
            else:
                # Second normalization with same channels if no downsampling
                self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)

        # Add downsampling convolution if enabled
        if self.downsample:
            # Convolution for downsampling in residual path
            self.conv_pool_residual = nn.Conv1d(
                in_channels=dim_in, 
                out_channels=dim_in, 
                kernel_size=2,
                stride=2
            )

    def _residual(self, x):
        """
        Process input through the residual block's main path.
        
        This method implements the main processing path of the residual block, which includes:
        1. Optional normalization and activation
        2. First convolution
        3. Optional downsampling through dual-path (max pooling + convolution)
        4. Optional normalization and activation
        5. Second convolution
        
        The downsampling path combines features from both max pooling and convolutional downsampling
        to preserve more information during the downsampling process.
        
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

    def forward(self, x):
        """
        Forward pass through the residual block.
        
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
        # Process input through the residual path
        x = self._residual(x)
        return x

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer for style transfer.
    
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
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        # Instance normalization without learnable parameters
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        
        # Fully connected layer to generate adaptive parameters
        # Outputs 2*num_features parameters (gamma and beta for each feature)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        """
        Forward pass of the AdaIN layer.
        
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
            torch.Tensor: Style-modulated features of shape (batch_size, num_features, length)
        """
        # Generate style parameters from the style vector
        h = self.fc(s)
        
        # Reshape to match feature dimensions (batch_size, num_features*2, 1)
        h = h.view(h.size(0), h.size(1), 1)
        
        # Split into gamma and beta parameters
        # gamma: scaling parameters, beta: shifting parameters
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        
        # Apply adaptive instance normalization:
        # 1. Normalize features using instance normalization
        # 2. Scale by (1 + gamma) to preserve some original feature statistics
        # 3. Add style-specific bias (beta)
        return (1 + gamma) * self.norm(x) + beta

class AdainResBlk(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) Residual Block for style transfer.
    
    This block implements a residual connection architecture with AdaIN for style transfer,
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
        actv (nn.Module): Activation function. Default: LeakyReLU(0.2)
        upsample (bool): Whether to upsample the input. Default: False
        generator (bool): Whether to use AdaIN (True) or standard normalization (False). Default: True
        upsample_scale (int): Scale factor for upsampling. Default: 2
    """
    def __init__(self, dim_in, dim_out, k=3, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False, generator=True, upsample_scale=2):
        super(AdainResBlk, self).__init__()
        # Store configuration parameters
        self.generator = generator    # Whether to use AdaIN or standard normalization
        self.k = k                    # Kernel size for convolutions
        self.w_hpf = w_hpf           # High-pass filter width
        self.actv = actv             # Activation function
        self.upsample = upsample     # Whether to upsample
        self.upsample_scale = upsample_scale  # Scale factor for upsampling
        
        # Build the network weights (convolutional and normalization layers)
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        """
        Build the network weights for the AdaIN residual block.
        
        This method creates the convolutional layers, normalization layers, and upsampling
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
            self.conv1 = nn.Conv1d(dim_in*2, dim_out, self.k, 1, self.k//2)
        else:
            # Without upsampling, input is same as specified
            self.conv1 = nn.Conv1d(dim_in, dim_out, self.k, 1, self.k//2)

        # Second convolution layer
        self.conv2 = nn.Conv1d(dim_out, dim_out, self.k, 1, self.k//2)
        
        # Add normalization layers based on generator mode
        if self.generator:
            # AdaIN layers for style transfer
            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_out)
        else:
            # Standard instance normalization
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_out, affine=True)
            
        # Add transposed convolution for upsampling if enabled
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
                bias=False
            )

    def _residual(self, x, s):
        """
        Process input through the residual block's main path.
        
        This method implements the main processing path of the AdaIN residual block, which includes:
        1. First normalization and activation
        2. Optional upsampling through dual-path (nearest neighbor + transposed convolution)
        3. First convolution
        4. Second normalization and activation
        5. Final convolution
        
        The upsampling path combines features from both nearest neighbor interpolation and
        transposed convolution to preserve more information during the upsampling process.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in, length)
            s (torch.Tensor): Style vector of shape (batch_size, style_dim) for AdaIN
            
        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, dim_out, length) or
                         (batch_size, dim_out, length*upsample_scale) if upsampling is enabled
        """
        # First normalization and activation
        if self.generator:
            # Apply AdaIN with style vector
            x = self.norm1(x, s)
        else:
            # Apply standard instance normalization
            x = self.norm1(x)
        x = self.actv(x)
        
        # Upsampling through dual-path if enabled
        if self.upsample:
            # Path 1: Nearest neighbor upsampling
            x_up = F.interpolate(x, scale_factor=self.upsample_scale, mode='nearest')
            # Path 2: Transposed convolution upsampling
            x_trans = self.transpose_residual(x)
            # Combine features from both paths
            x = torch.cat([x_up, x_trans], dim=1)
            
        # First convolution
        x = self.conv1(x)
        
        # Second normalization and activation
        if self.generator:
            # Apply AdaIN with style vector
            x = self.norm2(x, s)
        else:
            # Apply standard instance normalization
            x = self.norm2(x)
        x = self.actv(x)
        
        # Final convolution
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        """
        Forward pass through the AdaIN residual block.
        
        This method implements the forward pass of the residual block, which processes
        the input through the main residual path. The residual connection is implemented
        in the parent network architecture.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in, length)
            s (torch.Tensor): Style vector of shape (batch_size, style_dim) for AdaIN
            
        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, dim_out, length) or
                         (batch_size, dim_out, length*upsample_scale) if upsampling is enabled
                         
        Note:
            The actual residual connection (adding the input to the output) is handled
            by the parent network architecture, not within this block. This method only
            implements the main processing path through the block.
        """
        # Process input through the residual path
        out = self._residual(x, s)
        return out

class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with optional dropout.
    
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
        hidden_features (int, optional): Number of hidden features. If None, uses in_features
        out_features (int, optional): Number of output features. If None, uses in_features
        act_layer (nn.Module, optional): Activation function. Default: nn.GELU
        drop (float, optional): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # Set output and hidden dimensions if not specified
        out_features = out_features or in_features  # Use input dimension if output not specified
        hidden_features = hidden_features or in_features  # Use input dimension if hidden not specified
        
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # Activation function
        self.act = act_layer()
        
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # Dropout layer for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass through the MLP network.
        
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
        # First fully connected layer
        x = self.fc1(x)
        
        # Apply activation function
        x = self.act(x)
        
        # Apply dropout after activation
        x = self.drop(x)
        
        # Second fully connected layer
        x = self.fc2(x)
        
        # Apply final dropout
        x = self.drop(x)
        
        return x


def window_partition(x, window_size):
    """
    Partition input tensor into non-overlapping windows for window-based attention.
    
    This function takes a 1D input tensor and divides it into non-overlapping windows
    of specified size. This is a key operation in the Swin Transformer architecture
    that enables local attention computation within windows.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, W, C) where:
            - B: Batch size
            - W: Sequence length (width)
            - C: Number of channels/features
        window_size (int): Size of each window. Must divide W evenly.
    
    Returns:
        torch.Tensor: Reshaped tensor of shape (num_windows*B, window_size, C) where:
            - num_windows = W // window_size
            - Each window contains window_size consecutive elements
            - Windows are flattened across the batch dimension
    
    Example:
        If input shape is (4, 100, 64) and window_size=10:
        - Creates 10 windows per sequence (100/10)
        - Output shape will be (40, 10, 64) (4 batches * 10 windows)
    """
    # Get input dimensions
    B, W, C = x.shape
    
    # Reshape input to separate windows:
    # (B, W, C) -> (B, W//window_size, window_size, C)
    # This creates a new dimension for window_size
    x = x.view(B, W // window_size, window_size, C)
    
    # Permute and reshape to flatten windows across batch dimension:
    # (B, W//window_size, window_size, C) -> (B*W//window_size, window_size, C)
    # This makes it easier to process all windows in parallel
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    
    return windows


def window_reverse(windows, window_size, W):
    """
    Reverse the window partitioning operation to reconstruct the original input tensor.
    
    This function takes windowed features and reconstructs them back into the original
    sequence format. It is the inverse operation of window_partition and is used after
    processing windows in the Swin Transformer architecture.
    
    Args:
        windows (torch.Tensor): Windowed features of shape (num_windows*B, window_size, C) where:
            - num_windows: Number of windows per sequence (W/window_size)
            - B: Batch size
            - window_size: Size of each window
            - C: Number of channels/features
        window_size (int): Size of each window. Must divide W evenly.
        W (int): Original sequence length (width) of the input tensor.
    
    Returns:
        torch.Tensor: Reconstructed tensor of shape (B, W, C) where:
            - B: Batch size
            - W: Original sequence length
            - C: Number of channels/features
    
    Example:
        If windows shape is (40, 10, 64), window_size=10, and W=100:
        - Input represents 4 batches (40/10 windows)
        - Output shape will be (4, 100, 64)
    """
    # Calculate batch size from windows shape and sequence length
    # num_windows = W/window_size, so B = windows.shape[0] / num_windows
    B = int(windows.shape[0] / (W / window_size))
    
    # Reshape windows back to original sequence format:
    # (num_windows*B, window_size, C) -> (B, W//window_size, window_size, C)
    # This separates windows back into their original positions
    x = windows.view(B, W // window_size, window_size, -1)
    
    # Permute and reshape to reconstruct original sequence:
    # (B, W//window_size, window_size, C) -> (B, W, C)
    # This combines windows back into continuous sequences
    x = x.permute(0, 1, 2, 3).contiguous().view(B, W, -1)
    
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Initialize the Window-based Multi-head Self-Attention (W-MSA) module.
        
        This module implements attention computation within local windows, which is a key
        component of the Swin Transformer architecture. It includes relative position bias
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
        
        # Store basic configuration
        self.dim = dim                # Input dimension
        self.window_size = window_size  # Size of attention window
        self.num_heads = num_heads    # Number of attention heads
        
        # Calculate dimension per head
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # Scaling factor for attention scores

        # Initialize relative position bias table
        # Shape: (2*window_size-1, num_heads)
        # This table stores learnable position biases for each relative position
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))

        # Generate relative position indices for all positions in the window
        # This creates a mapping from relative positions to indices in the bias table
        coords_w = torch.arange(self.window_size)  # [0, 1, ..., window_size-1]
        coords = torch.stack(torch.meshgrid([coords_w], indexing='ij'))  # [1, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [1, window_size]
        
        # Calculate relative positions between all pairs of positions
        # Shape: [window_size, window_size, 1]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        # Shift indices to be non-negative
        relative_coords[:, :, 0] += self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [window_size, window_size]
        self.register_buffer("relative_position_index", relative_position_index)

        # Initialize QKV projection layer
        # Projects input to query, key, and value vectors
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Initialize dropout layers
        self.attn_drop = nn.Dropout(attn_drop)  # For attention weights
        self.proj = nn.Linear(dim, dim)         # Output projection
        self.proj_drop = nn.Dropout(proj_drop)  # For output

        # Initialize relative position bias table with truncated normal distribution
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        # Initialize softmax for attention score normalization
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Forward pass of the Window-based Multi-head Self-Attention (W-MSA) module.
        
        This method implements the attention computation within local windows, including:
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
        # Get input dimensions
        B_, N, C = x.shape  # B_ = num_windows*B
        
        # Project input to query, key, and value vectors
        # Shape: (num_windows*B, N, 3*C) -> (3, num_windows*B, num_heads, N, C//num_heads)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separate Q, K, V
        
        # Scale query vectors for stable attention scores
        q = q * self.scale
        
        # Compute attention scores: Q @ K^T
        # Shape: (num_windows*B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias to attention scores
        # Shape: (window_size, window_size, num_heads) -> (num_heads, window_size, window_size)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask if provided
        if mask is not None:
            nW = mask.shape[0]  # number of windows
            # Reshape attention scores to separate windows
            # Shape: (num_windows*B, num_heads, N, N) -> (B, num_windows, num_heads, N, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # Reshape back to original form
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Normalize attention scores with softmax
        attn = self.softmax(attn)
        
        # Apply dropout to attention weights
        attn = self.attn_drop(attn)
        
        # Compute weighted sum of values
        # Shape: (num_windows*B, num_heads, N, C//num_heads) -> (num_windows*B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Project to output dimension and apply dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def extra_repr(self) -> str:
        """
        Generate a string representation of the WindowAttention module's key parameters.
        
        This method is used by PyTorch's print() and str() functions to display
        important configuration parameters of the module. It helps in debugging
        and model inspection by showing the module's configuration at a glance.
        
        Returns:
            str: A formatted string containing the module's key parameters:
                - dim: Input/Output dimension
                - window_size: Size of attention window
                - num_heads: Number of attention heads
        """
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """
        Calculate the number of floating-point operations (FLOPs) for the attention computation.
        
        This method estimates the computational complexity of the attention mechanism
        for a given number of tokens. It's useful for analyzing the model's efficiency
        and comparing different architectures.
        
        Args:
            N (int): Number of tokens in the input sequence (window_size)
            
        Returns:
            int: Total number of FLOPs for one window's attention computation, including:
                - QKV projection: N * dim * 3 * dim
                - Attention matrix computation: num_heads * N * (dim/num_heads) * N
                - Attention-weighted sum: num_heads * N * N * (dim/num_heads)
                - Output projection: N * dim * dim
        """
        # Calculate FLOPs for each major operation
        
        # 1. QKV projection: input -> query, key, value vectors
        # Shape: (N, dim) -> (N, 3*dim)
        flops = N * self.dim * 3 * self.dim
        
        # 2. Attention matrix computation: Q @ K^T
        # Shape: (N, dim/num_heads) @ (dim/num_heads, N) -> (N, N)
        # Multiplied by num_heads for all attention heads
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        
        # 3. Attention-weighted sum: (N, N) @ (N, dim/num_heads) -> (N, dim/num_heads)
        # Multiplied by num_heads for all attention heads
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        
        # 4. Output projection: (N, dim) -> (N, dim)
        flops += N * self.dim * self.dim
        
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path: Union[float, List[float]] = 0.,
                 act_layer=nn.GELU, norm_layer=nn.Module, style_dim: Optional[int]=64):
        """
        Initialize a Swin Transformer Block with optional shifted window attention.
        
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
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default: 4.0
            qkv_bias (bool, optional): If True, add learnable bias to QKV. Default: True
            qk_scale (float, optional): Override default QK scale. Default: None
            drop (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop (float, optional): Dropout rate for attention. Default: 0.0
            drop_path (Union[float, List[float]], optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()
        
        # Store basic configuration
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
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        # Initialize normalization layers
        if style_dim is not None:
            # Use AdaIN for style transfer
            self.norm1 = norm_layer(style_dim, dim)
            self.norm2 = norm_layer(style_dim, dim)
        else:
            # Use standard layer normalization
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            
        # Initialize window attention module
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
        # Initialize stochastic depth
        if isinstance(drop_path, list):
            self.drop_path = DropPath(drop_path[0]) if drop_path[0] > 0. else nn.Identity()
        else:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Initialize MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
                      
        # Generate attention mask for shifted window attention
        if self.shift_size > 0:
            # Create base mask tensor
            W = self.input_resolution
            img_mask = torch.zeros((1, W, 1))  # [1, W, 1]
            
            # Define window slices for shifted attention
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
                       
            # Assign different values to each slice to create mask
            cnt = 0
            for w in w_slices:
                img_mask[:, w, :] = cnt
                cnt += 1
                
            # Partition mask into windows
            mask_windows = window_partition(img_mask, self.window_size)  # [nW, window_size, 1]
            mask_windows = mask_windows.view(-1, self.window_size)  # [nW, window_size]
            
            # Create attention mask by computing differences between positions
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # Convert differences to binary mask (0 for same window, -inf for different)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        # Register attention mask as buffer (not parameter)
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, s=None):
        """
        Forward pass of the Swin Transformer Block.
        
        This method implements the complete forward pass of the transformer block, including:
        1. Layer normalization
        2. Window-based attention (W-MSA) or shifted window attention (SW-MSA)
        3. Residual connection
        4. MLP processing
        5. Final residual connection
        
        The block can operate in two modes:
        - Regular mode (shift_size=0): Standard window-based attention
        - Shifted mode (shift_size>0): Attention with shifted windows for cross-window connections
        
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
            The input sequence length L must match the input_resolution specified in __init__.
        """
        # Get input dimensions and validate sequence length
        W = self.input_resolution
        B, L, C = x.shape
        assert L == W, "input feature has wrong size"
        
        # Store input for residual connection
        shortcut = x
        
        # First normalization and attention block
        if s is not None:
            # Apply AdaIN normalization with style vector
            x = self.norm1(x.transpose(1,2), s)
            x = x.transpose(1, 2)
        else:
            # Apply standard layer normalization
            x = self.norm1(x.transpose(1, 2))
            x = x.transpose(1, 2)
            
        # Reshape for window-based processing
        x = x.view(B, W, C)
        
        # Apply cyclic shift if using shifted window attention
        if self.shift_size > 0:
            # Shift the sequence by -shift_size positions
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x
            
        # Partition windows for attention computation
        # Shape: (B, W, C) -> (num_windows*B, window_size, C)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size, C)
        
        # Apply window attention
        # Shape: (num_windows*B, window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows back to sequence
        # Shape: (num_windows*B, window_size, C) -> (B, W, C)
        attn_windows = attn_windows.view(-1, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, W)
        
        # Reverse cyclic shift if using shifted window attention
        if self.shift_size > 0:
            # Shift the sequence back by shift_size positions
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x
            
        # Reshape back to original format
        x = x.view(B, W, C)
        
        # First residual connection
        x = shortcut + self.drop_path(x)
        
        # Second normalization and MLP block
        if s is not None:
            # Apply AdaIN normalization with style vector
            x = x + self.drop_path(self.mlp(self.norm2(x.transpose(1, 2), s).transpose(1, 2)))
        else:
            # Apply standard layer normalization
            x = x + self.drop_path(self.mlp(self.norm2(x.transpose(1, 2)).transpose(1, 2)))
            
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchMerging(nn.Module):
    """
    Patch Merging Layer for Swin Transformer.
    
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

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, style_dim: Optional[int]=64):
        """
        Initialize the Patch Merging layer.
        
        Args:
            input_resolution (int): Input sequence length
            dim (int): Number of input channels
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()
        
        # Store configuration
        self.input_resolution = input_resolution
        self.dim = dim
        
        # Initialize reduction layer
        # Projects concatenated features to new dimension
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        
        # Initialize normalization layer
        # Uses AdaIN if style_dim is provided, otherwise uses standard normalization
        norm_style_dim = style_dim if style_dim is not None else 64
        self.norm = norm_layer(norm_style_dim, 2 * dim)

    def forward(self, x, s):
        """
        Forward pass of the Patch Merging layer.
        
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
        # Get input dimensions and validate sequence length
        W = self.input_resolution
        B, L, C = x.shape
        assert L == W, "input feature has wrong size"
        assert W % 2 == 0, f"x size ({W}) are not even."
        
        # Reshape input for patch merging
        x = x.view(B, W, C)
        
        # Extract even and odd indexed patches
        x0 = x[:, 0::2, :]  # [B, W/2, C] - even indices
        x1 = x[:, 1::2, :]  # [B, W/2, C] - odd indices
        
        # Concatenate patches along feature dimension
        # Shape: [B, W/2, 2*C]
        x = torch.cat([x0, x1], -1)
        x = x.view(B, -1, 2 * C)
        
        # Apply normalization and projection
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        x = self.reduction(x)
        
        return x

    def extra_repr(self) -> str:
        """
        Generate a string representation of the PatchMerging layer's configuration.
        
        Returns:
            str: A formatted string containing the layer's key parameters:
                - input_resolution: Input sequence length
                - dim: Number of input channels
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """
        Calculate the number of floating-point operations (FLOPs) for the patch merging.
        
        Returns:
            int: Total number of FLOPs for the patch merging operation, including:
                - Feature concatenation
                - Normalization
                - Linear projection
        """
        H, W = self.input_resolution
        # FLOPs for feature concatenation
        flops = H * W * self.dim
        # FLOPs for linear projection
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    """
    Patch Expansion Layer for Swin Transformer.
    
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

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm, style_dim: Optional[int]=64):
        """
        Initialize the Patch Expansion layer.
        
        Args:
            input_resolution (int): Input sequence length
            dim (int): Number of input channels
            dim_scale (int, optional): Scale factor for dimension reduction. Default: 2
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()
        
        # Store configuration
        self.input_resolution = input_resolution
        self.dim = dim
        
        # Initialize expansion layer
        # If dim_scale >= 2, project to lower dimension, otherwise use identity
        self.expand = nn.Linear(dim, dim // dim_scale, bias=False) if dim_scale >= 2 else nn.Identity()
        
        # Initialize normalization layer
        # Uses AdaIN if style_dim is provided, otherwise uses standard normalization
        norm_style_dim = style_dim if style_dim is not None else 64
        self.norm = norm_layer(norm_style_dim, dim // (dim_scale*2))

    def forward(self, x, s):
        """
        Forward pass of the Patch Expansion layer.
        
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
        # Get input dimensions and validate sequence length
        W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == W, "input feature has wrong size"
        
        # Reshape input for patch expansion
        x = x.view(B, W, C)
        
        # Rearrange features to increase sequence length
        # Shape: [B, W, C] -> [B, W*2, C/2]
        x = rearrange(x, 'b w (p1 c)-> b (w p1) c', p1=2, c=C // 2)
        x = x.view(B, -1, C // 2)
        
        # Apply normalization
        x = self.norm(x.transpose(1,2), s).transpose(1,2)
        
        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Final Patch Expansion Layer for Swin Transformer with 4x upsampling.
    
    This layer implements the final upsampling operation in the Swin Transformer decoder,
    expanding patches by a factor of 4 to achieve the desired output resolution. It's
    specifically designed for the final stage of the decoder to match the input resolution.
    
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

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.Module, style_dim: Optional[int]=64):
        """
        Initialize the Final Patch Expansion layer.
        
        Args:
            input_resolution (int): Input sequence length
            dim (int): Number of input channels
            dim_scale (int, optional): Scale factor for dimension reduction. Default: 4
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()
        
        # Store configuration
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        
        # Initialize expansion layer
        # Projects input features to a new dimension
        self.expand = nn.Linear(dim, dim, bias=False)
        
        # Calculate output dimension after scaling
        self.output_dim = (dim // dim_scale)
        
        # Initialize normalization layer
        # Uses AdaIN if style_dim is provided, otherwise uses standard normalization
        self.norm = norm_layer(style_dim, self.output_dim)

    def forward(self, x, s):
        """
        Forward pass of the Final Patch Expansion layer.
        
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
        # Get input dimensions and validate sequence length
        W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == W, "input feature has wrong size"
        
        # Reshape input for patch expansion
        x = x.view(B, W, C)
        
        # Rearrange features to increase sequence length by 4x
        # Shape: [B, W, C] -> [B, W*4, C/4]
        x = rearrange(x, 'b w (p1 c)-> b (w p1) c', p1=self.dim_scale,
                      c=C // (self.dim_scale))
        x = x.view(B, self.output_dim, -1)
        
        # Apply normalization
        x = self.norm(x, s)
        
        return x


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer Layer for one stage in the encoder.
    
    This layer implements a complete stage of the Swin Transformer encoder, consisting of:
    1. Multiple Swin Transformer blocks with alternating window attention
    2. Optional patch merging for downsampling
    3. Support for gradient checkpointing to save memory
    
    The layer processes features through a series of transformer blocks, where each block
    alternates between regular and shifted window attention. This allows for both local
    and cross-window feature interactions.
    
    Args:
        dim (int): Number of input channels
        input_resolution (int): Input sequence length
        depth (int): Number of transformer blocks in this layer
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool, optional): If True, add learnable bias to QKV. Default: True
        qk_scale (float, optional): Override default QK scale. Default: None
        drop (float, optional): Dropout rate for MLP. Default: 0.0
        attn_drop (float, optional): Dropout rate for attention. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer. Default: None
        use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default: False
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path: Union[float, List[float]] = 0., norm_layer=nn.Module, downsample=None, use_checkpoint=False, style_dim: Optional[int]=64):
        """
        Initialize the Basic Swin Transformer Layer.
        
        Args:
            dim (int): Number of input channels
            input_resolution (int): Input sequence length
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            window_size (int): Size of attention window
            mlp_ratio (float, optional): MLP expansion ratio. Default: 4.0
            qkv_bias (bool, optional): Whether to use bias in QKV projection. Default: True
            qk_scale (float, optional): Scale factor for attention scores. Default: None
            drop (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop (float, optional): Dropout rate for attention. Default: 0.0
            drop_path (Union[float, List[float]], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            downsample (nn.Module | None, optional): Downsample layer. Default: None
            use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default: False
            style_dim (Optional[int], optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()
        
        # Store basic configuration
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build transformer blocks
        # Each block alternates between regular and shifted window attention
        self.blocks = nn.ModuleList([
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
                # Apply stochastic depth if specified
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, 
                style_dim=style_dim
            )
            for i in range(depth)
        ])

        # Initialize patch merging layer if specified
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, 
                dim=dim, 
                norm_layer=norm_layer, 
                style_dim=style_dim
            )
        else:
            self.downsample = None

    def forward(self, x, s=None):
        """
        Forward pass through the Basic Swin Transformer Layer.
        
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
            if self.use_checkpoint:
                # Use gradient checkpointing to save memory
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, s)
                
        # Apply patch merging if specified
        if self.downsample is not None:
            x = self.downsample(x, s)
            
        return x

    def extra_repr(self) -> str:
        """
        Generate a string representation of the layer's configuration.
        
        Returns:
            str: A formatted string containing the layer's key parameters:
                - dim: Number of channels
                - input_resolution: Input sequence length
                - depth: Number of transformer blocks
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """
        Calculate the number of floating-point operations (FLOPs) for this layer.
        
        This method estimates the computational complexity of the layer by summing
        the FLOPs of all transformer blocks and the downsampling layer if present.
        
        Returns:
            int: Total number of FLOPs for the layer
        """
        flops = 0
        # Sum FLOPs from all transformer blocks
        for blk in self.blocks:
            flops += blk.flops()
        # Add FLOPs from downsampling if present
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """
    Basic Swin Transformer Layer for one stage in the decoder.
    
    This layer implements a complete stage of the Swin Transformer decoder, consisting of:
    1. Multiple Swin Transformer blocks with alternating window attention
    2. Optional patch expansion for upsampling
    3. Support for gradient checkpointing to save memory
    
    The layer processes features through a series of transformer blocks, where each block
    alternates between regular and shifted window attention. This allows for both local
    and cross-window feature interactions. After the transformer blocks, features can be
    upsampled using patch expansion.
    
    Args:
        dim (int): Number of input channels
        input_resolution (int): Input sequence length
        depth (int): Number of transformer blocks in this layer
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool, optional): If True, add learnable bias to QKV. Default: True
        qk_scale (float, optional): Override default QK scale. Default: None
        drop (float, optional): Dropout rate for MLP. Default: 0.0
        attn_drop (float, optional): Dropout rate for attention. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
        upsample (nn.Module | None, optional): Upsample layer. Default: None
        use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default: False
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        dim_scale (int, optional): Scale factor for dimension adjustment. Default: 2
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path: Union[float, List[float]] = 0., norm_layer=nn.Module, upsample=None, use_checkpoint=False, 
                 style_dim: Optional[int]=64, dim_scale=2):
        """
        Initialize the Basic Swin Transformer Layer for decoder.
        
        Args:
            dim (int): Number of input channels
            input_resolution (int): Input sequence length
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            window_size (int): Size of attention window
            mlp_ratio (float, optional): MLP expansion ratio. Default: 4.0
            qkv_bias (bool, optional): Whether to use bias in QKV projection. Default: True
            qk_scale (float, optional): Scale factor for attention scores. Default: None
            drop (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop (float, optional): Dropout rate for attention. Default: 0.0
            drop_path (Union[float, List[float]], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.Module
            upsample (nn.Module | None, optional): Upsample layer. Default: None
            use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default: False
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
            dim_scale (int, optional): Scale factor for dimension adjustment. Default: 2
        """
        super().__init__()
        
        # Store basic configuration
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build transformer blocks
        # Each block alternates between regular and shifted window attention
        self.blocks = nn.ModuleList([
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
                # Apply stochastic depth if specified
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, 
                style_dim=style_dim
            )
            for i in range(depth)
        ])

        # Initialize patch expansion layer if specified
        if upsample is not None:
            self.upsample = PatchExpand(
                input_resolution, 
                dim=dim, 
                dim_scale=dim_scale, 
                norm_layer=norm_layer, 
                style_dim=style_dim
            )
        else:
            self.upsample = None

    def forward(self, x, s):
        """
        Forward pass through the Basic Swin Transformer Layer in decoder.
        
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
            if self.use_checkpoint:
                # Use gradient checkpointing to save memory
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, s)
                
        # Apply patch expansion if specified
        if self.upsample is not None:
            x = self.upsample(x, s)
            
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding Layer for Swin Transformer.
    
    This layer implements the initial embedding of input signals into patches, which is
    the first step in the Swin Transformer architecture. It divides the input signal into
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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, style_dim=64):
        """
        Initialize the Patch Embedding layer.
        
        Args:
            img_size (int, optional): Input signal length. Default: 224
            patch_size (int, optional): Size of each patch. Default: 4
            in_chans (int, optional): Number of input channels. Default: 3
            embed_dim (int, optional): Dimension of patch embeddings. Default: 96
            norm_layer (nn.Module, optional): Normalization layer. Default: None
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()
        
        # Store configuration
        img_size = img_size
        patch_size = patch_size
        patch_resolution = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution

        # Store channel dimensions
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Initialize patch embedding projection
        # Uses convolution to project patches into embedding space
        self.proj = nn.Conv1d(
            in_chans,           # Input channels
            embed_dim,          # Output embedding dimension
            kernel_size=patch_size,  # Patch size
            stride=patch_size,       # Non-overlapping patches
            padding=0
        )
        
        # Initialize normalization layer if specified
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Forward pass of the Patch Embedding layer.
        
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
        # Get input dimensions and validate signal length
        B, C, W = x.shape
        assert W == self.img_size, \
            f"Input signal length ({W}) doesn't match model ({self.img_size})."
            
        # Project input into patch embeddings
        # Shape: (B, C, W) -> (B, embed_dim, W/patch_size)
        x = self.proj(x)
        
        # Apply normalization if specified
        if self.norm is not None:
            x = self.norm(x)
            # Transpose to sequence format
            # Shape: (B, embed_dim, W/patch_size) -> (B, W/patch_size, embed_dim)
            x = x.transpose(1, 2)
            
        return x

    def flops(self):
        """
        Calculate the number of floating-point operations (FLOPs) for the patch embedding.
        
        This method estimates the computational complexity of the patch embedding process,
        including both the projection and normalization operations.
        
        Returns:
            int: Total number of FLOPs for the patch embedding, including:
                - Convolutional projection: Ho * Wo * embed_dim * in_chans * patch_size
                - Normalization (if used): Ho * Wo * embed_dim
                where Ho, Wo are the output height and width
        """
        # Calculate output dimensions
        Ho, Wo = self.patches_resolution
        
        # Calculate FLOPs for convolutional projection
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size * self.patch_size)
        
        # Add FLOPs for normalization if used
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
            
        return flops


class SwinTransformerSysAdaIn(nn.Module):
    """
    Swin Transformer with Adaptive Instance Normalization (AdaIN) for style transfer.
    
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
        depths (list[int], optional): Number of transformer blocks in each encoder layer. Default: [2, 2, 2, 2]
        depths_decoder (list[int], optional): Number of transformer blocks in each decoder layer. Default: [1, 2, 2, 2]
        num_heads (list[int], optional): Number of attention heads in each layer. Default: [3, 6, 12, 24]
        window_size (int, optional): Size of attention window. Default: 7
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool, optional): Whether to use bias in QKV projection. Default: True
        qk_scale (float, optional): Scale factor for attention scores. Default: None
        drop_rate (float, optional): Dropout rate for MLP. Default: 0.0
        attn_drop_rate (float, optional): Dropout rate for attention. Default: 0.0
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1
        norm_layer_encoder (nn.Module, optional): Normalization layer for encoder. Default: nn.Module
        norm_layer_decoder (nn.Module, optional): Normalization layer for decoder. Default: nn.Module
        ape (bool, optional): Whether to use absolute position embedding. Default: False
        patch_norm (bool, optional): Whether to normalize after patch embedding. Default: True
        use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default: False
        final_upsample (str, optional): Final upsampling strategy. Default: "expand_first"
        style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer_encoder=nn.Module, norm_layer_decoder=nn.Module, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", style_dim=64,**kwargs):
        """
        Initialize the Swin Transformer with AdaIN.
        
        Args:
            img_size (int, optional): Input signal length. Default: 224
            patch_size (int, optional): Size of patches. Default: 4
            in_chans (int, optional): Number of input channels. Default: 3
            num_classes (int, optional): Number of output classes. Default: 1000
            embed_dim (int, optional): Initial embedding dimension. Default: 96
            depths (list[int], optional): Number of transformer blocks in each encoder layer. Default: [2, 2, 2, 2]
            depths_decoder (list[int], optional): Number of transformer blocks in each decoder layer. Default: [1, 2, 2, 2]
            num_heads (list[int], optional): Number of attention heads in each layer. Default: [3, 6, 12, 24]
            window_size (int, optional): Size of attention window. Default: 7
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Default: 4.0
            qkv_bias (bool, optional): Whether to use bias in QKV projection. Default: True
            qk_scale (float, optional): Scale factor for attention scores. Default: None
            drop_rate (float, optional): Dropout rate for MLP. Default: 0.0
            attn_drop_rate (float, optional): Dropout rate for attention. Default: 0.0
            drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1
            norm_layer_encoder (nn.Module, optional): Normalization layer for encoder. Default: nn.Module
            norm_layer_decoder (nn.Module, optional): Normalization layer for decoder. Default: nn.Module
            ape (bool, optional): Whether to use absolute position embedding. Default: False
            patch_norm (bool, optional): Whether to normalize after patch embedding. Default: True
            use_checkpoint (bool, optional): Whether to use gradient checkpointing. Default: False
            final_upsample (str, optional): Final upsampling strategy. Default: "expand_first"
            style_dim (int, optional): Dimension of style vector for AdaIN. Default: 64
        """
        super().__init__()

        # Print configuration for debugging
        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        # Store basic configuration
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

        # Initialize patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer_encoder if self.patch_norm else None, style_dim=style_dim)
        num_patches = self.patch_embed.num_patches
        patch_resolution = self.patch_embed.patch_resolution
        self.patch_resolution = patch_resolution

        # Initialize absolute position embedding if enabled
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # Initialize dropout layer
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Generate stochastic depth rates
        dpr: List[float] = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # type: ignore

        # Build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path_slice = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=patch_resolution // (2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_slice if len(drop_path_slice) > 1 else drop_path_slice[0],
                norm_layer=norm_layer_encoder,  # norm_layer is a class, not an instance
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint, style_dim=None
            )
            self.layers.append(layer)

        # Build decoder layers
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers_decoder):
            drop_path_slice = dpr[sum(depths[:(self.num_layers_decoder - 1 - i_layer)]):sum(
                depths[:(self.num_layers_decoder - 1 - i_layer) + 1])]
            if i_layer==0:
                # First decoder layer with special configuration
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers_decoder - 1 - i_layer)),
                    input_resolution=patch_resolution // (2 ** (self.num_layers_decoder - 1 - i_layer)),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers_decoder - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_slice if len(drop_path_slice) > 1 else drop_path_slice[0],
                    norm_layer=norm_layer_decoder,  # norm_layer is a class, not an instance
                    upsample=True if (i_layer < self.num_layers_decoder - 1) else None,
                    use_checkpoint=use_checkpoint, style_dim=style_dim, dim_scale=1
                )
            else:
                # Standard decoder layers
                layer_up = BasicLayer_up(
                    dim=2*int(embed_dim * 2 ** (self.num_layers_decoder - 1 - i_layer)),
                    input_resolution=patch_resolution // (2 ** (self.num_layers_decoder - 1 - i_layer)),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers_decoder - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_slice if len(drop_path_slice) > 1 else drop_path_slice[0],
                    norm_layer=norm_layer_decoder,  # norm_layer is a class, not an instance
                    upsample=True if (i_layer < self.num_layers_decoder - 1) else None,
                    use_checkpoint=use_checkpoint, style_dim=style_dim
                )
            self.layers_up.append(layer_up)
            
        # Initialize normalization layers
        self.norm = norm_layer_encoder(self.num_features)
        if len(self.layers_up) > 1:
            self.norm_up = norm_layer_decoder(self.style_dim, self.embed_dim*2)
        else:
            self.norm_up = norm_layer_decoder(self.style_dim, self.embed_dim)

        # Initialize final upsampling layer
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            if len(self.layers_up) > 1:
                self.up = FinalPatchExpand_X4(
                    input_resolution=img_size // patch_size,
                    dim_scale=patch_size, 
                    dim=embed_dim * 2, 
                    norm_layer=AdaIN, 
                    style_dim=style_dim
                )
            else:
                self.up = FinalPatchExpand_X4(
                    input_resolution=img_size // patch_size,
                    dim_scale=patch_size, 
                    dim=embed_dim, 
                    norm_layer=AdaIN,
                    style_dim=style_dim
                )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize the weights of the model.
        
        This method initializes the weights of linear layers and normalization layers
        using appropriate initialization strategies.
        
        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            # Initialize linear layers with truncated normal distribution
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize normalization layers
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore()
    def no_weight_decay(self):
        """
        Specify parameters that should not have weight decay applied.
        
        Returns:
            set: Set of parameter names that should not have weight decay
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore()
    def no_weight_decay_keywords(self):
        """
        Specify parameter keywords that should not have weight decay applied.
        
        Returns:
            set: Set of parameter keywords that should not have weight decay
        """
        return {'relative_position_bias_table'}

    def forward_features(self, x, s=None):
        """
        Forward pass through the encoder and bottleneck.
        
        This method processes input features through:
        1. Patch embedding
        2. Position embedding (if enabled)
        3. Encoder layers with downsampling
        4. Final normalization
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, W)
            s (torch.Tensor, optional): Style vector for AdaIN. Default: None
        
        Returns:
            tuple: (x, x_downsample) where:
                - x: Bottleneck features
                - x_downsample: List of features from each encoder layer for skip connections
        """
        # Initial patch embedding
        x = self.patch_embed(x)
        
        # Add absolute position embedding if enabled
        if self.ape:
            x = x + self.absolute_pos_embed
            
        # Apply dropout
        x = self.pos_drop(x)
        
        # Store features for skip connections
        x_downsample = []

        # Process through encoder layers
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                x_downsample.insert(0, x)
            x = layer(x)
            idx += 1

        # Final normalization
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        
        return x, x_downsample

    def forward_up_features(self, x, x_downsample, s):
        """
        Forward pass through the decoder with skip connections.
        
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
                # Add skip connection
                x = torch.cat([x, x_downsample[inx]], -1)
            
        # Final normalization
        x = self.norm_up(x.transpose(1,2), s).transpose(1,2)
        
        return x

    def up_x4(self, x, s):
        """
        Final upsampling to match input resolution.
        
        This method performs the final upsampling operation to match the input resolution,
        using the specified upsampling strategy.
        
        Args:
            x (torch.Tensor): Features from decoder
            s (torch.Tensor): Style vector for AdaIN
        
        Returns:
            torch.Tensor: Upsampled features
        """
        # Validate input dimensions
        W = self.patch_resolution
        B, L, C = x.shape
        assert L == W, "input features has wrong size"

        # Apply final upsampling
        if self.final_upsample == "expand_first":
            x = self.up(x, s)
            
        return x

    def forward(self, x, s):
        """
        Complete forward pass through the network.
        
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

    def flops(self):
        """
        Calculate the number of floating-point operations (FLOPs).
        
        This method estimates the computational complexity of the network by summing
        the FLOPs of all major components.
        
        Returns:
            int: Total number of FLOPs
        """
        flops = 0
        # Add FLOPs from patch embedding
        flops += self.patch_embed.flops()
        
        # Add FLOPs from each encoder layer
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
            
        # Add FLOPs from final operations
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        
        return flops
# --- End Approximation Model code --- 

# =====================
# 4. Refinement Model (BPModel and related classes)
# =====================

# --- Begin Refinement Model code ---

class Mlp_BP(nn.Module):
    """Multi-layer perceptron (MLP) for blood pressure prediction.
    
    This class implements a two-layer MLP with dropout and activation functions,
    specifically designed for blood pressure prediction tasks. The network can be
    configured with different input, hidden, and output dimensions.
    
    Architecture:
        Input -> Dropout -> Linear1 -> Activation -> Dropout -> Linear2 -> Output
    
    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features. If None, uses in_features
        out_features (int, optional): Number of output features. If None, uses in_features
        act_layer (nn.Module, optional): Activation function to use. Defaults to nn.GELU
        drop (float, optional): Dropout probability. Defaults to 0.0
    
    Example:
        >>> model = Mlp_BP(in_features=512, hidden_features=256, out_features=1)
        >>> x = torch.randn(32, 512)  # batch_size=32, features=512
        >>> output = model(x)  # shape: (32, 1)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # Set default values if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Activation function (default: GELU)
        self.act = act_layer()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout layer for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Apply dropout to input
        x = self.drop(x)
        # First linear transformation
        x = self.fc1(x)
        # Apply activation function
        x = self.act(x)
        # Apply dropout after activation
        x = self.drop(x)
        # Final linear transformation
        x = self.fc2(x)
        return x

class WaveformEncoder(nn.Module):
    """Encoder for processing waveform data (ECG/PPG signals).
    
    This class implements a waveform encoder using the PatchTSMixer architecture,
    which is specifically designed for processing time series data like ECG and PPG signals.
    The encoder can be configured with different context lengths and model dimensions.
    
    Architecture:
        Uses PatchTSMixer model with configurable:
        - Context length (sequence length)
        - Number of input channels
        - Model dimension
        - Number of layers
        - Expansion factor
    
    Args:
        model_name (str, optional): Name of the model. Defaults to None
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True
        trainable (bool, optional): Whether to make model parameters trainable. Defaults to True
        context_length (int, optional): Length of input sequences. Defaults to 1280
        num_input_channels (int, optional): Number of input channels. Defaults to 1
        d_model (int, optional): Dimension of the model. Defaults to 64
    
    Example:
        >>> encoder = WaveformEncoder(context_length=1024, d_model=64)
        >>> x = torch.randn(32, 1, 1024)  # batch_size=32, channels=1, seq_len=1024
        >>> output = encoder(x)  # Returns PatchTSMixer output
    """
    def __init__(self, model_name=None, pretrained=True, 
                 trainable=True, context_length=1280, num_input_channels=1, d_model=64):
        super().__init__()
        # Initialize PatchTSMixer model with specified configuration
        self.model = PatchTSMixerModel(PatchTSMixerConfig(
            context_length=context_length,  # Length of input sequences
            num_input_channels=num_input_channels,  # Number of input channels
            d_model=d_model,  # Model dimension
            num_layers=15,  # Number of transformer layers
            expansion_factor=5  # Expansion factor for feed-forward networks
        ))
        
        # Configure parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """Forward pass through the waveform encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_channels, context_length)
            
        Returns:
            torch.Tensor: Output from the PatchTSMixer model
        """
        return self.model(x).last_hidden_state

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
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        super().__init__()
        # Initialize DistilBERT model with specified configuration
        if pretrained:
            # Load pretrained model if specified
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            # Initialize model with default configuration
            self.model = DistilBertModel(config=DistilBertConfig())
        
        # Index of the target token to extract from the output
        # Using [CLS] token (index 0) as the default representation
        self.target_token_idx = 0
        
        # Configure parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        """Forward pass through the text encoder.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length)
                where 1 indicates tokens to attend to and 0 indicates padding tokens
            
        Returns:
            torch.Tensor: Extracted embeddings from the target token position
                of shape (batch_size, hidden_size)
        """
        # Process input through DistilBERT model
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Extract embeddings from the target token position (default: [CLS] token)
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
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
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

    def forward(self, x):
        """Forward pass through the projection head.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            torch.Tensor: Projected embeddings of shape (batch_size, projection_dim)
                with residual connection and layer normalization applied
        """
        # Initial projection
        projected = self.projection(x)
        # Apply GELU activation
        x = self.gelu(projected)
        # Second linear transformation
        x = self.fc(x)
        # Apply dropout
        x = self.dropout(x)
        # Add residual connection
        x = x + projected
        # Apply layer normalization
        x = self.layer_norm(x)
        return x


class BPModel(BaseModel):
    """
    Unified blood pressure estimation model.
    
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
    
    The model uses a combination of:
        - PatchTSMixer for waveform processing
        - DistilBERT for text processing
        - Multi-layer perceptrons for BP prediction
        - Weighted contrastive learning for better representations
    
    Args:
        pi (bool, optional): Whether to use patient information. Defaults to True
        temperature (float, optional): Temperature parameter for contrastive loss.
            Defaults to 4.0
        image_embedding (int, optional): Size of image embedding dimension.
            Defaults to None (computed from d_model and w_length)
        text_embedding (int, optional): Size of text embedding dimension.
            Defaults to 768 (DistilBERT default)
        wcl_age_threshold (float, optional): Threshold for age in weighted contrastive loss.
            Defaults to 0.2231
        wcl (bool, optional): Whether to use weighted contrastive loss.
            Defaults to True
        w_length (int, optional): Length of waveform input.
            Defaults to 1024
        normalized_bp (bool, optional): Whether BP values are normalized.
            Defaults to True
        projection_dim (int, optional): Dimension of projected embeddings.
            Defaults to 512
        dropout (float, optional): Dropout probability for regularization.
            Defaults to 0.1
        d_model (int, optional): Dimension of the model.
            Defaults to 64
    """
    def __init__(self, 
                 pi: bool = True,
                 temperature: float = 4.0,
                 image_embedding: Optional[int] = None,
                 text_embedding: int = 768,
                 wcl_age_threshold: Optional[float] = 0.2231,
                 wcl: bool = True,
                 w_length: int = 1024,
                 normalized_bp: bool = True,
                 projection_dim: int = 512,
                 dropout: float = 0.1,
                 d_model: int = 64,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__()
        
        # Pass supports_multi_directional from config to parent
        super(BPModel, self).__init__(*args, **kwargs)
        
        # Parameter validation
        if text_embedding <= 0:
            raise ValueError("text_embedding must be positive")
        if w_length <= 0:
            raise ValueError("w_length must be positive")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if wcl_age_threshold is not None and wcl_age_threshold < 0:
            raise ValueError("wcl_age_threshold must be non-negative if provided")
        self.pi = pi
        self.temperature = temperature
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.wcl_age_threshold = wcl_age_threshold
        self.wcl = wcl
        self.w_length = w_length
        self.normalized_bp = normalized_bp
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.d_model = d_model

        # Calculate image embedding size based on model dimension and waveform length
        # The PatchTSMixer encoder reduces the sequence length by a factor of 8 (3 downsampling layers)
        # so the output shape is (batch, d_model, w_length // 8)
        if image_embedding is None:
            image_embedding = (d_model * w_length) // 8
        self.image_embedding = image_embedding

        # --- Waveform Encoders ---
        # These encoders process the raw ECG and PPG signals
        if pi:
            # With patient information: also initialize text encoders and fusion encoders
            self.ppg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.ecg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.text_encoder = TextEncoder()
            # Fusion encoders for combining waveform and text features
            self.ppg_text_encoder = WaveformEncoder(context_length=projection_dim, num_input_channels=2)
            self.ecg_text_encoder = WaveformEncoder(context_length=projection_dim, num_input_channels=2)
        else:
            # Without patient information: only waveform encoders and BP-specific encoders
            self.ppg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.ppg_bp_encoder = WaveformEncoder(context_length=projection_dim, d_model=d_model)
            self.ecg_encoder = WaveformEncoder(context_length=w_length, d_model=d_model)
            self.ecg_bp_encoder = WaveformEncoder(context_length=projection_dim, d_model=d_model)

        # --- Projection Heads ---
        # These heads transform the encoder outputs to a common embedding space
        self.ppg_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim, dropout=dropout)
        self.ecg_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim, dropout=dropout)
        if pi:
            self.text_projection = ProjectionHead(embedding_dim=text_embedding, projection_dim=projection_dim, dropout=dropout)

        # --- BP Prediction Heads ---
        # These MLPs predict SBP and DBP from the final features
        in_features = (projection_dim * 2 * d_model) // 8 if pi else (projection_dim * d_model) // 8
        self.ppg_SBP_head = Mlp_BP(in_features=in_features, hidden_features=in_features // 2, out_features=1, drop=0.2)
        self.ppg_DBP_head = Mlp_BP(in_features=in_features, hidden_features=in_features // 2, out_features=1, drop=0.2)
        self.ecg_SBP_head = Mlp_BP(in_features=in_features, hidden_features=in_features // 2, out_features=1, drop=0.2)
        self.ecg_DBP_head = Mlp_BP(in_features=in_features, hidden_features=in_features // 2, out_features=1, drop=0.2)

        # --- L1 Loss (for legacy, but not used in forward if handled in trainer) ---
        self.l1 = torch.nn.L1Loss()

        # --- Store age threshold for weighted contrastive loss (if needed elsewhere) ---
        self.wcl_age_threshold = wcl_age_threshold

    def extract_input(self, batch_dict: Dict[str, torch.Tensor], ecg_label: int = 0, ppg_label: int = 1) -> Dict[str, torch.Tensor]:
        """
        Extract and prepare input for BPModel from unified batch structure
        
        This method handles the unified input processing for BPModel, including:
        - ECG and PPG waveform extraction using src_idxs and src_mask
        - Target channel extraction using tgt_idxs
        - Domain shift target generation
        
        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs
            ecg_label: ECG channel index (legacy fallback)
            ppg_label: PPG channel index (legacy fallback)
        Returns:
            dict: Prepared input with keys:
                - ecg: ECG waveform tensor (B, L, 1)
                - ppg: PPG waveform tensor (B, L, 1)
                - SBP/DBP: Raw BP values
                - SBP_n/DBP_n: Normalized BP values
                - input_ids/attention_mask: Text encoding (if available)
                - age/gender: Demographic info (if available)
        """
        waveform = batch_dict["waveform"]
        
        # Check if using unified batch structure
        # if "src_idxs" in batch_dict and "src_mask" in batch_dict and "tgt_idxs" in batch_dict:
        #     # Unified mode: use src_idxs, src_mask, tgt_idxs
        #     src_idxs = batch_dict["src_idxs"]      # [B, S_max]
        #     src_mask = batch_dict["src_mask"]      # [B, S_max]
        #     tgt_idxs = batch_dict["tgt_idxs"]     # [B]
            
        #     # For BPModel, we need both ECG and PPG channels
        #     # This is a special case where we extract specific channels regardless of direction
        #     if hasattr(self, 'vital2chan') and self.vital2chan is not None:
        #         # Use cached layout if available
        #         ecg_ch = self.vital2chan[self.vital_names.index('ECG')]
        #         ppg_ch = self.vital2chan[self.vital_names.index('PPG')]
        #     else:
        #         # Fallback to legacy labels
        #         ecg_ch = ecg_label
        #         ppg_ch = ppg_label
            
        #     ecg = waveform[:, ecg_ch:ecg_ch+1].permute(0, 2, 1)  # [B, L, 1]
        #     ppg = waveform[:, ppg_ch:ppg_ch+1].permute(0, 2, 1)  # [B, L, 1]
            
        # else:
        #     # Legacy mode: fallback to old channel-based approach
        ecg = waveform[:, ecg_label:ecg_label+1].permute(0, 2, 1)
        ppg = waveform[:, ppg_label:ppg_label+1].permute(0, 2, 1)
    
        # Extract BP values (maintain existing logic)
        # bp = batch_dict.get("bp")
        # batch_obj = batch_dict.get("batch")
        
        # if bp is not None and batch_obj is not None:
        #     result = {
        #         "ecg": ecg,
        #         "ppg": ppg,
        #         "SBP": batch_obj.bp_raw[:, 0:1],
        #         "DBP": batch_obj.bp_raw[:, 1:2],
        #         "SBP_n": bp[:, 0:1],
        #         "DBP_n": bp[:, 1:2],
        #     }
        # else:
        # Fallback if BP data not available
        result = {
            "ecg": ecg,
            "ppg": ppg,
        }
        
        # Handle text encoding if available
        if "text" in batch_dict:
            result["input_ids"] = batch_dict["text"]["input_ids"]
            result["attention_mask"] = batch_dict["text"]["attention_mask"]
        
        # # Handle demographics if available
        # if "demographics" in batch_dict:
        #     demographics = batch_dict["demographics"]
        #     if demographics.shape[1] >= 2:
        #         result["age"] = demographics[:, 0:1]
        #         result["gender"] = demographics[:, 1:2]
        
        return result

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Forward pass of the blood pressure estimation model.
        This method implements the complete forward pass of the model, processing both
        waveform data (ECG/PPG) and optionally patient information to predict blood
        pressure values. The method can operate in two modes:
        1. With patient information (PI): Uses both waveform and text data
        2. Without patient information: Uses only waveform data
        Args:
            batch (dict): Dictionary containing:
                - "ecg" (torch.Tensor): ECG signal data of shape (batch_size, channels, sequence_length)
                - "ppg" (torch.Tensor): PPG signal data of shape (batch_size, channels, sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if using PI
        Returns:
            dict: Dictionary containing predictions and intermediate features:
                - "ecg_sbp": ECG-based SBP predictions
                - "ecg_dbp": ECG-based DBP predictions  
                - "ppg_sbp": PPG-based SBP predictions
                - "ppg_dbp": PPG-based DBP predictions
                - "ecg_embeddings": ECG embeddings for WCL (if enabled)
                - "ppg_embeddings": PPG embeddings for WCL (if enabled)
                - "text_embeddings": Text embeddings for WCL (if enabled and PI)
        """
        # Always extract and prepare input from the raw batch dict
        x = self.extract_input(batch)

        # --- Feature extraction ---
        # Pass ECG and PPG through their respective encoders
        ecg_features = self.ecg_encoder(x["ecg"])
        ppg_features = self.ppg_encoder(x["ppg"])
        # Flatten encoder outputs and project to embedding space
        # ecg_embeddings = self.ecg_projection(ecg_features[0].view(ecg_features.shape[0], -1))
        # ppg_embeddings = self.ppg_projection(ppg_features[0].view(ppg_features.shape[0], -1))
        ecg_embeddings = self.ecg_projection(ecg_features.view(ecg_features.shape[0], -1)).unsqueeze(-1)
        ppg_embeddings = self.ppg_projection(ppg_features.view(ppg_features.shape[0], -1)).unsqueeze(-1)

        text_embeddings = None
        if self.pi and "input_ids" in x and "attention_mask" in x:
            # Process text data if present
            text_features = self.text_encoder(x["input_ids"], x["attention_mask"])
            text_embeddings = self.text_projection(text_features).unsqueeze(-1)

        if self.pi and text_embeddings is not None:
            # Combine waveform and text features for fusion
            ecg_text_features = torch.cat((ecg_embeddings, text_embeddings), dim=-1)
            ppg_text_features = torch.cat((ppg_embeddings, text_embeddings), dim=-1)
            # Pass through fusion encoders and flatten
            ecg_final = self.ecg_text_encoder(ecg_text_features).view(ecg_text_features.shape[0], -1)
            ppg_final = self.ppg_text_encoder(ppg_text_features).view(ppg_text_features.shape[0], -1)
        else:
            # If not using PI, optionally pass through BP-specific encoders
            ecg_final = self.ecg_bp_encoder(ecg_embeddings).view(ecg_embeddings.shape[0], -1) if hasattr(self, 'ecg_bp_encoder') else ecg_embeddings
            ppg_final = self.ppg_bp_encoder(ppg_embeddings).view(ppg_embeddings.shape[0], -1) if hasattr(self, 'ppg_bp_encoder') else ppg_embeddings

        # --- BP prediction ---
        y_ecg_SBP = self.ecg_SBP_head(ecg_final)
        y_ecg_DBP = self.ecg_DBP_head(ecg_final)
        y_ppg_SBP = self.ppg_SBP_head(ppg_final)
        y_ppg_DBP = self.ppg_DBP_head(ppg_final)

        y_SBP = (y_ecg_SBP + y_ppg_SBP) / 2
        y_DBP = (y_ecg_DBP + y_ppg_DBP) / 2

        y_bp = torch.cat((y_SBP, y_DBP), dim=-1)

        # --- Prepare output dict ---
        result = {
            "y_pred": y_bp,
            "y_pred_sbp": y_SBP,
            "y_pred_dbp": y_DBP,
            "ecg_sbp": y_ecg_SBP,
            "ecg_dbp": y_ecg_DBP,
            "ppg_sbp": y_ppg_SBP,
            "ppg_dbp": y_ppg_DBP,
            "ecg_embeddings": ecg_embeddings.squeeze(-1),
            "ppg_embeddings": ppg_embeddings.squeeze(-1),
        }
        if text_embeddings is not None:
            result["text_embeddings"] = text_embeddings.squeeze(-1)
        return result

    def _compute_wcl_losses(self, embeddings, batch):
        """Compute weighted contrastive losses for waveform embeddings.
        
        This method computes weighted contrastive losses for waveform embeddings (ECG/PPG)
        based on blood pressure values (SBP and DBP). It helps learn waveform representations
        that are similar for similar blood pressure values.
        
        The loss computation:
        1. Compute SBP-based WCL using regression parameters
        2. Compute DBP-based WCL using regression parameters
        3. Combine losses with appropriate scaling based on BP normalization
        
        Args:
            embeddings (torch.Tensor): Waveform embeddings of shape (batch_size, embedding_dim)
            batch (dict): Dictionary containing:
                - "SBP" (torch.Tensor): Systolic blood pressure values of shape (batch_size,)
                - "DBP" (torch.Tensor): Diastolic blood pressure values of shape (batch_size,)
        
        Returns:
            torch.Tensor: Combined weighted contrastive loss for waveform embeddings
            
        Note:
            - Both SBP and DBP WCL use temperature=4.0 and threshold=0.0235 for regression
            - Losses are scaled by 1e-3 when BP values are normalized
            - The same parameters are used for both SBP and DBP to ensure consistent learning
        """
        # Compute SBP-based weighted contrastive loss
        # Uses regression parameters (temperature=4.0, threshold=0.0235)
        wcl_SBP = self.wcl_criterion(embeddings, batch["SBP"])
        
        # Compute DBP-based weighted contrastive loss
        # Uses regression parameters (temperature=4.0, threshold=0.0235)
        wcl_DBP = self.wcl_criterion(embeddings, batch["DBP"])
        
        # Combine SBP and DBP losses with appropriate scaling
        if self.normalized_bp:
            # Scale losses by 1e-3 when BP values are normalized
            return wcl_SBP * 1e-3 + wcl_DBP * 1e-3
        # Use raw losses when BP values are not normalized
        return wcl_SBP + wcl_DBP

    def _compute_text_wcl(self, text_embeddings, batch):
        """Compute weighted contrastive loss for text embeddings.
        
        This method computes weighted contrastive losses for text embeddings based on
        demographic information (gender and age). It helps learn text representations
        that are similar for similar demographic characteristics.
        
        The loss computation:
        1. Compute gender-based WCL using binary classification
        2. If age threshold is set:
           - Compute age-based WCL using regression
           - Combine gender and age losses with appropriate scaling
        3. If no age threshold:
           - Use only gender-based WCL
        4. Apply normalization if BP values are normalized
        
        Args:
            text_embeddings (torch.Tensor): Text embeddings of shape (batch_size, embedding_dim)
            batch (dict): Dictionary containing:
                - "gender" (torch.Tensor): Gender values of shape (batch_size,)
                - "age" (torch.Tensor): Age values of shape (batch_size,)
        
        Returns:
            torch.Tensor: Combined weighted contrastive loss for text embeddings
            
        Note:
            - Gender WCL uses temperature=1.0 and threshold=1.0 for binary classification
            - Age WCL uses temperature=4.0 and threshold=0.0235 for regression
            - Losses are scaled by 1e-2 when BP values are normalized
        """
        # Compute gender-based weighted contrastive loss
        # Uses binary classification parameters (temperature=1.0, threshold=1.0)
        text_wcl_gender = self.text_wcl_gender(text_embeddings, batch["gender"])
        
        if self.wcl_age_threshold is not None:
            # Compute age-based weighted contrastive loss
            # Uses regression parameters (temperature=4.0, threshold=0.0235)
            text_wcl_age = self.text_wcl_age(text_embeddings, batch["age"])
            
            # Combine gender and age losses with appropriate scaling
            if self.normalized_bp:
                # Scale losses by 1e-2 when BP values are normalized
                text_wcl = text_wcl_age * 1e-2 + text_wcl_gender * 1e-2
            else:
                # Use raw losses when BP values are not normalized
                text_wcl = text_wcl_age + text_wcl_gender
        else:
            # Use only gender-based loss if no age threshold is set
            if self.normalized_bp:
                # Scale loss by 1e-2 when BP values are normalized
                text_wcl = text_wcl_gender * 1e-2
            else:
                # Use raw loss when BP values are not normalized
                text_wcl = text_wcl_gender

        return text_wcl

    def get_SBP_DBP_fromPPG(self, batch):
        """Get blood pressure predictions from PPG signal.
        
        This method processes Photoplethysmogram (PPG) signals to predict both systolic (SBP)
        and diastolic (DBP) blood pressure values. It can operate in two modes:
        1. With patient information (PI): Uses both PPG and text data
        2. Without patient information: Uses only PPG data
        
        Args:
            batch (dict): Dictionary containing:
                - "ppg" (torch.Tensor): PPG signal data of shape (batch_size, channels, sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if using PI
        
        Returns:
            tuple: (SBP_predictions, DBP_predictions)
                - SBP_predictions (torch.Tensor): Predicted systolic blood pressure values
                - DBP_predictions (torch.Tensor): Predicted diastolic blood pressure values
        
        Example:
            >>> batch = {
            ...     "ppg": torch.randn(32, 1, 1024),  # batch_size=32, channels=1, seq_len=1024
            ...     "input_ids": torch.randint(0, 1000, (32, 128)),  # if using PI
            ...     "attention_mask": torch.ones(32, 128)  # if using PI
            ... }
            >>> sbp, dbp = model.get_SBP_DBP_fromPPG(batch)
        """
        # Use the forward method to get predictions
        model_outputs = self.forward(batch)
        return (model_outputs["ppg_sbp"], model_outputs["ppg_dbp"])

    def get_SBP_DBP_fromECG(self, batch):
        """Get blood pressure predictions from ECG signal.
        
        This method processes ECG signals to predict both systolic (SBP) and diastolic (DBP)
        blood pressure values. It can operate in two modes:
        1. With patient information (PI): Uses both ECG and text data
        2. Without patient information: Uses only ECG data
        
        Args:
            batch (dict): Dictionary containing:
                - "ecg" (torch.Tensor): ECG signal data of shape (batch_size, channels, sequence_length)
                - "input_ids" (torch.Tensor, optional): Text token IDs if using PI
                - "attention_mask" (torch.Tensor, optional): Text attention mask if using PI
        
        Returns:
            tuple: (SBP_predictions, DBP_predictions)
                - SBP_predictions (torch.Tensor): Predicted systolic blood pressure values
                - DBP_predictions (torch.Tensor): Predicted diastolic blood pressure values
        
        Example:
            >>> batch = {
            ...     "ecg": torch.randn(32, 1, 1024),  # batch_size=32, channels=1, seq_len=1024
            ...     "input_ids": torch.randint(0, 1000, (32, 128)),  # if using PI
            ...     "attention_mask": torch.ones(32, 128)  # if using PI
            ... }
            >>> sbp, dbp = model.get_SBP_DBP_fromECG(batch)
        """
        # Use the forward method to get predictions
        model_outputs = self.forward(batch)
        return (model_outputs["ecg_sbp"], model_outputs["ecg_dbp"])



# --- End Refinement Model code --- 
cs = ConfigStore.instance()
cs.store(name="base_unet_swin_unet", group="model", node=UNet_SwinUnetConfig)
cs.store(name="base_bp_model", group="model", node=BPModelConfig)