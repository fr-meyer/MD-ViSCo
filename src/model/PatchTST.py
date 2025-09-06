"""
PatchTST (Patch Time Series Transformer) implementation for time series regression.
This module provides two implementations:
1. ApproximationPatchTST: General time series regression model
2. RefinementPatchTST: Specialized model for blood pressure prediction
"""

# Standard library imports
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Third-party imports
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForRegression
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

# Local imports
from .base_model import BaseModel, BaseModelConfig

@dataclass
class PatchTSTModelConfig(BaseModelConfig):
    """Configuration class for PatchTST model architecture parameters.
    
    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the PatchTST model, independently of training or dataset.
    
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
    _target_: str = "src.model.PatchTST.PatchTST"
    supports_multi_directional: bool = False  # PatchTST only supports single-direction
    model_name: str = "PatchTST"
    
    # Model Architecture configuration
    # REMOVED: context_length field - now inherited from BaseModelConfig
    patch_len: int = 16  # Length of each patch
    stride: int = 8  # Stride between patches
    d_model: int = MISSING  # Dimension of model
    num_encoder_layers: int = MISSING  # Number of encoder layers
    num_heads: int = MISSING  # Number of attention heads
    dropout: float = 0.1  # Attention dropout rate
    fc_dropout: float = 0.1  # Dropout rate for fully connected layers
    head_dropout: float = 0.1  # Dropout rate for prediction head
    in_channels: int = 1  # Number of input channels
    num_targets: int = MISSING  # Number of target values to predict
    use_cls_token: bool = False  # Whether to use CLS token for regression
    use_demographics: bool = MISSING  # Whether to fuse demographic information
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
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
        if self.input_length <= 0:  # UPDATED
            raise ValueError("input_length must be positive")

class PatchTST(BaseModel):
    """
    Unified PatchTST model for time series regression.
    
    This implementation is based on the paper "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    and uses Hugging Face's transformers library. It supports both approximation and refinement use cases
    through configuration parameters.
    
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
        use_demographics (bool): Whether to fuse demographic information with waveform data
    """
    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 640,
        num_encoder_layers: int = 10,
        num_heads: int = 10,
        dropout: float = 0.1,
        fc_dropout: float = 0.1,
        head_dropout: float = 0.1,
        in_channels: int = 1,
        num_targets: int = 1,
        use_cls_token: bool = False,
        use_demographics: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Store parameters as attributes for easy access
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
        if self.input_length <= 0:  # UPDATED
            raise ValueError("input_length must be positive")
        # Create PatchTST configuration
        self.patchtst_config = PatchTSTConfig(
            num_input_channels=in_channels,
            num_targets=num_targets,
            context_length=self.input_length,  # UPDATED
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            num_hidden_layers=num_encoder_layers,
            num_attention_heads=num_heads,
            attention_dropout=dropout,
            ff_dropout=fc_dropout,
            path_dropout=head_dropout,
            # Additional configuration options from the paper
            share_embedding=True,           # Share embedding across channels
            channel_attention=False,        # No channel attention
            norm_type='batchnorm',         # Use batch normalization
            activation_function='gelu',     # GELU activation
            pre_norm=True,                 # Apply normalization before attention
            positional_encoding_type='sincos',  # Sinusoidal positional encoding
            use_cls_token=use_cls_token
        )
        
        # Initialize PatchTST model with regression head
        self.model = PatchTSTForRegression(self.patchtst_config)

    def extract_input(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        """
        Extract and prepare input for PatchTST model from unified batch structure.
        
        This method handles input processing for both approximation and refinement modes:
        - Approximation mode: Channel selection using src_idxs and src_mask
        - Refinement mode: Waveform + demographics fusion with masking
        
        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs
            
        Returns:
            x: Tensor [B, L, C] ready for PatchTST
            past_observed_mask: BoolTensor [B, L, C] or None for missing value handling
        """
        if self.use_demographics:
            # Refinement mode: Waveform + demographics fusion
            waveform = batch_dict["waveform"]  # [B, C, L]
            B, C, L = waveform.shape
            
            x = waveform
            mask = torch.ones((B, C, L), dtype=torch.bool, device=waveform.device)

            if "demographics" in batch_dict:
                demographics = batch_dict["demographics"]  # [B, D]
                demo_mask = ~torch.isnan(demographics)     # [B, D]
                demographics = torch.nan_to_num(demographics, nan=0.0)  # [B, D]
                demo_exp = demographics.unsqueeze(-1).expand(-1, -1, L)  # [B, D, L]
                demo_mask = demo_mask.unsqueeze(-1).expand(-1, -1, L)    # [B, D, L]

                x = torch.cat([x, demo_exp], dim=1)       # [B, C+D, L]
                mask = torch.cat([mask, demo_mask.to(waveform.device)], dim=1)  # [B, C+D, L]

            x = x.transpose(1, 2)      # [B, L, C]
            mask = mask.transpose(1, 2)  # [B, L, C]
            
            return x, mask
        else:
            # Approximation mode: Use unified batch structure only
            # Parent class validates required fields and extracts input
            x = super().extract_input(batch_dict)

            # Convert to PatchTST expected format [B, L, C]
            if x.dim() == 3:
                x = x.transpose(1, 2)  # [B, T, S_max]

            return x, None  # No mask needed for PatchTST

    def forward(self, batch_dict: Dict[str, torch.Tensor], target_values=None, past_observed_mask=None):
        """
        Forward pass of the unified PatchTST model.
        
        Args:
            batch_dict: Dictionary containing at least 'waveform'. May contain 'demographics' if use_demographics=True.
            target_values: Ground truth values [B, num_targets]
            past_observed_mask: Optional externally-supplied mask [B, L, C]

        Returns:
            Regression predictions [B, num_targets]
        """
        # Handle dict inputs
        if isinstance(batch_dict, dict):
            x, generated_mask = self.extract_input(batch_dict)
            
        past_observed_mask = past_observed_mask if past_observed_mask is not None else generated_mask

        output = self.model(
            past_values=x,
            target_values=target_values,
            past_observed_mask=past_observed_mask
        )

        outputs = {}

        if self._dictionary_key == "y_pred_bp":
            outputs["y_pred_sbp"] = output.regression_outputs[:, 0].unsqueeze(-1)
            outputs["y_pred_dbp"] = output.regression_outputs[:, 1].unsqueeze(-1)
            outputs["y_pred"] = output.regression_outputs
        else:
            outputs[self._dictionary_key] = output.regression_outputs.unsqueeze(1)

        return outputs

cs = ConfigStore.instance()
cs.store(name="base_patchtst", group="model", node=PatchTSTModelConfig)