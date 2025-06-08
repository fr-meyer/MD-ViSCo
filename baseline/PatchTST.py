"""
PatchTST (Patch Time Series Transformer) implementation for time series regression.
This module provides two implementations:
1. ApproximationPatchTST: General time series regression model
2. RefinementPatchTST: Specialized model for blood pressure prediction
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForRegression

class ApproximationPatchTST(nn.Module):
    """
    PatchTST model for time series regression.
    
    This implementation is based on the paper "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    and uses Hugging Face's transformers library.
    
    Args:
        input_size (int): Number of input channels/features
        output_size (int): Number of target values to regress
        context_length (int, optional): Length of input sequence. Defaults to 512.
        patch_len (int, optional): Length of each patch. Defaults to 16.
        stride (int, optional): Stride between patches. Defaults to 8.
        d_model (int, optional): Dimension of model. Defaults to 128.
        num_encoder_layers (int, optional): Number of encoder layers. Defaults to 3.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Attention dropout rate. Defaults to 0.1.
        fc_dropout (float, optional): Dropout rate for fully connected layers. Defaults to 0.1.
        head_dropout (float, optional): Dropout rate for regression head. Defaults to 0.1.
    """
    def __init__(
        self,
        input_size,
        output_size,
        context_length=512,
        patch_len=16,
        stride=8,
        d_model=128,
        num_encoder_layers=3,
        num_heads=8,
        dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.1
    ):
        super().__init__()
        
        # Create PatchTST configuration
        self.config = PatchTSTConfig(
            num_input_channels=input_size,
            num_targets=output_size,  # Changed from prediction_length to num_targets
            context_length=context_length,
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
            positional_encoding_type='sincos'  # Sinusoidal positional encoding
        )
        
        # Initialize PatchTST model with regression head
        self.model = PatchTSTForRegression(self.config)

    def forward(self, x, target_values=None, past_observed_mask=None):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, context_length, num_input_channels)
            target_values (torch.Tensor, optional): Target values for training of shape (batch_size, num_targets)
            past_observed_mask (torch.BoolTensor, optional): Boolean mask for missing values
                Shape: (batch_size, context_length, num_input_channels)
                1 for observed values, 0 for missing values
        
        Returns:
            During training (with target_values):
                ModelOutput with loss and regression_outputs
            During inference:
                ModelOutput with regression_outputs of shape (batch_size, num_targets)
        """
        outputs = self.model(
            past_values=x,
            target_values=target_values,
            past_observed_mask=past_observed_mask
        )
        return outputs.regression_outputs


class RefinementPatchTST(nn.Module):
    """
    PatchTST model for blood pressure prediction from waveform and demographic data.
    
    This implementation uses the Hugging Face's PatchTST model to predict SBP and DBP values
    from physiological waveform (e.g., PPG, ECG) combined with demographic information.
    
    Args:
        input_size (int): Number of input channels (1 waveform + 5 demographics)
        output_size (int): Fixed to 2 for SBP and DBP prediction
        context_length (int): Length of input sequence
        patch_len (int): Length of each patch
        stride (int): Stride between patches
        d_model (int): Dimension of model
        num_encoder_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        dropout (float): Attention dropout rate
        fc_dropout (float): Dropout rate for fully connected layers
        head_dropout (float): Dropout rate for prediction head
        use_cls_token (bool): Whether to use CLS token for regression
    """
    def __init__(
        self,
        input_size=6,  # 1 waveform + 5 demographics
        output_size=2,  # SBP and DBP
        context_length=1280,
        patch_len=16,
        stride=8,
        d_model=640,
        num_encoder_layers=10,
        num_heads=10,
        dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.1,
        use_cls_token=True
    ):
        super().__init__()
        
        # Create PatchTST configuration
        self.config = PatchTSTConfig(
            num_input_channels=input_size,
            num_targets=output_size,
            context_length=context_length,
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            num_hidden_layers=num_encoder_layers,
            num_attention_heads=num_heads,
            attention_dropout=dropout,
            ff_dropout=fc_dropout,
            path_dropout=head_dropout,
            # Configuration for BP prediction
            share_embedding=True,           # Share embedding across channels
            channel_attention=False,        # No channel attention
            norm_type='batchnorm',         # Use batch normalization
            activation_function='gelu',     # GELU activation
            pre_norm=True,                 # Apply normalization before attention
            positional_encoding_type='sincos',  # Sinusoidal positional encoding
            use_cls_token=use_cls_token    # Use the parameter here
        )
        
        # Initialize PatchTST model for regression
        self.model = PatchTSTForRegression(self.config)

    def forward(self, x, target_values=None, past_observed_mask=None):
        """
        Forward pass of the BP prediction model.
        
        Args:
            x (torch.Tensor): Input tensor combining waveform and demographics
                Shape: (batch_size, sequence_length, input_channels)
                Where input_channels includes waveform + demographic features
            target_values (torch.Tensor, optional): Target BP values for training 
                Shape: (batch_size, 2) where values are [SBP, DBP]
            past_observed_mask (torch.BoolTensor, optional): Boolean mask for missing values
                Shape: (batch_size, sequence_length, input_channels)
                1 for observed values, 0 for missing demographic values
        
        Returns:
            Tensor of shape (batch_size, 2) containing SBP and DBP predictions
        """
        outputs = self.model(
            past_values=x,
            target_values=target_values,
            past_observed_mask=past_observed_mask
        )
        return outputs.regression_outputs
