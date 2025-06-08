# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .UNet_1DCNN_pytorch import UNet
from .MLPRegressor_pytorch import MLPRegressor

class ShallowUNet(nn.Module):
    """ShallowUNet model combining UNet for waveform prediction and MLPs for BP prediction"""
    
    def __init__(self, input_size, feature_number=1024, hidden_size=100, 
                 model_depth=1, num_channel=1,
                 model_width=128, kernel_size=3, problem_type='Regression', 
                 output_nums=1, ds=0, ae=1, ag=0, is_transconv=False,
                 mlp_activation='relu', mlp_alpha=0.0001,
                 feature_extraction_only=False, alpha=1, lstm=0):
        """Initialize ShallowUNet
        
        Args:
            input_size (int): Size of input signal
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
        super().__init__()
        
        # Store configuration parameters as attributes
        self.input_size = input_size
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
        self.unet = UNet(
            length=self.input_size,
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
        
        # Initialize MLPs for SBP and DBP prediction
        self.mlp_sbp = MLPRegressor(
            input_size=self.feature_number,
            hidden_size=self.hidden_size,
            activation=self.mlp_activation,
            alpha=self.mlp_alpha
        )
        
        self.mlp_dbp = MLPRegressor(
            input_size=self.feature_number,
            hidden_size=self.hidden_size,
            activation=self.mlp_activation,
            alpha=self.mlp_alpha
        )
        
    def forward(self, x, return_abp=False):
        """Forward pass
        Args:
            x (torch.Tensor): Input signal of shape (batch_size, 1, signal_length)
            return_features (bool): Whether to return extracted features
        Returns:
            tuple: (waveform, features) if return_features=True
                  (waveform, sbp, dbp) if return_features=False
        """
        if return_abp:
            # Get waveform prediction
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
    
