# https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction/tree/main

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import torch
import torch.autograd as autograd
import torch.nn as nn

# Local imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Dict
from src.utils.bp_utils import extract_bp_values
from .base_model import BaseModel, BaseModelConfig

# ----------
#  Configuration
# ----------

@dataclass
class P2EWGANConfig(BaseModelConfig):
    """Configuration class for P2E-WGAN model architecture parameters.
    
    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the P2E-WGAN model, independently of training or dataset.
    
    Training parameters (lambda_gp, n_critic) remain in dataset configs.
    """
    _target_: str = "src.model.P2E_WGAN.P2EWGAN"
    supports_multi_directional: bool = False  # P2E-WGAN only supports single-direction
    model_name: str = "P2EWGAN"
    
    # Model Architecture configuration
    in_channels: int = 1
    out_channels: int = 1
    
    # Network capacity configuration
    generator_init_filters: int = MISSING  # Controls generator network capacity (approximation default)
    discriminator_init_filters: int = MISSING  # Controls discriminator network capacity (approximation default)


# ----------
#  U-NET
# ----------

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        # Use padding='same' to maintain size/(stride) for any input size
        padding = ksize // 2
        layers = [nn.Conv1d(in_size, out_size, kernel_size=ksize,
                            stride=stride, bias=False, padding=padding, padding_mode='replicate')]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, dropout=0.0):
        super(UNetUp, self).__init__()
        # Use padding='same' for the transposed convolution
        padding = ksize // 2
        output_padding = stride - 1
        layers = [
            nn.ConvTranspose1d(in_size, out_size, kernel_size=ksize,
                               stride=stride, padding=padding, 
                               output_padding=output_padding, bias=False),
            nn.InstanceNorm1d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        
        # Center crop the larger tensor to match sizes
        if x.size(2) != skip_input.size(2):
            if x.size(2) > skip_input.size(2):
                diff = x.size(2) - skip_input.size(2)
                x = x[:, :, diff//2:-(diff-diff//2)]
            else:
                diff = skip_input.size(2) - x.size(2)
                skip_input = skip_input[:, :, diff//2:-(diff-diff//2)]
        
        x = torch.cat((x, skip_input), 1)
        return x


# ----------
#  Generator
# ----------

class GeneratorUNet(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, init_filters=128, *args, **kwargs):
        """U-Net Generator with configurable initial filter size
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            init_filters: Initial filter size for the first layer
        """
        super(GeneratorUNet, self).__init__(*args, **kwargs)
        
        # Parameter validation
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if init_filters <= 0:
            raise ValueError("init_filters must be positive")
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters

        # Calculate filter sizes based on init_filters
        f1 = init_filters          # First level filters
        f2 = init_filters * 2      # Second level filters  
        f3 = init_filters * 4      # Third level filters
        f4 = init_filters * 4      # Bottom level filters

        # Downsampling path - will work with any input size
        self.down1 = UNetDown(in_channels, f1, normalize=False)    # size -> size/2
        self.down2 = UNetDown(f1, f2)                              # size/2 -> size/4
        self.down3 = UNetDown(f2, f3, dropout=0.5)                 # size/4 -> size/8
        self.down4 = UNetDown(f3, f4, dropout=0.5, normalize=False)  # size/8 -> size/16

        # Upsampling path
        self.up1 = UNetUp(f4, f3, dropout=0.5)           # size/16 -> size/8
        self.up2 = UNetUp(f3*2, f2)                      # size/8 -> size/4
        self.up3 = UNetUp(f2*2, f1)                      # size/4 -> size/2

        # Modified final convolution with dynamic padding
        final_conv_size = 4
        final_padding = final_conv_size // 2  # This will be 2 for kernel size 4
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(f1*2, out_channels, kernel_size=final_conv_size, 
                     padding=final_padding, padding_mode='replicate'),
            nn.Tanh(),
        )

        # Automatically initialize weights on creation
        self.apply(weights_init_normal)

    def extract_input(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract input for P2E-WGAN.
        
        This method handles the unified input processing for P2E-WGAN, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, 1, L] for P2E-WGAN
        
        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask
        
        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        # Get common input extraction from parent class
        x = super().extract_input(batch_dict)
        
        return {
            "x": x
        }

    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        """
        Forward pass for GeneratorUNet. Expects a standardized batch dict and source_channel.
        Args:
            batch_dict: Output of DataLoader collate_fn (standardized dict) or pre-processed tensor
            source_channel: Index of the source channel to use as input (ignored if batch_dict is tensor)
        Returns:
            torch.Tensor: Generated output
        """
        
        x = self.extract_input(batch_dict)
            
        # Store original size
        original_size = x["x"].size(2)
        d1 = self.down1(x["x"])
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        output = self.final(u3)
        # Ensure output size matches input size
        if output.size(2) != original_size:
            # Center crop if needed
            if output.size(2) > original_size:
                diff = output.size(2) - original_size
                output = output[:, :, diff//2:diff//2 + original_size]
            else:
                raise ValueError(f"Output size {output.size(2)} is smaller than input size {original_size}")
        
        outputs = {}
        outputs["real_a"] = x["x"]
        outputs["fake_b"] = output
        if self._dictionary_key == "y_pred_bp":
            if output.shape[-1] == 1280:
                sbp, dbp = extract_bp_values(output[:, :, 15:-15])
            else:
                sbp, dbp = extract_bp_values(output)
                
            outputs["y_pred_sbp"] = sbp
            outputs["y_pred_dbp"] = dbp
        outputs["y_pred"] = output

        return outputs


# --------------
#  Discriminator
# --------------


class Discriminator(BaseModel):
    def __init__(self, in_channels=1, init_filters=64, *args, **kwargs):
        """Discriminator with configurable initial filter size
        
        Args:
            in_channels: Number of input channels per signal
            init_filters: Initial filter size for the first layer
        """
        super(Discriminator, self).__init__(*args, **kwargs)
        
        # Parameter validation
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if init_filters <= 0:
            raise ValueError("init_filters must be positive")
        
        # Store parameters
        self.in_channels = in_channels
        self.init_filters = init_filters

        def discriminator_block(in_filters, out_filters, ksize=4, stride=2, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            # Calculate padding to maintain size/stride
            padding = ksize // 2
            layers = [nn.Conv1d(in_filters, out_filters, ksize,
                               stride=stride, padding=padding, padding_mode='replicate')]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Calculate filter sizes based on init_filters
        f1 = init_filters          # First level filters
        f2 = init_filters * 2      # Second level filters  
        f3 = init_filters * 4      # Third level filters
        f4 = init_filters * 8      # Fourth level filters

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, f1, normalization=False),  # size -> size/2
            *discriminator_block(f1, f2),                                    # size/2 -> size/4
            *discriminator_block(f2, f3),                                    # size/4 -> size/8
            *discriminator_block(f3, f4),                                    # size/8 -> size/16
            nn.Conv1d(f4, 1, 4, stride=1, padding=1, padding_mode='replicate')  # size/16 -> size/16
        )

        # Automatically initialize weights on creation
        self.apply(weights_init_normal)

    def extract_input(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract input for P2E-WGAN.
        
        This method handles the unified input processing for P2E-WGAN, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, 1, L] for P2E-WGAN
        
        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask
        
        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        # Get common input extraction from parent class
        x = super().extract_input(batch_dict)
        
        return {
            "x": x
        }

    def forward(self, batch_dict, signal_B):
        """
        Forward pass for Discriminator using the standardized interface.
        Args:
            batch_dict: Output of DataLoader collate_fn (standardized dict) or pre-processed tensor
            signal_B: The target waveform (real or fake) [B, 1, L]
            source_channel: Index of the source channel (for condition signal, ignored if batch_dict is tensor)
            target_channel: Index of the target channel (for real/fake signal, ignored if batch_dict is tensor)
        Returns:
            torch.Tensor: Discriminator output
        """
        if isinstance(batch_dict, torch.Tensor):
            signal_A = batch_dict
        else:
            x = self.extract_input(batch_dict)
            signal_A = x["x"]
            
        signal_input = torch.cat((signal_A, signal_B), 1)
        return self.model(signal_input)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(D, real_samples, fake_samples, real_A, patch, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) 
                   * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, real_A)
    fake = torch.full(
        (real_samples.shape[0], *patch), 1, dtype=torch.float, device=device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class P2EWGAN(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, generator_init_filters=128, discriminator_init_filters=64, *args, **kwargs):
        super(P2EWGAN, self).__init__(*args, **kwargs)
        
        # Parameter validation
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if generator_init_filters <= 0:
            raise ValueError("generator_init_filters must be positive")
        if discriminator_init_filters <= 0:
            raise ValueError("discriminator_init_filters must be positive")
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.generator_init_filters = generator_init_filters
        self.discriminator_init_filters = discriminator_init_filters
        
        self.generator = GeneratorUNet(in_channels, out_channels, generator_init_filters, *args, **kwargs)
        self.discriminator = Discriminator(in_channels, discriminator_init_filters, *args, **kwargs)

cs = ConfigStore.instance()
cs.store(name="base_p2ewgan", group="model", node=P2EWGANConfig)