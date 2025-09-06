"""
Wasserstein Gradient Penalty criterion for WGAN-GP.

Implements the gradient penalty as a modular criterion for use in composite GAN losses.
"""

import torch
import torch.autograd as autograd
from dataclasses import dataclass
from typing import Optional, Tuple
from .base_criterion import BaseCriterion, ReductionType, CriterionBaseConfig

@dataclass
class WGANGradientPenaltyConfig(CriterionBaseConfig):
    """
    Configuration for WGAN Gradient Penalty criterion.

    Attributes:
        lambda_gp (float): Gradient penalty weight. Default: 10.0
    """
    _target_: str = "src.criterions.gradient_penalty.WGANGradientPenalty"
    name: str = "wgan_gradient_penalty"
    reduction: ReductionType = ReductionType.MEAN
    lambda_gp: float = 10.0

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'lambda_gp': self.lambda_gp,
            'device': self.device,
            'name': self.name,
            'log_loss': self.log_loss,
            'enabled': self.enabled
        }

    def __str__(self) -> str:
        return f"WGANGradientPenaltyConfig(lambda_gp={self.lambda_gp}, enabled={self.enabled})"

class WGANGradientPenalty(BaseCriterion):
    """
    WGAN-GP Gradient Penalty criterion.

    Args:
        lambda_gp (float): Gradient penalty weight. Default: 10.0
        device (Optional[torch.device]): Device to compute the loss on. Default: None
        name (Optional[str]): Name for logging. Default: None
        log_loss (bool): Whether to log loss values. Default: False
    """
    def __init__(
        self,
        lambda_gp: float = 10.0,
        *args,
        **kwargs
    ):
        # Call parent constructor with explicit parameters
        super().__init__(
            *args,
            **kwargs
        )
        
        # Store additional fields specific to this criterion
        self.lambda_gp = lambda_gp

    def get_interpolates(self, real_samples: torch.Tensor, fake_samples: torch.Tensor, discriminator: torch.nn.Module, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate interpolated samples and compute discriminator outputs for gradient penalty.
        
        Args:
            real_samples: Real target samples [B, 1, L] 
            fake_samples: Fake target samples [B, 1, L]
            discriminator: Discriminator model
            condition: Condition signal (source) [B, 1, L]
            
        Returns:
            interpolates: Interpolated samples [B, 1, L]
            d_interpolates: Discriminator output on interpolated samples [B, 1, L]
        """
        # Create random interpolation weights
        alpha = torch.rand((real_samples.size(0), 1, 1), device=real_samples.device)
        
        # Create interpolated samples between real and fake TARGET data
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Call discriminator with vanilla signature: D(interpolates, real_A)
        d_interpolates = discriminator(interpolates, condition)
        
        return interpolates, d_interpolates
    
    def forward(
        self,
        real_samples: torch.Tensor,
        interpolates: torch.Tensor,
        d_interpolates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient penalty for WGAN-GP.

        Args:
            real_samples (torch.Tensor): Real samples [B, 1, L]
            interpolates (torch.Tensor): Interpolated samples
            d_interpolates (torch.Tensor): Discriminator output on interpolated samples

        Returns:
            torch.Tensor: Scaled gradient penalty (lambda_gp * penalty)
        """
        # Extract output shape from real_samples
        output_shape = d_interpolates.shape[1:]
        fake = torch.ones((real_samples.shape[0], *output_shape), device=real_samples.device, dtype=real_samples.dtype)
        # fake = torch.ones_like(d_interpolates)
        
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

        scaled_penalty = self.lambda_gp * gradient_penalty
        return scaled_penalty

# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_wgan_gradient_penalty", node=WGANGradientPenaltyConfig) 