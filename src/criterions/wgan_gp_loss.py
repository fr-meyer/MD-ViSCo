"""
Composite WGAN-GP Loss for GAN training.

Combines adversarial, sample (MSE), and gradient penalty losses using modular sub-criterions.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
from omegaconf import MISSING
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType
from src.criterions.adversarial_loss import WGANAdversarialLossConfig, WGANAdversarialLoss
from src.criterions.gradient_penalty import WGANGradientPenaltyConfig, WGANGradientPenalty
from src.criterions.mse_loss import MSELossConfig, MSELoss


@dataclass
class WGANGPLossConfig(CriterionBaseConfig):
    adv_config: WGANAdversarialLossConfig = MISSING
    mse_config: MSELossConfig = MISSING
    gp_config: WGANGradientPenaltyConfig = MISSING
    lambda_sample: float = 50.0
    _target_: str = "src.criterions.wgan_gp_loss.WGANGPLoss"
    name: str = "wgan_gp_loss"
    reduction: ReductionType = ReductionType.MEAN

    def to_dict(self):
        return {
            "adv_config": self.adv_config,
            "mse_config": self.mse_config,
            "gp_config": self.gp_config,
            "lambda_sample": self.lambda_sample,
        }

    def __str__(self):
        return f"WGANGPLossConfig(lambda_sample={self.lambda_sample})"

class WGANGPLoss(BaseCriterion):
    """
    Composite WGAN-GP Loss for GAN training.

    Args:
        adv_config: Instantiated WGANAdversarialLoss object
        mse_config: Instantiated MSE loss criterion object
        gp_config: Instantiated WGANGradientPenalty object
        lambda_sample: Weight for MSE loss
    """
    def __init__(
        self,
        adv_config: WGANAdversarialLoss,
        mse_config: MSELoss,
        gp_config: WGANGradientPenalty,
        lambda_sample: float = 50.0,
        *args,
        **kwargs
    ):
        # Call parent constructor with explicit parameters
        super().__init__(
            *args,
            **kwargs
        )
        
        # Store references to instantiated sub-criteria directly
        # Hydra will have already instantiated these objects before calling this constructor
        self.adv_loss = adv_config
        self.mse_loss = mse_config
        self.gp_loss = gp_config
        self.lambda_sample = lambda_sample

    def generator_loss(
        self, fake_pred: torch.Tensor, fake_data: torch.Tensor, real_data: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        adv = self.adv_loss(fake_pred)
        mse = self.mse_loss(fake_data, real_data)
        total = adv + self.lambda_sample * mse
        return total, {"adv_loss": adv.item(), "mse_loss": mse.item(), "total_g_loss": total.item()}

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        real_data: torch.Tensor,
        interpolates: torch.Tensor,
        d_interpolates: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        wasserstein = self.adv_loss(real_pred, fake_pred)
        gp = self.gp_loss(
            real_data, interpolates, d_interpolates
        )
        total = wasserstein + gp
        return total, {
            "wasserstein_loss": wasserstein.item(),
            "gp_loss": gp.item(),
            "total_d_loss": total.item(),
        }

    def forward(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        interpolates: torch.Tensor = None,
        d_interpolates: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute WGAN-GP loss with simplified parameters.
        
        Args:
            real_data: Real target data tensor
            fake_data: Generated fake data tensor
            real_pred: Discriminator prediction on real data
            fake_pred: Discriminator prediction on fake data
            interpolates: Interpolated samples for gradient penalty
            d_interpolates: Discriminator output on interpolated samples
            
        Returns:
            torch.Tensor: Computed loss (generator or discriminator)
        """
        if interpolates is None or d_interpolates is None:
            g_loss, g_dict = self.generator_loss(fake_pred, fake_data, real_data)
            return g_loss
        else:
            d_loss, d_dict = self.discriminator_loss(
                real_pred, fake_pred, real_data, interpolates, d_interpolates
            )
            return d_loss
        

# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_wgan_gp_loss", node=WGANGPLossConfig) 