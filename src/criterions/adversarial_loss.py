"""
Adversarial Loss criterion for WGAN (Wasserstein GAN).

This module provides a simple adversarial loss implementation for use in GAN training.
"""

import torch
from dataclasses import dataclass
from typing import Optional
from .base_criterion import BaseCriterion, CriterionBaseConfig, ReductionType

@dataclass
class WGANAdversarialLossConfig(CriterionBaseConfig):
    """
    Configuration for WGAN Adversarial Loss criterion.

    Attributes:
        reduction (ReductionType): Reduction method. Default: ReductionType.MEAN
    """
    _target_: str = "src.criterions.adversarial_loss.WGANAdversarialLoss"
    name: str = "wgan_adversarial_loss"
    reduction: ReductionType = ReductionType.MEAN

    def __post_init__(self):
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(f"reduction must be a ReductionType enum, got {type(self.reduction)}")

    def to_dict(self) -> dict:
        return {
            'reduction': self.reduction.value,
            'device': self.device,
            'name': self.name,
            'log_loss': self.log_loss,
            'enabled': self.enabled
        }

    def __str__(self) -> str:
        return (f"WGANAdversarialLossConfig(reduction='{self.reduction.value}', enabled={self.enabled})")

class WGANAdversarialLoss(BaseCriterion):
    """
    Wasserstein Adversarial Loss for GAN training.
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        # Call parent constructor with explicit parameters
        super().__init__(
            *args,
            **kwargs
        )

    def forward(self, fake_out: torch.Tensor, real_out: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the adversarial loss for the generator in WGAN.
        Args:
            fake_out (torch.Tensor): Discriminator output for fake samples
            real_out (torch.Tensor): Discriminator output for real samples
        Returns:
            torch.Tensor: Adversarial loss (scalar)
        """
        if real_out is None:
            adv_loss = self.generator_loss(fake_out)
        else:
            adv_loss = self.discriminator_loss(real_out, fake_out)
        return adv_loss

    def generator_loss(self, fake_out: torch.Tensor) -> torch.Tensor:
        """
        Compute the adversarial loss for the generator in WGAN.
        Args:
            fake_out (torch.Tensor): Discriminator output for fake samples
        Returns:
            torch.Tensor: Adversarial loss (scalar)
        """
        # adv_loss = -torch.mean(fake_out)
        return fake_out

    def discriminator_loss(self, real_out: torch.Tensor, fake_out: torch.Tensor) -> torch.Tensor:
        """
        Compute the adversarial loss for the discriminator in WGAN.
        Args:
            real_out (torch.Tensor): Discriminator output for real samples
            fake_out (torch.Tensor): Discriminator output for fake samples
        Returns:
            torch.Tensor: Adversarial loss (scalar)
        """
        # disc_loss = torch.mean(fake_out) - torch.mean(real_out)
        return fake_out - real_out

# Register config with Hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_wgan_adversarial_loss", node=WGANAdversarialLossConfig) 