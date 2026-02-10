"""Adversarial Loss criterion for WGAN (Wasserstein GAN).

This module provides a simple adversarial loss implementation for use in GAN training.
"""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType


@dataclass
class WGANAdversarialLossConfig(CriterionBaseConfig):
    """Configuration for WGAN Adversarial Loss criterion.

    Attributes:
        reduction: Reduction method. Default: ReductionType.MEAN.
    """

    _target_: str = "src.criterions.adversarial_loss.WGANAdversarialLoss"
    name: str = "wgan_adversarial_loss"
    reduction: ReductionType = ReductionType.MEAN

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(
                f"reduction must be a ReductionType enum, got {type(self.reduction)}"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "reduction": self.reduction.value,
            "device": self.device,
            "name": self.name,
            "log_loss": self.log_loss,
            "enabled": self.enabled,
        }

    def __str__(self) -> str:
        """Return string representation of configuration."""
        return (
            "WGANAdversarialLossConfig("
            f"reduction='{self.reduction.value}', enabled={self.enabled})"
        )


class WGANAdversarialLoss(BaseCriterion):
    """Wasserstein Adversarial Loss for GAN training."""

    def __init__(self, *args, **kwargs):
        """Initialize WGAN Adversarial Loss.

        Args:
            *args: Variable positional arguments passed to parent.
            **kwargs: Variable keyword arguments passed to parent.
        """
        super().__init__(*args, **kwargs)

    def forward(
        self, fake_out: torch.Tensor, real_out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the adversarial loss for the generator in WGAN.

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
        """Compute the adversarial loss for the generator in WGAN.

        Args:
            fake_out (torch.Tensor): Discriminator output for fake samples

        Returns:
            torch.Tensor: Adversarial loss (scalar) - negative of mean fake
            output to maximize critic score.
        """
        # Match P2E-WGAN implementation: loss_GAN = -mean(D(fake))
        g_loss = -fake_out
        return self._apply_reduction(g_loss)

    def discriminator_loss(
        self, real_out: torch.Tensor, fake_out: torch.Tensor
    ) -> torch.Tensor:
        """Compute the adversarial loss for the discriminator in WGAN.

        Args:
            real_out (torch.Tensor): Discriminator output for real samples
            fake_out (torch.Tensor): Discriminator output for fake samples

        Returns:
            torch.Tensor: Adversarial loss (scalar) - mean fake minus mean real
        """
        d_loss = fake_out - real_out
        return self._apply_reduction(d_loss)


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="criterion", name="base_wgan_adversarial_loss", node=WGANAdversarialLossConfig
)
