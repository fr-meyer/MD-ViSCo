"""Composite WGAN-GP Loss for GAN training.

Combines adversarial, sample (MSE), and gradient penalty losses using
modular sub-criterions.
"""

# Standard library imports
from dataclasses import dataclass

import torch

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Local imports
from src.criterions.adversarial_loss import WGANAdversarialLoss
from src.criterions.adversarial_loss import WGANAdversarialLossConfig
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType
from src.criterions.gradient_penalty import WGANGradientPenalty
from src.criterions.gradient_penalty import WGANGradientPenaltyConfig
from src.criterions.mse_loss import MSELoss
from src.criterions.mse_loss import MSELossConfig


@dataclass
class WGANGPLossConfig(CriterionBaseConfig):
    """Configuration for WGAN-GP Loss criterion.

    Attributes:
        adv_config: Configuration for adversarial loss component.
        mse_config: Configuration for MSE loss component.
        gp_config: Configuration for gradient penalty component.
        lambda_sample: Weight for MSE loss component.
    """

    adv_config: WGANAdversarialLossConfig = MISSING
    mse_config: MSELossConfig = MISSING
    gp_config: WGANGradientPenaltyConfig = MISSING
    lambda_sample: float = 50.0
    _target_: str = "src.criterions.wgan_gp_loss.WGANGPLoss"
    name: str = "wgan_gp_loss"
    reduction: ReductionType = ReductionType.MEAN

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "adv_config": self.adv_config,
            "mse_config": self.mse_config,
            "gp_config": self.gp_config,
            "lambda_sample": self.lambda_sample,
        }

    def __str__(self) -> str:
        """Return string representation of configuration."""
        return f"WGANGPLossConfig(lambda_sample={self.lambda_sample})"


class WGANGPLoss(BaseCriterion):
    """Composite WGAN-GP Loss for GAN training.

    Attributes:
        adv_loss: Instantiated WGANAdversarialLoss.
        mse_loss: Instantiated MSE loss criterion.
        gp_loss: Instantiated WGANGradientPenalty.
        lambda_sample: Weight for MSE loss component.
    """

    def __init__(
        self,
        adv_config: WGANAdversarialLoss,
        mse_config: MSELoss,
        gp_config: WGANGradientPenalty,
        lambda_sample: float = 50.0,
        *args,
        **kwargs,
    ):
        """Initialize WGAN-GP Loss.

        Args:
            adv_config: Instantiated WGANAdversarialLoss object.
            mse_config: Instantiated MSE loss criterion object.
            gp_config: Instantiated WGANGradientPenalty object.
            lambda_sample: Weight for MSE loss component.
            *args: Variable positional arguments passed to parent.
            **kwargs: Variable keyword arguments passed to parent.
        """
        super().__init__(*args, **kwargs)

        # Hydra instantiates sub-criteria before calling this constructor
        self.adv_loss = adv_config
        self.mse_loss = mse_config
        self.gp_loss = gp_config
        self.lambda_sample = lambda_sample

    def generator_loss(
        self, fake_pred: torch.Tensor, fake_data: torch.Tensor, real_data: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute generator loss combining adversarial and MSE components.

        Args:
            fake_pred: Discriminator prediction on fake data.
            fake_data: Generated fake data tensor.
            real_data: Real target data tensor.

        Returns:
            Tuple containing total loss and dictionary of loss components.
        """
        adv_out = self.adv_loss(fake_pred)
        adv: torch.Tensor = (
            adv_out["total_loss"]
            if isinstance(adv_out, dict)
            else (adv_out[0] if isinstance(adv_out, tuple) else adv_out)
        )
        mse_out = self.mse_loss(fake_data, real_data)
        mse: torch.Tensor = (
            mse_out["total_loss"]
            if isinstance(mse_out, dict)
            else (mse_out[0] if isinstance(mse_out, tuple) else mse_out)
        )
        total = adv + self.lambda_sample * mse
        return total, {
            "adv_loss": adv.item(),
            "mse_loss": mse.item(),
            "total_g_loss": total.item(),
        }

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        real_data: torch.Tensor,
        interpolates: torch.Tensor,
        d_interpolates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute discriminator loss (Wasserstein distance + gradient penalty).

        Args:
            real_pred: Discriminator prediction on real data.
            fake_pred: Discriminator prediction on fake data.
            real_data: Real target data tensor.
            interpolates: Interpolated samples for gradient penalty.
            d_interpolates: Discriminator output on interpolated samples.

        Returns:
            Tuple containing total loss and dictionary of loss components.
        """
        w_out = self.adv_loss(fake_pred, real_pred)
        wasserstein: torch.Tensor = (
            w_out["total_loss"]
            if isinstance(w_out, dict)
            else (w_out[0] if isinstance(w_out, tuple) else w_out)
        )
        gp_out = self.gp_loss(real_data, interpolates, d_interpolates)
        gp: torch.Tensor = (
            gp_out["total_loss"]
            if isinstance(gp_out, dict)
            else (gp_out[0] if isinstance(gp_out, tuple) else gp_out)
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
        interpolates: torch.Tensor | None = None,
        d_interpolates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute WGAN-GP loss with simplified parameters.

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
            g_loss, _g_dict = self.generator_loss(fake_pred, fake_data, real_data)
            return g_loss
        else:
            d_loss, _d_dict = self.discriminator_loss(
                real_pred, fake_pred, real_data, interpolates, d_interpolates
            )
            return d_loss


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(group="criterion", name="base_wgan_gp_loss", node=WGANGPLossConfig)
