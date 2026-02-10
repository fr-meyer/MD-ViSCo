"""Wasserstein Gradient Penalty criterion for WGAN-GP.

Implements the gradient penalty as a modular criterion for use in composite GAN losses.
"""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import torch
import torch.autograd as autograd
from hydra.core.config_store import ConfigStore

# Local imports
from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType


@dataclass
class WGANGradientPenaltyConfig(CriterionBaseConfig):
    """Configuration for WGAN Gradient Penalty criterion.

    Attributes:
        lambda_gp: Gradient penalty weight. Default: 10.0.
    """

    _target_: str = "src.criterions.gradient_penalty.WGANGradientPenalty"
    name: str = "wgan_gradient_penalty"
    reduction: ReductionType = ReductionType.MEAN
    lambda_gp: float = 10.0

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "reduction": self.reduction.value,
            "lambda_gp": self.lambda_gp,
            "device": self.device,
            "log_loss": self.log_loss,
            "enabled": self.enabled,
        }

    def __str__(self) -> str:
        """Return string representation of configuration."""
        return (
            f"WGANGradientPenaltyConfig(lambda_gp={self.lambda_gp}, "
            f"enabled={self.enabled})"
        )


class WGANGradientPenalty(BaseCriterion):
    """WGAN-GP Gradient Penalty criterion.

    Args:
        lambda_gp (float): Gradient penalty weight. Default: 10.0
        device (Optional[torch.device]): Device to compute the loss on. Default: None
        name (Optional[str]): Name for logging. Default: None
        log_loss (bool): Whether to log loss values. Default: False
    """

    def __init__(self, lambda_gp: float = 10.0, *args, **kwargs):
        """Initialize WGAN Gradient Penalty.

        Args:
            lambda_gp: Gradient penalty weight.
            *args: Variable positional arguments passed to parent.
            **kwargs: Variable keyword arguments passed to parent.
        """
        super().__init__(*args, **kwargs)

        # Store additional fields specific to this criterion
        self.lambda_gp = lambda_gp

    def get_interpolates(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        discriminator: torch.nn.Module,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate interpolated samples and discriminator outputs for gradient penalty.

        Args:
            real_samples: Real target samples [B, 1, L]
            fake_samples: Fake target samples [B, 1, L]
            discriminator: Discriminator model
            condition: Condition signal (source) [B, 1, L]

        Returns:
            interpolates: Interpolated samples [B, 1, L]
            d_interpolates: Discriminator output on interpolated samples [B, 1, L]
        """
        # Random alpha for WGAN-GP interpolated samples.
        alpha = torch.rand((real_samples.size(0), 1, 1), device=real_samples.device)

        # Interpolated samples for WGAN-GP gradient penalty.
        interpolates = (
            alpha * real_samples + (1 - alpha) * fake_samples
        ).requires_grad_(True)

        # D(interpolates, real_A) for WGAN-GP gradient penalty
        d_interpolates = discriminator(interpolates, condition)

        return interpolates, d_interpolates

    def forward(
        self,
        real_samples: torch.Tensor,
        interpolates: torch.Tensor,
        d_interpolates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the gradient penalty for WGAN-GP.

        Args:
            real_samples (torch.Tensor): Real samples [B, 1, L]
            interpolates (torch.Tensor): Interpolated samples
            d_interpolates (torch.Tensor): Discriminator output on interpolated samples

        Returns:
            torch.Tensor: Scaled gradient penalty (lambda_gp * penalty)
        """
        # Extract output shape from real_samples
        output_shape = d_interpolates.shape[1:]
        fake = torch.ones(
            (real_samples.shape[0], *output_shape),
            device=real_samples.device,
            dtype=real_samples.dtype,
        )

        grad_output = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        if grad_output is None:
            raise RuntimeError("autograd.grad returned None for gradient penalty")
        gradients = grad_output.view(grad_output.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        scaled_penalty = self.lambda_gp * gradient_penalty
        return scaled_penalty


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    group="criterion", name="base_wgan_gradient_penalty", node=WGANGradientPenaltyConfig
)
