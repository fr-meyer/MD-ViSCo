"""Adam optimizer implementation with Hydra configuration support."""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from .base import BaseOptimizer
from .base import BaseOptimizerConfig


@dataclass
class AdamConfig(BaseOptimizerConfig):
    """Configuration for Adam optimizer."""

    _target_: str = "src.optimizers.adam.AdamOptimizer"
    name: str = "Adam"
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


class AdamOptimizer(BaseOptimizer, torch.optim.Adam):
    """Adam optimizer with Hydra configuration support.

    Uses multiple inheritance: BaseOptimizer mixin + torch.optim.Adam implementation.
    """

    def __init__(
        self,
        params,
        name: str = "Adam",
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        **kwargs,
    ):
        """Initialize Adam optimizer.

        Args:
            params: Model parameters to optimize
            name: Name for logging
            lr: Learning rate
            betas: Coefficients for running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Whether to use the AMSGrad variant
            **kwargs: Additional keyword arguments forwarded to ``torch.optim.Adam``
                (e.g. ``maximize``, ``foreach``, ``capturable``, ``differentiable``,
                ``fused``).
        """
        BaseOptimizer.__init__(self, name=name)

        # Avoid passing duplicate arguments to torch.optim.Adam.__init__
        for duplicate_key in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
            kwargs.pop(duplicate_key, None)

        adam_kwargs = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        adam_kwargs.update(kwargs)
        torch.optim.Adam.__init__(self, params, **adam_kwargs)


cs = ConfigStore.instance()
cs.store(name="base_adam", node=AdamConfig, group="optimizer")
