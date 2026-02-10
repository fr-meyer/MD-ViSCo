"""SGD optimizer implementation with Hydra configuration support."""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from .base import BaseOptimizer
from .base import BaseOptimizerConfig


@dataclass
class SGDConfig(BaseOptimizerConfig):
    """Configuration for SGD optimizer."""

    _target_: str = "src.optimizers.sgd.SGDOptimizer"
    name: str = "SGD"
    lr: float = 0.001
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


class SGDOptimizer(BaseOptimizer, torch.optim.SGD):
    """SGD optimizer with Hydra configuration support.

    Uses multiple inheritance: BaseOptimizer mixin + torch.optim.SGD implementation.
    """

    def __init__(
        self,
        params,
        name: str = "SGD",
        lr: float = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        **kwargs,
    ):
        """Initialize SGD optimizer.

        Args:
            params: Model parameters to optimize
            name: Name for logging
            lr: Learning rate
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Enables Nesterov momentum
            **kwargs: Additional keyword arguments forwarded to ``torch.optim.SGD``
                (e.g. ``maximize``, ``foreach``, ``differentiable``, ``fused``).
        """
        BaseOptimizer.__init__(self, name=name)

        # Avoid passing duplicate arguments to torch.optim.SGD.__init__
        for duplicate_key in (
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
        ):
            kwargs.pop(duplicate_key, None)

        sgd_kwargs = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        sgd_kwargs.update(kwargs)
        torch.optim.SGD.__init__(self, params, **sgd_kwargs)


cs = ConfigStore.instance()
cs.store(name="base_sgd", node=SGDConfig, group="optimizer")
