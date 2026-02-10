"""ReduceLROnPlateau scheduler implementation with Hydra configuration support."""

# Standard library imports
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from .base import BaseScheduler
from .base import BaseSchedulerConfig


@dataclass
class ReduceLROnPlateauConfig(BaseSchedulerConfig):
    """Configuration for ReduceLROnPlateau scheduler."""

    _target_: str = "src.schedulers.reduce_lr_on_plateau.ReduceLROnPlateauScheduler"
    name: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8


class ReduceLROnPlateauScheduler(
    BaseScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
):
    """ReduceLROnPlateau scheduler with Hydra configuration support.

    Uses multiple inheritance: BaseScheduler mixin +
    torch.optim.lr_scheduler.ReduceLROnPlateau implementation.

    Attributes:
        name: Name for logging purposes (inherited from BaseScheduler)
        mode: One of 'min', 'max'. In 'min' mode, lr will be reduced when the
            quantity monitored has stopped decreasing
        factor: Factor by which the learning rate will be reduced
        patience: Number of epochs with no improvement after which learning rate
            will be reduced
        threshold: Threshold for measuring the new optimum
        threshold_mode: One of 'rel', 'abs'. In 'rel' mode,
            dynamic_threshold = best * (1 + threshold)
        cooldown: Number of epochs to wait before resuming normal operation after
            lr has been reduced
        min_lr: Lower bound on the learning rate
        eps: Minimal decay applied to lr

    Notes:
        This scheduler uses multiple inheritance to combine MD-ViSCo-specific
        functionality (name, logging) with PyTorch's ReduceLROnPlateau implementation.
        The scheduler is registered with Hydra's ConfigStore for configuration
        management. Instances are typically created via `trainer.create_scheduler()`
        which uses Hydra's partial instantiation pattern.

        Note: Scheduler mode is independent from EarlyStopping mode.
        You can have different modes for each (e.g., scheduler on loss, early
        stopping on accuracy).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        verbose: bool = False,
        *,
        name: str = "ReduceLROnPlateau",
        **kwargs: Any,
    ):
        """Initialize ReduceLROnPlateau scheduler.

        Args:
            optimizer: Optimizer to schedule
            name: Name for logging
            mode: One of 'min', 'max'. In 'min' mode, lr will be reduced when the
                quantity monitored has stopped decreasing
            factor: Factor by which the learning rate will be reduced
            patience: Number of epochs with no improvement after which learning
                rate will be reduced
            threshold: Threshold for measuring the new optimum
            threshold_mode: One of 'rel', 'abs'. In 'rel' mode,
                dynamic_threshold = best * (1 + threshold)
            cooldown: Number of epochs to wait before resuming normal operation
                after lr has been reduced
            min_lr: A scalar or a list of scalars. Lower bound on the learning rate
            eps: Minimal decay applied to lr
            verbose: If True, prints a message to stdout for each update
            **kwargs: Additional arguments for extensibility
        """
        BaseScheduler.__init__(self, name=name, **kwargs)

        torch.optim.lr_scheduler.ReduceLROnPlateau.__init__(
            self,
            optimizer,
            mode=cast("Literal['min', 'max']", mode),
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=cast("Literal['rel', 'abs']", threshold_mode),
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )


cs = ConfigStore.instance()
cs.store(
    name="base_reduce_lr_on_plateau", node=ReduceLROnPlateauConfig, group="scheduler"
)
