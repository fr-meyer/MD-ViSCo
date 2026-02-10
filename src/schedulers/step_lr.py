"""StepLR scheduler implementation with Hydra configuration support."""

# Standard library imports
from dataclasses import dataclass
from typing import Any

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from .base import BaseScheduler
from .base import BaseSchedulerConfig


@dataclass
class StepLRConfig(BaseSchedulerConfig):
    """Configuration for StepLR scheduler."""

    _target_: str = "src.schedulers.step_lr.StepLRScheduler"
    name: str = "StepLR"
    step_size: int = 30
    gamma: float = 0.1
    last_epoch: int = -1


class StepLRScheduler(BaseScheduler, torch.optim.lr_scheduler.StepLR):
    """StepLR scheduler with Hydra configuration support.

    Uses multiple inheritance: BaseScheduler mixin +
    torch.optim.lr_scheduler.StepLR implementation.

    Attributes:
        name: Name for logging purposes (inherited from BaseScheduler)
        last_epoch: The index of the last epoch (inherited from PyTorch scheduler)
        base_lrs: List of initial learning rates (inherited from PyTorch scheduler)
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay

    Notes:
        This scheduler uses multiple inheritance to combine MD-ViSCo-specific
        functionality (name, logging) with PyTorch's StepLR implementation.
        The scheduler is registered with Hydra's ConfigStore for configuration
        management. Instances are typically created via `trainer.create_scheduler()`
        which uses Hydra's partial instantiation pattern.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int = 30,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
        *,
        name: str = "StepLR",
        **kwargs: Any,
    ):
        """Initialize StepLR scheduler.

        Args:
            optimizer: Optimizer to schedule
            name: Name for logging
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
            **kwargs: Additional arguments for extensibility

        Note:
            verbose is passed to PyTorch as the string "true" or "false" for
            internal type-stub compatibility.
        """
        BaseScheduler.__init__(self, name=name, **kwargs)

        torch.optim.lr_scheduler.StepLR.__init__(
            self,
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
            verbose="true" if verbose else "false",  # stubs expect str
        )


cs = ConfigStore.instance()
cs.store(name="base_step_lr", node=StepLRConfig, group="scheduler")
