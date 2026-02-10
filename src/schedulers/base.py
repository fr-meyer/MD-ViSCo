"""Base scheduler class for PyTorch learning rate schedulers.

Includes common functionality like name handling and logging. All custom
schedulers should inherit from BaseScheduler.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any

# Third-party imports
from omegaconf import MISSING


@dataclass
class BaseSchedulerConfig:
    """Pure data container for scheduler configuration.

    Attributes:
        _target_: Hydra instantiation target (e.g., module path to scheduler class).
        _partial_: If True, Hydra uses partial instantiation; inherited by subclasses.
        name: Human-readable name for logging. Default: "Scheduler".
    """

    _target_: str = MISSING
    _partial_: bool = True
    name: str = "Scheduler"


class BaseScheduler:
    """Base mixin class for all schedulers in the MD-ViSCo project.

    This class provides a common interface and functionality for all schedulers,
    using multiple inheritance with PyTorch scheduler classes.

    Attributes:
        name: Name for logging purposes
        last_epoch: The index of the last epoch (inherited from PyTorch scheduler)
        base_lrs: List of initial learning rates (inherited from PyTorch scheduler)

    Notes:
        This class uses cooperative multiple inheritance with PyTorch scheduler classes.
        The PyTorch scheduler's __init__ method should be called after this mixin's
        __init__ to ensure proper initialization order. This pattern allows schedulers
        to inherit both the MD-ViSCo-specific functionality (name, logging) and the
        PyTorch scheduler implementation.
    """

    def __init__(self, name: str = "Scheduler", *args: Any, **kwargs: Any):
        """Initialize the base scheduler mixin.

        Args:
            name: Name for logging purposes.
            *args: Additional positional arguments passed to parent scheduler class.
            **kwargs: Additional keyword arguments passed to parent scheduler class.

        Note:
            Uses cooperative multiple inheritance. The PyTorch scheduler's
            __init__ method should be called after this mixin's __init__.
        """
        self.name = name

    def log_state(self) -> dict[str, Any]:
        """Log current scheduler state for debugging/monitoring.

        Returns:
            Dictionary containing scheduler state information
        """
        state = {
            "name": self.name,
            "last_epoch": getattr(self, "last_epoch", None),
        }

        # Add scheduler-specific state if available (getattr for mixin typing)
        base_lrs = getattr(self, "base_lrs", None)
        if base_lrs is not None:
            state["base_lrs"] = base_lrs
        get_last_lr = getattr(self, "get_last_lr", None)
        if get_last_lr is not None:
            state["last_lr"] = get_last_lr()

        return state
