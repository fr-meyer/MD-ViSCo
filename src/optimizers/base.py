"""Base optimizer class for PyTorch optimizers.

Includes common functionality like name handling and logging. All custom
optimizers should inherit from BaseOptimizer.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any
from typing import Protocol

# Third-party imports
from omegaconf import MISSING


@dataclass
class BaseOptimizerConfig:
    """Pure data container for optimizer configuration.

    Attributes:
        _target_: Hydra instantiation target (e.g., module path to optimizer class).
        _partial_: If True, Hydra uses partial instantiation; inherited by subclasses.
        name: Human-readable name for logging. Default: "Optimizer".
        lr: Learning rate. Default: 0.001.
    """

    _target_: str = MISSING
    _partial_: bool = True
    name: str = "Optimizer"
    lr: float = 0.001


class OptimizerProtocol(Protocol):
    """Protocol for PyTorch optimizer interface."""

    param_groups: list[dict[str, Any]]


class BaseOptimizer:
    """Base mixin class for all optimizers in the MD-ViSCo project.

    This class provides a common interface and functionality for all optimizers,
    using multiple inheritance with PyTorch optimizer classes.
    """

    name: str
    param_groups: list[dict[str, Any]]  # Populated by torch.optim.Optimizer

    def __init__(self, name: str = "Optimizer", *args, **kwargs):
        """Initialize the base optimizer mixin.

        Args:
            name: Name for logging purposes.
            *args: Additional positional arguments passed to parent optimizer class.
            **kwargs: Additional keyword arguments passed to parent optimizer class.

        Note:
            Uses cooperative multiple inheritance. The PyTorch optimizer's
            __init__ method should be called after this mixin's __init__.
        """
        self.name = name

    def log_state(self) -> dict[str, object]:
        """Log current optimizer state for debugging/monitoring.

        Returns:
            Dictionary containing optimizer state information
        """
        state = {
            "name": self.name,
            "param_groups": len(self.param_groups),
            "lr": self.param_groups[0]["lr"] if len(self.param_groups) > 0 else None,
        }
        return state
