"""Base criterion class for PyTorch loss functions.

Includes common functionality like reduction methods, device handling,
and logging. All custom loss functions should inherit from BaseCriterion.
"""

# Standard library imports
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

# Third-party imports
from omegaconf import MISSING
from torch.nn.modules.loss import _Loss


class ReductionType(Enum):
    """Enum for loss reduction types supported by all criteria."""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"

    @classmethod
    def from_string(cls, value: str) -> "ReductionType":
        """Convert string to enum value.

        Args:
            value: String to convert ('none', 'mean', or 'sum').

        Returns:
            ReductionType: The corresponding enum member.

        Raises:
            ValueError: If value is not a valid reduction type.
        """
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(
                f"Invalid reduction type '{value}'. Must be one of: "
                f"{[e.value for e in cls]}"
            ) from err

    def __str__(self) -> str:
        """Return string representation of the enum value."""
        return self.value


@dataclass
class CriterionBaseConfig:
    """Pure data container for criterion configuration.

    Attributes:
        name: Human-readable criterion name for logging.
        _target_: Full path to criterion class for Hydra instantiation.
        enabled: If True, criterion is included in loss computation.
        log_loss: If True, log loss values during computation.
        device: Optional device string for YAML compatibility (e.g. "cuda:0").
    """

    name: str = MISSING
    _target_: str = MISSING
    enabled: bool = True
    log_loss: bool = False
    device: str | None = None  # String-based for YAML compatibility


class BaseCriterion(_Loss):
    """Common interface for all criterion (loss) functions in the MD-ViSCo project.

    Initialization, reduction, device handling, and logging are shared.

    Attributes:
        reduction_enum: Reduction method for the loss (enum-typed).
        reduction: Reduction method string value (from _Loss, for compatibility).
        device: Device to compute the loss on.
        name: Name of the criterion for logging purposes.
        log_loss: Whether to log loss values during computation.
        logger: Logger instance for loss tracking.
    """

    def __init__(
        self,
        reduction: str | ReductionType = ReductionType.MEAN,
        device: torch.device | None = None,
        name: str | None = None,
        log_loss: bool = False,
        **kwargs,
    ):
        """Initialize the base criterion.

        Args:
            reduction: Reduction method for the loss. Options: 'none', 'mean',
                'sum' or ReductionType enum values. Default: ReductionType.MEAN
            device: Device to compute the loss on. If None, will use the device
                of the input tensors.
            name: Name of the criterion for logging purposes. If None, will use
                the class name.
            log_loss: Whether to log loss values during computation.
                Default: False
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__()

        if isinstance(reduction, str):
            self.reduction_enum = ReductionType.from_string(reduction)
        elif isinstance(reduction, ReductionType):
            self.reduction_enum = reduction
        else:
            raise ValueError(
                f"Invalid reduction type. Expected str or ReductionType, "
                f"got {type(reduction)}"
            )

        self.reduction = self.reduction_enum.value

        self.device = device
        self.name = name or self.__class__.__name__
        self.log_loss = log_loss

        self.logger: logging.Logger | None = (
            self._setup_logger() if self.log_loss else None
        )

        self.config = kwargs
        self.reset_stats()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for loss tracking."""
        logger = logging.getLogger(f"criterion.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def reset_stats(self):
        """Reset loss statistics."""
        self.loss_history = []
        self.total_loss = 0.0
        self.num_calls = 0

    def get_stats(self) -> dict[str, Any]:
        """Get current loss statistics."""
        if self.num_calls == 0:
            return {
                "total_loss": 0.0,
                "mean_loss": 0.0,
                "num_calls": 0,
                "loss_history": [],
            }

        return {
            "total_loss": self.total_loss,
            "mean_loss": self.total_loss / self.num_calls,
            "num_calls": self.num_calls,
            "loss_history": self.loss_history.copy(),
        }

    def _update_stats(self, loss_value: float):
        """Update loss statistics."""
        self.loss_history.append(loss_value)
        self.total_loss += loss_value
        self.num_calls += 1

        if self.log_loss and self.logger:
            self.logger.info("Loss: %.6f", loss_value)

    def _validate_inputs(self, *args, **kwargs) -> tuple[torch.Tensor, ...]:
        """Validate and prepare input tensors.

        Args:
            *args: Input tensors
            **kwargs: Additional keyword arguments

        Returns:
            Validated tensors (same device as criterion if set).

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If tensors are not on the same device.
        """
        tensors = []

        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                raise ValueError(f"Input {i} must be a torch.Tensor, got {type(arg)}")

            if self.device is not None and arg.device != self.device:
                arg = arg.to(self.device)

            tensors.append(arg)

        if len(tensors) > 1:
            devices = {t.device for t in tensors}
            if len(devices) > 1:
                raise RuntimeError(
                    f"All tensors must be on the same device. Found devices: {devices}"
                )

        return tuple(tensors)

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to the loss tensor.

        Args:
            loss: Loss tensor to reduce.

        Returns:
            Reduced loss tensor (scalar if reduction is mean/sum).
        """
        if self.reduction_enum == ReductionType.NONE:
            return loss
        elif self.reduction_enum == ReductionType.MEAN:
            return loss.mean()
        elif self.reduction_enum == ReductionType.SUM:
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction_enum}")

    def _check_gradients(self, loss: torch.Tensor) -> None:
        """Check for potential gradient issues.

        Args:
            loss: Loss tensor to check.
        """
        if not loss.requires_grad:
            warnings.warn(
                f"{self.name}: Loss tensor does not require gradients. "
                f"This might cause issues during backpropagation.",
                stacklevel=2,
            )

        if torch.isnan(loss).any():
            warnings.warn(f"{self.name}: Loss contains NaN values.", stacklevel=2)

        if torch.isinf(loss).any():
            warnings.warn(f"{self.name}: Loss contains infinite values.", stacklevel=2)

    def forward(
        self, *args, **kwargs
    ) -> torch.Tensor | dict[str, Any] | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass of the criterion.

        This method should be overridden by subclasses to implement
        the actual loss computation.

        Args:
            *args: Input tensors
            **kwargs: Additional keyword arguments

        Returns:
            Computed loss (Tensor, dict for multi-loss, or tuple for
            deep supervision).

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented by subclasses"
        )

    def compute_loss(
        self, *args, **kwargs
    ) -> torch.Tensor | dict[str, Any] | tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute the loss with validation and statistics tracking.

        Prefer calling this over forward when using reduction or stats. Handles
        input validation, device management, reduction, and statistics.
        When forward returns a non-Tensor (e.g. dict for multi-loss criteria),
        reduction, gradient checks, and stats are skipped and the result is
        returned directly.

        Args:
            *args: Input tensors (or mixed inputs for multi-loss criteria)
            **kwargs: Additional keyword arguments

        Returns:
            Computed loss (Tensor, dict, or tuple for deep supervision).
        """
        # Validate inputs only when all positional args are tensors
        if all(isinstance(a, torch.Tensor) for a in args):
            validated_args = self._validate_inputs(*args)
            loss = self.forward(*validated_args, **kwargs)
        else:
            loss = self.forward(*args, **kwargs)

        # Non-Tensor returns (dict/tuple) bypass reduction, gradient checks, and stats
        if not isinstance(loss, torch.Tensor):
            return loss

        loss = self._apply_reduction(loss)
        self._check_gradients(loss)
        if self.reduction_enum != ReductionType.NONE:
            loss_value = loss.item()
            self._update_stats(loss_value)

        return loss

    def extra_repr(self) -> str:
        """Return extra representation string."""
        extra_repr_parts = [
            f"reduction={self.reduction_enum.value}",
            f"name={self.name}",
            f"log_loss={self.log_loss}",
        ]

        if self.device is not None:
            extra_repr_parts.append(f"device={self.device}")

        if self.config:
            config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
            extra_repr_parts.append(f"config={{{config_str}}}")

        return ", ".join(extra_repr_parts)

    def to(self, *args, **kwargs) -> "BaseCriterion":
        """Move the criterion to the specified device/dtype.

        This method mirrors torch.nn.Module.to() and forwards all arguments
        to the parent class, updating self.device when a device is provided.

        Args:
            *args: Arguments forwarded to torch.nn.Module.to()
            **kwargs: Keyword arguments forwarded to torch.nn.Module.to()

        Returns:
            Self for chaining.
        """
        result = super().to(*args, **kwargs)

        # Sync criterion device with module when .to(device) is called
        if args:
            first_arg = args[0]
            if isinstance(first_arg, (torch.device, str)):
                self.device = (
                    torch.device(first_arg) if isinstance(first_arg, str) else first_arg
                )
            elif isinstance(first_arg, torch.dtype):
                pass
            elif isinstance(first_arg, torch.Tensor):
                self.device = first_arg.device

        return result

    def __call__(
        self, *args, **kwargs
    ) -> torch.Tensor | dict[str, Any] | tuple[torch.Tensor, list[torch.Tensor]]:
        """Call the criterion with validation and statistics tracking.

        Args:
            *args: Input tensors (or mixed inputs for multi-loss criteria)
            **kwargs: Additional keyword arguments

        Returns:
            Computed loss (Tensor, dict, or tuple).
        """
        return self.compute_loss(*args, **kwargs)
