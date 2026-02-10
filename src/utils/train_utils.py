"""Training utilities for early stopping and state management.

This module provides EarlyStopping for validation-loss-based stopping and
optional state serialization. Used by trainers to halt training when
validation metrics cease to improve.

Classes:
    - EarlyStopping: Early stops training when validation loss does not improve

Examples:
    >>> es = EarlyStopping(patience=10, mode='min')
    >>> es(val_loss)  # Call each epoch; check es.early_stop

See Also:
    - src.trainers.trainer: Trainer usage of EarlyStopping
"""

import logging
from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training if validation loss does not improve after given patience.

    Attributes:
        patience (int): How long to wait after last time validation loss
            improved
        verbose (bool): If True, prints a message for each validation loss
            improvement
        counter (int): The number of epochs since last improvement
        best_loss (float): Best observed validation loss
        early_stop (bool): True if training should stop
        delta (float): Minimum change in monitored quantity to qualify as an
            improvement
        threshold (float): Threshold for measuring the new optimum
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * (1 + threshold) in 'max' mode or
            best * (1 - threshold) in `min` mode. In `abs` mode,
            dynamic_threshold = best + threshold

    Note on threshold semantics in 'min' mode:
        When threshold > 0, values within the threshold are considered
        non-worsening. For example, if best_loss=1.0 and threshold=0.1, then
        values up to 1.1 are considered acceptable and will update best_loss
        to the latest value. This allows small increases before considering it
        as "no improvement".
    """

    def __init__(
        self,
        mode: str = "min",
        patience: int = 5,
        threshold: float = 0.0,
        threshold_mode: str = "rel",
        verbose: bool = False,
        delta: float = 0.0,
    ) -> None:
        """Initialize EarlyStopping with specified parameters.

        Args:
            mode: 'min' for loss minimization, 'max' for metric maximization
            patience: Number of epochs to wait after last improvement
            threshold: Threshold for measuring new optimum
            threshold_mode: 'rel' or 'abs' mode for threshold calculation
            verbose: If True, print messages for each improvement
            delta: Minimum change to qualify as improvement
        """
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss: float | None = None
        self.early_stop = False
        self.delta = delta
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        # Validate mode parameter
        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")

        # Validate threshold_mode parameter
        if self.threshold_mode not in ["rel", "abs"]:
            raise ValueError(
                f"threshold_mode must be 'rel' or 'abs', got {self.threshold_mode}"
            )

    def __call__(self, val_loss: float) -> None:
        """Check if validation loss improved and update early stopping state.

        Args:
            val_loss (float): Validation loss from current epoch
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        # Threshold from mode (min_delta, rel, etc.)
        if self.threshold_mode == "rel":
            threshold_value = self.best_loss * self.threshold
        else:  # 'abs'
            threshold_value = self.threshold

        # Mode-aware comparison
        if self.mode == "min":
            # For minimization: check if val_loss > best_loss + threshold -
            # delta
            is_worse = val_loss > self.best_loss + threshold_value - self.delta
        else:  # 'max'
            # For maximization: check if val_loss < best_loss - threshold +
            # delta
            is_worse = val_loss < self.best_loss - threshold_value + self.delta

        if is_worse:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "Early stopping counter: %d out of %d",
                    self.counter,
                    self.patience,
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the early stopping object as a dictionary."""
        return {
            "mode": self.mode,
            "patience": self.patience,
            "verbose": self.verbose,
            "counter": self.counter,
            "best_loss": self.best_loss,
            "early_stop": self.early_stop,
            "delta": self.delta,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the early stopping state from a dictionary.

        Args:
            state_dict (dict): Dictionary containing early stopping state
        """
        # Backward compatibility default
        self.mode = state_dict.get("mode", "min")
        self.patience = state_dict["patience"]
        self.verbose = state_dict["verbose"]
        self.counter = state_dict["counter"]
        self.best_loss = state_dict["best_loss"]
        self.early_stop = state_dict["early_stop"]
        self.delta = state_dict["delta"]
        self.threshold = state_dict["threshold"]
        self.threshold_mode = state_dict["threshold_mode"]


@dataclass
class EarlyStoppingConfig:
    """Config for EarlyStopping: halt training when val loss stops improving.

    Registered with Hydra ConfigStore so
    `early_stopping/base_early_stopping` can be composed in YAML defaults.
    """

    _target_: str = "src.utils.train_utils.EarlyStopping"

    # Early stopping parameters
    mode: str = "min"  # 'min' for loss minimization, 'max' for metric
    # maximization
    patience: int = 5
    threshold: float = 0.0
    threshold_mode: str = "rel"
    verbose: bool = False
    delta: float = 0.0


# Register with Hydra ConfigStore
# Register Hydra config at module import time so that entry points can
# simply import this module (e.g., `import src.utils.train_utils`) to
# ensure the config group is available before Hydra composes configs.
cs = ConfigStore.instance()
cs.store(
    group="early_stopping",
    name="base_early_stopping",
    node=EarlyStoppingConfig,
)
