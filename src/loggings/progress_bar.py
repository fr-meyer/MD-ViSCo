"""Progress bar with integrated WandB and CSV logging for training and evaluation.

See Also:
    src.loggings.csv_wrapper: CSVWrapper for run-level logging.
    src.loggings.wandb_wrapper: WandBWrapper for experiment tracking.
"""

# Standard library imports
import logging
from dataclasses import dataclass

# Third-party imports
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

# Local imports
from .csv_wrapper import CSVWrapper
from .csv_wrapper import CSVWrapperConfig
from .wandb_wrapper import WandBWrapper
from .wandb_wrapper import WandBWrapperConfig

logger = logging.getLogger(__name__)


@dataclass
class ProgressBarConfig:
    """Configuration for ProgressBar with Hydra-compatible defaults."""

    _target_: str = "src.loggings.progress_bar.ProgressBar"
    log_interval: int = 1
    wandb_log_interval: int = 1
    csv_log_interval: int = 1
    wandb_wrapper: WandBWrapperConfig | None = None
    csv_wrapper: CSVWrapperConfig | None = None


class ProgressBar:
    """Enhanced progress bar with integrated WandB and CSV logging.

    This class provides:
    - Real-time progress display
    - Integrated WandB logging
    - Integrated CSV logging
    - Run-level state management

    Step Management:
    - Training: Logs every batch with step=self._step (monotonically increasing)
    - Validation: Handled at epoch level by trainer (no step collision)

    Features:
    - Real-time metrics display in progress bar postfix
    - Proper interval control for different logging systems
    - Clean separation of concerns between step management and logging
    - Complete CSV logging for both training and validation
    - Step collision prevention for validation metrics

    Instantiation:
        Objects are instantiated via Hydra's automatic instantiation system
        using the _target_ field in configuration files.
    """

    def __init__(
        self,
        log_interval: int = 1,
        wandb_log_interval: int = 1,
        csv_log_interval: int = 1,
        wandb_wrapper: WandBWrapper | None = None,
        csv_wrapper: CSVWrapper | None = None,
    ):
        """Initialize progress bar without rank determination.

        Args:
            log_interval: Interval for progress bar updates
            wandb_log_interval: Interval for WandB logging
            csv_log_interval: Interval for CSV logging
            wandb_wrapper: Optional pre-configured WandBWrapper instance
            csv_wrapper: Optional pre-configured CSVWrapper instance

        Raises:
            ValueError: If parameters are invalid
        """
        self.log_interval = log_interval
        self.wandb_log_interval = wandb_log_interval
        self.csv_log_interval = csv_log_interval
        self._current_bar = None
        self._step = 0  # Simple step counter

        self._validate_config()

        # Use provided wrappers
        self.wandb = wandb_wrapper
        self.csv = csv_wrapper

        logger.debug(f"Initialized ProgressBar: {self}")

    def __repr__(self) -> str:
        """Return string representation for logging and debugging."""
        return (
            f"<ProgressBar(log_interval={self.log_interval}, "
            f"wandb_log_interval={self.wandb_log_interval}, "
            f"csv_log_interval={self.csv_log_interval})>"
        )

    def _validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.wandb_log_interval <= 0:
            raise ValueError("wandb_log_interval must be positive")
        if self.csv_log_interval <= 0:
            raise ValueError("csv_log_interval must be positive")

    def create_bar(self, total, description="", disable=False, **kwargs):
        """Create tqdm progress bar."""
        self._current_bar = tqdm(
            total=total, desc=description, disable=disable, ncols=125, **kwargs
        )

    def update(
        self,
        n=1,
        metrics_dict=None,
        step=None,
        is_rank0: bool = False,
        to_log: bool = False,
    ):
        """Update progress bar with simple step management."""
        if self._current_bar is None:
            return

        if step is not None:
            self._step = step
        else:
            self._step += n

        self._current_bar.update(n)

        if metrics_dict and self._step % self.log_interval == 0:
            self._update_metrics(metrics_dict, is_rank0, to_log)

    def _update_metrics(
        self, metrics_dict, is_rank0: bool = False, to_log: bool = False
    ):
        """Update metrics display, WandB, and CSV logging with interval control."""
        if self._current_bar is not None and hasattr(self._current_bar, "set_postfix"):
            postfix = {}
            for key, value in metrics_dict.items():
                if isinstance(value, float):
                    postfix[key] = f"{value:.4f}"
                else:
                    postfix[key] = str(value)
            self._current_bar.set_postfix(**postfix)

        # WandB logging (only on rank 0, at interval)
        if (
            self._step % self.wandb_log_interval == 0
            and self.wandb is not None
            and to_log
        ):
            self.wandb.log(metrics_dict, step=None, is_rank0=is_rank0)

        # CSV logging (only on rank 0, at interval)
        if self._step % self.csv_log_interval == 0 and self.csv is not None and to_log:
            self.csv.log_metrics(metrics_dict, step=self._step, is_rank0=is_rank0)

    def close(self):
        """Close progress bar, WandB, and CSV with run management."""
        if self._current_bar and hasattr(self._current_bar, "close"):
            self._current_bar.close()
        if self.wandb is not None:
            self.wandb.finish()
        if self.csv is not None:
            # Finish all active runs before closing
            for file_key in list(self.csv._run_started.keys()):
                self.csv.finish_run(file_key)
            self.csv.finish()

    def close_wandb(self):
        """Close WandB wrapper."""
        if self.wandb is not None:
            self.wandb.finish()

    def close_progress_bar(self):
        """Close progress bar."""
        if self._current_bar and hasattr(self._current_bar, "close"):
            self._current_bar.close()

    def close_csv(self):
        """Close CSV wrapper."""
        if self.csv is not None:
            self.csv.finish()


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_progress_bar", node=ProgressBarConfig, group="progress_bar")
