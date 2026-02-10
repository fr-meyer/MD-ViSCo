"""Global metrics management with distributed data parallel (DDP) support.

Provides a single global metrics instance for logging scalars and sums during
training and evaluation, with optional DDP reduction. Typical usage: call
``with metrics.aggregate(): ... log_scalar(...)`` then read or sync values.

See Also:
    src.loggings.meters: AverageMeter, SumMeter, MetersDict.
    src.trainers.trainer: Uses metrics for training/validation logging.
"""

# Standard library imports
from contextlib import contextmanager

# Third-party imports
import torch.distributed as dist

# Local imports
from .meters import AverageMeter
from .meters import MetersDict
from .meters import SumMeter


class MetricsManager:
    """Global metrics management with DDP support."""

    def __init__(self):
        """Initialize metrics manager."""
        self._aggregators = {}
        self._current_aggregator = None

    def _get_aggregator(self, name):
        """Get or create aggregator by name.

        Args:
            name: Aggregator name (e.g. "default").

        Returns:
            MetersDict: The aggregator for the given name.
        """
        if name not in self._aggregators:
            self._aggregators[name] = MetersDict()
        return self._aggregators[name]

    def log_scalar(self, name, value, sample_size=1, priority=None):
        """Log scalar metric with automatic aggregation.

        Args:
            name: Metric name.
            value: Scalar value to log.
            sample_size: Sample size for averaging. Defaults to 1.
            priority: Optional priority for meter ordering.

        Note:
            No-op if no aggregator is active (e.g. outside aggregate() context).
        """
        if self._current_aggregator is None:
            return

        if name not in self._current_aggregator:
            self._current_aggregator.add_meter(name, AverageMeter(), priority)

        self._current_aggregator[name].update(value, sample_size)

    def log_scalar_sum(self, name, value, sample_size=1, priority=None):
        """Log cumulative sum metric.

        Args:
            name: Metric name.
            value: Value to add to the sum.
            sample_size: Sample size for the update. Defaults to 1.
            priority: Optional priority for meter ordering.

        Note:
            No-op if no aggregator is active.
        """
        if self._current_aggregator is None:
            return

        if name not in self._current_aggregator:
            self._current_aggregator.add_meter(name, SumMeter(), priority)

        self._current_aggregator[name].update(value, sample_size)

    @contextmanager
    def aggregate(self, name=None, new_root=False):
        """Provide context manager for metric aggregation.

        Args:
            name: Aggregator name. Defaults to "default" if None.
            new_root: If True, create a new root aggregator for this name.

        Yields:
            MetersDict: The active aggregator for the context.
        """
        if name is None:
            name = "default"

        if new_root:
            aggregator = MetersDict()
            self._aggregators[name] = aggregator
        else:
            aggregator = self._get_aggregator(name)

        prev_aggregator = self._current_aggregator
        self._current_aggregator = aggregator

        try:
            yield aggregator
        finally:
            self._current_aggregator = prev_aggregator

    def get_smoothed_values(self, name="default"):
        """Get smoothed values from aggregator.

        Args:
            name: Aggregator name. Defaults to "default".

        Returns:
            dict: Smoothed metric values from the aggregator.
        """
        aggregator = self._get_aggregator(name)
        return aggregator.get_smoothed_values()

    def reset_meters(self, name="default"):
        """Reset all meters in aggregator.

        Args:
            name: Aggregator name. Defaults to "default".
        """
        aggregator = self._get_aggregator(name)
        for meter in aggregator.values():
            meter.reset()

    def sync_distributed(
        self, device=None, op="mean", fallback_to_cpu=False, world_size=None
    ):
        """Synchronize metrics across distributed processes with fallback strategies.

        Args:
            device: Device for tensor operations (default: current CUDA device or CPU)
            op: Reduction operation ('mean', 'sum', 'min', 'max')
            fallback_to_cpu: Whether to fallback to CPU if GPU reduction fails
            world_size: World size for DDP reduction (if None, from dist)
        """
        if not dist.is_initialized():
            return

        for _name, aggregator in self._aggregators.items():
            aggregator.reduce(
                device=device,
                op=op,
                fallback_to_cpu=fallback_to_cpu,
                world_size=world_size,
            )


# Global metrics instance
metrics = MetricsManager()
