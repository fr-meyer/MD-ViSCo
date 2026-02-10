"""Meter classes for tracking and aggregating metrics during training and evaluation.

See Also:
    src.loggings.metrics: MetricsManager and aggregate context for training/eval.
"""

from __future__ import annotations

# Standard library imports
import logging
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict

# Third-party imports
import torch

logger = logging.getLogger(__name__)


class Meter(ABC):
    """Base abstract class for all meters."""

    @abstractmethod
    def update(self, val, n=1):
        """Update meter with new value.

        Args:
            val: Value to record.
            n: Sample size for the update. Defaults to 1.
        """

    @abstractmethod
    def reset(self):
        """Reset meter to initial state."""

    @abstractmethod
    def val(self) -> float | int | None:
        """Return current meter value.

        Returns:
            Current value (average, sum, or meter-specific value).
        """


class AverageMeter(Meter):
    """Computes and stores the average and current value."""

    sum: float
    count: int

    def __init__(self):
        """Initialize average meter."""
        self.reset()

    def update(self, val, n=1):
        """See base class."""
        self.current = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        """See base class."""
        self.current = 0
        self.avg = 0
        self.sum = 0.0
        self.count = 0

    def val(self) -> float:
        """See base class."""
        return self.avg


class SumMeter(Meter):
    """Tracks cumulative sum."""

    def __init__(self):
        """Initialize sum meter."""
        self.reset()

    def update(self, val, n=1):
        """See base class."""
        self.sum += val * n

    def reset(self):
        """See base class."""
        self.sum = 0

    def val(self) -> float | int:
        """See base class."""
        return self.sum


class MetersDict(OrderedDict):
    """Priority-ordered dictionary that manages multiple meters."""

    def __init__(self, *args, **kwargs):
        """Initialize meters dictionary."""
        super().__init__(*args, **kwargs)
        self.priority = []

    def add_meter(self, name, meter, priority=None):
        """Add meter with optional priority.

        Args:
            name: Meter name (key in the dictionary).
            meter: Meter instance to add.
            priority: Optional priority for ordering. Defaults to None.
        """
        self[name] = meter
        if priority is not None:
            self.priority.append((priority, name))
            self.priority.sort()

    def get_smoothed_values(self):
        """Get smoothed values from all meters.

        Returns:
            dict: Mapping of meter name to smoothed value (avg or val()).
        """
        smoothed = {}
        for name, meter in self.items():
            if hasattr(meter, "avg"):
                smoothed[name] = meter.avg
            else:
                smoothed[name] = meter.val()
        return smoothed

    def reduce(self, device=None, op="mean", fallback_to_cpu=False, world_size=None):
        """Perform robust DDP reduction with comprehensive error handling.

        Args:
            device: Device for tensor operations (default: current CUDA device or CPU)
            op: Reduction operation ('mean', 'sum', 'min', 'max')
            fallback_to_cpu: Whether to fallback to CPU if GPU reduction fails
            world_size: World size for DDP reduction (if None, from dist)

        Returns:
            self: For method chaining
        """
        try:
            import torch.distributed as dist
        except ImportError:
            return self

        if not dist.is_initialized():
            return self

        if device is None:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        # Use provided world_size or fallback to distributed module
        if world_size is None:
            try:
                world_size = dist.get_world_size()
            except RuntimeError:
                world_size = 1

        try:
            # Map op string to ReduceOp
            op_map = {
                "sum": dist.ReduceOp.SUM,
                "mean": dist.ReduceOp.SUM,  # Will divide by world_size after reduction
                "min": dist.ReduceOp.MIN,
                "max": dist.ReduceOp.MAX,
            }
            reduce_op = op_map.get(op.lower(), dist.ReduceOp.SUM)

            for _meter_name, meter in self.items():
                if hasattr(meter, "avg"):
                    # CRITICAL FIX: Reduce sum and count, not the already-averaged value
                    # Correct global average regardless of uneven sample distribution
                    sum_t = torch.tensor(
                        float(meter.sum), device=device, dtype=torch.float64
                    )
                    cnt_t = torch.tensor(
                        float(meter.count), device=device, dtype=torch.float64
                    )

                    # Try GPU reduction first
                    try:
                        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
                        dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)

                        # Write back consistent state
                        meter.sum = sum_t.item()
                        meter.count = max(1.0, cnt_t.item())  # avoid div-by-zero
                        meter.avg = meter.sum / meter.count

                    except RuntimeError as e:
                        if fallback_to_cpu and device != "cpu":
                            # Fallback to CPU reduction
                            sum_t = sum_t.cpu()
                            cnt_t = cnt_t.cpu()
                            dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
                            dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)

                            # Write back consistent state
                            meter.sum = sum_t.item()
                            meter.count = max(1.0, cnt_t.item())  # avoid div-by-zero
                            meter.avg = meter.sum / meter.count
                        else:
                            raise e
                elif hasattr(meter, "sum"):
                    # Handle meters without avg attribute (e.g., SumMeter)
                    # Reduce sum across ranks and apply the requested operation
                    sum_t = torch.tensor(
                        float(meter.sum), device=device, dtype=torch.float64
                    )

                    # Try GPU reduction first
                    try:
                        dist.all_reduce(sum_t, op=reduce_op)

                        # Apply mean operation if requested (divide by world_size)
                        if op.lower() == "mean":
                            sum_t = sum_t / world_size

                        # Write back consistent state
                        meter.sum = sum_t.item()

                    except RuntimeError as e:
                        if fallback_to_cpu and device != "cpu":
                            # Fallback to CPU reduction
                            sum_t = sum_t.cpu()
                            dist.all_reduce(sum_t, op=reduce_op)

                            # Apply mean operation if requested (divide by world_size)
                            if op.lower() == "mean":
                                sum_t = sum_t / world_size

                            # Write back consistent state
                            meter.sum = sum_t.item()
                        else:
                            raise e
        except (RuntimeError, AttributeError) as e:
            logger.warning("DDP reduction failed, using local values: %s", e)

        return self
