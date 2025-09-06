import torch
import torch.distributed as dist
from typing import Dict, Any, Optional
from contextlib import contextmanager
from .meters import MetersDict, AverageMeter, SumMeter

class MetricsManager:
    """Global metrics management with DDP support"""
    
    def __init__(self):
        self._aggregators = {}
        self._current_aggregator = None
    
    def _get_aggregator(self, name):
        """Get or create aggregator"""
        if name not in self._aggregators:
            self._aggregators[name] = MetersDict()
        return self._aggregators[name]
    
    def log_scalar(self, name, value, sample_size=1, priority=None):
        """Log scalar metric with automatic aggregation"""
        if self._current_aggregator is None:
            return
        
        if name not in self._current_aggregator:
            self._current_aggregator.add_meter(name, AverageMeter(), priority)
        
        self._current_aggregator[name].update(value, sample_size)
    
    def log_scalar_sum(self, name, value, sample_size=1, priority=None):
        """Log cumulative sum metric"""
        if self._current_aggregator is None:
            return
        
        if name not in self._current_aggregator:
            self._current_aggregator.add_meter(name, SumMeter(), priority)
        
        self._current_aggregator[name].update(value, sample_size)
    
    @contextmanager
    def aggregate(self, name=None, new_root=False):
        """Context manager for metric aggregation"""
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
        """Get smoothed values from aggregator"""
        aggregator = self._get_aggregator(name)
        return aggregator.get_smoothed_values()
    
    def reset_meters(self, name="default"):
        """Reset all meters in aggregator"""
        aggregator = self._get_aggregator(name)
        for meter in aggregator.values():
            meter.reset()
    
    def sync_distributed(self, device=None, op='mean', fallback_to_cpu=False, world_size=None):
        """Enhanced distributed synchronization with fallback strategies
        
        Args:
            device: Device for tensor operations (default: current CUDA device or CPU)
            op: Reduction operation ('mean', 'sum', 'min', 'max')
            fallback_to_cpu: Whether to fallback to CPU if GPU reduction fails
            world_size: World size for DDP reduction (if None, will try to get from dist)
        """
        if not dist.is_initialized():
            return
        
        for name, aggregator in self._aggregators.items():
            aggregator.reduce(device=device, op=op, fallback_to_cpu=fallback_to_cpu, world_size=world_size)

# Global metrics instance
metrics = MetricsManager() 