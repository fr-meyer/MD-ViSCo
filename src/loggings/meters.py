import time
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from collections import OrderedDict

class Meter(ABC):
    """Base abstract class for all meters"""
    
    @abstractmethod
    def update(self, val, n=1):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def val(self):
        pass

class AverageMeter(Meter):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def val(self):
        return self.avg

class SumMeter(Meter):
    """Tracks cumulative sum"""
    
    def __init__(self):
        self.reset()
    
    def update(self, val, n=1):
        self.sum += val * n
    
    def reset(self):
        self.sum = 0
    
    def val(self):
        return self.sum

class MetersDict(OrderedDict):
    """Priority-ordered dictionary that manages multiple meters"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority = []
    
    def add_meter(self, name, meter, priority=None):
        self[name] = meter
        if priority is not None:
            self.priority.append((priority, name))
            self.priority.sort()
    
    def get_smoothed_values(self):
        """Get smoothed values from all meters"""
        smoothed = {}
        for name, meter in self.items():
            if hasattr(meter, 'avg'):
                smoothed[name] = meter.avg
            else:
                smoothed[name] = meter.val()
        return smoothed
    
    def reduce(self, device=None, op='mean', fallback_to_cpu=False, world_size=None):
        """Robust DDP reduction with comprehensive error handling
        
        Args:
            device: Device for tensor operations (default: current CUDA device or CPU)
            op: Reduction operation ('mean', 'sum', 'min', 'max')
            fallback_to_cpu: Whether to fallback to CPU if GPU reduction fails
            world_size: World size for DDP reduction (if None, will try to get from dist)
            
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
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        
        # Use provided world_size or fallback to distributed module
        if world_size is None:
            try:
                world_size = dist.get_world_size()
            except:
                world_size = 1
        
        try:
            for meter_name, meter in self.items():
                if hasattr(meter, 'avg'):
                    # CRITICAL FIX: Reduce sum and count, not the already-averaged value
                    # This ensures correct global average regardless of uneven sample distribution across ranks
                    sum_t = torch.tensor(float(meter.sum), device=device, dtype=torch.float64)
                    cnt_t = torch.tensor(float(meter.count), device=device, dtype=torch.float64)
                    
                    # Try GPU reduction first
                    try:
                        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
                        dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
                        
                        # Write back consistent state
                        meter.sum = sum_t.item()
                        meter.count = max(1.0, cnt_t.item())  # avoid div-by-zero
                        meter.avg = meter.sum / meter.count
                        
                    except RuntimeError as e:
                        if fallback_to_cpu and device != 'cpu':
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
        except Exception as e:
            print(f"Warning: DDP reduction failed, using local values: {e}")
        
        return self 