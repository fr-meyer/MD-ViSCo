import os
import csv
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Set
from dataclasses import dataclass, field
from omegaconf import DictConfig


logger = logging.getLogger(__name__)

@dataclass
class CSVWrapperConfig:
    """Configuration for CSVWrapper with Hydra-compatible defaults"""
    _target_: str = "src.loggings.csv_wrapper.CSVWrapper"
    output_dir: str = "./results"
    flush_every_n: int = 100
    # Remove: rank: int = 0

class CSVWrapper:
    """Pure utility CSV wrapper - completely agnostic of data semantics"""
    
    def __init__(self, output_dir: str = "./results", flush_every_n: int = 100):
        """Initialize without rank - rank will be determined at runtime
        
        Args:
            output_dir: Directory to save CSV files
            flush_every_n: Number of entries to buffer before flushing to disk
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.output_dir = output_dir
        self.flush_every_n = flush_every_n
        self._active_files = {}
        self._buffers = {}
        self._fieldnames = {}
        
        # Validate configuration
        self._validate_config()
        logger.debug(f"Initialized CSVWrapper: {self}")
    
    def __repr__(self) -> str:
        """String representation for logging and debugging"""
        return (f"<CSVWrapper(output_dir='{self.output_dir}', "
                f"flush_every_n={self.flush_every_n})>")
    
    def _validate_config(self) -> None:
        """Validate configuration settings
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.flush_every_n <= 0:
            raise ValueError("flush_every_n must be positive")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int = None, 
                   file_key: str = "default", is_rank0: bool = False):
        """Log metrics to CSV with buffering for real-time updates"""
        if not is_rank0:  # Only master process logs
            return
            
        if file_key not in self._buffers:
            self._buffers[file_key] = []
            self._fieldnames[file_key] = set()
            
        # Add timestamp and step
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step or 0,
            **metrics
        }
        
        # Robust fieldname detection
        self._fieldnames[file_key].update(log_entry.keys())
        
        self._buffers[file_key].append(log_entry)
        
        # Flush if buffer is full
        if len(self._buffers[file_key]) >= self.flush_every_n:
            self._flush_buffer(file_key)
    
    def save_dataframe(self, df: pd.DataFrame, filepath: str, is_rank0: bool = False) -> None:
        """Save any DataFrame to CSV - evaluators provide the DataFrame"""
        if not is_rank0:
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save DataFrame
        df.to_csv(filepath, index=False)
        logger.info(f"Saved CSV to {filepath}")
    
    def _flush_buffer(self, file_key: str):
        """Flush buffered data to CSV file"""
        if file_key not in self._buffers or not self._buffers[file_key]:
            return
            
        # Get or create file with timestamped filename
        if file_key not in self._active_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{file_key}_metrics_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            self._active_files[file_key] = open(filepath, 'a', newline='')
            
        # Use robust fieldname detection
        fieldnames = sorted(list(self._fieldnames[file_key]))
        writer = csv.DictWriter(self._active_files[file_key], fieldnames=fieldnames)
        
        if self._active_files[file_key].tell() == 0:  # New file
            writer.writeheader()
            
        writer.writerows(self._buffers[file_key])
        self._active_files[file_key].flush()
        
        # Clear buffer
        self._buffers[file_key] = []
    
    def finish(self):
        """Flush all buffers and close files"""
        for file_key in list(self._buffers.keys()):
            self._flush_buffer(file_key)
            
        for file_handle in self._active_files.values():
            file_handle.close()
        
        self._active_files.clear()
    
    @classmethod
    def from_config(cls, config: DictConfig) -> 'CSVWrapper':
        """Create CSVWrapper from main config with error handling
        
        Args:
            config: Main configuration object (DictConfig for better Hydra integration)
            
        Returns:
            CSVWrapper: Initialized wrapper
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Extract CSV wrapper config parameters
            csv_config = config.csv_wrapper
            return cls(
                output_dir=csv_config.output_dir,
                flush_every_n=csv_config.flush_every_n
            )
        except AttributeError:
            raise ValueError("Config must have 'csv_wrapper' attribute with appropriate CSV settings") 