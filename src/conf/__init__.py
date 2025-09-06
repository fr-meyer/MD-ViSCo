"""
Configuration package for MD-ViSCo project.

This package contains all configuration classes and utilities for the MD-ViSCo project,
including the main Config class and dataset-specific configurations.
"""

from hydra.core.config_store import ConfigStore
from .config import Config

# Register config at module import time (consistent with dataset configs)
if __name__ != "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)  # Generic name, not matching YAML file

__all__ = ['Config']
