"""
This module contains the Direction class, which is used to represent the direction of a signal.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from src.core.domain import Direction, DirectionConfig

@dataclass
class DirectionsConfig:
    """Configuration for Directions."""
    _target_: str = "src.core.direction.Directions"  # Fixed: single underscore
    active_directions: List[DirectionConfig] = MISSING
    # active_directions: List[DirectionConfig] = field(default_factory=lambda: [])
    # active_directions: List[DirectionConfig] = field(default_factory=list)

class Directions:
    """Dataset-agnostic list of allowed directions."""
    def __init__(self, active_directions: List[Direction]):
        _dirs = []
        for d in active_directions:
            # optional simple guard: disallow same-to-same
            # if d.src == d.tgt:
            #     raise ValueError(f"Self-direction not allowed: {d.key()}")
            _dirs.append(d)
        self._dirs: Tuple[Direction, ...] = tuple(_dirs)
        self._keys: Tuple[str, ...] = tuple(d.key() for d in self._dirs)

    @staticmethod
    def from_strings(keys: List[str]) -> "Directions":
        return Directions(Direction.parse(k) for k in keys)

    def __len__(self): return len(self._dirs)
    def __iter__(self): return iter(self._dirs)
    def __getitem__(self, idx): return self._dirs[idx]
    def __contains__(self, item): return item in self._dirs
    def keys(self) -> Tuple[str, ...]: return self._keys
    
    @property
    def directions(self) -> Tuple[Direction, ...]:
        """Property to access the directions (for backward compatibility)."""
        return self._dirs

def register_directions():
    """
    Register the Direction class to the ConfigStore.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_directions", node=DirectionsConfig, group="directions")


