"""Domain classes for the project."""
from enum import Enum
from dataclasses import dataclass
from omegaconf import MISSING

class Vital(str, Enum):
    """Vital signs."""
    ECG = "ECG"
    PPG = "PPG"
    ABP = "ABP"

@dataclass(frozen=True)
class DirectionConfig:
    """Configuration for Direction."""
    _target_: str = "src.core.domain.Direction"
    src: Vital = MISSING
    tgt: Vital = MISSING

class Direction:
    """Direction between two vital signs."""
    def __init__(self, src: Vital, tgt: Vital):
        self.src = src
        self.tgt = tgt
    
    def key(self) -> str: return f"{self.src}2{self.tgt}"
    
    @staticmethod
    def parse(s: str) -> "Direction":
        s = s.upper()
        if "2" not in s: raise ValueError(f"Invalid direction format: {s}")
        a,b = s.split("2",1)
        return Direction(Vital[a], Vital[b])

def register_direction():
    """Register the Direction class to the ConfigStore."""
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="base_direction", node=DirectionConfig, group="direction")