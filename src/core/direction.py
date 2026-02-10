"""Source→target direction definitions and multi-direction config.

Defines Direction configs and a Directions container for allowed
source→target mappings. register_directions() stores DirectionsConfig
under the Hydra group "directions" with name "base_directions".
"""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Local imports
from src.core.domain import Direction
from src.core.domain import DirectionConfig


@dataclass
class DirectionsConfig:
    """Hydra config node for a list of allowed source→target directions.

    Attributes:
        _target_: Hydra instantiation target (Directions class).
        active_directions: List of DirectionConfig objects; required, no default.

    Example:
        In YAML defaults: ``directions: base_directions`` with
        ``active_directions: [ppg2abp, ecg2abp]`` (or equivalent config list).
    """

    _target_: str = "src.core.direction.Directions"
    active_directions: list[DirectionConfig] = MISSING


class Directions:
    """Dataset-agnostic container of allowed source→target directions."""

    def __init__(self, active_directions: list[Direction]):
        """Initialize Directions container with active source→target directions.

        Args:
            active_directions: List of Direction objects defining allowed
                transformations.

        Raises:
            ValueError: If active_directions is empty.
        """
        if not active_directions:
            raise ValueError(
                "active_directions must be non-empty; provide at least one "
                "direction in config."
            )
        _dirs = list(active_directions)
        self._dirs: tuple[Direction, ...] = tuple(_dirs)
        self._keys: tuple[str, ...] = tuple(d.key() for d in self._dirs)

    @staticmethod
    def from_strings(keys: list[str]) -> "Directions":
        """Build Directions from key strings.

        Args:
            keys: List of direction key strings (e.g. ["PPG2ABP", "ECG2ABP"]).

        Returns:
            Directions: Container of parsed Direction objects.
        """
        return Directions([Direction.parse(k) for k in keys])

    def __len__(self) -> int:
        """Return the number of directions."""
        return len(self._dirs)

    def __iter__(self):
        """Iterate over directions."""
        return iter(self._dirs)

    def __getitem__(self, idx: int) -> Direction:
        """Get direction by index.

        Args:
            idx: Index of the direction to retrieve.

        Returns:
            Direction object at the given index.
        """
        return self._dirs[idx]

    def __contains__(self, item: Direction) -> bool:
        """Check if direction is in the container.

        Args:
            item: Direction object to check for.

        Returns:
            True if direction is in the container, False otherwise.
        """
        return item in self._dirs

    def keys(self) -> tuple[str, ...]:
        """Return tuple of direction key strings.

        Returns:
            Tuple of direction keys (e.g., ("PPG2ABP", "ECG2ABP")).
        """
        return self._keys

    @property
    def directions(self) -> tuple[Direction, ...]:
        """Tuple of Direction objects for backward compatibility."""
        return self._dirs


def register_directions():
    """Register DirectionsConfig as base_directions in group 'directions'.

    Side effect only: registers with Hydra ConfigStore. Call before
    Hydra composes configs that reference the directions group.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_directions", node=DirectionsConfig, group="directions")
