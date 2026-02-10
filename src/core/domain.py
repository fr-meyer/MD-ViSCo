"""Domain types for vitals and source→target directions.

Vital enum and Direction/DirectionConfig describe which signals are used
as sources and targets. register_direction() stores DirectionConfig in
Hydra's ConfigStore under group "direction" with name "base_direction".
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum

# Third-party imports
from omegaconf import MISSING


class Vital(str, Enum):
    """Vital signs (waveforms and scalar values).

    Members: ECG (electrocardiogram), PPG (photoplethysmogram), ABP (arterial
    blood pressure waveform), IMP (respiratory impedance), BP (blood pressure
    scalar: SBP, DBP, MAP).
    """

    ECG = "ECG"
    PPG = "PPG"
    ABP = "ABP"
    IMP = "IMP"
    BP = "BP"


@dataclass(frozen=True)
class DirectionConfig:
    """Hydra config for a single source→target direction.

    Hydra will automatically convert string values from YAML to Vital enums.

    YAML Examples:
        # Single source
        direction:
          source: [PPG]  # Hydra converts to [Vital.PPG]
          target: ABP  # Hydra converts to Vital.ABP

        # Multi-source
        direction:
          source: [PPG, ECG]  # Hydra converts to [Vital.PPG, Vital.ECG]
          target: ABP

    Python Examples:
        # Single source
        DirectionConfig(
            _target_="src.core.domain.Direction",
            source=[Vital.PPG],
            target=Vital.ABP,
        )

        # Multi-source
        DirectionConfig(
            _target_="src.core.domain.Direction",
            source=[Vital.PPG, Vital.ECG],
            target=Vital.ABP,
        )
    """

    _target_: str = "src.core.domain.Direction"
    _convert_: str = "all"
    source: list[Vital] = MISSING
    target: Vital = MISSING


class Direction:
    """Direction between vital signs (supports single or multiple sources).

    The `source` property always returns a list of Vital enums for consistency.
    For single-source directions, use `source[0]` to get the single vital.

    Examples:
        >>> # Single source
        >>> direction = Direction(source=[Vital.PPG], target=Vital.ABP)
        >>> direction.source  # [Vital.PPG] - always a list
        >>> print(direction.key())  # "PPG2ABP"

        >>> # Multiple sources
        >>> direction = Direction(source=[Vital.PPG, Vital.ECG], target=Vital.ABP)
        >>> direction.source  # [Vital.PPG, Vital.ECG]
        >>> print(direction.key())  # "PPG+ECG2ABP"
    """

    def __init__(self, source: list[Vital], target: Vital):
        """Initialize Direction with source and target vitals.

        Args:
            source: List of source Vital enums (must be non-empty).
            target: Target Vital enum.

        Raises:
            TypeError: If source is not a list or contains non-Vital items.
            TypeError: If target is not a Vital enum.
            ValueError: If source list is empty.
        """
        if not isinstance(source, list):
            raise TypeError(f"source must be List[Vital], got {type(source)}")
        self._source = list(source)
        if not self._source:
            raise ValueError("source cannot be empty")

        for i, item in enumerate(self._source):
            if not isinstance(item, Vital):
                raise TypeError(
                    f"All items in source must be of type Vital."
                    f" Item at index {i} is {type(item).__name__}: {item!r}"
                )

        if not isinstance(target, Vital):
            raise TypeError(f"target must be Vital, got {type(target)}")
        self._target = target

    @property
    def source(self) -> list[Vital]:
        """Return source vitals as a list (always).

        This is the canonical property for accessing source vitals. For
        single-source directions, returns a list with one element. Use
        `source[0]` if you need the single vital.
        """
        return self._source

    @property
    def target(self) -> Vital:
        """Return target vital."""
        return self._target

    def key(self) -> str:
        """Generate direction key string.

        Returns:
            String like "PPG2ABP" for single source or "PPG+ECG2ABP" for multi-source
        """
        if len(self._source) == 1:
            return f"{self._source[0].value}2{self._target.value}"
        else:
            source_str = "+".join(v.value for v in self._source)
            return f"{source_str}2{self._target.value}"

    @staticmethod
    def parse(s: str) -> "Direction":
        """Parse direction string to Direction object.

        Args:
            s: Direction string like "PPG2ABP" or "PPG+ECG2ABP"

        Returns:
            Direction object

        Raises:
            ValueError: For malformed strings (e.g. missing "2") or unknown vitals.

        Examples:
            >>> Direction.parse("PPG2ABP")
            Direction(source=[Vital.PPG], target=Vital.ABP)

            >>> Direction.parse("PPG+ECG2ABP")
            Direction(source=[Vital.PPG, Vital.ECG], target=Vital.ABP)
        """
        s = s.upper()
        if "2" not in s:
            raise ValueError(f"Invalid direction format: {s}")

        source_part, target_part = s.split("2", 1)
        allowed = [v.name for v in Vital]

        # Convert vital name string to Vital enum or raise ValueError
        def vital_or_raise(name: str) -> Vital:
            try:
                return Vital[name]
            except KeyError:
                raise ValueError(
                    f"Unknown vital '{name}'. Allowed vitals: {allowed}"
                ) from None

        target = vital_or_raise(target_part.strip())

        if "+" in source_part:
            source_strs = source_part.split("+")
            source = [vital_or_raise(part.strip()) for part in source_strs]
        else:
            source = [vital_or_raise(source_part.strip())]

        return Direction(source=source, target=target)


def register_direction():
    """Store DirectionConfig in ConfigStore (group "direction", name "base_direction").

    Note:
        Side effect only: registers with Hydra ConfigStore. Call before
        Hydra composes configs that reference the direction group.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_direction", node=DirectionConfig, group="direction")
