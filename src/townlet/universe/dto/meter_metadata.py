"""Meter metadata DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MeterInfo:
    """Metadata about a single meter."""

    name: str
    index: int
    critical: bool
    initial_value: float
    observable: bool
    description: str = ""


@dataclass(frozen=True)
class MeterMetadata:
    """Collection of all meter metadata."""

    meters: tuple[MeterInfo, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "meters", tuple(self.meters))

    def get_meter_by_name(self, name: str) -> MeterInfo:
        """Lookup meter by name."""
        for meter in self.meters:
            if meter.name == name:
                return meter
        raise KeyError(f"Meter '{name}' not found")
