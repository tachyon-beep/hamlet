"""Universe metadata DTO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class UniverseMetadata:
    """High-level metadata describing a compiled universe."""

    # Identification
    universe_name: str
    schema_version: str

    # Substrate + topology
    substrate_type: str
    position_dim: int

    # Meter metadata
    meter_count: int
    meter_names: tuple[str, ...] = field(default_factory=tuple)
    meter_name_to_index: Mapping[str, int] = field(default_factory=dict)

    # Affordance metadata
    affordance_count: int = 0
    affordance_ids: tuple[str, ...] = field(default_factory=tuple)
    affordance_id_to_index: Mapping[str, int] = field(default_factory=dict)

    # Action/observation space
    action_count: int = 0
    observation_dim: int = 0

    # Spatial metadata (for grid substrates)
    grid_size: int | None = None
    grid_cells: int | None = None

    # Economic metadata
    max_sustainable_income: float = 0.0
    total_affordance_costs: float = 0.0
    economic_balance: float = 0.0

    # Temporal metadata
    ticks_per_day: int = 24

    # Versioning + provenance
    config_version: str = "1.0"
    compiler_version: str = "0.0.0"
    compiled_at: str = ""
    config_hash: str = ""
    provenance_id: str = ""
    compiler_git_sha: str = ""
    python_version: str = ""
    torch_version: str = ""
    pydantic_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "meter_names", tuple(self.meter_names))
        object.__setattr__(
            self,
            "meter_name_to_index",
            MappingProxyType(dict(self.meter_name_to_index)),
        )
        object.__setattr__(self, "affordance_ids", tuple(self.affordance_ids))
        object.__setattr__(
            self,
            "affordance_id_to_index",
            MappingProxyType(dict(self.affordance_id_to_index)),
        )
