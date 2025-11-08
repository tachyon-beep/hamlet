"""Universe metadata DTO."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseMetadata:
    """High-level metadata describing a compiled universe."""

    universe_name: str
    schema_version: str
    compiled_at: str
    config_hash: str
    obs_dim: int
    action_dim: int
    num_meters: int
    num_affordances: int
    position_dim: int
