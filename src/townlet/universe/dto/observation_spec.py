"""Observation specification DTOs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal


def compute_observation_field_uuid(
    *,
    name: str,
    scope: str,
    description: str | None,
    dims: int,
    semantic_type: str | None,
) -> str:
    """Return a deterministic UUID for an observation field."""

    payload = "|".join(
        (
            scope,
            name,
            description or "",
            str(dims),
            semantic_type or "",
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ObservationField:
    """Single field in the flattened observation vector."""

    uuid: str | None
    name: str
    type: Literal["scalar", "vector", "categorical", "spatial_grid"]
    dims: int
    start_index: int
    end_index: int
    scope: Literal["global", "agent", "agent_private"]
    description: str
    semantic_type: str | None = None
    categorical_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.uuid:
            generated = compute_observation_field_uuid(
                name=self.name,
                scope=self.scope,
                description=self.description,
                dims=self.dims,
                semantic_type=self.semantic_type,
            )
            object.__setattr__(self, "uuid", generated)
        if self.dims <= 0:
            raise ValueError(f"Observation field '{self.name}' must have dims > 0")
        if self.end_index <= self.start_index:
            raise ValueError(f"Observation field '{self.name}' has invalid indices: {self.start_index}..{self.end_index}")
        if self.categorical_labels is not None:
            if not self.categorical_labels:
                raise ValueError(f"Observation field '{self.name}' categorical labels cannot be empty")
            object.__setattr__(self, "categorical_labels", tuple(self.categorical_labels))


@dataclass(frozen=True)
class ObservationSpec:
    """Complete observation specification emitted by the compiler."""

    total_dims: int
    fields: tuple[ObservationField, ...] = field(default_factory=tuple)
    encoding_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        if self.total_dims < 0:
            raise ValueError("ObservationSpec total_dims must be >= 0")
        uuids = [field.uuid for field in self.fields]
        if len(set(uuids)) != len(uuids):
            raise ValueError("ObservationSpec contains duplicate observation field UUIDs")

    def get_field_by_name(self, name: str) -> ObservationField:
        """Lookup field by name."""
        for obs_field in self.fields:
            if obs_field.name == name:
                return obs_field
        raise KeyError(f"Field '{name}' not found in observation spec")

    def get_fields_by_semantic_type(self, semantic: str) -> list[ObservationField]:
        """Return fields that share a semantic type (e.g., 'meter')."""
        return [obs_field for obs_field in self.fields if obs_field.semantic_type == semantic]

    @classmethod
    def from_fields(cls, fields: Sequence[ObservationField], encoding_version: str = "1.0") -> ObservationSpec:
        """Helper for building from a sequence while auto-computing dims."""
        total_dims = sum(field.dims for field in fields)
        return cls(total_dims=total_dims, fields=tuple(fields), encoding_version=encoding_version)
