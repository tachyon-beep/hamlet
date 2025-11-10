"""Adapters between VFS observation fields and compiler DTOs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from townlet.universe.dto.observation_spec import ObservationField as CompilerObservationField
from townlet.universe.dto.observation_spec import ObservationSpec, compute_observation_field_uuid
from townlet.vfs.schema import ObservationField as VFSObservationField


def _infer_field_type(field: VFSObservationField) -> Literal["scalar", "vector", "categorical", "spatial_grid"]:
    if not field.shape:
        return "scalar"
    if len(field.shape) == 1:
        return "vector"
    return "spatial_grid"


def _flatten_dims(shape: list[int]) -> int:
    if not shape:
        return 1
    dims = 1
    for dim in shape:
        dims *= dim
    return dims


def _semantic_from_name(name: str) -> str | None:
    lowered = name.lower()
    if "position" in lowered:
        return "position"
    if any(token in lowered for token in ["energy", "health", "satiation", "mood", "fitness", "hygiene", "money"]):
        return "meter"
    if "affordance" in lowered:
        return "affordance"
    if "time" in lowered or "temporal" in lowered:
        return "temporal"
    return None


def _scope_from_metadata(
    exposed_to: list[str] | None,
    variable_scope: str | None,
) -> Literal["global", "agent", "agent_private"]:
    candidates = exposed_to or []
    if variable_scope == "global":
        return "global"
    if variable_scope == "agent_private":
        return "agent_private"
    if "global" in candidates:
        return "global"
    if "agent_private" in candidates:
        return "agent_private"
    if "agent" in candidates:
        return "agent"
    return "agent"


def vfs_to_observation_spec(
    fields: Iterable[VFSObservationField],
    variable_lookup: Mapping[str, str] | None = None,
) -> ObservationSpec:
    compiler_fields: list[CompilerObservationField] = []
    cursor = 0

    for field in fields:
        dims = _flatten_dims(field.shape)
        scope = _scope_from_metadata(
            getattr(field, "exposed_to", None),
            variable_lookup.get(field.source_variable) if variable_lookup else None,
        )
        semantic = _semantic_from_name(field.id)
        compiler_fields.append(
            CompilerObservationField(
                uuid=compute_observation_field_uuid(
                    name=field.id,
                    scope=scope,
                    description=field.source_variable,
                    dims=dims,
                    semantic_type=semantic,
                ),
                name=field.id,
                type=_infer_field_type(field),
                dims=dims,
                start_index=cursor,
                end_index=cursor + dims,
                scope=scope,
                description=field.source_variable,
                semantic_type=semantic,
            )
        )
        cursor += dims

    return ObservationSpec.from_fields(compiler_fields)
