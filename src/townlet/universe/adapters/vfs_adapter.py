"""Adapters between VFS observation fields and compiler DTOs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from townlet.universe.dto.observation_activity import ObservationActivity
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


class VFSAdapter:
    """Adapter for converting VFS schemas to compiler DTOs."""

    @staticmethod
    def build_observation_activity(
        observation_spec: list[VFSObservationField],
        field_uuids: dict[str, str],
    ) -> ObservationActivity:
        """Build ObservationActivity metadata from observation spec.

        Flattens observation fields into a boolean mask indicating active vs padding
        dimensions, computes group slices for semantic types, and collects active UUIDs.

        Args:
            observation_spec: List of observation fields with semantic_type and curriculum_active
            field_uuids: Dict mapping field id to UUID

        Returns:
            ObservationActivity with mask, slices, and UUIDs

        Example:
            >>> spec = [
            ...     ObservationField(id="health", shape=[1], semantic_type="bars", curriculum_active=True),
            ...     ObservationField(id="mood", shape=[1], semantic_type="bars", curriculum_active=False),
            ... ]
            >>> uuids = {"health": "uuid1", "mood": "uuid2"}
            >>> activity = VFSAdapter.build_observation_activity(spec, uuids)
            >>> activity.active_mask
            (True, False)
            >>> activity.active_dim_count
            1
        """
        if not observation_spec:
            return ObservationActivity(
                active_mask=(),
                group_slices={},
                active_field_uuids=(),
            )

        # Build flat mask by expanding each field's shape
        active_mask_list: list[bool] = []
        active_uuids_list: list[str] = []

        # Track group boundaries (start_idx for each semantic_type)
        group_boundaries: dict[str, int] = {}
        group_end_indices: dict[str, int] = {}
        current_idx = 0

        for field in observation_spec:
            # Record group start if first time seeing this semantic_type
            if field.semantic_type not in group_boundaries:
                group_boundaries[field.semantic_type] = current_idx

            # Flatten field into mask (one bool per dimension)
            field_dims = 1 if not field.shape else int(sum(field.shape))
            field_uuid = field_uuids.get(field.id, field.id)

            for _ in range(field_dims):
                active_mask_list.append(field.curriculum_active)
                if field.curriculum_active:
                    active_uuids_list.append(field_uuid)

            # Update group end
            current_idx += field_dims
            group_end_indices[field.semantic_type] = current_idx

        # Build group slices
        group_slices = {
            group_name: slice(group_boundaries[group_name], group_end_indices[group_name]) for group_name in group_boundaries.keys()
        }

        return ObservationActivity(
            active_mask=tuple(active_mask_list),
            group_slices=group_slices,
            active_field_uuids=tuple(active_uuids_list),
        )
