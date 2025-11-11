"""Observation activity metadata for curriculum masking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationActivity:
    """Metadata describing which observation dimensions are active.

    Used by structured encoders and RND to mask out padding dimensions
    that are present for portability but inactive in current curriculum level.

    Example:
        L0_0 has 8 meters but only uses health/energy:
        - active_mask: (True, True, False, False, False, False, False, False)
        - group_slices: {"bars": slice(0, 8)}
        - active_field_uuids: ("health_uuid", "energy_uuid")

    Attributes:
        active_mask: Tuple of bool indicating active (True) vs padding (False)
            for each observation dimension. Length must equal total_dims.
        group_slices: Dict mapping semantic group name to slice in observation vector.
            Keys: "bars", "spatial", "affordance", "temporal", "custom".
            Used by structured encoders to extract group features.
        active_field_uuids: Tuple of UUIDs for active fields only (where mask=True).
            Used for checkpoint compatibility validation.
    """

    active_mask: tuple[bool, ...]
    group_slices: dict[str, slice]
    active_field_uuids: tuple[str, ...]

    @property
    def total_dims(self) -> int:
        """Total observation dimensions (active + padding)."""
        return len(self.active_mask)

    @property
    def active_dim_count(self) -> int:
        """Count of active dimensions (True in mask)."""
        return sum(self.active_mask)

    def get_group_slice(self, group_name: str) -> slice | None:
        """Get slice for semantic group, or None if not present."""
        return self.group_slices.get(group_name)
