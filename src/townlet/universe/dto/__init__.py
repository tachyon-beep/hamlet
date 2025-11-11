"""Universe metadata DTOs shared between compiler and training systems."""

from townlet.universe.dto.action_metadata import ActionMetadata, ActionSpaceMetadata
from townlet.universe.dto.affordance_metadata import (
    AffordanceInfo,
    AffordanceMetadata,
)
from townlet.universe.dto.meter_metadata import MeterInfo, MeterMetadata
from townlet.universe.dto.observation_activity import ObservationActivity
from townlet.universe.dto.observation_spec import (
    ObservationField,
    ObservationSpec,
    compute_observation_field_uuid,
)
from townlet.universe.dto.universe_metadata import UniverseMetadata

__all__ = [
    "ActionMetadata",
    "ActionSpaceMetadata",
    "AffordanceInfo",
    "AffordanceMetadata",
    "MeterInfo",
    "MeterMetadata",
    "ObservationActivity",
    "ObservationField",
    "ObservationSpec",
    "compute_observation_field_uuid",
    "UniverseMetadata",
]
