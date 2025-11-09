"""Runtime-facing DTOs derived from CompiledUniverse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from townlet.config import HamletConfig
from townlet.environment.action_config import ActionSpaceConfig
from townlet.universe.dto import (
    ActionSpaceMetadata,
    AffordanceMetadata,
    MeterMetadata,
    ObservationSpec,
    UniverseMetadata,
)
from townlet.universe.optimization import OptimizationData
from townlet.vfs.schema import VariableDef


@dataclass(frozen=True)
class RuntimeUniverse:
    """Lightweight view of a compiled universe for runtime systems."""

    hamlet_config: HamletConfig
    global_actions: ActionSpaceConfig
    variables_reference: tuple[VariableDef, ...]
    metadata: UniverseMetadata
    observation_spec: ObservationSpec
    action_space_metadata: ActionSpaceMetadata
    meter_metadata: MeterMetadata
    affordance_metadata: AffordanceMetadata
    optimization_data: OptimizationData

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables_reference", tuple(self.variables_reference))

    @property
    def meter_name_to_index(self) -> Mapping[str, int]:
        """Expose meter lookup for engines that require it."""

        return self.metadata.meter_name_to_index

    @property
    def affordance_ids(self) -> tuple[str, ...]:
        """Convenience accessor for affordance ordering."""

        return self.metadata.affordance_ids
