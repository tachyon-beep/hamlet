"""Runtime-facing DTOs derived from CompiledUniverse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from townlet.config import HamletConfig
from townlet.environment.action_config import ActionSpaceConfig
from townlet.environment.cascade_config import EnvironmentConfig
from townlet.substrate.config import ActionLabelConfig, SubstrateConfig
from townlet.universe.dto import (
    ActionSpaceMetadata,
    AffordanceMetadata,
    MeterMetadata,
    ObservationActivity,
    ObservationSpec,
    UniverseMetadata,
)
from townlet.universe.optimization import OptimizationData
from townlet.vfs.schema import ObservationField as VfsObservationField
from townlet.vfs.schema import VariableDef


def _freeze_value(value: Any) -> Any:
    """Recursively wrap values in read-only views."""

    if isinstance(value, BaseModel):
        return _FrozenModelView(value)
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):  # pragma: no cover - defensive
        return tuple(sorted(_freeze_value(v) for v in value))
    return value


class _FrozenModelView:
    """Shallow proxy that exposes BaseModel attributes read-only."""

    __slots__ = ("_model",)

    def __init__(self, model: BaseModel) -> None:
        object.__setattr__(self, "_model", model)

    def __getattr__(self, item: str) -> Any:
        attr = getattr(self._model, item)
        return _freeze_value(attr)

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover - defensive
        raise AttributeError("RuntimeUniverse configuration views are read-only")


@dataclass(frozen=True)
class RuntimeUniverse:
    """Lightweight, read-only view of a compiled universe for runtime systems."""

    _hamlet_config: HamletConfig
    _global_actions: ActionSpaceConfig
    variables_reference: tuple[VariableDef, ...]
    config_dir: Path
    metadata: UniverseMetadata
    observation_spec: ObservationSpec
    observation_activity: ObservationActivity
    vfs_observation_fields: tuple[VfsObservationField, ...]
    action_space_metadata: ActionSpaceMetadata
    meter_metadata: MeterMetadata
    affordance_metadata: AffordanceMetadata
    optimization_data: OptimizationData
    _environment_config: EnvironmentConfig
    action_labels_config: ActionLabelConfig | None = None
    _hamlet_view: _FrozenModelView = field(init=False, repr=False)
    _global_actions_view: _FrozenModelView = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables_reference", tuple(self.variables_reference))
        object.__setattr__(self, "vfs_observation_fields", tuple(self.vfs_observation_fields))
        object.__setattr__(self, "_hamlet_view", _FrozenModelView(self._hamlet_config))
        object.__setattr__(self, "_global_actions_view", _FrozenModelView(self._global_actions))

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def hamlet_config(self) -> _FrozenModelView:
        """Return a read-only view of the hamlet configuration."""

        return self._hamlet_view

    @property
    def global_actions(self) -> _FrozenModelView:
        """Return a read-only view of the global action vocabulary."""

        return self._global_actions_view

    @property
    def meter_name_to_index(self) -> Mapping[str, int]:
        """Expose meter lookup for engines that require it."""

        return self.metadata.meter_name_to_index

    @property
    def affordance_ids(self) -> tuple[str, ...]:
        """Convenience accessor for affordance ordering."""

        return self.metadata.affordance_ids

    # ------------------------------------------------------------------
    # Clone helpers (runtime systems receive mutable copies as needed)
    # ------------------------------------------------------------------

    def clone_environment_config(self):
        return self._hamlet_config.environment.model_copy(deep=True)

    def clone_curriculum_config(self):
        return self._hamlet_config.curriculum.model_copy(deep=True)

    def clone_substrate_config(self) -> SubstrateConfig:
        return self._hamlet_config.substrate.model_copy(deep=True)

    def clone_affordance_configs(self):
        return tuple(aff.model_copy(deep=True) for aff in self._hamlet_config.affordances)

    def clone_global_actions(self) -> ActionSpaceConfig:
        return self._global_actions.model_copy(deep=True)

    def clone_action_labels_config(self) -> ActionLabelConfig | None:
        return self.action_labels_config.model_copy(deep=True) if self.action_labels_config else None

    def clone_environment_cascade_config(self) -> EnvironmentConfig:
        return self._environment_config.model_copy(deep=True)
