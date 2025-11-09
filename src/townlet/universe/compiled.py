"""Immutable CompiledUniverse artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import msgpack
import torch

from townlet.config import HamletConfig
from townlet.environment.action_config import ActionSpaceConfig
from townlet.universe.dto import (
    ActionMetadata,
    ActionSpaceMetadata,
    AffordanceInfo,
    AffordanceMetadata,
    MeterInfo,
    MeterMetadata,
    ObservationField,
    ObservationSpec,
    UniverseMetadata,
)
from townlet.universe.optimization import OptimizationData
from townlet.universe.runtime import RuntimeUniverse
from townlet.vfs.schema import VariableDef


@dataclass(frozen=True)
class CompiledUniverse:
    """Complete, immutable representation of a compiled universe."""

    hamlet_config: HamletConfig
    variables_reference: Sequence[VariableDef]
    global_actions: ActionSpaceConfig
    config_dir: Path
    metadata: UniverseMetadata
    observation_spec: ObservationSpec
    action_space_metadata: ActionSpaceMetadata
    meter_metadata: MeterMetadata
    affordance_metadata: AffordanceMetadata
    optimization_data: OptimizationData

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables_reference", tuple(self.variables_reference))
        object.__setattr__(self, "config_dir", Path(self.config_dir))
        if self.metadata.meter_count != len(self.hamlet_config.bars):
            raise ValueError(
                "Metadata meter_count does not match bars length. " f"{self.metadata.meter_count} vs {len(self.hamlet_config.bars)}"
            )
        if self.metadata.affordance_count != len(self.hamlet_config.affordances):
            raise ValueError(
                "Metadata affordance_count does not match affordances length. "
                f"{self.metadata.affordance_count} vs {len(self.hamlet_config.affordances)}"
            )

    # Convenience properties -------------------------------------------------

    @property
    def substrate(self):
        return self.hamlet_config.substrate

    @property
    def bars(self):
        return self.hamlet_config.bars

    @property
    def cascades(self):
        return self.hamlet_config.cascades

    @property
    def affordances(self):
        return self.hamlet_config.affordances

    @property
    def cues(self):
        return self.hamlet_config.cues

    @property
    def training(self):
        return self.hamlet_config.training

    # Runtime helpers -------------------------------------------------------

    def create_environment(self, num_agents: int, device: str = "cpu"):
        """Instantiate a VectorizedHamletEnv using this compiled universe."""

        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env_cfg = self.hamlet_config.environment
        curriculum = self.hamlet_config.curriculum

        return VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=env_cfg.grid_size,
            partial_observability=env_cfg.partial_observability,
            vision_range=env_cfg.vision_range,
            enable_temporal_mechanics=env_cfg.enable_temporal_mechanics,
            move_energy_cost=env_cfg.energy_move_depletion,
            wait_energy_cost=env_cfg.energy_wait_depletion,
            interact_energy_cost=env_cfg.energy_interact_depletion,
            agent_lifespan=curriculum.max_steps_per_episode,
            device=torch.device(device),
            enabled_affordances=env_cfg.enabled_affordances,
            config_pack_path=self.config_dir,
        )

    # Checkpoint compatibility -----------------------------------------------

    def check_checkpoint_compatibility(self, checkpoint: dict) -> tuple[bool, str]:
        """Validate whether a checkpoint can be used with this compiled universe."""

        checkpoint_hash = checkpoint.get("config_hash")
        if checkpoint_hash is None:
            return (
                False,
                "Checkpoint missing config_hash; retraining recommended.",
            )
        if checkpoint_hash != self.metadata.config_hash:
            return (
                False,
                "Config hash mismatch between checkpoint and compiled universe.",
            )

        checkpoint_obs_dim = checkpoint.get("observation_dim")
        if checkpoint_obs_dim is not None and checkpoint_obs_dim != self.metadata.observation_dim:
            return (
                False,
                "Observation dimension mismatch between checkpoint and compiled universe.",
            )

        checkpoint_action_dim = checkpoint.get("action_dim")
        if checkpoint_action_dim is not None and checkpoint_action_dim != self.metadata.action_count:
            return (
                False,
                "Action dimension mismatch between checkpoint and compiled universe.",
            )

        return True, "Checkpoint compatible."

    def to_runtime(self) -> RuntimeUniverse:
        """Create a runtime-facing DTO for environment and training systems."""

        return RuntimeUniverse(
            hamlet_config=deepcopy(self.hamlet_config),
            global_actions=deepcopy(self.global_actions),
            variables_reference=tuple(deepcopy(var) for var in self.variables_reference),
            config_dir=self.config_dir,
            metadata=self.metadata,
            observation_spec=self.observation_spec,
            action_space_metadata=self.action_space_metadata,
            meter_metadata=self.meter_metadata,
            affordance_metadata=self.affordance_metadata,
            optimization_data=self.optimization_data,
        )

    # Serialization -----------------------------------------------------------

    def save_to_cache(self, path: Path) -> None:
        """Serialize compiled universe to MessagePack file."""

        data = {
            "hamlet_config": self.hamlet_config.model_dump(),
            "variables_reference": [var.model_dump() for var in self.variables_reference],
            "global_actions": self.global_actions.model_dump(),
            "config_dir": str(self.config_dir),
            "metadata": _dataclass_to_plain(self.metadata),
            "observation_spec": _dataclass_to_plain(self.observation_spec),
            "action_space_metadata": _dataclass_to_plain(self.action_space_metadata),
            "meter_metadata": _dataclass_to_plain(self.meter_metadata),
            "affordance_metadata": _dataclass_to_plain(self.affordance_metadata),
            "optimization_data": {
                "base_depletions": self.optimization_data.base_depletions.cpu().tolist(),
                "cascade_data": self.optimization_data.cascade_data,
                "modulation_data": self.optimization_data.modulation_data,
                "action_mask_table": (
                    self.optimization_data.action_mask_table.cpu().tolist()
                    if self.optimization_data.action_mask_table is not None
                    else None
                ),
                "affordance_position_map": _serialize_affordance_positions(self.optimization_data.affordance_position_map),
            },
        }

        packed = msgpack.packb(data, use_bin_type=True)
        path.write_bytes(packed)

    @classmethod
    def load_from_cache(cls, path: Path) -> CompiledUniverse:
        """Deserialize a compiled universe from MessagePack."""

        payload = msgpack.unpackb(path.read_bytes(), raw=False)

        optimization_payload = payload["optimization_data"]
        action_mask = optimization_payload.get("action_mask_table")
        if action_mask is None:
            action_mask_tensor = torch.zeros((24, 0), dtype=torch.bool)
        else:
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
        return cls(
            hamlet_config=HamletConfig.model_validate(payload["hamlet_config"]),
            variables_reference=[VariableDef.model_validate(var) for var in payload["variables_reference"]],
            global_actions=ActionSpaceConfig.model_validate(payload["global_actions"]),
            config_dir=Path(payload["config_dir"]),
            metadata=UniverseMetadata(**payload["metadata"]),
            observation_spec=ObservationSpec(
                total_dims=payload["observation_spec"]["total_dims"],
                encoding_version=payload["observation_spec"]["encoding_version"],
                fields=tuple(ObservationField(**field) for field in payload["observation_spec"]["fields"]),
            ),
            action_space_metadata=ActionSpaceMetadata(
                total_actions=payload["action_space_metadata"]["total_actions"],
                actions=tuple(ActionMetadata(**entry) for entry in payload["action_space_metadata"]["actions"]),
            ),
            meter_metadata=MeterMetadata(meters=tuple(MeterInfo(**entry) for entry in payload["meter_metadata"]["meters"])),
            affordance_metadata=AffordanceMetadata(
                affordances=tuple(AffordanceInfo(**entry) for entry in payload["affordance_metadata"]["affordances"])
            ),
            optimization_data=OptimizationData(
                base_depletions=torch.tensor(optimization_payload["base_depletions"], dtype=torch.float32),
                cascade_data=optimization_payload["cascade_data"],
                modulation_data=optimization_payload["modulation_data"],
                action_mask_table=action_mask_tensor,
                affordance_position_map=_deserialize_affordance_positions(optimization_payload["affordance_position_map"]),
            ),
        )


def _dataclass_to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {key: _dataclass_to_plain(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_dataclass_to_plain(value) for value in obj]
    return obj


def _serialize_affordance_positions(position_map: dict[str, torch.Tensor | None]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in position_map.items():
        if isinstance(value, torch.Tensor):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized


def _deserialize_affordance_positions(payload: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    restored: dict[str, torch.Tensor | None] = {}
    for key, value in payload.items():
        if value is None:
            restored[key] = None
        else:
            restored[key] = torch.tensor(value)
    return restored
