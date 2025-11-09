"""Immutable CompiledUniverse artifact."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

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
