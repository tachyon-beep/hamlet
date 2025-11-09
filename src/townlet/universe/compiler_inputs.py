"""Shared helper for collecting compiler input DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from townlet.config import HamletConfig
from townlet.config.affordance import AffordanceConfig
from townlet.config.bar import BarConfig
from townlet.config.cascade import CascadeConfig
from townlet.config.cues import CompoundCueConfig, CuesConfig, SimpleCueConfig
from townlet.config.curriculum import CurriculumConfig
from townlet.config.environment import TrainingEnvironmentConfig
from townlet.config.exploration import ExplorationConfig
from townlet.config.population import PopulationConfig
from townlet.config.training import TrainingConfig
from townlet.environment.action_config import ActionSpaceConfig, load_global_actions_config
from townlet.substrate.config import SubstrateConfig
from townlet.vfs.schema import VariableDef, load_variables_reference_config


@dataclass(frozen=True)
class RawConfigs:
    """Container of all config DTOs the compiler needs for Stage 1."""

    hamlet_config: HamletConfig
    variables_reference: list[VariableDef]
    global_actions: ActionSpaceConfig

    # --- Convenience accessors -------------------------------------------------

    @property
    def training(self) -> TrainingConfig:
        return self.hamlet_config.training

    @property
    def environment(self) -> TrainingEnvironmentConfig:
        return self.hamlet_config.environment

    @property
    def population(self) -> PopulationConfig:
        return self.hamlet_config.population

    @property
    def curriculum(self) -> CurriculumConfig:
        return self.hamlet_config.curriculum

    @property
    def exploration(self) -> ExplorationConfig:
        return self.hamlet_config.exploration

    @property
    def bars(self) -> tuple[BarConfig, ...]:
        return self.hamlet_config.bars

    @property
    def cascades(self) -> tuple[CascadeConfig, ...]:
        return self.hamlet_config.cascades

    @property
    def affordances(self) -> tuple[AffordanceConfig, ...]:
        return self.hamlet_config.affordances

    @property
    def cues(self) -> tuple[SimpleCueConfig | CompoundCueConfig, ...]:
        cues_config: CuesConfig = self.hamlet_config.cues
        return tuple(cues_config.simple_cues + cues_config.compound_cues)

    @property
    def substrate(self) -> SubstrateConfig:
        return self.hamlet_config.substrate

    # --- Factory ---------------------------------------------------------------

    @classmethod
    def from_config_dir(cls, config_dir: Path) -> RawConfigs:
        """Load config DTOs from a pack directory."""

        hamlet_config = HamletConfig.load(config_dir)
        variables_reference = load_variables_reference_config(config_dir)
        global_actions = load_global_actions_config()
        return cls(
            hamlet_config=hamlet_config,
            variables_reference=variables_reference,
            global_actions=global_actions,
        )
