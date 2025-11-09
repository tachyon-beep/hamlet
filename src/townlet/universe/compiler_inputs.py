"""Shared helper for collecting compiler input DTOs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
import yaml
from pydantic import ValidationError

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
from townlet.environment.action_config import ActionConfig, ActionSpaceConfig, load_global_actions_config
from townlet.environment.cascade_config import EnvironmentConfig
from townlet.environment.cascade_config import load_environment_config as load_cascade_environment_config
from townlet.substrate.config import ActionLabelConfig, SubstrateConfig
from townlet.substrate.factory import SubstrateFactory
from townlet.universe.errors import CompilationErrorCollector
from townlet.universe.source_map import SourceMap
from townlet.vfs.schema import VariableDef, load_variables_reference_config

_T = TypeVar("_T")


@dataclass(frozen=True)
class RawConfigs:
    """Container of all config DTOs the compiler needs for Stage 1."""

    hamlet_config: HamletConfig
    variables_reference: list[VariableDef]
    global_actions: ActionSpaceConfig
    action_labels: ActionLabelConfig | None
    environment_config: EnvironmentConfig
    source_map: SourceMap
    config_dir: Path

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
        errors = CompilationErrorCollector(stage="Stage 1: Parse")

        def _load_config(
            description: str,
            loader: Callable[[], _T],
            *,
            missing_hint: str | None = None,
            optional: bool = False,
        ) -> _T | None:
            try:
                return loader()
            except FileNotFoundError as exc:
                if optional:
                    return None
                errors.add(f"{description}: {exc}")
                if missing_hint and missing_hint not in errors.hints:
                    errors.add_hint(missing_hint)
            except yaml.YAMLError as exc:
                errors.add(f"{description}: YAML syntax error - {exc}")
            except ValidationError as exc:
                errors.add(f"{description}: validation error - {exc}")
            except ValueError as exc:
                errors.add(f"{description}: {exc}")
            return None

        hamlet_config = _load_config(
            "hamlet_config",
            lambda: HamletConfig.load(config_dir),
            missing_hint=(
                "Ensure the pack includes training.yaml, environment.yaml, population.yaml, curriculum.yaml, "
                "exploration.yaml, bars.yaml, cascades.yaml, affordances.yaml, substrate.yaml, and cues.yaml."
            ),
        )

        variables_reference = _load_config(
            "variables_reference.yaml",
            lambda: load_variables_reference_config(config_dir),
            missing_hint=f"Add variables_reference.yaml to {config_dir} (see docs/tasks for schema).",
        )

        global_actions_path = Path("configs") / "global_actions.yaml"
        global_actions = _load_config(
            "global_actions.yaml",
            lambda: load_global_actions_config(global_actions_path),
            missing_hint="Ensure configs/global_actions.yaml exists (global action vocabulary).",
        )

        composed_actions: ActionSpaceConfig | None = None
        if hamlet_config is not None and global_actions is not None:
            composed_actions = cls._compose_action_space(
                hamlet_config=hamlet_config,
                custom_actions=global_actions,
                errors=errors,
            )

        action_labels = _load_config(
            "action_labels.yaml",
            lambda: cls._load_action_labels_config(config_dir),
            optional=True,
        )

        environment_config = _load_config(
            "bars/cascades environment_config",
            lambda: load_cascade_environment_config(config_dir),
        )

        errors.check_and_raise()

        if hamlet_config is None or variables_reference is None or composed_actions is None or environment_config is None:
            # Defensive: reaching here would indicate check_and_raise failed to trigger.
            raise RuntimeError("Stage 1: Parse succeeded without fully loaded configs.")

        source_map = SourceMap()
        source_map.track_cascades(config_dir / "cascades.yaml")
        source_map.track_affordances(config_dir / "affordances.yaml")
        source_map.track_actions(global_actions_path)
        source_map.track_training_environment_key(
            config_dir / "training.yaml",
            "environment.enabled_affordances",
        )

        return cls(
            hamlet_config=hamlet_config,
            variables_reference=variables_reference,
            global_actions=composed_actions,
            action_labels=action_labels,
            environment_config=environment_config,
            source_map=source_map,
            config_dir=config_dir,
        )

    @staticmethod
    def _compose_action_space(
        *,
        hamlet_config: HamletConfig,
        custom_actions: ActionSpaceConfig,
        errors: CompilationErrorCollector,
    ) -> ActionSpaceConfig | None:
        """Combine substrate default actions with global custom actions for validation."""

        try:
            substrate = SubstrateFactory.build(hamlet_config.substrate, torch.device("cpu"))
        except Exception as exc:  # pragma: no cover - defensive
            errors.add(f"substrate.yaml: failed to build substrate actions - {exc}")
            return None

        try:
            substrate_actions = substrate.get_default_actions()
        except Exception as exc:  # pragma: no cover - defensive
            errors.add(f"substrate.yaml: failed to derive default actions - {exc}")
            return None

        combined: list[ActionConfig] = []
        next_id = 0

        def _clone(action: ActionConfig) -> ActionConfig:
            nonlocal next_id
            cloned = action.model_copy(update={"id": next_id})
            next_id += 1
            return cloned

        for action in substrate_actions:
            combined.append(_clone(action))

        for action in custom_actions.actions:
            combined.append(_clone(action))

        return ActionSpaceConfig(actions=combined)

    @staticmethod
    def _load_action_labels_config(config_dir: Path) -> ActionLabelConfig:
        yaml_path = config_dir / "action_labels.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"action_labels.yaml not found in {config_dir}")

        with yaml_path.open() as handle:
            data = yaml.safe_load(handle) or {}

        payload = data.get("action_labels", data)
        if not isinstance(payload, dict):
            raise ValueError(f"action_labels.yaml must define a mapping, got {type(payload).__name__}")

        return ActionLabelConfig(**payload)
