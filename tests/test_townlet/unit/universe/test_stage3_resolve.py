"""Tests for Stage 3 reference resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.config.environment import TrainingEnvironmentConfig
from townlet.environment.action_config import ActionConfig, ActionSpaceConfig
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.errors import CompilationError, CompilationErrorCollector


@pytest.fixture(scope="module")
def base_raw_configs() -> RawConfigs:
    return RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))


def _clone_raw_configs(
    original: RawConfigs,
    *,
    hamlet_overrides: dict | None = None,
    global_actions: ActionSpaceConfig | None = None,
) -> RawConfigs:
    if hamlet_overrides:
        hamlet_config = original.hamlet_config.model_copy(update=hamlet_overrides)
    else:
        hamlet_config = original.hamlet_config
    return RawConfigs(
        hamlet_config=hamlet_config,
        variables_reference=original.variables_reference,
        global_actions=global_actions or original.global_actions,
    )


def _run_stage3(raw_configs: RawConfigs) -> list[str]:
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    errors = CompilationErrorCollector(stage="Stage 3: Resolve References")
    compiler._stage_3_resolve_references(raw_configs, symbol_table, errors)
    with pytest.raises(CompilationError) as exc_info:
        errors.check_and_raise("Stage 3: Resolve References")
    return exc_info.value.errors


def test_stage3_detects_dangling_cascade_meter(base_raw_configs: RawConfigs) -> None:
    bad_cascade = base_raw_configs.cascades[0].model_copy(update={"name": "broken", "source": "ghost_meter"})
    mutated_cascades = base_raw_configs.cascades + (bad_cascade,)
    mutated_raw = _clone_raw_configs(
        base_raw_configs,
        hamlet_overrides={"cascades": mutated_cascades},
    )

    errors = _run_stage3(mutated_raw)

    assert any("ghost_meter" in message for message in errors)


def test_stage3_detects_invalid_environment_affordance(base_raw_configs: RawConfigs) -> None:
    bad_environment: TrainingEnvironmentConfig = base_raw_configs.environment.model_copy(
        update={"enabled_affordances": ["Bed", "UnknownAffordance"]}
    )
    mutated_raw = _clone_raw_configs(
        base_raw_configs,
        hamlet_overrides={"environment": bad_environment},
    )

    errors = _run_stage3(mutated_raw)

    assert any("UnknownAffordance" in message for message in errors)


def test_stage3_detects_invalid_action_cost_meter(base_raw_configs: RawConfigs) -> None:
    actions = list(base_raw_configs.global_actions.actions)
    broken_action: ActionConfig = actions[0].model_copy(update={"costs": {**actions[0].costs, "phantom_meter": 0.25}})
    actions[0] = broken_action
    mutated_actions = ActionSpaceConfig(actions=actions)
    mutated_raw = _clone_raw_configs(base_raw_configs, global_actions=mutated_actions)

    errors = _run_stage3(mutated_raw)

    assert any("phantom_meter" in message for message in errors)
