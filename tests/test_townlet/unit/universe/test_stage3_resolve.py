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
        action_labels=original.action_labels,
        environment_config=original.environment_config,
        source_map=original.source_map,
        config_dir=original.config_dir,
    )


def _run_stage3(raw_configs: RawConfigs) -> CompilationError:
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    errors = CompilationErrorCollector(stage="Stage 3: Resolve References")
    compiler._stage_3_resolve_references(raw_configs, symbol_table, errors)
    with pytest.raises(CompilationError) as exc_info:
        errors.check_and_raise("Stage 3: Resolve References")
    return exc_info.value


def test_stage3_detects_dangling_cascade_meter(base_raw_configs: RawConfigs) -> None:
    bad_cascade = base_raw_configs.cascades[0].model_copy(update={"name": "broken", "source": "ghost_meter"})
    mutated_cascades = base_raw_configs.cascades + (bad_cascade,)
    mutated_raw = _clone_raw_configs(
        base_raw_configs,
        hamlet_overrides={"cascades": mutated_cascades},
    )

    error = _run_stage3(mutated_raw)

    assert any("[UAC-RES-001]" in message and "ghost_meter" in message for message in error.errors)
    assert any(issue.code == "UAC-RES-001" for issue in error.issues)
    assert any(issue.location and "cascades" in issue.location for issue in error.issues)


def test_stage3_detects_invalid_environment_affordance(base_raw_configs: RawConfigs) -> None:
    bad_environment: TrainingEnvironmentConfig = base_raw_configs.environment.model_copy(
        update={"enabled_affordances": ["Bed", "UnknownAffordance"]}
    )
    mutated_raw = _clone_raw_configs(
        base_raw_configs,
        hamlet_overrides={"environment": bad_environment},
    )

    error = _run_stage3(mutated_raw)

    assert any("[UAC-RES-004]" in message and "UnknownAffordance" in message for message in error.errors)


def test_stage3_detects_invalid_action_cost_meter(base_raw_configs: RawConfigs) -> None:
    actions = list(base_raw_configs.global_actions.actions)
    broken_action: ActionConfig = actions[0].model_copy(update={"costs": {**actions[0].costs, "phantom_meter": 0.25}})
    actions[0] = broken_action
    mutated_actions = ActionSpaceConfig(actions=actions)
    mutated_raw = _clone_raw_configs(base_raw_configs, global_actions=mutated_actions)

    error = _run_stage3(mutated_raw)

    assert any("[UAC-RES-005]" in message and "phantom_meter" in message for message in error.errors)


def _mutated_affordance_raw_configs(base: RawConfigs, update: dict) -> RawConfigs:
    affordances = list(base.affordances)
    affordances[0] = affordances[0].model_copy(update=update)
    return _clone_raw_configs(base, hamlet_overrides={"affordances": tuple(affordances)})


def test_stage3_detects_invalid_costs_per_tick_meter(base_raw_configs: RawConfigs) -> None:
    mutated_raw = _mutated_affordance_raw_configs(
        base_raw_configs,
        {"costs_per_tick": [{"meter": "invalid_cost_meter", "amount": 0.01}]},
    )

    error = _run_stage3(mutated_raw)

    assert any("invalid_cost_meter" in message for message in error.errors)


def test_stage3_detects_invalid_effects_per_tick_meter(base_raw_configs: RawConfigs) -> None:
    mutated_raw = _mutated_affordance_raw_configs(
        base_raw_configs,
        {"effects_per_tick": [{"meter": "invalid_effect_tick", "amount": 0.02}]},
    )

    error = _run_stage3(mutated_raw)

    assert any("invalid_effect_tick" in message for message in error.errors)


def test_stage3_detects_missing_meter_field(base_raw_configs: RawConfigs) -> None:
    mutated_raw = _mutated_affordance_raw_configs(
        base_raw_configs,
        {"effects": [{"amount": 0.5}]},
    )

    error = _run_stage3(mutated_raw)

    assert any("UAC-RES-003" in message for message in error.errors)


def test_stage3_detects_invalid_completion_bonus_meter(base_raw_configs: RawConfigs) -> None:
    mutated_raw = _mutated_affordance_raw_configs(
        base_raw_configs,
        {"completion_bonus": [{"meter": "invalid_bonus", "amount": 0.1}]},
    )

    error = _run_stage3(mutated_raw)

    assert any("invalid_bonus" in message for message in error.errors)


def test_stage3_detects_invalid_effect_pipeline_meter(base_raw_configs: RawConfigs) -> None:
    mutated_raw = _mutated_affordance_raw_configs(
        base_raw_configs,
        {
            "effect_pipeline": {
                "on_start": [{"meter": "invalid_pipeline", "amount": 0.1}],
            }
        },
    )

    error = _run_stage3(mutated_raw)

    assert any("invalid_pipeline" in message for message in error.errors)


def test_stage3_detects_invalid_capability_meter(base_raw_configs: RawConfigs) -> None:
    mutated_raw = _mutated_affordance_raw_configs(
        base_raw_configs,
        {"capabilities": [{"type": "meter_gated", "meter": "invalid_cap"}]},
    )

    error = _run_stage3(mutated_raw)

    assert any("invalid_cap" in message for message in error.errors)


def test_stage3_detects_invalid_availability_meter(base_raw_configs: RawConfigs) -> None:
    mutated_raw = _mutated_affordance_raw_configs(
        base_raw_configs,
        {"availability": [{"meter": "invalid_availability"}]},
    )

    error = _run_stage3(mutated_raw)

    assert any("invalid_availability" in message for message in error.errors)


def test_stage3_detects_invalid_action_effect_meter(base_raw_configs: RawConfigs) -> None:
    actions = list(base_raw_configs.global_actions.actions)
    broken_action: ActionConfig = actions[0].model_copy(update={"effects": {**actions[0].effects, "fake_meter": 0.5}})
    actions[0] = broken_action
    mutated_actions = ActionSpaceConfig(actions=actions)
    mutated_raw = _clone_raw_configs(base_raw_configs, global_actions=mutated_actions)

    error = _run_stage3(mutated_raw)

    assert any("fake_meter" in message for message in error.errors)
