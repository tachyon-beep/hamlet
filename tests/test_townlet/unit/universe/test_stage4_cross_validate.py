"""Tests for Stage 4 cross-validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.config.cues import CueCondition, CuesConfig, SimpleCueConfig, VisualCueConfig
from townlet.environment.action_config import ActionConfig, ActionSpaceConfig
from townlet.substrate.config import AspatialSubstrateConfig, SubstrateConfig
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
    hamlet_config = original.hamlet_config.model_copy(update=hamlet_overrides or {})
    return RawConfigs(
        hamlet_config=hamlet_config,
        variables_reference=original.variables_reference,
        global_actions=global_actions or original.global_actions,
        source_map=original.source_map,
        config_dir=original.config_dir,
    )


def _run_stage4_collector(raw_configs: RawConfigs) -> CompilationErrorCollector:
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    collector = CompilationErrorCollector(stage="Stage 4: Cross-Validation")
    compiler._stage_4_cross_validate(raw_configs, symbol_table, collector)
    return collector


def _run_stage4_expect_error(raw_configs: RawConfigs) -> list[str]:
    collector = _run_stage4_collector(raw_configs)
    with pytest.raises(CompilationError) as exc_info:
        collector.check_and_raise("Stage 4: Cross-Validation")
    return exc_info.value.errors


def test_stage4_detects_spatial_infeasibility(base_raw_configs: RawConfigs) -> None:
    env = base_raw_configs.environment.model_copy(update={"grid_size": 1, "enabled_affordances": ["Bed", "Hospital", "HomeMeal"]})
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"environment": env})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-001" in message for message in errors)


def test_stage4_detects_cascade_cycle(base_raw_configs: RawConfigs) -> None:
    cascades = list(base_raw_configs.cascades)
    cascades.append(cascades[0].model_copy(update={"name": "loop_one", "source": "energy", "target": "energy"}))
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"cascades": tuple(cascades)})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-003" in message for message in errors)


def test_stage4_detects_aspatial_movement_actions(base_raw_configs: RawConfigs) -> None:
    aspatial_substrate = SubstrateConfig(
        version="1.0",
        description="Aspatial",
        type="aspatial",
        aspatial=AspatialSubstrateConfig(),
    )
    actions = list(base_raw_configs.global_actions.actions)
    actions[0] = actions[0].model_copy(update={"type": "movement", "delta": [0, 1]})
    mutated_actions = ActionSpaceConfig(actions=actions)
    mutated_raw = _clone_raw_configs(
        base_raw_configs,
        hamlet_overrides={"substrate": aspatial_substrate},
        global_actions=mutated_actions,
    )

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-006" in message for message in errors)


def test_stage4_warns_on_economic_imbalance(base_raw_configs: RawConfigs) -> None:
    collector = _run_stage4_collector(base_raw_configs)
    assert any("UAC-VAL-002" in warning for warning in collector.warnings)


def test_stage4_detects_availability_meter_reference(base_raw_configs: RawConfigs) -> None:
    affs = list(base_raw_configs.affordances)
    affs[0] = affs[0].model_copy(update={"availability": [{"meter": "unknown", "min": 0.2}]})
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"affordances": tuple(affs)})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-007" in message for message in errors)


def test_stage4_detects_availability_min_max_order(base_raw_configs: RawConfigs) -> None:
    affs = list(base_raw_configs.affordances)
    affs[0] = affs[0].model_copy(update={"availability": [{"meter": "energy", "min": 0.8, "max": 0.2}]})
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"affordances": tuple(affs)})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("min" in message and "max" in message for message in errors)


def test_stage4_detects_visual_cue_range_gap(base_raw_configs: RawConfigs) -> None:
    cues = CuesConfig(
        version="1.0",
        simple_cues=[
            SimpleCueConfig(
                cue_id="tmp",
                name="Tmp",
                category="energy",
                visibility="public",
                condition=CueCondition(meter="energy", operator="<", threshold=0.2),
            )
        ],
        visual_cues={
            "energy": [
                VisualCueConfig(range=(0.0, 0.3), label="low"),
                VisualCueConfig(range=(0.5, 1.0), label="high"),
            ]
        },
    )
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"cues": cues})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-009" in message for message in errors)


def test_stage4_detects_capability_conflict(base_raw_configs: RawConfigs) -> None:
    affs = list(base_raw_configs.affordances)
    affs[0] = affs[0].model_copy(
        update={
            "interaction_type": "instant",
            "capabilities": [{"type": "multi_tick", "duration_ticks": 5}],
            "effect_pipeline": None,
        }
    )
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"affordances": tuple(affs)})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-008" in message for message in errors)


def test_stage4_detects_resumable_without_multi_tick(base_raw_configs: RawConfigs) -> None:
    affs = list(base_raw_configs.affordances)
    affs[0] = affs[0].model_copy(
        update={
            "capabilities": [{"type": "cooldown", "resumable": True}],
            "effect_pipeline": None,
        }
    )
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"affordances": tuple(affs)})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("'resumable' flag" in message for message in errors)


def test_stage4_detects_missing_effect_pipeline_for_multi_tick(base_raw_configs: RawConfigs) -> None:
    affs = list(base_raw_configs.affordances)
    affs[0] = affs[0].model_copy(
        update={
            "interaction_type": "multi",
            "capabilities": [{"type": "multi_tick", "duration_ticks": 3}],
            "effect_pipeline": None,
        }
    )
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"affordances": tuple(affs)})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-008" in message for message in errors)


def test_stage4_warns_on_early_exit_without_permission(base_raw_configs: RawConfigs) -> None:
    affs = list(base_raw_configs.affordances)
    aff = affs[0].model_copy(
        update={
            "capabilities": [{"type": "multi_tick", "duration_ticks": 3, "early_exit_allowed": False}],
            "effect_pipeline": {
                "on_completion": [{"meter": "energy", "amount": 0.1}],
                "on_early_exit": [{"meter": "mood", "amount": -0.1}],
            },
        }
    )
    affs[0] = aff
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"affordances": tuple(affs)})

    collector = _run_stage4_collector(mutated_raw)

    assert any("UAC-VAL-008" in warning for warning in collector.warnings)


def test_stage4_detects_position_out_of_bounds(base_raw_configs: RawConfigs) -> None:
    affs = list(base_raw_configs.affordances)
    affs[0] = affs[0].model_copy(update={"position": [10, 10]})
    mutated_raw = _clone_raw_configs(base_raw_configs, hamlet_overrides={"affordances": tuple(affs)})

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-010" in message for message in errors)


def test_stage4_detects_missing_square_movements(base_raw_configs: RawConfigs) -> None:
    # Keep only two movement actions to trigger validator
    limited_actions = [
        action
        for action in base_raw_configs.global_actions.actions
        if action.type != "movement" or tuple(action.delta or []) in {(0, 1), (1, 0)}
    ]
    limited_actions.append(
        ActionConfig(
            id=999,
            name="INTERACT",
            type="interaction",
            costs={},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            reads=[],
            writes=[],
        )
    )
    mutated_actions = ActionSpaceConfig(actions=limited_actions)
    mutated_raw = _clone_raw_configs(base_raw_configs, global_actions=mutated_actions)

    errors = _run_stage4_expect_error(mutated_raw)

    assert any("UAC-VAL-006" in message for message in errors)


def test_stage4_exception_includes_warnings(base_raw_configs: RawConfigs) -> None:
    limited_actions = [
        action
        for action in base_raw_configs.global_actions.actions
        if action.type != "movement" or tuple(action.delta or []) in {(0, 1), (1, 0)}
    ]
    limited_actions.append(
        ActionConfig(
            id=999,
            name="INTERACT",
            type="interaction",
            costs={},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            reads=[],
            writes=[],
        )
    )
    mutated_actions = ActionSpaceConfig(actions=limited_actions)
    mutated_raw = _clone_raw_configs(base_raw_configs, global_actions=mutated_actions)

    collector = _run_stage4_collector(mutated_raw)

    with pytest.raises(CompilationError) as exc_info:
        collector.check_and_raise("Stage 4: Cross-Validation")

    assert exc_info.value.warnings, "Warnings should be attached to the raised CompilationError"
