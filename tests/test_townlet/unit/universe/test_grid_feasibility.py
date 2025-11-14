"""Tests for grid feasibility validation with multi-agent populations.

BUG-20: Spatial feasibility check should account for num_agents, not assume +1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.errors import CompilationError, CompilationErrorCollector


@pytest.fixture(scope="module")
def base_raw_configs() -> RawConfigs:
    """Load minimal config as base."""
    return RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))


def _clone_raw_configs(
    original: RawConfigs,
    *,
    hamlet_overrides: dict | None = None,
) -> RawConfigs:
    """Clone RawConfigs with optional overrides."""
    hamlet_config = original.hamlet_config.model_copy(update=hamlet_overrides or {})
    return RawConfigs(
        hamlet_config=hamlet_config,
        variables_reference=original.variables_reference,
        global_actions=original.global_actions,
        action_labels=original.action_labels,
        environment_config=original.environment_config,
        source_map=original.source_map,
        config_dir=original.config_dir,
    )


def _run_stage4_collector(raw_configs: RawConfigs) -> CompilationErrorCollector:
    """Run Stage 4 cross-validation and return error collector."""
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    collector = CompilationErrorCollector(stage="Stage 4: Cross-Validation")
    compiler._stage_4_cross_validate(raw_configs, symbol_table, collector)
    return collector


def _run_stage4_expect_error(raw_configs: RawConfigs) -> CompilationError:
    """Run Stage 4 and expect a compilation error."""
    collector = _run_stage4_collector(raw_configs)
    with pytest.raises(CompilationError) as exc_info:
        collector.check_and_raise("Stage 4: Cross-Validation")
    return exc_info.value


def test_spatial_feasibility_single_agent_sufficient_space(base_raw_configs: RawConfigs) -> None:
    """Test that 3×3 grid with 1 affordance + 1 agent (2 cells) passes."""
    substrate = base_raw_configs.substrate.model_copy(deep=True)
    if substrate.grid is not None:
        substrate.grid.width = 3
        substrate.grid.height = 3

    env = base_raw_configs.environment.model_copy(update={"enabled_affordances": ["Bed"]})
    population = base_raw_configs.population.model_copy(update={"num_agents": 1})

    mutated_raw = _clone_raw_configs(
        base_raw_configs, hamlet_overrides={"environment": env, "substrate": substrate, "population": population}
    )

    collector = _run_stage4_collector(mutated_raw)
    # Should NOT raise - 9 cells >= 2 required (1 affordance + 1 agent)
    collector.check_and_raise("Stage 4: Cross-Validation")


def test_spatial_feasibility_multi_agent_insufficient_space(base_raw_configs: RawConfigs) -> None:
    """Test that 3×3 grid with 2 affordances + 8 agents (10 cells needed) fails.

    BUG-20: This test SHOULD fail but currently passes due to hard-coded +1.
    Grid has 9 cells but needs 10 (2 affordances + 8 agents).
    """
    substrate = base_raw_configs.substrate.model_copy(deep=True)
    if substrate.grid is not None:
        substrate.grid.width = 3
        substrate.grid.height = 3

    env = base_raw_configs.environment.model_copy(update={"enabled_affordances": ["Bed", "Hospital"]})
    population = base_raw_configs.population.model_copy(update={"num_agents": 8})

    mutated_raw = _clone_raw_configs(
        base_raw_configs, hamlet_overrides={"environment": env, "substrate": substrate, "population": population}
    )

    error = _run_stage4_expect_error(mutated_raw)

    # Should detect spatial impossibility: 9 cells < 10 required (2 affordances + 8 agents)
    assert any("UAC-VAL-001" in message for message in error.errors)
    assert any(issue.code == "UAC-VAL-001" for issue in error.issues)
    assert any("9 cells" in message and "10" in message for message in error.errors)


def test_spatial_feasibility_multi_agent_exact_capacity(base_raw_configs: RawConfigs) -> None:
    """Test that 3×3 grid with 1 affordance + 8 agents (9 cells needed) passes."""
    substrate = base_raw_configs.substrate.model_copy(deep=True)
    if substrate.grid is not None:
        substrate.grid.width = 3
        substrate.grid.height = 3

    env = base_raw_configs.environment.model_copy(update={"enabled_affordances": ["Bed"]})
    population = base_raw_configs.population.model_copy(update={"num_agents": 8})

    mutated_raw = _clone_raw_configs(
        base_raw_configs, hamlet_overrides={"environment": env, "substrate": substrate, "population": population}
    )

    collector = _run_stage4_collector(mutated_raw)
    # Should NOT raise - 9 cells >= 9 required (1 affordance + 8 agents)
    collector.check_and_raise("Stage 4: Cross-Validation")


def test_spatial_feasibility_multi_agent_over_capacity_by_one(base_raw_configs: RawConfigs) -> None:
    """Test that 3×3 grid with 0 affordances + 10 agents (10 cells needed) fails."""
    substrate = base_raw_configs.substrate.model_copy(deep=True)
    if substrate.grid is not None:
        substrate.grid.width = 3
        substrate.grid.height = 3

    env = base_raw_configs.environment.model_copy(update={"enabled_affordances": []})
    population = base_raw_configs.population.model_copy(update={"num_agents": 10})

    mutated_raw = _clone_raw_configs(
        base_raw_configs, hamlet_overrides={"environment": env, "substrate": substrate, "population": population}
    )

    error = _run_stage4_expect_error(mutated_raw)

    # Should detect spatial impossibility: 9 cells < 10 required (0 affordances + 10 agents)
    assert any("UAC-VAL-001" in message for message in error.errors)
    assert any(issue.code == "UAC-VAL-001" for issue in error.issues)
