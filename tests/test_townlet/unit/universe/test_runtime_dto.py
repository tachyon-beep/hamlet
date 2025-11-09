"""Tests for runtime DTO view of compiled universes."""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


def test_compiled_universe_to_runtime_view() -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    runtime = compiled.to_runtime()

    assert runtime.metadata is compiled.metadata
    assert runtime.observation_spec is compiled.observation_spec
    assert runtime.optimization_data is compiled.optimization_data
    assert runtime.hamlet_config is not compiled.hamlet_config
    assert runtime.global_actions is not compiled.global_actions
    assert runtime.config_dir == compiled.config_dir
    assert runtime.meter_name_to_index == compiled.metadata.meter_name_to_index
    assert runtime.affordance_ids == compiled.metadata.affordance_ids
    assert isinstance(runtime.variables_reference, tuple)


def test_runtime_views_are_read_only() -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    runtime = compiled.to_runtime()

    with pytest.raises(AttributeError):
        runtime.hamlet_config.environment.grid_size = 10  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        runtime.global_actions.actions = []  # type: ignore[attr-defined]


def test_runtime_clone_helpers_return_mutable_copies() -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    runtime = compiled.to_runtime()

    env_copy = runtime.clone_environment_config()
    env_copy.grid_size = env_copy.grid_size + 1  # Should succeed without raising

    actions_copy = runtime.clone_global_actions()
    actions_copy.actions[0].enabled = False

    cascade_copy = runtime.clone_environment_cascade_config()
    cascade_copy.bars.bars[0].description = "updated"
