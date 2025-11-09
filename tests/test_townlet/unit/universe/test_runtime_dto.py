"""Tests for runtime DTO view of compiled universes."""

from __future__ import annotations

from pathlib import Path

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
