"""Smoke tests for RawConfigs loader."""

from pathlib import Path

from townlet.universe.compiler_inputs import RawConfigs


def test_raw_configs_from_config_dir():
    config_dir = Path("configs/L0_0_minimal")
    raw = RawConfigs.from_config_dir(config_dir)

    assert raw.training.max_episodes > 0
    assert raw.environment.grid_size > 0
    assert len(raw.variables_reference) > 0
    assert len(raw.global_actions.actions) > 0
