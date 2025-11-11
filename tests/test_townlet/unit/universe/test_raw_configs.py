"""Smoke tests for RawConfigs loader."""

import shutil
from pathlib import Path

import yaml

from townlet.universe.compiler_inputs import RawConfigs


def test_raw_configs_from_config_dir():
    config_dir = Path("configs/L0_0_minimal")
    raw = RawConfigs.from_config_dir(config_dir)

    assert raw.training.max_episodes > 0
    # grid_size is now in substrate.yaml, not environment.yaml
    assert raw.substrate.grid is not None
    assert raw.substrate.grid.width > 0
    # variables_reference can be empty list (auto-generated only)
    assert raw.variables_reference is not None
    assert len(raw.global_actions.actions) > 0


def test_raw_configs_respects_training_enabled_actions(tmp_path):
    source = Path("configs/L0_0_minimal")
    target = tmp_path / "mask_pack"
    shutil.copytree(source, target)

    training_path = target / "training.yaml"
    data = yaml.safe_load(training_path.read_text())
    data.setdefault("training", {})["enabled_actions"] = ["INTERACT", "WAIT"]
    training_path.write_text(yaml.safe_dump(data, sort_keys=False))

    raw = RawConfigs.from_config_dir(target)

    enabled_names = {action.name for action in raw.global_actions.actions if action.enabled}
    assert enabled_names == {"INTERACT", "WAIT"}

    disabled_names = {action.name for action in raw.global_actions.actions if not action.enabled}
    assert "UP" in disabled_names  # Movement disabled via config
