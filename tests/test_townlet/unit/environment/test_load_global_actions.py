"""Tests for load_global_actions_config."""

from pathlib import Path

import pytest

from townlet.environment.action_config import ActionSpaceConfig, load_global_actions_config


def test_load_global_actions_config(tmp_path: Path):
    yaml_content = """
custom_actions:
  - name: REST
    type: passive
    costs: {energy: -0.01}
    effects: {}
  - name: MEDITATE
    type: passive
    costs: {energy: 0.001}
    effects: {mood: 0.02}
"""
    yaml_path = tmp_path / "global_actions.yaml"
    yaml_path.write_text(yaml_content)

    space = load_global_actions_config(yaml_path)

    assert isinstance(space, ActionSpaceConfig)
    assert [a.name for a in space.actions] == ["REST", "MEDITATE"]


def test_load_global_actions_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_global_actions_config(tmp_path / "missing.yaml")


def test_load_global_actions_requires_list(tmp_path: Path):
    yaml_path = tmp_path / "global_actions.yaml"
    yaml_path.write_text("{}\n")
    with pytest.raises(ValueError):
        load_global_actions_config(yaml_path)
