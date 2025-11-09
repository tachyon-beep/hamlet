"""Tests for load_variables_reference_config helper."""

from pathlib import Path

import pytest

from townlet.vfs.schema import load_variables_reference_config


def test_load_variables_reference_config_smoke(tmp_path: Path):
    """Loads variables_reference.yaml and returns VariableDef list."""
    yaml_content = """
variables:
  - id: energy
    scope: agent
    type: scalar
    lifetime: episode
    readable_by: ["agent"]
    writable_by: ["engine"]
    default: 1.0
"""
    config_dir = tmp_path
    (config_dir / "variables_reference.yaml").write_text(yaml_content)

    variables = load_variables_reference_config(config_dir)

    assert len(variables) == 1
    assert variables[0].id == "energy"
    assert variables[0].scope == "agent"


def test_load_variables_reference_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_variables_reference_config(tmp_path)


def test_load_variables_reference_missing_variables_key(tmp_path: Path):
    (tmp_path / "variables_reference.yaml").write_text("{}\n")
    with pytest.raises(ValueError):
        load_variables_reference_config(tmp_path)
