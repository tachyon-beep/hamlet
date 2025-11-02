"""Tests for configuration pack loading (affordances/bars/cascades per pack)."""

import shutil
from pathlib import Path

import pytest
import yaml

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def test_pack_dir(tmp_path: Path) -> Path:
    """Return a writable copy of the default test config pack."""
    source_pack = Path("configs/test")
    target_pack = tmp_path / "custom_pack"
    shutil.copytree(source_pack, target_pack)
    return target_pack


def test_vectorized_env_uses_pack_specific_bars(test_pack_dir: Path):
    """Ensure VectorizedHamletEnv reads bars.yaml from the selected pack."""
    bars_path = test_pack_dir / "bars.yaml"
    original = bars_path.read_text()

    if "base_depletion: 0.005" not in original:
        pytest.fail("Unexpected bars.yaml fixture content: missing base_depletion 0.005")

    modified = original.replace("base_depletion: 0.005", "base_depletion: 0.010", 1)
    bars_path.write_text(modified)

    # Update training config with custom energy costs
    training_path = test_pack_dir / "training.yaml"
    training_config = yaml.safe_load(training_path.read_text())
    env_cfg = training_config.setdefault("environment", {})
    env_cfg["energy_move_depletion"] = 0.02
    env_cfg["energy_wait_depletion"] = 0.015
    env_cfg["energy_interact_depletion"] = 0.001
    training_path.write_text(yaml.safe_dump(training_config))

    env_cfg_update = training_config["environment"]
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=env_cfg_update.get("grid_size", 8),
        partial_observability=env_cfg_update.get("partial_observability", False),
        vision_range=env_cfg_update.get("vision_range", 2),
        enable_temporal_mechanics=env_cfg_update.get("enable_temporal_mechanics", False),
        move_energy_cost=env_cfg_update["energy_move_depletion"],
        wait_energy_cost=env_cfg_update["energy_wait_depletion"],
        interact_energy_cost=env_cfg_update["energy_interact_depletion"],
        config_pack_path=test_pack_dir,
    )

    energy_base = env.meter_dynamics.cascade_engine.get_base_depletion("energy")
    assert energy_base == pytest.approx(0.010, rel=1e-6)

    assert env.move_energy_cost == pytest.approx(0.02, rel=1e-6)
    assert env.wait_energy_cost == pytest.approx(0.015, rel=1e-6)
    assert env.interact_energy_cost == pytest.approx(0.001, rel=1e-6)

    min_cost = min(env.move_energy_cost, env.wait_energy_cost, env.interact_energy_cost)
    expected_baseline = 1.0 / (energy_base + min_cost)
    assert env.calculate_baseline_survival() == pytest.approx(expected_baseline, rel=1e-6)


def test_baseline_uses_minimum_action_cost(test_pack_dir: Path):
    """Baseline should always be computed with the cheapest per-step action cost."""
    training_path = test_pack_dir / "training.yaml"
    training_config = yaml.safe_load(training_path.read_text())

    env_cfg = training_config.setdefault("environment", {})
    env_cfg["energy_move_depletion"] = 0.02
    env_cfg["energy_wait_depletion"] = 0.012  # Must remain less than move cost
    env_cfg["energy_interact_depletion"] = 0.05  # Highest cost
    training_path.write_text(yaml.safe_dump(training_config))

    env_cfg_update = training_config["environment"]
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=env_cfg_update.get("grid_size", 8),
        partial_observability=env_cfg_update.get("partial_observability", False),
        vision_range=env_cfg_update.get("vision_range", 2),
        enable_temporal_mechanics=env_cfg_update.get("enable_temporal_mechanics", False),
        move_energy_cost=env_cfg_update["energy_move_depletion"],
        wait_energy_cost=env_cfg_update["energy_wait_depletion"],
        interact_energy_cost=env_cfg_update["energy_interact_depletion"],
        config_pack_path=test_pack_dir,
    )

    energy_base = env.meter_dynamics.cascade_engine.get_base_depletion("energy")
    min_cost = min(env.move_energy_cost, env.wait_energy_cost, env.interact_energy_cost)
    assert min_cost == env.wait_energy_cost  # WAIT is the cheapest action in this scenario

    expected_baseline = 1.0 / (energy_base + min_cost)
    assert env.calculate_baseline_survival() == pytest.approx(expected_baseline, rel=1e-6)
