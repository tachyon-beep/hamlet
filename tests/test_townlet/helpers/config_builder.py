"""Shared helpers for building temporary config packs in tests."""

from __future__ import annotations

import copy
import shutil
from collections.abc import Callable
from pathlib import Path

import yaml

from tests.test_townlet.unit.config.fixtures import CONFIGS_DIR

SUPPORT_FILES = [
    "bars.yaml",
    "cascades.yaml",
    "affordances.yaml",
    "cues.yaml",
    "substrate.yaml",
    "variables_reference.yaml",
]
TEST_CONFIG_SRC = CONFIGS_DIR / "test"

BASE_CONFIG = {
    "environment": {
        "grid_size": 8,
        "partial_observability": False,
        "vision_range": 8,
        "enable_temporal_mechanics": False,
        "enabled_affordances": None,
        "randomize_affordances": True,
        "energy_move_depletion": 0.005,
        "energy_wait_depletion": 0.001,
        "energy_interact_depletion": 0.0,
    },
    "population": {
        "num_agents": 1,
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "replay_buffer_capacity": 1000,
        "network_type": "simple",
    },
    "curriculum": {
        "max_steps_per_episode": 50,
        "survival_advance_threshold": 0.7,
        "survival_retreat_threshold": 0.3,
        "entropy_gate": 0.5,
        "min_steps_at_stage": 10,
    },
    "exploration": {
        "embed_dim": 128,
        "initial_intrinsic_weight": 1.0,
        "variance_threshold": 100.0,
        "survival_window": 100,
    },
    "training": {
        "device": "cpu",
        "max_episodes": 5,
        "train_frequency": 4,
        "target_update_frequency": 100,
        "batch_size": 32,
        "sequence_length": 8,
        "max_grad_norm": 10.0,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
    },
}


def copy_support_files(dest_dir: Path) -> None:
    """Copy required YAML config files into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in SUPPORT_FILES:
        src = TEST_CONFIG_SRC / filename
        if not src.exists():
            raise FileNotFoundError(f"Required test config file '{filename}' missing in {TEST_CONFIG_SRC}")
        shutil.copy(src, dest_dir / filename)


def build_training_config() -> dict:
    """Return deep copy of base training configuration."""
    return copy.deepcopy(BASE_CONFIG)


def write_training_yaml(config_dir: Path, config_data: dict) -> None:
    """Write training.yaml using the provided data."""
    with open(config_dir / "training.yaml", "w") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)


def prepare_config_dir(tmp_path: Path, modifier: Callable[[dict], None] | None = None, name: str = "config") -> Path:
    """Prepare a full config pack in tmp_path with optional modifier."""
    config_dir = tmp_path / name
    config_dir.mkdir()
    config_data = build_training_config()
    if modifier is not None:
        modifier(config_data)
    write_training_yaml(config_dir, config_data)
    copy_support_files(config_dir)
    return config_dir


def mutate_training_yaml(config_dir: Path, mutator: Callable[[dict], None]) -> None:
    """Load training.yaml, apply mutator, and write back."""
    training_yaml = config_dir / "training.yaml"
    data = yaml.safe_load(training_yaml.read_text())
    mutator(data)
    write_training_yaml(config_dir, data)
