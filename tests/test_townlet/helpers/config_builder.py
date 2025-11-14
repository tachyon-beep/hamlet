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
    "drive_as_code.yaml",
    "brain.yaml",  # REQUIRED for all config packs
]
TEST_CONFIG_SRC = CONFIGS_DIR / "test"

BASE_CONFIG = {
    "environment": {
        # grid_size moved to substrate.yaml
        "partial_observability": False,
        "vision_range": 8,
        "enable_temporal_mechanics": False,
        "enabled_affordances": None,
        "randomize_affordances": True,
    },
    "population": {
        "num_agents": 1,
        "mask_unused_obs": False,
        # learning_rate, gamma, replay_buffer_capacity, network_type managed by brain.yaml
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
        "min_survival_fraction": 0.5,
    },
    "training": {
        "device": "cpu",
        "max_episodes": 5,
        "train_frequency": 4,
        "batch_size": 32,
        "sequence_length": 8,
        "max_grad_norm": 10.0,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "allow_unfeasible_universe": True,
        # target_update_frequency, use_double_dqn managed by brain.yaml
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


def mutate_brain_yaml(config_dir: Path, mutator: Callable[[dict], None]) -> None:
    """Load brain.yaml, apply mutator, and write back."""
    brain_yaml = config_dir / "brain.yaml"
    data = yaml.safe_load(brain_yaml.read_text())
    mutator(data)
    with open(brain_yaml, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def mutate_substrate_yaml(config_dir: Path, mutator: Callable[[dict], None]) -> None:
    """Load substrate.yaml, apply mutator, and write back."""
    substrate_yaml = config_dir / "substrate.yaml"
    data = yaml.safe_load(substrate_yaml.read_text())
    mutator(data)
    with open(substrate_yaml, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
