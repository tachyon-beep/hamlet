"""Test fixtures and utilities for configuration DTO tests.

This module provides reusable test data and utilities to avoid magic numbers
and ensure consistency across all config DTO tests.

Philosophy: Test data should be explicit, realistic, and reusable.
"""

from pathlib import Path
from typing import Any

# ==============================================================================
# PRODUCTION CONFIG PACK PATHS
# ==============================================================================

HAMLET_ROOT = Path(__file__).parent.parent.parent.parent.parent
CONFIGS_DIR = HAMLET_ROOT / "configs"

PRODUCTION_CONFIG_PACKS = {
    "L0_0_minimal": CONFIGS_DIR / "L0_0_minimal",
    "L0_5_dual_resource": CONFIGS_DIR / "L0_5_dual_resource",
    "L1_full_observability": CONFIGS_DIR / "L1_full_observability",
    "L2_partial_observability": CONFIGS_DIR / "L2_partial_observability",
    "L3_temporal_mechanics": CONFIGS_DIR / "L3_temporal_mechanics",
    "L1_3D_house": CONFIGS_DIR / "L1_3D_house",
    "L1_continuous_1D": CONFIGS_DIR / "L1_continuous_1D",
    "L1_continuous_2D": CONFIGS_DIR / "L1_continuous_2D",
    "L1_continuous_3D": CONFIGS_DIR / "L1_continuous_3D",
    "aspatial_test": CONFIGS_DIR / "aspatial_test",
    "test": CONFIGS_DIR / "test",
}

# ==============================================================================
# VALID DTO PARAMETERS (realistic values from production configs)
# ==============================================================================

VALID_TRAINING_PARAMS = {
    "device": "cuda",
    "max_episodes": 5000,
    "train_frequency": 4,
    "target_update_frequency": 100,
    "batch_size": 64,
    "max_grad_norm": 10.0,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "sequence_length": 8,
}

VALID_ENVIRONMENT_PARAMS = {
    "partial_observability": False,
    "vision_range": 8,
    "enable_temporal_mechanics": False,
    "enabled_affordances": None,  # None = all affordances
    "randomize_affordances": True,
    "energy_move_depletion": 0.005,
    "energy_wait_depletion": 0.001,
    "energy_interact_depletion": 0.0,
}

VALID_POPULATION_PARAMS = {
    "num_agents": 1,
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "replay_buffer_capacity": 10000,
    "network_type": "simple",
}

VALID_CURRICULUM_PARAMS = {
    "max_steps_per_episode": 500,
    "survival_advance_threshold": 0.7,
    "survival_retreat_threshold": 0.3,
    "entropy_gate": 0.5,
    "min_steps_at_stage": 1000,
}

VALID_EXPLORATION_PARAMS = {
    "embed_dim": 128,
    "initial_intrinsic_weight": 1.0,
    "variance_threshold": 100.0,
    "survival_window": 100,
    "min_survival_fraction": 0.5,
}

VALID_BAR_PARAMS = {
    "name": "energy",
    "index": 0,
    "tier": "pivotal",
    "range": [0.0, 1.0],
    "initial": 1.0,
    "base_depletion": 0.005,
}

VALID_CASCADE_PARAMS = {
    "name": "satiation_to_health",
    "description": "Starvation makes you sick",
    "source": "satiation",
    "target": "health",
    "threshold": 0.3,
    "strength": 0.004,
}

VALID_AFFORDANCE_PARAMS = {
    "id": "0",
    "name": "Bed",
    "costs": [{"meter": "money", "amount": 0.05}],
    "effects": [{"meter": "energy", "amount": 0.50}],
}

VALID_CUES_CONFIG = {
    "version": "1.0",
    "description": "Test cues",
    "status": "TEMPLATE",
    "simple_cues": [
        {
            "cue_id": "looks_tired",
            "name": "Looks Tired",
            "category": "energy",
            "visibility": "public",
            "condition": {"meter": "energy", "operator": "<", "threshold": 0.2},
        }
    ],
    "compound_cues": [],
}

# ==============================================================================
# INVALID PARAMETER VARIATIONS (for negative testing)
# ==============================================================================

# Device values
VALID_DEVICES = ["cpu", "cuda", "mps"]
INVALID_DEVICES = ["gpu", "invalid", ""]

# Epsilon constraints
EPSILON_MIN = 0.0
EPSILON_MAX = 1.0
EPSILON_DECAY_MIN_EXCLUSIVE = 0.0  # Must be > 0.0
EPSILON_DECAY_MAX_EXCLUSIVE = 1.0  # Must be < 1.0

# Network types
VALID_NETWORK_TYPES = ["simple", "recurrent"]
INVALID_NETWORK_TYPES = ["mlp", "lstm", "transformer"]

# Grid size
MIN_GRID_SIZE = 1
TYPICAL_GRID_SIZES = [3, 7, 8, 16]
LARGE_GRID_SIZE = 100

# Vision range
MIN_VISION_RANGE = 0
TYPICAL_VISION_RANGES = [0, 2, 4, 8]

# Thresholds (0.0 to 1.0)
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0
TYPICAL_THRESHOLDS = [0.3, 0.5, 0.7]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def make_valid_params(base_params: dict[str, Any], **overrides) -> dict[str, Any]:
    """Create valid parameters with optional overrides.

    Args:
        base_params: Base parameter dict (e.g., VALID_TRAINING_PARAMS)
        **overrides: Parameter overrides

    Returns:
        Merged parameter dict

    Example:
        >>> params = make_valid_params(VALID_TRAINING_PARAMS, device="cpu")
        >>> params["device"]
        'cpu'
        >>> params["max_episodes"]
        5000
    """
    return {**base_params, **overrides}


def make_temp_yaml(tmp_path: Path, section: str, data: dict[str, Any]) -> Path:
    """Create temporary YAML file for testing.

    Args:
        tmp_path: pytest tmp_path fixture
        section: Section name (e.g., "training", "bars")
        data: Data to write to section

    Returns:
        Path to created YAML file

    Example:
        >>> path = make_temp_yaml(tmp_path, "training", VALID_TRAINING_PARAMS)
        >>> # Creates tmp_path/training.yaml with training: {...}
    """
    import yaml

    yaml_path = tmp_path / f"{section}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({section: data}, f)
    return yaml_path


def make_temp_config_pack(tmp_path: Path) -> Path:
    """Create temporary config pack with all required files.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to config pack directory

    Example:
        >>> config_dir = make_temp_config_pack(tmp_path)
        >>> # Creates tmp_path/config_pack/ with training.yaml containing all sections
    """
    import yaml

    config_dir = tmp_path / "config_pack"
    config_dir.mkdir()

    # Create training.yaml with all sections
    training_yaml = config_dir / "training.yaml"
    with open(training_yaml, "w") as f:
        yaml.dump(
            {
                "training": VALID_TRAINING_PARAMS,
                "environment": VALID_ENVIRONMENT_PARAMS,
                "population": VALID_POPULATION_PARAMS,
                "curriculum": VALID_CURRICULUM_PARAMS,
                "exploration": VALID_EXPLORATION_PARAMS,
            },
            f,
        )

    # Create bars.yaml
    bars_yaml = config_dir / "bars.yaml"
    with open(bars_yaml, "w") as f:
        yaml.dump({"bars": [VALID_BAR_PARAMS]}, f)

    # Create cascades.yaml
    cascades_yaml = config_dir / "cascades.yaml"
    with open(cascades_yaml, "w") as f:
        yaml.dump({"cascades": [VALID_CASCADE_PARAMS]}, f)

    # Create affordances.yaml
    affordances_yaml = config_dir / "affordances.yaml"
    with open(affordances_yaml, "w") as f:
        yaml.dump({"affordances": [VALID_AFFORDANCE_PARAMS]}, f)

    # Create cues.yaml (optional but good for completeness)
    cues_yaml = config_dir / "cues.yaml"
    with open(cues_yaml, "w") as f:
        yaml.dump(VALID_CUES_CONFIG, f)

    return config_dir
