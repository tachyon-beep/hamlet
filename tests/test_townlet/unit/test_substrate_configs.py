"""Unit tests for substrate.yaml config files."""

from pathlib import Path

import pytest

from townlet.substrate.config import Grid2DSubstrateConfig, load_substrate_config


@pytest.mark.parametrize(
    "config_name,expected_width,expected_height,expected_obs_grid_dim",
    [
        ("L0_0_minimal", 3, 3, 9),  # 3×3 = 9
        ("L0_5_dual_resource", 7, 7, 49),  # 7×7 = 49
        ("L1_full_observability", 8, 8, 64),  # 8×8 = 64
        ("L2_partial_observability", 8, 8, 64),  # 8×8 = 64
        ("L3_temporal_mechanics", 8, 8, 64),  # 8×8 = 64
        ("templates", 8, 8, 64),  # 8×8 = 64
        ("test", 8, 8, 64),  # 8×8 = 64
    ],
)
def test_substrate_config_schema_valid(config_name, expected_width, expected_height, expected_obs_grid_dim):
    """Substrate config should load and pass schema validation."""
    config_path = Path("configs") / config_name / "substrate.yaml"

    # Load config (will raise ValidationError if schema invalid)
    config = load_substrate_config(config_path)

    # Verify basic structure
    assert config.version == "1.0"
    assert config.type == "grid"
    assert config.description  # Non-empty description required

    # Verify grid configuration
    assert isinstance(config.grid, Grid2DSubstrateConfig)
    assert config.grid.topology == "square"
    assert config.grid.width == expected_width
    assert config.grid.height == expected_height
    assert config.grid.boundary == "clamp"
    assert config.grid.distance_metric == "manhattan"

    # Verify observation dimension calculation
    # obs_dim = grid_size + 8 meters + 15 affordances + 4 temporal
    obs_grid_dim = config.grid.width * config.grid.height
    assert obs_grid_dim == expected_obs_grid_dim


@pytest.mark.parametrize(
    "config_name",
    [
        "L0_0_minimal",
        "L0_5_dual_resource",
        "L1_full_observability",
        "L2_partial_observability",
        "L3_temporal_mechanics",
        "templates",
        "test",
    ],
)
def test_substrate_config_behavioral_equivalence(config_name):
    """Substrate config should produce identical behavior to legacy hardcoded grid."""
    config_path = Path("configs") / config_name / "substrate.yaml"
    config = load_substrate_config(config_path)

    # All current configs use same spatial behavior (only size differs)
    assert config.grid.topology == "square"  # Standard 2D grid
    assert config.grid.boundary == "clamp"  # Hard walls (not wrap/bounce/sticky)
    assert config.grid.distance_metric == "manhattan"  # L1 norm (not euclidean)

    # Grid must be square (width == height) for current configs
    assert config.grid.width == config.grid.height


def test_substrate_config_no_defaults():
    """Substrate config should require all fields (no-defaults principle)."""

    # Attempt to load incomplete config (missing required fields)
    incomplete_yaml = """
version: "1.0"
type: "grid"
# Missing: description, grid section
"""
    incomplete_path = Path("/tmp/incomplete_substrate.yaml")
    incomplete_path.write_text(incomplete_yaml)

    # Should raise ValidationError (not fall back to defaults)
    with pytest.raises(Exception) as exc_info:
        load_substrate_config(incomplete_path)

    # Error message should mention missing field
    assert "description" in str(exc_info.value).lower()

    # Cleanup
    incomplete_path.unlink()


def test_substrate_config_file_exists():
    """All production config packs should have substrate.yaml."""
    production_configs = [
        "L0_0_minimal",
        "L0_5_dual_resource",
        "L1_full_observability",
        "L2_partial_observability",
        "L3_temporal_mechanics",
        "test",
    ]

    for config_name in production_configs:
        substrate_path = Path("configs") / config_name / "substrate.yaml"
        assert substrate_path.exists(), f"Missing substrate.yaml for {config_name}"
