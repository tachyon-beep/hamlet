"""
Tests for cascade configuration loading and validation.

Tests:
1. YAML file loading (syntax, structure)
2. Pydantic validation (types, constraints)
3. Value verification (match meter_dynamics.py)
4. Helper methods (get_bar_by_name, etc.)
5. Error handling (missing files, invalid data)
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from townlet.environment.cascade_config import (
    BarConfig,
    BarsConfig,
    CascadeConfig,
    CascadesConfig,
    EnvironmentConfig,
    ModulationConfig,
    TerminalCondition,
    load_bars_config,
    load_cascades_config,
    load_default_config,
    load_environment_config,
)


@pytest.fixture
def config_dir() -> Path:
    """Get path to configs directory."""
    return Path(__file__).parent.parent.parent / "configs"


@pytest.fixture
def bars_path(config_dir: Path) -> Path:
    """Get path to bars.yaml."""
    return config_dir / "bars.yaml"


@pytest.fixture
def cascades_path(config_dir: Path) -> Path:
    """Get path to cascades.yaml."""
    return config_dir / "cascades.yaml"


# ============================================================================
# YAML File Loading Tests
# ============================================================================


def test_bars_yaml_loads(bars_path: Path):
    """Test that bars.yaml is valid YAML and loads without errors."""
    assert bars_path.exists(), f"bars.yaml not found at {bars_path}"

    with open(bars_path) as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict)
    assert "version" in data
    assert "bars" in data
    assert "terminal_conditions" in data


def test_cascades_yaml_loads(cascades_path: Path):
    """Test that cascades.yaml is valid YAML and loads without errors."""
    assert cascades_path.exists(), f"cascades.yaml not found at {cascades_path}"

    with open(cascades_path) as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict)
    assert "version" in data
    assert "modulations" in data
    assert "cascades" in data


# ============================================================================
# BarsConfig Validation Tests
# ============================================================================


def test_bars_config_loads_successfully(bars_path: Path):
    """Test that bars.yaml validates successfully with Pydantic."""
    config = load_bars_config(bars_path)

    assert isinstance(config, BarsConfig)
    assert config.version == "1.0"
    assert len(config.bars) == 8
    assert len(config.terminal_conditions) == 2


def test_bars_config_has_all_meters(bars_path: Path):
    """Test that all 8 meters are defined with correct indices."""
    config = load_bars_config(bars_path)

    meter_names = {bar.name for bar in config.bars}
    expected_names = {
        "energy",
        "health",
        "satiation",
        "fitness",
        "mood",
        "hygiene",
        "social",
        "money",
    }

    assert meter_names == expected_names, f"Expected {expected_names}, got {meter_names}"

    indices = {bar.index for bar in config.bars}
    assert indices == {0, 1, 2, 3, 4, 5, 6, 7}, f"Expected indices 0-7, got {indices}"


def test_bars_config_validates_depletion_rates(bars_path: Path):
    """Test that base depletion rates match meter_dynamics.py."""
    config = load_bars_config(bars_path)

    # Expected values from meter_dynamics.py line 43-52
    expected_depletions = {
        "energy": 0.005,
        "hygiene": 0.003,
        "satiation": 0.004,
        "money": 0.0,
        "mood": 0.001,
        "social": 0.006,
        "health": 0.0,  # No base depletion - handled by fitness modulation
        "fitness": 0.002,
    }

    for bar in config.bars:
        expected = expected_depletions[bar.name]
        assert bar.base_depletion == expected, (
            f"{bar.name}: expected base_depletion={expected}, got {bar.base_depletion}"
        )


def test_bars_config_validates_terminal_conditions(bars_path: Path):
    """Test that terminal conditions are correct."""
    config = load_bars_config(bars_path)

    assert len(config.terminal_conditions) == 2

    meters = {tc.meter for tc in config.terminal_conditions}
    assert meters == {"health", "energy"}

    for tc in config.terminal_conditions:
        assert tc.operator == "<="
        assert tc.value == 0.0


def test_bar_config_rejects_invalid_range():
    """Test that BarConfig rejects non-standard ranges."""
    with pytest.raises(ValidationError):
        BarConfig(
            name="test",
            index=0,
            tier="pivotal",
            range=(0.0, 100.0),  # Invalid - must be (0.0, 1.0)
            initial=1.0,
            base_depletion=0.01,
            description="Test bar",
        )


def test_bars_config_rejects_duplicate_indices():
    """Test that BarsConfig rejects duplicate indices."""
    bars_data = {
        "version": "1.0",
        "description": "Test",
        "bars": [
            {
                "name": "bar1",
                "index": 0,
                "tier": "pivotal",
                "range": [0.0, 1.0],
                "initial": 1.0,
                "base_depletion": 0.01,
                "description": "Bar 1",
            },
            {
                "name": "bar2",
                "index": 0,  # Duplicate!
                "tier": "pivotal",
                "range": [0.0, 1.0],
                "initial": 1.0,
                "base_depletion": 0.01,
                "description": "Bar 2",
            },
        ]
        + [
            {
                "name": f"bar{i}",
                "index": i,
                "tier": "pivotal",
                "range": [0.0, 1.0],
                "initial": 1.0,
                "base_depletion": 0.01,
                "description": f"Bar {i}",
            }
            for i in range(2, 8)
        ],
        "terminal_conditions": [],
    }

    with pytest.raises(ValidationError, match="indices must be 0-7"):
        BarsConfig(**bars_data)


# ============================================================================
# CascadesConfig Validation Tests
# ============================================================================


def test_cascades_config_loads_successfully(cascades_path: Path):
    """Test that cascades.yaml validates successfully with Pydantic."""
    config = load_cascades_config(cascades_path)

    assert isinstance(config, CascadesConfig)
    assert config.version == "1.0"
    assert config.math_type == "gradient_penalty"
    assert len(config.modulations) == 1
    assert len(config.cascades) == 10


def test_cascades_config_validates_modulation(cascades_path: Path):
    """Test that fitness-health modulation is configured correctly."""
    config = load_cascades_config(cascades_path)

    assert len(config.modulations) == 1
    mod = config.modulations[0]

    assert mod.name == "fitness_health_modulation"
    assert mod.source == "fitness"
    assert mod.target == "health"
    assert mod.type == "depletion_multiplier"
    assert mod.base_multiplier == 0.5
    assert mod.range == 2.5
    assert mod.baseline_depletion == 0.001


def test_cascades_config_validates_cascade_strengths(cascades_path: Path):
    """Test that cascade strengths match meter_dynamics.py."""
    config = load_cascades_config(cascades_path)

    # Expected values from meter_dynamics.py
    expected_strengths = {
        "satiation_to_health": 0.004,
        "satiation_to_energy": 0.005,
        "mood_to_energy": 0.005,
        "hygiene_to_satiation": 0.002,
        "hygiene_to_fitness": 0.002,
        "hygiene_to_mood": 0.003,
        "social_to_mood": 0.004,
        "hygiene_to_health": 0.0005,
        "hygiene_to_energy": 0.0005,
        "social_to_energy": 0.0008,
    }

    cascades_by_name = {c.name: c for c in config.cascades}

    for name, expected_strength in expected_strengths.items():
        cascade = cascades_by_name[name]
        assert cascade.strength == expected_strength, (
            f"{name}: expected strength={expected_strength}, got {cascade.strength}"
        )


def test_cascades_config_validates_thresholds(cascades_path: Path):
    """Test that all cascades use 30% threshold."""
    config = load_cascades_config(cascades_path)

    for cascade in config.cascades:
        assert cascade.threshold == 0.3, (
            f"{cascade.name}: expected threshold=0.3, got {cascade.threshold}"
        )


def test_cascades_config_validates_execution_order(cascades_path: Path):
    """Test that execution order is defined correctly."""
    config = load_cascades_config(cascades_path)

    expected_order = [
        "modulations",
        "primary_to_pivotal",
        "secondary_to_primary",
        "secondary_to_pivotal_weak",
    ]

    assert config.execution_order == expected_order


def test_cascade_config_rejects_invalid_threshold():
    """Test that CascadeConfig rejects invalid thresholds."""
    with pytest.raises(ValidationError):
        CascadeConfig(
            name="test",
            description="Test",
            category="test",
            source="energy",
            source_index=0,
            target="health",
            target_index=6,
            threshold=1.5,  # Invalid - must be <= 1.0
            strength=0.01,
        )


def test_cascades_config_rejects_duplicate_names():
    """Test that CascadesConfig rejects duplicate cascade names."""
    cascades_data = {
        "version": "1.0",
        "description": "Test",
        "math_type": "gradient_penalty",
        "modulations": [],
        "cascades": [
            {
                "name": "cascade1",
                "description": "Test 1",
                "category": "test",
                "source": "energy",
                "source_index": 0,
                "target": "health",
                "target_index": 6,
                "threshold": 0.3,
                "strength": 0.01,
            },
            {
                "name": "cascade1",  # Duplicate!
                "description": "Test 2",
                "category": "test",
                "source": "mood",
                "source_index": 4,
                "target": "energy",
                "target_index": 0,
                "threshold": 0.3,
                "strength": 0.01,
            },
        ],
        "execution_order": [],
    }

    with pytest.raises(ValidationError, match="Duplicate cascade names"):
        CascadesConfig(**cascades_data)


# ============================================================================
# EnvironmentConfig Integration Tests
# ============================================================================


def test_environment_config_loads_successfully(config_dir: Path):
    """Test that complete environment config loads."""
    config = load_environment_config(config_dir)

    assert isinstance(config, EnvironmentConfig)
    assert isinstance(config.bars, BarsConfig)
    assert isinstance(config.cascades, CascadesConfig)


def test_environment_config_get_bar_by_name(config_dir: Path):
    """Test get_bar_by_name helper method."""
    config = load_environment_config(config_dir)

    energy = config.get_bar_by_name("energy")
    assert energy.name == "energy"
    assert energy.index == 0

    health = config.get_bar_by_name("health")
    assert health.name == "health"
    assert health.index == 6

    with pytest.raises(ValueError, match="not found"):
        config.get_bar_by_name("nonexistent")


def test_environment_config_get_bar_by_index(config_dir: Path):
    """Test get_bar_by_index helper method."""
    config = load_environment_config(config_dir)

    bar0 = config.get_bar_by_index(0)
    assert bar0.name == "energy"

    bar6 = config.get_bar_by_index(6)
    assert bar6.name == "health"

    with pytest.raises(ValueError, match="not found"):
        config.get_bar_by_index(99)


def test_environment_config_get_cascade_by_name(config_dir: Path):
    """Test get_cascade_by_name helper method."""
    config = load_environment_config(config_dir)

    cascade = config.get_cascade_by_name("satiation_to_health")
    assert cascade.name == "satiation_to_health"
    assert cascade.source == "satiation"
    assert cascade.target == "health"

    with pytest.raises(ValueError, match="not found"):
        config.get_cascade_by_name("nonexistent")


def test_load_default_config():
    """Test that default config loads from project root."""
    config = load_default_config()

    assert isinstance(config, EnvironmentConfig)
    assert len(config.bars.bars) == 8
    assert len(config.cascades.cascades) == 10


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_load_bars_config_missing_file():
    """Test that load_bars_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_bars_config(Path("/nonexistent/bars.yaml"))


def test_load_cascades_config_missing_file():
    """Test that load_cascades_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_cascades_config(Path("/nonexistent/cascades.yaml"))


def test_load_environment_config_missing_directory():
    """Test that load_environment_config raises FileNotFoundError for missing directory."""
    with pytest.raises(FileNotFoundError):
        load_environment_config(Path("/nonexistent/configs"))
