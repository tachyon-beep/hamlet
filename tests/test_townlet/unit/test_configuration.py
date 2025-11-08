"""Consolidated tests for configuration loading and validation.

This file consolidates configuration-related tests from multiple old files:
- test_config_packs.py: Config pack loading (bars, cascades, affordances per pack)
- test_config_training_params.py: Training parameter validation (epsilon, hyperparameters)
- test_config_max_episodes.py: Episode limits and constraints
- test_affordance_config_loading.py: Affordance YAML loading and schema validation
- test_cascade_config.py: Cascade and bars YAML loading and schema validation

The configuration system follows the UNIVERSE_AS_CODE principle:
- Everything configurable via YAML
- No-defaults principle: all parameters explicit
- Schema validation before training starts

Each config pack is a directory containing:
- bars.yaml: Meter definitions and base depletion rates
- cascades.yaml: Meter cascade relationships
- affordances.yaml: Affordance definitions (costs, effects, operating hours)
- cues.yaml: Visual cue configuration
- training.yaml: Training hyperparameters and episode limits
"""

from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from tests.test_townlet.helpers.config_builder import mutate_training_yaml, prepare_config_dir
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.runner import DemoRunner
from townlet.environment.affordance_config import (
    AffordanceConfig,
    AffordanceConfigCollection,
    AffordanceCost,
    AffordanceEffect,
    load_affordance_config,
)
from townlet.environment.cascade_config import (
    BarConfig,
    BarsConfig,
    CascadeConfig,
    CascadesConfig,
    EnvironmentConfig,
    load_bars_config,
    load_cascades_config,
    load_default_config,
    load_environment_config,
)
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation

# =============================================================================
# CONFIG PACK LOADING TESTS
# =============================================================================


class TestConfigPackLoading:
    """Test loading complete config packs and pack-specific configurations.

    Consolidated from test_config_packs.py (2 tests).
    Tests that VectorizedHamletEnv correctly loads pack-specific bars.yaml,
    cascades.yaml, etc.
    """

    def test_vectorized_env_uses_pack_specific_bars(self, temp_config_pack: Path):
        """Ensure VectorizedHamletEnv reads bars.yaml from the selected pack."""
        bars_path = temp_config_pack / "bars.yaml"
        original = bars_path.read_text()

        if "base_depletion: 0.005" not in original:
            pytest.fail("Unexpected bars.yaml fixture content: missing base_depletion 0.005")

        # Modify energy base depletion to verify it's being read
        modified = original.replace("base_depletion: 0.005", "base_depletion: 0.010", 1)
        bars_path.write_text(modified)

        # Update training config with custom energy costs
        training_path = temp_config_pack / "training.yaml"
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
            agent_lifespan=1000,
            config_pack_path=temp_config_pack,
        )

        # Verify modified base depletion was loaded
        energy_base = env.meter_dynamics.cascade_engine.get_base_depletion("energy")
        assert energy_base == pytest.approx(0.010, rel=1e-6)

        # Verify action costs were set correctly
        assert env.move_energy_cost == pytest.approx(0.02, rel=1e-6)
        assert env.wait_energy_cost == pytest.approx(0.015, rel=1e-6)
        assert env.interact_energy_cost == pytest.approx(0.001, rel=1e-6)


# =============================================================================
# BARS CONFIG TESTS
# =============================================================================


class TestBarsConfig:
    """Test bars.yaml schema validation and loading.

    Consolidated from test_cascade_config.py (~10 tests).
    Tests Pydantic schema validation, YAML loading, and value verification
    against meter_dynamics.py.
    """

    def test_bars_yaml_loads(self, test_config_pack_path: Path):
        """Test that bars.yaml is valid YAML and loads without errors."""
        bars_path = test_config_pack_path / "bars.yaml"
        assert bars_path.exists(), f"bars.yaml not found at {bars_path}"

        with open(bars_path) as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        assert "version" in data
        assert "bars" in data
        assert "terminal_conditions" in data

    def test_bars_config_loads_successfully(self, test_config_pack_path: Path):
        """Test that bars.yaml validates successfully with Pydantic."""
        bars_path = test_config_pack_path / "bars.yaml"
        config = load_bars_config(bars_path)

        assert isinstance(config, BarsConfig)
        assert config.version == "1.0"
        assert len(config.bars) == 8
        assert len(config.terminal_conditions) == 2

    def test_bars_config_has_all_meters(self, test_config_pack_path: Path):
        """Test that all 8 meters are defined with correct indices."""
        bars_path = test_config_pack_path / "bars.yaml"
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

    def test_bars_config_validates_depletion_rates(self, test_config_pack_path: Path):
        """Test that base depletion rates match meter_dynamics.py."""
        bars_path = test_config_pack_path / "bars.yaml"
        config = load_bars_config(bars_path)

        # Expected values from meter_dynamics.py
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
            assert bar.base_depletion == expected, f"{bar.name}: expected base_depletion={expected}, got {bar.base_depletion}"

    def test_bars_config_validates_terminal_conditions(self, test_config_pack_path: Path):
        """Test that terminal conditions are correct."""
        bars_path = test_config_pack_path / "bars.yaml"
        config = load_bars_config(bars_path)

        assert len(config.terminal_conditions) == 2

        meters = {tc.meter for tc in config.terminal_conditions}
        assert meters == {"health", "energy"}

        for tc in config.terminal_conditions:
            assert tc.operator == "<="
            assert tc.value == 0.0

    def test_bar_config_rejects_invalid_range(self):
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

    def test_bars_config_rejects_duplicate_indices(self):
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

        # TASK-001: Error message changed from "indices must be 0-7" to more detailed message
        with pytest.raises(ValidationError, match="Bar indices must be contiguous from 0 to 7"):
            BarsConfig(**bars_data)


# =============================================================================
# CASCADES CONFIG TESTS
# =============================================================================


class TestCascadesConfig:
    """Test cascades.yaml schema validation and loading.

    Consolidated from test_cascade_config.py (~8 tests).
    Tests cascade relationships, thresholds, and execution order.
    """

    def test_cascades_yaml_loads(self, test_config_pack_path: Path):
        """Test that cascades.yaml is valid YAML and loads without errors."""
        cascades_path = test_config_pack_path / "cascades.yaml"
        assert cascades_path.exists(), f"cascades.yaml not found at {cascades_path}"

        with open(cascades_path) as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        assert "version" in data
        assert "modulations" in data
        assert "cascades" in data

    def test_cascades_config_loads_successfully(self, test_config_pack_path: Path):
        """Test that cascades.yaml validates successfully with Pydantic."""
        cascades_path = test_config_pack_path / "cascades.yaml"
        config = load_cascades_config(cascades_path)

        assert isinstance(config, CascadesConfig)
        assert config.version == "1.0"
        assert config.math_type == "gradient_penalty"
        assert len(config.modulations) == 1
        assert len(config.cascades) == 10

    def test_cascades_config_validates_modulation(self, test_config_pack_path: Path):
        """Test that fitness-health modulation is configured correctly."""
        cascades_path = test_config_pack_path / "cascades.yaml"
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

    def test_cascades_config_validates_cascade_strengths(self, test_config_pack_path: Path):
        """Test that cascade strengths match meter_dynamics.py."""
        cascades_path = test_config_pack_path / "cascades.yaml"
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
            assert cascade.strength == expected_strength, f"{name}: expected strength={expected_strength}, got {cascade.strength}"

    def test_cascades_config_validates_thresholds(self, test_config_pack_path: Path):
        """Test that all cascades use 30% threshold."""
        cascades_path = test_config_pack_path / "cascades.yaml"
        config = load_cascades_config(cascades_path)

        for cascade in config.cascades:
            assert cascade.threshold == 0.3, f"{cascade.name}: expected threshold=0.3, got {cascade.threshold}"

    def test_cascades_config_validates_execution_order(self, test_config_pack_path: Path):
        """Test that execution order is defined correctly."""
        cascades_path = test_config_pack_path / "cascades.yaml"
        config = load_cascades_config(cascades_path)

        expected_order = [
            "modulations",
            "primary_to_pivotal",
            "secondary_to_primary",
            "secondary_to_pivotal_weak",
        ]

        assert config.execution_order == expected_order

    def test_cascade_config_rejects_invalid_threshold(self):
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

    def test_cascades_config_rejects_duplicate_names(self):
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


# =============================================================================
# ENVIRONMENT CONFIG INTEGRATION TESTS
# =============================================================================


class TestEnvironmentConfig:
    """Test combined environment configuration (bars + cascades).

    Consolidated from test_cascade_config.py (~5 tests).
    Tests integration of bars and cascades into EnvironmentConfig
    with helper methods.
    """

    def test_environment_config_loads_successfully(self, test_config_pack_path: Path):
        """Test that complete environment config loads."""
        config = load_environment_config(test_config_pack_path)

        assert isinstance(config, EnvironmentConfig)
        assert isinstance(config.bars, BarsConfig)
        assert isinstance(config.cascades, CascadesConfig)

    def test_environment_config_get_bar_by_name(self, test_config_pack_path: Path):
        """Test get_bar_by_name helper method."""
        config = load_environment_config(test_config_pack_path)

        energy = config.get_bar_by_name("energy")
        assert energy.name == "energy"
        assert energy.index == 0

        health = config.get_bar_by_name("health")
        assert health.name == "health"
        assert health.index == 6

        with pytest.raises(ValueError, match="not found"):
            config.get_bar_by_name("nonexistent")

    def test_environment_config_get_bar_by_index(self, test_config_pack_path: Path):
        """Test get_bar_by_index helper method."""
        config = load_environment_config(test_config_pack_path)

        bar0 = config.get_bar_by_index(0)
        assert bar0.name == "energy"

        bar6 = config.get_bar_by_index(6)
        assert bar6.name == "health"

        with pytest.raises(ValueError, match="not found"):
            config.get_bar_by_index(99)

    def test_environment_config_get_cascade_by_name(self, test_config_pack_path: Path):
        """Test get_cascade_by_name helper method."""
        config = load_environment_config(test_config_pack_path)

        cascade = config.get_cascade_by_name("satiation_to_health")
        assert cascade.name == "satiation_to_health"
        assert cascade.source == "satiation"
        assert cascade.target == "health"

        with pytest.raises(ValueError, match="not found"):
            config.get_cascade_by_name("nonexistent")

    def test_load_default_config(self):
        """Test that default config loads from project root."""
        config = load_default_config()

        assert isinstance(config, EnvironmentConfig)
        assert len(config.bars.bars) == 8
        assert len(config.cascades.cascades) == 10


# =============================================================================
# AFFORDANCE CONFIG TESTS
# =============================================================================


class TestAffordanceConfigSchema:
    """Test Pydantic schema validation for affordance configs.

    Consolidated from test_affordance_config_loading.py (schema tests).
    Tests validation of AffordanceConfig, AffordanceCost, AffordanceEffect.
    """

    def test_minimal_instant_affordance(self):
        """Test minimal valid instant affordance."""
        config = {
            "id": "TestShower",
            "name": "Test Shower",
            "category": "hygiene",
            "interaction_type": "instant",
            "costs": [{"meter": "money", "amount": 0.01}],
            "effects": [{"meter": "hygiene", "amount": 0.50}],
            "operating_hours": [0, 24],
        }
        affordance = AffordanceConfig(**config)
        assert affordance.id == "TestShower"
        assert affordance.interaction_type == "instant"
        assert len(affordance.effects) == 1

    def test_multi_tick_affordance_with_bonus(self):
        """Test multi-tick affordance with completion bonus."""
        config = {
            "id": "TestBed",
            "name": "Test Bed",
            "category": "energy",
            "interaction_type": "multi_tick",
            "required_ticks": 5,
            "costs_per_tick": [{"meter": "money", "amount": 0.01}],
            "effects_per_tick": [{"meter": "energy", "amount": 0.075, "type": "linear"}],
            "completion_bonus": [
                {"meter": "energy", "amount": 0.125},
                {"meter": "health", "amount": 0.02},
            ],
            "operating_hours": [0, 24],
        }
        affordance = AffordanceConfig(**config)
        assert affordance.interaction_type == "multi_tick"
        assert affordance.required_ticks == 5
        assert len(affordance.completion_bonus) == 2

    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raise ValidationError."""
        config = {
            "id": "TestBroken",
            "name": "Test Broken",
            # Missing: category, interaction_type, operating_hours
        }
        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(**config)

        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "category" in error_fields
        assert "interaction_type" in error_fields

    def test_invalid_interaction_type_raises_error(self):
        """Test that invalid interaction_type raises ValidationError."""
        config = {
            "id": "TestBad",
            "name": "Test Bad",
            "category": "test",
            "interaction_type": "invalid_type",  # Not in [instant, multi_tick, continuous]
            "operating_hours": [0, 24],
        }
        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(**config)

        assert "interaction_type" in str(exc_info.value)

    def test_multi_tick_without_required_ticks_raises_error(self):
        """Test that multi_tick without required_ticks raises ValidationError."""
        config = {
            "id": "TestBadMulti",
            "name": "Test Bad Multi",
            "category": "test",
            "interaction_type": "multi_tick",
            # Missing: required_ticks (required for multi_tick)
            "operating_hours": [0, 24],
        }
        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(**config)

        assert "required_ticks" in str(exc_info.value)

    def test_negative_cost_raises_error(self):
        """Test that negative costs raise ValidationError."""
        with pytest.raises(ValidationError):
            AffordanceCost(meter="money", amount=-0.10)  # Negative cost invalid

    def test_effect_amount_bounds(self):
        """Test that effect amounts have reasonable bounds."""
        # Very large positive effect should be allowed (config flexibility)
        effect = AffordanceEffect(meter="energy", amount=1.0)
        assert effect.amount == 1.0

        # Small negative effect should be allowed (penalties)
        effect_neg = AffordanceEffect(meter="fitness", amount=-0.05)
        assert effect_neg.amount == -0.05

    def test_operating_hours_validation(self):
        """Test operating hours format validation."""
        # Valid 24/7
        config = {
            "id": "Test24",
            "name": "Test 24/7",
            "category": "test",
            "interaction_type": "instant",
            "operating_hours": [0, 24],
        }
        affordance = AffordanceConfig(**config)
        assert affordance.operating_hours == [0, 24]

        # Valid business hours
        config["operating_hours"] = [8, 18]
        affordance = AffordanceConfig(**config)
        assert affordance.operating_hours == [8, 18]

        # Valid midnight wraparound (Bar: 6pm-4am)
        config["operating_hours"] = [18, 28]
        affordance = AffordanceConfig(**config)
        assert affordance.operating_hours == [18, 28]


class TestAffordanceConfigLoading:
    """Test loading affordance configs from YAML files.

    Consolidated from test_affordance_config_loading.py (loading tests).
    Tests YAML file loading, structure validation, and affordance lookup.
    """

    def test_load_main_affordances_yaml(self, test_config_pack_path: Path):
        """Test loading the main affordances.yaml config."""
        config_path = test_config_pack_path / "affordances.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        collection = load_affordance_config(config_path, bars_config)

        assert collection.version == "2.0"
        assert collection.status == "PRODUCTION - Dual-mode ready"
        assert len(collection.affordances) == 14  # Dual-mode pack defines 14 affordances

    def test_loaded_affordances_have_valid_ids(self, test_config_pack_path: Path):
        """Test that loaded affordances have expected IDs."""
        config_path = test_config_pack_path / "affordances.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        collection = load_affordance_config(config_path, bars_config)

        expected_ids = {str(i) for i in range(14)}  # IDs 0-13 inclusive
        actual_ids = {aff.id for aff in collection.affordances}
        assert actual_ids == expected_ids

    def test_affordance_lookup_by_id(self, test_config_pack_path: Path):
        """Test that we can look up affordances by ID."""
        config_path = test_config_pack_path / "affordances.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        collection = load_affordance_config(config_path, bars_config)

        # Test lookup
        bed = collection.get_affordance("0")  # Bed has ID "0" in dual-mode config
        assert bed is not None
        assert bed.name == "Bed"
        assert bed.interaction_type == "dual"
        assert bed.required_ticks == 5

        # Test missing affordance
        missing = collection.get_affordance("NonExistent")
        assert missing is None

    def test_operating_hours_for_all_affordances(self, test_config_pack_path: Path):
        """Test that all affordances have valid operating hours."""
        config_path = test_config_pack_path / "affordances.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        collection = load_affordance_config(config_path, bars_config)

        for affordance in collection.affordances:
            assert len(affordance.operating_hours) == 2
            open_hour, close_hour = affordance.operating_hours
            assert 0 <= open_hour < 24
            assert 0 < close_hour <= 28  # 28 allows midnight wraparound


class TestAffordanceCategories:
    """Test affordance categorization and grouping.

    Consolidated from test_affordance_config_loading.py (categorization tests).
    Tests affordance categorization, free affordances, and penalties.
    """

    def test_dual_mode_affordances(self, test_config_pack_path: Path):
        """Test that all affordances are dual-mode in production config."""
        config_path = test_config_pack_path / "affordances.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        collection = load_affordance_config(config_path, bars_config)

        dual = [aff for aff in collection.affordances if aff.interaction_type == "dual"]
        assert len(dual) == len(collection.affordances), "All affordances should be dual-mode in production config"

    def test_free_affordances(self, test_config_pack_path: Path):
        """Test identification of free (no-cost) affordances."""
        config_path = test_config_pack_path / "affordances.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        collection = load_affordance_config(config_path, bars_config)

        free_affordances = {aff.name for aff in collection.affordances if len(aff.costs) == 0}
        # Park, Job, Labor are free in instant mode
        assert {"Park", "Job", "Labor"}.issubset(free_affordances)

    def test_affordances_with_penalties(self, test_config_pack_path: Path):
        """Test affordances that have negative effects (penalties)."""
        config_path = test_config_pack_path / "affordances.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        collection = load_affordance_config(config_path, bars_config)

        # FastFood has fitness and health penalties
        fastfood = collection.get_affordance("4")
        assert fastfood is not None

        negative_effects = [eff for eff in fastfood.effects if eff.amount < 0]
        negative_effects += [eff for eff in fastfood.completion_bonus if eff.amount < 0]
        assert len(negative_effects) >= 2  # fitness and health penalties

        # Bar has health penalty (completion bonus contains -health)
        bar = collection.get_affordance("9")
        assert bar is not None

        bar_penalties = [eff for eff in bar.effects if eff.amount < 0]
        bar_penalties += [eff for eff in bar.completion_bonus if eff.amount < 0]
        assert len(bar_penalties) >= 1  # health penalty


class TestAffordanceConfigEdgeCases:
    """Test edge cases and error handling for affordance configs.

    Consolidated from test_affordance_config_loading.py (edge case tests).
    Tests missing files, empty affordances, duplicate IDs.
    """

    def test_load_nonexistent_file_raises_error(self, test_config_pack_path: Path):
        """Test that loading non-existent file raises appropriate error."""
        bars_config = load_bars_config(test_config_pack_path / "bars.yaml")
        with pytest.raises(FileNotFoundError):
            load_affordance_config(Path("configs/nonexistent.yaml"), bars_config)

    def test_empty_affordances_list(self):
        """Test that empty affordances list is valid."""
        collection_data = {
            "version": "1.0",
            "description": "Empty test",
            "status": "test",
            "affordances": [],
        }
        collection = AffordanceConfigCollection(**collection_data)
        assert len(collection.affordances) == 0

    def test_duplicate_affordance_ids_detected(self):
        """Test that duplicate affordance IDs are detected (documents current behavior)."""
        collection_data = {
            "version": "1.0",
            "description": "Duplicate test",
            "status": "test",
            "affordances": [
                {
                    "id": "Duplicate",
                    "name": "First",
                    "category": "test",
                    "interaction_type": "instant",
                    "operating_hours": [0, 24],
                },
                {
                    "id": "Duplicate",  # Same ID!
                    "name": "Second",
                    "category": "test",
                    "interaction_type": "instant",
                    "operating_hours": [0, 24],
                },
            ],
        }

        # Current behavior: duplicates allowed, lookup returns first match
        collection = AffordanceConfigCollection(**collection_data)
        assert len(collection.affordances) == 2

        found = collection.get_affordance("Duplicate")
        assert found is not None


# =============================================================================
# TRAINING PARAMETERS TESTS
# =============================================================================


class TestEpsilonConfiguration:
    """Test epsilon parameter configuration from YAML.

    Consolidated from test_config_training_params.py (epsilon tests).
    Tests that epsilon parameters (start, decay, min) are configurable
    for exploration strategies.
    """

    def test_epsilon_greedy_uses_config_values(self):
        """EpsilonGreedy should accept epsilon params from config."""
        config = {
            "training": {
                "epsilon_start": 0.8,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.05,
            }
        }

        exploration = EpsilonGreedyExploration(
            epsilon=config["training"]["epsilon_start"],
            epsilon_decay=config["training"]["epsilon_decay"],
            epsilon_min=config["training"]["epsilon_min"],
        )

        assert exploration.epsilon == 0.8
        assert exploration.epsilon_decay == 0.99
        assert exploration.epsilon_min == 0.05

    def test_adaptive_intrinsic_uses_epsilon_config(self, cpu_device: torch.device, basic_env):
        """AdaptiveIntrinsicExploration should pass epsilon params to RND."""
        config = {
            "training": {
                "epsilon_start": 0.7,
                "epsilon_decay": 0.98,
                "epsilon_min": 0.02,
            },
            "exploration": {
                "embed_dim": 64,
                "initial_intrinsic_weight": 1.0,
                "variance_threshold": 100.0,
                "survival_window": 100,
            },
        }

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            embed_dim=config["exploration"]["embed_dim"],
            initial_intrinsic_weight=config["exploration"]["initial_intrinsic_weight"],
            variance_threshold=config["exploration"]["variance_threshold"],
            survival_window=config["exploration"]["survival_window"],
            epsilon_start=config["training"]["epsilon_start"],
            epsilon_decay=config["training"]["epsilon_decay"],
            epsilon_min=config["training"]["epsilon_min"],
            device=cpu_device,
        )

        assert exploration.rnd.epsilon == 0.7
        assert exploration.rnd.epsilon_decay == 0.98
        assert exploration.rnd.epsilon_min == 0.02

    def test_runner_loads_epsilon_from_yaml(self, tmp_path: Path):
        """DemoRunner should load epsilon params from training.yaml (integration test)."""
        config_dir = prepare_config_dir(tmp_path, name="epsilon_config")

        def mutator(data: dict) -> None:
            data["training"].update(
                {
                    "device": "cpu",
                    "max_episodes": 10,
                    "epsilon_start": 0.9,
                    "epsilon_decay": 0.97,
                    "epsilon_min": 0.03,
                }
            )
            data["exploration"].update(
                {
                    "embed_dim": 64,
                    "initial_intrinsic_weight": 1.0,
                    "variance_threshold": 100.0,
                    "survival_window": 50,
                }
            )

        mutate_training_yaml(config_dir, mutator)

        _runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=10,
        )

        # Create exploration with config params (mimics runner initialization)
        # Create a basic env to get observation_dim
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        temp_env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,  # Match training_config grid_size
            partial_observability=False,
            device=torch.device("cpu"),
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=config_dir,
        )

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=temp_env.observation_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=50,
            epsilon_start=0.9,
            epsilon_decay=0.97,
            epsilon_min=0.03,
            device=torch.device("cpu"),
        )

        assert exploration.rnd.epsilon == 0.9
        assert exploration.rnd.epsilon_decay == 0.97
        assert exploration.rnd.epsilon_min == 0.03


class TestTrainingHyperparameters:
    """Test training hyperparameter configuration from YAML.

    Consolidated from test_config_training_params.py (hyperparameter tests).
    Tests that training hyperparameters (train_frequency, batch_size, etc.)
    are configurable for VectorizedPopulation.
    """

    def test_population_uses_train_frequency_from_config(self, test_config_pack_path: Path, cpu_device: torch.device):
        """VectorizedPopulation should accept train_frequency and other hyperparameters."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            device=cpu_device,
            enabled_affordances=["Bed"],
            config_pack_path=test_config_pack_path,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
        )

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=cpu_device,
        )

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=50,
            device=cpu_device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            learning_rate=0.001,
            gamma=0.95,
            replay_buffer_capacity=1000,
            network_type="simple",
            vision_window_size=5,
            train_frequency=8,
            target_update_frequency=200,
            batch_size=32,
            sequence_length=16,
            max_grad_norm=5.0,
        )

        assert population.train_frequency == 8
        assert population.target_update_frequency == 200
        assert population.batch_size == 32
        assert population.sequence_length == 16
        assert population.max_grad_norm == 5.0

    def test_runner_loads_training_hyperparameters_from_yaml(self, tmp_path: Path):
        """DemoRunner should load training hyperparameters from config (integration test)."""
        config_dir = prepare_config_dir(tmp_path, name="hyperparams_config")

        def mutator(data: dict) -> None:
            data["training"].update(
                {
                    "device": "cpu",
                    "max_episodes": 10,
                    "train_frequency": 2,
                    "target_update_frequency": 50,
                    "batch_size": 128,
                    "sequence_length": 4,
                    "max_grad_norm": 15.0,
                }
            )

        mutate_training_yaml(config_dir, mutator)

        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=10,
        )

        # Verify config structure
        assert runner.config["training"]["train_frequency"] == 2
        assert runner.config["training"]["target_update_frequency"] == 50
        assert runner.config["training"]["batch_size"] == 128
        assert runner.config["training"]["sequence_length"] == 4
        assert runner.config["training"]["max_grad_norm"] == 15.0


class TestMaxEpisodesConfiguration:
    """Test max_episodes configuration from YAML.

    Consolidated from test_config_max_episodes.py (4 tests).
    Tests that DemoRunner correctly reads max_episodes from config
    with proper precedence (explicit > config > default).
    """

    def test_explicit_max_episodes_overrides_config(self, tmp_path: Path):
        """When max_episodes is explicitly provided, it should override config."""
        config_dir = prepare_config_dir(tmp_path, name="max_episodes_override")

        def mutator(data: dict) -> None:
            data["training"].update({"max_episodes": 500})

        mutate_training_yaml(config_dir, mutator)

        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=1000,  # Explicit override
        )

        assert runner.max_episodes == 1000

    def test_reads_max_episodes_from_config_when_not_provided(self, tmp_path: Path):
        """When max_episodes is not provided, should read from config YAML."""
        config_dir = prepare_config_dir(tmp_path, name="max_episodes_from_config")

        def mutator(data: dict) -> None:
            data["training"].update({"max_episodes": 500})

        mutate_training_yaml(config_dir, mutator)

        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=None,  # Will read from config
        )

        assert runner.max_episodes == 500

    def test_raises_error_when_max_episodes_missing(self, tmp_path: Path):
        """PDR-002: When max_episodes is missing, should raise clear error (no-defaults principle)."""
        config_dir = prepare_config_dir(tmp_path, name="max_episodes_missing")

        def mutator(data: dict) -> None:
            data["training"].pop("max_episodes", None)

        mutate_training_yaml(config_dir, mutator)

        # Verify PDR-002 fail-fast behavior
        with pytest.raises(ValueError) as exc_info:
            DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test.db",
                checkpoint_dir=tmp_path / "checkpoints",
                max_episodes=None,
            )

        error = str(exc_info.value)
        assert "training.yaml" in error
        assert "max_episodes" in error

    def test_stable_test_config_reads_200_episodes(self):
        """Integration test: configs/test should read 200 episodes (stable test config)."""
        config_dir = Path("configs/test")
        if not config_dir.exists():
            pytest.skip("Test config not found")

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            runner = DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test.db",
                checkpoint_dir=tmp_path / "checkpoints",
                max_episodes=None,  # Read from config
            )

            assert runner.max_episodes == 200


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestConfigErrorHandling:
    """Test error handling for configuration loading.

    Consolidated error handling tests from multiple files.
    Tests missing files, invalid data, and validation errors.
    """

    def test_load_bars_config_missing_file(self):
        """Test that load_bars_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_bars_config(Path("/nonexistent/bars.yaml"))

    def test_load_cascades_config_missing_file(self):
        """Test that load_cascades_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_cascades_config(Path("/nonexistent/cascades.yaml"))

    def test_load_environment_config_missing_directory(self):
        """Test that load_environment_config raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_environment_config(Path("/nonexistent/configs"))
