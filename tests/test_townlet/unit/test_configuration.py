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

from tests.test_townlet.helpers.config_builder import mutate_training_yaml
from tests.test_townlet.utils.builders import make_vectorized_env_from_pack
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
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.fixture(scope="module")
def bars_config(test_config_pack_path: Path) -> BarsConfig:
    """Load bars.yaml once per module."""

    return load_bars_config(test_config_pack_path / "bars.yaml")


@pytest.fixture(scope="module")
def cascades_config(test_config_pack_path: Path) -> CascadesConfig:
    """Load cascades.yaml once per module."""

    return load_cascades_config(test_config_pack_path / "cascades.yaml")


@pytest.fixture(scope="module")
def environment_config(test_config_pack_path: Path) -> EnvironmentConfig:
    """Load the combined environment config once."""

    return load_environment_config(test_config_pack_path)


@pytest.fixture(scope="module")
def affordance_collection(test_config_pack_path: Path, bars_config: BarsConfig) -> AffordanceConfigCollection:
    """Load affordances.yaml once; skip tests if file missing."""

    config_path = test_config_pack_path / "affordances.yaml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    return load_affordance_config(config_path, bars_config)


# =============================================================================
# CONFIG PACK LOADING TESTS
# =============================================================================


class TestConfigPackLoading:
    """Test loading complete config packs and pack-specific configurations.

    Consolidated from test_config_packs.py (2 tests).
    Tests that VectorizedHamletEnv correctly loads pack-specific bars.yaml,
    cascades.yaml, etc.
    """

    def test_vectorized_env_uses_pack_specific_bars(self, temp_config_pack: Path, cpu_device: torch.device):
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

        env = make_vectorized_env_from_pack(
            temp_config_pack,
            num_agents=1,
            device=cpu_device,
        )

        # Verify modified base depletion was loaded
        energy_base = env.meter_dynamics.get_base_depletion("energy")
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

    def test_bars_config_loads_successfully(self, bars_config: BarsConfig):
        """Test that bars.yaml validates successfully with Pydantic."""
        assert isinstance(bars_config, BarsConfig)
        assert bars_config.version == "1.0"
        assert len(bars_config.bars) == 8
        assert len(bars_config.terminal_conditions) == 2

    @pytest.mark.parametrize(
        ("meter_name", "expected_index"),
        [
            ("energy", 0),
            ("health", 6),
            ("satiation", 2),
            ("fitness", 7),
            ("mood", 4),
            ("hygiene", 1),
            ("social", 5),
            ("money", 3),
        ],
    )
    def test_bars_config_has_meter_indices(self, bars_config: BarsConfig, meter_name: str, expected_index: int):
        bar = next((bar for bar in bars_config.bars if bar.name == meter_name), None)
        assert bar is not None, f"{meter_name} missing from bars configuration"
        assert bar.index == expected_index

    @pytest.mark.parametrize(
        ("meter_name", "expected_depletion"),
        [
            ("energy", 0.005),
            ("hygiene", 0.003),
            ("satiation", 0.004),
            ("money", 0.0),
            ("mood", 0.001),
            ("social", 0.006),
            ("health", 0.0),
            ("fitness", 0.002),
        ],
    )
    def test_bars_config_base_depletions(self, bars_config: BarsConfig, meter_name: str, expected_depletion: float):
        bar = next(bar for bar in bars_config.bars if bar.name == meter_name)
        assert bar.base_depletion == expected_depletion, f"{meter_name}: expected {expected_depletion}"

    @pytest.mark.parametrize("meter_name", ["health", "energy"])
    def test_bars_config_terminal_conditions(self, bars_config: BarsConfig, meter_name: str):
        tc = next(tc for tc in bars_config.terminal_conditions if tc.meter == meter_name)
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

    def test_cascades_config_loads_successfully(self, cascades_config: CascadesConfig):
        """Test that cascades.yaml validates successfully with Pydantic."""
        assert isinstance(cascades_config, CascadesConfig)
        assert cascades_config.version == "1.0"
        assert cascades_config.math_type == "gradient_penalty"
        assert len(cascades_config.modulations) == 1
        assert len(cascades_config.cascades) == 10

    def test_cascades_config_validates_modulation(self, cascades_config: CascadesConfig):
        """Test that fitness-health modulation is configured correctly."""
        assert len(cascades_config.modulations) == 1
        mod = cascades_config.modulations[0]

        assert mod.name == "fitness_health_modulation"
        assert mod.source == "fitness"
        assert mod.target == "health"
        assert mod.type == "depletion_multiplier"
        assert mod.base_multiplier == 0.5
        assert mod.range == 2.5
        assert mod.baseline_depletion == 0.001

    @pytest.mark.parametrize(
        ("cascade_name", "expected_strength"),
        [
            ("satiation_to_health", 0.004),
            ("satiation_to_energy", 0.005),
            ("mood_to_energy", 0.005),
            ("hygiene_to_satiation", 0.002),
            ("hygiene_to_fitness", 0.002),
            ("hygiene_to_mood", 0.003),
            ("social_to_mood", 0.004),
            ("hygiene_to_health", 0.0005),
            ("hygiene_to_energy", 0.0005),
            ("social_to_energy", 0.0008),
        ],
    )
    def test_cascades_config_strengths(self, cascades_config: CascadesConfig, cascade_name: str, expected_strength: float):
        cascade = next(c for c in cascades_config.cascades if c.name == cascade_name)
        assert cascade.strength == expected_strength

    def test_cascades_config_validates_thresholds(self, cascades_config: CascadesConfig):
        """Test that all cascades use 30% threshold."""
        for cascade in cascades_config.cascades:
            assert cascade.threshold == 0.3, f"{cascade.name}: expected threshold=0.3, got {cascade.threshold}"

    def test_cascades_config_validates_execution_order(self, cascades_config: CascadesConfig):
        """Test that execution order is defined correctly."""
        expected_order = [
            "modulations",
            "primary_to_pivotal",
            "secondary_to_primary",
            "secondary_to_pivotal_weak",
        ]

        assert cascades_config.execution_order == expected_order

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

    def test_environment_config_loads_successfully(self, environment_config: EnvironmentConfig):
        """Test that complete environment config loads."""
        assert isinstance(environment_config, EnvironmentConfig)
        assert isinstance(environment_config.bars, BarsConfig)
        assert isinstance(environment_config.cascades, CascadesConfig)

    @pytest.mark.parametrize(
        ("meter_name", "expected_index"),
        [("energy", 0), ("health", 6)],
    )
    def test_environment_config_get_bar_by_name(self, environment_config: EnvironmentConfig, meter_name: str, expected_index: int):
        bar = environment_config.get_bar_by_name(meter_name)
        assert bar.name == meter_name
        assert bar.index == expected_index

    @pytest.mark.parametrize(
        ("bar_index", "expected_name"),
        [(0, "energy"), (6, "health")],
    )
    def test_environment_config_get_bar_by_index(self, environment_config: EnvironmentConfig, bar_index: int, expected_name: str):
        bar = environment_config.get_bar_by_index(bar_index)
        assert bar.name == expected_name

    def test_environment_config_get_bar_invalid(self, environment_config: EnvironmentConfig):
        with pytest.raises(ValueError, match="not found"):
            environment_config.get_bar_by_name("nonexistent")
        with pytest.raises(ValueError, match="not found"):
            environment_config.get_bar_by_index(99)

    def test_environment_config_get_cascade_by_name(self, environment_config: EnvironmentConfig):
        cascade = environment_config.get_cascade_by_name("satiation_to_health")
        assert cascade.source == "satiation"
        assert cascade.target == "health"
        with pytest.raises(ValueError, match="not found"):
            environment_config.get_cascade_by_name("nonexistent")

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
    """Test Pydantic schema validation for affordance configs."""

    @pytest.mark.parametrize(
        "payload, expectations",
        [
            (
                {
                    "id": "TestShower",
                    "name": "Test Shower",
                    "category": "hygiene",
                    "interaction_type": "instant",
                    "costs": [{"meter": "money", "amount": 0.01}],
                    "effects": [{"meter": "hygiene", "amount": 0.50}],
                    "operating_hours": [0, 24],
                },
                {"interaction_type": "instant", "effects_len": 1},
            ),
            (
                {
                    "id": "TestBed",
                    "name": "Test Bed",
                    "category": "energy",
                    "interaction_type": "multi_tick",
                    "duration_ticks": 5,
                    "costs_per_tick": [{"meter": "money", "amount": 0.01}],
                    "effects_per_tick": [{"meter": "energy", "amount": 0.075, "type": "linear"}],
                    "completion_bonus": [
                        {"meter": "energy", "amount": 0.125},
                        {"meter": "health", "amount": 0.02},
                    ],
                    "operating_hours": [0, 24],
                },
                {"interaction_type": "multi_tick", "duration_ticks": 5, "completion_bonus_len": 2},
            ),
        ],
        ids=["instant_minimal", "multi_tick_bonus"],
    )
    def test_valid_affordance_payloads(self, payload: dict, expectations: dict):
        affordance = AffordanceConfig(**payload)
        assert affordance.interaction_type == expectations["interaction_type"]
        if "duration_ticks" in expectations:
            assert affordance.duration_ticks == expectations["duration_ticks"]
        if "effects_len" in expectations:
            assert len(affordance.effects) == expectations["effects_len"]
        if "completion_bonus_len" in expectations:
            assert len(affordance.completion_bonus) == expectations["completion_bonus_len"]

    @pytest.mark.parametrize(
        ("payload", "expected_fields"),
        [
            (
                {
                    "id": "TestBroken",
                    "name": "Test Broken",
                },
                {"category", "interaction_type", "operating_hours"},
            ),
            (
                {
                    "id": "TestBad",
                    "name": "Test Bad",
                    "category": "test",
                    "interaction_type": "invalid_type",
                    "operating_hours": [0, 24],
                },
                {"interaction_type"},
            ),
            (
                {
                    "id": "TestBadMulti",
                    "name": "Test Bad Multi",
                    "category": "test",
                    "interaction_type": "multi_tick",
                    "operating_hours": [0, 24],
                },
                {"duration_ticks"},
            ),
        ],
        ids=["missing_required_fields", "invalid_interaction_type", "missing_duration_ticks"],
    )
    def test_invalid_affordance_payloads(self, payload: dict, expected_fields: set[str]):
        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(**payload)

        error_text = str(exc_info.value)
        for field in expected_fields:
            assert field in error_text

    def test_negative_cost_raises_error(self):
        with pytest.raises(ValidationError):
            AffordanceCost(meter="money", amount=-0.10)

    @pytest.mark.parametrize(
        ("meter", "amount"),
        [("energy", 1.0), ("fitness", -0.05)],
        ids=["large_positive", "negative_penalty"],
    )
    def test_effect_amount_bounds(self, meter: str, amount: float):
        effect = AffordanceEffect(meter=meter, amount=amount)
        assert effect.amount == amount

    @pytest.mark.parametrize(
        "hours",
        ([0, 24], [8, 18], [18, 28]),
        ids=["all_day", "business_hours", "wraparound"],
    )
    def test_operating_hours_validation(self, hours: list[int]):
        payload = {
            "id": "TestHours",
            "name": "Test Hours",
            "category": "test",
            "interaction_type": "instant",
            "operating_hours": hours,
        }
        affordance = AffordanceConfig(**payload)
        assert affordance.operating_hours == hours


class TestAffordanceConfigLoading:
    """Test loading affordance configs from YAML files."""

    def test_load_main_affordances_yaml(self, affordance_collection: AffordanceConfigCollection):
        assert affordance_collection.version == "2.0"
        assert affordance_collection.status == "PRODUCTION - Dual-mode ready"
        assert len(affordance_collection.affordances) == 14

    def test_loaded_affordances_have_valid_ids(self, affordance_collection: AffordanceConfigCollection):
        expected_ids = {str(i) for i in range(14)}
        actual_ids = {aff.id for aff in affordance_collection.affordances}
        assert actual_ids == expected_ids

    def test_affordance_lookup_by_id(self, affordance_collection: AffordanceConfigCollection):
        bed = affordance_collection.get_affordance("0")
        assert bed is not None
        assert bed.name == "Bed"
        assert bed.interaction_type == "dual"
        assert bed.duration_ticks == 5

        assert affordance_collection.get_affordance("NonExistent") is None

    def test_operating_hours_for_all_affordances(self, affordance_collection: AffordanceConfigCollection):
        for affordance in affordance_collection.affordances:
            assert len(affordance.operating_hours) == 2
            open_hour, close_hour = affordance.operating_hours
            assert 0 <= open_hour < 24
            assert 0 < close_hour <= 28


class TestAffordanceCategories:
    """Test affordance categorization and grouping."""

    def test_dual_mode_affordances(self, affordance_collection: AffordanceConfigCollection):
        dual = [aff for aff in affordance_collection.affordances if aff.interaction_type == "dual"]
        assert len(dual) == len(affordance_collection.affordances)

    def test_free_affordances(self, affordance_collection: AffordanceConfigCollection):
        free_affordances = {aff.name for aff in affordance_collection.affordances if not aff.costs}
        assert {"Park", "Job", "Labor"}.issubset(free_affordances)

    def test_affordances_with_penalties(self, affordance_collection: AffordanceConfigCollection):
        fastfood = affordance_collection.get_affordance("4")
        assert fastfood is not None
        penalties = [eff for eff in fastfood.effects if eff.amount < 0]
        penalties += [eff for eff in fastfood.completion_bonus if eff.amount < 0]
        assert len(penalties) >= 2

        bar = affordance_collection.get_affordance("9")
        assert bar is not None
        bar_penalties = [eff for eff in bar.effects if eff.amount < 0]
        bar_penalties += [eff for eff in bar.completion_bonus if eff.amount < 0]
        assert bar_penalties


class TestAffordanceConfigEdgeCases:
    """Test edge cases and error handling for affordance configs.

    Consolidated from test_affordance_config_loading.py (edge case tests).
    Tests missing files, empty affordances, duplicate IDs.
    """

    def test_load_nonexistent_file_raises_error(self, bars_config: BarsConfig):
        """Test that loading non-existent file raises appropriate error."""
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
    """Test epsilon parameter configuration from YAML."""

    @pytest.mark.parametrize(
        ("epsilon_start", "epsilon_decay", "epsilon_min"),
        [
            (0.8, 0.99, 0.05),
            (0.7, 0.98, 0.02),
        ],
        ids=["epsilon_greedy", "adaptive_intrinsic"],
    )
    def test_exploration_uses_config_values(self, epsilon_start, epsilon_decay, epsilon_min, cpu_device, basic_env):
        greedy = EpsilonGreedyExploration(
            epsilon=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )
        assert greedy.epsilon == epsilon_start
        assert greedy.epsilon_decay == epsilon_decay
        assert greedy.epsilon_min == epsilon_min

        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=100,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            device=cpu_device,
        )
        assert adaptive.rnd.epsilon == epsilon_start
        assert adaptive.rnd.epsilon_decay == epsilon_decay
        assert adaptive.rnd.epsilon_min == epsilon_min

    def test_runner_loads_epsilon_from_yaml(self, tmp_path: Path, config_pack_factory):
        config_dir = config_pack_factory(name="epsilon_config")

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

        temp_env = make_vectorized_env_from_pack(
            config_dir,
            num_agents=1,
            device=torch.device("cpu"),
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
    """Test training hyperparameter configuration from YAML."""

    @pytest.mark.parametrize(
        ("train_frequency", "target_update_frequency", "batch_size", "sequence_length", "max_grad_norm"),
        [(8, 200, 32, 16, 5.0)],
    )
    def test_population_uses_train_frequency_from_config(
        self,
        test_config_pack_path: Path,
        cpu_device: torch.device,
        train_frequency: int,
        target_update_frequency: int,
        batch_size: int,
        sequence_length: int,
        max_grad_norm: float,
    ):
        env = make_vectorized_env_from_pack(
            test_config_pack_path,
            num_agents=1,
            device=cpu_device,
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
            train_frequency=train_frequency,
            target_update_frequency=target_update_frequency,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_grad_norm=max_grad_norm,
        )

        assert population.train_frequency == train_frequency
        assert population.target_update_frequency == target_update_frequency
        assert population.batch_size == batch_size
        assert population.sequence_length == sequence_length
        assert population.max_grad_norm == max_grad_norm

    def test_runner_loads_training_hyperparameters_from_yaml(self, tmp_path: Path, config_pack_factory):
        config_dir = config_pack_factory(name="hyperparams_config")

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

        assert runner.config["training"]["train_frequency"] == 2
        assert runner.config["training"]["target_update_frequency"] == 50
        assert runner.config["training"]["batch_size"] == 128
        assert runner.config["training"]["sequence_length"] == 4
        assert runner.config["training"]["max_grad_norm"] == 15.0


class TestMaxEpisodesConfiguration:
    """Test max_episodes configuration precedence."""

    @pytest.mark.parametrize(
        ("explicit_override", "config_value", "expected"),
        [
            (1000, 500, 1000),
            (None, 500, 500),
        ],
        ids=["explicit_overrides_config", "reads_from_config"],
    )
    def test_max_episodes_precedence(self, tmp_path: Path, config_pack_factory, explicit_override, config_value, expected):
        config_dir = config_pack_factory(
            modifier=lambda data: data["training"].update({"max_episodes": config_value}),
        )

        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=explicit_override,
        )

        assert runner.max_episodes == expected

    def test_raises_error_when_max_episodes_missing(self, tmp_path: Path, config_pack_factory):
        """PDR-002: When max_episodes is missing, should raise clear error (no-defaults principle)."""
        config_dir = config_pack_factory(name="max_episodes_missing")

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
    """Test error handling for configuration loading."""

    @pytest.mark.parametrize(
        ("loader", "path"),
        [
            (load_bars_config, Path("/nonexistent/bars.yaml")),
            (load_cascades_config, Path("/nonexistent/cascades.yaml")),
            (load_environment_config, Path("/nonexistent/configs")),
        ],
        ids=["bars_missing", "cascades_missing", "environment_dir_missing"],
    )
    def test_loaders_raise_error_for_missing_resources(self, loader, path):
        with pytest.raises(FileNotFoundError):
            loader(path)
