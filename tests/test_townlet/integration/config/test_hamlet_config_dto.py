"""Integration tests for HamletConfig DTO (Master config composer).

These tests verify that HamletConfig correctly:
1. Composes all 5 section DTOs (training, environment, population, curriculum, exploration)
2. Performs cross-config validation (batch_size vs buffer_capacity, etc.)
3. Loads successfully from all production config packs
4. Provides clear error messages for validation failures
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

# Add src to path for imports (integration tests may run standalone)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from tests.test_townlet.helpers.config_builder import mutate_brain_yaml, mutate_training_yaml
from tests.test_townlet.unit.config.fixtures import (
    PRODUCTION_CONFIG_PACKS,
    VALID_CURRICULUM_PARAMS,
    VALID_ENVIRONMENT_PARAMS,
    VALID_EXPLORATION_PARAMS,
    VALID_POPULATION_PARAMS,
    VALID_TRAINING_PARAMS,
)
from townlet.config import HamletConfig
from townlet.universe.compiler_inputs import RawConfigs


def _apply_valid_sections(config_dir: Path) -> None:
    """Rewrite training.yaml with canonical fixture sections."""

    def mutator(data: dict) -> None:
        data["training"] = copy.deepcopy(VALID_TRAINING_PARAMS)
        data["environment"] = copy.deepcopy(VALID_ENVIRONMENT_PARAMS)
        data["population"] = copy.deepcopy(VALID_POPULATION_PARAMS)
        data["curriculum"] = copy.deepcopy(VALID_CURRICULUM_PARAMS)
        data["exploration"] = copy.deepcopy(VALID_EXPLORATION_PARAMS)

    mutate_training_yaml(config_dir, mutator)


class TestHamletConfigComposition:
    """Test that HamletConfig composes all section DTOs correctly."""

    def test_all_sections_required(self, config_pack_factory):
        """All 5 sections must be present in config."""
        config_dir = config_pack_factory()
        mutate_training_yaml(config_dir, lambda data: data.pop("exploration", None))

        with pytest.raises(Exception) as exc_info:
            HamletConfig.load(config_dir)
        assert "exploration" in str(exc_info.value).lower()

    def test_load_complete_config(self, config_pack_factory):
        """Load complete config with all sections."""
        config_dir = config_pack_factory()
        _apply_valid_sections(config_dir)

        config = HamletConfig.load(config_dir)

        # Verify all sections loaded
        assert config.training.device == VALID_TRAINING_PARAMS["device"]
        assert config.population.num_agents == VALID_POPULATION_PARAMS["num_agents"]
        assert config.curriculum.max_steps_per_episode == VALID_CURRICULUM_PARAMS["max_steps_per_episode"]
        assert config.exploration.embed_dim == VALID_EXPLORATION_PARAMS["embed_dim"]
        assert len(config.bars) > 0
        assert len(config.affordances) > 0
        assert config.substrate.type in {"grid", "gridnd", "continuous", "continuousnd", "aspatial"}
        assert config.cues is not None

    def test_config_sections_are_dtos_not_dicts(self, config_pack_factory):
        """Verify sections are DTO objects, not raw dicts."""
        from townlet.config.curriculum import CurriculumConfig
        from townlet.config.environment import TrainingEnvironmentConfig
        from townlet.config.exploration import ExplorationConfig
        from townlet.config.population import PopulationConfig
        from townlet.config.training import TrainingConfig

        config_dir = config_pack_factory()
        _apply_valid_sections(config_dir)
        config = HamletConfig.load(config_dir)

        # Type checks
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.environment, TrainingEnvironmentConfig)
        assert isinstance(config.population, PopulationConfig)
        assert isinstance(config.curriculum, CurriculumConfig)
        assert isinstance(config.exploration, ExplorationConfig)

        # Should NOT be dicts
        assert not isinstance(config.training, dict)

    def test_missing_cues_file_raises(self, config_pack_factory):
        """cues.yaml is required – missing file must raise."""
        config_dir = config_pack_factory()
        (config_dir / "cues.yaml").unlink()

        with pytest.raises(FileNotFoundError):
            HamletConfig.load(config_dir)

    def test_training_section_can_be_overridden_via_explicit_path(self, tmp_path, config_pack_factory):
        """training_config_path override must control training hyperparameters."""
        config_dir = config_pack_factory()
        _apply_valid_sections(config_dir)

        override_path = tmp_path / "custom_training.yaml"
        with open(config_dir / "training.yaml") as f:
            data = yaml.safe_load(f)

        data["training"]["max_episodes"] = 123
        data["training"]["epsilon_start"] = 0.5
        # Mutate environment on override file to ensure only training section is overridden
        data["environment"]["grid_size"] = 42

        with open(override_path, "w") as f:
            yaml.safe_dump(data, f)

        config = HamletConfig.load(config_dir, training_config_path=override_path)

        assert config.training.max_episodes == 123
        assert config.training.epsilon_start == 0.5
        # Environment comes from config_dir (not override file)
        assert config.environment.partial_observability == VALID_ENVIRONMENT_PARAMS["partial_observability"]


class TestHamletConfigCrossValidation:
    """Test cross-config validation rules in HamletConfig."""

    def test_batch_size_must_not_exceed_buffer_capacity(self, config_pack_factory):
        """batch_size cannot exceed replay_buffer_capacity."""
        config_dir = config_pack_factory()
        _apply_valid_sections(config_dir)

        # Set batch_size in training.yaml
        def training_mutator(data: dict) -> None:
            data["training"]["batch_size"] = 10000

        mutate_training_yaml(config_dir, training_mutator)

        # Set replay_buffer_capacity in brain.yaml (managed by brain.yaml now)
        def brain_mutator(data: dict) -> None:
            data["replay"]["capacity"] = 1000

        mutate_brain_yaml(config_dir, brain_mutator)

        with pytest.raises(ValueError) as exc_info:
            HamletConfig.load(config_dir)

        error = str(exc_info.value)
        assert "batch_size" in error.lower()
        assert "replay_buffer_capacity" in error.lower() or "buffer" in error.lower()

    def test_batch_size_equal_to_buffer_capacity_allowed(self, config_pack_factory):
        """batch_size == replay_buffer_capacity is allowed (edge case)."""

        from townlet.agent.brain_config import load_brain_config

        config_dir = config_pack_factory()
        _apply_valid_sections(config_dir)

        # Set batch_size in training.yaml
        def training_mutator(data: dict) -> None:
            data["training"]["batch_size"] = 5000

        mutate_training_yaml(config_dir, training_mutator)

        # Set replay_buffer_capacity in brain.yaml (managed by brain.yaml now)
        def brain_mutator(data: dict) -> None:
            data["replay"]["capacity"] = 5000

        mutate_brain_yaml(config_dir, brain_mutator)

        config = HamletConfig.load(config_dir)
        # replay_buffer_capacity is now managed by brain.yaml, not training.yaml
        brain_config = load_brain_config(config_dir)
        assert config.training.batch_size == brain_config.replay.capacity


class TestHamletConfigProductionPacks:
    """Test loading from all production config packs."""

    @pytest.mark.parametrize("pack_name", sorted(PRODUCTION_CONFIG_PACKS.keys()))
    def test_load_pack(self, pack_name):
        """Ensure each config pack compiles via HamletConfig."""
        config_dir = PRODUCTION_CONFIG_PACKS[pack_name]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)

        assert len(config.bars) > 0
        assert len(config.affordances) > 0
        assert config.substrate.type in {"grid", "gridnd", "continuous", "continuousnd", "aspatial"}

        if pack_name == "L1_full_observability":
            assert config.environment.partial_observability is False
        if pack_name == "L2_partial_observability":
            assert config.environment.partial_observability is True
        if pack_name == "L3_temporal_mechanics":
            assert config.environment.enable_temporal_mechanics is True
        if pack_name == "aspatial_test":
            assert config.substrate.type == "aspatial"
        if pack_name in {"L2_partial_observability", "L3_temporal_mechanics"}:
            assert config.population.network_type == "recurrent"
        else:
            assert config.population.network_type == "simple"

    def test_load_L2_partial_observability(self):  # noqa: N802
        """Load L2_partial_observability config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L2_partial_observability"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)
        assert config.environment.partial_observability is True  # L2 is POMDP
        assert config.environment.vision_range == 2  # 5×5 window
        assert config.population.network_type == "recurrent"  # L2 uses LSTM

    def test_load_L3_temporal_mechanics(self):  # noqa: N802
        """Load L3_temporal_mechanics config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L3_temporal_mechanics"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)
        assert config.environment.enable_temporal_mechanics is True  # L3 has temporal
        assert config.population.network_type == "recurrent"  # L3 uses LSTM

    def test_all_production_configs_load_successfully(self):
        """Verify all production config packs load without errors."""
        for pack_name, config_dir in PRODUCTION_CONFIG_PACKS.items():
            if not config_dir.exists():
                pytest.skip(f"Config pack not found: {config_dir}")

            # Should load without raising
            config = HamletConfig.load(config_dir)

            # Basic sanity checks
            assert config.training.max_episodes > 0, f"{pack_name}: max_episodes must be positive"
            assert config.population.num_agents > 0, f"{pack_name}: num_agents must be positive"
            assert config.curriculum.max_steps_per_episode > 0, f"{pack_name}: max_steps must be positive"
            assert config.exploration.embed_dim > 0, f"{pack_name}: embed_dim must be positive"


class TestHamletConfigErrorMessages:
    """Test that HamletConfig provides clear error messages."""

    def test_missing_training_yaml_file(self, tmp_path):
        """Clear error when training.yaml file is missing."""
        config_dir = tmp_path / "empty_config"
        config_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            HamletConfig.load(config_dir)

        error = str(exc_info.value)
        assert "training.yaml" in error.lower()

    def test_invalid_field_value_shows_field_name(self, config_pack_factory):
        """Error message includes field name for invalid values."""
        config_dir = config_pack_factory()
        mutate_training_yaml(config_dir, lambda data: data["training"].update({"device": "invalid_device"}))

        with pytest.raises(ValueError) as exc_info:
            HamletConfig.load(config_dir)

        error = str(exc_info.value)
        assert "device" in error.lower()
        assert "training" in error.lower()


class TestRawConfigsIntegration:
    """Ensure RawConfigs loads for all production packs."""

    @pytest.mark.parametrize("pack_name", sorted(PRODUCTION_CONFIG_PACKS.keys()))
    def test_raw_configs_loads(self, pack_name):
        config_dir = PRODUCTION_CONFIG_PACKS[pack_name]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        raw = RawConfigs.from_config_dir(config_dir)

        assert isinstance(raw.variables_reference, list)  # Empty list is valid (auto-generated variables only)
        assert len(raw.global_actions.actions) > 0
        assert raw.training.max_episodes > 0
