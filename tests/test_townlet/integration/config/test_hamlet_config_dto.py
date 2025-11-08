"""Integration tests for HamletConfig DTO (Master config composer).

These tests verify that HamletConfig correctly:
1. Composes all 5 section DTOs (training, environment, population, curriculum, exploration)
2. Performs cross-config validation (batch_size vs buffer_capacity, etc.)
3. Loads successfully from all production config packs
4. Provides clear error messages for validation failures
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports (integration tests may run standalone)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from tests.test_townlet.unit.config.fixtures import (
    PRODUCTION_CONFIG_PACKS,
    VALID_CURRICULUM_PARAMS,
    VALID_ENVIRONMENT_PARAMS,
    VALID_EXPLORATION_PARAMS,
    VALID_POPULATION_PARAMS,
    VALID_TRAINING_PARAMS,
    make_temp_config_pack,
)
from townlet.config import HamletConfig


class TestHamletConfigComposition:
    """Test that HamletConfig composes all section DTOs correctly."""

    def test_all_sections_required(self, tmp_path):
        """All 5 sections must be present in config."""
        import yaml

        config_dir = tmp_path / "incomplete_config"
        config_dir.mkdir()

        # Missing exploration section
        training_yaml = config_dir / "training.yaml"
        with open(training_yaml, "w") as f:
            yaml.dump(
                {
                    "training": VALID_TRAINING_PARAMS,
                    "environment": VALID_ENVIRONMENT_PARAMS,
                    "population": VALID_POPULATION_PARAMS,
                    "curriculum": VALID_CURRICULUM_PARAMS,
                    # Missing: exploration
                },
                f,
            )

        # Need dummy files for other DTOs
        for filename in ["bars.yaml", "cascades.yaml", "affordances.yaml"]:
            (config_dir / filename).write_text("bars: []\ncascades: []\naffordances: []\n")

        with pytest.raises(Exception) as exc_info:
            HamletConfig.load(config_dir)
        assert "exploration" in str(exc_info.value).lower()

    def test_load_complete_config(self, tmp_path):
        """Load complete config with all sections."""
        config_dir = make_temp_config_pack(tmp_path)

        config = HamletConfig.load(config_dir)

        # Verify all sections loaded
        assert config.training.device == VALID_TRAINING_PARAMS["device"]
        assert config.environment.grid_size == VALID_ENVIRONMENT_PARAMS["grid_size"]
        assert config.population.num_agents == VALID_POPULATION_PARAMS["num_agents"]
        assert config.curriculum.max_steps_per_episode == VALID_CURRICULUM_PARAMS["max_steps_per_episode"]
        assert config.exploration.embed_dim == VALID_EXPLORATION_PARAMS["embed_dim"]

    def test_config_sections_are_dtos_not_dicts(self, tmp_path):
        """Verify sections are DTO objects, not raw dicts."""
        from townlet.config.curriculum import CurriculumConfig
        from townlet.config.environment import TrainingEnvironmentConfig
        from townlet.config.exploration import ExplorationConfig
        from townlet.config.population import PopulationConfig
        from townlet.config.training import TrainingConfig

        config_dir = make_temp_config_pack(tmp_path)
        config = HamletConfig.load(config_dir)

        # Type checks
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.environment, TrainingEnvironmentConfig)
        assert isinstance(config.population, PopulationConfig)
        assert isinstance(config.curriculum, CurriculumConfig)
        assert isinstance(config.exploration, ExplorationConfig)

        # Should NOT be dicts
        assert not isinstance(config.training, dict)


class TestHamletConfigCrossValidation:
    """Test cross-config validation rules in HamletConfig."""

    def test_batch_size_must_not_exceed_buffer_capacity(self, tmp_path):
        """batch_size cannot exceed replay_buffer_capacity."""
        import yaml

        config_dir = tmp_path / "invalid_batch_size"
        config_dir.mkdir()

        # Create config with batch_size > buffer_capacity
        training_yaml = config_dir / "training.yaml"
        invalid_training = {**VALID_TRAINING_PARAMS, "batch_size": 10000}  # batch_size=10000
        invalid_population = {**VALID_POPULATION_PARAMS, "replay_buffer_capacity": 1000}  # buffer=1000

        with open(training_yaml, "w") as f:
            yaml.dump(
                {
                    "training": invalid_training,
                    "environment": VALID_ENVIRONMENT_PARAMS,
                    "population": invalid_population,
                    "curriculum": VALID_CURRICULUM_PARAMS,
                    "exploration": VALID_EXPLORATION_PARAMS,
                },
                f,
            )

        # Create dummy files
        for filename in ["bars.yaml", "cascades.yaml", "affordances.yaml", "cues.yaml"]:
            (config_dir / filename).write_text("bars: []\ncascades: []\naffordances: []\ncues: []\n")

        with pytest.raises(ValueError) as exc_info:
            HamletConfig.load(config_dir)

        error = str(exc_info.value)
        assert "batch_size" in error.lower()
        assert "replay_buffer_capacity" in error.lower() or "buffer" in error.lower()

    def test_batch_size_equal_to_buffer_capacity_allowed(self, tmp_path):
        """batch_size == replay_buffer_capacity is allowed (edge case)."""
        import yaml

        config_dir = tmp_path / "edge_case_batch"
        config_dir.mkdir()

        # Create config with batch_size == buffer_capacity
        training_yaml = config_dir / "training.yaml"
        edge_training = {**VALID_TRAINING_PARAMS, "batch_size": 5000}
        edge_population = {**VALID_POPULATION_PARAMS, "replay_buffer_capacity": 5000}

        with open(training_yaml, "w") as f:
            yaml.dump(
                {
                    "training": edge_training,
                    "environment": VALID_ENVIRONMENT_PARAMS,
                    "population": edge_population,
                    "curriculum": VALID_CURRICULUM_PARAMS,
                    "exploration": VALID_EXPLORATION_PARAMS,
                },
                f,
            )

        # Create dummy files
        for filename in ["bars.yaml", "cascades.yaml", "affordances.yaml", "cues.yaml"]:
            (config_dir / filename).write_text("bars: []\ncascades: []\naffordances: []\ncues: []\n")

        # Should NOT raise (batch_size == buffer_capacity is valid)
        config = HamletConfig.load(config_dir)
        assert config.training.batch_size == config.population.replay_buffer_capacity


class TestHamletConfigProductionPacks:
    """Test loading from all production config packs."""

    def test_load_L0_0_minimal(self):  # noqa: N802
        """Load L0_0_minimal config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L0_0_minimal"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)
        assert config.training.device in ["cpu", "cuda", "mps"]
        assert config.environment.grid_size == 3  # L0 is 3×3
        assert config.environment.enable_temporal_mechanics is False  # L0 has no temporal

    def test_load_L0_5_dual_resource(self):  # noqa: N802
        """Load L0_5_dual_resource config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L0_5_dual_resource"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)
        assert config.environment.grid_size == 7  # L0.5 is 7×7
        assert config.environment.enable_temporal_mechanics is False

    def test_load_L1_full_observability(self):  # noqa: N802
        """Load L1_full_observability config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L1_full_observability"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)
        assert config.environment.grid_size == 8  # L1 is 8×8
        assert config.environment.partial_observability is False  # L1 is full obs
        assert config.population.network_type == "simple"  # L1 uses MLP

    def test_load_L2_partial_observability(self):  # noqa: N802
        """Load L2_partial_observability config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L2_partial_observability"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)
        assert config.environment.grid_size == 8  # L2 is 8×8
        assert config.environment.partial_observability is True  # L2 is POMDP
        assert config.environment.vision_range == 2  # 5×5 window
        assert config.population.network_type == "recurrent"  # L2 uses LSTM

    def test_load_L3_temporal_mechanics(self):  # noqa: N802
        """Load L3_temporal_mechanics config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L3_temporal_mechanics"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = HamletConfig.load(config_dir)
        assert config.environment.grid_size == 8  # L3 is 8×8
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
            assert config.environment.grid_size > 0, f"{pack_name}: grid_size must be positive"
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

    def test_invalid_field_value_shows_field_name(self, tmp_path):
        """Error message includes field name for invalid values."""
        import yaml

        config_dir = tmp_path / "invalid_field"
        config_dir.mkdir()

        # Create config with invalid device
        training_yaml = config_dir / "training.yaml"
        invalid_training = {**VALID_TRAINING_PARAMS, "device": "invalid_device"}

        with open(training_yaml, "w") as f:
            yaml.dump(
                {
                    "training": invalid_training,
                    "environment": VALID_ENVIRONMENT_PARAMS,
                    "population": VALID_POPULATION_PARAMS,
                    "curriculum": VALID_CURRICULUM_PARAMS,
                    "exploration": VALID_EXPLORATION_PARAMS,
                },
                f,
            )

        # Create dummy files
        for filename in ["bars.yaml", "cascades.yaml", "affordances.yaml", "cues.yaml"]:
            (config_dir / filename).write_text("bars: []\ncascades: []\naffordances: []\ncues: []\n")

        with pytest.raises(ValueError) as exc_info:
            HamletConfig.load(config_dir)

        error = str(exc_info.value)
        assert "device" in error.lower()
        assert "training" in error.lower()
