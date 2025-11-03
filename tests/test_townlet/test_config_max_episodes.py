"""Test that max_episodes is correctly read from config YAML."""

import pytest
import yaml
from pathlib import Path

from townlet.demo.runner import DemoRunner


class TestMaxEpisodesFromConfig:
    """Test that DemoRunner reads max_episodes from config when not explicitly provided."""

    def test_explicit_max_episodes_overrides_config(self, tmp_path):
        """When max_episodes is explicitly provided, it should override config."""
        # Arrange: Create config with max_episodes=500
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        training_config = {
            "environment": {
                "grid_size": 5,
                "enabled_affordances": ["Bed"],
            },
            "population": {
                "num_agents": 1,
                "learning_rate": 0.001,
                "gamma": 0.95,
                "replay_buffer_capacity": 1000,
                "network_type": "simple",
            },
            "curriculum": {
                "max_steps_per_episode": 100,
                "survival_advance_threshold": 0.7,
                "survival_retreat_threshold": 0.3,
                "entropy_gate": 0.5,
                "min_steps_at_stage": 100,
            },
            "exploration": {
                "embed_dim": 64,
                "initial_intrinsic_weight": 1.0,
                "variance_threshold": 100.0,
                "survival_window": 50,
            },
            "training": {
                "device": "cpu",
                "max_episodes": 500,  # Config says 500
            }
        }

        training_yaml = config_dir / "training.yaml"
        with open(training_yaml, "w") as f:
            yaml.dump(training_config, f)

        # Copy other required config files
        import shutil
        l0_config = Path("configs/L0_minimal")
        for yaml_file in ["affordances.yaml", "bars.yaml", "cascades.yaml", "cues.yaml"]:
            shutil.copy(l0_config / yaml_file, config_dir / yaml_file)

        # Act: Create runner with explicit max_episodes=1000
        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=1000,  # Explicit override
        )

        # Assert: Should use explicit value, not config
        assert runner.max_episodes == 1000

    def test_reads_max_episodes_from_config_when_not_provided(self, tmp_path):
        """When max_episodes is not provided, should read from config YAML."""
        # Arrange: Create config with max_episodes=500
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        training_config = {
            "environment": {
                "grid_size": 5,
                "enabled_affordances": ["Bed"],
            },
            "population": {
                "num_agents": 1,
                "learning_rate": 0.001,
                "gamma": 0.95,
                "replay_buffer_capacity": 1000,
                "network_type": "simple",
            },
            "curriculum": {
                "max_steps_per_episode": 100,
                "survival_advance_threshold": 0.7,
                "survival_retreat_threshold": 0.3,
                "entropy_gate": 0.5,
                "min_steps_at_stage": 100,
            },
            "exploration": {
                "embed_dim": 64,
                "initial_intrinsic_weight": 1.0,
                "variance_threshold": 100.0,
                "survival_window": 50,
            },
            "training": {
                "device": "cpu",
                "max_episodes": 500,  # Config says 500
            }
        }

        training_yaml = config_dir / "training.yaml"
        with open(training_yaml, "w") as f:
            yaml.dump(training_config, f)

        # Copy other required config files
        import shutil
        l0_config = Path("configs/L0_minimal")
        for yaml_file in ["affordances.yaml", "bars.yaml", "cascades.yaml", "cues.yaml"]:
            shutil.copy(l0_config / yaml_file, config_dir / yaml_file)

        # Act: Create runner without providing max_episodes
        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=None,  # Will read from config
        )

        # Assert: Should read from config YAML
        assert runner.max_episodes == 500

    def test_defaults_to_10000_when_not_in_config(self, tmp_path):
        """When max_episodes is not in config, should default to 10000."""
        # Arrange: Create config WITHOUT max_episodes
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        training_config = {
            "environment": {
                "grid_size": 5,
                "enabled_affordances": ["Bed"],
            },
            "population": {
                "num_agents": 1,
                "learning_rate": 0.001,
                "gamma": 0.95,
                "replay_buffer_capacity": 1000,
                "network_type": "simple",
            },
            "curriculum": {
                "max_steps_per_episode": 100,
                "survival_advance_threshold": 0.7,
                "survival_retreat_threshold": 0.3,
                "entropy_gate": 0.5,
                "min_steps_at_stage": 100,
            },
            "exploration": {
                "embed_dim": 64,
                "initial_intrinsic_weight": 1.0,
                "variance_threshold": 100.0,
                "survival_window": 50,
            },
            "training": {
                "device": "cpu",
                # No max_episodes specified
            }
        }

        training_yaml = config_dir / "training.yaml"
        with open(training_yaml, "w") as f:
            yaml.dump(training_config, f)

        # Copy other required config files
        import shutil
        l0_config = Path("configs/L0_minimal")
        for yaml_file in ["affordances.yaml", "bars.yaml", "cascades.yaml", "cues.yaml"]:
            shutil.copy(l0_config / yaml_file, config_dir / yaml_file)

        # Act: Create runner without providing max_episodes
        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=None,
        )

        # Assert: Should default to 10000
        assert runner.max_episodes == 10000

    def test_stable_test_config_reads_200_episodes(self):
        """Integration test: configs/test should read 200 episodes (stable test config)."""
        # Arrange: Use stable test config (never changes)
        config_dir = Path("configs/test")
        if not config_dir.exists():
            pytest.skip("Test config not found")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Act: Create runner without explicit max_episodes
            runner = DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test.db",
                checkpoint_dir=tmp_path / "checkpoints",
                max_episodes=None,  # Read from config
            )

            # Assert: Test config specifies 200 episodes
            assert runner.max_episodes == 200
