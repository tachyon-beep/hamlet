"""
Tests for runner integration with episode recording.

Tests that DemoRunner properly integrates with EpisodeRecorder for capturing episodes.
"""

import pytest
import tempfile
import time
from pathlib import Path
import torch


class TestRunnerRecording:
    """Test runner integration with recording system."""

    def test_runner_initializes_recorder_attribute(self):
        """Runner should have recorder attribute initialized to None."""
        from townlet.demo.runner import DemoRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create minimal training config
            config_content = """
environment:
  grid_size: 8
  partial_observability: false

population:
  num_agents: 1
  learning_rate: 0.00025

curriculum:
  max_steps_per_episode: 50

exploration:
  embed_dim: 128

training:
  device: cpu
  max_episodes: 2
"""
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            (config_dir / "training.yaml").write_text(config_content)

            db_path = tmpdir_path / "test.db"
            checkpoint_dir = tmpdir_path / "checkpoints"

            # Initialize runner
            runner = DemoRunner(
                config_dir=config_dir,
                db_path=db_path,
                checkpoint_dir=checkpoint_dir,
                max_episodes=2,
            )

            # Verify recorder attribute exists and starts as None
            assert hasattr(runner, 'recorder')
            assert runner.recorder is None

    def test_runner_config_has_recording_field(self):
        """Runner should load recording config when present."""
        from townlet.demo.runner import DemoRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create config WITH recording section
            config_content = """
environment:
  grid_size: 8
  partial_observability: false

population:
  num_agents: 1
  learning_rate: 0.00025

curriculum:
  max_steps_per_episode: 50

exploration:
  embed_dim: 128

training:
  device: cpu
  max_episodes: 2

recording:
  enabled: true
  output_dir: recordings
  max_queue_size: 100
  compression: lz4
"""
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            (config_dir / "training.yaml").write_text(config_content)

            db_path = tmpdir_path / "test.db"
            checkpoint_dir = tmpdir_path / "checkpoints"

            # Initialize runner
            runner = DemoRunner(
                config_dir=config_dir,
                db_path=db_path,
                checkpoint_dir=checkpoint_dir,
                max_episodes=2,
            )

            # Verify recording config is loaded
            assert "recording" in runner.config
            assert runner.config["recording"]["enabled"] is True
            assert runner.config["recording"]["output_dir"] == "recordings"
