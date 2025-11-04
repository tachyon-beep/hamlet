"""Integration tests for DemoRunner orchestration.

This module tests the runner's integration with core components:
- Episode recording (migrated tests)
- Episode loop execution
- Checkpoint saving at intervals
- Database logging after episodes
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest
import torch
import yaml


class TestRunnerRecordingIntegration:
    """Test runner integration with recording system (migrated from test_recording)."""

    def test_runner_initializes_recorder_attribute(self, tmp_path):
        """Runner should have recorder attribute initialized to None."""
        from townlet.demo.runner import DemoRunner

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
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "training.yaml").write_text(config_content)

        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"

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

    def test_runner_config_has_recording_field(self, tmp_path):
        """Runner should load recording config when present."""
        from townlet.demo.runner import DemoRunner

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
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "training.yaml").write_text(config_content)

        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"

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


class TestRunnerOrchestration:
    """Test runner orchestration of training loop, checkpointing, and database logging."""

    def test_runner_episode_loop_execution(self, tmp_path, test_config_pack_path, cpu_device):
        """Verify runner executes episode loop with env reset, steps, and done handling.

        Critical integration test: Runner orchestrates env reset, population steps,
        and episode completion. Tests the core training loop.
        """
        from townlet.demo.runner import DemoRunner

        # Setup: Create runner with minimal config (no affordances to ensure fast death)
        config_content = """
environment:
  grid_size: 8
  partial_observability: false
  enabled_affordances: []  # No affordances = agents die quickly

population:
  num_agents: 1
  learning_rate: 0.00025
  network_type: simple
  gamma: 0.99
  replay_buffer_capacity: 1000

curriculum:
  max_steps_per_episode: 50
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 10

exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 10

training:
  device: cpu
  max_episodes: 3
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 32
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.99
  epsilon_min: 0.01
"""
        # Write config to temp directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "training.yaml").write_text(config_content)

        # Copy test config pack files (bars, cascades, affordances, cues)
        import shutil
        for file in ["bars.yaml", "cascades.yaml", "affordances.yaml", "cues.yaml"]:
            src = test_config_pack_path / file
            dst = config_dir / file
            if src.exists():
                shutil.copy(src, dst)

        # Create runner
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        runner = DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=3,
        )

        # Run training
        runner.run()

        # Verify: Runner completed all episodes
        assert runner.current_episode == 3, "Runner should complete 3 episodes"

        # Verify: Database has episode records
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        episode_count = cursor.fetchone()[0]
        conn.close()

        assert episode_count == 3, f"Database should have 3 episode records, got {episode_count}"

    def test_runner_checkpoint_save_at_interval(self, tmp_path, test_config_pack_path, cpu_device):
        """Verify runner saves checkpoint every 100 episodes.

        Critical integration test: Runner should automatically save checkpoints
        at regular intervals (every 100 episodes) without manual intervention.
        """
        from townlet.demo.runner import DemoRunner

        # Setup: Create runner with short episode limit
        config_content = """
environment:
  grid_size: 8
  partial_observability: false
  enabled_affordances: []  # No affordances = fast death

population:
  num_agents: 1
  learning_rate: 0.00025
  network_type: simple
  gamma: 0.99
  replay_buffer_capacity: 1000

curriculum:
  max_steps_per_episode: 50
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 10

exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 10

training:
  device: cpu
  max_episodes: 150  # Run 150 episodes to trigger 100-episode checkpoint
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 32
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.99
  epsilon_min: 0.01
"""
        # Write config to temp directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "training.yaml").write_text(config_content)

        # Copy test config pack files
        import shutil
        for file in ["bars.yaml", "cascades.yaml", "affordances.yaml", "cues.yaml"]:
            src = test_config_pack_path / file
            dst = config_dir / file
            if src.exists():
                shutil.copy(src, dst)

        # Create runner
        checkpoint_dir = tmp_path / "checkpoints"
        db_path = tmp_path / "test.db"
        runner = DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=150,
        )

        # Run training
        runner.run()

        # Verify: Checkpoint file exists at checkpoint_dir
        checkpoints = list(checkpoint_dir.glob("checkpoint_ep*.pt"))

        # Should have at least checkpoint at episode 100 and final checkpoint at 150
        checkpoint_episodes = [int(cp.stem.replace("checkpoint_ep", "")) for cp in checkpoints]

        assert 100 in checkpoint_episodes, \
            f"Checkpoint should exist at episode 100, found: {checkpoint_episodes}"
        assert 150 in checkpoint_episodes, \
            f"Checkpoint should exist at final episode 150, found: {checkpoint_episodes}"

    def test_runner_database_logging_after_episode(self, tmp_path, test_config_pack_path, cpu_device):
        """Verify runner logs episode metrics to database after each episode.

        Critical integration test: After each episode, runner should write
        episode record to database with survival_time, rewards, stage, epsilon.
        """
        from townlet.demo.runner import DemoRunner

        # Setup: Create runner
        config_content = """
environment:
  grid_size: 8
  partial_observability: false
  enabled_affordances: []  # No affordances = fast death

population:
  num_agents: 1
  learning_rate: 0.00025
  network_type: simple
  gamma: 0.99
  replay_buffer_capacity: 1000

curriculum:
  max_steps_per_episode: 50
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 10

exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 10

training:
  device: cpu
  max_episodes: 5  # Run 5 episodes
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 32
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.99
  epsilon_min: 0.01
"""
        # Write config to temp directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "training.yaml").write_text(config_content)

        # Copy test config pack files
        import shutil
        for file in ["bars.yaml", "cascades.yaml", "affordances.yaml", "cues.yaml"]:
            src = test_config_pack_path / file
            dst = config_dir / file
            if src.exists():
                shutil.copy(src, dst)

        # Create runner
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        runner = DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=5,
        )

        # Run training
        runner.run()

        # Verify: Database has 5 episode records
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check episode count
        cursor.execute("SELECT COUNT(*) FROM episodes")
        episode_count = cursor.fetchone()[0]
        assert episode_count == 5, f"Should have 5 episode records, got {episode_count}"

        # Check required fields exist (database schema uses survival_time not survival_steps)
        cursor.execute(
            "SELECT survival_time, total_reward, curriculum_stage, epsilon "
            "FROM episodes LIMIT 1"
        )
        record = cursor.fetchone()
        assert record is not None, "Episode record should exist"

        survival_time, total_reward, curriculum_stage, epsilon = record

        # Verify all fields are populated (not NULL)
        assert survival_time is not None, "survival_time should be populated"
        assert total_reward is not None, "total_reward should be populated"
        assert curriculum_stage is not None, "curriculum_stage should be populated"
        assert epsilon is not None, "epsilon should be populated"

        # Verify types are reasonable
        assert isinstance(survival_time, int), "survival_time should be integer"
        assert isinstance(total_reward, float) or isinstance(total_reward, int), "total_reward should be numeric"
        assert isinstance(curriculum_stage, int), "curriculum_stage should be integer"
        assert isinstance(epsilon, float) or isinstance(epsilon, int), "epsilon should be numeric"

        conn.close()


class TestRunnerAffordanceTransitions:
    """Test runner tracking of affordance transitions (Phase 2)."""

    def test_runner_persists_transitions_to_database(self, tmp_path, test_config_pack_path, cpu_device):
        """Runner should persist affordance transitions to database after episode.

        Integration test: When agent uses affordances in sequence (Bed → Hospital → Job),
        runner should track transitions and save to database using insert_affordance_visits().
        """
        from townlet.demo.runner import DemoRunner

        # Create config with enabled affordances (Bed only for simplicity)
        config_content = """
environment:
  grid_size: 3
  partial_observability: false
  enabled_affordances: ["Bed"]  # Single affordance to test self-loops

population:
  num_agents: 1
  learning_rate: 0.00025
  network_type: simple
  gamma: 0.99
  replay_buffer_capacity: 1000

curriculum:
  max_steps_per_episode: 100  # Enough steps to interact multiple times
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 10

exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 10

training:
  device: cpu
  max_episodes: 50  # More episodes = higher chance of interactions
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 32
  max_grad_norm: 10.0
  epsilon_start: 0.9  # Higher epsilon = more exploration = more interactions
  epsilon_decay: 0.99
  epsilon_min: 0.01
"""
        # Write config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "training.yaml").write_text(config_content)

        # Copy test config pack files
        import shutil
        for file in ["bars.yaml", "cascades.yaml", "affordances.yaml", "cues.yaml"]:
            src = test_config_pack_path / file
            dst = config_dir / file
            if src.exists():
                shutil.copy(src, dst)

        # Create runner
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        runner = DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=50,
        )

        # Run training
        runner.run()

        # Verify: Database has affordance_visits records
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check that affordance_visits table has records
        cursor.execute("SELECT COUNT(*) FROM affordance_visits")
        transition_count = cursor.fetchone()[0]

        # We should have at least 1 transition recorded across 50 episodes
        # (Agent will use Bed multiple times, creating Bed→Bed transitions)
        assert transition_count > 0, \
            f"Expected at least 1 affordance transition recorded, got {transition_count}"

        # Verify transition structure
        cursor.execute(
            "SELECT episode_id, from_affordance, to_affordance, visit_count "
            "FROM affordance_visits LIMIT 1"
        )
        record = cursor.fetchone()

        if record is not None:
            episode_id, from_aff, to_aff, count = record
            assert isinstance(episode_id, int), "episode_id should be integer"
            assert isinstance(from_aff, str), "from_affordance should be string"
            assert isinstance(to_aff, str), "to_affordance should be string"
            assert isinstance(count, int) and count > 0, "visit_count should be positive integer"

        conn.close()
