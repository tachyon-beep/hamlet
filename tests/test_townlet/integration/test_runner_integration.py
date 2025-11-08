"""Integration tests for DemoRunner orchestration.

This module tests the runner's integration with core components:
- Episode recording (migrated tests)
- Episode loop execution
- Checkpoint saving at intervals
- Database logging after episodes
"""

import sqlite3

import pytest

from tests.test_townlet.helpers.config_builder import prepare_config_dir


class TestRunnerRecordingIntegration:
    """Test runner integration with recording system (migrated from test_recording)."""

    def test_runner_initializes_recorder_attribute(self, tmp_path):
        """Runner should have recorder attribute initialized to None."""
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            data["training"]["max_episodes"] = 2

        config_dir = prepare_config_dir(tmp_path, modifier)
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"

        # Initialize runner (use context manager to ensure cleanup)
        with DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=2,
        ) as runner:
            # Verify recorder attribute exists and starts as None
            assert hasattr(runner, "recorder")
            assert runner.recorder is None

    def test_runner_config_has_recording_field(self, tmp_path):
        """Runner should load recording config when present."""
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            data["training"]["max_episodes"] = 2
            data["recording"] = {
                "enabled": True,
                "output_dir": "recordings",
                "max_queue_size": 100,
                "compression": "lz4",
            }

        config_dir = prepare_config_dir(tmp_path, modifier)
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"

        # Initialize runner (use context manager to ensure cleanup)
        with DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=2,
        ) as runner:
            # Verify recording config is loaded
            assert "recording" in runner.config
            assert runner.config["recording"]["enabled"] is True
            assert runner.config["recording"]["output_dir"] == "recordings"


class TestRunnerOrchestration:
    """Test runner orchestration of training loop, checkpointing, and database logging."""

    def test_runner_episode_loop_execution(self, tmp_path):
        """Verify runner executes episode loop with env reset, steps, and done handling.

        Critical integration test: Runner orchestrates env reset, population steps,
        and episode completion. Tests the core training loop.
        """
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            env = data["environment"]
            env["enabled_affordances"] = ["Bed"]
            env["vision_range"] = 8
            data["exploration"]["survival_window"] = 10
            data["training"]["max_episodes"] = 3

        config_dir = prepare_config_dir(tmp_path, modifier)
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

    def test_runner_checkpoint_save_at_interval(self, tmp_path):
        """Verify runner saves checkpoint every 100 episodes.

        Critical integration test: Runner should automatically save checkpoints
        at regular intervals (every 100 episodes) without manual intervention.
        """
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            env = data["environment"]
            env["enabled_affordances"] = ["Bed"]
            env["vision_range"] = 8
            data["exploration"]["survival_window"] = 10
            data["training"]["max_episodes"] = 150

        config_dir = prepare_config_dir(tmp_path, modifier)
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

        assert 100 in checkpoint_episodes, f"Checkpoint should exist at episode 100, found: {checkpoint_episodes}"
        assert 150 in checkpoint_episodes, f"Checkpoint should exist at final episode 150, found: {checkpoint_episodes}"

    def test_runner_database_logging_after_episode(self, tmp_path):
        """Verify runner logs episode metrics to database after each episode.

        Critical integration test: After each episode, runner should write
        episode record to database with survival_time, rewards, stage, epsilon.
        """
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            env = data["environment"]
            env["enabled_affordances"] = ["Bed"]
            env["vision_range"] = 8
            data["exploration"]["survival_window"] = 10
            data["training"]["max_episodes"] = 5

        config_dir = prepare_config_dir(tmp_path, modifier)

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
        cursor.execute("SELECT survival_time, total_reward, curriculum_stage, epsilon FROM episodes LIMIT 1")
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

    def test_runner_persists_transitions_to_database(self, tmp_path):
        """Runner should persist affordance transitions to database after episode.

        Integration test: When agent uses affordances in sequence (Bed → Hospital → Job),
        runner should track transitions and save to database using insert_affordance_visits().
        """
        import random

        import numpy as np
        import torch

        from townlet.demo.runner import DemoRunner

        # Seed all RNGs for deterministic affordance placement and agent behavior
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        def modifier(data: dict) -> None:
            env = data["environment"]
            env["grid_size"] = 3
            env["vision_range"] = 3
            env["enabled_affordances"] = ["Bed"]
            data["curriculum"]["max_steps_per_episode"] = 100
            data["exploration"]["survival_window"] = 10
            data["training"].update(
                {
                    "max_episodes": 50,
                    "epsilon_start": 0.9,
                }
            )

        config_dir = prepare_config_dir(tmp_path, modifier)

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
        assert transition_count > 0, f"Expected at least 1 affordance transition recorded, got {transition_count}"

        # Verify transition structure
        cursor.execute("SELECT episode_id, from_affordance, to_affordance, visit_count FROM affordance_visits LIMIT 1")
        record = cursor.fetchone()

        if record is not None:
            episode_id, from_aff, to_aff, count = record
            assert isinstance(episode_id, int), "episode_id should be integer"
            assert isinstance(from_aff, str), "from_affordance should be string"
            assert isinstance(to_aff, str), "to_affordance should be string"
            assert isinstance(count, int) and count > 0, "visit_count should be positive integer"

        conn.close()


class TestRunnerConfigValidation:
    """Test runner config validation and error messages (PDR-002 compliance)."""

    def test_missing_required_environment_params_raises_helpful_error(self, tmp_path):
        """Missing required environment params should raise ValueError with helpful message.

        Verifies PDR-002 no-defaults enforcement: all UAC parameters must be explicit.
        Error message should show which params are missing and how to fix.
        """
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            env = data["environment"]
            for key in [
                "grid_size",
                "partial_observability",
                "vision_range",
                "energy_move_depletion",
                "energy_wait_depletion",
                "energy_interact_depletion",
            ]:
                env.pop(key, None)

        config_dir = prepare_config_dir(tmp_path, modifier)

        with pytest.raises(ValueError) as exc_info:
            DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test.db",
                checkpoint_dir=tmp_path / "checkpoints",
                max_episodes=1,
            )

        assert "environment" in str(exc_info.value).lower()

    def test_missing_required_training_params_raises_helpful_error(self, tmp_path):
        """Missing required training params should raise ValueError with helpful message."""
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            train = data["training"]
            for key in [
                "train_frequency",
                "target_update_frequency",
                "batch_size",
                "sequence_length",
                "max_grad_norm",
                "epsilon_start",
                "epsilon_decay",
                "epsilon_min",
            ]:
                train.pop(key, None)

        config_dir = prepare_config_dir(tmp_path, modifier)

        with pytest.raises(ValueError) as exc_info:
            DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test.db",
                checkpoint_dir=tmp_path / "checkpoints",
                max_episodes=1,
            )

        assert "training" in str(exc_info.value).lower()


class TestRunnerResourceCleanup:
    """Test runner resource cleanup and context manager protocol (QUICK-002)."""

    def test_context_manager_cleanup_on_normal_exit(self, tmp_path):
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            data["training"]["max_episodes"] = 1
            data["environment"]["enabled_affordances"] = ["Bed"]

        config_dir = prepare_config_dir(tmp_path, modifier)
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"

        with DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=1,
        ) as runner:
            assert runner.db is not None
            assert runner.tb_logger is not None

        assert runner.db._closed is True, "Database connection should be closed after context exit"

    def test_idempotent_cleanup_safe_to_call_multiple_times(self, tmp_path):
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            data["training"]["max_episodes"] = 1
            data["environment"]["enabled_affordances"] = ["Bed"]

        config_dir = prepare_config_dir(tmp_path, modifier)
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"

        with DemoRunner(
            config_dir=config_dir,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=1,
        ) as runner:
            pass

        runner._cleanup()
        runner._cleanup()
        runner._cleanup()

        assert runner.db._closed is True

    def test_context_manager_cleanup_on_exception(self, tmp_path):
        from townlet.demo.runner import DemoRunner

        def modifier(data: dict) -> None:
            data["training"]["max_episodes"] = 1
            data["environment"]["enabled_affordances"] = ["Bed"]

        config_dir = prepare_config_dir(tmp_path, modifier)
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"

        runner_ref = None
        with pytest.raises(RuntimeError, match="Intentional test exception"):
            with DemoRunner(
                config_dir=config_dir,
                db_path=db_path,
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            ) as runner:
                runner_ref = runner
                raise RuntimeError("Intentional test exception")

        assert runner_ref is not None
        assert runner_ref.db._closed is True, "Database should be closed even after exception"


class TestDatabaseDefensiveChecks:
    """Test database defensive checks after close() (Issue #5)."""

    def test_database_operations_after_close_raise_runtime_error(self, tmp_path):
        """Database operations after close() should raise RuntimeError with helpful message."""
        import pytest

        from townlet.demo.database import DemoDatabase

        db_path = tmp_path / "test.db"
        db = DemoDatabase(db_path)

        # Close the database
        db.close()

        # Verify all database operations raise RuntimeError after close
        with pytest.raises(RuntimeError, match="Database connection is closed"):
            db.insert_episode(
                episode_id=1,
                timestamp=1234567890.0,
                survival_time=100,
                total_reward=50.0,
                extrinsic_reward=45.0,
                intrinsic_reward=5.0,
                intrinsic_weight=0.5,
                curriculum_stage=1,
                epsilon=0.5,
            )

        with pytest.raises(RuntimeError, match="Database connection is closed"):
            db.get_latest_episodes()

        with pytest.raises(RuntimeError, match="Database connection is closed"):
            db.set_system_state("test_key", "test_value")

        with pytest.raises(RuntimeError, match="Database connection is closed"):
            db.get_system_state("test_key")

        with pytest.raises(RuntimeError, match="Database connection is closed"):
            db.insert_affordance_visits(1, {"Bed": {"Hospital": 3}})

        with pytest.raises(RuntimeError, match="Database connection is closed"):
            db.get_recording(1)

        with pytest.raises(RuntimeError, match="Database connection is closed"):
            db.list_recordings()

    def test_database_close_is_idempotent(self, tmp_path):
        """Database close() should be idempotent - safe to call multiple times."""
        from townlet.demo.database import DemoDatabase

        db_path = tmp_path / "test.db"
        db = DemoDatabase(db_path)

        # Close multiple times - should not raise exception
        db.close()
        db.close()
        db.close()

        # Verify state is still closed
        assert db._closed is True
