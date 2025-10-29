"""Tests for demo runner."""

import pytest
import tempfile
import torch
from pathlib import Path
from hamlet.demo.runner import DemoRunner
from hamlet.demo.database import DemoDatabase


def test_demo_runner_initialization():
    """DemoRunner should initialize from config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(__file__).parent.parent.parent / "configs" / "townlet" / "sparse_adaptive.yaml"
        runner = DemoRunner(
            config_path=config_path,
            db_path=Path(tmpdir) / "demo.db",
            checkpoint_dir=Path(tmpdir) / "checkpoints",
            max_episodes=100  # Override for testing
        )

        assert runner.max_episodes == 100
        assert runner.current_episode == 0
        assert runner.db is not None


def test_checkpoint_save_load():
    """Should save and load checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(__file__).parent.parent.parent / "configs" / "townlet" / "sparse_adaptive.yaml"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        runner = DemoRunner(
            config_path=config_path,
            db_path=Path(tmpdir) / "demo.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=100
        )

        # Run a few episodes
        runner.current_episode = 42

        # Save checkpoint
        runner.save_checkpoint()

        # Verify file exists
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_ep*.pt"))
        assert len(checkpoint_files) == 1
        assert "ep00042" in str(checkpoint_files[0])

        # Load checkpoint in new runner
        runner2 = DemoRunner(
            config_path=config_path,
            db_path=Path(tmpdir) / "demo.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=100
        )
        loaded_episode = runner2.load_checkpoint()

        assert loaded_episode == 42
