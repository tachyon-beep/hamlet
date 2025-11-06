"""Test pre-flight validation detects old checkpoints."""

from pathlib import Path

import pytest
import torch

from townlet.demo.runner import DemoRunner


def test_preflight_detects_old_checkpoints(tmp_path):
    """DemoRunner should detect and reject old checkpoints on startup."""
    # Create mock old checkpoint (missing substrate_metadata)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    old_checkpoint = checkpoint_dir / "checkpoint_ep00100.pt"
    torch.save(
        {
            "episode": 100,
            "network_state": {},
            "optimizer_state": {},
            # Missing substrate_metadata field (old format)
        },
        old_checkpoint,
    )

    # Attempting to create DemoRunner should detect old checkpoint
    with pytest.raises(ValueError, match="Old checkpoints detected"):
        DemoRunner(
            config_dir=Path("configs/L1_full_observability"),
            db_path=tmp_path / "test.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=10,
            training_config_path=Path("configs/L1_full_observability/training.yaml"),
        )


def test_preflight_allows_new_checkpoints(tmp_path):
    """DemoRunner should allow new checkpoints with substrate_metadata."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    new_checkpoint = checkpoint_dir / "checkpoint_ep00100.pt"
    torch.save(
        {
            "episode": 100,
            "network_state": {},
            "optimizer_state": {},
            "substrate_metadata": {"position_dim": 2},  # New format
        },
        new_checkpoint,
    )

    # Should not raise error
    runner = DemoRunner(
        config_dir=Path("configs/L1_full_observability"),
        db_path=tmp_path / "test.db",
        checkpoint_dir=checkpoint_dir,
        max_episodes=10,
        training_config_path=Path("configs/L1_full_observability/training.yaml"),
    )
    runner._cleanup()  # Clean up


def test_preflight_allows_empty_checkpoint_dir(tmp_path):
    """DemoRunner should allow empty checkpoint directory (fresh start)."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Should not raise error (no checkpoints to validate)
    runner = DemoRunner(
        config_dir=Path("configs/L1_full_observability"),
        db_path=tmp_path / "test.db",
        checkpoint_dir=checkpoint_dir,
        max_episodes=10,
        training_config_path=Path("configs/L1_full_observability/training.yaml"),
    )
    runner._cleanup()  # Clean up
