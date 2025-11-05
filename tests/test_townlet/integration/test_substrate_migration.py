"""Integration tests for substrate migration (Grid2D → Grid3D → Continuous)."""

from pathlib import Path

import pytest
import torch

from townlet.demo.runner import DemoRunner


def test_training_with_grid3d_substrate(tmp_path):
    """Training runs with 3D cubic grid."""
    config_dir = Path("configs/L1_3D_house")
    if not config_dir.exists():
        pytest.skip("L1_3D_house config not found")

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=5,
        training_config_path=config_dir / "training.yaml",
    ) as runner:
        runner.run()

        # Verify 3D positions
        assert runner.env.positions.shape[1] == 3
        assert runner.env.positions.dtype == torch.long

        # Verify Z dimension in bounds
        assert (runner.env.positions[:, 2] >= 0).all()
        assert (runner.env.positions[:, 2] < 3).all()
