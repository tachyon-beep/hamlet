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


def test_training_with_continuous1d_substrate(tmp_path):
    """Training runs with 1D continuous substrate."""
    config_dir = Path("configs/L1_continuous_1D")
    if not config_dir.exists():
        pytest.skip("L1_continuous_1D config not found")

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=5,
        training_config_path=config_dir / "training.yaml",
    ) as runner:
        runner.run()

        # Verify 1D continuous positions
        assert runner.env.positions.shape[1] == 1
        assert runner.env.positions.dtype == torch.float32

        # Verify X dimension in bounds [0, 10]
        assert (runner.env.positions[:, 0] >= 0.0).all()
        assert (runner.env.positions[:, 0] <= 10.0).all()

        # Verify action dim = 4 for 1D
        assert runner.env.action_dim == 4


def test_training_with_continuous2d_substrate(tmp_path):
    """Training runs with 2D continuous substrate."""
    config_dir = Path("configs/L1_continuous_2D")
    if not config_dir.exists():
        pytest.skip("L1_continuous_2D config not found")

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=5,
        training_config_path=config_dir / "training.yaml",
    ) as runner:
        runner.run()

        # Verify 2D continuous positions
        assert runner.env.positions.shape[1] == 2
        assert runner.env.positions.dtype == torch.float32

        # Verify X and Y dimensions in bounds [0, 10]
        assert (runner.env.positions[:, 0] >= 0.0).all()
        assert (runner.env.positions[:, 0] <= 10.0).all()
        assert (runner.env.positions[:, 1] >= 0.0).all()
        assert (runner.env.positions[:, 1] <= 10.0).all()

        # Verify action dim = 6 for 2D
        assert runner.env.action_dim == 6

        # Verify affordances are randomly placed in continuous space
        for affordance_name, affordance_pos in runner.env.affordances.items():
            assert affordance_pos.dtype == torch.float32
            assert affordance_pos.shape[0] == 2  # 2D positions


def test_training_with_continuous3d_substrate(tmp_path):
    """Training runs with 3D continuous substrate."""
    config_dir = Path("configs/L1_continuous_3D")
    if not config_dir.exists():
        pytest.skip("L1_continuous_3D config not found")

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=5,
        training_config_path=config_dir / "training.yaml",
    ) as runner:
        runner.run()

        # Verify 3D continuous positions
        assert runner.env.positions.shape[1] == 3
        assert runner.env.positions.dtype == torch.float32

        # Verify X, Y, Z dimensions in bounds [0, 10]
        assert (runner.env.positions[:, 0] >= 0.0).all()
        assert (runner.env.positions[:, 0] <= 10.0).all()
        assert (runner.env.positions[:, 1] >= 0.0).all()
        assert (runner.env.positions[:, 1] <= 10.0).all()
        assert (runner.env.positions[:, 2] >= 0.0).all()
        assert (runner.env.positions[:, 2] <= 10.0).all()

        # Verify action dim = 8 for 3D
        assert runner.env.action_dim == 8

        # Verify affordances are randomly placed in continuous space
        for affordance_name, affordance_pos in runner.env.affordances.items():
            assert affordance_pos.dtype == torch.float32
            assert affordance_pos.shape[0] == 3  # 3D positions


def test_continuous_proximity_interaction(tmp_path):
    """Verify proximity-based interaction works in continuous space."""
    from townlet.substrate.continuous import Continuous2DSubstrate

    # Create continuous substrate directly
    substrate = Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    # Place agent and affordance close together
    agent_pos = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
    affordance_pos = torch.tensor([[5.5, 5.5]], dtype=torch.float32)

    # Check proximity interaction
    # Distance = sqrt((5.5-5.0)^2 + (5.5-5.0)^2) = sqrt(0.5) ≈ 0.707 < 0.8
    on_affordance = substrate.is_on_position(agent_pos, affordance_pos)

    assert on_affordance[0], "Agent should be within interaction radius"

    # Now move affordance outside interaction radius
    affordance_pos_far = torch.tensor([[7.0, 7.0]], dtype=torch.float32)
    on_affordance_far = substrate.is_on_position(agent_pos, affordance_pos_far)

    assert not on_affordance_far[0], "Agent should be outside interaction radius"
