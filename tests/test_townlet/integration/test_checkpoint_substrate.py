"""Test checkpoint serialization with substrate metadata."""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def env_cpu(test_config_pack_path):
    """Create a CPU-based environment for testing."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
        config_pack_path=test_config_pack_path,
    )


def test_checkpoint_includes_position_dim(env_cpu):
    """Affordance positions checkpoint should include position_dim."""
    # Get affordance positions for checkpoint
    checkpoint_data = env_cpu.get_affordance_positions()

    # Should have position_dim field
    assert "position_dim" in checkpoint_data
    assert checkpoint_data["position_dim"] == env_cpu.substrate.position_dim

    # For 2D grid substrate, position_dim should be 2
    assert checkpoint_data["position_dim"] == 2


def test_checkpoint_validates_position_dim(env_cpu):
    """Loading checkpoint should validate position_dim compatibility."""
    # Create checkpoint with mismatched position_dim
    bad_checkpoint = {
        "positions": {"Bed": [2, 3, 0]},  # 3D position [x, y, z]
        "ordering": ["Bed"],
        "position_dim": 3,  # 3D!
    }

    # Should raise error when loading into 2D substrate
    with pytest.raises(ValueError, match="position_dim mismatch"):
        env_cpu.set_affordance_positions(bad_checkpoint)


def test_checkpoint_rejects_legacy_format(env_cpu):
    """BREAKING CHANGE: Legacy checkpoints (no position_dim) should be rejected."""
    # Create legacy checkpoint (no position_dim field)
    legacy_checkpoint = {
        "positions": {"Bed": [2, 3]},
        "ordering": ["Bed"],
        # No position_dim field!
    }

    # Should raise clear error for legacy format
    with pytest.raises(ValueError, match="legacy checkpoint.*pre-Phase 4"):
        env_cpu.set_affordance_positions(legacy_checkpoint)
