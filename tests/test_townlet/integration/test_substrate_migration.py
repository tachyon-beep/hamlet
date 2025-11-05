"""Integration tests for substrate.yaml migration (environment loading).

NOTE: These tests will FAIL until Phase 4 (Environment Integration) is complete.
They document expected behavior after VectorizedEnv is updated to load substrate.yaml.
"""

import pytest

# These tests will fail until Phase 4 - skip them for now
pytestmark = pytest.mark.skip(reason="Phase 4 (Environment Integration) not yet complete")


# NOTE: Commented out until VectorizedEnv supports substrate loading
# from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.mark.parametrize(
    "config_name,expected_obs_dim",
    [
        ("L0_0_minimal", 36),  # 9 grid + 8 meters + 15 affordances + 4 temporal
        ("L0_5_dual_resource", 76),  # 49 grid + 8 meters + 15 affordances + 4 temporal
        ("L1_full_observability", 91),  # 64 grid + 8 + 15 + 4
        ("L2_partial_observability", 91),  # Same as L1 (full obs_dim, not local window)
        ("L3_temporal_mechanics", 91),  # Same as L1
    ],
)
def test_env_observation_dim_unchanged(config_name, expected_obs_dim):
    """Environment with substrate.yaml should produce same obs dims as legacy."""
    # NOTE: This will work after Phase 4 when VectorizedEnv loads substrate.yaml
    pytest.skip("Phase 4 not complete - VectorizedEnv doesn't load substrate yet")

    # env = VectorizedHamletEnv(
    #     config_pack_path=Path("configs") / config_name,
    #     num_agents=1,
    #     device=torch.device("cpu"),
    # )
    #
    # # Observation dimension should match legacy hardcoded behavior
    # assert env.observation_dim == expected_obs_dim
    #
    # # Verify substrate loaded correctly
    # assert env.substrate is not None
    # assert env.substrate.position_dim == 2  # 2D grid


@pytest.mark.parametrize(
    "config_name,expected_grid_size",
    [
        ("L0_0_minimal", 3),
        ("L0_5_dual_resource", 7),
        ("L1_full_observability", 8),
        ("L2_partial_observability", 8),
        ("L3_temporal_mechanics", 8),
    ],
)
def test_env_substrate_dimensions(config_name, expected_grid_size):
    """Environment substrate should have correct grid dimensions."""
    pytest.skip("Phase 4 not complete - VectorizedEnv doesn't load substrate yet")

    # env = VectorizedHamletEnv(
    #     config_pack_path=Path("configs") / config_name,
    #     num_agents=1,
    #     device=torch.device("cpu"),
    # )
    #
    # # Substrate should be Grid2D with correct dimensions
    # assert env.substrate.width == expected_grid_size
    # assert env.substrate.height == expected_grid_size
    # assert env.substrate.width == env.substrate.height  # Square grid


def test_env_substrate_boundary_behavior():
    """Environment substrate should use clamp boundary (legacy behavior)."""
    pytest.skip("Phase 4 not complete - VectorizedEnv doesn't load substrate yet")

    # env = VectorizedHamletEnv(
    #     config_pack_path=Path("configs") / "L1_full_observability",
    #     num_agents=1,
    #     device=torch.device("cpu"),
    # )
    #
    # # Test boundary clamping (agent at edge trying to move out of bounds)
    # positions = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")  # Top-left corner
    # action_deltas = torch.tensor([[-1, -1]], dtype=torch.long, device="cpu")  # Try to move up-left
    #
    # new_positions = env.substrate.apply_movement(positions, action_deltas)
    #
    # # Should clamp to [0, 0] (not wrap or bounce)
    # assert (new_positions == torch.tensor([[0, 0]], dtype=torch.long)).all()


def test_env_substrate_distance_metric():
    """Environment substrate should use manhattan distance (legacy behavior)."""
    pytest.skip("Phase 4 not complete - VectorizedEnv doesn't load substrate yet")

    # env = VectorizedHamletEnv(
    #     config_pack_path=Path("configs") / "L1_full_observability",
    #     num_agents=1,
    #     device=torch.device("cpu"),
    # )
    #
    # # Test manhattan distance calculation
    # pos1 = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")
    # pos2 = torch.tensor([[3, 4]], dtype=torch.long, device="cpu")
    #
    # distance = env.substrate.compute_distance(pos1, pos2)
    #
    # # Manhattan distance: |3-0| + |4-0| = 7
    # assert distance.item() == 7.0
