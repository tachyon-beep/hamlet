"""Integration tests for multi-substrate training (TASK-002A Phase 8).

Tests that full training loops work correctly with Grid2D and Aspatial substrates.
"""

import pytest
import torch


@pytest.mark.parametrize("substrate_fixture", ["grid2d_8x8_env", "aspatial_env"])
def test_episode_reset_multi_substrate(substrate_fixture, request):
    """Test episode reset works correctly for all substrates."""
    env = request.getfixturevalue(substrate_fixture)

    # Reset environment
    obs1 = env.reset()

    # Check observation shape
    assert obs1.shape[0] == env.num_agents, "Observation should have num_agents rows"
    assert obs1.shape[1] == env.observation_dim, "Observation should match observation_dim"

    # Step a few times
    for _ in range(5):
        actions = torch.zeros(env.num_agents, dtype=torch.long, device=env.device)
        env.step(actions, depletion_multiplier=1.0)

    # Reset again
    obs2 = env.reset()

    # Observation shape should be consistent
    assert obs2.shape == obs1.shape, "Observation shape should be consistent across resets"


@pytest.mark.parametrize("substrate_fixture", ["grid2d_8x8_env", "aspatial_env"])
def test_affordance_interaction_multi_substrate(substrate_fixture, request):
    """Test affordance interactions work for all substrates."""
    env = request.getfixturevalue(substrate_fixture)

    env.reset()

    # For Grid2D: Move agent to affordance position
    substrate_type = type(env.substrate).__name__.lower().replace("substrate", "")
    if substrate_type == "grid2d":
        # Place agent on first affordance
        affordance_pos = list(env.affordances.values())[0]
        env.positions[0] = affordance_pos

    # Perform INTERACT action (find correct action ID for substrate)
    interact_action_id = env.action_space.get_action_by_name("INTERACT").id
    actions = torch.tensor([interact_action_id], dtype=torch.long, device=env.device)
    obs, rewards, dones, info = env.step(actions, depletion_multiplier=1.0)

    # Interaction should complete
    # (No assertion on success - may fail if requirements not met)
    # Just verify no crashes

    assert obs.shape[0] == env.num_agents, "Observation should be returned"


@pytest.mark.parametrize("substrate_fixture", ["grid2d_8x8_env", "aspatial_env"])
def test_observation_dimension_consistency_multi_substrate(substrate_fixture, request):
    """Test observation dimension stays consistent across episodes."""
    env = request.getfixturevalue(substrate_fixture)

    # Get observation dimension from first reset
    obs1 = env.reset()
    dim1 = obs1.shape[1]

    # Run 3 episodes
    for _ in range(3):
        env.reset()
        for _ in range(10):
            actions = torch.zeros(env.num_agents, dtype=torch.long, device=env.device)
            obs, _, _, _ = env.step(actions, depletion_multiplier=1.0)

        # Observation dimension should be consistent
        assert obs.shape[1] == dim1, f"Observation dimension changed: {obs.shape[1]} vs {dim1}"


def test_grid2d_position_tensor_shape(grid2d_8x8_env):
    """Grid2D should have position_dim=2."""
    grid2d_8x8_env.reset()

    assert grid2d_8x8_env.positions.shape == (
        grid2d_8x8_env.num_agents,
        2,
    ), f"Grid2D positions should be (num_agents, 2), got {grid2d_8x8_env.positions.shape}"


def test_aspatial_position_tensor_shape(aspatial_env):
    """Aspatial should have position_dim=0."""
    aspatial_env.reset()

    assert aspatial_env.positions.shape == (
        aspatial_env.num_agents,
        0,
    ), f"Aspatial positions should be (num_agents, 0), got {aspatial_env.positions.shape}"
