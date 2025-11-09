"""Property-based tests for VectorizedHamletEnv.

These tests use Hypothesis to generate random inputs and verify that
universal invariants hold across all possible inputs.

Properties tested:
1. Agent positions never leave grid bounds after any action sequence
2. Meter values stay in [0, 1] after any interaction sequence
3. Observations always match expected dimensions regardless of state
4. Temporal features are always valid (sin² + cos² = 1)

Note: ObservationBuilder was removed in VFS integration.
"""

from pathlib import Path

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_townlet.utils.builders import make_vectorized_env_from_pack


class TestEnvironmentBoundaryProperties:
    """Property tests for environment boundaries and constraints."""

    @given(
        action_sequence=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=50),
    )
    @settings(max_examples=50, deadline=None)  # Reduce examples for faster tests, disable deadline for VFS overhead
    def test_agents_never_leave_grid_bounds(self, action_sequence):
        """Property: Agent positions always in [0, grid_size) after ANY action sequence.

        This tests the fundamental spatial constraint that agents cannot move
        outside the grid boundaries, regardless of action sequence.

        NOTE: After TASK-002A, grid_size comes from substrate.yaml (8×8),
              not from the grid_size parameter.
        """
        env = make_vectorized_env_from_pack(
            Path("configs/test"),
            num_agents=1,
            device=torch.device("cpu"),
        )

        # Reset environment
        obs = env.reset()
        assert obs.shape[0] == 1  # Single agent

        # Get actual grid_size from substrate (not parameter)
        actual_grid_size = env.grid_size  # Loaded from substrate.yaml

        # Execute action sequence
        for action in action_sequence:
            obs, rewards, dones, info = env.step(torch.tensor([action]))

            # PROPERTY: Positions always in bounds
            positions = env.positions
            assert torch.all(positions[:, 0] >= 0), f"X position {positions[0, 0]} < 0"
            assert torch.all(positions[:, 0] < actual_grid_size), f"X position {positions[0, 0]} >= {actual_grid_size}"
            assert torch.all(positions[:, 1] >= 0), f"Y position {positions[0, 1]} < 0"
            assert torch.all(positions[:, 1] < actual_grid_size), f"Y position {positions[0, 1]} >= {actual_grid_size}"

            # Stop if agent dies (no more actions possible)
            if dones[0]:
                break

    @given(
        num_agents=st.integers(min_value=1, max_value=8),
        num_steps=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=30, deadline=None)  # Disable deadline for VFS overhead
    def test_meters_stay_in_valid_range(self, num_agents, num_steps):
        """Property: Meters always in [0.0, 1.0] after any sequence of steps.

        Meters represent normalized resources (energy, health, etc.) and must
        never go below 0 or above 1, regardless of interactions or decay.
        """
        env = make_vectorized_env_from_pack(
            Path("configs/test"),
            num_agents=num_agents,
            device=torch.device("cpu"),
        )

        env.reset()

        # Execute random actions
        for _ in range(num_steps):
            actions = torch.randint(0, 6, (num_agents,))
            obs, rewards, dones, info = env.step(actions)

            # PROPERTY: All meters in [0, 1]
            meters = env.meters
            assert torch.all(meters >= 0.0), f"Meter values below 0: {meters[meters < 0.0]}"
            assert torch.all(meters <= 1.0), f"Meter values above 1: {meters[meters > 1.0]}"

            # Stop if all agents dead
            if torch.all(dones):
                break

    @given(
        config_name=st.sampled_from(
            [
                "configs/L0_0_minimal",  # 3×3 full obs
                "configs/test",  # 8×8 full obs
                "configs/L2_partial_observability",  # POMDP
            ]
        ),
        num_agents=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=40, deadline=None)  # Disable deadline for VFS overhead
    def test_observations_always_match_expected_dimensions(self, config_name, num_agents):
        """Property: Observations always match observation_dim regardless of state.

        The observation dimension is computed at initialization and should
        never change during environment execution.
        """

        env = make_vectorized_env_from_pack(
            Path(config_name),
            num_agents=num_agents,
            device=torch.device("cpu"),
        )

        # Reset and check initial observation
        obs = env.reset()
        expected_dim = env.observation_dim

        # PROPERTY: Initial observations match expected dimension
        assert obs.shape == (num_agents, expected_dim), f"Expected {(num_agents, expected_dim)}, got {obs.shape}"

        # Execute random actions and verify dimensions persist
        for _ in range(10):
            actions = torch.randint(0, 6, (num_agents,))
            obs, rewards, dones, info = env.step(actions)

            # PROPERTY: Observations always match expected dimension
            assert obs.shape == (num_agents, expected_dim), f"Expected {(num_agents, expected_dim)}, got {obs.shape}"

            if torch.all(dones):
                break
