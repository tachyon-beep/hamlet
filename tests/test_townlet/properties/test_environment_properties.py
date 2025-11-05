"""Property-based tests for VectorizedHamletEnv and ObservationBuilder.

These tests use Hypothesis to generate random inputs and verify that
universal invariants hold across all possible inputs.

Properties tested:
1. Agent positions never leave grid bounds after any action sequence
2. Meter values stay in [0, 1] after any interaction sequence
3. Observations always match expected dimensions regardless of state
4. Temporal features are always valid (sin² + cos² = 1)
"""

from pathlib import Path

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from townlet.environment.observation_builder import ObservationBuilder
from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestEnvironmentBoundaryProperties:
    """Property tests for environment boundaries and constraints."""

    @given(
        action_sequence=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=50),
    )
    @settings(max_examples=50)  # Reduce examples for faster tests
    def test_agents_never_leave_grid_bounds(self, action_sequence):
        """Property: Agent positions always in [0, grid_size) after ANY action sequence.

        This tests the fundamental spatial constraint that agents cannot move
        outside the grid boundaries, regardless of action sequence.

        NOTE: After TASK-002A, grid_size comes from substrate.yaml (8×8),
              not from the grid_size parameter.
        """
        # Create environment
        # grid_size loaded from substrate.yaml (8×8)
        project_root = Path(__file__).parent.parent.parent.parent
        config_pack = project_root / "configs" / "test"

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,  # NOTE: Ignored! Actual grid_size comes from substrate.yaml
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
            config_pack_path=config_pack,
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
    @settings(max_examples=30)
    def test_meters_stay_in_valid_range(self, num_agents, num_steps):
        """Property: Meters always in [0.0, 1.0] after any sequence of steps.

        Meters represent normalized resources (energy, health, etc.) and must
        never go below 0 or above 1, regardless of interactions or decay.
        """
        project_root = Path(__file__).parent.parent.parent.parent
        config_pack = project_root / "configs" / "test"

        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
            config_pack_path=config_pack,
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
        grid_size=st.integers(min_value=5, max_value=12),  # Min 5 for affordances and POMDP
        num_agents=st.integers(min_value=1, max_value=8),
        partial_observability=st.booleans(),
    )
    @settings(max_examples=40)
    def test_observations_always_match_expected_dimensions(self, grid_size, num_agents, partial_observability):
        """Property: Observations always match observation_dim regardless of state.

        The observation dimension is computed at initialization and should
        never change during environment execution.
        """

        project_root = Path(__file__).parent.parent.parent.parent
        config_pack = project_root / "configs" / "test"

        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=grid_size,
            partial_observability=partial_observability,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
            config_pack_path=config_pack,
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


class TestObservationBuilderProperties:
    """Property tests for observation construction."""

    @given(
        grid_size=st.integers(min_value=3, max_value=16),
        num_agents=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=30)
    def test_grid_encoding_has_correct_dimensions(self, grid_size, num_agents):
        """Property: Full observability grid encoding always has grid_size² dimensions.

        The one-hot grid encoding should always be exactly grid_size × grid_size,
        regardless of agent positions or affordance placements.
        """
        device = torch.device("cpu")
        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=grid_size,  # Full visibility
            enable_temporal_mechanics=False,
            num_affordance_types=3,
            affordance_names=["Bed", "Hospital", "Job"],
        )

        # Create random positions and meters
        positions = torch.randint(0, grid_size, (num_agents, 2), device=device)
        meters = torch.rand(num_agents, 8, device=device)
        affordances = {
            "Bed": torch.randint(0, grid_size, (2,), device=device),
            "Hospital": torch.randint(0, grid_size, (2,), device=device),
            "Job": torch.randint(0, grid_size, (2,), device=device),
        }

        obs = builder.build_observations(positions=positions, meters=meters, affordances=affordances)

        # PROPERTY: Grid component is exactly grid_size²
        # Observation structure: [grid | meters | affordance_encoding | temporal]
        # Grid is first grid_size² dimensions
        grid_component = obs[:, : grid_size * grid_size]
        assert grid_component.shape == (num_agents, grid_size * grid_size)

    @given(
        time_of_day=st.integers(min_value=0, max_value=23),
    )
    @settings(max_examples=24)  # Test all hours
    def test_temporal_features_are_cyclic(self, time_of_day):
        """Property: Time encoding satisfies sin² + cos² = 1 for cyclical representation.

        The time_of_day is encoded as [sin(angle), cos(angle)] which should
        always satisfy the Pythagorean identity.
        """
        device = torch.device("cpu")
        num_agents = 1

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=8,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=True,
            num_affordance_types=3,
            affordance_names=["Bed", "Hospital", "Job"],
        )

        positions = torch.tensor([[0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {
            "Bed": torch.tensor([2, 3], device=device),
            "Hospital": torch.tensor([5, 5], device=device),
            "Job": torch.tensor([7, 7], device=device),
        }

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            time_of_day=time_of_day,
        )

        # Extract temporal features (last 4 dimensions: time_sin, time_cos, progress, lifetime)
        temporal = obs[0, -4:]
        time_sin = temporal[0].item()
        time_cos = temporal[1].item()

        # PROPERTY: sin² + cos² = 1 (within floating point tolerance)
        identity = time_sin**2 + time_cos**2
        assert abs(identity - 1.0) < 1e-6, f"sin²={time_sin**2}, cos²={time_cos**2}, sum={identity}"

    @given(
        grid_size=st.integers(min_value=5, max_value=16),
        vision_range=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=30)
    def test_vision_window_never_exceeds_bounds(self, grid_size, vision_range):
        """Property: POMDP vision window always produces valid observations.

        When agents are near boundaries, the vision window should handle
        edge cases gracefully without accessing out-of-bounds cells.
        """
        # Ensure grid is large enough for vision window
        window_size = 2 * vision_range + 1
        assume(grid_size >= window_size)

        device = torch.device("cpu")
        num_agents = 4

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=True,
            vision_range=vision_range,
            enable_temporal_mechanics=False,
            num_affordance_types=3,
            affordance_names=["Bed", "Hospital", "Job"],
        )

        # Test corner positions (most likely to cause boundary issues)
        corner_positions = torch.tensor(
            [[0, 0], [0, grid_size - 1], [grid_size - 1, 0], [grid_size - 1, grid_size - 1]],
            device=device,
        )

        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {
            "Bed": torch.tensor([2, 3], device=device),
            "Hospital": torch.tensor([5, 5], device=device),
            "Job": torch.tensor([7, 7], device=device),
        }

        # PROPERTY: Building observations at corners should not crash
        # and should produce valid observation dimensions
        obs = builder.build_observations(positions=corner_positions, meters=meters, affordances=affordances)

        # Verify shape is correct
        expected_local_grid = window_size * window_size  # Local vision
        expected_position = 2  # (x, y)
        expected_meters = 8
        expected_affordance = 4  # 3 types + 1 "none"
        expected_temporal = 4
        expected_dim = expected_local_grid + expected_position + expected_meters + expected_affordance + expected_temporal

        assert obs.shape == (num_agents, expected_dim), f"Expected {(num_agents, expected_dim)}, got {obs.shape}"

        # PROPERTY: All observation values should be valid (no NaN, no Inf)
        assert torch.all(torch.isfinite(obs)), f"Non-finite values in observation: {obs[~torch.isfinite(obs)]}"
