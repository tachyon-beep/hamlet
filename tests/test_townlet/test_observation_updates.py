"""Tests for observation builder updates: affordances in grid, temporal features, lifetime progress.

These tests cover the following changes:
1. Full observability grid now includes affordance positions (0=empty, 1=agent/affordance, 2=both)
2. Temporal features (time_sin, time_cos, interaction_progress, lifetime_progress) always included
3. Agent lifespan and retirement mechanics
"""

import math

import pytest
import torch

from townlet.environment.observation_builder import ObservationBuilder
from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestFullObservabilityGrid:
    """Tests for affordance positions in full observability grid encoding."""

    def test_grid_shows_affordances_in_full_observability(self):
        """Full observability: grid marks affordance positions with value=1.0."""
        num_agents = 1
        grid_size = 8
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            num_affordance_types=3,
        )

        # Agent at (0, 0), affordances at (2, 3) and (5, 5)
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
        )

        # Grid is first 64 dimensions (8×8)
        grid = obs[0, :64]

        # Agent position (0, 0) should be marked (value depends on if on affordance)
        agent_idx = 0 * grid_size + 0
        assert grid[agent_idx] == 1.0  # Agent at (0,0), no affordance there

        # Affordance positions should be marked
        bed_idx = 3 * grid_size + 2
        assert grid[bed_idx] == 1.0

        hospital_idx = 5 * grid_size + 5
        assert grid[hospital_idx] == 1.0

        job_idx = 7 * grid_size + 7
        assert grid[job_idx] == 1.0

        # Empty cell should be 0
        empty_idx = 4 * grid_size + 4
        assert grid[empty_idx] == 0.0

    def test_grid_marks_agent_on_affordance_with_value_2(self):
        """When agent is on affordance, grid cell has value=2.0 (agent + affordance)."""
        num_agents = 1
        grid_size = 8
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            num_affordance_types=1,
        )

        # Agent at (2, 3), bed also at (2, 3)
        positions = torch.tensor([[2, 3]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {"Bed": torch.tensor([2, 3], device=device)}

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
        )

        grid = obs[0, :64]

        # Agent is ON the bed, should have value 2.0
        position_idx = 3 * grid_size + 2
        assert grid[position_idx] == 2.0

    def test_partial_observability_still_shows_affordances_in_local_window(self):
        """Partial observability: local 5×5 window marks affordances within range."""
        num_agents = 1
        grid_size = 8
        vision_range = 2  # 5×5 window
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=True,
            vision_range=vision_range,
            enable_temporal_mechanics=False,
            num_affordance_types=1,
        )

        # Agent at (4, 4), bed at (5, 5) - within 5×5 window
        positions = torch.tensor([[4, 4]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {"Bed": torch.tensor([5, 5], device=device)}

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
        )

        # First 25 dims are the local 5×5 grid
        local_grid = obs[0, :25]

        # Bed at (5, 5) relative to agent at (4, 4) is offset (+1, +1)
        # In local coords: center is (2, 2), so bed is at (3, 3)
        bed_local_idx = 3 * 5 + 3
        assert local_grid[bed_local_idx] == 1.0

        # Center (agent position) is not marked in partial obs
        center_idx = 2 * 5 + 2
        assert local_grid[center_idx] == 0.0  # Agent position not in local grid


class TestTemporalFeatures:
    """Tests for temporal features always included in observations."""

    def test_temporal_features_always_present_even_when_disabled(self):
        """Temporal features (time_sin, time_cos, interaction_progress, lifetime_progress) always in obs."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,  # Disabled!
        )

        env.reset()
        obs = env._get_observations()

        # Observation should still include 4 temporal features at the end
        # Full obs: 64 grid + 8 meters + 15 affordance + 4 temporal = 91
        expected_dim = 64 + 8 + 15 + 4
        assert obs.shape == (1, expected_dim)

        # Last 4 dimensions are temporal features
        time_sin = obs[0, -4]
        time_cos = obs[0, -3]
        interaction_progress = obs[0, -2]
        lifetime_progress = obs[0, -1]

        # When temporal mechanics disabled, interaction_progress defaults to 0
        assert interaction_progress == 0.0

        # time_sin and time_cos should still be valid (time cycles naturally)
        assert -1.0 <= time_sin <= 1.0
        assert -1.0 <= time_cos <= 1.0

        # lifetime_progress should be 0 at start
        assert lifetime_progress == 0.0

    def test_time_encoding_is_cyclical_sincos(self):
        """Time is encoded as sin/cos so 23:00 and 00:00 are close."""
        num_agents = 1
        grid_size = 8
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=True,
            num_affordance_types=14,
        )

        positions = torch.tensor([[0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {}

        # Test midnight (0:00)
        obs_midnight = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            time_of_day=0,
        )

        time_sin_0 = obs_midnight[0, -4]
        time_cos_0 = obs_midnight[0, -3]

        # At midnight: angle = 0, so sin=0, cos=1
        assert abs(time_sin_0 - 0.0) < 1e-5
        assert abs(time_cos_0 - 1.0) < 1e-5

        # Test noon (12:00)
        obs_noon = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            time_of_day=12,
        )

        time_sin_12 = obs_noon[0, -4]
        time_cos_12 = obs_noon[0, -3]

        # At noon: angle = π, so sin≈0, cos≈-1
        assert abs(time_sin_12 - 0.0) < 1e-5
        assert abs(time_cos_12 - (-1.0)) < 1e-5

        # Test 18:00 (6pm)
        obs_evening = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            time_of_day=18,
        )

        time_sin_18 = obs_evening[0, -4]
        time_cos_18 = obs_evening[0, -3]

        # At 18:00: angle = 3π/2, so sin≈-1, cos≈0
        expected_angle = (18.0 / 24.0) * 2 * math.pi
        assert abs(time_sin_18 - math.sin(expected_angle)) < 1e-5
        assert abs(time_cos_18 - math.cos(expected_angle)) < 1e-5

    def test_interaction_progress_is_normalized(self):
        """interaction_progress is normalized to [0, 1] range (divided by 10)."""
        num_agents = 1
        grid_size = 8
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=True,
            num_affordance_types=14,
        )

        positions = torch.tensor([[0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {}

        # interaction_progress = 5 ticks (out of 10 max)
        interaction_progress = torch.tensor([5.0], device=device)

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            time_of_day=0,
            interaction_progress=interaction_progress,
        )

        # Normalized: 5 / 10 = 0.5
        assert obs[0, -2] == 0.5


class TestLifetimeProgress:
    """Tests for lifetime_progress observation and retirement mechanics."""

    def test_lifetime_progress_starts_at_zero(self):
        """lifetime_progress is 0.0 at episode start."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            agent_lifespan=1000,
        )

        env.reset()
        obs = env._get_observations()

        # Last dimension is lifetime_progress
        lifetime_progress = obs[0, -1]
        assert lifetime_progress == 0.0

    def test_lifetime_progress_increases_linearly(self):
        """lifetime_progress increases from 0 to 1 over agent_lifespan steps."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            agent_lifespan=100,  # Short lifespan for testing
        )

        env.reset()

        # Step 0: lifetime_progress = 0/100 = 0.0
        obs = env._get_observations()
        assert obs[0, -1] == 0.0

        # Take 50 steps
        actions = torch.tensor([4], device=torch.device("cpu"))  # WAIT action
        for _ in range(50):
            env.step(actions)

        # Step 50: lifetime_progress = 50/100 = 0.5
        obs = env._get_observations()
        assert abs(obs[0, -1] - 0.5) < 1e-5

        # Take another 49 steps (total 99)
        for _ in range(49):
            env.step(actions)

        # Step 99: lifetime_progress = 99/100 = 0.99
        obs = env._get_observations()
        assert abs(obs[0, -1] - 0.99) < 1e-2

    def test_retirement_at_agent_lifespan(self):
        """Episode ends when agent reaches agent_lifespan."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            agent_lifespan=50,  # Very short lifespan
        )

        env.reset()

        actions = torch.tensor([4], device=torch.device("cpu"))  # WAIT action

        # Step 49 times (not done yet)
        for _ in range(49):
            _, _, dones, _ = env.step(actions)
            assert not dones[0], f"Agent died early at step {env.step_counts[0]}"

        # Step 50: should retire
        _, rewards, dones, _ = env.step(actions)
        assert dones[0], "Agent should retire at agent_lifespan"
        assert env.step_counts[0] == 50

    def test_retirement_bonus_reward(self):
        """Agent receives +1 reward when retiring at agent_lifespan."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            agent_lifespan=50,
        )

        env.reset()

        actions = torch.tensor([4], device=torch.device("cpu"))

        # Step to retirement
        for _ in range(49):
            env.step(actions)

        # Final step: should get +1 retirement bonus
        _, rewards, dones, _ = env.step(actions)

        # Base reward is +1.0 for survival, +1.0 for retirement = +2.0 total
        assert rewards[0] >= 1.0, "Should receive at least +1 retirement bonus"
        assert dones[0], "Should be done after retirement"

    def test_agent_lifespan_default_value(self):
        """agent_lifespan defaults to 1000 if not specified."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            # agent_lifespan not specified
        )

        assert env.agent_lifespan == 1000

    def test_custom_agent_lifespan(self):
        """agent_lifespan can be customized."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            agent_lifespan=500,
        )

        assert env.agent_lifespan == 500

    def test_lifetime_progress_clamped_at_one(self):
        """lifetime_progress never exceeds 1.0 (clamped)."""
        num_agents = 1
        grid_size = 8
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            num_affordance_types=14,
        )

        positions = torch.tensor([[0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {}

        # Simulate lifetime_progress > 1.0 (should be clamped)
        # In practice this doesn't happen (episode ends at lifespan), but test the clamp
        lifetime_progress = torch.tensor([1.5], device=device)

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            lifetime_progress=lifetime_progress,
        )

        # Should be clamped to 1.0
        assert obs[0, -1] == 1.0


class TestObservationDimensions:
    """Tests for correct observation dimensions with new features."""

    def test_full_observability_dimension_with_temporal_and_lifetime(self):
        """Full observability: 64 grid + 8 meters + 15 affordance + 4 temporal = 91."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

        env.reset()
        obs = env._get_observations()

        # 64 grid + 8 meters + 15 affordance (14 types + 1 none) + 4 temporal
        expected_dim = 64 + 8 + 15 + 4
        assert obs.shape == (1, expected_dim)
        assert obs.shape[1] == 91

    def test_partial_observability_dimension_with_temporal_and_lifetime(self):
        """Partial observability: 25 local + 2 position + 8 meters + 15 affordance + 4 temporal = 54."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=True,
        )

        env.reset()
        obs = env._get_observations()

        # 25 local grid + 2 position + 8 meters + 15 affordance + 4 temporal
        expected_dim = 25 + 2 + 8 + 15 + 4
        assert obs.shape == (1, expected_dim)
        assert obs.shape[1] == 54

    def test_observation_dim_property_matches_actual_shape(self):
        """env.observation_dim property matches actual observation shape."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=True,
        )

        env.reset()
        obs = env._get_observations()

        assert obs.shape[1] == env.observation_dim


class TestBackwardCompatibility:
    """Tests that changes maintain backward compatibility with existing systems."""

    def test_temporal_features_default_to_zero_when_none(self):
        """When interaction_progress/lifetime_progress=None, they default to 0."""
        num_agents = 1
        grid_size = 8
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            num_affordance_types=14,
        )

        positions = torch.tensor([[0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {}

        # Don't pass interaction_progress or lifetime_progress
        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            time_of_day=0,
            # interaction_progress=None (default)
            # lifetime_progress=None (default)
        )

        # Should default to 0
        interaction_progress = obs[0, -2]
        lifetime_progress = obs[0, -1]

        assert interaction_progress == 0.0
        assert lifetime_progress == 0.0

    def test_multiple_agents_lifetime_progress(self):
        """lifetime_progress works correctly with multiple agents."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            agent_lifespan=100,
        )

        env.reset()

        # Step 10 times
        actions = torch.tensor([4, 4, 4], device=torch.device("cpu"))
        for _ in range(10):
            env.step(actions)

        obs = env._get_observations()

        # All agents should have lifetime_progress = 10/100 = 0.1
        for i in range(3):
            assert abs(obs[i, -1] - 0.1) < 1e-5
