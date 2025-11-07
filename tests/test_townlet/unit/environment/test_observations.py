"""Consolidated tests for observation construction across all modes.

This file consolidates observation tests from multiple old files:
- test_observation_builder.py: Basic construction and dimensions
- test_observation_dimensions.py: Dimension matching with networks
- test_observation_updates.py: Grid encoding, temporal features, lifetime progress

Tests cover:
- Full observability mode (8×8 grid one-hot encoding)
- Partial observability mode (5×5 vision window)
- Temporal feature encoding (time_sin, time_cos, interaction_progress, lifetime_progress)
- Observation updates across environment steps
- Multi-agent observation independence
- Vision window mechanics for POMDP
"""

import math
from pathlib import Path

import pytest
import torch
import yaml

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate


class TestFullObservability:
    """Test observation construction in full observability mode.

    Full observability observations now include a global occupancy grid:
    - Grid encoding: width × height cells (flattened occupancy map)
    - Position encoding: 2 dims (normalized x, y coordinates [0, 1])
    - Meters: 8 dims (normalized meter values)
    - Affordance encoding: 15 dims (14 types + 1 "none")
    - Temporal features: 4 dims (time_sin, time_cos, interaction_progress, lifetime_progress)
    """

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
    def test_dimension_matches_expected_formula(self, basic_env):
        """Full observability: grid + position + meters + affordance + temporal."""
        obs = basic_env.reset()

        grid_dim = basic_env.substrate.get_observation_dim()  # includes grid + position features
        expected_dim = grid_dim + basic_env.meter_count + (basic_env.num_affordance_types + 1) + 4
        assert obs.shape == (1, expected_dim)
        assert obs.shape[1] == expected_dim

    def test_observation_dim_property_matches_actual_shape(self, basic_env):
        """env.observation_dim property should match actual observation shape."""
        obs = basic_env.reset()
        assert obs.shape[1] == basic_env.observation_dim

    # REMOVED: test_grid_shows_agent_position - tested obsolete one-hot grid encoding
    # REMOVED: test_grid_shows_affordances_at_positions - tested obsolete one-hot grid encoding
    # REMOVED: test_agent_on_affordance_marked_with_value_2 - tested obsolete one-hot grid encoding

    def test_meters_are_included_and_normalized(self, basic_env):
        """Meter values should be included in observation (8 values after position)."""
        obs = basic_env.reset()

        grid_dim = basic_env.substrate.get_observation_dim()
        meters = obs[0, grid_dim : grid_dim + basic_env.meter_count]

        # Should have 8 meter values
        assert meters.shape[0] == 8

        # Meters should be normalized to [0, 1] range
        assert (meters >= 0.0).all()
        assert (meters <= 1.0).all()

    def test_affordance_encoding_is_one_hot(self, basic_env):
        """Affordance encoding should be one-hot (15 dims: 14 types + 1 "none")."""
        obs = basic_env.reset()

        grid_dim = basic_env.substrate.get_observation_dim()

        # Affordance encoding sits after grid + meters
        start = grid_dim + basic_env.meter_count
        end = start + basic_env.num_affordance_types + 1
        affordance = obs[0, start:end]

        # Should have 15 values (14 affordance types + 1 "none")
        assert affordance.shape[0] == basic_env.num_affordance_types + 1

        # Should be one-hot: sum = 1.0, all values 0 or 1
        assert affordance.sum() == 1.0
        assert ((affordance == 0.0) | (affordance == 1.0)).all()

    def test_grid_encoding_marks_agent_and_affordances(self, basic_env):
        """Global grid encoding should mark both agent and affordance locations."""
        obs = basic_env.reset()

        width = basic_env.substrate.width
        height = basic_env.substrate.height
        grid_cells = width * height

        grid_encoding = obs[0, :grid_cells]

        agent_x, agent_y = basic_env.positions[0].tolist()
        agent_idx = int(agent_y) * width + int(agent_x)

        assert grid_encoding[agent_idx] >= 1.0

        for affordance_pos in basic_env.affordances.values():
            if affordance_pos.numel() == 0:
                continue

            aff_x = int(affordance_pos[0].item())
            aff_y = int(affordance_pos[1].item())
            if not (0 <= aff_x < width and 0 <= aff_y < height):
                continue

            idx = aff_y * width + aff_x
            assert grid_encoding[idx] >= 1.0


class TestPartialObservability:
    """Test observation construction in POMDP mode (5×5 vision).

    Partial observability observations:
    - Local grid: 25 dims (5×5 vision window)
    - Position: 2 dims (normalized x, y)
    - Meters: 8 dims (normalized meter values)
    - Affordance encoding: 15 dims (14 types + 1 "none")
    - Temporal features: 4 dims (time_sin, time_cos, interaction_progress, lifetime_progress)
    Total: 25 + 2 + 8 + 15 + 4 = 54 dims
    """

    def test_dimension_matches_expected_formula(self, pomdp_env):
        """POMDP: 25 local + 2 position + 8 meters + 15 affordance + 4 temporal = 54."""
        obs = pomdp_env.reset()

        # Expected: 25 (local grid) + 2 (position) + 8 (meters) + 15 (affordance) + 4 (temporal)
        expected_dim = 25 + 2 + 8 + 15 + 4
        assert obs.shape == (1, expected_dim)
        assert obs.shape[1] == 54

    def test_observation_dim_property_matches_actual_shape(self, pomdp_env):
        """env.observation_dim property should match actual observation shape."""
        obs = pomdp_env.reset()
        assert obs.shape[1] == pomdp_env.observation_dim

    def test_vision_window_shows_nearby_affordances(self, pomdp_env):
        """POMDP: local 5×5 window marks affordances within range."""
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
            affordance_names=["Bed"],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
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
        assert local_grid[center_idx] == 0.0

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
    def test_vision_window_does_not_show_distant_affordances(self, pomdp_env):
        """POMDP: affordances outside vision window should not be visible."""
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
            affordance_names=["Hospital"],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
        )

        # Agent at (0, 0), hospital at (7, 7) - outside 5×5 window
        positions = torch.tensor([[0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {"Hospital": torch.tensor([7, 7], device=device)}

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
        )

        # First 25 dims are the local 5×5 grid
        local_grid = obs[0, :25]

        # Hospital is too far away, should not appear in local grid
        assert (local_grid == 0.0).all()

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
    def test_position_is_normalized(self, pomdp_env):
        """POMDP: agent position should be normalized to [0, 1] range."""
        obs = pomdp_env.reset()

        # Position is indices 25:27
        position = obs[0, 25:27]

        # Should have 2 values (x, y)
        assert position.shape[0] == 2

        # Should be normalized to [0, 1]
        assert (position >= 0.0).all()
        assert (position <= 1.0).all()

    def test_vision_window_size_is_5x5(self, pomdp_env):
        """POMDP with vision_range=2 should produce 5×5 window (2*2+1)."""
        obs = pomdp_env.reset()

        # First 25 dims are the local grid (5×5)
        local_grid = obs[0, :25]
        assert local_grid.shape[0] == 25


class TestPartialObservabilityWindowDimensions:
    """Validate POMDP window sizing across substrates."""

    def test_grid3d_window_dimension_matches_position_dim(self, temp_config_pack: Path, device: torch.device) -> None:
        """Grid3D should produce W³ local footprint for vision window."""
        substrate_path = temp_config_pack / "substrate.yaml"
        cubic_config = {
            "version": "1.0",
            "description": "Test cubic substrate for window dimension checks",
            "type": "grid",
            "grid": {
                "topology": "cubic",
                "width": 5,
                "height": 5,
                "depth": 5,
                "boundary": "clamp",
                "distance_metric": "manhattan",
                "observation_encoding": "relative",
            },
        }
        with substrate_path.open("w") as fh:
            yaml.safe_dump(cubic_config, fh)

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=True,
            vision_range=1,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=100,
            config_pack_path=temp_config_pack,
            device=device,
        )

        obs = env.reset()
        window_size = 2 * env.vision_range + 1
        expected_window_dim = window_size**env.substrate.position_dim
        expected_total_dim = expected_window_dim + env.substrate.position_dim + env.meter_count + (env.num_affordance_types + 1) + 4

        assert env.substrate.position_dim == 3
        assert env.observation_dim == expected_total_dim
        assert obs.shape == (env.num_agents, expected_total_dim)
        assert obs.shape[1] == expected_total_dim

    def test_aspatial_partial_observability_rejected(self, temp_config_pack: Path, device: torch.device) -> None:
        """Aspatial substrates must reject partial observability."""
        substrate_path = temp_config_pack / "substrate.yaml"
        aspatial_config = {
            "version": "1.0",
            "description": "Aspatial substrate for validation",
            "type": "aspatial",
            "aspatial": {},
        }
        with substrate_path.open("w") as fh:
            yaml.safe_dump(aspatial_config, fh)

        with pytest.raises(ValueError, match="Partial observability \\(POMDP\\) is not supported for aspatial substrates"):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=1,
                partial_observability=True,
                vision_range=1,
                enable_temporal_mechanics=False,
                move_energy_cost=0.005,
                wait_energy_cost=0.001,
                interact_energy_cost=0.0,
                agent_lifespan=100,
                config_pack_path=temp_config_pack,
                device=device,
            )

    def test_continuous_partial_observability_rejected(self, temp_config_pack: Path, device: torch.device) -> None:
        """Continuous substrates must reject partial observability."""
        substrate_path = temp_config_pack / "substrate.yaml"
        continuous_config = {
            "version": "1.0",
            "description": "Continuous substrate for validation",
            "type": "continuous",
            "continuous": {
                "dimensions": 1,
                "bounds": [[0.0, 10.0]],
                "boundary": "clamp",
                "movement_delta": 1.0,
                "interaction_radius": 0.5,
                "distance_metric": "euclidean",
                "observation_encoding": "relative",
            },
        }
        with substrate_path.open("w") as fh:
            yaml.safe_dump(continuous_config, fh)

        with pytest.raises(ValueError, match="Partial observability \\(POMDP\\) is not supported for continuous substrates"):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=1,
                partial_observability=True,
                vision_range=1,
                enable_temporal_mechanics=False,
                move_energy_cost=0.005,
                wait_energy_cost=0.001,
                interact_energy_cost=0.0,
                agent_lifespan=100,
                config_pack_path=temp_config_pack,
                device=device,
            )


class TestTemporalFeatures:
    """Test temporal feature encoding (time_of_day, interaction_progress, lifetime_progress).

    Temporal features (always present for forward compatibility):
    - time_sin: sin(2π * time_of_day / 24) - cyclical time encoding
    - time_cos: cos(2π * time_of_day / 24) - cyclical time encoding
    - interaction_progress: ticks_completed / 10 - normalized to [0, 1]
    - lifetime_progress: step_count / agent_lifespan - normalized to [0, 1]
    """

    def test_temporal_features_always_present(self, basic_env):
        """Temporal features (4 dims) always present, even when temporal mechanics disabled."""
        # basic_env has enable_temporal_mechanics=False
        obs = basic_env.reset()

        # Observation should still include 4 temporal features at the end
        expected_dim = basic_env.observation_dim
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
            affordance_names=[
                "Bed",
                "Bathroom",
                "Shower",
                "Fridge",
                "Microwave",
                "Gym",
                "Hospital",
                "Job",
                "Bar",
                "CoffeeShop",
                "Restaurant",
                "Park",
                "SocialClub",
                "MeditationCenter",
            ],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
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

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
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
            affordance_names=[
                "Bed",
                "Bathroom",
                "Shower",
                "Fridge",
                "Microwave",
                "Gym",
                "Hospital",
                "Job",
                "Bar",
                "CoffeeShop",
                "Restaurant",
                "Park",
                "SocialClub",
                "MeditationCenter",
            ],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
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

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
    def test_lifetime_progress_starts_at_zero(self, basic_env):
        """lifetime_progress is 0.0 at episode start."""
        obs = basic_env.reset()

        # Last dimension is lifetime_progress
        lifetime_progress = obs[0, -1]
        assert lifetime_progress == 0.0

    def test_lifetime_progress_is_clamped_at_one(self):
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
            affordance_names=[
                "Bed",
                "Bathroom",
                "Shower",
                "Fridge",
                "Microwave",
                "Gym",
                "Hospital",
                "Job",
                "Bar",
                "CoffeeShop",
                "Restaurant",
                "Park",
                "SocialClub",
                "MeditationCenter",
            ],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
        )

        positions = torch.tensor([[0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {}

        # Simulate lifetime_progress > 1.0 (should be clamped)
        lifetime_progress = torch.tensor([1.5], device=device)

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
            lifetime_progress=lifetime_progress,
        )

        # Should be clamped to 1.0
        assert obs[0, -1] == 1.0


class TestObservationUpdates:
    """Test that observations change correctly across environment steps.

    Tests behavioral changes in observations:
    - Movement updates grid position (full obs) or vision window (POMDP)
    - Interaction updates meters
    - Time progresses through steps
    - Lifetime progress increases linearly
    """

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
    def test_movement_updates_grid_position_full_obs(self, test_config_pack_path, cpu_device):
        """Full observability: moving agent updates position encoding."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        # Use CPU device for determinism
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        env.reset()
        # Force agent to center position where all movements are valid
        env.positions[0] = torch.tensor([4, 4], device=cpu_device, dtype=torch.long)

        obs1 = env._get_observations()
        grid_cells = env.substrate.width * env.substrate.height
        grid_dim_total = env.substrate.get_observation_dim()
        position_slice = slice(grid_cells, grid_dim_total)
        position1 = obs1[0, position_slice]

        # Move UP (guaranteed valid from center)
        actions = torch.tensor([0], device=cpu_device)
        obs2, _, _, _ = env.step(actions)
        position2 = obs2[0, position_slice]

        # Position MUST change (agent moved from (4,4) to (3,4))
        assert not torch.equal(position1, position2), "Position should update after movement"

    def test_movement_updates_vision_window_pomdp(self, test_config_pack_path, cpu_device):
        """POMDP: moving agent updates vision window contents."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        # Use CPU device for determinism
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=True,
            vision_range=2,  # 5×5 window
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        env.reset()
        # Force agent to center position
        env.positions[0] = torch.tensor([4, 4], device=cpu_device, dtype=torch.long)

        obs1 = env._get_observations()
        _local_grid1 = obs1[0, :25]
        position1 = obs1[0, 25:27]

        # Move RIGHT (guaranteed valid from center)
        actions = torch.tensor([3], device=cpu_device)  # RIGHT
        obs2, _, _, _ = env.step(actions)
        _local_grid2 = obs2[0, :25]
        position2 = obs2[0, 25:27]

        # Position MUST change (moved from (4,4) to (4,5))
        assert not torch.equal(position1, position2), "Position should update"

    def test_meters_update_after_interactions(self, basic_env):
        """Interacting with affordances should change meter values."""
        obs1 = basic_env.reset()
        grid_dim_total = basic_env.substrate.get_observation_dim()
        meter_slice = slice(grid_dim_total, grid_dim_total + basic_env.meter_count)
        meters1 = obs1[0, meter_slice]

        # Take several steps to allow interactions
        for _ in range(10):
            obs, _, dones, _ = basic_env.step(torch.tensor([4], device=basic_env.device))  # INTERACT
            if dones[0]:
                break

        meters_final = obs[0, meter_slice]

        # Meters should have changed (energy cost, possible interactions)
        # At minimum, energy should have decreased
        assert meters_final[0] < meters1[0]  # Energy decreased

    def test_lifetime_progress_increases_linearly(self, basic_env, test_config_pack_path):
        """lifetime_progress increases from 0 to 1 over agent_lifespan steps."""
        # Note: basic_env has default agent_lifespan=1000, too long for test
        # Create custom env with short lifespan
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=basic_env.device,
            partial_observability=False,
            agent_lifespan=100,
            config_pack_path=test_config_pack_path,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
        )

        env.reset()

        # Step 0: lifetime_progress = 0/100 = 0.0
        obs = env._get_observations()
        assert obs[0, -1] == 0.0

        # Take 50 steps
        actions = torch.tensor([4], device=env.device)  # INTERACT
        for _ in range(50):
            env.step(actions)

        # Step 50: lifetime_progress = 50/100 = 0.5
        obs = env._get_observations()
        assert abs(obs[0, -1] - 0.5) < 1e-5

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
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
            affordance_names=[
                "Bed",
                "Bathroom",
                "Shower",
                "Fridge",
                "Microwave",
                "Gym",
                "Hospital",
                "Job",
                "Bar",
                "CoffeeShop",
                "Restaurant",
                "Park",
                "SocialClub",
                "MeditationCenter",
            ],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
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
        )

        # Should default to 0
        interaction_progress = obs[0, -2]
        lifetime_progress = obs[0, -1]

        assert interaction_progress == 0.0
        assert lifetime_progress == 0.0


class TestMultiAgentObservations:
    """Test that agents have independent observations in multi-agent scenarios.

    Tests:
    - Different agent positions produce different observations (POMDP)
    - Same positions produce same observations
    - Batch dimension is correct
    - All agents receive valid observations simultaneously
    """

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
    def test_different_positions_produce_different_observations_pomdp(self):
        """POMDP: agents at different positions should have different observations."""
        num_agents = 2
        grid_size = 8
        vision_range = 2
        device = torch.device("cpu")

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=True,
            vision_range=vision_range,
            enable_temporal_mechanics=False,
            num_affordance_types=1,
            affordance_names=["Bed"],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
        )

        # Agents at different positions, bed near first agent
        positions = torch.tensor([[3, 3], [7, 7]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {"Bed": torch.tensor([4, 4], device=device)}

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
        )

        # Agents should have different positions
        position_0 = obs[0, 25:27]
        position_1 = obs[1, 25:27]
        assert not torch.equal(position_0, position_1)

        # First agent at (3,3) should see bed at (4,4) - within range
        # Second agent at (7,7) should NOT see bed - out of range
        local_grid_0 = obs[0, :25]
        local_grid_1 = obs[1, :25]

        # At least one should have the bed visible
        assert local_grid_0.sum() > 0.0 or local_grid_1.sum() > 0.0

    @pytest.mark.skip(reason="TODO VFS: Rewrite to test via environment")
    def test_same_position_produces_same_observations_full_obs(self):
        """Full observability: agents at same position should have identical observations."""
        num_agents = 2
        grid_size = 8
        device = torch.device("cpu")
        num_affordance_types = 1

        builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            num_affordance_types=num_affordance_types,
            affordance_names=["Bed"],
            substrate=Grid2DSubstrate(width=grid_size, height=grid_size, boundary="clamp", distance_metric="manhattan"),
        )

        # Both agents at same position (0, 0)
        positions = torch.tensor([[0, 0], [0, 0]], device=device)
        meters = torch.zeros(num_agents, 8, device=device)
        affordances = {"Bed": torch.tensor([4, 4], device=device)}

        obs = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
        )

        # With global grid encoding, observations should be identical across agents
        grid_dim_total = builder.substrate.get_observation_dim()
        expected_dim = grid_dim_total + meters.shape[1] + (num_affordance_types + 1) + 4
        assert obs.shape == (2, expected_dim)
        # Both observations should be identical (same position, same meters, same affordance)
        assert torch.equal(obs[0], obs[1])

    def test_batch_dimension_is_correct(self, multi_agent_env):
        """Multi-agent environment should produce batched observations."""
        obs = multi_agent_env.reset()

        # Should have 4 agents
        assert obs.shape[0] == 4

        # Each observation should have correct dimension
        assert obs.shape[1] == multi_agent_env.observation_dim

    def test_all_agents_receive_valid_observations(self, multi_agent_env):
        """All agents should receive valid observations simultaneously."""
        obs = multi_agent_env.reset()

        # All observations should be valid (no NaN, no inf)
        assert not torch.isnan(obs).any()
        assert not torch.isinf(obs).any()

        # All observations should have reasonable values
        # Grid: 0, 1, or 2
        # Meters: [0, 1]
        # Affordance: 0 or 1
        # Temporal: [-1, 1] for sin/cos, [0, 1] for progress
        assert obs.min() >= -1.0
        assert obs.max() <= 2.0

    def test_multiple_agents_lifetime_progress(self, test_config_pack_path):
        """lifetime_progress works correctly with multiple agents."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=100,
            device=torch.device("cpu"),
            config_pack_path=test_config_pack_path,
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


class TestDimensionConsistency:
    """Test dimension consistency across different modes and configurations.

    Validates that observation dimensions match expected formulas and remain
    consistent across environment resets and configuration changes.
    """

    def test_full_observability_with_temporal_mechanics(self, temporal_env):
        """Full obs + temporal retains grid, position, meter, affordance, temporal dims."""
        obs = temporal_env.reset()

        # Same dimension as without temporal (temporal features always present)
        expected_dim = temporal_env.observation_dim
        assert obs.shape == (1, expected_dim)

    def test_pomdp_with_temporal_mechanics(self, test_config_pack_path):
        """POMDP + temporal: 25 local + 2 pos + 8 meters + 15 affordance + 4 temporal = 54."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=True,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
            config_pack_path=test_config_pack_path,
        )

        obs = env.reset()

        # 25 local grid + 2 position + 8 meters + 15 affordance + 4 temporal
        expected_dim = 25 + 2 + 8 + 15 + 4
        assert obs.shape == (1, expected_dim)

    def test_observation_dim_matches_across_resets(self, basic_env):
        """Observation dimension should remain consistent across environment resets."""
        obs1 = basic_env.reset()
        dim1 = obs1.shape[1]

        obs2 = basic_env.reset()
        dim2 = obs2.shape[1]

        assert dim1 == dim2
        assert dim1 == basic_env.observation_dim

    def test_affordance_encoding_size_matches_vocabulary(self, basic_env):
        """Affordance encoding should match full vocabulary size (not deployed affordances)."""
        # basic_env may have fewer affordances deployed than in vocabulary
        # But affordance encoding should always be (num_affordance_types + 1)

        obs = basic_env.reset()

        # Affordance encoding is 15 dims (14 types + 1 "none")
        # Indices 10:25 (after 2 position + 8 meters)
        affordance = obs[0, 10:25]
        assert affordance.shape[0] == basic_env.num_affordance_types + 1
        assert affordance.shape[0] == 15
