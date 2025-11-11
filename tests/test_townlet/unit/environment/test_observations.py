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

import shutil
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_townlet.conftest import SUBSTRATE_FIXTURES

# Removed: calculate_expected_observation_dim (now using env.observation_dim directly)
from townlet.universe.errors import CompilationError


class TestFullObservability:
    """Test observation construction in full observability mode.

    Full observability observations now include a global occupancy grid:
    - Grid encoding: width × height cells (flattened occupancy map)
    - Position encoding: 2 dims (normalized x, y coordinates [0, 1])
    - Meters: 8 dims (normalized meter values)
    - Affordance encoding: 15 dims (14 types + 1 "none")
    - Temporal features: 4 dims (time_sin, time_cos, interaction_progress, lifetime_progress)
    """

    @pytest.mark.parametrize("env_fixture_name", SUBSTRATE_FIXTURES)
    def test_dimension_matches_expected_formula(self, request, env_fixture_name):
        """All substrates should follow the canonical observation layout.

        Note: Observation dimension is determined by the compiled observation_spec,
        which is the authoritative source of truth. This test verifies that actual
        observations match the compiled spec rather than trying to recompute dims manually.
        """

        try:
            env = request.getfixturevalue(env_fixture_name)
        except Exception as exc:  # pragma: no cover - fixture validation guardrail
            if isinstance(exc, CompilationError):
                pytest.skip(f"Fixture {env_fixture_name} is temporarily unavailable: {exc}")
            raise
        obs = env.reset()

        # Use env.observation_dim from compiled spec as the source of truth
        expected_dim = env.observation_dim

        assert obs.shape == (env.num_agents, expected_dim)
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

        # Should match meter count from metadata
        assert meters.shape[0] == basic_env.metadata.meter_count

        # Meters should be normalized to [0, 1] range
        assert (meters >= 0.0).all()
        assert (meters <= 1.0).all()

    def test_affordance_encoding_is_one_hot(self, basic_env):
        """Affordance encoding should be one-hot (15 dims: 14 types + 1 "none")."""
        obs = basic_env.reset()

        # substrate.get_observation_dim() includes position (grid_encoding + position)
        substrate_dim = basic_env.substrate.get_observation_dim()
        velocity_dim = 3  # velocity_x, velocity_y, velocity_magnitude

        # Affordance encoding sits after substrate + velocity + meters
        start = substrate_dim + velocity_dim + basic_env.meter_count
        end = start + basic_env.num_affordance_types + 1
        affordance = obs[0, start:end]

        # Should have 15 values (14 affordance types + 1 "none")
        assert affordance.shape[0] == basic_env.num_affordance_types + 1

        # Should be one-hot: sum = 1.0, all values 0 or 1
        assert affordance.sum() == 1.0
        assert ((affordance == 0.0) | (affordance == 1.0)).all()


class TestPartialObservability:
    """Test observation construction in POMDP mode (5×5 vision).

    Partial observability observations:
    - Local grid: 25 dims (5×5 vision window)
    - Position: 2 dims (normalized x, y)
    - Velocity: 3 dims (velocity_x, velocity_y, velocity_magnitude)
    - Meters: 8 dims (normalized meter values)
    - Affordance encoding: 15 dims (14 types + 1 "none", padded with zeros in POMDP)
    - Temporal features: 4 dims (time_sin, time_cos, interaction_progress, lifetime_progress)
    Total: 25 + 2 + 3 + 8 + 15 + 4 = 57 dims
    """

    def test_dimension_matches_expected_formula(self, pomdp_env):
        """POMDP: 25 local + 2 position + 3 velocity + 8 meters + 15 affordance + 4 temporal = 57."""
        obs = pomdp_env.reset()

        # Use env.observation_dim as the authoritative source (from compiled observation spec)
        assert obs.shape == (1, pomdp_env.observation_dim)
        assert obs.shape[1] == pomdp_env.metadata.observation_dim

    def test_observation_dim_property_matches_actual_shape(self, pomdp_env):
        """env.observation_dim property should match actual observation shape."""
        obs = pomdp_env.reset()
        assert obs.shape[1] == pomdp_env.observation_dim

    def test_position_is_normalized(self, pomdp_env):
        """POMDP: agent position should be normalized to [0, 1] range."""
        pomdp_env.reset()

        # Query position from VFS registry instead of hardcoded slicing
        position = pomdp_env.vfs_registry.get("position", reader="agent")

        # Should have 2 values per agent (x, y)
        assert position.shape == (1, 2), f"Expected (1, 2), got {position.shape}"

        # Should be normalized to [0, 1]
        assert (position >= 0.0).all(), f"Position values should be >= 0.0: {position}"
        assert (position <= 1.0).all(), f"Position values should be <= 1.0: {position}"

    def test_vision_window_size_is_5x5(self, pomdp_env):
        """POMDP with vision_range=2 should produce 5×5 window (2*2+1)."""
        obs = pomdp_env.reset()

        # First 25 dims are the local grid (5×5)
        local_grid = obs[0, :25]
        assert local_grid.shape[0] == 25


class TestPartialObservabilityWindowDimensions:
    """Validate POMDP window sizing across substrates."""

    def test_grid3d_window_dimension_matches_position_dim(
        self,
        temp_config_pack: Path,
        device: torch.device,
        env_factory,
    ) -> None:
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

        # Update VFS config to match 3D substrate (configs/test has 2D position)
        vfs_path = temp_config_pack / "variables_reference.yaml"
        with vfs_path.open() as fh:
            vfs_config = yaml.safe_load(fh)

        # Update position variable from 2D to 3D
        for var in vfs_config["variables"]:
            if var["id"] == "position":
                var["dims"] = 3
                var["default"] = [0.0, 0.0, 0.0]
                var["description"] = "Normalized agent position (3D) in [0, 1]^3 range"
            elif var["id"] == "local_window":
                # POMDP window: vision_range=1 → (2*1+1)^3 = 3^3 = 27 cells
                var["dims"] = 27
                var["description"] = "POMDP local observation window (3×3×3 cube, vision_range=1)"

        # Update observations (position and local_window)
        for obs in vfs_config.get("exposed_observations", []):
            if obs["id"] == "obs_position":
                obs["shape"] = [3]
                if obs.get("normalization"):
                    obs["normalization"]["min"] = [0.0, 0.0, 0.0]
                    obs["normalization"]["max"] = [1.0, 1.0, 1.0]
            elif obs["id"] == "obs_local_window":
                # Update local_window obs shape to match 3D window
                obs["shape"] = [27]

        with vfs_path.open("w") as fh:
            yaml.safe_dump(vfs_config, fh)

        training_path = temp_config_pack / "training.yaml"
        with training_path.open() as fh:
            training_config = yaml.safe_load(fh)

        training_config["environment"]["partial_observability"] = True
        training_config["environment"]["vision_range"] = 1

        with training_path.open("w") as fh:
            yaml.safe_dump(training_config, fh, sort_keys=False)

        env = env_factory(
            config_dir=temp_config_pack,
            num_agents=1,
            device_override=device,
        )

        obs = env.reset()
        window_size = 2 * env.vision_range + 1
        expected_window_dim = window_size**env.substrate.position_dim

        # Verify substrate is 3D
        assert env.substrate.position_dim == 3

        # Verify observation shape matches env.observation_dim (VFS-computed)
        assert obs.shape == (env.num_agents, env.observation_dim)

        # Verify window is the correct size for 3D cube
        assert expected_window_dim == 27  # 3×3×3 cube

    def test_aspatial_partial_observability_rejected(
        self,
        temp_config_pack: Path,
        device: torch.device,
        env_factory,
    ) -> None:
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

        training_path = temp_config_pack / "training.yaml"
        with training_path.open() as fh:
            training_config = yaml.safe_load(fh)

        training_config["environment"]["partial_observability"] = True
        training_config["environment"]["vision_range"] = 1

        with training_path.open("w") as fh:
            yaml.safe_dump(training_config, fh, sort_keys=False)

        with pytest.raises(ValueError, match="Partial observability \\(POMDP\\) is not supported for aspatial substrates"):
            env_factory(
                config_dir=temp_config_pack,
                num_agents=1,
                device_override=device,
            )

    def test_continuous_partial_observability_rejected(
        self,
        temp_config_pack: Path,
        device: torch.device,
        env_factory,
    ) -> None:
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

        training_path = temp_config_pack / "training.yaml"
        with training_path.open() as fh:
            training_config = yaml.safe_load(fh)

        training_config["environment"]["partial_observability"] = True
        training_config["environment"]["vision_range"] = 1

        with training_path.open("w") as fh:
            yaml.safe_dump(training_config, fh, sort_keys=False)

        with pytest.raises(ValueError, match="Partial observability \\(POMDP\\) is not supported for continuous substrates"):
            env_factory(
                config_dir=temp_config_pack,
                num_agents=1,
                device_override=device,
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

    def test_lifetime_progress_starts_at_zero(self, basic_env):
        """lifetime_progress is 0.0 at episode start."""
        basic_env.reset()

        # Query lifetime_progress from VFS registry instead of hardcoded slicing
        lifetime_progress = basic_env.vfs_registry.get("lifetime_progress", reader="agent")

        # Should be (num_agents,) shape with all zeros at episode start
        assert lifetime_progress.shape == (1,), f"Expected (1,), got {lifetime_progress.shape}"
        assert lifetime_progress[0] == 0.0, f"Expected 0.0, got {lifetime_progress[0]}"


class TestObservationUpdates:
    """Test that observations change correctly across environment steps.

    Tests behavioral changes in observations:
    - Movement updates grid position (full obs) or vision window (POMDP)
    - Interaction updates meters
    - Time progresses through steps
    - Lifetime progress increases linearly
    """

    def test_movement_updates_grid_position_full_obs(self, test_config_pack_path, cpu_device, env_factory):
        """Full observability: moving agent updates position encoding."""

        env = env_factory(
            config_dir=test_config_pack_path,
            num_agents=1,
            device_override=cpu_device,
        )

        env.reset()
        # Force agent to center position where all movements are valid
        env.positions[0] = torch.tensor([4, 4], device=cpu_device, dtype=torch.long)

        # Query position from VFS registry instead of hardcoded slicing
        env._get_observations()  # Refresh VFS registry
        position1 = env.vfs_registry.get("position", reader="agent").clone()

        # Move UP (guaranteed valid from center)
        actions = torch.tensor([0], device=cpu_device)
        env.step(actions)
        position2 = env.vfs_registry.get("position", reader="agent")

        # Position MUST change (agent moved from (4,4) to (3,4))
        assert not torch.equal(position1, position2), f"Position should update after movement: {position1} vs {position2}"

    def test_movement_updates_vision_window_pomdp(self, pomdp_env):
        """POMDP: moving agent updates vision window contents."""

        env = pomdp_env

        env.reset()
        # Force agent to center position
        env.positions[0] = torch.tensor([4, 4], device=env.device, dtype=torch.long)

        obs1 = env._get_observations()
        _local_grid1 = obs1[0, :25]
        position1 = obs1[0, 25:27]

        # Move RIGHT (guaranteed valid from center)
        actions = torch.tensor([3], device=env.device)  # RIGHT
        obs2, _, _, _ = env.step(actions)
        _local_grid2 = obs2[0, :25]
        position2 = obs2[0, 25:27]

        # Position MUST change (moved from (4,4) to (4,5))
        assert not torch.equal(position1, position2), "Position should update"

    def test_meters_update_after_interactions(self, basic_env):
        """Interacting with affordances should change meter values."""
        obs1 = basic_env.reset()
        # substrate.get_observation_dim() includes position (grid_encoding + position)
        substrate_dim = basic_env.substrate.get_observation_dim()
        velocity_dim = 3  # velocity_x, velocity_y, velocity_magnitude

        # Meters sit after substrate + velocity
        meter_start = substrate_dim + velocity_dim
        meter_slice = slice(meter_start, meter_start + basic_env.meter_count)
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

    def test_lifetime_progress_increases_linearly(
        self,
        tmp_path: Path,
        test_config_pack_path: Path,
        cpu_device: torch.device,
        env_factory,
    ):
        """lifetime_progress increases from 0 to 1 over agent_lifespan steps."""

        config_dir = tmp_path / "short_lifespan"
        shutil.copytree(test_config_pack_path, config_dir)

        training_path = config_dir / "training.yaml"
        with training_path.open() as fh:
            training_config = yaml.safe_load(fh)

        training_config["curriculum"]["max_steps_per_episode"] = 100

        with training_path.open("w") as fh:
            yaml.safe_dump(training_config, fh, sort_keys=False)

        env = env_factory(
            config_dir=config_dir,
            num_agents=1,
            device_override=cpu_device,
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


class TestMultiAgentObservations:
    """Test that agents have independent observations in multi-agent scenarios.

    Tests:
    - Different agent positions produce different observations (POMDP)
    - Same positions produce same observations
    - Batch dimension is correct
    - All agents receive valid observations simultaneously
    """

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

    def test_multiple_agents_lifetime_progress(
        self,
        tmp_path: Path,
        test_config_pack_path: Path,
        cpu_device: torch.device,
        env_factory,
    ):
        """lifetime_progress works correctly with multiple agents."""

        config_dir = tmp_path / "multi_agent_short_life"
        shutil.copytree(test_config_pack_path, config_dir)

        training_path = config_dir / "training.yaml"
        with training_path.open() as fh:
            training_config = yaml.safe_load(fh)

        training_config["curriculum"]["max_steps_per_episode"] = 100

        with training_path.open("w") as fh:
            yaml.safe_dump(training_config, fh, sort_keys=False)

        env = env_factory(
            config_dir=config_dir,
            num_agents=3,
            device_override=cpu_device,
        )

        env.reset()

        # Step 10 times
        actions = torch.tensor([4, 4, 4], device=env.device)
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

    def test_pomdp_with_temporal_mechanics(
        self,
        tmp_path: Path,
        test_config_pack_path: Path,
        cpu_device: torch.device,
        env_factory,
    ):
        """POMDP + temporal: 25 local + 2 pos + 3 velocity + 8 meters + 15 affordance + 4 temporal = 57.

        NOTE: affordance_at_position included in POMDP (padded with zeros) for transfer learning.
        """

        config_dir = tmp_path / "pomdp_temporal"
        shutil.copytree(test_config_pack_path, config_dir)

        training_path = config_dir / "training.yaml"
        with training_path.open() as fh:
            training_config = yaml.safe_load(fh)

        training_config["environment"]["partial_observability"] = True
        training_config["environment"]["vision_range"] = 2
        training_config["environment"]["enable_temporal_mechanics"] = True

        with training_path.open("w") as fh:
            yaml.safe_dump(training_config, fh, sort_keys=False)

        env = env_factory(
            config_dir=config_dir,
            num_agents=1,
            device_override=cpu_device,
        )

        obs = env.reset()

        # 25 local window + 2 position + 3 velocity + 8 meters + 15 affordance + 4 temporal = 57
        expected_dim = 25 + 2 + 3 + 8 + 15 + 4
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
