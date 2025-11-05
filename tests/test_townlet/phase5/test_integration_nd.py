"""Integration tests for GridND and ContinuousND with full system.

Tests verify that N-dimensional substrates work end-to-end with:
- Configuration loading and factory
- VectorizedHamletEnv
- ObservationBuilder
- Affordance interactions
- Action space handling
"""

import torch

from townlet.environment.observation_builder import ObservationBuilder
from townlet.substrate.config import SubstrateConfig
from townlet.substrate.continuousnd import ContinuousNDSubstrate
from townlet.substrate.factory import SubstrateFactory
from townlet.substrate.gridnd import GridNDSubstrate


class TestGridNDIntegration:
    """Integration tests for GridND with full system components."""

    def test_gridnd_4d_config_loading(self):
        """Test loading GridND config from YAML and factory creation."""
        config_dict = {
            "version": "1.0",
            "description": "4D grid for integration testing",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [8, 8, 8, 8],
                "boundary": "clamp",
                "distance_metric": "manhattan",
                "observation_encoding": "relative",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, GridNDSubstrate)
        assert substrate.position_dim == 4
        assert substrate.action_space_size == 10  # 2*4 + 2
        assert substrate.get_observation_dim() == 4

    def test_gridnd_with_observation_builder(self):
        """Test GridND observation encoding with ObservationBuilder."""
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        # Mock affordance data
        affordance_names = ["Bed", "Hospital", "Job", "Store"]
        affordances = {
            "Bed": torch.tensor([2, 2, 2, 2], dtype=torch.long),
            "Job": torch.tensor([4, 4, 4, 4], dtype=torch.long),
        }

        # Create observation builder
        builder = ObservationBuilder(
            num_agents=2,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            num_affordance_types=len(affordance_names),
            affordance_names=affordance_names,
            substrate=substrate,
        )

        # Test positions
        positions = torch.tensor(
            [[1, 1, 1, 1], [2, 2, 2, 2]],  # Agent 1: not on affordance
            dtype=torch.long,  # Agent 2: on Bed
        )

        # Build observations
        meters = torch.ones((2, 8))
        observations = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
        )

        # Verify observation shape
        # substrate encoding (4) + meters (8) + affordance encoding (5) + temporal (4)
        assert observations.shape == (2, 21)

        # Verify agent 2 is on Bed affordance
        # Bed is at index 0 in affordance_names
        agent2_affordance_idx = 4 + 8  # After substrate + meters (before temporal)
        assert observations[1, agent2_affordance_idx] == 1.0  # Bed is at index 0

    def test_gridnd_movement_integration(self):
        """Test GridND movement with action selection."""
        substrate = GridNDSubstrate(
            dimension_sizes=[6, 6, 6, 6],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        # Agent at [3, 3, 3, 3] in center
        positions = torch.tensor([[3, 3, 3, 3]], dtype=torch.long)

        # Move -1 in dimension 0
        # action = 2*dim + direction, where direction=0 is -1, direction=1 is +1
        delta = torch.zeros((1, 4), dtype=torch.float32)
        delta[0, 0] = -1.0

        new_positions = substrate.apply_movement(positions, delta)

        # Should move to [2, 3, 3, 3]
        assert torch.equal(new_positions, torch.tensor([[2, 3, 3, 3]], dtype=torch.long))

    def test_gridnd_distance_computation(self):
        """Test distance computation for affordance interactions."""
        substrate = GridNDSubstrate(
            dimension_sizes=[10, 10, 10, 10],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        agent_pos = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        affordance_pos = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

        distance = substrate.compute_distance(agent_pos, affordance_pos)

        # Manhattan: |2| + |3| + |4| + |5| = 14
        assert distance[0] == 14

    def test_gridnd_affordance_detection(self):
        """Test that agents detect affordances at their position."""
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        agents = torch.tensor(
            [[2, 2, 2, 2], [1, 1, 1, 1]],
            dtype=torch.long,
        )
        affordance = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        on_position = substrate.is_on_position(agents, affordance)

        assert on_position[0]  # Agent 0 is on affordance
        assert not on_position[1]  # Agent 1 is not

    def test_gridnd_multiple_encoding_modes(self):
        """Test that different observation encodings work."""
        dimension_sizes = [4, 5, 6, 7]

        # Test relative encoding
        substrate_rel = GridNDSubstrate(
            dimension_sizes=dimension_sizes,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        assert substrate_rel.get_observation_dim() == 4

        # Test scaled encoding
        substrate_scaled = GridNDSubstrate(
            dimension_sizes=dimension_sizes,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="scaled",
        )
        assert substrate_scaled.get_observation_dim() == 8  # 2 * num_dims

        # Test absolute encoding
        substrate_abs = GridNDSubstrate(
            dimension_sizes=dimension_sizes,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="absolute",
        )
        assert substrate_abs.get_observation_dim() == 4

        # Verify encodings produce different results
        positions = torch.tensor([[2, 2, 3, 3]], dtype=torch.long)

        enc_rel = substrate_rel.encode_observation(positions, {})
        enc_scaled = substrate_scaled.encode_observation(positions, {})

        # Relative and scaled should differ in shape
        assert enc_rel.shape != enc_scaled.shape
        # Relative encoding normalizes by (size - 1) for grids
        max_coords = torch.tensor([dim_size - 1 for dim_size in dimension_sizes], dtype=torch.float32)
        expected_rel = positions.float() / max_coords
        assert torch.allclose(enc_rel[0], expected_rel)

    def test_gridnd_boundary_modes_integration(self):
        """Test different boundary modes in movement."""
        # Test wrap boundary
        substrate_wrap = GridNDSubstrate(
            dimension_sizes=[3, 3, 3, 3],
            boundary="wrap",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        positions = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        delta = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        new_pos = substrate_wrap.apply_movement(positions, delta)

        # Should wrap to [2, 0, 0, 0]
        assert new_pos[0, 0] == 2

        # Test clamp boundary
        substrate_clamp = GridNDSubstrate(
            dimension_sizes=[3, 3, 3, 3],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        new_pos = substrate_clamp.apply_movement(positions, delta)

        # Should clamp to [0, 0, 0, 0]
        assert new_pos[0, 0] == 0

    def test_gridnd_action_space_adaptation(self):
        """Verify action space size adapts to dimensionality."""
        # 4D: 2*4 + 2 = 10
        substrate_4d = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        assert substrate_4d.action_space_size == 10

        # 6D: 2*6 + 2 = 14
        substrate_6d = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        assert substrate_6d.action_space_size == 14

        # 8D: 2*8 + 2 = 18
        substrate_8d = GridNDSubstrate(
            dimension_sizes=[4, 4, 4, 4, 4, 4, 4, 4],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        assert substrate_8d.action_space_size == 18


class TestContinuousNDIntegration:
    """Integration tests for ContinuousND with full system components."""

    def test_continuousnd_4d_config_loading(self):
        """Test loading ContinuousND config from YAML and factory creation."""
        config_dict = {
            "version": "1.0",
            "description": "4D continuous space for integration testing",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 4,
                "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                "boundary": "clamp",
                "movement_delta": 0.5,
                "interaction_radius": 1.0,
                "distance_metric": "euclidean",
                "observation_encoding": "relative",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, ContinuousNDSubstrate)
        assert substrate.position_dim == 4
        assert substrate.action_space_size == 10  # 2*4 + 2
        assert substrate.get_observation_dim() == 4

    def test_continuousnd_with_observation_builder(self):
        """Test ContinuousND observation encoding with ObservationBuilder."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        # Mock affordance data
        affordance_names = ["Bed", "Hospital", "Job", "Store"]
        affordances = {
            "Bed": torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float32),
            "Job": torch.tensor([8.0, 8.0, 8.0, 8.0], dtype=torch.float32),
        }

        # Create observation builder
        builder = ObservationBuilder(
            num_agents=2,
            grid_size=10,  # Not used for continuous, but needed for builder
            device=torch.device("cpu"),
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            num_affordance_types=len(affordance_names),
            affordance_names=affordance_names,
            substrate=substrate,
        )

        # Test positions
        positions = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            dtype=torch.float32,
        )

        # Build observations
        meters = torch.ones((2, 8))
        observations = builder.build_observations(
            positions=positions,
            meters=meters,
            affordances=affordances,
        )

        # Verify observation shape
        # substrate encoding (4) + meters (8) + affordance encoding (5) + temporal (4)
        assert observations.shape == (2, 21)

        # Verify agent 2 is on Bed affordance (within interaction_radius=1.0)
        # Euclidean distance from [2, 2, 2, 2] to [2, 2, 2, 2] is 0, so it's within radius
        agent2_affordance_idx = 4 + 8  # After substrate + meters (before temporal)
        assert observations[1, agent2_affordance_idx] == 1.0  # Bed is at index 0

    def test_continuousnd_movement_integration(self):
        """Test ContinuousND movement with action selection."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        # Agent at [5.0, 5.0, 5.0, 5.0]
        positions = torch.tensor([[5.0, 5.0, 5.0, 5.0]], dtype=torch.float32)

        # Move -1.0 in dimension 0 (will be scaled by movement_delta=0.5 to -0.5)
        delta = torch.zeros((1, 4), dtype=torch.float32)
        delta[0, 0] = -1.0

        new_positions = substrate.apply_movement(positions, delta)

        # Should move to [4.5, 5.0, 5.0, 5.0] (delta=-1.0 * 0.5 = -0.5)
        expected = torch.tensor([[4.5, 5.0, 5.0, 5.0]], dtype=torch.float32)
        assert torch.allclose(new_positions, expected)

    def test_continuousnd_distance_computation(self):
        """Test distance computation for affordance interactions in continuous space."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 100.0), (0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        agent_pos = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        affordance_pos = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float32)

        distance = substrate.compute_distance(agent_pos, affordance_pos)

        # Euclidean: sqrt(3^2 + 4^2 + 0^2 + 0^2) = 5.0
        assert torch.allclose(distance, torch.tensor([5.0]))

    def test_continuousnd_affordance_detection_radius(self):
        """Test that affordance detection respects interaction_radius."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.5,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        affordance = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float32)

        agents = torch.tensor(
            [
                [5.0, 5.0, 5.0, 5.0],  # Exact match
                [5.5, 5.5, 5.5, 5.5],  # sqrt(4*0.5^2) = sqrt(1.0) = 1.0 (within radius)
                [6.5, 6.5, 6.5, 6.5],  # sqrt(4*1.5^2) = sqrt(9.0) = 3.0 (outside radius)
            ],
            dtype=torch.float32,
        )

        on_position = substrate.is_on_position(agents, affordance)

        assert on_position[0]  # Exact match
        assert on_position[1]  # Within radius
        assert not on_position[2]  # Outside radius

    def test_continuousnd_multiple_encoding_modes(self):
        """Test different observation encodings for continuous space."""
        bounds = [(0.0, 100.0), (-50.0, 50.0), (10.0, 20.0), (-10.0, 30.0)]

        # Test relative encoding
        substrate_rel = ContinuousNDSubstrate(
            bounds=bounds,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )
        assert substrate_rel.get_observation_dim() == 4

        # Test scaled encoding
        substrate_scaled = ContinuousNDSubstrate(
            bounds=bounds,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="scaled",
        )
        assert substrate_scaled.get_observation_dim() == 8  # 2 * num_dims

        # Test absolute encoding
        substrate_abs = ContinuousNDSubstrate(
            bounds=bounds,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="absolute",
        )
        assert substrate_abs.get_observation_dim() == 4

        # Verify absolute encoding returns raw coordinates
        positions = torch.tensor([[50.0, 25.0, 15.0, 10.0]], dtype=torch.float32)
        enc_abs = substrate_abs.encode_observation(positions, {})
        assert torch.allclose(enc_abs, positions)

    def test_continuousnd_boundary_modes_integration(self):
        """Test different boundary modes in movement."""
        # Test wrap boundary
        substrate_wrap = ContinuousNDSubstrate(
            bounds=[(0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0)],
            boundary="wrap",
            movement_delta=1.0,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        positions = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        delta = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        new_pos = substrate_wrap.apply_movement(positions, delta)

        # Should wrap to [4.0, 0.0, 0.0, 0.0]
        assert torch.allclose(new_pos[0, 0], torch.tensor(4.0))

        # Test clamp boundary
        substrate_clamp = ContinuousNDSubstrate(
            bounds=[(0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0)],
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        new_pos = substrate_clamp.apply_movement(positions, delta)

        # Should clamp to [0.0, 0.0, 0.0, 0.0]
        assert torch.allclose(new_pos[0, 0], torch.tensor(0.0))

    def test_continuousnd_action_space_adaptation(self):
        """Verify action space size adapts to dimensionality."""
        # 4D: 2*4 + 2 = 10
        substrate_4d = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )
        assert substrate_4d.action_space_size == 10

        # 6D: 2*6 + 2 = 14
        substrate_6d = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 6,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )
        assert substrate_6d.action_space_size == 14

        # 8D: 2*8 + 2 = 18
        substrate_8d = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 8,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )
        assert substrate_8d.action_space_size == 18

    def test_continuousnd_random_initialization(self):
        """Test random position initialization respects bounds."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 100.0), (-50.0, 50.0), (10.0, 20.0), (-10.0, 30.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        positions = substrate.initialize_positions(num_agents=20, device=torch.device("cpu"))

        assert positions.shape == (20, 4)
        assert positions.dtype == torch.float32

        # Verify all positions are within bounds
        assert torch.all(positions[:, 0] >= 0.0) and torch.all(positions[:, 0] <= 100.0)
        assert torch.all(positions[:, 1] >= -50.0) and torch.all(positions[:, 1] <= 50.0)
        assert torch.all(positions[:, 2] >= 10.0) and torch.all(positions[:, 2] <= 20.0)
        assert torch.all(positions[:, 3] >= -10.0) and torch.all(positions[:, 3] <= 30.0)


class TestNDSubstrateInteroperability:
    """Test that N-D substrates interoperate with existing system components."""

    def test_gridnd_with_multiple_affordances(self):
        """Test GridND with multiple affordances at different positions."""
        substrate = GridNDSubstrate(
            dimension_sizes=[6, 6, 6, 6],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        affordances = {
            "Bed": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            "Hospital": torch.tensor([3, 3, 3, 3], dtype=torch.long),
            "Job": torch.tensor([5, 5, 5, 5], dtype=torch.long),
            "Store": torch.tensor([2, 2, 2, 2], dtype=torch.long),
        }

        agents = torch.tensor(
            [[1, 1, 1, 1], [3, 3, 3, 3], [5, 5, 5, 5], [0, 0, 0, 0]],
            dtype=torch.long,
        )

        # Check each agent's affordance
        for idx, agent_pos in enumerate(agents):
            for aff_name, aff_pos in affordances.items():
                on_affordance = substrate.is_on_position(agent_pos.unsqueeze(0), aff_pos)
                if idx == 0:
                    assert on_affordance[0] == (aff_name == "Bed")
                elif idx == 1:
                    assert on_affordance[0] == (aff_name == "Hospital")
                elif idx == 2:
                    assert on_affordance[0] == (aff_name == "Job")
                else:
                    assert not on_affordance[0]

    def test_continuousnd_with_multiple_affordances(self):
        """Test ContinuousND with multiple affordances at different positions."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.5,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        affordances = {
            "Bed": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            "Hospital": torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float32),
            "Job": torch.tensor([9.0, 9.0, 9.0, 9.0], dtype=torch.float32),
            "Store": torch.tensor([5.0, 1.0, 5.0, 1.0], dtype=torch.float32),
        }

        agents = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # On Bed
                [5.0, 5.0, 5.0, 5.0],  # On Hospital
                [9.0, 9.0, 9.0, 9.0],  # On Job
                [0.0, 0.0, 0.0, 0.0],  # On none
            ],
            dtype=torch.float32,
        )

        # Check each agent's affordances
        for idx, agent_pos in enumerate(agents):
            for aff_name, aff_pos in affordances.items():
                on_affordance = substrate.is_on_position(agent_pos.unsqueeze(0), aff_pos)
                if idx == 0:
                    assert on_affordance[0] == (aff_name == "Bed")
                elif idx == 1:
                    # Agent at [5, 5, 5, 5] is only on Hospital
                    # Distance to Hospital: 0
                    # Distance to Store [5, 1, 5, 1]: sqrt((5-5)^2 + (5-1)^2 + (5-5)^2 + (5-1)^2) = sqrt(32) > 0.5
                    assert on_affordance[0] == (aff_name == "Hospital")
                elif idx == 2:
                    assert on_affordance[0] == (aff_name == "Job")
                else:
                    assert not on_affordance[0]

    def test_gridnd_batch_operations(self):
        """Test GridND handles batch operations correctly."""
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        # Batch of 10 agent positions
        positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

        assert positions.shape == (10, 4)
        assert positions.dtype == torch.long
        assert torch.all(positions >= 0) and torch.all(positions < 5)

        # Apply movement to batch
        deltas = torch.randn((10, 4), dtype=torch.float32)
        new_positions = substrate.apply_movement(positions, deltas)

        assert new_positions.shape == (10, 4)
        assert torch.all(new_positions >= 0) and torch.all(new_positions < 5)

    def test_continuousnd_batch_operations(self):
        """Test ContinuousND handles batch operations correctly."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        # Batch of 10 agent positions
        positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

        assert positions.shape == (10, 4)
        assert positions.dtype == torch.float32

        # Apply movement to batch
        deltas = torch.randn((10, 4), dtype=torch.float32) * 0.5
        new_positions = substrate.apply_movement(positions, deltas)

        assert new_positions.shape == (10, 4)
        assert torch.all(new_positions >= 0.0) and torch.all(new_positions <= 10.0)

    def test_gridnd_distance_metrics_consistency(self):
        """Test that different distance metrics work consistently."""
        dimension_sizes = [8, 8, 8, 8]

        substrates = {
            "manhattan": GridNDSubstrate(
                dimension_sizes=dimension_sizes,
                boundary="clamp",
                distance_metric="manhattan",
                observation_encoding="relative",
            ),
            "euclidean": GridNDSubstrate(
                dimension_sizes=dimension_sizes,
                boundary="clamp",
                distance_metric="euclidean",
                observation_encoding="relative",
            ),
            "chebyshev": GridNDSubstrate(
                dimension_sizes=dimension_sizes,
                boundary="clamp",
                distance_metric="chebyshev",
                observation_encoding="relative",
            ),
        }

        pos1 = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

        distances = {name: substrate.compute_distance(pos1, pos2) for name, substrate in substrates.items()}

        # Manhattan: |2| + |3| + |4| + |5| = 14
        assert distances["manhattan"][0] == 14

        # Euclidean: sqrt(4 + 9 + 16 + 25) = sqrt(54) ≈ 7.35
        assert torch.allclose(distances["euclidean"], torch.tensor([7.35], dtype=torch.float32), atol=0.1)

        # Chebyshev: max(2, 3, 4, 5) = 5
        assert distances["chebyshev"][0] == 5

    def test_continuousnd_distance_metrics_consistency(self):
        """Test that different distance metrics work consistently in continuous space."""
        bounds = [(0.0, 10.0)] * 4

        substrates = {
            "manhattan": ContinuousNDSubstrate(
                bounds=bounds,
                boundary="clamp",
                movement_delta=0.5,
                interaction_radius=1.0,
                distance_metric="manhattan",
                observation_encoding="relative",
            ),
            "euclidean": ContinuousNDSubstrate(
                bounds=bounds,
                boundary="clamp",
                movement_delta=0.5,
                interaction_radius=1.0,
                distance_metric="euclidean",
                observation_encoding="relative",
            ),
            "chebyshev": ContinuousNDSubstrate(
                bounds=bounds,
                boundary="clamp",
                movement_delta=0.5,
                interaction_radius=1.0,
                distance_metric="chebyshev",
                observation_encoding="relative",
            ),
        }

        pos1 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        pos2 = torch.tensor([[2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)

        distances = {name: substrate.compute_distance(pos1, pos2) for name, substrate in substrates.items()}

        # Manhattan: |2| + |3| + |4| + |5| = 14.0
        assert torch.allclose(distances["manhattan"], torch.tensor([14.0]))

        # Euclidean: sqrt(4 + 9 + 16 + 25) = sqrt(54) ≈ 7.35
        assert torch.allclose(distances["euclidean"], torch.tensor([7.35], dtype=torch.float32), atol=0.1)

        # Chebyshev: max(2, 3, 4, 5) = 5.0
        assert torch.allclose(distances["chebyshev"], torch.tensor([5.0]))


class TestNDSubstrateObservationDim:
    """Test observation dimension calculations for N-D substrates."""

    def test_gridnd_observation_dim_relative(self):
        """GridND relative encoding returns position_dim dimensions."""
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        assert substrate.get_observation_dim() == 5

    def test_gridnd_observation_dim_scaled(self):
        """GridND scaled encoding returns 2*position_dim dimensions."""
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="scaled",
        )

        assert substrate.get_observation_dim() == 10

    def test_gridnd_observation_dim_absolute(self):
        """GridND absolute encoding returns position_dim dimensions."""
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="absolute",
        )

        assert substrate.get_observation_dim() == 5

    def test_continuousnd_observation_dim_relative(self):
        """ContinuousND relative encoding returns position_dim dimensions."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 5,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )

        assert substrate.get_observation_dim() == 5

    def test_continuousnd_observation_dim_scaled(self):
        """ContinuousND scaled encoding returns 2*position_dim dimensions."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 5,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="scaled",
        )

        assert substrate.get_observation_dim() == 10

    def test_continuousnd_observation_dim_absolute(self):
        """ContinuousND absolute encoding returns position_dim dimensions."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 5,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="absolute",
        )

        assert substrate.get_observation_dim() == 5


class TestNDSubstrateConfigRoundtrip:
    """Test that N-D substrate configs can be saved and loaded."""

    def test_gridnd_yaml_config_roundtrip(self):
        """Test creating GridND from config dict simulates YAML loading."""
        config_dict = {
            "version": "1.0",
            "description": "Test 4D GridND",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [6, 7, 8, 9],
                "boundary": "wrap",
                "distance_metric": "euclidean",
                "observation_encoding": "scaled",
            },
        }

        # Create substrate from config
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        # Verify all properties preserved
        assert substrate.dimension_sizes == [6, 7, 8, 9]
        assert substrate.boundary == "wrap"
        assert substrate.distance_metric == "euclidean"
        assert substrate.observation_encoding == "scaled"

    def test_continuousnd_yaml_config_roundtrip(self):
        """Test creating ContinuousND from config dict simulates YAML loading."""
        config_dict = {
            "version": "1.0",
            "description": "Test 4D ContinuousND",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 4,
                "bounds": [(0.0, 100.0), (-50.0, 50.0), (10.0, 20.0), (-10.0, 30.0)],
                "boundary": "bounce",
                "movement_delta": 2.0,
                "interaction_radius": 3.5,
                "distance_metric": "chebyshev",
                "observation_encoding": "absolute",
            },
        }

        # Create substrate from config
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        # Verify all properties preserved
        assert substrate.bounds == [(0.0, 100.0), (-50.0, 50.0), (10.0, 20.0), (-10.0, 30.0)]
        assert substrate.boundary == "bounce"
        assert substrate.movement_delta == 2.0
        assert substrate.interaction_radius == 3.5
        assert substrate.distance_metric == "chebyshev"
        assert substrate.observation_encoding == "absolute"
