from pathlib import Path

from townlet.demo.live_inference import LiveInferenceServer
from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.continuous import Continuous2DSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.gridnd import GridNDSubstrate


class MockEnv:
    """Mock environment for testing metadata building."""

    def __init__(self, substrate):
        self.substrate = substrate


def test_metadata_includes_grid2d_topology():
    """WebSocket metadata should include topology for Grid2D."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="square",
    )

    # Mock the server just enough to test _build_substrate_metadata
    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        step_delay=0.2,
        total_episodes=1,
        training_config_path=None,
        config_dir=TEST_CONFIG_DIR,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "grid2d"
    assert metadata["topology"] == "square"
    assert metadata["position_dim"] == 2
    assert metadata["width"] == 8
    assert metadata["height"] == 8


def test_metadata_includes_grid3d_topology():
    """WebSocket metadata should include topology for Grid3D."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="cubic",
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        step_delay=0.2,
        total_episodes=1,
        training_config_path=None,
        config_dir=TEST_CONFIG_DIR,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "grid3d"
    assert metadata["topology"] == "cubic"
    assert metadata["position_dim"] == 3
    assert metadata["depth"] == 3


def test_metadata_includes_gridnd_topology():
    """WebSocket metadata should include topology for GridND."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="hypercube",
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        step_delay=0.2,
        total_episodes=1,
        training_config_path=None,
        config_dir=TEST_CONFIG_DIR,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "gridnd"
    assert metadata["topology"] == "hypercube"
    assert metadata["position_dim"] == 7
    assert metadata["dimension_sizes"] == [5, 5, 5, 5, 5, 5, 5]


def test_metadata_omits_topology_for_continuous():
    """WebSocket metadata should omit topology for Continuous substrates."""
    substrate = Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        step_delay=0.2,
        total_episodes=1,
        training_config_path=None,
        config_dir=TEST_CONFIG_DIR,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "continuous2d"
    assert "topology" not in metadata  # Should be omitted, not None
    assert metadata["position_dim"] == 2


def test_metadata_omits_topology_for_aspatial():
    """WebSocket metadata should omit topology for Aspatial substrate."""
    substrate = AspatialSubstrate()

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        step_delay=0.2,
        total_episodes=1,
        training_config_path=None,
        config_dir=TEST_CONFIG_DIR,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    assert metadata["type"] == "aspatial"
    assert "topology" not in metadata  # Should be omitted, not None
    assert metadata["position_dim"] == 0


def test_metadata_topology_respects_substrate_attribute():
    """WebSocket metadata should read topology from substrate, not hardcode."""
    # This test verifies we're reading from substrate.topology, not hardcoding
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="square",  # Explicitly provided
    )

    server = LiveInferenceServer(
        checkpoint_dir="",
        port=8766,
        step_delay=0.2,
        total_episodes=1,
        training_config_path=None,
        config_dir=TEST_CONFIG_DIR,
    )
    server.env = MockEnv(substrate)

    metadata = server._build_substrate_metadata()

    # Should read from substrate.topology, not hardcode "square"
    assert metadata["topology"] == substrate.topology


TEST_CONFIG_DIR = Path("configs/test")
