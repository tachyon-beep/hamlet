"""Integration tests for topology metadata propagation.

Tests end-to-end flow: config → factory → substrate → WebSocket metadata.
"""

import tempfile
from pathlib import Path

import torch
import yaml

from townlet.demo.live_inference import LiveInferenceServer
from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory


class MockEnv:
    """Mock environment for testing metadata building."""

    def __init__(self, substrate):
        self.substrate = substrate


def test_topology_propagates_from_config_to_websocket_metadata_grid2d():
    """Integration test: topology flows from config → factory → substrate → metadata."""
    # Step 1: Create substrate.yaml config
    config_data = {
        "version": "1.0",
        "description": "Integration test Grid2D config",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 2: Load config
        config = load_substrate_config(config_path)
        assert config.grid.topology == "square"

        # Step 3: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))
        assert hasattr(substrate, "topology")
        assert substrate.topology == "square"

        # Step 4: Build WebSocket metadata
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

        # Step 5: Verify topology in metadata
        assert metadata["topology"] == "square"
        assert metadata["type"] == "grid2d"


def test_topology_propagates_from_config_to_websocket_metadata_gridnd():
    """Integration test: GridND topology flows through entire pipeline."""
    # Step 1: Create substrate.yaml config for GridND
    config_data = {
        "version": "1.0",
        "description": "Integration test GridND config",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [5, 5, 5, 5, 5, 5, 5],
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
            "topology": "hypercube",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 2: Load config
        config = load_substrate_config(config_path)
        assert config.gridnd.topology == "hypercube"

        # Step 3: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))
        assert hasattr(substrate, "topology")
        assert substrate.topology == "hypercube"

        # Step 4: Build WebSocket metadata
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

        # Step 5: Verify topology in metadata
        assert metadata["topology"] == "hypercube"
        assert metadata["type"] == "gridnd"
        assert metadata["position_dim"] == 7


def test_continuous_substrate_has_no_topology_in_metadata():
    """Integration test: Continuous substrates omit topology throughout pipeline."""
    # Step 1: Create substrate.yaml config for Continuous2D
    config_data = {
        "version": "1.0",
        "description": "Integration test Continuous2D config",
        "type": "continuous",
        "continuous": {
            "dimensions": 2,
            "bounds": [[0.0, 10.0], [0.0, 10.0]],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 0.8,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 2: Load config
        config = load_substrate_config(config_path)
        assert not hasattr(config.continuous, "topology")

        # Step 3: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))
        assert not hasattr(substrate, "topology")

        # Step 4: Build WebSocket metadata
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

        # Step 5: Verify topology NOT in metadata
        assert "topology" not in metadata
        assert metadata["type"] == "continuous2d"


# Action Label Integration Tests for N-Dimensional Substrates


def test_gridnd_substrate_initializes_with_action_labels():
    """GridND substrate should initialize with action labels (no ValueError)."""
    config_data = {
        "version": "1.0",
        "description": "7D GridND integration test",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [3, 3, 3, 3, 3, 3, 3],
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
            "topology": "hypercube",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 1: Load config
        config = load_substrate_config(config_path)

        # Step 2: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        # Step 3: Get action labels (should not raise ValueError)
        from townlet.environment.action_labels import get_labels

        labels = get_labels(preset="math", substrate_position_dim=substrate.position_dim)

        # Step 4: Verify labels
        assert labels.get_action_count() == 16  # 2*7 + 2
        assert labels.get_label(0) == "D0_NEG"
        assert labels.get_label(7) == "D0_POS"
        assert labels.get_label(14) == "INTERACT"
        assert labels.get_label(15) == "WAIT"


def test_continuousnd_substrate_initializes_with_action_labels():
    """ContinuousND substrate should initialize with action labels."""
    config_data = {
        "version": "1.0",
        "description": "5D ContinuousND integration test",
        "type": "continuousnd",
        "continuous": {  # Note: Uses 'continuous' field for N-dimensional continuous
            "dimensions": 5,
            "bounds": [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 0.8,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 1: Load config
        config = load_substrate_config(config_path)

        # Step 2: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        # Step 3: Get action labels (should not raise ValueError)
        from townlet.environment.action_labels import get_labels

        labels = get_labels(preset="gaming", substrate_position_dim=substrate.position_dim)

        # Step 4: Verify labels
        assert labels.get_action_count() == 12  # 2*5 + 2
        assert labels.get_label(0) == "D0_NEG"
        assert labels.get_label(5) == "D0_POS"
        assert labels.get_label(10) == "INTERACT"
        assert labels.get_label(11) == "WAIT"


def test_4d_gridnd_custom_action_labels():
    """4D GridND with custom action labels."""
    config_data = {
        "version": "1.0",
        "description": "4D GridND with custom labels",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [5, 5, 5, 5],
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
            "topology": "hypercube",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "substrate.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Step 1: Load config
        config = load_substrate_config(config_path)

        # Step 2: Build substrate via factory
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        # Step 3: Use custom robotics labels
        from townlet.environment.action_labels import get_labels

        custom_labels = {
            0: "TRANS_X_NEG",
            1: "TRANS_Y_NEG",
            2: "TRANS_Z_NEG",
            3: "ROT_ROLL_NEG",
            4: "TRANS_X_POS",
            5: "TRANS_Y_POS",
            6: "TRANS_Z_POS",
            7: "ROT_ROLL_POS",
            8: "INTERACT",
            9: "WAIT",
        }
        labels = get_labels(custom_labels=custom_labels, substrate_position_dim=substrate.position_dim)

        # Step 4: Verify custom labels work
        assert labels.get_action_count() == 10  # 2*4 + 2
        assert labels.get_label(0) == "TRANS_X_NEG"
        assert labels.get_label(3) == "ROT_ROLL_NEG"
        assert labels.get_label(7) == "ROT_ROLL_POS"
        assert labels.get_label(8) == "INTERACT"


TEST_CONFIG_DIR = Path("configs/test")
