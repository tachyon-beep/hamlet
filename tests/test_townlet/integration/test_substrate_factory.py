"""Test substrate factory integration."""

import torch

from townlet.substrate.config import SubstrateConfig
from townlet.substrate.factory import SubstrateFactory


def test_factory_passes_observation_encoding_to_grid2d():
    """Test factory passes observation_encoding to Grid2D substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test",
        type="grid",
        grid={
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "scaled",  # Non-default
        },
    )
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))
    assert substrate.observation_encoding == "scaled"


def test_factory_passes_observation_encoding_to_continuous():
    """Test factory passes observation_encoding to Continuous substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test",
        type="continuous",
        continuous={
            "dimensions": 2,
            "bounds": [(0.0, 10.0), (0.0, 10.0)],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 1.0,
            "distance_metric": "euclidean",
            "observation_encoding": "absolute",  # Non-default
        },
    )
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))
    assert substrate.observation_encoding == "absolute"
