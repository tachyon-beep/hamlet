"""Tests for RND (Random Network Distillation)."""

import pytest
import torch
from townlet.exploration.rnd import RNDNetwork


def test_rnd_network_forward():
    """RNDNetwork should transform observations to embeddings."""
    obs_dim = 70
    embed_dim = 128

    network = RNDNetwork(obs_dim=obs_dim, embed_dim=embed_dim)

    # Test single observation
    obs = torch.randn(1, obs_dim)
    embedding = network(obs)

    assert embedding.shape == (1, embed_dim)

    # Test batch
    obs_batch = torch.randn(32, obs_dim)
    embeddings = network(obs_batch)

    assert embeddings.shape == (32, embed_dim)


def test_rnd_network_architecture():
    """RNDNetwork should have 3-layer MLP architecture."""
    network = RNDNetwork(obs_dim=70, embed_dim=128)

    # Should have 3 linear layers
    linear_layers = [m for m in network.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3

    # Verify dimensions: 70 -> 256 -> 128 -> 128
    assert linear_layers[0].in_features == 70
    assert linear_layers[0].out_features == 256
    assert linear_layers[1].in_features == 256
    assert linear_layers[1].out_features == 128
    assert linear_layers[2].in_features == 128
    assert linear_layers[2].out_features == 128


def test_rnd_fixed_network_frozen():
    """Fixed network parameters should be frozen (no gradients)."""
    from townlet.exploration.rnd import RNDExploration

    rnd = RNDExploration(obs_dim=70, embed_dim=128, device=torch.device('cpu'))

    # All fixed network parameters should have requires_grad=False
    for param in rnd.fixed_network.parameters():
        assert not param.requires_grad, "Fixed network should be frozen"

    # Predictor network should have requires_grad=True
    for param in rnd.predictor_network.parameters():
        assert param.requires_grad, "Predictor network should be trainable"
