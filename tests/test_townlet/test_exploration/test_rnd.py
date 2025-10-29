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


def test_rnd_novelty_decreases_with_training():
    """Prediction error should decrease for repeated states."""
    from townlet.exploration.rnd import RNDExploration

    rnd = RNDExploration(obs_dim=70, embed_dim=128, training_batch_size=32, device=torch.device('cpu'))

    # Create a fixed observation
    obs = torch.randn(1, 70)

    # Initial novelty (high, predictor untrained)
    initial_novelty = rnd.compute_intrinsic_rewards(obs).item()

    # Train predictor on this observation repeatedly (multiple training rounds)
    # Add same observation to buffer and train multiple times
    for round_idx in range(10):
        for _ in range(32):
            rnd.obs_buffer.append(obs.squeeze(0))
        loss = rnd.update_predictor()

    # Final novelty (should be much lower)
    final_novelty = rnd.compute_intrinsic_rewards(obs).item()

    # Novelty should decrease significantly (more lenient threshold due to random init)
    assert final_novelty < initial_novelty * 0.8, \
        f"Novelty should decrease with training: {initial_novelty:.4f} -> {final_novelty:.4f}"


def test_rnd_predictor_loss_decreases():
    """Predictor training loss should decrease over multiple updates."""
    from townlet.exploration.rnd import RNDExploration

    rnd = RNDExploration(obs_dim=70, embed_dim=128, training_batch_size=32, device=torch.device('cpu'))

    # Generate a fixed set of observations (same 32 observations repeated)
    obs_data = [torch.randn(70) for _ in range(32)]

    losses = []

    # Train on same observations for multiple batches (should decrease loss)
    for batch_idx in range(4):
        # Add same observations to buffer
        for obs in obs_data:
            rnd.obs_buffer.append(obs)

        # Train on this batch
        loss = rnd.update_predictor()
        losses.append(loss)

    assert len(losses) == 4

    # Loss should generally decrease (later losses < earlier losses)
    avg_early = sum(losses[:2]) / 2
    avg_late = sum(losses[2:]) / 2

    assert avg_late < avg_early, \
        f"Loss should decrease with training: {avg_early:.4f} -> {avg_late:.4f}"
