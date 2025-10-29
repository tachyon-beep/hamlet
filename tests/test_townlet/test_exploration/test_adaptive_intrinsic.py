"""Tests for AdaptiveIntrinsicExploration."""

import pytest
import torch
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.training.state import BatchedAgentState


def test_adaptive_intrinsic_construction():
    """AdaptiveIntrinsic should initialize with RND instance."""
    adaptive = AdaptiveIntrinsicExploration(
        obs_dim=70,
        embed_dim=128,
        device=torch.device('cpu'),
    )

    assert adaptive.current_intrinsic_weight == 1.0
    assert adaptive.min_intrinsic_weight == 0.0
    assert hasattr(adaptive, 'rnd')
    assert len(adaptive.survival_history) == 0


def test_adaptive_annealing_triggers_on_low_variance():
    """Intrinsic weight should decay when variance < threshold."""
    adaptive = AdaptiveIntrinsicExploration(
        variance_threshold=10.0,
        survival_window=10,
        decay_rate=0.9,
        device=torch.device('cpu'),
    )

    # Add consistent survival times (low variance)
    for _ in range(10):
        adaptive.update_on_episode_end(survival_time=100.0)

    # Variance should be 0 (all same value)
    assert adaptive.should_anneal()

    initial_weight = adaptive.current_intrinsic_weight
    adaptive.anneal_weight()

    # Weight should decrease
    assert adaptive.current_intrinsic_weight < initial_weight
    assert adaptive.current_intrinsic_weight == initial_weight * 0.9


def test_adaptive_no_annealing_on_high_variance():
    """Intrinsic weight should NOT decay when variance > threshold."""
    adaptive = AdaptiveIntrinsicExploration(
        variance_threshold=10.0,
        survival_window=10,
        decay_rate=0.9,
        device=torch.device('cpu'),
    )

    # Add highly variable survival times (high variance)
    for i in range(10):
        adaptive.update_on_episode_end(survival_time=float(i * 50))

    # Variance should be high
    assert not adaptive.should_anneal()

    # Weight should not change
    initial_weight = adaptive.current_intrinsic_weight
    if not adaptive.should_anneal():
        # Don't anneal if variance too high
        pass

    assert adaptive.current_intrinsic_weight == initial_weight


def test_adaptive_weight_floor():
    """Intrinsic weight should not go below min_intrinsic_weight."""
    adaptive = AdaptiveIntrinsicExploration(
        initial_intrinsic_weight=1.0,
        min_intrinsic_weight=0.1,
        decay_rate=0.5,
        device=torch.device('cpu'),
    )

    # Anneal many times
    for _ in range(10):
        adaptive.anneal_weight()

    # Weight should floor at 0.1
    assert adaptive.current_intrinsic_weight >= 0.1


def test_adaptive_composition_delegates_to_rnd():
    """AdaptiveIntrinsic should delegate intrinsic reward computation to RND."""
    adaptive = AdaptiveIntrinsicExploration(
        obs_dim=70,
        embed_dim=128,
        initial_intrinsic_weight=0.5,
        device=torch.device('cpu'),
    )

    obs = torch.randn(10, 70)

    # Get RND raw novelty
    rnd_novelty = adaptive.rnd.compute_intrinsic_rewards(obs)

    # Get adaptive intrinsic (should be scaled by weight)
    adaptive_intrinsic = adaptive.compute_intrinsic_rewards(obs)

    # Should be RND novelty * weight
    expected = rnd_novelty * 0.5
    assert torch.allclose(adaptive_intrinsic, expected, atol=1e-5)
