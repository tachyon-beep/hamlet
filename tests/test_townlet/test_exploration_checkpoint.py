"""
Tests for exploration checkpoint state completeness.

Verifies P3.2: Exploration Checkpoint Audit
Ensures all exploration state is saved and restored correctly.
"""

import pytest
import torch

from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.exploration.rnd import RNDExploration


@pytest.fixture
def device():
    """PyTorch device for testing."""
    return torch.device("cpu")


class TestRNDCheckpoint:
    """Test RND exploration checkpoint completeness."""

    def test_rnd_saves_predictor_optimizer(self, device):
        """RND should save predictor optimizer state."""
        rnd = RNDExploration(
            obs_dim=72,
            embed_dim=128,
            learning_rate=1e-4,
            device=device,
        )

        # Checkpoint state should include optimizer
        state = rnd.checkpoint_state()
        assert "optimizer" in state, "RND checkpoint must include optimizer state"

    def test_rnd_saves_epsilon(self, device):
        """RND should save epsilon decay parameters."""
        rnd = RNDExploration(
            obs_dim=72,
            epsilon_start=0.8,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            device=device,
        )

        state = rnd.checkpoint_state()
        assert "epsilon" in state, "RND checkpoint must include epsilon"
        assert "epsilon_min" in state, "RND checkpoint must include epsilon_min"
        assert "epsilon_decay" in state, "RND checkpoint must include epsilon_decay"
        assert state["epsilon"] == 0.8
        assert state["epsilon_min"] == 0.05
        assert state["epsilon_decay"] == 0.99

    def test_rnd_saves_network_architecture_params(self, device):
        """RND should save architecture parameters for reconstruction."""
        rnd = RNDExploration(
            obs_dim=72,
            embed_dim=256,
            device=device,
        )

        state = rnd.checkpoint_state()
        assert "obs_dim" in state, "RND checkpoint must include obs_dim"
        assert "embed_dim" in state, "RND checkpoint must include embed_dim"
        assert state["obs_dim"] == 72
        assert state["embed_dim"] == 256

    def test_rnd_saves_network_weights(self, device):
        """RND should save both fixed and predictor network weights."""
        rnd = RNDExploration(obs_dim=72, device=device)

        state = rnd.checkpoint_state()
        assert "fixed_network" in state, "RND checkpoint must include fixed_network"
        assert "predictor_network" in state, "RND checkpoint must include predictor_network"

        # Should be state dicts
        assert isinstance(state["fixed_network"], dict)
        assert isinstance(state["predictor_network"], dict)

    def test_rnd_restores_epsilon(self, device):
        """RND should restore epsilon from checkpoint."""
        rnd = RNDExploration(obs_dim=72, epsilon_start=1.0, device=device)

        # Decay epsilon
        for _ in range(10):
            rnd.epsilon *= rnd.epsilon_decay

        decayed_epsilon = rnd.epsilon
        state = rnd.checkpoint_state()

        # Create new RND and restore
        rnd2 = RNDExploration(obs_dim=72, epsilon_start=1.0, device=device)
        rnd2.load_state(state)

        assert abs(rnd2.epsilon - decayed_epsilon) < 1e-6, "Epsilon should be restored"

    def test_rnd_restores_network_weights(self, device):
        """RND should restore network weights from checkpoint."""
        rnd = RNDExploration(obs_dim=72, device=device)

        # Train predictor on some data
        obs = torch.randn(128, 72, device=device)
        target = rnd.fixed_network(obs)
        pred = rnd.predictor_network(obs)
        loss = torch.nn.functional.mse_loss(pred, target)
        rnd.optimizer.zero_grad()
        loss.backward()
        rnd.optimizer.step()

        # Save state
        state = rnd.checkpoint_state()

        # Create new RND and restore
        rnd2 = RNDExploration(obs_dim=72, device=device)
        rnd2.load_state(state)

        # Networks should produce same output
        test_obs = torch.randn(10, 72, device=device)
        pred1 = rnd.predictor_network(test_obs)
        pred2 = rnd2.predictor_network(test_obs)

        assert torch.allclose(pred1, pred2), "Predictor network weights should be restored"

    def test_rnd_restores_optimizer_state(self, device):
        """RND should restore optimizer state from checkpoint."""
        rnd = RNDExploration(obs_dim=72, learning_rate=1e-3, device=device)

        # Train for a few steps to build optimizer state
        for _ in range(5):
            obs = torch.randn(128, 72, device=device)
            target = rnd.fixed_network(obs)
            pred = rnd.predictor_network(obs)
            loss = torch.nn.functional.mse_loss(pred, target)
            rnd.optimizer.zero_grad()
            loss.backward()
            rnd.optimizer.step()

        # Save state
        state = rnd.checkpoint_state()
        optimizer_state_keys = set(state["optimizer"]["state"].keys())

        # Create new RND and restore
        rnd2 = RNDExploration(obs_dim=72, learning_rate=1e-3, device=device)
        rnd2.load_state(state)

        # Optimizer should have same state
        restored_state_keys = set(rnd2.optimizer.state_dict()["state"].keys())
        assert optimizer_state_keys == restored_state_keys, "Optimizer state should be restored"


class TestAdaptiveIntrinsicCheckpoint:
    """Test AdaptiveIntrinsic exploration checkpoint completeness."""

    def test_adaptive_saves_intrinsic_weight(self, device):
        """AdaptiveIntrinsic should save current intrinsic weight."""
        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=72,
            initial_intrinsic_weight=0.5,
            device=device,
        )

        state = adaptive.checkpoint_state()
        assert "current_intrinsic_weight" in state
        assert state["current_intrinsic_weight"] == 0.5

    def test_adaptive_saves_annealing_params(self, device):
        """AdaptiveIntrinsic should save all annealing parameters."""
        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=72,
            variance_threshold=50.0,
            survival_window=200,
            decay_rate=0.95,
            min_intrinsic_weight=0.05,
            device=device,
        )

        state = adaptive.checkpoint_state()
        assert "variance_threshold" in state
        assert "survival_window" in state
        assert "decay_rate" in state
        assert "min_intrinsic_weight" in state
        assert state["variance_threshold"] == 50.0
        assert state["survival_window"] == 200
        assert state["decay_rate"] == 0.95
        assert state["min_intrinsic_weight"] == 0.05

    def test_adaptive_saves_survival_history(self, device):
        """AdaptiveIntrinsic should save survival history for annealing."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=72, device=device)

        # Simulate some survival times
        adaptive.survival_history = [10.0, 20.0, 30.0, 40.0, 50.0]

        state = adaptive.checkpoint_state()
        assert "survival_history" in state
        assert state["survival_history"] == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_adaptive_saves_rnd_state(self, device):
        """AdaptiveIntrinsic should save underlying RND state."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=72, device=device)

        state = adaptive.checkpoint_state()
        assert "rnd_state" in state
        assert isinstance(state["rnd_state"], dict)

        # RND state should contain network weights
        rnd_state = state["rnd_state"]
        assert "predictor_network" in rnd_state
        assert "optimizer" in rnd_state

    def test_adaptive_restores_intrinsic_weight(self, device):
        """AdaptiveIntrinsic should restore intrinsic weight from checkpoint."""
        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=72,
            initial_intrinsic_weight=1.0,
            device=device,
        )

        # Anneal weight
        adaptive.current_intrinsic_weight = 0.3

        state = adaptive.checkpoint_state()

        # Create new instance and restore
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=72, device=device)
        adaptive2.load_state(state)

        assert abs(adaptive2.current_intrinsic_weight - 0.3) < 1e-6

    def test_adaptive_restores_survival_history(self, device):
        """AdaptiveIntrinsic should restore survival history from checkpoint."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=72, device=device)

        # Set survival history
        adaptive.survival_history = [15.0, 25.0, 35.0]

        state = adaptive.checkpoint_state()

        # Create new instance and restore
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=72, device=device)
        adaptive2.load_state(state)

        assert adaptive2.survival_history == [15.0, 25.0, 35.0]

    def test_adaptive_restores_rnd_state(self, device):
        """AdaptiveIntrinsic should restore RND state from checkpoint."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=72, device=device)

        # Train RND predictor
        obs = torch.randn(128, 72, device=device)
        adaptive.rnd.obs_buffer = [obs[i] for i in range(128)]
        adaptive.rnd.update_predictor()

        # Get predictor output before checkpoint
        test_obs = torch.randn(10, 72, device=device)
        pred_before = adaptive.rnd.predictor_network(test_obs)

        # Save and restore
        state = adaptive.checkpoint_state()
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=72, device=device)
        adaptive2.load_state(state)

        # Predictor should produce same output
        pred_after = adaptive2.rnd.predictor_network(test_obs)
        assert torch.allclose(pred_before, pred_after)


class TestEpsilonGreedyCheckpoint:
    """Test EpsilonGreedy exploration checkpoint completeness."""

    def test_epsilon_greedy_saves_epsilon(self, device):
        """EpsilonGreedy should save epsilon parameters."""
        epsilon = EpsilonGreedyExploration(
            epsilon=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.999,
        )

        state = epsilon.checkpoint_state()
        assert "epsilon" in state
        assert "epsilon_min" in state
        assert "epsilon_decay" in state
        assert abs(state["epsilon"] - 0.9) < 1e-6
        assert abs(state["epsilon_min"] - 0.1) < 1e-6
        assert abs(state["epsilon_decay"] - 0.999) < 1e-6

    def test_epsilon_greedy_restores_epsilon(self, device):
        """EpsilonGreedy should restore epsilon from checkpoint."""
        epsilon = EpsilonGreedyExploration(epsilon=1.0)

        # Decay epsilon
        for _ in range(20):
            epsilon.epsilon *= epsilon.epsilon_decay

        decayed = epsilon.epsilon
        state = epsilon.checkpoint_state()

        # Restore
        epsilon2 = EpsilonGreedyExploration(epsilon=1.0)
        epsilon2.load_state(state)

        assert abs(epsilon2.epsilon - decayed) < 1e-6


class TestCheckpointRoundTrip:
    """Test full checkpoint save/restore cycle."""

    def test_rnd_roundtrip_preserves_behavior(self, device):
        """RND should have identical behavior after save/restore."""
        rnd = RNDExploration(obs_dim=72, device=device)

        # Train predictor
        for _ in range(5):
            obs = torch.randn(128, 72, device=device)
            rnd.obs_buffer.extend([obs[i] for i in range(128)])
            rnd.update_predictor()

        # Compute intrinsic rewards before checkpoint
        test_obs = torch.randn(10, 72, device=device)
        rewards_before = rnd.compute_intrinsic_rewards(test_obs)

        # Save and restore
        state = rnd.checkpoint_state()
        rnd2 = RNDExploration(obs_dim=72, device=device)
        rnd2.load_state(state)

        # Compute intrinsic rewards after restore
        rewards_after = rnd2.compute_intrinsic_rewards(test_obs)

        # Should be identical
        assert torch.allclose(rewards_before, rewards_after), "RND behavior should be preserved"

    def test_adaptive_roundtrip_preserves_behavior(self, device):
        """AdaptiveIntrinsic should have identical behavior after save/restore."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=72, device=device)

        # Train and anneal
        for _ in range(3):
            obs = torch.randn(128, 72, device=device)
            adaptive.rnd.obs_buffer.extend([obs[i] for i in range(128)])
            adaptive.rnd.update_predictor()

        adaptive.survival_history = [20.0, 30.0, 40.0]
        adaptive.current_intrinsic_weight = 0.5

        # Compute intrinsic rewards before checkpoint
        test_obs = torch.randn(10, 72, device=device)
        rewards_before = adaptive.compute_intrinsic_rewards(test_obs)

        # Save and restore
        state = adaptive.checkpoint_state()
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=72, device=device)
        adaptive2.load_state(state)

        # Compute intrinsic rewards after restore
        rewards_after = adaptive2.compute_intrinsic_rewards(test_obs)

        # Should be identical
        assert torch.allclose(rewards_before, rewards_after), "Adaptive behavior should be preserved"
        assert abs(adaptive2.current_intrinsic_weight - 0.5) < 1e-6


class TestCheckpointMissingFields:
    """Test handling of missing checkpoint fields (backwards compatibility)."""

    def test_rnd_handles_missing_optimizer(self, device):
        """RND should handle checkpoints without optimizer state (legacy)."""
        rnd = RNDExploration(obs_dim=72, device=device)

        state = rnd.checkpoint_state()
        # Simulate legacy checkpoint without optimizer
        del state["optimizer"]

        # Should not crash (though training may be suboptimal)
        rnd2 = RNDExploration(obs_dim=72, device=device)
        try:
            rnd2.load_state(state)
        except KeyError:
            pytest.fail("RND should handle missing optimizer gracefully")

    def test_adaptive_handles_missing_survival_history(self, device):
        """AdaptiveIntrinsic should handle checkpoints without survival history."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=72, device=device)

        state = adaptive.checkpoint_state()
        # Simulate legacy checkpoint without survival history
        del state["survival_history"]

        # Should not crash (annealing may restart)
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=72, device=device)
        try:
            adaptive2.load_state(state)
        except KeyError:
            pytest.fail("Adaptive should handle missing survival_history gracefully")
