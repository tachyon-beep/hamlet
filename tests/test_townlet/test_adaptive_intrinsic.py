"""
Tests for Adaptive Intrinsic Exploration.

Focus on variance-based annealing logic and RND integration.
"""

import torch

from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.training.state import BatchedAgentState


class TestAnnealingLogic:
    """Test variance-based annealing conditions."""

    def test_should_not_anneal_with_insufficient_data(self):
        """Should not anneal before survival_window filled."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            survival_window=100,
            variance_threshold=100.0,
            device=torch.device("cpu"),
        )

        # Add only 50 episodes (less than window)
        for _ in range(50):
            exploration.update_on_episode_end(survival_time=100.0)

        # Should not have annealed yet
        assert not exploration.should_anneal()
        assert abs(exploration.current_intrinsic_weight - 1.0) < 1e-6

    def test_should_not_anneal_with_high_variance(self):
        """Should not anneal when survival variance is high."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            survival_window=100,
            variance_threshold=100.0,
            device=torch.device("cpu"),
        )

        # Add 100 episodes with high variance (0-200 steps)
        for i in range(100):
            exploration.update_on_episode_end(survival_time=float(i * 2))

        # Should not anneal (high variance)
        assert not exploration.should_anneal()
        assert abs(exploration.current_intrinsic_weight - 1.0) < 1e-6

    def test_should_not_anneal_consistent_failure(self):
        """Should NOT anneal when consistently failing (low variance, low survival)."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            survival_window=100,
            variance_threshold=100.0,
            device=torch.device("cpu"),
        )

        # Add 100 episodes of consistent failure (5-15 steps, mean=10)
        for _ in range(100):
            exploration.update_on_episode_end(survival_time=10.0)

        # Low variance but LOW survival - should NOT anneal
        # This was the bug: premature annealing for "consistent failure"
        assert not exploration.should_anneal()
        assert abs(exploration.current_intrinsic_weight - 1.0) < 1e-6

    def test_should_anneal_consistent_success(self):
        """Should anneal when consistently succeeding (low variance, high survival)."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            survival_window=100,
            variance_threshold=100.0,
            decay_rate=0.99,
            device=torch.device("cpu"),
        )

        # Add 100 episodes of consistent success (95-105 steps, mean=100)
        for i in range(100):
            exploration.update_on_episode_end(survival_time=100.0 + (i % 10))

        # Low variance AND high survival - should anneal
        assert exploration.should_anneal()

        # Weight should have been reduced (called in update_on_episode_end)
        assert exploration.current_intrinsic_weight < 1.0

    def test_anneal_weight_exponential_decay(self):
        """anneal_weight should apply exponential decay."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            initial_intrinsic_weight=1.0,
            decay_rate=0.9,
            min_intrinsic_weight=0.1,
            device=torch.device("cpu"),
        )

        # First anneal
        exploration.anneal_weight()
        assert abs(exploration.current_intrinsic_weight - 0.9) < 1e-6

        # Second anneal
        exploration.anneal_weight()
        assert abs(exploration.current_intrinsic_weight - 0.81) < 1e-6

        # Third anneal
        exploration.anneal_weight()
        assert abs(exploration.current_intrinsic_weight - 0.729) < 1e-6

    def test_anneal_weight_respects_minimum(self):
        """anneal_weight should not go below minimum."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            initial_intrinsic_weight=0.15,
            decay_rate=0.5,
            min_intrinsic_weight=0.1,
            device=torch.device("cpu"),
        )

        # This would normally give 0.075, but should floor at 0.1
        exploration.anneal_weight()
        assert abs(exploration.current_intrinsic_weight - 0.1) < 1e-6

        # Should stay at floor
        exploration.anneal_weight()
        assert abs(exploration.current_intrinsic_weight - 0.1) < 1e-6

    def test_update_on_episode_end_maintains_window(self):
        """update_on_episode_end should maintain sliding window."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            survival_window=10,
            device=torch.device("cpu"),
        )

        # Add 15 episodes
        for i in range(15):
            exploration.update_on_episode_end(survival_time=float(i))

        # Should only keep last 10
        assert len(exploration.survival_history) == 10
        assert abs(exploration.survival_history[0] - 5.0) < 1e-6
        assert abs(exploration.survival_history[-1] - 14.0) < 1e-6


class TestIntrinsicRewardScaling:
    """Test intrinsic reward computation with weight scaling."""

    def test_compute_intrinsic_rewards_scales_by_weight(self):
        """Intrinsic rewards should be scaled by current weight."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            initial_intrinsic_weight=0.5,
            device=torch.device("cpu"),
        )

        observations = torch.randn(4, 10)

        # Get rewards with weight=0.5
        rewards_half = exploration.compute_intrinsic_rewards(observations)

        # All rewards should be non-negative (MSE property)
        assert torch.all(rewards_half >= 0)

        # Now set weight to 1.0
        exploration.current_intrinsic_weight = 1.0
        rewards_full = exploration.compute_intrinsic_rewards(observations)

        # Should be exactly 2x (weight doubled)
        assert torch.allclose(rewards_full, rewards_half * 2.0, atol=1e-5)

    def test_compute_intrinsic_rewards_zero_weight(self):
        """Zero weight should give zero rewards."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            initial_intrinsic_weight=0.0,
            device=torch.device("cpu"),
        )

        observations = torch.randn(4, 10)
        rewards = exploration.compute_intrinsic_rewards(observations)

        # All should be zero
        assert torch.allclose(rewards, torch.zeros(4))


class TestActionSelection:
    """Test action selection delegation to RND."""

    def test_select_actions_delegates_to_rnd(self):
        """select_actions should delegate to RND."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            epsilon_start=0.0,  # Greedy for deterministic test
            device=torch.device("cpu"),
        )

        q_values = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.5, 1.5],
                [0.0, 5.0, 1.0, 2.0, 3.0],
            ]
        )

        device = torch.device("cpu")
        agent_states = BatchedAgentState(
            observations=torch.zeros((2, 10)),
            actions=torch.zeros(2, dtype=torch.long),
            rewards=torch.zeros(2),
            dones=torch.zeros(2, dtype=torch.bool),
            epsilons=torch.zeros(2),
            intrinsic_rewards=torch.zeros(2),
            survival_times=torch.zeros(2),
            curriculum_difficulties=torch.zeros(2),
            device=device,
        )

        actions = exploration.select_actions(q_values, agent_states)

        # Should select argmax (epsilon=0.0)
        assert actions[0] == 2  # Max is index 2
        assert actions[1] == 1  # Max is index 1


class TestRNDUpdate:
    """Test RND predictor update delegation."""

    def test_update_delegates_to_rnd(self):
        """update should delegate to RND predictor."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            rnd_training_batch_size=4,
            device=torch.device("cpu"),
        )

        # Create batch with enough observations to trigger training
        batch = {
            "observations": torch.randn(4, 10),
        }

        # Update should work without error
        exploration.update(batch)

        # RND buffer should have accumulated observations
        # (This implicitly tests the delegation worked)


class TestEpsilonDecay:
    """Test epsilon decay delegation."""

    def test_decay_epsilon_delegates_to_rnd(self):
        """decay_epsilon should delegate to RND."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            epsilon_start=1.0,
            epsilon_decay=0.9,
            epsilon_min=0.01,
            device=torch.device("cpu"),
        )

        # Initial epsilon should be 1.0
        assert abs(exploration.rnd.epsilon - 1.0) < 1e-6

        # Decay
        exploration.decay_epsilon()
        assert abs(exploration.rnd.epsilon - 0.9) < 1e-6

        # Decay again
        exploration.decay_epsilon()
        assert abs(exploration.rnd.epsilon - 0.81) < 1e-6


class TestStatePersistence:
    """Test checkpoint and restore functionality."""

    def test_checkpoint_state_includes_all_config(self):
        """checkpoint_state should include RND state and annealing config."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            initial_intrinsic_weight=0.7,
            min_intrinsic_weight=0.05,
            variance_threshold=50.0,
            survival_window=200,
            decay_rate=0.95,
            device=torch.device("cpu"),
        )

        # Add some survival history
        exploration.update_on_episode_end(100.0)
        exploration.update_on_episode_end(110.0)

        state = exploration.checkpoint_state()

        # Should contain all config
        assert "rnd_state" in state
        assert abs(state["current_intrinsic_weight"] - 0.7) < 1e-6
        assert abs(state["min_intrinsic_weight"] - 0.05) < 1e-6
        assert abs(state["variance_threshold"] - 50.0) < 1e-6
        assert state["survival_window"] == 200
        assert abs(state["decay_rate"] - 0.95) < 1e-6
        assert len(state["survival_history"]) == 2
        assert abs(state["survival_history"][0] - 100.0) < 0.1
        assert abs(state["survival_history"][1] - 110.0) < 0.1

    def test_load_state_restores_all_config(self):
        """load_state should restore RND and annealing config."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            device=torch.device("cpu"),
        )

        # Create state to load
        rnd_state = exploration.rnd.checkpoint_state()
        new_state = {
            "rnd_state": rnd_state,
            "current_intrinsic_weight": 0.3,
            "min_intrinsic_weight": 0.01,
            "variance_threshold": 25.0,
            "survival_window": 150,
            "decay_rate": 0.97,
            "survival_history": [50.0, 60.0, 70.0],
        }

        exploration.load_state(new_state)

        # Should have updated all fields
        assert abs(exploration.current_intrinsic_weight - 0.3) < 1e-6
        assert abs(exploration.min_intrinsic_weight - 0.01) < 1e-6
        assert abs(exploration.variance_threshold - 25.0) < 1e-6
        assert exploration.survival_window == 150
        assert abs(exploration.decay_rate - 0.97) < 1e-6
        assert exploration.survival_history == [50.0, 60.0, 70.0]

    def test_checkpoint_restore_roundtrip(self):
        """Checkpoint and restore should preserve exact state."""
        original = AdaptiveIntrinsicExploration(
            obs_dim=10,
            initial_intrinsic_weight=0.42,
            min_intrinsic_weight=0.08,
            variance_threshold=75.0,
            survival_window=80,
            decay_rate=0.88,
            epsilon_start=0.8,
            device=torch.device("cpu"),
        )

        # Add some history and modify state
        for i in range(20):
            original.update_on_episode_end(float(50 + i))
        original.anneal_weight()

        # Save state
        state = original.checkpoint_state()

        # Create new exploration and restore
        restored = AdaptiveIntrinsicExploration(obs_dim=10, device=torch.device("cpu"))
        restored.load_state(state)

        # Should match original
        assert abs(restored.current_intrinsic_weight - original.current_intrinsic_weight) < 1e-6
        assert abs(restored.min_intrinsic_weight - original.min_intrinsic_weight) < 1e-6
        assert abs(restored.variance_threshold - original.variance_threshold) < 1e-6
        assert restored.survival_window == original.survival_window
        assert abs(restored.decay_rate - original.decay_rate) < 1e-6
        assert restored.survival_history == original.survival_history

        # RND epsilon should also match
        assert abs(restored.rnd.epsilon - original.rnd.epsilon) < 1e-6


class TestIntrinsicWeightGetter:
    """Test intrinsic weight getter."""

    def test_get_intrinsic_weight_returns_current_value(self):
        """get_intrinsic_weight should return current weight."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=10,
            initial_intrinsic_weight=0.65,
            device=torch.device("cpu"),
        )

        assert abs(exploration.get_intrinsic_weight() - 0.65) < 1e-6

        # Anneal and check again
        exploration.anneal_weight()
        expected = 0.65 * 0.99  # Default decay_rate
        assert abs(exploration.get_intrinsic_weight() - expected) < 1e-6
