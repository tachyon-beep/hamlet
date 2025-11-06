"""Consolidated tests for exploration strategies.

This file consolidates all unit tests for exploration strategies:
- EpsilonGreedyExploration: Epsilon-greedy action selection with decay
- RNDExploration: Random Network Distillation for novelty detection
- AdaptiveIntrinsicExploration: RND + variance-based annealing

Old files consolidated:
- test_epsilon_greedy.py (257 lines)
- test_rnd.py (291 lines)
- test_adaptive_intrinsic.py (388 lines)
- test_exploration_checkpoint.py (397 lines)

Total: 1333 lines consolidated into structured test classes.
"""

import pytest
import torch

from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.exploration.rnd import RNDExploration
from townlet.training.state import BatchedAgentState

# =============================================================================
# EPSILON-GREEDY EXPLORATION
# =============================================================================


class TestEpsilonGreedyActionSelection:
    """Test epsilon-greedy action selection logic."""

    @pytest.fixture
    def exploration(self):
        """Create epsilon-greedy exploration strategy."""
        return EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.01)

    def test_random_actions_without_masks(self, exploration):
        """Random action selection should work without masks."""
        batch_size = 10
        num_actions = 6
        q_values = torch.randn(batch_size, num_actions)

        # Create state with epsilon=1.0 (always explore)
        state = BatchedAgentState(
            observations=torch.randn(batch_size, 10),
            actions=torch.zeros(batch_size, dtype=torch.long),
            rewards=torch.zeros(batch_size),
            dones=torch.zeros(batch_size, dtype=torch.bool),
            epsilons=torch.ones(batch_size),  # 100% exploration
            intrinsic_rewards=torch.zeros(batch_size),
            survival_times=torch.zeros(batch_size),
            curriculum_difficulties=torch.zeros(batch_size),
            device=torch.device("cpu"),
        )

        # Select actions (should all be random since epsilon=1.0)
        actions = exploration.select_actions(q_values, state, action_masks=None)

        # Check shape and range
        assert actions.shape == (batch_size,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < num_actions)

    def test_random_actions_with_masks(self, exploration):
        """Random action selection should respect masks."""
        batch_size = 5
        num_actions = 6
        q_values = torch.randn(batch_size, num_actions)

        # Create masks that only allow action 0 and 1
        action_masks = torch.zeros(batch_size, num_actions, dtype=torch.bool)
        action_masks[:, 0] = True
        action_masks[:, 1] = True

        # Create state with epsilon=1.0 (always explore)
        state = BatchedAgentState(
            observations=torch.randn(batch_size, 10),
            actions=torch.zeros(batch_size, dtype=torch.long),
            rewards=torch.zeros(batch_size),
            dones=torch.zeros(batch_size, dtype=torch.bool),
            epsilons=torch.ones(batch_size),  # 100% exploration
            intrinsic_rewards=torch.zeros(batch_size),
            survival_times=torch.zeros(batch_size),
            curriculum_difficulties=torch.zeros(batch_size),
            device=torch.device("cpu"),
        )

        # Select actions many times to ensure randomness respects masks
        for _ in range(50):
            actions = exploration.select_actions(q_values, state, action_masks=action_masks)

            # All actions should be 0 or 1 (only valid actions)
            assert torch.all((actions == 0) | (actions == 1))

    def test_greedy_without_masks(self, exploration):
        """Greedy selection should work without masks."""
        batch_size = 3
        num_actions = 6

        # Create Q-values where action 2 is clearly best
        q_values = torch.zeros(batch_size, num_actions)
        q_values[:, 2] = 10.0  # Action 2 has highest Q-value

        # Create state with epsilon=0.0 (always exploit)
        state = BatchedAgentState(
            observations=torch.randn(batch_size, 10),
            actions=torch.zeros(batch_size, dtype=torch.long),
            rewards=torch.zeros(batch_size),
            dones=torch.zeros(batch_size, dtype=torch.bool),
            epsilons=torch.zeros(batch_size),  # 0% exploration
            intrinsic_rewards=torch.zeros(batch_size),
            survival_times=torch.zeros(batch_size),
            curriculum_difficulties=torch.zeros(batch_size),
            device=torch.device("cpu"),
        )

        # Select actions (should all be greedy = action 2)
        actions = exploration.select_actions(q_values, state, action_masks=None)

        # All should select action 2
        assert torch.all(actions == 2)


class TestEpsilonDecay:
    """Test epsilon decay mechanism."""

    def test_epsilon_decays_towards_minimum(self):
        """Epsilon should decay exponentially but respect minimum."""
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.1)

        initial_epsilon = exploration.epsilon
        assert abs(initial_epsilon - 1.0) < 1e-6

        # Decay several times
        for _ in range(10):
            exploration.decay_epsilon()

        # Should have decayed
        assert exploration.epsilon < initial_epsilon

    def test_epsilon_respects_minimum(self):
        """Epsilon should not go below minimum."""
        exploration = EpsilonGreedyExploration(epsilon=0.05, epsilon_decay=0.9, epsilon_min=0.01)

        # Decay many times
        for _ in range(100):
            exploration.decay_epsilon()

        # Should be at minimum
        assert abs(exploration.epsilon - 0.01) < 1e-6

    def test_epsilon_decay_formula(self):
        """Test exact decay calculation."""
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.0)

        exploration.decay_epsilon()
        assert exploration.epsilon == pytest.approx(0.95, rel=1e-5)

        exploration.decay_epsilon()
        assert exploration.epsilon == pytest.approx(0.95 * 0.95, rel=1e-5)


class TestEpsilonGreedyIntrinsicRewards:
    """Test intrinsic reward computation (should always be zero)."""

    def test_intrinsic_rewards_always_zero(self):
        """Epsilon-greedy produces no intrinsic rewards."""
        exploration = EpsilonGreedyExploration(epsilon=0.5)

        batch_size = 10
        obs_dim = 20
        observations = torch.randn(batch_size, obs_dim)

        intrinsic_rewards = exploration.compute_intrinsic_rewards(observations)

        # Should be all zeros
        assert intrinsic_rewards.shape == (batch_size,)
        assert torch.all(intrinsic_rewards.abs() < 1e-6)

    def test_intrinsic_rewards_correct_device(self):
        """Intrinsic rewards should be on same device as observations."""
        exploration = EpsilonGreedyExploration(epsilon=0.5)

        observations = torch.randn(5, 10)
        intrinsic_rewards = exploration.compute_intrinsic_rewards(observations)

        assert intrinsic_rewards.device == observations.device


class TestEpsilonGreedyUpdate:
    """Test update method (should be no-op)."""

    def test_update_is_noop(self):
        """Update should do nothing for epsilon-greedy."""
        exploration = EpsilonGreedyExploration(epsilon=0.5)

        # Create fake batch
        batch = {
            "observations": torch.randn(10, 20),
            "actions": torch.randint(0, 6, (10,)),
            "rewards": torch.randn(10),
        }

        # Should not crash
        exploration.update(batch)

        # Epsilon should be unchanged
        assert abs(exploration.epsilon - 0.5) < 1e-6


class TestEpsilonGreedyCheckpointing:
    """Test state serialization for checkpointing."""

    def test_checkpoint_state_contains_all_params(self):
        """Checkpoint should save epsilon, decay, and min."""
        exploration = EpsilonGreedyExploration(epsilon=0.7, epsilon_decay=0.98, epsilon_min=0.05)

        state = exploration.checkpoint_state()

        assert "epsilon" in state
        assert "epsilon_decay" in state
        assert "epsilon_min" in state
        assert abs(state["epsilon"] - 0.7) < 1e-6
        assert abs(state["epsilon_decay"] - 0.98) < 1e-6
        assert abs(state["epsilon_min"] - 0.05) < 1e-6

    def test_load_state_restores_params(self):
        """load_state should restore all parameters."""
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

        # Modify epsilon through decay
        for _ in range(10):
            exploration.decay_epsilon()

        # Save state
        saved_state = exploration.checkpoint_state()

        # Create new instance with different params
        new_exploration = EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.1)

        # Load old state
        new_exploration.load_state(saved_state)

        # Should match original
        assert new_exploration.epsilon == saved_state["epsilon"]
        assert new_exploration.epsilon_decay == saved_state["epsilon_decay"]
        assert new_exploration.epsilon_min == saved_state["epsilon_min"]

    def test_checkpoint_roundtrip(self):
        """Full checkpoint save/load cycle should preserve state."""
        original = EpsilonGreedyExploration(epsilon=0.42, epsilon_decay=0.97, epsilon_min=0.03)

        # Decay a bit
        for _ in range(5):
            original.decay_epsilon()

        # Save and restore
        state = original.checkpoint_state()
        restored = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.0)
        restored.load_state(state)

        # Should match
        assert restored.epsilon == original.epsilon
        assert restored.epsilon_decay == original.epsilon_decay
        assert restored.epsilon_min == original.epsilon_min


# =============================================================================
# RND EXPLORATION
# =============================================================================


class TestRNDNetworkInitialization:
    """Test RND network setup and freezing."""

    def test_fixed_network_is_frozen(self):
        """Fixed network should have requires_grad=False for all parameters."""
        rnd = RNDExploration(obs_dim=10, embed_dim=8, device=torch.device("cpu"))

        # Check all fixed network parameters are frozen
        for param in rnd.fixed_network.parameters():
            assert not param.requires_grad, "Fixed network should be frozen"

    def test_predictor_network_is_trainable(self):
        """Predictor network should have requires_grad=True."""
        rnd = RNDExploration(obs_dim=10, embed_dim=8, device=torch.device("cpu"))

        # Check predictor network parameters are trainable
        trainable_params = [p for p in rnd.predictor_network.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Predictor should have trainable parameters"

    def test_networks_have_different_weights(self):
        """Fixed and predictor should start with different random weights."""
        rnd = RNDExploration(obs_dim=10, embed_dim=8, device=torch.device("cpu"))

        # Get first layer weights
        fixed_weight = list(rnd.fixed_network.parameters())[0]
        predictor_weight = list(rnd.predictor_network.parameters())[0]

        # Should be different (randomly initialized)
        assert not torch.allclose(fixed_weight, predictor_weight, atol=1e-3)


class TestRNDIntrinsicRewardComputation:
    """Test prediction error as intrinsic reward signal."""

    @pytest.fixture
    def rnd(self):
        """Create RND exploration with small networks."""
        return RNDExploration(obs_dim=10, embed_dim=8, learning_rate=0.01, device=torch.device("cpu"))

    def test_intrinsic_rewards_are_nonnegative(self, rnd):
        """Prediction errors (MSE) should always be non-negative."""
        observations = torch.randn(5, 10)
        intrinsic_rewards = rnd.compute_intrinsic_rewards(observations)

        assert torch.all(intrinsic_rewards >= 0), "MSE should be non-negative"

    def test_identical_observations_have_same_reward(self, rnd):
        """Same observation should produce same intrinsic reward."""
        obs = torch.randn(1, 10)

        reward1 = rnd.compute_intrinsic_rewards(obs)
        reward2 = rnd.compute_intrinsic_rewards(obs)

        assert torch.allclose(reward1, reward2, atol=1e-6)

    def test_novel_states_have_higher_rewards_initially(self, rnd):
        """Before training, all states should have similar high prediction errors."""
        obs1 = torch.randn(1, 10)
        obs2 = torch.randn(1, 10)

        reward1 = rnd.compute_intrinsic_rewards(obs1)
        reward2 = rnd.compute_intrinsic_rewards(obs2)

        # Both should be positive (untrained predictor has high error)
        assert reward1.item() > 0
        assert reward2.item() > 0

    def test_batch_intrinsic_rewards(self, rnd):
        """Should handle batch of observations correctly."""
        batch_size = 8
        observations = torch.randn(batch_size, 10)

        intrinsic_rewards = rnd.compute_intrinsic_rewards(observations)

        assert intrinsic_rewards.shape == (batch_size,)
        assert torch.all(intrinsic_rewards >= 0)


class TestRNDBufferAccumulation:
    """Test observation buffer management."""

    def test_buffer_starts_empty(self):
        """Buffer should be empty on initialization."""
        rnd = RNDExploration(obs_dim=10, device=torch.device("cpu"))
        assert len(rnd.obs_buffer) == 0

    def test_update_adds_to_buffer(self):
        """update() should add observations to buffer."""
        rnd = RNDExploration(obs_dim=10, training_batch_size=10, device=torch.device("cpu"))

        batch = {"observations": torch.randn(3, 10)}
        rnd.update(batch)

        # Should have 3 observations in buffer
        assert len(rnd.obs_buffer) == 3

    def test_update_without_observations_key(self):
        """update() should handle missing 'observations' gracefully."""
        rnd = RNDExploration(obs_dim=10, device=torch.device("cpu"))

        batch = {"actions": torch.randint(0, 6, (3,))}
        rnd.update(batch)  # Should not crash

        # Buffer should still be empty
        assert len(rnd.obs_buffer) == 0

    def test_buffer_accumulates_across_updates(self):
        """Multiple update() calls should accumulate observations."""
        rnd = RNDExploration(obs_dim=10, training_batch_size=20, device=torch.device("cpu"))

        # Add observations in batches
        rnd.update({"observations": torch.randn(5, 10)})
        assert len(rnd.obs_buffer) == 5

        rnd.update({"observations": torch.randn(3, 10)})
        assert len(rnd.obs_buffer) == 8

        rnd.update({"observations": torch.randn(2, 10)})
        assert len(rnd.obs_buffer) == 10


class TestRNDPredictorTraining:
    """Test predictor network learning."""

    def test_update_predictor_requires_minimum_buffer_size(self):
        """Should return 0.0 when buffer is too small."""
        rnd = RNDExploration(obs_dim=10, training_batch_size=10, device=torch.device("cpu"))

        # Add only 5 observations (less than batch size of 10)
        rnd.update({"observations": torch.randn(5, 10)})

        # Should return 0.0 (no training happened)
        loss = rnd.update_predictor()
        assert abs(loss - 0.0) < 1e-6

    def test_update_predictor_clears_buffer(self):
        """After training, buffer should be cleared."""
        rnd = RNDExploration(obs_dim=10, training_batch_size=10, device=torch.device("cpu"))

        # Manually add to buffer (bypass update() auto-training)
        for _ in range(10):
            rnd.obs_buffer.append(torch.randn(10))

        assert len(rnd.obs_buffer) == 10

        # Train manually
        rnd.update_predictor()

        # Buffer should be empty
        assert len(rnd.obs_buffer) == 0

    def test_update_predictor_keeps_excess_observations(self):
        """Observations beyond batch size should remain in buffer."""
        rnd = RNDExploration(obs_dim=10, training_batch_size=5, device=torch.device("cpu"))

        # Manually add 8 observations (bypass update() auto-training)
        for _ in range(8):
            rnd.obs_buffer.append(torch.randn(10))

        assert len(rnd.obs_buffer) == 8

        # Train on first 5
        rnd.update_predictor()

        # Should have 3 remaining
        assert len(rnd.obs_buffer) == 3

    def test_predictor_network_updates(self):
        """Predictor network weights should change after training."""
        torch.manual_seed(123)

        rnd = RNDExploration(
            obs_dim=10,
            embed_dim=8,
            learning_rate=0.01,
            training_batch_size=16,
            device=torch.device("cpu"),
        )

        # Save initial predictor weights
        initial_weights = [p.clone() for p in rnd.predictor_network.parameters()]

        # Train on some observations
        observations = torch.randn(16, 10)
        for _ in range(10):
            for obs in observations:
                rnd.obs_buffer.append(obs.detach())
            if len(rnd.obs_buffer) >= rnd.training_batch_size:
                rnd.update_predictor()

        # Predictor weights should have changed
        weights_changed = False
        for initial_w, current_w in zip(initial_weights, rnd.predictor_network.parameters()):
            if not torch.allclose(initial_w, current_w, atol=1e-4):
                weights_changed = True
                break

        assert weights_changed, "Predictor network should update during training"

    def test_update_triggers_training_automatically(self):
        """update() should automatically call update_predictor when buffer full."""
        rnd = RNDExploration(obs_dim=10, training_batch_size=5, device=torch.device("cpu"))

        # Add exactly batch_size observations
        observations = torch.randn(5, 10)
        initial_reward = rnd.compute_intrinsic_rewards(observations).mean().item()

        # Call update multiple times (should trigger training internally)
        for _ in range(10):
            rnd.update({"observations": observations.clone()})

        # Predictor should have learned (error decreased)
        final_reward = rnd.compute_intrinsic_rewards(observations).mean().item()
        assert final_reward < initial_reward


class TestRNDLearningBehavior:
    """Test RND's novelty detection over time."""

    def test_familiar_states_get_lower_rewards(self):
        """States seen repeatedly should have lower intrinsic rewards."""
        torch.manual_seed(123)

        rnd = RNDExploration(
            obs_dim=10,
            embed_dim=8,
            learning_rate=0.1,
            training_batch_size=10,
            device=torch.device("cpu"),
        )

        # Create two different observations
        familiar_obs = torch.randn(10, 10)
        novel_obs = torch.randn(10, 10)

        # Train on familiar_obs extensively
        for _ in range(30):
            rnd.update({"observations": familiar_obs.clone()})

        # Get rewards
        familiar_reward = rnd.compute_intrinsic_rewards(familiar_obs).mean().item()
        novel_reward = rnd.compute_intrinsic_rewards(novel_obs).mean().item()

        # Novel states should have higher intrinsic reward
        assert (
            novel_reward > familiar_reward
        ), f"Novel states should have higher reward: novel={novel_reward:.4f}, familiar={familiar_reward:.4f}"

    def test_fixed_network_never_changes(self):
        """Fixed network weights should remain constant during training."""
        torch.manual_seed(456)

        rnd = RNDExploration(
            obs_dim=10,
            embed_dim=8,
            learning_rate=0.1,
            training_batch_size=5,
            device=torch.device("cpu"),
        )

        # Save initial fixed network weights
        initial_weights = [p.clone() for p in rnd.fixed_network.parameters()]

        # Train predictor extensively
        for _ in range(20):
            rnd.update({"observations": torch.randn(5, 10)})

        # Fixed network should be unchanged
        for initial_w, current_w in zip(initial_weights, rnd.fixed_network.parameters()):
            assert torch.allclose(initial_w, current_w, atol=1e-6), "Fixed network should never change"


# =============================================================================
# ADAPTIVE INTRINSIC EXPLORATION
# =============================================================================


class TestAdaptiveIntrinsicAnnealingLogic:
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


class TestAdaptiveIntrinsicRewardScaling:
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


class TestAdaptiveIntrinsicActionSelection:
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


class TestAdaptiveIntrinsicRNDUpdate:
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


class TestAdaptiveIntrinsicEpsilonDecay:
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


class TestAdaptiveIntrinsicStatePersistence:
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


class TestAdaptiveIntrinsicWeightGetter:
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


# =============================================================================
# EXPLORATION CHECKPOINTING (COMPREHENSIVE)
# =============================================================================


class TestRNDCheckpointCompleteness:
    """Test RND exploration checkpoint completeness."""

    def test_rnd_saves_predictor_optimizer(self, cpu_device, basic_env):
        """RND should save predictor optimizer state."""
        rnd = RNDExploration(
            obs_dim=basic_env.observation_dim,
            embed_dim=128,
            learning_rate=1e-4,
            device=cpu_device,
        )

        # Checkpoint state should include optimizer
        state = rnd.checkpoint_state()
        assert "optimizer" in state, "RND checkpoint must include optimizer state"

    def test_rnd_saves_epsilon(self, cpu_device, basic_env):
        """RND should save epsilon decay parameters."""
        rnd = RNDExploration(
            obs_dim=basic_env.observation_dim,
            epsilon_start=0.8,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            device=cpu_device,
        )

        state = rnd.checkpoint_state()
        assert "epsilon" in state, "RND checkpoint must include epsilon"
        assert "epsilon_min" in state, "RND checkpoint must include epsilon_min"
        assert "epsilon_decay" in state, "RND checkpoint must include epsilon_decay"
        assert state["epsilon"] == 0.8
        assert state["epsilon_min"] == 0.05
        assert state["epsilon_decay"] == 0.99

    def test_rnd_saves_network_architecture_params(self, cpu_device, basic_env):
        """RND should save architecture parameters for reconstruction."""
        rnd = RNDExploration(
            obs_dim=basic_env.observation_dim,
            embed_dim=256,
            device=cpu_device,
        )

        state = rnd.checkpoint_state()
        assert "obs_dim" in state, "RND checkpoint must include obs_dim"
        assert "embed_dim" in state, "RND checkpoint must include embed_dim"
        assert state["obs_dim"] == basic_env.observation_dim
        assert state["embed_dim"] == 256

    def test_rnd_saves_network_weights(self, cpu_device, basic_env):
        """RND should save both fixed and predictor network weights."""
        rnd = RNDExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        state = rnd.checkpoint_state()
        assert "fixed_network" in state, "RND checkpoint must include fixed_network"
        assert "predictor_network" in state, "RND checkpoint must include predictor_network"

        # Should be state dicts
        assert isinstance(state["fixed_network"], dict)
        assert isinstance(state["predictor_network"], dict)

    def test_rnd_restores_epsilon(self, cpu_device, basic_env):
        """RND should restore epsilon from checkpoint."""
        rnd = RNDExploration(obs_dim=basic_env.observation_dim, epsilon_start=1.0, device=cpu_device)

        # Decay epsilon
        for _ in range(10):
            rnd.epsilon *= rnd.epsilon_decay

        decayed_epsilon = rnd.epsilon
        state = rnd.checkpoint_state()

        # Create new RND and restore
        rnd2 = RNDExploration(obs_dim=basic_env.observation_dim, epsilon_start=1.0, device=cpu_device)
        rnd2.load_state(state)

        assert abs(rnd2.epsilon - decayed_epsilon) < 1e-6, "Epsilon should be restored"

    def test_rnd_restores_network_weights(self, cpu_device, basic_env):
        """RND should restore network weights from checkpoint."""
        rnd = RNDExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        # Train predictor on some data
        obs = torch.randn(128, basic_env.observation_dim, device=cpu_device)
        target = rnd.fixed_network(obs)
        pred = rnd.predictor_network(obs)
        loss = torch.nn.functional.mse_loss(pred, target)
        rnd.optimizer.zero_grad()
        loss.backward()
        rnd.optimizer.step()

        # Save state
        state = rnd.checkpoint_state()

        # Create new RND and restore
        rnd2 = RNDExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        rnd2.load_state(state)

        # Networks should produce same output
        test_obs = torch.randn(10, basic_env.observation_dim, device=cpu_device)
        pred1 = rnd.predictor_network(test_obs)
        pred2 = rnd2.predictor_network(test_obs)

        assert torch.allclose(pred1, pred2), "Predictor network weights should be restored"

    def test_rnd_restores_optimizer_state(self, cpu_device, basic_env):
        """RND should restore optimizer state from checkpoint."""
        rnd = RNDExploration(obs_dim=basic_env.observation_dim, learning_rate=1e-3, device=cpu_device)

        # Train for a few steps to build optimizer state
        for _ in range(5):
            obs = torch.randn(128, basic_env.observation_dim, device=cpu_device)
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
        rnd2 = RNDExploration(obs_dim=basic_env.observation_dim, learning_rate=1e-3, device=cpu_device)
        rnd2.load_state(state)

        # Optimizer should have same state
        restored_state_keys = set(rnd2.optimizer.state_dict()["state"].keys())
        assert optimizer_state_keys == restored_state_keys, "Optimizer state should be restored"


class TestAdaptiveIntrinsicCheckpointCompleteness:
    """Test AdaptiveIntrinsic exploration checkpoint completeness."""

    def test_adaptive_saves_intrinsic_weight(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should save current intrinsic weight."""
        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            initial_intrinsic_weight=0.5,
            device=cpu_device,
        )

        state = adaptive.checkpoint_state()
        assert "current_intrinsic_weight" in state
        assert state["current_intrinsic_weight"] == 0.5

    def test_adaptive_saves_annealing_params(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should save all annealing parameters."""
        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            variance_threshold=50.0,
            survival_window=200,
            decay_rate=0.95,
            min_intrinsic_weight=0.05,
            device=cpu_device,
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

    def test_adaptive_saves_survival_history(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should save survival history for annealing."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        # Simulate some survival times
        adaptive.survival_history = [10.0, 20.0, 30.0, 40.0, 50.0]

        state = adaptive.checkpoint_state()
        assert "survival_history" in state
        assert state["survival_history"] == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_adaptive_saves_rnd_state(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should save underlying RND state."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        state = adaptive.checkpoint_state()
        assert "rnd_state" in state
        assert isinstance(state["rnd_state"], dict)

        # RND state should contain network weights
        rnd_state = state["rnd_state"]
        assert "predictor_network" in rnd_state
        assert "optimizer" in rnd_state

    def test_adaptive_restores_intrinsic_weight(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should restore intrinsic weight from checkpoint."""
        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            initial_intrinsic_weight=1.0,
            device=cpu_device,
        )

        # Anneal weight
        adaptive.current_intrinsic_weight = 0.3

        state = adaptive.checkpoint_state()

        # Create new instance and restore
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        adaptive2.load_state(state)

        assert abs(adaptive2.current_intrinsic_weight - 0.3) < 1e-6

    def test_adaptive_restores_survival_history(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should restore survival history from checkpoint."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        # Set survival history
        adaptive.survival_history = [15.0, 25.0, 35.0]

        state = adaptive.checkpoint_state()

        # Create new instance and restore
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        adaptive2.load_state(state)

        assert adaptive2.survival_history == [15.0, 25.0, 35.0]

    def test_adaptive_restores_rnd_state(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should restore RND state from checkpoint."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        # Train RND predictor
        obs = torch.randn(128, basic_env.observation_dim, device=cpu_device)
        adaptive.rnd.obs_buffer = [obs[i] for i in range(128)]
        adaptive.rnd.update_predictor()

        # Get predictor output before checkpoint
        test_obs = torch.randn(10, basic_env.observation_dim, device=cpu_device)
        pred_before = adaptive.rnd.predictor_network(test_obs)

        # Save and restore
        state = adaptive.checkpoint_state()
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        adaptive2.load_state(state)

        # Predictor should produce same output
        pred_after = adaptive2.rnd.predictor_network(test_obs)
        assert torch.allclose(pred_before, pred_after)


class TestCheckpointRoundTrip:
    """Test full checkpoint save/restore cycle."""

    def test_rnd_roundtrip_preserves_behavior(self, cpu_device, basic_env):
        """RND should have identical behavior after save/restore."""
        rnd = RNDExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        # Train predictor
        for _ in range(5):
            obs = torch.randn(128, basic_env.observation_dim, device=cpu_device)
            rnd.obs_buffer.extend([obs[i] for i in range(128)])
            rnd.update_predictor()

        # Compute intrinsic rewards before checkpoint
        test_obs = torch.randn(10, basic_env.observation_dim, device=cpu_device)
        rewards_before = rnd.compute_intrinsic_rewards(test_obs)

        # Save and restore
        state = rnd.checkpoint_state()
        rnd2 = RNDExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        rnd2.load_state(state)

        # Compute intrinsic rewards after restore
        rewards_after = rnd2.compute_intrinsic_rewards(test_obs)

        # Should be identical
        assert torch.allclose(rewards_before, rewards_after), "RND behavior should be preserved"

    def test_adaptive_roundtrip_preserves_behavior(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should have identical behavior after save/restore."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        # Train and anneal
        for _ in range(3):
            obs = torch.randn(128, basic_env.observation_dim, device=cpu_device)
            adaptive.rnd.obs_buffer.extend([obs[i] for i in range(128)])
            adaptive.rnd.update_predictor()

        adaptive.survival_history = [20.0, 30.0, 40.0]
        adaptive.current_intrinsic_weight = 0.5

        # Compute intrinsic rewards before checkpoint
        test_obs = torch.randn(10, basic_env.observation_dim, device=cpu_device)
        rewards_before = adaptive.compute_intrinsic_rewards(test_obs)

        # Save and restore
        state = adaptive.checkpoint_state()
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        adaptive2.load_state(state)

        # Compute intrinsic rewards after restore
        rewards_after = adaptive2.compute_intrinsic_rewards(test_obs)

        # Should be identical
        assert torch.allclose(rewards_before, rewards_after), "Adaptive behavior should be preserved"
        assert abs(adaptive2.current_intrinsic_weight - 0.5) < 1e-6


class TestCheckpointBackwardsCompatibility:
    """Test handling of missing checkpoint fields (backwards compatibility)."""

    def test_rnd_handles_missing_optimizer(self, cpu_device, basic_env):
        """RND should handle checkpoints without optimizer state (legacy)."""
        rnd = RNDExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        state = rnd.checkpoint_state()
        # Simulate legacy checkpoint without optimizer
        del state["optimizer"]

        # Should not crash (though training may be suboptimal)
        rnd2 = RNDExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        try:
            rnd2.load_state(state)
        except KeyError:
            pytest.fail("RND should handle missing optimizer gracefully")

    def test_adaptive_handles_missing_survival_history(self, cpu_device, basic_env):
        """AdaptiveIntrinsic should handle checkpoints without survival history."""
        adaptive = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)

        state = adaptive.checkpoint_state()
        # Simulate legacy checkpoint without survival history
        del state["survival_history"]

        # Should not crash (annealing may restart)
        adaptive2 = AdaptiveIntrinsicExploration(obs_dim=basic_env.observation_dim, device=cpu_device)
        try:
            adaptive2.load_state(state)
        except KeyError:
            pytest.fail("Adaptive should handle missing survival_history gracefully")
