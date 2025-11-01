"""
Test Suite: Random Network Distillation (RND) Exploration

Tests for RND intrinsic motivation to ensure:
1. Fixed network remains frozen during training
2. Prediction error reflects novelty (high for new, low for repeated)
3. Predictor network learns to match fixed network
4. Buffer accumulation and training triggers work correctly

Coverage Target: exploration/rnd.py (20% -> ~70%)

Focus: Core RND logic, not duplicated epsilon-greedy behavior
"""

import pytest
import torch

from townlet.exploration.rnd import RNDExploration


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


class TestIntrinsicRewardComputation:
    """Test prediction error as intrinsic reward signal."""

    @pytest.fixture
    def rnd(self):
        """Create RND exploration with small networks."""
        return RNDExploration(
            obs_dim=10, embed_dim=8, learning_rate=0.01, device=torch.device("cpu")
        )

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


class TestBufferAccumulation:
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

        batch = {"actions": torch.randint(0, 5, (3,))}
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


class TestPredictorTraining:
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
        torch.manual_seed(123)  # Different seed

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
        assert novel_reward > familiar_reward, (
            f"Novel states should have higher reward: "
            f"novel={novel_reward:.4f}, familiar={familiar_reward:.4f}"
        )

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
            assert torch.allclose(initial_w, current_w, atol=1e-6), (
                "Fixed network should never change"
            )
