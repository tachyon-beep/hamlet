"""
Test Suite: Epsilon-Greedy Exploration Strategy

Tests for epsilon-greedy action selection to ensure:
1. Random action sampling respects action masks
2. Epsilon decay works correctly
3. Intrinsic rewards are always zero
4. State serialization works for checkpointing

Coverage Target: exploration/epsilon_greedy.py (75% -> ~95%)
"""

import pytest
import torch

from src.townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from src.townlet.training.state import BatchedAgentState


class TestEpsilonGreedyActionSelection:
    """Test epsilon-greedy action selection logic."""

    @pytest.fixture
    def exploration(self):
        """Create epsilon-greedy exploration strategy."""
        return EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.01)

    def test_random_actions_without_masks(self, exploration):
        """Random action selection should work without masks."""
        batch_size = 10
        num_actions = 5
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
        """Random action selection should respect masks (line 80)."""
        batch_size = 5
        num_actions = 5
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
        """Greedy selection should work without masks (line 66)."""
        batch_size = 3
        num_actions = 5

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


class TestIntrinsicRewards:
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


class TestUpdateMethod:
    """Test update method (should be no-op)."""

    def test_update_is_noop(self):
        """Update should do nothing for epsilon-greedy."""
        exploration = EpsilonGreedyExploration(epsilon=0.5)

        # Create fake batch
        batch = {
            "observations": torch.randn(10, 20),
            "actions": torch.randint(0, 5, (10,)),
            "rewards": torch.randn(10),
        }

        # Should not crash
        exploration.update(batch)

        # Epsilon should be unchanged
        assert abs(exploration.epsilon - 0.5) < 1e-6


class TestStateCheckpointing:
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
