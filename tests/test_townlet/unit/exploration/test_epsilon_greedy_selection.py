"""Tests for epsilon_greedy_action_selection utility function.

This file tests the shared action selection utility used by multiple exploration strategies.

Coverage Target: epsilon_greedy_action_selection() in townlet.exploration.action_selection

Critical Behaviors:
1. Greedy actions selected with probability (1 - epsilon)
2. Random valid actions selected with probability epsilon
3. Action masking respected (invalid actions never selected)
4. Edge case: all-False action_masks should return greedy action (BUG-23)
"""

import torch

from townlet.exploration.action_selection import epsilon_greedy_action_selection


class TestEpsilonGreedyActionSelection:
    """Test epsilon-greedy action selection with masking."""

    def test_all_false_action_masks_returns_greedy_action(self):
        """When all actions are invalid (dead agents), should return greedy action instead of sampling.

        BUG-23: Previously, the defensive epsilon add (probs + 1e-8) created a uniform
        distribution over all actions, including invalid ones. This test ensures we fall
        back to the greedy action when no valid actions exist.
        """
        batch_size = 3
        num_actions = 8

        # Setup Q-values - agent 1 has highest Q for action 2
        q_values = torch.randn(batch_size, num_actions)
        q_values[1, 2] = 100.0  # Force greedy action = 2 for agent 1

        # Epsilon values - agent 1 will explore (epsilon=1.0)
        epsilons = torch.tensor([0.0, 1.0, 0.0])  # Only agent 1 explores

        # Action masks - agent 1 has ALL actions invalid (dead agent)
        action_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
        action_masks[1, :] = False  # Agent 1: all False

        # Act
        actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)

        # Assert
        # Agent 0: epsilon=0.0, should select greedy (argmax with masking)
        assert actions[0] in range(num_actions)

        # Agent 1: epsilon=1.0 but ALL actions invalid
        # Should fall back to greedy action (action 2, the argmax of q_values[1])
        # NOT sample uniformly from invalid actions
        assert actions[1] == 2, f"Expected greedy action 2 for dead agent, got {actions[1]}"

        # Agent 2: epsilon=0.0, should select greedy
        assert actions[2] in range(num_actions)

    def test_epsilon_zero_always_greedy(self):
        """With epsilon=0, should always select greedy action (argmax)."""
        batch_size = 4
        num_actions = 8

        q_values = torch.randn(batch_size, num_actions)
        epsilons = torch.zeros(batch_size)
        action_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Compute expected greedy actions
        expected_greedy = torch.argmax(q_values, dim=1)

        # Select multiple times - should always be identical (deterministic)
        for _ in range(10):
            actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)
            assert torch.equal(actions, expected_greedy)

    def test_epsilon_one_samples_only_valid_actions(self):
        """With epsilon=1, should only sample from valid (True) actions."""
        batch_size = 2
        num_actions = 8

        q_values = torch.randn(batch_size, num_actions)
        epsilons = torch.ones(batch_size)  # Always explore

        # Mask out actions 0, 1, 2 for all agents
        action_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
        action_masks[:, 0:3] = False

        valid_actions = [3, 4, 5, 6, 7]

        # Sample many times - should NEVER see actions 0, 1, 2
        for _ in range(100):
            actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)
            assert all(a.item() in valid_actions for a in actions)

    def test_no_action_masks_samples_all_actions(self):
        """Without action_masks, should sample uniformly from all actions."""
        batch_size = 2
        num_actions = 6

        q_values = torch.randn(batch_size, num_actions)
        epsilons = torch.ones(batch_size)  # Always random

        # No action masks provided
        actions_seen = set()
        for _ in range(100):
            actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks=None)
            actions_seen.update(actions.tolist())

        # Should see variety (all actions valid)
        assert len(actions_seen) >= 4  # Probabilistic, but very likely

    def test_mixed_epsilon_values(self):
        """Different agents can have different epsilon values."""
        batch_size = 3
        num_actions = 8

        q_values = torch.randn(batch_size, num_actions)
        epsilons = torch.tensor([0.0, 0.5, 1.0])
        action_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Just verify no errors and valid action indices
        actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)

        assert actions.shape == (batch_size,)
        assert all(0 <= a < num_actions for a in actions.tolist())

    def test_action_masking_respects_boundaries(self):
        """Action masking should prevent invalid boundary actions."""
        batch_size = 2
        num_actions = 6

        q_values = torch.randn(batch_size, num_actions)
        q_values[:, 0] = 100.0  # Force action 0 to be highest Q-value

        epsilons = torch.zeros(batch_size)  # Greedy only

        # Mask out action 0 (boundary violation)
        action_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
        action_masks[:, 0] = False

        # Select many times
        for _ in range(50):
            actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)
            # Should NEVER select action 0 (masked)
            assert all(a != 0 for a in actions.tolist())
