"""Tests for shared action selection utilities."""

import torch

from townlet.exploration.action_selection import epsilon_greedy_action_selection


class TestEpsilonGreedyActionSelection:
    """Test the shared epsilon-greedy action selection utility."""

    def test_greedy_selection_when_epsilon_zero(self):
        """When epsilon=0, should always select best action."""
        q_values = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.5, 0.1],  # Best: action 2
                [0.5, 0.1, 1.0, 2.0, 3.0],  # Best: action 4
            ]
        )
        epsilons = torch.zeros(2)

        actions = epsilon_greedy_action_selection(q_values, epsilons)

        assert actions[0] == 2
        assert actions[1] == 4

    def test_respects_action_masks(self):
        """Should only select from valid actions."""
        q_values = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.5, 0.1],  # Best: action 2, but it's masked
            ]
        )
        action_masks = torch.tensor(
            [
                [True, True, False, True, True],  # Action 2 invalid
            ]
        )
        epsilons = torch.zeros(1)  # Greedy

        actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)

        # Should select action 1 (second best, and valid)
        assert actions[0] == 1

    def test_random_selection_when_epsilon_one(self):
        """When epsilon=1, should select random action."""
        torch.manual_seed(42)
        q_values = torch.randn(10, 5)
        epsilons = torch.ones(10)

        actions = epsilon_greedy_action_selection(q_values, epsilons)

        # Should be random (not all the same)
        unique_actions = torch.unique(actions)
        assert len(unique_actions) > 1

    def test_per_agent_epsilon(self):
        """Each agent can have different epsilon value."""
        q_values = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.5, 0.1],
                [1.0, 2.0, 3.0, 0.5, 0.1],
            ]
        )
        epsilons = torch.tensor([0.0, 0.0])  # Both greedy

        actions = epsilon_greedy_action_selection(q_values, epsilons)

        # Both should select action 2 (best)
        assert torch.all(actions == 2)

    def test_batch_processing(self):
        """Should handle arbitrary batch sizes."""
        batch_size = 32
        num_actions = 5

        q_values = torch.randn(batch_size, num_actions)
        epsilons = torch.rand(batch_size) * 0.5  # Varied epsilons

        actions = epsilon_greedy_action_selection(q_values, epsilons)

        assert actions.shape == (batch_size,)
        assert actions.dtype == torch.long
        assert torch.all((actions >= 0) & (actions < num_actions))

    def test_no_masking_fallback(self):
        """Should work without action masks."""
        q_values = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.5, 0.1],
            ]
        )
        epsilons = torch.zeros(1)

        actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks=None)

        assert actions[0] == 2  # Best action

    def test_random_respects_masks(self):
        """Random exploration should only sample valid actions."""
        torch.manual_seed(42)
        q_values = torch.randn(100, 5)
        epsilons = torch.ones(100)  # Always explore
        action_masks = torch.tensor([[True, True, False, False, False]] * 100)

        actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)

        # Should only select actions 0 or 1
        assert torch.all((actions == 0) | (actions == 1))

    def test_handles_single_valid_action(self):
        """Should handle case where only one action is valid."""
        q_values = torch.randn(5, 5)
        epsilons = torch.ones(5)  # Always explore
        action_masks = torch.zeros(5, 5, dtype=torch.bool)
        action_masks[:, 3] = True  # Only action 3 is valid

        actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)

        # All should select action 3
        assert torch.all(actions == 3)

    def test_fallback_when_no_valid_actions(self):
        """Should fallback to action 0 if no valid actions (shouldn't happen in practice)."""
        q_values = torch.randn(2, 5)
        epsilons = torch.ones(2)  # Always explore
        action_masks = torch.zeros(2, 5, dtype=torch.bool)  # No valid actions!

        actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)

        # Should fallback to 0
        assert torch.all(actions == 0)
