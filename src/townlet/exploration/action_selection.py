"""
Shared action selection utilities for exploration strategies.

This module contains common action selection logic used across multiple
exploration strategies to avoid duplication.
"""

import torch


def epsilon_greedy_action_selection(
    q_values: torch.Tensor,  # [batch, num_actions]
    epsilons: torch.Tensor,  # [batch] per-agent epsilon values
    action_masks: torch.Tensor | None = None,  # [batch, num_actions] bool
) -> torch.Tensor:
    """
    Select actions using epsilon-greedy strategy with action masking.

    This is a shared utility used by multiple exploration strategies:
    - EpsilonGreedyExploration
    - RNDExploration
    - AdaptiveIntrinsicExploration
    - VectorizedPopulation (for inference)

    Args:
        q_values: Q-values for each action [batch, num_actions]
        epsilons: Per-agent exploration rates [batch] (0.0 = greedy, 1.0 = random)
        action_masks: Optional validity masks [batch, num_actions] bool (True = valid)

    Returns:
        actions: [batch] selected action indices (long)

    Example:
        >>> q_values = torch.randn(4, 5)  # 4 agents, 5 actions
        >>> epsilons = torch.tensor([0.1, 0.5, 0.0, 1.0])
        >>> action_masks = torch.ones(4, 5, dtype=torch.bool)
        >>> actions = epsilon_greedy_action_selection(q_values, epsilons, action_masks)
        >>> actions.shape
        torch.Size([4])
    """
    batch_size, num_actions = q_values.shape
    device = q_values.device

    # Apply action masking to Q-values if provided
    if action_masks is not None:
        masked_q_values = q_values.clone()
        masked_q_values[~action_masks] = float("-inf")
    else:
        masked_q_values = q_values

    # Greedy actions (argmax of masked Q-values)
    greedy_actions = torch.argmax(masked_q_values, dim=1)

    # Random actions (sample only from valid actions)
    if action_masks is not None:
        # Sample from valid actions per agent
        random_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            valid_actions = torch.where(action_masks[i])[0]
            if len(valid_actions) == 0:
                # Fallback: if no valid actions (shouldn't happen), use action 0
                random_actions[i] = 0
            else:
                random_idx = torch.randint(0, len(valid_actions), (1,), device=device)
                random_actions[i] = valid_actions[random_idx]
    else:
        # No masking: sample uniformly from all actions
        random_actions = torch.randint(0, num_actions, (batch_size,), device=device)

    # Epsilon mask: True = explore (random), False = exploit (greedy)
    explore_mask = torch.rand(batch_size, device=device) < epsilons

    # Select based on mask
    actions = torch.where(explore_mask, random_actions, greedy_actions)

    return actions
