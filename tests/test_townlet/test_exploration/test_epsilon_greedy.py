"""Tests for EpsilonGreedyExploration (vectorized)."""

import pytest
import torch

from townlet.training.state import BatchedAgentState


def test_epsilon_greedy_select_actions():
    """EpsilonGreedy should select actions based on epsilon."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(
        epsilon=0.0,  # Greedy (always pick max Q)
        epsilon_decay=1.0,
    )

    # Q-values with clear max
    q_values = torch.tensor([
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Action 1 is best
        [0.0, 0.0, 2.0, 0.0, 0.0],  # Action 2 is best
    ])

    state = BatchedAgentState(
        observations=torch.randn(2, 70),
        actions=torch.zeros(2, dtype=torch.long),
        rewards=torch.zeros(2),
        dones=torch.zeros(2, dtype=torch.bool),
        epsilons=torch.zeros(2),  # Greedy
        intrinsic_rewards=torch.zeros(2),
        survival_times=torch.zeros(2, dtype=torch.long),
        curriculum_difficulties=torch.zeros(2),
        device=torch.device('cpu'),
    )

    actions = exploration.select_actions(q_values, state)

    assert actions[0] == 1  # Greedy selects action 1
    assert actions[1] == 2  # Greedy selects action 2


def test_epsilon_greedy_exploration():
    """High epsilon should produce random actions."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=1.0)

    q_values = torch.zeros(100, 5)  # All Q-values = 0
    q_values[:, 0] = 10.0  # Action 0 is "best"

    state = BatchedAgentState(
        observations=torch.randn(100, 70),
        actions=torch.zeros(100, dtype=torch.long),
        rewards=torch.zeros(100),
        dones=torch.zeros(100, dtype=torch.bool),
        epsilons=torch.ones(100),  # Full exploration
        intrinsic_rewards=torch.zeros(100),
        survival_times=torch.zeros(100, dtype=torch.long),
        curriculum_difficulties=torch.zeros(100),
        device=torch.device('cpu'),
    )

    actions = exploration.select_actions(q_values, state)

    # With epsilon=1.0, should get diverse actions (not all 0)
    unique_actions = torch.unique(actions)
    assert len(unique_actions) > 1  # Not all the same action


def test_epsilon_greedy_no_intrinsic_rewards():
    """EpsilonGreedy should return zero intrinsic rewards."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=0.995)

    observations = torch.randn(10, 70)
    intrinsic_rewards = exploration.compute_intrinsic_rewards(observations)

    assert intrinsic_rewards.shape == (10,)
    assert torch.all(intrinsic_rewards == 0.0)


def test_epsilon_greedy_checkpoint():
    """EpsilonGreedy checkpoint should include epsilon."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(epsilon=0.8, epsilon_decay=0.99)

    state = exploration.checkpoint_state()

    assert isinstance(state, dict)
    assert state['epsilon'] == 0.8
    assert state['epsilon_decay'] == 0.99

    # Restore
    new_exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=1.0)
    new_exploration.load_state(state)

    assert new_exploration.epsilon == 0.8
    assert new_exploration.epsilon_decay == 0.99
