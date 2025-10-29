"""Tests for AdversarialCurriculum."""

import pytest
import torch
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.training.state import BatchedAgentState


def test_adversarial_curriculum_construction():
    """AdversarialCurriculum should initialize with stage 1 defaults."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=torch.device('cpu'),
    )

    assert curriculum.max_steps_per_episode == 500
    assert curriculum.device.type == 'cpu'
    assert curriculum.tracker is None  # Not initialized until population set

    # Stage 1 specs (stage helpers should work)
    assert curriculum._get_active_meters(1) == ['energy', 'hygiene']
    assert curriculum._get_depletion_multiplier(1) == 0.2
    assert curriculum._get_reward_mode(1) == 'shaped'


def test_advancement_when_mastery_achieved():
    """Agents should advance when survival high + learning positive + entropy low."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        survival_advance_threshold=0.7,
        entropy_gate=0.5,
        min_steps_at_stage=100,  # Low for testing
        device=torch.device('cpu'),
    )
    curriculum.initialize_population(num_agents=3)

    # Simulate mastery conditions
    curriculum.tracker.episode_steps = torch.tensor([400.0, 450.0, 420.0])  # High survival
    curriculum.tracker.episode_rewards = torch.tensor([80.0, 90.0, 85.0])  # Good rewards
    curriculum.tracker.prev_avg_reward = torch.tensor([0.1, 0.1, 0.1])  # Learning progress
    curriculum.tracker.steps_at_stage = torch.tensor([150.0, 150.0, 150.0])  # Enough steps

    # Mock agent states with low entropy (converged policy)
    agent_states = BatchedAgentState(
        observations=torch.zeros(3, 70),
        actions=torch.zeros(3, dtype=torch.long),
        rewards=torch.zeros(3),
        dones=torch.zeros(3, dtype=torch.bool),
        epsilons=torch.tensor([0.1, 0.1, 0.1]),  # Low exploration
        intrinsic_rewards=torch.zeros(3),
        survival_times=torch.tensor([400.0, 450.0, 420.0]),
        curriculum_difficulties=torch.ones(3),
        device=torch.device('cpu'),
    )

    # Mock entropy calculation (will be implemented in Task 4)
    def mock_calculate_entropy(states):
        return torch.tensor([0.3, 0.3, 0.3])  # Low entropy

    curriculum._calculate_action_entropy = mock_calculate_entropy

    # Get decisions and check for advancement
    decisions = curriculum.get_batch_decisions(agent_states, ['agent_0', 'agent_1', 'agent_2'])

    # Should advance to stage 2
    assert curriculum.tracker.agent_stages[0].item() == 2
    assert decisions[0].difficulty_level == 0.25  # Stage 2 -> (2-1)/4 = 0.25
    assert decisions[0].active_meters == ['energy', 'hygiene', 'satiation']
