"""Tests for StaticCurriculum (trivial implementation)."""

import pytest
import torch

from townlet.training.state import BatchedAgentState, CurriculumDecision


def test_static_curriculum_returns_same_decision():
    """StaticCurriculum should always return the same decision."""
    from townlet.curriculum.static import StaticCurriculum

    curriculum = StaticCurriculum(
        difficulty_level=0.5,
        reward_mode='shaped',
    )

    # Create mock state
    state = BatchedAgentState(
        observations=torch.randn(3, 70),
        actions=torch.zeros(3, dtype=torch.long),
        rewards=torch.zeros(3),
        dones=torch.zeros(3, dtype=torch.bool),
        epsilons=torch.ones(3),
        intrinsic_rewards=torch.zeros(3),
        survival_times=torch.randint(0, 1000, (3,)),
        curriculum_difficulties=torch.full((3,), 0.5),
        device=torch.device('cpu'),
    )

    decisions = curriculum.get_batch_decisions(
        state, agent_ids=['agent_0', 'agent_1', 'agent_2']
    )

    assert len(decisions) == 3
    for decision in decisions:
        assert isinstance(decision, CurriculumDecision)
        assert decision.difficulty_level == 0.5
        assert decision.reward_mode == 'shaped'


def test_static_curriculum_checkpoint():
    """StaticCurriculum checkpoint should be serializable."""
    from townlet.curriculum.static import StaticCurriculum

    curriculum = StaticCurriculum(difficulty_level=0.8, reward_mode='sparse')

    state = curriculum.checkpoint_state()

    assert isinstance(state, dict)
    assert state['difficulty_level'] == 0.8
    assert state['reward_mode'] == 'sparse'

    # Should be able to restore
    new_curriculum = StaticCurriculum(difficulty_level=0.0, reward_mode='shaped')
    new_curriculum.load_state(state)

    assert new_curriculum.difficulty_level == 0.8
    assert new_curriculum.reward_mode == 'sparse'
