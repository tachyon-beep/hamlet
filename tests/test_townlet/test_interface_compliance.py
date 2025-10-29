"""
Interface compliance tests.

Ensures all implementations satisfy their interface contracts.
Parameterized to automatically test new implementations as they're added.
"""

import pytest
import torch

from townlet.training.state import BatchedAgentState, CurriculumDecision
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration


# Curriculum Manager Compliance Tests
# (Will be populated as implementations are added in Phase 1+)

@pytest.mark.parametrize("curriculum_class", [
    StaticCurriculum,
    # Add more implementations here:
    # AdversarialCurriculum,
])
def test_curriculum_manager_compliance(curriculum_class):
    """Verify curriculum implementations satisfy interface contract."""
    curriculum = curriculum_class()

    # Should have all required methods
    assert hasattr(curriculum, 'get_batch_decisions')
    assert hasattr(curriculum, 'checkpoint_state')
    assert hasattr(curriculum, 'load_state')

    # get_batch_decisions should return list of CurriculumDecisions
    state = create_mock_batched_state(batch_size=2)
    decisions = curriculum.get_batch_decisions(state, ['agent_0', 'agent_1'])

    assert isinstance(decisions, list)
    assert len(decisions) == 2

    for decision in decisions:
        assert isinstance(decision, CurriculumDecision)

    # checkpoint/restore should work
    checkpoint = curriculum.checkpoint_state()
    assert isinstance(checkpoint, dict)

    curriculum.load_state(checkpoint)  # Should not raise


# Exploration Strategy Compliance Tests

@pytest.mark.parametrize("exploration_class", [
    EpsilonGreedyExploration,
    # Add more implementations here:
    # RNDExploration,
])
def test_exploration_strategy_compliance(exploration_class):
    """Verify exploration implementations satisfy interface contract."""
    exploration = exploration_class()

    # Should have all required methods
    assert hasattr(exploration, 'select_actions')
    assert hasattr(exploration, 'compute_intrinsic_rewards')
    assert hasattr(exploration, 'update')

    # select_actions should return tensor
    q_values = torch.randn(3, 5)
    state = create_mock_batched_state(batch_size=3)
    actions = exploration.select_actions(q_values, state)

    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (3,)

    # compute_intrinsic_rewards should return tensor
    observations = torch.randn(3, 70)
    intrinsic = exploration.compute_intrinsic_rewards(observations)

    assert isinstance(intrinsic, torch.Tensor)
    assert intrinsic.shape == (3,)

    # update should not raise
    batch = {'states': torch.randn(10, 70)}
    exploration.update(batch)  # Should not raise

    # checkpoint/restore should work
    checkpoint = exploration.checkpoint_state()
    assert isinstance(checkpoint, dict)
    exploration.load_state(checkpoint)  # Should not raise


# Population Manager Compliance Tests

@pytest.mark.parametrize("population_class", [
    # Add implementations here:
    # VectorizedPopulation,
])
def test_population_manager_compliance(population_class):
    """Verify population implementations satisfy interface contract."""
    pytest.skip("No population implementations yet (Phase 1)")


# Helper: Create mock BatchedAgentState for testing

def create_mock_batched_state(batch_size: int = 1) -> BatchedAgentState:
    """Create mock BatchedAgentState for interface testing."""
    return BatchedAgentState(
        observations=torch.randn(batch_size, 70),
        actions=torch.zeros(batch_size, dtype=torch.long),
        rewards=torch.zeros(batch_size),
        dones=torch.zeros(batch_size, dtype=torch.bool),
        epsilons=torch.full((batch_size,), 0.5),
        intrinsic_rewards=torch.zeros(batch_size),
        survival_times=torch.randint(0, 1000, (batch_size,)),
        curriculum_difficulties=torch.full((batch_size,), 0.5),
        device=torch.device('cpu'),
    )
