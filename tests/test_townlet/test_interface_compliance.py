"""
Interface compliance tests.

Ensures all implementations satisfy their interface contracts.
Parameterized to automatically test new implementations as they're added.
"""

import pytest
import torch

from townlet.training.state import BatchedAgentState, CurriculumDecision
from townlet.curriculum.static import StaticCurriculum


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
    # Add implementations here:
    # EpsilonGreedyExploration,
    # RNDExploration,
])
def test_exploration_strategy_compliance(exploration_class):
    """Verify exploration implementations satisfy interface contract."""
    pytest.skip("No exploration implementations yet (Phase 1)")


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
