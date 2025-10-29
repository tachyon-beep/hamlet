"""Tests for VectorizedPopulation (population coordinator)."""

import pytest
import torch

from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration


def test_vectorized_population_construction():
    """VectorizedPopulation should construct with components."""
    from townlet.population.vectorized import VectorizedPopulation
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=0.995)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0', 'agent_1'],
        device=torch.device('cpu'),
    )

    assert population.num_agents == 2
    assert len(population.agent_ids) == 2


def test_vectorized_population_step():
    """VectorizedPopulation should coordinate training step."""
    from townlet.population.vectorized import VectorizedPopulation
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.1, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0'],
        device=torch.device('cpu'),
    )

    # Reset
    population.reset()

    # Step (will use dummy Q-network for now)
    state = population.step_population(env)

    from townlet.training.state import BatchedAgentState
    assert isinstance(state, BatchedAgentState)
    assert state.batch_size == 1
