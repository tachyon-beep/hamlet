"""
Integration tests for full training loop.

Tests end-to-end integration of all Phase 1 components:
- VectorizedHamletEnv
- StaticCurriculum
- EpsilonGreedyExploration
- VectorizedPopulation
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.mark.integration
def test_train_one_episode_n1():
    """Should train single agent for one episode."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.3, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0'],
        device=torch.device('cpu'),
    )

    # Run episode
    population.reset()

    episode_rewards = []
    done = False
    step_count = 0
    max_steps = 500

    while not done and step_count < max_steps:
        state = population.step_population(env)
        episode_rewards.append(state.rewards[0].item())
        done = state.dones[0].item()
        step_count += 1

    # Assertions
    assert len(episode_rewards) > 0
    assert len(episode_rewards) <= max_steps

    # Should accumulate some reward
    total_reward = sum(episode_rewards)
    print(f"Episode completed in {len(episode_rewards)} steps, total reward: {total_reward:.2f}")


@pytest.mark.integration
def test_train_multiple_agents():
    """Should train multiple agents in parallel."""
    num_agents = 5

    env = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.2, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f'agent_{i}' for i in range(num_agents)],
        device=torch.device('cpu'),
    )

    # Run episode
    population.reset()

    all_done = False
    step_count = 0
    max_steps = 500

    while not all_done and step_count < max_steps:
        state = population.step_population(env)
        all_done = torch.all(state.dones)
        step_count += 1

    print(f"All {num_agents} agents completed in {step_count} steps")

    # All agents should have stepped
    assert step_count > 0
    assert step_count <= max_steps


@pytest.mark.integration
def test_checkpoint_save_restore():
    """Should save and restore population checkpoint."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.8, epsilon_decay=0.995)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0'],
        device=torch.device('cpu'),
    )

    # Get checkpoint
    checkpoint = population.get_checkpoint()

    # Verify structure
    assert checkpoint.num_agents == 1
    assert len(checkpoint.agent_ids) == 1
    assert 'global' in checkpoint.curriculum_states
    assert 'global' in checkpoint.exploration_states

    # Verify exploration state
    exploration_state = checkpoint.exploration_states['global']
    assert exploration_state['epsilon'] == 0.8

    # Should be serializable
    json_str = checkpoint.model_dump_json()
    assert 'num_agents' in json_str

    # Should be deserializable
    from townlet.training.state import PopulationCheckpoint
    restored = PopulationCheckpoint.model_validate_json(json_str)
    assert restored.num_agents == checkpoint.num_agents


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_on_gpu():
    """Should train on GPU if available."""
    device = torch.device('cuda')

    env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=device)
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0', 'agent_1'],
        device=device,
    )

    # Run a few steps
    population.reset()

    for _ in range(10):
        state = population.step_population(env)

        # Verify tensors are on GPU
        assert state.observations.device.type == 'cuda'
        assert state.rewards.device.type == 'cuda'


@pytest.mark.integration
def test_integration_with_adversarial_curriculum():
    """VectorizedPopulation should work with AdversarialCurriculum."""
    from townlet.curriculum.adversarial import AdversarialCurriculum

    device = torch.device('cpu')
    num_agents = 3

    # Create curriculum
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        survival_advance_threshold=0.7,
        min_steps_at_stage=10,  # Low for fast testing
        device=device,
    )

    # Create environment
    env = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)

    # Initialize curriculum with population size
    curriculum.initialize_population(num_agents=num_agents)

    # Create exploration strategy
    exploration = EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=1.0)

    # Create population with curriculum
    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f'agent_{i}' for i in range(num_agents)],
        device=device,
    )

    # Reset and run steps
    population.reset()

    # Run 50 steps
    for step in range(50):
        agent_state = population.step_population(env)

        # Verify state is returned
        assert agent_state is not None
        assert agent_state.observations.shape[0] == num_agents

        # Update curriculum tracker (REQUIRED for adversarial curriculum)
        population.update_curriculum_tracker(agent_state.rewards, agent_state.dones)

    # Verify curriculum decisions are being made
    assert hasattr(population, 'current_curriculum_decisions'), \
        "VectorizedPopulation must store current_curriculum_decisions"
    assert len(population.current_curriculum_decisions) == num_agents, \
        f"Expected {num_agents} curriculum decisions, got {len(population.current_curriculum_decisions)}"

    # Verify decisions have correct structure
    for decision in population.current_curriculum_decisions:
        assert hasattr(decision, 'difficulty_level')
        assert hasattr(decision, 'active_meters')
        assert hasattr(decision, 'reward_mode')
        assert decision.difficulty_level >= 0.0

    # Verify curriculum tracker is being updated
    assert curriculum.tracker is not None
    assert curriculum.tracker.episode_steps.sum() > 0, "Curriculum tracker should have recorded steps"
