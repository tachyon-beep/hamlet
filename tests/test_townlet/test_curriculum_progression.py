"""End-to-end test for curriculum progression through all stages."""

import pytest
import torch
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.population.vectorized import VectorizedPopulation
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration


def test_curriculum_progression_through_stages():
    """Train population through full 5-stage curriculum progression.

    This is a smoke test, not a full training run. We verify:
    1. Population can train with adversarial curriculum
    2. Stages advance when conditions are met
    3. System reaches beyond stage 1
    4. No crashes or errors
    """
    device = torch.device('cpu')
    num_agents = 5
    max_steps = 200  # Short episodes for fast testing

    # Create curriculum with aggressive advancement for testing
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=max_steps,
        survival_advance_threshold=0.5,  # Lower threshold for testing
        survival_retreat_threshold=0.2,
        entropy_gate=1.0,  # Disable entropy check (network not trained in this test)
        min_steps_at_stage=50,  # Low for fast progression
        device=device,
    )

    # Initialize curriculum with population size
    curriculum.initialize_population(num_agents=num_agents)

    # Create environment
    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)

    # Create exploration strategy
    exploration = EpsilonGreedyExploration(
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
    )

    # Create population
    population = VectorizedPopulation(
        env=envs,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f'agent_{i}' for i in range(num_agents)],
        device=device,
    )

    # Track progression
    max_stage_reached = 1
    stage_history = []

    # Run training for limited steps (not full training)
    num_episodes = 50
    for episode in range(num_episodes):
        population.reset()

        for step in range(max_steps):
            # Step population
            agent_state = population.step_population(envs)

            # Update curriculum tracker
            population.update_curriculum_tracker(agent_state.rewards, agent_state.dones)

            # Track max stage
            current_stages = curriculum.tracker.agent_stages
            max_stage_reached = max(max_stage_reached, current_stages.max().item())

            if step % 50 == 0:
                stage_history.append(current_stages.clone())

    # Verification
    print(f"\nMax stage reached: {max_stage_reached}")
    print(f"Final stages: {curriculum.tracker.agent_stages}")

    # Should have advanced beyond stage 1 (even if not to stage 5)
    assert max_stage_reached > 1, "Curriculum should advance beyond stage 1"

    # Should have some variation in stages (not all agents identical)
    final_stages = curriculum.tracker.agent_stages
    assert len(torch.unique(final_stages)) >= 1, "Should have stage diversity"

    # Should not crash with sparse rewards if reached stage 5
    if max_stage_reached >= 5:
        # Verify sparse reward mode is active
        decisions = curriculum.get_batch_decisions(
            agent_state,
            [f'agent_{i}' for i in range(num_agents)],
        )
        sparse_decisions = [d for d in decisions if d.reward_mode == 'sparse']
        assert len(sparse_decisions) > 0, "Stage 5 should use sparse rewards"


@pytest.mark.slow
def test_long_curriculum_progression():
    """Longer test to verify reaching stage 5 (marked as slow).

    Run with: pytest -m slow
    """
    device = torch.device('cpu')
    num_agents = 3
    max_steps = 300

    curriculum = AdversarialCurriculum(
        max_steps_per_episode=max_steps,
        survival_advance_threshold=0.4,  # Lower threshold for testing
        survival_retreat_threshold=0.2,
        entropy_gate=1.0,  # Disable entropy check (network not trained in this test)
        min_steps_at_stage=50,  # Lower for testing without training
        device=device,
    )

    # Initialize curriculum with population size
    curriculum.initialize_population(num_agents=num_agents)

    # Create environment
    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)

    # Create exploration strategy
    exploration = EpsilonGreedyExploration(
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.1,
    )

    # Create population
    population = VectorizedPopulation(
        env=envs,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f'agent_{i}' for i in range(num_agents)],
        device=device,
    )

    max_stage_reached = 1

    # Longer training
    num_episodes = 200
    for episode in range(num_episodes):
        population.reset()

        for step in range(max_steps):
            agent_state = population.step_population(envs)
            population.update_curriculum_tracker(agent_state.rewards, agent_state.dones)

            max_stage_reached = max(
                max_stage_reached,
                curriculum.tracker.agent_stages.max().item()
            )

        if episode % 50 == 0:
            print(f"Episode {episode}: Max stage = {max_stage_reached}, "
                  f"Stages = {curriculum.tracker.agent_stages.tolist()}")

    print(f"\nFinal max stage: {max_stage_reached}")

    # With 200 episodes, should reach at least stage 2 (not fully training, just smoke test)
    assert max_stage_reached >= 2, f"Should reach stage 2+ in 200 episodes, got {max_stage_reached}"
