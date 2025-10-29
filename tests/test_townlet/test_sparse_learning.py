"""End-to-end test for sparse reward learning with intrinsic motivation."""

import pytest
import torch
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.population.vectorized import VectorizedPopulation
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration


@pytest.mark.slow
def test_sparse_learning_with_intrinsic():
    """Agent should learn sparse reward task with intrinsic motivation.

    This is a long-running test (10K episodes, ~30 minutes) that validates
    the complete Phase 3 system can enable sparse reward learning.

    Expected: Avg survival > 100 steps (better than random baseline ~50 steps)
    """
    device = torch.device('cpu')
    num_agents = 1
    max_steps = 500

    # Adversarial curriculum (will progress to sparse rewards at stage 5)
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=max_steps,
        survival_advance_threshold=0.7,
        survival_retreat_threshold=0.3,
        entropy_gate=0.5,
        min_steps_at_stage=1000,
        device=device,
    )
    curriculum.initialize_population(num_agents)

    # Adaptive intrinsic exploration (RND + annealing)
    exploration = AdaptiveIntrinsicExploration(
        obs_dim=70,
        embed_dim=128,
        initial_intrinsic_weight=1.0,
        min_intrinsic_weight=0.0,
        variance_threshold=10.0,
        survival_window=100,
        decay_rate=0.99,
        device=device,
    )

    # Environment
    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)

    # Population with replay buffer
    population = VectorizedPopulation(
        env=envs,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f"agent_{i}" for i in range(num_agents)],
        device=device,
        obs_dim=70,
        action_dim=5,
        replay_buffer_capacity=10000,
    )

    # Track metrics
    survival_times = []
    intrinsic_weights = []

    # Train for 10K episodes
    num_episodes = 10000
    for episode in range(num_episodes):
        envs.reset()
        population.reset()

        episode_steps = 0
        for step in range(max_steps):
            agent_state = population.step_population(envs)
            population.update_curriculum_tracker(agent_state.rewards, agent_state.dones)

            episode_steps += 1

            if agent_state.dones.any():
                break

        survival_times.append(episode_steps)
        intrinsic_weights.append(exploration.get_intrinsic_weight())

        # Log progress
        if episode % 1000 == 0:
            avg_survival = sum(survival_times[-100:]) / min(len(survival_times), 100)
            print(f"Episode {episode}: Avg survival = {avg_survival:.1f}, "
                  f"Intrinsic weight = {intrinsic_weights[-1]:.3f}, "
                  f"Stage = {curriculum.tracker.agent_stages[0].item()}")

    # Final metrics
    final_avg_survival = sum(survival_times[-100:]) / 100
    final_intrinsic_weight = intrinsic_weights[-1]
    final_stage = curriculum.tracker.agent_stages[0].item()

    print(f"\nFinal Results:")
    print(f"  Avg survival (last 100): {final_avg_survival:.1f} steps")
    print(f"  Intrinsic weight: {final_intrinsic_weight:.3f}")
    print(f"  Curriculum stage: {final_stage}/5")

    # Assertions
    assert final_avg_survival > 100, \
        f"Agent should survive >100 steps with intrinsic motivation, got {final_avg_survival:.1f}"

    # Intrinsic weight should have decreased
    assert final_intrinsic_weight < 0.5, \
        f"Intrinsic weight should anneal below 0.5, got {final_intrinsic_weight:.3f}"

    # Should reach at least stage 3
    assert final_stage >= 3, \
        f"Agent should reach at least stage 3, got stage {final_stage}"


def test_sparse_learning_baseline_comparison():
    """Compare adaptive intrinsic vs pure epsilon-greedy (shorter test).

    Run for 1K episodes to verify intrinsic motivation provides benefit.
    """
    device = torch.device('cpu')
    num_agents = 1
    max_steps = 300
    num_episodes = 1000

    # Test 1: Pure epsilon-greedy (baseline)
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
    from townlet.curriculum.static import StaticCurriculum

    baseline_exploration = EpsilonGreedyExploration()
    baseline_curriculum = StaticCurriculum(difficulty_level=0.5)

    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)

    baseline_population = VectorizedPopulation(
        env=envs,
        curriculum=baseline_curriculum,
        exploration=baseline_exploration,
        agent_ids=[f"agent_{i}" for i in range(num_agents)],
        device=device,
        obs_dim=70,
        action_dim=5,
        replay_buffer_capacity=10000,
    )

    baseline_survivals = []
    for episode in range(num_episodes):
        envs.reset()
        baseline_population.reset()

        episode_steps = 0
        for step in range(max_steps):
            agent_state = baseline_population.step_population(envs)
            episode_steps += 1
            if agent_state.dones.any():
                break

        baseline_survivals.append(episode_steps)

    baseline_avg = sum(baseline_survivals[-100:]) / 100

    # Test 2: Adaptive intrinsic
    adaptive_exploration = AdaptiveIntrinsicExploration(
        obs_dim=70,
        initial_intrinsic_weight=1.0,
        device=device,
    )
    adaptive_curriculum = StaticCurriculum(difficulty_level=0.5)

    # Re-create environment for second test
    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)

    adaptive_population = VectorizedPopulation(
        env=envs,
        curriculum=adaptive_curriculum,
        exploration=adaptive_exploration,
        agent_ids=[f"agent_{i}" for i in range(num_agents)],
        device=device,
        obs_dim=70,
        action_dim=5,
        replay_buffer_capacity=10000,
    )

    adaptive_survivals = []
    for episode in range(num_episodes):
        envs.reset()
        adaptive_population.reset()

        episode_steps = 0
        for step in range(max_steps):
            agent_state = adaptive_population.step_population(envs)
            episode_steps += 1
            if agent_state.dones.any():
                break

        adaptive_survivals.append(episode_steps)

    adaptive_avg = sum(adaptive_survivals[-100:]) / 100

    print(f"\nBaseline (epsilon-greedy): {baseline_avg:.1f} steps")
    print(f"Adaptive intrinsic: {adaptive_avg:.1f} steps")

    # Adaptive should be better than baseline
    assert adaptive_avg > baseline_avg, \
        f"Adaptive intrinsic ({adaptive_avg:.1f}) should outperform baseline ({baseline_avg:.1f})"
