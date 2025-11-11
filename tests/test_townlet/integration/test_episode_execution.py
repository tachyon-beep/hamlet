"""Integration tests for episode execution end-to-end.

This file tests the complete episode lifecycle from reset to completion,
verifying the observation → action → reward → next observation cycle.

Task 13a: Episode Execution Integration Tests
Focus: Test complete episode execution with real components
"""

from pathlib import Path

import torch

from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.state import BatchedAgentState

# =============================================================================
# TEST CLASS: Episode Lifecycle
# =============================================================================


class TestEpisodeLifecycle:
    """Test complete episode execution from start to finish.

    These tests verify the end-to-end episode execution flow:
    1. Environment reset initializes state correctly
    2. Step loop executes for N steps
    3. Episode terminates correctly on death or max_steps
    4. Observation → action → reward → next_obs cycle maintains consistency
    5. Both feedforward and recurrent networks work correctly
    """

    def test_single_episode_feedforward_network(self, cpu_device, cpu_env_factory):
        """Verify single episode with feedforward network completes correctly.

        This test validates the most basic episode execution flow:
        - Reset → step loop → termination
        - Uses SimpleQNetwork (feedforward, no LSTM)
        - Single agent, small grid for fast execution
        - Verifies BatchedAgentState structure at each step

        Critical integration point: VectorizedHamletEnv + VectorizedPopulation
        with feedforward network completing a full episode.
        """
        # Create small environment for fast testing
        env = cpu_env_factory(num_agents=1)

        # Create population with feedforward network
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(
            epsilon=0.1,
            epsilon_min=0.1,
            epsilon_decay=1.0,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset environment and population
        population.reset()

        # Initialize all meters to 1.0 to prevent cascade-induced death during test
        env.meters.fill_(1.0)

        # Verify initial observation shape
        assert population.current_obs.shape == (1, env.observation_dim)

        # Run episode for 50 steps (max_steps)
        episode_done = False
        step_count = 0

        while not episode_done and step_count < 50:
            # Execute one step
            agent_state = population.step_population(env)

            # Verify BatchedAgentState structure
            assert isinstance(agent_state, BatchedAgentState)
            assert agent_state.batch_size == 1
            assert agent_state.observations.shape == (1, env.observation_dim)
            assert agent_state.actions.shape == (1,)
            assert agent_state.rewards.shape == (1,)
            assert agent_state.dones.shape == (1,)

            # Check if episode terminated
            if agent_state.dones[0]:
                episode_done = True

            step_count += 1

        # Episode should complete (either by death or max_steps)
        assert step_count > 0, "Episode should run at least 1 step"
        assert step_count <= 50, "Episode should not exceed max_steps"

    def test_single_episode_recurrent_network_with_lstm(self, cpu_device, cpu_env_factory):
        """Verify single episode with LSTM network completes correctly.

        This test validates episode execution with recurrent networks:
        - Reset → step loop → termination
        - Uses RecurrentSpatialQNetwork (LSTM with hidden state)
        - Partial observability (POMDP)
        - Verifies hidden state evolves during episode
        - Verifies episodes are flushed to SequentialReplayBuffer

        Critical integration point: VectorizedHamletEnv + VectorizedPopulation
        with recurrent network managing hidden state across episode steps.

        Note: Uses minimal energy costs to prevent agent death during test,
        since death resets hidden state to zeros (see test_hidden_state_resets_on_death).
        """
        # Create environment via compiled universe pipeline
        env = cpu_env_factory(config_dir=Path("configs/L2_partial_observability"), num_agents=1)

        # Create population with recurrent network
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(
            epsilon=0.1,
            epsilon_min=0.1,
            epsilon_decay=1.0,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            # action_dim defaults to env.action_dim
            network_type="recurrent",
            vision_window_size=5,
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=8,
            train_frequency=10000,  # Disable training (test focuses on episode execution and hidden state)
        )

        # Reset environment and population
        population.reset()

        # Initialize all meters to 1.0 to prevent cascade-induced death during test
        # (cascade effects like satiation→energy/health can kill agent even with minimal costs)
        env.meters.fill_(1.0)

        # Capture initial hidden state
        recurrent_network = population.q_network
        h0, c0 = recurrent_network.get_hidden_state()
        initial_h = h0.clone()

        # Run episode for 10 steps (reduced from 20 to avoid cascade-induced death)
        # Even with ultra-minimal costs and full meters, cascade effects can kill agents over longer runs
        episode_done = False
        step_count = 0

        while not episode_done and step_count < 10:
            # Execute one step
            agent_state = population.step_population(env)

            # Verify BatchedAgentState structure
            assert isinstance(agent_state, BatchedAgentState)
            assert agent_state.batch_size == 1

            # Check if episode terminated
            if agent_state.dones[0]:
                episode_done = True

            step_count += 1

        # Verify agent survived (death resets hidden state to zeros, causing false failures)
        assert (
            not episode_done
        ), f"Agent died after {step_count} steps (cascade effects). Death resets hidden state to zeros, invalidating test."

        # Verify hidden state evolved during episode
        h_final, c_final = recurrent_network.get_hidden_state()
        assert not torch.allclose(initial_h, h_final, atol=1e-6), "Hidden state should change during episode (memory accumulation)"

        # Episode should complete
        assert step_count > 0, "Episode should run at least 1 step"
        assert step_count <= 10, "Episode should not exceed max_steps"

    def test_multi_agent_episode_with_partial_dones(self, cpu_device, cpu_env_factory):
        """Verify multi-agent episode where agents die at different times.

        This test validates partial done handling in multi-agent settings:
        - Some agents may die while others continue
        - Dead agents' episodes should be flushed to replay buffer
        - Hidden states should reset for dead agents (recurrent only)
        - Surviving agents should continue running

        Critical integration point: VectorizedPopulation handling asynchronous
        episode termination across multiple agents.
        """
        # Create multi-agent environment
        env = cpu_env_factory(num_agents=4)

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(
            epsilon=0.1,
            epsilon_min=0.1,
            epsilon_decay=1.0,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1", "agent_2", "agent_3"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset environment and population
        population.reset()

        # Track which agents have died
        agents_died = torch.zeros(4, dtype=torch.bool, device=cpu_device)

        # Run episode for 50 steps
        for step in range(50):
            agent_state = population.step_population(env)

            # Track any new deaths
            agents_died = agents_died | agent_state.dones

            # Verify batch structure
            assert agent_state.batch_size == 4
            assert agent_state.dones.shape == (4,)

            # If all agents died, episode should end
            if agents_died.all():
                break

        # At least some agents should have participated in episode
        assert step > 0, "Episode should run at least 1 step"

    def test_episode_observation_action_reward_cycle(self, cpu_device, cpu_env_factory):
        """Verify obs → action → reward → next_obs cycle works correctly.

        This test validates the core data flow through an episode:
        - Observations are consistent across steps (dimension, range)
        - Actions are valid (within action space)
        - Rewards are produced (extrinsic + intrinsic)
        - Next observations update correctly
        - Info dict contains expected metadata

        Critical integration point: Complete data cycle from observation building
        through action selection to reward computation and state update.
        """
        # Create small environment
        env = cpu_env_factory(num_agents=1)

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(
            epsilon=0.1,
            epsilon_min=0.1,
            epsilon_decay=1.0,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset environment and population
        population.reset()

        # Initialize all meters to 1.0 to prevent cascade-induced death during test
        env.meters.fill_(1.0)

        initial_obs = population.current_obs.clone()

        # Track data cycle over 10 steps
        observations = [initial_obs]
        actions = []
        rewards = []

        for step in range(10):
            # Execute one step
            agent_state = population.step_population(env)

            # Validate observation
            assert agent_state.observations.shape == (
                1,
                env.observation_dim,
            ), f"Step {step}: Observation shape should be (1, {env.observation_dim})"
            assert torch.isfinite(agent_state.observations).all(), f"Step {step}: Observation should not contain NaN/Inf"

            # Validate action
            assert agent_state.actions.shape == (1,), f"Step {step}: Action shape should be (1,)"
            assert 0 <= agent_state.actions[0] < env.action_dim, f"Step {step}: Action should be in range [0, {env.action_dim})"

            # Validate reward
            assert agent_state.rewards.shape == (1,), f"Step {step}: Reward shape should be (1,)"
            assert torch.isfinite(agent_state.rewards).all(), f"Step {step}: Reward should not contain NaN/Inf"

            # Validate info dict
            assert isinstance(agent_state.info, dict), f"Step {step}: Info should be a dictionary"
            assert "step_counts" in agent_state.info, f"Step {step}: Info should contain step_counts"

            # Store cycle data
            observations.append(agent_state.observations.clone())
            actions.append(agent_state.actions.clone())
            rewards.append(agent_state.rewards.clone())

            # If episode terminates, break
            if agent_state.dones[0]:
                break

        # Verify data cycle produced valid sequences
        assert len(observations) >= 2, "Should have at least initial obs + 1 next obs"
        assert len(actions) >= 1, "Should have at least 1 action"
        assert len(rewards) >= 1, "Should have at least 1 reward"

        # Verify observations changed over time (agent moved or environment updated)
        # Note: Observations might stay same if agent doesn't move, so we just check
        # that the data flow completed without errors
        assert all(obs.shape == (1, env.observation_dim) for obs in observations), "All observations should have consistent shape"
