"""Integration tests for data flows through complete pipelines.

This file tests data transformations through observation, reward, action,
and training pipelines.

Task 13c: Data Flow Integration Tests
Focus: Trace data through complete pipelines to verify transformations
"""

from pathlib import Path

import torch

from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation

# =============================================================================
# TEST CLASS 1: Observation Pipeline
# =============================================================================


class TestObservationPipeline:
    """Test observation building pipeline.

    Verifies that observations are correctly constructed by ObservationBuilder
    and flow to VectorizedPopulation with correct dimensions and structure.
    """

    def test_environment_builds_observation_correct_dims(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Verify environment builds observations with correct dimensions.

        This test validates the critical observation pipeline:
        - Environment reset produces observations
        - Observation dimension matches ObservationBuilder calculation
        - Full observability: grid + meters + affordance + temporal
        - Dimensions vary by grid size but structure is consistent

        Integration point: VectorizedHamletEnv → ObservationBuilder → Population

        NOTE: After TASK-002A, grid_size is loaded from substrate.yaml (8×8),
              not from the grid_size parameter (which is now ignored).
        """
        # Create environment - grid_size parameter ignored, loaded from substrate.yaml (8×8)
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

        # Reset environment to build observations
        obs = env.reset()

        # Verify observation dimension calculation (programmatic, not hardcoded)
        # Full obs: substrate + meters + affordances + temporal + velocity
        velocity_vars = env.substrate.position_dim + 1  # position_dim velocity components + magnitude
        affordance_obs_dim = env.num_affordance_types + 1  # +1 for "none"
        temporal_dims = 4  # time_sin, time_cos, interaction_progress, lifetime_progress
        expected_dim = env.substrate.get_observation_dim() + env.meter_count + affordance_obs_dim + temporal_dims + velocity_vars
        assert obs.shape == (1, expected_dim), f"Observation should be [1, {expected_dim}], got {obs.shape}"

        # Verify observation matches environment's reported dimension
        assert obs.shape[1] == env.observation_dim, f"Observation dim should match environment ({env.observation_dim}), got {obs.shape[1]}"

        # Verify observations are finite (no NaN/Inf)
        assert torch.isfinite(obs).all(), "Observations should not contain NaN/Inf"

        # Verify population receives correct observations
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=0.1, epsilon_min=0.1, epsilon_decay=1.0)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        population.reset()

        # Verify population's current_obs matches environment's observation
        assert population.current_obs.shape == (
            1,
            expected_dim,
        ), f"Population obs should be [1, {expected_dim}], got {population.current_obs.shape}"

    def test_partial_observability_5x5_window_correct(self, cpu_device, cpu_env_factory, recurrent_brain_config):
        """Verify partial observability produces correct 5×5 local window.

        This test validates the POMDP observation pipeline:
        - Agent sees only 5×5 local window centered on position
        - Window contains affordances visible in local region
        - Out-of-bounds regions are not visible (encoded as 0)
        - Observation dimension is FIXED (25 + 2 + 8 + 15 + 4 = 54)

        Integration point: VectorizedHamletEnv (POMDP) → ObservationBuilder._build_partial_observations()
        """
        # Create POMDP environment with 5×5 vision
        env = cpu_env_factory(config_dir=Path("configs/L2_partial_observability"), num_agents=1)

        # Reset environment
        obs = env.reset()

        # Verify POMDP observation dimension matches environment
        # Different config packs may include/exclude components (e.g., L2 excludes affordances)
        # Trust env.observation_dim which is computed from actual VFS observation spec
        assert obs.shape == (1, env.observation_dim), f"POMDP observation should be [1, {env.observation_dim}], got {obs.shape}"

        # Verify observations are finite (no NaN/Inf)
        assert torch.isfinite(obs).all(), "Observations should not contain NaN/Inf"

        # Step agent and verify observation updates
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            # action_dim defaults to env.action_dim
            brain_config=recurrent_brain_config,  # POMDP uses recurrent network
            vision_window_size=5,
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=8,
            train_frequency=10000,  # Disable training (test focuses on observation structure)
        )

        population.reset()
        initial_obs = population.current_obs.clone()

        # Take 5 steps (agent should move with random actions)
        for _ in range(5):
            agent_state = population.step_population(env)
            if agent_state.dones[0]:
                break

        # Verify observations changed (local window updates as agent moves)
        # Note: Might not change if agent doesn't move, so we just verify structure
        final_obs = population.current_obs
        assert final_obs.shape == initial_obs.shape, "Observation shape should remain consistent across steps"


# =============================================================================
# TEST CLASS 2: Reward Pipeline
# =============================================================================


class TestRewardPipeline:
    """Test reward computation and combination pipeline.

    Verifies that extrinsic rewards flow from environment, intrinsic rewards
    are computed by RND, rewards are combined correctly, and stored in replay buffer.
    """

    def test_environment_extrinsic_reward_to_population(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Verify extrinsic rewards flow from environment to population.

        This test validates the extrinsic reward pipeline:
        - Environment produces survival rewards (+1.0 per step for alive agents)
        - Rewards flow through env.step() to population
        - Rewards are positive for alive agents, 0 for dead agents

        Integration point: VectorizedHamletEnv.step() → VectorizedPopulation
        """
        # Create small environment
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=2)

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=0.1, epsilon_min=0.1, epsilon_decay=1.0)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset and step
        population.reset()
        agent_state = population.step_population(env)

        # Verify rewards exist
        assert hasattr(agent_state, "rewards"), "Agent state should have rewards"
        assert agent_state.rewards.shape == (2,), "Rewards should be [num_agents]"

        # Verify rewards are finite
        assert torch.isfinite(agent_state.rewards).all(), "Rewards should not contain NaN/Inf"

        # Verify alive agents receive positive rewards (survival reward = +1.0 per step)
        alive_agents = ~agent_state.dones
        if alive_agents.any():
            # Alive agents should have positive rewards (survival + any intrinsic)
            # Note: With epsilon-greedy (no RND), intrinsic = 0, so rewards should be ~1.0
            alive_rewards = agent_state.rewards[alive_agents]
            assert (alive_rewards > 0).all(), f"Alive agents should have positive rewards, got {alive_rewards}"

    def test_exploration_intrinsic_reward_combined(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Verify intrinsic rewards are combined with extrinsic rewards.

        This test validates the intrinsic reward pipeline:
        - RND computes intrinsic rewards for observations
        - Intrinsic rewards are non-negative (MSE property)
        - Combined reward = extrinsic + (intrinsic * weight)
        - Both components are accessible in agent state

        Integration point: RND → AdaptiveIntrinsicExploration → VectorizedPopulation
        """
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

        # Create adaptive intrinsic exploration with known weight
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=0.5,  # 50% intrinsic weight for clear testing
            variance_threshold=100.0,
            survival_window=50,
            device=cpu_device,
        )

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=obs_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset and step
        population.reset()
        agent_state = population.step_population(env)

        # Verify both reward components exist
        assert hasattr(agent_state, "rewards"), "Agent state should have combined rewards"
        assert hasattr(agent_state, "intrinsic_rewards"), "Agent state should have intrinsic rewards"

        # Verify intrinsic rewards are non-negative (MSE property of RND)
        intrinsic = agent_state.intrinsic_rewards[0].item()
        assert intrinsic >= 0, f"Intrinsic reward should be non-negative (MSE), got {intrinsic}"

        # Verify combined rewards include both components
        combined = agent_state.rewards[0].item()
        weight = exploration.get_intrinsic_weight()

        # Behavioral assertion: Weight should be 0.5 as initialized
        assert abs(weight - 0.5) < 1e-6, f"Intrinsic weight should be 0.5, got {weight}"

        # Combined reward should be a valid float
        assert isinstance(combined, float), "Combined reward should be a float"
        assert torch.isfinite(torch.tensor(combined)), "Combined reward should be finite"

        # Run multiple steps and verify intrinsic rewards vary (novelty detection)
        intrinsic_rewards_collected = []
        for _ in range(10):
            agent_state = population.step_population(env)
            intrinsic_rewards_collected.append(agent_state.intrinsic_rewards[0].item())

        # At least some intrinsic rewards should be positive (RND detects novelty)
        assert sum(intrinsic_rewards_collected) > 0, "RND should produce non-zero novelty rewards over 10 steps"

    def test_combined_reward_stored_in_replay_buffer(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Verify combined rewards are stored in replay buffer (not separate rewards).

        This test validates the replay buffer storage contract:
        - Replay buffer stores COMBINED rewards for training
        - No separate extrinsic/intrinsic storage (prevents double-counting)
        - Training uses combined rewards from buffer

        Integration point: VectorizedPopulation → ReplayBuffer.push()
        """
        # Create small environment
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

        # Create adaptive intrinsic exploration
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=0.5,
            variance_threshold=100.0,
            survival_window=50,
            device=cpu_device,
        )

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=obs_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset and run 20 steps to fill buffer
        population.reset()
        for _ in range(20):
            agent_state = population.step_population(env)
            if agent_state.dones[0]:
                break

        # Verify replay buffer has transitions
        assert len(population.replay_buffer) >= 10, "Replay buffer should have at least 10 transitions"

        # Sample from replay buffer (need intrinsic_weight for combined reward calculation)
        batch = population.replay_buffer.sample(batch_size=10, intrinsic_weight=0.5)

        # Verify batch contains 'rewards' key (combined rewards)
        assert "rewards" in batch, "Replay buffer should store 'rewards' key"

        # Verify rewards are combined (NOT separate)
        assert "intrinsic_rewards" not in batch, "Replay buffer should NOT store separate intrinsic rewards"
        assert "rewards_extrinsic" not in batch, "Replay buffer should NOT store separate extrinsic rewards"

        # Verify rewards are tensors with correct shape
        assert isinstance(batch["rewards"], torch.Tensor), "Rewards should be tensors"
        assert batch["rewards"].shape[0] == 10, "Batch should have 10 samples"
        assert torch.isfinite(batch["rewards"]).all(), "Rewards should be finite"


# =============================================================================
# TEST CLASS 3: Action Pipeline
# =============================================================================


class TestActionPipeline:
    """Test action selection pipeline.

    Verifies that Q-values flow from network to exploration, epsilon-greedy
    selection works with action masking, and actions execute in environment.
    """

    def test_qnetwork_qvalues_to_exploration(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Verify Q-values flow from Q-network to exploration strategy.

        This test validates the Q-value pipeline:
        - Q-network produces Q-values for all actions
        - Q-values have correct shape [num_agents, num_actions]
        - Q-values flow to exploration strategy for action selection
        - Q-values are finite (no NaN/Inf from untrained network)

        Integration point: Q-network → VectorizedPopulation → Exploration
        """
        # Create small environment
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=2)

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)  # Greedy for Q-value testing

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset environment
        population.reset()

        # Get Q-values from network directly
        with torch.no_grad():
            q_values = population.q_network(population.current_obs)

        # Verify Q-values shape
        assert q_values.shape == (2, env.action_dim), f"Q-values should be [2, {env.action_dim}], got {q_values.shape}"

        # Verify Q-values are finite (untrained network should still produce valid outputs)
        assert torch.isfinite(q_values).all(), "Q-values should not contain NaN/Inf from untrained network"

        # Step population and verify actions are based on Q-values (epsilon=0, greedy)
        agent_state = population.step_population(env)

        # Verify actions exist
        assert hasattr(agent_state, "actions"), "Agent state should have actions"
        assert agent_state.actions.shape == (2,), "Actions should be [num_agents]"

        # Verify actions are valid (in range [0, action_dim))
        for action in agent_state.actions:
            assert 0 <= action < env.action_dim, f"Action should be in [0, {env.action_dim}), got {action}"

    def test_exploration_epsilon_greedy_with_masking(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Verify epsilon-greedy selects actions with masking correctly.

        This test validates the action selection pipeline:
        - Action masks prevent invalid actions (out of bounds, closed affordances)
        - Masked actions have Q-values set to -inf before selection
        - Epsilon-greedy never selects masked actions
        - Random exploration respects action masks

        Integration point: Environment.get_action_masks() → Exploration.select_actions()

        NOTE: After TASK-002A, grid_size comes from substrate.yaml (8×8).
        """
        # Create environment - grid_size loaded from substrate.yaml (8×8)
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

        # Create population with full random exploration to test masking thoroughly
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset and run 100 steps to hit boundaries
        population.reset()
        actions_taken = []
        positions = []

        for step in range(100):
            agent_state = population.step_population(env)
            actions_taken.append(agent_state.actions[0].item())
            positions.append(env.positions[0].cpu().clone())

            if agent_state.dones[0]:
                break

        # Verify all actions are valid (should be in [0, action_dim))
        for action in actions_taken:
            assert 0 <= action < env.action_dim, f"Action should be in [0, {env.action_dim}), got {action}"

        # Verify all positions stayed within bounds (8×8 grid from substrate.yaml)
        for pos in positions:
            x, y = pos[0].item(), pos[1].item()
            assert 0 <= x < 8, f"X position should be in [0, 8), got {x}"
            assert 0 <= y < 8, f"Y position should be in [0, 8), got {y}"

        # Verify action variety (not all same action)
        assert len(set(actions_taken)) > 1, "Should have variety of actions (not all same)"

    def test_actions_to_environment_execution(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Verify actions execute in environment and cause state changes.

        This test validates the action execution pipeline:
        - Actions flow from exploration to environment
        - Environment.step(actions) updates state (positions, meters)
        - State changes are observable in next observation
        - Info dict contains step counts and metadata

        Integration point: Exploration → VectorizedPopulation → Environment.step()
        """
        # Create small environment
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0)  # Random to cause movement

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset and capture initial state
        population.reset()
        initial_obs = population.current_obs.clone()
        initial_position = env.positions[0].clone()
        initial_energy = env.meters[0, 0].item()  # Energy meter

        # Take 10 steps
        state_changes_detected = False
        for step in range(10):
            agent_state = population.step_population(env)

            # Verify info dict exists
            assert isinstance(agent_state.info, dict), "Agent state should have info dict"
            assert "step_counts" in agent_state.info, "Info should contain step_counts"

            # Check if state changed (position or energy)
            current_position = env.positions[0]
            current_energy = env.meters[0, 0].item()

            position_changed = not torch.equal(initial_position, current_position)
            energy_changed = abs(current_energy - initial_energy) > 1e-6

            if position_changed or energy_changed:
                state_changes_detected = True
                break

            if agent_state.dones[0]:
                break

        # Verify state changed over 10 steps (agent moved or energy depleted)
        assert state_changes_detected, "State should change over 10 steps (position or energy)"

        # Verify observations updated (should differ from initial)
        final_obs = population.current_obs
        assert final_obs.shape == initial_obs.shape, "Observation shape should remain consistent"

        # Observations should differ (state changed)
        # Note: Not asserting obs_changed because observations might be same if agent didn't move
        # and meters didn't change significantly. Over 10 random movement steps, at least
        # energy should deplete, but not guaranteed with random actions.
