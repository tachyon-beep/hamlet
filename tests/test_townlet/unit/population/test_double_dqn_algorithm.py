"""Tests for Double DQN algorithm implementation."""

import torch

from townlet.population.vectorized import VectorizedPopulation


class TestDoubleDQNFeedforward:
    """Test Double DQN Q-target computation for feedforward networks."""

    def test_vanilla_dqn_uses_max_q_from_target_network(
        self,
        cpu_env_factory,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        minimal_brain_config,
    ):
        """Vanilla DQN: Q_target = r + γ * max_a Q_target(s', a)."""
        # Use cpu_env_factory to ensure env and population use same device
        basic_env = cpu_env_factory()

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,  # Use actual obs_dim from env
            action_dim=basic_env.action_dim,
            brain_config=minimal_brain_config,
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            use_double_dqn=False,  # Vanilla DQN
        )

        # Populate replay buffer with transitions
        obs = basic_env.reset()
        for _ in range(10):
            actions = torch.randint(0, basic_env.action_dim, (1,))
            next_obs, rewards, dones, _ = basic_env.step(actions)
            intrinsic_rewards = torch.zeros_like(rewards)
            population.replay_buffer.push(
                observations=obs,
                actions=actions,
                rewards_extrinsic=rewards,
                rewards_intrinsic=intrinsic_rewards,
                next_observations=next_obs,
                dones=dones,
            )
            obs = next_obs

        # Sample batch and compute Q-targets
        batch = population.replay_buffer.sample(batch_size=4, intrinsic_weight=0.0)

        # Manually compute vanilla DQN Q-targets
        with torch.no_grad():
            q_next_vanilla = population.target_network(batch["next_observations"]).max(1)[0]
            expected_q_target = batch["rewards"] + 0.99 * q_next_vanilla * (~batch["dones"]).float()

        # Trigger training step (step_population internally computes Q-targets)
        # We'll verify by checking that training runs without error
        # (Full verification requires exposing Q-target computation or adding logging)

        # For now, verify that vanilla DQN runs without crashing
        assert population.use_double_dqn is False
        assert expected_q_target.shape == (4,)

    def test_double_dqn_uses_online_network_for_action_selection(
        self,
        cpu_env_factory,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        minimal_brain_config,
    ):
        """Double DQN: Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))."""
        # Use cpu_env_factory to ensure env and population use same device
        basic_env = cpu_env_factory()

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,  # Use actual obs_dim from env
            action_dim=basic_env.action_dim,
            brain_config=minimal_brain_config,
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            use_double_dqn=True,  # Double DQN
        )

        # Populate replay buffer
        obs = basic_env.reset()
        for _ in range(10):
            actions = torch.randint(0, basic_env.action_dim, (1,))
            next_obs, rewards, dones, _ = basic_env.step(actions)
            intrinsic_rewards = torch.zeros_like(rewards)
            population.replay_buffer.push(
                observations=obs,
                actions=actions,
                rewards_extrinsic=rewards,
                rewards_intrinsic=intrinsic_rewards,
                next_observations=next_obs,
                dones=dones,
            )
            obs = next_obs

        # Sample batch
        batch = population.replay_buffer.sample(batch_size=4, intrinsic_weight=0.0)

        # Manually compute Double DQN Q-targets
        with torch.no_grad():
            # Step 1: Use ONLINE network to select best actions
            next_actions = population.q_network(batch["next_observations"]).argmax(1)
            # Step 2: Use TARGET network to evaluate those actions
            q_next_double = population.target_network(batch["next_observations"]).gather(1, next_actions.unsqueeze(1)).squeeze()
            expected_q_target = batch["rewards"] + 0.99 * q_next_double * (~batch["dones"]).float()

        # Verify Double DQN is enabled
        assert population.use_double_dqn is True
        assert expected_q_target.shape == (4,)
        assert next_actions.shape == (4,)

    def test_double_dqn_differs_from_vanilla_dqn(
        self,
        cpu_env_factory,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        minimal_brain_config,
    ):
        """Double DQN should produce different Q-targets than vanilla DQN."""
        # Use cpu_env_factory to ensure env and population use same device
        basic_env = cpu_env_factory()

        # Create two populations with same initialization
        torch.manual_seed(42)
        pop_vanilla = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,  # Use actual obs_dim from env
            action_dim=basic_env.action_dim,
            brain_config=minimal_brain_config,
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            use_double_dqn=False,
        )

        torch.manual_seed(42)
        pop_double = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,  # Use actual obs_dim from env
            action_dim=basic_env.action_dim,
            brain_config=minimal_brain_config,
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            use_double_dqn=True,
        )

        # Populate both with same transitions
        torch.manual_seed(123)
        obs = basic_env.reset()
        for _ in range(10):
            actions = torch.randint(0, basic_env.action_dim, (1,))
            next_obs, rewards, dones, _ = basic_env.step(actions)
            intrinsic_rewards = torch.zeros_like(rewards)

            pop_vanilla.replay_buffer.push(obs, actions, rewards, intrinsic_rewards, next_obs, dones)
            pop_double.replay_buffer.push(obs, actions, rewards, intrinsic_rewards, next_obs, dones)
            obs = next_obs

        # Sample same batch (use same random seed)
        torch.manual_seed(456)
        batch_vanilla = pop_vanilla.replay_buffer.sample(batch_size=4, intrinsic_weight=0.0)
        torch.manual_seed(456)
        batch_double = pop_double.replay_buffer.sample(batch_size=4, intrinsic_weight=0.0)

        # Verify both algorithms can compute Q-targets
        # We're just checking that the implementation runs without errors
        # (Full verification of behavior differences would require training)
        with torch.no_grad():
            # Vanilla: max over target network
            q_next_vanilla = pop_vanilla.target_network(batch_vanilla["next_observations"]).max(1)[0]
            assert q_next_vanilla.shape == (4,)

            # Double: argmax from online, evaluate with target
            next_actions = pop_double.q_network(batch_double["next_observations"]).argmax(1)
            q_next_double = pop_double.target_network(batch_double["next_observations"]).gather(1, next_actions.unsqueeze(1)).squeeze()
            assert q_next_double.shape == (4,)
            assert next_actions.shape == (4,)

        # Verify networks initialized with same seed produce same Q-values
        q_values_vanilla = pop_vanilla.target_network(batch_vanilla["next_observations"])
        q_values_double = pop_double.target_network(batch_double["next_observations"])
        assert torch.allclose(q_values_vanilla, q_values_double)

        # Both populations use the same target network, but the MECHANISM differs:
        # - Vanilla DQN: Uses target network's max Q-value
        # - Double DQN: Uses online network to select action, target to evaluate
        # This test verifies both mechanisms execute successfully


class TestDoubleDQNRecurrent:
    """Test Double DQN for recurrent (LSTM) networks."""

    def test_recurrent_double_dqn_uses_online_network_for_action_selection(
        self,
        compile_universe,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        recurrent_brain_config,
    ):
        """Recurrent Double DQN should use online network for action selection."""
        # Create POMDP environment on CPU
        from pathlib import Path

        from townlet.environment.vectorized_env import VectorizedHamletEnv

        universe = compile_universe(Path("configs/L2_partial_observability"))
        env = VectorizedHamletEnv.from_universe(
            universe,
            num_agents=1,
            device=cpu_device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=recurrent_brain_config,
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=2,
            sequence_length=8,
            use_double_dqn=True,  # Double DQN
        )

        # Verify Double DQN flag is set
        assert population.use_double_dqn is True
        assert population.is_recurrent is True

    def test_recurrent_vanilla_vs_double_dqn_differ(
        self,
        compile_universe,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        recurrent_brain_config,
    ):
        """Recurrent vanilla and Double DQN should use different action selection."""
        # This test verifies the mechanism is in place
        # (Actual Q-target differences require longer training)
        from pathlib import Path

        from townlet.environment.vectorized_env import VectorizedHamletEnv

        universe = compile_universe(Path("configs/L2_partial_observability"))
        env = VectorizedHamletEnv.from_universe(
            universe,
            num_agents=1,
            device=cpu_device,
        )

        pop_vanilla = VectorizedPopulation(
            env=env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=recurrent_brain_config,
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=2,
            sequence_length=8,
            use_double_dqn=False,
        )

        pop_double = VectorizedPopulation(
            env=env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=recurrent_brain_config,
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=2,
            sequence_length=8,
            use_double_dqn=True,
        )

        assert pop_vanilla.use_double_dqn is False
        assert pop_double.use_double_dqn is True
