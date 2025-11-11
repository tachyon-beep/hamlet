"""Tests for VectorizedPopulation Double DQN configuration."""

from townlet.population.vectorized import VectorizedPopulation


class TestDoubleDQNConfiguration:
    """Test Double DQN parameter plumbing."""

    def test_population_accepts_use_double_dqn_parameter(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """VectorizedPopulation should accept use_double_dqn parameter."""
        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=32,
            use_double_dqn=True,  # NEW PARAMETER
        )

        assert population.use_double_dqn is True

    def test_population_defaults_to_vanilla_dqn_when_false(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """VectorizedPopulation with use_double_dqn=False uses vanilla DQN."""
        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=32,
            use_double_dqn=False,
        )

        assert population.use_double_dqn is False

    def test_population_stores_use_double_dqn_attribute(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """use_double_dqn should be stored as instance attribute."""
        pop_vanilla = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=32,
            use_double_dqn=False,
        )

        pop_double = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            learning_rate=0.001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=32,
            use_double_dqn=True,
        )

        assert pop_vanilla.use_double_dqn is False
        assert pop_double.use_double_dqn is True
