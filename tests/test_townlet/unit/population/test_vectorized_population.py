"""Tests for VectorizedPopulation Double DQN configuration."""

import pytest

from townlet.agent.brain_config import (
    ArchitectureConfig,
    BrainConfig,
    FeedforwardConfig,
    LossConfig,
    OptimizerConfig,
    QLearningConfig,
    ReplayConfig,
    ScheduleConfig,
)
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


@pytest.fixture
def simple_brain_config():
    """Create a simple BrainConfig for testing."""
    return BrainConfig(
        version="1.0",
        description="Test brain config",
        architecture=ArchitectureConfig(
            type="feedforward",
            feedforward=FeedforwardConfig(
                hidden_layers=[128, 64],
                activation="relu",
                dropout=0.0,
                layer_norm=False,
            ),
        ),
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
            schedule=ScheduleConfig(type="constant"),
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
        replay=ReplayConfig(
            capacity=10000,
            prioritized=False,
        ),
    )


class TestBrainConfigIntegration:
    """Test BrainConfig parameter plumbing."""

    def test_population_accepts_brain_config_parameter(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        simple_brain_config,
    ):
        """VectorizedPopulation should accept brain_config parameter."""
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
            brain_config=simple_brain_config,
        )

        assert population.brain_config is simple_brain_config

    def test_population_builds_network_from_brain_config(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        simple_brain_config,
    ):
        """VectorizedPopulation should build Q-network from brain_config."""
        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            brain_config=simple_brain_config,
        )

        # Network should be built from config (not hardcoded)
        # Verify network architecture matches brain_config.architecture.feedforward
        import torch.nn as nn

        assert isinstance(population.q_network, nn.Sequential)
        # Config has hidden_layers=[128, 64], so we expect:
        # Linear(obs_dim -> 128), ReLU, Linear(128 -> 64), ReLU, Linear(64 -> action_dim)
        # Total: 5 layers (2 linear + 2 activation + 1 output linear)
        assert len(population.q_network) == 5

    def test_population_builds_optimizer_from_brain_config(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
        simple_brain_config,
    ):
        """VectorizedPopulation should build optimizer from brain_config."""
        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            brain_config=simple_brain_config,
        )

        # Optimizer should be Adam with config parameters
        import torch.optim as optim

        assert isinstance(population.optimizer, optim.Adam)
        # Verify learning rate matches config
        assert population.optimizer.param_groups[0]["lr"] == 0.001

    def test_brain_config_overrides_q_learning_parameters(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """brain_config q_learning fields should override constructor parameters."""
        # Create brain_config with specific q_learning values
        brain_config = BrainConfig(
            version="1.0",
            description="Test Q-learning override",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(type="constant"),
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.90,  # Different from constructor
                target_update_frequency=250,  # Different from constructor
                use_double_dqn=True,  # Different from constructor
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        # Constructor has different values
        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            gamma=0.99,  # Constructor says 0.99
            target_update_frequency=100,  # Constructor says 100
            use_double_dqn=False,  # Constructor says False
            brain_config=brain_config,
        )

        # brain_config should win
        assert population.gamma == 0.90
        assert population.target_update_frequency == 250
        assert population.use_double_dqn is True

    def test_configured_loss_function_is_used(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """Configured loss function should be used in training."""
        import torch.nn as nn

        # Create brain_config with Huber loss
        brain_config = BrainConfig(
            version="1.0",
            description="Test Huber loss usage",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(type="constant"),
            ),
            loss=LossConfig(type="huber", huber_delta=2.0),  # Huber loss with delta=2.0
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            brain_config=brain_config,
        )

        # Verify loss function is HuberLoss with correct delta
        assert isinstance(population.loss_fn, nn.HuberLoss)
        assert population.loss_fn.delta == 2.0


class TestRecurrentNetworkSupport:
    """Test recurrent network integration (Phase 2)."""

    def test_population_builds_recurrent_network_from_brain_config(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """VectorizedPopulation should build RecurrentSpatialQNetwork from recurrent config."""
        from townlet.agent.brain_config import (
            CNNEncoderConfig,
            LSTMConfig,
            MLPEncoderConfig,
            RecurrentConfig,
        )
        from townlet.agent.networks import RecurrentSpatialQNetwork

        brain_config = BrainConfig(
            version="1.0",
            description="Test recurrent config",
            architecture=ArchitectureConfig(
                type="recurrent",
                recurrent=RecurrentConfig(
                    vision_encoder=CNNEncoderConfig(
                        channels=[16, 32],
                        kernel_sizes=[3, 3],
                        strides=[1, 1],
                        padding=[1, 1],
                        activation="relu",
                    ),
                    position_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    meter_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    affordance_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    lstm=LSTMConfig(
                        hidden_size=256,
                        num_layers=1,
                        dropout=0.0,
                    ),
                    q_head=MLPEncoderConfig(
                        hidden_sizes=[128],
                        activation="relu",
                    ),
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.0001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(type="constant"),
            ),
            loss=LossConfig(type="huber"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=True,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",  # Should be ignored when brain_config present
            brain_config=brain_config,
        )

        # Should build RecurrentSpatialQNetwork
        assert isinstance(population.q_network, RecurrentSpatialQNetwork)
        assert isinstance(population.target_network, RecurrentSpatialQNetwork)
        assert population.is_recurrent is True

    def test_is_recurrent_flag_comes_from_brain_config_not_network_type(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """CRITICAL: is_recurrent flag must come from brain_config.architecture.type, not network_type parameter."""
        from townlet.agent.brain_config import (
            CNNEncoderConfig,
            LSTMConfig,
            MLPEncoderConfig,
            RecurrentConfig,
        )

        # Create recurrent brain_config
        recurrent_config = BrainConfig(
            version="1.0",
            description="Test is_recurrent flag correctness",
            architecture=ArchitectureConfig(
                type="recurrent",
                recurrent=RecurrentConfig(
                    vision_encoder=CNNEncoderConfig(
                        channels=[16, 32],
                        kernel_sizes=[3, 3],
                        strides=[1, 1],
                        padding=[1, 1],
                        activation="relu",
                    ),
                    position_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    meter_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    affordance_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    lstm=LSTMConfig(
                        hidden_size=256,
                        num_layers=1,
                        dropout=0.0,
                    ),
                    q_head=MLPEncoderConfig(
                        hidden_sizes=[128],
                        activation="relu",
                    ),
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.0001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(type="constant"),
            ),
            loss=LossConfig(type="huber"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=True,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        # Pass network_type="simple" but brain_config with recurrent architecture
        # The is_recurrent flag should come from brain_config, not network_type
        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",  # MISLEADING - should be ignored!
            brain_config=recurrent_config,
        )

        # CRITICAL: is_recurrent should be True (from brain_config.architecture.type)
        # NOT False (from network_type="simple")
        assert population.is_recurrent is True, (
            "is_recurrent flag must come from brain_config.architecture.type, not network_type parameter. "
            f"Expected True (from brain_config), got {population.is_recurrent} (from network_type)"
        )

    def test_is_recurrent_flag_uses_network_type_when_no_brain_config(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """When brain_config is None, is_recurrent should come from network_type parameter."""
        # Test feedforward network
        population_feedforward = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="simple",
            brain_config=None,
        )
        assert population_feedforward.is_recurrent is False

        # Test recurrent network
        population_recurrent = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            network_type="recurrent",
            brain_config=None,
        )
        assert population_recurrent.is_recurrent is True

    def test_recurrent_network_has_correct_dimensions(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """Recurrent network should have dimensions from config."""
        from townlet.agent.brain_config import (
            CNNEncoderConfig,
            LSTMConfig,
            MLPEncoderConfig,
            RecurrentConfig,
        )

        brain_config = BrainConfig(
            version="1.0",
            description="Test recurrent dimensions",
            architecture=ArchitectureConfig(
                type="recurrent",
                recurrent=RecurrentConfig(
                    vision_encoder=CNNEncoderConfig(
                        channels=[16, 32],
                        kernel_sizes=[3, 3],
                        strides=[1, 1],
                        padding=[1, 1],
                        activation="relu",
                    ),
                    position_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    meter_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    affordance_encoder=MLPEncoderConfig(
                        hidden_sizes=[32],
                        activation="relu",
                    ),
                    lstm=LSTMConfig(
                        hidden_size=128,  # Different from hardcoded 256
                        num_layers=1,
                        dropout=0.0,
                    ),
                    q_head=MLPEncoderConfig(
                        hidden_sizes=[128],
                        activation="relu",
                    ),
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.0001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(type="constant"),
            ),
            loss=LossConfig(type="huber"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=True,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=brain_config,
        )

        # LSTM hidden size should come from config (128), not hardcoded (256)
        assert population.q_network.lstm.hidden_size == 128


class TestSchedulerIntegration:
    """Test learning rate scheduler integration (Phase 2)."""

    def test_population_unpacks_scheduler_from_optimizer_factory(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """VectorizedPopulation should unpack (optimizer, scheduler) tuple."""
        from torch.optim.lr_scheduler import StepLR

        brain_config = BrainConfig(
            version="1.0",
            description="Test step decay schedule",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(
                    type="step_decay",
                    step_size=100,
                    gamma=0.1,
                ),
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=brain_config,
        )

        # Should unpack scheduler from OptimizerFactory.build()
        assert hasattr(population, "scheduler")
        assert isinstance(population.scheduler, StepLR)

    def test_population_has_no_scheduler_for_constant_schedule(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """VectorizedPopulation should have scheduler=None for constant schedule."""
        brain_config = BrainConfig(
            version="1.0",
            description="Test constant schedule",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(type="constant"),
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=brain_config,
        )

        # Constant schedule should result in scheduler=None
        assert hasattr(population, "scheduler")
        assert population.scheduler is None

    def test_exponential_scheduler_support(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """VectorizedPopulation should support ExponentialLR scheduler."""
        from torch.optim.lr_scheduler import ExponentialLR

        brain_config = BrainConfig(
            version="1.0",
            description="Test exponential schedule",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(
                    type="exponential",
                    gamma=0.9999,
                ),
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=brain_config,
        )

        assert isinstance(population.scheduler, ExponentialLR)

    def test_scheduler_state_persists_across_checkpoint_save_load(
        self,
        compile_universe,
        test_config_pack_path,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """Scheduler state should be saved and restored in checkpoints."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        # Create CPU-based environment
        universe = compile_universe(test_config_pack_path)
        env = VectorizedHamletEnv.from_universe(
            universe,
            num_agents=1,
            device=cpu_device,
        )

        brain_config = BrainConfig(
            version="1.0",
            description="Test scheduler state persistence",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(
                    type="step_decay",
                    step_size=100,
                    gamma=0.1,
                ),
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        # Create population with scheduler
        population1 = VectorizedPopulation(
            env=env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,  # Use actual observation dimension
            brain_config=brain_config,
        )

        # Initialize curriculum
        adversarial_curriculum.initialize_population(1)

        # Take training steps to advance scheduler
        # Add some transitions to replay buffer first
        population1.reset()
        for _ in range(10):
            state = population1.step_population(env)
            if state.dones.any():
                break

        # Advance scheduler by triggering training steps
        # Fill replay buffer with enough transitions
        for _ in range(100):
            state = population1.step_population(env)
            if state.dones.any():
                env.reset()
                population1.reset()

        # Get scheduler step count before checkpoint
        initial_step_count = population1.scheduler.last_epoch

        # Save checkpoint
        checkpoint = population1.get_checkpoint_state()

        # Verify scheduler state is in checkpoint
        assert "scheduler" in checkpoint
        assert checkpoint["scheduler"] is not None
        assert checkpoint["scheduler"]["last_epoch"] == initial_step_count

        # Create new population
        population2 = VectorizedPopulation(
            env=env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,  # Use actual observation dimension
            brain_config=brain_config,
        )

        # Verify new population starts at step 0
        assert population2.scheduler.last_epoch == 0

        # Load checkpoint
        population2.load_checkpoint_state(checkpoint)

        # Verify scheduler state restored
        assert population2.scheduler.last_epoch == initial_step_count

    def test_checkpoint_without_scheduler_state_is_backward_compatible(
        self,
        basic_env,
        adversarial_curriculum,
        epsilon_greedy_exploration,
        cpu_device,
    ):
        """Loading old checkpoints without scheduler state should not crash."""
        brain_config = BrainConfig(
            version="1.0",
            description="Test backward compatibility",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(
                    type="step_decay",
                    step_size=100,
                    gamma=0.1,
                ),
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
            replay=ReplayConfig(
                capacity=10000,
                prioritized=False,
            ),
        )

        # Create population
        population = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=brain_config,
        )

        # Create checkpoint without scheduler state (simulating old checkpoint)
        checkpoint = population.get_checkpoint_state()
        del checkpoint["scheduler"]  # Remove scheduler state to simulate old checkpoint

        # Create new population
        population2 = VectorizedPopulation(
            env=basic_env,
            curriculum=adversarial_curriculum,
            exploration=epsilon_greedy_exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            brain_config=brain_config,
        )

        # Load checkpoint should not crash
        population2.load_checkpoint_state(checkpoint)

        # Scheduler should remain at initial state (step 0)
        assert population2.scheduler.last_epoch == 0
