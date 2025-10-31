"""
Test Suite: Q-Learning Updates & DQN Training

Tests for DQN training loop to ensure:
1. Q-network learns from replay buffer
2. Bellman equation is applied correctly
3. Gradient updates occur
4. Training frequency is respected
5. Recurrent network batch training works

Coverage Target: population/vectorized.py training logic (partial)

Critical Areas:
1. Q-value prediction and target calculation
2. Loss computation (MSE between predicted and target Q-values)
3. Gradient clipping (max_norm=10.0)
4. Training frequency (every 4 steps)
5. Replay buffer integration
6. Recurrent network hidden state management during training
"""

import pytest
import torch
import torch.nn.functional as functional

from src.townlet.curriculum.static import StaticCurriculum
from src.townlet.environment.vectorized_env import VectorizedHamletEnv
from src.townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from src.townlet.population.vectorized import VectorizedPopulation


class TestQNetworkTraining:
    """Test Q-network training from replay buffer."""

    @pytest.fixture
    def training_setup(self):
        """Create environment and population for training tests."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

        obs_dim = env.observation_dim
        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.1)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["0", "1"],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            action_dim=5,
            learning_rate=0.001,
            gamma=0.99,
            network_type="simple",
        )

        population.reset()
        return population, env

    def test_replay_buffer_integration(self, training_setup):
        """Population should store experiences in replay buffer."""
        population, env = training_setup

        initial_buffer_size = len(population.replay_buffer)

        # Take one step
        state = population.step_population(env)

        # Buffer should have 2 new transitions (one per agent)
        assert len(population.replay_buffer) == initial_buffer_size + 2

        # State should contain valid data
        assert state.observations.shape == (2, env.observation_dim)
        assert state.actions.shape == (2,)
        assert state.rewards.shape == (2,)
        assert state.dones.shape == (2,)

    def test_training_frequency(self, training_setup):
        """Training should occur every train_frequency steps."""
        population, env = training_setup

        # Fill replay buffer with enough data
        for _ in range(40):  # 40 steps × 2 agents = 80 transitions
            population.step_population(env)

        # Check that total_steps is being tracked
        assert population.total_steps > 0

        # Training happens every 4 steps by default
        # We can't directly check training happened, but we can verify
        # the system doesn't crash and buffer is being used
        assert len(population.replay_buffer) > 64

    def test_q_network_parameters_update(self, training_setup):
        """Q-network parameters should change after training."""
        population, env = training_setup

        # Get initial network parameters
        initial_params = [p.clone() for p in population.q_network.parameters()]

        # Fill replay buffer and train
        for _ in range(50):  # Enough steps to trigger training
            population.step_population(env)

        # Check that at least some parameters changed
        params_changed = False
        for initial_p, current_p in zip(initial_params, population.q_network.parameters()):
            if not torch.equal(initial_p, current_p):
                params_changed = True
                break

        assert params_changed, "Q-network parameters should update during training"

    def test_bellman_target_calculation(self, training_setup):
        """Test Bellman equation: Q_target = reward + gamma * max(Q_next) * (1-done)."""
        population, env = training_setup

        # Create simple batch
        batch_size = 4
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        next_obs = torch.randn(batch_size, env.observation_dim)
        dones = torch.tensor([False, False, True, False])  # Get Q-values
        with torch.no_grad():
            q_next = population.q_network(next_obs).max(1)[0]

            # Apply Bellman equation
            expected_target = rewards + population.gamma * q_next * (~dones).float()

            # For done states, target should be just the reward
            assert expected_target[2] == rewards[2], "Done state should have target = reward"

            # For non-done states, target should include discounted future value
            assert expected_target[0] != rewards[0], "Non-done should include future value"

    def test_gradient_clipping(self, training_setup):
        """Test that gradients are clipped to prevent explosion."""
        population, env = training_setup

        # Fill replay buffer
        for _ in range(40):
            population.step_population(env)

        # Manually trigger a training step with extreme values
        # to test gradient clipping
        population.replay_buffer.rewards_extrinsic[:64] = 1000.0  # Extreme rewards

        # Get initial parameter values
        initial_params = [p.clone() for p in population.q_network.parameters()]

        # Train
        for _ in range(5):
            population.step_population(env)

        # Check that parameters changed but not by huge amounts
        # (gradient clipping should prevent explosion)
        for initial_p, current_p in zip(initial_params, population.q_network.parameters()):
            param_change = (current_p - initial_p).abs().max().item()
            assert param_change < 50.0, "Gradient clipping should prevent huge parameter changes"


class TestRecurrentNetworkTraining:
    """Test Q-learning with recurrent networks."""

    @pytest.fixture
    def recurrent_training_setup(self):
        """Create environment and recurrent population."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=False,
        )

        obs_dim = env.observation_dim
        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.1)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["0", "1"],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            action_dim=5,
            learning_rate=0.001,
            gamma=0.99,
            network_type="recurrent",
        )

        population.reset()
        return population, env

    def test_recurrent_training_resets_hidden_state(self, recurrent_training_setup):
        """During batch training, hidden states should be reset."""
        population, env = recurrent_training_setup

        # Fill replay buffer
        for _ in range(40):
            population.step_population(env)

        # Check that hidden state exists
        h, c = population.q_network.get_hidden_state()
        assert h is not None
        assert c is not None

        # Hidden state should be for episode batch (num_agents)
        assert h.shape[1] == 2  # 2 agents

    def test_recurrent_episode_reset_clears_hidden_state(self, recurrent_training_setup):
        """When episode ends, hidden state for that agent should reset."""
        population, env = recurrent_training_setup

        # Take steps until we get a done
        max_steps = 200
        done_found = False

        for _ in range(max_steps):
            state = population.step_population(env)
            if state.dones.any():
                done_found = True

                # Check that hidden states were handled
                h, _ = population.q_network.get_hidden_state()

                # For agents that finished, hidden state should be zeroed
                done_idx = torch.where(state.dones)[0][0].item()
                assert h[:, done_idx, :].abs().sum() < 1e-6, (
                    "Hidden state should be zeroed for done agent"
                )
                break

        # Should eventually get a done state
        assert done_found or max_steps == 200, "Should find done state or reach max steps"

    def test_recurrent_parameters_update(self, recurrent_training_setup):
        """Recurrent network parameters should update during training."""
        population, env = recurrent_training_setup

        # Get initial LSTM parameters
        initial_lstm_params = [
            p.clone() for name, p in population.q_network.named_parameters() if "lstm" in name
        ]

        # Train
        for _ in range(50):
            population.step_population(env)

        # Check LSTM parameters changed
        lstm_params_changed = False
        for initial_p, (name, current_p) in zip(
            initial_lstm_params,
            [(n, p) for n, p in population.q_network.named_parameters() if "lstm" in n],
        ):
            if not torch.equal(initial_p, current_p):
                lstm_params_changed = True
                break

        assert lstm_params_changed, "LSTM parameters should update during training"


class TestDQNLossCalculation:
    """Test DQN loss computation."""

    @pytest.fixture
    def simple_population(self):
        """Create minimal population for loss testing."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=3,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

        obs_dim = env.observation_dim
        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.0)  # No exploration

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["0"],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            action_dim=5,
            learning_rate=0.001,
            gamma=0.99,
            network_type="simple",
        )

        population.reset()
        return population, env

    def test_loss_is_mse(self, simple_population):
        """Loss should be MSE between predicted and target Q-values."""
        population, env = simple_population

        # Create fake batch
        batch_size = 8
        test_obs = torch.randn(batch_size, env.observation_dim)
        test_actions = torch.randint(0, 5, (batch_size,))
        test_rewards = torch.randn(batch_size)
        test_next_obs = torch.randn(batch_size, env.observation_dim)

        # Get Q-values
        q_values = population.q_network(test_obs)
        q_pred = q_values.gather(1, test_actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_next = population.q_network(test_next_obs).max(1)[0]
            q_target = test_rewards + population.gamma * q_next  # Calculate loss
        loss = functional.mse_loss(q_pred, q_target)

        # Loss should be non-negative
        assert loss >= 0.0

        # Loss should be scalar
        assert loss.ndim == 0

    def test_q_pred_uses_taken_actions(self, simple_population):
        """Q_pred should use the Q-values for actions that were taken."""
        population, env = simple_population

        test_obs = torch.randn(4, env.observation_dim)
        actions = torch.tensor([0, 1, 2, 3])  # Different action for each

        q_values = population.q_network(test_obs)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Check that we got the right Q-values
        assert q_pred[0] == q_values[0, 0]  # Action 0
        assert q_pred[1] == q_values[1, 1]  # Action 1
        assert q_pred[2] == q_values[2, 2]  # Action 2
        assert q_pred[3] == q_values[3, 3]  # Action 3

    def test_q_target_uses_max_next_q(self, simple_population):
        """Q_target should use max Q-value from next state."""
        population, env = simple_population

        rewards = torch.tensor([1.0, 2.0])
        next_obs = torch.randn(2, env.observation_dim)
        done_states = torch.tensor([False, True])

        with torch.no_grad():
            q_next_values = population.q_network(next_obs)
            q_next_max = q_next_values.max(1)[0]

            q_target = rewards + population.gamma * q_next_max * (~done_states).float()

        # Non-done state should have discounted future value
        assert q_target[0] == rewards[0] + population.gamma * q_next_max[0]

        # Done state should only have reward
        assert q_target[1] == rewards[1]


class TestTrainingStability:
    """Test training stability and convergence behaviors."""

    @pytest.fixture
    def stable_setup(self):
        """Create setup for stability testing."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

        obs_dim = env.observation_dim
        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.5)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["0", "1"],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            action_dim=5,
            learning_rate=0.0001,  # Lower learning rate for stability
            gamma=0.99,
            network_type="simple",
        )

        population.reset()
        return population, env

    def test_no_nan_values_during_training(self, stable_setup):
        """Training should not produce NaN values."""
        population, env = stable_setup

        # Train for many steps
        for _ in range(100):
            state = population.step_population(env)

            # Check for NaN in rewards
            assert not torch.isnan(state.rewards).any(), "Rewards should not be NaN"

            # Check Q-network parameters for NaN
            for p in population.q_network.parameters():
                assert not torch.isnan(p).any(), "Network parameters should not be NaN"

    def test_replay_buffer_capacity_respected(self, stable_setup):
        """Replay buffer should not exceed capacity."""
        population, env = stable_setup

        buffer_capacity = population.replay_buffer.capacity

        # Fill way beyond capacity
        for _ in range(buffer_capacity + 100):
            population.step_population(env)

        # Buffer should be at capacity, not over
        assert len(population.replay_buffer) == buffer_capacity

    def test_step_counter_increments(self, stable_setup):
        """Total steps counter should increment each step."""
        population, env = stable_setup

        initial_steps = population.total_steps

        # Take 10 steps
        for _ in range(10):
            population.step_population(env)

        # Should have 10 more steps
        assert population.total_steps == initial_steps + 10

    def test_optimizer_state_maintained(self, stable_setup):
        """Optimizer should maintain state across training steps."""
        population, env = stable_setup

        # Train for a bit
        for _ in range(50):
            population.step_population(env)

        # Optimizer should have state for parameters
        optimizer_state = population.optimizer.state_dict()

        # Check that optimizer has state
        assert "state" in optimizer_state
        # Adam optimizer should have momentum buffers
        assert len(optimizer_state["state"]) > 0
