"""
Integration tests for recurrent networks (LSTM) in Townlet.

Tests LSTM hidden state management, batch training, and POMDP integration.
Covers both migrated tests from test_lstm_temporal_learning.py and new critical tests.

Test Organization:
- TestLSTMHiddenStatePersistence: Hidden state lifecycle during episodes (4 tests)
- TestLSTMBatchTraining: Batch training with sequences (3 tests - 1 new + 2 migrated)
- TestLSTMForwardPass: Forward pass with POMDP (1 test)
"""

import torch
import torch.nn.functional as functional

from townlet.agent.networks import RecurrentSpatialQNetwork
from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.sequential_replay_buffer import SequentialReplayBuffer

# =============================================================================
# TEST CLASS 1: LSTM Hidden State Persistence
# =============================================================================


class TestLSTMHiddenStatePersistence:
    """Test LSTM hidden state lifecycle during episode execution."""

    def test_hidden_state_persists_across_10_steps_within_episode(self, test_config_pack_path, cpu_device):
        """
        Verify hidden state evolves during episode rollout.

        Hidden state should change across steps as the agent accumulates
        experience, demonstrating that memory is being built up.
        """
        # Create small POMDP environment for fast testing
        # Use VERY low energy costs to ensure agent survives 10 steps
        # (cascade effects from satiation/mood could kill agent otherwise)
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=True,
            vision_range=2,  # 5×5 window
            enable_temporal_mechanics=False,
            move_energy_cost=0.0001,  # Reduced from 0.005 to prevent death
            wait_energy_cost=0.00001,  # Reduced from 0.001 to prevent death
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
        )

        # Create recurrent population
        curriculum = StaticCurriculum()
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
            action_dim=6,  # Match test environment action space (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
            network_type="recurrent",
            vision_window_size=5,
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=8,
        )

        # Reset environment and population
        population.reset()

        # Initialize all meters to 1.0 to prevent cascade-induced death during test
        env.meters.fill_(1.0)

        # Capture initial hidden state
        recurrent_network = population.q_network
        h0, c0 = recurrent_network.get_hidden_state()
        hidden_states = [(h0.clone(), c0.clone())]

        # Run 10 steps and verify agent survives
        for step_num in range(10):
            state = population.step_population(env)
            # Ensure agent didn't die (would reset hidden state and break test)
            assert not state.dones[0], (
                f"Agent died at step {step_num + 1}/10. "
                f"Energy: {env.meters[0, 0]:.3f}, Health: {env.meters[0, 6]:.3f}. "
                "This test requires agent to survive 10 steps to verify hidden state persistence. "
                "If this fails, energy costs may need to be reduced further."
            )
            h, c = recurrent_network.get_hidden_state()
            hidden_states.append((h.clone(), c.clone()))

        # Verify hidden state changed across steps
        for i in range(len(hidden_states) - 1):
            h_curr, c_curr = hidden_states[i]
            h_next, c_next = hidden_states[i + 1]

            # Hidden states should be different (memory evolving)
            assert not torch.allclose(h_curr, h_next, atol=1e-6), f"Hidden state should change between steps {i} and {i+1}"
            assert not torch.allclose(c_curr, c_next, atol=1e-6), f"Cell state should change between steps {i} and {i+1}"

    def test_hidden_state_resets_on_death(self, test_config_pack_path, cpu_device):
        """
        Verify hidden state resets when agent dies.

        When agent dies, hidden state should be reset to zeros to prevent
        memory contamination across episodes.
        """
        # Create environment with NO affordances (agents die quickly)
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
        )

        # Disable all affordances in environment (force death)
        env.enabled_affordances = []
        env.deployed_affordances = {}

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(
            epsilon=0.0,  # Greedy for determinism
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            action_dim=6,  # Match test environment action space
            network_type="recurrent",
            vision_window_size=5,
        )

        population.reset()

        # Run until death (energy drains to 0)
        max_steps = 200  # Safety limit
        done = False
        for step in range(max_steps):
            state = population.step_population(env)
            if state.dones[0]:
                done = True
                break

        assert done, "Agent should have died"

        # After death, hidden state should be zeros
        recurrent_network = population.q_network
        h, c = recurrent_network.get_hidden_state()

        assert torch.allclose(h, torch.zeros_like(h)), "Hidden state should be zeros after death"
        assert torch.allclose(c, torch.zeros_like(c)), "Cell state should be zeros after death"

    def test_hidden_state_resets_after_flush_on_max_steps(self, test_config_pack_path, cpu_device):
        """
        Verify hidden state resets after flush_episode() on max_steps survival.

        When agent survives max_steps, flush_episode() should reset hidden state
        to prevent memory leakage into next episode.
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
        )

        curriculum = StaticCurriculum()
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
            action_dim=6,  # Match test environment action space
            network_type="recurrent",
            vision_window_size=5,
        )

        population.reset()

        # Initialize all meters to 1.0 to prevent cascade-induced death during test
        env.meters.fill_(1.0)

        # Run 50 steps (no death expected)
        for _ in range(50):
            population.step_population(env)

        # Flush episode (max_steps survival)
        population.flush_episode(agent_idx=0)

        # Hidden state should be zeros after flush
        recurrent_network = population.q_network
        h, c = recurrent_network.get_hidden_state()

        assert torch.allclose(h, torch.zeros_like(h)), "Hidden state should be zeros after flush"
        assert torch.allclose(c, torch.zeros_like(c)), "Cell state should be zeros after flush"

    def test_hidden_state_shape_correct_during_episode(self, test_config_pack_path, cpu_device):
        """
        Verify hidden state shape during multi-agent rollout.

        Hidden state shape should be [1, num_agents, 256] throughout episode.
        """
        # Create environment with 2 agents
        # Use VERY low energy costs to ensure agents survive 10 steps
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.0001,  # Reduced from 0.005 to prevent death
            wait_energy_cost=0.00001,  # Reduced from 0.001 to prevent death
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
        )

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(
            epsilon=0.1,
            epsilon_min=0.1,
            epsilon_decay=1.0,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1"],
            device=cpu_device,
            action_dim=6,  # Match test environment action space
            network_type="recurrent",
            vision_window_size=5,
        )

        population.reset()

        # Initialize all meters to 1.0 to prevent cascade-induced death during test
        env.meters.fill_(1.0)

        # Verify initial shape
        recurrent_network = population.q_network
        h, c = recurrent_network.get_hidden_state()
        assert h.shape == (1, 2, 256), f"Expected shape (1, 2, 256), got {h.shape}"
        assert c.shape == (1, 2, 256), f"Expected shape (1, 2, 256), got {c.shape}"

        # Run 10 steps and verify shape remains consistent
        for _ in range(10):
            population.step_population(env)
            h, c = recurrent_network.get_hidden_state()
            assert h.shape == (1, 2, 256), f"Hidden state shape should be (1, 2, 256), got {h.shape}"
            assert c.shape == (1, 2, 256), f"Cell state shape should be (1, 2, 256), got {c.shape}"


# =============================================================================
# TEST CLASS 2: LSTM Batch Training
# =============================================================================


class TestLSTMBatchTraining:
    """Test LSTM batch training with sequence sampling."""

    def test_hidden_state_batch_size_correct_during_training(self, test_config_pack_path, cpu_device):
        """
        Verify hidden state shape changes from num_agents to batch_size during training.

        During episode: [1, num_agents, 256]
        During training: [1, batch_size, 256]
        After training: [1, num_agents, 256]
        """
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
        )

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(
            epsilon=0.1,
            epsilon_min=0.1,
            epsilon_decay=1.0,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1"],
            device=cpu_device,
            action_dim=6,  # Match test environment action space
            network_type="recurrent",
            vision_window_size=5,
            batch_size=8,
            train_frequency=4,
        )

        population.reset()

        # Initialize all meters to 1.0 to prevent cascade-induced death during test
        env.meters.fill_(1.0)

        # Verify initial shape (episode batch size = num_agents)
        recurrent_network = population.q_network
        h, c = recurrent_network.get_hidden_state()
        assert h.shape == (1, 2, 256), f"Initial hidden state should be (1, 2, 256), got {h.shape}"

        # Run enough steps to trigger training (need 16+ episodes in buffer)
        # Each episode needs to die to be stored
        for episode in range(20):
            # Run until death
            for step in range(100):
                state = population.step_population(env)
                if state.dones.any():
                    break

        # After training, hidden state should be back to num_agents
        h, c = recurrent_network.get_hidden_state()
        assert h.shape == (1, 2, 256), f"After training, hidden state should be (1, 2, 256), got {h.shape}"

    def test_lstm_training_with_sequences(self, cpu_device):
        """
        Test that LSTM can learn a simple temporal pattern: A → B → C.

        Migrated from test_lstm_temporal_learning.py::test_lstm_learns_temporal_sequence.

        Task: Learn that action sequences matter
        - State 0 (start) → State 1: Take action 0
        - State 1 → State 2 (goal): Take action 1
        - Reward only given at State 2

        An MLP would struggle (state 1 looks the same whether you came from 0 or elsewhere).
        An LSTM should learn the temporal dependency.
        """
        # Simple network for testing
        network = RecurrentSpatialQNetwork(
            action_dim=2,  # Actions: 0 (A→B) or 1 (B→C)
            window_size=1,  # Minimal spatial observation
            num_meters=2,  # Minimal state (just state_id encoded)
            num_affordance_types=1,
            enable_temporal_features=False,
            hidden_dim=32,  # Small for fast training
        ).to(cpu_device)

        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        # Create synthetic episode: State 0 → State 1 → State 2
        # Observation encoding: [grid (1), pos (2), meters (2), affordance (2)] = 7 dims
        # State 0: meters = [1.0, 0.0]
        # State 1: meters = [0.5, 0.5]
        # State 2: meters = [0.0, 1.0]

        def make_observation(state_id):
            """Create synthetic observation for a state."""
            grid = torch.zeros(1)  # Minimal grid
            pos = torch.tensor([0.0, 0.0])  # Fixed position

            if state_id == 0:
                meters = torch.tensor([1.0, 0.0])
            elif state_id == 1:
                meters = torch.tensor([0.5, 0.5])
            else:  # state_id == 2
                meters = torch.tensor([0.0, 1.0])

            affordance = torch.zeros(2)  # No affordance

            return torch.cat([grid, pos, meters, affordance])

        # Create training episode: 0 → 1 → 2
        episode = {
            "observations": torch.stack(
                [
                    make_observation(0),
                    make_observation(1),
                    make_observation(2),
                ]
            ),
            "actions": torch.tensor([0, 1, 0]),  # A→B, B→C, done
            "rewards_extrinsic": torch.tensor([0.0, 0.0, 1.0]),  # Reward only at end
            "rewards_intrinsic": torch.tensor([0.0, 0.0, 0.0]),
            "dones": torch.tensor([False, False, True]),
        }

        # Store episode in replay buffer
        replay_buffer = SequentialReplayBuffer(capacity=100, device=cpu_device)
        replay_buffer.store_episode(episode)

        # Store a few more episodes for sampling variety
        for _ in range(15):
            replay_buffer.store_episode(episode)

        # Create target network (for stable temporal learning)
        target_network = RecurrentSpatialQNetwork(
            action_dim=2,
            window_size=1,
            num_meters=2,
            num_affordance_types=1,
            enable_temporal_features=False,
            hidden_dim=32,
        ).to(cpu_device)
        target_network.load_state_dict(network.state_dict())
        target_network.eval()

        # Train for multiple iterations
        losses = []
        for iteration in range(500):  # More iterations for convergence
            # Sample sequence
            batch = replay_buffer.sample_sequences(
                batch_size=8,
                seq_len=3,
                intrinsic_weight=0.0,
            )

            gamma = 0.99

            # PASS 1: Collect Q-predictions from online network
            network.reset_hidden_state(batch_size=8, device=cpu_device)
            q_pred_list = []

            for t in range(3):
                q_values, _ = network(batch["observations"][:, t, :])
                q_pred = q_values.gather(1, batch["actions"][:, t].unsqueeze(1)).squeeze()
                q_pred_list.append(q_pred)

            # PASS 2: Collect Q-targets from target network (maintains hidden state!)
            with torch.no_grad():
                target_network.reset_hidden_state(batch_size=8, device=cpu_device)
                q_target_list = []
                q_values_list = []

                # First, unroll through entire sequence to get Q-values at each step
                for t in range(3):
                    q_values, _ = target_network(batch["observations"][:, t, :])
                    q_values_list.append(q_values)

                # Now compute targets using the Q-values from next timestep
                for t in range(3):
                    if t < 2:
                        # Use Q-values from t+1 (already computed with hidden state from t)
                        q_next = q_values_list[t + 1].max(1)[0]
                        q_target = batch["rewards"][:, t] + gamma * q_next * (~batch["dones"][:, t]).float()
                    else:
                        # Terminal state
                        q_target = batch["rewards"][:, t]

                    q_target_list.append(q_target)

            # Compute loss with masking (P2.2)
            q_pred_all = torch.stack(q_pred_list, dim=1)
            q_target_all = torch.stack(q_target_list, dim=1)

            # Apply mask to prevent post-terminal gradients
            losses_raw = functional.mse_loss(q_pred_all, q_target_all, reduction="none")
            mask = batch["mask"].float()
            loss = (losses_raw * mask).sum() / mask.sum().clamp_min(1)

            # Backprop through time
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            # Update target network every 10 steps
            if iteration % 10 == 0:
                target_network.load_state_dict(network.state_dict())

            losses.append(loss.item())

        # Verify loss decreased (learning happened)
        initial_loss = sum(losses[:10]) / 10
        final_loss = sum(losses[-10:]) / 10

        assert final_loss < initial_loss * 0.5, (
            f"Loss should decrease during training. " f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
        )

    def test_lstm_memory_persistence_in_training(self, cpu_device):
        """
        Test that LSTM hidden state persists across timesteps during sequence training.

        Migrated from test_lstm_temporal_learning.py::test_lstm_memory_persistence_in_training.

        This validates that the training loop correctly maintains hidden state
        when unrolling through sequences (as opposed to resetting each timestep).
        """
        network = RecurrentSpatialQNetwork(
            action_dim=2,
            window_size=1,
            num_meters=2,
            num_affordance_types=1,
            enable_temporal_features=False,
            hidden_dim=16,
        ).to(cpu_device)

        # Create a sequence of 3 observations
        obs_sequence = torch.randn(4, 3, 7)  # [batch=4, seq_len=3, obs_dim=7]

        # Reset hidden state once for the batch
        network.reset_hidden_state(batch_size=4, device=cpu_device)

        # Unroll through sequence, tracking hidden states
        hidden_states = []

        for t in range(3):
            q_values, hidden = network(obs_sequence[:, t, :])
            h, c = hidden
            hidden_states.append((h.clone(), c.clone()))

        # Verify hidden state changed across timesteps
        h0, c0 = hidden_states[0]
        h1, c1 = hidden_states[1]
        h2, c2 = hidden_states[2]

        # Hidden states should be different (memory is evolving)
        assert not torch.allclose(h0, h1), "Hidden state should change between timesteps"
        assert not torch.allclose(h1, h2), "Hidden state should change between timesteps"

        # Cell states should also be different
        assert not torch.allclose(c0, c1), "Cell state should change between timesteps"
        assert not torch.allclose(c1, c2), "Cell state should change between timesteps"


# =============================================================================
# TEST CLASS 3: LSTM Forward Pass
# =============================================================================


class TestLSTMForwardPass:
    """Test LSTM forward pass with POMDP observations."""

    def test_partial_observability_5x5_window_to_lstm(self, test_config_pack_path, cpu_device):
        """
        Verify POMDP environment → LSTM data flow.

        POMDP env (vision_range=2) produces 5×5 window.
        LSTM forward pass should work correctly and produce Q-values.
        """
        # Create POMDP environment
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=True,
            vision_range=2,  # 5×5 window
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
        )

        # Create recurrent network
        network = RecurrentSpatialQNetwork(
            action_dim=6,
            window_size=5,
            num_meters=8,
            num_affordance_types=env.num_affordance_types,
            enable_temporal_features=False,
            hidden_dim=256,
        ).to(cpu_device)

        # Reset environment and get observation
        obs = env.reset()

        # Verify observation shape (POMDP)
        # 5×5 grid (25) + position (2) + meters (8) + affordance (15) + temporal (4) = 54
        expected_obs_dim = 25 + 2 + 8 + 15 + 4
        assert obs.shape == (1, expected_obs_dim), f"Expected obs shape (1, {expected_obs_dim}), got {obs.shape}"

        # Reset hidden state
        network.reset_hidden_state(batch_size=1, device=cpu_device)

        # Forward pass
        q_values, new_hidden = network(obs)

        # Verify Q-values shape
        assert q_values.shape == (1, 6), f"Expected Q-values shape (1, 6), got {q_values.shape}"

        # Verify hidden state shape
        h, c = new_hidden
        assert h.shape == (1, 1, 256), f"Expected h shape (1, 1, 256), got {h.shape}"
        assert c.shape == (1, 1, 256), f"Expected c shape (1, 1, 256), got {c.shape}"

        # Verify Q-values are finite
        assert torch.isfinite(q_values).all(), "Q-values should be finite"

        # Verify hidden state changed from initial zeros
        initial_h, initial_c = network.get_hidden_state()
        assert not torch.allclose(h, initial_h, atol=1e-6), "Hidden state should change after forward pass"
        assert not torch.allclose(c, initial_c, atol=1e-6), "Cell state should change after forward pass"
