"""Tests for population-level action selection.

This file tests VectorizedPopulation action selection methods:
- Greedy selection (argmax of masked Q-values)
- Epsilon-greedy exploration (random + greedy with masks)
- Q-value masking (-inf for invalid actions)
- Recurrent network handling (tuple unpacking)

Extracted from test_action_selection.py lines 207-560.

NOTE: Environment-level masking tests are in unit/environment/test_action_masking.py
(lines 31-205 from original test_action_selection.py).

Coverage Target: VectorizedPopulation.select_greedy_actions() and
                 VectorizedPopulation.select_epsilon_greedy_actions()

Critical Behaviors:
1. Greedy selection chooses argmax of masked Q-values (deterministic)
2. Epsilon-greedy mixes exploitation (greedy) and exploration (random valid actions)
3. Invalid actions get -inf Q-value before argmax (masking)
4. Recurrent networks return (q_values, hidden_state) tuple - must unpack
5. Random exploration only samples from valid actions (respects masks)
"""

import pytest
import torch

from townlet.agent.networks import SimpleQNetwork
from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


class TestGreedyActionSelection:
    """Test greedy action selection with Q-value masking."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple network + environment."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
        )

        obs_dim = env.observation_dim
        network = SimpleQNetwork(obs_dim=obs_dim, action_dim=env.action_dim, hidden_dim=128)

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.1)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0, 1],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            # action_dim defaults to env.action_dim
            learning_rate=0.001,
            gamma=0.99,
            network_type="simple",
        )

        population.reset()
        return population, env, network

    def test_greedy_selection_respects_masks(self, simple_setup):
        """Greedy selection should never choose masked actions."""
        population, env, _ = simple_setup

        # Place agents at top-left corner (UP and LEFT masked)
        env.positions = torch.tensor([[0, 0], [0, 0]], device=env.device)

        # Select actions 100 times - should never see UP (0) or LEFT (2)
        for _ in range(100):
            actions = population.select_greedy_actions(env)
            assert actions.shape == (2,)

            # Should only be DOWN (1), RIGHT (3), INTERACT (4), or WAIT (5)
            assert all(a in [1, 3, 4, 5] for a in actions.tolist())

    def test_greedy_selects_highest_q_value(self, simple_setup):
        """Greedy selection should choose action with highest Q-value."""
        population, env, network = simple_setup

        # Place agents in center (all actions valid)
        env.positions = torch.tensor([[2, 2], [2, 2]], device=env.device)

        # Test that greedy selection uses Q-values
        # We can't easily control untrained network outputs, but we can verify:
        # 1. Actions are selected (no errors)
        # 2. Same observation gives same action (deterministic)

        actions1 = population.select_greedy_actions(env)
        actions2 = population.select_greedy_actions(env)

        # Greedy should be deterministic for same state
        assert torch.equal(actions1, actions2)

        # Actions should be valid (0-5 including WAIT)
        assert all(0 <= a < 6 for a in actions1.tolist())

    def test_masked_actions_get_negative_infinity(self, simple_setup):
        """Masked actions should get -inf Q-value before argmax."""
        population, env, _network = simple_setup

        obs = env.reset()
        population.current_obs = obs

        # Place agent at top-left corner
        env.positions = torch.tensor([[0, 0], [2, 2]], device=env.device)

        with torch.no_grad():
            q_values = population.q_network(obs)
            if isinstance(q_values, tuple):  # Recurrent network
                q_values = q_values[0]

            # Get masks
            action_masks = env.get_action_masks()

            # Mask Q-values (this is what select_greedy_actions does internally)
            masked_q_values = q_values.clone()
            masked_q_values[~action_masks] = float("-inf")

            # For agent 0 (corner), UP and LEFT should be -inf
            assert masked_q_values[0, 0] == float("-inf")  # UP
            assert masked_q_values[0, 2] == float("-inf")  # LEFT

            # For agent 1 (center), at least DOWN, LEFT, RIGHT, UP should be valid (not -inf)
            # INTERACT and WAIT may or may not be valid depending on affordances
            movement_actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
            assert all(masked_q_values[1, a] > float("-inf") for a in movement_actions)


class TestEpsilonGreedyActionSelection:
    """Test epsilon-greedy action selection."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple network + environment."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
        )

        obs_dim = env.observation_dim

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.5)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0, 1],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            # action_dim defaults to env.action_dim
            learning_rate=0.001,
            gamma=0.99,
            network_type="simple",
        )

        population.reset()
        return population, env

    def test_epsilon_zero_is_greedy(self, simple_setup):
        """With epsilon=0, should always select greedy action."""
        population, env = simple_setup

        # Place agents in center
        env.positions = torch.tensor([[2, 2], [2, 2]], device=env.device)

        # Select with epsilon=0
        actions_list = []
        for _ in range(10):
            actions = population.select_epsilon_greedy_actions(env, epsilon=0.0)
            actions_list.append(actions.clone())

        # All selections should be identical (greedy)
        for actions in actions_list[1:]:
            assert torch.equal(actions, actions_list[0])

    def test_epsilon_one_is_random(self, simple_setup):
        """With epsilon=1, should always select random valid action."""
        population, env = simple_setup

        # Place agents in center (all 6 actions valid)
        env.positions = torch.tensor([[2, 2], [2, 2]], device=env.device)

        # Select with epsilon=1.0 many times
        actions_seen = set()
        for _ in range(50):
            actions = population.select_epsilon_greedy_actions(env, epsilon=1.0)
            actions_seen.update(actions.tolist())

        # Should see variety of actions (probabilistic, but with 50 tries very likely)
        assert len(actions_seen) >= 3  # Should see at least 3 different actions

    def test_epsilon_greedy_respects_masks(self, simple_setup):
        """Random exploration should only choose valid actions."""
        population, env = simple_setup

        # Place agents at top-left corner (UP and LEFT masked)
        env.positions = torch.tensor([[0, 0], [0, 0]], device=env.device)

        # Select with epsilon=1.0 (always random)
        for _ in range(100):
            actions = population.select_epsilon_greedy_actions(env, epsilon=1.0)

            # Should never see UP (0) or LEFT (2)
            # Valid actions: DOWN (1), RIGHT (3), INTERACT (4), WAIT (5)
            assert all(a in [1, 3, 4, 5] for a in actions.tolist())

    def test_epsilon_greedy_mixes_exploration_exploitation(self, simple_setup):
        """With 0 < epsilon < 1, should see mix of greedy and random."""
        population, env = simple_setup

        # Place agents in center
        env.positions = torch.tensor([[2, 2], [2, 2]], device=env.device)

        # Select with epsilon=0.5
        actions_list = []
        for _ in range(50):
            actions = population.select_epsilon_greedy_actions(env, epsilon=0.5)
            actions_list.append(actions.clone())

        # Should see some variation (not all identical)
        unique_action_pairs = set()
        for actions in actions_list:
            unique_action_pairs.add(tuple(actions.tolist()))

        # With epsilon=0.5 and 50 trials, should see multiple different action pairs
        assert len(unique_action_pairs) > 1


class TestRecurrentNetworkActionSelection:
    """Test action selection with recurrent networks."""

    @pytest.fixture
    def recurrent_setup(self):
        """Create recurrent network + environment."""
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
            device=torch.device("cpu"),
        )

        obs_dim = env.observation_dim

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.1)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0, 1],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            # action_dim defaults to env.action_dim
            learning_rate=0.001,
            gamma=0.99,
            network_type="recurrent",
        )

        population.reset()
        return population, env

    def test_recurrent_greedy_selection(self, recurrent_setup):
        """Recurrent network should handle tuple output in action selection."""
        population, env = recurrent_setup

        # Place agents in center
        env.positions = torch.tensor([[2, 2], [2, 2]], device=env.device)

        # Should not raise error (handles tuple unpacking)
        actions = population.select_greedy_actions(env)

        assert actions.shape == (2,)
        assert actions.dtype == torch.long

    def test_recurrent_epsilon_greedy_selection(self, recurrent_setup):
        """Recurrent network should work with epsilon-greedy."""
        population, env = recurrent_setup

        # Place agents in center
        env.positions = torch.tensor([[2, 2], [2, 2]], device=env.device)

        # Should not raise error
        actions = population.select_epsilon_greedy_actions(env, epsilon=0.5)

        assert actions.shape == (2,)
        assert actions.dtype == torch.long

    def test_recurrent_respects_masks_at_boundary(self, recurrent_setup):
        """Recurrent network should respect action masks."""
        population, env = recurrent_setup

        # Place agents at top boundary
        env.positions = torch.tensor([[2, 0], [2, 0]], device=env.device)

        # Select actions many times
        for _ in range(50):
            actions = population.select_greedy_actions(env)

            # Should never select UP (0)
            assert all(a != 0 for a in actions.tolist())


class TestActionSelectionEdgeCases:
    """Test edge cases in action selection."""

    @pytest.fixture
    def minimal_env(self):
        """Create minimal 5×5 environment with single agent."""
        return VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
        )

    def test_single_agent_at_corner(self, minimal_env):
        """Should handle case where some actions are masked at corner."""
        obs_dim = minimal_env.observation_dim

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.1)

        population = VectorizedPopulation(
            env=minimal_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            # action_dim defaults to env.action_dim
            learning_rate=0.001,
            gamma=0.99,
            network_type="simple",
        )

        population.reset()

        # Place agent at top-left corner
        # UP and LEFT are masked
        minimal_env.positions = torch.tensor([[0, 0]], device=minimal_env.device)

        # Should select one of the valid actions (not UP or LEFT)
        actions = population.select_greedy_actions(minimal_env)
        assert actions[0] not in [0, 2]  # Should NOT be UP or LEFT

    def test_all_movement_actions_valid_at_center(self):
        """Test case where all movement actions are valid (center position)."""
        # This verifies masking logic works correctly when no movements are masked

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            device=torch.device("cpu"),
        )

        env.reset()  # Initialize environment

        # Place in center - all movements valid
        env.positions = torch.tensor([[2, 2]], device=env.device)
        masks = env.get_action_masks()

        # All movement actions should be valid
        assert masks[0, 0]  # UP
        assert masks[0, 1]  # DOWN
        assert masks[0, 2]  # LEFT
        assert masks[0, 3]  # RIGHT
        # INTERACT and WAIT validity depends on affordances/temporal mechanics
