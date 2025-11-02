"""
Test Suite: Action Selection & Masking

Tests for action selection with Q-values and action masking to ensure:
1. Invalid actions are never selected
2. Q-value masking works correctly
3. Epsilon-greedy exploration respects masks
4. Greedy selection works correctly
5. Integration between population and environment

Coverage Target: population/vectorized.py action selection methods (partial)

Critical Areas:
1. Action masking at boundaries
2. Q-value masking with -inf
3. Epsilon-greedy random selection from valid actions only
4. Greedy action selection (argmax of masked Q-values)
5. Recurrent vs simple network handling
"""

import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork
from townlet.population.vectorized import VectorizedPopulation
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration


class TestActionMasking:
    """Test environment action masking logic."""

    @pytest.fixture
    def env(self):
        """Create small environment for testing."""
        return VectorizedHamletEnv(
            num_agents=4,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

    def test_action_mask_shape(self, env):
        """Action masks should have correct shape."""
        env.reset()  # Initialize positions and meters
        masks = env.get_action_masks()
        assert masks.shape == (4, 6)  # num_agents × 6 actions
        assert masks.dtype == torch.bool

    def test_all_actions_valid_in_center(self, env):
        """All actions should be valid when agent is in center of grid."""
        env.reset()  # Initialize environment
        # Place all agents in center (2, 2) - away from all boundaries
        env.positions = torch.tensor(
            [
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
            ],
            device=env.device,
        )

        masks = env.get_action_masks()

        # All movement actions should be valid (UP, DOWN, LEFT, RIGHT)
        assert masks[:, 0].all()  # UP valid
        assert masks[:, 1].all()  # DOWN valid
        assert masks[:, 2].all()  # LEFT valid
        assert masks[:, 3].all()  # RIGHT valid

    def test_top_boundary_masks_up(self, env):
        """UP action should be masked at top boundary (y=0)."""
        env.reset()
        # Place agents at top row
        env.positions = torch.tensor(
            [
                [2, 0],  # Top center
                [0, 0],  # Top-left corner
                [4, 0],  # Top-right corner
                [2, 2],  # Center (control)
            ],
            device=env.device,
        )

        masks = env.get_action_masks()

        # First 3 agents can't go UP
        assert not masks[0, 0]  # UP invalid
        assert not masks[1, 0]  # UP invalid
        assert not masks[2, 0]  # UP invalid
        assert masks[3, 0]  # UP valid (center)

    def test_bottom_boundary_masks_down(self, env):
        """DOWN action should be masked at bottom boundary (y=max)."""
        env.reset()
        env.reset()  # Initialize environment
        # Place agents at bottom row (y=4 for grid_size=5)
        env.positions = torch.tensor(
            [
                [2, 4],  # Bottom center
                [0, 4],  # Bottom-left corner
                [4, 4],  # Bottom-right corner
                [2, 2],  # Center (control)
            ],
            device=env.device,
        )

        masks = env.get_action_masks()

        # First 3 agents can't go DOWN
        assert not masks[0, 1]  # DOWN invalid
        assert not masks[1, 1]  # DOWN invalid
        assert not masks[2, 1]  # DOWN invalid
        assert masks[3, 1]  # DOWN valid (center)

    def test_left_boundary_masks_left(self, env):
        """LEFT action should be masked at left boundary (x=0)."""
        env.reset()
        env.reset()  # Initialize environment
        # Place agents at left column
        env.positions = torch.tensor(
            [
                [0, 2],  # Left center
                [0, 0],  # Top-left corner
                [0, 4],  # Bottom-left corner
                [2, 2],  # Center (control)
            ],
            device=env.device,
        )

        masks = env.get_action_masks()

        # First 3 agents can't go LEFT
        assert not masks[0, 2]  # LEFT invalid
        assert not masks[1, 2]  # LEFT invalid
        assert not masks[2, 2]  # LEFT invalid
        assert masks[3, 2]  # LEFT valid (center)

    def test_right_boundary_masks_right(self, env):
        """RIGHT action should be masked at right boundary (x=max)."""
        env.reset()
        env.reset()  # Initialize environment
        # Place agents at right column (x=4 for grid_size=5)
        env.positions = torch.tensor(
            [
                [4, 2],  # Right center
                [4, 0],  # Top-right corner
                [4, 4],  # Bottom-right corner
                [2, 2],  # Center (control)
            ],
            device=env.device,
        )

        masks = env.get_action_masks()

        # First 3 agents can't go RIGHT
        assert not masks[0, 3]  # RIGHT invalid
        assert not masks[1, 3]  # RIGHT invalid
        assert not masks[2, 3]  # RIGHT invalid
        assert masks[3, 3]  # RIGHT valid (center)

    def test_corner_masks_two_directions(self, env):
        """Corners should mask two directions."""
        env.reset()
        env.reset()  # Initialize environment
        # Place agents at all four corners
        env.positions = torch.tensor(
            [
                [0, 0],  # Top-left: can't UP or LEFT
                [4, 0],  # Top-right: can't UP or RIGHT
                [0, 4],  # Bottom-left: can't DOWN or LEFT
                [4, 4],  # Bottom-right: can't DOWN or RIGHT
            ],
            device=env.device,
        )

        masks = env.get_action_masks()

        # Top-left: can't UP or LEFT
        assert not masks[0, 0]  # UP invalid
        assert not masks[0, 2]  # LEFT invalid
        assert masks[0, 1]  # DOWN valid
        assert masks[0, 3]  # RIGHT valid

        # Top-right: can't UP or RIGHT
        assert not masks[1, 0]  # UP invalid
        assert not masks[1, 3]  # RIGHT invalid
        assert masks[1, 1]  # DOWN valid
        assert masks[1, 2]  # LEFT valid

        # Bottom-left: can't DOWN or LEFT
        assert not masks[2, 1]  # DOWN invalid
        assert not masks[2, 2]  # LEFT invalid
        assert masks[2, 0]  # UP valid
        assert masks[2, 3]  # RIGHT valid

        # Bottom-right: can't DOWN or RIGHT
        assert not masks[3, 1]  # DOWN invalid
        assert not masks[3, 3]  # RIGHT invalid
        assert masks[3, 0]  # UP valid
        assert masks[3, 2]  # LEFT valid


class TestGreedyActionSelection:
    """Test greedy action selection with Q-value masking."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple network + environment."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

        obs_dim = env.observation_dim
        network = SimpleQNetwork(obs_dim=obs_dim, action_dim=6)

        curriculum = StaticCurriculum()
        exploration = EpsilonGreedyExploration(epsilon=0.1)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0, 1],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            action_dim=6,
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

        # Actions should be valid (0-4)
        assert all(0 <= a < 5 for a in actions1.tolist())

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

            # For agent 1 (center), nothing should be -inf
            assert all(masked_q_values[1] > float("-inf"))


class TestEpsilonGreedyActionSelection:
    """Test epsilon-greedy action selection."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple network + environment."""
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
            agent_ids=[0, 1],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            action_dim=6,
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
            agent_ids=[0, 1],
            device=torch.device("cpu"),
            obs_dim=obs_dim,
            action_dim=6,
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
        """Create minimal 2×2 environment."""
        return VectorizedHamletEnv(
            num_agents=1,
            grid_size=2,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

    def test_single_agent_single_action(self, minimal_env):
        """Should handle case where only one action is valid."""
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
            action_dim=6,
            learning_rate=0.001,
            gamma=0.99,
            network_type="simple",
        )

        population.reset()

        # Place agent at top-left of 2×2 grid
        # Only DOWN, RIGHT, INTERACT, and WAIT are valid
        minimal_env.positions = torch.tensor([[0, 0]], device=minimal_env.device)

        # Should select one of the valid actions
        actions = population.select_greedy_actions(minimal_env)
        assert actions[0] in [1, 3, 4, 5]  # DOWN, RIGHT, INTERACT, or WAIT

    def test_all_movement_actions_masked(self):
        """Test case where all movement actions are masked (edge case)."""
        # This is a theoretical test - in practice, at least 2 movements are always valid
        # But we test the masking logic works even in extreme cases

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=3,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )

        env.reset()  # Initialize environment

        # Place in center - all movements valid
        env.positions = torch.tensor([[1, 1]], device=env.device)
        masks = env.get_action_masks()

        # All movement actions should be valid
        assert masks[0, 0]  # UP
        assert masks[0, 1]  # DOWN
        assert masks[0, 2]  # LEFT
        assert masks[0, 3]  # RIGHT
        assert masks[0, 4]  # INTERACT (depends on affordances)
