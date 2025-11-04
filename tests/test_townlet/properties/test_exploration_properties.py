"""Property-based tests for exploration strategies.

These tests verify universal properties of exploration strategies that should
hold for all possible configurations and action sequences.

Properties tested:
1. Epsilon always stays in [epsilon_min, 1.0] after any number of decays
2. Selected actions are always valid (in [0, num_actions))
3. Epsilon decay is deterministic and monotonic
4. Action selection respects action masks when provided
"""

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from townlet.exploration.action_selection import epsilon_greedy_action_selection
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration


class TestEpsilonDecayProperties:
    """Property tests for epsilon decay behavior."""

    @given(
        epsilon_start=st.floats(min_value=0.5, max_value=1.0),
        epsilon_decay=st.floats(min_value=0.9, max_value=0.9999),
        epsilon_min=st.floats(min_value=0.001, max_value=0.1),
        num_decays=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50)
    def test_epsilon_stays_in_valid_range(self, epsilon_start, epsilon_decay, epsilon_min, num_decays):
        """Property: Epsilon always in [epsilon_min, 1.0] after any number of decays.

        No matter how many times we decay epsilon, it should never go below
        epsilon_min or above the starting value.
        """
        # Ensure epsilon_min < epsilon_start (valid configuration)
        assume(epsilon_min < epsilon_start)

        exploration = EpsilonGreedyExploration(epsilon=epsilon_start, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

        # Decay epsilon num_decays times
        for _ in range(num_decays):
            exploration.decay_epsilon()

            # PROPERTY: Epsilon is always in valid range
            assert exploration.epsilon >= epsilon_min, f"Epsilon {exploration.epsilon} < min {epsilon_min}"
            assert exploration.epsilon <= epsilon_start, f"Epsilon {exploration.epsilon} > start {epsilon_start}"

    @given(
        epsilon_decay=st.floats(min_value=0.9, max_value=0.999),
        num_decays=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=40)
    def test_epsilon_decay_is_monotonic(self, epsilon_decay, num_decays):
        """Property: Epsilon never increases (monotonically decreasing).

        Each decay step should either decrease epsilon or keep it at epsilon_min,
        but never increase it.
        """
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=epsilon_decay, epsilon_min=0.01)

        previous_epsilon = exploration.epsilon

        for _ in range(num_decays):
            exploration.decay_epsilon()

            # PROPERTY: Epsilon never increases
            assert exploration.epsilon <= previous_epsilon, f"Epsilon increased from {previous_epsilon} to {exploration.epsilon}"

            previous_epsilon = exploration.epsilon

    @given(
        epsilon_decay=st.floats(min_value=0.9, max_value=0.999),
        epsilon_min=st.floats(min_value=0.001, max_value=0.1),
        num_decays=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30)
    def test_epsilon_decay_is_deterministic(self, epsilon_decay, epsilon_min, num_decays):
        """Property: Same decay schedule produces same epsilon values.

        Given the same initial epsilon, decay rate, and number of steps,
        we should always get the same final epsilon value.
        """
        exploration1 = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

        exploration2 = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

        # Decay both the same number of times
        for _ in range(num_decays):
            exploration1.decay_epsilon()
            exploration2.decay_epsilon()

        # PROPERTY: Both should have identical epsilon
        assert (
            abs(exploration1.epsilon - exploration2.epsilon) < 1e-10
        ), f"Epsilons differ: {exploration1.epsilon} vs {exploration2.epsilon}"

    @given(
        epsilon_min=st.floats(min_value=0.001, max_value=0.1),
        num_decays=st.integers(min_value=5000, max_value=10000),  # More decays needed
    )
    @settings(max_examples=20)
    def test_epsilon_eventually_reaches_minimum(self, epsilon_min, num_decays):
        """Property: After enough decays, epsilon reaches epsilon_min.

        With exponential decay (0.995 per step), epsilon should eventually reach
        the minimum and stay there. After 5000+ decays, epsilon should be at minimum.
        """
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=0.995, epsilon_min=epsilon_min)

        # Decay many times
        for _ in range(num_decays):
            exploration.decay_epsilon()

        # PROPERTY: Epsilon should be at minimum (exact equality after clamping)
        assert exploration.epsilon == epsilon_min, f"Epsilon {exploration.epsilon} not at min {epsilon_min}"

        # PROPERTY: Further decays don't change epsilon
        exploration.decay_epsilon()
        assert exploration.epsilon == epsilon_min


class TestActionSelectionProperties:
    """Property tests for epsilon-greedy action selection."""

    @given(
        num_agents=st.integers(min_value=1, max_value=32),
        num_actions=st.integers(min_value=2, max_value=10),
        epsilon=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_selected_actions_are_always_valid(self, num_agents, num_actions, epsilon):
        """Property: Selected actions are always in [0, num_actions).

        Regardless of Q-values or epsilon, selected actions should always be
        valid indices in the action space.
        """
        # Create random Q-values
        q_values = torch.randn(num_agents, num_actions)

        # Create per-agent epsilons
        epsilons = torch.full((num_agents,), epsilon)

        # Select actions
        actions = epsilon_greedy_action_selection(q_values=q_values, epsilons=epsilons)

        # PROPERTY: All actions are valid indices
        assert torch.all(actions >= 0), f"Invalid action {actions.min()} < 0"
        assert torch.all(actions < num_actions), f"Invalid action {actions.max()} >= {num_actions}"

        # PROPERTY: Actions tensor has correct shape
        assert actions.shape == (num_agents,)

        # PROPERTY: Actions are integers
        assert actions.dtype == torch.long

    @given(
        num_agents=st.integers(min_value=1, max_value=16),
        num_actions=st.integers(min_value=3, max_value=8),
    )
    @settings(max_examples=40)
    def test_epsilon_zero_selects_greedy_actions(self, num_agents, num_actions):
        """Property: With epsilon=0, always select action with highest Q-value.

        When epsilon is 0 (no exploration), the selected action should always
        be the greedy action (argmax Q-value).
        """
        # Create Q-values where greedy action is obvious
        q_values = torch.randn(num_agents, num_actions)

        # For verification, compute greedy actions
        greedy_actions = q_values.argmax(dim=1)

        # Create per-agent epsilons (all zero)
        epsilons = torch.zeros(num_agents)

        # Select actions
        actions = epsilon_greedy_action_selection(q_values=q_values, epsilons=epsilons)

        # PROPERTY: All actions should be greedy
        assert torch.equal(actions, greedy_actions), f"Expected greedy {greedy_actions}, got {actions}"

    @given(
        num_agents=st.integers(min_value=1, max_value=16),
        num_actions=st.integers(min_value=3, max_value=8),
        num_masked=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=40)
    def test_action_masking_respected(self, num_agents, num_actions, num_masked):
        """Property: Masked actions are never selected.

        When action masks are provided, the selected actions should never
        include masked (invalid) actions.
        """
        # Ensure we don't mask all actions
        assume(num_masked < num_actions)

        q_values = torch.randn(num_agents, num_actions)

        # Create action masks (True = valid, False = masked)
        action_masks = torch.ones(num_agents, num_actions, dtype=torch.bool)

        # Mask some actions for each agent
        for agent_idx in range(num_agents):
            masked_actions = torch.randperm(num_actions)[:num_masked]
            action_masks[agent_idx, masked_actions] = False

        # Create per-agent epsilons
        epsilons = torch.full((num_agents,), 0.5)

        # Select actions with masks
        actions = epsilon_greedy_action_selection(q_values=q_values, epsilons=epsilons, action_masks=action_masks)

        # PROPERTY: No selected action should be masked
        for agent_idx in range(num_agents):
            selected_action = actions[agent_idx].item()
            assert action_masks[agent_idx, selected_action], f"Agent {agent_idx} selected masked action {selected_action}"


class TestExplorationCheckpointingProperties:
    """Property tests for exploration state serialization."""

    @given(
        epsilon=st.floats(min_value=0.01, max_value=1.0),
        epsilon_decay=st.floats(min_value=0.9, max_value=0.999),
        epsilon_min=st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=30)
    def test_checkpoint_restore_preserves_state(self, epsilon, epsilon_decay, epsilon_min):
        """Property: Checkpointing and restoring preserves exact state.

        After checkpointing and loading state, the exploration strategy should
        have identical parameters and produce identical behavior.
        """
        # Ensure epsilon >= epsilon_min
        assume(epsilon >= epsilon_min)

        exploration = EpsilonGreedyExploration(epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

        # Checkpoint state
        state = exploration.checkpoint_state()

        # Create new exploration and load state
        new_exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.1)
        new_exploration.load_state(state)

        # PROPERTY: All parameters are preserved
        assert new_exploration.epsilon == exploration.epsilon
        assert new_exploration.epsilon_decay == exploration.epsilon_decay
        assert new_exploration.epsilon_min == exploration.epsilon_min

        # PROPERTY: Both produce identical behavior
        for _ in range(10):
            exploration.decay_epsilon()
            new_exploration.decay_epsilon()

            assert abs(new_exploration.epsilon - exploration.epsilon) < 1e-10
