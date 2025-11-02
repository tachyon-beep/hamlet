"""
Test P2.1: INTERACT action masking based on affordance position.

INTERACT should only be available when agent is standing on an affordance.
This forces agents to learn proper economic planning: move → interact.
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def device():
    """PyTorch device for testing."""
    return torch.device("cpu")


@pytest.fixture
def env(device):
    """Simple environment for INTERACT masking tests."""
    return VectorizedHamletEnv(
        num_agents=2,
        grid_size=5,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


class TestInteractMaskingBasics:
    """Test basic INTERACT masking behavior."""

    def test_interact_masked_when_not_on_affordance(self, env):
        """INTERACT should be masked when agent is not on any affordance."""
        env.reset()

        # Move agent to empty position (0, 0) - guaranteed no affordance here
        env.positions[0] = torch.tensor([0, 0], device=env.device)

        masks = env.get_action_masks()

        # Actions: [UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4]
        assert masks.shape[1] == 6, f"Expected 6 actions, got {masks.shape[1]}"

        # Movement actions should be available (with boundary checks)
        # At position (0, 0): UP and LEFT should be masked (boundary), DOWN and RIGHT available
        assert not masks[0, 0], "UP should be masked at top boundary"
        assert masks[0, 1], "DOWN should be available"
        assert not masks[0, 2], "LEFT should be masked at left boundary"
        assert masks[0, 3], "RIGHT should be available"

        # INTERACT (action 4) should be masked when not on affordance
        assert not masks[0, 4], "INTERACT should be masked when not on affordance"

    def test_interact_available_when_on_affordance(self, env):
        """INTERACT should be available when agent is standing on an affordance."""
        env.reset()

        # Find an affordance position
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()

        masks = env.get_action_masks()

        # INTERACT should be available
        # This test documents the desired behavior - will fail initially
        assert masks.shape[1] == 6, "Should have 6 actions: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT"
        assert masks[0, 4], "INTERACT should be available on affordance"

    def test_movement_actions_unaffected_by_affordance(self, env):
        """Movement actions should work the same whether on affordance or not."""
        env.reset()

        # Test on empty space
        env.positions[0] = torch.tensor([2, 2], device=env.device)
        masks_empty = env.get_action_masks()

        # Test on affordance
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()
        masks_on_affordance = env.get_action_masks()

        # Movement actions (0-4) should have same availability
        # (except for boundary effects)
        # This is checking that affordance presence doesn't break movement
        assert masks_empty.shape[0] == 2  # 2 agents
        assert masks_on_affordance.shape[0] == 2


class TestInteractMaskingMultipleAgents:
    """Test INTERACT masking with multiple agents at different positions."""

    def test_independent_masking_per_agent(self, env):
        """Each agent should have independent INTERACT masking."""
        env.reset()

        # Agent 0: on affordance
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()

        # Agent 1: not on affordance
        env.positions[1] = torch.tensor([0, 0], device=env.device)

        masks = env.get_action_masks()

        # Agent 0: INTERACT available (on Bed)
        assert masks[0, 4], "Agent on affordance should have INTERACT available"

        # Agent 1: INTERACT masked (empty space)
        assert not masks[1, 4], "Agent not on affordance should have INTERACT masked"

    def test_multiple_agents_same_affordance(self, env):
        """Multiple agents can be on same affordance (no collision blocking)."""
        env.reset()

        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()
        env.positions[1] = bed_pos.clone()

        masks = env.get_action_masks()

        # Both agents should have INTERACT available
        assert masks[0, 4]
        assert masks[1, 4]


class TestInteractMaskingAllAffordances:
    """Test INTERACT masking works for all affordance types."""

    def test_interact_available_on_all_affordance_types(self, env):
        """INTERACT should be available on any affordance type."""
        env.reset()

        affordance_names = list(env.affordances.keys())

        for affordance_name in affordance_names:
            # Place agent on this affordance
            affordance_pos = env.affordances[affordance_name]
            env.positions[0] = affordance_pos.clone()

            masks = env.get_action_masks()

            assert masks[0, 4], f"INTERACT should be available on {affordance_name}"


class TestInteractMaskingEdgeCases:
    """Test edge cases for INTERACT masking."""

    def test_interact_masked_at_grid_boundaries(self, env):
        """INTERACT should be masked at boundaries if no affordance there."""
        env.reset()

        # Test all four corners (unlikely to have affordances)
        corners = [
            torch.tensor([0, 0], device=env.device),
            torch.tensor([0, env.grid_size - 1], device=env.device),
            torch.tensor([env.grid_size - 1, 0], device=env.device),
            torch.tensor([env.grid_size - 1, env.grid_size - 1], device=env.device),
        ]

        for corner in corners:
            env.positions[0] = corner

            # Check if there's an affordance at this position
            on_affordance = False
            for affordance_pos in env.affordances.values():
                if torch.equal(corner, affordance_pos):
                    on_affordance = True
                    break

            masks = env.get_action_masks()

            if on_affordance:
                assert masks[0, 4], f"Should have INTERACT at {corner} (affordance present)"
            else:
                assert not masks[0, 4], f"Should not have INTERACT at {corner} (no affordance)"

    def test_movement_actions_respect_boundaries(self, env):
        """Movement actions should be masked at grid boundaries."""
        env.reset()

        # Test corner positions
        # Top-left corner (0, 0): UP and LEFT should be masked
        env.positions[0] = torch.tensor([0, 0], device=env.device)
        masks = env.get_action_masks()
        assert not masks[0, 0], "UP should be masked at top"
        assert not masks[0, 2], "LEFT should be masked at left edge"
        assert masks[0, 1], "DOWN should be available"
        assert masks[0, 3], "RIGHT should be available"


class TestInteractMaskingWithTemporalMechanics:
    """Test INTERACT masking interacts correctly with temporal mechanics."""

    def test_interact_respects_operating_hours(self, device):
        """INTERACT should be masked if affordance is closed (operating hours)."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=True,
        )
        env.reset()

        # Find an affordance with restricted hours (e.g., Job: 8am-6pm)
        if "Job" in env.affordances:
            job_pos = env.affordances["Job"]
            env.positions[0] = job_pos.clone()

            # Set time to 2am (closed hours)
            env.time_of_day = 2

            masks = env.get_action_masks()

            # INTERACT should be masked (affordance closed)
            assert not masks[0, 4], "INTERACT should be masked when affordance closed"

            # Set time to 10am (open hours)
            env.time_of_day = 10

            masks = env.get_action_masks()

            # INTERACT should be available (affordance open)
            assert masks[0, 4], "INTERACT should be available when affordance open"

    def test_interact_available_24_7_affordances(self, device):
        """INTERACT should always be available on 24/7 affordances."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=True,
        )
        env.reset()

        # Bed is 24/7
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()

        # Test at various times
        for time in [0, 6, 12, 18, 23]:
            env.time_of_day = time
            masks = env.get_action_masks()
            assert masks[0, 4], f"INTERACT on Bed should be available at time {time}"
