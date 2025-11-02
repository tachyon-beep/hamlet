"""
Comprehensive tests for all 6 actions in the Hamlet environment.

Tests cover:
- Movement actions (UP, DOWN, LEFT, RIGHT)
- Interaction action (INTERACT)
- Wait action (WAIT)
- Action costs, boundaries, masking, and integration
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
    """Standard environment with 2 agents."""
    return VectorizedHamletEnv(
        num_agents=2,
        grid_size=8,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


class TestActionSpace:
    """Test overall action space structure."""

    def test_action_space_has_six_actions(self, env):
        """Action space should have exactly 6 actions."""
        assert env.action_dim == 6

    def test_action_indices(self, env):
        """Verify action index mapping."""
        # UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4, WAIT=5
        env.reset()

        # Test each action index is valid
        for action_idx in range(6):
            actions = torch.full((env.num_agents,), action_idx, device=env.device)
            env.step(actions)  # Should not crash


class TestMovementActions:
    """Test movement actions: UP, DOWN, LEFT, RIGHT."""

    def test_up_action_moves_north(self, env):
        """UP action (0) should decrease y coordinate."""
        env.reset()
        initial_pos = torch.tensor([4, 4], device=env.device)
        env.positions[0] = initial_pos.clone()

        actions = torch.tensor([0, 5], device=env.device)  # Agent 0: UP, Agent 1: WAIT
        env.step(actions)

        expected_pos = torch.tensor([4, 3], device=env.device)  # y decreased
        assert torch.equal(env.positions[0], expected_pos), f"Expected {expected_pos}, got {env.positions[0]}"

    def test_down_action_moves_south(self, env):
        """DOWN action (1) should increase y coordinate."""
        env.reset()
        initial_pos = torch.tensor([4, 4], device=env.device)
        env.positions[0] = initial_pos.clone()

        actions = torch.tensor([1, 5], device=env.device)  # Agent 0: DOWN
        env.step(actions)

        expected_pos = torch.tensor([4, 5], device=env.device)  # y increased
        assert torch.equal(env.positions[0], expected_pos)

    def test_left_action_moves_west(self, env):
        """LEFT action (2) should decrease x coordinate."""
        env.reset()
        initial_pos = torch.tensor([4, 4], device=env.device)
        env.positions[0] = initial_pos.clone()

        actions = torch.tensor([2, 5], device=env.device)  # Agent 0: LEFT
        env.step(actions)

        expected_pos = torch.tensor([3, 4], device=env.device)  # x decreased
        assert torch.equal(env.positions[0], expected_pos)

    def test_right_action_moves_east(self, env):
        """RIGHT action (3) should increase x coordinate."""
        env.reset()
        initial_pos = torch.tensor([4, 4], device=env.device)
        env.positions[0] = initial_pos.clone()

        actions = torch.tensor([3, 5], device=env.device)  # Agent 0: RIGHT
        env.step(actions)

        expected_pos = torch.tensor([5, 4], device=env.device)  # x increased
        assert torch.equal(env.positions[0], expected_pos)

    def test_multiple_movements(self, env):
        """Test sequence of movements."""
        env.reset()
        env.positions[0] = torch.tensor([4, 4], device=env.device)

        # UP, RIGHT, DOWN, LEFT (should return to start)
        for action in [0, 3, 1, 2]:
            actions = torch.tensor([action, 5], device=env.device)
            env.step(actions)

        expected_pos = torch.tensor([4, 4], device=env.device)
        assert torch.equal(env.positions[0], expected_pos), "Should return to starting position"

    def test_movement_costs_energy(self, env):
        """All movement actions should cost energy."""
        # Test each movement action
        for action in [0, 1, 2, 3]:  # UP, DOWN, LEFT, RIGHT
            env.reset()
            initial = env.meters[0, 0].item()
            actions = torch.tensor([action, 5], device=env.device)
            env.step(actions)
            final = env.meters[0, 0].item()

            assert final < initial, f"Action {action} should drain energy"

    def test_all_movements_cost_same_energy(self, env):
        """All movement actions should have equal energy cost."""
        env.reset()
        energy_drains = []

        for action in [0, 1, 2, 3]:  # UP, DOWN, LEFT, RIGHT
            env.reset()
            initial = env.meters[0, 0].item()
            actions = torch.tensor([action, 5], device=env.device)
            env.step(actions)
            final = env.meters[0, 0].item()
            energy_drains.append(initial - final)

        # All drains should be equal (within floating point tolerance)
        for drain in energy_drains[1:]:
            assert abs(drain - energy_drains[0]) < 1e-6, "All movements should cost same energy"


class TestBoundaryHandling:
    """Test action masking and clamping at grid boundaries."""

    def test_up_masked_at_top_edge(self, env):
        """UP should be masked when agent is at y=0."""
        env.reset()
        env.positions[0] = torch.tensor([4, 0], device=env.device)  # Top edge

        masks = env.get_action_masks()

        assert not masks[0, 0], "UP should be masked at top edge"
        assert masks[0, 1], "DOWN should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_down_masked_at_bottom_edge(self, env):
        """DOWN should be masked when agent is at y=grid_size-1."""
        env.reset()
        env.positions[0] = torch.tensor([4, 7], device=env.device)  # Bottom edge (grid_size=8)

        masks = env.get_action_masks()

        assert not masks[0, 1], "DOWN should be masked at bottom edge"
        assert masks[0, 0], "UP should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_left_masked_at_left_edge(self, env):
        """LEFT should be masked when agent is at x=0."""
        env.reset()
        env.positions[0] = torch.tensor([0, 4], device=env.device)  # Left edge

        masks = env.get_action_masks()

        assert not masks[0, 2], "LEFT should be masked at left edge"
        assert masks[0, 3], "RIGHT should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_right_masked_at_right_edge(self, env):
        """RIGHT should be masked when agent is at x=grid_size-1."""
        env.reset()
        env.positions[0] = torch.tensor([7, 4], device=env.device)  # Right edge

        masks = env.get_action_masks()

        assert not masks[0, 3], "RIGHT should be masked at right edge"
        assert masks[0, 2], "LEFT should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_corner_masks_two_directions(self, env):
        """At corners, two movement directions should be masked."""
        env.reset()
        env.positions[0] = torch.tensor([0, 0], device=env.device)  # Top-left corner

        masks = env.get_action_masks()

        assert not masks[0, 0], "UP should be masked at corner"
        assert not masks[0, 2], "LEFT should be masked at corner"
        assert masks[0, 1], "DOWN should be available"
        assert masks[0, 3], "RIGHT should be available"
        assert masks[0, 5], "WAIT should be available"

    def test_movement_clamped_at_boundaries(self, env):
        """Movement beyond boundaries should be clamped."""
        env.reset()
        env.positions[0] = torch.tensor([0, 0], device=env.device)

        # Try to move UP (should clamp to y=0)
        actions = torch.tensor([0, 5], device=env.device)
        env.step(actions)

        assert env.positions[0, 1].item() == 0, "Y should be clamped at 0"

        # Try to move LEFT (should clamp to x=0)
        env.positions[0] = torch.tensor([0, 0], device=env.device)
        actions = torch.tensor([2, 5], device=env.device)
        env.step(actions)

        assert env.positions[0, 0].item() == 0, "X should be clamped at 0"


class TestInteractAction:
    """Test INTERACT action (4)."""

    def test_interact_action_no_movement(self, env):
        """INTERACT should not change agent position."""
        env.reset()
        initial_pos = env.positions[0].clone()

        # Place on Bed to make INTERACT valid
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()
        initial_pos = bed_pos.clone()

        actions = torch.tensor([4, 5], device=env.device)  # INTERACT
        env.step(actions)

        assert torch.equal(env.positions[0], initial_pos), "INTERACT should not move agent"

    def test_interact_masked_when_not_on_affordance(self, env):
        """INTERACT should be masked when not on an affordance."""
        env.reset()
        env.positions[0] = torch.tensor([4, 4], device=env.device)  # Empty cell

        masks = env.get_action_masks()

        assert not masks[0, 4], "INTERACT should be masked off affordance"

    def test_interact_available_on_affordable_affordance(self, env):
        """INTERACT should be available when on affordable affordance."""
        env.reset()

        # Place on Bed (free affordance)
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()
        env.meters[0, 3] = 1.0  # Max money

        masks = env.get_action_masks()

        assert masks[0, 4], "INTERACT should be available on Bed"

    def test_interact_restores_energy_on_bed(self, env):
        """INTERACT on Bed should restore energy."""
        env.reset()

        # Place on Bed with low energy
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()
        env.meters[0, 0] = 0.3  # Low energy

        initial_energy = env.meters[0, 0].item()

        actions = torch.tensor([4, 5], device=env.device)  # INTERACT
        env.step(actions)

        final_energy = env.meters[0, 0].item()

        # Bed should restore energy (not drain it)
        assert final_energy > initial_energy, "Bed should restore energy"


class TestWaitAction:
    """Test WAIT action (5)."""

    def test_wait_action_no_movement(self, env):
        """WAIT should not change agent position."""
        env.reset()
        initial_pos = env.positions[0].clone()

        actions = torch.tensor([5, 5], device=env.device)  # Both WAIT
        env.step(actions)

        assert torch.equal(env.positions[0], initial_pos), "WAIT should not move agent"

    def test_wait_always_available(self, env):
        """WAIT should always be available for alive agents."""
        env.reset()

        # Test at various positions
        test_positions = [
            [0, 0],  # Corner
            [7, 7],  # Opposite corner
            [4, 4],  # Center
            [0, 4],  # Edge
        ]

        for pos in test_positions:
            env.positions[0] = torch.tensor(pos, device=env.device)
            masks = env.get_action_masks()
            assert masks[0, 5], f"WAIT should be available at {pos}"

    def test_wait_costs_less_than_movement(self, env):
        """WAIT should cost less energy than movement."""
        env.reset()

        # Measure WAIT cost
        initial_wait = env.meters[0, 0].item()
        actions = torch.tensor([5, 5], device=env.device)
        env.step(actions)
        wait_cost = initial_wait - env.meters[0, 0].item()

        # Measure movement cost
        env.reset()
        initial_move = env.meters[0, 0].item()
        actions = torch.tensor([0, 5], device=env.device)  # UP
        env.step(actions)
        move_cost = initial_move - env.meters[0, 0].item()

        assert wait_cost < move_cost, f"WAIT cost ({wait_cost}) should be < movement cost ({move_cost})"


class TestActionCosts:
    """Test energy and meter costs for all actions."""

    def test_all_actions_deplete_energy(self, env):
        """All actions should deplete energy (including passive depletion)."""
        env.reset()

        for action in range(6):  # Test all 6 actions
            env.reset()
            initial_energy = env.meters[0, 0].item()

            # Place on Bed if testing INTERACT
            if action == 4:
                bed_pos = env.affordances["Bed"]
                env.positions[0] = bed_pos.clone()
                initial_energy = env.meters[0, 0].item()  # Re-measure after move

            actions = torch.tensor([action, 5], device=env.device)
            env.step(actions)

            final_energy = env.meters[0, 0].item()

            # Note: INTERACT on Bed restores energy, so it's special
            if action != 4:
                assert final_energy < initial_energy, f"Action {action} should deplete energy"

    def test_movement_drains_multiple_meters(self, env):
        """Movement should drain energy, hygiene, and satiation."""
        env.reset()

        initial_energy = env.meters[0, 0].item()
        initial_hygiene = env.meters[0, 1].item()
        initial_satiation = env.meters[0, 2].item()

        actions = torch.tensor([0, 5], device=env.device)  # UP
        env.step(actions)

        # Energy, hygiene, satiation should all decrease
        assert env.meters[0, 0].item() < initial_energy, "Movement should drain energy"
        assert env.meters[0, 1].item() < initial_hygiene, "Movement should drain hygiene"
        assert env.meters[0, 2].item() < initial_satiation, "Movement should drain satiation"

    def test_wait_only_drains_energy(self, env):
        """WAIT should only drain energy, not other meters."""
        env.reset()

        initial_hygiene = env.meters[0, 1].item()

        actions = torch.tensor([5, 5], device=env.device)  # WAIT
        env.step(actions)

        final_hygiene = env.meters[0, 1].item()

        # Hygiene decreases from passive depletion only (no action cost)
        # The decrease should be minimal (passive only)
        hygiene_decrease = initial_hygiene - final_hygiene

        # Compare with movement hygiene decrease
        env.reset()
        initial_hygiene_move = env.meters[0, 1].item()
        actions = torch.tensor([0, 5], device=env.device)  # UP
        env.step(actions)
        move_hygiene_decrease = initial_hygiene_move - env.meters[0, 1].item()

        # WAIT should drain less hygiene than movement
        assert hygiene_decrease < move_hygiene_decrease, "WAIT should drain less hygiene than movement"


class TestActionSequences:
    """Test sequences of actions."""

    def test_movement_sequence(self, env):
        """Test agent can execute movement sequence."""
        env.reset()
        env.positions[0] = torch.tensor([4, 4], device=env.device)

        sequence = [0, 0, 3, 3, 1, 1, 2, 2]  # UP UP RIGHT RIGHT DOWN DOWN LEFT LEFT

        for action in sequence:
            actions = torch.tensor([action, 5], device=env.device)
            env.step(actions)

        # Should be back at (4, 4)
        expected = torch.tensor([4, 4], device=env.device)
        assert torch.equal(env.positions[0], expected), "Should complete movement square"

    def test_mixed_action_sequence(self, env):
        """Test mixing different action types."""
        env.reset()
        initial_pos = torch.tensor([4, 4], device=env.device)
        env.positions[0] = initial_pos.clone()

        # Move, Wait, Move, Wait
        sequence = [0, 5, 1, 5]  # UP, WAIT, DOWN, WAIT

        for action in sequence:
            actions = torch.tensor([action, 5], device=env.device)
            env.step(actions)

        # Should be back at starting position
        assert torch.equal(env.positions[0], initial_pos)


class TestMultiAgentActions:
    """Test multiple agents taking actions simultaneously."""

    def test_agents_can_take_different_actions(self, env):
        """Different agents can take different actions simultaneously."""
        env.reset()
        env.positions[0] = torch.tensor([4, 4], device=env.device)
        env.positions[1] = torch.tensor([3, 3], device=env.device)

        # Agent 0: UP, Agent 1: DOWN
        actions = torch.tensor([0, 1], device=env.device)
        env.step(actions)

        assert env.positions[0, 1].item() == 3, "Agent 0 should move up"
        assert env.positions[1, 1].item() == 4, "Agent 1 should move down"

    def test_agents_can_take_same_action(self, env):
        """Multiple agents can take the same action."""
        env.reset()
        env.positions[0] = torch.tensor([4, 4], device=env.device)
        env.positions[1] = torch.tensor([3, 3], device=env.device)

        # Both agents: RIGHT
        actions = torch.tensor([3, 3], device=env.device)
        env.step(actions)

        assert env.positions[0, 0].item() == 5, "Agent 0 should move right"
        assert env.positions[1, 0].item() == 4, "Agent 1 should move right"

    def test_agents_can_occupy_same_cell(self, env):
        """Agents can move to and occupy the same cell."""
        env.reset()
        env.positions[0] = torch.tensor([4, 3], device=env.device)
        env.positions[1] = torch.tensor([4, 5], device=env.device)

        # Both move to center
        actions = torch.tensor([1, 0], device=env.device)  # DOWN, UP
        env.step(actions)

        # Both should be at [4, 4]
        assert torch.equal(env.positions[0], env.positions[1]), "Agents can occupy same cell"


class TestActionMaskingIntegration:
    """Test that action masking works correctly across all actions."""

    def test_dead_agents_have_all_actions_masked(self, env):
        """Dead agents should have all 6 actions masked."""
        env.reset()

        # Kill agent 0
        env.meters[0, 6] = 0.0  # Health = 0

        masks = env.get_action_masks()

        for action in range(6):
            assert not masks[0, action], f"Action {action} should be masked for dead agent"

    def test_alive_agents_have_valid_actions(self, env):
        """Alive agents in center should have most actions available."""
        env.reset()
        env.positions[0] = torch.tensor([4, 4], device=env.device)  # Center

        masks = env.get_action_masks()

        # All movement actions should be available
        assert masks[0, 0], "UP should be available"
        assert masks[0, 1], "DOWN should be available"
        assert masks[0, 2], "LEFT should be available"
        assert masks[0, 3], "RIGHT should be available"

        # WAIT should always be available
        assert masks[0, 5], "WAIT should be available"

        # INTERACT depends on affordance location (probably masked)
        # Don't assert about INTERACT since it's position-dependent


class TestConfigurableActionCosts:
    """Test configurable action costs."""

    def test_custom_movement_cost(self, device):
        """Custom move_energy_cost should affect all movement actions."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=device,
            move_energy_cost=0.02,  # 2% instead of default 0.5%
        )
        env.reset()
        env.positions[0] = torch.tensor([4, 4], device=env.device)

        initial_energy = env.meters[0, 0].item()

        # Test UP
        actions = torch.tensor([0], device=device)
        env.step(actions)

        energy_after = env.meters[0, 0].item()
        drain = initial_energy - energy_after

        # Should be approximately 2% + 0.5% passive = 2.5%
        expected = 0.025
        assert abs(drain - expected) < 0.001, f"Expected ~{expected}, got {drain}"

    def test_all_movement_actions_use_same_cost(self, device):
        """All four movement actions should use move_energy_cost."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=device,
            move_energy_cost=0.02,
        )

        drains = []
        for action in [0, 1, 2, 3]:  # UP, DOWN, LEFT, RIGHT
            env.reset()
            env.positions[0] = torch.tensor([4, 4], device=env.device)
            initial = env.meters[0, 0].item()

            actions = torch.tensor([action], device=device)
            env.step(actions)

            final = env.meters[0, 0].item()
            drains.append(initial - final)

        # All drains should be equal
        for drain in drains[1:]:
            assert abs(drain - drains[0]) < 1e-6, "All movements should have equal cost"
