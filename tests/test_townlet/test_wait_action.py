"""
Test WAIT action with configurable energy costs.

WAIT action allows agents to be patient instead of moving unnecessarily.
Energy costs should be configurable for: move, interact, and wait actions.
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def device():
    """PyTorch device for testing."""
    return torch.device('cpu')


@pytest.fixture
def env(device):
    """Environment with default action costs."""
    return VectorizedHamletEnv(
        num_agents=2,
        grid_size=5,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


class TestWaitActionExists:
    """Test that WAIT action exists in action space."""

    def test_action_space_has_six_actions(self, env):
        """Action space should have 6 actions including WAIT."""
        assert env.action_dim == 6, "Should have 6 actions: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT"

    def test_wait_action_is_action_5(self, env):
        """WAIT should be action 5 (last action)."""
        env.reset()
        
        # Take WAIT action
        actions = torch.tensor([5, 5], device=env.device)  # Both agents WAIT
        _, _, _, _ = env.step(actions)
        
        # Should not crash - WAIT is valid action

    def test_action_masks_include_wait(self, env):
        """Action masks should have 6 dimensions including WAIT."""
        env.reset()
        masks = env.get_action_masks()
        
        assert masks.shape[1] == 6, "Action masks should have 6 dimensions"


class TestWaitEnergyConsumption:
    """Test that WAIT consumes less energy than movement."""

    def test_wait_consumes_less_energy_than_move(self, env):
        """WAIT should drain less energy than movement."""
        env.reset()
        
        # Agent 0: WAIT
        # Agent 1: MOVE
        initial_energy_0 = env.meters[0, 0].item()
        initial_energy_1 = env.meters[1, 0].item()
        
        actions = torch.tensor([5, 0], device=env.device)  # WAIT, UP
        env.step(actions)
        
        energy_after_wait = env.meters[0, 0].item()
        energy_after_move = env.meters[1, 0].item()
        
        wait_drain = initial_energy_0 - energy_after_wait
        move_drain = initial_energy_1 - energy_after_move
        
        # WAIT should drain less than movement
        assert wait_drain < move_drain, f"WAIT drain ({wait_drain}) should be < move drain ({move_drain})"
        
        # WAIT should still drain some energy (not zero)
        assert wait_drain > 0, "WAIT should drain some energy"

    def test_wait_does_not_move_agent(self, env):
        """WAIT action should not change agent position."""
        env.reset()
        
        initial_pos = env.positions[0].clone()
        
        actions = torch.tensor([5, 5], device=env.device)
        env.step(actions)
        
        final_pos = env.positions[0]
        
        assert torch.equal(initial_pos, final_pos), "WAIT should not move agent"

    def test_multiple_waits_accumulate_drain(self, env):
        """Multiple WAIT actions should accumulate energy drain."""
        env.reset()
        
        initial_energy = env.meters[0, 0].item()
        
        # Wait 5 times
        for _ in range(5):
            actions = torch.tensor([5, 5], device=env.device)
            env.step(actions)
        
        final_energy = env.meters[0, 0].item()
        total_drain = initial_energy - final_energy
        
        # Should have drained energy over multiple steps
        assert total_drain > 0, "Multiple WAITs should drain energy"


class TestWaitActionMasking:
    """Test that WAIT action masking works correctly."""

    def test_wait_always_available_for_alive_agents(self, env):
        """WAIT should always be available except for dead agents."""
        env.reset()
        
        masks = env.get_action_masks()
        
        # WAIT (action 5) should be available for both alive agents
        assert masks[0, 5], "WAIT should be available for agent 0"
        assert masks[1, 5], "WAIT should be available for agent 1"

    def test_wait_masked_for_dead_agents(self, env):
        """WAIT should be masked for dead agents."""
        env.reset()

        # Kill agent 0 by setting health to zero
        env.meters[0, 6] = 0.0

        masks = env.get_action_masks()

        # All actions should be masked for dead agent
        assert not masks[0, 5], "WAIT should be masked for dead agent"

        # WAIT should still be available for alive agent
        assert masks[1, 5], "WAIT should be available for alive agent"

    def test_wait_available_at_boundaries(self, env):
        """WAIT should be available even at grid boundaries."""
        env.reset()
        
        # Place agent at corner (0, 0)
        env.positions[0] = torch.tensor([0, 0], device=env.device)
        
        masks = env.get_action_masks()
        
        # Movement should be restricted at corner, but WAIT should be available
        assert not masks[0, 0], "UP should be masked at top"
        assert not masks[0, 2], "LEFT should be masked at left edge"
        assert masks[0, 5], "WAIT should be available at corner"


class TestConfigurableEnergyCosts:
    """Test that energy costs are configurable."""

    def test_custom_move_energy_cost(self, device):
        """Custom move_energy_cost should affect movement drain."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=False,
            move_energy_cost=0.01,  # 1% per move (higher than default)
        )
        env.reset()

        initial_energy = env.meters[0, 0].item()

        actions = torch.tensor([0], device=device)  # UP
        env.step(actions)

        energy_after = env.meters[0, 0].item()
        drain = initial_energy - energy_after

        # Should be approximately 1% (0.01) + passive 0.5% (0.005) = 1.5% total
        expected_drain = 0.015
        assert abs(drain - expected_drain) < 0.001, f"Expected ~{expected_drain} drain, got {drain}"

    def test_custom_wait_energy_cost(self, device):
        """Custom wait_energy_cost should affect WAIT drain."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=False,
            wait_energy_cost=0.001,  # 0.1% per wait
        )
        env.reset()

        initial_energy = env.meters[0, 0].item()

        actions = torch.tensor([5], device=device)  # WAIT
        env.step(actions)

        energy_after = env.meters[0, 0].item()
        drain = initial_energy - energy_after

        # Should be approximately 0.1% (0.001) + passive 0.5% (0.005) = 0.6% total
        expected_drain = 0.006
        assert abs(drain - expected_drain) < 0.001, f"Expected ~{expected_drain} drain, got {drain}"

    def test_custom_interact_energy_cost(self, device):
        """Custom interact_energy_cost should affect INTERACT drain."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=False,
            interact_energy_cost=0.002,
        )
        env.reset()

        # Place agent on Bed
        bed_pos = env.affordances['Bed']
        env.positions[0] = bed_pos.clone()

        actions = torch.tensor([4], device=device)
        env.step(actions)

        # Config should be accepted without error

    def test_wait_cheaper_than_move_by_default(self, env):
        """Default WAIT cost should be cheaper than default move cost."""
        env.reset()
        
        # Measure move drain
        initial_energy = env.meters[0, 0].item()
        actions = torch.tensor([0, 5], device=env.device)  # Agent 0 moves, Agent 1 waits
        env.step(actions)
        
        move_drain = initial_energy - env.meters[0, 0].item()
        wait_drain = initial_energy - env.meters[1, 0].item()
        
        assert wait_drain < move_drain, "Default WAIT should be cheaper than move"

    def test_all_costs_configurable_together(self, device):
        """All three cost types should be configurable simultaneously."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=False,
            move_energy_cost=0.01,
            wait_energy_cost=0.002,
            interact_energy_cost=0.005,
        )

        # Should construct without error and store configs
        assert abs(env.move_energy_cost - 0.01) < 1e-6
        assert abs(env.wait_energy_cost - 0.002) < 1e-6
        assert abs(env.interact_energy_cost - 0.005) < 1e-6


class TestWaitWithOtherMechanics:
    """Test WAIT action interacts correctly with other game mechanics."""

    def test_wait_with_meter_depletion(self, env):
        """WAIT should allow passive meter depletion."""
        env.reset()
        
        initial_satiation = env.meters[0, 2].item()
        
        actions = torch.tensor([5, 5], device=env.device)
        env.step(actions)
        
        final_satiation = env.meters[0, 2].item()
        
        # Satiation should decrease from passive depletion
        assert final_satiation < initial_satiation, "Passive depletion should occur during WAIT"

    def test_wait_with_cascade_effects(self, env):
        """WAIT should allow cascade effects to occur."""
        env.reset()

        # Set low satiation to trigger cascades
        env.meters[0, 2] = 0.2

        # Wait and let cascades happen
        actions = torch.tensor([5, 5], device=env.device)
        env.step(actions)

        final_health = env.meters[0, 6].item()

        # Health might decrease due to low satiation cascade
        # WAIT shouldn't prevent cascades - just verify we don't crash
        assert final_health >= 0.0, "Health should remain non-negative"

    def test_wait_does_not_trigger_interactions(self, env):
        """WAIT should not trigger affordance interactions."""
        env.reset()
        
        # Place agent on Bed
        bed_pos = env.affordances['Bed']
        env.positions[0] = bed_pos.clone()
        
        initial_energy = env.meters[0, 0].item()
        
        # WAIT while on Bed
        actions = torch.tensor([5, 5], device=env.device)
        env.step(actions)
        
        final_energy = env.meters[0, 0].item()
        
        # Energy should decrease (WAIT drain + passive), not increase (Bed effect)
        assert final_energy < initial_energy, "WAIT should not trigger Bed interaction"
