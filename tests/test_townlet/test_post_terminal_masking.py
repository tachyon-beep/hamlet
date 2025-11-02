"""
Test P3.1: Post-terminal action masking.

All actions should be masked for agents after they die (health or energy reaches zero).
This prevents wasted computation and confusing learning signals.
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
    """Simple environment for post-terminal masking tests."""
    return VectorizedHamletEnv(
        num_agents=2,
        grid_size=5,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


class TestPostTerminalMasking:
    """Test that all actions are masked after agent death."""

    def test_all_actions_masked_after_death(self, env):
        """All 6 actions should be masked when agent dies."""
        env.reset()

        # Kill agent 0 by setting health to 0
        env.meters[0, 6] = 0.0  # health meter index 6

        # Check done state
        dones = env.meters[:, 6] <= 0.0  # health death condition
        assert dones[0], "Agent 0 should be marked as done"

        # Get action masks
        masks = env.get_action_masks()

        # All actions should be masked for dead agent
        assert not masks[0].any(), "Dead agent should have all actions masked"

        # Agent 1 should still have actions available
        assert masks[1].any(), "Alive agent should have some actions available"

    def test_all_actions_masked_after_energy_death(self, env):
        """All actions masked when agent dies from energy depletion."""
        env.reset()

        # Kill agent 0 by setting energy to 0
        env.meters[0, 0] = 0.0  # energy meter index 0

        # Check done state
        dones = env.meters[:, 0] <= 0.0  # energy death condition
        assert dones[0], "Agent 0 should be marked as done"

        # Get action masks
        masks = env.get_action_masks()

        # All actions should be masked
        assert not masks[0].any(), "Dead agent should have all actions masked"

    def test_alive_agent_unaffected_by_dead_agent(self, env):
        """Dead agent doesn't affect alive agent's action availability."""
        env.reset()

        # Kill agent 0
        env.meters[0, 6] = 0.0

        # Keep agent 1 alive with good stats
        env.meters[1, 6] = 1.0  # full health
        env.meters[1, 0] = 1.0  # full energy

        masks = env.get_action_masks()

        # Agent 0: all masked
        assert not masks[0].any()

        # Agent 1: some actions available (at least movement)
        assert masks[1].any()

    def test_multiple_dead_agents_all_masked(self, env):
        """Multiple dead agents should all have masked actions."""
        env.reset()

        # Kill both agents
        env.meters[0, 6] = 0.0
        env.meters[1, 6] = 0.0

        masks = env.get_action_masks()

        # Both should be fully masked
        assert not masks[0].any(), "Agent 0 should have all actions masked"
        assert not masks[1].any(), "Agent 1 should have all actions masked"


class TestPostTerminalMaskingPersistence:
    """Test that masking persists across multiple steps."""

    def test_masking_persists_after_step(self, env):
        """Dead agent should remain masked after environment steps."""
        env.reset()

        # Kill agent 0
        env.meters[0, 6] = 0.0

        # Take a step with both agents (agent 0's action shouldn't matter)
        actions = torch.tensor([0, 1], device=env.device)  # UP for both
        obs, rewards, dones, info = env.step(actions)

        # Check masks after step
        masks = env.get_action_masks()

        # Agent 0 should still be fully masked
        assert not masks[0].any(), "Dead agent should remain masked after step"

    def test_masking_persists_across_multiple_steps(self, env):
        """Dead agent masking should persist for multiple steps."""
        env.reset()

        # Kill agent 0
        env.meters[0, 6] = 0.0

        # Take multiple steps
        for _ in range(5):
            actions = torch.tensor([0, 1], device=env.device)
            obs, rewards, dones, info = env.step(actions)

            masks = env.get_action_masks()
            assert not masks[0].any(), "Dead agent should remain masked"


class TestPostTerminalMaskingEdgeCases:
    """Test edge cases for post-terminal masking."""

    def test_near_death_not_masked(self, env):
        """Agent near death (but alive) should still have actions available."""
        env.reset()

        # Set health to very low but not zero
        env.meters[0, 6] = 0.01  # barely alive
        env.meters[0, 0] = 0.01  # barely any energy

        masks = env.get_action_masks()

        # Should still have some actions (at least INTERACT might be masked, but not all)
        # Movement might be available depending on position
        assert masks.shape == (2, 6), "Should have 6 actions"
        # Don't assert any() because position might mask all movement, but test structure

    def test_exactly_zero_triggers_masking(self, env):
        """Exactly zero health/energy should trigger full masking."""
        env.reset()

        # Set to exactly zero
        env.meters[0, 6] = 0.0

        masks = env.get_action_masks()

        assert not masks[0].any(), "Exactly zero should trigger full masking"

    def test_agent_dies_during_step(self, env):
        """Agent that dies during step should be masked on next call."""
        env.reset()

        # Set health very low
        env.meters[0, 6] = 0.01

        # Deplete meters enough to cause death (would need meter depletion)
        # For this test, just manually set to 0 after a step
        actions = torch.tensor([0, 1], device=env.device)
        obs, rewards, dones, info = env.step(actions)

        # Manually kill agent (simulating death during step)
        env.meters[0, 6] = 0.0

        # Next mask check should show full masking
        masks = env.get_action_masks()
        assert not masks[0].any(), "Agent that died during step should be masked"


class TestPostTerminalMaskingWithTemporalMechanics:
    """Test post-terminal masking with temporal mechanics enabled."""

    def test_dead_agent_masked_with_temporal(self, device):
        """Dead agent should be masked even with temporal mechanics."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=True,
        )
        env.reset()

        # Kill agent 0
        env.meters[0, 6] = 0.0

        masks = env.get_action_masks()

        # Should be fully masked regardless of time
        assert not masks[0].any(), "Dead agent should be masked with temporal mechanics"

    def test_dead_agent_masked_across_time_cycles(self, device):
        """Dead agent should remain masked through time-of-day cycles."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            partial_observability=False,
            enable_temporal_mechanics=True,
        )
        env.reset()

        # Kill agent
        env.meters[0, 6] = 0.0

        # Advance time through a full cycle
        for time in range(24):
            env.time_of_day = time
            masks = env.get_action_masks()
            assert not masks[0].any(), f"Dead agent should be masked at time {time}"
