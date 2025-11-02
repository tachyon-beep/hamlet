"""
Tests for P1.4: INTERACT de-masking (Remove affordability gating).

INTERACT should be available when on an affordance, regardless of money.
Affordability check happens inside interaction handler, not in masking.

This enables economic planning: agents can learn to visit Job before Hospital.
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
    """Standard environment with 1 agent."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


@pytest.fixture
def env_temporal(device):
    """Environment with temporal mechanics enabled for testing closed affordances."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=True,
    )


class TestInteractDemasking:
    """Test that INTERACT is available on affordances regardless of money."""

    def test_interact_available_on_bed_when_broke(self, env, device):
        """INTERACT should be available on Bed even with $0."""
        env.reset()

        # Place on Bed with $0
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()
        env.meters[0, 3] = 0.0  # Money = 0

        masks = env.get_action_masks()

        assert masks[0, 4], "INTERACT should be available on Bed even when broke"

    def test_interact_available_on_hospital_when_broke(self, env, device):
        """INTERACT should be available on Hospital even with $0."""
        env.reset()

        # Place on Hospital with $0
        hospital_pos = env.affordances["Hospital"]
        env.positions[0] = hospital_pos.clone()
        env.meters[0, 3] = 0.0  # Money = 0

        masks = env.get_action_masks()

        assert masks[0, 4], "INTERACT should be available on Hospital even when broke"

    def test_interact_available_on_job_always(self, env, device):
        """INTERACT should be available on Job (generates money)."""
        env.reset()

        # Place on Job
        job_pos = env.affordances["Job"]
        env.positions[0] = job_pos.clone()
        env.meters[0, 3] = 0.0  # Money = 0

        masks = env.get_action_masks()

        assert masks[0, 4], "INTERACT should be available on Job"

    def test_interact_unavailable_off_affordance(self, env, device):
        """INTERACT should still be masked when not on any affordance."""
        env.reset()

        # Place on empty cell (not on any affordance)
        env.positions[0] = torch.tensor([4, 4], device=device)

        # Ensure not on any affordance
        on_any_affordance = False
        for affordance_name, affordance_pos in env.affordances.items():
            if torch.equal(env.positions[0], affordance_pos):
                on_any_affordance = True
                break

        if not on_any_affordance:
            masks = env.get_action_masks()
            assert not masks[0, 4], "INTERACT should be masked when not on affordance"

    def test_interact_no_effect_when_broke(self, env, device):
        """INTERACT on expensive affordance should have no effect when broke."""
        env.reset()

        # Place on Hospital with $0
        hospital_pos = env.affordances["Hospital"]
        env.positions[0] = hospital_pos.clone()
        env.meters[0, 3] = 0.0  # Money = 0
        env.meters[0, 6] = 0.5  # Health = 50%

        initial_health = env.meters[0, 6].item()

        # Try to interact (should be allowed but have no effect)
        actions = torch.tensor([4], device=device)  # INTERACT
        env.step(actions)

        final_health = env.meters[0, 6].item()

        # Health should not increase (can't afford)
        # May decrease due to passive depletion
        assert final_health <= initial_health + 0.01, "Hospital should have no effect when broke"

    def test_interact_works_when_affordable(self, env, device):
        """INTERACT on Hospital should work when affordable."""
        env.reset()

        # Place on Hospital with money
        hospital_pos = env.affordances["Hospital"]
        env.positions[0] = hospital_pos.clone()
        env.meters[0, 3] = 1.0  # Money = 100%
        env.meters[0, 6] = 0.3  # Health = 30% (low)

        initial_health = env.meters[0, 6].item()

        # Interact (should work)
        actions = torch.tensor([4], device=device)  # INTERACT
        env.step(actions)

        final_health = env.meters[0, 6].item()

        # Health should increase
        assert final_health > initial_health, "Hospital should restore health when affordable"


class TestInteractTimeCosts:
    """Test that attempting INTERACT when broke still consumes time (passive decay)."""

    def test_broke_interact_consumes_time(self, env, device):
        """Attempting INTERACT when broke should still consume a timestep."""
        env.reset()

        # Place on Hospital with $0
        hospital_pos = env.affordances["Hospital"]
        env.positions[0] = hospital_pos.clone()
        env.meters[0, 3] = 0.0  # Money = 0
        env.meters[0, 0] = 1.0  # Energy = 100%
        env.meters[0, 2] = 1.0  # Satiation = 100%

        initial_energy = env.meters[0, 0].item()
        initial_satiation = env.meters[0, 2].item()

        # Try to interact
        actions = torch.tensor([4], device=device)  # INTERACT
        env.step(actions)

        final_energy = env.meters[0, 0].item()
        final_satiation = env.meters[0, 2].item()

        # Passive decay should have occurred
        assert final_energy < initial_energy, "Passive energy decay should occur even on failed interact"
        assert final_satiation < initial_satiation, "Passive satiation decay should occur even on failed interact"


class TestInteractMaskingStillWorksForPhysical:
    """Test that INTERACT is still masked for physically impossible cases."""

    def test_interact_still_masked_at_closed_affordance(self, env_temporal, device):
        """INTERACT should be masked at closed affordances (temporal mechanics)."""
        env_temporal.reset()

        # Place on Job at night (closed - Job operates 8am-6pm)
        job_pos = env_temporal.affordances["Job"]
        env_temporal.positions[0] = job_pos.clone()
        env_temporal.time_of_day = 20  # 8pm (Job closed)

        masks = env_temporal.get_action_masks()

        # INTERACT should be masked (closed)
        assert not masks[0, 4], "INTERACT should be masked at closed affordance"

        # Verify time_of_day is exposed in observation
        obs = env_temporal._get_observations()
        # Observation should include time_of_day as last 2 features (time, interaction_progress)
        assert obs.shape[1] > env_temporal.grid_size * env_temporal.grid_size + 8, (
            "Temporal observations should include time_of_day and interaction_progress"
        )

    def test_interact_masked_for_dead_agent(self, env, device):
        """INTERACT should be masked for dead agents."""
        env.reset()

        # Place on Bed but kill agent
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos.clone()
        env.meters[0, 6] = 0.0  # Health = 0 (dead)

        masks = env.get_action_masks()

        # All actions should be masked (dead)
        assert not torch.any(masks[0]), "All actions should be masked for dead agent"
