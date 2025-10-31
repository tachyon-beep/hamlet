# tests/test_townlet/test_temporal_integration.py
"""Integration tests for temporal mechanics system."""

import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_full_24_hour_cycle():
    """Verify 24-hour cycle completes and wraps correctly."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()
    assert env.time_of_day == 0

    # Step through 24 hours
    for expected_time in range(24):
        assert env.time_of_day == expected_time
        env.step(torch.tensor([0]))  # UP action

    # Should wrap back to 0
    assert env.time_of_day == 0


def test_observation_dimensions_with_temporal():
    """Verify observation includes temporal features."""
    env = VectorizedHamletEnv(
        num_agents=2,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    obs = env.reset()

    # Full observability: 64 (grid) + 8 (meters) + (num_affordance_types + 1) + 2 (temporal)
    # num_affordance_types = 15 (including CoffeeShop), encoding = 16
    expected_dim = 64 + 8 + (env.num_affordance_types + 1) + 2
    assert obs.shape == (2, expected_dim)

    # Last two features are time_of_day and interaction_progress
    time_feature = obs[0, -2]
    progress_feature = obs[0, -1]

    assert 0.0 <= time_feature <= 1.0
    assert progress_feature == 0.0  # No interaction yet


def test_multi_tick_job_completion():
    """Verify Job completion over 4 ticks with money gain."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([6, 6])  # On Job
    env.meters[0, 3] = 0.5  # Start with $50
    env.time_of_day = 10  # 10am (Job open 8-18)

    initial_money = env.meters[0, 3].item()

    # Complete 4 ticks of Job
    for i in range(4):
        env.step(torch.tensor([4]))  # INTERACT
        # Progress: 1, 2, 3, then 0 (completes on 4th)
        if i < 3:
            assert env.interaction_progress[0] == (i + 1)
        else:
            assert env.interaction_progress[0] == 0  # Completed

    final_money = env.meters[0, 3].item()

    # Job pays: 4 × $4.21875 (linear) + $5.625 (completion) = ~$22.5 total
    # Normalized: +0.225
    assert (final_money - initial_money) > 0.19  # At least 19% gain (accounting for time)


def test_operating_hours_mask_job():
    """Verify Job is masked out after 6pm."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([6, 6])  # On Job
    env.meters[0, 3] = 1.0

    # 10am: Job open
    env.time_of_day = 10
    masks = env.get_action_masks()
    assert masks[0, 4] == True  # INTERACT allowed

    # 7pm: Job closed
    env.time_of_day = 19
    masks = env.get_action_masks()
    assert masks[0, 4] == False  # INTERACT blocked


def test_early_exit_from_interaction():
    """Verify agent can exit early and keep linear benefits."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.meters[0, 0] = 0.3  # Low energy
    env.positions[0] = torch.tensor([1, 1])  # On Bed

    initial_energy = env.meters[0, 0].item()

    # Do 2 ticks of Bed (requires 5 for completion)
    env.step(torch.tensor([4]))
    env.step(torch.tensor([4]))
    assert env.interaction_progress[0] == 2

    energy_after_2 = env.meters[0, 0].item()

    # Move away
    env.step(torch.tensor([0]))  # UP
    assert env.interaction_progress[0] == 0  # Progress reset
    assert env.last_interaction_affordance[0] is None

    # Energy should have increased from 2 ticks
    assert (energy_after_2 - initial_energy) > 0.1


def test_temporal_mechanics_disabled_fallback():
    """Verify environment works without temporal mechanics (legacy mode)."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=False,  # Legacy mode
    )

    obs = env.reset()

    # Without temporal: 64 (grid) + 8 (meters) + 15 (affordance) = 87
    assert obs.shape == (1, 87)

    # No time tracking
    assert not hasattr(env, "time_of_day")

    # Interactions work (legacy single-shot mode)
    env.positions[0] = torch.tensor([1, 1])  # On Bed
    env.meters[0, 0] = 0.3  # Start low to see increase

    initial_energy = env.meters[0, 0].item()

    env.step(torch.tensor([4]))  # INTERACT

    final_energy = env.meters[0, 0].item()
    # Legacy mode: single-shot benefit (+50% energy from Bed)
    # Even with depletion, should see significant increase
    assert (final_energy - initial_energy) > 0.4  # At least 40% gain
