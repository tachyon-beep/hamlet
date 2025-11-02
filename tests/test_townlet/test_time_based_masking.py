# tests/test_townlet/test_time_based_masking.py
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_job_closed_outside_business_hours():
    """Verify Job is masked out after 6pm."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([6, 6])  # On Job (default position)
    env.meters[0, 3] = 1.0  # Full money (can afford)

    # Tick 10 (10am): Job is open (8-18)
    env.time_of_day = 10
    masks = env.get_action_masks()
    assert masks[0, 4]  # INTERACT allowed

    # Tick 19 (7pm): Job is closed
    env.time_of_day = 19
    masks = env.get_action_masks()
    assert not masks[0, 4]  # INTERACT blocked


def test_bar_open_after_6pm():
    """Verify Bar opens at 6pm."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([7, 0])  # On Bar (default position)
    env.meters[0, 3] = 1.0

    # Tick 12 (noon): Bar is closed (opens at 18)
    env.time_of_day = 12
    masks = env.get_action_masks()
    assert not masks[0, 4]

    # Tick 20 (8pm): Bar is open
    env.time_of_day = 20
    masks = env.get_action_masks()
    assert masks[0, 4]


def test_bar_wraparound_midnight():
    """Verify Bar hours wrap midnight (18-4)."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()
    env.positions[0] = torch.tensor([7, 0])  # On Bar
    env.meters[0, 3] = 1.0

    # Tick 2 (2am): Bar is still open (wraps to 4am)
    env.time_of_day = 2
    masks = env.get_action_masks()
    assert masks[0, 4]

    # Tick 5 (5am): Bar is closed
    env.time_of_day = 5
    masks = env.get_action_masks()
    assert not masks[0, 4]
