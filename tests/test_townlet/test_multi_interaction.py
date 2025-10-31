# tests/test_townlet/test_multi_interaction.py
import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def env():
    """Environment with Bed at default position (1,1)."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device('cpu'),
        enable_temporal_mechanics=True,
    )


def test_progressive_benefit_accumulation(env):
    """Verify linear benefits accumulate per tick."""
    env.reset()
    env.meters[0, 0] = 0.3  # Start at 30% energy to avoid clamping
    env.positions[0] = torch.tensor([1, 1])  # On Bed (default position)

    initial_energy = env.meters[0, 0].item()

    # Bed config: 5 ticks, +7.5% energy per tick (linear)
    # But also -0.5% depletion per tick
    # Net per tick: +7.0%
    # First INTERACT
    env.step(torch.tensor([4]))

    energy_after_1 = env.meters[0, 0].item()
    # Expected: +0.075 (benefit) - 0.005 (depletion) + cascading ~= +0.070
    assert abs((energy_after_1 - initial_energy) - 0.070) < 0.01

    # Second INTERACT
    env.step(torch.tensor([4]))

    energy_after_2 = env.meters[0, 0].item()
    assert abs((energy_after_2 - initial_energy) - 0.140) < 0.02


def test_completion_bonus(env):
    """Verify 25% bonus on full completion."""
    env.reset()
    env.meters[0, 0] = 0.3  # Start at 30% energy
    env.meters[0, 6] = 0.7  # Start at 70% health (to see bonus)
    env.positions[0] = torch.tensor([1, 1])

    initial_energy = env.meters[0, 0].item()
    initial_health = env.meters[0, 6].item()

    # Complete all 5 ticks
    for _ in range(5):
        env.step(torch.tensor([4]))

    final_energy = env.meters[0, 0].item()
    final_health = env.meters[0, 6].item()

    # Total energy: 5 × 7.5% (linear) + 12.5% (completion) = 50%
    # Minus 5 × 0.5% depletion = -2.5%
    # Net: ~47.5%, but cascading effects apply
    # Check that energy increased significantly (at least 40%)
    assert (final_energy - initial_energy) > 0.40

    # Health bonus only on completion: +2%
    # (cascading effects from energy/satiation also apply)
    assert (final_health - initial_health) > 0.015


def test_early_exit_keeps_progress(env):
    """Verify agent keeps linear benefits if exiting early."""
    env.reset()
    env.meters[0, 0] = 0.3  # Start at 30% energy
    env.positions[0] = torch.tensor([1, 1])

    initial_energy = env.meters[0, 0].item()

    # Do 3 ticks, then move away
    for _ in range(3):
        env.step(torch.tensor([4]))

    energy_after_3 = env.meters[0, 0].item()

    # Move away (UP action)
    env.step(torch.tensor([0]))

    final_energy = env.meters[0, 0].item()

    # Energy should be at approximately 3 × 7% = 21% gain
    # (3 ticks of benefit minus depletion, no completion bonus)
    assert abs((energy_after_3 - initial_energy) - 0.21) < 0.03


def test_money_charged_per_tick(env):
    """Verify cost charged each tick, not on completion."""
    env.reset()
    env.positions[0] = torch.tensor([1, 1])
    env.meters[0, 3] = 0.50  # Start with $50

    # Bed costs $1/tick = 0.01 normalized
    env.step(torch.tensor([4]))

    money_after_1 = env.meters[0, 3].item()
    assert abs(money_after_1 - 0.49) < 0.001  # $50 - $1 = $49

    env.step(torch.tensor([4]))

    money_after_2 = env.meters[0, 3].item()
    assert abs(money_after_2 - 0.48) < 0.001  # $49 - $1 = $48
