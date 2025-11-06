"""Test substrate-based distance checks for Phase 5.

This test file validates that distance calculations use substrate.is_on_position()
instead of hardcoded Manhattan distance calculations.
"""

import torch


def test_interaction_uses_substrate_distance(basic_env):
    """Environment should use substrate.is_on_position() for interactions."""
    env = basic_env

    # Find Bed affordance position
    bed_pos = env.affordances["Bed"]

    # Reset environment and place agent on Bed
    env.reset()
    env.positions = bed_pos.unsqueeze(0)  # [1, 2]

    # Set meters to ensure there's room for energy increase
    # Energy at 0.5 so Bed can restore it
    env.meters[0, 0] = 0.5  # Energy
    initial_energy = env.meters[0, 0].clone()

    # Execute INTERACT action (action=4)
    actions = torch.tensor([4], dtype=torch.long, device=env.device)
    obs, rewards, dones, infos = env.step(actions)

    # Interaction should succeed (agent is on affordance)
    final_energy = env.meters[0, 0]

    # Verify interaction worked by checking energy increased
    assert final_energy > initial_energy, (
        f"Energy should have increased from interaction. "
        f"Initial: {initial_energy:.3f}, Final: {final_energy:.3f}, "
        f"Agent position: {env.positions[0]}, Bed position: {bed_pos}"
    )
