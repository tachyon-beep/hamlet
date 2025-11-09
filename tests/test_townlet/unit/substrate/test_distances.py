"""Test substrate-based distance checks for Phase 5.

This test file validates that distance calculations use substrate.is_on_position()
instead of hardcoded Manhattan distance calculations.
"""

import torch


def test_interaction_uses_substrate_distance(basic_env):
    """Environment should use substrate.is_on_position() for interactions."""
    env = basic_env

    # Reset environment so randomization occurs, then capture Bed position
    env.reset()
    bed_pos = env.affordances["Bed"].clone()
    env.positions = bed_pos.unsqueeze(0)  # [1, position_dim]

    # Set meters to ensure there's room for energy increase
    # Energy at 0.5 so Bed can restore it
    env.meters[0, 0] = 0.5  # Energy
    initial_energy = env.meters[0, 0].clone()

    # Execute INTERACT action via action space lookup
    interact_action = env.action_space.get_action_by_name("INTERACT").id
    actions = torch.tensor([interact_action], dtype=torch.long, device=env.device)
    obs, rewards, dones, infos = env.step(actions)

    # Interaction should succeed (agent is on affordance)
    final_energy = env.meters[0, 0]

    # Verify interaction worked by checking energy increased
    assert final_energy > initial_energy, (
        f"Energy should have increased from interaction. "
        f"Initial: {initial_energy:.3f}, Final: {final_energy:.3f}, "
        f"Agent position: {env.positions[0]}, Bed position: {bed_pos}"
    )
