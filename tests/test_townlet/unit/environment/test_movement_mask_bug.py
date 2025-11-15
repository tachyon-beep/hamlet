"""Test movement mask bug for dynamic action spaces.

This test suite demonstrates the P1 bug where the hardcoded movement_mask
(actions < 4) incorrectly flags INTERACT/WAIT as movement for substrates
with fewer than 4 movement actions.

Bug Location: src/townlet/environment/vectorized_env.py:595
See: docs/bugs/movement-mask-dynamic-action-spaces.md
"""

import torch


def _action_tensor(env, action_name: str) -> torch.Tensor:
    """Helper to build one-step action tensors on the correct device."""

    action_id = env.action_space.get_action_by_name(action_name).id
    return torch.tensor([action_id], dtype=torch.long, device=env.device)


def test_aspatial_interact_should_not_pay_movement_cost(aspatial_env):
    """Aspatial INTERACT should pay base_depletion + interaction cost (not movement costs).

    Energy costs breakdown (JANK-03):
    - Base depletion: 0.5% (existence cost)
    - Interaction cost: 0.5% (from bars.yaml: base_interaction_cost)
    - Movement cost: 0% (INTERACT is not movement)
    - Total: 1.0%

    This test verifies the movement mask bug is fixed - INTERACT no longer
    flagged as movement for aspatial substrates.
    """
    env = aspatial_env
    env.reset()

    # Remove affordance side-effects so we isolate pure action costs.
    env._handle_interactions = lambda interact_mask: {}

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes INTERACT action (action 0 for aspatial)
    interact_action = _action_tensor(env, "INTERACT")
    env.step(interact_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay base_depletion + base_interaction_cost (from bars.yaml)
    expected_cost = 0.010  # 0.005 base + 0.005 interaction
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"INTERACT should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_aspatial_wait_should_not_pay_movement_cost(aspatial_env):
    """Aspatial WAIT should pay base_depletion only (not movement costs).

    Energy costs breakdown (JANK-03):
    - Base depletion: 0.5% (from bars.yaml - existence cost)
    - Movement cost: 0% (WAIT is not movement)
    - Total: 0.5%

    This test verifies WAIT no longer pays movement costs for aspatial substrates.
    """
    env = aspatial_env
    env.reset()

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes WAIT action (action 1 for aspatial)
    wait_action = _action_tensor(env, "WAIT")
    env.step(wait_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay only base_depletion (WAIT has no additional cost - JANK-03)
    base_depletion = env.base_depletions[env.energy_idx].item()
    expected_cost = base_depletion  # 0.005
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"WAIT should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_1d_interact_should_not_pay_movement_cost(continuous1d_env):
    """1D INTERACT should pay base_depletion + interaction cost (not movement costs).

    Energy costs breakdown (JANK-03):
    - Base depletion: 0.5% (existence cost)
    - Interaction cost: 0.5% (from bars.yaml: base_interaction_cost)
    - Movement cost: 0% (INTERACT is not movement)
    - Total: 1.0%

    This test verifies INTERACT no longer pays movement costs for 1D substrates.
    """
    env = continuous1d_env
    env.reset()

    env._handle_interactions = lambda interact_mask: {}

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes INTERACT action (action 2 for 1D)
    interact_action = _action_tensor(env, "INTERACT")
    env.step(interact_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay base_depletion + base_interaction_cost (from bars.yaml)
    expected_cost = 0.010  # 0.005 base + 0.005 interaction
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"INTERACT should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_1d_wait_should_not_pay_movement_cost(continuous1d_env):
    """1D WAIT should pay base_depletion only (not movement costs).

    Energy costs breakdown (JANK-03):
    - Base depletion: 0.5% (existence cost)
    - Movement cost: 0% (WAIT is not movement)
    - Total: 0.5%

    This test verifies WAIT no longer pays movement costs for 1D substrates.
    """
    env = continuous1d_env
    env.reset()

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes WAIT action (action 3 for 1D)
    wait_action = _action_tensor(env, "WAIT")
    env.step(wait_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay only base_depletion (WAIT has no additional cost - JANK-03)
    expected_cost = 0.005  # base_depletion only
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"WAIT should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_aspatial_hygiene_satiation_only_pay_base_depletion(aspatial_env):
    """Aspatial INTERACT should not drain movement-specific hygiene/satiation penalties.

    Hygiene/satiation costs:
    - Base depletion: 0.3% hygiene, 0.4% satiation (from bars.yaml)
    - Movement penalties: 0% (INTERACT is not movement)

    This test verifies hygiene/satiation only pay base_depletion, not the
    movement-specific penalties (0.3% hygiene, 0.4% satiation).
    """
    env = aspatial_env
    env.reset()

    # Record initial meters
    initial_hygiene = env.meters[0, env.hygiene_idx].item() if env.hygiene_idx is not None else None
    initial_satiation = env.meters[0, env.satiation_idx].item() if env.satiation_idx is not None else None

    # Agent takes INTERACT action
    interact_action = _action_tensor(env, "INTERACT")
    env.step(interact_action)

    # Check hygiene/satiation changes match base_depletion (not movement penalties)
    if env.hygiene_idx is not None:
        final_hygiene = env.meters[0, env.hygiene_idx].item()
        hygiene_cost = initial_hygiene - final_hygiene
        expected_hygiene_cost = 0.003  # base_depletion from bars.yaml
        assert (
            abs(hygiene_cost - expected_hygiene_cost) < 1e-6
        ), f"Hygiene should cost {expected_hygiene_cost:.3%} (base only), but cost {hygiene_cost:.3%}"

    if env.satiation_idx is not None:
        final_satiation = env.meters[0, env.satiation_idx].item()
        satiation_cost = initial_satiation - final_satiation
        expected_satiation_cost = 0.004  # base_depletion from bars.yaml
        assert (
            abs(satiation_cost - expected_satiation_cost) < 1e-6
        ), f"Satiation should cost {expected_satiation_cost:.3%} (base only), but cost {satiation_cost:.3%}"


def test_1d_movement_should_pay_movement_cost(continuous1d_env):
    """1D movement actions (LEFT/RIGHT) should pay both base_depletion and movement cost.

    Energy costs breakdown:
    - Base depletion: 0.5%
    - Movement cost: 0.5% (move_energy_cost)
    - Total: 1.0%

    This verifies actual movement actions are correctly charged, and shows
    the difference between movement (1.0%) and INTERACT (0.5%).
    """
    env = continuous1d_env
    env.reset()

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes LEFT action (action 0 for 1D)
    left_action = _action_tensor(env, "LEFT")
    env.step(left_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay base_depletion + move_energy_cost
    expected_cost = 0.010  # 0.005 base + 0.005 movement
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"LEFT movement should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_3d_interact_should_not_pay_movement_cost(continuous3d_env):
    """3D INTERACT (action 6) should pay base depletion + interaction cost (JANK-03)."""
    env = continuous3d_env
    env.reset()

    # Remove affordance side-effects to isolate action costs
    env._handle_interactions = lambda interact_mask: {}

    initial_energy = env.meters[0, env.energy_idx].item()

    interact_action = _action_tensor(env, "INTERACT")
    env.step(interact_action)

    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    expected_cost = 0.010  # 0.005 base + 0.005 interaction
    assert abs(energy_cost - expected_cost) < 1e-6, f"3D INTERACT should cost {expected_cost:.3%}, but cost {energy_cost:.3%}"


def test_3d_wait_should_not_pay_movement_cost(continuous3d_env):
    """3D WAIT (action 7) should pay base depletion only (JANK-03)."""
    env = continuous3d_env
    env.reset()

    initial_energy = env.meters[0, env.energy_idx].item()

    wait_action = _action_tensor(env, "WAIT")
    env.step(wait_action)

    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    expected_cost = 0.005  # base_depletion only (WAIT has no additional cost)
    assert abs(energy_cost - expected_cost) < 1e-6, f"3D WAIT should cost {expected_cost:.3%}, but cost {energy_cost:.3%}"


def test_3d_vertical_movement_should_pay_movement_cost(continuous3d_env):
    """3D UP_Z (action 4) should pay both base depletion and movement cost.

    Note: Environment applies uniform move_energy_cost to all movement actions.
    The substrate's per-action costs (UP_Z=0.008) are not currently enabled.
    """
    env = continuous3d_env
    env.reset()

    initial_energy = env.meters[0, env.energy_idx].item()

    up_z_action = _action_tensor(env, "UP_Z")
    env.step(up_z_action)

    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    expected_cost = 0.010  # 0.005 base + 0.005 movement (uniform cost)
    assert abs(energy_cost - expected_cost) < 1e-6, f"3D UP_Z should cost {expected_cost:.3%}, but cost {energy_cost:.3%}"
