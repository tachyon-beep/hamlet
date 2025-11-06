"""Test movement mask bug for dynamic action spaces.

This test suite demonstrates the P1 bug where the hardcoded movement_mask
(actions < 4) incorrectly flags INTERACT/WAIT as movement for substrates
with fewer than 4 movement actions.

Bug Location: src/townlet/environment/vectorized_env.py:595
See: docs/bugs/movement-mask-dynamic-action-spaces.md
"""

from pathlib import Path

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def aspatial_env(tmp_path):
    """Create aspatial environment for testing."""
    import shutil

    # Create config pack
    config_pack = tmp_path / "aspatial_test"
    config_pack.mkdir()

    # Create aspatial substrate.yaml
    substrate_yaml = config_pack / "substrate.yaml"
    substrate_yaml.write_text(
        """
version: "1.0"
description: "Aspatial substrate for testing action costs"
type: "aspatial"
aspatial: {}
"""
    )

    # Copy complete config files from test config
    test_config = Path("configs/test")
    shutil.copy(test_config / "bars.yaml", config_pack / "bars.yaml")
    shutil.copy(test_config / "affordances.yaml", config_pack / "affordances.yaml")
    shutil.copy(test_config / "cascades.yaml", config_pack / "cascades.yaml")

    # Create environment
    return VectorizedHamletEnv(
        config_pack_path=config_pack,
        num_agents=1,
        grid_size=8,  # Ignored for aspatial
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        enabled_affordances=[],  # No positioned affordances for aspatial
        move_energy_cost=0.005,  # 0.5%
        wait_energy_cost=0.001,  # 0.1%
        interact_energy_cost=0.003,  # 0.3%
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )


@pytest.fixture
def continuous1d_env(tmp_path):
    """Create 1D continuous environment for testing."""
    import shutil

    # Create config pack
    config_pack = tmp_path / "continuous1d_test"
    config_pack.mkdir()

    # Create 1D continuous substrate.yaml
    substrate_yaml = config_pack / "substrate.yaml"
    substrate_yaml.write_text(
        """
version: "1.0"
description: "1D continuous substrate for testing action costs"
type: "continuous"
continuous:
  dimensions: 1
  bounds: [[0.0, 10.0]]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"
  observation_encoding: "relative"
"""
    )

    # Copy complete config files from test config
    test_config = Path("configs/test")
    shutil.copy(test_config / "bars.yaml", config_pack / "bars.yaml")
    shutil.copy(test_config / "affordances.yaml", config_pack / "affordances.yaml")
    shutil.copy(test_config / "cascades.yaml", config_pack / "cascades.yaml")

    # Create environment
    return VectorizedHamletEnv(
        config_pack_path=config_pack,
        num_agents=1,
        grid_size=8,  # Ignored for continuous
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        enabled_affordances=[],  # No affordances for testing
        move_energy_cost=0.005,  # 0.5%
        wait_energy_cost=0.001,  # 0.1%
        interact_energy_cost=0.003,  # 0.3%
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )


@pytest.fixture
def continuous3d_env(tmp_path):
    """Create 3D continuous environment for testing."""
    import shutil

    # Create config pack
    config_pack = tmp_path / "continuous3d_test"
    config_pack.mkdir()

    # Create 3D continuous substrate.yaml
    substrate_yaml = config_pack / "substrate.yaml"
    substrate_yaml.write_text(
        """
version: "1.0"
description: "3D continuous substrate for testing action costs"
type: "continuous"
continuous:
  dimensions: 3
  bounds:
    - [0.0, 10.0]
    - [0.0, 10.0]
    - [0.0, 10.0]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"
  observation_encoding: "relative"
"""
    )

    # Copy complete config files from test config
    test_config = Path("configs/test")
    shutil.copy(test_config / "bars.yaml", config_pack / "bars.yaml")
    shutil.copy(test_config / "affordances.yaml", config_pack / "affordances.yaml")
    shutil.copy(test_config / "cascades.yaml", config_pack / "cascades.yaml")

    # Create environment
    return VectorizedHamletEnv(
        config_pack_path=config_pack,
        num_agents=1,
        grid_size=8,  # Ignored for continuous substrates
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        enabled_affordances=[],  # No affordances for testing
        move_energy_cost=0.005,  # 0.5%
        wait_energy_cost=0.001,  # 0.1%
        interact_energy_cost=0.003,  # 0.3%
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )


def test_aspatial_interact_should_not_pay_movement_cost(aspatial_env):
    """Aspatial INTERACT should pay base_depletion only (not movement-specific costs).

    Energy costs breakdown:
    - Base depletion: 0.5% (from bars.yaml, happens every step)
    - Movement cost: 0% (INTERACT is not movement)
    - Total: 0.5%

    This test verifies the movement mask bug is fixed - INTERACT no longer
    flagged as movement for aspatial substrates.
    """
    env = aspatial_env
    env.reset()

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes INTERACT action (action 0 for aspatial)
    interact_action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))
    env.step(interact_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should only pay base_depletion (0.5%), NOT movement costs
    expected_cost = 0.005  # base_depletion from bars.yaml
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"INTERACT should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_aspatial_wait_should_not_pay_movement_cost(aspatial_env):
    """Aspatial WAIT should pay base_depletion + wait_energy_cost (not movement costs).

    Energy costs breakdown:
    - Base depletion: 0.5% (from bars.yaml)
    - Wait cost: 0.1% (wait_energy_cost)
    - Movement cost: 0% (WAIT is not movement)
    - Total: 0.6%

    This test verifies WAIT no longer pays movement costs for aspatial substrates.
    """
    env = aspatial_env
    env.reset()

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes WAIT action (action 1 for aspatial)
    wait_action = torch.tensor([1], dtype=torch.long, device=torch.device("cpu"))
    env.step(wait_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay base_depletion + wait_cost, NOT movement costs
    expected_cost = 0.006  # 0.005 base + 0.001 wait
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"WAIT should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_1d_interact_should_not_pay_movement_cost(continuous1d_env):
    """1D INTERACT should pay base_depletion only (not movement costs).

    Energy costs breakdown:
    - Base depletion: 0.5%
    - Movement cost: 0% (INTERACT is not movement)
    - Total: 0.5%

    This test verifies INTERACT no longer pays movement costs for 1D substrates.
    """
    env = continuous1d_env
    env.reset()

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes INTERACT action (action 2 for 1D)
    interact_action = torch.tensor([2], dtype=torch.long, device=torch.device("cpu"))
    env.step(interact_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should only pay base_depletion, NOT movement costs
    expected_cost = 0.005  # base_depletion
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"INTERACT should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_1d_wait_should_not_pay_movement_cost(continuous1d_env):
    """1D WAIT should pay base_depletion + wait_energy_cost (not movement costs).

    Energy costs breakdown:
    - Base depletion: 0.5%
    - Wait cost: 0.1%
    - Movement cost: 0% (WAIT is not movement)
    - Total: 0.6%

    This test verifies WAIT no longer pays movement costs for 1D substrates.
    """
    env = continuous1d_env
    env.reset()

    # Record initial energy
    initial_energy = env.meters[0, env.energy_idx].item()

    # Agent takes WAIT action (action 3 for 1D)
    wait_action = torch.tensor([3], dtype=torch.long, device=torch.device("cpu"))
    env.step(wait_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay base_depletion + wait_cost, NOT movement costs
    expected_cost = 0.006  # 0.005 base + 0.001 wait
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
    interact_action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))
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
    left_action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))
    env.step(left_action)

    # Check energy cost
    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    # Should pay base_depletion + move_energy_cost
    expected_cost = 0.010  # 0.005 base + 0.005 movement
    actual_cost = energy_cost

    assert abs(actual_cost - expected_cost) < 1e-6, f"LEFT movement should cost {expected_cost:.3%}, but cost {actual_cost:.3%}"


def test_3d_interact_should_not_pay_movement_cost(continuous3d_env):
    """3D INTERACT (action 6) should only pay base depletion."""
    env = continuous3d_env
    env.reset()

    initial_energy = env.meters[0, env.energy_idx].item()

    interact_action = torch.tensor([6], dtype=torch.long, device=torch.device("cpu"))
    env.step(interact_action)

    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    expected_cost = 0.005  # base_depletion
    assert abs(energy_cost - expected_cost) < 1e-6, f"3D INTERACT should cost {expected_cost:.3%}, but cost {energy_cost:.3%}"


def test_3d_wait_should_not_pay_movement_cost(continuous3d_env):
    """3D WAIT (action 7) should pay base depletion + wait cost only."""
    env = continuous3d_env
    env.reset()

    initial_energy = env.meters[0, env.energy_idx].item()

    wait_action = torch.tensor([7], dtype=torch.long, device=torch.device("cpu"))
    env.step(wait_action)

    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    expected_cost = 0.006  # 0.005 base + 0.001 wait
    assert abs(energy_cost - expected_cost) < 1e-6, f"3D WAIT should cost {expected_cost:.3%}, but cost {energy_cost:.3%}"


def test_3d_vertical_movement_should_pay_movement_cost(continuous3d_env):
    """3D UP_Z (action 4) should pay both base depletion and movement cost.

    Note: Environment applies uniform move_energy_cost to all movement actions.
    The substrate's per-action costs (UP_Z=0.008) are not currently enabled.
    """
    env = continuous3d_env
    env.reset()

    initial_energy = env.meters[0, env.energy_idx].item()

    up_z_action = torch.tensor([4], dtype=torch.long, device=torch.device("cpu"))
    env.step(up_z_action)

    final_energy = env.meters[0, env.energy_idx].item()
    energy_cost = initial_energy - final_energy

    expected_cost = 0.010  # 0.005 base + 0.005 movement (uniform cost)
    assert abs(energy_cost - expected_cost) < 1e-6, f"3D UP_Z should cost {expected_cost:.3%}, but cost {energy_cost:.3%}"
