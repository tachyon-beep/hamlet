"""
Tests for CascadeEngine - config-driven cascade system.

Tests:
1. Engine initialization and data structures
2. Base depletion application
3. Modulation effects (fitness → health)
4. Threshold cascades (gradient penalties)
5. Terminal conditions
6. Full cascade sequence
7. Equivalence with hardcoded meter_dynamics.py
"""

from pathlib import Path

import pytest
import torch

from townlet.environment.cascade_config import load_environment_config
from townlet.environment.cascade_engine import CascadeEngine


@pytest.fixture
def device():
    """Get torch device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config():
    """Load default environment configuration."""
    config_dir = Path(__file__).parent.parent.parent / "configs" / "test"
    return load_environment_config(config_dir)


@pytest.fixture
def engine(config, device):
    """Create cascade engine with default config."""
    return CascadeEngine(config, device)


@pytest.fixture
def meters(device):
    """Create sample meter tensor for testing."""
    # 4 agents with various meter states
    return torch.tensor(
        [
            # Agent 0: All meters at 100%
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            # Agent 1: Low satiation (20% - below 30% threshold)
            [1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0],
            # Agent 2: Low fitness (10%)
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.1],
            # Agent 3: Multiple low meters
            [0.8, 0.2, 0.2, 0.5, 0.2, 0.2, 0.8, 0.2],
        ],
        device=device,
    )


# ============================================================================
# Initialization Tests
# ============================================================================


def test_engine_initialization(engine, config):
    """Test that engine initializes correctly with config."""
    assert engine.config == config
    assert engine.device is not None

    # Check lookup maps built
    assert len(engine._bar_name_to_idx) == 8
    assert len(engine._bar_idx_to_name) == 8
    assert engine._bar_name_to_idx["energy"] == 0
    assert engine._bar_name_to_idx["health"] == 6

    # Check base depletions tensor built
    assert engine._base_depletions.shape == (8,)
    assert engine._base_depletions[0] == 0.005  # energy
    # health (handled by fitness modulation, not base depletion)
    assert torch.isclose(engine._base_depletions[6], torch.tensor(0.0, device=engine.device))

    # Check cascade data built
    assert len(engine._cascade_data) > 0
    assert "primary_to_pivotal" in engine._cascade_data
    assert (
        len(engine._cascade_data["primary_to_pivotal"]) == 3
    )  # satiation→health, satiation→energy, mood→energy

    # Check modulation data built
    assert len(engine._modulation_data) == 1
    assert engine._modulation_data[0]["source_idx"] == 7  # fitness
    assert engine._modulation_data[0]["target_idx"] == 6  # health

    # Check terminal data built
    assert len(engine._terminal_data) == 2


def test_engine_helper_methods(engine):
    """Test bar name/index lookup helpers."""
    assert engine.get_bar_name(0) == "energy"
    assert engine.get_bar_name(6) == "health"

    assert engine.get_bar_index("energy") == 0
    assert engine.get_bar_index("health") == 6


# ============================================================================
# Base Depletion Tests
# ============================================================================


def test_base_depletions_applied(engine, meters):
    """Test that base depletions are applied correctly."""
    initial_meters = meters.clone()
    result = engine.apply_base_depletions(meters)

    # Check shape preserved
    assert result.shape == initial_meters.shape

    # Check depletions applied (agent 0, all at 100%)
    assert result[0, 0] < initial_meters[0, 0]  # energy depleted
    # health NOT depleted by base_depletions (handled by fitness modulation)
    assert result[0, 6] == initial_meters[0, 6]

    # Check specific values (agent 0)
    expected_energy = 1.0 - 0.005
    expected_health = 1.0  # No base depletion for health
    assert torch.isclose(result[0, 0], torch.tensor(expected_energy))
    assert torch.isclose(result[0, 6], torch.tensor(expected_health))


def test_base_depletions_respect_bounds(engine, device):
    """Test that base depletions clamp to [0, 1]."""
    # Create meters near 0
    meters = torch.zeros(2, 8, device=device)
    result = engine.apply_base_depletions(meters)

    # Should clamp to 0, not go negative
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


def test_base_depletions_match_config(engine, config):
    """Test that depletion rates match config exactly."""
    for bar in config.bars.bars:
        assert engine._base_depletions[bar.index] == bar.base_depletion


# ============================================================================
# Modulation Tests
# ============================================================================


def test_fitness_modulation_healthy_agent(engine, device):
    """Test fitness modulation when agent is healthy (fitness=100%)."""
    # Agent with 100% fitness
    meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)

    initial_health = meters[0, 6].item()
    result = engine.apply_modulations(meters)

    # Health should deplete at 0.5x rate (base_multiplier=0.5)
    # depletion = 0.001 * 0.5 = 0.0005
    expected_health = initial_health - 0.0005
    assert torch.isclose(result[0, 6], torch.tensor(expected_health), atol=1e-5)


def test_fitness_modulation_unfit_agent(engine, device):
    """Test fitness modulation when agent is unfit (fitness=0%)."""
    # Agent with 0% fitness
    meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0]], device=device)

    initial_health = meters[0, 6].item()
    result = engine.apply_modulations(meters)

    # Health should deplete at 3.0x rate (base + range = 0.5 + 2.5)
    # depletion = 0.001 * 3.0 = 0.003
    expected_health = initial_health - 0.003
    assert torch.isclose(result[0, 6], torch.tensor(expected_health), atol=1e-5)


def test_fitness_modulation_gradient(engine, device):
    """Test that fitness modulation is smooth gradient."""
    # Test multiple fitness levels
    fitness_levels = [1.0, 0.75, 0.5, 0.25, 0.0]
    expected_multipliers = [0.5, 1.125, 1.75, 2.375, 3.0]

    for fitness, expected_mult in zip(fitness_levels, expected_multipliers):
        meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, fitness]], device=device)
        initial_health = meters[0, 6].item()
        result = engine.apply_modulations(meters)

        expected_depletion = 0.001 * expected_mult
        expected_health = initial_health - expected_depletion

        assert torch.isclose(result[0, 6], torch.tensor(expected_health), atol=1e-5), (
            f"fitness={fitness}, mult={expected_mult}"
        )


# ============================================================================
# Threshold Cascade Tests
# ============================================================================


def test_threshold_cascade_above_threshold(engine, device):
    """Test that cascades don't apply when source is above threshold."""
    # Agent with satiation at 50% (above 30% threshold)
    meters = torch.tensor([[1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)

    initial_health = meters[0, 6].item()
    result = engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

    # Health should be unchanged (no cascade)
    assert result[0, 6] == initial_health


def test_threshold_cascade_below_threshold(engine, device):
    """Test that cascades apply when source is below threshold."""
    # Agent with satiation at 20% (below 30% threshold)
    meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)

    initial_health = meters[0, 6].item()
    result = engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

    # Health should decrease (cascade applied)
    assert result[0, 6] < initial_health


def test_threshold_cascade_gradient_penalty(engine, device):
    """Test that cascade penalty is proportional to deficit."""
    # Test satiation_to_health cascade (threshold=0.3, strength=0.004)

    # Case 1: satiation = 0.2 (deficit = 0.1 / 0.3 = 0.333)
    meters1 = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)
    result1 = engine.apply_threshold_cascades(meters1, ["primary_to_pivotal"])
    deficit1 = (0.3 - 0.2) / 0.3
    expected_penalty1 = 0.004 * deficit1
    expected_health1 = 1.0 - expected_penalty1
    assert torch.isclose(result1[0, 6], torch.tensor(expected_health1), atol=1e-5)

    # Case 2: satiation = 0.1 (deficit = 0.2 / 0.3 = 0.667)
    meters2 = torch.tensor([[1.0, 1.0, 0.1, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)
    result2 = engine.apply_threshold_cascades(meters2, ["primary_to_pivotal"])
    deficit2 = (0.3 - 0.1) / 0.3
    expected_penalty2 = 0.004 * deficit2
    expected_health2 = 1.0 - expected_penalty2
    assert torch.isclose(result2[0, 6], torch.tensor(expected_health2), atol=1e-5)

    # Penalty should be larger for lower satiation
    assert expected_penalty2 > expected_penalty1


def test_threshold_cascade_categories(engine, device):
    """Test that different cascade categories work."""
    # Test primary_to_pivotal (satiation → health/energy)
    meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)
    result = engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])
    assert result[0, 6] < 1.0  # health affected
    assert result[0, 0] < 1.0  # energy affected

    # Test secondary_to_primary (hygiene → mood)
    meters = torch.tensor([[1.0, 0.2, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)
    result = engine.apply_threshold_cascades(meters, ["secondary_to_primary"])
    assert result[0, 4] < 1.0  # mood affected


def test_threshold_cascade_multiple_sources(engine, device):
    """Test that multiple cascades from different sources accumulate."""
    # Agent with low satiation AND low mood (both affect energy)
    meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 0.2, 1.0, 1.0, 1.0]], device=device)

    result = engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

    # Energy should be affected by BOTH satiation and mood
    # satiation→energy: deficit=0.333, penalty=0.005*0.333=0.00167
    # mood→energy: deficit=0.333, penalty=0.005*0.333=0.00167
    # Total: ~0.00334
    expected_energy = 1.0 - 0.00167 - 0.00167
    assert torch.isclose(result[0, 0], torch.tensor(expected_energy), atol=1e-4)


# ============================================================================
# Terminal Condition Tests
# ============================================================================


def test_terminal_conditions_healthy_agent(engine, device):
    """Test that healthy agent is not terminal."""
    meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)
    dones = torch.zeros(1, dtype=torch.bool, device=device)

    result = engine.check_terminal_conditions(meters, dones)
    assert not result[0]  # Not dead


def test_terminal_conditions_zero_health(engine, device):
    """Test that zero health triggers death."""
    meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0]], device=device)
    dones = torch.zeros(1, dtype=torch.bool, device=device)

    result = engine.check_terminal_conditions(meters, dones)
    assert result[0]  # Dead (health=0)


def test_terminal_conditions_zero_energy(engine, device):
    """Test that zero energy triggers death."""
    meters = torch.tensor([[0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)
    dones = torch.zeros(1, dtype=torch.bool, device=device)

    result = engine.check_terminal_conditions(meters, dones)
    assert result[0]  # Dead (energy=0)


def test_terminal_conditions_batch(engine, device):
    """Test terminal conditions on batch of agents."""
    meters = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Healthy
            [0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Dead (energy=0)
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0],  # Dead (health=0)
            [0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0],  # Dead (both=0)
        ],
        device=device,
    )
    dones = torch.zeros(4, dtype=torch.bool, device=device)

    result = engine.check_terminal_conditions(meters, dones)

    assert not result[0]  # Agent 0 alive
    assert result[1]  # Agent 1 dead
    assert result[2]  # Agent 2 dead
    assert result[3]  # Agent 3 dead


# ============================================================================
# Full Cascade Sequence Tests
# ============================================================================


def test_full_cascade_sequence(engine, device):
    """Test that full cascade applies all stages in order."""
    # Agent with low satiation
    meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)
    initial_meters = meters.clone()

    result = engine.apply_full_cascade(meters)

    # Meters should change
    assert not torch.equal(result, initial_meters)

    # Health and energy should be affected by satiation cascade
    assert result[0, 6] < initial_meters[0, 6]  # health
    assert result[0, 0] < initial_meters[0, 0]  # energy


def test_full_cascade_respects_execution_order(engine, config):
    """Test that cascades execute in config-specified order."""
    # Execution order should be: modulations, primary_to_pivotal, secondary_to_primary, secondary_to_pivotal_weak
    assert config.cascades.execution_order == [
        "modulations",
        "primary_to_pivotal",
        "secondary_to_primary",
        "secondary_to_pivotal_weak",
    ]


# ============================================================================
# Equivalence Tests with MeterDynamics
# ============================================================================


def test_equivalence_with_meter_dynamics_healthy(engine, device):
    """Test that CascadeEngine produces same results as MeterDynamics for healthy agent."""
    from townlet.environment.meter_dynamics import MeterDynamics

    # Healthy agent
    meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)

    # Apply with MeterDynamics
    md = MeterDynamics(1, device)
    meters_md = meters.clone()
    meters_md = md.deplete_meters(meters_md)  # Includes base depletions + fitness modulation
    meters_md = md.apply_secondary_to_primary_effects(meters_md)
    meters_md = md.apply_tertiary_to_secondary_effects(meters_md)
    meters_md = md.apply_tertiary_to_primary_effects(meters_md)

    # Apply with CascadeEngine
    # Note: deplete_meters() does base depletions + fitness modulation together
    # So we do: base_depletions + apply_full_cascade (which includes modulations)
    meters_ce = meters.clone()
    meters_ce = engine.apply_base_depletions(meters_ce)  # Base depletions (health=0 at this point)
    meters_ce = engine.apply_full_cascade(meters_ce)  # Modulations (fitness→health) + cascades

    # Results should be very close (within floating point tolerance)
    assert torch.allclose(meters_md, meters_ce, atol=1e-5)


def test_equivalence_with_meter_dynamics_low_satiation(engine, device):
    """Test equivalence for agent with low satiation."""
    from townlet.environment.meter_dynamics import MeterDynamics

    # Agent with low satiation (triggers cascades)
    meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=device)

    # Apply with MeterDynamics
    md = MeterDynamics(1, device)
    meters_md = meters.clone()
    meters_md = md.deplete_meters(meters_md)
    meters_md = md.apply_secondary_to_primary_effects(meters_md)
    meters_md = md.apply_tertiary_to_secondary_effects(meters_md)
    meters_md = md.apply_tertiary_to_primary_effects(meters_md)

    # Apply with CascadeEngine (base depletions + full cascade)
    meters_ce = meters.clone()
    meters_ce = engine.apply_base_depletions(meters_ce)
    meters_ce = engine.apply_full_cascade(meters_ce)

    # Results should match
    assert torch.allclose(meters_md, meters_ce, atol=1e-5)
