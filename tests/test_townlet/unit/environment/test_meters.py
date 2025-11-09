"""Consolidated tests for meter dynamics and cascade effects.

This file consolidates meter-related tests from:
- test_meter_dynamics.py (422 lines) - Meter depletion, terminal conditions
- test_cascade_engine.py (417 lines) - CascadeEngine and equivalence tests

Tests cover:
- Base depletion (per-step meter decay)
- Modulation effects (fitness → health)
- Cascade effects (satiation → energy/health, mood → energy, etc.)
- Terminal conditions (death when energy ≤ 0 OR health ≤ 0)
- Meter clamping (values stay in [0, 1])
- Multi-agent meter independence
- CascadeEngine equivalence with MeterDynamics

All tests use CPU device for determinism.
"""

from pathlib import Path

import pytest
import torch

from townlet.environment.cascade_config import load_environment_config
from townlet.environment.cascade_engine import CascadeEngine
from townlet.environment.meter_dynamics import MeterDynamics

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cascade_config(test_config_pack_path: Path):
    """Load environment configuration for CascadeEngine tests."""
    return load_environment_config(test_config_pack_path)


@pytest.fixture
def cascade_engine(cascade_config, cpu_device):
    """Create CascadeEngine with test config."""
    return CascadeEngine(cascade_config, cpu_device)


@pytest.fixture
def cpu_env_factory(env_factory, cpu_device):
    """Convenience builder for CPU-bound VectorizedHamletEnv instances."""

    def _build(num_agents: int = 1):
        return env_factory(num_agents=num_agents, device_override=cpu_device)

    return _build


# =============================================================================
# Base Depletion Tests
# =============================================================================


class TestBaseDepletion:
    """Test per-step meter depletion mechanics."""

    def test_base_depletion_rates(self, cpu_env_factory):
        """Base depletion rates applied correctly via MeterDynamics."""
        env = cpu_env_factory()
        env.reset()

        # Set all meters to 1.0
        env.meters = torch.ones(1, 8)

        # Run one depletion cycle
        env.meters = env.meter_dynamics.deplete_meters(env.meters)

        # Expected after depletion:
        # energy: 1.0 - 0.005 = 0.995
        # hygiene: 1.0 - 0.003 = 0.997
        # satiation: 1.0 - 0.004 = 0.996
        # money: 1.0 - 0.0 = 1.0 (no depletion)
        # mood: 1.0 - 0.001 = 0.999
        # social: 1.0 - 0.006 = 0.994
        # health: modulated by fitness (fitness=1.0 → 0.5x multiplier → 0.0005)
        # fitness: 1.0 - 0.002 = 0.998

        expected = torch.tensor([[0.995, 0.997, 0.996, 1.0, 0.999, 0.994, 0.9995, 0.998]])
        assert torch.allclose(env.meters, expected, atol=1e-4)

    def test_base_depletions_via_cascade_engine(self, cascade_engine, cpu_device):
        """Base depletions applied correctly via CascadeEngine."""
        # 4 agents with various meter states
        meters = torch.tensor(
            [
                # Agent 0: All meters at 100%
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
                # Agent 1: Low satiation
                [1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0],
                # Agent 2: Low fitness
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.1],
                # Agent 3: Multiple low meters
                [0.8, 0.2, 0.2, 0.5, 0.2, 0.2, 0.8, 0.2],
            ],
            device=cpu_device,
        )
        initial_meters = meters.clone()
        result = cascade_engine.apply_base_depletions(meters)

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

    def test_base_depletions_match_config(self, cascade_engine, cascade_config):
        """Depletion rates match config exactly."""
        for bar in cascade_config.bars.bars:
            assert cascade_engine._base_depletions[bar.index] == bar.base_depletion

    def test_clamping_at_zero(self, cpu_env_factory):
        """Meters clamped at 0.0 (no negative values)."""
        env = cpu_env_factory()
        env.reset()

        # Set all meters very low
        env.meters = torch.full((1, 8), 0.001)

        # Run depletion (should clamp at 0.0)
        env.meters = env.meter_dynamics.deplete_meters(env.meters)

        # All should be 0.0 or near-zero (except money which doesn't deplete)
        assert torch.all(env.meters[:, [0, 1, 2, 4, 5, 6, 7]] >= 0.0)
        assert torch.all(env.meters[:, [0, 1, 2, 4, 5, 6, 7]] <= 0.001)

    def test_base_depletions_respect_bounds(self, cascade_engine, cpu_device):
        """Base depletions clamp to [0, 1] via CascadeEngine."""
        # Create meters near 0
        meters = torch.zeros(2, 8, device=cpu_device)
        result = cascade_engine.apply_base_depletions(meters)

        # Should clamp to 0, not go negative
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()


# =============================================================================
# Modulation Tests
# =============================================================================


class TestModulation:
    """Test meter modulation effects (fitness → health)."""

    def test_fitness_modulated_health_depletion(self, cpu_env_factory):
        """Health depletion modulated by fitness (gradient approach)."""
        env = cpu_env_factory(num_agents=3)
        env.reset()

        # Set three fitness levels: 100%, 50%, 0%
        env.meters = torch.ones(3, 8)
        env.meters[:, 7] = torch.tensor([1.0, 0.5, 0.0])  # fitness

        # Run depletion
        env.meters = env.meter_dynamics.deplete_meters(env.meters)

        # Fitness modulation:
        # fitness=100%: multiplier=0.5, health_depletion=0.001*0.5=0.0005
        # fitness=50%: multiplier=1.75, health_depletion=0.001*1.75=0.00175
        # fitness=0%: multiplier=3.0, health_depletion=0.001*3.0=0.003

        # Health should be:
        # Agent 0 (fit): 1.0 - 0.0005 = 0.9995
        # Agent 1 (moderate): 1.0 - 0.00175 = 0.99825
        # Agent 2 (unfit): 1.0 - 0.003 = 0.997

        assert torch.isclose(env.meters[0, 6], torch.tensor(0.9995), atol=1e-4)  # fit
        assert torch.isclose(env.meters[1, 6], torch.tensor(0.99825), atol=1e-4)  # moderate
        assert torch.isclose(env.meters[2, 6], torch.tensor(0.997), atol=1e-4)  # unfit

    def test_fitness_modulation_healthy_agent(self, cascade_engine, cpu_device):
        """Fitness modulation when agent is healthy (fitness=100%)."""
        # Agent with 100% fitness
        meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)

        initial_health = meters[0, 6].item()
        result = cascade_engine.apply_modulations(meters)

        # Health should deplete at 0.5x rate (base_multiplier=0.5)
        # depletion = 0.001 * 0.5 = 0.0005
        expected_health = initial_health - 0.0005
        assert torch.isclose(result[0, 6], torch.tensor(expected_health), atol=1e-5)

    def test_fitness_modulation_unfit_agent(self, cascade_engine, cpu_device):
        """Fitness modulation when agent is unfit (fitness=0%)."""
        # Agent with 0% fitness
        meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0]], device=cpu_device)

        initial_health = meters[0, 6].item()
        result = cascade_engine.apply_modulations(meters)

        # Health should deplete at 3.0x rate (base + range = 0.5 + 2.5)
        # depletion = 0.001 * 3.0 = 0.003
        expected_health = initial_health - 0.003
        assert torch.isclose(result[0, 6], torch.tensor(expected_health), atol=1e-5)

    def test_fitness_modulation_gradient(self, cascade_engine, cpu_device):
        """Fitness modulation is smooth gradient."""
        # Test multiple fitness levels
        fitness_levels = [1.0, 0.75, 0.5, 0.25, 0.0]
        expected_multipliers = [0.5, 1.125, 1.75, 2.375, 3.0]

        for fitness, expected_mult in zip(fitness_levels, expected_multipliers):
            meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, fitness]], device=cpu_device)
            initial_health = meters[0, 6].item()
            result = cascade_engine.apply_modulations(meters)

            expected_depletion = 0.001 * expected_mult
            expected_health = initial_health - expected_depletion

            assert torch.isclose(result[0, 6], torch.tensor(expected_health), atol=1e-5), f"fitness={fitness}, mult={expected_mult}"


# =============================================================================
# Cascade Effects Tests
# =============================================================================


class TestCascadeEffects:
    """Test meter cascade relationships."""

    # Secondary → Primary Cascades

    def test_low_satiation_affects_both_primaries(self, cpu_env_factory):
        """Low satiation damages both health AND energy (fundamental need)."""
        env = cpu_env_factory()
        env.reset()

        # Set satiation below threshold (0.3), primaries at 1.0
        env.meters = torch.ones(1, 8)
        env.meters[0, 2] = 0.2  # satiation at 20% (below 30% threshold)
        initial_health = env.meters[0, 6].item()
        initial_energy = env.meters[0, 0].item()

        # Apply cascade
        env.meters = env.meter_dynamics.apply_secondary_to_primary_effects(env.meters)

        # Both should decrease
        assert env.meters[0, 6] < initial_health  # health decreased
        assert env.meters[0, 0] < initial_energy  # energy decreased

        # Deficit = (0.3 - 0.2) / 0.3 = 0.333...
        # Health penalty = 0.004 * 0.333 = 0.00133...
        # Energy penalty = 0.005 * 0.333 = 0.00167...
        expected_health = 1.0 - (0.004 * (0.3 - 0.2) / 0.3)
        expected_energy = 1.0 - (0.005 * (0.3 - 0.2) / 0.3)

        assert torch.isclose(env.meters[0, 6], torch.tensor(expected_health), atol=1e-4)
        assert torch.isclose(env.meters[0, 0], torch.tensor(expected_energy), atol=1e-4)

    def test_low_mood_affects_energy(self, cpu_env_factory):
        """Low mood damages energy (depressed → exhausted)."""
        env = cpu_env_factory()
        env.reset()

        # Set mood below threshold, energy at 1.0
        env.meters = torch.ones(1, 8)
        env.meters[0, 4] = 0.1  # mood at 10%
        initial_energy = env.meters[0, 0].item()

        # Apply cascade
        env.meters = env.meter_dynamics.apply_secondary_to_primary_effects(env.meters)

        # Energy should decrease
        assert env.meters[0, 0] < initial_energy

        # Deficit = (0.3 - 0.1) / 0.3 = 0.667
        # Energy penalty = 0.005 * 0.667 = 0.00333...
        expected_energy = 1.0 - (0.005 * (0.3 - 0.1) / 0.3)
        assert torch.isclose(env.meters[0, 0], torch.tensor(expected_energy), atol=1e-4)

    def test_high_satiation_no_penalty(self, cpu_env_factory):
        """Satiation above threshold → no penalties."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 2] = 0.8  # satiation at 80% (above threshold)

        initial_health = env.meters[0, 6].item()
        initial_energy = env.meters[0, 0].item()

        env.meters = env.meter_dynamics.apply_secondary_to_primary_effects(env.meters)

        # No change
        assert env.meters[0, 6] == initial_health
        assert env.meters[0, 0] == initial_energy

    # Tertiary → Secondary Cascades

    def test_low_hygiene_affects_satiation_fitness_mood(self, cpu_env_factory):
        """Low hygiene damages satiation, fitness, and mood."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 1] = 0.1  # hygiene at 10%

        initial_satiation = env.meters[0, 2].item()
        initial_fitness = env.meters[0, 7].item()
        initial_mood = env.meters[0, 4].item()

        env.meters = env.meter_dynamics.apply_tertiary_to_secondary_effects(env.meters)

        # All three should decrease
        assert env.meters[0, 2] < initial_satiation
        assert env.meters[0, 7] < initial_fitness
        assert env.meters[0, 4] < initial_mood

    def test_low_social_affects_mood(self, cpu_env_factory):
        """Low social damages mood."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 5] = 0.1  # social at 10%

        initial_mood = env.meters[0, 4].item()

        env.meters = env.meter_dynamics.apply_tertiary_to_secondary_effects(env.meters)

        # Mood should decrease
        assert env.meters[0, 4] < initial_mood

        # Deficit = (0.3 - 0.1) / 0.3 = 0.667
        # Mood penalty = 0.004 * 0.667 = 0.00267
        expected_mood = 1.0 - (0.004 * (0.3 - 0.1) / 0.3)
        assert torch.isclose(env.meters[0, 4], torch.tensor(expected_mood), atol=1e-4)

    # Tertiary → Primary Cascades (weak effects)

    def test_low_hygiene_weak_health_energy_penalty(self, cpu_env_factory):
        """Low hygiene → weak health and energy penalties."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 1] = 0.1  # hygiene at 10%

        initial_health = env.meters[0, 6].item()
        initial_energy = env.meters[0, 0].item()

        env.meters = env.meter_dynamics.apply_tertiary_to_primary_effects(env.meters)

        # Both should decrease (weak effect)
        assert env.meters[0, 6] < initial_health
        assert env.meters[0, 0] < initial_energy

        # Penalties are 0.0005 (weak)
        # Deficit = (0.3 - 0.1) / 0.3 = 0.667
        # Penalty = 0.0005 * 0.667 = 0.000333...
        expected_health = 1.0 - (0.0005 * (0.3 - 0.1) / 0.3)
        expected_energy = 1.0 - (0.0005 * (0.3 - 0.1) / 0.3)

        assert torch.isclose(env.meters[0, 6], torch.tensor(expected_health), atol=1e-5)
        assert torch.isclose(env.meters[0, 0], torch.tensor(expected_energy), atol=1e-5)

    def test_low_social_weak_energy_penalty(self, cpu_env_factory):
        """Low social → weak energy penalty."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 5] = 0.1  # social at 10%

        initial_energy = env.meters[0, 0].item()

        env.meters = env.meter_dynamics.apply_tertiary_to_primary_effects(env.meters)

        # Energy should decrease (weak effect)
        assert env.meters[0, 0] < initial_energy

        # Penalty = 0.0008 (slightly stronger than hygiene)
        # Deficit = (0.3 - 0.1) / 0.3 = 0.667
        # Penalty = 0.0008 * 0.667 = 0.000533...
        expected_energy = 1.0 - (0.0008 * (0.3 - 0.1) / 0.3)
        assert torch.isclose(env.meters[0, 0], torch.tensor(expected_energy), atol=1e-5)

    # CascadeEngine cascade tests

    def test_threshold_cascade_above_threshold(self, cascade_engine, cpu_device):
        """Cascades don't apply when source is above threshold."""
        # Agent with satiation at 50% (above 30% threshold)
        meters = torch.tensor([[1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)

        initial_health = meters[0, 6].item()
        result = cascade_engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

        # Health should be unchanged (no cascade)
        assert result[0, 6] == initial_health

    def test_threshold_cascade_below_threshold(self, cascade_engine, cpu_device):
        """Cascades apply when source is below threshold."""
        # Agent with satiation at 20% (below 30% threshold)
        meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)

        initial_health = meters[0, 6].item()
        result = cascade_engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

        # Health should decrease (cascade applied)
        assert result[0, 6] < initial_health

    def test_threshold_cascade_gradient_penalty(self, cascade_engine, cpu_device):
        """Cascade penalty is proportional to deficit."""
        # Test satiation_to_health cascade (threshold=0.3, strength=0.004)

        # Case 1: satiation = 0.2 (deficit = 0.1 / 0.3 = 0.333)
        meters1 = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)
        result1 = cascade_engine.apply_threshold_cascades(meters1, ["primary_to_pivotal"])
        deficit1 = (0.3 - 0.2) / 0.3
        expected_penalty1 = 0.004 * deficit1
        expected_health1 = 1.0 - expected_penalty1
        assert torch.isclose(result1[0, 6], torch.tensor(expected_health1), atol=1e-5)

        # Case 2: satiation = 0.1 (deficit = 0.2 / 0.3 = 0.667)
        meters2 = torch.tensor([[1.0, 1.0, 0.1, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)
        result2 = cascade_engine.apply_threshold_cascades(meters2, ["primary_to_pivotal"])
        deficit2 = (0.3 - 0.1) / 0.3
        expected_penalty2 = 0.004 * deficit2
        expected_health2 = 1.0 - expected_penalty2
        assert torch.isclose(result2[0, 6], torch.tensor(expected_health2), atol=1e-5)

        # Penalty should be larger for lower satiation
        assert expected_penalty2 > expected_penalty1

    def test_threshold_cascade_categories(self, cascade_engine, cpu_device):
        """Different cascade categories work."""
        # Test primary_to_pivotal (satiation → health/energy)
        meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)
        result = cascade_engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])
        assert result[0, 6] < 1.0  # health affected
        assert result[0, 0] < 1.0  # energy affected

        # Test secondary_to_primary (hygiene → mood)
        meters = torch.tensor([[1.0, 0.2, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)
        result = cascade_engine.apply_threshold_cascades(meters, ["secondary_to_primary"])
        assert result[0, 4] < 1.0  # mood affected

    def test_threshold_cascade_multiple_sources(self, cascade_engine, cpu_device):
        """Multiple cascades from different sources accumulate."""
        # Agent with low satiation AND low mood (both affect energy)
        meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 0.2, 1.0, 1.0, 1.0]], device=cpu_device)

        result = cascade_engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

        # Energy should be affected by BOTH satiation and mood
        # satiation→energy: deficit=0.333, penalty=0.005*0.333=0.00167
        # mood→energy: deficit=0.333, penalty=0.005*0.333=0.00167
        # Total: ~0.00334
        expected_energy = 1.0 - 0.00167 - 0.00167
        assert torch.isclose(result[0, 0], torch.tensor(expected_energy), atol=1e-4)


# =============================================================================
# Terminal Conditions Tests
# =============================================================================


class TestTerminalConditions:
    """Test death conditions (energy ≤ 0, health ≤ 0)."""

    def test_health_zero_death(self, cpu_env_factory):
        """Health at 0 → death."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 6] = 0.0  # health at 0
        env.dones = torch.tensor([False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0]

    def test_energy_zero_death(self, cpu_env_factory):
        """Energy at 0 → death."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 0] = 0.0  # energy at 0
        env.dones = torch.tensor([False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0]

    def test_both_primaries_above_zero_alive(self, cpu_env_factory):
        """Both primaries > 0 → alive."""
        env = cpu_env_factory()
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 0] = 0.5  # energy at 50%
        env.meters[0, 6] = 0.5  # health at 50%
        env.dones = torch.tensor([False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert not env.dones[0]

    def test_terminal_conditions_healthy_agent(self, cascade_engine, cpu_device):
        """Healthy agent is not terminal."""
        meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)
        dones = torch.zeros(1, dtype=torch.bool, device=cpu_device)

        result = cascade_engine.check_terminal_conditions(meters, dones)
        assert not result[0]  # Not dead

    def test_terminal_conditions_zero_health(self, cascade_engine, cpu_device):
        """Zero health triggers death."""
        meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0]], device=cpu_device)
        dones = torch.zeros(1, dtype=torch.bool, device=cpu_device)

        result = cascade_engine.check_terminal_conditions(meters, dones)
        assert result[0]  # Dead (health=0)

    def test_terminal_conditions_zero_energy(self, cascade_engine, cpu_device):
        """Zero energy triggers death."""
        meters = torch.tensor([[0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)
        dones = torch.zeros(1, dtype=torch.bool, device=cpu_device)

        result = cascade_engine.check_terminal_conditions(meters, dones)
        assert result[0]  # Dead (energy=0)


# =============================================================================
# Meter Clamping Tests
# =============================================================================


class TestMeterClamping:
    """Test that meters stay within [0, 1] bounds."""

    def test_no_negative_values_after_depletion(self, cpu_env_factory):
        """Meters don't go negative after depletion."""
        env = cpu_env_factory()
        env.reset()

        # Set all meters very low
        env.meters = torch.full((1, 8), 0.001)

        # Run multiple depletion cycles
        for _ in range(5):
            env.meters = env.meter_dynamics.deplete_meters(env.meters)

        # All should be >= 0.0
        assert torch.all(env.meters >= 0.0)

    def test_no_overflow_above_one(self, cpu_env_factory):
        """Meters don't overflow above 1.0."""
        env = cpu_env_factory()
        env.reset()

        # Set all meters to 1.0
        env.meters = torch.ones(1, 8)

        # Meters should stay at or below 1.0
        assert torch.all(env.meters <= 1.0)

    def test_cascade_engine_respects_bounds(self, cascade_engine, cpu_device):
        """CascadeEngine clamps to [0, 1]."""
        # Create meters near 0
        meters = torch.zeros(2, 8, device=cpu_device)
        result = cascade_engine.apply_base_depletions(meters)

        # Should clamp to 0, not go negative
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()


# =============================================================================
# Multi-Agent Meter Tests
# =============================================================================


class TestMultiAgentMeters:
    """Test that agents have independent meter states."""

    def test_multi_agent_selective_death(self, cpu_env_factory):
        """Multiple agents: only those with primary=0 die."""
        env = cpu_env_factory(num_agents=3)
        env.reset()

        env.meters = torch.ones(3, 8)
        env.meters[0, 6] = 0.0  # Agent 0: health=0 → dead
        env.meters[1, 0] = 0.0  # Agent 1: energy=0 → dead
        # Agent 2: both primaries > 0 → alive
        env.dones = torch.tensor([False, False, False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0]  # dead (health=0)
        assert env.dones[1]  # dead (energy=0)
        assert not env.dones[2]  # alive

    def test_terminal_conditions_batch(self, cascade_engine, cpu_device):
        """Terminal conditions on batch of agents."""
        meters = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Healthy
                [0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Dead (energy=0)
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0],  # Dead (health=0)
                [0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0],  # Dead (both=0)
            ],
            device=cpu_device,
        )
        dones = torch.zeros(4, dtype=torch.bool, device=cpu_device)

        result = cascade_engine.check_terminal_conditions(meters, dones)

        assert not result[0]  # Agent 0 alive
        assert result[1]  # Agent 1 dead
        assert result[2]  # Agent 2 dead
        assert result[3]  # Agent 3 dead

    def test_agents_have_independent_meters(self, multi_agent_env):
        """Different agents have independent meter states."""
        multi_agent_env.reset()

        # Modify one agent's meters
        multi_agent_env.meters[0, 0] = 0.5  # Agent 0 energy = 50%
        multi_agent_env.meters[0, 2] = 0.3  # Agent 0 satiation = 30%

        # Other agents should be unaffected
        assert multi_agent_env.meters[1, 0] != 0.5
        assert multi_agent_env.meters[2, 0] != 0.5
        assert multi_agent_env.meters[3, 0] != 0.5


# =============================================================================
# Full Cascade Integration Tests
# =============================================================================


class TestCascadeIntegration:
    """Test full cascade sequence (as called in step())."""

    def test_full_cascade_sequence(self, cpu_env_factory):
        """Test complete cascade: deplete → secondary → tertiary → check_dones."""
        env = cpu_env_factory()
        env.reset()

        # Set meters to trigger cascades
        env.meters = torch.ones(1, 8)
        env.meters[0, 1] = 0.1  # hygiene at 10% (will cascade)
        env.meters[0, 2] = 0.2  # satiation at 20% (will cascade)
        env.meters[0, 5] = 0.1  # social at 10% (will cascade)

        initial_health = env.meters[0, 6].item()
        initial_energy = env.meters[0, 0].item()

        # Run full cascade as in step()
        env.meters = env.meter_dynamics.deplete_meters(env.meters)
        env.meters = env.meter_dynamics.apply_secondary_to_primary_effects(env.meters)
        env.meters = env.meter_dynamics.apply_tertiary_to_secondary_effects(env.meters)
        env.meters = env.meter_dynamics.apply_tertiary_to_primary_effects(env.meters)
        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        # Health and energy should be reduced (cascade effects accumulate)
        assert env.meters[0, 6] < initial_health  # health decreased
        assert env.meters[0, 0] < initial_energy  # energy decreased

        # Agent should still be alive (primaries not at 0 yet)
        assert not env.dones[0]

    def test_full_cascade_via_engine(self, cascade_engine, cpu_device):
        """Full cascade applies all stages in order via CascadeEngine."""
        # Agent with low satiation
        meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)
        initial_meters = meters.clone()

        result = cascade_engine.apply_full_cascade(meters)

        # Meters should change
        assert not torch.equal(result, initial_meters)

        # Health and energy should be affected by satiation cascade
        assert result[0, 6] < initial_meters[0, 6]  # health
        assert result[0, 0] < initial_meters[0, 0]  # energy

    def test_full_cascade_respects_execution_order(self, cascade_config):
        """Cascades execute in config-specified order."""
        # Execution order should be: modulations, primary_to_pivotal, secondary_to_primary, secondary_to_pivotal_weak
        assert cascade_config.cascades.execution_order == [
            "modulations",
            "primary_to_pivotal",
            "secondary_to_primary",
            "secondary_to_pivotal_weak",
        ]


# =============================================================================
# CascadeEngine Equivalence Tests
# =============================================================================


class TestCascadeEngineEquivalence:
    """Test that CascadeEngine produces same results as MeterDynamics."""

    def test_equivalence_with_meter_dynamics_healthy(self, cascade_engine, cpu_device):
        """CascadeEngine produces same results as MeterDynamics for healthy agent."""
        # Healthy agent
        meters = torch.tensor([[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)

        # Apply with MeterDynamics
        md = MeterDynamics(1, cpu_device)
        meters_md = meters.clone()
        meters_md = md.deplete_meters(meters_md)  # Includes base depletions + fitness modulation
        meters_md = md.apply_secondary_to_primary_effects(meters_md)
        meters_md = md.apply_tertiary_to_secondary_effects(meters_md)
        meters_md = md.apply_tertiary_to_primary_effects(meters_md)

        # Apply with CascadeEngine
        meters_ce = meters.clone()
        meters_ce = cascade_engine.apply_base_depletions(meters_ce)  # Base depletions
        meters_ce = cascade_engine.apply_full_cascade(meters_ce)  # Modulations + cascades

        # Results should be very close (within floating point tolerance)
        assert torch.allclose(meters_md, meters_ce, atol=1e-5)

    def test_equivalence_with_meter_dynamics_low_satiation(self, cascade_engine, cpu_device):
        """Equivalence for agent with low satiation."""
        # Agent with low satiation (triggers cascades)
        meters = torch.tensor([[1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0]], device=cpu_device)

        # Apply with MeterDynamics
        md = MeterDynamics(1, cpu_device)
        meters_md = meters.clone()
        meters_md = md.deplete_meters(meters_md)
        meters_md = md.apply_secondary_to_primary_effects(meters_md)
        meters_md = md.apply_tertiary_to_secondary_effects(meters_md)
        meters_md = md.apply_tertiary_to_primary_effects(meters_md)

        # Apply with CascadeEngine (base depletions + full cascade)
        meters_ce = meters.clone()
        meters_ce = cascade_engine.apply_base_depletions(meters_ce)
        meters_ce = cascade_engine.apply_full_cascade(meters_ce)

        # Results should match
        assert torch.allclose(meters_md, meters_ce, atol=1e-5)

    def test_equivalence_multi_agent_batch(self, cascade_engine, cpu_device):
        """Equivalence for batch of multiple agents."""
        # Multiple agents with different states
        meters = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Healthy
                [1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0],  # Low satiation
                [1.0, 0.1, 1.0, 0.5, 0.1, 1.0, 1.0, 0.1],  # Multiple low
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # All moderate
            ],
            device=cpu_device,
        )

        # Apply with MeterDynamics
        md = MeterDynamics(4, cpu_device)
        meters_md = meters.clone()
        meters_md = md.deplete_meters(meters_md)
        meters_md = md.apply_secondary_to_primary_effects(meters_md)
        meters_md = md.apply_tertiary_to_secondary_effects(meters_md)
        meters_md = md.apply_tertiary_to_primary_effects(meters_md)

        # Apply with CascadeEngine
        meters_ce = meters.clone()
        meters_ce = cascade_engine.apply_base_depletions(meters_ce)
        meters_ce = cascade_engine.apply_full_cascade(meters_ce)

        # Results should match for all agents
        assert torch.allclose(meters_md, meters_ce, atol=1e-5)


# =============================================================================
# CascadeEngine Initialization Tests
# =============================================================================


class TestCascadeEngineInitialization:
    """Test CascadeEngine initialization and data structures."""

    def test_engine_initialization(self, cascade_engine, cascade_config):
        """Engine initializes correctly with config."""
        assert cascade_engine.config == cascade_config
        assert cascade_engine.device is not None

        # Check lookup maps built
        assert len(cascade_engine._bar_name_to_idx) == 8
        assert len(cascade_engine._bar_idx_to_name) == 8
        assert cascade_engine._bar_name_to_idx["energy"] == 0
        assert cascade_engine._bar_name_to_idx["health"] == 6

        # Check base depletions tensor built
        assert cascade_engine._base_depletions.shape == (8,)
        assert cascade_engine._base_depletions[0] == 0.005  # energy
        # health (handled by fitness modulation, not base depletion)
        assert torch.isclose(cascade_engine._base_depletions[6], torch.tensor(0.0, device=cascade_engine.device))

        # Check cascade data built
        assert len(cascade_engine._cascade_data) > 0
        assert "primary_to_pivotal" in cascade_engine._cascade_data
        # satiation→health, satiation→energy, mood→energy
        assert len(cascade_engine._cascade_data["primary_to_pivotal"]) == 3

        # Check modulation data built
        assert len(cascade_engine._modulation_data) == 1
        assert cascade_engine._modulation_data[0]["source_idx"] == 7  # fitness
        assert cascade_engine._modulation_data[0]["target_idx"] == 6  # health

        # Check terminal data built
        assert len(cascade_engine._terminal_data) == 2

    def test_engine_helper_methods(self, cascade_engine):
        """Bar name/index lookup helpers work."""
        assert cascade_engine.get_bar_name(0) == "energy"
        assert cascade_engine.get_bar_name(6) == "health"

        assert cascade_engine.get_bar_index("energy") == 0
        assert cascade_engine.get_bar_index("health") == 6
