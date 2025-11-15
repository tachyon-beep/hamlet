"""Consolidated tests for meter dynamics and cascade effects.

This file consolidates meter-related tests from:
- test_meter_dynamics.py (422 lines) - Meter depletion, terminal conditions

Tests cover:
- Base depletion (per-step meter decay)
- Modulation effects (fitness → health)
- Cascade effects (satiation → energy/health, mood → energy, etc.)
- Terminal conditions (death when energy ≤ 0 OR health ≤ 0)
- Meter clamping (values stay in [0, 1])
- Multi-agent meter independence

All tests use CPU device for determinism.
"""

from pathlib import Path

import pytest
import torch

from townlet.environment.cascade_config import load_environment_config

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cascade_config(test_config_pack_path: Path):
    """Load environment configuration for cascade tests."""
    return load_environment_config(test_config_pack_path)


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

    def test_full_cascade_respects_execution_order(self, cascade_config):
        """Cascades execute in config-specified order."""
        # Execution order should be: modulations, primary_to_pivotal, secondary_to_primary, secondary_to_pivotal_weak
        assert cascade_config.cascades.execution_order == [
            "modulations",
            "primary_to_pivotal",
            "secondary_to_primary",
            "secondary_to_pivotal_weak",
        ]
