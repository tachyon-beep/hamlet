"""
Characterization tests for meter dynamics (RED phase baseline).

These tests document the actual behavior of meter depletion and cascade effects
BEFORE extraction, to ensure zero behavioral changes during refactoring.
"""

import torch
import pytest

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestBaseDepletion:
    """Test _deplete_meters() method."""

    def test_base_depletion_rates(self):
        """Base depletion rates applied correctly."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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

    def test_fitness_modulated_health_depletion(self):
        """Health depletion modulated by fitness (gradient approach)."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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

        # After base fitness depletion (0.002), health should be:
        # Agent 0 (fit): 1.0 - 0.0005 = 0.9995
        # Agent 1 (moderate): 1.0 - 0.00175 = 0.99825
        # Agent 2 (unfit): 1.0 - 0.003 = 0.997

        # But fitness also depletes (0.002), so actual fitness values are:
        # Agent 0: 1.0 - 0.002 = 0.998
        # Agent 1: 0.5 - 0.002 = 0.498
        # Agent 2: 0.0 - 0.002 = 0.0 (clamped)

        # Health depletion uses INITIAL fitness values, then clamps
        assert torch.isclose(env.meters[0, 6], torch.tensor(0.9995), atol=1e-4)  # fit
        assert torch.isclose(env.meters[1, 6], torch.tensor(0.99825), atol=1e-4)  # moderate
        assert torch.isclose(env.meters[2, 6], torch.tensor(0.997), atol=1e-4)  # unfit

    def test_clamping_at_zero(self):
        """Meters clamped at 0.0 (no negative values)."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set all meters very low
        env.meters = torch.full((1, 8), 0.001)

        # Run depletion (should clamp at 0.0)
        env.meters = env.meter_dynamics.deplete_meters(env.meters)

        # All should be 0.0 or near-zero (except money which doesn't deplete)
        assert torch.all(env.meters[:, [0, 1, 2, 4, 5, 6, 7]] >= 0.0)
        assert torch.all(env.meters[:, [0, 1, 2, 4, 5, 6, 7]] <= 0.001)


class TestSecondaryCascades:
    """Test _apply_secondary_to_primary_effects() method."""

    def test_low_satiation_affects_both_primaries(self):
        """Low satiation damages both health AND energy (fundamental need)."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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

    def test_low_mood_affects_energy(self):
        """Low mood damages energy (depressed → exhausted)."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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

    def test_high_satiation_no_penalty(self):
        """Satiation above threshold → no penalties."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 2] = 0.8  # satiation at 80% (above threshold)

        initial_health = env.meters[0, 6].item()
        initial_energy = env.meters[0, 0].item()

        env.meters = env.meter_dynamics.apply_secondary_to_primary_effects(env.meters)

        # No change
        assert env.meters[0, 6] == initial_health
        assert env.meters[0, 0] == initial_energy


class TestTertiaryToSecondaryCascades:
    """Test _apply_tertiary_to_secondary_effects() method."""

    def test_low_hygiene_affects_satiation_fitness_mood(self):
        """Low hygiene damages satiation, fitness, and mood."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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

    def test_low_social_affects_mood(self):
        """Low social damages mood."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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


class TestTertiaryToPrimaryCascades:
    """Test _apply_tertiary_to_primary_effects() method."""

    def test_low_hygiene_weak_health_energy_penalty(self):
        """Low hygiene → weak health and energy penalties."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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

    def test_low_social_weak_energy_penalty(self):
        """Low social → weak energy penalty."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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
    """Test _check_dones() method."""

    def test_health_zero_death(self):
        """Health at 0 → death."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 6] = 0.0  # health at 0
        env.dones = torch.tensor([False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0] == True

    def test_energy_zero_death(self):
        """Energy at 0 → death."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 0] = 0.0  # energy at 0
        env.dones = torch.tensor([False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0] == True

    def test_both_primaries_above_zero_alive(self):
        """Both primaries > 0 → alive."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        env.meters = torch.ones(1, 8)
        env.meters[0, 0] = 0.5  # energy at 50%
        env.meters[0, 6] = 0.5  # health at 50%
        env.dones = torch.tensor([False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0] == False

    def test_multi_agent_selective_death(self):
        """Multiple agents: only those with primary=0 die."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        env.meters = torch.ones(3, 8)
        env.meters[0, 6] = 0.0  # Agent 0: health=0 → dead
        env.meters[1, 0] = 0.0  # Agent 1: energy=0 → dead
        # Agent 2: both primaries > 0 → alive
        env.dones = torch.tensor([False, False, False])

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0] == True  # dead (health=0)
        assert env.dones[1] == True  # dead (energy=0)
        assert env.dones[2] == False  # alive


class TestCascadeIntegration:
    """Test full cascade sequence (as called in step())."""

    def test_full_cascade_sequence(self):
        """Test complete cascade: deplete → secondary → tertiary → check_dones."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
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

        # Actual cascade reduction varies by meter
        # Health: ~0.22% (smaller - only tertiary effects)
        # Energy: ~0.76% (larger - hit by base depletion + satiation + mood + hygiene + social)
        assert env.meters[0, 6] > initial_health - 0.003  # health: less than 0.3% reduction
        assert env.meters[0, 0] > initial_energy - 0.008  # energy: less than 0.8% reduction

        # Agent should still be alive (primaries not at 0 yet)
        assert env.dones[0] == False
