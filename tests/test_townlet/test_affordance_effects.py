"""
Tests for affordance effects in vectorized_env.py.

Tests affordance interactions that modify agent meters:
- Doctor/Hospital: Health restoration (lines 768-780)
- Therapist: Mood restoration (lines 781-786)
- Park: Free fitness/social/mood (lines 750-762)
- Bar: Social hub with penalties (lines 712-730)
- FastFood: Quick satiation with health costs (lines 694-709)
- Job/Labor: Income generation (lines 670-693)
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def env():
    """Create a simple environment for testing affordance effects."""
    return VectorizedHamletEnv(
        num_agents=3,
        grid_size=8,
        device=torch.device("cpu"),
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


class TestHealthRestorationAffordances:
    """Test Doctor and Hospital health restoration."""

    def test_doctor_restores_health_for_money(self, env):
        """Doctor should restore +25% health for $8."""
        """Doctor should restore +25% health for $8."""
        env.reset()

        # Place agent at Doctor location (5, 1)
        env.positions[0] = env.affordances["Doctor"].clone()

        # Set low health, adequate money
        env.meters[0, 6] = 0.30  # Health 30%
        env.meters[0, 3] = 0.50  # Money $50

        initial_health = env.meters[0, 6].item()
        initial_money = env.meters[0, 3].item()

        # Interact with Doctor
        actions = torch.tensor([4])  # INTERACT action
        env.step(actions)

        # Should restore +25% health
        assert env.meters[0, 6] > initial_health
        assert abs(env.meters[0, 6] - (initial_health + 0.25)) < 0.01

        # Should cost $8
        assert env.meters[0, 3] < initial_money
        assert abs(initial_money - env.meters[0, 3] - 0.08) < 0.01

    def test_hospital_restores_more_health_for_more_money(self, env):
        """Hospital should restore +40% health for $15 (intensive care)."""
        # Place agent at Hospital location
        env.reset()
        """Hospital should restore +40% health for $15 (intensive care)."""
        # Place agent at Hospital location
        hospital_pos = env.affordances["Hospital"]
        env.positions[0] = hospital_pos.clone()

        # Set critical health, adequate money
        env.meters[0, 6] = 0.15  # Health 15% (critical!)
        env.meters[0, 3] = 0.80  # Money $80

        initial_health = env.meters[0, 6].item()
        initial_money = env.meters[0, 3].item()

        # Interact with Hospital
        actions = torch.tensor([4])  # INTERACT action
        env.step(actions)

        # Should restore +40% health (intensive care)
        assert env.meters[0, 6] > initial_health
        assert abs(env.meters[0, 6] - (initial_health + 0.40)) < 0.01

        # Should cost $15 (expensive)
        assert env.meters[0, 3] < initial_money
        assert abs(initial_money - env.meters[0, 3] - 0.15) < 0.01

    def test_health_clamped_at_100_percent(self, env):
        """Health restoration should not exceed 100%."""
        env.reset()
        """Health restoration should not exceed 100%."""
        doctor_pos = env.affordances["Doctor"]
        env.positions[0] = doctor_pos.clone()

        # Already at 90% health
        env.meters[0, 6] = 0.90
        env.meters[0, 3] = 0.50

        # Interact (would restore +25% → 115% without clamp)
        actions = torch.tensor([4])
        env.step(actions)

        # Should clamp at 1.0 (100%)
        assert env.meters[0, 6] <= 1.0
        assert abs(env.meters[0, 6] - 1.0) < 0.01


class TestMoodRestorationAffordances:
    """Test Therapist mood restoration."""

    def test_therapist_restores_mood(self, env):
        """Therapist should restore +40% mood for $15."""
        env.reset()
        """Therapist should restore +40% mood for $15."""
        therapist_pos = env.affordances["Therapist"]
        env.positions[0] = therapist_pos.clone()

        # Set low mood, adequate money
        env.meters[0, 4] = 0.20  # Mood 20% (depressed)
        env.meters[0, 3] = 0.50  # Money $50

        initial_mood = env.meters[0, 4].item()
        initial_money = env.meters[0, 3].item()

        # Interact with Therapist
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +40% mood (intensive therapy)
        assert env.meters[0, 4] > initial_mood
        assert abs(env.meters[0, 4] - (initial_mood + 0.40)) < 0.01

        # Should cost $15
        assert abs(initial_money - env.meters[0, 3] - 0.15) < 0.01


class TestParkAffordance:
    """Test Park - free fitness/social/mood builder."""

    def test_park_is_free(self, env):
        """Park should cost $0 (FREE!)."""
        env.reset()
        """Park should cost $0 (FREE!)."""
        park_pos = env.affordances["Park"]
        env.positions[0] = park_pos.clone()

        initial_money = env.meters[0, 3].item()

        # Interact with Park
        actions = torch.tensor([4])
        env.step(actions)

        # Money should be unchanged (FREE!)
        assert abs(env.meters[0, 3] - initial_money) < 0.01

    def test_park_builds_fitness(self, env):
        """Park should restore +20% fitness."""
        env.reset()
        """Park should restore +20% fitness."""
        park_pos = env.affordances["Park"]
        env.positions[0] = park_pos.clone()

        env.meters[0, 7] = 0.30  # Fitness 30%
        initial_fitness = env.meters[0, 7].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +20% fitness
        assert env.meters[0, 7] > initial_fitness
        assert abs(env.meters[0, 7] - (initial_fitness + 0.20)) < 0.01

    def test_park_builds_social(self, env):
        """Park should restore +15% social."""
        env.reset()
        """Park should restore +15% social."""
        park_pos = env.affordances["Park"]
        env.positions[0] = park_pos.clone()

        env.meters[0, 5] = 0.40  # Social 40%
        initial_social = env.meters[0, 5].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +15% social
        assert env.meters[0, 5] > initial_social
        assert abs(env.meters[0, 5] - (initial_social + 0.15)) < 0.01

    def test_park_improves_mood(self, env):
        """Park should restore +15% mood."""
        env.reset()
        """Park should restore +15% mood."""
        park_pos = env.affordances["Park"]
        env.positions[0] = park_pos.clone()

        env.meters[0, 4] = 0.25  # Mood 25%
        initial_mood = env.meters[0, 4].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +15% mood
        assert env.meters[0, 4] > initial_mood
        assert abs(env.meters[0, 4] - (initial_mood + 0.15)) < 0.01

    def test_park_costs_energy(self, env):
        """Park should cost 15% energy (time/effort)."""
        env.reset()
        """Park should cost 15% energy (time/effort)."""
        park_pos = env.affordances["Park"]
        env.positions[0] = park_pos.clone()

        env.meters[0, 0] = 0.80  # Energy 80%
        initial_energy = env.meters[0, 0].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 15% energy
        assert env.meters[0, 0] < initial_energy
        assert abs(initial_energy - env.meters[0, 0] - 0.15) < 0.01


class TestBarAffordance:
    """Test Bar - social hub with health penalties."""

    def test_bar_best_for_social(self, env):
        """Bar should restore +50% social (BEST in game)."""
        env.reset()
        """Bar should restore +50% social (BEST in game)."""
        bar_pos = env.affordances["Bar"]
        env.positions[0] = bar_pos.clone()

        env.meters[0, 5] = 0.20  # Social 20%
        env.meters[0, 3] = 0.50  # Money for bar
        initial_social = env.meters[0, 5].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +50% social (BEST)
        assert env.meters[0, 5] > initial_social
        assert abs(env.meters[0, 5] - (initial_social + 0.50)) < 0.01

    def test_bar_improves_mood(self, env):
        """Bar should restore +25% mood."""
        env.reset()
        """Bar should restore +25% mood."""
        bar_pos = env.affordances["Bar"]
        env.positions[0] = bar_pos.clone()

        env.meters[0, 4] = 0.30  # Mood 30%
        env.meters[0, 3] = 0.50  # Money
        initial_mood = env.meters[0, 4].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +25% mood
        assert env.meters[0, 4] > initial_mood
        assert abs(env.meters[0, 4] - (initial_mood + 0.25)) < 0.01

    def test_bar_has_health_penalty(self, env):
        """Bar should cost 5% health (late nights, drinking)."""
        env.reset()
        """Bar should cost 5% health (late nights, drinking)."""
        bar_pos = env.affordances["Bar"]
        env.positions[0] = bar_pos.clone()

        env.meters[0, 6] = 0.80  # Health 80%
        env.meters[0, 3] = 0.50  # Money
        initial_health = env.meters[0, 6].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 5% health
        assert env.meters[0, 6] < initial_health
        assert abs(initial_health - env.meters[0, 6] - 0.05) < 0.01

    def test_bar_costs_money(self, env):
        """Bar should cost $15."""
        env.reset()
        """Bar should cost $15."""
        bar_pos = env.affordances["Bar"]
        env.positions[0] = bar_pos.clone()

        env.meters[0, 3] = 0.50  # Money $50
        initial_money = env.meters[0, 3].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost $15
        assert abs(initial_money - env.meters[0, 3] - 0.15) < 0.01


class TestFastFoodAffordance:
    """Test FastFood - quick satiation with health costs."""

    def test_fastfood_restores_satiation(self, env):
        """FastFood should restore +45% satiation."""
        env.reset()
        """FastFood should restore +45% satiation."""
        fastfood_pos = env.affordances["FastFood"]
        env.positions[0] = fastfood_pos.clone()

        env.meters[0, 2] = 0.30  # Satiation 30%
        env.meters[0, 3] = 0.50  # Money
        initial_satiation = env.meters[0, 2].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +45% satiation
        assert env.meters[0, 2] > initial_satiation
        assert abs(env.meters[0, 2] - (initial_satiation + 0.45)) < 0.01

    def test_fastfood_gives_energy_boost(self, env):
        """FastFood should restore +15% energy."""
        env.reset()
        """FastFood should restore +15% energy."""
        fastfood_pos = env.affordances["FastFood"]
        env.positions[0] = fastfood_pos.clone()

        env.meters[0, 0] = 0.40  # Energy 40%
        env.meters[0, 3] = 0.50  # Money
        initial_energy = env.meters[0, 0].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should restore +15% energy (sugar rush)
        assert env.meters[0, 0] > initial_energy
        assert abs(env.meters[0, 0] - (initial_energy + 0.15)) < 0.01

    def test_fastfood_has_fitness_penalty(self, env):
        """FastFood should cost 3% fitness (unhealthy food)."""
        env.reset()
        """FastFood should cost 3% fitness (unhealthy food)."""
        fastfood_pos = env.affordances["FastFood"]
        env.positions[0] = fastfood_pos.clone()

        env.meters[0, 7] = 0.80  # Fitness 80%
        env.meters[0, 3] = 0.50  # Money
        initial_fitness = env.meters[0, 7].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 3% fitness (junk food)
        assert env.meters[0, 7] < initial_fitness
        assert abs(initial_fitness - env.meters[0, 7] - 0.03) < 0.01

    def test_fastfood_has_health_penalty(self, env):
        """FastFood should cost 2% health (junk food)."""
        env.reset()
        """FastFood should cost 2% health (junk food)."""
        fastfood_pos = env.affordances["FastFood"]
        env.positions[0] = fastfood_pos.clone()

        env.meters[0, 6] = 0.80  # Health 80%
        env.meters[0, 3] = 0.50  # Money
        initial_health = env.meters[0, 6].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 2% health (junk food)
        assert env.meters[0, 6] < initial_health
        assert abs(initial_health - env.meters[0, 6] - 0.02) < 0.01


class TestJobAffordances:
    """Test Job and Labor income generation."""

    def test_job_generates_income(self, env):
        """Job should generate $22.50."""
        env.reset()
        """Job should generate $22.50."""
        job_pos = env.affordances["Job"]
        env.positions[0] = job_pos.clone()

        env.meters[0, 3] = 0.20  # Money $20
        initial_money = env.meters[0, 3].item()

        # Interact (work)
        actions = torch.tensor([4])
        env.step(actions)

        # Should generate $22.50 (0.225 in normalized units)
        assert env.meters[0, 3] > initial_money
        assert abs(env.meters[0, 3] - (initial_money + 0.225)) < 0.01

    def test_job_costs_energy(self, env):
        """Job should cost 15% energy (office work)."""
        env.reset()
        """Job should cost 15% energy (office work)."""
        job_pos = env.affordances["Job"]
        env.positions[0] = job_pos.clone()

        env.meters[0, 0] = 0.80  # Energy 80%
        initial_energy = env.meters[0, 0].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 15% energy
        assert env.meters[0, 0] < initial_energy
        assert abs(initial_energy - env.meters[0, 0] - 0.15) < 0.01

    def test_labor_generates_more_income(self, env):
        """Labor should generate $30 (33% more than Job)."""
        env.reset()
        """Labor should generate $30 (33% more than Job)."""
        labor_pos = env.affordances["Labor"]
        env.positions[0] = labor_pos.clone()

        env.meters[0, 3] = 0.20  # Money $20
        initial_money = env.meters[0, 3].item()

        # Interact (work)
        actions = torch.tensor([4])
        env.step(actions)

        # Should generate $30
        assert env.meters[0, 3] > initial_money
        assert abs(env.meters[0, 3] - (initial_money + 0.150)) < 0.01

    def test_labor_costs_more_energy(self, env):
        """Labor should cost 20% energy (exhausting physical work)."""
        env.reset()
        """Labor should cost 20% energy (exhausting physical work)."""
        labor_pos = env.affordances["Labor"]
        env.positions[0] = labor_pos.clone()

        env.meters[0, 0] = 0.80  # Energy 80%
        initial_energy = env.meters[0, 0].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 20% energy (more exhausting than Job)
        assert env.meters[0, 0] < initial_energy
        assert abs(initial_energy - env.meters[0, 0] - 0.20) < 0.01

    def test_labor_has_fitness_penalty(self, env):
        """Labor should cost 5% fitness (physical wear and tear)."""
        env.reset()
        """Labor should cost 5% fitness (physical wear and tear)."""
        labor_pos = env.affordances["Labor"]
        env.positions[0] = labor_pos.clone()

        env.meters[0, 7] = 0.80  # Fitness 80%
        initial_fitness = env.meters[0, 7].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 5% fitness (injury risk)
        assert env.meters[0, 7] < initial_fitness
        assert abs(initial_fitness - env.meters[0, 7] - 0.05) < 0.01

    def test_labor_has_health_penalty(self, env):
        """Labor should cost 5% health (injury risk)."""
        env.reset()
        """Labor should cost 5% health (injury risk)."""
        labor_pos = env.affordances["Labor"]
        env.positions[0] = labor_pos.clone()

        env.meters[0, 6] = 0.80  # Health 80%
        initial_health = env.meters[0, 6].item()

        # Interact
        actions = torch.tensor([4])
        env.step(actions)

        # Should cost 5% health (injury risk)
        assert env.meters[0, 6] < initial_health
        assert abs(initial_health - env.meters[0, 6] - 0.05) < 0.01
