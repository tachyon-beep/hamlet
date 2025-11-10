"""Consolidated tests for affordance interactions.

This file consolidates environment-level affordance mechanics from:
- test_affordance_engine.py: Operating hours, cost application, multi-tick mechanics
- test_affordance_effects.py: All instant affordance effects
- test_affordance_equivalence.py: Equivalence validation (75%/25% split)
- test_affordance_integration.py: Batching tests (integration tests excluded)

Test Coverage:
1. Affordance availability (operating hours, affordability checks)
2. Instant affordance effects (meter changes, costs)
3. Multi-tick interactions (completion, early exit, per-tick effects)
4. Batch processing (multiple agents, simultaneous interactions)
5. Equivalence validation (instant vs multi-tick correctness)

Old files consolidated: test_affordance_engine.py, test_affordance_effects.py,
test_affordance_equivalence.py, test_affordance_integration.py
"""

import shutil
from pathlib import Path

import pytest
import torch
import yaml


class TestAffordanceAvailability:
    """Test affordance availability checks (operating hours, affordability)."""

    def test_job_business_hours_only(self, basic_env):
        """Job should only be available 8am-6pm."""
        basic_env.reset()

        # Job operating hours: [8, 18] (8am-6pm)
        engine = basic_env.affordance_engine

        # 6am - CLOSED
        assert not engine.is_affordance_open("Job", time_of_day=6)

        # 8am - OPEN
        assert engine.is_affordance_open("Job", time_of_day=8)

        # 12pm - OPEN
        assert engine.is_affordance_open("Job", time_of_day=12)

        # 6pm (18:00) - CLOSED (exclusive end)
        assert not engine.is_affordance_open("Job", time_of_day=18)

        # 10pm - CLOSED
        assert not engine.is_affordance_open("Job", time_of_day=22)

    def test_bar_midnight_wraparound(self, basic_env):
        """Bar operating hours should wrap around midnight (6pm-4am)."""
        basic_env.reset()
        engine = basic_env.affordance_engine

        # Bar operating hours: [18, 28] (6pm to 4am, wraps midnight)

        # 4pm - CLOSED
        assert not engine.is_affordance_open("Bar", time_of_day=16)

        # 6pm - OPEN
        assert engine.is_affordance_open("Bar", time_of_day=18)

        # 11pm - OPEN
        assert engine.is_affordance_open("Bar", time_of_day=23)

        # Midnight (0) - OPEN (wrapped)
        assert engine.is_affordance_open("Bar", time_of_day=0)

        # 2am - OPEN (wrapped)
        assert engine.is_affordance_open("Bar", time_of_day=2)

        # 4am - CLOSED (exclusive end: 28 % 24 = 4)
        assert not engine.is_affordance_open("Bar", time_of_day=4)

        # 5am - CLOSED
        assert not engine.is_affordance_open("Bar", time_of_day=5)

    def test_bed_always_open(self, basic_env):
        """Bed should be available 24/7."""
        basic_env.reset()
        engine = basic_env.affordance_engine

        # Bed operating hours: [0, 24]
        for hour in range(24):
            assert engine.is_affordance_open("Bed", time_of_day=hour), f"Bed should be open at {hour}:00"

    def test_insufficient_money_blocks_interaction(self, basic_env):
        """Insufficient money should prevent interaction."""
        basic_env.reset()

        # Agent with no money at Shower
        basic_env.positions[0] = basic_env.affordances["Shower"]
        basic_env.meters[0, 1] = 0.2  # Low hygiene
        basic_env.meters[0, 3] = 0.00  # No money (Shower costs $3)

        initial_hygiene = basic_env.meters[0, 1].item()
        initial_money = basic_env.meters[0, 3].item()

        # Try to interact (should be blocked by affordability check)
        actions = torch.tensor([4], device=basic_env.device)  # INTERACT
        basic_env.step(actions)

        # Hygiene should NOT increase (interaction blocked)
        # Note: passive depletion still occurs
        assert basic_env.meters[0, 1] <= initial_hygiene, "Hygiene should not increase without money"
        assert abs(basic_env.meters[0, 3] - initial_money) < 1e-6, "Money should not change"

    def test_sufficient_money_allows_interaction(self, instant_env):
        """Sufficient money should allow interaction."""
        instant_env.reset()

        # Agent with money at Shower
        instant_env.positions[0] = instant_env.affordances["Shower"]
        instant_env.meters[0, 1] = 0.2  # Low hygiene
        instant_env.meters[0, 3] = 0.50  # Has money

        initial_hygiene = instant_env.meters[0, 1].item()
        initial_money = instant_env.meters[0, 3].item()

        # Interact
        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Hygiene should increase, money should decrease
        assert instant_env.meters[0, 1] > initial_hygiene, "Hygiene should increase with money"
        assert instant_env.meters[0, 3] < initial_money, "Money should decrease"

    def test_park_is_free(self, instant_env):
        """Park should work even with $0 (free affordance)."""
        instant_env.reset()

        # Agent at Park with NO money
        instant_env.positions[0] = instant_env.affordances["Park"]
        instant_env.meters[0, 7] = 0.3  # Low fitness
        instant_env.meters[0, 3] = 0.00  # $0 (Park is FREE!)

        initial_fitness = instant_env.meters[0, 7].item()
        initial_money = instant_env.meters[0, 3].item()

        # Interact
        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Fitness should increase, money unchanged
        assert instant_env.meters[0, 7] > initial_fitness, "Park should work for free"
        assert abs(instant_env.meters[0, 3] - initial_money) < 1e-6, "Money should not change (Park is free)"


class TestInstantAffordanceEffects:
    """Test instant affordance effects on meters and resources."""

    # Health Restoration Affordances

    def test_doctor_restores_health(self, instant_env):
        """Doctor should restore +25% health for $8."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Doctor"]
        instant_env.meters[0, 6] = 0.30  # Health 30%
        instant_env.meters[0, 3] = 0.50  # Money $50

        initial_health = instant_env.meters[0, 6].item()
        initial_money = instant_env.meters[0, 3].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Should restore health and cost money
        assert instant_env.meters[0, 6] > initial_health
        assert abs(instant_env.meters[0, 6] - (initial_health + 0.25)) < 0.01
        assert instant_env.meters[0, 3] < initial_money
        assert abs(initial_money - instant_env.meters[0, 3] - 0.08) < 0.01

    def test_hospital_restores_more_health(self, instant_env):
        """Hospital should restore +40% health for $15 (intensive care)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Hospital"]
        instant_env.meters[0, 6] = 0.15  # Critical health
        instant_env.meters[0, 3] = 0.80  # Money

        initial_health = instant_env.meters[0, 6].item()
        initial_money = instant_env.meters[0, 3].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Should restore more health for higher cost
        assert instant_env.meters[0, 6] > initial_health
        assert abs(instant_env.meters[0, 6] - (initial_health + 0.40)) < 0.01
        assert abs(initial_money - instant_env.meters[0, 3] - 0.15) < 0.01

    def test_health_clamped_at_100_percent(self, instant_env):
        """Health restoration should not exceed 100%."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Doctor"]
        instant_env.meters[0, 6] = 0.90  # Already high
        instant_env.meters[0, 3] = 0.50

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Should clamp at 1.0
        assert instant_env.meters[0, 6] <= 1.0
        assert abs(instant_env.meters[0, 6] - 1.0) < 0.01

    # Mood Restoration

    def test_therapist_restores_mood(self, instant_env):
        """Therapist should restore +40% mood for $15."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Therapist"]
        instant_env.meters[0, 4] = 0.20  # Low mood
        instant_env.meters[0, 3] = 0.50

        initial_mood = instant_env.meters[0, 4].item()
        initial_money = instant_env.meters[0, 3].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 4] > initial_mood
        assert abs(instant_env.meters[0, 4] - (initial_mood + 0.40)) < 0.01
        assert abs(initial_money - instant_env.meters[0, 3] - 0.15) < 0.01

    # Park - Free Multi-Effect

    def test_park_builds_fitness(self, instant_env):
        """Park should restore +20% fitness."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Park"]
        instant_env.meters[0, 7] = 0.30

        initial_fitness = instant_env.meters[0, 7].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 7] > initial_fitness
        assert abs(instant_env.meters[0, 7] - (initial_fitness + 0.20)) < 0.01

    def test_park_builds_social(self, instant_env):
        """Park should restore +15% social."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Park"]
        instant_env.meters[0, 5] = 0.40

        initial_social = instant_env.meters[0, 5].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 5] > initial_social
        assert abs(instant_env.meters[0, 5] - (initial_social + 0.15)) < 0.01

    def test_park_improves_mood(self, instant_env):
        """Park should restore +15% mood."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Park"]
        instant_env.meters[0, 4] = 0.25

        initial_mood = instant_env.meters[0, 4].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 4] > initial_mood
        assert abs(instant_env.meters[0, 4] - (initial_mood + 0.15)) < 0.01

    def test_park_costs_energy(self, instant_env):
        """Park should cost 15% energy (time/effort)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Park"]
        instant_env.meters[0, 0] = 0.80

        initial_energy = instant_env.meters[0, 0].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Energy decreases from Park cost + passive depletion
        assert instant_env.meters[0, 0] < initial_energy
        # Park: -15%, passive depletion adds more
        expected_min_decrease = 0.15
        assert (initial_energy - instant_env.meters[0, 0]) >= expected_min_decrease - 0.01

    # Bar - Social Hub with Penalties

    def test_bar_best_for_social(self, instant_env):
        """Bar should restore +50% social (BEST in game)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Bar"]
        instant_env.meters[0, 5] = 0.20
        instant_env.meters[0, 3] = 0.50

        initial_social = instant_env.meters[0, 5].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 5] > initial_social
        assert abs(instant_env.meters[0, 5] - (initial_social + 0.50)) < 0.01

    def test_bar_improves_mood(self, instant_env):
        """Bar should restore +25% mood."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Bar"]
        instant_env.meters[0, 4] = 0.30
        instant_env.meters[0, 3] = 0.50

        initial_mood = instant_env.meters[0, 4].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 4] > initial_mood
        assert abs(instant_env.meters[0, 4] - (initial_mood + 0.25)) < 0.01

    def test_bar_has_health_penalty(self, instant_env):
        """Bar should cost 5% health (late nights, drinking)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Bar"]
        instant_env.meters[0, 6] = 0.80
        instant_env.meters[0, 3] = 0.50

        initial_health = instant_env.meters[0, 6].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Health penalty from Bar
        assert instant_env.meters[0, 6] < initial_health
        assert abs(initial_health - instant_env.meters[0, 6] - 0.05) < 0.01

    def test_bar_costs_money(self, instant_env):
        """Bar should cost $15."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Bar"]
        instant_env.meters[0, 3] = 0.50

        initial_money = instant_env.meters[0, 3].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert abs(initial_money - instant_env.meters[0, 3] - 0.15) < 0.01

    # FastFood - Quick Satiation with Penalties

    def test_fastfood_restores_satiation(self, instant_env):
        """FastFood should restore +45% satiation."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["FastFood"]
        instant_env.meters[0, 2] = 0.30
        instant_env.meters[0, 3] = 0.50

        initial_satiation = instant_env.meters[0, 2].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 2] > initial_satiation
        assert abs(instant_env.meters[0, 2] - (initial_satiation + 0.45)) < 0.01

    def test_fastfood_gives_energy_boost(self, instant_env):
        """FastFood should restore +15% energy."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["FastFood"]
        instant_env.meters[0, 0] = 0.40
        instant_env.meters[0, 3] = 0.50

        initial_energy = instant_env.meters[0, 0].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Energy boost from FastFood (sugar rush)
        # Note: passive depletion also occurs, but boost is larger
        assert instant_env.meters[0, 0] > initial_energy

    def test_fastfood_has_fitness_penalty(self, instant_env):
        """FastFood should cost 3% fitness (unhealthy food)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["FastFood"]
        instant_env.meters[0, 7] = 0.80
        instant_env.meters[0, 3] = 0.50

        initial_fitness = instant_env.meters[0, 7].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Fitness penalty
        assert instant_env.meters[0, 7] < initial_fitness
        assert abs(initial_fitness - instant_env.meters[0, 7] - 0.03) < 0.01

    def test_fastfood_has_health_penalty(self, instant_env):
        """FastFood should cost 2% health (junk food)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["FastFood"]
        instant_env.meters[0, 6] = 0.80
        instant_env.meters[0, 3] = 0.50

        initial_health = instant_env.meters[0, 6].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Health penalty
        assert instant_env.meters[0, 6] < initial_health
        assert abs(initial_health - instant_env.meters[0, 6] - 0.02) < 0.01

    # Job - Income Generation

    def test_job_generates_income(self, instant_env):
        """Job should generate $28.13 in instant mode (per_tick + completion)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Job"]
        instant_env.meters[0, 3] = 0.20

        initial_money = instant_env.meters[0, 3].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Money should increase by per_tick (0.05625 * 4) + completion (0.05625) = 0.28125
        assert instant_env.meters[0, 3] > initial_money
        assert abs(instant_env.meters[0, 3] - (initial_money + 0.28125)) < 0.01

    def test_job_costs_energy(self, instant_env):
        """Job should cost 15% energy (office work)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Job"]
        instant_env.meters[0, 0] = 0.80

        initial_energy = instant_env.meters[0, 0].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Energy cost + passive depletion
        assert instant_env.meters[0, 0] < initial_energy
        expected_decrease = 0.15
        assert (initial_energy - instant_env.meters[0, 0]) >= expected_decrease - 0.01

    # Labor - High-Pay Physical Work

    def test_labor_generates_income(self, instant_env):
        """Labor should generate income (different trade-off from Job)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Labor"]
        instant_env.meters[0, 3] = 0.20

        initial_money = instant_env.meters[0, 3].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Labor generates income (may be different from Job based on config)
        money_gained = instant_env.meters[0, 3] - initial_money
        assert money_gained > 0  # Should increase money

    def test_labor_costs_more_energy(self, instant_env):
        """Labor should cost more energy than Job (exhausting)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Labor"]
        instant_env.meters[0, 0] = 0.80

        initial_energy = instant_env.meters[0, 0].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # More exhausting than Job (20% vs 15%)
        energy_lost = initial_energy - instant_env.meters[0, 0]
        assert energy_lost >= 0.20 - 0.01

    def test_labor_has_fitness_penalty(self, instant_env):
        """Labor should cost 5% fitness (physical wear)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Labor"]
        instant_env.meters[0, 7] = 0.80

        initial_fitness = instant_env.meters[0, 7].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 7] < initial_fitness
        assert abs(initial_fitness - instant_env.meters[0, 7] - 0.05) < 0.01

    def test_labor_has_health_penalty(self, instant_env):
        """Labor should cost 5% health (injury risk)."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Labor"]
        instant_env.meters[0, 6] = 0.80

        initial_health = instant_env.meters[0, 6].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        assert instant_env.meters[0, 6] < initial_health
        assert abs(initial_health - instant_env.meters[0, 6] - 0.05) < 0.01

    # Basic Affordances

    def test_shower_restores_hygiene(self, instant_env):
        """Shower should restore hygiene and cost money."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Shower"]
        instant_env.meters[0, 1] = 0.2  # Low hygiene
        instant_env.meters[0, 3] = 0.50  # Money

        initial_hygiene = instant_env.meters[0, 1].item()
        initial_money = instant_env.meters[0, 3].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Hygiene increases, money decreases
        assert instant_env.meters[0, 1] > initial_hygiene
        assert instant_env.meters[0, 3] < initial_money

    def test_home_meal_restores_satiation_and_health(self, instant_env):
        """HomeMeal should restore satiation and health."""
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["HomeMeal"]
        instant_env.meters[0, 2] = 0.2  # Low satiation
        instant_env.meters[0, 6] = 0.6  # Moderate health
        instant_env.meters[0, 3] = 0.50

        initial_satiation = instant_env.meters[0, 2].item()
        initial_health = instant_env.meters[0, 6].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Both should increase
        assert instant_env.meters[0, 2] > initial_satiation
        assert instant_env.meters[0, 6] > initial_health

    def test_bed_instant_mode_restores_energy(self, instant_env):
        """Bed in instant mode should restore energy."""
        # Note: This tests instant mode (not temporal mechanics)
        instant_env.reset()

        instant_env.positions[0] = instant_env.affordances["Bed"]
        instant_env.meters[0, 0] = 0.3  # Low energy
        instant_env.meters[0, 3] = 0.50

        initial_energy = instant_env.meters[0, 0].item()

        actions = torch.tensor([4], device=instant_env.device)
        instant_env.step(actions)

        # Energy should increase significantly
        assert instant_env.meters[0, 0] > initial_energy


class TestMultiTickInteractions:
    """Test multi-tick interaction mechanics (temporal mode)."""

    def test_bed_multi_tick_progression(self, temporal_env):
        """Bed should apply per-tick effects and completion bonus."""
        temporal_env.reset()

        # Place agent at Bed
        temporal_env.positions[0] = temporal_env.affordances["Bed"]
        temporal_env.meters[0, 0] = 0.2  # Low energy
        temporal_env.meters[0, 3] = 0.50  # Money

        initial_energy = temporal_env.meters[0, 0].item()

        # Start multi-tick interaction (Bed takes 5 ticks)
        actions = torch.tensor([4], device=temporal_env.device)  # INTERACT

        # Interact for 5 ticks to complete
        for _ in range(5):
            temporal_env.step(actions)

        # After completion: should have linear effects + completion bonus
        # Linear: 5 ticks * (0.50 * 0.75 / 5) = 0.375
        # Bonus: 0.50 * 0.25 = 0.125
        # Total: 0.50 energy restoration
        final_energy = temporal_env.meters[0, 0].item()
        energy_gain = final_energy - initial_energy

        # Should be close to full restoration (accounting for passive depletion)
        assert energy_gain > 0.40, f"Expected ~0.50 energy gain, got {energy_gain}"

    def test_bed_early_exit_no_bonus(self, temporal_env):
        """Early exit from multi-tick interaction should skip completion bonus."""
        temporal_env.reset()

        temporal_env.positions[0] = temporal_env.affordances["Bed"]
        temporal_env.meters[0, 0] = 0.2
        temporal_env.meters[0, 3] = 0.50

        initial_energy = temporal_env.meters[0, 0].item()

        # Only interact for 3 ticks (not full 5)
        actions = torch.tensor([4], device=temporal_env.device)
        for _ in range(3):
            temporal_env.step(actions)

        # Exit early by moving
        move_action = torch.tensor([0], device=temporal_env.device)  # UP
        temporal_env.step(move_action)

        final_energy = temporal_env.meters[0, 0].item()
        energy_gain = final_energy - initial_energy

        # Should have linear benefits only (no 25% bonus)
        # 3 ticks * (0.50 * 0.75 / 5) = 0.225
        # Should be less than full completion
        assert energy_gain < 0.40, f"Early exit should not give full restoration, got {energy_gain}"

    def test_multi_tick_per_tick_costs(self, temporal_env):
        """Multi-tick interactions should apply per-tick costs."""
        temporal_env.reset()

        temporal_env.positions[0] = temporal_env.affordances["Bed"]
        temporal_env.meters[0, 0] = 0.5
        temporal_env.meters[0, 3] = 0.50  # Starting money

        initial_money = temporal_env.meters[0, 3].item()

        # Complete 5-tick interaction
        actions = torch.tensor([4], device=temporal_env.device)
        for _ in range(5):
            temporal_env.step(actions)

        # Money should decrease by per-tick cost * ticks
        # Bed: $0.01 per tick * 5 ticks = $0.05
        money_spent = initial_money - temporal_env.meters[0, 3].item()

        assert abs(money_spent - 0.05) < 0.01, f"Expected $0.05 cost for 5 ticks, got {money_spent}"


class TestAffordanceBatching:
    """Test batch processing of affordances (multi-agent scenarios)."""

    def test_multiple_agents_different_affordances(self, multi_agent_env):
        """Multiple agents should interact with different affordances simultaneously."""
        multi_agent_env.reset()

        # Position agents at different affordances
        multi_agent_env.positions[0] = multi_agent_env.affordances["Bed"]
        multi_agent_env.positions[1] = multi_agent_env.affordances["Shower"]
        multi_agent_env.positions[2] = multi_agent_env.affordances["HomeMeal"]
        multi_agent_env.positions[3] = multi_agent_env.affordances["Park"]

        # Set initial meters
        multi_agent_env.meters[:, 3] = 0.50  # All have money
        multi_agent_env.meters[0, 0] = 0.2  # Agent 0: low energy
        multi_agent_env.meters[1, 1] = 0.3  # Agent 1: low hygiene
        multi_agent_env.meters[2, 2] = 0.4  # Agent 2: low satiation
        multi_agent_env.meters[3, 7] = 0.3  # Agent 3: low fitness

        # All interact simultaneously
        actions = torch.tensor([4, 4, 4, 4], device=multi_agent_env.device)
        multi_agent_env.step(actions)

        # Verify each agent got correct effects
        assert multi_agent_env.meters[0, 0] > 0.2, "Agent 0 should restore energy"
        assert multi_agent_env.meters[1, 1] > 0.3, "Agent 1 should restore hygiene"
        assert multi_agent_env.meters[2, 2] > 0.4, "Agent 2 should restore satiation"
        assert multi_agent_env.meters[3, 7] > 0.3, "Agent 3 should restore fitness"

    def test_batch_affordability_masking(self, multi_agent_env):
        """Batch affordability check should mask agents without money."""
        multi_agent_env.reset()

        # All agents at Shower
        multi_agent_env.positions[:] = multi_agent_env.affordances["Shower"]

        # Set varied money levels
        multi_agent_env.meters[0, 3] = 0.50  # Can afford
        multi_agent_env.meters[1, 3] = 0.00  # Cannot afford
        multi_agent_env.meters[2, 3] = 0.50  # Can afford
        multi_agent_env.meters[3, 3] = 0.01  # Cannot afford ($0.01 < $0.03)

        # Set low hygiene for all
        multi_agent_env.meters[:, 1] = 0.2

        initial_hygiene = multi_agent_env.meters[:, 1].clone()

        # All try to interact
        actions = torch.tensor([4, 4, 4, 4], device=multi_agent_env.device)
        multi_agent_env.step(actions)

        # Only agents 0 and 2 should have hygiene increase
        assert multi_agent_env.meters[0, 1] > initial_hygiene[0], "Agent 0 should interact (has money)"
        assert multi_agent_env.meters[1, 1] <= initial_hygiene[1] + 0.01, "Agent 1 should not interact (no money)"
        assert multi_agent_env.meters[2, 1] > initial_hygiene[2], "Agent 2 should interact (has money)"
        assert multi_agent_env.meters[3, 1] <= initial_hygiene[3] + 0.01, "Agent 3 should not interact (insufficient money)"

    def test_batch_processing_preserves_individual_states(self, multi_agent_env):
        """Batch processing should not leak state between agents."""
        multi_agent_env.reset()

        # Agent 0 at Bed, others elsewhere
        multi_agent_env.positions[0] = multi_agent_env.affordances["Bed"]
        multi_agent_env.positions[1] = multi_agent_env.affordances["Shower"]
        multi_agent_env.positions[2] = multi_agent_env.affordances["HomeMeal"]
        multi_agent_env.positions[3] = multi_agent_env.affordances["Park"]

        multi_agent_env.meters[:, 3] = 0.50  # All have money

        # Store initial states
        initial_energy = multi_agent_env.meters[:, 0].clone()
        _initial_hygiene = multi_agent_env.meters[:, 1].clone()
        _initial_satiation = multi_agent_env.meters[:, 2].clone()

        # Only agent 0 interacts
        actions = torch.tensor([4, 0, 1, 2], device=multi_agent_env.device)  # 0=INTERACT, others=MOVE
        multi_agent_env.step(actions)

        # Agent 0 should have energy change, others should not
        assert multi_agent_env.meters[0, 0] != initial_energy[0], "Agent 0 should change (interacted)"
        # Other agents may have passive depletion but no affordance effects
        # Just verify they didn't get Bed's energy boost
        assert abs(multi_agent_env.meters[1, 0] - initial_energy[1]) < 0.1, "Agent 1 should not get Bed's effects"


@pytest.fixture
def bed_equivalence_envs(
    tmp_path: Path,
    test_config_pack_path: Path,
    env_factory,
    cpu_device: torch.device,
):
    """Provide instant + temporal mechanics envs sharing the same config pack."""

    def _build_env(enable_temporal: bool):
        target = tmp_path / ("temporal" if enable_temporal else "instant")
        shutil.copytree(test_config_pack_path, target)

        training_path = target / "training.yaml"
        with training_path.open() as fh:
            training_config = yaml.safe_load(fh)

        training_config["environment"]["enable_temporal_mechanics"] = enable_temporal

        with training_path.open("w") as fh:
            yaml.safe_dump(training_config, fh, sort_keys=False)

        return env_factory(
            config_dir=target,
            num_agents=1,
            device_override=cpu_device,
        )

    return _build_env(False), _build_env(True)


class TestAffordanceEquivalence:
    """Test equivalence between instant and multi-tick modes (75%/25% split validation)."""

    def test_bed_instant_vs_multitick_total_equals(self, bed_equivalence_envs, cpu_device):
        """Bed instant mode should equal multi-tick total (75% linear + 25% bonus)."""
        instant_env, temporal_env = bed_equivalence_envs

        # Test Bed interaction equivalence
        instant_env.reset()
        temporal_env.reset()

        # Set identical initial states
        instant_env.positions[0] = instant_env.affordances["Bed"]
        temporal_env.positions[0] = temporal_env.affordances["Bed"]

        instant_env.meters[0, 0] = 0.3  # Energy
        temporal_env.meters[0, 0] = 0.3
        instant_env.meters[0, 3] = 0.50  # Money
        temporal_env.meters[0, 3] = 0.50

        # Instant mode: single interaction
        instant_initial = instant_env.meters[0, 0].item()
        instant_env.step(torch.tensor([4], device=cpu_device))
        instant_final = instant_env.meters[0, 0].item()
        instant_gain = instant_final - instant_initial

        # Multi-tick mode: complete 5-tick interaction
        temporal_initial = temporal_env.meters[0, 0].item()
        for _ in range(5):
            temporal_env.step(torch.tensor([4], device=cpu_device))
        temporal_final = temporal_env.meters[0, 0].item()
        temporal_gain = temporal_final - temporal_initial

        # Gains should be approximately equal (accounting for passive depletion differences)
        # Instant: 1 step of depletion
        # Multi-tick: 5 steps of depletion
        # So temporal will have ~4 extra steps of depletion
        passive_diff = 4 * 0.005  # 4 extra steps * 0.5% per step

        assert (
            abs(instant_gain - temporal_gain - passive_diff) < 0.05
        ), f"Instant gain ({instant_gain}) should equal temporal gain ({temporal_gain}) + passive diff ({passive_diff})"

    def test_all_affordances_have_consistent_effects(self, basic_env, cpu_device):
        """All affordances should produce deterministic, consistent effects."""
        # This ensures affordance engine produces reliable results

        affordances_to_test = [
            "Bed",
            "Shower",
            "HomeMeal",
            "FastFood",
            "Bar",
            "Doctor",
            "Hospital",
            "Therapist",
            "Park",
            "Gym",
            "Job",
            "Labor",
            "Recreation",
        ]

        for aff_name in affordances_to_test:
            # Reset and position
            basic_env.reset()
            basic_env.positions[0] = basic_env.affordances[aff_name]
            basic_env.meters[0] = torch.tensor([0.5] * 8, dtype=torch.float32, device=cpu_device)
            basic_env.meters[0, 3] = 0.50  # Ensure money

            # Store initial
            initial_meters = basic_env.meters[0].clone()

            # Interact
            basic_env.step(torch.tensor([4], device=basic_env.device))

            # Verify some effect occurred
            meter_diff = (basic_env.meters[0] - initial_meters).abs().sum()
            assert meter_diff > 1e-6, f"{aff_name} should modify meters (got diff={meter_diff})"
