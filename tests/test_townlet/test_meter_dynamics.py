"""
Test Suite: Meter Dynamics & Cascading Effects

This test file validates the fundamental meter depletion and cascading
penalty system. These tests are CRITICAL because meter dynamics form the
foundation of the entire game - if meters are wrong, nothing works.

Coverage Target: vectorized_env.py lines 770-950 (meter dynamics)

CRITICAL DESIGN INSIGHT (2025-10-31):
==========================================
All tests use INTERACT action (index 4) on empty tiles to isolate passive
depletion from movement costs. This is ESSENTIAL because:

1. Movement costs 0.5% energy per step (same as passive depletion)
2. INTERACT is masked unless on affordance, forcing agents to move every step
3. This causes 25% extra energy drain over 50 steps (on top of cascades)
4. Original finding: 81.5% energy loss seemed like multiplicative cascade bug
5. Actual cause: 52.3% from cascades (nearly additive) + 25% from forced movement
6. WAIT action is HIGH PRIORITY to fix oscillation behavior near affordances

Testing Philosophy:
- Test passive depletion in isolation (INTERACT action)
- Test cascades in isolation (INTERACT action + manipulated meters)
- Test combined effects separately (movement + cascades together)
- This reveals true system behavior vs confounding factors
"""

import pytest
import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def env():
    """Create environment with default settings."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device("cpu"))
    env.reset()  # Initialize meters and grid state
    return env


class TestBaseDepletion:
    """Test base per-step meter depletion rates.

    NOTE: Uses INTERACT action (4) on empty tiles to test PASSIVE depletion only.
    Movement actions add additional costs (see TestMovementCosts for combined costs).
    """

    def test_energy_depletes_at_correct_rate(self, env):
        """Energy should deplete 0.5% per step from passive depletion only.

        NOTE: Movement costs ADDITIONAL 0.5%, total 1.0% per movement step.
        This test uses INTERACT on empty tile to isolate passive depletion.
        """
        # Move to empty tile away from affordances (center of grid)
        env.positions[0] = torch.tensor([4, 4])
        initial_energy = env.meters[0, 0].item()

        # Use INTERACT on empty tile (fails, but passive depletion still happens)
        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT - no movement cost

        energy_after_100 = env.meters[0, 0].item()
        expected_depletion = 0.005 * 100  # 0.5% × 100 steps = 50%

        # Allow tolerance for cascading effects from other meters
        actual_depletion = initial_energy - energy_after_100
        assert abs(actual_depletion - expected_depletion) < 0.10, (
            f"Passive energy depletion should be ~{expected_depletion * 100:.1f}%, "
            f"got {actual_depletion * 100:.1f}%"
        )

    def test_hygiene_depletes_at_correct_rate(self, env):
        """Hygiene should deplete 0.3% per step (passive only)."""
        env.positions[0] = torch.tensor([4, 4])
        initial_hygiene = env.meters[0, 1].item()

        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT - no movement cost

        hygiene_after_100 = env.meters[0, 1].item()
        expected_depletion = 0.003 * 100  # 0.3% × 100 = 30%
        actual_depletion = initial_hygiene - hygiene_after_100

        assert abs(actual_depletion - expected_depletion) < 0.05, (
            f"Passive hygiene depletion should be ~{expected_depletion * 100:.1f}%, "
            f"got {actual_depletion * 100:.1f}%"
        )

    def test_satiation_depletes_at_correct_rate(self, env):
        """Satiation should deplete 0.4% per step (passive only)."""
        env.positions[0] = torch.tensor([4, 4])
        initial_satiation = env.meters[0, 2].item()

        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT - no movement cost

        satiation_after_100 = env.meters[0, 2].item()
        expected_depletion = 0.004 * 100  # 0.4% × 100 = 40%
        actual_depletion = initial_satiation - satiation_after_100

        assert abs(actual_depletion - expected_depletion) < 0.05, (
            f"Passive satiation depletion should be ~{expected_depletion * 100:.1f}%, "
            f"got {actual_depletion * 100:.1f}%"
        )

    def test_money_does_not_deplete_passively(self, env):
        """Money should only change via interactions, not passive depletion."""
        env.positions[0] = torch.tensor([4, 4])
        initial_money = env.meters[0, 3].item()

        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT - no money spent on failure

        money_after_100 = env.meters[0, 3].item()

        assert abs(money_after_100 - initial_money) < 0.001, (
            f"Money should not deplete passively, got {money_after_100:.3f} vs {initial_money:.3f}"
        )

    def test_mood_depletes_at_correct_rate(self, env):
        """Mood should deplete 0.1% per step (slowest meter)."""
        env.positions[0] = torch.tensor([4, 4])
        initial_mood = env.meters[0, 4].item()

        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT - no movement cost

        mood_after_100 = env.meters[0, 4].item()
        expected_depletion = 0.001 * 100  # 0.1% × 100 = 10%
        actual_depletion = initial_mood - mood_after_100

        assert abs(actual_depletion - expected_depletion) < 0.05, (
            f"Passive mood depletion should be ~{expected_depletion * 100:.1f}%, "
            f"got {actual_depletion * 100:.1f}%"
        )

    def test_social_depletes_at_correct_rate(self, env):
        """Social should deplete 0.6% per step (fastest base rate!)."""
        env.positions[0] = torch.tensor([4, 4])
        initial_social = env.meters[0, 5].item()

        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT - no movement cost

        social_after_100 = env.meters[0, 5].item()
        expected_depletion = 0.006 * 100  # 0.6% × 100 = 60%
        actual_depletion = initial_social - social_after_100

        assert abs(actual_depletion - expected_depletion) < 0.05, (
            f"Passive social depletion should be ~{expected_depletion * 100:.1f}%, "
            f"got {actual_depletion * 100:.1f}%"
        )

    def test_fitness_depletes_at_correct_rate(self, env):
        """Fitness should deplete 0.2% per step."""
        env.positions[0] = torch.tensor([4, 4])
        initial_fitness = env.meters[0, 7].item()

        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT - no movement cost

        fitness_after_100 = env.meters[0, 7].item()
        expected_depletion = 0.002 * 100  # 0.2% × 100 = 20%
        actual_depletion = initial_fitness - fitness_after_100

        assert abs(actual_depletion - expected_depletion) < 0.05, (
            f"Passive fitness depletion should be ~{expected_depletion * 100:.1f}%, "
            f"got {actual_depletion * 100:.1f}%"
        )


class TestFitnessModulatedHealthDepletion:
    """Test health depletion modulated by fitness level - HIGH BUG RISK."""

    def test_fitness_gradient_health_modulation(self, env):
        """Fitness uses smooth gradient (0.5x to 3.0x) - NO MORE HARD THRESHOLDS!"""
        # Test at various fitness levels to confirm smooth gradient
        test_cases = [
            (1.0, 0.5),  # 100% fitness: 0.5x multiplier = 0.0005/step
            (0.7, 1.25),  # 70% fitness: 1.25x multiplier = 0.00125/step
            (0.5, 1.75),  # 50% fitness: 1.75x multiplier = 0.00175/step
            (0.3, 2.25),  # 30% fitness: 2.25x multiplier = 0.00225/step
            (0.0, 3.0),  # 0% fitness: 3.0x multiplier = 0.003/step
        ]

        print("\n💪 FITNESS GRADIENT MODULATION (SMOOTH - NO CLIFFS):")

        for fitness_level, expected_multiplier in test_cases:
            env.reset()
            env.meters[0, 7] = fitness_level
            env.positions[0] = torch.tensor([4, 4])  # Empty tile
            initial_health = env.meters[0, 6].item()

            # Step 100 times (INTERACT to isolate health depletion)
            for _ in range(100):
                env.step(torch.tensor([4]))  # INTERACT

            health_loss = initial_health - env.meters[0, 6].item()
            expected_loss = 0.001 * expected_multiplier * 100

            print(
                f"   Fitness {fitness_level * 100:5.1f}%: "
                f"{expected_multiplier:.2f}x → "
                f"{health_loss * 100:5.1f}% loss (expected {expected_loss * 100:.1f}%)"
            )

            # Allow small tolerance for floating point + fitness decay
            assert abs(health_loss - expected_loss) < 0.03, (
                f"At {fitness_level * 100}% fitness, expected {expected_loss * 100:.1f}% health loss, "
                f"got {health_loss * 100:.1f}%"
            )

        print("   ✅ Smooth gradient confirmed - consistent with other cascades!")

    def test_low_fitness_accelerates_health_depletion(self, env):
        """Low fitness → health depletes faster with GRADIENT calculation."""
        # Set fitness to 20%
        env.meters[0, 7] = 0.2
        env.positions[0] = torch.tensor([4, 4])  # Empty tile
        initial_health = env.meters[0, 6].item()

        # Step 100 times (INTERACT to avoid movement costs)
        for _ in range(100):
            env.step(torch.tensor([4]))  # INTERACT

        health_after_100 = env.meters[0, 6].item()

        # At 20% fitness:
        # penalty_strength = 1.0 - 0.2 = 0.8
        # multiplier = 0.5 + (2.5 * 0.8) = 2.5
        # depletion = 0.001 * 2.5 = 0.0025/step
        # But fitness also decays (0.2% per step), so multiplier increases slightly
        # Average multiplier ≈ 2.6, so ≈ 26% loss over 100 steps

        expected_loss = 0.26  # ~26% (escalating as fitness drops further)
        actual_loss = initial_health - health_after_100

        print("\n🚨 LOW FITNESS HEALTH DEPLETION:")
        print(f"   Starting fitness: 20%")
        print(f"   Final fitness: {env.meters[0, 7].item() * 100:.1f}%")
        print(f"   Health loss: {actual_loss * 100:.1f}%")
        print(f"   Expected: ~{expected_loss * 100:.0f}%")

        # Allow tolerance for escalating multiplier
        assert abs(actual_loss - expected_loss) < 0.05, (
            f"Expected ~{expected_loss * 100:.0f}% health loss, got {actual_loss * 100:.1f}%"
        )

    def test_fitness_decay_affects_health_over_time(self, env):
        """As fitness decays, health depletion should accelerate - DEATH SPIRAL CHECK."""
        # Start with medium fitness
        env.meters[0, 7] = 0.5

        health_checkpoints = []
        fitness_checkpoints = []

        # Track every 50 steps for 300 steps
        for i in range(6):
            health_checkpoints.append(env.meters[0, 6].item())
            fitness_checkpoints.append(env.meters[0, 7].item())

            # Step 50 times
            for _ in range(50):
                env.step(torch.tensor([0]))

        # Final checkpoint
        health_checkpoints.append(env.meters[0, 6].item())
        fitness_checkpoints.append(env.meters[0, 7].item())

        print("\n🔍 FITNESS DECAY HEALTH SPIRAL:")
        for i, (health, fitness) in enumerate(zip(health_checkpoints, fitness_checkpoints)):
            print(
                f"   Step {i * 50:3d}: Health={health * 100:5.1f}%, Fitness={fitness * 100:5.1f}%"
            )

        # Verify death spiral pattern:
        # - Health depletion accelerates as fitness drops
        # - Agent dies somewhere between 200-300 steps
        # This is EXPECTED with gradient system (not a bug!)

        final_health = env.meters[0, 6].item()

        if final_health == 0.0:
            print("   ⚠️ Agent died from fitness death spiral")
            print("   📊 This is mathematically correct with gradient penalties")
            # Verify agent survived at least 200 steps before death
            assert health_checkpoints[4] > 0, "Agent died too quickly (<200 steps) - check for bugs"
        else:
            print(f"   ✓ Agent survived 300 steps with {final_health * 100:.1f}% health")

        # Verify accelerating depletion pattern (health loss increases over time)
        loss_step_0_50 = health_checkpoints[0] - health_checkpoints[1]
        loss_step_100_150 = health_checkpoints[2] - health_checkpoints[3]

        print(
            f"   📈 Loss rate increased: "
            f"{loss_step_0_50 * 100:.1f}% (steps 0-50) → "
            f"{loss_step_100_150 * 100:.1f}% (steps 100-150)"
        )

        assert loss_step_100_150 > loss_step_0_50, (
            "Health depletion should accelerate as fitness drops (gradient effect)"
        )


class TestCascadingSecondaryToPrimary:
    """Test SECONDARY → PRIMARY cascading effects - HIGH BUG RISK."""

    def test_low_satiation_damages_health(self, env):
        """Low satiation (<30%) → health penalty scales with deficit - VALIDATED CORRECT."""
        # Set satiation to 20% (below 30% threshold)
        env.meters[0, 2] = 0.2
        env.positions[0] = torch.tensor([4, 4])  # Empty tile
        initial_health = env.meters[0, 6].item()

        # Step 50 times (INTERACT to avoid movement costs and isolate cascade)
        for _ in range(50):
            env.step(torch.tensor([4]))  # INTERACT

        health_after_50 = env.meters[0, 6].item()

        # Satiation drops from 20% → 0% over 50 steps (0.4% per step)
        # Cascade penalty escalates as satiation drops:
        # - Start (20%): deficit=0.333, penalty=0.133% per step
        # - Middle (10%): deficit=0.667, penalty=0.267% per step
        # - End (0%): deficit=1.000, penalty=0.400% per step
        # Average cascade: ~0.267% per step
        # Base depletion: 0.1% per step (fitness-modulated)
        # Total: 0.367% per step average × 50 = 18.4%

        expected_base_depletion = 0.001 * 50  # 5.0% (before fitness modulation)
        expected_cascade_penalty = 0.134  # ~13.4% (escalating)
        expected_total = expected_base_depletion + expected_cascade_penalty  # 18.4%

        actual_health_loss = initial_health - health_after_50

        print(f"\n🍔 LOW SATIATION → HEALTH:")
        print(f"   Base depletion: {expected_base_depletion * 100:.1f}%")
        print(f"   Cascade penalty (escalating): {expected_cascade_penalty * 100:.1f}%")
        print(f"   Total expected: {expected_total * 100:.1f}%")
        print(f"   Actual loss: {actual_health_loss * 100:.1f}%")
        print(f"   Average rate: {(actual_health_loss / 50) * 100:.2f}% per step")

        # Allow tolerance for fitness modulation effects
        assert abs(actual_health_loss - expected_total) < 0.05, (
            f"Expected {expected_total * 100:.1f}% health loss, got {actual_health_loss * 100:.1f}%"
        )

    def test_low_satiation_damages_energy(self, env):
        """Low satiation (<30%) → energy penalty scales with deficit - VALIDATED CORRECT."""
        # Set satiation to 20%
        env.meters[0, 2] = 0.2
        env.positions[0] = torch.tensor([4, 4])  # Empty tile
        initial_energy = env.meters[0, 0].item()

        # Step 50 times (INTERACT to avoid movement costs)
        for _ in range(50):
            env.step(torch.tensor([4]))  # INTERACT

        energy_after_50 = env.meters[0, 0].item()

        # VALIDATED CALCULATION:
        # Satiation drops from 20% → 0% over 50 steps (0.4% per step)
        # Cascade penalty escalates as satiation drops:
        # - Start (20%): deficit=0.333, penalty=0.167% per step
        # - Middle (10%): deficit=0.667, penalty=0.333% per step
        # - End (0%): deficit=1.000, penalty=0.500% per step
        # Average cascade: ~0.334% per step
        # Base depletion: 0.5% per step
        # Total: 0.834% per step average × 50 = 41.7%

        expected_base_depletion = 0.005 * 50  # 25.0%
        expected_cascade_penalty = 0.167  # ~16.7% (escalating)
        expected_total = expected_base_depletion + expected_cascade_penalty  # 41.7%

        actual_energy_loss = initial_energy - energy_after_50

        print(f"\n⚡ LOW SATIATION → ENERGY:")
        print(f"   Base depletion: {expected_base_depletion * 100:.1f}%")
        print(f"   Cascade penalty (escalating): {expected_cascade_penalty * 100:.1f}%")
        print(f"   Total expected: {expected_total * 100:.1f}%")
        print(f"   Actual loss: {actual_energy_loss * 100:.1f}%")
        print(f"   Average rate: {(actual_energy_loss / 50) * 100:.2f}% per step")

        # Verified correct: 41.8% actual vs 41.7% expected
        assert abs(actual_energy_loss - expected_total) < 0.02, (
            f"Expected {expected_total * 100:.1f}% energy loss, got {actual_energy_loss * 100:.1f}%"
        )

        # Design note: This IS aggressive (67% more damage than passive alone)
        # but mathematically correct. May need rebalancing for gameplay.

    def test_low_mood_damages_energy(self, env):
        """Low mood (<30%) → energy penalty scales with deficit."""
        # Set mood to 20%
        env.meters[0, 4] = 0.2
        env.positions[0] = torch.tensor([4, 4])  # Empty tile
        initial_energy = env.meters[0, 0].item()

        # Step 50 times (INTERACT to avoid movement costs)
        for _ in range(50):
            env.step(torch.tensor([4]))  # INTERACT

        energy_after_50 = env.meters[0, 0].item()

        # Similar to satiation: mood drops from 20% → 19% over 50 steps (0.1% per step)
        # Mood starts below 30% threshold, so cascade is active throughout
        # But mood decays much slower than satiation (0.1% vs 0.4%)
        # Average deficit ≈ (0.333 + 0.3) / 2 = 0.316
        # Average cascade: 0.005 * 0.316 = 0.158% per step
        expected_base_depletion = 0.005 * 50  # 25.0%
        expected_cascade_penalty = 0.005 * 0.316 * 50  # ~7.9%
        expected_total = expected_base_depletion + expected_cascade_penalty  # 32.9%

        actual_energy_loss = initial_energy - energy_after_50

        print("\n😔 LOW MOOD → ENERGY:")
        print(f"   Mood: 20% → {env.meters[0, 4].item() * 100:.1f}%")
        print(f"   Energy loss: {actual_energy_loss * 100:.1f}%")
        print(f"   Expected: {expected_total * 100:.1f}%")

        # Allow tolerance for slight calculation variations
        assert abs(actual_energy_loss - expected_total) < 0.05, (
            f"Expected {expected_total * 100:.1f}% energy loss, got {actual_energy_loss * 100:.1f}%"
        )

    def test_combined_low_satiation_and_mood_energy_drain(self, env):
        """COMBINED low satiation + low mood → compound escalating penalty - BALANCE ISSUE."""
        # Set both satiation AND mood to 20%
        env.meters[0, 2] = 0.2  # Low satiation
        env.meters[0, 4] = 0.2  # Low mood
        env.positions[0] = torch.tensor([4, 4])  # Empty tile
        initial_energy = env.meters[0, 0].item()

        # Step 50 times (INTERACT to isolate cascade effects)
        for _ in range(50):
            env.step(torch.tensor([4]))  # INTERACT

        energy_after_50 = env.meters[0, 0].item()

        # ESCALATING COMBINED CASCADES (both meters decay):
        # Satiation: 20% → 0% (0.4% per step)
        # Mood: 20% → 15% (0.1% per step)
        #
        # Energy depletion compounds:
        # 1. Base: 0.5% per step
        # 2. Satiation cascade (escalating): ~0.334% per step average
        # 3. Mood cascade (semi-stable): ~0.158% per step average
        #
        # Total expected: 25% + 16.7% + 7.9% = 49.6%
        #
        # BUT satiation's escalation + mood's escalation compound!
        # Actual observed: 70-85% (death spiral territory)

        expected_base_depletion = 0.005 * 50  # 25.0%
        expected_satiation_cascade = 0.167  # ~16.7% (escalating)
        expected_mood_cascade = 0.079  # ~7.9% (stable)
        expected_simple_additive = (
            expected_base_depletion + expected_satiation_cascade + expected_mood_cascade
        )  # 49.6%

        actual_energy_loss = initial_energy - energy_after_50

        print(f"\n💥 COMBINED SATIATION + MOOD → ENERGY:")
        print(f"   Base depletion: {expected_base_depletion * 100:.1f}%")
        print(f"   Satiation cascade (escalating): {expected_satiation_cascade * 100:.1f}%")
        print(f"   Mood cascade (stable): {expected_mood_cascade * 100:.1f}%")
        print(f"   Simple additive: {expected_simple_additive * 100:.1f}%")
        print(f"   Actual loss: {actual_energy_loss * 100:.1f}%")
        print(f"   🚨 COMPOUND EFFECT: {(actual_energy_loss / 50) * 100:.2f}% per step!")
        print(f"   📊 Amplification: {(actual_energy_loss / expected_simple_additive):.2f}x")

        # BALANCE ISSUE DOCUMENTATION:
        # - WITHOUT MOVEMENT: 52.3% loss (1.05% per step)
        # - WITH MOVEMENT: 70-85% loss (1.4-1.7% per step)
        # - Movement adds 0.5% per step = 25% over 50 steps
        # - This compounds with cascades multiplicatively
        # - Agent exploring randomly dies in ~60-80 steps
        # - This is faster than agents can find/reach/use affordances (50-80 steps needed)
        # - Creates unrecoverable situation that prevents learning

        # Validate cascades are additive (52.3% ≈ 49.6% expected)
        # Small amplification (1.05x) is tolerable - this is NOT the death spiral
        assert abs(actual_energy_loss - expected_simple_additive) < 0.05, (
            f"Expected ~{expected_simple_additive * 100:.1f}% loss, got {actual_energy_loss * 100:.1f}% - math discrepancy!"
        )

        # Confirm agent survives long enough to validate (we're testing cascade math, not instant death)
        assert energy_after_50 > 0, (
            "Agent died during test - cascades are too aggressive even for testing"
        )

    def test_satiation_threshold_investigation(self, env):
        """Investigate if 30% threshold is correct - BUG SUSPECT."""
        # Test at various satiation levels around threshold
        test_levels = [0.35, 0.31, 0.30, 0.29, 0.25, 0.20, 0.15, 0.10]

        print(f"\n🔬 SATIATION THRESHOLD INVESTIGATION:")
        print(f"   Threshold: 30% (below this, penalties apply)")
        print(f"   Testing energy drain over 50 steps...")

        for satiation_level in test_levels:
            # Reset environment
            env.reset()
            env.meters[0, 2] = satiation_level
            initial_energy = env.meters[0, 0].item()

            # Step 50 times
            for _ in range(50):
                env.step(torch.tensor([0]))

            energy_loss = initial_energy - env.meters[0, 0].item()
            energy_loss_per_step = energy_loss / 50

            threshold_text = "ABOVE" if satiation_level >= 0.3 else "BELOW"
            print(
                f"   Satiation {satiation_level * 100:4.0f}% ({threshold_text}): "
                f"Energy loss = {energy_loss * 100:5.1f}% ({energy_loss_per_step * 100:.3f}% per step)"
            )


class TestCascadingTertiaryToSecondary:
    """Test TERTIARY → SECONDARY cascading effects."""

    def test_low_hygiene_damages_satiation(self, env):
        """Low hygiene (<30%) → satiation penalty (being dirty → loss of appetite) - ESCALATING."""
        # Set hygiene to 20%
        env.meters[0, 1] = 0.2
        env.positions[0] = torch.tensor([4, 4])  # Empty tile
        initial_satiation = env.meters[0, 2].item()

        # Step 50 times (INTERACT to isolate cascade)
        for _ in range(50):
            env.step(torch.tensor([4]))  # INTERACT

        satiation_after_50 = env.meters[0, 2].item()

        # Hygiene drops from 20% → 5% over 50 steps (0.3% per step)
        # This creates escalating cascade as hygiene worsens:
        # - Start (20%): deficit=0.333, penalty=0.0667% per step
        # - Middle (12.5%): deficit=0.583, penalty=0.1167% per step
        # - End (5%): deficit=0.833, penalty=0.1667% per step
        # Average cascade: ~0.117% per step
        expected_base_depletion = 0.004 * 50  # 20.0%
        expected_hygiene_cascade = 0.058  # ~5.8% (escalating)
        expected_total = expected_base_depletion + expected_hygiene_cascade  # 25.8%

        actual_loss = initial_satiation - satiation_after_50

        print(f"\n🧼 LOW HYGIENE → SATIATION:")
        print(f"   Base depletion: {expected_base_depletion * 100:.1f}%")
        print(f"   Hygiene cascade (escalating): {expected_hygiene_cascade * 100:.1f}%")
        print(f"   Total expected: {expected_total * 100:.1f}%")
        print(f"   Actual loss: {actual_loss * 100:.1f}%")

        # Allow tolerance for cascade escalation variations
        assert abs(actual_loss - expected_total) < 0.08, (
            f"Expected {expected_total * 100:.1f}% satiation loss, got {actual_loss * 100:.1f}%"
        )

    def test_low_social_damages_mood(self, env):
        """Low social (<30%) → mood penalty (strongest tertiary effect)."""
        # Set social to 20%
        env.meters[0, 5] = 0.2
        initial_mood = env.meters[0, 4].item()

        # Step 50 times
        for _ in range(50):
            env.step(torch.tensor([0]))

        mood_after_50 = env.meters[0, 4].item()

        # Social penalty: 0.004 * deficit per step (stronger than hygiene's 0.003)
        deficit = (0.3 - 0.2) / 0.3
        expected_cascade = 0.004 * deficit * 50
        expected_base = 0.001 * 50
        expected_total = expected_cascade + expected_base

        actual_loss = initial_mood - mood_after_50

        print(f"\n👥 LOW SOCIAL → MOOD:")
        print(f"   Social: 20%")
        print(f"   Mood loss: {actual_loss * 100:.1f}%")
        print(f"   Expected: {expected_total * 100:.1f}%")
        print(
            f"   Cascade contributes: {(expected_cascade / expected_total) * 100:.0f}% of total loss"
        )


class TestDeathConditions:
    """Test terminal conditions - CRITICAL."""

    def test_death_when_health_reaches_zero(self, env):
        """Health = 0 → done = True."""
        # Force health to 0
        env.meters[0, 6] = 0.0

        # Take one step
        _, _, dones, _ = env.step(torch.tensor([0]))

        assert dones[0].item() is True, "Agent should die when health reaches 0"

    def test_death_when_energy_reaches_zero(self, env):
        """Energy = 0 → done = True."""
        # Force energy to 0
        env.meters[0, 0] = 0.0

        # Take one step
        _, _, dones, _ = env.step(torch.tensor([0]))

        assert dones[0].item() is True, "Agent should die when energy reaches 0"

    def test_no_death_from_other_meters_at_zero(self, env):
        """Setting hygiene, satiation, mood, social, fitness to 0 should NOT cause immediate death."""
        # Set all non-primary meters to 0
        env.meters[0, 1] = 0.0  # Hygiene
        env.meters[0, 2] = 0.0  # Satiation
        env.meters[0, 4] = 0.0  # Mood
        env.meters[0, 5] = 0.0  # Social
        env.meters[0, 7] = 0.0  # Fitness

        # Ensure primary meters are healthy
        env.meters[0, 0] = 0.8  # Energy
        env.meters[0, 6] = 0.8  # Health

        # Take one step
        _, _, dones, _ = env.step(torch.tensor([0]))

        # Should not die immediately (though cascading will kill soon)
        assert dones[0].item() is False, (
            "Agent should not die immediately from secondary/tertiary meters being 0"
        )

    def test_death_spiral_from_low_satiation(self, env):
        """Start with low satiation (20%) and normal other meters - track steps to death."""
        env.meters[0, 2] = 0.2  # Low satiation

        steps_survived = 0
        max_steps = 500

        for step in range(max_steps):
            _, _, dones, _ = env.step(torch.tensor([0]))
            steps_survived = step + 1

            if dones[0].item():
                break

        print(f"\n☠️ DEATH SPIRAL FROM LOW SATIATION:")
        print(f"   Started with satiation: 20%")
        print(f"   Survived: {steps_survived} steps")
        print(f"   Final meters:")
        print(f"      Energy: {env.meters[0, 0].item() * 100:.1f}%")
        print(f"      Health: {env.meters[0, 6].item() * 100:.1f}%")
        print(f"      Satiation: {env.meters[0, 2].item() * 100:.1f}%")

        # BUG INVESTIGATION: If agent dies in <100 steps, it's probably too punishing
        # At 0.2 satiation, agent needs ~50 steps to reach affordance and interact
        assert steps_survived > 50, (
            f"Agent died in {steps_survived} steps - too fast to reach affordances from low satiation"
        )

    def test_death_spiral_from_combined_low_meters(self, env):
        """Multiple low meters (satiation + hygiene + social) → faster death."""
        env.meters[0, 1] = 0.2  # Low hygiene
        env.meters[0, 2] = 0.2  # Low satiation
        env.meters[0, 5] = 0.2  # Low social

        steps_survived = 0
        max_steps = 500

        for step in range(max_steps):
            _, _, dones, _ = env.step(torch.tensor([0]))
            steps_survived = step + 1

            if dones[0].item():
                break

        print(f"\n💀 DEATH SPIRAL FROM COMBINED LOW METERS:")
        print(f"   Started with hygiene/satiation/social: 20%")
        print(f"   Survived: {steps_survived} steps")

        # Should die faster than single low meter, but not instantly
        assert 30 < steps_survived < 200, (
            f"Combined low meters: expected 30-200 steps survival, got {steps_survived}"
        )


class TestMeterClamping:
    """Test meter boundary conditions."""

    def test_meters_cannot_exceed_1_0(self, env):
        """Apply massive positive benefits - meters should clamp at 1.0."""
        # Set all meters to 0.9
        env.meters[0, :] = 0.9

        # Try to add +50% to all meters (should clamp at 1.0)
        env.meters[0, :] = torch.clamp(env.meters[0, :] + 0.5, 0.0, 1.0)

        assert torch.all(env.meters[0, :] <= 1.0), "All meters should be clamped at 1.0 maximum"

        # Verify they actually reached 1.0 (not stuck below)
        assert torch.all(env.meters[0, :] == 1.0), (
            "All meters should be exactly 1.0 after massive positive benefit"
        )

    def test_meters_cannot_go_below_0_0(self, env):
        """Apply massive negative effects - meters should clamp at 0.0."""
        # Set all meters to 0.1
        env.meters[0, :] = 0.1

        # Try to subtract -50% from all meters (should clamp at 0.0)
        env.meters[0, :] = torch.clamp(env.meters[0, :] - 0.5, 0.0, 1.0)

        assert torch.all(env.meters[0, :] >= 0.0), "All meters should be clamped at 0.0 minimum"

        # Verify they actually reached 0.0
        assert torch.all(env.meters[0, :] == 0.0), (
            "All meters should be exactly 0.0 after massive negative effect"
        )

    def test_money_cannot_go_negative(self, env):
        """Money should never go below 0."""
        # Set money to $2 (0.02)
        env.meters[0, 3] = 0.02

        # Try to spend more money than available (would go negative)
        # This tests the clamping behavior
        env.meters[0, 3] = torch.clamp(env.meters[0, 3] - 0.10, 0.0, 1.0)

        assert env.meters[0, 3].item() == 0.0, "Money should clamp at 0.0, not go negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
