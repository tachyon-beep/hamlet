#!/usr/bin/env python3
"""Quick validation test for curriculum difficulty and reward fixes."""

import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.runtime_registry import AgentRuntimeRegistry


def test_baseline_calculation():
    """Test that baseline survival is calculated correctly."""
    print("=" * 60)
    print("Test 1: Baseline Survival Calculation")
    print("=" * 60)

    device = torch.device("cpu")
    env = VectorizedHamletEnv(num_agents=1, device=device)

    # Stage 1: 0.2 multiplier
    baseline_stage1 = env.calculate_baseline_survival(0.2)
    print(f"\n✓ Stage 1 (20% difficulty): {baseline_stage1:.1f} steps")
    print(f"  Expected: ~167 steps")
    assert 160 < baseline_stage1 < 175, f"Stage 1 baseline out of range: {baseline_stage1}"

    # Stage 2: 0.5 multiplier
    baseline_stage2 = env.calculate_baseline_survival(0.5)
    print(f"\n✓ Stage 2 (50% difficulty): {baseline_stage2:.1f} steps")
    print(f"  Expected: ~133 steps")
    assert 130 < baseline_stage2 < 140, f"Stage 2 baseline out of range: {baseline_stage2}"

    # Stage 4: 1.0 multiplier
    baseline_stage4 = env.calculate_baseline_survival(1.0)
    print(f"\n✓ Stage 4 (100% difficulty): {baseline_stage4:.1f} steps")
    print(f"  Expected: ~100 steps")
    assert 95 < baseline_stage4 < 105, f"Stage 4 baseline out of range: {baseline_stage4}"

    print("\n✅ All baseline calculations correct!\n")


def test_reward_function():
    """Test that reward function works correctly."""
    print("=" * 60)
    print("Test 2: Reward Function")
    print("=" * 60)

    device = torch.device("cpu")
    env = VectorizedHamletEnv(num_agents=3, device=device)
    registry = AgentRuntimeRegistry(agent_ids=[f"agent-{i}" for i in range(3)], device=device)
    env.attach_runtime_registry(registry)

    # Set Stage 1 baseline
    baseline_tensor = env.update_baseline_for_curriculum(0.2)
    registry.set_baselines(baseline_tensor)
    baseline = baseline_tensor[0].item()
    print(f"\n✓ Baseline set to: {baseline:.1f} steps")

    # Test different survival times
    step_counts = torch.tensor([100, 167, 250], device=device)
    dones = torch.tensor([True, True, True], device=device)

    rewards = env.reward_strategy.calculate_rewards(step_counts, dones, registry.get_baseline_tensor())

    print(f"\n✓ Agent 1: 100 steps → reward = {rewards[0]:.1f}")
    print(f"  Expected: ~-67 (below baseline)")

    print(f"\n✓ Agent 2: 167 steps → reward = {rewards[1]:.1f}")
    print(f"  Expected: ~0 (at baseline)")

    print(f"\n✓ Agent 3: 250 steps → reward = {rewards[2]:.1f}")
    print(f"  Expected: ~+83 (above baseline, POSITIVE!)")

    # Verify signs
    assert rewards[0] < 0, "Below-baseline reward should be negative"
    assert abs(rewards[1]) < 10, "At-baseline reward should be near zero"
    assert rewards[2] > 0, "Above-baseline reward should be POSITIVE"

    print("\n✅ Reward function working correctly!\n")


def test_curriculum_multiplier():
    """Test that curriculum multiplier is applied to depletion."""
    print("=" * 60)
    print("Test 3: Curriculum Difficulty Multiplier")
    print("=" * 60)

    device = torch.device("cpu")
    env = VectorizedHamletEnv(num_agents=1, device=device)

    env.reset()
    initial_energy = env.meters[0, 0].item()
    print(f"\n✓ Initial energy: {initial_energy:.4f}")

    # Step with Stage 1 difficulty (0.2 multiplier)
    actions = torch.tensor([4], device=device)  # INTERACT (no movement)
    env.step(actions, depletion_multiplier=0.2)

    energy_after_stage1 = env.meters[0, 0].item()
    stage1_depletion = initial_energy - energy_after_stage1

    print(f"\n✓ Stage 1 (20% difficulty):")
    print(f"  Energy after 1 step: {energy_after_stage1:.4f}")
    print(f"  Depletion: {stage1_depletion:.4f}")
    print(f"  Expected: ~0.001 (0.005 * 0.2)")

    # Reset and test full difficulty
    env.reset()
    env.step(actions, depletion_multiplier=1.0)

    energy_after_full = env.meters[0, 0].item()
    full_depletion = initial_energy - energy_after_full

    print(f"\n✓ Stage 4 (100% difficulty):")
    print(f"  Energy after 1 step: {energy_after_full:.4f}")
    print(f"  Depletion: {full_depletion:.4f}")
    print(f"  Expected: ~0.005 (0.005 * 1.0)")

    # Verify multiplier effect
    ratio = full_depletion / stage1_depletion
    print(f"\n✓ Depletion ratio: {ratio:.1f}x")
    print(f"  Expected: ~5.0x (1.0 / 0.2)")

    assert 4.5 < ratio < 5.5, f"Multiplier not applied correctly: ratio={ratio}"

    print("\n✅ Curriculum multiplier working correctly!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VALIDATION TESTS: Curriculum Difficulty & Reward Fixes")
    print("=" * 60 + "\n")

    try:
        test_baseline_calculation()
        test_reward_function()
        test_curriculum_multiplier()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSystem is ready for training. Run:")
        print("  python run_demo.py --config configs/level_1_full_observability.yaml --episodes 100")
        print("\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise
