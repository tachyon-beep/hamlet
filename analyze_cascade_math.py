#!/usr/bin/env python3
"""
Analyze if the 41.8% energy loss from low satiation is mathematically correct.
"""

# Energy loss sources:
# 1. Base passive depletion: 0.5% per step
# 2. Low satiation cascade: 0.5% * deficit per step (when satiation < 30%)


def calculate_expected_energy_loss():
    """Calculate step-by-step expected energy loss."""

    energy = 100.0
    satiation = 20.0  # Starting satiation (already below 30% threshold)

    total_passive = 0.0
    total_cascade = 0.0

    print("Step-by-step breakdown:")
    print("-" * 80)

    for step in range(51):
        if step % 10 == 0:
            deficit = max(0.0, 0.3 - satiation / 100.0)
            cascade_penalty = 0.5 * deficit if satiation < 30.0 else 0.0

            print(
                f"Step {step:2d}: Energy={energy:5.1f}%, Satiation={satiation:4.1f}%, "
                f"Deficit={deficit:.3f}, Cascade={cascade_penalty:.4f}%"
            )

        # Calculate penalties for this step
        passive_loss = 0.5

        # Calculate deficit (max 30%, since threshold is 30%)
        deficit = max(0.0, 0.3 - satiation / 100.0)
        cascade_loss = 0.5 * deficit if satiation < 30.0 else 0.0

        # Apply penalties
        energy -= passive_loss
        energy -= cascade_loss

        # Track totals
        total_passive += passive_loss
        total_cascade += cascade_loss

        # Satiation also depletes (0.4% per step)
        satiation -= 0.4
        satiation = max(0.0, satiation)

    print("-" * 80)
    print(f"\nFinal Energy: {energy:.1f}%")
    print(f"Total Energy Loss: {100.0 - energy:.1f}%")
    print(f"\nBreakdown:")
    print(f"  Passive depletion (0.5% × 50): {total_passive:.1f}%")
    print(f"  Cascade penalties (variable):  {total_cascade:.1f}%")
    print(f"  Total:                          {total_passive + total_cascade:.1f}%")

    return 100.0 - energy, total_passive, total_cascade


if __name__ == "__main__":
    print("=" * 80)
    print("EXPECTED ENERGY LOSS CALCULATION")
    print("=" * 80)
    print()

    loss, passive, cascade = calculate_expected_energy_loss()

    print("\n" + "=" * 80)
    print("COMPARISON WITH ACTUAL TEST RESULTS")
    print("=" * 80)
    print(f"\nExpected total loss: {loss:.1f}%")
    print(f"Actual test loss:    41.8%")
    print(f"Difference:          {abs(loss - 41.8):.1f}%")
    print()

    if abs(loss - 41.8) < 1.0:
        print("✅ MATCH: Cascade calculation is CORRECT!")
        print("   The 41.8% loss is mathematically expected.")
    else:
        print("❌ MISMATCH: Cascade calculation may have a bug!")
        print(f"   Expected {loss:.1f}% but got 41.8%")
