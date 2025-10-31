"""Debug low satiation cascading to energy."""

import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv

print("=" * 70)
print("LOW SATIATION → ENERGY CASCADE TEST")
print("=" * 70)

# Test with INTERACT (no movement cost)
env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device("cpu"))
env.reset()
env.positions[0] = torch.tensor([4, 4])  # Empty tile
env.meters[0, 2] = 0.2  # Set satiation to 20% (below 30% threshold)

print(f"\nInitial state:")
print(f"  Satiation: {env.meters[0, 2].item() * 100:.1f}%")
print(f"  Energy: {env.meters[0, 0].item() * 100:.1f}%")
print(f"\nExpected per-step costs (at 20% satiation):")
print(f"  Base energy depletion: 0.5%")
print(f"  Cascade penalty: 0.5% * (0.3-0.2)/0.3 = {0.005 * ((0.3 - 0.2) / 0.3) * 100:.2f}%")
print(f"  Total: {(0.005 + 0.005 * ((0.3 - 0.2) / 0.3)) * 100:.2f}% per step")
print(f"\nOver 50 steps:")
print(f"  Expected total: {(0.005 + 0.005 * ((0.3 - 0.2) / 0.3)) * 50 * 100:.1f}%")

print("\n" + "-" * 70)
print("Step-by-step tracking:")
print("-" * 70)

for step in range(10):
    before_energy = env.meters[0, 0].item()
    before_satiation = env.meters[0, 2].item()

    env.step(torch.tensor([4]))  # INTERACT - no movement cost

    after_energy = env.meters[0, 0].item()
    after_satiation = env.meters[0, 2].item()

    energy_loss = before_energy - after_energy
    satiation_loss = before_satiation - after_satiation

    print(
        f"Step {step + 1:2d}: Energy {before_energy * 100:5.1f}% → {after_energy * 100:5.1f}% "
        f"(loss: {energy_loss * 100:.2f}%) | "
        f"Satiation {before_satiation * 100:5.1f}% → {after_satiation * 100:5.1f}%"
    )

# Run remaining 40 steps
for _ in range(40):
    env.step(torch.tensor([4]))

print(f"\n" + "=" * 70)
print(f"After 50 steps (INTERACT only, no movement cost):")
print(f"  Energy: {env.meters[0, 0].item() * 100:.1f}%")
print(f"  Satiation: {env.meters[0, 2].item() * 100:.1f}%")
print(f"  Total energy loss: {(1.0 - env.meters[0, 0].item()) * 100:.1f}%")
print(f"  Average per step: {(1.0 - env.meters[0, 0].item()) / 50 * 100:.2f}%")
print("=" * 70)

# Now test with movement
print("\n" + "=" * 70)
print("COMPARISON: WITH MOVEMENT COST")
print("=" * 70)

env2 = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device("cpu"))
env2.reset()
env2.positions[0] = torch.tensor([0, 0])  # Top-left corner
env2.meters[0, 2] = 0.2  # Set satiation to 20%

initial_energy = env2.meters[0, 0].item()

for _ in range(50):
    env2.step(torch.tensor([0]))  # UP (blocked by boundary, but costs energy)

final_energy = env2.meters[0, 0].item()
total_loss = initial_energy - final_energy

print(f"\nInitial energy: {initial_energy * 100:.1f}%")
print(f"Final energy: {final_energy * 100:.1f}%")
print(f"Total loss: {total_loss * 100:.1f}%")
print(f"Per step: {total_loss / 50 * 100:.2f}%")
print(f"\nExpected per-step with movement:")
print(f"  Base: 0.5% + Movement: 0.5% + Cascade: 0.17% = 1.17% per step")
print(f"  Over 50 steps: ~58.3%")
print("=" * 70)
