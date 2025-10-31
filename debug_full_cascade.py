"""Debug ALL cascading effects on energy."""

import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv

print("=" * 70)
print("FULL CASCADE ANALYSIS - LOW SATIATION")
print("=" * 70)

env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device("cpu"))
env.reset()
env.positions[0] = torch.tensor([4, 4])
env.meters[0, 2] = 0.2  # Satiation at 20%

print("\nInitial all meters:")
meter_names = ["Energy", "Hygiene", "Satiation", "Money", "Mood", "Social", "Health", "Fitness"]
for i, name in enumerate(meter_names):
    print(f"  {name:10s}: {env.meters[0, i].item() * 100:5.1f}%")

print("\n" + "-" * 70)
print("Tracking ALL meters over 50 steps:")
print("-" * 70)

for step in [0, 10, 20, 30, 40, 50]:
    if step > 0:
        for _ in range(10):
            env.step(torch.tensor([4]))  # INTERACT

    print(f"\nStep {step}:")
    for i, name in enumerate(meter_names):
        print(f"  {name:10s}: {env.meters[0, i].item() * 100:5.1f}%", end="")
        if name in ["Hygiene", "Social"] and env.meters[0, i].item() < 0.3:
            print(" ⚠️  BELOW THRESHOLD", end="")
        print()

print("\n" + "=" * 70)
print("CASCADE ANALYSIS:")
print("=" * 70)
print("\nEnergy can be damaged by:")
print("  1. Base depletion: 0.5% per step")
print("  2. Movement cost: 0.5% per step (not in this test)")
print("  3. Low satiation (<30%): 0.5% * deficit per step")
print("  4. Low mood (<30%): 0.5% * deficit per step")
print("  5. Low hygiene (<30%): 0.05% * deficit per step (weak)")
print("  6. Low social (<30%): 0.08% * deficit per step (weak, indirect)")

print("\n🔍 INVESTIGATION:")
hygiene_final = env.meters[0, 1].item()
social_final = env.meters[0, 5].item()

if hygiene_final < 0.3:
    print(f"  ⚠️  Hygiene dropped to {hygiene_final * 100:.1f}% (below 30%)")
    print(f"      Adding weak energy drain!")

if social_final < 0.3:
    print(f"  ⚠️  Social dropped to {social_final * 100:.1f}% (below 30%)")
    print(f"      Cascading to mood → then mood cascades to energy!")
print("=" * 70)
