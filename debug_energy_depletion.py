"""Quick debugging script to understand energy depletion."""

import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv

# Create environment
env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device("cpu"))
env.reset()

print("Initial meters:")
print(f"  Energy: {env.meters[0, 0].item() * 100:.1f}%")

# Step 10 times and track energy
for step in range(10):
    before_energy = env.meters[0, 0].item()
    action = torch.tensor([0])  # UP action
    obs, reward, done, info = env.step(action)
    after_energy = env.meters[0, 0].item()

    energy_loss = before_energy - after_energy
    print(
        f"Step {step + 1}: Energy {before_energy * 100:.2f}% → "
        f"{after_energy * 100:.2f}% (loss: {energy_loss * 100:.3f}%)"
    )

    if done[0]:
        print(f"  DIED at step {step + 1}")
        break

print(f"\nFinal energy: {env.meters[0, 0].item() * 100:.1f}%")
