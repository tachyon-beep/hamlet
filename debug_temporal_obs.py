#!/usr/bin/env python3
"""Debug temporal observation dimensions."""

import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv

# Create POMDP environment with temporal mechanics
env = VectorizedHamletEnv(
    num_agents=1,
    grid_size=8,
    partial_observability=True,
    vision_range=2,  # 5×5 window
    enable_temporal_mechanics=True,  # +2 dims
    device=torch.device("cpu"),
)

print(f"Environment observation_dim: {env.observation_dim}")
print(f"Expected: 25 (grid) + 2 (pos) + 8 (meters) + 16 (affordances) + 2 (temporal) = 53")

# Reset and get actual observation
obs = env.reset()
print(f"Actual observation shape: {obs.shape}")
print(f"Actual observation dims: {obs.shape[1]}")

# Let's also check without temporal
env_no_temporal = VectorizedHamletEnv(
    num_agents=1,
    grid_size=8,
    partial_observability=True,
    vision_range=2,  # 5×5 window
    enable_temporal_mechanics=False,  # No temporal
    device=torch.device("cpu"),
)

print(f"\nWithout temporal:")
print(f"Environment observation_dim: {env_no_temporal.observation_dim}")
print(f"Expected: 25 (grid) + 2 (pos) + 8 (meters) + 16 (affordances) = 51")

obs_no_temporal = env_no_temporal.reset()
print(f"Actual observation shape: {obs_no_temporal.shape}")
print(f"Actual observation dims: {obs_no_temporal.shape[1]}")

print(f"\n Difference: {obs.shape[1] - obs_no_temporal.shape[1]} (should be 2)")
