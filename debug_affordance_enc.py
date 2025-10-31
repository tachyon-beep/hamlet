#!/usr/bin/env python3
"""Debug affordance encoding dimensions."""

import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv

# Create POMDP environment
env = VectorizedHamletEnv(
    num_agents=1,
    grid_size=8,
    partial_observability=True,
    vision_range=2,
    enable_temporal_mechanics=False,
    device=torch.device("cpu"),
)

print(f"Affordance names (types): {env.affordance_names}")
print(f"Number of affordance types: {env.num_affordance_types}")
print(f"Affordance encoding should be: num_types + 1 = {env.num_affordance_types + 1}")

# Get affordance encoding
obs = env.reset()
affordance_enc = env._get_current_affordance_encoding()
print(f"\nAffordance encoding shape: {affordance_enc.shape}")
print(f"Affordance encoding dimensions: {affordance_enc.shape[1]}")

# Break down observation
window_size = 5
grid_size = window_size * window_size
print(f"\nObservation breakdown:")
print(f"- Grid: {grid_size}")
print(f"- Position: 2")
print(f"- Meters: 8")
print(f"- Affordance: {affordance_enc.shape[1]}")
print(f"- Total: {grid_size + 2 + 8 + affordance_enc.shape[1]}")
print(f"- Actual observation: {obs.shape[1]}")
