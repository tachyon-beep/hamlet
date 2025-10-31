#!/usr/bin/env python3
"""Debug temporal environment affordance count."""

import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv

# Create env with temporal mechanics (like the failing tests)
env = VectorizedHamletEnv(
    num_agents=2,
    grid_size=8,
    device=torch.device("cpu"),
    enable_temporal_mechanics=True,
)

obs = env.reset()

print(f"Number of affordance types: {env.num_affordance_types}")
print(f"Affordance names: {env.affordance_names}")
print(f"Affordance encoding size (types + 1): {env.num_affordance_types + 1}")
print(f"\nObservation dimension: {env.observation_dim}")
print(f"Actual observation shape: {obs.shape}")
print(f"\nBreakdown:")
print(f"  Grid: 64")
print(f"  Meters: 8")
print(f"  Affordance encoding: {env.num_affordance_types + 1}")
print(f"  Temporal: 2")
print(f"  Total: {64 + 8 + (env.num_affordance_types + 1) + 2}")
