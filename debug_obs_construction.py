#!/usr/bin/env python3
"""Debug temporal observation construction."""

import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv

# Create env with temporal mechanics
env = VectorizedHamletEnv(
    num_agents=1,
    grid_size=8,
    device=torch.device("cpu"),
    enable_temporal_mechanics=True,
)

obs = env.reset()

# Get affordance encoding separately
affordance_enc = env._get_current_affordance_encoding()

# Get full obs without temporal
env_no_temporal = VectorizedHamletEnv(
    num_agents=1,
    grid_size=8,
    device=torch.device("cpu"),
    enable_temporal_mechanics=False,
)
obs_no_temporal = env_no_temporal.reset()

print(f"Affordance encoding shape: {affordance_enc.shape}")
print(f"Observation without temporal: {obs_no_temporal.shape}")
print(f"Observation with temporal: {obs.shape}")
print(f"Difference: {obs.shape[1] - obs_no_temporal.shape[1]} (should be 2)")
print(f"\nEnvironment says observation_dim: {env.observation_dim}")
print(f"Actual observation dim: {obs.shape[1]}")
print(f"Mismatch: {obs.shape[1] - env.observation_dim}")
