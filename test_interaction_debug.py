"""Minimal test to debug why interactions aren't happening in L0_5 training."""

from pathlib import Path

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.universe.compiler import UniverseCompiler

# Compile L0_5 config
config_dir = Path("configs/L0_5_dual_resource")
compiler = UniverseCompiler()
compiled_universe = compiler.compile(config_dir)

# Create environment
env = VectorizedHamletEnv.from_universe(compiled_universe, num_agents=1, device="cpu")

# Reset environment
obs = env.reset()

print("\n=== Environment Setup ===")
print(f"Grid size: {env.substrate.width}x{env.substrate.height}")
print(f"Number of affordances: {len(env.affordances)}")
print("Affordance positions:")
for name, pos in env.affordances.items():
    print(f"  {name}: {pos.tolist()}")
print(f"Initial agent position: {env.positions[0].tolist()}")
print(f"Initial money: ${env.meters[0, env.money_idx].item() * 100:.1f}")

# Test 1: Manual placement + INTERACT
print("\n=== Test 1: Manual Placement + INTERACT ===")
env.reset()

# Place agent ON Bed affordance
bed_pos = env.affordances["Bed"]
env.positions[0] = bed_pos.clone()
env.meters[0, env.energy_idx] = 0.3  # LOW ENERGY (so we can see restoration!)
env.meters[0, env.money_idx] = 0.5  # Ensure enough money

print(f"Agent position: {env.positions[0].tolist()}")
print(f"Bed position: {bed_pos.tolist()}")
print(f"Agent on Bed: {torch.equal(env.positions[0], bed_pos)}")

# Check initial energy
initial_energy = env.meters[0, env.energy_idx].item()
initial_money = env.meters[0, env.money_idx].item()
print(f"Initial energy: {initial_energy:.3f}")
print(f"Initial money: ${initial_money * 100:.1f}")

# Check Bed config
bed_config = env.affordance_engine.affordance_map.get("Bed")
print(f"Bed interaction type: {bed_config.interaction_type}")
print(f"Bed duration_ticks: {bed_config.duration_ticks}")
print(f"Bed cost (instant mode): ${env.affordance_engine.get_affordance_cost('Bed', 'instant') * 100:.1f}")
print(f"Bed has effect_pipeline: {hasattr(bed_config, 'effect_pipeline') and bed_config.effect_pipeline is not None}")
if hasattr(bed_config, "effect_pipeline") and bed_config.effect_pipeline:
    pipeline = bed_config.effect_pipeline
    print(f"  - per_tick effects: {len(pipeline.per_tick)} effects")
    print(f"  - on_completion effects: {len(pipeline.on_completion)} effects")
    print(f"  - on_start: {getattr(pipeline, 'on_start', 'MISSING')}")
    print(f"  - not pipeline.on_start: {not getattr(pipeline, 'on_start', None)}")

    # Check which branch would be taken in apply_interaction
    has_on_start = hasattr(pipeline, "on_start") and pipeline.on_start
    takes_dual_branch = bed_config.interaction_type == "dual" and not has_on_start and (pipeline.per_tick or pipeline.on_completion)
    print(f"  - Would take dual-mode instant branch: {takes_dual_branch}")
    if not takes_dual_branch:
        print(f"    WHY NOT: interaction_type={bed_config.interaction_type}, has_on_start={has_on_start}")
print(f"Temporal mechanics enabled: {env.enable_temporal_mechanics}")

# Take INTERACT action (id=4 for Grid2D)
interact_action = env.interact_action_idx
print(f"\nINTERACT action index: {interact_action}")

actions = torch.tensor([interact_action], dtype=torch.long, device=env.device)
obs, rewards, dones, info = env.step(actions)

final_energy = env.meters[0, env.energy_idx].item()
final_money = env.meters[0, env.money_idx].item()
print(f"\nFinal energy: {final_energy:.3f}")
print(f"Final money: ${final_money * 100:.1f}")
print(f"Energy change: {final_energy - initial_energy:+.3f}")
print(f"Money change: ${(final_money - initial_money) * 100:+.1f}")
print(f"Successful interactions: {info.get('successful_interactions', {})}")

if final_energy > initial_energy:
    print("✅ Test 1 PASSED - Interaction worked!")
else:
    print("❌ Test 1 FAILED - No interaction detected")

# Test 2: Random exploration
print("\n=== Test 2: Random Exploration (100 steps) ===")
env.reset()

interaction_count = 0
for step in range(100):
    # Random action
    action = torch.randint(0, env.action_dim, (1,), device=env.device)
    obs, rewards, dones, info = env.step(action)

    interactions = info.get("successful_interactions", {})
    if interactions:
        interaction_count += len(interactions)
        print(f"Step {step}: Interaction! {interactions}")

print(f"\nTotal interactions in 100 random steps: {interaction_count}")
if interaction_count > 0:
    print("✅ Test 2 PASSED - Random exploration found interactions")
else:
    print("⚠️  Test 2: Zero interactions (statistically unlikely but possible)")
