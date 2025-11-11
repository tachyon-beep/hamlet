# Continuous Directional Movement System Design

**Status**: Design Phase
**Date**: 2025-11-11
**Context**: Add velocity tracking + continuous directional control for continuous substrates

## Overview

Enable agents in continuous substrates to:
1. **Observe their own velocity** (direction + magnitude) for navigation memory
2. **Output continuous directional actions** (variable direction + magnitude) instead of 8-way discrete movement
3. **Self-discover affordance locations** through exploration (no automatic proximity vectors)

## Current Architecture

### Observation Side (VFS)
- **Grid encoding**: Flattened occupancy grid (for Grid2D/3D)
- **Position**: Normalized [x, y] or [x, y, z] coordinates
- **Meters**: Energy, health, satiation, etc.
- **Affordance at position**: One-hot encoding of what's at agent's location
- **Temporal**: time_sin, time_cos, progress metrics

### Action Side (Discrete)
- **Q-Network**: Outputs Q-values for each discrete action
- **Action Selection**: argmax(Q-values) → action_id
- **Action Execution**: `action_configs[action_id].delta` → `substrate.apply_movement(delta)`
- **Grid2D actions**: 6 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
- **Continuous actions**: 6 actions (±X, ±Y, ±Z depending on dimensions, INTERACT, WAIT)

**Problem**: Discrete 8-way movement loses directional information in continuous space.

---

## Design: Velocity Observation

### VFS Variables (Auto-Generated)

Add to `_auto_generate_standard_variables()` for continuous/grid substrates:

```python
# Velocity tracking (for spatial substrates with movement)
variables.extend([
    VariableDef(
        id="velocity_x",
        scope="agent",
        type="scalar",
        lifetime="tick",
        readable_by=["agent", "engine"],
        writable_by=["engine"],
        default=0.0,
        description="X-component of agent velocity (movement since last step)",
    ),
    VariableDef(
        id="velocity_y",
        scope="agent",
        type="scalar",
        lifetime="tick",
        readable_by=["agent", "engine"],
        writable_by=["engine"],
        default=0.0,
        description="Y-component of agent velocity (movement since last step)",
    ),
    VariableDef(
        id="velocity_magnitude",
        scope="agent",
        type="scalar",
        lifetime="tick",
        readable_by=["agent", "engine"],
        writable_by=["engine"],
        default=0.0,
        description="Speed (magnitude of velocity vector)",
    ),
])

# 3D substrates also get velocity_z
if is_3d:
    variables.append(
        VariableDef(
            id="velocity_z",
            scope="agent",
            type="scalar",
            lifetime="tick",
            readable_by=["agent", "engine"],
            writable_by=["engine"],
            default=0.0,
            description="Z-component of agent velocity",
        )
    )
```

### Velocity Calculation

In `VectorizedHamletEnv`, track previous positions:

```python
# In __init__
self.prev_positions = self.positions.clone()

# After apply_movement in step()
velocity = self.positions - self.prev_positions
self.vfs_registry.set("velocity_x", velocity[:, 0].unsqueeze(1), writer="engine")
self.vfs_registry.set("velocity_y", velocity[:, 1].unsqueeze(1), writer="engine")

magnitude = torch.norm(velocity, dim=1, keepdim=True)
self.vfs_registry.set("velocity_magnitude", magnitude, writer="engine")

self.prev_positions = self.positions.clone()
```

### Observation Impact

**Before** (Continuous2D, 8×8 equivalent spacing):
- Observation dim: ~29 (2 position + 8 meters + 15 affordances + 4 temporal)

**After**:
- Observation dim: ~32 (2 position + 2 velocity + 1 magnitude + 8 meters + 15 affordances + 4 temporal)

VFS auto-exposes these, network adapts automatically via `obs_dim` parameter.

---

## Design: Continuous Action Space

### Option A: Discretized Continuous (Recommended - DQN Compatible)

**Keep Q-network architecture**, discretize the continuous space into bins.

#### Action Space Design

**Directional Actions** (16-32 directions):
```python
# 16 directions (22.5° resolution)
directions = [
    (1.0, 0.0),      # 0° East
    (0.924, 0.383),  # 22.5° ENE
    (0.707, 0.707),  # 45° NE
    (0.383, 0.924),  # 67.5° NNE
    (0.0, 1.0),      # 90° North
    # ... 16 total directions
]

# Magnitude levels (5 bins: 0%, 25%, 50%, 75%, 100%)
magnitudes = [0.0, 0.25, 0.50, 0.75, 1.0]

# Total movement actions = 16 directions × 5 magnitudes = 80 actions
# Plus non-movement: INTERACT, WAIT = 82 total actions
```

**Action Config Generation**:
```python
# In ActionSpaceBuilder for continuous substrates
for dir_idx, (dx, dy) in enumerate(directions):
    for mag_idx, magnitude in enumerate(magnitudes):
        if magnitude == 0.0:
            # Only create one "STOP" action, not 16 variants
            if dir_idx == 0:
                actions.append(ActionConfig(
                    id=action_id,
                    name="STOP",
                    type="passive",
                    delta=None,  # No movement
                    costs={},
                    effects={},
                    enabled=True,
                    source="substrate",
                ))
                action_id += 1
        else:
            # Scaled delta: direction × magnitude × movement_delta
            scaled_dx = dx * magnitude * movement_delta
            scaled_dy = dy * magnitude * movement_delta

            actions.append(ActionConfig(
                id=action_id,
                name=f"MOVE_{dir_idx}_{mag_idx}",  # MOVE_0_1, MOVE_0_2, etc.
                type="movement",
                delta=[scaled_dx, scaled_dy],  # Float deltas now!
                costs={"energy": base_move_cost * magnitude},  # Scaled by magnitude
                effects={},
                enabled=True,
                source="substrate",
            ))
            action_id += 1
```

**Config Parameter** (`substrate.yaml`):
```yaml
continuous:
  dimensions: 2
  bounds: [[0.0, 10.0], [0.0, 10.0]]
  boundary: "clamp"
  movement_delta: 0.5  # Maximum movement per step
  interaction_radius: 0.8
  distance_metric: "euclidean"

  # New: Continuous action discretization
  action_discretization:
    num_directions: 16  # 8, 16, or 32
    num_magnitudes: 5   # 3, 5, or 7
```

**Pros**:
- ✅ DQN compatible (no training loop changes)
- ✅ Deterministic action selection
- ✅ Simple to implement
- ✅ Fine-grained control (80+ movement options)

**Cons**:
- ❌ Action space explosion (16×5 = 80 actions)
- ❌ Not truly continuous (binning artifacts)
- ❌ Network must learn Q-values for many similar actions

---

### Option B: True Continuous Actions (Requires Policy Gradient)

**Switch to Actor-Critic** architecture (PPO/SAC).

#### Network Architecture

**Actor** (Policy Network):
```python
class ContinuousActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dims: list[int]):
        # action_dims = [2, 1] for [direction_xy, magnitude]
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Direction head: outputs angle in [-π, π]
        self.direction_mean = nn.Linear(128, 1)
        self.direction_log_std = nn.Parameter(torch.zeros(1))

        # Magnitude head: outputs speed in [0, 1]
        self.magnitude_mean = nn.Linear(128, 1)
        self.magnitude_log_std = nn.Parameter(torch.zeros(1))

        # Discrete action head (INTERACT, WAIT)
        self.discrete_head = nn.Linear(128, 2)

    def forward(self, obs):
        features = self.shared(obs)

        # Continuous outputs (for movement action)
        dir_mean = self.direction_mean(features)
        dir_std = torch.exp(self.direction_log_std)

        mag_mean = torch.sigmoid(self.magnitude_mean(features))  # [0, 1]
        mag_std = torch.exp(self.magnitude_log_std)

        # Discrete logits (INTERACT vs WAIT)
        discrete_logits = self.discrete_head(features)

        return (dir_mean, dir_std), (mag_mean, mag_std), discrete_logits
```

**Critic** (Value Network):
```python
class ContinuousCriticNetwork(nn.Module):
    def __init__(self, obs_dim: int):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),  # V(s)
        )

    def forward(self, obs):
        return self.net(obs)
```

**Action Execution**:
```python
# Sample from distributions
direction_dist = Normal(dir_mean, dir_std)
direction = direction_dist.sample()  # angle in radians

magnitude_dist = Normal(mag_mean, mag_std)
magnitude = torch.clamp(magnitude_dist.sample(), 0.0, 1.0)

# Convert to delta
dx = torch.cos(direction) * magnitude * movement_delta
dy = torch.sin(direction) * magnitude * movement_delta
delta = torch.stack([dx, dy], dim=-1)

# Apply movement
new_positions = substrate.apply_movement(positions, delta)
```

**Pros**:
- ✅ Truly continuous control
- ✅ No action space explosion
- ✅ Smoother movement trajectories
- ✅ Industry-standard for continuous control

**Cons**:
- ❌ Requires PPO/SAC training loop (major change)
- ❌ More complex training (clip ratios, entropy bonuses)
- ❌ Stochastic policies (harder to debug)
- ❌ Not compatible with current DQN infrastructure

---

### Option C: Hybrid Discrete-Continuous

**Keep 8 discrete directions**, add continuous magnitude.

#### Network Architecture

```python
class HybridQNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_directions: int = 8):
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Discrete Q-values for direction+meta-actions
        # Actions: 8 directions + INTERACT + WAIT = 10
        self.direction_q_head = nn.Linear(128, num_directions + 2)

        # Continuous magnitude output [0, 1]
        self.magnitude_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs):
        features = self.shared(obs)
        direction_q_values = self.direction_q_head(features)
        magnitude = self.magnitude_head(features)
        return direction_q_values, magnitude
```

**Action Selection**:
```python
# Standard argmax for direction
direction_idx = torch.argmax(direction_q_values, dim=-1)

# Use predicted magnitude
magnitude = magnitude_output.squeeze(-1)

# Combine into movement delta
if direction_idx < 8:  # Movement action
    base_delta = DIRECTION_VECTORS[direction_idx]  # (dx, dy)
    delta = base_delta * magnitude * movement_delta
else:  # INTERACT or WAIT
    delta = None
```

**Pros**:
- ✅ Mostly DQN compatible
- ✅ Fine-grained speed control
- ✅ Small action space (10 discrete actions)
- ✅ Magnitude gradient flows through network

**Cons**:
- ❌ Mixed discrete-continuous (unusual)
- ❌ Direction still discretized (8-way)
- ❌ Loss function needs two components (Q-loss + magnitude MSE)

---

## Recommended Implementation Path

### Phase 1: Velocity Observation (Week 1)
**Goal**: Add velocity tracking without changing action space.

1. Add VFS variables (`velocity_x`, `velocity_y`, `velocity_magnitude`)
2. Track `prev_positions` in environment
3. Compute velocity each step
4. Write to VFS registry
5. Auto-exposed as observations via VFS

**Test**: Train on Continuous2D with velocity observations, verify agents learn to use velocity info.

### Phase 2A: Discretized Continuous Actions (Week 2-3) - **Recommended First**
**Goal**: Add fine-grained directional control with minimal architectural change.

1. Add `action_discretization` config to `substrate.yaml`
2. Modify `ActionSpaceBuilder` to generate discretized continuous actions
3. Change `ActionConfig.delta` from `list[int]` to `list[float]`
4. Test with 16 directions × 5 magnitudes = 80 actions

**Test**: Train on Continuous2D, verify agents can navigate to arbitrary locations.

### Phase 2B: True Continuous Actions (Week 4-6) - **Optional Advanced**
**Goal**: Full continuous control with policy gradient.

1. Implement `ContinuousActorNetwork` + `ContinuousCriticNetwork`
2. Implement PPO training loop
3. Modify `VectorizedPopulation` to support continuous action sampling
4. Add config flag: `training.action_mode: "discrete" | "continuous"`

**Test**: Compare PPO continuous vs DQN discretized on navigation benchmarks.

---

## Integration Points

### 1. VFS Variables (Compiler)
**File**: `src/townlet/universe/compiler.py`
**Method**: `_auto_generate_standard_variables()`
**Change**: Add velocity variables for spatial substrates

### 2. Velocity Tracking (Environment)
**File**: `src/townlet/environment/vectorized_env.py`
**Method**: `__init__()`, `step()` (or internal movement method)
**Change**: Track `prev_positions`, compute velocity, write to VFS

### 3. Action Config Schema
**File**: `src/townlet/environment/action_config.py`
**Change**: `delta: list[int] | None` → `delta: list[float] | None`

### 4. Action Space Builder
**File**: `src/townlet/environment/action_space.py` (if exists) or `vectorized_env.py`
**Method**: `_build_action_space()`
**Change**: Generate discretized continuous actions for continuous substrates

### 5. Substrate Config Schema
**File**: `src/townlet/substrate/config.py`
**Change**: Add `action_discretization` field to `ContinuousConfig`

### 6. Network Architecture (Phase 2B only)
**File**: `src/townlet/agent/networks.py`
**Change**: Add `ContinuousActorNetwork`, `ContinuousCriticNetwork`

### 7. Training Loop (Phase 2B only)
**File**: `src/townlet/population/vectorized.py`
**Change**: Add PPO training loop option

---

## Configuration Example

### Continuous Substrate with Discretized Actions

```yaml
# configs/L1_continuous_2D_directional/substrate.yaml
version: "1.0"
description: "Continuous 2D with fine-grained directional control"

type: "continuous"

continuous:
  dimensions: 2
  bounds: [[0.0, 10.0], [0.0, 10.0]]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"

  # Discretized continuous actions
  action_discretization:
    num_directions: 16  # 22.5° resolution
    num_magnitudes: 5   # 0%, 25%, 50%, 75%, 100%
```

### Training Config

```yaml
# configs/L1_continuous_2D_directional/training.yaml
# No changes needed! Action space auto-generated from substrate config.
population:
  num_agents: 64

training:
  lr: 0.0003
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  # ... standard DQN hyperparams
```

---

## Testing Strategy

### Unit Tests
- `test_velocity_calculation()`: Verify velocity = position_delta
- `test_discretized_actions()`: Verify 16×5 = 80 actions generated
- `test_continuous_delta()`: Verify float deltas work with substrates

### Integration Tests
- `test_velocity_observation()`: Train agent, verify velocity in observations
- `test_directional_movement()`: Agent navigates to arbitrary (x,y)
- `test_magnitude_control()`: Agent learns when to move fast vs slow

### Benchmarks
- **Precision Navigation**: Reach target (x,y) within ε=0.1 radius
- **Speed Control**: Minimize energy by using low magnitudes when near target
- **Path Efficiency**: Compare path length discrete 8-way vs 16-way

---

## Migration Guide

### Backward Compatibility

**Grid2D/3D**: No changes, continue using discrete 8-way actions.

**Existing Continuous Configs**: Add `action_discretization` block, or default to 8 directions × 3 magnitudes.

**Checkpoints**: Incompatible if action space changes. Mark as breaking change in release notes.

---

## Future Extensions

### 1. Variable-Resolution Discretization
Allow curriculum progression: L1 (8 dirs), L2 (16 dirs), L3 (32 dirs).

### 2. Learned Action Parameterization
Network outputs direction + magnitude directly (hybrid approach).

### 3. Hierarchical Actions
High-level: "Navigate to fridge", Low-level: Directional movement.

### 4. Multi-Agent Collision Avoidance
Velocity observations enable learning collision-free navigation.

---

## Decision Matrix

| Criterion | Option A (Discretized) | Option B (True Continuous) | Option C (Hybrid) |
|-----------|----------------------|---------------------------|------------------|
| DQN Compatible | ✅ Yes | ❌ No (needs PPO) | ⚠️ Mostly |
| Fine-grained control | ✅ 16-32 directions | ✅ Infinite resolution | ⚠️ 8 directions |
| Action space size | ❌ 80-160 actions | ✅ 3D continuous | ✅ 10 discrete |
| Implementation effort | ✅ Low (1-2 weeks) | ❌ High (4-6 weeks) | ⚠️ Medium (2-3 weeks) |
| Training stability | ✅ DQN is stable | ⚠️ PPO needs tuning | ⚠️ Mixed losses |
| Industry standard | ⚠️ Custom approach | ✅ Standard (PPO/SAC) | ❌ Unusual pattern |

**Recommendation**: Start with **Option A** (Discretized), evaluate performance, optionally upgrade to **Option B** if needed.

---

## Open Questions

1. **Should velocity be normalized?** (Currently raw delta)
2. **Should we add acceleration?** (velocity_delta between steps)
3. **Energy costs**: Linear with magnitude or quadratic? (Currently linear)
4. **Action masking**: Discretized continuous still needs boundary checking?
5. **Network architecture**: Does DQN need dueling heads for 80+ actions?

---

## Sign-Off

- [ ] Architecture reviewed
- [ ] Performance benchmarks defined
- [ ] Migration path approved
- [ ] Implementation roadmap agreed
