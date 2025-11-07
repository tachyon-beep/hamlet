# VFS Variables Configuration

**Status**: Phase 1 Implementation (TASK-002C)
**Version**: 1.0
**Last Updated**: 2025-11-07

## Overview

The Variable & Feature System (VFS) uses declarative YAML configuration to define state space variables, observation specs, and action dependencies. This document describes the `variables.yaml` schema and configuration patterns.

## File Location

Variables are defined in `configs/<config_pack>/variables.yaml` (future) or `variables_reference.yaml` (current test infrastructure).

## Schema Structure

```yaml
version: "1.0"

variables:
  - id: "energy"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["actions", "engine"]
    default: 1.0
    description: "Energy level [0.0-1.0]"

exposed_observations:
  - id: "obs_energy"
    source_variable: "energy"
    exposed_to: ["agent"]
    shape: []
    normalization:
      kind: "minmax"
      min: 0.0
      max: 1.0
```

## Variable Definition Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique variable identifier (e.g., "energy", "position") |
| `scope` | enum | Storage scope: "global", "agent", "agent_private" |
| `type` | enum | Data type: "scalar", "bool", "vec2i", "vec3i", "vecNi", "vecNf" |
| `lifetime` | enum | Lifecycle: "tick" (recomputed), "episode" (persistent) |
| `readable_by` | list[string] | Access control readers: ["agent", "engine", "acs", "bac"] |
| `writable_by` | list[string] | Access control writers: ["engine", "actions", "bac"] |
| `default` | varies | Initial value (scalar: float/bool, vector: list) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Human-readable description |
| `dims` | int | Dimension count (required for vecNi/vecNf types) |

## Scope Semantics

### Global Scope
**Storage**: Single value shared by all agents
**Shape**: `[]` (scalar) or `[dims]` (vector)
**Use Case**: Time-of-day, global flags, environment state

**Example**:
```yaml
- id: "time_sin"
  scope: "global"
  type: "scalar"
  lifetime: "tick"
  readable_by: ["agent", "engine"]
  writable_by: ["engine"]
  default: 0.0
  description: "sin(2π * time_of_day / 24) for cyclical time encoding"
```

### Agent Scope
**Storage**: Per-agent values, observable by all
**Shape**: `[num_agents]` (scalar) or `[num_agents, dims]` (vector)
**Use Case**: Meters, position, visible state

**Example**:
```yaml
- id: "energy"
  scope: "agent"
  type: "scalar"
  lifetime: "episode"
  readable_by: ["agent", "engine", "acs"]
  writable_by: ["actions", "engine"]
  default: 1.0
  description: "Energy level [0.0-1.0]"
```

### Agent_Private Scope
**Storage**: Per-agent values, observable only by owner
**Shape**: `[num_agents]` (scalar) or `[num_agents, dims]` (vector)
**Use Case**: Hidden state, private rewards, theory of mind

**Example**:
```yaml
- id: "internal_motivation"
  scope: "agent_private"
  type: "scalar"
  lifetime: "episode"
  readable_by: ["engine"]  # Agent cannot observe own motivation directly
  writable_by: ["engine"]
  default: 0.5
  description: "Internal motivation level (not exposed to agent)"
```

## Type System

### Scalar Types

**scalar**: Single float value
**Storage**: `torch.float32`
**Default Example**: `1.0`

**bool**: Boolean flag
**Storage**: `torch.bool`
**Default Example**: `true`

### Vector Types

**vec2i**: 2D integer vector (e.g., grid position)
**Storage**: `torch.long`, shape `[2]` (global) or `[num_agents, 2]` (agent)
**Default Example**: `[0, 0]`

**vec3i**: 3D integer vector
**Storage**: `torch.long`, shape `[3]` (global) or `[num_agents, 3]` (agent)
**Default Example**: `[0, 0, 0]`

**vecNi**: N-dimensional integer vector
**Storage**: `torch.long`, shape `[dims]` or `[num_agents, dims]`
**Requires**: `dims` field
**Default Example**: `[0, 0, 0, 0, 0]` (5D)

**vecNf**: N-dimensional float vector
**Storage**: `torch.float32`, shape `[dims]` or `[num_agents, dims]`
**Requires**: `dims` field
**Default Example**: `[0.0, 0.0, 0.0]` (3D)

## Lifetime Semantics

### tick
**Behavior**: Recomputed every tick (derived state)
**Use Case**: Grid encoding, affordance at position, interaction progress
**Example**: Substrate observation encoding (changes with agent position)

### episode
**Behavior**: Persistent across ticks (stateful)
**Use Case**: Meters, position, accumulated rewards
**Example**: Energy level (drains over time, restored by interactions)

## Access Control

### Readers
- **agent**: Agent networks can read this variable for decision-making
- **engine**: Environment engine can read for dynamics/rendering
- **acs**: Adversarial Curriculum System can read for difficulty adjustment
- **bac**: Behavioral Action Compiler can read for action execution

### Writers
- **engine**: Environment engine updates this variable (substrate dynamics)
- **actions**: Actions modify this variable (movement, interactions)
- **bac**: Behavioral Action Compiler writes computed values

**Permission Validation**: Registry enforces access control at runtime via `get()` and `set()` methods.

## Observation Exposure

Observations map variables to agent observation space with optional normalization.

### Observation Field Schema

```yaml
exposed_observations:
  - id: "obs_position"
    source_variable: "position"
    exposed_to: ["agent"]
    shape: [2]
    normalization:
      kind: "minmax"
      min: [0.0, 0.0]
      max: [1.0, 1.0]
```

### Normalization Options

**minmax**: Scale to [0, 1] or custom range
```yaml
normalization:
  kind: "minmax"
  min: 0.0  # scalar or list for vectors
  max: 1.0
```

**zscore**: Standardize to zero mean, unit variance
```yaml
normalization:
  kind: "zscore"
  mean: 0.5  # scalar or list for vectors
  std: 0.2
```

**null**: No normalization (raw values)
```yaml
normalization: null
```

## Complete Example: L1_full_observability

```yaml
version: "1.0"

variables:
  # Substrate Encoding (66 dims)
  - id: "grid_encoding"
    scope: "agent"
    type: "vecNf"
    dims: 64
    lifetime: "tick"
    readable_by: ["agent", "engine"]
    writable_by: ["engine"]
    default: [0.0, 0.0, ...]  # 64 zeros
    description: "8×8 grid encoding (0=empty, 1=agent, 2=affordance, 3=both)"

  - id: "position"
    scope: "agent"
    type: "vecNf"
    dims: 2
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["actions", "engine"]
    default: [0.0, 0.0]
    description: "Normalized agent position (x, y) in [0, 1] range"

  # Meters (8 dims)
  - id: "energy"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["actions", "engine"]
    default: 1.0
    description: "Energy level [0.0-1.0]"

  # ... (other 7 meters)

  # Affordance at Position (15 dims)
  - id: "affordance_at_position"
    scope: "agent"
    type: "vecNf"
    dims: 15
    lifetime: "tick"
    readable_by: ["agent", "engine"]
    writable_by: ["engine"]
    default: [0.0, 0.0, ..., 1.0]  # 15-element one-hot
    description: "One-hot affordance type at agent position"

  # Temporal Features (4 dims)
  - id: "time_sin"
    scope: "global"
    type: "scalar"
    lifetime: "tick"
    readable_by: ["agent", "engine"]
    writable_by: ["engine"]
    default: 0.0
    description: "sin(2π * time_of_day / 24)"

exposed_observations:
  - id: "obs_grid_encoding"
    source_variable: "grid_encoding"
    exposed_to: ["agent"]
    shape: [64]
    normalization: null

  - id: "obs_position"
    source_variable: "position"
    exposed_to: ["agent"]
    shape: [2]
    normalization:
      kind: "minmax"
      min: [0.0, 0.0]
      max: [1.0, 1.0]

  - id: "obs_energy"
    source_variable: "energy"
    exposed_to: ["agent"]
    shape: []
    normalization:
      kind: "minmax"
      min: 0.0
      max: 1.0

  # ... (other observations)
```

**Total Observation Dimension**: 66 + 8 + 15 + 4 = **93 dims**

## Configuration Patterns

### Pattern 1: Simple Meter
```yaml
- id: "health"
  scope: "agent"
  type: "scalar"
  lifetime: "episode"
  readable_by: ["agent", "engine"]
  writable_by: ["engine"]
  default: 1.0
  description: "Health level [0.0-1.0]"
```

### Pattern 2: Normalized Position
```yaml
- id: "position"
  scope: "agent"
  type: "vecNf"
  dims: 2
  lifetime: "episode"
  readable_by: ["agent", "engine", "acs"]
  writable_by: ["actions", "engine"]
  default: [0.0, 0.0]
  description: "Normalized position [0, 1]²"
```

### Pattern 3: Global Time Signal
```yaml
- id: "time_cos"
  scope: "global"
  type: "scalar"
  lifetime: "tick"
  readable_by: ["agent", "engine"]
  writable_by: ["engine"]
  default: 1.0
  description: "cos(2π * time_of_day / 24)"
```

### Pattern 4: Derived State (tick lifetime)
```yaml
- id: "interaction_progress"
  scope: "agent"
  type: "scalar"
  lifetime: "tick"
  readable_by: ["agent", "engine"]
  writable_by: ["engine"]
  default: 0.0
  description: "Normalized interaction progress [0-1]"
```

## Best Practices

### 1. Checkpoint Compatibility
**CRITICAL**: Changing observation dimensions breaks all existing checkpoints!

- Always run regression tests after modifying variables
- Use `pytest tests/test_townlet/unit/vfs/test_observation_dimension_regression.py`
- Expected dimensions documented in test file

### 2. Access Control Design
- **Principle of Least Privilege**: Only grant necessary read/write access
- **Agent-readable**: Variables agent needs for decision-making
- **Engine-writable**: Variables controlled by environment dynamics
- **ACS-readable**: Variables needed for curriculum adjustment

### 3. Scope Selection
- **global**: Time, weather, global events (rare, use sparingly)
- **agent**: Most game state (meters, position, observable state)
- **agent_private**: Hidden state, internal motivation (advanced use)

### 4. Type Selection
- **scalar**: Single values (meters, flags, progress)
- **vecNf**: Continuous vectors (normalized positions, velocities)
- **vecNi**: Discrete vectors (grid coordinates, indices)

### 5. Normalization Strategy
- **Meters [0, 1]**: Use minmax with min=0, max=1
- **Positions**: Normalized by substrate automatically
- **Time signals**: Already in [-1, 1], use normalization=null
- **One-hot encodings**: No normalization needed

## Validation

### Compile-Time Validation (Pydantic)
- Schema validation on YAML load
- Type checking (scalar must not have dims, vecNi must have dims)
- Scope validation (global/agent/agent_private only)

### Runtime Validation
- Access control enforcement in VariableRegistry
- Shape validation on tensor operations
- Device consistency (CPU/CUDA)

### Regression Tests
- Dimension compatibility tests (Cycle 5)
- Integration tests (Cycle 6)
- End-to-end pipeline validation

## Integration with Other Systems

### ActionConfig Integration (Phase 1)
Actions declare variable dependencies via `reads` and `writes` fields:

```yaml
# In global_actions.yaml
actions:
  - id: 0
    name: "MOVE_UP"
    type: "movement"
    delta: [0, -1]
    costs: {energy: 0.005}
    reads: ["position", "energy"]  # VFS integration
    writes:
      - variable_id: "position"
        expression: "position + delta"
```

### BAC Integration (Phase 2 - Future)
Behavioral Action Compiler will:
1. Parse `reads`/`writes` specifications
2. Generate efficient tensor operations
3. Enforce variable dependencies at compile time
4. Optimize batch operations

## Migration Guide

### From Hardcoded Observations to VFS

**Before** (hardcoded in environment):
```python
def _get_observation(self):
    obs = torch.cat([
        self.substrate.get_encoding(),
        self.meters.get_values(),
        # ... hardcoded concatenation
    ])
    return obs
```

**After** (VFS-driven):
```python
def _get_observation(self):
    builder = VFSObservationSpecBuilder()
    spec = builder.build_observation_spec(self.variables, self.exposures)
    obs = []
    for field in spec:
        value = self.registry.get(field.source_variable, reader="agent")
        obs.append(value)
    return torch.cat(obs)
```

## Reference Files

Current test infrastructure uses reference variable files:
- `configs/L0_0_minimal/variables_reference.yaml` (38 dims)
- `configs/L0_5_dual_resource/variables_reference.yaml` (78 dims)
- `configs/L1_full_observability/variables_reference.yaml` (93 dims)
- `configs/L2_partial_observability/variables_reference.yaml` (54 dims)
- `configs/L3_temporal_mechanics/variables_reference.yaml` (93 dims)

**Note**: These files are part of the regression test infrastructure and should NOT be deleted. They validate VFS dimension calculations match current hardcoded implementation.

## See Also

- `docs/plans/2025-11-06-variables-and-features-system.md` - VFS design document
- `src/townlet/vfs/schema.py` - Pydantic schema definitions
- `src/townlet/vfs/registry.py` - Runtime variable storage
- `src/townlet/vfs/observation_builder.py` - Observation spec builder
- `tests/test_townlet/unit/vfs/` - VFS unit tests
- `tests/test_townlet/integration/test_vfs_integration.py` - Integration tests
