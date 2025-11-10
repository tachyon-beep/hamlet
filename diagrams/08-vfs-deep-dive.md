# VFS (Variable & Feature System) Deep Dive

## VFS Architecture Overview

```mermaid
graph TB
    subgraph "Configuration Layer"
        variables_yaml["variables_reference.yaml"]
        variable_defs["VariableDef entries"]
        exposure_spec["exposed_observations entries"]
    end

    subgraph "Compilation Stage"
        vfs_schema["VFS Schema<br/>Pydantic Models"]
        vfs_obs_builder["VFSObservationSpecBuilder"]
        vfs_adapter["vfs_to_observation_spec()<br/>Adapter"]
        observation_spec["ObservationSpec<br/>CompilerDTO"]
    end

    subgraph "Runtime Layer"
        vfs_registry["VariableRegistry<br/>Runtime Storage"]
        storage["Storage Tensors<br/>{id: Tensor}"]
        access_control["Access Control<br/>readable_by / writable_by"]
    end

    subgraph "Update Sources"
        engine_updates["Engine Updates<br/>position, meters, affordances"]
        action_updates["Action Updates<br/>custom action effects"]
        temporal_updates["Temporal Updates<br/>time_sin, time_cos"]
    end

    subgraph "Observation Construction"
        read_vars["Read Variables<br/>with access control"]
        normalize["Apply Normalization<br/>minmax / zscore"]
        concatenate["Concatenate Fields<br/>In order"]
        observations["Observations<br/>[num_agents, obs_dim]"]
    end

    variables_yaml --> variable_defs
    variables_yaml --> exposure_spec

    variable_defs --> vfs_schema
    exposure_spec --> vfs_schema

    vfs_schema --> vfs_obs_builder
    vfs_obs_builder --> vfs_adapter
    vfs_adapter --> observation_spec

    variable_defs --> vfs_registry
    vfs_registry --> storage
    vfs_registry --> access_control

    engine_updates --> storage
    action_updates --> storage
    temporal_updates --> storage

    storage --> read_vars
    access_control --> read_vars
    read_vars --> normalize
    normalize --> concatenate
    concatenate --> observations

    style vfs_schema fill:#e1f5fe
    style vfs_registry fill:#c8e6c9
    style observations fill:#fff9c4
```

## Variable Scope Hierarchy

```mermaid
graph TB
    subgraph "Global Scope"
        global_vars["Global Variables<br/>Single value, all agents share"]
        time_sin["time_sin: scalar<br/>Shape: []"]
        time_cos["time_cos: scalar<br/>Shape: []"]
        global_storage["Storage: Tensor([value])<br/>No agent dimension"]
    end

    subgraph "Agent Scope"
        agent_vars["Agent Variables<br/>Per-agent, publicly observable"]
        energy["energy: scalar<br/>Shape: [num_agents]"]
        position["position: vec2i<br/>Shape: [num_agents, 2]"]
        meters["meters (8): scalar<br/>Shape: [num_agents]"]
        agent_storage["Storage: Tensor([agent0, agent1, ...])<br/>Agent dimension first"]
    end

    subgraph "Agent-Private Scope"
        private_vars["Agent-Private Variables<br/>Per-agent, only owner observes"]
        home_pos["home_position: vec2i<br/>Shape: [num_agents, 2]"]
        internal_state["internal_state: scalar<br/>Shape: [num_agents]"]
        private_storage["Storage: Tensor([agent0, agent1, ...])<br/>Same as agent scope storage"]
    end

    global_vars --> time_sin
    global_vars --> time_cos
    time_sin --> global_storage
    time_cos --> global_storage

    agent_vars --> energy
    agent_vars --> position
    agent_vars --> meters
    energy --> agent_storage
    position --> agent_storage
    meters --> agent_storage

    private_vars --> home_pos
    private_vars --> internal_state
    home_pos --> private_storage
    internal_state --> private_storage

    style global_vars fill:#fff9c4
    style agent_vars fill:#c8e6c9
    style private_vars fill:#f3e5f5
```

## Access Control Matrix

```mermaid
graph TB
    subgraph "Readers (Who can read)"
        reader_agent["agent"]
        reader_engine["engine"]
        reader_acs["acs (Action Config System)"]
        reader_bac["bac (Brain-as-Code)"]
    end

    subgraph "Writers (Who can write)"
        writer_engine["engine"]
        writer_actions["actions (custom actions)"]
        writer_bac["bac (learned behaviors)"]
    end

    subgraph "Access Control Examples"
        energy_access["energy<br/>Readable: [agent, engine, acs, bac]<br/>Writable: [engine]"]
        position_access["position<br/>Readable: [agent, engine, acs, bac]<br/>Writable: [engine, actions]"]
        private_access["home_position<br/>Readable: [agent]  (owner only)<br/>Writable: [engine]"]
        global_access["time_sin<br/>Readable: [agent, engine, acs, bac]<br/>Writable: [engine]"]
    end

    reader_agent --> energy_access
    reader_engine --> energy_access
    reader_acs --> energy_access
    reader_bac --> energy_access

    reader_agent --> position_access
    reader_engine --> position_access
    reader_acs --> position_access
    reader_bac --> position_access
    writer_engine --> position_access
    writer_actions --> position_access

    reader_agent --> private_access
    writer_engine --> private_access

    reader_agent --> global_access
    reader_engine --> global_access
    writer_engine --> global_access

    style energy_access fill:#c8e6c9
    style position_access fill:#e1f5fe
    style private_access fill:#f3e5f5
    style global_access fill:#fff9c4
```

## Variable Type System

```mermaid
graph TB
    subgraph "Scalar Types"
        scalar["scalar<br/>Single float value<br/>default: 1.0"]
        bool_type["bool<br/>Single boolean value<br/>default: true"]
    end

    subgraph "Vector Types (Fixed)"
        vec2i["vec2i<br/>2D integer vector<br/>default: [0, 0]"]
        vec3i["vec3i<br/>3D integer vector<br/>default: [0, 0, 0]"]
    end

    subgraph "Vector Types (Dynamic)"
        vecNi["vecNi<br/>N-dimensional integer<br/>requires: dims field"]
        vecNf["vecNf<br/>N-dimensional float<br/>requires: dims field"]
    end

    subgraph "Storage Shapes"
        global_scalar["Global scalar: []<br/>tensor(1.0)"]
        agent_scalar["Agent scalar: [num_agents]<br/>tensor([1.0, 1.0, 1.0, 1.0])"]
        global_vec["Global vec2i: [2]<br/>tensor([0, 0])"]
        agent_vec["Agent vec2i: [num_agents, 2]<br/>tensor([[0,0], [0,0], [0,0], [0,0]])"]
    end

    scalar --> global_scalar
    scalar --> agent_scalar
    bool_type --> global_scalar
    bool_type --> agent_scalar

    vec2i --> global_vec
    vec2i --> agent_vec
    vec3i --> global_vec
    vec3i --> agent_vec

    vecNi --> agent_vec
    vecNf --> agent_vec

    style scalar fill:#c8e6c9
    style vec2i fill:#e1f5fe
    style vecNi fill:#fff9c4
```

## Observation Field Construction

```mermaid
flowchart TD
    start[Build Observations]

    subgraph "For Each exposed_observation"
        get_spec["Get ObservationField spec<br/>from variables_reference.yaml"]
        lookup_var["Lookup source_variable<br/>in VariableRegistry"]
        check_reader{"Reader<br/>allowed?"}
        read_tensor["Read tensor from storage<br/>registry.get(var_id, reader='agent')"]
        check_norm{"Normalization<br/>specified?"}
        apply_norm["Apply Normalization<br/>minmax or zscore"]
        shape_check["Verify shape matches spec"]
        field_ready["Field ready"]
    end

    concat["Concatenate all fields<br/>in spec order"]
    verify_dim["Verify total_dims matches<br/>metadata.observation_dim"]
    final_obs["Final Observations<br/>[num_agents, obs_dim]"]

    start --> get_spec
    get_spec --> lookup_var
    lookup_var --> check_reader
    check_reader -->|Yes| read_tensor
    check_reader -->|No| error1([Access Denied])
    read_tensor --> check_norm
    check_norm -->|Yes| apply_norm
    check_norm -->|No| shape_check
    apply_norm --> shape_check
    shape_check --> field_ready
    field_ready --> concat

    concat --> verify_dim
    verify_dim --> final_obs

    style read_tensor fill:#c8e6c9
    style apply_norm fill:#e1f5fe
    style final_obs fill:#fff9c4
```

## Normalization Types

```mermaid
graph TB
    subgraph "MinMax Normalization"
        minmax_scalar["Scalar:<br/>normalized = (value - min) / (max - min)<br/>Example: energy [0.0, 1.0] → [0.0, 1.0]"]
        minmax_vector["Vector:<br/>normalized[i] = (value[i] - min[i]) / (max[i] - min[i])<br/>Example: position [0,0] to [7,7] → [0,0] to [1,1]"]
    end

    subgraph "Z-Score Normalization"
        zscore_scalar["Scalar:<br/>normalized = (value - mean) / std<br/>Example: meter (mean=0.5, std=0.2)"]
        zscore_vector["Vector:<br/>normalized[i] = (value[i] - mean[i]) / std[i]<br/>Example: velocity distribution"]
    end

    subgraph "Configuration"
        minmax_spec["NormalizationSpec:<br/>kind: 'minmax'<br/>min: 0.0<br/>max: 1.0"]
        zscore_spec["NormalizationSpec:<br/>kind: 'zscore'<br/>mean: 0.5<br/>std: 0.2"]
    end

    minmax_spec --> minmax_scalar
    minmax_spec --> minmax_vector
    zscore_spec --> zscore_scalar
    zscore_spec --> zscore_vector

    style minmax_scalar fill:#c8e6c9
    style zscore_scalar fill:#e1f5fe
```

## Runtime Update Cycle

```mermaid
sequenceDiagram
    participant E as Environment
    participant R as VariableRegistry
    participant M as MeterDynamics
    participant A as AffordanceEngine
    participant T as TemporalMechanics
    participant O as ObservationBuilder

    Note over E,O: Environment Step

    E->>R: Initialize/Reset Variables
    R->>R: Set defaults for all vars

    loop Each Step
        E->>R: Update positions<br/>set("position", new_pos, writer="engine")
        R->>R: Validate writer permission
        R->>R: Store tensor[num_agents, 2]

        M->>R: Update meters<br/>set("energy", new_energy, writer="engine")
        R->>R: Store tensor[num_agents]

        A->>R: Update affordance_at_pos<br/>set("affordance_at_pos", aff_ids, writer="engine")
        R->>R: Store tensor[num_agents]

        alt Temporal Mechanics Enabled
            T->>R: Update time_sin<br/>set("time_sin", sin(tick), writer="engine")
            T->>R: Update time_cos<br/>set("time_cos", cos(tick), writer="engine")
            R->>R: Store global tensors
        end

        E->>O: Build Observations
        O->>R: Read all exposed variables<br/>get(var_id, reader="agent")
        R-->>O: Return tensors
        O->>O: Apply normalizations
        O->>O: Concatenate fields
        O-->>E: Observations [num_agents, obs_dim]
    end
```

## Example Variable Definitions

```mermaid
graph TB
    subgraph "Position Variable"
        pos_def["VariableDef:<br/>id: 'position'<br/>scope: 'agent'<br/>type: 'vec2i'<br/>lifetime: 'episode'<br/>readable_by: [agent, engine, acs, bac]<br/>writable_by: [engine, actions]<br/>default: [0, 0]"]
        pos_storage["Storage:<br/>Shape: [num_agents, 2]<br/>dtype: int64<br/>device: cuda"]
        pos_obs["Observation:<br/>id: 'obs_position'<br/>source_variable: 'position'<br/>exposed_to: [agent]<br/>shape: [2]<br/>normalization: minmax [0,0] to [7,7]"]
    end

    subgraph "Energy Variable"
        energy_def["VariableDef:<br/>id: 'energy'<br/>scope: 'agent'<br/>type: 'scalar'<br/>lifetime: 'episode'<br/>readable_by: [agent, engine, acs, bac]<br/>writable_by: [engine]<br/>default: 1.0"]
        energy_storage["Storage:<br/>Shape: [num_agents]<br/>dtype: float32<br/>device: cuda"]
        energy_obs["Observation:<br/>id: 'obs_energy'<br/>source_variable: 'energy'<br/>exposed_to: [agent]<br/>shape: []<br/>normalization: minmax [0.0, 1.0]"]
    end

    subgraph "Time Variable"
        time_def["VariableDef:<br/>id: 'time_sin'<br/>scope: 'global'<br/>type: 'scalar'<br/>lifetime: 'episode'<br/>readable_by: [agent, engine, acs, bac]<br/>writable_by: [engine]<br/>default: 0.0"]
        time_storage["Storage:<br/>Shape: []<br/>dtype: float32<br/>device: cuda"]
        time_obs["Observation:<br/>id: 'obs_time_sin'<br/>source_variable: 'time_sin'<br/>exposed_to: [agent]<br/>shape: []<br/>normalization: none"]
    end

    pos_def --> pos_storage
    pos_storage --> pos_obs

    energy_def --> energy_storage
    energy_storage --> energy_obs

    time_def --> time_storage
    time_storage --> time_obs

    style pos_def fill:#e1f5fe
    style energy_def fill:#c8e6c9
    style time_def fill:#fff9c4
```

## Full Observability Example

```mermaid
flowchart LR
    subgraph "Variables (Storage)"
        pos["position [4,2]<br/>[[2,3], [4,5], [1,1], [6,7]]"]
        energy["energy [4]<br/>[0.8, 0.6, 0.9, 0.4]"]
        health["health [4]<br/>[1.0, 0.7, 0.8, 0.5]"]
        aff["affordance_at_pos [4]<br/>[0, 3, 0, 7]  (IDs)"]
        time_sin["time_sin []<br/>0.5"]
        time_cos["time_cos []<br/>0.866"]
    end

    subgraph "Normalization"
        pos_norm["Normalize position<br/>to [0,1] range"]
        energy_norm["Already normalized<br/>[0,1]"]
        health_norm["Already normalized<br/>[0,1]"]
        aff_encode["One-hot encode<br/>15 affordance types"]
        time_identity["No normalization"]
    end

    subgraph "Observation Fields"
        obs_pos["position: [4, 2]"]
        obs_energy["energy: [4]"]
        obs_health["health: [4]"]
        obs_aff["affordance: [4, 15]"]
        obs_time["temporal: [4, 2]<br/>(broadcast global)"]
    end

    concat["Concatenate along dim=1<br/>[4, 2+1+1+15+2] = [4, 21]"]
    final["Observations<br/>[4, 21]"]

    pos --> pos_norm --> obs_pos
    energy --> energy_norm --> obs_energy
    health --> health_norm --> obs_health
    aff --> aff_encode --> obs_aff
    time_sin --> time_identity --> obs_time
    time_cos --> time_identity --> obs_time

    obs_pos --> concat
    obs_energy --> concat
    obs_health --> concat
    obs_aff --> concat
    obs_time --> concat

    concat --> final

    style pos fill:#e1f5fe
    style final fill:#c8e6c9
```

## POMDP (Partial Observability) Example

```mermaid
flowchart LR
    subgraph "Variables (Storage)"
        local_window["local_window [4,25]<br/>5×5 flattened grid<br/>around each agent"]
        pos["position [4,2]<br/>global coordinates"]
        energy["energy [4]"]
        health["health [4]"]
        aff["affordance_at_pos [4]"]
        time_sin["time_sin []"]
    end

    subgraph "Observation Fields"
        obs_window["local_window: [4, 25]<br/>NOT normalized<br/>(grid cell IDs)"]
        obs_pos["position: [4, 2]<br/>normalized to [0,1]"]
        obs_energy["energy: [4]<br/>already [0,1]"]
        obs_health["health: [4]<br/>already [0,1]"]
        obs_aff["affordance: [4, 15]<br/>one-hot encoded"]
        obs_time["temporal: [4, 2]<br/>broadcast global"]
    end

    concat["Concatenate<br/>[4, 25+2+1+1+15+2] = [4, 46]"]
    final_pomdp["POMDP Observations<br/>[4, 46]"]

    local_window --> obs_window
    pos --> obs_pos
    energy --> obs_energy
    health --> obs_health
    aff --> obs_aff
    time_sin --> obs_time

    obs_window --> concat
    obs_pos --> concat
    obs_energy --> concat
    obs_health --> concat
    obs_aff --> concat
    obs_time --> concat

    concat --> final_pomdp

    style local_window fill:#f3e5f5
    style final_pomdp fill:#c8e6c9
```

## Variable Lifecycle States

```mermaid
stateDiagram-v2
    [*] --> Defined: variables_reference.yaml

    Defined --> Validated: Schema validation<br/>(Pydantic)

    Validated --> Compiled: VFSObservationSpecBuilder<br/>builds spec

    Compiled --> Allocated: VariableRegistry.__init__<br/>allocate tensors

    Allocated --> Initialized: Set default values

    Initialized --> Active: Episode start

    Active --> Updated: Environment step<br/>engine writes

    Updated --> Read: Observation construction<br/>agent reads

    Read --> Updated: Next step

    Updated --> Reset: Episode end

    Reset --> Initialized: New episode

    note right of Validated
        Type checking:
        - scope: global/agent/agent_private
        - type: scalar/vec2i/vec3i/vecNi/vecNf/bool
        - Access control lists valid
    end note

    note right of Allocated
        Storage tensors created:
        - Global: shape [] or [dims]
        - Agent: shape [num_agents] or [num_agents, dims]
        - Device: cuda or cpu
    end note

    note right of Updated
        Writers update storage:
        - engine: most variables
        - actions: custom effects
        - bac: learned behaviors
    end note
```

## Performance Characteristics

### Memory Usage (4 agents)

| Variable Type | Scope | Storage Size | Example |
|---------------|-------|--------------|---------|
| scalar | global | 4 bytes | `time_sin: tensor(0.5)` |
| scalar | agent | 16 bytes | `energy: tensor([0.8, 0.6, 0.9, 0.4])` |
| vec2i | global | 16 bytes | `global_pos: tensor([3, 5])` |
| vec2i | agent | 32 bytes | `position: tensor([[2,3], [4,5], [1,1], [6,7]])` |
| bool | global | 1 byte | `daytime: tensor(True)` |
| bool | agent | 4 bytes | `alive: tensor([T, T, F, T])` |

### Access Control Overhead

- **Read**: O(1) dictionary lookup + permission check
- **Write**: O(1) dictionary lookup + permission check + tensor assignment
- **Initialization**: O(num_variables) tensor allocation

### Normalization Overhead

- **MinMax**: 2 ops per value (subtract, divide)
- **Z-Score**: 2 ops per value (subtract, divide)
- **Batch operations**: Vectorized across `num_agents`

## Common VFS Patterns

### Pattern 1: Adding a New Meter

```yaml
# variables_reference.yaml
variables:
  - id: "stamina"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs", "bac"]
    writable_by: ["engine"]
    default: 1.0

exposed_observations:
  - id: "obs_stamina"
    source_variable: "stamina"
    exposed_to: ["agent"]
    shape: []
    normalization:
      kind: "minmax"
      min: 0.0
      max: 1.0
```

### Pattern 2: Adding Temporal Features

```yaml
# Global temporal variables
variables:
  - id: "hour_of_day"
    scope: "global"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine"]
    writable_by: ["engine"]
    default: 0.0

exposed_observations:
  - id: "obs_hour"
    source_variable: "hour_of_day"
    exposed_to: ["agent"]
    shape: []
    normalization:
      kind: "minmax"
      min: 0.0
      max: 23.0
```

### Pattern 3: Private Agent State

```yaml
# Agent-private variable (hidden from other agents)
variables:
  - id: "internal_motivation"
    scope: "agent_private"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent"]  # Only owner can read
    writable_by: ["bac"]    # BAC can write
    default: 0.5
```

## Error Handling

### Access Control Violations

```python
# Attempting to read without permission
try:
    value = registry.get("energy", reader="unauthorized")
except PermissionError as e:
    # "'unauthorized' is not allowed to read variable 'energy'"
    # "Readable by: ['agent', 'engine', 'acs', 'bac']"
```

### Missing Variables

```python
# Attempting to read non-existent variable
try:
    value = registry.get("nonexistent", reader="agent")
except KeyError as e:
    # "Variable 'nonexistent' not found in registry"
```

### Type Mismatches

```python
# Attempting to write wrong-shaped tensor
try:
    registry.set("position", wrong_shape_tensor, writer="engine")
except ValueError as e:
    # "Expected shape [num_agents, 2], got [num_agents, 3]"
```

## Future Extensions (Phase 2)

1. **Expression Parsing**: Parse `WriteSpec.expression` into AST for validation
2. **Derivation Graphs**: Compute variables from other variables (e.g., `speed = ||velocity||`)
3. **Complex Types**: Structured types (objects, arrays)
4. **Dynamic Shapes**: Variables with runtime-determined dimensions
5. **Temporal Queries**: Access historical variable values (e.g., `energy[t-5]`)
6. **Conditional Normalization**: Apply different normalization based on conditions
7. **Multi-Agent Coordination**: Cross-agent variable dependencies
