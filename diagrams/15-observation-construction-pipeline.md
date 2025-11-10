# Observation Construction Pipeline

## Overview

This document details how observations flow from YAML configuration through compilation into runtime tensors that neural networks consume. The pipeline has three phases:

1. **Compile-Time**: VFS schema validation → observation spec generation
2. **Initialization-Time**: Registry setup → substrate encoding configuration
3. **Step-Time**: State update → VFS registry → observation tensor construction

## Pipeline Stages

```mermaid
graph TB
    subgraph "Phase 1: Compile-Time (UniverseCompiler)"
        yaml["variables_reference.yaml<br/>Variable definitions + exposures"]
        schema_validation["VFS Schema Validation<br/>pydantic models"]
        spec_builder["VFSObservationSpecBuilder<br/>Generate observation fields"]
        obs_spec["Observation Spec<br/>List[ObservationField]"]
        metadata["CompiledUniverse Metadata<br/>observation_dim, meter_count"]
    end
    
    subgraph "Phase 2: Initialization-Time (VectorizedHamletEnv)"
        registry_init["VariableRegistry.__init__()<br/>Initialize storage"]
        substrate_init["Substrate Setup<br/>Configure encoding mode"]
        action_space["ActionSpace Setup<br/>Build action vocabulary"]
        affordance_vocab["Affordance Vocabulary<br/>Fixed 14 affordances"]
    end
    
    subgraph "Phase 3: Step-Time (Environment Step)"
        state_update["Environment State Update<br/>positions, meters, time"]
        vfs_write["VFS Registry Write<br/>Update variable values"]
        read_and_norm["Read + Normalize<br/>Apply minmax/zscore"]
        concatenate["Concatenate<br/>Build observation tensor"]
        output["Observations<br/>[num_agents, obs_dim]"]
    end
    
    yaml --> schema_validation
    schema_validation --> spec_builder
    spec_builder --> obs_spec
    obs_spec --> metadata
    
    metadata --> registry_init
    obs_spec --> registry_init
    registry_init --> substrate_init
    substrate_init --> action_space
    action_space --> affordance_vocab
    
    affordance_vocab --> state_update
    state_update --> vfs_write
    vfs_write --> read_and_norm
    read_and_norm --> concatenate
    concatenate --> output
    
    style yaml fill:#d1c4e9
    style obs_spec fill:#c8e6c9
    style output fill:#e1f5fe
```

## Phase 1: Compile-Time (YAML → Observation Spec)

### VFS Variable Definition

```mermaid
sequenceDiagram
    participant Y as variables_reference.yaml
    participant S as VFSSchema
    participant B as VFSObservationSpecBuilder
    participant O as ObservationField[]
    
    Y->>S: Parse YAML
    S->>S: Validate variable definitions
    S->>S: Validate exposure config
    
    Note over S: Check readable_by, writable_by<br/>Check scope (global/agent/agent_private)<br/>Check type (scalar/vec2i/vecNf)
    
    S->>B: Pass validated variables + exposures
    
    loop For each exposure
        B->>B: Resolve source_variable
        B->>B: Infer shape from variable type
        B->>B: Build normalization spec (if provided)
        B->>B: Validate normalization shape
        B->>O: Create ObservationField
    end
    
    O->>Y: Return observation spec
```

### Variable Type → Observation Shape Inference

```mermaid
graph TB
    var_type["Variable Type"]
    
    scalar["scalar → []<br/>(1 dim)"]
    bool["bool → []<br/>(1 dim)"]
    vec2i["vec2i → [2]<br/>(2 dims)"]
    vec3i["vec3i → [3]<br/>(3 dims)"]
    vecNi["vecNi(N) → [N]<br/>(N dims)"]
    vecNf["vecNf(N) → [N]<br/>(N dims)"]
    
    var_type --> scalar
    var_type --> bool
    var_type --> vec2i
    var_type --> vec3i
    var_type --> vecNi
    var_type --> vecNf
    
    style var_type fill:#d1c4e9
    style scalar fill:#c8e6c9
```

### ObservationField Schema

```mermaid
classDiagram
    class ObservationField {
        <<pydantic>>
        +id: str
        +source_variable: str
        +exposed_to: list[str]
        +shape: list[int]
        +normalization: NormalizationSpec
    }
    
    class NormalizationSpec {
        <<pydantic>>
        +kind: str
        +min: float | list[float]
        +max: float | list[float]
        +mean: float | list[float]
        +std: float | list[float]
    }
    
    ObservationField *-- NormalizationSpec
    
    style ObservationField fill:#c8e6c9
    style NormalizationSpec fill:#fff9c4
```

## Phase 2: Initialization-Time (Registry Setup)

### VFS Registry Initialization

```mermaid
flowchart TD
    start[VectorizedHamletEnv.__init__]
    
    load_spec[Load observation spec<br/>from CompiledUniverse]
    compute_dim[Compute observation_dim<br/>sum(field.shape)]
    validate_dim{Matches<br/>metadata?}
    error[ERROR: Dimension mismatch]
    
    init_registry[Initialize VariableRegistry<br/>Create storage for each variable]
    setup_substrate[Setup substrate encoding<br/>Configure observation_encoding mode]
    build_action_space[Build ActionSpace<br/>Substrate + custom actions]
    build_affordance_vocab[Build affordance vocabulary<br/>Fixed 14 types + "none"]
    
    complete[Initialization Complete<br/>Ready for episodes]
    
    start --> load_spec
    load_spec --> compute_dim
    compute_dim --> validate_dim
    validate_dim -->|No| error
    validate_dim -->|Yes| init_registry
    
    init_registry --> setup_substrate
    setup_substrate --> build_action_space
    build_action_space --> build_affordance_vocab
    build_affordance_vocab --> complete
    
    style error fill:#ffccbc
    style complete fill:#c8e6c9
```

### Substrate Encoding Mode Selection

```mermaid
graph TB
    substrate_config["Substrate Config<br/>observation_encoding field"]
    
    relative["relative (default)<br/>Normalized [0,1] coords<br/>Transfer learning friendly"]
    scaled["scaled<br/>Normalized coords + range metadata<br/>Network learns grid size"]
    absolute["absolute<br/>Raw unnormalized coords<br/>Physical simulation"]
    
    substrate_config --> relative
    substrate_config --> scaled
    substrate_config --> absolute
    
    subgraph "Observation Dimensions (Grid2D 8×8)"
        relative_dims["relative: 2 dims<br/>(x/7, y/7)"]
        scaled_dims["scaled: 4 dims<br/>(x/7, y/7, 8, 8)"]
        absolute_dims["absolute: 2 dims<br/>(x, y)"]
    end
    
    relative --> relative_dims
    scaled --> scaled_dims
    absolute --> absolute_dims
    
    style relative fill:#c8e6c9
    style scaled fill:#fff9c4
```

## Phase 3: Step-Time (Observation Construction)

### Main Observation Construction Flow

```mermaid
flowchart TD
    step_start[Environment Step Complete<br/>positions, meters, time updated]
    
    subgraph "Update VFS Registry"
        write_grid[Write: grid_encoding or local_window]
        write_position[Write: position]
        write_meters[Write: energy, hygiene, etc.]
        write_affordance[Write: affordance_at_position]
        write_temporal[Write: time_sin, time_cos, etc.]
    end
    
    subgraph "Build Observation Tensor"
        init_list[observations = []]
        
        loop_start[For each ObservationField]
        read_var[Read variable from registry]
        check_perm{Readable by<br/>agent?}
        error_perm[ERROR: Permission denied]
        
        apply_norm{Has<br/>normalization?}
        minmax[Apply minmax normalization]
        zscore[Apply zscore normalization]
        
        ensure_2d[Ensure [num_agents, *] shape]
        append[Append to observations list]
        loop_end[Next field]
        
        concat[torch.cat(observations, dim=1)]
        output[Return observations<br/>[num_agents, obs_dim]]
    end
    
    step_start --> write_grid
    write_grid --> write_position
    write_position --> write_meters
    write_meters --> write_affordance
    write_affordance --> write_temporal
    
    write_temporal --> init_list
    init_list --> loop_start
    loop_start --> read_var
    read_var --> check_perm
    check_perm -->|No| error_perm
    check_perm -->|Yes| apply_norm
    
    apply_norm -->|minmax| minmax
    apply_norm -->|zscore| zscore
    apply_norm -->|None| ensure_2d
    
    minmax --> ensure_2d
    zscore --> ensure_2d
    ensure_2d --> append
    append --> loop_end
    loop_end -->|More fields| loop_start
    loop_end -->|Done| concat
    concat --> output
    
    style error_perm fill:#ffccbc
    style output fill:#c8e6c9
```

### Spatial Encoding: Full vs Partial Observability

```mermaid
graph TB
    observability["Observability Mode"]
    
    full["Full Observability<br/>(POMDP disabled)"]
    partial["Partial Observability<br/>(POMDP enabled)"]
    
    observability --> full
    observability --> partial
    
    subgraph "Full Observability Pipeline"
        full_grid["substrate._encode_full_grid()<br/>Returns global occupancy grid"]
        full_position["substrate.normalize_positions()<br/>Returns normalized [0,1] coords"]
        full_concat["Concatenate:<br/>grid + position + meters + affordances"]
        full_dims["Grid2D 8×8:<br/>64 grid + 2 position = 66 dims"]
    end
    
    subgraph "Partial Observability Pipeline"
        partial_window["substrate.encode_partial_observation()<br/>Returns local 5×5 window"]
        partial_position["substrate.normalize_positions()<br/>Returns normalized [0,1] coords"]
        partial_concat["Concatenate:<br/>local_window + position + meters + affordances"]
        partial_dims["Grid2D (vision_range=2):<br/>25 local + 2 position = 27 dims"]
    end
    
    full --> full_grid
    full_grid --> full_position
    full_position --> full_concat
    full_concat --> full_dims
    
    partial --> partial_window
    partial_window --> partial_position
    partial_position --> partial_concat
    partial_concat --> partial_dims
    
    style full fill:#c8e6c9
    style partial fill:#e1f5fe
```

### Grid Encoding (Full Observability)

```mermaid
graph TB
    input[Agent Positions<br/>[num_agents, 2]]
    affordances[Affordance Positions<br/>{name: [2]}]
    
    init_grid["Initialize empty grid<br/>[width × height]"]
    mark_affordances["Mark affordances<br/>grid[aff_y * width + aff_x] = 1.0"]
    broadcast["Broadcast to all agents<br/>[num_agents, width × height]"]
    mark_agents["Mark agent positions<br/>grid[agent_y * width + agent_x] += 1.0"]
    clamp["Clamp to [0, 2]<br/>0=empty, 1=affordance, 2=agent"]
    
    output["Grid Encoding<br/>[num_agents, width × height]"]
    
    input --> broadcast
    affordances --> init_grid
    init_grid --> mark_affordances
    mark_affordances --> broadcast
    broadcast --> mark_agents
    mark_agents --> clamp
    clamp --> output
    
    style init_grid fill:#d1c4e9
    style output fill:#c8e6c9
```

### Local Window Encoding (Partial Observability)

```mermaid
graph TB
    input[Agent Positions<br/>[num_agents, 2]]
    affordances[Affordance Positions<br/>{name: [2]}]
    vision_range[Vision Range<br/>e.g., 2 for 5×5 window]
    
    init_windows["Initialize local grids<br/>[num_agents, window_size, window_size]"]
    
    loop_start["For each agent"]
    extract_window["Extract window<br/>[agent_x-range : agent_x+range+1]<br/>[agent_y-range : agent_y+range+1]"]
    mark_local_affs["Mark affordances in window<br/>If (aff_x, aff_y) in window"]
    loop_end["Next agent"]
    
    flatten["Flatten windows<br/>[num_agents, window_size²]"]
    output["Local Window Encoding<br/>[num_agents, 25] for 5×5"]
    
    input --> init_windows
    affordances --> loop_start
    vision_range --> loop_start
    init_windows --> loop_start
    loop_start --> extract_window
    extract_window --> mark_local_affs
    mark_local_affs --> loop_end
    loop_end -->|More agents| loop_start
    loop_end -->|Done| flatten
    flatten --> output
    
    style extract_window fill:#d1c4e9
    style output fill:#e1f5fe
```

### Meter Encoding

```mermaid
graph TB
    meters["Meters Tensor<br/>[num_agents, num_meters]"]
    
    subgraph "Per-Meter Write"
        energy["energy: meters[:, 0]"]
        hygiene["hygiene: meters[:, 1]"]
        satiation["satiation: meters[:, 2]"]
        money["money: meters[:, 3]"]
        health["health: meters[:, 4]"]
        fitness["fitness: meters[:, 5]"]
        mood["mood: meters[:, 6]"]
        social["social: meters[:, 7]"]
    end
    
    registry["VFS Registry<br/>Store each meter individually"]
    
    read_meters["Read all meters from registry<br/>According to observation spec"]
    normalize["Apply normalization<br/>(usually minmax [0,1])"]
    output["Meter Observations<br/>[num_agents, 8]"]
    
    meters --> energy
    meters --> hygiene
    meters --> satiation
    meters --> money
    meters --> health
    meters --> fitness
    meters --> mood
    meters --> social
    
    energy --> registry
    hygiene --> registry
    satiation --> registry
    money --> registry
    health --> registry
    fitness --> registry
    mood --> registry
    social --> registry
    
    registry --> read_meters
    read_meters --> normalize
    normalize --> output
    
    style meters fill:#d1c4e9
    style output fill:#c8e6c9
```

### Affordance Encoding (One-Hot)

```mermaid
graph TB
    positions[Agent Positions<br/>[num_agents, 2]]
    affordances[Deployed Affordances<br/>{name: [2]}]
    vocab["Affordance Vocabulary<br/>14 types from YAML"]
    
    init["Initialize encoding<br/>[num_agents, num_types + 1]<br/>Last dim is 'none'"]
    default["Set all to 'none'<br/>encoding[:, -1] = 1.0"]
    
    loop_start["For each affordance in vocabulary"]
    check_deployed{Is deployed<br/>on grid?}
    check_agents["Check which agents on affordance<br/>substrate.is_on_position()"]
    mark["Mark agents<br/>encoding[on_aff, idx] = 1.0<br/>encoding[on_aff, -1] = 0.0"]
    loop_end["Next affordance"]
    
    output["Affordance Encoding<br/>[num_agents, 15]<br/>14 types + 1 'none'"]
    
    positions --> init
    affordances --> loop_start
    vocab --> loop_start
    
    init --> default
    default --> loop_start
    loop_start --> check_deployed
    check_deployed -->|Yes| check_agents
    check_deployed -->|No| loop_end
    check_agents --> mark
    mark --> loop_end
    loop_end -->|More affordances| loop_start
    loop_end -->|Done| output
    
    style vocab fill:#d1c4e9
    style output fill:#c8e6c9
```

**Key Insight**: Encoding uses FULL vocabulary (14 types) even if not all are deployed. This ensures observation dimensions stay constant across curriculum levels, enabling checkpoint transfer.

### Temporal Encoding

```mermaid
graph TB
    time_of_day["time_of_day<br/>0-23 integer"]
    
    subgraph "Cyclic Encoding"
        angle["angle = (time / 24) × 2π"]
        sin_component["time_sin = sin(angle)"]
        cos_component["time_cos = cos(angle)"]
    end
    
    interaction_progress["interaction_progress<br/>0-10 ticks"]
    normalize_progress["normalized = progress / 10"]
    
    lifetime["step_counts / agent_lifespan"]
    clamp["clamp(0.0, 1.0)"]
    lifetime_progress["lifetime_progress"]
    
    registry["VFS Registry"]
    
    output["Temporal Features<br/>[time_sin, time_cos,<br/>interaction_progress,<br/>lifetime_progress]<br/>4 dims total"]
    
    time_of_day --> angle
    angle --> sin_component
    angle --> cos_component
    
    sin_component --> registry
    cos_component --> registry
    
    interaction_progress --> normalize_progress
    normalize_progress --> registry
    
    lifetime --> clamp
    clamp --> lifetime_progress
    lifetime_progress --> registry
    
    registry --> output
    
    style angle fill:#d1c4e9
    style output fill:#c8e6c9
```

**Rationale**: Sin/cos encoding preserves cyclical nature of time (23:00 is close to 00:00).

### Normalization Application

```mermaid
flowchart TD
    value["Raw Variable Value<br/>from VFS Registry"]
    
    check_norm{Has<br/>normalization?}
    
    no_norm["Use raw value"]
    
    check_kind{Normalization<br/>kind?}
    
    minmax_flow["MinMax Normalization"]
    minmax_formula["value = (value - min) / (max - min + ε)"]
    minmax_result["Normalized to [0, 1]"]
    
    zscore_flow["Z-Score Normalization"]
    zscore_formula["value = (value - mean) / (std + ε)"]
    zscore_result["Standardized (μ=0, σ=1)"]
    
    ensure_2d["Ensure 2D shape<br/>[num_agents, *]"]
    output["Normalized Value"]
    
    value --> check_norm
    check_norm -->|No| no_norm
    check_norm -->|Yes| check_kind
    
    check_kind -->|minmax| minmax_flow
    check_kind -->|zscore| zscore_flow
    
    minmax_flow --> minmax_formula
    minmax_formula --> minmax_result
    minmax_result --> ensure_2d
    
    zscore_flow --> zscore_formula
    zscore_formula --> zscore_result
    zscore_result --> ensure_2d
    
    no_norm --> ensure_2d
    ensure_2d --> output
    
    style minmax_result fill:#c8e6c9
    style zscore_result fill:#e1f5fe
```

## Observation Dimension Breakdown

### Level 1: Full Observability (Grid2D 8×8)

```mermaid
graph TB
    obs_dim["observation_dim = 29"]
    
    grid["Grid Encoding: 64 dims<br/>(8 × 8 flattened)"]
    position["Position: 2 dims<br/>(normalized x, y)"]
    meters["Meters: 8 dims<br/>(energy, hygiene, ...)"]
    affordance["Affordance: 15 dims<br/>(14 types + none)"]
    temporal["Temporal: 4 dims<br/>(time_sin, time_cos,<br/>interaction_progress,<br/>lifetime_progress)"]
    
    note["Note: Grid encoding NOT included<br/>in observation_dim for L1<br/>(VFS exposures don't include it)<br/><br/>Actual: 2 + 8 + 15 + 4 = 29"]
    
    obs_dim --> grid
    obs_dim --> position
    obs_dim --> meters
    obs_dim --> affordance
    obs_dim --> temporal
    obs_dim --> note
    
    style obs_dim fill:#d1c4e9
    style note fill:#fff9c4
```

### Level 2: Partial Observability (Grid2D, vision_range=2)

```mermaid
graph TB
    obs_dim["observation_dim = 54"]
    
    local_window["Local Window: 25 dims<br/>(5 × 5 local grid)"]
    position["Position: 2 dims<br/>(normalized x, y)"]
    meters["Meters: 8 dims<br/>(energy, hygiene, ...)"]
    affordance["Affordance: 15 dims<br/>(14 types + none)"]
    temporal["Temporal: 4 dims<br/>(time_sin, time_cos,<br/>interaction_progress,<br/>lifetime_progress)"]
    
    total["Total: 25 + 2 + 8 + 15 + 4 = 54"]
    
    obs_dim --> local_window
    obs_dim --> position
    obs_dim --> meters
    obs_dim --> affordance
    obs_dim --> temporal
    obs_dim --> total
    
    style obs_dim fill:#d1c4e9
    style total fill:#c8e6c9
```

## VFS Access Control During Observation Construction

```mermaid
sequenceDiagram
    participant E as Environment
    participant R as VFS Registry
    participant V as VariableDef
    participant O as ObservationField
    
    Note over E: Step complete, build observations
    
    loop For each ObservationField
        E->>R: get(source_variable, reader="agent")
        R->>V: Check variable.readable_by
        
        alt "agent" in readable_by
            R->>E: Return value tensor
            E->>E: Apply normalization
            E->>E: Append to observations list
        else "agent" NOT in readable_by
            R->>E: ERROR: Permission denied
        end
    end
    
    E->>E: torch.cat(observations, dim=1)
    E->>E: Return [num_agents, obs_dim]
```

## Summary

### Observation Construction Path

```
YAML Config → VFS Schema Validation → ObservationField Spec → VariableRegistry → 
Environment State Update → VFS Write → VFS Read → Normalization → Concatenation → 
Observation Tensor [num_agents, obs_dim]
```

### Key Design Principles

1. **Fixed Observation Vocabulary**: All levels observe same 14 affordances (for transfer learning)
2. **VFS-Driven Construction**: Observations built declaratively from VFS registry
3. **Access Control Enforcement**: Only variables marked `readable_by: [agent]` are exposed
4. **Normalization Flexibility**: MinMax or Z-score normalization configurable per field
5. **Substrate Abstraction**: Encoding mode (relative/scaled/absolute) controlled by substrate config
6. **POMDP Support**: Local window encoding for partial observability

### Observation Dimensions by Level

| Level | Grid/Window | Position | Meters | Affordance | Temporal | **Total** |
|-------|-------------|----------|--------|------------|----------|-----------|
| L0 (3×3) | 9 | 2 | 8 | 15 | 4 | **38** |
| L0.5 (7×7) | 49 | 2 | 8 | 15 | 4 | **78** |
| L1 (8×8, full) | 64 | 2 | 8 | 15 | 4 | **93** |
| L2 (8×8, POMDP) | 25 (local) | 2 | 8 | 15 | 4 | **54** |
| L3 (8×8, temporal) | 64 | 2 | 8 | 15 | 4 | **93** |

**Note**: Actual observation dimensions depend on VFS exposure configuration. The table above shows common configurations.

### Performance Considerations

- **Hot Path**: `_get_observations()` called every step for all agents
- **GPU Tensors**: All operations on GPU (no CPU transfers)
- **Batched Operations**: Single tensor operation for all agents
- **No Python Loops**: Vectorized affordance encoding using `is_on_position()`
- **Registry Overhead**: Acceptable (write once, read once per observation field)
