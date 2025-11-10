# Townlet Compilation Pipeline

## Compilation Stages

```mermaid
flowchart TD
    start([Config Pack Directory])

    subgraph "Stage 1: Parse"
        parse[Parse YAML Files]
        load_substrate[Load substrate.yaml]
        load_bars[Load bars.yaml]
        load_cascades[Load cascades.yaml]
        load_affordances[Load affordances.yaml]
        load_training[Load training.yaml]
        load_vfs[Load variables_reference.yaml]
        load_actions[Load global_actions.yaml]
        raw_configs[RawConfigs DTO]
    end

    subgraph "Stage 2: Symbol Tables"
        build_symbols[Build Symbol Table]
        register_meters[Register Meters]
        register_vars[Register Variables]
        register_actions[Register Actions]
        register_cascades[Register Cascades]
        register_affordances[Register Affordances]
        symbol_table[UniverseSymbolTable]
    end

    subgraph "Stage 3: Resolve References"
        resolve[Resolve Cross-File Refs]
        check_meter_refs[Check Meter References]
        check_affordance_refs[Check Affordance References]
        check_action_refs[Check Action References]
        errors1{Errors?}
    end

    subgraph "Stage 4: Cross-Validation"
        validate[Cross-Validate]
        spatial_feasibility[Spatial Feasibility]
        economic_balance[Economic Balance]
        cascade_cycles[Cascade Cycles]
        operating_hours[Operating Hours]
        availability[Availability Constraints]
        capabilities[Capabilities & Pipelines]
        substrate_compat[Substrate-Action Compatibility]
        errors2{Errors?}
    end

    subgraph "Stage 5: Metadata"
        compute_metadata[Compute Metadata]
        build_obs_spec[Build Observation Spec]
        build_vfs[Build VFS Registry]
        compute_dims[Compute Dimensions]
        compute_hash[Compute Config Hash]
        compute_provenance[Compute Provenance ID]
        metadata[UniverseMetadata]
        obs_spec[ObservationSpec]
    end

    subgraph "Stage 6: Optimization"
        optimize[Pre-compute Tensors]
        base_depletions[Base Depletions Tensor]
        cascade_data[Cascade Data Structures]
        action_masks[Action Mask Table<br/>24h × affordances]
        position_map[Affordance Position Map]
        opt_data[OptimizationData]
    end

    subgraph "Stage 7: Emit"
        emit[Emit CompiledUniverse]
        freeze_check[Verify Frozen<br/>Dataclass]
        cache[Write Cache]
        compiled[CompiledUniverse]
    end

    start --> parse
    parse --> load_substrate
    parse --> load_bars
    parse --> load_cascades
    parse --> load_affordances
    parse --> load_training
    parse --> load_vfs
    parse --> load_actions
    load_substrate --> raw_configs
    load_bars --> raw_configs
    load_cascades --> raw_configs
    load_affordances --> raw_configs
    load_training --> raw_configs
    load_vfs --> raw_configs
    load_actions --> raw_configs

    raw_configs --> build_symbols
    build_symbols --> register_meters
    build_symbols --> register_vars
    build_symbols --> register_actions
    build_symbols --> register_cascades
    build_symbols --> register_affordances
    register_meters --> symbol_table
    register_vars --> symbol_table
    register_actions --> symbol_table
    register_cascades --> symbol_table
    register_affordances --> symbol_table

    symbol_table --> resolve
    resolve --> check_meter_refs
    resolve --> check_affordance_refs
    resolve --> check_action_refs
    check_meter_refs --> errors1
    check_affordance_refs --> errors1
    check_action_refs --> errors1
    errors1 -->|Yes| fail1([Compilation Error])
    errors1 -->|No| validate

    validate --> spatial_feasibility
    validate --> economic_balance
    validate --> cascade_cycles
    validate --> operating_hours
    validate --> availability
    validate --> capabilities
    validate --> substrate_compat
    spatial_feasibility --> errors2
    economic_balance --> errors2
    cascade_cycles --> errors2
    operating_hours --> errors2
    availability --> errors2
    capabilities --> errors2
    substrate_compat --> errors2
    errors2 -->|Yes| fail2([Compilation Error])
    errors2 -->|No| compute_metadata

    compute_metadata --> build_obs_spec
    compute_metadata --> build_vfs
    compute_metadata --> compute_dims
    compute_metadata --> compute_hash
    compute_metadata --> compute_provenance
    build_obs_spec --> obs_spec
    build_vfs --> obs_spec
    compute_dims --> metadata
    compute_hash --> metadata
    compute_provenance --> metadata

    metadata --> optimize
    obs_spec --> optimize
    optimize --> base_depletions
    optimize --> cascade_data
    optimize --> action_masks
    optimize --> position_map
    base_depletions --> opt_data
    cascade_data --> opt_data
    action_masks --> opt_data
    position_map --> opt_data

    opt_data --> emit
    metadata --> emit
    obs_spec --> emit
    emit --> freeze_check
    freeze_check --> cache
    cache --> compiled

    style parse fill:#e1f5fe
    style build_symbols fill:#fff9c4
    style resolve fill:#f3e5f5
    style validate fill:#ffe0b2
    style compute_metadata fill:#c8e6c9
    style optimize fill:#ffccbc
    style emit fill:#d1c4e9
```

## Stage Details

### Stage 1: Parse Individual Files
- **Input**: Config pack directory path
- **Process**:
  - Load each YAML file using Pydantic DTOs
  - Validate schema compliance
  - Build source map for error reporting
- **Output**: `RawConfigs` object containing all parsed configs
- **Key Classes**:
  - `RawConfigs.from_config_dir()`
  - `SubstrateConfig`, `BarConfig`, `CascadeConfig`, etc.

### Stage 2: Build Symbol Tables
- **Input**: `RawConfigs`
- **Process**:
  - Register all meters from `bars.yaml`
  - Register all variables from `variables_reference.yaml`
  - Register all actions from `global_actions.yaml`
  - Register all cascades from `cascades.yaml`
  - Register all affordances from `affordances.yaml`
  - Register all cues from `cues.yaml`
- **Output**: `UniverseSymbolTable`
- **Purpose**: Create global namespace for cross-file reference resolution

### Stage 3: Resolve References
- **Input**: `RawConfigs`, `UniverseSymbolTable`
- **Process**:
  - Validate all meter references in cascades
  - Validate all meter references in affordances (costs, effects, capabilities)
  - Validate enabled_affordances list
  - Validate action costs/effects meter references
- **Output**: List of reference errors (if any)
- **Error Codes**: `UAC-RES-001` through `UAC-RES-005`

### Stage 4: Cross-Validation
- **Input**: `RawConfigs`, `UniverseSymbolTable`
- **Validations**:
  1. **Spatial Feasibility**: Grid has enough cells for affordances + agents
  2. **Economic Balance**: Income-generating affordances exist and are available
  3. **Cascade Cycles**: No circular dependencies in meter cascades
  4. **Operating Hours**: Valid 0-23 hour ranges
  5. **Availability Constraints**: Valid min/max bounds [0.0, 1.0]
  6. **Capabilities**: multi_tick requires per_tick or on_completion effects
  7. **Substrate-Action Compatibility**: Actions valid for substrate type
  8. **Capacity & Sustainability**: Critical meters have restoration paths
- **Output**: List of validation errors/warnings
- **Error Codes**: `UAC-VAL-001` through `UAC-VAL-010`

### Stage 5: Compute Metadata
- **Input**: `RawConfigs`, `UniverseSymbolTable`
- **Process**:
  - Build VFS observation spec from variables
  - Compute observation dimensions
  - Compute grid dimensions (if applicable)
  - Compute economic metrics (max income, total costs, balance)
  - Generate config hash (SHA256 of normalized YAMLs)
  - Generate provenance ID (hash of config + compiler + environment)
- **Output**: `UniverseMetadata`, `ObservationSpec`, VFS fields
- **Key Fields**:
  - `observation_dim`: Total observation dimensions
  - `action_count`: Number of actions
  - `meter_count`, `affordance_count`: Counts
  - `config_hash`, `provenance_id`: Provenance tracking

### Stage 6: Optimization
- **Input**: `RawConfigs`, `UniverseMetadata`
- **Process**:
  - Pre-compute base depletion tensor `[meter_count]`
  - Pre-compute cascade data structures (sorted by target)
  - Pre-compute action mask table `[24 hours, affordance_count]`
  - Pre-compute affordance position tensors
- **Output**: `OptimizationData`
- **Purpose**: Avoid runtime computation, enable GPU-native operations

### Stage 7: Emit CompiledUniverse
- **Input**: All computed artifacts
- **Process**:
  - Construct `CompiledUniverse` frozen dataclass
  - Verify immutability (frozen=True)
  - Write cache to `.compiled/universe.msgpack`
- **Output**: `CompiledUniverse`
- **Invariants**:
  - Must be frozen (immutable after creation)
  - Must be cacheable (deterministic from configs)

## Cache Management

```mermaid
flowchart LR
    check[Check Cache]
    load[Load from Cache]
    validate_hash[Validate Config Hash]
    validate_prov[Validate Provenance ID]
    recompile[Recompile]
    use_cache[Use Cached Universe]

    check --> load
    load --> validate_hash
    validate_hash -->|Match| validate_prov
    validate_hash -->|Mismatch| recompile
    validate_prov -->|Match| use_cache
    validate_prov -->|Mismatch| recompile

    style use_cache fill:#c8e6c9
    style recompile fill:#ffccbc
```

## Error Handling

```mermaid
flowchart TD
    error[Compilation Error]
    collect[CompilationErrorCollector]
    errors[Error Messages]
    warnings[Warning Messages]
    hints[Hints]
    format[Format with Source Map]
    display[Display to User]

    error --> collect
    collect --> errors
    collect --> warnings
    collect --> hints
    errors --> format
    warnings --> format
    hints --> format
    format --> display
```

### Error Code Ranges
- **UAC-RES-001 to UAC-RES-005**: Reference resolution errors
- **UAC-VAL-001 to UAC-VAL-010**: Cross-validation errors

### Source Map
- Tracks YAML file line numbers for each config element
- Enables precise error location reporting
- Format: `filename:key:subkey[index]`

## Provenance Tracking

The compiler generates a provenance ID to track exact compilation conditions:

```python
provenance_id = SHA256(
    config_hash +           # SHA256 of all YAML files
    compiler_version +      # e.g., "0.1.0"
    compiler_git_sha +      # Git commit SHA
    python_version +        # e.g., "3.11.5"
    torch_version +         # e.g., "2.1.0"
    pydantic_version        # e.g., "2.5.0"
)
```

This enables:
- Deterministic cache invalidation
- Reproducibility tracking
- Version migration detection
