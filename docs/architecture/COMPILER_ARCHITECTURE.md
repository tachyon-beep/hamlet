# Compiler Architecture: UAC, BAC, and System Orchestration

**Document Type**: Architecture Specification
**Status**: Design Phase
**Version**: 1.0
**Last Updated**: 2025-11-07
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing compiler infrastructure, researchers extending the system
**Technical Level**: Advanced (compiler design, type systems, data contracts)
**Estimated Reading Time**: 15 min for skim | 45 min for full read | 90 min for deep study

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
The compiler architecture that transforms declarative YAML configurations (Universe as Code, Brain as Code) into executable runtime artifacts. Covers the three-tier compilation system: outer orchestrator (UAC → BAC → Training), UAC internal architecture (7 sub-compilers with dependency resolution), and data contracts between components.

**Why This Document Exists**:
Establishes the compilation pipeline that enables configuration-driven AI development. Without this infrastructure, the system would rely on ad-hoc config loading, hardcoded dimensions, and runtime error discovery. The compiler catches configuration errors at compile-time, enables checkpoint transfer learning, and provides reproducibility through universe versioning.

**Who Should Read This**:
- **Must Read**: Engineers implementing TASK-003 (UAC Core DTOs), TASK-004A (Compiler Implementation), TASK-005 (Brain as Code)
- **Should Read**: Researchers extending UAC with new subsystems, operators creating custom configurations
- **Optional**: Frontend developers (only §6 on observation specs), curriculum designers (skim §1-2)

**Reading Strategy**:
- **Quick Scan** (15 min): Read §1-2 for system overview and compilation pipeline
- **Partial Read** (30 min): Add §3-4 for UAC internals and data contracts
- **Deep Study** (90 min): Read all sections for implementation-level detail

---

## Document Scope

**In Scope**:
- **System orchestration**: How UAC, BAC, and Training components coordinate
- **UAC internal architecture**: 7-stage compilation pipeline with sub-compilers
- **Data contracts**: ObservationSpec, CompiledUniverse, ActionSpaceMetadata
- **Compilation stages**: Parse → Validate → Resolve → Cross-validate → Compute → Optimize → Emit
- **Serialization format**: MessagePack for caching compiled universes
- **Cues as communication**: Theory of Mind through observable behavioral cues
- **Checkpoint compatibility**: Universe versioning via config hashing

**Out of Scope**:
- **Individual DTO schemas**: See TASK-003 (UAC Core DTOs) for Pydantic specifications
- **VFS implementation**: See TASK-002C (Variable & Feature System) for typed variables
- **BAC compiler internals**: See TASK-005 (Brain as Code) for network compilation
- **Runtime execution**: See §6 (Runtime Engine Components) in HLD
- **Deployment**: See operational documentation

**Critical Boundary**:
This document describes the **compilation architecture** (how configs become runtime artifacts), not the **configuration schemas** (what configs contain). For schema details, see TASK-003 (DTOs), TASK-002A (Substrates), TASK-002B (Actions), TASK-002C (VFS).

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**:
  - [UNIVERSE_AS_CODE.md](UNIVERSE_AS_CODE.md) - UAC design philosophy
  - [BRAIN_AS_CODE.md](BRAIN_AS_CODE.md) - BAC design philosophy
  - [01-executive-summary.md](hld/01-executive-summary.md) - Framework overview
- **Builds On**:
  - TASK-002A (Configurable Spatial Substrates)
  - TASK-002B (Composable Action Space)
  - TASK-002C (Variable & Feature System)
  - TASK-003 (UAC Core DTOs)
- **Related**:
  - TASK-004A (Compiler Implementation) - Implementation plan
  - TASK-005 (Brain as Code) - BAC compiler specification
- **Next**: Implementation tasks (TASK-004A, TASK-005)

**Architecture Layer**: Infrastructure (foundational compilation pipeline)

---

## Keywords for Discovery

**Primary Keywords**: compiler, UAC, BAC, compilation pipeline, data contracts, ObservationSpec, CompiledUniverse
**Secondary Keywords**: sub-compilers, symbol table, cross-validation, serialization, MessagePack, checkpoint compatibility
**Subsystems**: UniverseCompiler, BrainCompiler, TrainingRunner, SubstrateCompiler, VFSCompiler, CuesCompiler
**Design Patterns**: Multi-stage compilation, dependency resolution, immutable artifacts, cached compilation

**Quick Search Hints**:
- Looking for "how compilers coordinate"? → See §1 (System Overview)
- Looking for "UAC internal structure"? → See §2 (UAC Architecture)
- Looking for "data contracts"? → See §3 (Data Handoffs)
- Looking for "compilation stages"? → See §4 (Compilation Pipeline)
- Looking for "Theory of Mind"? → See §5 (Cues as Communication)
- Looking for "checkpoint versioning"? → See §7 (Checkpoint Compatibility)

---

## Version History

**Version 1.0** (2025-11-07): Initial compiler architecture design
- Three-tier orchestration (UAC → BAC → Training)
- Seven-stage UAC pipeline with sub-compilers
- Data contracts: ObservationSpec, CompiledUniverse, ActionSpaceMetadata
- Cues as first-class communication modality (Theory of Mind)
- MessagePack serialization with config hashing

---

# 1. System Overview: Three-Tier Compilation

## 1.1 The Compilation Stack

The HAMLET system uses a **three-tier compilation architecture** that transforms declarative configurations into executable training runs:

```
┌─────────────────────────────────────────────────┐
│         HamletOrchestrator (Main Process)       │
├─────────────────────────────────────────────────┤
│                                                 │
│  Tier 1: UniverseCompiler (UAC)                 │
│    ├─ Input:  config_pack_dir/ (7 YAML files)  │
│    └─ Output: CompiledUniverse (immutable DTO)  │
│           │                                     │
│           ▼                                     │
│  Tier 2: BrainCompiler (BAC)                    │
│    ├─ Input:  brain.yaml + CompiledUniverse    │
│    └─ Output: CompiledBrain (network + metadata)│
│           │                                     │
│           ▼                                     │
│  Tier 3: TrainingRunner                         │
│    ├─ Input:  CompiledUniverse + CompiledBrain │
│    └─ Output: Trained agent + checkpoints      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Key Principles

**1. Compile Once, Execute Many Times**
- Configuration errors discovered at compile-time, not runtime
- Compiled universe can be cached and reused across runs
- No repeated parsing/validation during training

**2. Explicit Dependencies**
- BAC depends on UAC (needs obs_dim, action_dim from universe)
- Training depends on both (needs universe + brain to instantiate environment)
- Dependencies enforced via function signatures (type-safe)

**3. Immutable Artifacts**
- `CompiledUniverse` is immutable (frozen after compilation)
- `CompiledBrain` is immutable (frozen after compilation)
- Prevents accidental modification during training

**4. Serialization for Caching**
- CompiledUniverse serialized to MessagePack (`.compiled_universe.msgpack`)
- Hash-based cache invalidation (recompile only if configs change)
- Dramatically improves iteration speed during development

## 1.2 Orchestrator Implementation

```python
class HamletOrchestrator:
    """Main orchestrator for compilation and training."""

    def __init__(self, config_pack_dir: Path, brain_config_path: Path):
        self.config_pack_dir = config_pack_dir
        self.brain_config_path = brain_config_path
        self.universe_compiler = UniverseCompiler()
        self.brain_compiler = BrainCompiler()

    def compile_and_train(self) -> None:
        """Sequential pipeline: UAC → BAC → Training."""

        # === TIER 1: Compile Universe ===
        print("🔧 Compiling universe...")
        compiled_universe = self.universe_compiler.compile(
            self.config_pack_dir,
            use_cache=True  # Use cached compilation if configs unchanged
        )
        print(f"✅ Universe compiled (obs_dim={compiled_universe.metadata.obs_dim}, "
              f"action_dim={compiled_universe.metadata.action_dim})")

        # === TIER 2: Compile Brain ===
        print("🧠 Compiling brain architecture...")
        compiled_brain = self.brain_compiler.compile(
            self.brain_config_path,
            obs_spec=compiled_universe.observation_spec,
            action_dim=compiled_universe.metadata.action_dim
        )
        print(f"✅ Brain compiled ({compiled_brain.total_params} parameters)")

        # === TIER 3: Run Training ===
        print("🚀 Starting training...")
        runner = TrainingRunner(compiled_universe, compiled_brain)
        runner.train()

    def run_inference(self, checkpoint_path: Path, port: int) -> None:
        """Run inference server for visualization."""

        # Compile universe only (no brain needed for inference)
        compiled_universe = self.universe_compiler.compile(self.config_pack_dir)

        # Check checkpoint compatibility
        compiled_universe.check_checkpoint_compatibility(checkpoint_path)

        # Start inference server
        inference_server = InferenceServer(
            compiled_universe,
            checkpoint_path,
            port
        )
        inference_server.serve()
```

### Design Rationale

**Why Sequential (Not Parallel)?**
- Clear data dependencies (BAC needs UAC output)
- Easy to debug (linear execution flow)
- Compilation is fast (<1s), no need for parallelization

**Why Separate Tiers (Not Monolithic)?**
- **Separation of concerns**: Universe compilation independent of brain architecture
- **Reusability**: Same compiled universe can be used with multiple brains
- **Testability**: Each compiler can be tested independently
- **Caching**: Can cache compiled universe, recompile only brain during architecture search

**Why Immutable Artifacts?**
- **Prevents bugs**: Can't accidentally modify universe during training
- **Enables caching**: Immutable objects are safe to serialize/deserialize
- **Reproducibility**: Guarantees same universe used throughout training run

---

# 2. UAC Internal Architecture: Seven Sub-Compilers

## 2.1 Sub-Compiler Inventory

The Universe Compiler is composed of **seven specialized sub-compilers**, each responsible for one aspect of universe configuration:

| Sub-Compiler | Input File | Output | Dependencies |
|--------------|-----------|--------|--------------|
| **SubstrateCompiler** | substrate.yaml | SpatialSubstrate | None (independent) |
| **VFSCompiler** | variables.yaml | VariableRegistry | Substrate (for position types) |
| **BarCompiler** | bars.yaml | BarsConfig | VFS (meters are variables) |
| **CascadeCompiler** | cascades.yaml | CascadesConfig | Bars (references meters) |
| **CuesCompiler** | cues.yaml | CuesConfig | Bars (references meters) |
| **ActionSpaceComposer** | substrate + global_actions.yaml | ComposedActionSpace | Substrate, VFS |
| **AffordanceCompiler** | affordances.yaml | AffordanceConfigs | Bars, Actions |

## 2.2 Dependency Graph

```
Substrate (independent)
    ↓
VFS (position variables)
    ↓
┌───┴────┐
│        │
Bars     ActionSpace
│        │
├────────┤
│        │
Cascades Cues
│        │
└───┬────┘
    ↓
Affordances
    ↓
ObservationBuilder (computes obs_dim)
```

**Critical Dependencies**:
- **VFS → Substrate**: Position variables need substrate dimensionality (2D, 3D, ND)
- **Bars → VFS**: Meters are typed variables in VFS registry
- **Cascades → Bars**: Cascade effects reference meter names
- **Cues → Bars**: Visual cues map meter ranges to observable states
- **Actions → Substrate + VFS**: Substrate provides movement actions, VFS provides read/write contracts
- **Affordances → Bars + Actions**: Affordance effects modify meters, trigger actions
- **ObservationBuilder → VFS + Substrate**: Observation dimensions computed from variables + substrate encoding

## 2.3 Seven-Stage Compilation Pipeline

The UniverseCompiler executes a **seven-stage pipeline** that progressively validates and builds the compiled universe:

### **Stage 1: Parse Individual Files**

**Goal**: Load all YAML files and validate individual schemas.

**Operations**:
- Load substrate.yaml → validate SubstrateConfig schema
- Load variables.yaml → validate VariableDefinitions schema
- Load bars.yaml → validate BarsConfig schema
- Load cascades.yaml → validate CascadesConfig schema
- Load cues.yaml → validate CuesConfig schema
- Load affordances.yaml → validate AffordanceConfigs schema
- Run all DTOs with `ConfigDict(extra="forbid")` so stray keys fail fast
- Capture `training.enabled_actions` from HamletConfig for downstream action masking

**Errors Caught**:
- File not found
- YAML syntax errors
- Pydantic validation failures (missing fields, wrong types)

**Output**: `RawConfigs` (container of all loaded configs)

### **Stage 2: Build Symbol Tables**

**Goal**: Register all named entities for cross-file reference resolution.

**Operations**:
- Register meters (from bars.yaml) in symbol table
- Register variables (from variables.yaml) in symbol table
- Register affordances (from affordances.yaml) in symbol table
- Register actions (from substrate + global_actions.yaml) in symbol table

**Errors Caught**:
- Duplicate meter names
- Duplicate variable names
- Duplicate affordance IDs

**Output**: `UniverseSymbolTable` (central registry)

### **Stage 3: Resolve References**

**Goal**: Validate that all cross-file references point to valid entities.

**Operations**:
- Validate cascades reference valid meters
- Validate cues reference valid meters
- Validate affordances reference valid bars
- Validate affordances reference valid actions
- Validate action read/write contracts reference valid variables

**Errors Caught**:
- Dangling meter references (cascade affects non-existent meter)
- Dangling action references (affordance triggers non-existent action)
- Dangling variable references (action reads non-existent variable)

**Output**: `CompilationErrorCollector` (list of errors, or empty if valid)

### **Stage 4: Cross-Validate**

**Goal**: Validate cross-file constraints and semantic correctness.

**Operations**:
- Validate spatial feasibility (N affordances + agents must fit on the configured grid)
- Validate economic balance 2.0 (total income ≥ total costs, jobs available enough hours per day)
- Enforce safety ceilings (`MAX_METERS`, `MAX_AFFORDANCES`, `MAX_CASCADES`, `MAX_ACTIONS`, `MAX_VARIABLES`) to guard against config injection
- Check for circular cascade dependencies (A drains B, B drains A)
- Validate cue range coverage (visual cues span [0.0, 1.0] without gaps/overlaps)
- Validate cue-action consistency (movement/capability declarations align with action metadata)
- Validate operating hours and income windows (jobs must be open when agents can work)
- Validate depletion sustainability (critical meters need restoring affordances with sufficient throughput)
- Validate multi-agent capacity (critical path affordances must handle the configured population)
- Emit structured diagnostics with `UAC-VAL-*` codes, downgrading to warnings when `training.allow_unfeasible_universe` is true

**Errors Caught**:
- Circular dependencies
- Economic imbalance or zero income availability despite costs
- Spatial impossibility (too many affordances for grid size)
- Cue definition gaps/overlaps
- Security limit violations (excessive meters/affordances/cascades/actions/variables)
- Critical meter depletion with no viable restoration
- Income-hour gaps (jobs never open) and capacity bottlenecks in multi-agent scenarios

**Output**: `CompilationErrorCollector` (errors raised if any found)

**Stage 4 Diagnostic Codes**

| Code        | Condition Detected                                                                    | Default Response |
|-------------|----------------------------------------------------------------------------------------|------------------|
| `UAC-VAL-001` | Grid too small for the configured affordances/agents                                  | Error            |
| `UAC-VAL-002` | Economic imbalance (no income, insufficient income hours, or total costs > income)   | Error or warning when imbalance is non-fatal |
| `UAC-VAL-003` | Cascade circularity                                                                  | Error            |
| `UAC-VAL-004` | Cue coverage gaps/overlaps                                                           | Error            |
| `UAC-VAL-005` | Critical meter sustainability or cue meter typos                                     | Error            |
| `UAC-VAL-006` | Security guard rails exceeded (meters/affordances/cascades/actions/variables)        | Error            |
| `UAC-VAL-007` | Invalid operating-hour declarations or availability bounds                           | Error            |
| `UAC-VAL-008` | Capability/effect mismatches (e.g., instant affordance declaring multi-tick effects) | Error            |
| `UAC-VAL-009` | Visual cue definitions missing range coverage                                        | Error            |
| `UAC-VAL-010` | Affordance positions outside substrate bounds                                        | Error            |

All diagnostics flow through `CompilationMessage`, so SourceMap locations are attached automatically and the CLI can filter/search by code. Operators can downgrade hard failures to warnings for intentional experiments by setting `training.allow_unfeasible_universe=true`; the compiler still emits the same codes so CI dashboards stay searchable.

### **Stage 5: Compute Metadata**

**Goal**: Calculate derived metadata for BAC and training system.

**Operations**:
- Instantiate substrate to compute position encoding dimensions
- Compose action space (substrate actions + custom actions)
- Build observation specification from VFS + substrate + cues
- Compute observation dimension (sum of all observation fields)
- Compute action dimension (total actions including disabled)
- Apply `training.enabled_actions` masks so disabled actions stay in vocabulary but surface as `enabled=False`
- Compute meter count, affordance count
- Generate universe config hash (hash of all YAML file contents)

**Errors Caught**: None (purely computational)

**Output**: `UniverseMetadata` (obs_dim, action_dim, config_hash, etc.)

### **Stage 6: Optimize (Pre-Compute)**

**Goal**: Pre-compute runtime data structures for performance.

**Operations**:
- Build cascade update matrix (sparse matrix for meter updates)
- Build affordance effect tensors (pre-allocated GPU tensors)
- Build action cost lookup table (avoid runtime dictionary lookups)
- Pre-compute cue observation encodings (one-hot → categorical indices)

**Errors Caught**: None (purely computational)

**Output**: `OptimizationData` (pre-computed structures)

### **Stage 7: Emit Compiled Universe**

**Goal**: Assemble immutable CompiledUniverse artifact.

**Operations**:
- Package all configs, metadata, and optimization data
- Freeze artifact (make immutable)
- Serialize to MessagePack for caching

**Errors Caught**: None (assembly only)

**Output**: `CompiledUniverse` (final artifact)

### 2.5 Symbol Table Contract (Implementation Note)

Before coding Stage 2, standardise the symbol table surface so each downstream stage knows what to expect. The `UniverseSymbolTable` must expose at least:

- `register_meter(BarConfig)` / `get_meter(name)` → provides meter metadata + index
- `register_variable(VariableDef)` / `get_variable(id)`
- `register_action(ActionConfig)` / `get_action(name)`
- iteration helpers (`meters`, `variables`, `actions`) for validation passes

All register calls should be idempotent and raise `CompilationError` on duplicates, keeping error reporting consistent with the Stage‑2 plan.

### 2.6 Diagnostics & Source Maps

Every diagnostic now flows through `townlet.universe.errors.CompilationMessage`, which captures the error code, human-readable text, and SourceMap location in one object. `CompilationErrorCollector` simply aggregates these messages until a stage completes, then raises a `CompilationError` whose string form lists stage → `[CODE] file:line - message` entries plus any hints/warnings.

- **Codes**: All compiler-generated issues use the `UAC-VAL-*`, `UAC-SEC-*`, or `UAC-ACT-*` namespaces. Search tooling and CI dashboards can filter by code, enabling fast triage.
- **Locations**: Stage 1 builds a `SourceMap` covering every YAML file/section. Later stages request location handles (e.g., `facilities.yaml:Bed`) so user-facing diagnostics always include clickable pointers.
- **Hints vs Warnings vs Errors**: `_record_feasibility_issue()` downgrades certain feasibility failures to warnings when `training.allow_unfeasible_universe` is set, but the codes remain identical.
- **Extensibility**: New stages only need to call `errors.add(formatter("UAC-VAL-XXX", ...))` to inherit code/location rendering.

### 2.7 Implementation Structure

```python
class UniverseCompiler:
    """Multi-stage universe compiler with cross-validation."""

    def __init__(self):
        # Sub-compilers
        self.substrate_compiler = SubstrateCompiler()
        self.vfs_compiler = VFSCompiler()
        self.bar_compiler = BarCompiler()
        self.cascade_compiler = CascadeCompiler()
        self.cues_compiler = CuesCompiler()
        self.action_composer = ActionSpaceComposer()
        self.affordance_compiler = AffordanceCompiler()

    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """
        Seven-stage compilation pipeline.

        Args:
            config_dir: Directory containing YAML config files
            use_cache: If True, use cached compilation if configs unchanged

        Returns:
            Immutable CompiledUniverse ready for training

        Raises:
            CompilationError: If universe is invalid
        """
        # Check cache
        if use_cache:
            cached = self._try_load_cache(config_dir)
            if cached is not None:
                return cached

        # === STAGE 1: Parse Individual Files ===
        raw_configs = self._stage_1_parse_individual_files(config_dir)

        # === STAGE 2: Build Symbol Tables ===
        symbol_table = self._stage_2_build_symbol_tables(raw_configs)

        # === STAGE 3: Resolve References ===
        errors = CompilationErrorCollector()
        self._stage_3_resolve_references(raw_configs, symbol_table, errors)
        errors.check_and_raise("Stage 3: Reference Resolution")

        # === STAGE 4: Cross-Validate ===
        self._stage_4_cross_validate(raw_configs, symbol_table, errors)
        errors.check_and_raise("Stage 4: Cross-Validation")

        # === STAGE 5: Compute Metadata ===
        metadata = self._stage_5_compute_metadata(raw_configs, symbol_table)

        # === STAGE 6: Optimize (Pre-Compute) ===
        optimization_data = self._stage_6_optimize(raw_configs, metadata)

        # === STAGE 7: Emit Compiled Universe ===
        universe = self._stage_7_emit_compiled_universe(
            raw_configs, metadata, optimization_data
        )

        # Cache for next time
        self._save_cache(universe, config_dir)

        return universe
```

---

# 3. Data Contracts Between Components

## 3.1 The Handoff Problem

The compilation system must pass data between three independent components:

1. **UAC → BAC**: Brain compiler needs universe metadata to build network architecture
2. **UAC → Training**: Training system needs universe to instantiate environment
3. **BAC → Training**: Training system needs brain to instantiate agent

**Design Constraint**: These components are developed independently and may be extended by different researchers. The data contracts must be:
- **Type-safe**: Prevent runtime type errors
- **Versioned**: Support backward compatibility
- **Rich**: Provide semantic metadata, not just raw dimensions
- **Inspectable**: Enable debugging and logging

## 3.2 Handoff 1: UAC → BAC (ObservationSpec)

### What BAC Needs

The Brain Compiler must construct neural network architectures that consume observations. It needs:

1. **Observation dimension** (scalar) - for input layer size
2. **Observation structure** (rich) - for custom encoder architectures
3. **Field semantics** (metadata) - to build domain-specific encoders (vision, meters, temporal)

### ObservationSpec Data Structure

```python
@dataclass
class ObservationField:
    """Single field in observation vector."""
    name: str  # e.g., "energy", "position", "local_grid"
    type: Literal["scalar", "vector", "categorical", "spatial_grid"]
    dims: int  # Number of dimensions this field occupies
    start_index: int  # Index in flat observation vector
    end_index: int    # Exclusive end index (for slicing)
    scope: Literal["global", "agent", "agent_private"]
    description: str

    # Semantic metadata for custom encoders
    semantic_type: str | None = None  # "position", "meter", "affordance", "cue", "temporal", "vision"
    categorical_labels: list[str] | None = None  # For one-hot encodings

@dataclass
class ObservationSpec:
    """Complete observation specification."""
    total_dims: int  # Sum of all field dims (this is obs_dim)
    fields: list[ObservationField]  # All observation fields
    encoding_version: str = "1.0"  # For checkpoint compatibility

    # === Query Methods ===

    def get_field_by_name(self, name: str) -> ObservationField:
        """Lookup field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Field '{name}' not found in observation spec")

    def get_fields_by_type(self, field_type: str) -> list[ObservationField]:
        """Get all fields of given type (e.g., all 'spatial_grid' fields)."""
        return [f for f in self.fields if f.type == field_type]

    def get_fields_by_semantic_type(self, semantic: str) -> list[ObservationField]:
        """Get fields by semantic meaning (e.g., all 'meter' fields)."""
        return [f for f in self.fields if f.semantic_type == semantic]
```

### Example: L1 Full Observability

```python
obs_spec = ObservationSpec(
    total_dims=29,
    fields=[
        ObservationField(
            name="position",
            type="vector",
            dims=2,
            start_index=0,
            end_index=2,
            scope="agent",
            semantic_type="position",
            description="Normalized agent position (x, y) in [0, 1]"
        ),
        ObservationField(
            name="energy",
            type="scalar",
            dims=1,
            start_index=2,
            end_index=3,
            scope="agent",
            semantic_type="meter",
            description="Energy level [0, 1]"
        ),
        # ... (7 more meters: health, satiation, money, mood, social, fitness, hygiene)

        ObservationField(
            name="affordance_at_position",
            type="categorical",
            dims=15,
            start_index=10,
            end_index=25,
            scope="agent",
            semantic_type="affordance",
            categorical_labels=["Bed", "Hospital", "HomeMeal", "Job", ..., "none"],
            description="One-hot encoding of affordance under agent"
        ),
        ObservationField(
            name="time_sin",
            type="scalar",
            dims=1,
            start_index=25,
            end_index=26,
            scope="global",
            semantic_type="temporal",
            description="sin(2π * time_of_day / 24)"
        ),
        # ... (3 more temporal features)
    ]
)
```

### How BAC Uses ObservationSpec

**Operator-defined brain architecture** (brain.yaml):

```yaml
architecture:
  type: "custom"
  encoders:
    - name: "position_encoder"
      input_fields: ["position"]  # References ObservationSpec
      type: "mlp"
      hidden_layers: [32]

    - name: "meter_encoder"
      input_fields: ["energy", "health", "satiation", "money", "mood", "social", "fitness", "hygiene"]
      type: "mlp"
      hidden_layers: [64]

    - name: "temporal_encoder"
      input_fields: ["time_sin", "time_cos", "interaction_progress", "lifetime_progress"]
      type: "mlp"
      hidden_layers: [16]

  fusion:
    type: "concatenate"
    hidden_layers: [256, 128]
```

**BAC Compilation**:

```python
class BrainCompiler:
    def compile(self, brain_yaml: Path, obs_spec: ObservationSpec, action_dim: int):
        """Compile brain architecture."""

        config = load_brain_config(brain_yaml)

        # Build encoders using ObservationSpec
        encoders = []
        for encoder_config in config.architecture.encoders:
            # Look up fields in ObservationSpec
            input_fields = [
                obs_spec.get_field_by_name(name)
                for name in encoder_config.input_fields
            ]

            # Compute input dimension
            input_dim = sum(field.dims for field in input_fields)

            # Get slice indices for runtime
            start_idx = input_fields[0].start_index
            end_idx = input_fields[-1].end_index

            # Create encoder network
            encoder = Encoder(
                encoder_type=encoder_config.type,
                input_dim=input_dim,
                slice_indices=(start_idx, end_idx),
                **encoder_config.params
            )

            encoders.append(encoder)

        return CompiledBrain(encoders=encoders, obs_spec=obs_spec, action_dim=action_dim)
```

**Runtime Usage**:

```python
# During forward pass
obs = env.get_observation()  # [batch, 29]

# Position encoder gets obs[:, 0:2]
position_features = position_encoder(obs[:, 0:2])

# Meter encoder gets obs[:, 2:10]
meter_features = meter_encoder(obs[:, 2:10])

# Temporal encoder gets obs[:, 25:29]
temporal_features = temporal_encoder(obs[:, 25:29])

# Fusion
combined = torch.cat([position_features, meter_features, temporal_features], dim=1)
q_values = q_head(combined)
```

## 3.3 Handoff 2: UAC → Training (CompiledUniverse)

### What Training System Needs

The training system needs to:
1. Instantiate environment (VectorizedHamletEnv)
2. Log per-meter metrics
3. Track affordance usage
4. Apply action masks
5. Verify checkpoint compatibility

### CompiledUniverse Data Structure

```python
@dataclass
class UniverseMetadata:
    """High-level metadata about compiled universe."""
    universe_name: str  # e.g., "L1_full_observability"
    schema_version: str  # e.g., "1.0"
    compiled_at: str  # ISO timestamp
    config_hash: str  # Hash of all config files (BLOCKER 2 solution!)

    # Dimensions (for checkpoint compatibility)
    obs_dim: int
    action_dim: int
    num_meters: int
    num_affordances: int
    position_dim: int  # Substrate-specific (2 for Grid2D, 7 for GridND-7D)

@dataclass
class ActionMetadata:
    """Metadata for single action."""
    id: int
    name: str
    type: Literal["movement", "interaction", "passive", "custom"]
    enabled: bool
    source: Literal["substrate", "custom", "affordance"]
    costs: dict[str, float]  # meter_name → cost
    description: str

@dataclass
class MeterInfo:
    """Single meter metadata."""
    name: str
    index: int  # Index in meters tensor
    critical: bool  # Agent dies if reaches 0?
    initial_value: float
    observable: bool  # In observation space?
    description: str

class CompiledUniverse:
    """Immutable compiled universe artifact."""

    def __init__(
        self,
        metadata: UniverseMetadata,
        substrate: SpatialSubstrate,
        bars_config: BarsConfig,
        cascades_config: CascadesConfig,
        affordances_config: AffordanceConfigCollection,
        cues_config: CuesConfig,
        action_space: ActionSpaceMetadata,
        observation_spec: ObservationSpec,
        meter_metadata: MeterMetadata,
        affordance_metadata: AffordanceMetadata,
        optimization_data: OptimizationData | None = None,
    ):
        # Store all components
        self.metadata = metadata
        self.substrate = substrate
        self.bars_config = bars_config
        self.cascades_config = cascades_config
        self.affordances_config = affordances_config
        self.cues_config = cues_config
        self.action_space = action_space
        self.observation_spec = observation_spec
        self.meter_metadata = meter_metadata
        self.affordance_metadata = affordance_metadata
        self.optimization_data = optimization_data

        # Make immutable
        self._frozen = True

    # === Factory Methods ===

    def create_environment(
        self,
        num_agents: int,
        device: torch.device,
        **env_kwargs
    ) -> VectorizedHamletEnv:
        """Instantiate environment from compiled universe."""
        return VectorizedHamletEnv(
            num_agents=num_agents,
            device=device,
            substrate=self.substrate,
            bars_config=self.bars_config,
            cascades_config=self.cascades_config,
            affordances_config=self.affordances_config,
            cues_config=self.cues_config,
            action_space=self.action_space,
            observation_spec=self.observation_spec,
            **env_kwargs
        )

    # === Checkpoint Compatibility ===

    def check_checkpoint_compatibility(self, checkpoint_path: Path) -> None:
        """Verify checkpoint is compatible with this universe."""
        checkpoint = torch.load(checkpoint_path)

        if checkpoint['obs_dim'] != self.metadata.obs_dim:
            raise ValueError(
                f"Checkpoint expects obs_dim={checkpoint['obs_dim']}, "
                f"but universe has obs_dim={self.metadata.obs_dim}"
            )

        if checkpoint['action_dim'] != self.metadata.action_dim:
            raise ValueError(
                f"Checkpoint expects action_dim={checkpoint['action_dim']}, "
                f"but universe has action_dim={self.metadata.action_dim}"
            )

        # Config hash check (BLOCKER 2 solution)
        if 'universe_hash' in checkpoint:
            if checkpoint['universe_hash'] != self.metadata.config_hash:
                import warnings
                warnings.warn(
                    f"Checkpoint trained on different universe config. "
                    f"Transfer learning may behave unexpectedly.",
                    UserWarning
                )
```

### How Training System Uses CompiledUniverse

```python
class TrainingRunner:
    """Main training loop."""

    def train(
        self,
        universe: CompiledUniverse,
        brain: CompiledBrain,
        training_config: TrainingConfig
    ):
        """Run training."""

        # Instantiate environment
        env = universe.create_environment(
            num_agents=training_config.num_agents,
            device=training_config.device
        )

        # Initialize agent
        agent = DQNAgent(
            network=brain.network,
            obs_dim=universe.metadata.obs_dim,
            action_dim=universe.metadata.action_dim,
            learning_rate=training_config.learning_rate,
        )

        # Setup logging with meter names
        logger = MetricsLogger(
            meter_names=[m.name for m in universe.meter_metadata.meters],
            affordance_names=[a.name for a in universe.affordance_metadata.affordances],
        )

        # Training loop
        for episode in range(training_config.max_episodes):
            obs = env.reset()

            while not done:
                # Get base action mask from universe
                action_mask = universe.action_space.get_action_mask(
                    num_agents=training_config.num_agents,
                    device=training_config.device
                )

                actions = agent.select_actions(obs, action_mask)
                obs, rewards, done, info = env.step(actions)

                # Log per-meter values
                for meter in universe.meter_metadata.meters:
                    logger.log_meter(
                        meter.name,
                        env.meters[:, meter.index].mean().item()
                    )

            # Save checkpoint with universe hash
            if episode % 100 == 0:
                self.save_checkpoint(
                    agent=agent,
                    universe_hash=universe.metadata.config_hash,
                    obs_dim=universe.metadata.obs_dim,
                    action_dim=universe.metadata.action_dim,
                )
```

---

# 4. Serialization and Caching

## 4.1 Serialization Format: MessagePack

**Why MessagePack?**

| Format | Size | Load Time | Human-Readable | Binary Support | Cross-Language |
|--------|------|-----------|----------------|----------------|----------------|
| JSON | 100% | 1.0x | ✅ Yes | ❌ No | ✅ Yes |
| MessagePack | 50-70% | 0.3x | ⚠️ With tools | ✅ Yes | ✅ Yes |
| Pickle | 40-60% | 0.2x | ❌ No | ✅ Yes | ❌ Python-only |
| Protobuf | 30-50% | 0.15x | ❌ No | ✅ Yes | ✅ Yes |

**Verdict**: MessagePack is the sweet spot (fast, compact, inspectable, cross-language).

## 4.2 Cache Implementation

```python
class UniverseCompiler:
    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile universe with optional caching."""

        cache_path = config_dir / ".compiled_universe.msgpack"

        # Check cache
        if use_cache and cache_path.exists():
            # Compute hash of all config files
            current_hash = self._compute_config_hash(config_dir)

            # Load cached universe
            cached = CompiledUniverse.load(cache_path)

            if cached.metadata.config_hash == current_hash:
                print(f"✅ Using cached compilation (hash {current_hash[:8]}...)")
                return cached
            else:
                print(f"⚠️  Cache stale (hash mismatch), recompiling...")

        # Full compilation
        print("🔧 Compiling universe...")
        universe = self._full_compile(config_dir)

        # Cache for next time
        universe.save(cache_path)
        print(f"💾 Cached compilation to {cache_path}")

        return universe

    def _compute_config_hash(self, config_dir: Path) -> str:
        """Compute hash of all config files."""
        import hashlib

        files = [
            "substrate.yaml",
            "variables.yaml",
            "bars.yaml",
            "cascades.yaml",
            "cues.yaml",
            "affordances.yaml",
        ]

        content = []
        for filename in sorted(files):  # Sorted for deterministic hash
            path = config_dir / filename
            if path.exists():
                content.append(path.read_text())

        combined = "".join(content)
        return hashlib.sha256(combined.encode()).hexdigest()
```

**Cache Invalidation Strategy**:
- Hash all YAML file contents
- If hash matches cached universe, reuse cache
- If hash differs, recompile and update cache
- Cache stored alongside config pack (`.compiled_universe.msgpack`)

**Performance Impact**:
- **Cold start** (no cache): ~500-1000ms (full compilation)
- **Warm start** (cache hit): ~10-50ms (load MessagePack)
- **Cache miss** (stale hash): ~500-1000ms (recompile)

---

# 5. Cues as Communication Modality (Theory of Mind)

## 5.1 The Theory of Mind Problem

In multi-agent scenarios (L5+), agents need to model each other's internal states to cooperate effectively. There are three approaches:

**1. Full Telepathy** (Unrealistic):
```python
# Agent A directly observes Agent B's meters
obs_agent_a = {
    'other_agent_energy': 0.35,  # TELEPATHY - unrealistic
    'other_agent_health': 0.82,
}
```

**2. No Information** (Too Hard):
```python
# Agent A sees only position
obs_agent_a = {
    'other_agent_position': (3, 5),  # Can't infer internal state
}
```

**3. Cue-Based Inference** (Realistic + Pedagogical):
```python
# Agent A observes behavioral cues and must INFER internal state
obs_agent_a = {
    'other_agent_position': (3, 5),
    'other_agent_cue': 'exhausted',        # Visual appearance
    'other_agent_movement_speed': 0.5,     # Observable behavior
    'other_agent_limping': True,           # Health-based behavior
}
# Must infer: "Moving slowly + limping → probably low energy + low health"
```

## 5.2 Cues Configuration (Extended for Theory of Mind)

**Current cues.yaml** (visualization only):
```yaml
energy:
  color_map:
    - [0.0, 0.2]: "#FF0000"  # Red
    - [0.2, 0.8]: "#FFFF00"  # Yellow
    - [0.8, 1.0]: "#00FF00"  # Green
  icon: "⚡"
```

**Future cues.yaml** (communication modality):
```yaml
energy:
  visual_cues:
    - range: [0.0, 0.2]
      label: "exhausted"
      icon: "😴"
      observable_effects:
        movement_speed_modifier: 0.5   # Moves at half speed (observable!)
        action_delay: 2                # Takes 2x longer to interact

    - range: [0.2, 0.5]
      label: "tired"
      icon: "😐"
      observable_effects:
        movement_speed_modifier: 0.8

    - range: [0.5, 1.0]
      label: "energetic"
      icon: "😊"
      observable_effects:
        movement_speed_modifier: 1.0

health:
  visual_cues:
    - range: [0.0, 0.2]
      label: "critical"
      icon: "🩹"
      observable_effects:
        limping: true               # Observable behavioral change
        action_success_rate: 0.7    # Fails interactions 30% of time

    - range: [0.5, 1.0]
      label: "healthy"
      icon: "💪"
```

## 5.3 CuesCompiler Validation (First-Class Component)

**Critical Validations**:
1. **Cues reference valid meters** (cross-file validation)
2. **Ranges cover full [0.0, 1.0] domain** (no gaps)
3. **Ranges don't overlap** (unambiguous mapping)
4. **Observable effects are consistent with actions** (movement_speed_modifier doesn't contradict action definitions)

```python
class CuesCompiler:
    """Compiles cues.yaml into communication protocol."""

    def validate_references(self, cues_config, symbol_table, errors):
        """Validate cues reference valid meters."""
        for meter_name, cue_spec in cues_config.items():
            # Check meter exists
            if meter_name not in symbol_table.meters:
                errors.add(
                    f"Cue '{meter_name}' references non-existent meter",
                    hint=f"Available meters: {list(symbol_table.meters.keys())}"
                )

            # Check ranges cover full [0.0, 1.0] domain
            ranges = [cue['range'] for cue in cue_spec.visual_cues]
            if not self._ranges_cover_domain(ranges):
                errors.add(
                    f"Cue '{meter_name}' ranges don't cover full [0.0, 1.0] domain. "
                    f"Every meter value must map to exactly one cue."
                )

            # Check for overlapping ranges
            if self._ranges_overlap(ranges):
                errors.add(f"Cue '{meter_name}' has overlapping ranges")

    def _ranges_cover_domain(self, ranges: list[tuple[float, float]]) -> bool:
        """Check if ranges cover [0.0, 1.0] without gaps."""
        sorted_ranges = sorted(ranges, key=lambda r: r[0])

        # Check starts at 0.0
        if sorted_ranges[0][0] != 0.0:
            return False

        # Check ends at 1.0
        if sorted_ranges[-1][1] != 1.0:
            return False

        # Check no gaps between ranges
        for i in range(len(sorted_ranges) - 1):
            if sorted_ranges[i][1] != sorted_ranges[i+1][0]:
                return False

        return True

    def build_cue_observation_spec(self, cues_config, multi_agent: bool):
        """Build observation spec for cues.

        Single-agent: Returns empty spec (cues not observable)
        Multi-agent: Returns spec for observing other agents' cues
        """
        if not multi_agent:
            return []  # Cues not in observation space

        # In multi-agent mode, each agent observes others' cues
        cue_features = []
        for meter_name, cue_spec in cues_config.items():
            num_cue_states = len(cue_spec.visual_cues)
            cue_features.append(
                ObservationField(
                    name=f"other_agent_{meter_name}_cue",
                    type="categorical",
                    num_categories=num_cue_states,
                    description=f"Visual cue for other agent's {meter_name}"
                )
            )

        return cue_features
```

## 5.4 Pedagogical Value

**Question for Students**: "Your teammate is moving slowly and limping. What should you do?"

**Without Cues (Telepathy)**:
```python
# Student just reads teammate.energy directly
if teammate.energy < 0.3:
    share_food(teammate)
# NO INFERENCE REQUIRED - boring!
```

**With Cues (Theory of Mind)**:
```python
# Student must INFER internal state from observations
if teammate.movement_speed < 0.6 and teammate.limping:
    # Could be low energy OR low health OR both
    # Must build mental model through repeated observations
    # Must learn: "limping + slow → probably need help"
    approach_and_offer_help(teammate)
# INFERENCE REQUIRED - teaches ToM!
```

**Key Insight**: Cues force agents to build **mental models** of other agents rather than just reading their state directly. This is critical for cooperative multi-agent learning.

---

# 6. Checkpoint Compatibility and Versioning

## 6.1 The Checkpoint Transfer Problem

**Scenario**: Train agent on L1 (full observability), transfer to L2 (POMDP).

**Problem**: Different observation dimensions cause weight mismatch:
- L1: obs_dim = 29
- L2: obs_dim = 54

**Without versioning**: Silent failure or cryptic error during checkpoint load.

**With versioning**: Clear error message and compatibility check.

## 6.2 Universe Config Hash (BLOCKER 2 Solution)

From TASK-002C:

> **BLOCKER 2**: Curriculum-driven world model adaptation requires agent to observe `world_config_hash` (which physics rules apply), but no mechanism exists to expose this.

**Solution**: Store universe config hash in both CompiledUniverse and checkpoints.

```python
# When compiling universe
universe = UniverseCompiler().compile(config_dir)
# universe.metadata.config_hash = "a3b8c9d2..." (hash of all YAML files)

# When saving checkpoint
def save_checkpoint(agent, universe_hash, obs_dim, action_dim):
    torch.save({
        'network_state': agent.network.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'universe_hash': universe_hash,  # NEW: Config hash
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'episode': episode,
    }, checkpoint_path)

# When loading checkpoint
def load_checkpoint(checkpoint_path, universe):
    checkpoint = torch.load(checkpoint_path)

    # Check dimensions (HARD CONSTRAINT)
    universe.check_checkpoint_compatibility(checkpoint_path)

    # Check config hash (SOFT WARNING)
    if checkpoint['universe_hash'] != universe.metadata.config_hash:
        print(f"⚠️  Transfer learning: checkpoint trained on different config")
        print(f"   Checkpoint hash: {checkpoint['universe_hash'][:8]}...")
        print(f"   Current hash:    {universe.metadata.config_hash[:8]}...")
```

## 6.3 Compatibility Checks

```python
class CompiledUniverse:
    def check_checkpoint_compatibility(self, checkpoint_path: Path) -> None:
        """Verify checkpoint is compatible with this universe."""
        checkpoint = torch.load(checkpoint_path)

        # HARD CONSTRAINT: Observation dimension must match
        if checkpoint['obs_dim'] != self.metadata.obs_dim:
            raise ValueError(
                f"Checkpoint incompatible: obs_dim mismatch\n"
                f"  Checkpoint expects: {checkpoint['obs_dim']}\n"
                f"  Universe provides:  {self.metadata.obs_dim}\n"
                f"Cannot transfer weights across different observation dimensions."
            )

        # HARD CONSTRAINT: Action dimension must match
        if checkpoint['action_dim'] != self.metadata.action_dim:
            raise ValueError(
                f"Checkpoint incompatible: action_dim mismatch\n"
                f"  Checkpoint expects: {checkpoint['action_dim']}\n"
                f"  Universe provides:  {self.metadata.action_dim}\n"
                f"Cannot transfer weights across different action dimensions."
            )

        # SOFT WARNING: Config hash mismatch (transfer learning)
        if 'universe_hash' in checkpoint:
            if checkpoint['universe_hash'] != self.metadata.config_hash:
                import warnings
                warnings.warn(
                    f"Checkpoint trained on different universe config:\n"
                    f"  Checkpoint hash: {checkpoint['universe_hash']}\n"
                    f"  Current hash:    {self.metadata.config_hash}\n"
                    f"Transfer learning may behave unexpectedly (different physics/affordances/cascades).",
                    UserWarning
                )
```

**Pedagogical Use Case**:

```python
# Train agent on L1 universe
universe_L1 = UniverseCompiler().compile("configs/L1_full_observability")
# universe_L1.metadata.config_hash = "a3b8c9d2..."

agent = train(universe_L1)
save_checkpoint(agent, universe_hash=universe_L1.metadata.config_hash)

# Load checkpoint into L2 universe (POMDP)
universe_L2 = UniverseCompiler().compile("configs/L2_partial_observability")
# universe_L2.metadata.config_hash = "f4e7d1a9..."

load_checkpoint("checkpoint_L1.pth", universe_L2)
# ❌ Raises error: obs_dim mismatch (29 vs 54)

# Load checkpoint into L1 with TWEAKED cascades
universe_L1_tweaked = UniverseCompiler().compile("configs/L1_tweaked_cascades")
# universe_L1_tweaked.metadata.config_hash = "b7c2a4f8..."  # Different!

load_checkpoint("checkpoint_L1.pth", universe_L1_tweaked)
# ⚠️  Warning: universe hash mismatch, transfer learning may behave unexpectedly
# ✅ Loads successfully (same obs_dim, action_dim)
```

---

# 7. Future Extensibility

## 7.1 Adding New Sub-Compilers

The sequential sub-compiler architecture makes it easy to extend the UAC with new subsystems:

**Example: Adding RewardCompiler** (for custom reward shaping)

1. Create `src/townlet/universe/subcompilers/reward.py`:

```python
class RewardCompiler:
    """Compiles reward.yaml into reward function."""

    def load(self, config_path: Path) -> RewardConfig:
        """Load and validate reward configuration."""
        # Pydantic validation

    def validate_references(self, reward_config, symbol_table, errors):
        """Validate reward function references valid meters."""
        # Check meter names exist

    def build_reward_function(self, reward_config, symbol_table):
        """Build executable reward function."""
        # Pre-compute reward matrix
```

2. Add to UniverseCompiler:

```python
class UniverseCompiler:
    def __init__(self):
        # Existing sub-compilers
        self.substrate_compiler = SubstrateCompiler()
        # ...
        self.reward_compiler = RewardCompiler()  # NEW

    def compile(self, config_dir: Path) -> CompiledUniverse:
        # Stage 1: Parse
        reward_config = self.reward_compiler.load(config_dir / "reward.yaml")

        # Stage 3: Validate references
        self.reward_compiler.validate_references(reward_config, symbol_table, errors)

        # Stage 7: Emit
        universe = CompiledUniverse(
            # ...
            reward_config=reward_config,  # Include in artifact
        )
```

**Backward Compatibility**: Make reward.yaml optional with sensible defaults.

## 7.2 Adding Custom Validation Rules

The error collector pattern makes it easy to add new cross-validation rules:

```python
def _stage_4_cross_validate(self, raw_configs, symbol_table, errors):
    """Stage 4: Cross-validate."""

    # Existing validations
    self._validate_cascade_cycles(raw_configs.cascades, errors)
    self._validate_economic_balance(raw_configs.affordances, errors)
    self._validate_spatial_feasibility(raw_configs.substrate, raw_configs.affordances, errors)

    # NEW: Custom validation
    self._validate_reward_achievability(raw_configs.reward, raw_configs.affordances, errors)
```

## 7.3 Alternative Orchestrators

The data contract design supports alternative orchestration patterns:

**Example: Hyperparameter Search Orchestrator**

```python
class HyperparameterSearchOrchestrator:
    """Run multiple training runs with different brain architectures."""

    def run_search(self, universe_config_dir: Path, brain_configs: list[Path]):
        # Compile universe ONCE
        universe = UniverseCompiler().compile(universe_config_dir)

        results = []
        for brain_yaml in brain_configs:
            # Compile different brain
            brain = BrainCompiler().compile(
                brain_yaml,
                obs_spec=universe.observation_spec,
                action_dim=universe.metadata.action_dim
            )

            # Train
            runner = TrainingRunner(universe, brain)
            metrics = runner.train()
            results.append((brain_yaml, metrics))

        return results
```

---

# 8. Implementation Checklist

## 8.1 Priority 1: Core Data Contracts (Week 1)

- [ ] Create `src/townlet/universe/dto/observation_spec.py`
  - [ ] `ObservationField` dataclass
  - [ ] `ObservationSpec` dataclass with query methods
  - [ ] Unit tests for field lookup, slicing

- [ ] Create `src/townlet/universe/dto/action_metadata.py`
  - [ ] `ActionMetadata` dataclass
  - [ ] `ActionSpaceMetadata` dataclass with masking methods
  - [ ] Unit tests for action lookup, masking

- [ ] Create `src/townlet/universe/dto/meter_metadata.py`
  - [ ] `MeterInfo` dataclass
  - [ ] `MeterMetadata` dataclass
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/dto/affordance_metadata.py`
  - [ ] `AffordanceInfo` dataclass
  - [ ] `AffordanceMetadata` dataclass
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/dto/universe_metadata.py`
  - [ ] `UniverseMetadata` dataclass
  - [ ] Config hash computation
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/compiled.py`
  - [ ] `CompiledUniverse` class
  - [ ] Immutability enforcement
  - [ ] `create_environment()` factory method
  - [ ] `check_checkpoint_compatibility()` method
  - [ ] `save()` / `load()` serialization (MessagePack)
  - [ ] Unit tests

## 8.2 Priority 2: Sub-Compilers (Week 2)

- [ ] Create `src/townlet/universe/subcompilers/substrate.py`
  - [ ] `SubstrateCompiler` class
  - [ ] `load()` and `validate()` methods
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/subcompilers/vfs.py`
  - [ ] `VFSCompiler` class (TASK-002C integration)
  - [ ] `build_observation_spec()` method
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/subcompilers/bars.py`
  - [ ] `BarCompiler` class
  - [ ] Meter metadata extraction
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/subcompilers/cascades.py`
  - [ ] `CascadeCompiler` class
  - [ ] `validate_references()` method
  - [ ] `check_circular_dependencies()` method
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/subcompilers/cues.py`
  - [ ] `CuesCompiler` class
  - [ ] `validate_range_coverage()` method
  - [ ] `build_cue_observation_spec()` method (multi-agent)
  - [ ] Unit tests

- [ ] Refactor `src/townlet/environment/action_builder.py`
  - [ ] Extract into `src/townlet/universe/subcompilers/actions.py`
  - [ ] `ActionSpaceComposer` class
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/subcompilers/affordances.py`
  - [ ] `AffordanceCompiler` class
  - [ ] `validate_references()` method
  - [ ] Unit tests

## 8.3 Priority 3: Main Compiler (Week 2-3)

- [ ] Create `src/townlet/universe/compiler.py`
  - [ ] `UniverseCompiler` class
  - [ ] Seven-stage pipeline implementation
  - [ ] Cache management
  - [ ] Config hash computation
  - [ ] Integration tests

- [ ] Create `src/townlet/universe/symbol_table.py`
  - [ ] `UniverseSymbolTable` class
  - [ ] Entity registration methods
  - [ ] Reference resolution methods
  - [ ] Unit tests

- [ ] Create `src/townlet/universe/errors.py`
  - [ ] `CompilationError` exception
  - [ ] `CompilationErrorCollector` class
  - [ ] Error formatting with hints
  - [ ] Unit tests

## 8.4 Priority 4: Integration (Week 3-4)

- [ ] Update `src/townlet/environment/vectorized_env.py`
  - [ ] Accept `CompiledUniverse` in constructor
  - [ ] Use `observation_spec` for buffer allocation
  - [ ] Use `action_space` for masking
  - [ ] Remove ad-hoc config loading
  - [ ] Integration tests

- [ ] Update `src/townlet/demo/runner.py`
  - [ ] Use `UniverseCompiler` instead of direct config loading
  - [ ] Use `CompiledUniverse.create_environment()`
  - [ ] Store universe hash in checkpoints
  - [ ] Integration tests

- [ ] Update checkpoint saving/loading
  - [ ] Add `universe_hash` to checkpoint metadata
  - [ ] Add compatibility checks on load
  - [ ] Update inference server to check compatibility
  - [ ] Integration tests

---

# 9. Success Criteria

## 9.1 Functional Requirements

- [ ] **UAC compiles all existing config packs** (L0_0_minimal, L0_5_dual_resource, L1, L2, L3)
- [ ] **Cross-validation catches errors**:
  - [ ] Dangling meter references (cascades → non-existent meter)
  - [ ] Circular cascade dependencies
  - [ ] Economic imbalance (income < costs)
  - [ ] Incomplete cue ranges (gaps in [0.0, 1.0])
- [ ] **Cache works correctly**:
  - [ ] Cache hit when configs unchanged (~10-50ms load time)
  - [ ] Cache miss when configs changed (~500-1000ms recompile)
  - [ ] Cache invalidated correctly (hash mismatch)
- [ ] **Checkpoint compatibility enforced**:
  - [ ] Hard error on obs_dim mismatch
  - [ ] Hard error on action_dim mismatch
  - [ ] Soft warning on universe hash mismatch
- [ ] **BAC compiler uses ObservationSpec**:
  - [ ] Can build custom encoders by semantic type
  - [ ] Can slice observation tensor by field indices
  - [ ] Validates field names exist
- [ ] **Training system uses CompiledUniverse**:
  - [ ] Can instantiate environment via factory method
  - [ ] Can log per-meter metrics by name
  - [ ] Can apply action masks
  - [ ] Can check checkpoint compatibility

## 9.2 Non-Functional Requirements

- [ ] **Performance**:
  - [ ] Cold compilation < 1s (no cache)
  - [ ] Warm compilation < 50ms (cache hit)
  - [ ] MessagePack serialization < 100ms
- [ ] **Error Messages**:
  - [ ] Clear, actionable error messages with hints
  - [ ] Point to exact file + line where error occurred
  - [ ] Suggest fixes (e.g., "Available meters: [energy, health]")
- [ ] **Backward Compatibility**:
  - [ ] Existing config packs work without modification
  - [ ] Existing checkpoints can be loaded (with compatibility check)

---

# 10. References

## 10.1 Implementation Tasks

- **TASK-002A**: Configurable Spatial Substrates (substrate abstraction)
- **TASK-002B**: Composable Action Space (action system)
- **TASK-002C**: Variable & Feature System (VFS, typed variables)
- **TASK-003**: UAC Core DTOs (Pydantic schemas)
- **TASK-004A**: Compiler Implementation (7-stage pipeline)
- **TASK-004B**: UAC Capabilities (affordance effects)
- **TASK-005**: Brain as Code (BAC compiler)
- **TASK-006**: Substrate-Agnostic Visualization (frontend integration)

## 10.2 Research Documents

- **RESEARCH-UNIVERSE-COMPILER-DESIGN.md**: UAC compiler design philosophy
- **RESEARCH-UAC-COMPILER-INFRASTRUCTURE.md**: Infrastructure patterns

## 10.3 Architecture Documents

- **UNIVERSE_AS_CODE.md**: UAC design philosophy
- **BRAIN_AS_CODE.md**: BAC design philosophy
- **hld/01-executive-summary.md**: Framework overview
- **hld/02-brain-as-code.md**: Detailed BAC specification

---

# Glossary

**UAC (Universe as Code)**: Declarative YAML-based universe configuration system

**BAC (Brain as Code)**: Declarative YAML-based agent architecture configuration system

**CompiledUniverse**: Immutable artifact produced by UAC compiler, contains all universe metadata and configs

**ObservationSpec**: Rich specification of observation structure, consumed by BAC compiler

**Sub-Compiler**: Specialized compiler for one config file (SubstrateCompiler, CuesCompiler, etc.)

**Symbol Table**: Central registry of all named entities (meters, affordances, variables) for cross-file reference resolution

**Config Hash**: SHA-256 hash of all YAML file contents, used for cache invalidation and checkpoint versioning

**Cues**: Observable behavioral manifestations of internal state (movement speed, limping, etc.) that enable Theory of Mind without telepathy

**MessagePack**: Binary serialization format used for caching compiled universes

---

**End of Document**
