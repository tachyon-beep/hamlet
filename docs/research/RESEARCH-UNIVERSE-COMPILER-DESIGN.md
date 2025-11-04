# Research: Universe Compiler Design and Refactoring

## Executive Summary

This research examines **how to build a best-practice universe compiler** for HAMLET's UAC system and identifies **critical refactoring considerations** for moving from ad-hoc config loading to a robust compilation pipeline.

**KEY FINDINGS**:

1. **Multi-Pass Architecture** - Compiler should use 4-7 passes (parse → resolve → validate → optimize → emit)
2. **Symbol Tables Are Essential** - Need universe-wide symbol resolution (meter names, affordance IDs, action names)
3. **Error Recovery Matters** - Collect ALL errors, don't stop at first failure
4. **Lazy vs Eager Compilation** - Eager compilation (all-at-once) fits UAC use case better than incremental
5. **Immutability After Compilation** - Compiled universe should be frozen, not modified during training
6. **Caching Strategy Critical** - Compilation is expensive, must cache compiled artifacts
7. **Current System Needs Major Refactoring** - Independent loaders → unified compiler pipeline

**RECOMMENDATIONS**:

1. **Build UniverseCompiler with 7-stage pipeline** (11-16 hours)
2. **Implement SymbolTable for cross-file resolution** (4-6 hours)
3. **Create CompiledUniverse immutable artifact** (3-4 hours)
4. **Add comprehensive error collection and reporting** (4-6 hours)
5. **Implement compilation cache** (4-6 hours)

**Total Estimated Effort**: 26-38 hours (3-5 days)

---

## Part 1: Compiler Architecture Patterns

### Multi-Pass Compilation

**Concept**: Break compilation into distinct phases, each with clear responsibility.

**Classic Compiler Passes**:

```
Source Code
    ↓
PASS 1: Lexical Analysis (tokenize)
    ↓
PASS 2: Syntax Analysis (parse to AST)
    ↓
PASS 3: Semantic Analysis (type checking, symbol resolution)
    ↓
PASS 4: Optimization (constant folding, dead code elimination)
    ↓
PASS 5: Code Generation (emit target code)
```

**Applied to UAC**:

```
YAML Files
    ↓
STAGE 1: Parse & Validate Individual Files (bars, cascades, affordances, actions, substrate)
    ↓
STAGE 2: Build Symbol Tables (meter names, affordance IDs, action names)
    ↓
STAGE 3: Resolve References (cascade meter references, affordance cost meters)
    ↓
STAGE 4: Cross-Validate (spatial feasibility, economic balance, temporal conflicts)
    ↓
STAGE 5: Compute Metadata (obs_dim, action_dim, affordance vocab size)
    ↓
STAGE 6: Optimize (pre-compute tensors, build lookup tables)
    ↓
STAGE 7: Emit CompiledUniverse (immutable artifact)
```

**Why Multi-Pass?**

- ✅ **Separation of concerns**: Each pass has single responsibility
- ✅ **Better error messages**: Can pinpoint which stage failed
- ✅ **Easier testing**: Test each pass independently
- ✅ **Extensibility**: Add new passes without rewriting existing ones
- ✅ **Optimization opportunities**: Can reorder passes or skip unnecessary ones

**Why NOT Single-Pass?**

- ❌ **Forward references**: Can't resolve cascade meter names until bars are loaded
- ❌ **Cross-file dependencies**: Actions depend on substrate, affordances depend on bars
- ❌ **Error quality**: Single-pass must fail fast, can't collect multiple errors

**Recommendation**: **7-stage multi-pass compiler** (as outlined above)

---

### Symbol Tables and Reference Resolution

**Problem**: Configs reference entities by name (meter names, affordance IDs, action names), but these are defined across multiple files.

**Example References**:

```yaml
# cascades.yaml
cascades:
  - source: "mood"        # References meter from bars.yaml
    target: "energy"      # References meter from bars.yaml

# affordances.yaml
affordances:
  - costs:
      - meter: "money"    # References meter from bars.yaml
```

**Without Symbol Table**:

- Validate each file independently → miss dangling references
- Or embed validation in each loader → tight coupling, code duplication

**With Symbol Table**:

```python
class UniverseSymbolTable:
    """Central registry of all named entities in universe."""

    def __init__(self):
        self.meters: dict[str, BarConfig] = {}
        self.affordances: dict[str, AffordanceConfig] = {}
        self.actions: dict[str, ActionConfig] = {}
        self.cascades: dict[str, CascadeConfig] = {}

    def register_meter(self, name: str, config: BarConfig):
        """Register meter for later reference resolution."""
        if name in self.meters:
            raise CompilationError(f"Duplicate meter name: {name}")
        self.meters[name] = config

    def resolve_meter_reference(self, name: str, location: str) -> BarConfig:
        """Resolve meter reference, raising clear error on failure."""
        if name not in self.meters:
            raise CompilationError(
                f"{location}: References non-existent meter '{name}'. "
                f"Valid meters: {list(self.meters.keys())}"
            )
        return self.meters[name]
```

**Usage in Compiler**:

```python
# STAGE 1: Load bars, populate symbol table
bars_config = load_bars_config(config_dir / "bars.yaml")
for bar in bars_config.bars:
    symbol_table.register_meter(bar.name, bar)

# STAGE 3: Resolve cascade references
for cascade in cascades_config.cascades:
    source_bar = symbol_table.resolve_meter_reference(
        cascade.source,
        location=f"cascades.yaml:{cascade.name}"
    )
    target_bar = symbol_table.resolve_meter_reference(
        cascade.target,
        location=f"cascades.yaml:{cascade.name}"
    )
```

**Benefits**:

- ✅ **Clear error messages**: "cascades.yaml:low_mood_hits_energy: References non-existent meter 'moodiness'. Valid meters: [energy, health, mood, ...]"
- ✅ **Centralized resolution**: All reference resolution logic in one place
- ✅ **Extensible**: Easy to add new entity types (actions, stages, etc.)

**Recommendation**: **Implement UniverseSymbolTable for Stage 2-3**

---

### Error Collection vs Fail-Fast

**Two Approaches**:

**1. Fail-Fast** (current Pydantic behavior):

```python
# PROBLEM: Stops at first error
config = BarsConfig(**data)  # ValidationError: "index must be 0-7"
# User fixes index error, runs again...
# ValidationError: "Duplicate name 'energy'"
# User fixes duplicate, runs again...
# ValidationError: "Initial value must be 0.0-1.0"
# Annoying!
```

**2. Error Collection**:

```python
errors = []

# Collect ALL errors before failing
for bar in bars_config.bars:
    if bar.index < 0 or bar.index > 31:
        errors.append(f"{bar.name}: index out of range (got {bar.index})")

    if bar.initial < 0.0 or bar.initial > 1.0:
        errors.append(f"{bar.name}: initial value out of range (got {bar.initial})")

if errors:
    raise CompilationError("\n".join([
        "bars.yaml validation failed with 3 errors:",
        "  1. energy: index out of range (got 32)",
        "  2. mood: initial value out of range (got 1.5)",
        "  3. health: duplicate name 'health' found in bars"
    ]))
```

**Tradeoffs**:

| Approach | Pros | Cons |
|----------|------|------|
| **Fail-Fast** | Simple implementation, stops early | Frustrating UX (fix one error → hit next error) |
| **Error Collection** | Better UX (see all errors at once), faster iteration | More complex implementation, must track all errors |

**Best Practice**: **Use hybrid approach**:

- Fail-fast for **critical errors** (file not found, YAML parse error)
- Collect errors for **validation** (index out of range, dangling reference)

**Implementation**:

```python
class CompilationErrorCollector:
    """Collect multiple validation errors before failing."""

    def __init__(self):
        self.errors: list[str] = []

    def add_error(self, error: str):
        """Record validation error."""
        self.errors.append(error)

    def check_and_raise(self, stage: str):
        """If errors exist, raise CompilationError with all messages."""
        if self.errors:
            error_list = "\n  ".join([f"{i+1}. {e}" for i, e in enumerate(self.errors)])
            raise CompilationError(
                f"{stage} failed with {len(self.errors)} errors:\n  {error_list}"
            )

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0

# Usage
errors = CompilationErrorCollector()

for cascade in cascades_config.cascades:
    if cascade.source not in symbol_table.meters:
        errors.add_error(f"{cascade.name}: source meter '{cascade.source}' not found")
    if cascade.target not in symbol_table.meters:
        errors.add_error(f"{cascade.name}: target meter '{cascade.target}' not found")

errors.check_and_raise("Stage 3: Reference Resolution")
```

**Recommendation**: **Implement CompilationErrorCollector for Stage 3-4 validation**

---

### Eager vs Lazy Compilation

**Two Strategies**:

**1. Eager Compilation** (compile entire universe upfront):

```python
# Load all YAMLs, validate everything, emit compiled artifact
universe = UniverseCompiler.compile("configs/L1_full_observability")

# Compiled universe is ready to use immediately
env = VectorizedHamletEnv(compiled_universe=universe)
```

**2. Lazy Compilation** (load configs on-demand):

```python
# Load only what's needed
bars = load_bars_config("configs/L1_full_observability/bars.yaml")
# Don't load cascades until they're actually used
```

**Tradeoffs**:

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Eager** | Catch all errors upfront, can cache compiled universe, startup penalty amortized across runs | Slower first startup, wastes work if config rarely used | Production, repeated training runs |
| **Lazy** | Fast initial load, only pay for what you use | Errors discovered late (during training!), hard to cache, complex dependency management | Development, exploratory work |

**Analysis for UAC**:

✅ **Eager compilation fits UAC better**:

- Training runs are long (thousands of episodes) → startup cost negligible
- Want to catch config errors before training starts (fail fast)
- Can cache compiled universe for 10-50x faster subsequent loads
- Universe is small (~5 YAML files) → compilation is fast anyway (<100ms)

❌ **Lazy compilation doesn't help**:

- Delayed error discovery (find cascade error 10 minutes into training)
- Can't pre-compute tensors (need full universe)
- Complexity of dependency tracking not worth the savings

**Recommendation**: **Eager compilation with caching**

**Compilation Cache Implementation**:

```python
class UniverseCompiler:
    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile universe with optional caching."""

        cache_path = config_dir / ".compiled" / "universe.msgpack"

        # Try cache first
        if use_cache and cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime

            # Check if any source YAML is newer than cache
            yaml_files = list(config_dir.glob("*.yaml"))
            if all(f.stat().st_mtime < cache_mtime for f in yaml_files):
                # Cache is fresh, load it
                return CompiledUniverse.load_from_cache(cache_path)

        # Cache miss or stale, do full compilation
        universe = self._compile_from_source(config_dir)

        # Save to cache
        cache_path.parent.mkdir(exist_ok=True)
        universe.save_to_cache(cache_path)

        return universe
```

**Benefits**:

- First run: ~50-100ms compilation
- Cached runs: ~1-5ms load (50-100x speedup)
- Cache invalidation: automatic via mtime comparison

---

## Part 2: Immutability and Compiled Artifacts

### Why Immutability Matters

**Problem**: Mutable configs allow accidental modification during training.

**Example of Danger**:

```python
# Load config
bars_config = load_bars_config("configs/L1/bars.yaml")

# Somewhere deep in code...
bars_config.bars[0].initial = 0.5  # OOPS! Accidentally modified

# Now training starts with different initial values than YAML specifies
# Results are not reproducible!
```

**Solution**: **Immutable CompiledUniverse**

```python
from dataclasses import dataclass

@dataclass(frozen=True)  # Immutable!
class CompiledUniverse:
    """Immutable artifact representing a fully validated universe."""

    # Config components
    bars: BarsConfig
    substrate: SubstrateConfig
    actions: ActionSpaceConfig
    cascades: CascadesConfig
    affordances: AffordanceConfigCollection
    training: TrainingConfig

    # Computed metadata (derived from configs)
    metadata: UniverseMetadata

    # Pre-computed optimization data
    optimization_data: OptimizationData

    def __post_init__(self):
        """Validate universe is complete and consistent (runs once at construction)."""
        # This is the ONLY time we can validate (frozen after this)
        if self.metadata.meter_count != len(self.bars.bars):
            raise ValueError("Metadata meter_count doesn't match bars")

        # Additional invariant checks...

# Trying to modify raises error:
universe = compiled_universe
universe.bars.bars[0].initial = 0.5  # AttributeError: cannot assign to field
```

**Benefits**:

- ✅ **No accidental mutation** - Python enforces immutability at runtime
- ✅ **Safe to share** - Can pass same universe to multiple environments
- ✅ **Cacheable** - Immutable objects can be safely serialized
- ✅ **Reproducible** - Universe matches YAML exactly, no hidden state
- ✅ **Thread-safe** - No locks needed (immutable = safe)

**Metadata Examples**:

```python
@dataclass(frozen=True)
class UniverseMetadata:
    """Derived metadata about the universe."""

    # Computed from configs
    meter_count: int  # len(bars.bars)
    meter_names: list[str]  # Sorted by index
    meter_name_to_index: dict[str, int]

    affordance_count: int
    affordance_ids: list[str]
    affordance_id_to_index: dict[str, int]

    action_count: int
    action_names: list[str]

    # Observation space metadata
    observation_dim: int  # grid_size² + meter_count + affordances + extras

    # Spatial metadata
    grid_cells: int  # substrate.dimensions[0] * substrate.dimensions[1]

    # Economic metadata
    max_sustainable_income: float
    total_affordance_costs: float
    economic_balance: float  # income / costs (>1.0 = sustainable)

    # Temporal metadata
    ticks_per_day: int  # Usually 24

    # Version tracking
    config_version: str
    compiler_version: str
    compiled_at: str  # ISO timestamp
```

**Recommendation**: **Use frozen dataclass for CompiledUniverse with comprehensive metadata**

---

## Part 3: Current System Refactoring

### Current Architecture (Problems)

**Current Loading Pattern**:

```python
# In VectorizedHamletEnv.__init__()

# Load affordances (independent)
affordance_config = load_affordance_config(config_path)

# Load meter dynamics (independent, loads bars + cascades internally)
self.meter_dynamics = MeterDynamics(cascade_config_dir=self.config_pack_path)

# Hardcoded observation dim calculation
self.observation_dim = grid_size * grid_size + 8 + (num_affordance_types + 1)
#                                                ^^^ HARDCODED!

# Hardcoded action dim
self.action_dim = 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
#                ^^^ HARDCODED!
```

**Problems**:

1. **No Central Orchestration**
   - Each component loads its own configs
   - No validation that configs are compatible
   - Can't cache compiled universe

2. **Hidden Dependencies**
   - `MeterDynamics` secretly loads bars + cascades
   - `AffordanceEngine` validates meter names against hardcoded dict
   - Observation builder doesn't know about meter count

3. **Hardcoded Constants**
   - Meter count: 8 (line 120, 124)
   - Action count: 6 (line 157)
   - Can't detect these from config

4. **No Cross-Validation**
   - Cascades can reference non-existent meters
   - Too many affordances for grid size → silent failure
   - Economic imbalance → discovered during training

5. **Tight Coupling**
   - `VectorizedHamletEnv` does config loading
   - Can't unit test compilation without full environment
   - Can't pre-compile configs offline

---

### Target Architecture (Solution)

**New Loading Pattern**:

```python
# STEP 1: Compile universe (can happen offline, cached)
compiler = UniverseCompiler()
universe = compiler.compile("configs/L1_full_observability")
# ^ This does ALL validation, reference resolution, metadata computation

# STEP 2: Create environment from compiled universe
env = VectorizedHamletEnv(universe=universe)
# ^ No config loading, no validation, just use pre-compiled artifact

# Environment just reads metadata
obs_dim = universe.metadata.observation_dim
action_dim = universe.metadata.action_count
meter_count = universe.metadata.meter_count
```

**Benefits**:

✅ **Separation of Concerns**:

- `UniverseCompiler`: Load, validate, optimize configs
- `VectorizedHamletEnv`: Execute universe, no config logic

✅ **Fail-Fast**:

- All config errors caught during compilation
- Training never starts with invalid universe

✅ **Cacheable**:

- Compiled universe saved to disk
- Subsequent runs: ~50x faster startup

✅ **Testable**:

- Can test compiler without environment
- Can test environment with mock universes

✅ **Flexible**:

- Can compile offline: `python -m townlet.compiler compile configs/L1`
- Can inspect compiled universe: `python -m townlet.compiler inspect .compiled/universe.msgpack`

---

### Refactoring Steps

**Phase 1: Extract Compilation Logic** (6-8 hours)

Move config loading from `VectorizedHamletEnv` to `UniverseCompiler`:

```python
# Before: In VectorizedHamletEnv.__init__()
affordance_config = load_affordance_config(config_path)
meter_dynamics = MeterDynamics(cascade_config_dir=config_pack_path)

# After: In UniverseCompiler.compile()
class UniverseCompiler:
    def compile(self, config_dir: Path) -> CompiledUniverse:
        # Stage 1: Load all YAMLs
        bars = load_bars_config(config_dir / "bars.yaml")
        cascades = load_cascades_config(config_dir / "cascades.yaml")
        affordances = load_affordance_config(config_dir / "affordances.yaml")
        training = load_training_config(config_dir / "training.yaml")

        # Stage 2-7: Validate, resolve, optimize, emit
        ...

        return CompiledUniverse(...)
```

**Tests**:

- Can compile valid universe
- Compilation fails on invalid universe
- Error messages are clear

---

**Phase 2: Implement Symbol Table** (4-6 hours)

Add reference resolution:

```python
# Stage 2: Build symbol tables
symbol_table = UniverseSymbolTable()

for bar in bars.bars:
    symbol_table.register_meter(bar.name, bar)

for aff in affordances.affordances:
    symbol_table.register_affordance(aff.id, aff)

# Stage 3: Resolve references
for cascade in cascades.cascades:
    try:
        source = symbol_table.resolve_meter_reference(cascade.source)
        target = symbol_table.resolve_meter_reference(cascade.target)
    except ReferenceError as e:
        errors.add_error(str(e))
```

**Tests**:

- Symbol table registers entities
- Symbol table resolves valid references
- Symbol table raises error on dangling references

---

**Phase 3: Add Cross-Validation** (4-6 hours)

Validate cross-file constraints:

```python
# Stage 4: Cross-validate
validator = CrossValidator(symbol_table, errors)

validator.check_spatial_feasibility(substrate, affordances, training)
validator.check_economic_balance(affordances, training)
validator.check_cascade_circularity(cascades)
validator.check_temporal_conflicts(affordances)

errors.check_and_raise("Stage 4: Cross-Validation")
```

**Tests**:

- Spatial validation catches too many affordances
- Economic validation warns on poverty traps
- Cascade validation catches circular dependencies

---

**Phase 4: Compute Metadata** (3-4 hours)

Calculate observation dim, action dim, etc.:

```python
# Stage 5: Compute metadata
metadata = UniverseMetadata(
    meter_count=len(bars.bars),
    meter_names=[bar.name for bar in sorted(bars.bars, key=lambda b: b.index)],
    observation_dim=self._compute_observation_dim(substrate, bars, affordances),
    action_count=len(actions.actions),
    ...
)

def _compute_observation_dim(self, substrate, bars, affordances) -> int:
    grid_dim = substrate.dimensions[0] * substrate.dimensions[1]
    meter_dim = len(bars.bars)  # DYNAMIC!
    affordance_dim = len(affordances.affordances) + 1  # +1 for "none"
    extras_dim = 4  # time_sin, time_cos, progress, lifetime

    return grid_dim + meter_dim + affordance_dim + extras_dim
```

**Tests**:

- Observation dim correct for 4-meter universe
- Observation dim correct for 8-meter universe
- Observation dim correct for 12-meter universe

---

**Phase 5: Pre-Compute Optimizations** (4-6 hours)

Build lookup tables, tensors:

```python
# Stage 6: Optimize
optimization_data = OptimizationData(
    base_depletions=self._build_base_depletion_tensor(bars),
    cascade_data=self._build_cascade_lookup(cascades),
    modulation_data=self._build_modulation_lookup(cascades),
    action_mask_table=self._build_action_mask_table(affordances),
    affordance_position_map=self._build_affordance_map(affordances),
)
```

**Tests**:

- Base depletion tensor has correct shape [meter_count]
- Cascade lookup contains all cascades by category
- Action mask table has shape [24, num_affordances]

---

**Phase 6: Emit Compiled Universe** (2-3 hours)

Create immutable artifact:

```python
# Stage 7: Emit
universe = CompiledUniverse(
    bars=bars,
    cascades=cascades,
    affordances=affordances,
    training=training,
    metadata=metadata,
    optimization_data=optimization_data,
)

# Validate immutability
assert universe.__dataclass_fields__['bars'].frozen
return universe
```

**Tests**:

- Compiled universe is frozen (can't modify)
- Compiled universe contains all components
- Compiled universe metadata is correct

---

**Phase 7: Update VectorizedHamletEnv** (3-4 hours)

Refactor to accept compiled universe:

```python
class VectorizedHamletEnv:
    def __init__(
        self,
        universe: CompiledUniverse,  # NEW: Accept compiled universe
        num_agents: int,
        device: str = "cuda",
        # Remove: config_pack_path, enabled_affordances (in universe.training)
    ):
        self.universe = universe
        self.num_agents = num_agents
        self.device = device

        # Read metadata instead of computing
        self.observation_dim = universe.metadata.observation_dim
        self.action_dim = universe.metadata.action_count
        self.meter_count = universe.metadata.meter_count

        # Initialize components with universe
        self.meter_dynamics = MeterDynamics(universe=universe, num_agents=num_agents, device=device)
        self.affordance_engine = AffordanceEngine(universe=universe, num_agents=num_agents, device=device)

        # Use pre-computed optimization data
        self.base_depletions = universe.optimization_data.base_depletions
        ...
```

**Tests**:

- Environment accepts compiled universe
- Environment reads metadata correctly
- Environment behavior unchanged (integration test)

---

## Part 4: Compilation Stages in Detail

### Stage 1: Parse & Validate Individual Files

**Goal**: Load YAMLs, validate basic syntax and schema.

**Operations**:

```python
def stage_1_parse_individual_files(self, config_dir: Path) -> RawConfigs:
    """Parse and validate individual YAML files."""

    try:
        bars = load_bars_config(config_dir / "bars.yaml")
    except FileNotFoundError:
        raise CompilationError("bars.yaml not found")
    except yaml.YAMLError as e:
        raise CompilationError(f"bars.yaml is malformed: {e}")
    except ValidationError as e:
        raise CompilationError(f"bars.yaml validation failed: {e}")

    # Same for cascades, affordances, actions, substrate, training
    ...

    return RawConfigs(
        bars=bars,
        cascades=cascades,
        affordances=affordances,
        actions=actions,
        substrate=substrate,
        training=training,
    )
```

**Error Examples**:

- `FileNotFoundError`: "bars.yaml not found in configs/L1_full_observability"
- `yaml.YAMLError`: "bars.yaml line 15: unexpected indent"
- `ValidationError`: "bars.yaml: bar 'energy' has index=-1 (must be >= 0)"

**Tests**:

- Valid configs parse successfully
- Missing files raise clear error
- Malformed YAML raises parse error
- Invalid schema raises validation error

---

### Stage 2: Build Symbol Tables

**Goal**: Register all named entities for reference resolution.

**Operations**:

```python
def stage_2_build_symbol_tables(self, raw_configs: RawConfigs) -> UniverseSymbolTable:
    """Build symbol tables from raw configs."""

    symbol_table = UniverseSymbolTable()

    # Register meters
    for bar in raw_configs.bars.bars:
        symbol_table.register_meter(bar.name, bar)

    # Register affordances
    for aff in raw_configs.affordances.affordances:
        symbol_table.register_affordance(aff.id, aff)

    # Register actions
    for action in raw_configs.actions.actions:
        symbol_table.register_action(action.id, action)

    return symbol_table
```

**Error Examples**:

- `DuplicateSymbolError`: "Duplicate meter name 'energy' found in bars.yaml"
- `DuplicateSymbolError`: "Duplicate affordance ID 'Bed' found in affordances.yaml"

**Tests**:

- Symbol table registers all entities
- Symbol table detects duplicates
- Symbol table provides lookup by name

---

### Stage 3: Resolve References

**Goal**: Validate all cross-file references.

**Operations**:

```python
def stage_3_resolve_references(
    self,
    raw_configs: RawConfigs,
    symbol_table: UniverseSymbolTable,
    errors: CompilationErrorCollector
):
    """Resolve and validate all cross-file references."""

    # Resolve cascade meter references
    for cascade in raw_configs.cascades.cascades:
        if cascade.source not in symbol_table.meters:
            errors.add_error(
                f"cascades.yaml:{cascade.name}: "
                f"source meter '{cascade.source}' not found. "
                f"Valid meters: {list(symbol_table.meters.keys())}"
            )

        if cascade.target not in symbol_table.meters:
            errors.add_error(
                f"cascades.yaml:{cascade.name}: "
                f"target meter '{cascade.target}' not found"
            )

    # Resolve affordance meter references
    for aff in raw_configs.affordances.affordances:
        for cost in aff.costs + aff.costs_per_tick:
            if cost.meter not in symbol_table.meters:
                errors.add_error(
                    f"affordances.yaml:{aff.id}: "
                    f"cost meter '{cost.meter}' not found"
                )

        for effect in aff.effects + aff.effects_per_tick + aff.completion_bonus:
            if effect.meter not in symbol_table.meters:
                errors.add_error(
                    f"affordances.yaml:{aff.id}: "
                    f"effect meter '{effect.meter}' not found"
                )

    # Resolve action requirements
    # (e.g., movement actions require spatial substrate)

    errors.check_and_raise("Stage 3: Reference Resolution")
```

**Error Examples**:

- `ReferenceError`: "cascades.yaml:low_mood_hits_energy: source meter 'moodiness' not found. Valid meters: [energy, health, mood, money, satiation, hygiene, social, fitness]"
- `ReferenceError`: "affordances.yaml:Bed: effect meter 'stamina' not found"

**Tests**:

- Valid references resolve successfully
- Invalid meter references caught
- Invalid affordance references caught
- Error messages list valid alternatives

---

### Stage 4: Cross-Validate

**Goal**: Validate constraints that span multiple configs.

**Operations**:

```python
def stage_4_cross_validate(
    self,
    raw_configs: RawConfigs,
    symbol_table: UniverseSymbolTable,
    errors: CompilationErrorCollector
):
    """Validate cross-file constraints."""

    # 1. Spatial feasibility
    grid_cells = raw_configs.substrate.dimensions[0] * raw_configs.substrate.dimensions[1]
    enabled_affordances = raw_configs.training.enabled_affordances
    required_cells = len(enabled_affordances) + 1  # +1 for agent

    if required_cells > grid_cells:
        errors.add_error(
            f"Spatial impossibility: Grid has {grid_cells} cells but need {required_cells} "
            f"({len(enabled_affordances)} affordances + 1 agent)"
        )

    # 2. Economic balance
    total_income = sum(
        aff.effects[0].amount  # Assuming first effect is income
        for aff in raw_configs.affordances.affordances
        if aff.category == "income"
    )

    total_costs = sum(
        cost.amount
        for aff in raw_configs.affordances.affordances
        for cost in aff.costs
    )

    if total_income < total_costs:
        errors.add_error(
            f"Economic imbalance: Total income ({total_income}) < total costs ({total_costs}). "
            f"Universe may be poverty trap."
        )

    # 3. Cascade circularity detection
    cascade_graph = self._build_cascade_graph(raw_configs.cascades)
    cycles = self._detect_cycles(cascade_graph)

    if cycles:
        for cycle in cycles:
            errors.add_error(
                f"Cascade circularity detected: {' → '.join(cycle + [cycle[0]])}"
            )

    # 4. Temporal conflicts
    for aff in raw_configs.affordances.affordances:
        open_hour, close_hour = aff.operating_hours
        if open_hour < 0 or open_hour > 23:
            errors.add_error(
                f"affordances.yaml:{aff.id}: open_hour must be 0-23, got {open_hour}"
            )
        if close_hour < 1 or close_hour > 28:
            errors.add_error(
                f"affordances.yaml:{aff.id}: close_hour must be 1-28, got {close_hour}"
            )

    # 5. Action-substrate compatibility
    if raw_configs.substrate.type == "aspatial":
        movement_actions = [a for a in raw_configs.actions.actions if a.type == "movement"]
        if movement_actions:
            errors.add_error(
                f"Action-substrate incompatibility: Aspatial substrate cannot have movement actions. "
                f"Found: {[a.id for a in movement_actions]}"
            )

    errors.check_and_raise("Stage 4: Cross-Validation")
```

**Error Examples**:

- `SpatialError`: "Spatial impossibility: Grid has 9 cells but need 11 (10 affordances + 1 agent)"
- `EconomicWarning`: "Economic imbalance: Total income (20.0) < total costs (30.0). Universe may be poverty trap."
- `CircularityError`: "Cascade circularity detected: mood → energy → health → mood"
- `TemporalError`: "affordances.yaml:Bar: close_hour must be 1-28, got 30"
- `CompatibilityError`: "Action-substrate incompatibility: Aspatial substrate cannot have movement actions. Found: [move_up, move_down]"

**Tests**:

- Spatial validation catches over-crowded grids
- Economic validation warns on imbalances
- Circularity detection catches cycles
- Temporal validation catches invalid hours
- Action-substrate validation catches incompatibilities

---

### Stage 5: Compute Metadata

**Goal**: Calculate derived properties (obs_dim, action_dim, etc.)

**Operations**:

```python
def stage_5_compute_metadata(
    self,
    raw_configs: RawConfigs,
    symbol_table: UniverseSymbolTable
) -> UniverseMetadata:
    """Compute metadata from validated configs."""

    # Meter metadata
    meter_count = len(raw_configs.bars.bars)
    meter_names = [bar.name for bar in sorted(raw_configs.bars.bars, key=lambda b: b.index)]
    meter_name_to_index = {bar.name: bar.index for bar in raw_configs.bars.bars}

    # Affordance metadata
    affordance_count = len(raw_configs.affordances.affordances)
    affordance_ids = [aff.id for aff in raw_configs.affordances.affordances]
    affordance_id_to_index = {aff.id: i for i, aff in enumerate(raw_configs.affordances.affordances)}

    # Action metadata
    action_count = len(raw_configs.actions.actions)
    action_names = [action.id for action in raw_configs.actions.actions]

    # Observation dimension (complex calculation)
    if raw_configs.training.partial_observability:
        window_size = 2 * raw_configs.training.vision_range + 1
        obs_dim = (
            window_size * window_size +  # Local grid
            2 +                           # Agent position (x, y)
            meter_count +                 # DYNAMIC meter count!
            affordance_count + 1 +        # Affordance at position (+ "none")
            4                             # Temporal extras
        )
    else:
        grid_dim = raw_configs.substrate.dimensions[0] * raw_configs.substrate.dimensions[1]
        obs_dim = (
            grid_dim +                    # Full grid
            meter_count +                 # DYNAMIC meter count!
            affordance_count + 1 +        # Affordance at position
            4                             # Temporal extras
        )

    # Spatial metadata
    grid_cells = raw_configs.substrate.dimensions[0] * raw_configs.substrate.dimensions[1]

    # Economic metadata
    max_income = self._compute_max_sustainable_income(raw_configs.affordances)
    total_costs = self._compute_total_affordance_costs(raw_configs.affordances)
    economic_balance = max_income / total_costs if total_costs > 0 else float('inf')

    # Version tracking
    config_version = raw_configs.bars.version
    compiler_version = "1.0.0"
    compiled_at = datetime.now().isoformat()

    return UniverseMetadata(
        meter_count=meter_count,
        meter_names=meter_names,
        meter_name_to_index=meter_name_to_index,
        affordance_count=affordance_count,
        affordance_ids=affordance_ids,
        affordance_id_to_index=affordance_id_to_index,
        action_count=action_count,
        action_names=action_names,
        observation_dim=obs_dim,
        grid_cells=grid_cells,
        max_sustainable_income=max_income,
        total_affordance_costs=total_costs,
        economic_balance=economic_balance,
        ticks_per_day=24,
        config_version=config_version,
        compiler_version=compiler_version,
        compiled_at=compiled_at,
    )
```

**Tests**:

- Metadata has correct meter_count for variable-size configs
- Observation dim scales with meter_count
- Observation dim correct for partial observability
- Economic metadata computed correctly

---

### Stage 6: Optimize (Pre-Compute)

**Goal**: Build lookup tables and tensors for fast runtime execution.

**Operations**:

```python
def stage_6_optimize(
    self,
    raw_configs: RawConfigs,
    metadata: UniverseMetadata,
    device: str = "cpu"
) -> OptimizationData:
    """Pre-compute optimization data for fast runtime."""

    # 1. Base depletion tensor [meter_count]
    base_depletions = torch.zeros(metadata.meter_count, device=device)
    for bar in raw_configs.bars.bars:
        base_depletions[bar.index] = bar.base_depletion

    # 2. Cascade lookup by category
    cascade_data = {}
    for cascade in raw_configs.cascades.cascades:
        category = cascade.category
        if category not in cascade_data:
            cascade_data[category] = []

        cascade_data[category].append({
            "source_idx": metadata.meter_name_to_index[cascade.source],
            "target_idx": metadata.meter_name_to_index[cascade.target],
            "threshold": cascade.threshold,
            "strength": cascade.strength,
        })

    # Sort by target_idx for cache locality
    for category in cascade_data:
        cascade_data[category].sort(key=lambda c: c["target_idx"])

    # 3. Modulation lookup
    modulation_data = []
    for mod in raw_configs.cascades.modulations:
        modulation_data.append({
            "source_idx": metadata.meter_name_to_index[mod.source],
            "target_idx": metadata.meter_name_to_index[mod.target],
            "base_multiplier": mod.base_multiplier,
            "range": mod.range,
            "baseline_depletion": mod.baseline_depletion,
        })

    # 4. Action mask table [24, num_affordances]
    action_mask_table = torch.zeros((24, metadata.affordance_count), dtype=torch.bool, device=device)
    for hour in range(24):
        for aff_idx, aff in enumerate(raw_configs.affordances.affordances):
            open_hour, close_hour = aff.operating_hours
            action_mask_table[hour, aff_idx] = self._is_open(hour, open_hour, close_hour)

    # 5. Affordance position map (will be populated at reset)
    affordance_position_map = {}
    for aff in raw_configs.affordances.affordances:
        affordance_position_map[aff.id] = None  # Populated dynamically

    return OptimizationData(
        base_depletions=base_depletions,
        cascade_data=cascade_data,
        modulation_data=modulation_data,
        action_mask_table=action_mask_table,
        affordance_position_map=affordance_position_map,
    )
```

**Tests**:

- Base depletion tensor has correct shape
- Cascade data sorted by target index
- Action mask table correct for 24-hour cycle
- Optimization data complete

---

### Stage 7: Emit CompiledUniverse

**Goal**: Create immutable artifact with all components.

**Operations**:

```python
def stage_7_emit_compiled_universe(
    self,
    raw_configs: RawConfigs,
    metadata: UniverseMetadata,
    optimization_data: OptimizationData
) -> CompiledUniverse:
    """Emit immutable compiled universe."""

    universe = CompiledUniverse(
        bars=raw_configs.bars,
        cascades=raw_configs.cascades,
        affordances=raw_configs.affordances,
        actions=raw_configs.actions,
        substrate=raw_configs.substrate,
        training=raw_configs.training,
        metadata=metadata,
        optimization_data=optimization_data,
    )

    # Validate immutability
    if not universe.__dataclass_fields__['bars'].frozen:
        raise CompilationError("CompiledUniverse must be frozen (immutable)")

    return universe
```

**Tests**:

- Compiled universe contains all components
- Compiled universe is frozen (immutable)
- Compiled universe metadata is correct

---

## Part 5: Caching Strategy

### Cache File Format

**Options**:

| Format | Pros | Cons |
|--------|------|------|
| **JSON** | Human-readable, widely supported | Slow, large files, no tensor support |
| **YAML** | Human-readable | Even slower, no tensor support |
| **Pickle** | Fast, Python-native, tensor support | Security risk (arbitrary code execution), Python-only |
| **MessagePack** | Fast, compact, cross-language | Requires library, less human-readable |
| **HDF5** | Tensor-optimized, very fast | Complex, overkill for small data |

**Recommendation**: **MessagePack** for compiled universe cache

**Rationale**:

- ✅ Fast deserialization (~10x faster than JSON)
- ✅ Compact (50-70% smaller than JSON)
- ✅ Safe (no arbitrary code execution like Pickle)
- ✅ Supports tensors (with custom serializer)
- ✅ Cross-language (future Python→Rust migration)

**Implementation**:

```python
import msgpack

class CompiledUniverse:
    def save_to_cache(self, path: Path):
        """Serialize compiled universe to MessagePack."""

        # Convert to dict (Pydantic models → dict)
        data = {
            "bars": self.bars.model_dump(),
            "cascades": self.cascades.model_dump(),
            "affordances": self.affordances.model_dump(),
            "actions": self.actions.model_dump(),
            "substrate": self.substrate.model_dump(),
            "training": self.training.model_dump(),
            "metadata": dataclasses.asdict(self.metadata),
            "optimization_data": {
                "base_depletions": self.optimization_data.base_depletions.cpu().numpy().tolist(),
                "cascade_data": self.optimization_data.cascade_data,
                "modulation_data": self.optimization_data.modulation_data,
                "action_mask_table": self.optimization_data.action_mask_table.cpu().numpy().tolist(),
            }
        }

        # Serialize to MessagePack
        packed = msgpack.packb(data, use_bin_type=True)

        with open(path, "wb") as f:
            f.write(packed)

    @classmethod
    def load_from_cache(cls, path: Path) -> "CompiledUniverse":
        """Deserialize compiled universe from MessagePack."""

        with open(path, "rb") as f:
            packed = f.read()

        data = msgpack.unpackb(packed, raw=False)

        # Reconstruct Pydantic models
        bars = BarsConfig(**data["bars"])
        cascades = CascadesConfig(**data["cascades"])
        affordances = AffordanceConfigCollection(**data["affordances"])
        ...

        # Reconstruct tensors
        optimization_data = OptimizationData(
            base_depletions=torch.tensor(data["optimization_data"]["base_depletions"]),
            cascade_data=data["optimization_data"]["cascade_data"],
            ...
        )

        return cls(
            bars=bars,
            cascades=cascades,
            affordances=affordances,
            actions=actions,
            substrate=substrate,
            training=training,
            metadata=UniverseMetadata(**data["metadata"]),
            optimization_data=optimization_data,
        )
```

---

### Cache Invalidation

**Problem**: How to know when cache is stale?

**Strategy**: **mtime-based invalidation**

```python
class UniverseCompiler:
    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile universe with mtime-based cache invalidation."""

        cache_path = config_dir / ".compiled" / "universe.msgpack"

        if use_cache and cache_path.exists():
            # Get cache modification time
            cache_mtime = cache_path.stat().st_mtime

            # Get all YAML modification times
            yaml_files = list(config_dir.glob("*.yaml"))
            yaml_mtimes = [f.stat().st_mtime for f in yaml_files]

            # Cache is fresh if it's newer than ALL source files
            if all(yaml_mtime < cache_mtime for yaml_mtime in yaml_mtimes):
                logger.info(f"Loading cached universe from {cache_path}")
                return CompiledUniverse.load_from_cache(cache_path)
            else:
                logger.info(f"Cache stale (source YAML modified), recompiling...")

        # Cache miss or stale, do full compilation
        logger.info(f"Compiling universe from {config_dir}")
        universe = self._compile_from_source(config_dir)

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        universe.save_to_cache(cache_path)
        logger.info(f"Saved compiled universe to {cache_path}")

        return universe
```

**Benefits**:

- ✅ Automatic invalidation (no manual cache clearing)
- ✅ Fast (just stat calls, no file reading)
- ✅ Reliable (any YAML change triggers recompilation)

---

### Cache Performance Targets

**Benchmarks**:

| Operation | Without Cache | With Cache | Speedup |
|-----------|---------------|------------|---------|
| **Load YAML** | 10-20ms | 0ms (skipped) | ∞ |
| **Parse YAML** | 10-30ms | 0ms (skipped) | ∞ |
| **Validate Pydantic** | 5-10ms | 0ms (skipped) | ∞ |
| **Build Symbol Table** | 1-2ms | 0ms (skipped) | ∞ |
| **Resolve References** | 2-5ms | 0ms (skipped) | ∞ |
| **Cross-Validate** | 5-10ms | 0ms (skipped) | ∞ |
| **Compute Metadata** | 1-2ms | 0ms (skipped) | ∞ |
| **Optimize (Tensors)** | 5-10ms | 0ms (skipped) | ∞ |
| **Deserialize Cache** | - | 1-5ms | - |
| **Total** | **50-100ms** | **1-5ms** | **10-100x** |

**Goal**: Cached load < 5ms (target: 1-2ms)

---

## Part 6: Error Reporting Best Practices

### Error Message Quality

**Bad Error**:

```
ValidationError: 1 validation error for BarsConfig
bars
  Value error, Expected 8 bars, got 4 [type=value_error, ...]
```

**Good Error**:

```
Universe Compilation Failed (Stage 1: Parse Individual Files)

bars.yaml validation failed:
  Expected exactly 8 bars, got 4.

  Found bars: [energy, health, money, mood]
  Missing bars: [hygiene, satiation, social, fitness]

  Hint: If you're creating a simplified universe, this is a known limitation.
  See docs/TASK-005-VARIABLE-SIZE-METER-SYSTEM.md for variable-size meter support.
```

**Principles**:

1. **Context**: What stage failed? What file?
2. **Specificity**: What was expected vs what was found?
3. **Actionability**: How to fix it?
4. **Hints**: Common causes, related documentation

---

### Error Collection Example

```python
class CompilationError(Exception):
    """Raised when universe compilation fails."""

    def __init__(self, stage: str, errors: list[str], hints: list[str] | None = None):
        self.stage = stage
        self.errors = errors
        self.hints = hints or []

        # Build comprehensive error message
        message_parts = [
            f"Universe Compilation Failed ({stage})",
            "",
            f"Found {len(errors)} error(s):",
            "",
        ]

        for i, error in enumerate(errors, 1):
            message_parts.append(f"  {i}. {error}")

        if hints:
            message_parts.append("")
            message_parts.append("Hints:")
            for hint in hints:
                message_parts.append(f"  - {hint}")

        super().__init__("\n".join(message_parts))

# Usage
errors = CompilationErrorCollector()

for cascade in cascades.cascades:
    if cascade.source not in symbol_table.meters:
        errors.add_error(
            f"cascades.yaml:{cascade.name}: "
            f"References non-existent meter '{cascade.source}'. "
            f"Valid meters: {list(symbol_table.meters.keys())}"
        )

if errors.has_errors():
    raise CompilationError(
        stage="Stage 3: Reference Resolution",
        errors=errors.errors,
        hints=[
            "Check for typos in meter names (case-sensitive)",
            "Ensure bars.yaml defines all referenced meters",
            "See docs/UNIVERSE_AS_CODE.md for meter naming conventions"
        ]
    )
```

---

## Part 7: Testing Strategy

### Test Pyramid for Compiler

```
                    ▲
                  /   \
                /  E2E  \
              /  (Integration)  \
            /                   \
          /     Unit Tests       \
        /  (Stage, Validation)    \
      /                             \
    /      Fixture-Based Tests       \
  /  (Valid/Invalid Config Packs)    \
 /___________________________________\
```

**Layer 1: Fixture-Based Tests** (Foundation)

Create test config packs:

- `fixtures/valid_universe/` - Should compile successfully
- `fixtures/invalid_dangling_ref/` - Should fail at Stage 3 (reference resolution)
- `fixtures/invalid_spatial/` - Should fail at Stage 4 (spatial impossibility)
- `fixtures/invalid_circular/` - Should fail at Stage 4 (cascade circularity)

**Layer 2: Unit Tests** (Stage Isolation)

Test each compilation stage independently:

```python
def test_stage_1_parse_valid_bars():
    """Stage 1 should parse valid bars.yaml."""
    compiler = UniverseCompiler()
    bars = compiler.stage_1_parse_bars(Path("fixtures/valid_universe/bars.yaml"))
    assert len(bars.bars) == 8

def test_stage_2_build_symbol_table():
    """Stage 2 should register all meters."""
    symbol_table = UniverseSymbolTable()
    bars = load_bars_config(Path("fixtures/valid_universe/bars.yaml"))

    for bar in bars.bars:
        symbol_table.register_meter(bar.name, bar)

    assert "energy" in symbol_table.meters
    assert symbol_table.meters["energy"].index == 0

def test_stage_3_detect_dangling_reference():
    """Stage 3 should catch dangling meter references."""
    compiler = UniverseCompiler()

    with pytest.raises(CompilationError, match="non-existent meter"):
        compiler.compile(Path("fixtures/invalid_dangling_ref"))
```

**Layer 3: Integration Tests** (End-to-End)

Test full compilation pipeline:

```python
def test_compile_valid_universe():
    """Should compile valid universe successfully."""
    compiler = UniverseCompiler()
    universe = compiler.compile(Path("fixtures/valid_universe"))

    assert universe.metadata.meter_count == 8
    assert universe.metadata.observation_dim > 0
    assert universe.optimization_data.base_depletions.shape == (8,)

def test_compile_with_cache():
    """Should use cache on second compile."""
    compiler = UniverseCompiler()

    # First compile (no cache)
    start = time.time()
    universe1 = compiler.compile(Path("fixtures/valid_universe"))
    first_compile_time = time.time() - start

    # Second compile (with cache)
    start = time.time()
    universe2 = compiler.compile(Path("fixtures/valid_universe"))
    cached_compile_time = time.time() - start

    assert cached_compile_time < first_compile_time / 5  # At least 5x faster
    assert universe1.metadata == universe2.metadata

def test_compiled_universe_immutable():
    """Compiled universe should be immutable."""
    compiler = UniverseCompiler()
    universe = compiler.compile(Path("fixtures/valid_universe"))

    with pytest.raises(AttributeError, match="frozen"):
        universe.bars.bars[0].initial = 0.5
```

---

## Part 8: Refactoring Risks and Mitigations

### Risk 1: Breaking Existing Code

**Risk**: Refactoring `VectorizedHamletEnv` breaks existing training scripts.

**Mitigation**: **Dual API (temporary)**

```python
class VectorizedHamletEnv:
    def __init__(
        self,
        # NEW API: Accept compiled universe
        universe: CompiledUniverse | None = None,

        # OLD API: Accept config pack path (deprecated)
        config_pack_path: Path | None = None,
        num_agents: int = 1,
        device: str = "cuda",
        ...
    ):
        if universe is not None:
            # New API: Use compiled universe
            self._init_from_universe(universe, num_agents, device)
        elif config_pack_path is not None:
            # Old API: Compile on-the-fly (with deprecation warning)
            warnings.warn(
                "Passing config_pack_path is deprecated. "
                "Use UniverseCompiler.compile() and pass universe instead.",
                DeprecationWarning
            )
            compiler = UniverseCompiler()
            universe = compiler.compile(config_pack_path)
            self._init_from_universe(universe, num_agents, device)
        else:
            raise ValueError("Must provide either 'universe' or 'config_pack_path'")
```

**Timeline**:

- v1.0: Introduce new API, keep old API with deprecation warning
- v1.5: Log error if old API used
- v2.0: Remove old API completely

---

### Risk 2: Cache Corruption

**Risk**: Corrupted cache file causes crashes.

**Mitigation**: **Graceful degradation**

```python
def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
    """Compile with graceful cache degradation."""

    cache_path = config_dir / ".compiled" / "universe.msgpack"

    if use_cache and cache_path.exists():
        try:
            # Try to load from cache
            universe = CompiledUniverse.load_from_cache(cache_path)
            logger.info(f"Loaded from cache: {cache_path}")
            return universe

        except Exception as e:
            # Cache corrupted, fall back to full compilation
            logger.warning(
                f"Cache load failed ({e}), recompiling from source. "
                f"Cache will be regenerated."
            )

    # Full compilation (cache miss or load failed)
    universe = self._compile_from_source(config_dir)

    try:
        # Try to save cache (might fail due to permissions, disk full, etc.)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        universe.save_to_cache(cache_path)
        logger.info(f"Saved to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Cache save failed ({e}), continuing without cache")

    return universe
```

---

### Risk 3: Performance Regression

**Risk**: Compilation overhead slows down training startup.

**Mitigation**: **Benchmarking + caching**

```python
# Add benchmarking to compiler
class UniverseCompiler:
    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile with performance tracking."""

        timings = {}

        start = time.time()

        # Stage 1
        stage_start = time.time()
        raw_configs = self.stage_1_parse_individual_files(config_dir)
        timings["stage_1_parse"] = time.time() - stage_start

        # Stage 2
        stage_start = time.time()
        symbol_table = self.stage_2_build_symbol_tables(raw_configs)
        timings["stage_2_symbols"] = time.time() - stage_start

        # ... (same for all stages)

        timings["total"] = time.time() - start

        logger.info(f"Compilation timings: {timings}")

        return universe

# Benchmark test
def test_compilation_performance():
    """Compilation should be < 100ms for typical universe."""
    compiler = UniverseCompiler()

    start = time.time()
    universe = compiler.compile(Path("configs/L1_full_observability"), use_cache=False)
    compile_time = time.time() - start

    assert compile_time < 0.1, f"Compilation took {compile_time:.3f}s (expected <0.1s)"
```

**Performance Budget**:

- Stage 1 (Parse): < 30ms
- Stage 2 (Symbols): < 5ms
- Stage 3 (Resolve): < 10ms
- Stage 4 (Validate): < 20ms
- Stage 5 (Metadata): < 5ms
- Stage 6 (Optimize): < 20ms
- Stage 7 (Emit): < 5ms
- **Total: < 100ms**

---

## Part 9: Implementation Priorities

### HIGH PRIORITY (Must Have)

**1. Multi-Pass Compiler** (11-16 hours)

- Implement 7-stage compilation pipeline
- Stage 1: Parse individual files
- Stage 2: Build symbol tables
- Stage 3: Resolve references
- Stage 4: Cross-validate
- Stage 5: Compute metadata
- Stage 6: Optimize (pre-compute)
- Stage 7: Emit CompiledUniverse

**2. Symbol Table** (4-6 hours)

- Implement UniverseSymbolTable
- Register meters, affordances, actions
- Resolve references with clear error messages

**3. Error Collection** (4-6 hours)

- Implement CompilationErrorCollector
- Collect all errors before failing
- Provide actionable error messages with hints

**4. CompiledUniverse Artifact** (3-4 hours)

- Immutable dataclass with frozen=True
- Contains all configs + metadata + optimization data
- Validates consistency at construction

**Total High Priority**: 22-32 hours (3-4 days)

---

### MEDIUM PRIORITY (Should Have)

**5. Compilation Cache** (4-6 hours)

- MessagePack serialization
- mtime-based invalidation
- Graceful degradation on cache corruption

**6. Cross-Validation** (4-6 hours)

- Spatial feasibility check
- Economic balance check
- Cascade circularity detection
- Temporal conflict detection
- Action-substrate compatibility

**7. Metadata Computation** (3-4 hours)

- Dynamic observation_dim calculation
- Dynamic action_dim calculation
- Economic metadata
- Version tracking

**Total Medium Priority**: 11-16 hours (1-2 days)

---

### LOW PRIORITY (Nice to Have)

**8. CLI Tool** (2-3 hours)

- `python -m townlet.compiler compile configs/L1`
- `python -m townlet.compiler inspect .compiled/universe.msgpack`
- `python -m townlet.compiler validate configs/L1`

**9. Optimization Pre-Compute** (3-4 hours)

- Action mask table [24, num_affordances]
- Cascade pre-sorting by target index
- Base depletion tensor

**10. Performance Benchmarking** (2-3 hours)

- Per-stage timing
- Performance budget enforcement
- Cache speedup measurement

**Total Low Priority**: 7-10 hours (1 day)

---

## Conclusion

**Universe compiler is the "bones" of UAC** - it must be robust before we worry about reward models or economic tuning.

**Key Architectural Decisions**:

1. ✅ **Multi-pass compilation** (7 stages: parse → resolve → validate → optimize → emit)
2. ✅ **Symbol tables for reference resolution** (meters, affordances, actions)
3. ✅ **Error collection, not fail-fast** (show all errors at once)
4. ✅ **Eager compilation with caching** (compile once, cache for subsequent runs)
5. ✅ **Immutable CompiledUniverse** (frozen dataclass, safe to share)
6. ✅ **Comprehensive metadata** (observation_dim, action_dim, economic_balance)
7. ✅ **Graceful degradation** (cache corruption → recompile, not crash)

**Critical Refactoring**:

- Move config loading from `VectorizedHamletEnv` to `UniverseCompiler`
- Replace hardcoded observation_dim with dynamic calculation
- Replace hardcoded action_dim with metadata lookup
- Implement cross-file validation (dangling references, spatial feasibility)

**Estimated Total Effort**: 26-38 hours (3-5 days)

**Priority Order**:

1. HIGH: Multi-pass compiler (11-16h)
2. HIGH: Symbol table (4-6h)
3. HIGH: Error collection (4-6h)
4. HIGH: CompiledUniverse (3-4h)
5. MEDIUM: Compilation cache (4-6h)
6. MEDIUM: Cross-validation (4-6h)
7. MEDIUM: Metadata computation (3-4h)
8. LOW: CLI tool (2-3h)
9. LOW: Optimization (3-4h)
10. LOW: Benchmarking (2-3h)

**Slogan**: "Compile universes with best practices - fail fast, cache aggressively, error beautifully."
