# TASK-006: Universe Compiler Implementation

**Status**: Planned
**Priority**: HIGH (Foundational for UAC system integrity)
**Estimated Effort**: 46-66 hours (6-8 days) - UPDATED from 40-58h (+6-8h for capability system validation)
**Dependencies**: TASK-005 (Variable-Size Meter System - recommended but not required)
**Enables**: All future UAC work (robust compilation is foundation)

---

## Problem Statement

### Current Architecture (Problems)

**Ad-Hoc Config Loading**:
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
```

**Critical Problems**:

1. **No Central Orchestration**
   - Each component loads its own configs independently
   - No validation that configs are compatible with each other
   - Can't cache compiled universe (loading repeated every run)

2. **Hidden Dependencies**
   - `MeterDynamics` secretly loads bars + cascades internally
   - `AffordanceEngine` validates meter names against hardcoded dict
   - Observation builder doesn't know about actual meter count

3. **Hardcoded Constants**
   - Meter count: 8 (hardcoded in multiple places)
   - Action count: 6 (hardcoded)
   - Can't detect these values from config

4. **No Cross-Validation**
   - Cascades can reference non-existent meters (discovered during training)
   - Too many affordances for grid size → silent failure or runtime error
   - Economic imbalance (costs > income) → discovered after training starts
   - Circular cascade dependencies → infinite loops

5. **Tight Coupling**
   - `VectorizedHamletEnv` does config loading AND execution
   - Can't unit test compilation without full environment
   - Can't pre-compile configs offline
   - Can't share compiled universes between runs

6. **Late Error Discovery**
   - Config errors discovered during training (10 minutes in)
   - Dangling meter references crash during cascade application
   - Spatial impossibility crashes during reset

7. **No Reproducibility Guarantees**
   - Mutable configs can be accidentally modified
   - No version tracking
   - No metadata about when/how universe was compiled

**From Research**: This is **foundational infrastructure debt**. We need a proper compiler before building more UAC features.

---

## Solution Overview

### Design Principle

**"Compile Once, Execute Many Times"**

Build a **multi-pass compiler** that:
1. Loads all configs
2. Validates individual files
3. Resolves cross-file references
4. Validates cross-file constraints
5. Computes metadata (obs_dim, action_dim, etc.)
6. Pre-computes optimization data
7. Emits immutable `CompiledUniverse` artifact

### Architecture Changes

**Before** (current):
```
VectorizedHamletEnv.__init__()
    ↓
Load configs independently
    ↓
Hardcoded obs_dim/action_dim
    ↓
Start training (discover errors during execution)
```

**After** (target):
```
UniverseCompiler.compile()
    ↓
7-stage compilation pipeline
    ↓
CompiledUniverse (immutable, cached)
    ↓
VectorizedHamletEnv(universe=compiled)
    ↓
Start training (all errors caught upfront)
```

### Key Components

1. **UniverseCompiler**: 7-stage compilation pipeline
2. **UniverseSymbolTable**: Central registry for cross-file references
3. **CompilationErrorCollector**: Collect all errors before failing
4. **CompiledUniverse**: Immutable artifact with metadata + optimization data
5. **Compilation Cache**: MessagePack serialization for fast subsequent loads

---

## Implementation Plan

### Phase 1: Core Compiler Infrastructure (11-16 hours)

**Goal**: Implement 7-stage compilation pipeline with basic validation.

#### 1.1: Create UniverseCompiler Skeleton

**File**: `src/townlet/universe/compiler.py` (NEW)

```python
from dataclasses import dataclass
from pathlib import Path

from townlet.environment.cascade_config import load_bars_config, load_cascades_config
from townlet.environment.affordance_config import load_affordance_config


class UniverseCompiler:
    """Multi-stage universe compiler with cross-validation."""

    def compile(self, config_dir: Path, use_cache: bool = True) -> "CompiledUniverse":
        """
        Compile universe from YAML configs.

        Args:
            config_dir: Directory containing YAML config files
            use_cache: If True, use cached compilation if available

        Returns:
            Immutable CompiledUniverse ready for training

        Raises:
            CompilationError: If universe is invalid
        """
        # Stage 1: Parse individual files
        raw_configs = self._stage_1_parse_individual_files(config_dir)

        # Stage 2: Build symbol tables
        symbol_table = self._stage_2_build_symbol_tables(raw_configs)

        # Stage 3: Resolve references
        errors = CompilationErrorCollector()
        self._stage_3_resolve_references(raw_configs, symbol_table, errors)
        errors.check_and_raise("Stage 3: Reference Resolution")

        # Stage 4: Cross-validate
        self._stage_4_cross_validate(raw_configs, symbol_table, errors)
        errors.check_and_raise("Stage 4: Cross-Validation")

        # Stage 5: Compute metadata
        metadata = self._stage_5_compute_metadata(raw_configs, symbol_table)

        # Stage 6: Optimize (pre-compute)
        optimization_data = self._stage_6_optimize(raw_configs, metadata)

        # Stage 7: Emit compiled universe
        universe = self._stage_7_emit_compiled_universe(
            raw_configs, metadata, optimization_data
        )

        return universe
```

#### 1.2: Implement Stage 1 (Parse Individual Files)

```python
@dataclass
class RawConfigs:
    """Container for raw config objects loaded from YAML."""
    bars: BarsConfig
    cascades: CascadesConfig
    affordances: AffordanceConfigCollection
    training: TrainingConfig
    # Future: substrate, actions when TASK-000/002 implemented


def _stage_1_parse_individual_files(self, config_dir: Path) -> RawConfigs:
    """
    Stage 1: Load and validate individual YAML files.

    Validates:
    - File exists
    - YAML is well-formed
    - Pydantic schema is valid

    Does NOT validate cross-file references (Stage 3).
    """
    try:
        bars = load_bars_config(config_dir / "bars.yaml")
    except FileNotFoundError:
        raise CompilationError(
            stage="Stage 1: Parse",
            errors=[f"bars.yaml not found in {config_dir}"],
            hints=["Ensure config pack contains bars.yaml"]
        )
    except yaml.YAMLError as e:
        raise CompilationError(
            stage="Stage 1: Parse",
            errors=[f"bars.yaml is malformed: {e}"],
            hints=["Check YAML syntax with a validator"]
        )
    except ValidationError as e:
        raise CompilationError(
            stage="Stage 1: Parse",
            errors=[f"bars.yaml validation failed: {e}"],
            hints=["Check field types and constraints"]
        )

    # Same for cascades, affordances, training
    cascades = load_cascades_config(config_dir / "cascades.yaml")
    affordances = load_affordance_config(config_dir / "affordances.yaml")
    training = load_training_config(config_dir / "training.yaml")

    return RawConfigs(
        bars=bars,
        cascades=cascades,
        affordances=affordances,
        training=training,
    )
```

**Success Criteria**:
- [ ] Compiler loads all YAML files
- [ ] File not found raises clear error
- [ ] Malformed YAML raises parse error
- [ ] Invalid schema raises validation error

---

### Phase 2: Symbol Table & Reference Resolution (4-6 hours)

**Goal**: Implement cross-file reference resolution with clear error messages.

#### 2.1: Create UniverseSymbolTable

**File**: `src/townlet/universe/symbol_table.py` (NEW)

```python
class UniverseSymbolTable:
    """Central registry of all named entities in universe."""

    def __init__(self):
        self.meters: dict[str, BarConfig] = {}
        self.affordances: dict[str, AffordanceConfig] = {}
        # Future: actions, stages when TASK-003 (Action Space) implemented

    def register_meter(self, name: str, config: BarConfig):
        """Register meter for later reference resolution."""
        if name in self.meters:
            raise CompilationError(
                stage="Stage 2: Symbol Registration",
                errors=[f"Duplicate meter name: '{name}'"],
                hints=["Each meter must have unique name"]
            )
        self.meters[name] = config

    def register_affordance(self, id: str, config: AffordanceConfig):
        """Register affordance for later reference resolution."""
        if id in self.affordances:
            raise CompilationError(
                stage="Stage 2: Symbol Registration",
                errors=[f"Duplicate affordance ID: '{id}'"],
                hints=["Each affordance must have unique ID"]
            )
        self.affordances[id] = config

    def resolve_meter_reference(self, name: str, location: str) -> BarConfig:
        """
        Resolve meter reference, raising clear error on failure.

        Args:
            name: Meter name to resolve
            location: Where reference appears (for error messages)

        Returns:
            Resolved BarConfig

        Raises:
            ReferenceError: If meter doesn't exist
        """
        if name not in self.meters:
            raise ReferenceError(
                f"{location}: References non-existent meter '{name}'. "
                f"Valid meters: {list(self.meters.keys())}"
            )
        return self.meters[name]

    @property
    def meter_names(self) -> list[str]:
        """List of all registered meter names."""
        return list(self.meters.keys())

    @property
    def affordance_ids(self) -> list[str]:
        """List of all registered affordance IDs."""
        return list(self.affordances.keys())
```

#### 2.2: Implement Stage 2 (Build Symbol Tables)

```python
def _stage_2_build_symbol_tables(self, raw_configs: RawConfigs) -> UniverseSymbolTable:
    """
    Stage 2: Build symbol tables from raw configs.

    Registers all named entities for reference resolution.
    """
    symbol_table = UniverseSymbolTable()

    # Register meters
    for bar in raw_configs.bars.bars:
        symbol_table.register_meter(bar.name, bar)

    # Register affordances
    for aff in raw_configs.affordances.affordances:
        symbol_table.register_affordance(aff.id, aff)

    return symbol_table
```

#### 2.3: Implement Stage 3 (Resolve References)

```python
def _stage_3_resolve_references(
    self,
    raw_configs: RawConfigs,
    symbol_table: UniverseSymbolTable,
    errors: CompilationErrorCollector
):
    """
    Stage 3: Resolve and validate all cross-file references.

    Validates:
    - Cascade meter references exist in bars.yaml
    - Affordance meter references exist in bars.yaml
    - Training enabled_affordances exist in affordances.yaml
    """
    # Resolve cascade meter references
    for cascade in raw_configs.cascades.cascades:
        try:
            symbol_table.resolve_meter_reference(
                cascade.source,
                location=f"cascades.yaml:{cascade.name}"
            )
        except ReferenceError as e:
            errors.add_error(str(e))

        try:
            symbol_table.resolve_meter_reference(
                cascade.target,
                location=f"cascades.yaml:{cascade.name}"
            )
        except ReferenceError as e:
            errors.add_error(str(e))

    # Resolve affordance meter references
    for aff in raw_configs.affordances.affordances:
        # Check costs
        for cost in aff.costs + aff.costs_per_tick:
            try:
                symbol_table.resolve_meter_reference(
                    cost.meter,
                    location=f"affordances.yaml:{aff.id}:cost"
                )
            except ReferenceError as e:
                errors.add_error(str(e))

        # Check effects
        for effect in aff.effects + aff.effects_per_tick + aff.completion_bonus:
            try:
                symbol_table.resolve_meter_reference(
                    effect.meter,
                    location=f"affordances.yaml:{aff.id}:effect"
                )
            except ReferenceError as e:
                errors.add_error(str(e))

    # Resolve training enabled_affordances
    for aff_id in raw_configs.training.enabled_affordances:
        if aff_id not in symbol_table.affordances:
            errors.add_error(
                f"training.yaml:enabled_affordances: "
                f"References non-existent affordance '{aff_id}'. "
                f"Valid affordances: {symbol_table.affordance_ids}"
            )

    # Resolve action cost meter references (NEW - from research Gap 2)
    for action in raw_configs.actions.actions:
        for cost in action.costs:
            try:
                symbol_table.resolve_meter_reference(
                    cost.meter,
                    location=f"actions.yaml:{action.name}:cost"
                )
            except ReferenceError as e:
                errors.add_error(str(e))

    # Resolve capability meter references (NEW - from research Finding 1)
    for aff in raw_configs.affordances.affordances:
        if aff.capabilities:
            for capability in aff.capabilities:
                if capability.type == "meter_gated":
                    try:
                        symbol_table.resolve_meter_reference(
                            capability.meter,
                            location=f"affordances.yaml:{aff.id}:meter_gated capability"
                        )
                    except ReferenceError as e:
                        errors.add_error(str(e))

        # Resolve effect pipeline meter references
        if aff.effect_pipeline:
            for stage_name, effects in [
                ("on_start", aff.effect_pipeline.on_start),
                ("per_tick", aff.effect_pipeline.per_tick),
                ("on_completion", aff.effect_pipeline.on_completion),
                ("on_early_exit", aff.effect_pipeline.on_early_exit),
                ("on_failure", aff.effect_pipeline.on_failure),
            ]:
                for effect in effects:
                    try:
                        symbol_table.resolve_meter_reference(
                            effect.meter,
                            location=f"affordances.yaml:{aff.id}:effect_pipeline.{stage_name}"
                        )
                    except ReferenceError as e:
                        errors.add_error(str(e))

        # Resolve availability constraint meter references (NEW - from research Gap 1)
        if aff.availability:
            for constraint in aff.availability:
                try:
                    symbol_table.resolve_meter_reference(
                        constraint.meter,
                        location=f"affordances.yaml:{aff.id}:availability"
                    )
                except ReferenceError as e:
                    errors.add_error(str(e))
```

**Success Criteria**:
- [ ] Symbol table registers all meters and affordances
- [ ] Duplicate names detected and reported
- [ ] Dangling references caught with clear error messages
- [ ] Error messages list valid alternatives

---

### Phase 3: Error Collection Infrastructure (4-6 hours)

**Goal**: Collect ALL validation errors before failing (better UX).

#### 3.1: Create CompilationErrorCollector

**File**: `src/townlet/universe/errors.py` (NEW)

```python
class CompilationError(Exception):
    """Raised when universe compilation fails."""

    def __init__(
        self,
        stage: str,
        errors: list[str],
        hints: list[str] | None = None
    ):
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


class CompilationErrorCollector:
    """Collect multiple validation errors before failing."""

    def __init__(self):
        self.errors: list[str] = []

    def add_error(self, error: str):
        """Record validation error."""
        self.errors.append(error)

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0

    def check_and_raise(self, stage: str, hints: list[str] | None = None):
        """
        If errors exist, raise CompilationError with all messages.

        Args:
            stage: Compilation stage name (for error message)
            hints: Optional list of hints for fixing errors
        """
        if self.errors:
            raise CompilationError(stage=stage, errors=self.errors, hints=hints)

    def clear(self):
        """Clear all collected errors (for reuse)."""
        self.errors.clear()
```

**Success Criteria**:
- [ ] Error collector accumulates multiple errors
- [ ] check_and_raise formats errors clearly
- [ ] Error messages include stage, error list, and hints

---

### Phase 4: Cross-Validation (10-14 hours) ← UPDATED: was 4-6h

**⚠️ UPDATED AFTER RESEARCH**: Added capability system validation (+6-8h)

**Goal**: Validate constraints that span multiple config files.

#### 4.1: Implement Stage 4 (Cross-Validate)

```python
def _stage_4_cross_validate(
    self,
    raw_configs: RawConfigs,
    symbol_table: UniverseSymbolTable,
    errors: CompilationErrorCollector
):
    """
    Stage 4: Validate cross-file constraints.

    Validates:
    - Spatial feasibility (grid has enough cells)
    - Economic balance (income >= costs)
    - Cascade circularity (no infinite loops)
    - Temporal conflicts (valid operating hours)
    - **Substrate-action compatibility** (NEW - from research)
    """
    # 1. Spatial feasibility
    grid_size = raw_configs.training.grid_size
    grid_cells = grid_size * grid_size
    enabled_affordances = raw_configs.training.enabled_affordances
    required_cells = len(enabled_affordances) + 1  # +1 for agent

    if required_cells > grid_cells:
        errors.add_error(
            f"Spatial impossibility: Grid has {grid_cells} cells "
            f"({grid_size}×{grid_size}) but need {required_cells} "
            f"({len(enabled_affordances)} affordances + 1 agent)"
        )

    # 2. Economic balance (warning, not error)
    total_income = self._compute_max_income(raw_configs.affordances)
    total_costs = self._compute_total_costs(raw_configs.affordances)

    if total_income < total_costs:
        errors.add_error(
            f"Economic imbalance (WARNING): Total income ({total_income:.2f}) < "
            f"total costs ({total_costs:.2f}). Universe may be poverty trap."
        )

    # 3. Cascade circularity
    cascade_graph = self._build_cascade_graph(raw_configs.cascades)
    cycles = self._detect_cycles(cascade_graph)

    if cycles:
        for cycle in cycles:
            cycle_str = " → ".join(cycle + [cycle[0]])
            errors.add_error(
                f"Cascade circularity detected: {cycle_str}. "
                f"This will cause infinite cascade loops."
            )

    # 4. Temporal conflicts (operating hours)
    for aff in raw_configs.affordances.affordances:
        # Validate operating hours in modes
        if aff.modes:
            for mode_name, mode_config in aff.modes.items():
                if mode_config.hours:
                    open_hour, close_hour = mode_config.hours
                    if open_hour < 0 or open_hour > 23:
                        errors.add_error(
                            f"affordances.yaml:{aff.id}:modes:{mode_name}: "
                            f"open_hour must be 0-23, got {open_hour}"
                        )
                    if close_hour < 1 or close_hour > 28:
                        errors.add_error(
                            f"affordances.yaml:{aff.id}:modes:{mode_name}: "
                            f"close_hour must be 1-28, got {close_hour}"
                        )

        # Validate availability constraints (NEW - from research Gap 1)
        if aff.availability:
            for constraint in aff.availability:
                if constraint.min is not None and constraint.max is not None:
                    if constraint.min >= constraint.max:
                        errors.add_error(
                            f"affordances.yaml:{aff.id}:availability: "
                            f"min ({constraint.min}) must be < max ({constraint.max})"
                        )

    # 5. Capability conflicts (NEW - from research Finding 1)
    for aff in raw_configs.affordances.affordances:
        if not aff.capabilities:
            continue

        capability_types = {cap.type for cap in aff.capabilities}

        # Mutually exclusive capabilities
        if "instant" in capability_types and "multi_tick" in capability_types:
            errors.add_error(
                f"affordances.yaml:{aff.id}: "
                f"Cannot have both 'instant' and 'multi_tick' capabilities (mutually exclusive)"
            )

        # Dependent capabilities
        for cap in aff.capabilities:
            if cap.type == "multi_tick" and cap.resumable:
                # Check that multi_tick capability exists
                if "multi_tick" not in capability_types:
                    errors.add_error(
                        f"affordances.yaml:{aff.id}: "
                        f"'resumable' flag requires 'multi_tick' capability"
                    )

        # Effect pipeline consistency (NEW - from research Finding 1)
        if aff.effect_pipeline:
            has_multi_tick = "multi_tick" in capability_types
            has_multi_tick_effects = (
                bool(aff.effect_pipeline.per_tick) or
                bool(aff.effect_pipeline.on_completion)
            )

            if has_multi_tick and not has_multi_tick_effects:
                errors.add_error(
                    f"affordances.yaml:{aff.id}: "
                    f"'multi_tick' capability requires per_tick or on_completion effects"
                )

            # Warn if using early_exit effects without early_exit_allowed
            if aff.effect_pipeline.on_early_exit:
                has_early_exit_allowed = any(
                    cap.type == "multi_tick" and cap.early_exit_allowed
                    for cap in aff.capabilities
                )
                if not has_early_exit_allowed:
                    errors.add_warning(
                        f"affordances.yaml:{aff.id}: "
                        f"on_early_exit effects defined but early_exit_allowed=False "
                        f"(effects will never trigger)"
                    )

    # 5. Substrate-Action Compatibility (NEW - from research)
    # See: docs/research/RESEARCH-ACTION-COMPATIBILITY-VALIDATION.md
    from townlet.environment.substrate_action_validator import SubstrateActionValidator

    validator = SubstrateActionValidator(raw_configs.substrate, raw_configs.actions)
    result = validator.validate()

    if not result.valid:
        errors.add_error(
            f"Substrate-Action Compatibility Failed:\n" +
            f"  Substrate: {raw_configs.substrate.type} " +
            f"({raw_configs.substrate.grid.topology if raw_configs.substrate.type == 'grid' else 'N/A'})\n" +
            "\n".join(f"  • {err}" for err in result.errors)
        )

    # Add warnings (non-fatal)
    for warning in result.warnings:
        errors.add_warning(f"Substrate-Action Warning: {warning}")

    # 6. Affordance Position Bounds Validation (NEW - from research)
    # See: docs/research/RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md
    for aff in raw_configs.affordances.affordances:
        if aff.position is not None:
            # Validate position is within substrate bounds
            if not raw_configs.substrate.is_position_in_bounds(aff.position):
                errors.add_error(
                    f"affordances.yaml:{aff.id}: Position out of bounds: {aff.position}\n" +
                    f"  Substrate bounds: {raw_configs.substrate.get_bounds_description()}\n" +
                    f"  Valid positions must be within substrate dimensions."
                )


def _build_cascade_graph(self, cascades: CascadesConfig) -> dict[str, list[str]]:
    """Build directed graph of cascade dependencies."""
    graph = {}
    for cascade in cascades.cascades:
        if cascade.source not in graph:
            graph[cascade.source] = []
        graph[cascade.source].append(cascade.target)
    return graph


def _detect_cycles(self, graph: dict[str, list[str]]) -> list[list[str]]:
    """Detect cycles in cascade dependency graph using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()

    def dfs(node: str, path: list[str]):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path.copy())
            elif neighbor in rec_stack:
                # Cycle detected
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                cycles.append(cycle)

        rec_stack.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node, [])

    return cycles
```

**Success Criteria**:
- [ ] Spatial validation catches over-crowded grids
- [ ] Economic validation warns on poverty traps
- [ ] Circularity detection catches cascade cycles
- [ ] Temporal validation catches invalid operating hours
- [ ] **Substrate-action validation catches incompatible action spaces** (NEW)
  - [ ] Square grid requires 4-way movement
  - [ ] Cubic grid requires 6-way movement
  - [ ] Hexagonal grid requires 6 hex directions
  - [ ] Aspatial forbids movement actions
  - [ ] Missing INTERACT action detected
- [ ] **Affordance position bounds validation** (NEW)
  - [ ] Catches affordances positioned outside substrate bounds
  - [ ] Works for 2D grid: position [x, y] within [0, width) × [0, height)
  - [ ] Works for 3D grid: position [x, y, z] within bounds
  - [ ] Works for hex grid: position {q, r} within axial bounds
  - [ ] Works for graph: node_id < num_nodes

---

### Phase 5: Metadata Computation (3-4 hours)

**Goal**: Calculate derived properties (obs_dim, action_dim, etc.)

#### 5.1: Create UniverseMetadata

**File**: `src/townlet/universe/metadata.py` (NEW)

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseMetadata:
    """Derived metadata about compiled universe."""

    # Meter metadata
    meter_count: int
    meter_names: list[str]  # Sorted by index
    meter_name_to_index: dict[str, int]

    # Affordance metadata
    affordance_count: int
    affordance_ids: list[str]
    affordance_id_to_index: dict[str, int]

    # Action metadata
    action_count: int  # Currently hardcoded to 6

    # Observation space metadata
    observation_dim: int  # Computed from grid, meters, affordances

    # Spatial metadata
    grid_size: int
    grid_cells: int

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

#### 5.2: Implement Stage 5 (Compute Metadata)

```python
from datetime import datetime


def _stage_5_compute_metadata(
    self,
    raw_configs: RawConfigs,
    symbol_table: UniverseSymbolTable
) -> UniverseMetadata:
    """
    Stage 5: Compute metadata from validated configs.

    Calculates:
    - Meter count (dynamic based on bars.yaml)
    - Observation dimension (depends on meter count)
    - Action count (currently hardcoded to 6)
    - Economic metadata (income, costs, balance)
    """
    # Meter metadata
    meter_count = len(raw_configs.bars.bars)
    meter_names = [
        bar.name
        for bar in sorted(raw_configs.bars.bars, key=lambda b: b.index)
    ]
    meter_name_to_index = {bar.name: bar.index for bar in raw_configs.bars.bars}

    # Affordance metadata
    affordance_count = len(raw_configs.affordances.affordances)
    affordance_ids = [aff.id for aff in raw_configs.affordances.affordances]
    affordance_id_to_index = {
        aff.id: i
        for i, aff in enumerate(raw_configs.affordances.affordances)
    }

    # Action metadata (currently hardcoded, will be dynamic after TASK-003 Action Space)
    action_count = 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT

    # Observation dimension (complex calculation)
    grid_size = raw_configs.training.grid_size

    if raw_configs.training.partial_observability:
        # POMDP: local window + position + meters + affordance + extras
        vision_range = raw_configs.training.vision_range
        window_size = 2 * vision_range + 1
        obs_dim = (
            window_size * window_size +  # Local grid
            2 +                           # Agent position (x, y)
            meter_count +                 # DYNAMIC meter count!
            affordance_count + 1 +        # Affordance at position (+ "none")
            4                             # Temporal extras
        )
    else:
        # Full observability: full grid + meters + affordance + extras
        obs_dim = (
            grid_size * grid_size +       # Full grid
            meter_count +                 # DYNAMIC meter count!
            affordance_count + 1 +        # Affordance at position
            4                             # Temporal extras
        )

    # Spatial metadata
    grid_cells = grid_size * grid_size

    # Economic metadata
    max_income = self._compute_max_income(raw_configs.affordances)
    total_costs = self._compute_total_costs(raw_configs.affordances)
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
        observation_dim=obs_dim,
        grid_size=grid_size,
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

**Success Criteria**:
- [ ] Metadata has correct meter_count for variable-size configs
- [ ] Observation dim scales with meter_count
- [ ] Observation dim correct for partial observability
- [ ] Economic metadata computed correctly

---

### Phase 6: Optimization & CompiledUniverse (4-6 hours)

**Goal**: Pre-compute optimization data and emit immutable artifact.

#### 6.1: Create OptimizationData

**File**: `src/townlet/universe/optimization.py` (NEW)

```python
import torch
from dataclasses import dataclass


@dataclass
class OptimizationData:
    """Pre-computed optimization data for fast runtime execution."""

    # Base depletion tensor [meter_count]
    base_depletions: torch.Tensor

    # Cascade lookup by category
    cascade_data: dict[str, list[dict]]

    # Modulation lookup
    modulation_data: list[dict]

    # Action mask table [24, num_affordances]
    # Pre-computed affordance availability by hour
    action_mask_table: torch.Tensor

    # Affordance position map (populated at reset)
    affordance_position_map: dict[str, torch.Tensor | None]
```

#### 6.2: Implement Stage 6 (Optimize)

```python
def _stage_6_optimize(
    self,
    raw_configs: RawConfigs,
    metadata: UniverseMetadata,
    device: str = "cpu"
) -> OptimizationData:
    """
    Stage 6: Pre-compute optimization data for fast runtime.

    Pre-computes:
    - Base depletion tensor
    - Cascade lookup tables
    - Action mask table (affordance availability by hour)
    """
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
    action_mask_table = torch.zeros(
        (24, metadata.affordance_count),
        dtype=torch.bool,
        device=device
    )

    for hour in range(24):
        for aff_idx, aff in enumerate(raw_configs.affordances.affordances):
            open_hour, close_hour = aff.operating_hours
            action_mask_table[hour, aff_idx] = self._is_open(hour, open_hour, close_hour)

    # 5. Affordance position map (populated at reset)
    affordance_position_map = {aff.id: None for aff in raw_configs.affordances.affordances}

    return OptimizationData(
        base_depletions=base_depletions,
        cascade_data=cascade_data,
        modulation_data=modulation_data,
        action_mask_table=action_mask_table,
        affordance_position_map=affordance_position_map,
    )
```

#### 6.3: Create CompiledUniverse

**File**: `src/townlet/universe/compiled.py` (NEW)

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledUniverse:
    """Immutable artifact representing a fully validated universe."""

    # Config components
    bars: BarsConfig
    cascades: CascadesConfig
    affordances: AffordanceConfigCollection
    training: TrainingConfig

    # Computed metadata
    metadata: UniverseMetadata

    # Pre-computed optimization data
    optimization_data: OptimizationData

    def __post_init__(self):
        """Validate universe is complete and consistent."""
        # Validate meter count matches
        if self.metadata.meter_count != len(self.bars.bars):
            raise ValueError(
                f"Metadata meter_count ({self.metadata.meter_count}) "
                f"doesn't match bars count ({len(self.bars.bars)})"
            )

        # Validate observation dim is positive
        if self.metadata.observation_dim <= 0:
            raise ValueError(
                f"Observation dim must be positive, got {self.metadata.observation_dim}"
            )

        # Validate action count is positive
        if self.metadata.action_count <= 0:
            raise ValueError(
                f"Action count must be positive, got {self.metadata.action_count}"
            )
```

#### 6.4: Implement Stage 7 (Emit)

```python
def _stage_7_emit_compiled_universe(
    self,
    raw_configs: RawConfigs,
    metadata: UniverseMetadata,
    optimization_data: OptimizationData
) -> CompiledUniverse:
    """
    Stage 7: Emit immutable compiled universe.

    Creates frozen dataclass that cannot be modified.
    """
    universe = CompiledUniverse(
        bars=raw_configs.bars,
        cascades=raw_configs.cascades,
        affordances=raw_configs.affordances,
        training=raw_configs.training,
        metadata=metadata,
        optimization_data=optimization_data,
    )

    # Validate immutability
    if not universe.__dataclass_fields__['bars'].frozen:
        raise CompilationError(
            stage="Stage 7: Emit",
            errors=["CompiledUniverse must be frozen (immutable)"],
            hints=["Check @dataclass(frozen=True) decorator"]
        )

    return universe
```

**Success Criteria**:
- [ ] Base depletion tensor has correct shape [meter_count]
- [ ] Cascade data sorted by target index
- [ ] Action mask table has correct shape [24, num_affordances]
- [ ] CompiledUniverse is frozen (immutable)
- [ ] CompiledUniverse validates consistency

---

### Phase 7: Compilation Cache (4-6 hours)

**Goal**: Add MessagePack caching for 10-100x faster subsequent loads.

#### 7.1: Implement Cache Serialization

```python
import msgpack
from pathlib import Path


class CompiledUniverse:
    # ... (existing code)

    def save_to_cache(self, path: Path):
        """Serialize compiled universe to MessagePack."""
        # Convert to dict
        data = {
            "bars": self.bars.model_dump(),
            "cascades": self.cascades.model_dump(),
            "affordances": self.affordances.model_dump(),
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
        training = TrainingConfig(**data["training"])

        # Reconstruct tensors
        optimization_data = OptimizationData(
            base_depletions=torch.tensor(data["optimization_data"]["base_depletions"]),
            cascade_data=data["optimization_data"]["cascade_data"],
            modulation_data=data["optimization_data"]["modulation_data"],
            action_mask_table=torch.tensor(data["optimization_data"]["action_mask_table"]),
            affordance_position_map={},
        )

        return cls(
            bars=bars,
            cascades=cascades,
            affordances=affordances,
            training=training,
            metadata=UniverseMetadata(**data["metadata"]),
            optimization_data=optimization_data,
        )
```

#### 7.2: Implement Cache Invalidation

```python
class UniverseCompiler:
    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile universe with mtime-based cache invalidation."""
        cache_path = config_dir / ".compiled" / "universe.msgpack"

        if use_cache and cache_path.exists():
            try:
                # Get cache modification time
                cache_mtime = cache_path.stat().st_mtime

                # Get all YAML modification times
                yaml_files = list(config_dir.glob("*.yaml"))
                yaml_mtimes = [f.stat().st_mtime for f in yaml_files]

                # Cache is fresh if newer than ALL source files
                if all(yaml_mtime < cache_mtime for yaml_mtime in yaml_mtimes):
                    logger.info(f"Loading cached universe from {cache_path}")
                    return CompiledUniverse.load_from_cache(cache_path)
                else:
                    logger.info("Cache stale (source YAML modified), recompiling...")

            except Exception as e:
                # Cache corrupted, fall back to full compilation
                logger.warning(f"Cache load failed ({e}), recompiling from source")

        # Cache miss or stale, do full compilation
        logger.info(f"Compiling universe from {config_dir}")
        universe = self._compile_full(config_dir)

        # Save to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            universe.save_to_cache(cache_path)
            logger.info(f"Saved compiled universe to {cache_path}")
        except Exception as e:
            logger.warning(f"Cache save failed ({e}), continuing without cache")

        return universe

    def _compile_full(self, config_dir: Path) -> CompiledUniverse:
        """Do full compilation (all 7 stages)."""
        # This is the existing compile() logic (stages 1-7)
        ...
```

**Success Criteria**:
- [ ] First compile saves cache
- [ ] Second compile loads from cache (10-100x faster)
- [ ] Cache invalidated when YAML modified
- [ ] Cache corruption handled gracefully

---

### Phase 8: VectorizedHamletEnv Refactoring (3-4 hours)

**Goal**: Refactor environment to accept compiled universe.

#### 8.1: Update VectorizedHamletEnv Constructor

```python
class VectorizedHamletEnv:
    def __init__(
        self,
        # NEW API: Accept compiled universe
        universe: CompiledUniverse,
        num_agents: int,
        device: str = "cuda",
        # REMOVED: config_pack_path, enabled_affordances (in universe.training)
    ):
        self.universe = universe
        self.num_agents = num_agents
        self.device = device

        # Read metadata instead of computing
        self.observation_dim = universe.metadata.observation_dim
        self.action_dim = universe.metadata.action_count
        self.meter_count = universe.metadata.meter_count
        self.grid_size = universe.metadata.grid_size

        # Initialize meter dynamics with universe
        self.meter_dynamics = MeterDynamics(
            universe=universe,
            num_agents=num_agents,
            device=device
        )

        # Initialize affordance engine with universe
        self.affordance_engine = AffordanceEngine(
            universe=universe,
            num_agents=num_agents,
            device=device
        )

        # Use pre-computed optimization data
        self.base_depletions = universe.optimization_data.base_depletions

        # State tensors (initialized in reset)
        self.positions = torch.zeros((num_agents, 2), dtype=torch.long, device=device)
        self.meters = torch.zeros(
            (num_agents, self.meter_count),  # DYNAMIC!
            dtype=torch.float32,
            device=device
        )
        self.dones = torch.zeros(num_agents, dtype=torch.bool, device=device)
        ...
```

#### 8.2: Update Training Scripts

```python
# BEFORE (old API)
env = VectorizedHamletEnv(
    num_agents=4,
    grid_size=8,
    config_pack_path=Path("configs/L1_full_observability"),
    enabled_affordances=["Bed", "Hospital", "HomeMeal", "Job"],
    device="cuda"
)

# AFTER (new API)
from townlet.universe.compiler import UniverseCompiler

compiler = UniverseCompiler()
universe = compiler.compile(Path("configs/L1_full_observability"))

env = VectorizedHamletEnv(
    universe=universe,
    num_agents=4,
    device="cuda"
)
```

**Success Criteria**:
- [ ] Environment accepts compiled universe
- [ ] Environment reads metadata correctly
- [ ] Environment behavior unchanged (integration test)
- [ ] Training scripts updated

---

## Benefits

### 1. Fail-Fast Error Detection

**Before**: Errors discovered during training (10 minutes in)
```
Training episode 1234...
Traceback: KeyError: 'moodiness' not found in meters
```

**After**: Errors caught at compilation (before training starts)
```
Universe Compilation Failed (Stage 3: Reference Resolution)

Found 1 error(s):
  1. cascades.yaml:low_mood_hits_energy: References non-existent meter 'moodiness'.
     Valid meters: [energy, health, mood, money, satiation, hygiene, social, fitness]

Hints:
  - Check for typos in meter names (case-sensitive)
  - Ensure bars.yaml defines all referenced meters
```

### 2. Clear Error Messages

All errors collected and reported with:
- **Context**: Which stage failed? Which file? Which line?
- **Specificity**: What was expected vs what was found?
- **Actionability**: How to fix it? What are valid alternatives?
- **Hints**: Common causes, related documentation

### 3. Fast Startup (10-100x with Cache)

| Operation | Without Cache | With Cache | Speedup |
|-----------|---------------|------------|---------|
| Load YAML | 10-20ms | 0ms | ∞ |
| Parse YAML | 10-30ms | 0ms | ∞ |
| Validate | 5-10ms | 0ms | ∞ |
| Resolve References | 2-5ms | 0ms | ∞ |
| Cross-Validate | 5-10ms | 0ms | ∞ |
| Deserialize Cache | - | 1-5ms | - |
| **Total** | **50-100ms** | **1-5ms** | **10-100x** |

### 4. Immutability Guarantees

- CompiledUniverse is frozen (cannot be modified)
- Safe to share between multiple environments
- Thread-safe (no locks needed)
- Reproducible (universe matches YAML exactly)

### 5. Metadata Available

Environment no longer computes obs_dim, action_dim:
```python
# Before (hardcoded)
obs_dim = grid_size * grid_size + 8 + affordances + extras
#                                  ^^^ HARDCODED!

# After (from metadata)
obs_dim = universe.metadata.observation_dim
# Automatically scales with meter_count!
```

### 6. Pre-Computed Optimizations

- Base depletion tensor pre-computed
- Cascade lookup tables pre-built
- Action mask table pre-computed [24, num_affordances]
- No per-step computation overhead

### 7. Cross-File Validation

Catches errors across multiple files:
- Dangling meter references
- Spatial impossibility (too many affordances)
- Economic imbalance (costs > income)
- Cascade circularity
- Temporal conflicts

---

## Dependencies

### Required

**None** - This is foundational infrastructure.

### Recommended (for full benefit)

**TASK-002A (UAC Core DTOs)**:
- Provides core configuration DTOs (TrainingConfig, EnvironmentConfig, etc.)
- Compiler uses these DTOs for type-safe config loading
- Without it, compiler loads raw YAML dicts (less safe)
- With it, configs are validated at load time

**TASK-002B (UAC Capabilities)** (optional):
- Provides capability system DTOs (multi_tick, cooldown, etc.)
- Compiler validates capability composition rules
- Optional - compiler works without it for basic configs

**TASK-005 (Variable-Size Meter System)**:
- Enables dynamic meter_count in metadata
- Without it, meter_count is always 8 (but compiler still works)
- With it, observation_dim scales correctly for 4-meter, 12-meter universes

### Enables

All future UAC work depends on robust compilation:
- **TASK-000** (Spatial Substrates): Compiler validates substrate configs
- **TASK-003** (Action Space): Compiler validates action-substrate compatibility
- **TASK-005** (BRAIN_AS_CODE): Compiler provides obs_dim, action_dim metadata

---

## Success Criteria

### Core Compiler

- [ ] Compiler loads all YAML files
- [ ] Stage 1 catches file not found, malformed YAML, invalid schema
- [ ] Stage 2 registers all meters and affordances
- [ ] Stage 3 detects dangling references with clear error messages
- [ ] Stage 4 validates spatial feasibility, economic balance, cascade circularity
- [ ] Stage 5 computes correct observation_dim (scales with meter_count)
- [ ] Stage 6 pre-computes optimization data (tensors, lookup tables)
- [ ] Stage 7 emits immutable CompiledUniverse

### Error Reporting

- [ ] Error messages include stage, file, location
- [ ] Error messages list valid alternatives
- [ ] Error messages include actionable hints
- [ ] All errors collected before failing (not fail-fast)

### Caching

- [ ] First compile saves cache
- [ ] Second compile loads from cache (10-100x faster)
- [ ] Cache invalidated when YAML modified (mtime check)
- [ ] Cache corruption handled gracefully (fallback to full compile)

### Environment Integration

- [ ] VectorizedHamletEnv accepts CompiledUniverse
- [ ] Environment reads metadata (obs_dim, action_dim, meter_count)
- [ ] Environment behavior unchanged (integration tests pass)
- [ ] Training scripts updated to use compiler

### Testing

- [ ] Unit tests for each compilation stage
- [ ] Integration tests for full compilation pipeline
- [ ] Fixture-based tests (valid/invalid config packs)
- [ ] Performance tests (compilation < 100ms without cache)

---

## Effort Estimate

### Breakdown

**Core Implementation (Original Scope)**:
- **Phase 1** (Core Compiler): 11-16 hours
- **Phase 2** (Symbol Table): 4-6 hours
- **Phase 3** (Error Collection): 4-6 hours
- **Phase 4** (Cross-Validation - original): 4-6 hours
- **Phase 5** (Metadata): 3-4 hours
- **Phase 6** (Optimization & CompiledUniverse): 4-6 hours
- **Phase 7** (Caching): 4-6 hours
- **Phase 8** (Environment Refactor): 3-4 hours
- **Subtotal (Original)**: 37-54 hours

**Research Integration Extensions**:
- **Phase 4 Extension** (Capability system validation): +6-8 hours
  - Affordance availability validation (Gap 1)
  - Action cost meter references (Gap 2)
  - Capability conflict detection (Finding 1)
  - Effect pipeline consistency (Finding 1)

**Total**: 43-62 hours → rounded to **46-66 hours** (6-8 days)

**Note**: +6-8h represents +16-15% increase from original estimate due to capability system complexity.

**Updated from research estimate (40-58 hours) based on detailed breakdown**

**Confidence**: High (well-defined scope, clear interfaces)

---

## Risks & Mitigations

### Risk 1: Breaking Existing Code

**Risk**: Refactoring VectorizedHamletEnv breaks existing training scripts.

**Mitigation**: Dual API (temporary)
```python
class VectorizedHamletEnv:
    def __init__(
        self,
        universe: CompiledUniverse | None = None,
        config_pack_path: Path | None = None,  # Deprecated
        ...
    ):
        if universe is not None:
            self._init_from_universe(universe)
        elif config_pack_path is not None:
            warnings.warn("config_pack_path deprecated, use compiler", DeprecationWarning)
            universe = UniverseCompiler().compile(config_pack_path)
            self._init_from_universe(universe)
```

**Timeline**:
- v1.0: Both APIs work (new recommended, old deprecated)
- v1.5: Old API logs error
- v2.0: Old API removed

---

### Risk 2: Cache Corruption

**Risk**: Corrupted cache file causes crashes.

**Mitigation**: Graceful degradation
```python
try:
    return CompiledUniverse.load_from_cache(cache_path)
except Exception as e:
    logger.warning(f"Cache load failed ({e}), recompiling from source")
    return self._compile_full(config_dir)
```

**Likelihood**: Low
**Impact**: Low (handled gracefully)

---

### Risk 3: Performance Regression

**Risk**: Compilation overhead slows down training startup.

**Mitigation**:
- Benchmarking + performance budget (<100ms without cache)
- Caching (1-5ms with cache)
- Can compile offline: `python -m townlet.compiler compile configs/L1`

**Likelihood**: Very low (compilation is fast, caching is very fast)
**Impact**: Negligible

---

### Risk 4: Incomplete Error Coverage

**Risk**: Missing some validation rules, errors slip through.

**Mitigation**:
- Start with high-value validations (Stage 3-4)
- Add more validations incrementally
- Collect production errors, add validation rules

**Likelihood**: Medium (validation is never "complete")
**Impact**: Low (can add validations incrementally)

---

## Testing Strategy

### Test Pyramid

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

### Layer 1: Fixture-Based Tests

Create test config packs in `tests/test_townlet/fixtures/`:

**Valid Configs**:
- `valid_8meter/` - Standard 8-meter universe (should compile)
- `valid_4meter/` - 4-meter tutorial universe (should compile)
- `valid_12meter/` - 12-meter complex universe (should compile)

**Invalid Configs**:
- `invalid_dangling_ref/` - Cascade references non-existent meter (should fail Stage 3)
- `invalid_spatial/` - Too many affordances for grid size (should fail Stage 4)
- `invalid_circular/` - Circular cascade dependency (should fail Stage 4)
- `invalid_temporal/` - Invalid operating hours (should fail Stage 4)
- `invalid_missing_file/` - Missing bars.yaml (should fail Stage 1)
- `invalid_malformed/` - Malformed YAML (should fail Stage 1)

**Tests**:
```python
def test_compile_valid_8meter():
    compiler = UniverseCompiler()
    universe = compiler.compile(Path("fixtures/valid_8meter"))
    assert universe.metadata.meter_count == 8

def test_compile_invalid_dangling_ref():
    compiler = UniverseCompiler()
    with pytest.raises(CompilationError, match="non-existent meter"):
        compiler.compile(Path("fixtures/invalid_dangling_ref"))
```

---

### Layer 2: Unit Tests (Stage Isolation)

Test each compilation stage independently:

```python
def test_stage_1_parse_valid_bars():
    """Stage 1 should parse valid bars.yaml."""
    compiler = UniverseCompiler()
    bars = compiler._stage_1_parse_bars(Path("fixtures/valid_8meter/bars.yaml"))
    assert len(bars.bars) == 8

def test_stage_2_build_symbol_table():
    """Stage 2 should register all meters."""
    compiler = UniverseCompiler()
    bars = load_bars_config(Path("fixtures/valid_8meter/bars.yaml"))

    symbol_table = UniverseSymbolTable()
    for bar in bars.bars:
        symbol_table.register_meter(bar.name, bar)

    assert "energy" in symbol_table.meters

def test_stage_3_detect_dangling_reference():
    """Stage 3 should catch dangling meter references."""
    compiler = UniverseCompiler()
    with pytest.raises(CompilationError, match="non-existent meter"):
        compiler.compile(Path("fixtures/invalid_dangling_ref"))

def test_stage_4_detect_spatial_impossibility():
    """Stage 4 should catch grid too small for affordances."""
    compiler = UniverseCompiler()
    with pytest.raises(CompilationError, match="Spatial impossibility"):
        compiler.compile(Path("fixtures/invalid_spatial"))

def test_stage_5_compute_observation_dim():
    """Stage 5 should compute correct observation dimension."""
    compiler = UniverseCompiler()
    universe = compiler.compile(Path("fixtures/valid_4meter"))
    # 4-meter universe should have smaller obs_dim than 8-meter
    assert universe.metadata.observation_dim < 91  # 8-meter obs_dim
```

---

### Layer 3: Integration Tests (End-to-End)

Test full compilation pipeline:

```python
def test_compile_and_use_universe():
    """Should compile universe and use it in environment."""
    compiler = UniverseCompiler()
    universe = compiler.compile(Path("fixtures/valid_8meter"))

    # Create environment with compiled universe
    env = VectorizedHamletEnv(universe=universe, num_agents=4, device="cpu")

    # Run one episode
    obs = env.reset()
    assert obs.shape == (4, universe.metadata.observation_dim)

    action = torch.randint(0, universe.metadata.action_count, (4,))
    obs, reward, done, info = env.step(action)

    assert obs.shape == (4, universe.metadata.observation_dim)

def test_compile_with_cache():
    """Should use cache on second compile."""
    compiler = UniverseCompiler()

    # First compile (no cache)
    start = time.time()
    universe1 = compiler.compile(Path("fixtures/valid_8meter"))
    first_time = time.time() - start

    # Second compile (with cache)
    start = time.time()
    universe2 = compiler.compile(Path("fixtures/valid_8meter"))
    cached_time = time.time() - start

    # Cache should be at least 5x faster
    assert cached_time < first_time / 5
    assert universe1.metadata == universe2.metadata

def test_compiled_universe_immutable():
    """Compiled universe should be immutable."""
    compiler = UniverseCompiler()
    universe = compiler.compile(Path("fixtures/valid_8meter"))

    with pytest.raises(AttributeError, match="frozen"):
        universe.bars.bars[0].initial = 0.5
```

---

### Layer 4: Performance Tests

```python
def test_compilation_performance():
    """Compilation should be < 100ms without cache."""
    compiler = UniverseCompiler()

    start = time.time()
    universe = compiler.compile(Path("fixtures/valid_8meter"), use_cache=False)
    compile_time = time.time() - start

    assert compile_time < 0.1, f"Compilation took {compile_time:.3f}s (expected <0.1s)"

def test_cache_performance():
    """Cache load should be < 5ms."""
    compiler = UniverseCompiler()

    # Warm up cache
    compiler.compile(Path("fixtures/valid_8meter"))

    # Measure cached load
    start = time.time()
    universe = compiler.compile(Path("fixtures/valid_8meter"))
    cached_time = time.time() - start

    assert cached_time < 0.005, f"Cache load took {cached_time:.3f}s (expected <0.005s)"
```

---

## Documentation Updates

After implementation, update:

### 1. UNIVERSE_AS_CODE.md

Add section on compilation:
```markdown
## Universe Compilation

Universes are compiled before training using a 7-stage pipeline:

1. Parse individual YAML files
2. Build symbol tables
3. Resolve cross-file references
4. Validate cross-file constraints
5. Compute metadata (obs_dim, action_dim)
6. Pre-compute optimization data
7. Emit immutable CompiledUniverse

### Usage

```python
from townlet.universe.compiler import UniverseCompiler

# Compile universe
compiler = UniverseCompiler()
universe = compiler.compile("configs/L1_full_observability")

# Use compiled universe
env = VectorizedHamletEnv(universe=universe, num_agents=4)
```

### Caching

Compilation is cached for fast subsequent loads:
- First compile: ~50-100ms
- Cached compile: ~1-5ms (10-100x speedup)
- Cache invalidated automatically when YAML files modified
```

### 2. CLAUDE.md

Update architecture section:
```markdown
### Universe Compilation

**UniverseCompiler** (`src/townlet/universe/compiler.py`):
- 7-stage compilation pipeline
- Cross-file validation (dangling references, spatial feasibility)
- MessagePack caching (10-100x speedup)

**CompiledUniverse** (`src/townlet/universe/compiled.py`):
- Immutable artifact (frozen dataclass)
- Contains configs + metadata + optimization data
- Shared across multiple environments

**Usage**:
```python
compiler = UniverseCompiler()
universe = compiler.compile("configs/L1_full_observability")
env = VectorizedHamletEnv(universe=universe, num_agents=4)
```
```

### 3. Create New Documentation

**docs/UNIVERSE-COMPILER.md** (NEW):
- Detailed compilation pipeline documentation
- Error message reference
- Performance characteristics
- Troubleshooting guide

---

## Follow-Up Work

After completing this task:

### 1. Implement Remaining TASKs

Compiler is foundation for:
- **TASK-000**: Validate substrate configs
- **TASK-003**: Validate action-substrate compatibility
- **TASK-005**: Use compiled obs_dim, action_dim

### 2. Add More Validation Rules

Start with high-value validations, add more incrementally:
- Cascade strength bounds checking
- Modulation parameter validation
- Economic equilibrium analysis
- Temporal mechanics consistency

### 3. CLI Tool (Optional)

```bash
# Compile universe
python -m townlet.compiler compile configs/L1

# Inspect compiled universe
python -m townlet.compiler inspect configs/L1/.compiled/universe.msgpack

# Validate without caching
python -m townlet.compiler validate configs/L1
```

### 4. Performance Optimization

If compilation is slow (>100ms):
- Profile each stage
- Optimize bottlenecks
- Consider parallel YAML loading

### 5. Advanced Features (Future)

- Incremental compilation (recompile only changed files)
- Distributed compilation (compile multiple universes in parallel)
- Compilation plugins (custom validation rules)

---

## Conclusion

**Universe compiler is the "bones" of UAC** - it must be robust before we worry about reward models or economic tuning.

**Impact**:
- ✅ Fail-fast error detection (catch errors before training)
- ✅ Clear error messages (context, specificity, actionability, hints)
- ✅ Fast startup (10-100x with cache)
- ✅ Immutability guarantees (safe to share, reproducible)
- ✅ Metadata available (dynamic obs_dim, action_dim)
- ✅ Cross-file validation (dangling refs, spatial feasibility, circularity)

**Effort**: 37-54 hours (5-7 days)

**Risk**: Low (well-defined scope, clear interfaces, incremental implementation)

**Priority**: HIGH (foundational for all UAC work)

**Slogan**: "Compile universes with best practices - fail fast, cache aggressively, error beautifully."
