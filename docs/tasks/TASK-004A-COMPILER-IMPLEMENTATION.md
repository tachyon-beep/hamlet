# TASK-004A: Universe Compiler Implementation

**Status**: Planned (Aligned with COMPILER_ARCHITECTURE.md v1.0)
**Priority**: HIGH (Foundational for UAC system integrity)
**Estimated Effort**: 52-72 hours (6.5-9 days)
  - **UPDATED** from 37-54h (original) to align with COMPILER_ARCHITECTURE.md
  - **Additions**: CuesCompiler (+3-4h), Capability Validation (+6-8h), ObservationSpec (+2h), Rich Metadata (+4h)
  - **Total additions**: +15-18h (+40-33% increase)
**Dependencies**: TASK-003 (UAC Core DTOs - COMPLETE)
**Enables**: All future UAC work + TASK-005 (BAC - requires ObservationSpec)
**Authoritative Reference**: `docs/architecture/COMPILER_ARCHITECTURE.md`

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
    """Container for raw config objects loaded from YAML.

    Per COMPILER_ARCHITECTURE.md §2.1: All core universe configs.
    """
    substrate: SubstrateConfig  # Spatial structure (grid, continuous, aspatial)
    variables: VariablesConfig  # VFS variable definitions
    bars: BarsConfig  # Meter definitions
    cascades: CascadesConfig  # Meter relationships
    affordances: AffordanceConfigCollection  # Interaction definitions
    cues: CuesConfig  # Theory of Mind cue definitions
    actions: ActionSpaceConfig  # Global action vocabulary (substrate + custom)
    training: TrainingConfig  # Training hyperparameters


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

    # Load all core configs (Per COMPILER_ARCHITECTURE.md §2.3 Stage 1)
    substrate = load_substrate_config(config_dir / "substrate.yaml")
    variables = load_variables_config(config_dir / "variables.yaml")
    cascades = load_cascades_config(config_dir / "cascades.yaml")
    affordances = load_affordance_config(config_dir / "affordances.yaml")
    cues = load_cues_config(config_dir / "cues.yaml")
    actions = load_action_space_config(config_dir / "global_actions.yaml")
    training = load_training_config(config_dir / "training.yaml")

    return RawConfigs(
        substrate=substrate,
        variables=variables,
        bars=bars,
        cascades=cascades,
        affordances=affordances,
        cues=cues,
        actions=actions,
        training=training,
    )
```

**Success Criteria**:

- [ ] Compiler loads all YAML files
- [ ] File not found raises clear error
- [ ] Malformed YAML raises parse error
- [ ] Invalid schema raises validation error

---

### Phase 2: Symbol Table & Reference Resolution (7-10 hours)

**Goal**: Implement cross-file reference resolution with clear error messages.

**UPDATED** from 4-6h: Added cues loading and validation (+3-4h) per COMPILER_ARCHITECTURE.md §2.1

#### 2.1: Create UniverseSymbolTable

**File**: `src/townlet/universe/symbol_table.py` (NEW)

```python
class UniverseSymbolTable:
    """Central registry of all named entities in universe.

    Per COMPILER_ARCHITECTURE.md §2.3: Symbol table for Stage 2.
    """

    def __init__(self):
        self.variables: dict[str, VariableDef] = {}  # VFS variables
        self.meters: dict[str, BarConfig] = {}  # Meters (subset of variables)
        self.affordances: dict[str, AffordanceConfig] = {}  # Interactions
        self.cues: dict[str, CueConfig] = {}  # Theory of Mind cues
        self.actions: dict[str, ActionConfig] = {}  # Global action vocabulary

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

    def register_cue(self, meter_name: str, config: CueConfig):
        """Register cue for later validation."""
        if meter_name in self.cues:
            raise CompilationError(
                stage="Stage 2: Symbol Registration",
                errors=[f"Duplicate cue for meter: '{meter_name}'"],
                hints=["Each meter can have at most one cue definition"]
            )
        self.cues[meter_name] = config

    def register_variable(self, var_id: str, config: VariableDef):
        """Register VFS variable for later reference resolution."""
        if var_id in self.variables:
            raise CompilationError(
                stage="Stage 2: Symbol Registration",
                errors=[f"Duplicate variable ID: '{var_id}'"],
                hints=["Each variable must have unique ID"]
            )
        self.variables[var_id] = config

    def register_action(self, action_id: int, config: ActionConfig):
        """Register action for later reference resolution."""
        if action_id in self.actions:
            raise CompilationError(
                stage="Stage 2: Symbol Registration",
                errors=[f"Duplicate action ID: {action_id}"],
                hints=["Each action must have unique ID"]
            )
        self.actions[action_id] = config

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

    Per COMPILER_ARCHITECTURE.md §2.3 Stage 2: Register all named entities.
    """
    symbol_table = UniverseSymbolTable()

    # Register VFS variables
    for var in raw_configs.variables.variables:
        symbol_table.register_variable(var.id, var)

    # Register meters (subset of variables)
    for bar in raw_configs.bars.bars:
        symbol_table.register_meter(bar.name, bar)

    # Register affordances
    for aff in raw_configs.affordances.affordances:
        symbol_table.register_affordance(aff.id, aff)

    # Register cues
    for meter_name, cue in raw_configs.cues.items():
        symbol_table.register_cue(meter_name, cue)

    # Register actions
    for action in raw_configs.actions.actions:
        symbol_table.register_action(action.id, action)

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
- [ ] Symbol table registers all cues (NEW)
- [ ] Duplicate names detected and reported
- [ ] Dangling references caught with clear error messages
- [ ] Error messages list valid alternatives
- [ ] Cues loaded from cues.yaml and validated (NEW)

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

    # 5. Cues validation (NEW - Per COMPILER_ARCHITECTURE.md §5.3)
    for meter_name, cue_config in raw_configs.cues.items():
        # Validate cues reference valid meters
        if meter_name not in symbol_table.meters:
            errors.add_error(
                f"cues.yaml:{meter_name}: References non-existent meter. "
                f"Available meters: {list(symbol_table.meters.keys())}"
            )
            continue

        # Validate ranges cover full [0.0, 1.0] domain
        ranges = [(cue.min_value, cue.max_value) for cue in cue_config.visual_cues]
        if not self._ranges_cover_domain(ranges, 0.0, 1.0):
            errors.add_error(
                f"cues.yaml:{meter_name}: Cue ranges don't cover full [0.0, 1.0] domain. "
                f"Every meter value must map to exactly one cue."
            )

        # Check for overlapping ranges
        if self._ranges_overlap(ranges):
            errors.add_error(
                f"cues.yaml:{meter_name}: Cue ranges overlap. "
                f"Each meter value must map to exactly ONE cue."
            )

    # 6. Capability conflicts (NEW - from research Finding 1)
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
    # See: docs/research/RESEARCH-TASK-002A-UNSOLVED-PROBLEMS-CONSOLIDATED.md
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


def _ranges_cover_domain(
    self,
    ranges: list[tuple[float, float]],
    domain_min: float,
    domain_max: float
) -> bool:
    """
    Check if ranges cover the full domain without gaps.

    Args:
        ranges: List of (min, max) tuples
        domain_min: Minimum value of domain (e.g., 0.0)
        domain_max: Maximum value of domain (e.g., 1.0)

    Returns:
        True if ranges cover [domain_min, domain_max] without gaps
    """
    # Sort ranges by start value
    sorted_ranges = sorted(ranges, key=lambda r: r[0])

    # Check first range starts at domain_min
    if sorted_ranges[0][0] != domain_min:
        return False

    # Check ranges are contiguous (no gaps)
    for i in range(len(sorted_ranges) - 1):
        current_end = sorted_ranges[i][1]
        next_start = sorted_ranges[i + 1][0]
        if current_end != next_start:
            return False

    # Check last range ends at domain_max
    if sorted_ranges[-1][1] != domain_max:
        return False

    return True


def _ranges_overlap(self, ranges: list[tuple[float, float]]) -> bool:
    """
    Check if any ranges overlap.

    Args:
        ranges: List of (min, max) tuples

    Returns:
        True if any ranges overlap
    """
    # Sort ranges by start value
    sorted_ranges = sorted(ranges, key=lambda r: r[0])

    # Check for overlaps
    for i in range(len(sorted_ranges) - 1):
        current_end = sorted_ranges[i][1]
        next_start = sorted_ranges[i + 1][0]
        # Overlap if current range extends past start of next range
        if current_end > next_start:
            return True

    return False
```

**Success Criteria**:

- [ ] Spatial validation catches over-crowded grids
- [ ] Economic validation warns on poverty traps
- [ ] Circularity detection catches cascade cycles
- [ ] Temporal validation catches invalid operating hours
- [ ] **Cues validation** (NEW - Per COMPILER_ARCHITECTURE.md §5.3)
  - [ ] Validates cues reference valid meters
  - [ ] Validates ranges cover full [0.0, 1.0] domain
  - [ ] Detects overlapping ranges
  - [ ] Helper methods `_ranges_cover_domain()` and `_ranges_overlap()` work correctly
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

### Phase 5: Metadata Computation (5-6 hours)

**Goal**: Calculate derived properties (obs_dim, action_dim, etc.) + build ObservationSpec

**UPDATED** from 3-4h: Added ObservationSpec (+2h) per COMPILER_ARCHITECTURE.md §3.2 (BLOCKS TASK-005 BAC)

#### 5.1: Create UniverseMetadata

**File**: `src/townlet/universe/metadata.py` (NEW)

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseMetadata:
    """Derived metadata about compiled universe.

    Per COMPILER_ARCHITECTURE.md §3.1: Core metadata contract.
    """

    # Universe identification (NEW - Per §3.1)
    universe_name: str  # e.g., "L1_full_observability"
    schema_version: str  # UAC schema version (e.g., "1.0")

    # Substrate metadata (NEW - Per §3.1)
    substrate_type: str  # e.g., "grid2d", "continuous3d", "aspatial"
    position_dim: int  # Substrate dimensionality (0 for aspatial, 2 for grid2d, 3 for grid3d, etc.)

    # Meter metadata
    meter_count: int
    meter_names: list[str]  # Sorted by index
    meter_name_to_index: dict[str, int]

    # Affordance metadata
    affordance_count: int
    affordance_ids: list[str]
    affordance_id_to_index: dict[str, int]

    # Action metadata (FIXED - was hardcoded)
    action_count: int  # Composed from substrate + custom actions

    # Observation space metadata (FIXED - was computed from training params)
    observation_dim: int  # Built from VFS ObservationSpecBuilder

    # Spatial metadata (legacy - can be removed if substrate provides)
    grid_size: int | None = None  # Only for grid substrates
    grid_cells: int | None = None  # Only for grid substrates

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
    config_hash: str  # SHA-256 hash of all config file contents (for checkpoint compatibility)
```

#### 5.1b: Create ObservationSpec (NEW - Per COMPILER_ARCHITECTURE.md §3.2)

**File**: `src/townlet/universe/observation_spec.py` (NEW)

**Critical**: Required for UAC → BAC data contract. Without this, BAC can only build simple MLPs, not custom encoders.

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class ObservationField:
    """Single field in observation vector."""
    name: str  # e.g., "energy", "position", "local_grid"
    type: Literal["scalar", "vector", "categorical", "spatial_grid"]
    dims: int  # Number of dimensions this field occupies
    start_index: int  # Index in flat observation vector
    end_index: int  # Exclusive end index (for slicing)
    scope: Literal["global", "agent", "agent_private"]
    description: str

    # Semantic metadata for custom encoders
    semantic_type: str | None = None  # "position", "meter", "affordance", "cue", "temporal", "vision"
    categorical_labels: list[str] | None = None  # For one-hot encodings


@dataclass
class ObservationSpec:
    """Complete observation specification (UAC → BAC contract)."""
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

**Integration**: Use existing VFS `VFSObservationSpecBuilder` - infrastructure already exists!

```python
# In Stage 5 implementation:
from townlet.vfs.observation_builder import VFSObservationSpecBuilder

obs_spec_builder = VFSObservationSpecBuilder(
    variable_registry=raw_configs.vfs_registry,
    substrate=raw_configs.substrate,
)
observation_spec = obs_spec_builder.build_spec()
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

    # Action metadata (currently hardcoded, will be dynamic after TASK-002A Action Space)
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

    # Config hash (for checkpoint compatibility - Per COMPILER_ARCHITECTURE.md §4.2)
    config_hash = self._compute_config_hash(config_dir)

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
        config_hash=config_hash,
    )
```

**Success Criteria**:

- [ ] Metadata has correct meter_count for variable-size configs
- [ ] Observation dim scales with meter_count
- [ ] Observation dim correct for partial observability
- [ ] Economic metadata computed correctly
- [ ] **Config hash computed correctly** (NEW - Per COMPILER_ARCHITECTURE.md §4.2)
  - [ ] SHA-256 hash includes all config YAML file contents
  - [ ] Identical configs produce identical hashes
  - [ ] Any config change produces different hash
  - [ ] Hash enables checkpoint compatibility validation
- [ ] **ObservationSpec built correctly** (NEW - Per COMPILER_ARCHITECTURE.md §3.2)
  - [ ] ObservationSpec.total_dims matches computed observation_dim
  - [ ] All observation fields have correct start/end indices
  - [ ] VFS integration uses existing VFSObservationSpecBuilder
  - [ ] ObservationSpec enables custom neural encoders (BLOCKS TASK-005)

---

### Phase 6: Optimization & CompiledUniverse (8-10 hours)

**Goal**: Pre-compute optimization data, build rich metadata, and emit immutable artifact.

**UPDATED** from 4-6h: Added rich metadata structures (+4h) per COMPILER_ARCHITECTURE.md §3.3

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

#### 6.1b: Create Rich Metadata Structures (NEW - Per COMPILER_ARCHITECTURE.md §3.3)

**File**: `src/townlet/universe/rich_metadata.py` (NEW)

**Purpose**: Enable training system to log per-meter metrics, track affordance usage, apply action masks.

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class MeterInfo:
    """Single meter metadata."""
    name: str
    index: int  # Index in meters tensor
    critical: bool  # Agent dies if reaches 0?
    initial_value: float
    observable: bool  # In observation space?
    description: str


@dataclass
class MeterMetadata:
    """Collection of all meter metadata."""
    meters: list[MeterInfo]

    def get_meter_by_name(self, name: str) -> MeterInfo:
        """Lookup meter by name."""
        for meter in self.meters:
            if meter.name == name:
                return meter
        raise KeyError(f"Meter '{name}' not found")


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
class ActionSpaceMetadata:
    """Collection of all action metadata."""
    actions: list[ActionMetadata]
    action_dim: int  # Total actions (including disabled)

    def get_action_by_id(self, action_id: int) -> ActionMetadata:
        """Lookup action by ID."""
        for action in self.actions:
            if action.id == action_id:
                return action
        raise KeyError(f"Action ID {action_id} not found")


@dataclass
class AffordanceInfo:
    """Single affordance metadata."""
    name: str
    id: str
    index: int  # Index in affordance list
    type: str  # "resource", "service", "hazard", etc.
    description: str


@dataclass
class AffordanceMetadata:
    """Collection of all affordance metadata."""
    affordances: list[AffordanceInfo]

    def get_affordance_by_id(self, aff_id: str) -> AffordanceInfo:
        """Lookup affordance by ID."""
        for aff in self.affordances:
            if aff.id == aff_id:
                return aff
        raise KeyError(f"Affordance '{aff_id}' not found")
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
    """Immutable artifact representing a fully validated universe.

    Per COMPILER_ARCHITECTURE.md §3.3: Complete training hand-off contract.
    """

    # Core config components (Per §3.3)
    substrate: SubstrateConfig  # Spatial structure
    variables: VariablesConfig  # VFS variable definitions
    bars: BarsConfig  # Meter definitions
    cascades: CascadesConfig  # Meter relationships
    affordances: AffordanceConfigCollection  # Interactions
    cues: CuesConfig  # Theory of Mind cues
    actions: ActionSpaceConfig  # Global action vocabulary
    training: TrainingConfig  # Training hyperparameters

    # Computed metadata (Per §3.1)
    metadata: UniverseMetadata  # Core metadata contract

    # UAC → BAC data contracts (Per §3.2)
    observation_spec: ObservationSpec  # Observation field metadata for custom encoders

    # UAC → Training data contracts (Per §3.3)
    action_space_metadata: ActionSpaceMetadata  # Action costs and labels
    meter_metadata: MeterMetadata  # Meter ranges and semantic info
    affordance_metadata: AffordanceMetadata  # Affordance descriptions and categories

    # Pre-computed optimization data (legacy)
    optimization_data: OptimizationData  # Tensors for fast meter dynamics

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

    def create_environment(self, num_agents: int, device: str = "cuda"):
        """
        Create VectorizedHamletEnv from compiled universe.

        Per COMPILER_ARCHITECTURE.md §3.3: Training hand-off helper.

        Args:
            num_agents: Number of agents to simulate
            device: Device for tensors ("cuda" or "cpu")

        Returns:
            Configured VectorizedHamletEnv instance
        """
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        return VectorizedHamletEnv(
            universe=self,
            num_agents=num_agents,
            device=device
        )

    def check_checkpoint_compatibility(self, checkpoint: dict) -> tuple[bool, str]:
        """
        Check if checkpoint is compatible with this universe.

        Per COMPILER_ARCHITECTURE.md §6.2: Checkpoint validation helper.

        Args:
            checkpoint: Loaded checkpoint dict

        Returns:
            (is_compatible, message) tuple
        """
        # Check config hash
        if checkpoint['config_hash'] != self.metadata.config_hash:
            return (
                False,
                f"Config hash mismatch:\n"
                f"  Checkpoint: {checkpoint['config_hash'][:16]}...\n"
                f"  Current:    {self.metadata.config_hash[:16]}...\n"
                f"Transfer learning may fail if configs differ significantly."
            )

        # Check architecture dimensions
        if checkpoint['observation_dim'] != self.metadata.observation_dim:
            return (
                False,
                f"Observation dim mismatch: "
                f"checkpoint={checkpoint['observation_dim']}, "
                f"current={self.metadata.observation_dim}"
            )

        if checkpoint['action_dim'] != self.metadata.action_count:
            return (
                False,
                f"Action dim mismatch: "
                f"checkpoint={checkpoint['action_dim']}, "
                f"current={self.metadata.action_count}"
            )

        return (True, "Checkpoint compatible")
```

#### 6.4: Implement Stage 7 (Emit)

```python
def _stage_7_emit_compiled_universe(
    self,
    raw_configs: RawConfigs,
    metadata: UniverseMetadata,
    observation_spec: ObservationSpec,
    action_space_metadata: ActionSpaceMetadata,
    meter_metadata: MeterMetadata,
    affordance_metadata: AffordanceMetadata,
    optimization_data: OptimizationData
) -> CompiledUniverse:
    """
    Stage 7: Emit immutable compiled universe.

    Per COMPILER_ARCHITECTURE.md §2.3 Stage 7: Create frozen artifact.
    """
    universe = CompiledUniverse(
        substrate=raw_configs.substrate,
        variables=raw_configs.variables,
        bars=raw_configs.bars,
        cascades=raw_configs.cascades,
        affordances=raw_configs.affordances,
        cues=raw_configs.cues,
        actions=raw_configs.actions,
        training=raw_configs.training,
        metadata=metadata,
        observation_spec=observation_spec,
        action_space_metadata=action_space_metadata,
        meter_metadata=meter_metadata,
        affordance_metadata=affordance_metadata,
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
- [ ] **Rich metadata structures built correctly** (NEW - Per COMPILER_ARCHITECTURE.md §3.3)
  - [ ] MeterMetadata contains all meters with proper indexing
  - [ ] ActionSpaceMetadata contains all actions with costs
  - [ ] AffordanceMetadata contains all affordances with descriptions
  - [ ] Metadata lookup methods work correctly (get_meter_by_name, get_action_by_id, etc.)
  - [ ] Rich metadata enables training system integration

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

#### 7.2: Implement Hash-Based Cache Invalidation

**Per COMPILER_ARCHITECTURE.md §4.2**: Use SHA-256 hash of config file contents for robust cache invalidation.

**Why hash-based over mtime-based:**
- **Robust across filesystems**: mtime can be unreliable (git checkout, CI runners, network filesystems)
- **Content-aware**: Only invalidates when config *actually* changes, not just timestamps
- **Portable**: Works across different machines and deployment environments
- **Deterministic**: Same content always produces same hash

```python
import hashlib
from pathlib import Path


class UniverseCompiler:
    def _compute_config_hash(self, config_dir: Path) -> str:
        """
        Compute SHA-256 hash of all config file contents.

        Args:
            config_dir: Directory containing YAML config files

        Returns:
            Hexadecimal SHA-256 hash string

        Note: Per COMPILER_ARCHITECTURE.md §4.2
        """
        hasher = hashlib.sha256()

        # Get all YAML files in sorted order (for determinism)
        yaml_files = sorted(config_dir.glob("*.yaml"))

        for yaml_file in yaml_files:
            # Hash filename (for uniqueness)
            hasher.update(yaml_file.name.encode('utf-8'))

            # Hash file contents
            with open(yaml_file, 'rb') as f:
                hasher.update(f.read())

        return hasher.hexdigest()

    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile universe with hash-based cache invalidation."""
        cache_path = config_dir / ".compiled" / "universe.msgpack"

        if use_cache and cache_path.exists():
            try:
                # Load cached universe
                cached_universe = CompiledUniverse.load_from_cache(cache_path)

                # Compute current config hash
                current_hash = self._compute_config_hash(config_dir)

                # Compare hashes
                if cached_universe.metadata.config_hash == current_hash:
                    logger.info(f"Loading cached universe (hash={current_hash[:8]}...)")
                    return cached_universe
                else:
                    logger.info(
                        f"Cache stale (config changed):\n"
                        f"  Cached:  {cached_universe.metadata.config_hash[:8]}...\n"
                        f"  Current: {current_hash[:8]}...\n"
                        f"Recompiling..."
                    )

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
            logger.info(
                f"Saved compiled universe to {cache_path}\n"
                f"  Hash: {universe.metadata.config_hash[:16]}..."
            )
        except Exception as e:
            logger.warning(f"Cache save failed ({e}), continuing without cache")

        return universe

    def _compile_full(self, config_dir: Path) -> CompiledUniverse:
        """Do full compilation (all 7 stages)."""
        # This is the existing compile() logic (stages 1-7)
        ...
```

**Success Criteria**:

- [ ] First compile saves cache with config_hash
- [ ] Second compile loads from cache (10-100x faster)
- [ ] **Hash-based invalidation** (NEW - Per COMPILER_ARCHITECTURE.md §4.2)
  - [ ] Cache invalidated when config content changes (not just mtime)
  - [ ] Identical configs produce identical hashes (deterministic)
  - [ ] Hash comparison works across filesystems and git operations
  - [ ] `_compute_config_hash()` includes all YAML files in sorted order
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

#### 8.3: Checkpoint Compatibility Validation (NEW - Per COMPILER_ARCHITECTURE.md §6.2)

**Critical**: Use `universe.metadata.config_hash` to validate checkpoint compatibility during transfer learning.

```python
class TrainingRunner:
    def save_checkpoint(self, universe: CompiledUniverse, q_network, path: Path):
        """Save checkpoint with config_hash for compatibility checking."""
        torch.save({
            'q_network_state': q_network.state_dict(),
            'config_hash': universe.metadata.config_hash,  # NEW: Store hash
            'observation_dim': universe.metadata.observation_dim,
            'action_dim': universe.metadata.action_count,
            'meter_count': universe.metadata.meter_count,
        }, path)

    def load_checkpoint(self, universe: CompiledUniverse, q_network, path: Path):
        """Load checkpoint with soft-warning for config mismatches."""
        checkpoint = torch.load(path)

        # Soft-warning for config mismatch (Per COMPILER_ARCHITECTURE.md §6.2)
        if checkpoint['config_hash'] != universe.metadata.config_hash:
            logger.warning(
                f"⚠️  Checkpoint config mismatch (transfer learning):\n"
                f"  Checkpoint hash: {checkpoint['config_hash'][:16]}...\n"
                f"  Current hash:    {universe.metadata.config_hash[:16]}...\n"
                f"  This may cause training issues if configs differ significantly.\n"
                f"  Recommendation: Use checkpoints trained on identical configs."
            )

        # Load weights (may fail if architecture changed)
        try:
            q_network.load_state_dict(checkpoint['q_network_state'])
        except RuntimeError as e:
            raise ValueError(
                f"Failed to load checkpoint (architecture mismatch):\n"
                f"  Expected obs_dim={universe.metadata.observation_dim}, "
                f"got {checkpoint['observation_dim']}\n"
                f"  Expected action_dim={universe.metadata.action_count}, "
                f"got {checkpoint['action_dim']}\n"
                f"  Original error: {e}"
            )
```

**Success Criteria**:

- [ ] Environment accepts compiled universe
- [ ] Environment reads metadata correctly
- [ ] Environment behavior unchanged (integration test)
- [ ] Training scripts updated
- [ ] **Checkpoint compatibility validation** (NEW - Per COMPILER_ARCHITECTURE.md §6.2)
  - [ ] Checkpoints save config_hash from universe.metadata
  - [ ] Soft-warning displayed when loading checkpoint with different config_hash
  - [ ] Hard error when architecture dimensions mismatch
  - [ ] Transfer learning warnings enable debugging config drift

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

**TASK-003 (UAC Core DTOs)**:

- Provides core configuration DTOs (TrainingConfig, EnvironmentConfig, etc.)
- Compiler uses these DTOs for type-safe config loading
- Without it, compiler loads raw YAML dicts (less safe)
- With it, configs are validated at load time

**TASK-004B (UAC Capabilities)** (optional):

- Provides capability system DTOs (multi_tick, cooldown, etc.)
- Compiler validates capability composition rules
- Optional - compiler works without it for basic configs

**TASK-005 (Variable-Size Meter System)**:

- Enables dynamic meter_count in metadata
- Without it, meter_count is always 8 (but compiler still works)
- With it, observation_dim scales correctly for 4-meter, 12-meter universes

### Enables

All future UAC work depends on robust compilation:

- **TASK-002A** (Spatial Substrates): Compiler validates substrate configs
- **TASK-002A** (Action Space): Compiler validates action-substrate compatibility
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

**COMPILER_ARCHITECTURE.md Additions** (Aligned with authoritative reference):

- **Phase 2 Extension** (CuesCompiler): +3-4 hours
  - Cue loading from cues.yaml
  - Cue registration in symbol table
  - Cue meter reference validation
- **Phase 4 Extension** (Capability system validation): +6-8 hours
  - Affordance availability validation (Gap 1)
  - Action cost meter references (Gap 2)
  - Capability conflict detection (Finding 1)
  - Effect pipeline consistency (Finding 1)
  - Cues validation (range coverage, overlaps)
- **Phase 5 Extension** (ObservationSpec): +2 hours
  - UAC → BAC data contract (BLOCKS TASK-005)
  - VFS integration with VFSObservationSpecBuilder
- **Phase 6 Extension** (Rich Metadata): +4 hours
  - MeterMetadata, ActionSpaceMetadata, AffordanceMetadata
  - Training system integration structures

**Updated Breakdown**:

- **Phase 1** (Core Compiler): 11-16 hours
- **Phase 2** (Symbol Table + CuesCompiler): 7-10 hours
- **Phase 3** (Error Collection): 4-6 hours
- **Phase 4** (Cross-Validation + Cues): 10-14 hours
- **Phase 5** (Metadata + ObservationSpec): 5-6 hours
- **Phase 6** (Optimization + Rich Metadata): 8-10 hours
- **Phase 7** (Caching): 4-6 hours
- **Phase 8** (Environment Refactor): 3-4 hours

**Total**: 52-72 hours (6.5-9 days)

**Note**: +15-18h increase from original 37-54h estimate (+40-33%) due to:
- CuesCompiler integration (+3-4h)
- ObservationSpec UAC→BAC contract (+2h)
- Rich Metadata structures (+4h)
- Capability system validation (+6-8h)

**Updated per COMPILER_ARCHITECTURE.md (authoritative reference)**

**Confidence**: High (well-defined scope, clear interfaces, aligned with architecture)

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
- **TASK-002A**: Validate substrate configs
- **TASK-002A**: Validate action-substrate compatibility
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
