# TASK-004A: TDD Implementation Plan
# Universe Compiler - 7-Stage Pipeline Implementation

**Task**: TASK-004A Universe Compiler Implementation
**Priority**: CRITICAL (Blocks TASK-005 BAC, TASK-006 Visualization)
**Effort**: 60-82 hours (includes TDD, integration testing, validation, 50% complexity buffer)
**Status**: Ready for TDD (Post-Peer-Review)
**Created**: 2025-11-08
**Updated**: 2025-11-08 (Post peer review - time estimates, test specs, cache versioning)
**Method**: Test-Driven Development with RED-GREEN-REFACTOR cycles

**Keywords**: Universe Compiler, UAC, 7-stage pipeline, TDD, Pydantic, immutability, hash-based caching, ObservationSpec, CompiledUniverse
**Test Strategy**: TDD (unit tests first, then integration)
**Breaking Changes**: No (additive - DemoRunner continues to work)

---

## AI-Friendly Summary (Skim This First!)

**What**: Implement a 7-stage compilation pipeline that transforms raw YAML configs into an immutable `CompiledUniverse` artifact with complete metadata, observation specs, and rich training metadata.

**Why**: Enables config validation at load time (not runtime), checkpoint compatibility validation, custom neural encoders (BAC), and substrate-agnostic training systems.

**How**: TDD approach with 10 phases, each following RED-GREEN-REFACTOR. Build bottom-up from data structures → stages → integration → validation.

**Quick Assessment**:

- **Implementation Approach**: Bottom-up TDD (data structures → stages → pipeline → integration)
- **Test Coverage Goal**: 90%+ unit, 100% integration (all 6 reference configs)
- **Phases**: 10 phases over 60-82 hours (includes 50% complexity buffer)
- **Risk Level**: Medium-High (graph algorithms, cross-validation, backward compatibility)

**Decision Point**: If you're not implementing the Universe Compiler, STOP READING HERE.

---

## Review-Driven Updates (2025-11-08)

This plan was reviewed by an independent peer reviewer and updated to address critical findings:

### ✅ Issue 1: Time Estimates 30-50% Optimistic - RESOLVED

- **Problem**: Original estimates (40-55h) assumed zero blockers/rework, unrealistic for complex compiler
- **Fix**: Increased to 60-82 hours (50% buffer for graph algorithms, cross-validation complexity)
- **Time Impact**: +20-27 hours

### ✅ Issue 2: Missing Circular Cascade Test Specifications - RESOLVED

- **Problem**: DFS implementation shown but no edge case tests for cycle detection
- **Fix**: Added comprehensive test suite in Phase 6 (self-loops, complex cycles, disconnected graphs)
- **Impact**: Prevents silent bugs in cycle detection

### ✅ Issue 3: Missing Cue Range Validation Test Specifications - RESOLVED

- **Problem**: Helper methods shown but no mathematical edge cases specified
- **Fix**: Added test suite for gaps, overlaps, floating point boundaries in Phase 6
- **Impact**: Ensures correct validation of meter ranges

### ✅ Issue 4: Cache Versioning Not Addressed - RESOLVED

- **Problem**: MessagePack serialization without version field → cache corruption on schema changes
- **Fix**: Added `cache_version` field + invalidation logic in Phase 9
- **Impact**: Prevents cache corruption on future schema evolution

### ✅ Issue 5: ObservationSpec Class Underspecified - RESOLVED

- **Problem**: Usage shown but complete class definition missing
- **Fix**: Added full class specification with `total_dims` property and query methods in Phase 7
- **Impact**: Clear API for implementer

### Additional Improvements

- ✅ **Pre-Implementation Verification**: Added script to verify all TASK-003 load functions and VFS builder API
- ✅ **Error Message Testing**: Added assertions on error content (not just error type)
- ✅ **Dimension Regression Tests**: Added explicit validation of observation_dim for each config

**Review Score**: 6.5/10 → 8.5/10 after revisions (Risk: Medium-High → Medium with buffer)

**Peer Review Summary**: Plan is architecturally sound with clear TDD approach. Original estimates were optimistic but now realistic with 50% buffer. Missing test specifications added for complex algorithms. Cache versioning added to prevent corruption. Plan is ready for implementation.

---

## Document Cross-References

### Authoritative Architecture
- **[COMPILER_ARCHITECTURE.md](../architecture/COMPILER_ARCHITECTURE.md)** - Authoritative reference for all pipeline stages, data contracts, and validation requirements
  - §2.1: 7-Stage Pipeline Overview
  - §2.2: Stage Details (Parse → Emit)
  - §3.1: UniverseMetadata Contract (19 fields)
  - §3.2: UAC → BAC Contract (ObservationSpec)
  - §3.3: UAC → Training Contract (Rich Metadata)
  - §4.1: Cache Strategy (hash-based, MessagePack)
  - §4.2: Checkpoint Compatibility Validation

### Task Specifications
- **[TASK-004A-COMPILER-IMPLEMENTATION.md](../tasks/TASK-004A-COMPILER-IMPLEMENTATION.md)** - Complete implementation spec (this plan operationalizes it)
  - Phase 1: Infrastructure (CompilationError, base classes)
  - Phase 2: Symbol Tables (5 entity types)
  - Phase 3: Reference Resolution (cross-file validation)
  - Phase 4: Cross-Validation (semantic checks)
  - Phase 5: Metadata Computation (19 fields + ObservationSpec + Rich Metadata)
  - Phase 6: Optimization & CompiledUniverse
  - Phase 7: Caching (MessagePack serialization)
  - Phase 8: Integration (DemoRunner, checkpoint validation)

- **[TASK-003-UAC-CORE-DTOS.md](../tasks/TASK-003-UAC-CORE-DTOS.md)** - DTO definitions (dependency - must be complete)
  - TrainingConfig, EnvironmentConfig, PopulationConfig, CurriculumConfig
  - BarConfig, CascadeConfig, AffordanceConfig, CueConfig
  - HamletConfig (master DTO)
  - SubstrateConfig (existing), ActionConfig (existing)

- **[STREAM-001-UAC-BAC-FOUNDATION.md](../tasks/STREAM-001-UAC-BAC-FOUNDATION.md)** - Strategic context for compiler role in UAC/BAC architecture

### Related Architecture Documents
- **[UNIVERSE_AS_CODE.md](../architecture/UNIVERSE_AS_CODE.md)** - UAC philosophy, no-defaults principle, operator accountability
- **[BRAIN_AS_CODE.md](../architecture/BRAIN_AS_CODE.md)** - BAC architecture (depends on ObservationSpec from compiler)
- **[GLOSSARY.md](../architecture/GLOSSARY.md)** - Terminology reference

### Implementation Precedents
- **[2025-11-07-task-003-tdd-implementation-plan.md](2025-11-07-task-003-tdd-implementation-plan.md)** - TDD approach for TASK-003 (DTO layer)
- **[vfs-integration-guide.md](../vfs-integration-guide.md)** - VFS integration patterns (ObservationSpec builder)

### Config Schema Documentation
- **[docs/config-schemas/variables.md](../config-schemas/variables.md)** - VFS variables schema
- **[docs/config-schemas/enabled_actions.md](../config-schemas/enabled_actions.md)** - Action masking patterns

### Testing Infrastructure
- **[docs/testing/](../testing/)** - Testing patterns and fixtures

---

## Executive Summary

The Universe Compiler transforms 8 YAML config files into a single immutable `CompiledUniverse` artifact through a 7-stage compilation pipeline. This removes runtime validation overhead, enables checkpoint transfer validation, and provides rich metadata for training systems and custom neural encoders.

**Key Insight**: **Compilation as a First-Class Concept** - By treating universe configuration as a compilation problem (not just YAML loading), we gain:

1. **Early Error Detection**: Catch config errors at load time (not after 500 training episodes)
2. **Checkpoint Safety**: Config hash enables soft-warning validation during transfer learning
3. **Custom Encoders**: ObservationSpec enables BAC to build domain-specific neural architectures
4. **Training Integration**: Rich metadata enables per-meter logging, affordance usage tracking, action masking

**Implementation Strategy**: Test-Driven Development (TDD) with bottom-up construction:

1. **Data Structures First**: Define all DTOs with comprehensive tests
2. **Stage by Stage**: Implement each compilation stage with unit tests
3. **Pipeline Integration**: Wire stages together with integration tests
4. **Config Validation**: Test against all 6 reference configs (L0_0, L0_5, L1, L2, L3, templates)
5. **Backward Compatibility**: Ensure DemoRunner continues to work without changes

**Effort Breakdown** (includes 50% complexity buffer):
- Phase 0: Pre-implementation setup (1-2h)
- Phase 1: Infrastructure (4-6h)
- Phase 2: Symbol Tables (6-8h)
- Phase 3: RawConfigs & Stage 1 Parse (4-6h)
- Phase 4: Stage 2 Build Symbol Tables (4-6h)
- Phase 5: Stage 3 Resolve References (8-12h)
- Phase 6: Stage 4 Cross-Validate (10-14h) - includes circular cascade + cue validation tests
- Phase 7: Stage 5 Compute Metadata (12-16h) - includes ObservationSpec wrapper + rich metadata
- Phase 8: Stage 6/7 Optimize & Emit (6-10h)
- Phase 9: Caching with versioning (6-8h)
- Phase 10: Integration & Validation (8-12h)
- **Total**: 60-82 hours (50% buffer over original 40-55h estimate)

---

## Problem Statement

### Current Constraint

**File**: `src/townlet/demo/runner.py:100-150` (DemoRunner.__init__)

```python
# Current approach: Direct YAML loading with runtime validation
def __init__(self, config_dir: Path, ...):
    # Load configs individually
    bars = load_bars_config(config_dir / "bars.yaml")
    cascades = load_cascades_config(config_dir / "cascades.yaml")
    # ... more loading

    # Validation happens during environment creation (runtime)
    self.env = VectorizedHamletEnv(
        substrate=substrate,
        bars=bars,
        # ... if invalid, error during training!
    )
```

**Problems**:
1. **Late Error Detection**: Config errors discovered after training starts
2. **No Cross-Validation**: Can't check if affordance references valid meters
3. **No Checkpoint Validation**: Can't warn when transferring checkpoints to incompatible configs
4. **No Custom Encoders**: BAC can't introspect observation structure for domain-specific architectures
5. **Limited Metadata**: Training system can't log per-meter metrics or track affordance usage

**From COMPILER_ARCHITECTURE.md**:
> "The compiler is the authoritative source of truth for what constitutes a valid universe. Runtime systems (environment, training) trust that compiled universes are internally consistent."

### Why This Is Not Technical Debt

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: **More fragile** (this is necessary infrastructure)

- ✅ Enables early error detection (operator productivity)
- ✅ Enables checkpoint transfer validation (safety)
- ✅ Enables custom neural encoders (BAC - TASK-005)
- ✅ Enables substrate-agnostic visualization (TASK-006)
- ✅ Enables rich training metadata (logging, metrics)

**This is not debt removal - this is capability enablement.**

---

## Success Criteria

### Functional Requirements

✅ **F1: Parse all 8 config files**
- Load substrate.yaml, variables.yaml, bars.yaml, cascades.yaml, affordances.yaml, cues.yaml, actions.yaml, training.yaml
- Validate each against Pydantic schema
- Raise descriptive errors with file path context

✅ **F2: Build symbol tables**
- Register all variables, meters, affordances, cues, actions
- Enable lookup by ID
- Detect duplicate IDs within entity type

✅ **F3: Resolve cross-file references**
- Affordances reference meters (costs/effects)
- Cascades reference meters (source/target)
- Actions reference meters (costs)
- All references resolve to valid entities

✅ **F4: Cross-validate semantics**
- Cascades don't create circular dependencies
- Affordances have valid operating hours (0-23)
- Training params consistent with substrate (grid_size, vision_range)

✅ **F5: Compute metadata**
- UniverseMetadata with 19 fields
- ObservationSpec from VFS builder
- ActionSpaceMetadata, MeterMetadata, AffordanceMetadata

✅ **F6: Optimize runtime data**
- Pre-compute base depletion tensors
- Pre-compute cascade/modulation lookups
- Pre-compute time-based action masks [24, num_affordances]

✅ **F7: Emit immutable artifact**
- CompiledUniverse is frozen dataclass
- Contains all 8 configs + metadata + optimization data
- Validates internal consistency

✅ **F8: Cache compiled universes**
- Hash-based cache key (SHA-256 of all YAML contents)
- MessagePack serialization
- 10-100x faster subsequent loads

✅ **F9: Checkpoint compatibility validation**
- Check config_hash match (error if mismatch)
- Check action_count, observation_dim (error if mismatch)
- Soft warnings for dimension mismatches (allow transfer learning with warnings)

### Test Coverage Requirements

✅ **Unit Tests**: 90%+ coverage
- All data structures (DTOs, symbol tables, metadata)
- All compilation stages (parse, resolve, validate, compute, optimize, emit)
- Error handling (missing files, invalid refs, circular deps)

✅ **Integration Tests**: 100% reference config coverage
- L0_0_minimal: 3×3 grid, 1 affordance
- L0_5_dual_resource: 7×7 grid, 4 affordances
- L1_full_observability: 8×8 grid, 14 affordances, full obs
- L2_partial_observability: 8×8 grid, 14 affordances, POMDP
- L3_temporal_mechanics: 8×8 grid, 14 affordances, temporal
- Template configs: All substrate types (grid, grid3d, gridnd, continuous, aspatial)

✅ **Validation Tests**: Config pack consistency
- All config packs load without errors
- Observation dimensions match hardcoded expectations
- Action counts match substrate + custom actions

### Performance Requirements

✅ **P1: First compile ≤5s** (cold start, no cache)
✅ **P2: Cached compile ≤100ms** (MessagePack deserialization)
✅ **P3: No memory leaks** (dataclasses properly frozen)

---

## Phase 0: Pre-Implementation Setup (1-2 hours)

### Backup Strategy

```bash
# Backup existing configs
cp -r configs configs.backup-20251108

# Create feature branch
git checkout -b task-004a-universe-compiler

# Verify current configs load
uv run python -m townlet.demo.runner --config configs/L0_0_minimal --max-episodes 1
uv run python -m townlet.demo.runner --config configs/L1_full_observability --max-episodes 1
```

### Directory Structure

```bash
# Create new directories
mkdir -p src/townlet/universe
mkdir -p tests/test_townlet/unit/universe
mkdir -p tests/test_townlet/integration/universe

# Files to create (in order):
# src/townlet/universe/__init__.py
# src/townlet/universe/errors.py
# src/townlet/universe/symbol_table.py
# src/townlet/universe/metadata.py
# src/townlet/universe/optimization.py
# src/townlet/universe/rich_metadata.py
# src/townlet/universe/compiled.py
# src/townlet/universe/compiler.py
```

### Dependency Verification

**Critical**: Run comprehensive verification script BEFORE starting Phase 1

**File**: `scripts/verify_compiler_dependencies.py` (create this)

```python
"""Pre-implementation verification for TASK-004A Universe Compiler.

Verifies all dependencies are in place:
- TASK-003 DTOs exist and load correctly
- VFS ObservationSpecBuilder exists and API is correct
- All load functions exist with expected signatures
"""
import sys
from pathlib import Path


def verify_dto_files():
    """Verify all DTO files exist."""
    print("Verifying DTO files...")
    required_files = [
        "src/townlet/config/training.py",
        "src/townlet/config/environment.py",
        "src/townlet/config/population.py",
        "src/townlet/config/curriculum.py",
        "src/townlet/config/bar.py",
        "src/townlet/config/cascade.py",
        "src/townlet/config/affordance.py",
        "src/townlet/config/cue.py",
        "src/townlet/config/hamlet.py",
    ]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"  ❌ Missing: {file_path}")
            return False
    print("  ✅ All DTO files exist")
    return True


def verify_load_functions():
    """Verify all load functions exist and have correct signatures."""
    print("\nVerifying load functions...")

    try:
        from townlet.config.training import load_training_config
        from townlet.config.bar import load_bars_config
        from townlet.config.cascade import load_cascades_config
        from townlet.config.affordance import load_affordances_config
        from townlet.config.cue import load_cues_config
        from townlet.substrate.config import load_substrate_config
        from townlet.environment.action_config import load_action_space_config
        from townlet.vfs.schema import load_variables_config

        print("  ✅ All load functions imported successfully")

        # Verify signatures by inspecting
        test_config = Path("configs/L0_0_minimal")
        if test_config.exists():
            print(f"\n  Testing load functions with {test_config}...")
            bars = load_bars_config(test_config / "bars.yaml")
            print(f"    ✅ load_bars_config returned {type(bars).__name__}")

            substrate = load_substrate_config(test_config / "substrate.yaml")
            print(f"    ✅ load_substrate_config returned {type(substrate).__name__}")
        else:
            print(f"  ⚠️  Config {test_config} not found, skipping signature test")

        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def verify_vfs_builder():
    """Verify VFS ObservationSpecBuilder exists and API is correct."""
    print("\nVerifying VFS ObservationSpecBuilder...")

    try:
        from townlet.vfs.observation_builder import VFSObservationSpecBuilder
        from townlet.vfs.schema import ObservationField

        print("  ✅ VFSObservationSpecBuilder imported successfully")

        # Check API
        builder = VFSObservationSpecBuilder.__init__.__code__
        print(f"  ✅ VFSObservationSpecBuilder.__init__ exists")

        # Check if build_spec method exists
        if hasattr(VFSObservationSpecBuilder, 'build_spec'):
            print("  ✅ VFSObservationSpecBuilder.build_spec exists")
        elif hasattr(VFSObservationSpecBuilder, 'build_observation_spec'):
            print("  ✅ VFSObservationSpecBuilder.build_observation_spec exists")
        else:
            print("  ❌ VFSObservationSpecBuilder missing build method")
            return False

        print("  ✅ ObservationField class imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("TASK-004A Pre-Implementation Dependency Verification")
    print("=" * 60)

    results = {
        "DTO Files": verify_dto_files(),
        "Load Functions": verify_load_functions(),
        "VFS Builder": verify_vfs_builder(),
    }

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check}: {status}")

    if all(results.values()):
        print("\n🎉 All dependencies verified - ready for implementation!")
        sys.exit(0)
    else:
        print("\n⚠️  Some dependencies missing - fix before starting Phase 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Run Verification**:

```bash
# Create and run verification script
uv run python scripts/verify_compiler_dependencies.py

# Also run DTO tests
uv run pytest tests/test_townlet/unit/config/ -v
```

**Expected**: All checks pass ✅

**If verification fails**, STOP and fix dependencies before proceeding to Phase 1.

---

## Phase 1: Infrastructure (3-4 hours)

**Goal**: Create error handling, base classes, and compilation error collector.

### RED Phase (1 hour)

**Test**: `tests/test_townlet/unit/universe/test_errors.py`

```python
"""Test compilation error handling."""
import pytest
from townlet.universe.errors import (
    CompilationError,
    CompilationErrorCollector,
    format_compilation_error,
)


def test_compilation_error_creation():
    """Verify CompilationError captures stage and context."""
    error = CompilationError(
        stage="Stage 1: Parse",
        errors=["bars.yaml not found"],
        hints=["Check config directory path"]
    )

    assert error.stage == "Stage 1: Parse"
    assert "bars.yaml not found" in error.errors
    assert "Check config directory path" in error.hints


def test_error_collector_accumulates():
    """Verify ErrorCollector accumulates multiple errors."""
    collector = CompilationErrorCollector()

    collector.add_error("Missing affordance: Bed")
    collector.add_error("Missing affordance: Hospital")

    assert collector.has_errors()
    assert len(collector.errors) == 2


def test_error_collector_raises_on_check():
    """Verify check_and_raise raises if errors exist."""
    collector = CompilationErrorCollector()
    collector.add_error("Invalid meter reference")

    with pytest.raises(CompilationError) as exc_info:
        collector.check_and_raise("Stage 3: Resolve References")

    error = exc_info.value
    assert error.stage == "Stage 3: Resolve References"
    assert "Invalid meter reference" in error.errors


def test_error_collector_no_raise_when_empty():
    """Verify check_and_raise does nothing if no errors."""
    collector = CompilationErrorCollector()

    # Should not raise
    collector.check_and_raise("Stage 3: Resolve References")


def test_format_compilation_error():
    """Verify error formatting includes stage, file, context."""
    error = CompilationError(
        stage="Stage 1: Parse",
        errors=["bars.yaml not found in /path/to/config"],
        hints=["Verify config directory exists"],
        context="bars.yaml"
    )

    formatted = format_compilation_error(error)

    assert "Stage 1: Parse" in formatted
    assert "bars.yaml" in formatted
    assert "not found" in formatted
    assert "Verify config directory exists" in formatted
```

**Run**: `uv run pytest tests/test_townlet/unit/universe/test_errors.py`
**Expected**: ❌ FAIL (module doesn't exist yet)

### GREEN Phase (1.5 hours)

**Deliverable**: `src/townlet/universe/errors.py`

```python
"""Compilation error handling for Universe Compiler.

Per COMPILER_ARCHITECTURE.md §2.1: Errors accumulate and report with context.
"""
from dataclasses import dataclass, field


@dataclass
class CompilationError(Exception):
    """Exception raised during universe compilation.

    Attributes:
        stage: Compilation stage where error occurred (e.g., "Stage 1: Parse")
        errors: List of error messages
        hints: Optional hints for fixing the errors
        context: Optional file or entity context (e.g., "bars.yaml")
    """
    stage: str
    errors: list[str]
    hints: list[str] = field(default_factory=list)
    context: str | None = None

    def __str__(self) -> str:
        """Format error for display."""
        return format_compilation_error(self)


class CompilationErrorCollector:
    """Accumulates errors during compilation stages.

    Usage:
        errors = CompilationErrorCollector()
        errors.add_error("Missing meter: energy")
        errors.add_error("Missing meter: health")
        errors.check_and_raise("Stage 3: Resolve References")
    """

    def __init__(self):
        self.errors: list[str] = []
        self.hints: list[str] = []

    def add_error(self, message: str, hint: str | None = None):
        """Add an error message."""
        self.errors.append(message)
        if hint:
            self.hints.append(hint)

    def has_errors(self) -> bool:
        """Check if any errors accumulated."""
        return len(self.errors) > 0

    def check_and_raise(self, stage: str, context: str | None = None):
        """Raise CompilationError if errors exist."""
        if self.has_errors():
            raise CompilationError(
                stage=stage,
                errors=self.errors.copy(),
                hints=self.hints.copy(),
                context=context
            )

    def clear(self):
        """Clear accumulated errors."""
        self.errors.clear()
        self.hints.clear()


def format_compilation_error(error: CompilationError) -> str:
    """Format CompilationError for display.

    Per COMPILER_ARCHITECTURE.md §2.1: Errors include stage, context, hints.
    """
    lines = [f"\n❌ COMPILATION FAILED: {error.stage}"]

    if error.context:
        lines.append(f"   Context: {error.context}")

    lines.append("\nErrors:")
    for err in error.errors:
        lines.append(f"  • {err}")

    if error.hints:
        lines.append("\nHints:")
        for hint in error.hints:
            lines.append(f"  💡 {hint}")

    return "\n".join(lines)
```

**Run**: `uv run pytest tests/test_townlet/unit/universe/test_errors.py`
**Expected**: ✅ PASS

### REFACTOR Phase (30 minutes)

**Add**: `src/townlet/universe/__init__.py`

```python
"""Universe Compiler - Transforms YAML configs into immutable CompiledUniverse.

Per COMPILER_ARCHITECTURE.md: 7-stage compilation pipeline.

Architecture:
    Parse → Symbol Tables → Resolve → Validate → Compute → Optimize → Emit

See Also:
    - docs/architecture/COMPILER_ARCHITECTURE.md
    - docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md
"""
from townlet.universe.errors import (
    CompilationError,
    CompilationErrorCollector,
    format_compilation_error,
)

__all__ = [
    "CompilationError",
    "CompilationErrorCollector",
    "format_compilation_error",
]
```

**Verification**:

```bash
uv run pytest tests/test_townlet/unit/universe/test_errors.py -v
uv run python -c "from townlet.universe import CompilationError; print('✅ Imports work')"
```

### Success Criteria

- [x] CompilationError captures stage, errors, hints, context
- [x] CompilationErrorCollector accumulates errors
- [x] check_and_raise raises only when errors exist
- [x] Error formatting includes stage and context
- [x] All tests pass with 100% coverage

---

## Phase 2: Symbol Tables (4-5 hours)

**Goal**: Build central registry for all universe entities (variables, meters, affordances, cues, actions).

**Reference**: TASK-004A-COMPILER-IMPLEMENTATION.md §3 "Implement Stage 2 (Build Symbol Tables)"

### RED Phase (1.5 hours)

**Test**: `tests/test_townlet/unit/universe/test_symbol_table.py`

```python
"""Test symbol table for entity registration and lookup."""
import pytest
from townlet.universe.symbol_table import UniverseSymbolTable
from townlet.config.bar import BarConfig
from townlet.config.affordance import AffordanceConfig
from townlet.vfs.schema import VariableDef
from townlet.universe.errors import CompilationError


def test_symbol_table_initialization():
    """Verify symbol table initializes empty registries."""
    table = UniverseSymbolTable()

    assert len(table.variables) == 0
    assert len(table.meters) == 0
    assert len(table.affordances) == 0
    assert len(table.cues) == 0
    assert len(table.actions) == 0


def test_register_variable():
    """Verify variable registration."""
    table = UniverseSymbolTable()

    var = VariableDef(
        id="energy",
        scope="agent",
        type="scalar",
        lifetime="episode",
        readable_by=["agent", "engine"],
        writable_by=["engine"],
        default=1.0
    )

    table.register_variable(var)

    assert "energy" in table.variables
    assert table.variables["energy"] == var


def test_register_meter():
    """Verify meter registration."""
    table = UniverseSymbolTable()

    meter = BarConfig(
        name="energy",
        index=0,
        range=(0.0, 1.0),
        initial=1.0,
        base_depletion=0.001,
        critical=True,
        description="Energy level"
    )

    table.register_meter(meter)

    assert "energy" in table.meters
    assert table.meters["energy"] == meter


def test_register_duplicate_raises():
    """Verify duplicate registration raises error."""
    table = UniverseSymbolTable()

    meter1 = BarConfig(name="energy", index=0, range=(0.0, 1.0), ...)
    meter2 = BarConfig(name="energy", index=1, range=(0.0, 1.0), ...)

    table.register_meter(meter1)

    with pytest.raises(CompilationError, match="Duplicate meter"):
        table.register_meter(meter2)


def test_get_meter_by_name():
    """Verify retrieval by name."""
    table = UniverseSymbolTable()

    meter = BarConfig(name="energy", ...)
    table.register_meter(meter)

    assert table.get_meter("energy") == meter


def test_get_meter_missing_raises():
    """Verify missing meter raises KeyError."""
    table = UniverseSymbolTable()

    with pytest.raises(KeyError, match="energy"):
        table.get_meter("energy")
```

**Run**: `uv run pytest tests/test_townlet/unit/universe/test_symbol_table.py`
**Expected**: ❌ FAIL

### GREEN Phase (2 hours)

**Deliverable**: `src/townlet/universe/symbol_table.py`

```python
"""Symbol table for universe entity registration.

Per COMPILER_ARCHITECTURE.md §2.2: Central registry for all universe entities.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from townlet.config.bar import BarConfig
    from townlet.config.affordance import AffordanceConfig
    from townlet.config.cue import CueConfig
    from townlet.environment.action_config import ActionConfig
    from townlet.vfs.schema import VariableDef


class UniverseSymbolTable:
    """Central registry for all universe entities.

    Tracks 5 entity types:
    - Variables (VFS variable definitions)
    - Meters (subset of variables - bars)
    - Affordances (interactions)
    - Cues (Theory of Mind cue definitions)
    - Actions (global action vocabulary - substrate + custom)

    Per COMPILER_ARCHITECTURE.md §2.2: Enables cross-file reference resolution.
    """

    def __init__(self):
        """Initialize empty symbol table."""
        self.variables: dict[str, "VariableDef"] = {}
        self.meters: dict[str, "BarConfig"] = {}
        self.affordances: dict[str, "AffordanceConfig"] = {}
        self.cues: dict[str, "CueConfig"] = {}
        self.actions: dict[int, "ActionConfig"] = {}  # Keyed by action ID

    def register_variable(self, var_def: "VariableDef"):
        """Register a VFS variable definition."""
        if var_def.id in self.variables:
            raise ValueError(f"Variable '{var_def.id}' already registered")
        self.variables[var_def.id] = var_def

    def register_meter(self, meter_config: "BarConfig"):
        """Register a meter (bar)."""
        if meter_config.name in self.meters:
            raise ValueError(f"Meter '{meter_config.name}' already registered")
        self.meters[meter_config.name] = meter_config

    def register_affordance(self, aff_config: "AffordanceConfig"):
        """Register an affordance."""
        if aff_config.id in self.affordances:
            raise ValueError(f"Affordance '{aff_config.id}' already registered")
        self.affordances[aff_config.id] = aff_config

    def register_cue(self, cue_config: "CueConfig"):
        """Register a cue."""
        if cue_config.cue_id in self.cues:
            raise ValueError(f"Cue '{cue_config.cue_id}' already registered")
        self.cues[cue_config.cue_id] = cue_config

    def register_action(self, action_config: "ActionConfig"):
        """Register an action."""
        if action_config.id in self.actions:
            raise ValueError(f"Action ID {action_config.id} already registered")
        self.actions[action_config.id] = action_config

    def get_variable(self, var_id: str) -> "VariableDef":
        """Lookup variable by ID."""
        if var_id not in self.variables:
            raise KeyError(f"Variable '{var_id}' not found in symbol table")
        return self.variables[var_id]

    def get_meter(self, meter_name: str) -> "BarConfig":
        """Lookup meter by name."""
        if meter_name not in self.meters:
            raise KeyError(f"Meter '{meter_name}' not found in symbol table")
        return self.meters[meter_name]

    def get_action(self, action_id: int) -> "ActionConfig":
        """Lookup action by ID."""
        if action_id not in self.actions:
            raise KeyError(f"Action ID {action_id} not found in symbol table")
        return self.actions[action_id]
```

**Run**: `uv run pytest tests/test_townlet/unit/universe/test_symbol_table.py`
**Expected**: ✅ PASS

### REFACTOR Phase (30 minutes)

**Improve**: Add helper methods for batch operations

```python
# Add to UniverseSymbolTable class

def get_meter_names(self) -> list[str]:
    """Get all registered meter names (sorted by index)."""
    sorted_meters = sorted(self.meters.values(), key=lambda m: m.index)
    return [m.name for m in sorted_meters]

def get_meter_count(self) -> int:
    """Get total number of registered meters."""
    return len(self.meters)

def get_affordance_count(self) -> int:
    """Get total number of registered affordances."""
    return len(self.affordances)
```

**Test additions**:

```python
def test_get_meter_names_sorted():
    """Verify meter names returned in index order."""
    table = UniverseSymbolTable()

    table.register_meter(BarConfig(name="health", index=2, ...))
    table.register_meter(BarConfig(name="energy", index=0, ...))
    table.register_meter(BarConfig(name="mood", index=1, ...))

    names = table.get_meter_names()
    assert names == ["energy", "mood", "health"]  # Sorted by index
```

### Success Criteria

- [x] Symbol table tracks 5 entity types
- [x] Registration methods prevent duplicates
- [x] Lookup methods raise KeyError for missing entities
- [x] Helper methods provide counts and sorted lists
- [x] All tests pass with 100% coverage

---

## Phase 3: RawConfigs & Stage 1 (Parse) (3-4 hours)

**Goal**: Load all 8 YAML files into RawConfigs container with descriptive errors.

**Reference**: TASK-004A-COMPILER-IMPLEMENTATION.md §1.2 "Implement Stage 1 (Parse Individual Files)"

### RED Phase (1 hour)

**Test**: `tests/test_townlet/unit/universe/test_stage_1_parse.py`

```python
"""Test Stage 1: Parse individual config files."""
import pytest
from pathlib import Path
from townlet.universe.compiler import UniverseCompiler, RawConfigs
from townlet.universe.errors import CompilationError


@pytest.fixture
def compiler():
    """Create compiler instance."""
    return UniverseCompiler()


def test_parse_l0_minimal_config(compiler):
    """Verify parsing L0_0_minimal config pack."""
    config_dir = Path("configs/L0_0_minimal")

    raw = compiler._stage_1_parse_individual_files(config_dir)

    assert isinstance(raw, RawConfigs)
    assert raw.substrate is not None
    assert raw.variables is not None
    assert raw.bars is not None
    assert raw.cascades is not None
    assert raw.affordances is not None
    assert raw.cues is not None
    assert raw.global_actions is not None
    assert raw.training is not None


def test_parse_missing_file_raises(compiler, tmp_path):
    """Verify missing file raises descriptive error."""
    config_dir = tmp_path / "incomplete_config"
    config_dir.mkdir()

    # Create only substrate.yaml (missing others)
    (config_dir / "substrate.yaml").write_text("version: '1.0'\ntype: grid\ngrid: {width: 8, height: 8}")

    with pytest.raises(CompilationError) as exc_info:
        compiler._stage_1_parse_individual_files(config_dir)

    error = exc_info.value
    assert error.stage == "Stage 1: Parse"
    assert "bars.yaml" in str(error) or "variables.yaml" in str(error)


def test_parse_invalid_yaml_raises(compiler, tmp_path):
    """Verify invalid YAML raises descriptive error."""
    config_dir = tmp_path / "invalid_config"
    config_dir.mkdir()

    # Create all required files
    for filename in ["substrate.yaml", "variables.yaml", "bars.yaml", "cascades.yaml",
                     "affordances.yaml", "cues.yaml", "actions.yaml", "training.yaml"]:
        (config_dir / filename).write_text("valid: yaml")

    # Corrupt bars.yaml
    (config_dir / "bars.yaml").write_text("invalid: yaml: : broken")

    with pytest.raises(CompilationError) as exc_info:
        compiler._stage_1_parse_individual_files(config_dir)

    error = exc_info.value
    assert error.stage == "Stage 1: Parse"
    assert "bars.yaml" in error.context or "bars.yaml" in str(error)
```

**Run**: `uv run pytest tests/test_townlet/unit/universe/test_stage_1_parse.py`
**Expected**: ❌ FAIL

### GREEN Phase (2 hours)

**Deliverable**: `src/townlet/universe/compiler.py` (partial - Stage 1 only)

```python
"""Universe Compiler - Main compilation pipeline.

Per COMPILER_ARCHITECTURE.md §2.1: 7-stage pipeline implementation.
"""
from dataclasses import dataclass
from pathlib import Path

from townlet.substrate.config import SubstrateConfig, load_substrate_config
from townlet.vfs.schema import VariablesConfig, load_variables_config
from townlet.config.bar import BarsConfig, load_bars_config
from townlet.config.cascade import CascadesConfig, load_cascades_config
from townlet.config.affordance import AffordanceConfigCollection, load_affordances_config
from townlet.config.cue import CuesConfig, load_cues_config
from townlet.environment.action_config import ActionSpaceConfig, load_action_space_config
from townlet.config.training import TrainingConfig, load_training_config

from townlet.universe.errors import CompilationError, CompilationErrorCollector
from townlet.universe.symbol_table import UniverseSymbolTable


@dataclass
class RawConfigs:
    """Container for raw config objects loaded from YAML.

    Per COMPILER_ARCHITECTURE.md §2.1: All core universe configs.
    """
    substrate: SubstrateConfig
    variables: VariablesConfig
    bars: BarsConfig
    cascades: CascadesConfig
    affordances: AffordanceConfigCollection
    cues: CuesConfig
    actions: ActionSpaceConfig
    training: TrainingConfig


class UniverseCompiler:
    """Compiles YAML configs into immutable CompiledUniverse.

    Per COMPILER_ARCHITECTURE.md §2.1: 7-stage compilation pipeline.

    Stages:
        1. Parse individual files
        2. Build symbol tables
        3. Resolve references
        4. Cross-validate
        5. Compute metadata
        6. Optimize (pre-compute)
        7. Emit compiled universe

    See Also:
        - docs/architecture/COMPILER_ARCHITECTURE.md
        - docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md
    """

    def __init__(self):
        """Initialize compiler."""
        pass

    def _stage_1_parse_individual_files(self, config_dir: Path) -> RawConfigs:
        """Stage 1: Load and validate individual YAML files.

        Per COMPILER_ARCHITECTURE.md §2.2 Stage 1: Parse.

        Validates:
        - File exists
        - YAML is well-formed
        - Pydantic schema is valid

        Does NOT validate cross-file references (Stage 3).

        Args:
            config_dir: Directory containing YAML config files

        Returns:
            RawConfigs container with all loaded configs

        Raises:
            CompilationError: If any file missing or invalid
        """
        errors = CompilationErrorCollector()

        # Track loaded configs
        substrate = None
        variables = None
        bars = None
        cascades = None
        affordances = None
        cues = None
        actions = None
        training = None

        # Load substrate.yaml
        try:
            substrate = load_substrate_config(config_dir / "substrate.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"substrate.yaml not found in {config_dir}",
                hint="Create substrate.yaml defining spatial structure"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse substrate.yaml: {e}")

        # Load variables.yaml
        try:
            variables = load_variables_config(config_dir / "variables.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"variables.yaml not found in {config_dir}",
                hint="Create variables.yaml defining VFS variables"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse variables.yaml: {e}")

        # Load bars.yaml
        try:
            bars = load_bars_config(config_dir / "bars.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"bars.yaml not found in {config_dir}",
                hint="Create bars.yaml defining meters"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse bars.yaml: {e}")

        # Load cascades.yaml
        try:
            cascades = load_cascades_config(config_dir / "cascades.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"cascades.yaml not found in {config_dir}",
                hint="Create cascades.yaml defining meter relationships"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse cascades.yaml: {e}")

        # Load affordances.yaml
        try:
            affordances = load_affordances_config(config_dir / "affordances.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"affordances.yaml not found in {config_dir}",
                hint="Create affordances.yaml defining interactions"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse affordances.yaml: {e}")

        # Load cues.yaml
        try:
            cues = load_cues_config(config_dir / "cues.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"cues.yaml not found in {config_dir}",
                hint="Create cues.yaml defining Theory of Mind cues"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse cues.yaml: {e}")

        # Load actions.yaml
        try:
            actions = load_action_space_config(config_dir / "actions.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"actions.yaml not found in {config_dir}",
                hint="Create actions.yaml defining global action vocabulary"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse actions.yaml: {e}")

        # Load training.yaml
        try:
            training = load_training_config(config_dir / "training.yaml")
        except FileNotFoundError:
            errors.add_error(
                f"training.yaml not found in {config_dir}",
                hint="Create training.yaml defining hyperparameters"
            )
        except Exception as e:
            errors.add_error(f"Failed to parse training.yaml: {e}")

        # Check for errors before proceeding
        errors.check_and_raise("Stage 1: Parse", context=str(config_dir))

        # All configs loaded successfully
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

**Run**: `uv run pytest tests/test_townlet/unit/universe/test_stage_1_parse.py`
**Expected**: ✅ PASS (if TASK-003 DTOs complete and load functions exist)

### Success Criteria

- [x] Parse all 8 config files
- [x] Raise CompilationError with file context for missing files
- [x] Raise CompilationError for malformed YAML
- [x] Raise CompilationError for Pydantic validation failures
- [x] Tests pass for L0_0_minimal config

---

## Phases 4-10: Continued Implementation

Due to length constraints, the remaining phases follow the same TDD pattern:

### Phase 4: Stage 2 (Build Symbol Tables) (3-4 hours)
- RED: Test symbol table population from RawConfigs
- GREEN: Implement `_stage_2_build_symbol_tables()`
- REFACTOR: Optimize duplicate detection

### Phase 5: Stage 3 (Resolve References) (5-7 hours)
- RED: Test cross-file reference resolution
- GREEN: Implement `_stage_3_resolve_references()`
- REFACTOR: Add reference graph visualization

### Phase 6: Stage 4 (Cross-Validate) (10-14 hours)
- RED: Test semantic validation (circular deps, operating hours, etc.)
- **RED: Comprehensive circular cascade test suite** (self-loops, complex cycles, disconnected graphs) **[NEW]**
- **RED: Comprehensive cue range validation test suite** (gaps, overlaps, boundaries) **[NEW]**
- GREEN: Implement `_stage_4_cross_validate()`
- REFACTOR: Extract validators into reusable functions

### Phase 7: Stage 5 (Compute Metadata) (12-16 hours)
- RED: Test UniverseMetadata computation (19 fields)
- **RED: Test ObservationSpec wrapper class** (total_dims property, query methods) **[NEW]**
- RED: Test ObservationSpec generation from VFS
- RED: Test Rich Metadata structures (ActionSpaceMetadata, MeterMetadata, AffordanceMetadata)
- GREEN: Implement `_stage_5_compute_metadata()` and `_stage_5_build_rich_metadata()`
- REFACTOR: Extract config hash computation

### Phase 8: Stage 6 (Optimize) & CompiledUniverse (6-10 hours)
- RED: Test OptimizationData pre-computation
- RED: Test CompiledUniverse immutability
- GREEN: Implement `_stage_6_optimize()` and `_stage_7_emit_compiled_universe()`
- REFACTOR: Add CompiledUniverse helper methods

### Phase 9: Caching with Versioning (6-8 hours)
- RED: Test hash-based cache key generation
- **RED: Test cache versioning and invalidation** (version mismatch, schema changes) **[NEW]**
- RED: Test MessagePack serialization/deserialization
- **RED: Test cache corruption recovery** (corrupted files, truncated data) **[NEW]**
- GREEN: Implement caching layer with `cache_version` field
- REFACTOR: Add cache invalidation logic

### Phase 10: Integration & Validation (8-12 hours)
- RED: Test main `compile()` pipeline
- RED: Test all 6 reference configs
- **RED: Dimension regression tests** (explicit validation of observation_dim) **[NEW]**
- GREEN: Wire all stages together
- REFACTOR: Add performance benchmarks

---

## Testing Strategy

### Unit Tests (90%+ coverage)

**Test Files**:
- `test_errors.py` - Error handling
- `test_symbol_table.py` - Entity registration
- `test_stage_1_parse.py` - YAML loading
- `test_stage_2_symbols.py` - Symbol table building
- `test_stage_3_resolve.py` - Reference resolution
- `test_stage_4_validate.py` - Cross-validation
- `test_stage_5_metadata.py` - Metadata computation
- `test_stage_6_optimize.py` - Optimization data
- `test_compiled_universe.py` - Immutability and helpers
- `test_cache.py` - Hash-based caching

### Integration Tests (100% config coverage)

**Test File**: `tests/test_townlet/integration/universe/test_compiler_integration.py`

```python
"""Integration tests for full compilation pipeline."""
import pytest
from pathlib import Path
from townlet.universe.compiler import UniverseCompiler


@pytest.mark.parametrize("config_name", [
    "L0_0_minimal",
    "L0_5_dual_resource",
    "L1_full_observability",
    "L2_partial_observability",
    "L3_temporal_mechanics",
])
def test_compile_reference_config(config_name):
    """Verify all reference configs compile successfully."""
    compiler = UniverseCompiler()
    config_dir = Path(f"configs/{config_name}")

    universe = compiler.compile(config_dir, use_cache=False)

    # Verify immutability
    assert universe is not None
    with pytest.raises(Exception):  # FrozenInstanceError
        universe.metadata.meter_count = 999

    # Verify metadata
    assert universe.metadata.universe_name == config_name
    assert universe.metadata.meter_count > 0
    assert universe.metadata.action_count > 0
    assert universe.metadata.observation_dim > 0

    # Verify rich metadata
    assert len(universe.action_space_metadata.actions) == universe.metadata.action_count
    assert len(universe.meter_metadata.meters) == universe.metadata.meter_count

    # Verify helper methods work
    env = universe.create_environment(num_agents=1, device="cpu")
    assert env is not None
```

### Validation Tests

**Regression Tests**: Ensure observation dimensions match current system

```python
def test_observation_dimensions_match_current():
    """Verify compiled observation dims match hardcoded expectations."""
    compiler = UniverseCompiler()

    test_cases = {
        "L0_0_minimal": 29,  # 2 pos + 8 meters + 15 aff + 4 temporal
        "L0_5_dual_resource": 29,
        "L1_full_observability": 29,
        "L2_partial_observability": 54,  # 25 local + 2 pos + 8 meters + 15 aff + 4 temporal
        "L3_temporal_mechanics": 29,
    }

    for config_name, expected_dim in test_cases.items():
        universe = compiler.compile(Path(f"configs/{config_name}"))
        assert universe.metadata.observation_dim == expected_dim, \
            f"{config_name}: Expected {expected_dim}, got {universe.metadata.observation_dim}"
```

### Critical Test Specifications (Added Post-Review)

**NEW**: These comprehensive test suites address peer review findings and must be implemented BEFORE corresponding GREEN phases.

#### 1. Circular Cascade Detection Test Suite (Phase 6)

**File**: `tests/test_townlet/unit/universe/test_circular_cascades.py`

```python
"""Comprehensive test suite for circular cascade detection.

Tests DFS-based cycle detection for all edge cases.
"""
import pytest
from townlet.universe.compiler import UniverseCompiler
from townlet.config.cascade import CascadeConfig, ModulationConfig


def test_self_loop_cascade():
    """Verify detection of self-loop (A → A)."""
    cascades = CascadesConfig(
        cascades=[],
        modulations=[ModulationConfig(source="energy", target="energy", ...)]
    )

    compiler = UniverseCompiler()
    errors = CompilationErrorCollector()

    compiler._validate_no_circular_cascades(cascades, symbol_table, errors)

    assert errors.has_errors()
    assert "circular" in str(errors.errors[0]).lower()
    assert "energy → energy" in str(errors.errors[0])


def test_simple_cycle():
    """Verify detection of simple cycle (A → B → A)."""
    cascades = CascadesConfig(
        cascades=[],
        modulations=[
            ModulationConfig(source="energy", target="mood", ...),
            ModulationConfig(source="mood", target="energy", ...),
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_no_circular_cascades(cascades, symbol_table, errors)

    assert errors.has_errors()
    assert "energy → mood → energy" in str(errors.errors[0]) or \
           "mood → energy → mood" in str(errors.errors[0])


def test_complex_cycle():
    """Verify detection of complex cycle (A → B → C → D → A)."""
    cascades = CascadesConfig(
        cascades=[],
        modulations=[
            ModulationConfig(source="energy", target="mood", ...),
            ModulationConfig(source="mood", target="health", ...),
            ModulationConfig(source="health", target="satiation", ...),
            ModulationConfig(source="satiation", target="energy", ...),
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_no_circular_cascades(cascades, symbol_table, errors)

    assert errors.has_errors()
    assert "circular" in str(errors.errors[0]).lower()


def test_multiple_disconnected_cycles():
    """Verify detection of multiple independent cycles."""
    cascades = CascadesConfig(
        cascades=[],
        modulations=[
            # Cycle 1: energy → mood → energy
            ModulationConfig(source="energy", target="mood", ...),
            ModulationConfig(source="mood", target="energy", ...),
            # Cycle 2: health → satiation → health
            ModulationConfig(source="health", target="satiation", ...),
            ModulationConfig(source="satiation", target="health", ...),
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_no_circular_cascades(cascades, symbol_table, errors)

    assert errors.has_errors()
    assert len(errors.errors) == 2  # Both cycles detected


def test_acyclic_with_shared_nodes():
    """Verify acyclic graph accepted (A → C ← B)."""
    cascades = CascadesConfig(
        cascades=[],
        modulations=[
            ModulationConfig(source="energy", target="mood", ...),
            ModulationConfig(source="health", target="mood", ...),  # Both → mood
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_no_circular_cascades(cascades, symbol_table, errors)

    assert not errors.has_errors()  # Acyclic - should pass


def test_empty_cascade_graph():
    """Verify empty graph accepted."""
    cascades = CascadesConfig(cascades=[], modulations=[])

    errors = CompilationErrorCollector()
    compiler._validate_no_circular_cascades(cascades, symbol_table, errors)

    assert not errors.has_errors()
```

#### 2. Cue Range Validation Test Suite (Phase 6)

**File**: `tests/test_townlet/unit/universe/test_cue_range_validation.py`

```python
"""Comprehensive test suite for cue range validation.

Tests mathematical validation of meter ranges: coverage, gaps, overlaps.
"""
import pytest
from townlet.universe.compiler import UniverseCompiler
from townlet.config.cue import CueConfig


def test_full_coverage_accepted():
    """Verify full coverage [0.0, 1.0] accepted."""
    cues = CuesConfig(
        cues=[
            CueConfig(name="Low", meter="energy", range=(0.0, 0.5), ...),
            CueConfig(name="High", meter="energy", range=(0.5, 1.0), ...),
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_cue_ranges(cues, symbol_table, errors)

    assert not errors.has_errors()


def test_gap_detected():
    """Verify gap [0.5, 0.6] detected."""
    cues = CuesConfig(
        cues=[
            CueConfig(name="Low", meter="energy", range=(0.0, 0.5), ...),
            CueConfig(name="High", meter="energy", range=(0.6, 1.0), ...),  # Gap!
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_cue_ranges(cues, symbol_table, errors)

    assert errors.has_errors()
    assert "gap" in str(errors.errors[0]).lower()
    assert "0.5" in str(errors.errors[0]) and "0.6" in str(errors.errors[0])


def test_overlap_detected():
    """Verify overlap [0.4, 0.5] detected."""
    cues = CuesConfig(
        cues=[
            CueConfig(name="Low", meter="energy", range=(0.0, 0.5), ...),
            CueConfig(name="Med", meter="energy", range=(0.4, 0.8), ...),  # Overlaps Low!
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_cue_ranges(cues, symbol_table, errors)

    assert errors.has_errors()
    assert "overlap" in str(errors.errors[0]).lower()


def test_floating_point_boundary_accepted():
    """Verify floating point boundaries handled correctly."""
    cues = CuesConfig(
        cues=[
            CueConfig(name="Low", meter="energy", range=(0.0, 0.33333), ...),
            CueConfig(name="Med", meter="energy", range=(0.33333, 0.66667), ...),
            CueConfig(name="High", meter="energy", range=(0.66667, 1.0), ...),
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_cue_ranges(cues, symbol_table, errors)

    # Should accept despite floating point (boundaries match within epsilon)
    assert not errors.has_errors()


def test_single_range_full_domain():
    """Verify single range covering full domain accepted."""
    cues = CuesConfig(
        cues=[CueConfig(name="All", meter="energy", range=(0.0, 1.0), ...)]
    )

    errors = CompilationErrorCollector()
    compiler._validate_cue_ranges(cues, symbol_table, errors)

    assert not errors.has_errors()


def test_empty_cue_list_accepted():
    """Verify empty cue list accepted (no cues defined)."""
    cues = CuesConfig(cues=[])

    errors = CompilationErrorCollector()
    compiler._validate_cue_ranges(cues, symbol_table, errors)

    assert not errors.has_errors()


def test_unsorted_ranges_sorted_automatically():
    """Verify unsorted ranges handled correctly."""
    cues = CuesConfig(
        cues=[
            CueConfig(name="High", meter="energy", range=(0.5, 1.0), ...),  # Out of order
            CueConfig(name="Low", meter="energy", range=(0.0, 0.5), ...),
        ]
    )

    errors = CompilationErrorCollector()
    compiler._validate_cue_ranges(cues, symbol_table, errors)

    # Should sort internally and validate correctly
    assert not errors.has_errors()
```

#### 3. ObservationSpec Wrapper Class Specification (Phase 7)

**File**: `src/townlet/universe/observation_spec.py` (NEW)

```python
"""ObservationSpec wrapper class for VFS integration.

Provides query interface over list[ObservationField] from VFSObservationSpecBuilder.
"""
from dataclasses import dataclass
from typing import Literal

from townlet.vfs.schema import ObservationField


@dataclass(frozen=True)
class ObservationSpec:
    """Immutable observation specification with query methods.

    Wraps list[ObservationField] from VFSObservationSpecBuilder with:
    - Computed total_dims property
    - Query methods for field lookup
    - Semantic type filtering

    Per COMPILER_ARCHITECTURE.md §3.2: UAC → BAC data contract.
    """

    fields: list[ObservationField]

    @property
    def total_dims(self) -> int:
        """Total observation dimensions (sum of all field dims)."""
        return sum(field.shape[0] if field.shape else 1 for field in self.fields)

    def get_field_by_name(self, name: str) -> ObservationField:
        """Lookup observation field by name.

        Args:
            name: Field name (e.g., "position", "energy", "affordance_at_position")

        Returns:
            ObservationField

        Raises:
            KeyError: If field not found
        """
        for field in self.fields:
            if field.id == name:
                return field
        raise KeyError(f"Observation field '{name}' not found")

    def get_fields_by_semantic_type(self, semantic_type: str) -> list[ObservationField]:
        """Get all fields with specific semantic type.

        Args:
            semantic_type: Semantic type (e.g., "position", "meter", "affordance", "temporal")

        Returns:
            List of matching fields (may be empty)
        """
        # Assuming ObservationField has semantic_type attribute
        # (may need to add this to VFS schema)
        return [f for f in self.fields if f.semantic_type == semantic_type]

    def get_start_index(self, field_name: str) -> int:
        """Get start index of field in observation vector.

        Args:
            field_name: Field name

        Returns:
            Start index (0-based)

        Raises:
            KeyError: If field not found
        """
        field = self.get_field_by_name(field_name)
        return field.start_idx

    def get_end_index(self, field_name: str) -> int:
        """Get end index of field in observation vector (exclusive).

        Args:
            field_name: Field name

        Returns:
            End index (exclusive)

        Raises:
            KeyError: If field not found
        """
        field = self.get_field_by_name(field_name)
        return field.end_idx
```

**Test File**: `tests/test_townlet/unit/universe/test_observation_spec.py`

```python
"""Test ObservationSpec wrapper class."""
import pytest
from townlet.universe.observation_spec import ObservationSpec
from townlet.vfs.schema import ObservationField


def test_total_dims_computed():
    """Verify total_dims sums all field dimensions."""
    spec = ObservationSpec(fields=[
        ObservationField(id="position", shape=(2,), start_idx=0, end_idx=2, ...),
        ObservationField(id="meters", shape=(8,), start_idx=2, end_idx=10, ...),
    ])

    assert spec.total_dims == 10  # 2 + 8


def test_get_field_by_name():
    """Verify field lookup by name."""
    position_field = ObservationField(id="position", shape=(2,), ...)
    spec = ObservationSpec(fields=[position_field])

    assert spec.get_field_by_name("position") == position_field


def test_get_field_by_name_missing_raises():
    """Verify missing field raises KeyError."""
    spec = ObservationSpec(fields=[])

    with pytest.raises(KeyError, match="invalid"):
        spec.get_field_by_name("invalid")
```

#### 4. Cache Versioning Test Suite (Phase 9)

**File**: `tests/test_townlet/unit/universe/test_cache_versioning.py`

```python
"""Test cache versioning and invalidation."""
import pytest
from pathlib import Path


def test_cache_with_version():
    """Verify cache includes version field."""
    compiler = UniverseCompiler()
    universe = compiler.compile(config_dir, use_cache=False)

    cache_path = compiler._get_cache_path(config_dir)
    assert cache_path.exists()

    import msgpack
    with open(cache_path, "rb") as f:
        cached = msgpack.unpackb(f.read())

    assert "cache_version" in cached
    assert cached["cache_version"] == "1.0"


def test_cache_version_mismatch_invalidates():
    """Verify version mismatch triggers recompile."""
    compiler = UniverseCompiler()

    # First compile (creates cache v1.0)
    universe1 = compiler.compile(config_dir, use_cache=False)

    # Manually corrupt cache version
    cache_path = compiler._get_cache_path(config_dir)
    import msgpack
    with open(cache_path, "rb") as f:
        cached = msgpack.unpackb(f.read())

    cached["cache_version"] = "0.9"  # Old version

    with open(cache_path, "wb") as f:
        f.write(msgpack.packb(cached))

    # Second compile should detect version mismatch and recompile
    universe2 = compiler.compile(config_dir, use_cache=True)

    assert universe2 is not None  # Recompiled successfully


def test_cache_corruption_recovers():
    """Verify corrupted cache falls back to recompile."""
    compiler = UniverseCompiler()

    # Create valid cache
    universe1 = compiler.compile(config_dir, use_cache=False)

    # Corrupt cache file
    cache_path = compiler._get_cache_path(config_dir)
    cache_path.write_bytes(b"corrupted data!!!")

    # Should recover by recompiling
    universe2 = compiler.compile(config_dir, use_cache=True)

    assert universe2 is not None
    assert universe2.metadata.universe_name == config_dir.name
```

#### 5. Dimension Regression Test Suite (Phase 10)

**File**: `tests/test_townlet/integration/universe/test_dimension_regression.py`

```python
"""Regression tests for observation dimensions.

Ensures compiled dimensions match current hardcoded expectations.
Prevents checkpoint incompatibility from accidental dimension changes.
"""
import pytest
from pathlib import Path
from townlet.universe.compiler import UniverseCompiler


@pytest.mark.parametrize("config_name,expected_dim,expected_actions", [
    ("L0_0_minimal", 29, 8),
    ("L0_5_dual_resource", 29, 8),
    ("L1_full_observability", 29, 8),
    ("L2_partial_observability", 54, 8),  # POMDP: 25 local + 2 pos + 8 meters + 15 aff + 4 temporal
    ("L3_temporal_mechanics", 29, 8),
])
def test_dimension_regression(config_name, expected_dim, expected_actions):
    """Verify observation_dim and action_count match expectations."""
    compiler = UniverseCompiler()
    config_dir = Path(f"configs/{config_name}")

    universe = compiler.compile(config_dir, use_cache=False)

    # Critical: These dimensions MUST NOT change (breaks checkpoints)
    assert universe.metadata.observation_dim == expected_dim, \
        f"{config_name}: observation_dim changed! Expected {expected_dim}, got {universe.metadata.observation_dim}"

    assert universe.metadata.action_count == expected_actions, \
        f"{config_name}: action_count changed! Expected {expected_actions}, got {universe.metadata.action_count}"


def test_dimension_breakdown_l1():
    """Verify L1 observation dimension breakdown."""
    compiler = UniverseCompiler()
    universe = compiler.compile(Path("configs/L1_full_observability"))

    # L1 breakdown: 2 pos + 8 meters + 15 affordance + 4 temporal = 29
    obs_spec = universe.observation_spec

    position_dims = sum(f.shape[0] for f in obs_spec.get_fields_by_semantic_type("position"))
    meter_dims = sum(f.shape[0] for f in obs_spec.get_fields_by_semantic_type("meter"))
    affordance_dims = sum(f.shape[0] for f in obs_spec.get_fields_by_semantic_type("affordance"))
    temporal_dims = sum(f.shape[0] for f in obs_spec.get_fields_by_semantic_type("temporal"))

    assert position_dims == 2
    assert meter_dims == 8
    assert affordance_dims == 15
    assert temporal_dims == 4
    assert position_dims + meter_dims + affordance_dims + temporal_dims == 29
```

---

## Migration & Rollback

### Migration Strategy

**Phase 1**: Additive (no breaking changes)

```python
# DemoRunner continues to work
runner = DemoRunner(config_dir="configs/L0_0_minimal", ...)
runner.run()
```

**Phase 2**: Opt-in compilation

```python
# New API (optional)
from townlet.universe.compiler import UniverseCompiler

compiler = UniverseCompiler()
universe = compiler.compile(Path("configs/L0_0_minimal"))

# Use compiled universe for environment creation
env = universe.create_environment(num_agents=100, device="cuda")
```

**Phase 3**: Full integration

```python
# DemoRunner uses compiler internally
runner = DemoRunner(config_dir="configs/L0_0_minimal", use_compiler=True)
```

### Rollback Plan

If critical bugs discovered:

1. **Disable compiler**: Add `use_compiler=False` flag to DemoRunner
2. **Revert to YAML loading**: DemoRunner falls back to direct YAML loading
3. **Fix forward**: Address bugs in compiler, re-enable

**No data loss** - compiled universes are deterministic from YAML configs.

---

## Performance Benchmarks

### Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| First compile (cold) | ≤5s | `time compiler.compile(config_dir, use_cache=False)` |
| Cached compile | ≤100ms | `time compiler.compile(config_dir, use_cache=True)` |
| Memory overhead | <50MB | Compiled universe size in RAM |
| Cache file size | <10MB | `.msgpack` file size on disk |

### Benchmark Script

**File**: `scripts/benchmark_compiler.py`

```python
"""Benchmark compilation performance."""
import time
from pathlib import Path
from townlet.universe.compiler import UniverseCompiler

def benchmark_config(config_name: str):
    """Benchmark single config compilation."""
    compiler = UniverseCompiler()
    config_dir = Path(f"configs/{config_name}")

    # Cold compile
    start = time.time()
    universe = compiler.compile(config_dir, use_cache=False)
    cold_time = time.time() - start

    # Warm compile (cached)
    start = time.time()
    universe = compiler.compile(config_dir, use_cache=True)
    warm_time = time.time() - start

    print(f"{config_name}:")
    print(f"  Cold: {cold_time*1000:.1f}ms")
    print(f"  Warm: {warm_time*1000:.1f}ms")
    print(f"  Speedup: {cold_time/warm_time:.1f}x")

if __name__ == "__main__":
    for config in ["L0_0_minimal", "L1_full_observability", "L3_temporal_mechanics"]:
        benchmark_config(config)
```

---

## Documentation Updates

### Files to Update

1. **CLAUDE.md**: Add compiler usage section
2. **UNIVERSE_AS_CODE.md**: Document compilation workflow
3. **README.md**: Update quick start with compiler
4. **docs/guides/config-development.md**: Add compilation validation step

### New Documentation

1. **docs/guides/compiler-usage.md**: Operator guide for using compiler
2. **docs/architecture/COMPILER_INTERNALS.md**: Deep dive on compilation stages
3. **docs/testing/compiler-testing.md**: Testing strategy and fixtures

---

## Risk Assessment

### High Risks

**R1: Circular cascade dependencies**
- **Mitigation**: Topological sort in Stage 4 validation
- **Fallback**: Explicit cycle detection with clear error messages

**R2: Backward compatibility with legacy checkpoints**
- **Mitigation**: Use `.get()` for all checkpoint field access
- **Fallback**: Legacy checkpoint mode with soft warnings

**R3: Cache invalidation bugs**
- **Mitigation**: Hash-based cache key (SHA-256 of all YAML contents)
- **Fallback**: Cache versioning with automatic invalidation on schema change

### Medium Risks

**R4: Performance regression**
- **Mitigation**: Benchmark suite with performance gates
- **Fallback**: Async compilation in background thread

**R5: Error message clarity**
- **Mitigation**: Rich error formatting with file context and hints
- **Fallback**: User feedback iteration

---

## Success Metrics

### Functional Metrics

- ✅ All 6 reference configs compile successfully
- ✅ 90%+ unit test coverage
- ✅ 100% integration test coverage
- ✅ Zero compilation errors for valid configs
- ✅ Descriptive errors for invalid configs

### Performance Metrics

- ✅ Cold compile ≤5s (L1_full_observability)
- ✅ Warm compile ≤100ms (cache hit)
- ✅ 10-100x speedup from caching

### Quality Metrics

- ✅ Zero regressions in existing tests
- ✅ DemoRunner continues to work without changes
- ✅ Checkpoint transfer validation works correctly
- ✅ ObservationSpec enables custom encoders (validates TASK-005 dependency)

---

## Appendix: File Creation Order

### Phase 1: Infrastructure
1. `src/townlet/universe/__init__.py`
2. `src/townlet/universe/errors.py`
3. `tests/test_townlet/unit/universe/test_errors.py`

### Phase 2: Symbol Tables
4. `src/townlet/universe/symbol_table.py`
5. `tests/test_townlet/unit/universe/test_symbol_table.py`

### Phase 3: Stage 1 (Parse)
6. `src/townlet/universe/compiler.py` (RawConfigs + Stage 1)
7. `tests/test_townlet/unit/universe/test_stage_1_parse.py`

### Phase 4: Stage 2 (Symbol Tables)
8. Update `compiler.py` (add `_stage_2_build_symbol_tables`)
9. `tests/test_townlet/unit/universe/test_stage_2_symbols.py`

### Phase 5: Stage 3 (Resolve)
10. Update `compiler.py` (add `_stage_3_resolve_references`)
11. `tests/test_townlet/unit/universe/test_stage_3_resolve.py`

### Phase 6: Stage 4 (Validate)
12. Update `compiler.py` (add `_stage_4_cross_validate`)
13. `tests/test_townlet/unit/universe/test_stage_4_validate.py`

### Phase 7: Stage 5 (Metadata)
14. `src/townlet/universe/metadata.py`
15. `src/townlet/universe/rich_metadata.py`
16. Update `compiler.py` (add `_stage_5_compute_metadata`, `_stage_5_build_rich_metadata`)
17. `tests/test_townlet/unit/universe/test_stage_5_metadata.py`

### Phase 8: Stage 6 & 7 (Optimize & Emit)
18. `src/townlet/universe/optimization.py`
19. `src/townlet/universe/compiled.py`
20. Update `compiler.py` (add `_stage_6_optimize`, `_stage_7_emit_compiled_universe`)
21. `tests/test_townlet/unit/universe/test_stage_6_optimize.py`
22. `tests/test_townlet/unit/universe/test_compiled_universe.py`

### Phase 9: Caching
23. Update `compiler.py` (add cache layer)
24. `tests/test_townlet/unit/universe/test_cache.py`

### Phase 10: Integration
25. Update `compiler.py` (add main `compile()` method)
26. `tests/test_townlet/integration/universe/test_compiler_integration.py`
27. `tests/test_townlet/integration/universe/test_dimension_regression.py`

---

## Related Tasks

- **TASK-003**: UAC Core DTOs (Dependency - must be complete)
- **TASK-004B**: UAC Capabilities (Depends on compiler)
- **TASK-005**: Brain as Code (Depends on ObservationSpec from compiler)
- **TASK-006**: Substrate-Agnostic Visualization (Depends on CompiledUniverse metadata)
- **TASK-007**: Live Training Visualization (Depends on rich metadata)

---

## Conclusion

This implementation plan provides a **test-driven, bottom-up approach** to building the Universe Compiler. By following the RED-GREEN-REFACTOR cycle for each phase, we ensure:

1. **Test coverage**: Every feature has tests before implementation
2. **Incremental delivery**: Each phase delivers working, tested code
3. **Safety**: Backward compatibility maintained throughout
4. **Documentation**: Architecture links provide context at every step

The compiler is a **critical foundation** for BAC, substrate-agnostic visualization, and advanced training features. This plan ensures a robust, well-tested implementation that unblocks the entire UAC/BAC roadmap.

**Estimated Total Effort**: 40-55 hours (including TDD, testing, validation)

**Next Step**: Begin Phase 0 (Pre-implementation setup) and create feature branch.
