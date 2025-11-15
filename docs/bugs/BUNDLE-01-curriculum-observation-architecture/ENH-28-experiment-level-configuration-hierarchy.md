Title: Experiment-level configuration hierarchy for observation policy and cross-curriculum settings

Severity: medium
Status: design-v2.1-complete

Subsystem: universe/compiler + config
Affected Version/Branch: main

**DESIGN v2.1 COMPLETE**: See `target-config-design-v2.md` for complete architecture.
**Implementation Reference**: See `reference-config-v2.1-complete.yaml` for comprehensive example with all options.
**Code Review**: Approved 100/100 confidence, ready for implementation.

Affected Files:
- `src/townlet/universe/compiler.py` (Stage 1: load experiment structure, Stage 2: cross-curriculum validation, Stage 5: observation spec)
- `src/townlet/config/` (new schema files: ExperimentConfig, StratumConfig, EnvironmentConfig, AgentConfig)
- `configs/` (entire hierarchy restructured)

## Summary

After BUG-43 enabled curriculum masking for transfer learning, we identified the need for experiment-level configuration hierarchy. Through brainstorming, we developed a clean four-layer architecture separating **what exists** (vocabulary, breaks checkpoints) from **how it behaves** (parameters, curriculum-safe).

**Core Principle**: WHAT vs HOW split enables curriculum progression without breaking checkpoint portability.

## Problem Statement

- BUG-43 forces all configs to pay obs_dim cost of curriculum superset, even single-level experiments
- Some settings are **experiment-level concerns** (apply to entire curriculum) vs **curriculum-level concerns** (vary per level)
- Current flat structure doesn't enforce vocabulary consistency across curriculum levels
- Unclear where cross-curriculum settings like grid size and meter names should live

## Target Architecture

**See `target-config-design.md` for complete specification.**

### Four-Layer Hierarchy

1. **Experiment** (`experiment.yaml`) - Metadata
2. **Stratum** (`stratum.yaml`) - World shape (substrate type, grid size, temporal mechanics)
3. **Environment** (`environment.yaml`) - World vocabulary (bars, affordances, VFS variables)
4. **Agent** (`agent.yaml`) - Perception + Drive + Brain
5. **Curriculum** (`levels/L*/`) - Behavioral parameters (depletion rates, costs, effects)

### Example Structure

```
configs/default_curriculum/
├── experiment.yaml      # Metadata (name, description, author)
├── stratum.yaml         # World shape (Grid2D 8×8, temporal enabled)
├── environment.yaml     # Vocabulary (8 bars, 14 affordances, VFS vars)
├── agent.yaml           # Perception + Drive + Brain
└── levels/
    ├── L0_0_minimal/
    │   ├── bars.yaml         # Bar parameters + cascades
    │   ├── affordances.yaml  # Affordance parameters
    │   └── training.yaml     # Runtime orchestration
    ├── L1_full_observability/
    └── L2_partial_observability/
```

### Key Design Decisions

**WHAT vs HOW Split**:
- `environment.yaml` defines WHAT exists (vocabulary - breaks checkpoints)
- `levels/L*/bars.yaml` defines HOW bars behave (parameters - doesn't break)
- `levels/L*/affordances.yaml` defines HOW affordances behave (parameters - doesn't break)

**Observation Control Simplified**:
- No need for `observation_policy` modes (curriculum_superset, minimal, explicit)
- `agent.yaml: perception.partial_observability` handles this directly:
  - `false` → full grid_encoding (no local_window)
  - `true` → local_window only (no grid_encoding)

**File Consolidation**:
- `brain.yaml` + `drive_as_code.yaml` → `agent.yaml`
- `cascades.yaml` → merged into `bars.yaml`
- `substrate.yaml` → split into `stratum.yaml` (shape) + removed

**Compiler Validation**:
- Cross-curriculum vocabulary consistency enforced
- All levels must have same bars/affordances as `environment.yaml`
- Compiler error if vocabulary mismatch detected

## Implementation Plan

See `target-config-design.md` for:
- Complete file schemas
- Compiler changes (load structure, validate vocabulary, build observation spec)
- Migration path (legacy support → migration script → deprecation)
- Benefits and open questions

## Status

- **Design v2.1**: COMPLETE (2025-11-15)
  - Code review round 1: Addressed (Support/Active pattern)
  - Code review round 2: Addressed (normalized vision_range, observation_encoding clarification)
  - Code review round 3: APPROVED 100/100 confidence
- **Implementation Reference**: COMPLETE (`reference-config-v2.1-complete.yaml`)
- **Implementation**: Not started (ready to begin)
- **Migration Script**: Not started
- **Tests**: Not started

## Owner

compiler + config subsystems

## Links

- **Design Document**: `target-config-design-v2.md` (complete v2.1 architecture)
- **Implementation Reference**: `reference-config-v2.1-complete.yaml` (600+ line complete example)
- **BUG-43**: Partial observability global view masking (enabled this work)
- **Historical Documents**: `archive/README.md` (design iterations, brainstorming artifacts)
