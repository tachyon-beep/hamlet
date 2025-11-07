# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## 2025-11-07 - TASK-002C: VFS Phase 1 Integration (COMPLETE)

**Status**: ✅ Complete (Nov 2025)
**Branch**: `claude/review-tdd-implementation-plan-011CUsrSgdXVvvK5pbMigkQf`
**Scope**: Pre-release cleanup and integration

### Summary

Integrated Variable & Feature System (VFS) Phase 1 into production environment, replacing legacy hardcoded observation system. Centralized test utilities to eliminate duplication and establish best practices for VFS testing.

**Key Achievements:**

- VFS fully integrated into `VectorizedHamletEnv` (replaces 212-line `ObservationBuilder`)
- All 5 curriculum levels validated (dimension compatibility maintained)
- Training validated (L0_0_minimal tested successfully, checkpoints working)
- Test suite refactored (centralized utilities, eliminated ~100 lines of duplication)
- 88 VFS tests passing (76 unit + 12 integration)

### Added

**VFS Integration** (Production):

- `VectorizedHamletEnv` now uses VFS for observation generation
  - Loads `variables_reference.yaml` from config packs (required)
  - Initializes `VariableRegistry` with GPU tensor backing
  - Builds observation spec from variables + exposures
  - `_get_observations()` method rewritten to use VFS registry
  - Exposure filtering based on observability mode (grid_encoding vs local_window)
  - Dynamic normalization (minmax/zscore) from variable config

**Test Utilities** (Test Infrastructure):

- `tests/test_townlet/unit/vfs/test_helpers.py` (135 lines)
  - `EXPECTED_DIMENSIONS`: Centralized constant for all 5 curriculum levels
  - `CONFIG_PATHS`: Centralized config paths (no magic strings)
  - `load_variables_from_config()`: Load VFS variables from YAML
  - `load_exposures_from_config()`: Load exposure configuration
  - `calculate_vfs_observation_dim()`: Calculate total observation dimensions
  - `assert_dimension_equivalence()`: Assert with checkpoint incompatibility warnings

### Changed

**BREAKING CHANGES:**

- All config packs **MUST** include `variables_reference.yaml`
- Missing file triggers clear error with quick-fix instructions
- Legacy observation system fully removed (no backward compatibility)

**Test Refactoring:**

- `test_vfs_legacy_equivalence.py`: 209 → 106 lines (49% reduction)
  - Replaced 5 repetitive test methods with single `@pytest.mark.parametrize` test
  - Import shared utilities instead of local copies
  - All tests use centralized constants and helpers

- `test_observation_dimension_regression.py`: Minor cleanup
  - Import shared constants (`EXPECTED_DIMENSIONS`, `CONFIG_PATHS`)
  - Use `assert_dimension_equivalence()` for better error messages
  - Kept `compute_vfs_observation_dim_from_agent_readable()` local (different logic)

**VectorizedHamletEnv Changes:**

- Added VFS imports: `VariableRegistry`, `VFSObservationSpecBuilder`, `VariableDef`
- `__init__()` loads variables and builds observation spec
- `observation_dim` calculated from VFS spec (dynamic, not hardcoded)
- `_get_observations()` completely rewritten (VFS registry-based)
- Tensor type conversions for normalization (YAML lists → torch.Tensor)
- Exposure filtering for POMDP vs full observability

### Removed

**Legacy System:**

- `src/townlet/environment/observation_builder.py` (212 lines)
  - Hardcoded observation construction replaced by VFS
  - All imports removed from codebase
  - 9 tests skipped (unit tests of legacy internals)

**Test Duplication:**

- ~100 lines of duplicated helper functions removed
- Local `EXPECTED_DIMENSIONS` constants consolidated
- Hardcoded config paths replaced with shared constant

### Fixed

**VFS Integration Issues:**

- YAML import shadowing (removed duplicate conditional import)
- Normalization type mismatch (added tensor conversion for min/max/mean/std)
- Grid encoding dimension duplication (use `_encode_full_grid()` instead of `encode_observation()`)
- Missing `variables_reference.yaml` in test config (added with both grid_encoding and local_window)
- POMDP variable name mismatch (conditional logic for grid_encoding vs local_window)
- ObservationBuilder imports in tests (removed, added skip decorators)

### Testing

**Dimension Validation:**

- All 5 curriculum levels validated:
  - L0_0_minimal: 38 dims ✅
  - L0_5_dual_resource: 78 dims ✅
  - L1_full_observability: 93 dims ✅
  - L2_partial_observability: 54 dims ✅
  - L3_temporal_mechanics: 93 dims ✅

**Training Validation:**

- L0_0_minimal: 5-episode smoke test passed
- Checkpoint save/load working (2.7MB checkpoints)
- No dimension mismatches or tensor shape errors

**Test Coverage:**

- 88 VFS tests passing (76 unit + 12 integration)
- 13 dimension validation tests (7 integration + 6 regression)
- Test helpers: 100% utilization (all 6 utilities actively used)
- Test execution: ~24 function calls across test suites

### Performance Impact

**Memory:**

- VFS registry: Minimal overhead (reuses existing tensors)
- Observation spec: Built once at initialization (O(1) per episode)

**Runtime:**

- Observation generation: Same performance as legacy (registry lookups are O(1))
- No performance regressions detected in training

### Migration Guide

**For Operators:**

1. All config packs MUST have `variables_reference.yaml`
2. Copy reference: `cp configs/L1_full_observability/variables_reference.yaml configs/your_config/`
3. Edit variables to match your configuration
4. See `docs/config-schemas/variables.md` for schema details

**For Developers:**

1. Use `VFS registry.get(var_id, reader="agent")` for observations
2. Use `registry.set(var_id, value, writer="engine")` to update state
3. Observation dimensions calculated dynamically from VFS spec
4. Access control enforced by registry (reader/writer permissions)

### Test Quality Improvements

**Best Practices Implemented:**

- DRY principle: Single source of truth for test constants
- No magic numbers: Centralized `EXPECTED_DIMENSIONS` and `CONFIG_PATHS`
- Parametrization: 5 repetitive tests → 1 parametrized test
- Shared scaffolding: Consistent helpers across test suites
- Type hints and docstrings throughout test utilities

**Test Helper Utilization:**

- Integration tests: 6/6 utilities used (100%)
- Regression tests: 4/6 utilities used (66%, has specialized logic)
- Overall: 6/6 utilities actively used by at least one suite

### Commits

**VFS Integration:**
- `74792b4` - feat(vfs): Integrate VFS into VectorizedHamletEnv, remove legacy ObservationBuilder
- `c28a2ee` - docs: Update CLAUDE.md with VFS integration status

**Test Refactoring:**
- `667f39a` - refactor(tests): Centralize VFS test utilities to eliminate duplication

### References

- VFS Design: `docs/plans/2025-11-06-variables-and-features-system.md`
- Implementation plan: `docs/tasks/TASK-002-variables-and-features-system.md`
- Configuration guide: `docs/config-schemas/variables.md`
- Migration guide: `docs/vfs-integration-guide.md` (if exists)

---

## 2025-11-07 - TASK-002B: Composable Action Space (COMPLETE)

**Status**: ✅ All 5 Phases Complete (Nov 2025)
**Branch**: `002b-composable-action-space`
**Scope**: 26-37h estimated → ~15h actual (under budget, efficient subagent-driven development)

### Summary

Replaced hardcoded action space with composable architecture supporting substrate actions + custom actions from global vocabulary. Enables curriculum transfer via fixed action_dim, substrate-agnostic custom actions (REST, MEDITATE), and runtime action masking.

**Key Achievements:**

- Composable action space: Substrate (6-14) + Custom (2) = 8-16 total actions
- Global vocabulary (`global_actions.yaml`) shared across all configs
- Checkpoint transfer enabled (same action_dim across L0→L1→L2→L3)
- 2 substrate-agnostic custom actions: REST (recovery), MEDITATE (mood)
- 350+ tests passing (12 action builder, 3 custom actions, 2 curriculum transfer)
- 78 test fixes for dynamic action_dim (no more hardcoded action_dim=6)

### Added

**Action Space Components** (Phase 0-2):

- `ActionConfig`: Pydantic schema for action definitions (costs, effects, delta, teleport_to, enabled)
- `ComposedActionSpace`: Container tracking substrate + custom + affordance actions with metadata
- `ActionSpaceBuilder`: Composes actions from substrate.get_default_actions() + global_actions.yaml
- `configs/global_actions.yaml`: Global vocabulary with 2 substrate-agnostic custom actions (REST, MEDITATE)
  - **Design Decision**: Removed TELEPORT_HOME and SPRINT from original plan (broke on non-Grid2D substrates)
  - Global vocabulary must work on ALL substrates for curriculum transfer (Grid2D/3D/ND, Continuous, Aspatial)
  - Substrate-specific actions can be added in per-config files (future extensibility)

**Substrate Default Actions** (Phase 1):

- `Grid2DSubstrate.get_default_actions()`: 6 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
- `Grid3DSubstrate.get_default_actions()`: 8 actions (adds UP_Z, DOWN_Z)
- `GridNDSubstrate.get_default_actions()`: 2N+2 actions (DIM{i}_NEG/POS pattern for arbitrary dimensions)
- `Continuous1D/2D/3DSubstrate.get_default_actions()`: 4/6/8 actions (integer deltas scaled by movement_delta)
- `ContinuousNDSubstrate.get_default_actions()`: 2N+2 actions (same pattern as GridND)
- `AspatialSubstrate.get_default_actions()`: 2 actions (INTERACT, WAIT only, no movement)

**Environment Integration** (Phase 4):

- `VectorizedHamletEnv` uses `ActionSpaceBuilder` for composed action space
- `get_action_masks()` integrates base masking from `action_space.get_base_action_mask()`
- `_apply_custom_action()` dispatches costs/effects/teleportation for custom actions
- `_get_meter_index()` maps meter names to tensor indices
- Cached INTERACT/WAIT action indices for performance (completes deferred Task 1.6)
- Dynamic movement deltas from ActionConfig (replaces hardcoded arrays)

**Custom Actions** (Phase 3-4):

- **REST**: Passive recovery (energy: -0.002, mood: -0.01), available anywhere, substrate-agnostic
- **MEDITATE**: Mental health action (energy: +0.001, mood effects: +0.02), substrate-agnostic
- **Note**: TELEPORT_HOME and SPRINT removed from global vocabulary (Grid2D-specific, breaks universal curriculum)

**Testing & Documentation** (Phase 5):

- `tests/test_townlet/unit/test_action_builder.py`: 12 tests for ActionConfig + ComposedActionSpace + ActionSpaceBuilder
- `tests/test_townlet/integration/test_custom_actions.py`: 3 tests for REST, MEDITATE, TELEPORT_HOME
- `tests/test_townlet/integration/test_curriculum_transfer.py`: 2 tests verifying L0↔L1 checkpoint transfer
- `docs/config-schemas/enabled_actions.md`: Pattern documentation with examples
- `CLAUDE.md`: Updated with composable action space architecture, action count tables, checkpoint transfer benefits

### Changed

**Action Space Formula** (Phase 1):

- Grid2D: 6 substrate + 2 custom = **8 actions** (was 6)
- Grid3D: 8 substrate + 2 custom = **10 actions** (was 8)
- GridND(7D): 14 substrate + 2 custom = **16 actions** (was 14)
- Aspatial: 2 substrate + 2 custom = **4 actions** (was 2)

**VectorizedHamletEnv** (Phase 4):

- `__init__()` builds action_space via ActionSpaceBuilder (replaces `self.action_dim = substrate.action_space_size`)
- `_build_movement_deltas()` uses `action_space.actions` instead of `substrate.get_default_actions()`
- `get_action_masks()` starts with `action_space.get_base_action_mask()` (disabled actions = False)
- `_execute_actions()` dispatches custom actions before substrate actions (early dispatch pattern)

**VectorizedPopulation** (Phase 4):

- `action_dim` parameter now defaults to `None` (uses `env.action_dim` if not specified)
- Checkpoint metadata uses `self.action_dim` instead of hardcoded 6

**Network Architecture** (Phase 5):

- SimpleQNetwork: 29 input → **8 output** dims (~28K params, was ~26K)
- RecurrentSpatialQNetwork: 256 hidden → **8 output** dims (~620K params, was ~600K)
- All Grid2D configs have same architecture (29→8), enabling checkpoint transfer!

### Fixed

**Empty List Semantic Bug** (Phase 2):

- `ActionSpaceBuilder`: Fixed `enabled_action_names=[]` to correctly disable all actions (was enabling all)
- Changed from truthiness check (`if enabled_action_names`) to identity check (`if enabled_action_names is not None`)

**Hardcoded Action Assumptions** (Phase 4):

- Fixed 78 occurrences of hardcoded `action_dim=6` across 11 test files
- Tests now use `env.action_dim` dynamically (future-proof for any action space size)
- Fixed dimension mismatch errors: Q-values [batch, 6] vs action masks [batch, 10]

**Canonical Action Ordering** (Phase 1):

- All substrates emit actions in canonical order: [movement...], INTERACT (second-to-last), WAIT (last)
- Enables downstream systems to identify meta-actions by position
- 8 tests enforce ordering contract for all substrate types

**Integer Delta Pattern** (Phase 1):

- Continuous substrates use integer deltas (±1) scaled by `movement_delta` (not float deltas as originally planned)
- Benefits: simpler schema, dynamic scaling, type safety, consistency with grid substrates

### Technical Details

**Action Composition Pipeline**:

1. Substrate provides default actions via `get_default_actions()` (6-14 actions)
2. ActionSpaceBuilder loads `configs/global_actions.yaml` (4 custom actions)
3. Builder assigns sequential IDs: substrate (0-N), custom (N+1 onward)
4. Builder marks disabled actions with `enabled=False` (based on training.yaml - future)
5. ComposedActionSpace tracks metadata (substrate_count, custom_count, enabled_count)

**Action Dispatch Flow**:

1. `get_action_masks()` returns base mask (disabled actions = False)
2. Boundary constraints, dead agents, affordance availability applied on top
3. `_execute_actions()` checks if action >= `substrate_action_count` (custom action)
4. Custom actions call `_apply_custom_action()` (costs/effects/teleportation)
5. Substrate actions use movement deltas from ActionConfig (no hardcoding)

**Checkpoint Transfer Mechanism**:

- L0_0_minimal: 3×3 grid, 1 affordance, 8 actions (6 substrate + 2 custom)
- L0_5_dual_resource: 7×7 grid, 4 affordances, 8 actions (6 substrate + 2 custom)
- L1_full_observability: 8×8 grid, 14 affordances, 8 actions (6 substrate + 2 custom)
- L3_temporal_mechanics: 8×8 grid, temporal mechanics, 8 actions (6 substrate + 2 custom)

All have:
- Observation dim: 29 (constant via relative position encoding)
- Action dim: 8 (constant via global vocabulary - substrate-agnostic actions only)
- Network architecture: 29→256→128→8 (SimpleQNetwork)

Result: **Q-network trained on L0 transfers to L1-L3 without retraining!**

### Performance Impact

**Memory**:

- SimpleQNetwork: +2K params (26K → 28K, +8% for 2 custom actions)
- RecurrentSpatialQNetwork: +20K params (600K → 620K, +3%)

**Runtime**:

- Action space build: O(1) at initialization (10-20 actions typical)
- Action masking: O(action_dim) per step (~10 actions, negligible)
- Custom action dispatch: O(custom_count) per step (~0-2 custom actions/step, minimal)

**Test Coverage**:

- action_builder.py: 91% coverage (6 uncovered lines are unreachable error paths)
- 350+ tests passing (no regressions from action_dim change)

### Migration Guide

**For Operators**:

1. Action counts changed for all substrate types (+2 substrate-agnostic custom actions)
2. Grid2D configs: `action_dim=6` → `action_dim=8`
3. Tests: Remove hardcoded `action_dim=6`, use `env.action_dim` instead
4. Networks: Update output dim from 6 to 8 for Grid2D configs

**For Developers**:

1. Use `action_space.get_action_by_name(name)` for reliable action lookups
2. Use `substrate.get_default_actions()` instead of hardcoding action lists
3. Never hardcode action indices (use cached `interact_action_idx`, `wait_action_idx`)
4. Custom actions must define `costs`/`effects`/`delta`/`teleport_to` in global_actions.yaml

### Commits

**Phase 0-1 (Substrate Actions)**:
- `c2ac784` - feat(actions): add ActionConfig with enabled field
- `6cdd794` - feat(actions): Grid2D default actions
- `e9a783d` - feat(actions): Grid3D default actions
- `aa25cef` - feat(actions): GridND default actions with DIM{i}_{NEG|POS} pattern
- `d04e086` - feat(actions): Continuous substrate default actions
- `27cb2b1` - feat(actions): Aspatial default actions (INTERACT, WAIT only)
- `cf1422b` - test: verify substrates emit canonical action ordering
- `e6ca865` - refactor: use ActionConfig deltas instead of hardcoded arrays

**Phase 2 (ActionSpaceBuilder)**:
- `dd686c9` - feat(actions): add ComposedActionSpace class
- `1ca3951` - feat(actions): add ActionSpaceBuilder with global vocabulary
- `d874bf6` - fix(actions): correct empty list behavior in ActionSpaceBuilder

**Phase 3 (Global Actions Config)**:
- `7083bc0` - feat(config): add global_actions.yaml with 2 substrate-agnostic custom actions
- `d859288` - docs(config): document enabled_actions pattern

**Phase 4 (Environment Integration)**:
- `634d33d` - feat(env): integrate ActionSpaceBuilder with backward-compatible action_dim
- `320ac07` - feat(env): integrate disabled action masking into get_action_masks()
- `6da9c71` - feat(env): implement custom action dispatch
- `c9a4a69` - fix(tests): remove hardcoded action_dim=6 from integration tests

**Phase 5 (Testing & Docs)**:
- `13a1a5d` - test(curriculum): verify L0 and L1 have same action_dim
- `585df04` - docs: document composable action space architecture

### References

- Implementation plan: `/home/john/hamlet/docs/plans/2025-11-06-composable-action-space.md`
- Pattern documentation: `/home/john/hamlet/docs/config-schemas/enabled_actions.md`
- Global vocabulary: `/home/john/hamlet/configs/global_actions.yaml`

---

## 2025-11-06 - TASK-002A: Configurable Spatial Substrates (COMPLETE)

**Status**: ✅ All 8 Phases Complete (Nov 2024 - Nov 2025)
**Branch**: `task-002a-configurable-spatial-substrates`
**Scope**: 15-22h estimated → 81-107h actual (+368-586% growth)

### Summary

Replaced hardcoded 2D grid assumptions with config-driven spatial substrate system supporting 1D-100D spaces. Enables 3D environments, continuous spaces, aspatial universes, and N-dimensional RL research.

**Key Achievements:**

- 6 substrate types implemented (Grid2D/3D/ND, Continuous1D/2D/3D/ND, Aspatial)
- 3 observation encoding modes (relative, scaled, absolute)
- 4 boundary modes (clamp, wrap, bounce, sticky)
- 1,159 tests passing (77% coverage, 85%+ substrate modules)
- Transfer learning enabled (train on 3×3, run on 8×8)

### Added

**Substrate Types** (6 total):

- `Grid2DSubstrate`: 2D discrete grids (original behavior preserved)
- `Grid3DSubstrate`: 3D cubic grids for multi-floor environments
- `GridNDSubstrate`: 4D-100D hypercube grids (supports up to 100 dimensions)
- `Continuous1D/2D/3DSubstrate`: Smooth movement for robotics
- `ContinuousNDSubstrate`: 4D-100D continuous spaces for abstract RL
- `AspatialSubstrate`: Position-less universes (pure resource management)

**Configuration System** (Phases 1-3):

- `substrate.yaml` schema with Pydantic validation
- `observation_encoding` modes: relative (transfer learning), scaled (size-aware), absolute (physics)
- Boundary modes: clamp (walls), wrap (toroidal), bounce (elastic), sticky (adhesive)
- Distance metrics: manhattan (L1), euclidean (L2), chebyshev (L∞)
- Action labels: 4 presets (gaming, 6dof, cardinal, math) + custom labels
- Template configs at `configs/templates/substrate*.yaml`
- All curriculum levels (L0, L0.5, L1, L2, L3) have `substrate.yaml`

**Position Management** (Phases 4-5):

- Abstract `SpatialSubstrate` interface for polymorphic operations
- `substrate.initialize_positions()` - random agent/affordance placement
- `substrate.apply_movement()` - boundary-aware movement
- `substrate.compute_distance()` - metric-specific distance
- `substrate.normalize_positions()` - observation encoding
- `substrate.action_space_size` - dynamic action space (2N+2 for N dimensions)
- VectorizedHamletEnv fully substrate-agnostic

**Observation Encoding** (Phase 6):

- Coordinate encoding: 3D (8×8×3) = 512 dims → 3 dims (170× reduction)
- Substrate-specific `encode_observation()` and `encode_partial_observation()`
- Enables transfer learning (same network works on different grid sizes)
- Observation dimension formula: `position_encoding + meters + affordances + temporal`
- Grid2D (relative): 29 dims (2 coords + 8 meters + 15 affordances + 4 temporal)
- Aspatial: 13 dims (0 coords + 4 meters + 5 affordances + 4 temporal with test config)

**Frontend Multi-Substrate Rendering** (Phase 7):

- `AspatialView.vue`: Meters-only dashboard (no fake grid)
- `Grid.vue`: SVG-based 2D grid with heat maps (existing)
- WebSocket substrate metadata protocol
- Substrate type detection and renderer dispatch
- Live inference server routes by substrate type

**Testing Framework** (Phase 8):

- Property-based tests (Hypothesis): 8 tests for substrate contracts
- Integration tests: 8 parameterized tests (Grid2D + Aspatial)
- Regression tests: 7 tests for backward compatibility
- **Total**: 1,159 tests passing (23 substrate-specific)
- **Coverage**: 77% overall, 85%+ substrate modules

**Checkpoint Format V3**:

- Added `substrate_metadata` with `position_dim` and `substrate_type`
- Validates substrate compatibility on load
- Pre-flight validation for legacy checkpoints
- Migration guide in error messages

### Changed

**BREAKING CHANGES:**

- Checkpoint format Version 2 → Version 3 (substrate metadata required)
- Legacy checkpoints unsupported (clear migration path)
- ObservationBuilder requires substrate parameter
- All position tensors now variable-length: `(num_agents, position_dim)` where `position_dim` ∈ [0, 100]

**Position Management:**

- Removed 2D hardcoded assumptions (was `positions.shape = (N, 2)`)
- Now substrate-aware (Grid2D: 2, Grid3D: 3, GridND: N, Aspatial: 0)
- All movement operations use `substrate.apply_movement()` (was `torch.clamp`)
- Distance checks use `substrate.compute_distance()` (was hardcoded Manhattan)

**Observation Encoding:**

- Full observability: coordinate encoding (was one-hot grid cells)
- POMDP: substrate-aware local windows (was hardcoded 2D)
- Dimension calculation: `substrate.position_dim` (was `grid_size²`)
- Transfer learning enabled: train on 3×3, deploy on 8×8 with same network

**Action Space:**

- Dynamic sizing: `substrate.action_space_size` (was hardcoded if-else)
- Grid2D: 6 actions, Grid3D: 8 actions, GridND: 2N+2 actions
- Aspatial: 2 actions (INTERACT + WAIT only)

**Recording & Visualization:**

- Variable-length position tuples: `tuple[int, ...]` (was `tuple[int, int]`)
- Frontend routing by substrate type (was single 2D renderer)
- WebSocket includes substrate metadata

### Removed

- Hardcoded 2D position assumptions throughout codebase
- Manual grid iteration for affordance randomization (55+ lines → substrate method)
- One-hot grid encoding for large grids (prevents 3D substrates)
- 15 lines of action space sizing logic (→ substrate property)
- Backward compatibility with Version 2 checkpoints

### Scope Evolution

**Original Plan** (Nov 2024): 3 substrate types, 5 phases, 15-22h
**Revised Plan** (Nov 4): 3 substrate types, 8 phases, 51-65h (+140-195%)
**Final Implementation**: 6 substrate types, 8 phases, 81-107h (+368-586%)

**Additions Beyond Original Scope:**

- GridND/ContinuousND (4D-100D support) - generalization was cheap
- 3 encoding modes (was 1) - research paradigm flexibility
- Action label system - pedagogical value (semantics are arbitrary)
- Frontend Phase 7 - aspatial needed different visualization
- Testing Phase 8 - production quality requirements

**Why Scope Expanded:**

- Generalization cheap: Grid3D → GridND was 2-3h
- Research value: robotics, abstract RL, high-dimensional spaces
- Pedagogical completeness: deep insights about RL representation
- Production quality: property-based testing, regression coverage

**Was It Worth It?**

- ✅ Supports 1D-100D substrates (future-proof)
- ✅ Transfer learning works (practical benefit)
- ✅ Aspatial reveals insight (positioning optional)
- ⚠️ N-dimensional rarely used (but cheap to add)

### Testing

**Final Test Results:**

- 1,159/1,159 tests passing (100%)
- 23 substrate-specific tests (property-based, integration, regression)
- 77% code coverage overall
- 85%+ coverage for substrate modules (grid2d: 92%, continuous: 95%, continuousnd: 96%)

**Test Categories:**

- Property tests (Hypothesis): substrate contracts, invariants
- Integration tests: multi-substrate parameterization
- Regression tests: Grid2D behavioral equivalence
- Unit tests: already substrate-aware from Phase 6

---

## [0.2.0] - 2025-11-05 (Post-0.1.0 Improvements)

### Added (TASK-001 - Variable-Size Meter System)

**Status**: ✅ Complete (PR #1, merged to main 2025-11-04)

- **Variable-Size Meter System**:
  - Support for 1-32 meters (was hardcoded to 8)
  - Dynamic tensor sizing based on `meter_count` from `bars.yaml`
  - Checkpoint metadata includes `meter_count` for compatibility validation
  - Enables minimal tutorials (4 meters) and complex simulations (12-32 meters)
  - Example configs: 4-meter pedagogy, 12-meter research, 32-meter neurotransmitters

- **Dynamic Observation Dimensions**:
  - Observation size calculated dynamically from meter count
  - Network architectures adapt automatically to variable meters
  - Transfer learning enabled across different meter configurations

- **Meter Index Flexibility**:
  - RewardStrategy uses configurable meter indices (not hardcoded energy=0, health=6)
  - Action masking uses dynamic meter indices
  - Action costs use dynamic tensors (not hardcoded 8-element)
  - Recurrent networks use dynamic meter count

### Fixed (TASK-001)

- Action masking IndexError on non-8-meter configs (used hardcoded indices)
- Action costs RuntimeError on non-8-meter configs (hardcoded 8-element tensors)
- Recurrent network feature parsing on variable meters (hardcoded `num_meters=8`)

### Testing (TASK-001)

- 35 new tests covering variable meters (4, 8, 12, 32 meter configs)
- All tests passing, no regressions

---

### Added (QUICK-001 - Affordance Transition Database)

**Status**: ✅ Complete (merged to main 2025-11-04)

- **Affordance Transition Tracking**:
  - `insert_affordance_visits()` method in DemoDatabase
  - Tracks affordance transition patterns (Bed→Hospital→Job sequences)
  - Database schema: `affordance_visits` table (episode_id, from_affordance, to_affordance, visit_count)
  - Enables behavioral pattern analysis, reward hacking detection, Markov chain analysis

- **Runner Integration**:
  - Transition tracking in `runner.py` during episode execution
  - Per-episode persistence to database
  - Data structures: `affordance_transitions`, `last_affordance` state tracking

### Testing (QUICK-001)

- 4 new tests (3 unit + 1 integration)
- Coverage: database.py 87% (+38), runner.py 75% (+2)
- All tests passing

---

### Added (QUICK-002 - DemoRunner Resource Cleanup)

**Status**: ✅ Complete (PR #2, merged to main 2025-11-05)

- **Context Manager Support**:
  - `__enter__` and `__exit__` methods for DemoRunner
  - `_cleanup()` method (idempotent, handles partial initialization)
  - Guarantees resource cleanup (database, TensorBoard, recorder)
  - Fixes SQLite connection leaks in tests

- **DemoDatabase Improvements**:
  - Idempotent `close()` method with `_closed` flag tracking
  - Multiple close() calls safe

### Changed (QUICK-002)

- DemoRunner can now be used as context manager: `with DemoRunner(...) as runner:`
- `run()` method uses `_cleanup()` instead of duplicating cleanup code
- 3 existing tests converted to use context manager pattern

### Documentation (QUICK-002)

- Added "Using DemoRunner for Checkpoint Operations" section to CLAUDE.md
- Documents when to use context manager vs. calling `run()`
- Shows ✅ GOOD and ❌ BAD usage patterns

### Testing (QUICK-002)

- 3 new tests for context manager behavior
- 22/22 checkpointing tests passing
- Zero ResourceWarnings with `-W error::ResourceWarning`

---

### Added (QUICK-003 - PDR-002 No-Defaults Whitelist Cleanup)

**Status**: ✅ Complete (merged to main 2025-11-05, completed alongside QUICK-002)

- **No-Defaults Principle Implementation**:
  - Removed all UAC (UNIVERSE_AS_CODE) defaults from environment initialization
  - Removed all BAC (BRAIN_AS_CODE) defaults from network architectures
  - All behavioral parameters must now be explicitly specified in configs
  - 100% PDR-002 compliance achieved

- **Removed Defaults** (environment):
  - VectorizedHamletEnv: grid_size, partial_observability, vision_range, enable_temporal_mechanics, energy costs, agent_lifespan (8 UAC defaults removed)

- **Removed Defaults** (networks):
  - SimpleQNetwork: hidden_dim (was 128)
  - RecurrentSpatialQNetwork: action_dim, window_size, num_meters, num_affordance_types, enable_temporal_features, hidden_dim (6 BAC defaults removed)
  - RecurrentSpatialQNetwork.reset_hidden_state: batch_size default removed

- **Removed Defaults** (reward):
  - RewardStrategy: num_agents, meter_count defaults removed

- **Retained Infrastructure Defaults** (whitelisted):
  - device=torch.device("cpu") (infrastructure fallback)
  - config_pack_path=None (infrastructure fallback)
  - enabled_affordances=None (semantic: "all affordances" - not a hidden default)

- **Compliance Artifacts**:
  - Created `.defaults-whitelist-compliant.txt` (165 lines, infrastructure-only defaults)
  - Replaced old whitelist (85% UAC/BAC violations) with compliant version (0% violations)
  - Updated 7 fixtures in conftest.py with explicit parameters
  - Updated 6 integration tests with explicit parameters

### Documentation (QUICK-003)

- Added PDR-002 compliance documentation to all affected docstrings
- Created PLAN-QUICK-003-PHASE-{1,2,3,4}.md execution plans
- Moved QUICK-002 and QUICK-003 to `docs/tasks/completed/`
- Updated whitelist documentation with compliant patterns

### Changed (QUICK-003)

- **BREAKING**: All production configs must be complete (no silent fallback to code defaults)
- Config validation fails fast with clear error messages on missing UAC/BAC parameters
- isinstance() calls updated to use PEP 604 union syntax (X | Y)
- All UAC/BAC parameters must be explicit in config files (self-documenting configs)

---

### Added (Documentation Improvements)

**Status**: ✅ Complete (merged to main 2025-11-04)

- **AI-Friendly Documentation Structure**:
  - All architecture docs now include structured YAML frontmatter
  - AI-Friendly Summary sections (What/Why/Who/Reading Strategy)
  - Scope boundaries (In scope / Out of scope / Boundaries)
  - Related documents and reading order guidance
  - Token-efficient navigation for AI assistants

- **Phase-Based Task Organization**:
  - Restructured task-002a plans into 8 phase-based documents
  - Phase breakdown: 1-Foundation, 2-Foundation, 3-Config Migration, 4-Environment Integration, 5-Position Management, 6-Observation Builder, 7-Frontend Visualization, 8-Testing Verification
  - Added architecture templates and glossary

- **Task Documentation**:
  - Created comprehensive task templates (TASK-TEMPLATE, QUICK-TEMPLATE, BUG-TEMPLATE)
  - Added README.md for docs/tasks/ structure
  - Moved completed tasks to `docs/tasks/completed/`

- **Archive Organization**:
  - Archived obsolete investigations
  - Archived QUICK-003 plans to `docs/plans/archive/`

---

### Added (Repository Cleanup - Phase 1-3)

**Status**: ✅ Complete (from 0.1.0)

- **Repository Cleanup**:
  - LICENSE file (MIT) with proper copyright notice
  - CHANGELOG.md (this file) following Keep a Changelog format
  - CONTRIBUTING.md with comprehensive development guidelines
  - CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
  - SECURITY.md with responsible disclosure policy
  - GitHub issue templates (bug report, feature request)
  - GitHub pull request template
  - .python-version file (3.13) for version management
  - frontend/.nvmrc file (Node 20) for consistent Node.js versions

- **Documentation**:
  - Documentation structure overhaul with comprehensive docs/README.md
  - Observation space documentation in main README
  - Complete pyproject.toml metadata (keywords, classifiers, URLs, pytest markers)
  - All missing dependencies added to pyproject.toml

### Changed (Repository Cleanup)

- Replaced print() statements with proper logging in:
  - src/townlet/demo/live_inference.py
  - src/townlet/recording/**main**.py
- Updated pyproject.toml with all top-level dependencies

### Fixed (Repository Cleanup)

- README affordance count (14, not 15) - CoffeeShop commented out
- README test count (644+, not 387) with correct recording test count (73)
- README entry point paths now correctly point to scripts/run_demo.py
- Documentation paths now correctly point to docs/manual/ and docs/architecture/
- .gitignore patterns:
  - Changed `__pycache__/` to `**/__pycache__/` for recursive matching
  - Consolidated database patterns (*.db,*.db-shm, *.db-wal,*.sqlite, *.sqlite3)
  - Removed duplicate patterns
- CI workflow now includes mypy type checking

---

## [0.1.0] - 2025-11-04

### Added

- **Phase 3 Complete**: Vectorized GPU training environment
- **Level 1-3 Training**: Progressive complexity (L0→L1→L2→L3)
  - L0: Temporal credit assignment (single affordance)
  - L0.5: Multiple resource management (4 affordances)
  - L1: Full observability baseline (14 affordances, MLP network)
  - L2: Partial observability with LSTM (5×5 local vision, POMDP)
  - L3: Temporal mechanics (24-tick day/night cycle, operating hours)
- **Adversarial Curriculum**: 5-stage adaptive difficulty progression
- **Intrinsic Motivation**: RND-based exploration with variance-based annealing
- **Recording System**: Episode recording, replay, and video export (73 tests)
- **Unified Server**: Combined training + inference server
- **Vue 3 Frontend**: Live visualization with WebSocket streaming
- **Test Suite**: 644+ tests with 70% coverage
- **TensorBoard Integration**: Training metrics visualization
- **SQLite Metrics**: Episode tracking and analysis

### Architecture

- **Vectorized Environment**: GPU-native batched training
- **Two Network Types**:
  - SimpleQNetwork (~26K-70K params) for full observability
  - RecurrentSpatialQNetwork (~600K params) with LSTM for POMDP
- **Fixed Affordance Vocabulary**: All 14 affordances in observation space for transfer learning
- **UNIVERSE_AS_CODE**: All game mechanics defined in YAML configs

### Documentation

- 90+ markdown files across 9 categories
- Comprehensive architecture documentation (TOWNLET_HLD.md, ROADMAP.md)
- User guides for training, replay, visualization
- Pedagogical "teachable moments" collection
- Research-driven development methodology documented

### Known Issues

- CoffeeShop affordance commented out in configs (14 active, not 15)
- Legacy src/hamlet/ code fully migrated to src/townlet/
- Entry point temporarily in demo/ (will move to training/)

---

## Release Notes Format

### Added

- New features or functionality

### Changed

- Changes to existing functionality

### Deprecated

- Features that will be removed in future releases

### Removed

- Features removed in this release

### Fixed

- Bug fixes

### Security

- Security-related changes
