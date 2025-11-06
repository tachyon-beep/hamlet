# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
