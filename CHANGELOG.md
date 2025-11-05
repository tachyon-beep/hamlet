# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (TASK-002A - Configurable Spatial Substrates, Phase 1-5)

**Status**: ✅ Phase 5 Complete (on branch `task-002a-configurable-spatial-substrates`)

- **Substrate Abstraction System** (Phases 1-4):
  - Abstract `SpatialSubstrate` interface for polymorphic position management
  - `Grid2DSubstrate`: 2D square grids with configurable boundaries (clamp, wrap, bounce, sticky)
  - `Grid3DSubstrate`: 3D cubic grids for multi-floor environments (8×8×3 tested)
  - `AspatialSubstrate`: Position-less universes (pure resource management, no spatial reasoning)
  - Distance metrics: Manhattan (L1), Euclidean (L2), Chebyshev (L∞)
  - Substrate factory for building from `substrate.yaml` configuration
  - Pydantic schema validation for substrate configuration

- **Environment Integration** (Phase 4):
  - VectorizedHamletEnv loads substrate from `substrate.yaml`
  - All position operations use substrate methods:
    - `substrate.initialize_positions()` - random agent/affordance placement
    - `substrate.apply_movement()` - boundary-aware movement
    - `substrate.is_on_position()` - position-based interaction checks
    - `substrate.get_all_positions()` - position enumeration for affordances
  - Observation encoding uses substrate for position representation

- **Observation Encoding** (Phase 5):
  - **Substrate-Based Encoding**:
    - `substrate.encode_observation()` for full observability
    - `substrate.encode_partial_observation()` for POMDP local windows
    - `substrate.get_observation_dim()` for network architecture sizing
    - ObservationBuilder uses substrate for all position encoding
  - **Coordinate Encoding** (enables 3D/large grids):
    - 3D (8×8×3): 512 dims → 3 dims via normalized coordinates
    - Replaces one-hot encoding for grids >8×8 (prevents dimension explosion)
    - Enables transfer learning (same network works on different grid sizes)
  - **Variable-Dimensionality Support**:
    - 2D positions: (x, y) encoding
    - 3D positions: (x, y, z) encoding
    - Aspatial: no position encoding (position_dim=0)

- **Checkpoint Format V3** (Phase 5):
  - Added `substrate_metadata` field with `position_dim` and `substrate_type`
  - Validates position dimensionality on load (2D vs 3D vs aspatial)
  - Pre-flight validation detects legacy checkpoints on startup
  - Clear error messages guide users to delete old checkpoints
  - DemoRunner saves Version 3 checkpoints with full substrate context

- **Recording System** (Phase 5 Task 5.9):
  - Variable-length position recording: `tuple[int, ...]` type hints
  - Handles 2D (x, y), 3D (x, y, z), and aspatial () positions
  - EpisodeRecorder converts positions dynamically based on dimensionality
  - RecordedStep and EpisodeMetadata support flexible position tuples
  - Affordance layout recording with variable-dimension positions
  - 3 new tests for substrate-aware recording

- **Visualization System** (Phase 5 Task 5.8):
  - Live inference server sends substrate metadata to frontend
  - `_build_grid_data()` method routes by substrate type
  - WebSocket state updates include substrate type and position_dim
  - Affordance position handling for variable dimensions (x, y, z)
  - Frontend can detect substrate and route to appropriate renderer
  - Aspatial rendering support (meters-only, no grid)

- **Test Suite Updates** (Phase 5 Task 5.10):
  - All tests updated for substrate-agnostic position assertions
  - Checkpoint tests use config_pack_path for substrate loading
  - Property tests instantiate Grid2DSubstrate for ObservationBuilder
  - 826/827 tests passing (99.88% pass rate)
  - All substrate-specific tests passing (core, migration, integration)

- **Configuration System** (Phase 3):
  - Added `substrate.yaml` to all curriculum levels (L0, L0.5, L1, L2, L3)
  - Template with comprehensive documentation at `configs/templates/substrate.yaml`
  - Examples: 2D grids, 3D cubic, toroidal wraparound, aspatial

- **Phase 5B Features** (3D Cubic Grid):
  - 3D cubic grid support (8×8×3 tested, configurable depth)
  - Configurable action labels (UP/DOWN/LEFT/RIGHT/UP_FLOOR/DOWN_FLOOR/INTERACT)
  - Grid3DSubstrate with full 3D movement and positioning
  - Validation script for substrate integration testing

### Changed (TASK-002A)

- **BREAKING:** Checkpoint format Version 2 → Version 3 (substrate_metadata field added)
- **BREAKING:** Legacy checkpoints no longer supported without migration
- **BREAKING:** ObservationBuilder now requires substrate parameter
- All position operations now use substrate methods (removed hardcoded 2D grid assumptions)
- Observation dimensions now substrate-aware:
  - Full observability: `substrate.get_observation_dim()` (was `grid_size²`)
  - Partial observability: uses `substrate.position_dim` (was hardcoded `2`)
  - Enables aspatial universes with `position_dim=0`
- Recording system now handles variable-length positions (2D, 3D, aspatial)
- Visualization system routes by substrate type for rendering
- Type hints updated to support flexible position tuples: `tuple[int, ...]`

### Removed (TASK-002A)

- Hardcoded 2D position assumptions throughout codebase
- Backward compatibility with Version 2 checkpoints
- Manual grid iteration for affordance randomization (55+ lines)
- Hardcoded grid encoding in observation builder

### Testing (TASK-002A)

- **Comprehensive Test Coverage**:
  - 826/827 tests passing (99.88% pass rate)
  - 14 substrate-specific tests (core, migration, integration)
  - 3 recording system tests for variable-length positions
  - 6 property tests validating substrate-agnostic invariants
  - 23 checkpoint tests with Version 3 format validation
  - All integration tests updated for substrate abstraction
- **Test Categories**:
  - Core substrate functionality (Grid2D, Grid3D, Aspatial)
  - Distance metrics (Manhattan, Euclidean, Chebyshev)
  - Boundary conditions (clamp, wrap, bounce, sticky)
  - Observation encoding (full/partial, 2D/3D/aspatial)
  - Checkpoint save/load with substrate validation
  - Recording and visualization with variable dimensions

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
  - src/townlet/recording/__main__.py
- Updated pyproject.toml with all top-level dependencies

### Fixed (Repository Cleanup)

- README affordance count (14, not 15) - CoffeeShop commented out
- README test count (644+, not 387) with correct recording test count (73)
- README entry point paths now correctly point to scripts/run_demo.py
- Documentation paths now correctly point to docs/manual/ and docs/architecture/
- .gitignore patterns:
  - Changed `__pycache__/` to `**/__pycache__/` for recursive matching
  - Consolidated database patterns (*.db, *.db-shm, *.db-wal, *.sqlite, *.sqlite3)
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
