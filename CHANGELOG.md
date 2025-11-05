# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Phase 5 - TASK-002A)

- **Substrate Abstraction**: Polymorphic spatial position management
  - Abstract `SpatialSubstrate` interface for position operations
  - `Grid2DSubstrate`: 2D square grids with configurable boundaries (clamp, wrap, bounce, sticky)
  - `AspatialSubstrate`: Position-less universes (pure state machines)
  - Distance metrics: Manhattan (L1), Euclidean (L2), Chebyshev (L∞)
  - Enables future 3D grids, continuous spaces, graph topologies
- **Checkpoint Format V3**: Added `position_dim` field for substrate validation
  - Validates position dimensionality on load (2D vs 3D vs aspatial)
  - Pre-flight validation detects legacy checkpoints on startup
  - Clear error messages guide users to delete old checkpoints
- **Substrate-Based Observation Encoding**:
  - `substrate.encode_observation()` for full observability
  - `substrate.encode_partial_observation()` for POMDP local windows
  - `substrate.get_observation_dim()` for network architecture
  - Removed 55+ lines of hardcoded grid logic

### Changed (Phase 5 - TASK-002A)

- **BREAKING:** Checkpoint format Version 2 → Version 3
- **BREAKING:** Legacy checkpoints no longer supported (no backward compatibility)
- All position operations now use substrate methods:
  - Position initialization: `substrate.initialize_positions()`
  - Movement application: `substrate.apply_movement()`
  - Distance calculations: `substrate.is_on_position()`
  - Affordance randomization: `substrate.get_all_positions()`
  - Observation encoding: `substrate.encode_observation()` / `encode_partial_observation()`
- Observation dimensions now substrate-aware:
  - Full observability: `substrate.get_observation_dim()` (was `grid_size²`)
  - Partial observability: `substrate.position_dim` (was hardcoded `2`)
  - Enables aspatial universes with `position_dim=0`

### Removed (Phase 5 - TASK-002A)

- Hardcoded 2D position assumptions throughout codebase
- Backward compatibility with Version 2 checkpoints
- Manual grid iteration for affordance randomization
- Hardcoded grid encoding in observation builder (55+ lines)

---

### Added (Phase 1-3 - Repository Cleanup)

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

### Changed

- Replaced print() statements with proper logging in:
  - src/townlet/demo/live_inference.py
  - src/townlet/recording/**main**.py
- Updated pyproject.toml with all top-level dependencies

### Fixed

- README affordance count (14, not 15) - CoffeeShop commented out
- README test count (644+, not 387) with correct recording test count (73)
- README entry point paths now correctly point to scripts/run_demo.py
- Documentation paths now correctly point to docs/manual/ and docs/architecture/
- .gitignore patterns:
  - Changed **pycache**/ to **/**pycache**/ for recursive matching
  - Consolidated database patterns (*.db,*.db-shm, *.db-wal,*.sqlite, *.sqlite3)
  - Removed duplicate patterns
- CI workflow now includes mypy type checking

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
