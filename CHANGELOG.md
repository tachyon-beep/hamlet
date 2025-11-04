# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation structure overhaul with comprehensive docs/README.md
- Observation space documentation in main README
- Complete pyproject.toml metadata (keywords, classifiers, URLs)
- LICENSE file (MIT)
- CHANGELOG.md (this file)

### Fixed
- README affordance count (14, not 15)
- README test count (644+, not 387)
- Documentation paths now correctly point to docs/manual/ and docs/architecture/
- Entry point path corrected to scripts/run_demo.py

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
