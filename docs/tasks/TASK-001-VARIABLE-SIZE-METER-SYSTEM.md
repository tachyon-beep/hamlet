# TASK-001: Variable-Size Meter System

**Status**: ✅ Complete (PR #1)
**Priority**: High
**Type**: Feature - UNIVERSE_AS_CODE Foundation
**Assignee**: Claude Code
**Started**: 2025-11-04
**Completed**: 2025-11-04

## Overview

Implement variable-size meter system to support 1-32 meters instead of hardcoded 8 meters. This is foundational work for UNIVERSE_AS_CODE philosophy - enabling universes with different meter configurations.

**Current State**: System hardcodes 8 meters throughout (energy, hygiene, satiation, money, health, fitness, mood, social)

**Desired State**: System accepts 1-32 meters defined in `bars.yaml`, with dynamic tensor sizing and observation dimensions

## Problem Statement

The current implementation had 35+ locations with hardcoded assumptions about 8 meters:

- `BarsConfig` validation required exactly 8 meters
- Tensor allocations used `torch.zeros(8)`
- Observation dimension calculations used `+ 8` for meters
- Checkpoint system didn't validate meter count compatibility
- RewardStrategy assumed energy at index 0, health at index 6
- Action masking used hardcoded indices for energy/health
- Action costs used hardcoded 8-element tensors
- Recurrent networks hardcoded `num_meters=8`

This prevented:

- Creating minimal universes (e.g., 4 meters for pedagogy)
- Creating extended universes (e.g., 12+ meters for research)
- Transfer learning across different meter configurations
- Full UNIVERSE_AS_CODE implementation

## Objectives

### Primary Goals

- [x] Update config schema to accept 1-32 meters
- [x] Implement dynamic tensor sizing in engine layer
- [x] Calculate observation dimensions dynamically
- [x] Add checkpoint metadata validation
- [x] Fix all hardcoded meter indices (action masking, costs, recurrent networks)

### Success Criteria

- [x] BarsConfig validates 1-32 meters (was: exactly 8)
- [x] All tensors sized dynamically based on `meter_count`
- [x] Observation dimension computed from `meter_count`
- [x] Checkpoints include meter metadata and validate on load
- [x] All existing tests pass
- [x] New tests cover 4-meter, 12-meter, 32-meter configs
- [x] Recurrent networks work with variable meters
- [x] All Codex review bugs fixed

## Pull Request

**PR #1**: https://github.com/tachyon-beep/hamlet/pull/1

### Implementation Summary

- **Total Tests**: 35 (all passing)
- **Files Modified**: 12 (6 source, 6 test/docs)
- **Commits**: 9 (TDD RED-GREEN-REFACTOR + Codex fixes)

### Key Changes

1. Config schema accepts 1-32 meters
2. Dynamic tensor sizing throughout
3. Checkpoint metadata validation
4. RewardStrategy variable meter support
5. Action masking uses dynamic meter indices
6. Action costs use dynamic tensors
7. Recurrent networks use dynamic meter count
8. Comprehensive test coverage (35 tests)

### Codex Review Fixes (Post-Implementation)

Three critical bugs discovered and fixed:

1. **Action masking** (a75e5e5): Used hardcoded energy/health indices → IndexError on 4-meter configs
2. **Action costs** (de4dd24): Used hardcoded 8-element cost tensors → RuntimeError on 4-meter configs
3. **Recurrent networks** (6c5b32d): Used hardcoded `num_meters=8` → Wrong features parsed from observations

## Documentation

- **Planning**: [`docs/plans/plan-task-001-variable-size-meters-tdd-ready.md`](../plans/plan-task-001-variable-size-meters-tdd-ready.md)
- **Audit**: [`docs/reviews/TASK-001-AUDIT-RESULTS.md`](../reviews/TASK-001-AUDIT-RESULTS.md)
- **Status**: [`docs/status/TASK-001-STATUS-2025-11-04.md`](../status/TASK-001-STATUS-2025-11-04.md)

## Test Coverage

```bash
# Run all TASK-001 tests
pytest tests/test_townlet/unit/environment/test_variable_meters.py \
       tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py \
       tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints \
       tests/test_townlet/integration/test_variable_meters_e2e.py -v

# Expected: 35 tests passed in 5.92s
```

**Test Breakdown:**

- 12 config validation tests
- 11 engine dynamic sizing tests
- 4 checkpoint metadata tests
- 8 integration tests (including action masking, costs, recurrent networks)

## Success Metrics

- ✅ **Code Quality**: 35 new tests, all Codex review bugs fixed
- ✅ **Functionality**: 1-32 meters supported, dynamic obs_dim, checkpoint validation, recurrent networks
- ✅ **Documentation**: Planning, audit, status reports complete
- ✅ **Code Review**: 3 Codex bugs fixed (action masking, action costs, recurrent networks)
- ✅ **Pre-0.1 Status**: Fix-on-fail approach, breaking changes acceptable

**Status**: ✅ **COMPLETE** - Ready for review and merge

---

**Related**: PR #1, TASK-000 (UAC Action Space), TASK-002 (Universe Compilation)
