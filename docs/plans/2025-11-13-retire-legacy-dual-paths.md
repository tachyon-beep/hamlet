# Implementation Plan: Retire Legacy Dual Code Paths (WP-C2 + WP-C3)

**Task**: WP-C2 (Brain As Code Legacy Deprecation) + WP-C3 (Cascade System Consolidation)
**Priority**: CRITICAL (Blocking Public Release)
**Effort**: 24 hours (8h + 16h, includes audit, testing, deletion)
**Status**: Ready for Implementation
**Created**: 2025-11-13
**Updated**: 2025-11-13 (initial creation)
**Method**: Pre-Release Freedom (see CLAUDE.md "ZERO Backwards Compatibility Required")

**Keywords**: pre-release, breaking changes, technical debt, dual code paths, deprecation, cascade consolidation, Brain As Code, MeterDynamics
**Test Strategy**: Retrofit (update existing tests, no new test files needed)
**Breaking Changes**: YES - Hard break per pre-release policy (zero users = zero backward compatibility)

---

## AI-Friendly Summary (Skim This First!)

**What**: Delete all legacy dual code paths (Brain As Code fallbacks + CascadeEngine) and enforce single source of truth (brain.yaml + MeterDynamics).

**Why**: Pre-release status (zero users) enables clean breaks without backward compatibility burden. Dual paths double testing surface area, confuse developers, and violate "no backwards compatibility" principle.

**How**: Audit all dual path locations → update test fixtures to use modern path → delete legacy code → verify all tests pass.

**Quick Assessment**:

- **Implementation Approach**: Audit → Migrate → Delete (no TDD needed - tests already exist)
- **Test Coverage Goal**: Maintain existing >70% coverage, update ~12 test files
- **Phases**: 4 phases over 24 hours
- **Risk Level**: Low (pre-release freedom, extensive test coverage catches regressions)

**Decision Point**: If you're not implementing WP-C2/C3, STOP READING HERE.

---

## Executive Summary

HAMLET's pre-release status (zero users, zero downloads) provides unique freedom to eliminate technical debt through hard breaks. Two critical dual code path issues require immediate resolution before public release:

**WP-C2 (Brain As Code Legacy)**: Network/optimizer/loss instantiation has dual paths (brain_config vs legacy parameters). Legacy path exists when brain_config=None but has zero test coverage and violates single source of truth principle. This doubles maintenance burden and confuses future developers.

**WP-C3 (Cascade System)**: Both CascadeEngine (legacy config-driven, 331 lines) and MeterDynamics (modern tensor-driven, 187 lines) coexist. Different code paths use different engines, creating operator confusion about which system is active and potential bugs when behavior differs.

**Key Insight**: Pre-release means **freedom to break everything**. No deprecation warnings, no fallback mechanisms, no migration paths. Delete old code immediately and update references. Technical debt for non-existent users is inexcusable.

**Implementation Strategy**: Three-phase cleanup per CLAUDE.md antipatterns:
1. **Phase 1**: Comprehensive audit identifying all dual path locations
2. **Phase 2**: Migrate all references to modern path (brain.yaml, MeterDynamics)
3. **Phase 3**: Delete legacy code entirely (no commented-out code, no feature flags)
4. **Phase 4**: Verify via existing test suite (no new tests needed)

Total deletion: **~914 lines** of obsolete code (583 lines reward strategies already removed in TASK-004C, 331 lines CascadeEngine to remove).

---

## Review-Driven Updates (2025-11-13)

This plan follows the pre-release freedom principle from CLAUDE.md:

### Pre-Release Freedom Principle Applied

**From CLAUDE.md**:
> **THIS PROJECT IS PRE-RELEASE WITH ZERO USERS AND ZERO DOWNLOADS.**
>
> **ABSOLUTE RULES:**
> 1. **NO backwards compatibility arrangements** - Delete old code paths immediately
> 2. **NO fallback mechanisms** - Breaking changes are free and encouraged
> 3. **NO deprecation warnings** - Just break things and update references

**Application to WP-C2/C3**:
- ✅ **Hard break**: Delete legacy code, no "support both old and new"
- ✅ **No fallbacks**: Make brain_config REQUIRED, raise ValueError if None
- ✅ **Update all references**: Fix all test fixtures to use brain.yaml
- ✅ **Delete immediately**: No TODO comments, no feature flags, no "just in case"

---

## Problem Statement

### WP-C2: Brain As Code Dual Initialization Paths

**File**: `src/townlet/population/vectorized.py:143-192`

```python
# CURRENT (dual path - ANTIPATTERN)
def __init__(self, ...):
    if brain_config is not None:
        # Modern path: Use brain.yaml
        self.q_network = NetworkFactory.build(brain_config.architecture)
        self.optimizer = OptimizerFactory.build(brain_config.optimizer)
    else:
        # ❌ LEGACY PATH (zero test coverage!)
        self.q_network = SimpleQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,  # ❌ HARDCODED
        )
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=3e-4,  # ❌ HARDCODED
        )
```

**From CLAUDE.md - ANTIPATTERNS**:
> ❌ **ANTIPATTERN**: `if hasattr(obj, 'old_field')` checks for old vs new attributes
> - Why it's wrong: Maintaining dual code paths for zero users
> - Fix: Delete the old code path, update all references to new field

### WP-C3: Dual Cascade Systems

**Files**:
- `src/townlet/environment/cascade_engine.py` (331 lines, legacy)
- `src/townlet/environment/meter_dynamics.py` (187 lines, modern)

```python
# CURRENT (both systems exist - ANTIPATTERN)
class VectorizedHamletEnv:
    def __init__(self, ...):
        # Modern path uses MeterDynamics
        self.meter_dynamics = MeterDynamics(optimization_data)

        # ❌ LEGACY PATH still instantiated in some code paths
        self.cascade_engine = CascadeEngine(cascades_config)

    def step(self, actions):
        # Some paths call MeterDynamics
        self.meter_dynamics.apply_depletion_and_cascades(...)

        # ❌ Other paths call CascadeEngine
        self.cascade_engine.process_cascades(...)
```

**From Architecture Analysis**:
> **C3. Dual Cascade Systems**
> - **Impact**: Operators unsure which system active, bugs when cascades behave differently than expected
> - **Root Cause**: MeterDynamics introduced to replace CascadeEngine but migration incomplete

### Why This Is Technical Debt

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: **More expressive** (cleaner)

- ✅ Enables confident refactoring (single code path to modify)
- ✅ Enables faster development (half the testing surface area)
- ✅ Enables clearer documentation (one cascade system to explain)
- ✅ Enables operator confidence (no confusion about which system applies)
- ❌ Does NOT make system more fragile (modern path fully tested and production-ready)

**Conclusion**: Dual code paths are **technical debt masquerading as backward compatibility**, but backward compatibility is explicitly rejected per pre-release policy.

---

## Codebase Analysis Summary

### WP-C2: Brain As Code Legacy Path Locations

**Audit Commands**:
```bash
# Find brain_config None checks
grep -rn "brain_config is None\|brain_config=None\|if brain_config" src/townlet/

# Find legacy network_type usage
grep -rn "network_type.*=.*'simple'\|network_type.*=.*'recurrent'" src/townlet/ tests/

# Find hardcoded hyperparameters
grep -rn "hidden_dim=256\|lr=3e-4\|gamma=0.99" src/townlet/

# Find TODO(BRAIN_AS_CODE) comments
grep -rn "TODO.*BRAIN" src/townlet/
```

**Estimated Locations** (from architecture analysis):

| File | Lines | Issue | Change Required |
|------|-------|-------|-----------------|
| **population/vectorized.py** | 143-192 | Dual network init | Delete else branch, require brain_config |
| **agent/network_factory.py** | ~50 | legacy network_type arg | Remove parameter entirely |
| **tests/.../unit/population/** | Multiple | Test fixtures use legacy | Update to use brain.yaml fixtures |
| **tests/.../integration/** | Multiple | Test fixtures use legacy | Update to use brain.yaml fixtures |

**Total**: ~8-12 files requiring updates

### WP-C3: Cascade System Locations

**Audit Commands**:
```bash
# Find CascadeEngine usage
grep -rn "CascadeEngine\|cascade_engine" src/townlet/

# Find MeterDynamics usage
grep -rn "MeterDynamics\|meter_dynamics" src/townlet/

# Find cascade-related imports
grep -rn "from.*cascade_engine import\|from.*meter_dynamics import" src/townlet/
```

**Estimated Locations**:

| File | Lines | Issue | Change Required |
|------|-------|-------|-----------------|
| **environment/cascade_engine.py** | 1-331 | Entire legacy file | DELETE FILE |
| **environment/vectorized_env.py** | ~100-120 | CascadeEngine import/init | Delete, use MeterDynamics only |
| **environment/vectorized_env.py** | ~600-650 | Dual cascade calls | Use MeterDynamics.apply_depletion_and_cascades only |
| **tests/.../test_cascade_engine.py** | All | Tests for legacy | DELETE FILE |
| **tests/.../test_meter_dynamics.py** | All | Verify comprehensive | Ensure covers all cascade scenarios |

**Total**: ~5-8 files requiring updates

---

## Solution Architecture

### Design Principles

1. **Pre-Release Freedom**: Zero backward compatibility, delete immediately
2. **Single Source of Truth**: brain.yaml for networks, MeterDynamics for cascades
3. **Fail-Fast Validation**: Raise ValueError if brain_config=None, no silent fallbacks
4. **Clean Deletion**: No commented code, no TODO markers, no feature flags
5. **Test Coverage Maintained**: Update test fixtures, ensure all tests still pass

### Architecture Changes

**Layer 1: Config Layer (WP-C2)**

- REMOVE: legacy network_type parameter from all config DTOs
- ADD: Validation in PopulationConfig raising ValueError if brain_config path missing
- UPDATE: All test fixtures to provide brain.yaml paths

**Layer 2: Factory Layer (WP-C2)**

- REMOVE: NetworkFactory legacy network_type='simple'/'recurrent' code paths
- REMOVE: Hardcoded hidden_dim, learning_rate, gamma constants
- REQUIRE: brain_config parameter (no default, no None handling)

**Layer 3: Population Layer (WP-C2)**

- REMOVE: VectorizedPopulation.__init__ dual initialization paths
- REQUIRE: brain_config parameter (ValueError if None)
- DELETE: All TODO(BRAIN_AS_CODE) comments

**Layer 4: Environment Layer (WP-C3)**

- REMOVE: CascadeEngine import and instantiation
- USE: MeterDynamics.apply_depletion_and_cascades exclusively
- DELETE: All cascade_engine attribute references

**Layer 5: Test Layer (Both)**

- UPDATE: 12+ test fixtures to use brain.yaml configs
- DELETE: tests/test_townlet/unit/environment/test_cascade_engine.py
- VERIFY: tests/test_townlet/unit/environment/test_meter_dynamics.py covers all scenarios

---

## Implementation Plan

### Phase 0: Comprehensive Audit (2 hours)

#### Goal

Identify ALL locations with dual code paths before starting deletion.

#### 0.1: Run WP-C2 Audit (Brain As Code)

**Location**: Project root

```bash
# Find all brain_config None checks
grep -rn "brain_config is None\|brain_config=None\|if brain_config" \
    src/townlet/ tests/ > /tmp/wpc2_audit_brain_config.txt

# Find all legacy network_type usage
grep -rn "network_type.*=.*['\"]simple\|network_type.*=.*['\"]recurrent" \
    src/townlet/ tests/ > /tmp/wpc2_audit_network_type.txt

# Find hardcoded hyperparameters (legacy path indicators)
grep -rn "hidden_dim=256\|learning_rate=3e-4\|gamma=0.99" \
    src/townlet/ > /tmp/wpc2_audit_hardcoded.txt

# Find TODO comments referencing Brain As Code
grep -rn "TODO.*BRAIN\|FIXME.*brain_config\|XXX.*legacy" \
    src/townlet/ > /tmp/wpc2_audit_todos.txt

# Count total instances
echo "WP-C2 Audit Results:"
echo "brain_config checks: $(wc -l < /tmp/wpc2_audit_brain_config.txt)"
echo "network_type usage: $(wc -l < /tmp/wpc2_audit_network_type.txt)"
echo "hardcoded params: $(wc -l < /tmp/wpc2_audit_hardcoded.txt)"
echo "TODO comments: $(wc -l < /tmp/wpc2_audit_todos.txt)"
```

#### 0.2: Run WP-C3 Audit (Cascade Systems)

**Location**: Project root

```bash
# Find all CascadeEngine references
grep -rn "CascadeEngine\|cascade_engine" \
    src/townlet/ tests/ > /tmp/wpc3_audit_cascade_engine.txt

# Find all MeterDynamics references
grep -rn "MeterDynamics\|meter_dynamics" \
    src/townlet/ tests/ > /tmp/wpc3_audit_meter_dynamics.txt

# Find imports of cascade systems
grep -rn "from.*cascade_engine import\|from.*meter_dynamics import" \
    src/townlet/ tests/ > /tmp/wpc3_audit_imports.txt

# Find test files
find tests/ -name "*cascade*" -type f > /tmp/wpc3_audit_test_files.txt

# Count total instances
echo "WP-C3 Audit Results:"
echo "CascadeEngine refs: $(wc -l < /tmp/wpc3_audit_cascade_engine.txt)"
echo "MeterDynamics refs: $(wc -l < /tmp/wpc3_audit_meter_dynamics.txt)"
echo "Cascade imports: $(wc -l < /tmp/wpc3_audit_imports.txt)"
echo "Test files: $(wc -l < /tmp/wpc3_audit_test_files.txt)"
```

#### 0.3: Document All Findings

Create checklist in `docs/reviews/WP-C2-C3-AUDIT-RESULTS.md`:

```markdown
# WP-C2 + WP-C3 Audit Results

**Date**: 2025-11-13
**Scope**: All dual code path locations

## WP-C2: Brain As Code Legacy Paths

### Source Files
- [ ] src/townlet/population/vectorized.py:143-192 - Dual init path
- [ ] src/townlet/agent/network_factory.py:XX - Legacy network_type
- [ ] (Add all from grep results)

### Test Files
- [ ] tests/test_townlet/unit/population/test_vectorized.py - Update fixtures
- [ ] tests/test_townlet/integration/test_training.py - Update fixtures
- [ ] (Add all from grep results)

## WP-C3: Cascade System Dual Paths

### Files to Delete Entirely
- [ ] src/townlet/environment/cascade_engine.py (331 lines)
- [ ] tests/test_townlet/unit/environment/test_cascade_engine.py (if exists)

### Files to Update
- [ ] src/townlet/environment/vectorized_env.py - Remove CascadeEngine, use MeterDynamics only
- [ ] (Add all from grep results)

## Verification Checklist
- [ ] All grep searches executed and results documented
- [ ] Every file:line_number documented in checklist
- [ ] Estimated total lines to delete: ~600+ lines (331 CascadeEngine + legacy BAC paths)
```

#### Phase 0 Success Criteria

- [ ] All grep commands executed and saved to `/tmp/wpc*_audit_*.txt`
- [ ] Checklist created with every file:line requiring changes
- [ ] Audit results saved to `docs/reviews/WP-C2-C3-AUDIT-RESULTS.md`
- [ ] No false positives in checklist (manual verification)
- [ ] Estimated 8-15 files identified for updates, 2 files for deletion

**Estimated Time**: 2 hours

**Deliverable**: Complete audit checklist documenting all dual path locations

---

## Phase 1: WP-C2 - Deprecate Brain As Code Legacy (8 hours)

### Goal

Remove all legacy brain_config=None fallback paths and require brain.yaml for all training.

### 1.1: Identify Test Fixtures Needing Updates (1 hour)

**Action**: Scan all test files for legacy initialization patterns

```bash
# Find test files using legacy initialization
grep -rn "VectorizedPopulation\|SimpleQNetwork\|RecurrentSpatialQNetwork" \
    tests/test_townlet/ | grep -v "brain_config" > /tmp/legacy_test_patterns.txt

# Find test files with hardcoded network params
grep -rn "hidden_dim=\|learning_rate=\|gamma=" \
    tests/test_townlet/ > /tmp/hardcoded_test_params.txt
```

**Expected**: 10-15 test files needing fixture updates

### 1.2: Create Brain.yaml Test Fixtures (2 hours)

**Files**: `tests/test_townlet/conftest.py`

Add standardized brain.yaml fixtures for all test scenarios:

```python
# =============================================================================
# WP-C2: Brain As Code Test Fixtures (Standard)
# =============================================================================

@pytest.fixture
def minimal_brain_config(tmp_path):
    """Minimal brain.yaml for SimpleQNetwork testing.

    Use for: Unit tests requiring minimal network configuration.
    Architecture: SimpleQNetwork (MLP: obs_dim → 128 → 64 → action_dim)
    """
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text("""
architecture:
  type: simple_q
  hidden_dims: [128, 64]
  activation: relu

optimizer:
  type: adam
  learning_rate: 1e-3
  weight_decay: 0.0

q_learning:
  gamma: 0.99
  use_double_dqn: false
  target_update_frequency: 100

loss:
  type: smooth_l1
  beta: 1.0

replay:
  type: standard
  capacity: 10000
  batch_size: 32
""")
    return brain_yaml


@pytest.fixture
def recurrent_brain_config(tmp_path):
    """Recurrent brain.yaml for LSTM testing.

    Use for: POMDP tests requiring RecurrentSpatialQNetwork.
    Architecture: RecurrentSpatialQNetwork (CNN+LSTM)
    """
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text("""
architecture:
  type: recurrent_spatial_q
  lstm_hidden_size: 256
  activation: relu

optimizer:
  type: adam
  learning_rate: 3e-4
  weight_decay: 1e-5

q_learning:
  gamma: 0.99
  use_double_dqn: true
  target_update_frequency: 200

loss:
  type: huber
  delta: 1.0

replay:
  type: sequential
  capacity: 5000
  batch_size: 16
  sequence_length: 8
""")
    return brain_yaml


@pytest.fixture
def legacy_compatible_brain_config(tmp_path):
    """Brain.yaml matching old hardcoded defaults.

    Use for: Backward compatibility tests (legacy checkpoint loading).
    Matches old hardcoded values: hidden_dim=256, lr=3e-4, gamma=0.99
    """
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text("""
architecture:
  type: simple_q
  hidden_dims: [256, 128]
  activation: relu

optimizer:
  type: adam
  learning_rate: 3e-4  # Old hardcoded default
  weight_decay: 0.0

q_learning:
  gamma: 0.99  # Old hardcoded default
  use_double_dqn: false
  target_update_frequency: 100

loss:
  type: mse

replay:
  type: standard
  capacity: 50000
  batch_size: 64
""")
    return brain_yaml
```

**Test fixture loading**:

```bash
# Verify fixtures load correctly
pytest --collect-only tests/test_townlet/ | grep "brain_config"
```

Expected: 3 new fixtures appear in collection

### 1.3: Update Population Tests (2 hours)

**Files**: `tests/test_townlet/unit/population/test_vectorized.py`

**Before (legacy pattern)**:
```python
def test_population_creates_networks():
    """Population should create Q-networks."""
    population = VectorizedPopulation(
        num_agents=4,
        obs_dim=29,
        action_dim=8,
        device=torch.device("cpu"),
        # ❌ No brain_config - uses legacy path
    )
    assert population.q_network is not None
```

**After (modern pattern)**:
```python
def test_population_creates_networks(device_fixture, minimal_brain_config):
    """Population should create Q-networks from brain.yaml."""
    population = VectorizedPopulation(
        num_agents=4,
        obs_dim=29,
        action_dim=8,
        device=device_fixture,
        brain_config_path=minimal_brain_config,  # ✅ Required
    )
    assert population.q_network is not None
    assert isinstance(population.q_network, SimpleQNetwork)
```

**Repeat for all test files**:
- `tests/test_townlet/unit/population/test_vectorized.py`
- `tests/test_townlet/unit/population/test_training_loop.py`
- `tests/test_townlet/integration/test_training.py`
- `tests/test_townlet/integration/test_checkpointing.py`

**Run tests after each file**:
```bash
pytest tests/test_townlet/unit/population/ -v
```

### 1.4: Remove Legacy Code from VectorizedPopulation (1 hour)

**File**: `src/townlet/population/vectorized.py`

**Delete lines 143-192** (entire else branch):

```python
# BEFORE
def __init__(
    self,
    num_agents: int,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    brain_config: BrainConfig | None = None,  # ❌ Optional
    ...
):
    if brain_config is not None:
        # Modern path
        self.q_network = NetworkFactory.build(brain_config.architecture, ...)
        self.optimizer = OptimizerFactory.build(brain_config.optimizer, ...)
    else:
        # ❌ DELETE THIS ENTIRE BRANCH (lines 160-192)
        self.q_network = SimpleQNetwork(...)
        self.optimizer = torch.optim.Adam(...)
```

**AFTER**:

```python
def __init__(
    self,
    num_agents: int,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    brain_config: BrainConfig,  # ✅ REQUIRED (no None)
    ...
):
    """Initialize vectorized population.

    Args:
        brain_config: Brain configuration (REQUIRED). No fallback to legacy
            hardcoded values. All training must provide brain.yaml.

    Raises:
        ValueError: If brain_config is None (legacy path removed in WP-C2)
    """
    # Validate brain_config provided
    if brain_config is None:
        raise ValueError(
            "brain_config is required. Legacy initialization path removed. "
            "Provide brain.yaml configuration for all training runs. "
            "See docs/config-schemas/brain.md for examples."
        )

    # Modern path only (no dual paths!)
    self.q_network = NetworkFactory.build(
        brain_config.architecture,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    self.optimizer = OptimizerFactory.build(
        brain_config.optimizer,
        parameters=self.q_network.parameters(),
    )

    self.loss_fn = LossFactory.build(brain_config.loss)

    # ... rest of initialization
```

**Delete TODO comments**:

```bash
# Find and remove all TODO(BRAIN_AS_CODE) comments
grep -rn "TODO.*BRAIN" src/townlet/population/ | cut -d: -f1-2
# Manually edit each file to remove TODO lines
```

### 1.5: Remove Legacy Network Type Parameters (1 hour)

**File**: `src/townlet/agent/network_factory.py`

**Remove network_type parameter**:

```python
# BEFORE
class NetworkFactory:
    @staticmethod
    def build(
        architecture_config: ArchitectureConfig | None = None,
        network_type: str | None = None,  # ❌ LEGACY
        obs_dim: int | None = None,
        action_dim: int | None = None,
        ...
    ):
        if architecture_config is not None:
            # Modern path
            return NetworkFactory._build_from_config(architecture_config, ...)
        elif network_type is not None:
            # ❌ LEGACY PATH - DELETE
            if network_type == "simple":
                return SimpleQNetwork(...)
            elif network_type == "recurrent":
                return RecurrentSpatialQNetwork(...)

# AFTER
class NetworkFactory:
    @staticmethod
    def build(
        architecture_config: ArchitectureConfig,  # ✅ REQUIRED
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> nn.Module:
        """Build Q-network from architecture configuration.

        Args:
            architecture_config: Network architecture spec from brain.yaml.
                No fallback to legacy network_type parameter.

        Returns:
            Instantiated neural network.

        Raises:
            ValueError: If architecture_config is None (legacy removed).
        """
        if architecture_config is None:
            raise ValueError(
                "architecture_config required. Legacy network_type parameter removed. "
                "Provide brain.yaml with architecture section."
            )

        return NetworkFactory._build_from_config(
            architecture_config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
```

### 1.6: Run Full Test Suite (1 hour)

**Verify no regressions**:

```bash
# Run all population tests
pytest tests/test_townlet/unit/population/ -v

# Run all agent tests
pytest tests/test_townlet/unit/agent/ -v

# Run integration tests
pytest tests/test_townlet/integration/ -v --tb=short

# Run with coverage
pytest tests/test_townlet/ \
    --cov=townlet.population \
    --cov=townlet.agent \
    --cov-report=term-missing \
    -v
```

**Expected**:
- All tests pass (>70% coverage maintained)
- No legacy code path usage detected
- Coverage for modern path >90%

### 1.7: Commit WP-C2 Changes

```bash
git add -A
git commit -m "feat(wpc2): remove legacy Brain As Code initialization paths

BREAKING CHANGE: brain_config now required for VectorizedPopulation

Changes:
- DELETE: VectorizedPopulation legacy else branch (lines 143-192)
- REQUIRE: brain_config parameter (ValueError if None)
- DELETE: NetworkFactory network_type parameter
- DELETE: Hardcoded hidden_dim=256, lr=3e-4, gamma=0.99
- UPDATE: 12 test fixtures to use brain.yaml configs
- DELETE: All TODO(BRAIN_AS_CODE) comments

Rationale: Pre-release status (zero users) enables clean break.
No backward compatibility needed per CLAUDE.md policy.

Test Coverage: All tests passing, coverage >70% maintained.

Closes: WP-C2 (Architecture Analysis 2025-11-13)
Part of: Sprint 1 Critical Path"
```

### Phase 1 Success Criteria

- [ ] brain_config parameter required (no default=None)
- [ ] ValueError raised if brain_config=None with helpful message
- [ ] All test fixtures use brain.yaml configs
- [ ] Legacy else branches deleted from VectorizedPopulation
- [ ] Legacy network_type parameter deleted from NetworkFactory
- [ ] All TODO(BRAIN_AS_CODE) comments removed
- [ ] Full test suite passes (>70% coverage)
- [ ] No legacy code paths remain (verified via grep)

**Estimated Time**: 8 hours

---

## Phase 2: WP-C3 - Consolidate Cascade Systems (16 hours)

### Goal

Migrate all code paths to MeterDynamics and delete CascadeEngine entirely.

### 2.1: Verify MeterDynamics Feature Parity (3 hours)

**Action**: Comprehensive comparison of CascadeEngine vs MeterDynamics

**Create comparison checklist**:

```markdown
# CascadeEngine vs MeterDynamics Feature Parity

## Core Features

### Cascade Processing
- [ ] MeterDynamics: Applies (source → target) effects
- [ ] MeterDynamics: Respects cascade rates
- [ ] MeterDynamics: Handles multiple cascades per meter
- [ ] Test: `test_meter_dynamics.py::test_cascade_application`

### Circular Dependency Detection
- [ ] CascadeEngine: Detects A→B→A cycles (WHERE IS THIS?)
- [ ] MeterDynamics: Circular detection (VERIFY THIS EXISTS)
- [ ] Test: Need to add if missing

### Tensor Operations
- [ ] MeterDynamics: Uses optimization_data tensors (Stage 6)
- [ ] MeterDynamics: GPU-native operations
- [ ] MeterDynamics: Batched across all agents
- [ ] Test: `test_meter_dynamics.py::test_gpu_tensors`

### Edge Cases
- [ ] Zero cascade rate (no-op)
- [ ] Cascade to non-existent meter (should fail at compile)
- [ ] Cascade from critical meter at 0.0 (should not cascade)
- [ ] Test: Verify all edge cases covered
```

**Run comparison**:

```bash
# Read both implementations
code src/townlet/environment/cascade_engine.py
code src/townlet/environment/meter_dynamics.py

# Find circular detection in CascadeEngine
grep -n "circular\|cycle\|loop" src/townlet/environment/cascade_engine.py

# Verify MeterDynamics has same logic
grep -n "circular\|cycle\|loop" src/townlet/environment/meter_dynamics.py
```

**If MeterDynamics missing features**: Implement before proceeding (out of scope - escalate)

**Expected**: MeterDynamics has full parity (implemented during optimization phase)

### 2.2: Audit All CascadeEngine Call Sites (2 hours)

**Find all usages**:

```bash
# Find CascadeEngine class references
grep -rn "CascadeEngine" src/townlet/ > /tmp/cascade_engine_refs.txt

# Find cascade_engine attribute usage
grep -rn "self.cascade_engine\|env.cascade_engine" src/townlet/ > /tmp/cascade_engine_attrs.txt

# Find imports
grep -rn "from.*cascade_engine import" src/townlet/ > /tmp/cascade_engine_imports.txt

# Display results
echo "=== CascadeEngine References ==="
cat /tmp/cascade_engine_refs.txt
echo "=== Attribute Usage ==="
cat /tmp/cascade_engine_attrs.txt
echo "=== Imports ==="
cat /tmp/cascade_engine_imports.txt
```

**Expected locations**:
1. `src/townlet/environment/vectorized_env.py` - Import and instantiation
2. `src/townlet/environment/vectorized_env.py` - Call sites in step()
3. Potentially other environment modules

**Document each call site**:

```markdown
# CascadeEngine Call Sites

## File: vectorized_env.py

### Import (line ~15)
```python
from townlet.environment.cascade_engine import CascadeEngine  # DELETE
```

### Instantiation (line ~120)
```python
self.cascade_engine = CascadeEngine(cascades_config)  # DELETE
```

### Usage in step() (line ~640)
```python
self.cascade_engine.process_cascades(self.meters)  # REPLACE with MeterDynamics
```

## Migration Strategy
- DELETE: All CascadeEngine imports
- DELETE: cascade_engine attribute initialization
- REPLACE: process_cascades() calls with meter_dynamics.apply_depletion_and_cascades()
```

### 2.3: Migrate VectorizedEnv to MeterDynamics Only (4 hours)

**File**: `src/townlet/environment/vectorized_env.py`

**Step 1: Remove CascadeEngine import and init**

```python
# BEFORE (dual imports)
from townlet.environment.cascade_engine import CascadeEngine  # ❌ DELETE
from townlet.environment.meter_dynamics import MeterDynamics  # ✅ KEEP

class VectorizedHamletEnv:
    def __init__(self, ...):
        # Modern path
        self.meter_dynamics = MeterDynamics(
            optimization_data=compiled_universe.optimization_data,
            num_agents=num_agents,
            device=device,
        )

        # ❌ Legacy path - DELETE THIS
        self.cascade_engine = CascadeEngine(
            cascades_config=compiled_universe.cascades_config,
        )
```

**AFTER**:

```python
# AFTER (single import)
from townlet.environment.meter_dynamics import MeterDynamics

class VectorizedHamletEnv:
    def __init__(self, ...):
        # MeterDynamics only (no dual systems)
        self.meter_dynamics = MeterDynamics(
            optimization_data=compiled_universe.optimization_data,
            num_agents=num_agents,
            device=device,
        )
        # cascade_engine attribute removed entirely
```

**Step 2: Replace all cascade_engine.process_cascades() calls**

Find all calls:

```bash
grep -n "cascade_engine.process_cascades\|cascade_engine.apply" \
    src/townlet/environment/vectorized_env.py
```

**Before**:
```python
def step(self, actions):
    # ... execute actions ...

    # ❌ OLD: Dual cascade systems
    if hasattr(self, 'meter_dynamics'):
        self.meter_dynamics.apply_depletion_and_cascades(
            meters=self.meters,
            depletion_multiplier=depletion_multiplier,
        )
    else:
        self.cascade_engine.process_cascades(self.meters)
```

**After**:
```python
def step(self, actions):
    # ... execute actions ...

    # ✅ NEW: MeterDynamics only
    self.meter_dynamics.apply_depletion_and_cascades(
        meters=self.meters,
        depletion_multiplier=depletion_multiplier,
    )
```

**Step 3: Remove hasattr() checks for cascade systems**

```bash
# Find hasattr checks
grep -n "hasattr.*cascade\|hasattr.*meter_dynamics" \
    src/townlet/environment/vectorized_env.py
```

Delete all `hasattr()` defensive checks (legacy fallback pattern).

**Step 4: Verify no remaining CascadeEngine references**

```bash
# Should return zero results
grep -n "cascade_engine\|CascadeEngine" src/townlet/environment/vectorized_env.py
```

### 2.4: Delete CascadeEngine Implementation (1 hour)

**Files to delete**:

```bash
# Delete cascade_engine.py
git rm src/townlet/environment/cascade_engine.py

# Verify deletion
ls src/townlet/environment/cascade_engine.py
# Expected: No such file or directory
```

**Check for any remaining imports**:

```bash
# Should return zero results (except in deleted file)
grep -rn "from.*cascade_engine import\|import.*cascade_engine" src/townlet/
```

**If imports found**: Fix all import statements before proceeding.

### 2.5: Update/Delete Test Files (3 hours)

**Find cascade-related tests**:

```bash
# Find test files
find tests/test_townlet/ -name "*cascade*" -type f

# Expected:
# tests/test_townlet/unit/environment/test_cascade_engine.py (DELETE)
# tests/test_townlet/unit/environment/test_meter_dynamics.py (VERIFY COMPREHENSIVE)
```

**Action 1: Delete test_cascade_engine.py**

```bash
git rm tests/test_townlet/unit/environment/test_cascade_engine.py
```

**Action 2: Verify test_meter_dynamics.py comprehensive**

```python
# Required test coverage in test_meter_dynamics.py:
class TestMeterDynamics:
    """Comprehensive MeterDynamics tests."""

    def test_depletion_applied(self, device_fixture):
        """Base depletion reduces meters."""
        # Verify meters -= base_depletions × multiplier

    def test_cascades_applied(self, device_fixture):
        """Cascades transfer between meters."""
        # Verify source → target effects

    def test_multiple_cascades_per_meter(self, device_fixture):
        """Multiple cascades to same meter accumulate."""
        # Verify energy → health, hygiene → health both work

    def test_zero_cascade_rate_is_noop(self, device_fixture):
        """Zero cascade rate should not affect meters."""
        # Edge case

    def test_critical_meter_death_stops_cascades(self, device_fixture):
        """Dead agents don't cascade."""
        # Verify dones[i]=True → no cascade from agent i

    def test_batched_across_agents(self, device_fixture):
        """Operations batched across all agents."""
        # Verify GPU vectorization

    def test_uses_optimization_data_tensors(self, device_fixture):
        """Uses pre-computed tensors from compiler Stage 6."""
        # Verify optimization_data.base_depletions used
```

**Run comprehensive test**:

```bash
pytest tests/test_townlet/unit/environment/test_meter_dynamics.py -v --cov=townlet.environment.meter_dynamics --cov-report=term-missing
```

**Expected**: >90% coverage, all edge cases tested

**If coverage gaps found**: Add missing tests before proceeding.

**Action 3: Update integration tests**

```bash
# Find integration tests mentioning cascade
grep -rn "cascade" tests/test_townlet/integration/

# Update any tests checking cascade behavior to verify MeterDynamics
# No dual-system tests should remain
```

### 2.6: Update Documentation (2 hours)

**Files to update**:

1. `docs/config-schemas/cascades.md`
2. `CLAUDE.md`
3. `docs/architecture/TOWNLET_CURRICULUM_V4.md`

**File 1: cascades.md**

```markdown
# Cascade Configuration

**Status**: Production (MeterDynamics implementation)

**Cascades** define meter-to-meter effects where changes in one meter affect others.

## Implementation

**Active System**: MeterDynamics (GPU-native tensor processor)

**Deprecated**: CascadeEngine (removed in WP-C3, 2025-11-13)

Cascades are processed by `MeterDynamics.apply_depletion_and_cascades()`:
1. Apply base depletion: `meters -= base_depletions × depletion_multiplier`
2. Apply cascades: For each cascade, `meters[:, target] += meters[:, source] × rate`
3. Clamp meters to [0.0, 1.0]

Cascades use pre-computed tensors from UAC Stage 6 (optimization_data.cascade_data).

## Configuration
...
```

**File 2: CLAUDE.md**

Add to "Breaking Changes" section:

```markdown
## Recent Breaking Changes (2025-11-13)

### WP-C3: Cascade System Consolidation (2025-11-13)

**Change**: CascadeEngine deleted, MeterDynamics only

**Breaking**:
- `src/townlet/environment/cascade_engine.py` DELETED (331 lines)
- All cascade processing uses MeterDynamics.apply_depletion_and_cascades()
- No dual cascade systems

**Rationale**: Pre-release freedom eliminates technical debt. Modern GPU-native
tensor processor (MeterDynamics) fully tested and production-ready.

**Migration**: None needed (MeterDynamics already active in all curriculum levels)
```

**File 3: TOWNLET_CURRICULUM_V4.md**

Update cascade system references:

```bash
# Find cascade system mentions
grep -n "CascadeEngine\|cascade_engine" docs/architecture/TOWNLET_CURRICULUM_V4.md

# Replace all with MeterDynamics
```

### 2.7: Run Full Test Suite (1 hour)

**Comprehensive verification**:

```bash
# Run all environment tests
pytest tests/test_townlet/unit/environment/ -v

# Run cascade-specific tests
pytest tests/test_townlet/unit/environment/test_meter_dynamics.py -v --cov=townlet.environment.meter_dynamics --cov-report=term-missing

# Run integration tests
pytest tests/test_townlet/integration/ -v --tb=short

# Run full suite
pytest tests/test_townlet/ -v --tb=short

# Check coverage
pytest tests/test_townlet/ \
    --cov=townlet.environment \
    --cov-report=html \
    --cov-report=term-missing
```

**Expected**:
- All tests pass
- No CascadeEngine references remain
- MeterDynamics coverage >90%
- Integration tests verify cascade behavior

### 2.8: Commit WP-C3 Changes

```bash
git add -A
git commit -m "feat(wpc3): consolidate cascade systems to MeterDynamics

BREAKING CHANGE: CascadeEngine deleted, MeterDynamics only

Changes:
- DELETE: src/townlet/environment/cascade_engine.py (331 lines)
- DELETE: tests/test_townlet/unit/environment/test_cascade_engine.py
- REMOVE: CascadeEngine imports and instantiation from VectorizedEnv
- REMOVE: All hasattr() checks for dual cascade systems
- UPDATE: All cascade processing uses MeterDynamics.apply_depletion_and_cascades()
- UPDATE: Documentation (cascades.md, CLAUDE.md, curriculum)
- VERIFY: test_meter_dynamics.py comprehensive (>90% coverage)

Rationale: Pre-release status (zero users) enables clean break.
MeterDynamics is modern GPU-native tensor processor, fully tested.
Eliminates operator confusion about which cascade system is active.

Test Coverage: All tests passing, MeterDynamics coverage >90%.

Closes: WP-C3 (Architecture Analysis 2025-11-13)
Part of: Sprint 1 Critical Path"
```

### Phase 2 Success Criteria

- [ ] CascadeEngine.py deleted (331 lines removed)
- [ ] test_cascade_engine.py deleted
- [ ] All CascadeEngine imports removed
- [ ] VectorizedEnv uses MeterDynamics only (no dual systems)
- [ ] All hasattr() cascade checks removed
- [ ] test_meter_dynamics.py comprehensive (>90% coverage)
- [ ] Documentation updated (3 files)
- [ ] Full test suite passes
- [ ] Zero CascadeEngine references remain (verified via grep)

**Estimated Time**: 16 hours

---

## Phase 3: Final Verification (2 hours)

### Goal

Verify complete elimination of all dual code paths and ensure no regressions.

### 3.1: Comprehensive Grep Audit (30 min)

**Verify zero legacy references**:

```bash
# WP-C2: Brain As Code
grep -rn "brain_config is None\|brain_config=None" src/townlet/
# Expected: Zero results (only ValueError if None)

grep -rn "network_type.*=.*['\"]simple\|network_type.*=.*['\"]recurrent" src/townlet/
# Expected: Zero results

grep -rn "hidden_dim=256\|learning_rate=3e-4\|gamma=0.99" src/townlet/
# Expected: Zero results (only in comments/docs if any)

grep -rn "TODO.*BRAIN" src/townlet/
# Expected: Zero results

# WP-C3: Cascade Systems
grep -rn "CascadeEngine\|cascade_engine" src/townlet/
# Expected: Zero results

grep -rn "hasattr.*cascade\|hasattr.*meter_dynamics" src/townlet/
# Expected: Zero results

# Count deleted lines
echo "Lines deleted from CascadeEngine: 331"
echo "Lines deleted from legacy BAC paths: ~200-300"
echo "Total lines deleted: ~531-631"
```

### 3.2: Run Entire Test Suite (1 hour)

**Full test run with coverage**:

```bash
# Run all tests with verbose output
pytest tests/test_townlet/ -v --tb=short

# Generate coverage report
pytest tests/test_townlet/ \
    --cov=townlet \
    --cov-report=html \
    --cov-report=term-missing \
    -v

# Open coverage report
open htmlcov/index.html

# Verify specific modules
echo "=== Coverage Verification ==="
echo "population/vectorized.py: Should be >70%"
echo "agent/network_factory.py: Should be >80%"
echo "environment/meter_dynamics.py: Should be >90%"
echo "environment/vectorized_env.py: Should be >70%"
```

**Expected**:
- All tests pass (0 failures)
- Overall coverage >70%
- No skipped tests related to WP-C2/C3
- No warnings about legacy code paths

### 3.3: Integration Test with All Curriculum Levels (30 min)

**Verify all configs work**:

```bash
# Test each curriculum level loads and runs
for level in L0_0_minimal L0_5_dual_resource L1_full_observability L2_partial_observability L3_temporal_mechanics; do
    echo "Testing $level..."
    python -c "
from pathlib import Path
from townlet.universe.compiler import UniverseCompiler

compiler = UniverseCompiler()
universe = compiler.compile(Path('configs/$level'))
print(f'✓ {level}: obs_dim={universe.metadata.obs_dim}, action_dim={universe.metadata.action_dim}')
"
done
```

**Expected output**:
```
Testing L0_0_minimal...
✓ L0_0_minimal: obs_dim=29, action_dim=8
Testing L0_5_dual_resource...
✓ L0_5_dual_resource: obs_dim=29, action_dim=8
...
```

### 3.4: Document Completion (30 min)

**Update work package status**:

Create `docs/reviews/WP-C2-C3-COMPLETION-REPORT.md`:

```markdown
# WP-C2 + WP-C3 Completion Report

**Date**: 2025-11-13
**Status**: COMPLETE
**Total Effort**: 24 hours (8h WP-C2 + 16h WP-C3)

## Summary

Successfully eliminated all dual code paths from HAMLET codebase per pre-release
freedom principle. Zero users = zero backward compatibility burden.

## Changes Made

### WP-C2: Brain As Code Legacy Deprecation

**Lines Deleted**: ~250-300 lines
- VectorizedPopulation legacy else branch (lines 143-192)
- NetworkFactory network_type parameter and legacy path
- Hardcoded hyperparameters (hidden_dim=256, lr=3e-4, gamma=0.99)
- All TODO(BRAIN_AS_CODE) comments

**Files Modified**: 8 files
- src/townlet/population/vectorized.py
- src/townlet/agent/network_factory.py
- tests/test_townlet/conftest.py (added brain.yaml fixtures)
- tests/test_townlet/unit/population/*.py (5 test files updated)

**Breaking Changes**:
- brain_config parameter now REQUIRED (ValueError if None)
- network_type parameter removed from NetworkFactory
- All training requires brain.yaml configuration

**Test Coverage**: All tests passing, coverage >70% maintained

### WP-C3: Cascade System Consolidation

**Lines Deleted**: 331 lines
- src/townlet/environment/cascade_engine.py (entire file)
- tests/test_townlet/unit/environment/test_cascade_engine.py (entire file)

**Files Modified**: 5 files
- src/townlet/environment/vectorized_env.py
- tests/test_townlet/unit/environment/test_meter_dynamics.py
- docs/config-schemas/cascades.md
- docs/architecture/TOWNLET_CURRICULUM_V4.md
- CLAUDE.md

**Breaking Changes**:
- CascadeEngine class deleted
- All cascade processing uses MeterDynamics.apply_depletion_and_cascades()
- No dual cascade system references remain

**Test Coverage**: test_meter_dynamics.py >90%, all tests passing

## Verification

### Grep Audit Results
- brain_config=None checks: 0 found ✓
- network_type legacy usage: 0 found ✓
- CascadeEngine references: 0 found ✓
- hasattr() cascade checks: 0 found ✓
- TODO(BRAIN) comments: 0 found ✓

### Test Results
- Total tests: XXX
- Passed: XXX
- Failed: 0 ✓
- Coverage: >70% ✓

### Integration Tests
- All 5 curriculum levels load successfully ✓
- All configs compile without errors ✓
- No legacy code path warnings ✓

## Total Lines Deleted

- WP-C2 (BAC Legacy): ~250-300 lines
- WP-C3 (CascadeEngine): 331 lines
- **Total: ~581-631 lines**

## Impact

**Before**: Dual code paths in 2 critical subsystems, confusing documentation,
doubled testing burden, unclear which system active.

**After**: Single source of truth enforced, clean codebase, clear operator
guidance, simplified testing, ready for public release.

## Lessons Learned

1. **Pre-Release Freedom Works**: Zero backward compatibility burden enabled
   aggressive cleanup without fear of breaking users.

2. **Test Fixtures Critical**: Updating 12 test fixtures took 2 hours but
   enabled confident deletion of legacy code.

3. **Grep Audit Essential**: Comprehensive audit (Phase 0) found all dual path
   locations, preventing missed legacy code.

4. **Documentation Updates Matter**: Updating 5 docs files ensures operators
   know legacy paths removed, preventing confusion.

## Recommendations

1. **Apply Same Pattern to Future Cleanup**: Use pre-release freedom to
   eliminate other dual paths (if any found).

2. **Monitor for New Dual Paths**: Code review should catch any new
   `if X is None: fallback` patterns introduced.

3. **Maintain Test Fixture Quality**: Keep conftest.py fixtures well-organized
   as project grows.

## Sign-Off

- [ ] All tests passing
- [ ] Coverage >70%
- [ ] Grep audit clean (zero legacy refs)
- [ ] Documentation updated
- [ ] Integration tests pass
- [ ] Ready for Sprint 2 (WP-M2, WP-M4)

**Completed By**: [Name]
**Date**: 2025-11-13
```

### Phase 3 Success Criteria

- [ ] Grep audit returns zero legacy references
- [ ] Full test suite passes (0 failures)
- [ ] Coverage >70% maintained
- [ ] All 5 curriculum levels load successfully
- [ ] Completion report created and reviewed
- [ ] Ready for Sprint 2 medium-priority work packages

**Estimated Time**: 2 hours

---

## Success Criteria (Overall)

### WP-C2: Brain As Code Legacy Removed

- [ ] brain_config parameter required in VectorizedPopulation (no default=None)
- [ ] ValueError raised if brain_config=None with helpful error message
- [ ] Legacy else branch deleted from population/vectorized.py (lines 143-192)
- [ ] network_type parameter removed from NetworkFactory
- [ ] Hardcoded hyperparameters removed (hidden_dim, lr, gamma)
- [ ] All TODO(BRAIN_AS_CODE) comments deleted
- [ ] 12 test fixtures updated to use brain.yaml configs
- [ ] All tests passing, coverage >70%

### WP-C3: Cascade System Consolidated

- [ ] cascade_engine.py deleted (331 lines removed)
- [ ] test_cascade_engine.py deleted
- [ ] CascadeEngine imports removed from all files
- [ ] VectorizedEnv uses MeterDynamics only (no cascade_engine attribute)
- [ ] All hasattr() cascade checks removed
- [ ] test_meter_dynamics.py comprehensive (>90% coverage)
- [ ] Documentation updated (cascades.md, CLAUDE.md, curriculum docs)
- [ ] All tests passing

### Integration Verification

- [ ] Full test suite passes (0 failures)
- [ ] Overall coverage >70% maintained
- [ ] All 5 curriculum levels load without errors
- [ ] Grep audit clean (zero legacy references)
- [ ] Completion report created
- [ ] No warnings about legacy code paths

---

## Risks & Mitigations

### Risk 1: Breaking Existing Notebooks/Scripts

**Likelihood**: Medium
**Impact**: Low (pre-release, zero external users)
**Mitigation**:

- Pre-release policy explicitly allows breaking changes
- No external users to impact (zero downloads)
- Internal scripts updated as part of implementation
- Documentation clearly marks breaking changes

### Risk 2: Missing Dual Path Locations

**Likelihood**: Low
**Impact**: Medium (runtime failures if missed)
**Mitigation**:

- Comprehensive grep audit (Phase 0) finds all locations
- Multiple search patterns (brain_config, network_type, CascadeEngine)
- Test suite catches any missed references (runtime errors)
- Final verification phase (Phase 3) re-runs audit

### Risk 3: Test Fixture Migration Errors

**Likelihood**: Medium
**Impact**: Medium (tests fail, easy to fix)
**Mitigation**:

- Create brain.yaml fixtures in conftest.py first (Phase 1.2)
- Update tests incrementally, verify after each file
- Run test suite after each migration step
- Comprehensive fixtures (minimal, recurrent, legacy-compatible)

### Risk 4: MeterDynamics Missing Features

**Likelihood**: Low
**Impact**: High (if true, blocks WP-C3)
**Mitigation**:

- Feature parity verification step (Phase 2.1)
- Compare CascadeEngine vs MeterDynamics capabilities
- Escalate if MeterDynamics missing critical features
- Architecture analysis indicates parity already achieved

---

## Estimated Effort

| Phase | Description | Time | Notes |
|-------|-------------|------|-------|
| **Phase 0** | Comprehensive audit | 2h | Find all dual paths |
| **Phase 1** | WP-C2 - BAC legacy removal | 8h | brain.yaml required |
| **Phase 2** | WP-C3 - Cascade consolidation | 16h | Delete CascadeEngine |
| **Phase 3** | Final verification | 2h | Grep audit + tests |
| **Total** | | **28h** | Original estimate: 24h |

**Note**: 4 hour variance due to comprehensive verification (Phase 3) and documentation updates. Within acceptable range for critical work packages.

---

## Follow-Up Work (Post-Implementation)

1. **Monitor for New Dual Paths**:
   - Code review checklist: "Does this introduce dual code path?"
   - Pre-commit hook: Warn on `if X is None:` with fallback logic
   - Architecture reviews: Watch for new "support both" patterns

2. **Apply Pattern to Other Dual Paths** (if found):
   - Use same audit → migrate → delete → verify workflow
   - Leverage pre-release freedom while available
   - Document breaking changes in CLAUDE.md

3. **Update Contributor Guide**:
   - Add "Pre-Release Freedom Principle" section
   - Explain when to break vs when to maintain compatibility
   - Provide examples of acceptable breaking changes

4. **Sprint 2 Medium Priority** (next):
   - WP-M2: Consolidate POMDP Validation (8h)
   - WP-M4: Refactor Intrinsic Reward Coordination (16h)

---

## Running Tests

### Run All WP-C2 Tests

```bash
# Brain As Code tests
pytest tests/test_townlet/unit/population/ \
       tests/test_townlet/unit/agent/ \
       -v --tb=short

# Integration tests
pytest tests/test_townlet/integration/ -v --tb=short
```

### Run All WP-C3 Tests

```bash
# MeterDynamics tests
pytest tests/test_townlet/unit/environment/test_meter_dynamics.py \
       -v --cov=townlet.environment.meter_dynamics --cov-report=term-missing

# Environment tests
pytest tests/test_townlet/unit/environment/ -v
```

### Run Full Suite with Coverage

```bash
pytest tests/test_townlet/ \
       --cov=townlet.population \
       --cov=townlet.agent \
       --cov=townlet.environment \
       --cov-report=html \
       --cov-report=term-missing \
       -v
```

### Quick Smoke Test

```bash
# Verify brain_config required
python -c "
from townlet.population.vectorized import VectorizedPopulation
try:
    pop = VectorizedPopulation(num_agents=4, obs_dim=29, action_dim=8, device='cpu')
    print('❌ FAIL: Should require brain_config')
except ValueError as e:
    print(f'✓ PASS: {e}')
"

# Verify CascadeEngine deleted
python -c "
try:
    from townlet.environment.cascade_engine import CascadeEngine
    print('❌ FAIL: CascadeEngine should be deleted')
except ImportError:
    print('✓ PASS: CascadeEngine deleted')
"

# Verify MeterDynamics works
python -c "
from townlet.environment.meter_dynamics import MeterDynamics
print('✓ PASS: MeterDynamics importable')
"
```

---

## Conclusion

This implementation plan eliminates all dual code paths from HAMLET's codebase, leveraging pre-release freedom to make clean breaks without backward compatibility burden. Total deletion: **~581-631 lines** of obsolete code.

**Total effort**: 24-28 hours (2-3 days with testing)
**Risk**: Low (comprehensive audit + extensive test coverage)
**Priority**: CRITICAL (blocking public release per architecture analysis)
**Impact**: Cleaner codebase, faster development, clear operator guidance

**Next Steps**:
1. Begin Phase 0 - Comprehensive Audit (2h)
2. Execute Phase 1 - WP-C2 Brain As Code Deprecation (8h)
3. Execute Phase 2 - WP-C3 Cascade Consolidation (16h)
4. Complete Phase 3 - Final Verification (2h)

**Slogan**: "Pre-Release Freedom: Delete First, Ask Questions Never"

---

**Approval**: Ready for implementation ✓

**Prerequisites**: None (can start immediately)

**Blockers**: None

**Dependencies**: All test infrastructure exists, brain.yaml configs already in production (TASK-005 complete), MeterDynamics already implemented (optimization phase complete)
