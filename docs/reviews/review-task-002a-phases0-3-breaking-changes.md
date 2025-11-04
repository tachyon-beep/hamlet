# Phases 0-3 Review: Breaking Changes Authorization

**Date**: 2025-11-05
**Reviewer**: Claude Code
**Status**: Complete

---

## Requirement Change

User explicitly authorizes BREAKING CHANGES.

**Previous Requirement**: Maintain backward compatibility with configs lacking substrate.yaml
**New Requirement**: substrate.yaml is MANDATORY - fail fast if missing

**Impact**:
- NO backward compatibility required
- Configs without substrate.yaml will FAIL (not fallback to legacy mode)
- Deprecation warnings can be removed
- Legacy mode code can be deleted

---

## Phase-by-Phase Analysis

### Phase 0: Research Validation

**Impact**: **NONE**

**Analysis**:
Phase 0 only validates research findings through spot-checks. It performs no backward compatibility logic or legacy code paths. No changes needed.

**Tasks**:
- Task 0.1: Verify Research Findings ✓ No changes

---

### Phase 1: Substrate Interface

**Impact**: **NONE**

**Analysis**:
Phase 1 creates new code (abstract `SpatialSubstrate`, `Grid2DSubstrate`, `AspatialSubstrate`). Since it's greenfield development with no integration to existing systems, there are no backward compatibility concerns.

**Tasks**:
- Task 1.1: Create Substrate Module Structure ✓ No changes
- Task 1.2: Implement Grid2DSubstrate ✓ No changes
- Task 1.3: Implement AspatialSubstrate ✓ No changes

---

### Phase 2: Configuration Schema

**Impact**: **NONE**

**Analysis**:
Phase 2 creates Pydantic schemas (`SubstrateConfig`) and factory (`SubstrateFactory`). This is new infrastructure with no legacy code paths. No changes needed.

**Tasks**:
- Task 2.1: Create Substrate Config Schema ✓ No changes
- Task 2.2: Create Substrate Factory ✓ No changes

---

### Phase 3: Environment Integration

**Impact**: **SIGNIFICANT**

**Analysis**:
Task 3.1 is the ONLY location in Phases 0-3 with backward compatibility logic. The current plan includes:
1. **Legacy mode fallback** when substrate.yaml is missing
2. **Deprecation warning** to alert users
3. **Backward compatibility tests** to ensure legacy behavior works

With breaking changes authorization, ALL of this can be removed.

---

## Specific Changes Required

### Task 3.1: Add Substrate to VectorizedEnv (Load Only)

**Current Approach**: Falls back to legacy mode if substrate.yaml missing (lines 1372-1406)

**Current Code** (from plan):
```python
# Load substrate configuration (if exists)
substrate_config_path = config_pack_path / "substrate.yaml"
if substrate_config_path.exists():
    from townlet.substrate.config import load_substrate_config
    from townlet.substrate.factory import SubstrateFactory

    substrate_config = load_substrate_config(substrate_config_path)
    self.substrate = SubstrateFactory.build(substrate_config, device=self.device)

    # Update grid_size from substrate (for backward compatibility)
    if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
        if self.substrate.width != self.substrate.height:
            raise ValueError(
                f"Non-square grids not yet supported: "
                f"{self.substrate.width}×{self.substrate.height}"
            )
        self.grid_size = self.substrate.width
else:
    # Legacy mode: No substrate.yaml, use hardcoded behavior
    import warnings
    warnings.warn(
        f"No substrate.yaml found in {config_pack_path}. "
        f"Using legacy hardcoded grid substrate (grid_size={self.grid_size}). "
        f"This will be deprecated in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    self.substrate = None  # Legacy mode marker
```

---

**Revised Approach**: REQUIRE substrate.yaml, fail fast if missing

**Revised Code**:
```python
# Load substrate configuration (REQUIRED)
substrate_config_path = config_pack_path / "substrate.yaml"
if not substrate_config_path.exists():
    raise FileNotFoundError(
        f"substrate.yaml is required but not found in {config_pack_path}. "
        f"All config packs must define their spatial substrate. "
        f"See docs/examples/substrate.yaml for template."
    )

from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory

substrate_config = load_substrate_config(substrate_config_path)
self.substrate = SubstrateFactory.build(substrate_config, device=self.device)

# Update grid_size from substrate (for backward compatibility with other code)
if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
    if self.substrate.width != self.substrate.height:
        raise ValueError(
            f"Non-square grids not yet supported: "
            f"{self.substrate.width}×{self.substrate.height}"
        )
    self.grid_size = self.substrate.width
```

**Key Changes**:
1. ❌ **Removed**: `if substrate_config_path.exists()` check (now required)
2. ❌ **Removed**: `else` branch with legacy mode
3. ❌ **Removed**: Deprecation warning
4. ❌ **Removed**: `self.substrate = None` legacy mode marker
5. ✅ **Added**: Fail-fast error with helpful message
6. ✅ **Kept**: grid_size sync (needed for Phase 4 compatibility, will be removed in Phase 4)

---

### Task 3.1 Step 4: Test Backward Compatibility

**Current Plan** (lines 1409-1432):

```bash
# Test backward compatibility (no substrate.yaml)
python -c "
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
import warnings

# L0 doesn't have substrate.yaml yet
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    env = VectorizedHamletEnv(
        config_pack_path=Path('configs/L0_0_minimal'),
        num_agents=1,
        device='cpu',
    )

    assert len(w) == 1
    assert 'No substrate.yaml found' in str(w[0].message)
    assert env.substrate is None  # Legacy mode
    assert env.grid_size == 3  # From training.yaml
    print('✓ Backward compatibility works (legacy mode)')
"
```

**Revised Plan**: **DELETE THIS TEST ENTIRELY**

**Rationale**: This test validates legacy mode fallback behavior, which no longer exists. With breaking changes, missing substrate.yaml is an ERROR, not a warning.

**New Test** (optional, for error validation):

```bash
# Test that missing substrate.yaml fails fast
python -c "
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv

# L0 without substrate.yaml should raise FileNotFoundError
try:
    env = VectorizedHamletEnv(
        config_pack_path=Path('configs/L0_0_minimal'),
        num_agents=1,
        device='cpu',
    )
    print('✗ Should have raised FileNotFoundError!')
    exit(1)
except FileNotFoundError as e:
    assert 'substrate.yaml is required' in str(e)
    print('✓ Correctly fails fast when substrate.yaml missing')
"
```

---

### Task 3.1 Step 5: Commit Message

**Current Commit Message**:
```
feat: add substrate loading to VectorizedHamletEnv

Environment now loads substrate.yaml if present:
- Creates substrate instance via SubstrateFactory
- Updates grid_size from substrate (for compatibility)
- Falls back to legacy mode if substrate.yaml missing

Backward compatibility: Emits deprecation warning if no substrate.yaml.

Part of TASK-000 (Configurable Spatial Substrates).
```

**Revised Commit Message**:
```
feat: add substrate loading to VectorizedHamletEnv (BREAKING CHANGE)

Environment now REQUIRES substrate.yaml:
- Creates substrate instance via SubstrateFactory
- Updates grid_size from substrate (temporary, Phase 4 will remove)
- FAILS FAST if substrate.yaml missing (no legacy fallback)

BREAKING CHANGE: All config packs must now include substrate.yaml.
Configs without substrate.yaml will raise FileNotFoundError.

Part of TASK-002A Phase 3 (Configurable Spatial Substrates).
```

---

## Updated Effort Estimates

### Phase 3 Original Estimate
- Task 3.1: **2 hours** (includes backward compatibility logic + tests)

### Phase 3 Revised Estimate
- Task 3.1: **1.5 hours** (simpler code, no legacy paths)

### Savings
- **-0.5 hours** (25% reduction)

**Rationale**:
- Removed: `if/else` legacy branch (~10 lines)
- Removed: Deprecation warning import + warning call
- Removed: Legacy mode test script
- Removed: Mental overhead of maintaining two code paths
- Simplified: Fail-fast error is clearer than conditional warning

---

## New Risks Introduced

### Risk 1: Config Migration Required

**Description**: Existing config packs without substrate.yaml will immediately break.

**Mitigation**:
- Phase 6 already planned to "Create substrate.yaml for existing configs"
- Move Phase 6 config creation BEFORE Phase 3 integration
- Ensures all configs have substrate.yaml before environment requires it

**Revised Phase Order**:
1. Phase 0-2: Substrate infrastructure (no integration) ✓
2. **NEW: Phase 6 (moved earlier)**: Create substrate.yaml for all config packs
3. Phase 3: Environment integration (now safe, all configs have substrate.yaml)
4. Phase 4: Position management refactoring
5. Phase 5: Observation/network updates

---

### Risk 2: Developer Experience Impact

**Description**: New developers cloning repo and running training will see cryptic errors if substrate.yaml missing.

**Mitigation**:
- Error message includes helpful guidance: "See docs/examples/substrate.yaml for template"
- Add substrate.yaml to config pack requirements in CLAUDE.md
- Add pre-training validation script: `scripts/validate_configs.py`

**Example Validation Script**:
```python
"""Validate all config packs have required files."""
from pathlib import Path

REQUIRED_FILES = [
    "bars.yaml",
    "cascades.yaml",
    "affordances.yaml",
    "cues.yaml",
    "training.yaml",
    "substrate.yaml",  # NEW REQUIREMENT
]

for config_dir in Path("configs").iterdir():
    if not config_dir.is_dir():
        continue

    for required in REQUIRED_FILES:
        if not (config_dir / required).exists():
            print(f"❌ {config_dir.name} missing {required}")
            exit(1)

print("✅ All config packs valid")
```

---

### Risk 3: No Migration Path for Old Checkpoints

**Description**: Checkpoints saved before substrate implementation won't load (already a risk in Phase 4).

**Mitigation**:
- This is EXPECTED and ACCEPTED with breaking changes authorization
- Phase 4 already documents checkpoint incompatibility
- Users must delete old checkpoints and retrain (documented in Phase 4)

**No action needed** - risk already accepted.

---

## Recommendations

### 1. Update Phase 3 Plan Document

**File**: `docs/plans/plan-task-002a-configurable-spatial-substrates.md`

**Lines to Update**:

**Line 1372-1406** (Task 3.1 Step 3):
- Replace legacy fallback code with fail-fast code (see "Revised Code" above)
- Add FileNotFoundError with helpful message
- Remove deprecation warning

**Line 1409-1432** (Task 3.1 Step 4):
- DELETE backward compatibility test
- OPTIONALLY add fail-fast error test (see "New Test" above)

**Line 1436-1450** (Task 3.1 Step 5):
- Update commit message to include "BREAKING CHANGE" marker
- Remove mention of backward compatibility
- Update description

---

### 2. Reorder Phases: Move Phase 6 Before Phase 3

**Current Order**:
- Phase 3: Environment Integration (requires substrate.yaml)
- Phase 4: Position Management
- Phase 5: Observation/Network Updates
- Phase 6: Config Migration (creates substrate.yaml)

**Revised Order**:
- **Phase 3: Config Migration** (moved from Phase 6)
  - Create substrate.yaml for all existing config packs
  - MUST happen before environment integration
- **Phase 4: Environment Integration** (was Phase 3)
  - Now safe: all configs have substrate.yaml
- **Phase 5: Position Management** (was Phase 4)
- **Phase 6: Observation/Network Updates** (was Phase 5)

**Rationale**: Can't require substrate.yaml if config packs don't have it yet!

---

### 3. Add Pre-Training Validation

**Action**: Create `scripts/validate_configs.py` (see Risk 2 mitigation above)

**Purpose**: Catch missing substrate.yaml early, before training starts

**Usage**:
```bash
# Run before training
python scripts/validate_configs.py

# Or add to CI
uv run pytest tests/ && python scripts/validate_configs.py
```

---

### 4. Update CLAUDE.md

**File**: `CLAUDE.md`

**Section**: "Configuration System" (around line 165)

**Add**:
```markdown
### Required Config Files

Each config pack directory MUST contain:

- `bars.yaml` - Meter definitions
- `cascades.yaml` - Meter relationships
- `affordances.yaml` - Interaction definitions
- `cues.yaml` - UI metadata
- `training.yaml` - Hyperparameters
- **`substrate.yaml`** - Spatial topology (REQUIRED as of Phase 3)

Missing substrate.yaml will cause FileNotFoundError at environment initialization.
```

---

### 5. Update Error Message Template

**Current** (from revised code):
```python
raise FileNotFoundError(
    f"substrate.yaml is required but not found in {config_pack_path}. "
    f"All config packs must define their spatial substrate. "
    f"See docs/examples/substrate.yaml for template."
)
```

**Enhanced** (add troubleshooting):
```python
raise FileNotFoundError(
    f"substrate.yaml is required but not found in {config_pack_path}.\n\n"
    f"All config packs must define their spatial substrate.\n\n"
    f"Quick fix:\n"
    f"1. Copy template: cp docs/examples/substrate.yaml {config_pack_path}/\n"
    f"2. Edit substrate.yaml to match your grid_size from training.yaml\n"
    f"3. See CLAUDE.md 'Configuration System' for details"
)
```

---

## Summary of Changes

### Code Changes
1. **Remove**: Legacy fallback `if substrate_config_path.exists()` branch
2. **Remove**: Deprecation warning
3. **Remove**: `self.substrate = None` legacy marker
4. **Add**: Fail-fast FileNotFoundError with helpful message

### Test Changes
1. **Delete**: Backward compatibility test (Task 3.1 Step 4)
2. **Optional**: Add fail-fast error validation test

### Documentation Changes
1. **Update**: Phase 3 plan (lines 1372-1450)
2. **Update**: Commit message (mark BREAKING CHANGE)
3. **Update**: CLAUDE.md (add substrate.yaml requirement)

### Phase Order Changes
1. **Move**: Phase 6 → Phase 3 (config migration before integration)
2. **Rename**: Phase 3 → Phase 4 (environment integration)
3. **Rename**: Phase 4 → Phase 5 (position management)
4. **Rename**: Phase 5 → Phase 6 (observation updates)

### New Artifacts
1. **Create**: `scripts/validate_configs.py` (pre-training validation)
2. **Create**: `docs/examples/substrate.yaml` (template for operators)

---

## Effort Impact

| Phase | Original | Revised | Change | Rationale |
|-------|----------|---------|--------|-----------|
| Phase 0 | 1h | 1h | +0h | No changes |
| Phase 1 | 6h | 6h | +0h | No changes |
| Phase 2 | 3h | 3h | +0h | No changes |
| Phase 3 | 2h | 1.5h | **-0.5h** | Simpler code, no legacy paths |
| **Total** | **12h** | **11.5h** | **-0.5h** | **4% reduction** |

**Note**: Phase order change (moving config migration earlier) does NOT affect total effort, just sequencing.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation | Status |
|------|----------|------------|------------|--------|
| Configs break without substrate.yaml | High | High | Move Phase 6 before Phase 3 | ✅ Mitigated |
| Developer confusion | Medium | Medium | Better error messages + docs | ✅ Mitigated |
| Checkpoint incompatibility | High | High | Already accepted in Phase 4 | ✅ Accepted |

---

## Conclusion

**Breaking changes authorization simplifies Phase 3 significantly:**

✅ **Remove**: 20+ lines of legacy code
✅ **Remove**: Deprecation warning complexity
✅ **Remove**: Two-mode testing burden
✅ **Improve**: Clearer fail-fast behavior
✅ **Reduce**: Effort by 0.5 hours (25% in Phase 3)

**Key insight**: Backward compatibility was the ONLY complexity in Phases 0-3. Phases 0-2 are greenfield code with zero legacy concerns.

**Critical dependency**: Must reorder phases to create substrate.yaml (Phase 6) BEFORE requiring it (Phase 3).

**Recommendation**: **APPROVE** breaking changes for Phases 0-3. The simplification is significant, risks are well-mitigated, and Phase 4 already commits to breaking changes anyway.

---

**Next Steps**:

1. Update `docs/plans/plan-task-002a-configurable-spatial-substrates.md` (Phase 3 only)
2. Reorder phases (Phase 6 → Phase 3)
3. Create `docs/examples/substrate.yaml` template
4. Create `scripts/validate_configs.py` validation script
5. Update CLAUDE.md with substrate.yaml requirement
6. Proceed with implementation using simplified Phase 3 code
