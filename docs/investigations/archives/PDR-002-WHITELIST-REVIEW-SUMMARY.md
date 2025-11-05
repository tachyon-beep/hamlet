# PDR-002 Whitelist Review: Executive Summary

**Date**: 2025-11-05
**Status**: ⚠️ CRITICAL - 85% Non-Compliant
**Action Required**: Immediate whitelist replacement + phased refactoring

---

## TL;DR

**Current whitelist violates PDR-002 by exempting 200+ UAC/BAC defaults across 20+ files.**

**Solution**: Adopt compliant whitelist (ready to deploy) + systematic refactoring over 12-17 days.

**Impact**: Full config reproducibility, self-documenting configs, fail-fast validation.

---

## The Problem

**Current `.defaults-whitelist.txt`**:
- ✅ Whitelists entire modules (demo, environment, exploration, population)
- ❌ Hides 200+ UAC/BAC defaults (grid_size, epsilon, learning_rate, etc.)
- ❌ 85% of entries violate PDR-002 policy

**What PDR-002 Requires**:
- ✅ ALL UAC/BAC parameters must be explicit in config (no code defaults)
- ✅ Only infrastructure (device, port, log_dir) can have defaults
- ✅ Test: "Does param affect WHAT vs WHERE/HOW?" → WHAT = no defaults

---

## Key Findings

### Whitelist Analysis

| Category | Current | Should Be | Reason |
|----------|---------|-----------|--------|
| UAC/BAC exemptions | 17 entries | 0 entries | Violates PDR-002 (no exemptions for behavior) |
| Infrastructure exemptions | 2 entries | 17 entries | Acceptable (hardware, file paths, telemetry) |
| **Total compliance** | **15%** | **100%** | **85% non-compliant** |

### Violations by Module (Hidden by Current Whitelist)

| Module | Violations | Type | Refactoring Task |
|--------|------------|------|------------------|
| `demo/runner.py` | 50+ dict.get() | UAC/BAC | TASK-003 (DTOs) |
| `agent/networks.py` | 6 params | BAC | TASK-005 (BAC) |
| `environment/**` | 50+ params | UAC | TASK-004A (UAC) |
| `curriculum/**` | 20+ params | UAC | TASK-004A (UAC) |
| `exploration/**` | 50+ params | BAC | TASK-005 (BAC) |
| `population/**` | 30+ params | BAC | TASK-005 (BAC) |
| `training/**` | 20+ params | BAC | TASK-005 (BAC) |
| **TOTAL** | **226+** | **Mixed** | **1500+ LOC** |

---

## The Solution

### Compliant Whitelist (Ready to Deploy)

**File**: `.defaults-whitelist-compliant.txt`

**What's Changed**:
- ❌ Removed: 17 UAC/BAC exemptions (demo, environment, networks, exploration, etc.)
- ✅ Kept: 3 infrastructure exemptions (recording, tensorboard_logger, infrastructure params)
- ✅ Added: 15 granular exemptions (device, port, checkpoint_dir, metadata defaults)

**Result**: Only infrastructure/telemetry defaults whitelisted, all UAC/BAC violations exposed.

### Refactoring Roadmap (4 Phases)

**Phase 1: Config Loading (TASK-003)** - 2-3 days
- Replace 50+ `.get(key, default)` with Pydantic DTOs
- Files: `demo/runner.py`, `demo/unified_server.py`
- Impact: All training configs validated at load time

**Phase 2: Environment (TASK-004A)** - 3-4 days
- Remove UAC defaults from 8 environment files
- Files: `vectorized_env.py`, `affordance_config.py`, `cascade_config.py`, etc.
- Impact: Complete UAC reproducibility

**Phase 3: Networks (TASK-005)** - 2-3 days
- Remove BAC defaults from `networks.py`
- Add network architecture config (brain.yaml or training.yaml)
- Impact: Network architecture fully configurable

**Phase 4: Curriculum/Exploration/Population (TASK-004A, TASK-005)** - 5-7 days
- Remove remaining BAC/UAC defaults
- Files: 10+ files across curriculum, exploration, population
- Impact: Complete PDR-002 compliance

**Total Timeline**: 12-17 days full-time development

---

## Example: Before vs After

### Before (Non-Compliant)

**Code** (networks.py):
```python
def __init__(self, action_dim: int = 5, hidden_dim: int = 256):
    # ❌ Hidden defaults
```

**Whitelist**:
```
src/townlet/agent/networks.py:*  # ❌ Hides all defaults
```

**Result**: Operator doesn't know action_dim=5 is being used.

### After (Compliant)

**Code** (networks.py):
```python
def __init__(self, action_dim: int, hidden_dim: int):
    # ✅ Required params, no defaults
    if action_dim < 1:
        raise ValueError("action_dim must be >= 1. Add to config: action_dim: 5")
```

**Config** (training.yaml):
```yaml
network:
  action_dim: 5  # ✅ Explicit
  hidden_dim: 256  # ✅ Explicit
```

**Whitelist**:
```
# networks.py removed from whitelist (no longer needed)
```

**Result**: Operator sees action_dim=5 in config. Reproducible. Self-documenting.

---

## Benefits

### 1. Reproducibility ✅
- Old configs remain valid (values in YAML, not code)
- Same config + same code = identical behavior
- No "works on my machine" bugs

### 2. Transparency ✅
- All active parameters visible in config files
- No hidden behavior (operator knows actual values)
- Self-documenting configs

### 3. Fail-Fast ✅
- Missing params trigger clear errors (not silent fallbacks)
- Error messages show example config structure
- Forces conscious parameter choices

### 4. Domain Agnosticism ✅
- Any universe configurable (villages, factories, trading)
- No hardcoded assumptions in Python code
- UNIVERSE_AS_CODE philosophy realized

---

## Migration Plan

### Step 1: Adopt Compliant Whitelist (Immediate)

```bash
# Backup current whitelist
cp .defaults-whitelist.txt .defaults-whitelist.txt.backup

# Deploy compliant version
cp .defaults-whitelist-compliant.txt .defaults-whitelist.txt

# Verify violations now detected
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
# Expected: ~120 violations (this is CORRECT - they were hidden before)
```

**⚠️ CI will fail** - This is intentional. Violations must be fixed systematically.

### Step 2: Refactor by Priority (Phased)

**Priority order**:
1. Config loading (TASK-003) - highest impact, enables other refactoring
2. Environment UAC (TASK-004A) - core universe mechanics
3. Networks BAC (TASK-005) - agent architecture
4. Remaining modules (TASK-004A, TASK-005) - complete compliance

**Timeline**: 12-17 days total, phased over 3-4 sprints

### Step 3: Verify Compliance

```bash
# Linter passes
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
# Expected: 0 violations

# Configs load successfully
uv run pytest tests/test_integration/test_config_loading.py

# Training runs
uv run scripts/run_demo.py --config configs/L0_0_minimal
```

---

## Risks & Mitigations

### Risk 1: Breaking Changes to Configs

**Concern**: Existing configs might fail to load after refactoring.

**Mitigation**:
- All refactoring adds validation, doesn't change YAML structure
- Existing complete configs remain valid
- Incomplete configs will fail with clear error messages (this is desired)
- Migration scripts for batch updates if needed

### Risk 2: Development Velocity Impact

**Concern**: 12-17 days of refactoring delays feature work.

**Mitigation**:
- Phased approach allows parallel work (feature branches off main)
- Phase 1 (config loading) unblocks other refactoring
- Can be parallelized across team members
- Long-term velocity increase (faster debugging, fewer config bugs)

### Risk 3: Whitelist Bloat Over Time

**Concern**: New defaults added, whitelist grows unbounded.

**Mitigation**:
- Quarterly whitelist reviews (Q1, Q2, Q3, Q4)
- Linter enforces policy in CI (blocks PRs with violations)
- Clear exemption criteria in whitelist comments
- Target whitelist size: <20 entries

---

## Recommendation

**Immediate Action**:
1. ✅ Review findings (this document)
2. ✅ Adopt `.defaults-whitelist-compliant.txt` (replace current whitelist)
3. ✅ Create JIRA tickets for Phases 1-4
4. ✅ Assign Phase 1 (config loading) to start next sprint

**Expected Outcome**:
- Full PDR-002 compliance within 3-4 sprints
- Reproducible, self-documenting configs
- Domain-agnostic framework (UNIVERSE_AS_CODE realized)
- No hidden behavior (all params explicit)

---

## Questions?

**Q: Why is this urgent if code works today?**
A: Hidden defaults create non-reproducible configs. When defaults change, old configs break silently. PDR-002 prevents this by requiring explicit values.

**Q: Can we keep some defaults for convenience?**
A: Only for infrastructure (device, port, log_dir). PDR-002 explicitly forbids UAC/BAC defaults. Provide config templates instead.

**Q: What if we disagree with a whitelist removal?**
A: Review PDR-002 criteria: "Does param affect WHAT vs WHERE/HOW?" If WHAT (UAC/BAC), must remove. If WHERE/HOW (infrastructure), can whitelist.

**Q: How do we handle required fields in Pydantic DTOs?**
A: Use `Field(...)` (no default). Missing fields trigger validation errors with clear messages.

---

## Approval & Next Steps

**Reviewed By**: Claude (Sonnet 4.5)
**Approval Required From**: Architecture Team, Development Lead
**Decision Deadline**: 2025-11-08 (3 days)
**Implementation Start**: 2025-11-11 (next sprint)

**Action Items**:
1. [ ] Architecture team reviews findings (PDR-002-WHITELIST-REVIEW.md)
2. [ ] Dev lead approves migration plan
3. [ ] Create JIRA tickets for Phases 1-4
4. [ ] Deploy compliant whitelist to main branch
5. [ ] Begin Phase 1 refactoring (TASK-003)

---

**Full Analysis**: See `PDR-002-WHITELIST-REVIEW.md` for detailed per-entry review.
**Comparison**: See `PDR-002-WHITELIST-COMPARISON.md` for before/after comparison.
**Policy**: See `PDR-002-NO-DEFAULTS-PRINCIPLE.md` for policy rationale.
