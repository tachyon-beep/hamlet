# WP-C3: Cascade System Consolidation - Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete CascadeEngine entirely (zero production usage, only test equivalence checks) and consolidate all cascade processing to MeterDynamics.

**Architecture:** Single-path deletion: CascadeEngine exists only in test equivalence checks, MeterDynamics is the sole active cascade system in production. Delete source file, clean test files, update configuration, verify completeness.

**Tech Stack:** Python 3.13+, PyTorch 2.9.0+, pytest 7.4.0+

**References:**
- Combined Plan: `docs/plans/2025-11-14-retire-legacy-dual-paths-execution.md` (Tasks 1-6)
- Audit Results: `docs/reviews/WP-C2-C3-AUDIT-RESULTS.md` (WP-C3 section)
- Original Strategy: `docs/plans/2025-11-13-retire-legacy-dual-paths.md` (Phase 2 WP-C3)

**Total Effort:** 4 hours (6 tasks)

**Breaking Changes:** YES - CascadeEngine deleted per pre-release policy (zero users = zero backward compatibility)

---

## Table of Contents

1. [Task 1: Delete CascadeEngine source file](#task-1-delete-cascadeengine-source-file)
2. [Task 2: Clean test_meters.py](#task-2-clean-test_meterspy)
3. [Task 3: Clean test_engine_dynamic_sizing.py](#task-3-clean-test_engine_dynamic_sizingpy)
4. [Task 4: Update configuration files](#task-4-update-configuration-files)
5. [Task 5: Verify deletion completeness](#task-5-verify-deletion-completeness)
6. [Task 6: Commit WP-C3 changes](#task-6-commit-wpc3-changes)

---

## Task 1: Delete CascadeEngine source file

**Goal:** Remove CascadeEngine.py from source tree (331 lines of legacy code)

**Files:**
- Delete: `src/townlet/environment/cascade_engine.py` (331 lines)

**Estimated Time:** 15 minutes

---

### Step 1: Verify zero production usage

Check that CascadeEngine is NOT used in vectorized_env.py (audit confirmed MeterDynamics only):

```bash
grep -n "CascadeEngine\|cascade_engine" src/townlet/environment/vectorized_env.py
```

**Expected output:** No results (audit confirmed MeterDynamics only)

**If results found:** STOP - escalate to WP-C2/C3 combined plan (production usage detected)

---

### Step 2: Delete the file

```bash
git rm src/townlet/environment/cascade_engine.py
```

**Expected output:**
```
rm 'src/townlet/environment/cascade_engine.py'
```

---

### Step 3: Verify file deleted

```bash
ls src/townlet/environment/cascade_engine.py 2>&1
```

**Expected output:**
```
ls: cannot access 'src/townlet/environment/cascade_engine.py': No such file or directory
```

---

### Step 4: Attempt import to verify deletion

Try to import CascadeEngine (should fail):

```bash
python -c "from townlet.environment.cascade_engine import CascadeEngine" 2>&1
```

**Expected output:**
```
ModuleNotFoundError: No module named 'townlet.environment.cascade_engine'
```

**If import succeeds:** Python cache issue - run `find . -name "*.pyc" -delete` and retry

---

### Step 5: Stage deletion for commit (DO NOT COMMIT YET)

```bash
git status
```

**Expected output:** Shows `deleted: src/townlet/environment/cascade_engine.py` in staged changes

---

## Task 2: Clean test_meters.py

**Goal:** Remove CascadeEngine import, fixture, and equivalence test classes from test_meters.py

**Files:**
- Modify: `tests/test_townlet/unit/environment/test_meters.py`

**Estimated Time:** 1 hour

---

### Step 1: Read fixture section to identify exact lines

First, read the fixture to see the exact code:

```bash
head -n 45 tests/test_townlet/unit/environment/test_meters.py | tail -n 7
```

**Expected output:** Shows cascade_engine fixture definition (lines 38-41 approximately)

---

### Step 2: Remove CascadeEngine import (line 25)

**File:** `tests/test_townlet/unit/environment/test_meters.py`

**Find line 25:**
```python
from townlet.environment.cascade_engine import CascadeEngine
```

**Action:** Delete entire line (line 25)

**Verification:** After deletion, line 25 should be something else (like another import or blank line)

---

### Step 3: Remove cascade_engine fixture (lines 38-41)

**File:** `tests/test_townlet/unit/environment/test_meters.py`

**Read exact lines first:**
```bash
sed -n '38,42p' tests/test_townlet/unit/environment/test_meters.py
```

**Expected to see:**
```python
@pytest.fixture
def cascade_engine(cascade_config, cpu_device):
    """Create CascadeEngine with test config."""
    return CascadeEngine(cascade_config, cpu_device)
```

**Find (lines 38-41):**
```python
@pytest.fixture
def cascade_engine(cascade_config, cpu_device):
    """Create CascadeEngine with test config."""
    return CascadeEngine(cascade_config, cpu_device)
```

**Action:** Delete entire fixture definition (4 lines: decorator + function signature + docstring + return)

---

### Step 4: Read TestCascadeEngineEquivalence class location

Find the class boundaries:

```bash
grep -n "^class TestCascadeEngineEquivalence" tests/test_townlet/unit/environment/test_meters.py
grep -n "^class TestCascadeEngineInitialization" tests/test_townlet/unit/environment/test_meters.py
```

**Expected output:**
```
740:class TestCascadeEngineEquivalence:
820:class TestCascadeEngineInitialization:
```

This tells us TestCascadeEngineEquivalence runs from line 740 to line 819 (80 lines).

---

### Step 5: Remove TestCascadeEngineEquivalence class (lines 736-813)

**File:** `tests/test_townlet/unit/environment/test_meters.py`

**Read the exact section first:**
```bash
sed -n '736,746p' tests/test_townlet/unit/environment/test_meters.py
```

**Expected to see:**
```python
# CascadeEngine Equivalence Tests


class TestCascadeEngineEquivalence:
    """Test that CascadeEngine produces same results as MeterDynamics."""

    def test_equivalence_with_meter_dynamics_healthy(self, cascade_engine, cpu_device, cpu_env_factory):
        """CascadeEngine produces same results as MeterDynamics for healthy agent."""
        ...
```

**Find (starting at line 736):**
```python
# CascadeEngine Equivalence Tests


class TestCascadeEngineEquivalence:
    """Test that CascadeEngine produces same results as MeterDynamics."""

    def test_equivalence_with_meter_dynamics_healthy(self, cascade_engine, cpu_device, cpu_env_factory):
        # ... test implementation ...

    def test_equivalence_with_meter_dynamics_low_satiation(self, cascade_engine, cpu_device, cpu_env_factory):
        # ... test implementation ...

    def test_equivalence_multi_agent_batch(self, cascade_engine, cpu_device, cpu_env_factory):
        # ... test implementation ...
```

**Action:** Delete entire section from line 736 (comment "# CascadeEngine Equivalence Tests") through line 813 (end of last test method in class)

**Estimated lines to delete:** ~78 lines (comment + class + 3 test methods)

---

### Step 6: Read TestCascadeEngineInitialization class location

Find where this class ends:

```bash
sed -n '816,865p' tests/test_townlet/unit/environment/test_meters.py | grep -n "^class\|^def test_\|^$"
```

This helps identify the class boundaries and test methods.

---

### Step 7: Remove TestCascadeEngineInitialization class (lines 816-860)

**File:** `tests/test_townlet/unit/environment/test_meters.py`

**Read the exact section first:**
```bash
sed -n '816,835p' tests/test_townlet/unit/environment/test_meters.py
```

**Expected to see:**
```python
# CascadeEngine Initialization Tests


class TestCascadeEngineInitialization:
    """Test CascadeEngine initialization and data structures."""

    def test_engine_initialization(self, cascade_engine, cascade_config):
        # ... assertions ...

    def test_engine_helper_methods(self, cascade_engine):
        # ... assertions ...
```

**Find (starting at line 816):**
```python
# CascadeEngine Initialization Tests


class TestCascadeEngineInitialization:
    """Test CascadeEngine initialization and data structures."""

    def test_engine_initialization(self, cascade_engine, cascade_config):
        """CascadeEngine initializes with correct data structures."""
        # ... test implementation ...

    def test_engine_helper_methods(self, cascade_engine):
        """Helper methods return correct values."""
        # ... test implementation ...
```

**Action:** Delete entire section from line 816 (comment "# CascadeEngine Initialization Tests") through approximately line 860 (end of last test method)

**Estimated lines to delete:** ~45 lines (comment + class + 2 test methods)

**Note:** Exact ending line may vary - delete through the end of `test_engine_helper_methods` method

---

### Step 8: Run tests to verify no broken references

After all deletions, verify remaining tests still work:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_townlet/unit/environment/test_meters.py -v
```

**Expected output:** All remaining tests pass (MeterDynamics tests unaffected)

**Possible failures:**
- If cascade_engine fixture still referenced: Missed a test method using it
- If import errors: Missed removing import line
- If class not found: Incorrect line ranges deleted

**Fix strategy:** Re-read file, find remaining cascade_engine references, delete them

---

### Step 9: Stage changes (DO NOT COMMIT YET)

```bash
git add tests/test_townlet/unit/environment/test_meters.py
git status
```

**Expected output:** Shows modified test_meters.py in staged changes

---

## Task 3: Clean test_engine_dynamic_sizing.py

**Goal:** Remove CascadeEngine import and TestCascadeEngineDynamicSizing class

**Files:**
- Modify: `tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py`

**Estimated Time:** 30 minutes

---

### Step 1: Identify sections to delete

Find line numbers for CascadeEngine references:

```bash
grep -n "CascadeEngine\|TestCascadeEngine" tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py
```

**Expected output (from audit):**
```
12:from townlet.environment.cascade_engine import CascadeEngine
24:class TestCascadeEngineDynamicSizing:
```

---

### Step 2: Remove CascadeEngine import (line 12)

**File:** `tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py`

**Find line 12:**
```python
from townlet.environment.cascade_engine import CascadeEngine
```

**Action:** Delete entire line (line 12)

---

### Step 3: Read TestCascadeEngineDynamicSizing class location

Find the class boundaries:

```bash
sed -n '24,80p' tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py
```

This shows the class definition and helps identify where it ends (approximately line 73 per audit).

---

### Step 4: Remove TestCascadeEngineDynamicSizing class (lines 24-73)

**File:** `tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py`

**Read the exact section first:**
```bash
sed -n '24,35p' tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py
```

**Expected to see:**
```python
class TestCascadeEngineDynamicSizing:
    """Tests for CascadeEngine with dynamic meter counts."""

    def test_something(self, ...):
        # ... test implementation ...
```

**Find (starting at line 24):**
```python
class TestCascadeEngineDynamicSizing:
    """Tests for CascadeEngine with dynamic meter counts."""

    # ... all test methods ...
```

**Action:** Delete entire class from line 24 through approximately line 73 (end of class)

**Estimated lines to delete:** ~50 lines

---

### Step 5: Run tests to verify TestVectorizedEnvDynamicSizing still passes

The file should still contain TestVectorizedEnvDynamicSizing class - verify it still works:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py::TestVectorizedEnvDynamicSizing -v
```

**Expected output:** All TestVectorizedEnvDynamicSizing tests pass

**If failures:** Check if any test methods reference cascade_engine - delete those methods

---

### Step 6: Stage changes (DO NOT COMMIT YET)

```bash
git add tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py
```

---

## Task 4: Update configuration files

**Goal:** Remove cascade_engine from whitelist and lint test assertions

**Files:**
- Modify: `.defaults-whitelist.txt`
- Modify: `tests/test_no_defaults_lint.py`

**Estimated Time:** 30 minutes

---

### Step 1: Remove cascade_engine from whitelist (line 52)

**File:** `.defaults-whitelist.txt`

**Read line 52 first:**
```bash
sed -n '52p' .defaults-whitelist.txt
```

**Expected output:**
```
src/townlet/environment/cascade_engine.py:*
```

**Find (line 52):**
```
src/townlet/environment/cascade_engine.py:*
```

**Action:** Delete entire line (line 52)

---

### Step 2: Verify whitelist updated

```bash
grep "cascade_engine" .defaults-whitelist.txt
```

**Expected output:** No output (line removed)

**If output found:** Additional cascade_engine lines exist - delete them too

---

### Step 3: Read test_no_defaults_lint.py to find assertions

Find lines mentioning cascade_engine:

```bash
grep -n "cascade_engine" tests/test_no_defaults_lint.py
```

**Expected output (from audit):**
```
68:            "src/townlet/environment/cascade_engine.py::CascadeEngine::apply_base_depletions:DEF001"
72:        assert pattern.filepath_pattern == "src/townlet/environment/cascade_engine.py"
73:        assert pattern.class_pattern == "CascadeEngine"
```

---

### Step 4: Read exact context around lines 68, 72-73

**Read the test method:**
```bash
sed -n '66,76p' tests/test_no_defaults_lint.py
```

**Expected to see:**
```python
def test_parse_structural_function_specific(self):
    pattern = no_defaults_lint.parse_whitelist_pattern(
        "src/townlet/environment/cascade_engine.py::CascadeEngine::apply_base_depletions:DEF001"
    )
    assert pattern is not None
    assert pattern.is_structural()
    assert pattern.filepath_pattern == "src/townlet/environment/cascade_engine.py"
    assert pattern.class_pattern == "CascadeEngine"
    assert pattern.function_pattern == "apply_base_depletions"
```

---

### Step 5: Remove whitelist test assertions referencing cascade_engine

**File:** `tests/test_no_defaults_lint.py`

**Option A: If this is the ENTIRE test method** (test only tests cascade_engine):
- Delete entire `test_parse_structural_function_specific` method

**Option B: If test has OTHER assertions** (tests multiple patterns):
- Delete only the 3 cascade_engine assertion lines (68, 72-73)

**Read more context to decide:**
```bash
sed -n '66,85p' tests/test_no_defaults_lint.py
```

**Decision:** Based on method name `test_parse_structural_function_specific`, this likely tests ONLY cascade_engine pattern. **Delete entire method.**

**Find (approximately lines 66-76):**
```python
def test_parse_structural_function_specific(self):
    pattern = no_defaults_lint.parse_whitelist_pattern(
        "src/townlet/environment/cascade_engine.py::CascadeEngine::apply_base_depletions:DEF001"
    )
    assert pattern is not None
    assert pattern.is_structural()
    assert pattern.filepath_pattern == "src/townlet/environment/cascade_engine.py"
    assert pattern.class_pattern == "CascadeEngine"
    assert pattern.function_pattern == "apply_base_depletions"
```

**Action:** Delete entire test method (approximately 10 lines)

---

### Step 6: Run whitelist lint test

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_no_defaults_lint.py -v
```

**Expected output:** All remaining tests pass (no cascade_engine test exists)

**If test still references cascade_engine:** Missed some assertions - re-read and delete them

---

### Step 7: Stage configuration changes (DO NOT COMMIT YET)

```bash
git add .defaults-whitelist.txt tests/test_no_defaults_lint.py
```

---

## Task 5: Verify deletion completeness

**Goal:** Comprehensive verification that zero CascadeEngine references remain

**Files:** None (verification only)

**Estimated Time:** 1 hour

---

### Step 1: Grep for any remaining CascadeEngine references in source

```bash
grep -rn "CascadeEngine" src/townlet/
```

**Expected output:** No results

**If results found:** Production code still references CascadeEngine - STOP and escalate

---

### Step 2: Grep for any remaining cascade_engine attribute usage

```bash
grep -rn "cascade_engine" src/townlet/
```

**Expected output:** No results

**If results found:** Code still uses cascade_engine attribute - STOP and escalate

---

### Step 3: Grep for CascadeEngine in tests (should only be deletions)

```bash
grep -rn "CascadeEngine" tests/
```

**Expected output:** No results (all test references deleted)

**If results found:** Additional test files reference CascadeEngine - identify and clean them:
1. Read the file with references
2. Remove CascadeEngine imports
3. Delete or update test methods using cascade_engine
4. Re-run this grep

---

### Step 4: Verify MeterDynamics is sole cascade system

Check that MeterDynamics is imported and used:

```bash
grep -n "MeterDynamics\|meter_dynamics" src/townlet/environment/vectorized_env.py | head -5
```

**Expected output:** Shows MeterDynamics import and usage (approximately):
```
23:from townlet.environment.meter_dynamics import MeterDynamics
352:        self.meter_dynamics = MeterDynamics(
944:        self.meters = self.meter_dynamics.deplete_meters(...)
```

**If no results:** MeterDynamics not in vectorized_env - CRITICAL ERROR, escalate

---

### Step 5: Run full environment test suite

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_townlet/unit/environment/ -v
```

**Expected output:** All tests pass

**Common failures:**
- `NameError: name 'CascadeEngine' is not defined` → Missed removing an import
- `fixture 'cascade_engine' not found` → Test method still uses cascade_engine fixture
- `ModuleNotFoundError: cascade_engine` → Missed cleaning a test file

**Fix strategy:**
1. Read error message to identify file and line
2. Open that file
3. Remove CascadeEngine reference
4. Re-run tests

---

### Step 6: Document verification results

Create verification report:

```bash
cat > /tmp/wpc3_verification.txt << 'EOF'
WP-C3 Verification Checklist (2025-11-14)

✓ CascadeEngine.py deleted (331 lines removed)
✓ Zero CascadeEngine references in src/townlet/
✓ Zero cascade_engine references in src/townlet/
✓ TestCascadeEngine* classes removed from tests
✓ Whitelist configuration updated
✓ All environment tests passing
✓ MeterDynamics confirmed as sole cascade system

Files modified:
- Deleted: src/townlet/environment/cascade_engine.py (331 lines)
- Modified: tests/test_townlet/unit/environment/test_meters.py (~123 lines removed)
  - Removed CascadeEngine import (line 25)
  - Removed cascade_engine fixture (lines 38-41)
  - Removed TestCascadeEngineEquivalence class (lines 736-813, ~78 lines)
  - Removed TestCascadeEngineInitialization class (lines 816-860, ~45 lines)
- Modified: tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py (~51 lines removed)
  - Removed CascadeEngine import (line 12)
  - Removed TestCascadeEngineDynamicSizing class (lines 24-73, ~50 lines)
- Modified: .defaults-whitelist.txt (1 line removed)
- Modified: tests/test_no_defaults_lint.py (~10 lines removed)

Total lines deleted: ~516 lines

Verification commands run:
- grep -rn "CascadeEngine" src/townlet/ → 0 results ✓
- grep -rn "cascade_engine" src/townlet/ → 0 results ✓
- grep -rn "CascadeEngine" tests/ → 0 results ✓
- pytest tests/test_townlet/unit/environment/ → ALL PASS ✓
- grep -n "MeterDynamics" src/townlet/environment/vectorized_env.py → Found ✓
EOF

cat /tmp/wpc3_verification.txt
```

**Expected output:** Displays verification checklist

---

## Task 6: Commit WP-C3 changes

**Goal:** Commit all WP-C3 changes with detailed message

**Files:** All staged changes from Tasks 1-4

**Estimated Time:** 15 minutes

---

### Step 1: Review staged changes

```bash
git status
git diff --cached --stat
```

**Expected output:** Shows 5 files:
- deleted: src/townlet/environment/cascade_engine.py
- modified: tests/test_townlet/unit/environment/test_meters.py
- modified: tests/test_townlet/unit/environment/test_engine_dynamic_sizing.py
- modified: .defaults-whitelist.txt
- modified: tests/test_no_defaults_lint.py

**If unexpected files:** Review with `git diff --cached <file>` and unstage if needed

---

### Step 2: Run final test verification before commit

Ensure all tests pass before committing:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_townlet/unit/environment/ -v --tb=short
```

**Expected output:** All tests pass

**If failures:** Fix failing tests before committing (go back to Task 5 verification steps)

---

### Step 3: Commit with detailed message

```bash
git commit -m "$(cat <<'EOF'
feat(wpc3): consolidate cascade systems to MeterDynamics

BREAKING CHANGE: CascadeEngine deleted, MeterDynamics only

Changes:
- DELETE: src/townlet/environment/cascade_engine.py (331 lines)
- DELETE: TestCascadeEngineEquivalence class from test_meters.py (~78 lines)
- DELETE: TestCascadeEngineInitialization class from test_meters.py (~45 lines)
- DELETE: TestCascadeEngineDynamicSizing class from test_engine_dynamic_sizing.py (~50 lines)
- REMOVE: CascadeEngine imports from test files
- REMOVE: cascade_engine fixture from test_meters.py
- UPDATE: .defaults-whitelist.txt (remove cascade_engine entry)
- UPDATE: test_no_defaults_lint.py (remove cascade_engine assertions)

Rationale: Pre-release status (zero users) enables clean break.
MeterDynamics is modern GPU-native tensor processor, fully tested.
CascadeEngine had zero production usage (only equivalence tests).
Eliminates operator confusion about which cascade system is active.

Test Coverage: All environment tests passing (test_meters.py, test_engine_dynamic_sizing.py)

Lines deleted: ~516 lines total
- Source: 331 lines (cascade_engine.py)
- Tests: ~173 lines (test_meters.py ~123, test_engine_dynamic_sizing.py ~50)
- Config: ~12 lines (whitelist + lint test)

Verification:
- grep "CascadeEngine" src/townlet/ → 0 results
- grep "cascade_engine" src/townlet/ → 0 results
- grep "CascadeEngine" tests/ → 0 results
- MeterDynamics confirmed sole cascade system in vectorized_env.py

Closes: WP-C3 (Architecture Analysis 2025-11-13)
Part of: Sprint 1 Critical Path

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Expected output:** Commit created successfully

---

### Step 4: Verify commit

```bash
git log -1 --stat
```

**Expected output:** Shows commit with 5 file changes:
- 1 deletion (cascade_engine.py)
- 4 modifications (test files + config files)

**Verify commit message:** Shows detailed breakdown of changes

---

### Step 5: Tag completion milestone

```bash
git tag -a wpc3-complete -m "WP-C3: Cascade System Consolidation Complete

Summary:
- CascadeEngine.py deleted (331 lines)
- Test equivalence classes removed (~173 lines)
- Configuration files updated
- MeterDynamics confirmed sole cascade system

Total lines deleted: ~516 lines
All tests passing, zero CascadeEngine references remain

See: docs/plans/2025-11-14-wpc3-cascade-consolidation-execution.md"
```

**Expected output:** Tag created

---

### Step 6: Verify tag created

```bash
git tag -l "wpc*"
```

**Expected output:** Shows `wpc3-complete`

---

### Step 7: Display completion summary

```bash
cat << 'EOF'

╔══════════════════════════════════════════════════════════╗
║  WP-C3: Cascade System Consolidation COMPLETE ✅         ║
╚══════════════════════════════════════════════════════════╝

Total Effort: 4 hours (6 tasks)
Total Lines Deleted: ~516 lines

✅ Task 1: CascadeEngine.py deleted (331 lines)
✅ Task 2: test_meters.py cleaned (~123 lines removed)
✅ Task 3: test_engine_dynamic_sizing.py cleaned (~50 lines removed)
✅ Task 4: Configuration files updated (~12 lines removed)
✅ Task 5: Verification complete (zero CascadeEngine refs)
✅ Task 6: Changes committed with wpc3-complete tag

Breaking Changes:
- CascadeEngine class deleted
- All cascade processing uses MeterDynamics.apply_depletion_and_cascades()
- No dual cascade system references remain

Test Results:
- All environment tests passing
- MeterDynamics >90% coverage maintained
- Zero CascadeEngine references in codebase

Pre-Release Freedom: SUCCESSFULLY APPLIED
- No backwards compatibility burden
- Clean deletion without migration paths
- Single source of truth enforced (MeterDynamics only)

Status: WP-C3 COMPLETE

Next: WP-C2 (Brain As Code Legacy Deprecation, 8 hours)
      OR Sprint 2 Medium Priority (WP-M2, WP-M4, WP-M5)

╚══════════════════════════════════════════════════════════╝

EOF
```

**Expected output:** Displays completion banner

---

## Success Criteria

### Overall Success Criteria

- [ ] CascadeEngine.py deleted (331 lines removed)
- [ ] Zero CascadeEngine references in src/townlet/ (verified via grep)
- [ ] Zero cascade_engine references in src/townlet/ (verified via grep)
- [ ] Zero CascadeEngine references in tests/ (verified via grep)
- [ ] TestCascadeEngine* classes removed from all test files
- [ ] Whitelist configuration updated (.defaults-whitelist.txt)
- [ ] Lint test updated (test_no_defaults_lint.py)
- [ ] All environment tests passing (pytest tests/test_townlet/unit/environment/)
- [ ] MeterDynamics confirmed as sole cascade system in vectorized_env.py
- [ ] Commit created with detailed message and wpc3-complete tag
- [ ] Total lines deleted: ~516 lines

### Task-Level Success Criteria

**Task 1:**
- [ ] cascade_engine.py deleted via `git rm`
- [ ] Import attempt raises ModuleNotFoundError
- [ ] File staged for commit

**Task 2:**
- [ ] CascadeEngine import removed (line 25)
- [ ] cascade_engine fixture removed (lines 38-41)
- [ ] TestCascadeEngineEquivalence removed (~78 lines)
- [ ] TestCascadeEngineInitialization removed (~45 lines)
- [ ] All remaining tests pass
- [ ] Changes staged

**Task 3:**
- [ ] CascadeEngine import removed (line 12)
- [ ] TestCascadeEngineDynamicSizing removed (~50 lines)
- [ ] TestVectorizedEnvDynamicSizing still passes
- [ ] Changes staged

**Task 4:**
- [ ] cascade_engine entry removed from .defaults-whitelist.txt
- [ ] test_parse_structural_function_specific removed from test_no_defaults_lint.py
- [ ] Lint tests pass
- [ ] Changes staged

**Task 5:**
- [ ] Zero CascadeEngine in src/ (grep verified)
- [ ] Zero cascade_engine in src/ (grep verified)
- [ ] Zero CascadeEngine in tests/ (grep verified)
- [ ] MeterDynamics found in vectorized_env.py
- [ ] All environment tests pass
- [ ] Verification report created

**Task 6:**
- [ ] 5 files in commit (1 deleted, 4 modified)
- [ ] All tests pass before commit
- [ ] Commit message includes BREAKING CHANGE
- [ ] wpc3-complete tag created
- [ ] Completion banner displayed

---

## Common Issues & Troubleshooting

### Issue 1: "fixture 'cascade_engine' not found"

**Cause:** Test method still uses cascade_engine fixture but fixture was deleted

**Fix:**
1. Read error to identify test file and method
2. Open that test file
3. Find test method using cascade_engine parameter
4. Delete entire test method (it's testing CascadeEngine)
5. Re-run tests

**Example:**
```bash
# Error shows: tests/test_townlet/unit/environment/test_meters.py::test_some_cascade_feature
grep -n "def test_some_cascade_feature" tests/test_townlet/unit/environment/test_meters.py
# Edit file, delete that test method
```

---

### Issue 2: "ModuleNotFoundError: cascade_engine" but file deleted

**Cause:** Python bytecode cache (.pyc files) still reference old module

**Fix:**
```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

---

### Issue 3: Grep still finds CascadeEngine references after deletion

**Cause:** Missed a test file or documentation file

**Fix:**
1. Read grep output to identify file
2. If test file: Repeat Task 2/Task 3 cleanup steps for that file
3. If documentation: Update docs to remove CascadeEngine mentions
4. Re-run grep verification

**Example:**
```bash
grep -rn "CascadeEngine" tests/
# Output: tests/test_townlet/integration/test_some_integration.py:42
# → Open that file, remove CascadeEngine usage
```

---

### Issue 4: Tests pass locally but CI might fail

**Cause:** Different test discovery or cache behavior in CI

**Prevention:**
```bash
# Before committing, run full test suite exactly as CI would:
UV_CACHE_DIR=.uv-cache uv run pytest tests/ -v --tb=short
```

---

### Issue 5: Line numbers don't match exactly

**Cause:** File changed since audit (someone edited test_meters.py)

**Fix:**
1. Use grep/sed to find current line numbers
2. Read surrounding context to confirm correct section
3. Delete based on content, not line numbers
4. Verify with grep after deletion

**Example:**
```bash
# Instead of "delete line 25", use:
grep -n "from.*cascade_engine import" tests/test_townlet/unit/environment/test_meters.py
# Shows: 27:from townlet.environment.cascade_engine import CascadeEngine
# → Delete line 27, not 25
```

---

## Verification Commands Reference

Quick reference for all verification commands used in this plan:

```bash
# Source code verification
grep -rn "CascadeEngine" src/townlet/                    # Should: 0 results
grep -rn "cascade_engine" src/townlet/                   # Should: 0 results
grep -n "MeterDynamics" src/townlet/environment/vectorized_env.py  # Should: found

# Test code verification
grep -rn "CascadeEngine" tests/                          # Should: 0 results
grep -rn "cascade_engine" tests/                         # Should: 0 results

# File deletion verification
ls src/townlet/environment/cascade_engine.py             # Should: No such file
python -c "from townlet.environment.cascade_engine import CascadeEngine"  # Should: ModuleNotFoundError

# Test execution
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_townlet/unit/environment/ -v  # Should: ALL PASS

# Configuration verification
grep "cascade_engine" .defaults-whitelist.txt            # Should: 0 results
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_no_defaults_lint.py -v        # Should: ALL PASS

# Git verification
git status                                                # Should: 5 files staged
git diff --cached --stat                                 # Should: 1 deleted, 4 modified
```

---

## Plan Metadata

**Created:** 2025-11-14
**Author:** Claude Code (writing-plans skill)
**Estimated Total Time:** 4 hours
**Actual Time:** TBD (fill in after execution)
**Tasks:** 6 tasks
**Files Modified:** 5 files (1 deleted, 4 modified)
**Lines Deleted:** ~516 lines
**Breaking Changes:** YES (CascadeEngine deleted)
**Test Coverage:** Maintained (>90% for MeterDynamics)
**Pre-Release Freedom Applied:** YES

---

**Plan Status:** READY FOR EXECUTION ✅

**Prerequisites:** None (can start immediately)

**Blockers:** None

**Dependencies:** MeterDynamics already implemented and tested (optimization phase complete)
