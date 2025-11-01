# Checkpoint Directory Cleanup Analysis

**Date:** November 1, 2025  
**Issue:** Multiple checkpoint directories in project root that should be in `runs/`

---

## Current State

### Test Checkpoint Directories (16.4 MB total)

Old test artifacts from **October 27, 2025** - not created by current test suite:

```
5.6M    test_checkpoints                    # Oct 27 21:37
4.8M    test_checkpoints_spatial_dueling    # Oct 27 21:37
3.2M    test_checkpoints_spatial            # Oct 27 21:37
2.4M    test_checkpoints_dueling            # Oct 27 21:37 (ep3 updated Oct 31)
432K    test_checkpoints_qnetwork           # Oct 27 21:37
```

**Analysis:**

- ❌ Not tracked in git (already gitignored)
- ❌ Not created by current test suite
- ❌ No code references these directories
- ✅ Safe to delete (old manual test artifacts)

### Production Checkpoint Directories (1.3 GB total!)

Training runs from before `runs/` organization:

```
829M    checkpoints_level2                  # Old Level 2 training
450M    checkpoints_level2_validation       # Old validation run
 24M    checkpoints_spatial_dueling         # Old spatial dueling experiments
 12M    checkpoints_dueling                 # Old dueling experiments
112K    checkpoints                         # Miscellaneous old run
4.0K    demo_checkpoints                    # Empty/minimal
```

**Analysis:**

- ❌ Not tracked in git (already gitignored)
- ❌ Should be in `runs/<name>/checkpoints/`
- ⚠️ May contain valuable training data from past runs
- 🤔 Decision needed: Archive or delete?

---

## Recommendation

### 1. Delete Test Checkpoint Directories (Safe)

These are clearly old test artifacts with no value:

```bash
# Safe to delete immediately
rm -rf test_checkpoints*
```

**Benefit:** Frees 16.4 MB, cleans up project root

### 2. Archive Production Checkpoints (Preserve Data)

Move old training runs to `runs/archive/` to preserve but organize:

```bash
# Archive old training runs
mkdir -p runs/archive/old_checkpoints

# Move checkpoint directories
mv checkpoints runs/archive/old_checkpoints/misc
mv checkpoints_dueling runs/archive/old_checkpoints/dueling
mv checkpoints_spatial_dueling runs/archive/old_checkpoints/spatial_dueling
mv checkpoints_level2 runs/archive/old_checkpoints/level2
mv checkpoints_level2_validation runs/archive/old_checkpoints/level2_validation
mv demo_checkpoints runs/archive/old_checkpoints/demo

# Result: Clean root, data preserved in runs/archive/
```

**Benefit:**

- Clean project root
- Data preserved if needed for analysis
- Can delete later when safe (runs/archive/ already gitignored)

### 3. Alternative: Nuclear Option (If Data Not Needed)

If old checkpoints have no value:

```bash
# WARNING: Deletes 1.3 GB of training data!
rm -rf checkpoints* demo_checkpoints
```

**Only use if:**

- Don't need old training runs for comparison
- Can retrain if needed
- Want maximum cleanup

---

## Impact on New Integration Tests

**Good news:** New integration tests use `tempfile.mkdtemp()`

```python
@pytest.fixture
def temp_run_dir():
    """Create temporary directory for test run artifacts."""
    temp_dir = tempfile.mkdtemp(prefix="test_run_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)  # Auto-cleanup!
```

**This means:**

- ✅ No directories created in project root
- ✅ Automatic cleanup after tests
- ✅ No pollution of workspace
- ✅ No conflicts with production runs

---

## Updated .gitignore Verification

Let's verify .gitignore covers all cases:

```bash
# Check current patterns
grep -E "(checkpoint|runs)" .gitignore
```

**Current coverage:**

```
checkpoints/       # ✅ Covers checkpoints*/ directories
runs/              # ✅ Covers runs/ directory
*.pt               # ✅ Covers checkpoint files
*.pth              # ✅ Covers checkpoint files
*.ckpt             # ✅ Covers checkpoint files
```

**Good!** All checkpoint directories already gitignored.

---

## Recommended Action

### Conservative Approach (Archive First)

```bash
# 1. Delete worthless test artifacts
rm -rf test_checkpoints*

# 2. Archive potentially valuable training data
mkdir -p runs/archive/old_checkpoints
mv checkpoints* demo_checkpoints runs/archive/old_checkpoints/ 2>/dev/null || true

# 3. Verify clean root
ls -d checkpoints* demo_checkpoints test_checkpoints* 2>/dev/null
# Should show: "No such file or directory"

# 4. Optional: Delete archive later when safe
# rm -rf runs/archive/old_checkpoints
```

### Aggressive Approach (Clean Slate)

```bash
# Delete everything (1.3 GB freed)
rm -rf test_checkpoints* checkpoints* demo_checkpoints

# Verify clean
ls -d *checkpoint* 2>/dev/null
# Should show: "No such file or directory"
```

---

## Summary

**Test checkpoints (`test_checkpoints*`):**

- ❌ Old artifacts from Oct 27
- ❌ Not created by current tests
- ✅ Safe to delete immediately
- 💾 Size: 16.4 MB

**Production checkpoints (`checkpoints*`):**

- ⚠️ Old training runs (pre-organization)
- 🤔 May have value for analysis
- ✅ Should archive to `runs/archive/`
- 💾 Size: 1.3 GB

**New integration tests:**

- ✅ Use temp directories (auto-cleanup)
- ✅ No pollution of project root
- ✅ No conflicts with production

**Recommendation:**

1. Delete `test_checkpoints*` (no value)
2. Archive `checkpoints*` to `runs/archive/` (preserve data)
3. Delete archive later when safe
