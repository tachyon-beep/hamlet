# QUICK-001: DemoRunner Resource Cleanup (Context Manager)

**Status**: Planned
**Priority**: Medium
**Estimated Effort**: 1-2 hours
**Dependencies**: None
**Created**: 2025-11-04

**Keywords**: resource-leak, context-manager, database, cleanup, testing, TDD
**Subsystems**: runner, database, testing
**Files**: `src/townlet/demo/runner.py`, `tests/test_townlet/integration/test_checkpointing.py`
**Methodology**: Test-Driven Development (RED-GREEN-REFACTOR)

---

## AI-Friendly Summary (Skim This First!)

**What**: Add context manager support (`__enter__`/`__exit__`) to DemoRunner for deterministic resource cleanup
**Why**: Tests leak SQLite connections because DemoRunner opens DB in `__init__()` but only closes in `run()`'s finally block
**Scope**: Add context manager protocol to DemoRunner, update 3 test methods to use `with` statements

**Quick Assessment**:

- Current State: ❌ DemoRunner opens DB connection in constructor without cleanup guarantee
- Goal: ✅ DemoRunner works as context manager with guaranteed resource cleanup
- Impact: Tests won't leak connections, production code can safely create DemoRunner for checkpoint ops

**TDD Workflow Summary**:

```
🔴 RED (20 min)    → Write 3 failing tests for context manager
✅ GREEN (45 min)  → Implement _cleanup() + __enter__/__exit__, update tests
🔧 REFACTOR (10 min) → Polish code, verify zero warnings
```

---

## Problem Statement

### Context

`DemoRunner` creates a SQLite database connection in `__init__()` (line 57) but only closes it in the `finally` block of `run()` (line 719). Tests that instantiate `DemoRunner` for checkpoint operations—without calling `run()`—leak database connections, causing ResourceWarnings during test execution.

**Evidence**:

```
ResourceWarning: unclosed database in <sqlite3.Connection object at 0x7c3c277b8310>
ResourceWarning: unclosed database in <sqlite3.Connection object at 0x7c3bb6fd3f10>
```

### Current Limitations

**What Doesn't Work**:

- Tests create `DemoRunner` → manually set up components → call `save_checkpoint()` → test ends
- Database connection remains open until garbage collection (non-deterministic)
- 5 DemoRunner instances in test_checkpointing.py leak connections
- No documented cleanup method for test/one-off usage

**What We're Missing**:

- Context manager protocol for deterministic cleanup
- Clear pattern for "create runner, do checkpoint op, cleanup"
- Guarantee that resources are released regardless of `run()` being called

### Use Cases

**Primary Use Case**:
Tests that use DemoRunner for checkpoint operations (save/load) without running full training loop need guaranteed cleanup.

**Secondary Use Cases**:

- Production scripts that load checkpoints for analysis
- Inference servers that create DemoRunner to access checkpoint metadata
- Debugging tools that instantiate DemoRunner temporarily

---

## Solution Design

### Overview

Add Python context manager protocol (`__enter__`/`__exit__`) to `DemoRunner`. Extract cleanup logic into a private `_cleanup()` method called from both `__exit__` and `run()`'s finally block. Update affected tests to use `with` statements.

### Technical Approach

**Implementation Steps**:

1. **Extract cleanup logic** (`runner.py:719-722`) into `_cleanup()` method
2. **Add context manager** methods `__enter__()` and `__exit__()`
3. **Update `run()` finally** block to call `_cleanup()`
4. **Update 3 test methods** in `test_checkpointing.py` to use `with` statements

**Key Design Decisions**:

- **Use context manager**: Pythonic, explicit, guarantees cleanup via `with` statement
- **Extract _cleanup()**: Avoid code duplication between `run()` and `__exit__()`
- **Backward compatible**: Existing `run()` usage unchanged, context manager is optional for tests

### Edge Cases

**Must Handle**:

- ****exit** called when runner partially initialized**: Use `hasattr()` checks before cleanup
- **Exception during **init****: Database may not exist yet, _cleanup() must handle gracefully
- **Multiple **exit** calls**: Make _cleanup() idempotent (safe to call multiple times)

---

## Implementation Plan (TDD Approach)

### Phase 1: RED - Write Failing Tests (20 min)

**Goal**: Write tests that demonstrate desired behavior BEFORE implementing

**File**: `tests/test_townlet/integration/test_checkpointing.py` (add new test class at end)

**Add new test class**:

```python
class TestDemoRunnerResourceManagement:
    """Test DemoRunner context manager and resource cleanup."""

    def test_runner_closes_database_on_context_exit(self, cpu_device):
        """DemoRunner should close database when exiting context manager."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config = {
                "environment": {"grid_size": 8, "partial_observability": False},
                "population": {"num_agents": 1, "network_type": "simple"},
                "curriculum": {"max_steps_per_episode": 100},
                "exploration": {},
            }
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Create runner in context manager
            with DemoRunner(
                config_dir=config_path.parent,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
                training_config_path=config_path,
            ) as runner:
                # Database should be open
                assert hasattr(runner, 'db')
                assert runner.db.conn is not None
                # Store connection reference
                conn = runner.db.conn

            # After exiting context, connection should be closed
            # SQLite connection has no is_closed() but we can check it raises
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                conn.execute("SELECT 1")

    def test_runner_cleanup_is_idempotent(self, cpu_device):
        """Calling _cleanup() multiple times should be safe."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config = {
                "environment": {"grid_size": 8, "partial_observability": False},
                "population": {"num_agents": 1, "network_type": "simple"},
                "curriculum": {"max_steps_per_episode": 100},
                "exploration": {},
            }
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            runner = DemoRunner(
                config_dir=config_path.parent,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
                training_config_path=config_path,
            )

            # Call cleanup multiple times - should not raise
            runner._cleanup()
            runner._cleanup()  # Second call should be safe
            runner._cleanup()  # Third call should be safe

    def test_runner_context_manager_propagates_exceptions(self, cpu_device):
        """Context manager should propagate exceptions, not suppress them."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config = {
                "environment": {"grid_size": 8, "partial_observability": False},
                "population": {"num_agents": 1, "network_type": "simple"},
                "curriculum": {"max_steps_per_episode": 100},
                "exploration": {},
            }
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Exception inside with block should propagate
            with pytest.raises(ValueError, match="test exception"):
                with DemoRunner(
                    config_dir=config_path.parent,
                    db_path=tmpdir / "test.db",
                    checkpoint_dir=checkpoint_dir,
                    max_episodes=1,
                    training_config_path=config_path,
                ) as runner:
                    raise ValueError("test exception")
```

**Add import at top of file**:

```python
import sqlite3  # Add to imports
```

**Run tests and watch them FAIL**:

```bash
pytest tests/test_townlet/integration/test_checkpointing.py::TestDemoRunnerResourceManagement -v
```

**Expected failures**:

- `test_runner_closes_database_on_context_exit`: AttributeError: **enter** not found
- `test_runner_cleanup_is_idempotent`: AttributeError: _cleanup not found
- `test_runner_context_manager_propagates_exceptions`: AttributeError: **enter** not found

**Verification**:

- [ ] All 3 new tests fail with expected errors
- [ ] Test failures are clear and correct

---

### Phase 2: GREEN - Implement Context Manager (25 min)

**Goal**: Write minimal code to make tests pass

#### Step 2.1: Add _cleanup() method

**File**: `src/townlet/demo/runner.py` (after `_handle_shutdown` method, around line 110)

```python
def _cleanup(self):
    """Internal cleanup method for resources.

    Safe to call multiple times (idempotent).
    Handles partial initialization gracefully.
    """
    # Shutdown recorder if enabled
    if hasattr(self, 'recorder') and self.recorder is not None:
        logger.info("Shutting down episode recorder...")
        try:
            self.recorder.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down recorder: {e}")

    # Close database connection
    if hasattr(self, 'db'):
        try:
            self.db.close()
        except Exception as e:
            logger.warning(f"Error closing database: {e}")

    # Close TensorBoard logger
    if hasattr(self, 'tb_logger'):
        try:
            self.tb_logger.close()
        except Exception as e:
            logger.warning(f"Error closing TensorBoard logger: {e}")
```

#### Step 2.2: Add context manager protocol

**File**: `src/townlet/demo/runner.py` (after `_cleanup` method)

```python
def __enter__(self):
    """Enter context manager - return self for 'with' statement."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Exit context manager - cleanup resources.

    Args:
        exc_type: Exception type if exception occurred
        exc_val: Exception value if exception occurred
        exc_tb: Exception traceback if exception occurred

    Returns:
        False to propagate exceptions (don't suppress)
    """
    self._cleanup()
    return False  # Don't suppress exceptions
```

#### Step 2.3: Update run() to use _cleanup()

**File**: `src/townlet/demo/runner.py:697-722`

**Replace existing finally block**:

```python
def run(self):
    try:
        # ... training loop (no changes) ...
    finally:
        # Save final checkpoint
        logger.info("Training complete, saving final checkpoint...")
        self.save_checkpoint()

        # Phase 4 - Log final metrics with hyperparameters
        if self.population is not None:
            final_metrics = {
                "final_episode": self.current_episode,
                "total_training_steps": self.population.total_steps,
            }
            if hasattr(self, "hparams"):
                self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics=final_metrics)

        self.db.set_system_state("training_status", "completed")

        # Use extracted cleanup method
        self._cleanup()
```

**Run tests and watch them PASS**:

```bash
pytest tests/test_townlet/integration/test_checkpointing.py::TestDemoRunnerResourceManagement -v
```

**Verification**:

- [ ] All 3 new tests pass
- [ ] Existing tests still pass (no regressions)

---

### Phase 3: GREEN - Update Existing Tests (20 min)

**Goal**: Convert existing tests to use context manager (eliminate resource warnings)

**File**: `tests/test_townlet/integration/test_checkpointing.py`

**Update 3 test methods to use `with` statements**:

1. **test_runner_checkpoint_includes_all_components** (line 565-641)
2. **test_runner_checkpoint_preserves_episode_number** (line 643-714)
3. **test_runner_checkpoint_round_trip_preserves_training_state** (line 716-817)

**Pattern for each test**:

```python
# OLD:
runner = DemoRunner(...)
runner.env = VectorizedHamletEnv(...)
runner.save_checkpoint()

# NEW:
with DemoRunner(...) as runner:
    runner.env = VectorizedHamletEnv(...)
    runner.save_checkpoint()
```

**Example - test_runner_checkpoint_includes_all_components**:

```python
def test_runner_checkpoint_includes_all_components(self, cpu_device):
    """Runner checkpoint should include state from all components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint_dir = tmpdir / "checkpoints"
        checkpoint_dir.mkdir()

        # ... config setup (no change) ...

        # Create runner with context manager
        with DemoRunner(
            config_dir=config_path.parent,
            db_path=tmpdir / "test.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=1,
            training_config_path=config_path,
        ) as runner:
            # Manually initialize components
            runner.env = VectorizedHamletEnv(...)
            runner.curriculum = AdversarialCurriculum(...)
            # ... rest of setup ...

            # Save checkpoint
            runner.save_checkpoint()

            # Verify checkpoint structure
            checkpoint_file = list(checkpoint_dir.glob("*.pt"))[0]
            checkpoint = torch.load(checkpoint_file, weights_only=False)

            assert "episode" in checkpoint
            assert "population_state" in checkpoint
            # ... rest of assertions ...
```

**Run tests and verify NO ResourceWarnings**:

```bash
pytest tests/test_townlet/integration/test_checkpointing.py -W error::ResourceWarning
```

**Verification**:

- [ ] All updated tests pass
- [ ] Zero ResourceWarnings
- [ ] Tests fail with `-W error::ResourceWarning` if warnings present

---

### Phase 4: REFACTOR - Code Review & Polish (10 min)

**Goal**: Clean up, ensure code quality

**Checks**:

1. [ ] _cleanup() has proper docstring
2. [ ] Context manager methods documented clearly
3. [ ] hasattr() checks prevent AttributeError on partial init
4. [ ] Exception handling in _cleanup() prevents cascade failures
5. [ ] run() finally block uses _cleanup() (no duplication)
6. [ ] All tests green

**Run full test suite**:

```bash
# All checkpointing tests
pytest tests/test_townlet/integration/test_checkpointing.py -v

# Full suite to check for regressions
pytest tests/test_townlet/ -v

# Resource warning detection
pytest tests/test_townlet/integration/test_checkpointing.py -W error::ResourceWarning
```

**Final Verification**:

- [ ] 22 total tests pass (19 existing + 3 new)
- [ ] Zero ResourceWarnings
- [ ] No regressions in other test files
- [ ] Code is clean and well-documented

---

## Testing Strategy

**Test Requirements**:

- **New Unit Tests**: 3 tests for context manager behavior (Phase 1 - RED)
  - `test_runner_closes_database_on_context_exit`: Verify connection closed after `with` block
  - `test_runner_cleanup_is_idempotent`: Verify _cleanup() can be called multiple times
  - `test_runner_context_manager_propagates_exceptions`: Verify exceptions aren't suppressed
- **Existing Tests**: All 19 checkpointing tests continue to pass
- **Updated Tests**: 3 tests converted to use context manager (Phase 3 - GREEN)
- **Resource Warning Detection**: Run with `-W error::ResourceWarning` to fail on leaks

**Coverage Target**: No new coverage needed (fixes existing code, adds cleanup path)

**Test-Driven Development Flow**:

✅ **RED Phase**:

- [ ] Write 3 failing tests for context manager behavior
- [ ] Verify tests fail with expected errors (AttributeError, etc.)
- [ ] Confirm test failures demonstrate what we're building

✅ **GREEN Phase**:

- [ ] Implement _cleanup() method
- [ ] Add **enter** and **exit** methods
- [ ] Update run() to use _cleanup()
- [ ] Verify new tests pass
- [ ] Update 3 existing tests to use context manager
- [ ] Verify all 22 tests pass (19 existing + 3 new)

✅ **REFACTOR Phase**:

- [ ] Review code quality
- [ ] Ensure proper error handling
- [ ] Verify zero ResourceWarnings
- [ ] Confirm backward compatibility

---

## Acceptance Criteria

**Must Have (TDD Verified)**:

**RED Phase Complete**:

- [ ] 3 new tests written and failing correctly
- [ ] Test failures demonstrate desired behavior
- [ ] Tests clearly document requirements

**GREEN Phase Complete**:

- [ ] DemoRunner implements context manager protocol (`__enter__`, `__exit__`)
- [ ] `_cleanup()` method is idempotent and safe
- [ ] All 3 new tests pass
- [ ] All 3 affected existing tests updated to use `with` statements
- [ ] All 22 tests pass (19 existing + 3 new)
- [ ] No regressions in existing tests

**REFACTOR Phase Complete**:

- [ ] No ResourceWarnings when running test suite
- [ ] Tests run 100% clean with `-W error::ResourceWarning`
- [ ] Code quality: proper docstrings, error handling, hasattr() guards
- [ ] Backward compatible: `runner.run()` usage unchanged

**Success Metrics**:

- **Test Coverage**: 22/22 tests passing (100%)
- **Resource Safety**: Zero ResourceWarnings with `-W error::ResourceWarning`
- **Idempotency**: _cleanup() can be called multiple times safely
- **Exception Handling**: Context manager propagates exceptions correctly

---

## Risk Assessment

**Technical Risks**:

- ✅ **LOW**: Context manager is standard Python pattern, well-understood
- ✅ **LOW**: _cleanup() extracted from existing finally block (proven code)
- ⚠️ **MEDIUM**: Must ensure _cleanup() handles partial initialization gracefully

**Mitigation**:

- Use `hasattr()` checks before accessing `self.db`, `self.tb_logger`, `self.recorder`
- Wrap cleanup operations in try/except to prevent cascade failures
- Test with partially-initialized runner (e.g., exception during **init**)

---

## Future Enhancements (Out of Scope)

**Not Included**:

- Make `DemoDatabase` a context manager (lower-level abstraction)
- Add `__del__` destructor (non-deterministic, not needed with context manager)
- Refactor other components to use context managers (TensorBoardLogger, EpisodeRecorder)

**Rationale**: This task focuses on fixing immediate test resource leaks. Context manager support for other components can be added incrementally as needed.

---

## References

**Related Tasks**:

- None (standalone fix)

**Code Files**:

- `src/townlet/demo/runner.py:23-722` - DemoRunner class implementation
- `src/townlet/demo/database.py:8-341` - DemoDatabase class (where connection is created/closed)
- `tests/test_townlet/integration/test_checkpointing.py:565-817` - Affected test methods

**Documentation**:

- Python Context Managers: https://docs.python.org/3/reference/datamodel.html#context-managers
- PEP 343: The "with" Statement: https://peps.python.org/pep-0343/

---

**ROOT CAUSE ANALYSIS**:

DemoRunner architecture flaw: resources acquired in `__init__()` but only released in `run()`'s finally block. This violates resource management best practices - acquire/release should be paired symmetrically. Tests that use DemoRunner for checkpoint operations (without calling `run()`) have no way to trigger cleanup.

**EVIDENCE**:

- 5 DemoRunner instances created in test_checkpointing.py
- None call `run()` - they call `save_checkpoint()` or `load_checkpoint()` directly
- Database connections remain open until garbage collection
- ResourceWarnings appear asynchronously during later tests (GC timing)

**FIX**:
Context manager protocol provides explicit, deterministic resource cleanup via `with` statement, independent of whether `run()` is called.

---

**END OF TASK SPECIFICATION**
