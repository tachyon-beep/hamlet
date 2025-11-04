# Remaining MyPy Type Errors

## Summary

**Total Remaining**: 44 errors across 3 files
**Completed**: 16 errors fixed (27% done)

## Files Fixed ✅
- `database.py` (3 errors) - SQL parameter typing
- `replay.py` (4 errors) - Import stubs, database typing
- `recorder.py` (4 errors) - Import stubs, position tuple casting
- `video_export.py` (5 errors) - None checks, method name

## Remaining Errors by File

### 1. `vectorized.py` (14 errors) - PRIORITY: HIGH

**Union Type Issues** (8 errors):
- Lines 518, 634: `SequentialReplayBuffer` missing `push()` / `sample()`
- Line 548: `ReplayBuffer` missing `sample_sequences()`
- Need to narrow union types with `isinstance()` checks

**Tensor Type Issues** (6 errors):
- Lines 452, 559, 570, 631: `"Tensor" not callable`
- Lines 599, 642: Assigning Tensor to float variable
- Lines 606, 614, 647, 652: Calling `.item()` / `.backward()` on float

**Fix Strategy**:
```python
# Union narrowing example
if isinstance(self.replay_buffer, SequentialReplayBuffer):
    batch = self.replay_buffer.sample_sequences(batch_size)
else:
    batch = self.replay_buffer.sample(batch_size)

# Tensor typing example
loss: torch.Tensor = F.mse_loss(...)  # Not float!
loss_value: float = loss.item()  # Then extract float
```

### 2. `live_inference.py` (25 errors) - PRIORITY: MEDIUM

**None Check Needed** (20+ errors):
- Lines 319, 660, 663, 682, 686, 721, 738-739, 762, 768, 772, 779-780, 783
- `VectorizedPopulation | None` and `VectorizedHamletEnv | None` accessed without guards

**Missing Attributes** (3 errors):
- Line 770: `AFFORDANCE_CONFIGS` doesn't exist in module
- Lines 836-839: Dict indexing without None check

**Fix Strategy**:
```python
# Add assertions or guards
if self.population is None or self.env is None:
    raise RuntimeError("System not initialized")

# Then access safely
q_network = self.population.q_network
positions = self.env.positions
```

### 3. `unified_server.py` (6 errors) - PRIORITY: LOW

**None Checks Needed** (4 errors):
- Lines 350, 392, 479: `Path | None` accessed without check
- Line 490: Incompatible float assignment

**Missing Attributes** (2 errors):
- Lines 482, 490, 493, 495-496: `frontend_port` attribute doesn't exist

**Fix Strategy**:
```python
# Add type narrowing
if self.checkpoint_dir is None:
    raise ValueError("Checkpoint directory required")

# Use correct attribute name (likely `self._frontend_port` or similar)
```

### 4. `runner.py` (1 error) - PRIORITY: LOW

Line 417: Function body not checked (add `--check-untyped-defs` or add types)

## Recommended Fix Order

1. **vectorized.py** (Core training code - HIGH priority)
   - Fix Union types first (replay buffer conditional logic)
   - Fix Tensor/float type annotations
   - Estimated time: 30-45 mins

2. **live_inference.py** (Demo/visualization - MEDIUM priority)
   - Add initialization checks at start of methods
   - Add None guards throughout
   - Estimated time: 20-30 mins

3. **unified_server.py** (Server code - LOW priority)
   - Fix attribute names
   - Add path checks
   - Estimated time: 10-15 mins

4. **runner.py** (Entry point - LOW priority)
   - Add type annotations to untyped function
   - Estimated time: 5 mins

## Total Estimated Time: 1.5-2 hours

## Commands

```bash
# Check specific file
uv run mypy src/townlet/population/vectorized.py

# Check all with detailed output
uv run mypy src/townlet --show-error-codes

# Run after fixing
uv run mypy src/townlet && echo "✅ All type checks passing!"
```
