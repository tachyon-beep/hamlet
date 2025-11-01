# ACTION #9.2 COMPLETE: Sequential LSTM Training

**Date:** 2025-01-XX  
**Status:** ✅ COMPLETE - All 262 tests passing

## Problem Summary

RecurrentSpatialQNetwork was being trained incorrectly:

- Hidden state reset for each training batch
- Sampled random individual transitions (no temporal continuity)
- LSTM never learned to use memory across timesteps

## Solution Implemented

### 1. Dual Buffer System

```python
if self.is_recurrent:
    self.replay_buffer = SequentialReplayBuffer(...)  # Stores complete episodes
    self.current_episodes = [{...} for _ in range(num_agents)]  # Accumulators
else:
    self.replay_buffer = ReplayBuffer(...)  # Stores individual transitions
```

### 2. Episode Accumulation

During runtime (lines 283-300 in vectorized.py):

- Append each transition to `current_episodes[i]`
- Store complete episode when `done=True` (lines 438-453)
- Reset accumulator for next episode

### 3. Sequential Training Loop

When training (lines 332-388):

- Sample sequences: `sample_sequences(batch_size=16, seq_len=8)`
- Unroll LSTM through time maintaining hidden state
- Compute Q-learning loss for each timestep
- No hidden state reset between timesteps in sequence

### 4. Critical Bug Fix: Variable Scope

**Bug:** Training loop reused `next_obs` variable name (lines 367, 370), overwriting environment's `next_obs` from line 278.

**Symptom:** At step where training first happens:

```
[DEBUG after env.step] next_obs shape: torch.Size([4, 50])   # Correct from env
[DEBUG line 411] next_obs shape: torch.Size([16, 50])        # Wrong! Overwritten by training batch
```

**Fix:** Renamed to `next_obs_batch` in training loop to avoid collision.

## Testing

### Test Changes

**test_recurrent_parameters_update** (test_q_learning.py):

- Increased from 50 → 200 steps
- **Reason:** Episodes take ~70-80 steps to complete, need 16+ episodes in buffer before training
- First training happens around step 88 (when buffer reaches 16 episodes)
- Test now confirms LSTM parameters update during training ✅

### Test Results

All 262 tests passing:

- Core business logic: 92-100% coverage  
- Overall project: 68% coverage
- No regressions

## Files Modified

1. **src/townlet/population/vectorized.py** (487 lines):
   - Added dual buffer system (lines 89-117)
   - Added episode accumulation (lines 283-300)  
   - Added episode storage (lines 438-453)
   - Added sequential training loop (lines 332-388)
   - Fixed variable scope bug: `next_obs` → `next_obs_batch` (lines 367, 370, 374)

2. **tests/test_townlet/test_q_learning.py**:
   - test_recurrent_parameters_update: 50 → 200 steps (line 258)

## Validation

### Before (Broken)

- LSTM hidden state reset for each training batch
- Random transition sampling broke temporal continuity
- LSTM parameters never updated
- POMDP agents couldn't learn memory-dependent behaviors

### After (Working)

✅ Episodes accumulated correctly  
✅ Complete episodes stored in sequential buffer  
✅ Training samples sequences (8 timesteps each)  
✅ LSTM unrolled through time with persistent hidden state  
✅ LSTM parameters update during training  
✅ All 262 tests passing

## Next Steps

### Immediate: Validate POMDP Performance

- Run Level 2 POMDP training (10K episodes)
- Measure survival time improvement vs baseline
- Expected: 150+ steps (vs ~115 without proper memory)

### Future: ACTION #9 Completion

- If performance gains sufficient: ACTION #9 DONE
- If issues remain: Consider full network architecture redesign
- Monitor for LSTM gradient flow issues

## Technical Details

### Buffer Requirements

- Recurrent: 16 episodes minimum (vs 64 transitions for feedforward)
- Training frequency: Every 4 steps (same for both)
- Sequence length: 8 timesteps per sequence

### Memory Overhead

- Recurrent networks require full episode storage (vs single transitions)
- ~4x memory usage for episode buffer vs transition buffer
- Acceptable tradeoff for proper LSTM training

## Lessons Learned

1. **Variable Scope Matters**: Classic Python bug - reusing variable names in different scopes
2. **Debug Strategically**: Added instrumentation at critical points to trace state flow
3. **Test Duration**: Test assumptions about episode length - 50 steps wasn't enough!
4. **Buffer Size**: Min training threshold (16 vs 64) depends on what's being stored

---

**Result:** Sequential LSTM training now working correctly. POMDP agents can finally learn memory-dependent behaviors!
