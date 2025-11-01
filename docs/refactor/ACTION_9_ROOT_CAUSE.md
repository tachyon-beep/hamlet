# ACTION #9: LSTM Temporal Learning - ROOT CAUSE IDENTIFIED

**Date**: November 1, 2025  
**Status**: 🔴 ROOT CAUSE FOUND - Requires Target Network

---

## Summary

Created failing test (`test_lstm_learns_temporal_sequence`) that validates LSTM can learn temporal dependencies. Test FAILS ❌, confirming the documented issue that "LSTM doesn't learn effectively."

**Root Cause**: Q-target computation resets LSTM hidden state, breaking temporal credit assignment.

---

## The Problem

### Current Training Loop (vectorized.py lines 334-365)

```python
# PASS 1: Collect Q-predictions
self.q_network.reset_hidden_state(batch_size=batch_size)
for t in range(seq_len):
    q_values, _ = self.q_network(obs[t])  # Hidden state: h_0 → h_1 → h_2 ...
    q_pred = q_values[actions[t]]

# PASS 2: Collect Q-targets  
with torch.no_grad():
    for t in range(seq_len):
        # ❌ BUG: Resets hidden state EVERY timestep!
        self.q_network.reset_hidden_state(batch_size=batch_size)
        q_next, _ = self.q_network(next_obs[t])  # Always uses h_0!
        q_target = reward[t] + gamma * max(q_next)
```

**Problem**: Target computation uses `h_0` (zero state) for ALL timesteps, not the accumulated hidden state sequence. This breaks temporal dependencies.

---

## Why This Breaks LSTM Learning

### Example: 3-Step Sequence

**Correct Behavior**:

- t=0: Q(s_0, a_0 | h_0) predicts value with no history ✓
- t=1: Q(s_1, a_1 | h_1) predicts value knowing we came from s_0 ✓  
- t=2: Q(s_2, a_2 | h_2) predicts value knowing full history s_0→s_1 ✓

**Current Bug**:

- t=0: Q_pred uses h_0, Q_target uses h_0 (OK for first step)
- t=1: Q_pred uses h_1 (has history), Q_target uses h_0 (NO history!) ❌
- t=2: Q_pred uses h_2 (full history), Q_target uses h_0 (NO history!) ❌

**Result**: Predictions use temporal context, but targets don't. Network learns to IGNORE memory because targets don't reward using it.

---

## RED Test Evidence

Created `test_lstm_learns_temporal_sequence()` with simple 3-state MDP:

- State 0 → State 1 (action 0)
- State 1 → State 2 (action 1, reward=1.0)
- LSTM must use memory to know which action is correct from State 1

**Result after 100 training iterations**:

```
Q(state_0) = [1.079, 1.175]  # Should prefer action 0, but prefers action 1 ❌
Q(state_1) = [1.139, 0.766]  # Should prefer action 1, but prefers action 0 ❌
```

Q-values are INVERTED - network actively learned the WRONG policy!

---

## The Correct Solution: Target Network

### Option A: Target Network (Recommended)

**What**:  Two copies of Q-network:

1. **Online Network**: Updated every batch
2. **Target Network**: Frozen copy, updated every N steps

**Why**: Target network's hidden states can be properly unrolled without interfering with online network's gradient computation.

**Implementation**:

```python
# Create target network (copy of online network)
self.target_network = RecurrentSpatialQNetwork(...)
self.target_network.load_state_dict(self.q_network.state_dict())

# Training loop
for t in range(seq_len):
    # Online network: collect predictions (WITH gradients)
    q_pred, _ = self.q_network(obs[t])
    
    # Target network: collect targets (NO gradients)
    with torch.no_grad():
        q_target, _ = self.target_network(obs[t])  # Maintains hidden state!

# Update target network every 100 steps
if self.total_steps % 100 == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

**Pros**:

- Standard DQN practice (well-tested)
- Stable training (targets don't change every step)
- Properly handles LSTM hidden states
- ~100 lines of code

**Cons**:

- Slightly slower convergence (targets lag behind)
- Extra memory (2x network size)

### Option B: Detached Hidden States (Complex)

**What**: Store hidden states from first forward pass, use them for second pass.

**Why**: Avoid needing a separate network.

**Implementation**:

```python
# Pass 1: Collect predictions AND hidden states
hidden_states = []
for t in range(seq_len):
    q_pred, hidden = self.q_network(obs[t])
    hidden_states.append((hidden[0].detach(), hidden[1].detach()))

# Pass 2: Use saved hidden states for targets
with torch.no_grad():
    for t in range(seq_len):
        self.q_network.set_hidden_state(hidden_states[t])
        q_target, _ = self.q_network(obs[t+1])
```

**Pros**:

- No extra memory for target network
- Targets use exact same hidden states as predictions

**Cons**:

- Complex state management
- Easy to get wrong (detach/attach bugs)
- Non-standard approach
- Targets update every step (less stable)

---

## Recommended Action

**Implement Option A (Target Network)**:

1. Add `target_network` to `VectorizedPopulation.__init__`
2. Modify LSTM training loop (lines 334-375) to use target network for Q_target
3. Add target network update (soft update τ=0.001 or hard update every 100 steps)
4. Update tests to verify target network is used correctly
5. Run `test_lstm_learns_temporal_sequence` → should PASS ✅

**Estimated Time**: 2-3 hours (implementation + testing)

**Risk**: Low (standard DQN practice, well-documented)

---

## Files to Modify

1. **src/townlet/population/vectorized.py** (lines 28-110, 334-375):
   - Add `target_network` initialization
   - Use target network in LSTM training loop
   - Add target network update logic

2. **tests/test_townlet/test_lstm_temporal_learning.py** (NEW):
   - Keep as regression test
   - Should PASS after fix

3. **tests/test_townlet/test_q_learning.py** (extend):
   - Add test for target network updates
   - Verify target network is used in LSTM mode

---

## Success Criteria

✅ `test_lstm_learns_temporal_sequence` passes (network learns correct Q-values)  
✅ All existing 329 tests still pass  
✅ Coverage remains at 70%+  
✅ Multi-day training shows improved survival (115 → 150+ steps)

---

## Next Steps

1. Implement target network (2-3 hours)
2. Run temporal learning test → GREEN ✅
3. Run full test suite → all pass ✅
4. Start multi-day validation (Phase 3.5)

**This is the actual "network architecture redesign" mentioned in ACTION #9.** The infrastructure exists, but the training algorithm needs fixing.
