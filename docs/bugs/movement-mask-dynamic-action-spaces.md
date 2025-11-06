# Bug Analysis: Movement Mask for Dynamic Action Spaces

**Status**: 🔴 Critical Bug
**Priority**: P1
**Severity**: High - Breaks game mechanics for aspatial and 1D substrates
**Location**: `src/townlet/environment/vectorized_env.py:595`
**Discovered**: Code review, task-002a

## Executive Summary

The movement cost mask uses a hardcoded assumption (`actions < 4`) that only works for 2D/3D substrates. For substrates with fewer movement actions (aspatial, 1D), INTERACT and WAIT actions are incorrectly flagged as movement, causing agents to pay double costs.

## Root Cause

**Current Code** (Line 595):
```python
movement_mask = actions < 4
```

**Problem**: This assumes the action space has 4 movement actions (UP/DOWN/LEFT/RIGHT), which is only true for 2D substrates.

**Action Space Formula** (from `SpatialSubstrate.action_space_size`):
```
action_space_size = 2 * position_dim + 2  # (except aspatial = 2)
```

**Action Index Mapping**:
```
Actions [0 to 2*position_dim-1]: Movement (±1 per dimension)
Action [2*position_dim]:         INTERACT
Action [2*position_dim+1]:       WAIT
```

## Impact Analysis

### Aspatial Substrate (0D)

**Expected Action Space**:
- Action 0: INTERACT
- Action 1: WAIT
- **No movement actions**

**Current Behavior**:
- `actions < 4` matches BOTH actions (0, 1)
- Both INTERACT and WAIT pay full movement costs:
  - Energy: -0.5% (move_energy_cost)
  - Hygiene: -0.3%
  - Satiation: -0.4%
- Then WAIT also pays wait_energy_cost (-0.1%) → **double-charged**
- Then INTERACT also pays interact_energy_cost (-0.3%) → **double-charged**

**Consequence**: Every action in aspatial environments drains 1.2% energy plus hygiene/satiation penalties, making survival impossible.

### 1D Substrate (Continuous1D, hypothetical Grid1D)

**Expected Action Space**:
- Action 0: LEFT
- Action 1: RIGHT
- Action 2: INTERACT
- Action 3: WAIT

**Current Behavior**:
- `actions < 4` matches ALL actions (0, 1, 2, 3)
- INTERACT (action 2) and WAIT (action 3) are treated as movement
- They pay:
  - Movement penalty: -0.5% energy, -0.3% hygiene, -0.4% satiation
  - Then their specific cost on top of that
- **WAIT**: Pays 0.5% + 0.1% = 0.6% energy total (should be 0.1%)
- **INTERACT**: Pays 0.5% + 0.3% = 0.8% energy total (should be 0.3%)

**Consequence**: Agents are punished 5x-6x more than intended for waiting or interacting.

### 2D/3D Substrates (Grid2D, Grid3D, Continuous2D, Continuous3D)

**Expected Action Space (2D)**:
- Actions 0-3: UP, DOWN, LEFT, RIGHT
- Action 4: INTERACT
- Action 5: WAIT

**Current Behavior**:
- `actions < 4` correctly matches only movement actions (0-3)
- INTERACT (4) and WAIT (5) are NOT flagged as movement
- **Works correctly** ✅

**Expected Action Space (3D)**:
- Actions 0-5: ±X, ±Y, ±Z
- Action 6: INTERACT
- Action 7: WAIT

**Current Behavior**:
- `actions < 4` matches actions 0-3 (only ±X, ±Y)
- Actions 4-5 (±Z) are NOT flagged as movement → **incorrect**
- Movement in Z-direction doesn't pay movement costs! 🐛

**Consequence**: 3D substrates have a broken movement mask too (different bug).

## Correct Formula

**Fix**:
```python
# Calculate number of movement actions based on substrate dimensionality
num_movement_actions = 2 * self.substrate.position_dim  # 0 for aspatial, 2 for 1D, 4 for 2D, 6 for 3D

# Movement mask: only actions [0, num_movement_actions) are movement
movement_mask = actions < num_movement_actions
```

**Examples**:
- **Aspatial (0D)**: `actions < 0` → No actions flagged as movement ✅
- **1D**: `actions < 2` → Actions 0-1 (LEFT, RIGHT) flagged ✅
- **2D**: `actions < 4` → Actions 0-3 (UP, DOWN, LEFT, RIGHT) flagged ✅
- **3D**: `actions < 6` → Actions 0-5 (±X, ±Y, ±Z) flagged ✅
- **4D**: `actions < 8` → Actions 0-7 flagged ✅

## Verification Strategy

### Test Cases to Write

1. **Test: Aspatial INTERACT/WAIT don't pay movement costs**
   ```python
   def test_aspatial_interact_no_movement_cost():
       """Aspatial INTERACT should not pay movement cost."""
       # Create aspatial env
       # Agent takes INTERACT action (action 0)
       # Verify only interact_energy_cost deducted (not move_energy_cost)
       # Verify hygiene/satiation unchanged
   ```

2. **Test: 1D INTERACT/WAIT don't pay movement costs**
   ```python
   def test_1d_interact_no_movement_cost():
       """1D INTERACT should not pay movement cost."""
       # Create 1D env
       # Agent takes INTERACT action (action 2)
       # Verify only interact_energy_cost deducted
   ```

3. **Test: 3D Z-movement pays movement costs**
   ```python
   def test_3d_z_movement_pays_movement_cost():
       """3D Z-axis movement should pay movement cost."""
       # Create 3D env
       # Agent takes FORWARD action (action 4 or 5, ±Z)
       # Verify move_energy_cost deducted
   ```

4. **Test: 2D movement costs still work**
   ```python
   def test_2d_movement_costs_unchanged():
       """2D movement costs should work as before (regression test)."""
       # Create 2D env
       # Agent takes UP/DOWN/LEFT/RIGHT
       # Verify move_energy_cost deducted
       # Agent takes INTERACT/WAIT
       # Verify move_energy_cost NOT deducted
   ```

### Manual Verification

After fix, verify in live inference:
1. Run aspatial substrate with agents interacting
2. Monitor energy drain rate
3. Confirm INTERACT drains 0.3% (not 0.8%)
4. Confirm WAIT drains 0.1% (not 0.6%)

## Related Code

**Action Index Calculation** (vectorized_env.py:610-635):
```python
# WAIT action index
if self.substrate.position_dim == 1:
    wait_action_idx = 3
elif self.substrate.position_dim == 0:
    wait_action_idx = 1
else:  # 2D or 3D
    wait_action_idx = 5

# INTERACT action index
if self.substrate.position_dim == 1:
    interact_action_idx = 2
elif self.substrate.position_dim == 0:
    interact_action_idx = 0
else:  # 2D or 3D
    interact_action_idx = 4
```

**Note**: These hardcoded mappings are also fragile (fail for 4D+), but can be derived from formula:
```python
interact_action_idx = 2 * self.substrate.position_dim
wait_action_idx = 2 * self.substrate.position_dim + 1
```

## Implementation Notes

### Pre-Fix Checklist
- [ ] Write failing tests for aspatial, 1D, 3D substrates
- [ ] Verify tests fail with current code
- [ ] Document expected vs actual energy costs

### Fix Checklist
- [ ] Replace `movement_mask = actions < 4` with dynamic formula
- [ ] Update INTERACT/WAIT action index calculations to use formula
- [ ] Run all tests to verify fix
- [ ] Test with live inference on aspatial/1D substrates
- [ ] Update any related documentation

### Post-Fix Verification
- [ ] All substrate types charge correct action costs
- [ ] No regression in 2D substrate behavior
- [ ] 3D Z-movement now works correctly
- [ ] Aspatial/1D substrates are now playable

## Timeline

**Discovered**: 2025-11-06 (Code review, task-002a)
**Fix Applied**: 2025-11-06
**Status**: ✅ FIXED
**Verification**: Tests updated to account for base_depletion

## Fix Applied

**Changed Code** (vectorized_env.py:595-637):
```python
# OLD (hardcoded):
movement_mask = actions < 4

# NEW (dynamic):
num_movement_actions = 2 * self.substrate.position_dim
movement_mask = actions < num_movement_actions

# Also updated WAIT and INTERACT action indices to use formulas:
wait_action_idx = 2 * self.substrate.position_dim + 1
interact_action_idx = 2 * self.substrate.position_dim
```

**Key Insight During Testing**:
Energy has a `base_depletion: 0.005` (0.5% per step) which happens on EVERY action. Test expectations needed to account for this:
- Movement: base_depletion (0.5%) + move_energy_cost (0.5%) = 1.0%
- INTERACT: base_depletion (0.5%) + 0 (no movement cost) = 0.5%
- WAIT: base_depletion (0.5%) + wait_energy_cost (0.1%) = 0.6%

## References

- `src/townlet/substrate/base.py:59-84` - Action space formula
- `src/townlet/environment/vectorized_env.py:595-642` - Action cost handling
- TASK-002A: Configurable spatial substrates migration
