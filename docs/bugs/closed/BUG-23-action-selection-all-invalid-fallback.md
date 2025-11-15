Title: epsilon_greedy_action_selection samples uniformly when all actions are invalid

Severity: medium
Status: RESOLVED
Resolved Date: 2025-11-15

Subsystem: exploration/action-selection
Affected Version/Branch: main

Affected Files:
- `src/townlet/exploration/action_selection.py`

Description:
- When `action_masks` has a row with all False (e.g., dead agents), the code adds 1e-8 and normalizes, creating a uniform distribution across all actions (including invalid). Then `torch.multinomial` samples from that, yielding an invalid action.

Reproduction:
1) Provide `action_masks` with a row of all False.
2) Call `epsilon_greedy_action_selection` → returns some action index even though none are valid.

Expected Behavior:
- If no actions are valid, return a sentinel (e.g., WAIT if present) or keep greedy action unchanged; do not sample invalid actions.

Actual Behavior:
- Uniform sampling across all actions.

Root Cause:
- Defensive epsilon add and normalization path does not special-case zero-sum masks.

Proposed Fix (Breaking OK):
- Detect rows with `valid_count == 0`; set `random_actions` to the greedy action or a configured safe action (WAIT index if available), or leave unchanged and let env ignore.

Migration Impact:
- None for typical runs (env usually guarantees at least one valid action except for terminal agents).

Alternatives Considered:
- Skip selection for those agents (carry last action); complicates downstream code.

Tests:
- Add unit test for all-false row; assert fallback behavior.

Owner: exploration

---

## RESOLUTION

**Fixed By**: Systematic debugging process using superpowers:systematic-debugging skill
**Resolution Date**: 2025-11-15

### Implementation Summary

**Phase 1 (Root Cause Investigation)**: Identified two-part issue:
1. Defensive epsilon (`probs + 1e-8`) prevented division by zero but created non-zero probabilities for invalid actions
2. `torch.multinomial` then sampled from this uniform distribution over invalid actions
3. Dead agents (health ≤ 0 OR energy ≤ 0) have all actions masked in `VectorizedHamletEnv:923`

**Phase 2 (Pattern Analysis)**: Found that:
- Greedy action calculation on all `-inf` values returns argmax=0 (arbitrary)
- Need argmax of **unmasked** Q-values for dead agents instead
- WAIT action available but greedy action is more principled (respects learned policy)

**Phase 3 (Hypothesis)**: Root cause is twofold:
- Greedy actions use argmax of all `-inf` (returns 0), not argmax of original Q-values
- Random action sampling doesn't detect all-invalid rows and falls back to greedy

**Phase 4 (Implementation)**:

1. **Core Fix** (`src/townlet/exploration/action_selection.py`):
   - Lines 46-48: Added all-invalid row detection (`valid_count == 0`)
   - Lines 59-62: Fixed greedy action for all-invalid rows (use argmax of unmasked Q-values)
   - Lines 79-81: Fixed random action fallback for all-invalid rows

2. **Test Coverage** (`tests/test_townlet/unit/exploration/test_action_selection.py` - NEW):
   - `test_all_false_action_masks_returns_greedy_action()` - BUG-23 test case
   - 5 additional tests covering edge cases and normal behavior
   - Total: 6 comprehensive test cases

### Test Results
```
tests/test_townlet/unit/exploration/test_action_selection.py ...... [100%]
6 passed

tests/test_townlet/unit/population/test_action_selection.py ............ [100%]
12 passed

tests/test_townlet/unit/exploration/ ....................... [100%]
81 passed
```

**Coverage**: 100% line coverage (25/25 statements, 6/6 branches)

### Code Review
- Reviewer: feature-dev:code-reviewer subagent
- Status: ✅ APPROVED
- Score: 98/100 (Excellent)
- Findings:
  - Correctly handles all-False edge case
  - Appropriate fallback behavior (greedy from unmasked Q-values)
  - Minimal performance overhead (<1% typical, <5% worst case)
  - Excellent test coverage (6 comprehensive tests)
  - No new edge cases introduced

### Files Modified
1. `src/townlet/exploration/action_selection.py` - Core fix (7 lines added)
2. `tests/test_townlet/unit/exploration/test_action_selection.py` - New test suite (6 tests)

### Migration Notes
- Pure bug fix with no API changes
- Only affects edge case (dead agents with all actions invalid)
- Zero migration required

### Impact
- ✅ Eliminates invalid action sampling for dead agents
- ✅ Falls back to greedy action based on learned policy (principled)
- ✅ Maintains GPU-native vectorized operations (performance)
- ✅ Short-circuit optimization (no overhead when no dead agents)
- ✅ Comprehensive test coverage (100% line coverage)
