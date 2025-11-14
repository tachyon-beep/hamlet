Title: epsilon_greedy_action_selection samples uniformly when all actions are invalid

Severity: medium
Status: open

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
