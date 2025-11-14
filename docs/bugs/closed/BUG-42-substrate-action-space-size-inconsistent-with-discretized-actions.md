Title: Continuous2D discretized actions can exceed substrate.action_space_size contract

Severity: medium
Status: CLOSED - Already Fixed
Resolution Date: 2025-11-14
Resolution: ContinuousSubstrate overrides action_space_size (line 119) to dynamically return len(cached_actions), correctly handling discretization

Subsystem: substrate/continuous + environment/action_builder
Affected Version/Branch: main

Affected Files:
- `src/townlet/substrate/base.py:40`
- `src/townlet/substrate/continuous.py:516`
- `src/townlet/environment/action_builder.py:21`

Description:
- `SpatialSubstrate.action_space_size` defines a canonical contract: spatial substrates expose `2 * position_dim + 2` discrete actions (±movement per dimension + INTERACT + WAIT), and aspatial exposes 2 actions.
- `Continuous2DSubstrate.get_default_actions()` implements a discretized action scheme when `action_discretization` is provided:
  - It generates a `STOP` action, plus `num_directions * (num_magnitudes - 1)` movement actions, and then adds INTERACT and WAIT.
  - For example, `num_directions=32`, `num_magnitudes=7` yields `1 + 32 * 6 + 2 = 195` actions.
- However, `ContinuousSubstrate` does not override `action_space_size`, so `SpatialSubstrate.action_space_size` still reports `2 * position_dim + 2` (= 6 for 2D), which no longer matches the actual number of default actions when discretization is enabled.
- `ActionSpaceBuilder` and other consumers rely on `substrate.action_space_size` for bookkeeping (e.g., movement mask size, counts), so this mismatch can cause subtle inconsistencies between expected and actual action counts.

Reproduction:
1) Configure a continuous 2D substrate (`type: continuous`, `dimensions: 2`) with:
   - `action_discretization: { num_directions: 32, num_magnitudes: 7 }`.
2) Build the substrate via `SubstrateFactory.build()` and inspect:
   - `substrate.position_dim == 2` → `substrate.action_space_size` (via base class) reports `2 * 2 + 2 = 6`.
   - `len(substrate.get_default_actions())` returns 195 actions.
3) Downstream, `ActionSpaceBuilder` and environment logic assume the substrate space is size 6 when using `action_space_size`, but in practice use a much larger action vocabulary.

Expected Behavior:
- `action_space_size` should always reflect the true number of substrate-provided actions, including any discretization, so that:
  - `substrate.action_space_size == len(substrate.get_default_actions())` holds for all concrete substrates.
  - Higher-level components (action space builder, validators, masks) can rely on that contract without special casing continuous/discretized substrates.

Actual Behavior:
- For discretized Continuous2D substrates, `action_space_size` reflects the legacy 2D formula (6), while `get_default_actions()` returns a large, discretized action set (e.g., 195).
- This divergence is currently “papered over” because `ActionSpaceBuilder` uses `len(actions)` when constructing the composed space, but any code that uses `substrate.action_space_size` (e.g., future validators or masks) will see the wrong value and can go out of sync.

Root Cause:
- The base `SpatialSubstrate.action_space_size` implements the `2N + 2` formula for all spatial substrates.
- Discretized Continuous2D actions were added as an extended feature in `Continuous2DSubstrate` without overriding `action_space_size` to reflect the new, larger action vocabulary.

Proposed Fix (Breaking OK):
- Override `action_space_size` in `Continuous2DSubstrate` (and any other discretized substrates) to:
  - If `action_discretization is None`: return `2 * position_dim + 2` (legacy behavior).
  - Else: return `len(self.get_default_actions())` or compute `1 + num_directions * (num_magnitudes - 1) + 2` explicitly.
- Update docstrings and comments to make it clear that continuous substrates may expose “wider” action spaces when discretization is enabled.

Migration Impact:
- Code that relied on `action_space_size == 6` for Continuous2D will now see the true discretized size; any such consumers should be updated to not hardcode the legacy value.
- Default configs (without `action_discretization`) remain unaffected.

Alternatives Considered:
- Add a separate `effective_action_space_size` property for discretized action sets:
  - Rejected; increases complexity and ambiguity; better to make `action_space_size` reflect reality.

Tests:
- Extend substrate unit tests:
  - For Continuous2D with no discretization: assert `action_space_size == 6` and `len(get_default_actions()) == 6`.
  - For Continuous2D with `num_directions=8, num_magnitudes=3`: assert `action_space_size == len(get_default_actions()) == 1 + 8 * 2 + 2 = 19`.

Owner: substrate
Links:
- `src/townlet/substrate/base.py:action_space_size`
- `src/townlet/substrate/continuous.py:Continuous2DSubstrate.get_default_actions`
- `src/townlet/environment/action_builder.py:ActionSpaceBuilder`
