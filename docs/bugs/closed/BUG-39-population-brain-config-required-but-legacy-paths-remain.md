Title: VectorizedPopulation requires brain_config but still exposes unused legacy parameters and branches

Severity: medium
Status: closed
Date Closed: 2025-11-15
Resolution: Fixed in commit 02ebe3b (Nov 14, 2025)

Subsystem: training/population (VectorizedPopulation)
Affected Version/Branch: main (fixed)

Affected Files:
- `src/townlet/population/vectorized.py:90`
- `tests/test_townlet/_fixtures/training.py`

Description:
- The `VectorizedPopulation` constructor now enforces `brain_config` as required (per WP-C2), raising a `ValueError` when `brain_config is None`.
- However, the signature and implementation still carry legacy parameters and code paths that are effectively dead or misleading:
  - Parameters like `learning_rate`, `gamma`, `use_double_dqn`, `network_type`, `replay_buffer_capacity` are documented as "defaults" but are ignored whenever `brain_config` is passed (which is now mandatory).
  - Several branches guarded by `if brain_config is not None / else` are unreachable in production (the `else` branch can only be hit if someone bypasses or patches the `brain_config is required` check), yet they still contain non-trivial logic (manual network construction, optimizer/loss setup, replay capacity).
- This creates a confusing API surface: callers must supply `brain_config`, but the constructor advertises parameters and behavior that no longer apply, and tests/fixtures still pass values for them.

Reproduction:
- Instantiate `VectorizedPopulation` through current fixtures/tests:
  - In all current usages, a `BrainConfig` object is passed as `brain_config=minimal_brain_config`.
  - Despite this, the call sites still pass `learning_rate`, `gamma`, `replay_buffer_capacity`, etc., which are ignored and overridden by `brain_config` internals.
- Inspect `src/townlet/population/vectorized.py`:
  - The constructor raises if `brain_config is None` (line ~100).
  - Immediately afterwards, Q-learning parameters, network, optimizer, and loss are all derived from `brain_config`, with legacy fallback code guarded by `else` blocks that can never be reached in normal usage.

Expected Behavior:
- The constructor API and implementation should match the enforced contract:
  - If `brain_config` is truly required (no legacy path), legacy parameters should either be removed from the signature or clearly documented as ignored/deprecated.
  - Legacy branches for "no brain_config" behavior should be removed or guarded behind explicit feature flags to avoid dead code.

Actual Behavior:
- Public signature suggests two ways to configure Q-learning (constructor args *or* brain_config), but only the `brain_config` path is actually valid; the rest is dead or misleading.
- This increases the risk that future maintainers will accidentally re-activate or depend on legacy branches, confusing any attempt to reason about training configuration.

Root Cause:
- TASK-005 / WP-C2 migrated the system toward Brain As Code (brain.yaml) and made `brain_config` mandatory, but did not fully clean up the older parameter-based configuration path from `VectorizedPopulation`.

Proposed Fix (Breaking OK):
- Simplify `VectorizedPopulation.__init__` to align with Brain As Code:
  - Remove or deprecate constructor-level Q-learning hyperparameters (`learning_rate`, `gamma`, `use_double_dqn`, etc.) from the public signature, or mark them as ignored when `brain_config` is provided.
  - Remove unreachable `else` branches that construct networks/optimizers/losses without brain_config.
  - Update docs/tests/fixtures to treat `brain_config` as the *only* source of Q-learning and architecture parameters.

Migration Impact:
- Internal call sites that still pass legacy parameters will need to be updated (mostly tests/fixtures); external users are already required to provide `brain_config`, so the behavioral surface doesn’t change.

Alternatives Considered:
- Re-allow `brain_config=None` and support both configuration modes:
  - Rejected; it reintroduces legacy paths and contradicts the “Brain As Code only” decision.

Tests:
- Ensure all tests construct `VectorizedPopulation` via brain_config-only parameters and do not rely on legacy defaults.

Owner: training/population
Links:
- `docs/tasks/TASK-005-BRAIN-AS-CODE.md`

Fix Summary:
Commit 02ebe3b (Nov 14, 2025) "refactor(wpc2): delete legacy dual initialization paths" removed:
- Constructor parameters: learning_rate, gamma, replay_buffer_capacity, network_type, target_update_frequency, use_double_dqn
- All dual path logic (if brain_config / else branches)
- Q-network, optimizer, loss, and replay buffer fallback paths
- Net deletion: 84 lines of dead code
- All 18 tests passing after cleanup
