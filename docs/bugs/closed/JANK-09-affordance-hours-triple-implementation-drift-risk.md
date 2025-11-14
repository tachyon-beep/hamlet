Title: Affordance operating hours implemented three times with ACTUAL INCONSISTENCIES

Severity: HIGH (upgraded from medium)
Status: FIXED
Confirmed Date: 2025-11-14
Fixed Date: 2025-11-14
Resolution: Consolidated all three implementations into canonical temporal_utils.is_affordance_open()

Bugs Found (pre-fix):
  - affordance_config.py: Failed on "18-28" notation (returned False at 2am, should be True)
  - affordance_engine.py: Failed on "22-6" notation (returned False at 23:00, should be True)
  - compiler.py: Correct implementation (handled both notations)

Fix Applied:
  - Created src/townlet/environment/temporal_utils.py with canonical implementation
  - Deleted affordance_config.is_affordance_open() - now import from temporal_utils
  - Updated AffordanceEngine.is_affordance_open() to delegate to canonical
  - Updated UniverseCompiler to use canonical directly (no wrapper)
  - Added comprehensive test suite (9 tests, 100% coverage)
  - All existing tests pass, configs compile successfully

Ticket Type: JANK
Subsystem: environment/affordances + universe/compiler
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/affordance_config.py:205`
- `src/townlet/environment/affordance_engine.py:78`
- `src/townlet/environment/vectorized_env.py:520`
- `src/townlet/universe/compiler.py` (Stage 6 action_mask_table build)

Description:
- Affordance “open hours” semantics are currently implemented in three separate places:
  - `affordance_config.is_affordance_open(time_of_day, operating_hours)` – a standalone helper used in config/docs.
  - `AffordanceEngine.is_affordance_open(affordance_name, time_of_day)` – runtime check based on `operating_hours` from YAML.
  - `VectorizedHamletEnv._is_affordance_open(affordance_name, hour=None)` – runtime check that uses the compiler‑generated `action_mask_table` optimization tensor instead of `operating_hours`.
- The compiler also uses `operating_hours` when building `optimization_data.action_mask_table`, effectively embedding a fourth copy of the logic in Stage 6.
- While all implementations are currently consistent (midnight wraparound, `[open, close)` vs wrap ranges), this duplication makes it easy for future changes to alter behavior in one place and forget the others, leading to subtle “open vs closed” discrepancies between config, engine, and env‑level masking.

Reproduction:
1) Conceptual (no single failing test yet, but a realistic drift scenario):
   - Suppose `operating_hours` semantics are tightened in `AffordanceConfig.validate_operating_hours()` (e.g., allow `[open, 24]` only, or extend to tick‑based minutes) without updating:
     - `affordance_config.is_affordance_open`, and
     - Stage 6 compiler logic that builds `action_mask_table`.
   - The compiler may still generate an action mask table based on old assumptions, while engine‑level `AffordanceEngine.is_affordance_open` uses new ones.
2) At runtime:
   - `VectorizedHamletEnv.get_action_masks()` uses `_is_affordance_open` (action_mask_table) to gate INTERACT availability.
   - `_handle_instant_interactions` and `_handle_interactions` rely on `self._is_affordance_open` only when temporal mechanics are enabled.
   - Any divergence between table semantics and engine/config helpers would cause INTERACT to be masked/unmasked at times that disagree with the YAML definition.

Expected Behavior:
- There should be a single canonical definition of “is this affordance open at hour t?” that:
  - Is used by the compiler when precomputing `action_mask_table`.
  - Is used by `AffordanceEngine` and `VectorizedHamletEnv` when making runtime decisions.
- If performance requires a precomputed mask table, its construction should still go through a shared helper rather than embedding its own logic.

Actual Behavior:
- `AffordanceEngine.is_affordance_open` and `affordance_config.is_affordance_open` re‑implement the same wraparound logic based on `operating_hours`.
- `VectorizedHamletEnv._is_affordance_open` ignores `operating_hours` entirely and instead:
  - Uses `self.action_mask_table[hour_idx, idx]` where `idx` is a precomputed mask index.
  - Assumes `hours_per_day` based on `action_mask_table.shape[0]`, defaulting to 24 when the tensor is empty.
- Any future tweaks to “hours” semantics (e.g., non‑24‑hour days, finer resolution, different wrap conventions) must be made in at least three places to stay consistent.

Root Cause:
- Historical layering:
  - `affordance_config.is_affordance_open` predates the dedicated `AffordanceEngine` and was used directly in docs/examples.
  - `AffordanceEngine` added an object‑oriented helper for runtime checks.
  - The compiler later introduced `action_mask_table` as an optimization and `VectorizedHamletEnv._is_affordance_open` as a fast lookup against that table.
- No shared “hours contract” type or helper is used across these layers, so the logic drifted into multiple, similar but independent implementations.

Risk:
- Subtle, configuration‑dependent bugs where:
  - INTERACT is masked off even though the YAML hours say the affordance is open, or
  - INTERACT remains available in certain hours despite the config intending it to be closed.
- Operators and students may attribute weird training dynamics to exploration rather than a quiet disagreement between config and runtime.
- Refactors to temporal mechanics (e.g., minutes‑level granularity, variable day length) will be harder and more error‑prone.

Proposed Directions:
- Short‑term:
  - Add comments in `VectorizedHamletEnv._is_affordance_open` and `AffordanceEngine.is_affordance_open` explicitly pointing at a single source of truth helper (e.g., `affordance_config.is_affordance_open`) and cross‑reference tests.
  - Add a sanity test that compares:
    - `affordance_config.is_affordance_open(t, operating_hours)`
    - `AffordanceEngine.is_affordance_open(name, t)`
    - `VectorizedHamletEnv._is_affordance_open(name, t)` (via a small helper that maps names to mask indices)
    for a range of `t` and sample affordances.
- Medium‑term:
  - Extract a single helper/function that defines the contract (e.g., `compute_open_mask(operating_hours, hours_per_day)`), use it both in Stage 6 when building `action_mask_table` and in `AffordanceEngine`.
  - Make `VectorizedHamletEnv._is_affordance_open` a thin wrapper that consults the table but can fall back to the helper when optimization data is missing or mis‑shaped.
- Long‑term:
  - If temporal mechanics evolve beyond 24‑hour integer ticks, centralize temporal semantics in a dedicated `TimeOfDay` / `Schedule` abstraction to avoid a fourth re‑implementation.

Tests:
- Add regression tests that ensure:
  - For every affordance in a test pack and every hour `t ∈ [0, 23]`, all three implementations agree on “open vs closed”.
  - Any future change to the hours semantics that forgets to update one site will cause a test failure.

Owner: environment
Links:
- `src/townlet/environment/affordance_config.py:is_affordance_open`
- `src/townlet/environment/affordance_engine.py:is_affordance_open`
- `src/townlet/environment/vectorized_env.py:_is_affordance_open`
- `src/townlet/universe/compiler.py` (Stage 6 action_mask_table construction)
