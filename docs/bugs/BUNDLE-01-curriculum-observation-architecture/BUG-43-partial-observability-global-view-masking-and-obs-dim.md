Title: Partial observability swaps grid/local views and changes obs_dim instead of masking global slice

Severity: medium
Status: closed
Date Closed: 2025-11-15

Subsystem: universe/compiler + environment/vectorized + agent/architecture
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py` (observation spec construction)
- `src/townlet/universe/adapters/vfs_adapter.py:90` (ObservationActivity + active_mask)
- `src/townlet/environment/vectorized_env.py:680` (full vs partial obs grid/local encoding)
- `src/townlet/agent/networks.py:418` (StructuredQNetwork group_slices)

Description:
- The `partial_observability` flag is intended to control whether agents get a **global view** (full grid) or a **local window** (POMDP), while reusing the same observation layout so policies can transfer across curriculum levels (train with global view → graduate to POMDP).
- Today, the compiler and env implement this by **swapping out fields and changing obs_dim**, rather than masking a fixed “global view” slice:
  - L1_full_observability:
    - `obs_dim = 96`
    - Fields include `obs_grid_encoding` (64 dims) + meters + temporal, etc.
  - L2_partial_observability:
    - `obs_dim = 57`
    - Fields include `obs_local_window` (25 dims) instead of `obs_grid_encoding`; the full grid field is gone.
- This means:
  - You **cannot** train a single Q-network on L1 and then reuse it directly for L2, because `observation_dim` changes and the first slice of the vector has a different meaning.
  - The back-end `ObservationActivity.active_mask` and `population.mask_unused_obs` machinery (designed to mask unused dims without changing obs_dim) is not being used to bridge full→partial observability.

Reproduction:
1) Compile two packs:
   - `configs/L1_full_observability` (full obs)
   - `configs/L2_partial_observability` (POMDP)
2) Inspect the compiled universes:
   - L1:
     - `obs_dim: 96`
     - Fields include `obs_grid_encoding` (64 dims, desc='grid_encoding').
   - L2:
     - `obs_dim: 57`
     - Fields include `obs_local_window` (25 dims, desc='local_window') instead of `grid_encoding`.
3) Attempt to load a checkpoint trained on L1 into an environment compiled for L2 with the same brain config:
   - Q-network input size (96) no longer matches `metadata.observation_dim` (57), causing shape mismatches or requiring a separate network.

Expected Behavior:
- Partial observability should **change information content, not the observation layout**:
  - There should be a fixed superset observation spec across the curriculum (same `obs_dim`), with:
    - A dedicated “global view” slice (e.g., `grid_encoding` or typed world state) living at known indices.
    - POMDP/local views either in a separate slice or sharing part of the core features.
  - When `partial_observability: true`, the **global slice is masked/zeroed** via `ObservationActivity.active_mask` (and/or env logic), but remains present in the vector so the network shape is unchanged.
- This would allow:
  - Training a single network on full‑obs (global slice active).
  - Gradually annealing away the global slice (masking) while keeping obs_dim and architecture constant.

Actual Behavior:
- The compiler builds **different** observation specs for full vs partial observability:
  - Full obs uses `grid_encoding` and omits `local_window`.
  - Partial obs uses `local_window` and never includes `grid_encoding`.
- As a result:
  - `metadata.observation_dim` changes between L1 and L2.
  - Observation slices shift (first 25/64 dims mean different things), breaking weight reuse.
  - `ObservationActivity.active_mask` only masks “curriculum‑inactive” fields within a given spec; it does not enforce a cross‑level superset layout.

Root Cause:
- The observation spec is built directly from the current level’s notion of “what fields exist” (grid vs local) instead of from a **global contract** that spans the curriculum.
- `ObservationActivity` and `mask_unused_obs` were added for structured masking (e.g., hiding unused meters), but the full vs partial observability design pre‑dates that and still swaps fields out entirely instead of keeping them as masked slices.

Proposed Fix (Breaking OK):
- Define a **curriculum‑wide observation contract** that is a superset of both full and partial modes:
  - Always include a fixed slice reserved for “global spatial view” (e.g., grid_encoding or typed global map).
  - Always include slices for meters, temporal features, and any local/POMDP view.
- Update the compiler + VFS adapter to:
  - Always include the global view field(s) in the spec (same dims) regardless of `partial_observability`.
  - Set `field.curriculum_active` for those fields based on mode/level:
    - Full obs: global slice active (mask=True); local POMDP slice optional/inactive.
    - Partial obs: global slice inactive (mask=False); local window active.
  - Use `ObservationActivity.active_mask` to build an `active_mask` that zeroes out inactive dims while keeping `obs_dim` constant.
- In the env:
  - When global view is inactive, either:
    - Don’t populate the global slice (leave zeros), or
    - Explicitly zero that slice before returning observations.
  - Make sure this is consistent with `mask_unused_obs` semantics so structured/RND networks see the same masking.

Migration Impact:
- Any existing pack that expects different `observation_dim` between levels will see a stabilized obs_dim after the change:
  - Q-networks can then be shared across levels without reshaping.
  - Old checkpoints may not be compatible with the new layout and should be retrained (pre‑release this is acceptable).
- Configs that rely on the exact position of `grid_encoding` / `local_window` slices in the vector should be updated to refer to them via `ObservationActivity.group_slices` rather than hardcoded indices.

Alternatives Considered:
- Keep changing obs_dim and require separate networks per level:
  - Rejected for curriculum learning; contradicts the stated goal of “train with global view, then graduate to POMDP” using the same agent.
- Only stabilize obs_dim but not use `active_mask`:
  - Would leave unused dims “live” and encourage the network to rely on them even in POMDP; masking is needed to make the information regime clear.

Tests:
- Add unit/integration tests that:
  - Compile L1 and L2 after the change and assert:
    - `metadata.observation_dim` is identical across levels.
    - `ObservationActivity.total_dims` is constant, while `active_dim_count` changes according to which view is active.
  - Verify that:
    - In full‑obs, global slice is non‑zero and marked active.
    - In partial‑obs, global slice is zero (or constant) and `active_mask` is False on that slice, while local window dims are active.
  - Add a regression test on `StructuredQNetwork` / RND paths to ensure `active_mask` is honored and doesn’t break forward passes when switching levels.

Owner: compiler+env+agent
Links:
- `docs/plans/2025-11-11-quick-05-structured-obs-masking.md`
- `src/townlet/universe/compiler.py` (observation spec building)
- `src/townlet/universe/adapters/vfs_adapter.py:build_observation_activity`
- `src/townlet/environment/vectorized_env.py:_get_observations`
- `src/townlet/agent/networks.py:StructuredQNetwork`

Resolution:
Fixed via curriculum masking approach. Changes made:

1. **Variable Creation** (`compiler.py:364-418`):
   - Now ALWAYS creates both `grid_encoding` and `local_window` variables
   - Uses fixed vision_range for local_window in full obs mode (vision_range=2 for 2D → 5×5 window)
   - Ensures consistent local_window dimensions across all curriculum levels

2. **Exposure Marking** (`compiler.py:2303-2332`):
   - Removed field filtering logic that deleted grid_encoding OR local_window
   - Instead marks exposures with `curriculum_active` based on `partial_observability`:
     - Full obs: grid_encoding active, local_window inactive
     - Partial obs: local_window active, grid_encoding inactive

3. **VFS Builder** (`vfs/observation_builder.py:83-101`):
   - Updated to pass `curriculum_active` from exposure config to ObservationField

4. **Activity Building** (`compiler.py:2129-2139`):
   - Use original `vfs_observation_fields` for building ObservationActivity
   - Preserves `curriculum_active` metadata (was being lost in conversion)

**Results**:
- L1 and L2 now have **identical obs_dim** (121 dims)
- L1 active_dim_count = 96 (grid_encoding active)
- L2 active_dim_count = 57 (local_window active)
- Difference = 39 dims (64 - 25, as expected)
- Both configs have both spatial fields in observation spec
- `active_mask` correctly masks inactive field

**Tests**: `tests/test_townlet/unit/universe/test_partial_obs_curriculum_masking.py` (5 passing tests)
