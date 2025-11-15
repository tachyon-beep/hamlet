Title: GridND.get_all_positions wasteful for random sampling (generates all positions unnecessarily)

Severity: medium
Status: CONFIRMED
Confirmed Date: 2025-11-14
Guards Present: MemoryError at 10M positions, Warning at 100K
Issue: Wasteful design - generates all positions when only sampling a few for affordance placement

Ticket Type: JANK
Subsystem: substrate/gridnd + compiler
Affected Version/Branch: main

Affected Files:
- `src/townlet/substrate/gridnd.py:408`
- `src/townlet/universe/compiler.py` (Stage 6 affordance placement / feasibility)

Description:
- `GridNDSubstrate.get_all_positions()` enumerates all positions in an N-dimensional grid by:
  - Computing `total_positions = ∏ dimension_sizes`.
  - Raising `MemoryError` only when `total_positions > 10_000_000`.
  - Emitting a warning when `total_positions > 100_000`.
  - Then generating all positions via `itertools.product`, returning a Python list of lists.
- For moderate N and dimension sizes, this can produce very large in-memory lists (hundreds of thousands or millions of positions), which:
  - Are rarely needed in full (most use cases only want random sampling or feasibility checks).
  - Can cause significant slowdown and memory pressure during compilation or environment initialization.
- The compiler and environment currently rely on `get_all_positions()` for affordance randomization and feasibility checks, especially when randomizing affordance positions or validating capacity, but do not enforce a global cap based on use case.

Reproduction:
1) Configure a `gridnd` substrate with e.g. `dimension_sizes: [10, 10, 10, 10]` (4D, 10^4 = 10,000 positions):
   - `get_all_positions()` returns 10,000 positions; manageable but slow for repeated calls.
2) Increase dimensions or sizes:
   - `dimension_sizes: [10] * 6` → 10^6 = 1,000,000 positions; triggers a warning but still generates a million position vectors.
3) Use such a substrate with affordance randomization or any code path that calls `get_all_positions()` more than once; observe high memory usage and latency.

Expected Behavior:
- High-level systems (compiler and environment) should avoid constructing enormous Cartesian products when:
  - Only a subset of positions is actually needed (e.g., random sampling for affordance placement), or
  - Capacity feasibility can be checked analytically from dimension sizes without enumerating all positions.
- `get_all_positions()` should either:
  - Be clearly documented and used only in rare, bounded contexts (small grids), or
  - Enforce stricter limits (possibly configurable) for large N, with guidance to use alternative APIs.

Actual Behavior:
- `GridNDSubstrate.get_all_positions()` is available and appears harmless, but for high-dimensional or large grids it can:
  - Allocate very large intermediate Python structures.
  - Increase GC and memory pressure.
  - Add unexpected latency during compilation or initialization if called naively.

Root Cause:
- GridND was designed for research exploration of high-dimensional grids and exposes a generic enumeration API for completeness.
- The initial bounds (error at >10M positions, warning at >100K) were chosen heuristically without tight coupling to actual compiler/env usage patterns.

Proposed Directions:
- Short-term:
  - Call out in docs and code comments that `get_all_positions()` is intended only for small grids and diagnostic uses; large-N universes should avoid calling it.
  - Add a more conservative default cap (e.g., error at >1M positions) and make the threshold configurable via a module-level constant or environment variable if needed for specialized experiments.
- Medium-term:
  - Refactor compiler/env code paths that currently depend on `get_all_positions()` to use:
    - Analytical capacity checks (using `∏ dimension_sizes`) instead of full enumeration.
    - `initialize_positions()` for random sampling of affordance positions, as is already done for continuous substrates.
- Long-term:
  - Consider exposing a streaming iterator or generator API for GridND positions with back-pressure control, instead of building a full list in memory.

Tests:
- Add unit tests for `GridNDSubstrate.get_all_positions()` that:
  - Assert `MemoryError` for clearly huge grids (e.g., 11 dimensions of size 10).
  - Assert a warning for “large but allowed” grids near the threshold.
  - Verify that small grids still work and return the expected number of positions.

Owner: substrate+compiler
Links:
- `src/townlet/substrate/gridnd.py:get_all_positions`
- `src/townlet/environment/vectorized_env.py:randomize_affordance_positions`
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md` (Substrates concerns)
