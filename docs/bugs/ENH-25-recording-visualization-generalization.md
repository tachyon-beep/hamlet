Title: Generalize recording visualization for custom meters and substrates

Severity: low
Status: open

Subsystem: recording/video-export + live-inference
Affected Version/Branch: main

Affected Files:
- `src/townlet/recording/video_renderer.py:136`
- `src/townlet/demo/live_inference.py:880`
- `docs/WORK-PACKAGES.md:324`

Description:
- The current video export and live replay stack is tuned for the canonical 8×8 Grid2D, 8-meter HAMLET curriculum levels, and assumes 2D spatial substrates.
- `EpisodeVideoRenderer._render_meters()` hardcodes the standard eight meter names and one-to-one alignment with the `meters` array, which breaks down for custom meter configurations (e.g., fewer meters, reordered meters, or additional resource bars).
- `_render_grid()` assumes that `step_data["position"]` is a 2D `(x, y)` tuple and uses a fixed `grid_size`, which is derived from affordance positions or passed explicitly; this does not generalize cleanly to aspatial substrates (position_dim=0) or higher-dimensional/continuous substrates.
- `docs/WORK-PACKAGES.md` already flags this as WP-L1 (“Generalize Recording/Visualization for Custom Configs”), but there is no dedicated ENH ticket capturing the desired behavior and test plan.

Proposed Enhancement:
- Make recording visualization substrate- and meter-aware instead of hard-coded:
  - Derive meter names and ordering from compiled universe metadata or from the recorded episode metadata, instead of assuming the fixed list `["energy", "hygiene", "satiation", "money", "health", "fitness", "mood", "social"]`.
  - Use substrate metadata (already exposed in `LiveInferenceServer._build_substrate_metadata()`) to decide whether to render a grid, continuous space plot, or an aspatial meters-only dashboard.
  - For aspatial substrates (position_dim=0), skip `_render_grid()` entirely and focus on meters, temporal info, and Q-values.
  - For small grids (e.g., L0_0_minimal 3×3), infer width/height from substrate metadata rather than assuming an 8×8 grid.

Migration Impact:
- Existing L0–L3 configs with the standard 8 meters and Grid2D substrate should continue to render identically, aside from potential cosmetic tweaks.
- New configs with custom meter sets or aspatial substrates become first-class citizens for recording and video export, instead of relying on implicit assumptions that may cause crashes or misleading visuals.

Alternatives Considered:
- Keep the current “L1-only” visualization semantics and document that custom configs are unsupported:
  - Rejected; the entire system is designed to encourage experimentation with substrates and meter sets, and recording/visualization should keep up.

Tests:
- Add integration-style tests under `tests/test_townlet/integration`:
  - Export a video for `configs/L0_0_minimal` and assert that frames render without error and grid dimensions match the small substrate.
  - Export a video for an aspatial config (once available) and assert that the renderer produces a meters-only visualization without indexing into position data.
- Extend unit tests for `EpisodeVideoRenderer`:
  - Inject synthetic metadata with non-standard meter sets and ensure the renderer uses the correct labels and bar count.

Owner: recording/video-export
Links:
- `docs/WORK-PACKAGES.md:324` (WP-L1)
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md`
