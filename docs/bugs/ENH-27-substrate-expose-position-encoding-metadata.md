Title: Expose substrate position encoding metadata for downstream consumers

Severity: low
Status: open

Subsystem: substrate + environment/demo
Affected Version/Branch: main

Affected Files:
- `src/townlet/substrate/base.py`
- `src/townlet/substrate/{grid2d,grid3d,gridnd,continuous,continuousnd,aspatial}.py`
- `src/townlet/demo/live_inference.py`

Description:
- Substrates encode positions into observations using several modes:
  - `relative`: normalized coordinates.
  - `scaled`: normalized coordinates plus grid/bounds sizes.
  - `absolute`: raw coordinates.
- Each concrete substrate implements its own encoding helpers and `get_observation_dim()`, but there is no unified, introspectable metadata that describes:
  - How many position features are present (beyond `get_observation_dim()` which conflates grid+position).
  - Which slice of the observation corresponds to position vs grid vs other features.
  - What semantic meaning each coordinate axis has (e.g., Grid3D’s `coordinate_semantics`).
- Downstream consumers like:
  - Live inference/visualization,
  - Analysis tools, and
  - Curriculum/debug code
  often need to know “where the position lives” in the observation vector, but today they must re-derive this from substrate type and encoding mode, which is brittle and prone to duplication.

Proposed Enhancement:
- Add a small introspection API on substrates, for example:
  - `get_position_encoding_metadata() -> dict` with fields like:
    - `mode`: `"relative" | "scaled" | "absolute"`.
    - `position_dims`: int (number of position features).
    - `extra_dims`: int (e.g., range/bounds metadata).
    - Optional axis labels for Grid2D/3D (`["x", "y"]`, `["x", "y", "z"]`).
- Optionally, thread this metadata into:
  - `CompiledUniverse` or runtime metadata, so it’s available at demo/visualization layers.
  - The VFS observation spec, enabling structured slicing for visualization or structured networks.

Migration Impact:
- Purely additive; existing behavior is unchanged.
- Tools that currently hardcode assumptions about position encoding (e.g., some demo visualizations) can migrate to the metadata-based API over time to become substrate-agnostic.

Tests:
- Add substrate unit tests that:
  - Assert `get_position_encoding_metadata()` is consistent with `get_observation_dim()` and the actual encoding functions for Grid2D, Grid3D, Continuous, etc.
  - Verify that aspatial substrates report zero position dims and a suitable mode.

Owner: substrate+demo
Links:
- `src/townlet/substrate/grid2d.py:get_observation_dim`
- `src/townlet/substrate/grid3d.py:get_observation_dim`
- `src/townlet/substrate/continuous.py:get_observation_dim`
- `src/townlet/demo/live_inference.py` (visualization paths)
