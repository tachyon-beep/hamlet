# Aspatial Test Pack

This directory contains the minimal aspatial config pack used by unit tests
(`aspatial_test`). It exists to exercise substrate edge cases and therefore does
**not** satisfy the full TASK-004A compiler contract (missing spatial layout,
limited validation guarantees, etc.).

Because of those intentional omissions, CI skips this pack during
`python -m townlet.compiler validate` (see `scripts/validate_compiler_cli.py`).
Please keep it lightweight and dedicated to internal tests; production runs
should use one of the fully supported packs under `configs/L*`.
