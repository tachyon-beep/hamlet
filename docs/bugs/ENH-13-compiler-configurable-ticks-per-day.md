Title: Make ticks_per_day configurable and propagate to action_mask_table

Severity: low
Status: open

Subsystem: universe/compiler + metadata
Affected Version/Branch: main

Description:
- `UniverseMetadata.ticks_per_day` is fixed at 24, and optimization `action_mask_table` is hard-coded to shape (24, affordance_count).
- Some domains may want alternative day lengths (e.g., 12, 48) for curriculum experiments.

Proposed Enhancement:
- Introduce `ticks_per_day` in environment or training config and carry it through metadata.
- Build `action_mask_table` with that number of rows; runtime env already derives `hours_per_day` from the table shape.

Migration Impact:
- Default remains 24; no breaking change unless configs opt in.

Tests:
- Verify env uses the configured number of ticks and that affordance open/close tables align.

Owner: compiler
