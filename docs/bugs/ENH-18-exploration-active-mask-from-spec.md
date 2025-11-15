Title: Derive RND active_mask from observation spec/group activity

Severity: low
Status: open

Subsystem: exploration/RND + universe
Affected Version/Branch: main

Description:
- RND accepts an optional `active_mask` to zero out padded dims but callers must compute it manually.

Proposed Enhancement:
- Provide a helper to build `active_mask` from `ObservationSpec`/`ObservationActivity`, setting 0s for groups not exposed to agents or padding.

Migration Impact:
- Backwards compatible; makes usage simpler and robust to layout changes.

Tests:
- Build mask from a compiled universe and ensure the masked observation produces the same embedding dims.

Owner: exploration
