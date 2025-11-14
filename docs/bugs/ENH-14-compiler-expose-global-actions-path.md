Title: Record and expose the resolved global_actions path in compiled artifacts

Severity: low
Status: open

Subsystem: universe/compiler
Affected Version/Branch: main

Description:
- For debugging/provenance, it’s useful to know exactly which `global_actions.yaml` file was loaded.

Proposed Enhancement:
- Include the resolved path in `UniverseMetadata` (e.g., `global_actions_path`), and embed it in the cache fingerprint provenance.
- Expose via `RuntimeUniverse` read-only views.

Migration Impact:
- None; metadata extension only.

Tests:
- Check that `compiled.metadata.global_actions_path` points to the intended file.

Owner: compiler
