Title: Compiler hardcodes `configs/global_actions.yaml` path for loads and cache hashing

Severity: high
Status: open

Subsystem: universe/compiler
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler_inputs.py:144` (load_global_actions_config)
- `src/townlet/universe/compiler_inputs.py:203` (source_map.track_actions)
- `src/townlet/universe/compiler.py:2290` (hash)
- `src/townlet/universe/compiler.py:2334` (mtime)

Description:
- The compiler loads and hashes `configs/global_actions.yaml` via a relative path from CWD, not from the pack.
- This makes compiled artifacts depend on the caller’s working directory and external repo layout, hurting reproducibility and portability.

Reproduction:
1) Compile a pack from different working directories (or in an environment lacking `configs/global_actions.yaml`).
2) Observe differing behavior, load failures, or cache misses/hits unrelated to the pack contents.

Expected Behavior:
- Global actions should be resolved deterministically (e.g., path recorded within the config pack or an explicit absolute path), and cache fingerprints should use the same source actually loaded.

Actual Behavior:
- Relative path usage can point to different files or none at all depending on CWD.

Root Cause:
- Hardcoded use of `Path("configs")/"global_actions.yaml"` instead of the actual loaded file path or a pack-scoped resource.

Proposed Fix (Breaking OK):
- Make the global actions path a required part of the pack (e.g., `global_actions.yaml` within the pack) or accept an explicit path arg.
- For cache fingerprinting, hash the exact file that was loaded (recorded path), not a hard-coded location.

Migration Impact:
- Packs must include `global_actions.yaml` or provide a path. CI/scripts updated accordingly.

Alternatives Considered:
- Keep global actions external but record an absolute resolved path and use that consistently (still brittle in shared environments).

Tests:
- Compile from varying CWDs with the same pack → identical outputs/hashes.

Owner: compiler
