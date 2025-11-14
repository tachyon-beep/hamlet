Title: VFS variable scope and shape validation missing for env writes into registry

Severity: medium
Status: open

Subsystem: vfs/registry + environment/vectorized_env
Affected Version/Branch: main

Affected Files:
- `src/townlet/vfs/registry.py`
- `src/townlet/environment/vectorized_env.py:680`
- `configs/*/variables_reference.yaml`

Description:
- The `VariableRegistry` enforces per-variable shapes and dtypes when calling `set(variable_id, value, writer)`, but the environment writes into the registry based on implicit assumptions about variable definitions rather than any compiler-validated contract, particularly for standard variables like `grid_encoding`, `position`, `affordance_at_position`, `time_sin`, `time_cos`, `interaction_progress`, and `lifetime_progress`.
- If a `variables_reference.yaml` omits or misconfigures one of these standard variables (wrong type, wrong dims, missing `readable_by`/`writable_by` roles), the environment can hit runtime errors (KeyError, PermissionError, shape mismatch) deep inside the training loop, rather than failing fast at compile time.
- This makes VFS integration feel “kind of wired” but fragile: variables exist in schema but the actual runtime contract between compiler, registry, and env is enforced only indirectly and late.

Reproduction:
- Modify a `variables_reference.yaml` to:
  - Remove or rename a standard variable like `grid_encoding` or `position`, or
  - Change the type/dims of `position` from `vecNf` with `dims=2` to some incompatible configuration.
- Compile and run a training config that expects those variables:
  - Universe compile succeeds (it only validates variable definitions in isolation).
  - At runtime, `VectorizedHamletEnv` attempts to write or read those variables:
    - `set("grid_encoding", ...)` fails if there is no such variable or if shapes/dtypes mismatch.
    - `get("position", reader="agent")` fails if `readable_by` or scope are misconfigured.

Expected Behavior:
- The VFS contract between compiler/DTOs and env should be validated up front, so that:
  - Missing or misconfigured standard variables are caught during compilation of the config pack (Stage 2/3 of UAC).
  - Operators see a clear diagnostic pointing at `variables_reference.yaml`, not a runtime exception from `VariableRegistry` in the middle of training.

Actual Behavior:
- Validation of standard VFS variables is spread across:
  - The universe compiler building `VariableRegistry` (but not checking the specific variable set expected by the environment).
  - Ad hoc runtime checks in `VectorizedHamletEnv` (e.g., probing `self.vfs_registry._definitions` as in BUG-17).
- There is no centralized “VFS environment contract” that asserts presence and correct type/scope for required variables; failure modes only appear at runtime when env code interacts with the registry.

Root Cause:
- VFS Phase 1 focused on generic variable definitions and observation spec generation, leaving the mapping between “standard HAMLET universe variables” and env expectations implicit.
- The environment assumes certain variable IDs and shapes (e.g., `position`, `grid_encoding`, `affordance_at_position`, temporal vars), but those assumptions are not enforced in the compiler or DTO layer.

Proposed Fix (Breaking OK):
- Introduce a compiler-time validation pass for VFS variables:
  - Define a small contract list of required standard variables (IDs, types, scopes, dims) for the current universe (e.g., for Grid2D vs Aspatial).
  - In UAC Stage 3/4, validate that these variables are present and compatible with the env’s expectations; raise a clear error if they are not.
- Optionally:
  - Add a helper in `VariableRegistry` (e.g., `assert_has(var_id, scope, type, dims)`) used during universe compilation rather than in env.

Migration Impact:
- Misconfigured `variables_reference.yaml` files will start failing at compile time instead of during training:
  - This is desirable for operators, but existing hand-edited configs may need to be corrected to match the expected standard variable set.

Alternatives Considered:
- Keep the current “best-effort” runtime checks and rely on documentation to discourage changing standard variables:
  - Rejected; per project guidance, critical configuration contracts should be validated early and explicitly.

Tests:
- Add unit/integration tests for the compiler:
  - Provide minimal variables_reference.yaml with missing/incorrect standard variables and assert that compilation fails with helpful messages.

Owner: VFS + universe/compiler + env
Links:
- `docs/arch-analysis-2025-11-13-1532/04-final-report.md` (VFS Phase 1/2 discussion)
- `docs/audit/VFS-PHASE-1-AIS-AUDIT.md`
