Title: VFS global vector variables in observations have ambiguous semantics (should be forbidden)

Severity: medium
Status: CONFIRMED (reclassified from BUG-41 to JANK-11)
Confirmed Date: 2025-11-14
Recommendation: Implement Policy A (forbid global vectors in observations with compile-time error)

Subsystem: environment/vectorized_env + VFS
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/vectorized_env.py:700`
- `src/townlet/vfs/registry.py:140`
- `src/townlet/vfs/schema.py:89`

Description:
- `VectorizedHamletEnv._get_observations()` assumes that all **global** VFS variables used in the observation spec are scalars.
- For global variables, `VariableRegistry` initializes storage with shape `[]` for scalars and `[dims]` for vectors (`vecNi`/`vecNf` with `scope="global"`).
- `_get_observations()` only special-cases scalar globals (`value.ndim == 0` → broadcast to `[num_agents]`) and treats any 1‑D tensor as already having a leading agent dimension and simply `unsqueeze(1)`.
- If a config pack defines a global **vector** variable (e.g., `type: vecNf`, `scope: global`, `dims: 2`) and exposes it via an `ObservationField`, `_get_observations()` will see a value of shape `[dims]` and produce a tensor of shape `[dims, 1]`, which is incompatible with other per‑agent fields of shape `[num_agents, *]`.
- When `torch.cat(observations, dim=1)` runs, the first dimension of the global vector field (`dims`) will not match `num_agents`, causing a runtime error or, worse, silent misalignment if `dims == num_agents` happens accidentally.

Reproduction:
1) In `variables_reference.yaml`, define a global vector variable used in observations, for example:
   - `id: wind_vector`, `scope: global`, `type: vecNf`, `dims: 2`, `default: [0.0, 1.0]`.
   - Add an `ObservationField` that uses `source_variable: wind_vector` and exposes it to agents.
2) Compile the universe and instantiate `VectorizedHamletEnv` with `num_agents != 2`.
3) Call `env.reset()` or `env._get_observations()`:
   - `VariableRegistry.get("wind_vector", reader="agent")` returns a tensor of shape `[2]`.
   - `_get_observations()` converts it to shape `[2, 1]` and appends it to `observations`.
   - `torch.cat(observations, dim=1)` fails with a size mismatch on dimension 0 (`num_agents` vs `2`).

Expected Behavior:
- Global vector variables either:
  - Are **forbidden** in observation fields (with a clear compile‑time error), or
  - Are treated explicitly as broadcasted features so that agents see the same vector (shape `[num_agents, dims]`), mirroring the scalar broadcasting behavior.
- In all cases, `_get_observations()` should guarantee that every observation component has leading dimension `num_agents`, independent of variable scope.

Actual Behavior:
- Global vectors are treated as if their first dimension were the agent axis, even though `VariableRegistry` defines them as `[dims]`.
- This leads to shape mismatches whenever `dims != num_agents`, and fragile, accidental correctness when `dims == num_agents` by coincidence.

Root Cause:
- `_get_observations()` only has a special case for scalar globals:
  - `if value.ndim == 0: value = value.expand(self.num_agents).clone()`
- Any 1‑D tensor (including global vectors) is assumed to be per‑agent (`[num_agents]`) and simply unsqueezed:
  - `if value.ndim == 1: value = value.unsqueeze(1)`
- `VariableRegistry` intentionally distinguishes between global vectors (`[dims]`) and agent vectors (`[num_agents, dims]`), but the environment doesn’t reflect that in its observation construction logic.

Proposed Fix (Breaking OK):
- Decide on one of the following policies and enforce it explicitly:
  - **Policy A (simplest)**: Forbid global vector variables in observation fields:
    - Add a compile‑time validation pass that rejects `ObservationField` entries whose `source_variable` is a `vec*` with `scope="global"`.
    - Update docs/config‑schemas for VFS to explain that global vectors are internal only (e.g., for logging) and not directly observable.
  - **Policy B (broadcast)**: Support global vectors in observations:
    - In `_get_observations()`, treat 1‑D global variables as `[dims]` and broadcast to `[num_agents, dims]`:
      - `if var_def.scope == "global" and value.ndim == 1: value = value.unsqueeze(0).expand(self.num_agents, -1).clone()`
    - Keep the scalar logic as‑is, so all global variables become per‑agent features with identical rows.
- In both policies, add a short comment/docstring in `_get_observations()` explaining the contract for global variables so future maintainers don’t reintroduce the ambiguity.

Migration Impact:
- If Policy A (forbid) is chosen:
  - Any existing config packs using global vector variables in observations will fail to compile and must be refactored to agent‑scoped variables (or removed from observation fields).
- If Policy B (broadcast) is chosen:
  - Behavior becomes well‑defined for global vectors; previously failing configs will start working with intended semantics.
  - Existing configs that only used scalar globals remain unaffected.

Alternatives Considered:
- Treat all global variables as scalars only and ban vector types for `scope="global"` entirely:
  - Simpler to reason about but unnecessarily restrictive for future DAC/VFS extensions that may want global vector metadata.

Tests:
- Add a unit test for `_get_observations()` that:
  - Creates a `VariableRegistry` with a global vector variable (`vecNf` or `vecNi`) and an observation spec referring to it.
  - Asserts that observations have shape `[num_agents, observation_dim]` and that the global vector is either rejected (Policy A) or correctly broadcast (Policy B).
- Add a negative test case for misconfigured `variables_reference.yaml` if Policy A is chosen (compiler‑time failure).

Owner: env+vfs
Links:
- `src/townlet/environment/vectorized_env.py:_get_observations`
- `src/townlet/vfs/registry.py:_initialize_storage`
- `src/townlet/vfs/schema.py:VariableDef` (scope/type semantics)
