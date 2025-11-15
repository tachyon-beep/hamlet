# TASK-005B-VFS2: Variable & Feature System Phase 2 (Expressions, Contracts, Semantics)

**Status**: Planned
**Priority**: High
**Estimated Effort**: 60–80 hours (multi-phase, compiler + env refactor)
**Dependencies**: TASK-002C (VFS Phase 1), TASK-004A (Compiler), TASK-004C (DAC), BUG-36/37/38
**Enables**: TASK-008 (Model Abstraction and Export), future skill packs (L4+), richer DAC/VFS2 integration
**Created**: 2025-11-14
**Completed**: YYYY-MM-DD

**Keywords**: VFS, expressions, WriteSpec, VariableRegistry, observation_spec, curriculum_active, semantic_type, VFS2
**Subsystems**: `vfs/`, `environment/` (observation + actions), `universe/compiler`, `universe/adapters`, `agent/networks`
**Architecture Impact**: Major (extends VFS from “static schema + storage” to “expression-capable feature system”)
**Breaking Changes**: Yes (config validation becomes stricter; expression semantics become real instead of no-op)

---

## AI-Friendly Summary (Skim This First!)

**What**: Design and implement VFS Phase 2 (“VFS2”): real expression evaluation for variable updates and derived features, explicit contracts between compiler/env/VFS, and consistent semantic/masking metadata for structured encoders.

**Why**: Today VFS mostly defines *what* variables exist and *how* they are normalized, but not *how* they are derived or updated; `WriteSpec.expression` and action `writes` are validated yet never executed. That undermines the declarative story and leaves rich config surfaces as no-ops.

**Scope**:
- **Included**: Expression engine for `WriteSpec`, compiler/runtime contracts for standard variables, end-to-end threading of `semantic_type` and `curriculum_active`, adapter cleanup, and bug closure for BUG-36/BUG-37/BUG-38.
- **Excluded**: New frontend features, major DAC changes (beyond wiring to VFS2), multi-tenant runtime environments.

**Quick Assessment**:

- **Current Limitation**: VFS variables are essentially typed storage+normalization; derived variables and VFS-backed action effects cannot be expressed without code changes, and some exposed APIs (e.g., `writes`) are no-ops.
- **After Implementation**: Config authors can declare derived variables and variable updates as expressions; compiler validates the full VFS contract; env and DAC get a consistent, semantic-aware observation/masking layer.
- **Unblocks**: Skill packs / advanced curriculum levels using higher-level features (e.g., trends, urgency, spatial features), richer DAC strategies using VFS variables, safe experimentation with custom features.
- **Impact Radius**: VFS schema, compiler Stage 5 metadata, env observation construction, action execution, and structured network encoders.

**Decision Point**: If you are not touching variables/observations, DAC integrations, or structured encoders, you can skim Problem Statement and Acceptance Criteria and then stop.

---

## Problem Statement

### Current Constraint

1. **Expressions Are Schema-Only, Never Executed (BUG-36)**
   - `WriteSpec.expression` in `vfs/schema.py` and `ActionConfig.writes` in `environment/action_config.py` are defined and validated, but no runtime component parses or executes them.
   - `variables_reference.yaml` examples (L0_0_minimal et al.) sketch a rich expression language (derived features, temporal trends, spatial predicates) that cannot actually run.

   **Example**:

   ```yaml
   # variables_reference.yaml (today – illustrative only)
   variables:
     - id: "energy_deficit"
       type: "scalar"
       # Expression is just a string; not evaluated anywhere
       expression: "1.0 - bar['energy']"
   ```

   ```python
   # ActionConfig wires writes, but VectorizedHamletEnv never uses them
   class ActionConfig(BaseModel):
       writes: list[WriteSpec] = Field(default_factory=list)
   ```

2. **Standard Variable Contracts Are Implicit and Runtime-Only (BUG-37, BUG-17)**
   - Env assumes the existence, type, and scope of standard variables (`position`, `grid_encoding`, `affordance_at_position`, `time_sin`, `time_cos`, `interaction_progress`, `lifetime_progress`), but this is not enforced by the compiler; misconfigurations are caught only when `VariableRegistry.set/get` is called during training.

3. **Semantic & Curriculum Metadata Partially Used (BUG-38, BUG-28, BUG-27)**
   - `ObservationField.semantic_type` and `curriculum_active` are respected by `VFSAdapter.build_observation_activity()` and consumed by `StructuredQNetwork`, but `vfs_to_observation_spec` uses name heuristics and ignores explicit semantics, and Activity’s dimension flattening has known bugs (BUG-27).

### Why This Is Technical Debt, Not Design

**Test**: If we removed VFS Phase 2 surfaces (WriteSpec expressions, semantic/curriculum metadata) but left VFS1 (just normalization/storage), would the system be more expressive or more fragile?

**Answer**: More fragile.

- ✅ Enables: Declarative feature engineering (derived variables, temporal/spatial features).
- ✅ Enables: Clean integration between DAC and VFS (vfs_variable strategies with real semantics).
- ❌ Does NOT: Improve baseline L0–L3 behavior today (curriculum relies mostly on auto-generated variables).

**Conclusion**: These are partially-designed capabilities that we want; leaving them half‑wired (schema only, or “kind of used”) is technical debt.

### Impact of Current Constraint

**Cannot Create**:

- Derived VFS variables (`energy_deficit`, `energy_velocity`, `rush_hour`) that change over time without editing Python.
- VFS-backed action effects that update variables via expressions (`money += 10` when you work a Job).
- Stable, explicit observation semantics across compiler → env → networks (groups/masks currently rely on heuristics).

**Pedagogical Cost**:

- Limits teaching about feature engineering and representation learning; configs hint at capabilities (in comments and examples) that students cannot actually use.

**Research Cost**:

- Forces manual Python changes for new features; discourages exploration of complex reward/feature interactions that would otherwise be purely declarative.

---

## Solution Overview

### Design Principle

**Core Philosophy**: VFS should be the single declarative source of truth for variables and features—from declaration to update to observation—backed by a safe expression engine and explicit contracts with the environment and DAC.

**Key Insight**: We already have three pieces in place (schema, runtime storage, and observation spec builder); VFS2 mainly needs (a) an expression evaluator, and (b) formalized contracts, plus consistent metadata threading.

### Architecture Changes

1. **Expression Layer (VFS2 Core)**
   - Add an expression parser/evaluator for `WriteSpec.expression` and, optionally, variable-level expressions in `variables_reference.yaml`.
   - Integrate evaluation into env action execution and per-tick VFS updates (tick/episode lifetimes).

2. **Contracts & Validation (Compiler ↔ Env ↔ VFS)**
   - Define a compiler-time validation pass that checks standard variable presence and shape/type/scope against the env’s expectations.
   - Replace env’s direct access to registry internals with explicit `VariableRegistry` APIs and compiler-validated metadata.

3. **Observation Semantics & Activity**
   - Make `semantic_type` and `curriculum_active` the canonical source of grouping/masking semantics in all adapters and DTOs (no name heuristics where explicit metadata is present).
   - Fix dimension accounting so `ObservationActivity.total_dims` matches `ObservationSpec.total_dims` in all cases.

4. **Migration & Versioning**
   - Introduce a minimal VFS schema version flag (e.g., `variables_reference.version`) and/or expression version, so future expression changes are detectable.
   - Keep Phase 1 behavior intact for configs that do not use expressions or advanced metadata.

### Compatibility Strategy

**Backward Compatibility**:

- Existing L0–L3 curriculum packs that do not specify `writes` or custom expressions should behave identically.
- For configs that *do* specify expressions today, VFS2 will either:
  - Start rejecting them with clear errors (pre-VFS2 guardrail), or
  - Start honoring them once expression evaluation is implemented.

**Migration Path**:

- For early adopters of expression syntax, provide a small migration guide and, if necessary, a script to verify variable contracts and expression validity.

**Versioning**:

- Use schema-level versioning for VFS2 (and possibly an `expression_version` field) to avoid silently misinterpreting expressions in future iterations.

---

## Detailed Design

### Phase 1: Solidify VFS Contracts (8–12 hours)

**Objective**: Make the implicit compiler/env/VFS contract explicit and validated up front.

**Changes**:

- File: `src/townlet/universe/compiler.py`
  - Add a validation step that asserts presence and compatibility of standard variables (`position`, `grid_encoding`, `affordance_at_position`, `time_sin`, `time_cos`, `interaction_progress`, `lifetime_progress`, etc.) based on substrate type and temporal settings.
  - Surface clear errors that point back to `variables_reference.yaml` when a required variable is missing or misconfigured.

- File: `src/townlet/vfs/registry.py`
  - Add public helpers (e.g., `has(var_id) -> bool`) to replace direct `_definitions` access (tie-in with BUG-17).
  - Optionally add an assertion helper used only at compile time (`assert_has(var_id, type, scope, dims)`).

- File: `src/townlet/environment/vectorized_env.py`
  - Replace registry `_definitions` probes with the new public API.
  - Make writes to standard variables go through a contract verified by the compiler (fewer ad-hoc runtime checks).

**Tests**:

- Negative tests for misconfigured `variables_reference.yaml` that should now fail at compile time with actionable messages.
- Regression tests that confirm existing configs (L0–L3) still compile and run unchanged.

### Phase 2: Expression Engine for WriteSpec (VFS2 Core) (30–40 hours)

**Objective**: Turn `WriteSpec.expression` from a passive string into a safe, composable expression that updates VFS variables during action execution.

**Changes**:

- File: `src/townlet/vfs/schema.py`
  - Keep `WriteSpec` shape but document supported subset of expression syntax for Phase 2.
  - Optionally add a lightweight “expression kind” or metadata field if we need different execution paths.

- File: `src/townlet/vfs` (new module, e.g., `expression.py`)
  - Implement a small, safe expression language (likely AST-based, not `eval`) that can handle a subset of operations seen in `variables_reference.yaml` comments:
    - Arithmetic, min/max, simple conditionals, references to bars/variables, and maybe some temporal/spatial helpers.
  - Provide an API like:
    ```python
    evaluate_expression(expr: str, context: VFSContext) -> torch.Tensor
    ```

- File: `src/townlet/environment/action_config.py` + action execution path
  - Integrate `writes: list[WriteSpec]` into the env’s action handling: when an action fires, evaluate each `WriteSpec` and update the corresponding variable via `VariableRegistry.set`.
  - Ensure lifetime semantics (`tick` vs `episode`) are respected.

**Migration**:

- For configs already using `writes`, decide whether to:
  - Gate execution behind a config flag or schema version, or
  - Immediately interpret existing expressions under the new engine (likely fine for internal users only).

**Success Criteria**:

- At least one end-to-end example where a `drive_as_code.yaml` or action config uses a VFS-derived variable (e.g., `energy_deficit`) that is computed via `WriteSpec` and demonstrably affects behavior/rewards.

### Phase 3: Semantic & Curriculum Metadata Harmonization (12–16 hours)

**Objective**: Ensure `semantic_type` and `curriculum_active` are used consistently across VFS, compiler, env, and networks.

**Changes**:

- File: `src/townlet/universe/adapters/vfs_adapter.py`
  - Update `vfs_to_observation_spec` to prefer `field.semantic_type` (where present) over name heuristics (close BUG-28/BUG-38).
  - Fix dimension flattening/active mask calculation in `VFSAdapter.build_observation_activity` so it matches the observation spec (close BUG-27).

- File: `src/townlet/agent/networks.py`
  - Confirm that `StructuredQNetwork`’s use of `ObservationActivity.group_slices` aligns with the updated semantics; adjust or document group names if needed.

**Migration**:

- No changes required to existing configs that don’t use explicit `semantic_type`/`curriculum_active`.
- For configs that do, structured encoders and RND will now respect those semantics more faithfully.

**Success Criteria**:

- `ObservationActivity.total_dims == ObservationSpec.total_dims` in tests.
- Semantic grouping and active masks reflect explicit metadata in all relevant tests.

---

## Testing Strategy

### Test Coverage Requirements

**Unit Tests**:

- VFS expression parser/evaluator: high coverage (happy path + malformed expressions).
- Variable contracts: tests for standard variable presence/shape, both positive and negative.
- Adapter & activity: tests for semantic grouping, mask correctness, and dimension accounting.

**Integration Tests**:

- End-to-end run of a small config that uses:
  - A derived variable from VFS2 expression.
  - Action `writes` that update variables.
  - StructuredQNetwork with `ObservationActivity` masks.

**Regression Testing**:

- Ensure existing L0–L3 training runs still work and checkpoints resume correctly.
- Verify no significant performance regression from expression evaluation (or document and measure overhead).

---

## Migration Guide (Sketch)

### For Existing Configs

- **Configs without expressions/writes**: No change required; VFS2 features remain opt-in.
- **Configs using expressions in `variables_reference.yaml` or `writes`**:
  - Before VFS2: expressions are comments / no-ops; configs should be updated to match the new supported expression subset and validated against tests.
  - After VFS2: these expressions become live; behavior changes are expected and should be validated experimentally.

### For Existing Checkpoints

- VFS2 should not change the layout of observations or recorded variables for Phase 1 configs; checkpoints remain compatible as long as variable contracts are met.
- Any new variables introduced by VFS2 expressions are not expected to be baked into existing checkpoints (pre‑VFS2), but this is acceptable given pre-release status and zero backwards compatibility guarantees.

---

## Examples (High-Level Sketch)

### Example 1: Derived Variable for DAC

**Config Snippet (`variables_reference.yaml`)**:

```yaml
variables:
  - id: "energy_deficit"
    scope: "agent"
    type: "scalar"
    lifetime: "tick"
    readable_by: ["agent", "engine"]
    writable_by: ["engine"]
    default: 0.0
```

**Config Snippet (`drive_as_code.yaml`)**:

```yaml
extrinsic:
  type: "vfs_variable"
  variable: "energy_deficit"
```

**Runtime**: VFS2 expression engine updates `energy_deficit` each tick; DAC reads it via `VariableRegistry` to compute rewards.

### Example 2: Action-Driven Variable Update

**Config Snippet (`global_actions.yaml`)**:

```yaml
custom_actions:
  - name: "REST"
    type: "passive"
    costs: {}
    effects: {}
    reads: ["energy"]
    writes:
      - variable_id: "energy"
        expression: "min(1.0, energy + 0.1)"
```

**Runtime**: When REST is executed, VFS2 evaluates the expression and updates `energy` accordingly via `VariableRegistry.set`.

---

This task should be treated as the umbrella for VFS2 work: closing BUG-36/BUG-37/BUG-38 (and related VFS tickets) and delivering a coherent, documented, and test-backed VFS expression and contract story. Individual PRs can target the phases incrementally while referencing this task for architectural intent.
