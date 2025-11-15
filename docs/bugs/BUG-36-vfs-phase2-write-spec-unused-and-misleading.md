Title: VFS Phase 2 WriteSpec expressions are defined but never executed

Severity: medium
Status: open

Subsystem: vfs/schema + environment/action_config
Affected Version/Branch: main

Affected Files:
- `src/townlet/vfs/schema.py:92`
- `src/townlet/environment/action_config.py:77`
- `configs/*/variables_reference.yaml:1`
- `docs/arch-analysis-2025-11-13-1532/04-final-report.md:1069`

Description:
- The VFS schema exposes `WriteSpec` with a free-form `expression: str` intended for Phase 2 expression evaluation (derived variables and variable-based action effects), and `ActionConfig.writes: list[WriteSpec]` threads this into the action config layer.
- However, there is no implementation anywhere in the runtime (env, action executor, or VFS registry) that parses or executes `WriteSpec.expression`; the field is purely stored and never read at runtime.
- This means any YAML that configures action `writes` with expressions “kind of works” in the sense that it passes validation and can be inspected, but has **no effect** on the environment’s state, which is misleading for operators trying to use VFS Phase 2 features.

Reproduction:
- Define a custom action in `configs/global_actions.yaml` with a `writes` entry, e.g.:
  - `writes: [{variable_id: "energy_deficit", expression: "1.0 - bar['energy']"}]`
- Confirm that:
  - `ActionConfig` instances are created successfully (DTO validation passes).
  - The action executes without error in the environment.
  - The referenced VFS variable (`energy_deficit`) never changes in the `VariableRegistry` regardless of how often the action is taken.
- Inspect code:
  - `WriteSpec` is only used for schema validation in `vfs/schema.py` and as a field type in `ActionConfig.writes`.
  - No code path evaluates `expression` or applies it to the registry during env.step() or action execution.

Expected Behavior:
- Either:
  - VFS Phase 2 expression evaluation is implemented so that `WriteSpec.expression` is parsed into an AST and executed to mutate VFS variables at runtime, or
  - The `writes` / `WriteSpec` API is clearly marked as non-functional and rejected at config time (e.g., `writes` must be empty) until Phase 2 is implemented.

Actual Behavior:
- Configs with `writes` pass validation and appear “live” but do nothing:
  - No warnings or errors are emitted when `writes` is non-empty.
  - Operators can write complex expressions in YAML and reasonably expect them to affect variables, but the environment silently ignores them.

Root Cause:
- Phase 2 of VFS (expression evaluation for action effects and derived variables) was deferred after defining the DTOs, leaving a half-wired API surface that is reachable through YAML but not integrated into runtime execution.
- This is explicitly called out as technical debt in the architecture report (M3: VFS Phase 2 Features Unimplemented), but there is no bug ticket tracking the runtime no-op behavior.

Proposed Fix (Breaking OK):
- Short-term (pre‑Phase 2):
  - Add validation in `ActionConfig` or the universe compiler that rejects non-empty `writes` with a clear error: “VFS Phase 2 expressions are not yet supported; remove writes from this config.”
  - Alternatively, emit a loud warning at compile time when `writes` is non-empty to make the no-op explicit.
- Long-term (VFS2 / Phase 2):
  - Implement a safe expression engine that:
    - Parses `WriteSpec.expression` into an AST.
    - Evaluates expressions against VFS variables and environment state during action execution.
    - Updates `VariableRegistry` via a controlled API, with proper access control and shape/dtype validation.

Migration Impact:
- Any config currently using `writes` expecting behavior will either:
  - Start failing fast with a clear error (if we choose to reject), or
  - Continue to be accepted but emit explicit “no-op” warnings until real evaluation arrives.
- Current curriculum packs (L0–L3) do not depend on Phase 2 expressions, so the impact is mainly on experimental configs and documentation examples.

Alternatives Considered:
- Leave `writes` as a silent no-op and rely solely on documentation:
  - Rejected per project guidelines; exposed, non-functional features are considered bugs.

Tests:
- Extend `tests/test_townlet/unit/vfs/test_schema.py` and/or add a new test:
  - Assert that configs with non-empty `writes` either raise at compile/load time or cause a deliberate warning to be logged.

Owner: VFS + environment/actions
Links:
- `docs/arch-analysis-2025-11-13-1532/04-final-report.md:1069` (M3. VFS Phase 2 Features Unimplemented)
- `docs/plans/2025-11-11-quick-05-structured-obs-masking.md`
