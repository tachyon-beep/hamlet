Title: DAC shaping bonus kwargs contract is implicit and fragile

Severity: medium
Status: open

Ticket Type: JANK
Subsystem: environment/dac_engine + environment/vectorized_env
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/dac_engine.py:446`
- `src/townlet/environment/vectorized_env.py:1249`
- `docs/config-schemas/drive_as_code.md`
- `docs/arch-analysis-2025-11-13-1532/04-final-report.md:867`

Description:
- Each DAC shaping bonus closure (`approach_reward`, `completion_bonus`, `streak_bonus`, `diversity_bonus`, `timing_bonus`, etc.) expects a different set of keyword arguments (e.g., `agent_positions`, `affordance_positions`, `last_action_affordance`, `affordance_streak`, `unique_affordances_used`, `current_hour`), but:
  - The required kwargs are **not** expressed in the DAC config schema (there’s no way to know from `drive_as_code.yaml` which runtime fields must be provided by the environment).
  - The environment’s `VectorizedHamletEnv.calculate_rewards()` call site builds a fixed `kwargs` dict with specific keys; if that ever diverges from what a shaping bonus expects, the shaping bonus either silently returns zeros (due to null checks) or fails.
- This creates a brittle implicit contract between DAC and the environment: adding a new shaping bonus or tweaking kwargs on either side can easily break behavior in a way that manifests as “shaping silently disabled” rather than a clear error.

Reproduction:
1) Inspect `DACEngine._compile_shaping()`:
   - Each `create_*_bonus_fn` closure pulls different kwargs from `**kwargs` (e.g., `agent_positions`, `affordance_positions`, `last_action_affordance`, `affordance_streak`, `unique_affordances_used`, `current_hour`).
   - If a required kwarg is missing, many bonuses return `torch.zeros(...)` without logging.
2) Inspect `VectorizedHamletEnv.calculate_rewards()` call site:
   - It constructs a fixed `kwargs` dict with a hard-coded set of keys.
3) Consider future changes:
   - Adding a new shaping bonus type or repurposing an existing one requires coordinated changes to both DAC and env; forgetting to update `kwargs` yields no shaping effect, with no validation.

Expected Behavior:
- The DAC shaping system should have an explicit, verifiable contract for what contextual data each bonus type requires:
  - Either encoded in the config schema (e.g., fields that reference `agent_positions` or `current_hour`), or validated at engine initialization time.
  - Missing required context should raise a clear error during initialization or first call, not silently return zeros.

Actual Behavior:
- Shaping bonuses rely on free-form `**kwargs` and assume the environment has wired up matching keys.
- Missing or misnamed kwargs result in shaping functions returning zero bonuses by design (e.g., `if current_hour is None or last_action_affordance is None: return zeros`), making misconfigurations hard to detect.

Root Cause:
- Shaping bonuses were designed to be flexible and decoupled from environment internals, so a generic `**kwargs` mechanism was used.
- There is currently no schema layer tying DAC shaping configurations to the runtime context provided by the environment, so the interface is only documented in code comments.

Risk:
- Silent misconfiguration: new shaping bonuses may appear to be "enabled" in YAML but never have any effect at runtime due to missing context keys.
- Future refactors of `VectorizedHamletEnv` or DAC could break the shaping pipeline without immediate test failures if coverage is not exhaustive.

Proposed Directions:
- Short-term:
  - Add lightweight runtime validation in `DACEngine.__init__` or `_compile_shaping()` that inspects the active shaping configs and asserts that the environment will provide the required kwargs (e.g., via a known list of keys per bonus type).
  - At minimum, log warnings when a shaping bonus returns all-zero bonuses for many consecutive calls, hinting at missing context.
- Long-term:
  - Introduce a small schema or mapping that explicitly declares the context requirements per shaping bonus type (e.g., `{"approach_reward": ["agent_positions", "affordance_positions"], ...}`) and validate against the environment’s capabilities.
  - Consider threading a structured `RewardContext` object instead of raw `**kwargs` to make contracts type-checkable and more self-documenting.

Tests:
- Add unit/integration tests that intentionally omit required kwargs for a shaping bonus and assert that a clear error or warning is produced (instead of silent all-zero bonuses).

Owner: DAC / environment
Links:
- `docs/arch-analysis-2025-11-13-1532/04-final-report.md:867` (Shaping Bonus Kwargs Dependency noted as technical debt)
