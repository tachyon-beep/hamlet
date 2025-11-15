Title: Expose DAC extrinsic/intrinsic/shaping components via env.info or telemetry

Severity: low
Status: open

Subsystem: environment/vectorized_env + DAC
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/vectorized_env.py:1220`
- `src/townlet/environment/dac_engine.py`

Description:
- `VectorizedHamletEnv._calculate_shaped_rewards()` calls `DACEngine.calculate_rewards()` and receives both:
  - `total_rewards` (scalar per agent), and
  - `components` (a dict of per‑agent tensors for extrinsic, intrinsic_raw, intrinsic_weight, shaping, and sometimes modifier values).
- The method stores `components` only in a private attribute:
  - `self._last_reward_components = components`
- The public `step()` API returns `(observations, rewards, dones, info)` but `info` only contains:
  - `step_counts`, `positions`, and `successful_interactions`.
- This means the detailed DAC breakdown is effectively invisible to callers unless they reach into the private `_last_reward_components` attribute, which is fragile and not part of the documented environment contract.

Proposed Enhancement:
- Provide a supported path for consumers to inspect DAC components without depending on private attributes:
  - Option A: Attach a lightweight summary to `info`, e.g.:
    - `info["dac_components"] = { "extrinsic": extrinsic, "intrinsic": intrinsic_raw, "shaping": shaping }`
    - Optionally gated behind a config flag (e.g., `training.log_dac_components`) to avoid unnecessary tensor copies.
  - Option B: Feed selected components into `AgentRuntimeRegistry` as telemetry fields (e.g., `last_extrinsic`, `last_intrinsic_weight`, `last_shaping`), so they naturally propagate into live inference and database logging.
  - Option C: Add a small public accessor on the env, such as `get_last_reward_components()`, that returns a defensive copy of the last components dictionary.

Migration Impact:
- Backwards compatible if implemented as an additive feature:
  - Existing callers of `step()` are unaffected unless they opt into inspecting new `info["dac_components"]` or telemetry fields.
  - Internal users currently poking `_last_reward_components` can migrate to the new API and later treat `_last_reward_components` as an implementation detail.

Tests:
- Add a unit test to `tests/test_townlet/unit/environment/test_vectorized_env.py` that:
  - Configures a simple DAC setup with non‑trivial extrinsic and shaping components.
  - Calls `env.step()` once and asserts that:
    - `info` (or the new accessor) exposes non‑zero extrinsic and shaping components with shapes `[num_agents]`.
    - The sum of components matches `rewards` within numerical tolerance.

Owner: environment
Links:
- `src/townlet/environment/vectorized_env.py:_calculate_shaped_rewards`
- `src/townlet/environment/dac_engine.py:calculate_rewards`
