Title: Feedforward reward contract inconsistent across buffers and training path

Severity: medium
Status: open

Subsystem: training/replay-buffer + population
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/replay_buffer.py:121`
- `src/townlet/training/prioritized_replay_buffer.py:75`
- `src/townlet/population/vectorized.py:680`

Description:
- Standard buffer stores split rewards then combines at sample time via `intrinsic_weight`.
- PER stores combined reward at push.
- VectorizedPopulation (DAC path) pushes total reward as `rewards_extrinsic` with `rewards_intrinsic=0` and samples with weight=1.0.
- This divergence increases confusion and risk of double-counting/under-counting if code changes.

Reproduction:
- Review training flow in `VectorizedPopulation.step_population` for feedforward path and compare with buffer APIs.

Expected Behavior:
- Single reward contract for feedforward: store combined reward only; no `intrinsic_weight` in sampling.

Actual Behavior:
- Mixed approaches between buffers and training code.

Root Cause:
- Historical layering: DAC composition introduced after initial split-reward buffer.

Proposed Fix (Breaking OK):
- Standardize: Store only combined rewards for feedforward buffers (standard & PER).
- Remove `intrinsic_weight` parameter from `ReplayBuffer.sample` (breaking) and from callsites.
- Ensure DAC always composes total reward before pushing.

Migration Impact:
- Adjust callsites/tests that pass `intrinsic_weight` or assume split storage in standard buffer.

Alternatives Considered:
- Keep split; adds complexity and risk without benefit in feedforward learning.

Tests:
- Update integration tests that assert combined reward behavior to match new API (no intrinsic_weight everywhere).

Owner: training
Links:
- N/A
