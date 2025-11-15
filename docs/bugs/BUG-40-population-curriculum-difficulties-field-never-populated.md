Title: BatchedAgentState.curriculum_difficulties is always zeroed in VectorizedPopulation

Severity: low
Status: open

Subsystem: training/population + curriculum
Affected Version/Branch: main

Affected Files:
- `src/townlet/population/vectorized.py:616`
- `src/townlet/population/vectorized.py:976`
- `tests/test_townlet/unit/curriculum/test_curriculums.py`

Description:
- `BatchedAgentState` includes a `curriculum_difficulties` tensor intended to carry per-agent difficulty signals from the curriculum/system into decision-making and logging.
- In `VectorizedPopulation.step_population()`, the `BatchedAgentState` instances used both for:
  - Curriculum decision calls (temporary `temp_state`), and
  - The returned state at the end of the step,
  are constructed with `curriculum_difficulties=torch.zeros(self.num_agents, device=self.device)` and never updated.
- Meanwhile, curriculum logic (e.g., `AdversarialCurriculum`) computes continuous difficulty levels, but they are not wired into the `BatchedAgentState.curriculum_difficulties` field.

Reproduction:
- Inspect `VectorizedPopulation.step_population`:
  - For the temp state before curriculum decision:
    ```python
    temp_state = BatchedAgentState(
        ...
        curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
        ...
    )
    ```
  - For the final returned state:
    ```python
    state = BatchedAgentState(
        ...
        curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
        ...
    )
    ```
- The curriculum implementations in `curriculum/base.py` and concrete classes never see non-zero `curriculum_difficulties` from the population path.

Expected Behavior:
- If `curriculum_difficulties` is part of `BatchedAgentState`, it should carry meaningful data:
  - Either the current difficulty level per agent (as a float), or some derived signal.
  - Alternatively, if difficulty is always internal to the curriculum and not meant to be exposed, the field should be removed or clearly documented as unused.

Actual Behavior:
- The field is always zero in population-produced states, so any code that expects it to be informative (e.g., logging, future policies) would be misled.

Root Cause:
- Difficulty signals are produced and consumed inside curriculum classes, but no integration work was done to reflect them into `BatchedAgentState`.

Proposed Fix (Breaking OK):
- Decide whether `curriculum_difficulties` is meant to be:
  - A visible per-agent difficulty scalar (then populate it from `CurriculumDecision.difficulty_level` when decisions are made), or
  - An internal concept (then remove the field from `BatchedAgentState` to avoid confusion).
- If kept:
  - Update `VectorizedPopulation.step_population()` to fill `curriculum_difficulties` in both the temp state and final state based on active curriculum decisions.

Migration Impact:
- Tests that construct `BatchedAgentState` manually will need to either populate `curriculum_difficulties` appropriately or assert the new behavior.

Alternatives Considered:
- Leave the field unused and document it as “reserved for future use”:
  - Rejected; by current repo guidelines, fields that appear live but never carry information are considered jank/bugs.

Tests:
- Add assertions in curriculum-related tests that `curriculum_difficulties` reflects the configured difficulty level when using vectorized populations.

Owner: training/population + curriculum
Links:
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md` (curriculum integration)
