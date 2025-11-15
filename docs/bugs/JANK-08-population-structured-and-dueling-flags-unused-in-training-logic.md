Title: VectorizedPopulation structured/dueling flags exist but training path treats all networks identically

Severity: medium
Status: open

Ticket Type: JANK
Subsystem: training/population + agent/networks
Affected Version/Branch: main

Affected Files:
- `src/townlet/population/vectorized.py:151`
- `src/townlet/population/vectorized.py:210`
- `src/townlet/agent/networks.py:418`

Description:
- `VectorizedPopulation` supports multiple architecture types via `brain_config.architecture.type` and `network_type` (feedforward, recurrent, dueling, structured) and sets flags like `self.is_recurrent` and `self.is_dueling`.
- However, the core training loop in `step_population()` and the feedforward Q-update logic treat all non-recurrent architectures identically:
  - The same DQN loss/target computation and action selection code is used regardless of whether the underlying network is “structured” or “dueling”.
  - `self.is_dueling` is set but never consulted when computing Q-values or when interpreting the network output shape.
- For `StructuredQNetwork` in particular:
  - The network is built with `ObservationActivity` to handle semantic groups, but `VectorizedPopulation` simply calls `self.q_network(obs)` and `gather()`s over the resulting Q-values like any other network, with no special handling or tests validating group-wise masking behavior in the population context.

Reproduction:
- Use a `brain_config` whose architecture type is `"dueling"` or construct a `VectorizedPopulation` with `network_type="structured"` when `brain_config` is absent (in theory).
- Observe that:
  - `self.is_dueling` is set accordingly.
  - Training and action selection logic does not branch on `is_dueling` or architecture type; it always assumes a flat Q-vector per agent.

Expected Behavior:
- Either:
  - Architecture-specific flags like `is_dueling` and “structured” modes should drive concrete differences in training logic (e.g., where value vs advantage outputs are combined, or how masking is applied), **or**
  - These flags and legacy `network_type` entry points should be removed in favor of a single, well-tested Brain As Code path.

Actual Behavior:
- The code gives the impression that dueling/structured architectures are first-class citizens in population training, but the training logic is essentially architecture-agnostic and assumes a simple flat Q-output.

Root Cause:
- Architecture diversity was added via `NetworkFactory` and `brain_config`, but the population training loop was only minimally updated—enough to handle recurrent vs non-recurrent, not enough to reflect dueling/structured nuances.

Risk:
- Future changes to dueling or structured heads (e.g., multiple output heads, separate value/advantage) may require training logic that is not in place; the current flags may lull maintainers into thinking the path is fully supported.

Proposed Directions:
- Short-term:
  - Clarify in docs/comments that dueling/structured architectures are currently trained using the same scalar Q-value loss as simple networks, and that their additional structure is entirely inside the network.
  - Optionally, remove `is_dueling` if it remains unused after review, and route all architectural differences through `NetworkFactory`.
- Long-term:
  - Add architecture-aware training hooks if/when the dueling/structured networks adopt non-trivial output semantics that differ from simple Q-heads.

Tests:
- Add targeted tests for StructuredQNetwork and dueling architectures in the context of `VectorizedPopulation`, verifying that output shapes and training steps behave as expected (or explicitly document limitations).

Owner: training/population + agent/networks
Links:
- `docs/tasks/TASK-005-BRAIN-AS-CODE.md`
