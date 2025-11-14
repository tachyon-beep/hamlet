Title: Experiment-level configuration hierarchy for observation policy and cross-curriculum settings

Severity: medium
Status: open

Subsystem: universe/compiler + config
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py` (Stage 5: metadata generation)
- `src/townlet/config/hamlet.py` (HamletConfig loader)
- `configs/L*/` (all curriculum level directories)

Description:
- After BUG-43 (curriculum masking), all curriculum levels share a common observation superset (both grid_encoding and local_window always present, one masked).
- This enables transfer learning (train on L1, transfer to L2) but forces ALL configs to pay the obs_dim cost of the superset, even single-level experiments that don't need transfer.
- Example waste: L1 full obs has obs_dim=121 but only 96 active dims (25 dims wasted on masked local_window).
- Power users may want to optimize obs_dim for a single level rather than maintain curriculum compatibility.
- Additionally, some settings are fundamentally **experiment-level concerns** (apply to entire curriculum sequence) vs **curriculum-level concerns** (vary per level):
  - Experiment concerns: observation_policy, stratum.yaml, curriculum sequence/ordering, experiment metadata
  - Curriculum concerns: substrate.yaml, bars.yaml, affordances.yaml, training.yaml (per-level hyperparameters)

Current Structure (flat curriculum levels):
```
configs/
├── L0_0_minimal/
│   ├── substrate.yaml
│   ├── bars.yaml
│   └── ...
├── L1_full_observability/
├── L2_partial_observability/
└── stratum.yaml  # shared across levels, but location is ambiguous
```

Proposed Structure (experiment hierarchy):
```
experiments/
└── default_curriculum/
    ├── experiment.yaml         # NEW: experiment-level settings
    │   ├── observation_policy: "curriculum_superset" (default) | "minimal" | "explicit"
    │   ├── stratum: "../shared/stratum.yaml"
    │   ├── curriculum_sequence: [L0_0, L0_5, L1, L2, L3]
    │   └── metadata: (name, description, author)
    └── levels/
        ├── L0_0_minimal/
        │   ├── substrate.yaml
        │   ├── bars.yaml
        │   └── ...
        ├── L1_full_observability/
        └── L2_partial_observability/
```

Proposed Enhancement:

**1. Observation Policy Modes**

Add `observation_policy` to experiment.yaml with three modes:

**a) curriculum_superset (default, current BUG-43 behavior)**
- Includes all fields from all levels (both grid_encoding and local_window)
- Enables checkpoint transfer across curriculum levels
- obs_dim constant across levels (e.g., L1 and L2 both have 121)
- Example: L1 has obs_dim=121 (96 active + 25 masked local_window)

**b) minimal (power user optimization)**
- Only includes curriculum_active=True fields for each level
- Filters out masked fields before building ObservationSpec
- obs_dim varies per level (optimized for each)
- Example: L1 has obs_dim=96 (no masked local_window)
- **Cannot transfer checkpoints between levels** (different obs_dim)

**c) explicit (manual field specification)**
- User explicitly lists which fields to include
- Ignores curriculum_active entirely
- Complete control over observation vector
- Example:
  ```yaml
  observation_policy:
    mode: explicit
    fields:
      - grid_encoding
      - position
      - meters  # shorthand for all meter obs fields
      - affordances
      # (omit local_window, velocity, temporal)
  ```

**2. Experiment-Level Settings**

Move cross-curriculum concerns from per-level configs to experiment.yaml:
- `observation_policy` (new)
- `stratum.yaml` reference (currently lives in configs/ ambiguously)
- `curriculum_sequence` (ordering of levels for training)
- `shared_resources` (paths to shared configs like global_actions.yaml)
- Experiment metadata (name, description, author, tags)

**3. Backwards Compatibility**

- Default mode is "curriculum_superset" (current BUG-43 behavior)
- If no experiment.yaml exists, compiler infers from existing structure
- Existing flat configs/ directory structure continues to work
- Power users opt into experiment/ hierarchy explicitly

Implementation Notes:

**Compiler Changes (Stage 5: Metadata)**:
```python
# In compiler.py, Stage 5 metadata generation
if experiment_config.observation_policy.mode == "minimal":
    # Filter out curriculum_active=False fields before building ObservationSpec
    active_fields = [f for f in vfs_observation_fields if f.curriculum_active]
    observation_spec = ObservationSpec(fields=active_fields)
elif experiment_config.observation_policy.mode == "explicit":
    # Use only explicitly listed fields
    explicit_fields = filter_by_names(vfs_observation_fields, experiment_config.observation_policy.fields)
    observation_spec = ObservationSpec(fields=explicit_fields)
else:  # curriculum_superset (default)
    # Current BUG-43 behavior - include all fields (some masked)
    observation_spec = ObservationSpec(fields=vfs_observation_fields)
```

**Config Loader Changes**:
```python
# HamletConfig needs to understand experiment vs curriculum hierarchy
if (config_dir / "experiment.yaml").exists():
    # New experiment hierarchy
    experiment_config = load_experiment_yaml(config_dir / "experiment.yaml")
    curriculum_level = load_curriculum_level(config_dir / "levels" / level_name)
else:
    # Legacy flat hierarchy - infer defaults
    experiment_config = ExperimentConfig(observation_policy={"mode": "curriculum_superset"})
    curriculum_level = load_curriculum_level(config_dir)
```

Migration Impact:
- Existing configs/ directory structure continues to work (legacy mode)
- New experiments/ directory structure enables experiment-level control
- Power users can optimize obs_dim by switching to "minimal" mode
- Multi-experiment projects can share curriculum levels with different observation policies

Alternatives Considered:
- Add observation_policy to each curriculum level's substrate.yaml:
  - Rejected; observation policy is inherently cross-curriculum (affects transfer learning)
- Keep current flat structure, add observation_policy to each level's config:
  - Rejected; doesn't solve the "stratum.yaml location" problem or enable experiment metadata

Tests:
- Unit tests for ExperimentConfig schema validation
- Compiler tests for each observation_policy mode (superset, minimal, explicit)
- Integration tests verifying obs_dim for each mode:
  - Superset: L1 and L2 have same obs_dim (121)
  - Minimal: L1 has obs_dim=96, L2 has obs_dim=57
  - Explicit: obs_dim matches listed fields only
- Backwards compatibility test: existing flat configs/ directory still compiles

Owner: compiler + config
Links:
- BUG-43: Partial observability global view masking (curriculum masking implementation)
- docs/config-schemas/substrate.md (current observation encoding docs)
- src/townlet/universe/compiler.py:2303-2332 (curriculum_active marking)

Related Enhancements:
- ENH-XX: Stratum.yaml as experiment-level configuration (currently ambiguous location)
- ENH-XX: Multi-experiment workspace support (multiple experiment.yaml files in experiments/ dir)
