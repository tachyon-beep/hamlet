# TASK-002: Universe Compilation Pipeline with Cross-Validation

## Problem: No Cross-File Consistency Validation

### Current State

HAMLET configs are split across multiple YAML files:

- `bars.yaml` - Meter definitions (energy, health, money, etc.)
- `cascades.yaml` - Meter relationships (low satiation → low energy)
- `affordances.yaml` - Interactions available (Bed, Hospital, etc.)
- `cues.yaml` - UI metadata for visualization
- `actions.yaml` - Action space (UP, DOWN, INTERACT, etc.) [TASK-000]
- `training.yaml` - Hyperparameters and environment config

Each file is validated independently (via Pydantic DTOs), but **no cross-file validation** happens. This allows invalid universes:

**Example 1: Dangling References**

```yaml
# cascades.yaml
- source: "hunger"      # ❌ References non-existent meter
  target: "energy"      # ✅ Valid meter
  threshold: 0.5

# bars.yaml only defines: energy, health, satiation, money
# "hunger" doesn't exist!
```

**Example 2: Orphaned Affordances**

```yaml
# affordances.yaml
affordances:
  - name: "Bed"
    effects:
      - { meter: "energy", amount: 0.5 }

# actions.yaml
actions:
  - { id: 0, name: "UP", type: "movement" }
  - { id: 1, name: "WAIT", type: "passive" }
  # ❌ No INTERACT action! How does agent use affordances?
```

**Example 3: Invalid Meter References**

```yaml
# cues.yaml
cues:
  - meter: "stamina"    # ❌ References non-existent meter
    threshold: 0.3

# bars.yaml defines "energy" not "stamina"
```

These should be **compilation errors**, not runtime surprises.

## Solution: Multi-Stage Universe Compilation Pipeline

### Compilation Pipeline Architecture

The universe compiler validates configs in **dependency order**, ensuring all references resolve:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Foundation (Physics Layer)                         │
├─────────────────────────────────────────────────────────────┤
│ Load: bars.yaml                                             │
│ Output: BarsConfig (meter definitions)                      │
│ Validates:                                                  │
│   ✅ Meter names are unique                                 │
│   ✅ Indices are contiguous from 0                          │
│   ✅ Initial values in valid range                          │
│   ✅ Terminal conditions reference defined meters           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Structure (Topology Layer)                         │
├─────────────────────────────────────────────────────────────┤
│ Load: actions.yaml                                          │
│ Input: BarsConfig (for energy meter validation)            │
│ Output: ActionSpaceConfig                                   │
│ Validates:                                                  │
│   ✅ Action IDs contiguous from 0                           │
│   ✅ Energy costs reference "energy" meter (if exists)      │
│   ✅ Movement deltas match topology (2D for grid2d)         │
│   ✅ At least one action defined                            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Dynamics (Relationships Layer)                     │
├─────────────────────────────────────────────────────────────┤
│ Load: cascades.yaml                                         │
│ Input: BarsConfig (meter vocabulary)                        │
│ Output: CascadeConfig                                       │
│ Validates:                                                  │
│   ✅ All source/target meters exist in BarsConfig           │
│   ✅ No circular dependencies (A→B→A)                       │
│   ✅ Cascade thresholds in valid range [0.0, 1.0]           │
│   ✅ Modulation baseline_depletion matches bar base_depl    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Interactions (Affordance Layer)                    │
├─────────────────────────────────────────────────────────────┤
│ Load: affordances.yaml                                      │
│ Input: BarsConfig (meter vocabulary)                        │
│        ActionSpaceConfig (check for INTERACT action)        │
│ Output: AffordanceConfig                                    │
│ Validates:                                                  │
│   ✅ All meter references (costs/effects) exist             │
│   ✅ If affordances defined, INTERACT action must exist     │
│   ✅ Affordance IDs are unique                              │
│   ✅ Operating hours valid [0-24]                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Presentation (UI Layer)                            │
├─────────────────────────────────────────────────────────────┤
│ Load: cues.yaml                                             │
│ Input: BarsConfig (meter vocabulary)                        │
│ Output: CuesConfig                                          │
│ Validates:                                                  │
│   ✅ All meter references exist                             │
│   ✅ Threshold values in valid range                        │
│   ✅ Color codes are valid hex/named colors                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Runtime (Hyperparameters Layer)                    │
├─────────────────────────────────────────────────────────────┤
│ Load: training.yaml                                         │
│ Input: AffordanceConfig (for enabled_affordances)           │
│        ActionSpaceConfig (for action space size)            │
│ Output: TrainingConfig                                      │
│ Validates:                                                  │
│   ✅ enabled_affordances all exist in AffordanceConfig      │
│   ✅ grid_size² ≥ len(enabled_affordances) + 1             │
│   ✅ Network output_dim matches action_dim                  │
│   ✅ Epsilon decay in reasonable range                      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 7: Assembly (Universe Compilation)                    │
├─────────────────────────────────────────────────────────────┤
│ Combine all validated configs into UniverseConfig           │
│ Final cross-cutting validations:                            │
│   ✅ All enabled affordances fit on grid                    │
│   ✅ No orphaned definitions (defined but unused)           │
│   ✅ Observation dim calculable from configs                │
│   ✅ Network architecture compatible with obs/action space  │
│                                                              │
│ Output: UniverseConfig (fully validated universe)           │
└─────────────────────────────────────────────────────────────┘
                         ↓
                   ✅ COMPILATION SUCCESS
                   Runtime can begin safely
```

### Dependency Ordering Rationale

**Stage 1 (bars.yaml) must be first** because:

- Defines the "vocabulary" of the universe (what meters exist)
- All other configs reference these meters
- The "physics layer" - most fundamental

**Stage 2 (actions.yaml) after bars** because:

- Actions may reference energy meter for costs
- Defines topology that affordances must respect

**Stage 3 (cascades.yaml) after bars** because:

- All source/target references must be valid meters
- Modulations reference bar base_depletion values

**Stage 4 (affordances.yaml) after bars + actions** because:

- Affordance effects reference meters
- Must validate INTERACT action exists if affordances defined

**Stage 5 (cues.yaml) after bars** because:

- UI cues reference meters for display

**Stage 6 (training.yaml) after affordances + actions** because:

- enabled_affordances references affordance definitions
- Network architecture depends on action space size

**Stage 7 (assembly) after all** because:

- Cross-cutting validations require full universe view
- Spatial feasibility checks (grid size vs affordance count)

## Implementation: Universe Compiler

### Master Compiler Class

**File**: `src/townlet/config/universe_compiler.py`

```python
from pathlib import Path
from typing import TypedDict
import logging

logger = logging.getLogger(__name__)

class UniverseConfig(TypedDict):
    """Fully compiled and validated universe configuration."""
    bars: BarsConfig
    actions: ActionSpaceConfig
    cascades: CascadeConfig
    affordances: AffordanceConfig
    cues: CuesConfig
    training: TrainingConfig
    metadata: UniverseMetadata

class UniverseMetadata(BaseModel):
    """Computed metadata about the compiled universe."""
    observation_dim: int
    action_dim: int
    meter_count: int
    affordance_count: int
    enabled_affordance_count: int
    grid_cells: int
    topology: str

class UniverseCompiler:
    """
    Multi-stage compiler for UNIVERSE_AS_CODE configurations.

    Validates configs in dependency order, ensuring all cross-file
    references resolve correctly before allowing runtime execution.
    """

    def __init__(self, config_pack_dir: Path):
        self.config_dir = Path(config_pack_dir)
        if not self.config_dir.is_dir():
            raise ValueError(f"Config pack not found: {config_pack_dir}")

    def compile(self) -> UniverseConfig:
        """
        Compile universe configuration with full cross-validation.

        Returns:
            UniverseConfig: Fully validated universe

        Raises:
            UniverseCompilationError: If any validation fails
        """
        try:
            logger.info(f"🌍 Compiling universe: {self.config_dir.name}")

            # Stage 1: Foundation (Physics)
            bars_config = self._compile_bars()
            logger.info(f"  ✅ Stage 1: Bars ({len(bars_config.bars)} meters)")

            # Stage 2: Structure (Topology)
            actions_config = self._compile_actions(bars_config)
            logger.info(f"  ✅ Stage 2: Actions ({len(actions_config.actions)} actions)")

            # Stage 3: Dynamics (Relationships)
            cascades_config = self._compile_cascades(bars_config)
            logger.info(f"  ✅ Stage 3: Cascades ({len(cascades_config.cascades)} cascades)")

            # Stage 4: Interactions (Affordances)
            affordances_config = self._compile_affordances(bars_config, actions_config)
            logger.info(f"  ✅ Stage 4: Affordances ({len(affordances_config.affordances)} defined)")

            # Stage 5: Presentation (UI)
            cues_config = self._compile_cues(bars_config)
            logger.info(f"  ✅ Stage 5: Cues ({len(cues_config.cues)} cues)")

            # Stage 6: Runtime (Hyperparameters)
            training_config = self._compile_training(affordances_config, actions_config)
            logger.info(f"  ✅ Stage 6: Training (ε={training_config.epsilon_decay})")

            # Stage 7: Assembly (Cross-cutting validations)
            universe = self._assemble_universe(
                bars_config,
                actions_config,
                cascades_config,
                affordances_config,
                cues_config,
                training_config
            )
            logger.info(f"  ✅ Stage 7: Assembly complete")

            # Compute metadata
            metadata = self._compute_metadata(universe)
            universe["metadata"] = metadata

            logger.info(f"✅ Universe compiled successfully!")
            logger.info(f"   Observation dim: {metadata.observation_dim}")
            logger.info(f"   Action dim: {metadata.action_dim}")
            logger.info(f"   Topology: {metadata.topology}")

            return universe

        except Exception as e:
            logger.error(f"❌ UNIVERSE COMPILATION FAILED")
            raise UniverseCompilationError(f"Failed to compile {self.config_dir.name}: {e}") from e

    def _compile_bars(self) -> BarsConfig:
        """Stage 1: Load and validate bars.yaml"""
        bars_path = self.config_dir / "bars.yaml"
        if not bars_path.exists():
            raise FileNotFoundError(f"Required file missing: bars.yaml")

        bars_config = load_bars_config(bars_path)

        # Additional validations
        meter_names = [bar.name for bar in bars_config.bars]
        if len(meter_names) != len(set(meter_names)):
            raise ValueError("Meter names must be unique")

        return bars_config

    def _compile_actions(self, bars_config: BarsConfig) -> ActionSpaceConfig:
        """Stage 2: Load and validate actions.yaml"""
        actions_path = self.config_dir / "actions.yaml"
        if not actions_path.exists():
            raise FileNotFoundError(f"Required file missing: actions.yaml")

        actions_config = load_action_config(actions_path)

        # Cross-validate: If actions reference energy meter, ensure it exists
        meter_names = [bar.name for bar in bars_config.bars]
        for action in actions_config.actions:
            # Energy costs implicitly reference "energy" meter
            if "energy" not in meter_names and action.energy_cost != 0:
                logger.warning(f"Action '{action.name}' has energy_cost but 'energy' meter not defined")

        return actions_config

    def _compile_cascades(self, bars_config: BarsConfig) -> CascadeConfig:
        """Stage 3: Load and validate cascades.yaml"""
        cascades_path = self.config_dir / "cascades.yaml"
        if not cascades_path.exists():
            raise FileNotFoundError(f"Required file missing: cascades.yaml")

        cascades_config = load_cascade_config(cascades_path)

        # Cross-validate: All meter references must exist
        meter_names = {bar.name for bar in bars_config.bars}

        for cascade in cascades_config.cascades:
            source_name = bars_config.bars[cascade.source_index].name
            target_name = bars_config.bars[cascade.target_index].name

            if source_name not in meter_names:
                raise ValueError(f"Cascade references non-existent source meter: '{source_name}'")
            if target_name not in meter_names:
                raise ValueError(f"Cascade references non-existent target meter: '{target_name}'")

        for modulation in cascades_config.modulations:
            if modulation.source not in meter_names:
                raise ValueError(f"Modulation references non-existent source: '{modulation.source}'")
            if modulation.target not in meter_names:
                raise ValueError(f"Modulation references non-existent target: '{modulation.target}'")

        return cascades_config

    def _compile_affordances(
        self,
        bars_config: BarsConfig,
        actions_config: ActionSpaceConfig
    ) -> AffordanceConfig:
        """Stage 4: Load and validate affordances.yaml"""
        affordances_path = self.config_dir / "affordances.yaml"
        if not affordances_path.exists():
            raise FileNotFoundError(f"Required file missing: affordances.yaml")

        affordances_config = load_affordance_config(affordances_path)

        # Cross-validate: All meter references must exist
        meter_names = {bar.name for bar in bars_config.bars}

        for affordance in affordances_config.affordances:
            # Check costs
            for cost in affordance.costs:
                if cost.meter not in meter_names:
                    raise ValueError(
                        f"Affordance '{affordance.name}' cost references non-existent meter: '{cost.meter}'"
                    )

            # Check effects
            for effect in affordance.effects:
                if effect.meter not in meter_names:
                    raise ValueError(
                        f"Affordance '{affordance.name}' effect references non-existent meter: '{effect.meter}'"
                    )

        # CRITICAL VALIDATION: If affordances defined, INTERACT action must exist
        if len(affordances_config.affordances) > 0:
            action_types = [a.type for a in actions_config.actions]
            if "interaction" not in action_types:
                raise ValueError(
                    f"Affordances defined but no 'interaction' action exists! "
                    f"Agent has no mechanism to interact with affordances. "
                    f"Add an INTERACT action to actions.yaml."
                )

        return affordances_config

    def _compile_cues(self, bars_config: BarsConfig) -> CuesConfig:
        """Stage 5: Load and validate cues.yaml"""
        cues_path = self.config_dir / "cues.yaml"
        if not cues_path.exists():
            # Cues are optional for headless training
            logger.error("  ❌ cues.yaml missing – cues required for compiler pipeline")
            return CuesConfig(cues=[])

        cues_config = load_cues_config(cues_path)

        # Cross-validate: All meter references must exist
        meter_names = {bar.name for bar in bars_config.bars}
        for cue in cues_config.cues:
            if cue.meter not in meter_names:
                raise ValueError(
                    f"Cue references non-existent meter: '{cue.meter}'"
                )

        return cues_config

    def _compile_training(
        self,
        affordances_config: AffordanceConfig,
        actions_config: ActionSpaceConfig
    ) -> TrainingConfig:
        """Stage 6: Load and validate training.yaml"""
        training_path = self.config_dir / "training.yaml"
        if not training_path.exists():
            raise FileNotFoundError(f"Required file missing: training.yaml")

        training_config = load_training_config(training_path)

        # Cross-validate: enabled_affordances must exist
        affordance_names = {aff.name for aff in affordances_config.affordances}
        enabled = set(training_config.environment.enabled_affordances or [])

        for aff_name in enabled:
            if aff_name not in affordance_names:
                raise ValueError(
                    f"training.yaml enables non-existent affordance: '{aff_name}'. "
                    f"Available: {sorted(affordance_names)}"
                )

        return training_config

    def _assemble_universe(
        self,
        bars: BarsConfig,
        actions: ActionSpaceConfig,
        cascades: CascadeConfig,
        affordances: AffordanceConfig,
        cues: CuesConfig,
        training: TrainingConfig
    ) -> UniverseConfig:
        """Stage 7: Final cross-cutting validations and assembly"""

        # Validate spatial feasibility
        grid_size = training.environment.grid_size
        enabled_affordances = training.environment.enabled_affordances or []
        grid_cells = grid_size * grid_size
        required_cells = len(enabled_affordances) + 1  # +1 for agent

        if required_cells > grid_cells:
            raise ValueError(
                f"Spatial impossibility: {grid_size}×{grid_size} grid has {grid_cells} cells "
                f"but need {required_cells} ({len(enabled_affordances)} affordances + 1 agent). "
                f"Increase grid_size or reduce enabled_affordances."
            )

        # Warn about orphaned definitions (defined but never enabled)
        all_affordances = {aff.name for aff in affordances.affordances}
        unused = all_affordances - set(enabled_affordances)
        if unused:
            logger.warning(f"  ⚠️  Affordances defined but not enabled: {unused}")

        return {
            "bars": bars,
            "actions": actions,
            "cascades": cascades,
            "affordances": affordances,
            "cues": cues,
            "training": training,
        }

    def _compute_metadata(self, universe: UniverseConfig) -> UniverseMetadata:
        """Compute derived metadata about the universe"""
        training = universe["training"]
        bars = universe["bars"]
        actions = universe["actions"]
        affordances = universe["affordances"]

        # Calculate observation dimension
        if training.environment.partial_observability:
            window_size = 2 * training.environment.vision_range + 1
            obs_dim = (window_size * window_size) + 2 + len(bars.bars) + (len(affordances.affordances) + 1) + 4
        else:
            grid_size = training.environment.grid_size
            obs_dim = (grid_size * grid_size) + len(bars.bars) + (len(affordances.affordances) + 1) + 4

        enabled_count = len(training.environment.enabled_affordances or [])

        return UniverseMetadata(
            observation_dim=obs_dim,
            action_dim=len(actions.actions),
            meter_count=len(bars.bars),
            affordance_count=len(affordances.affordances),
            enabled_affordance_count=enabled_count,
            grid_cells=training.environment.grid_size ** 2,
            topology=actions.topology
        )

class UniverseCompilationError(Exception):
    """Raised when universe compilation fails."""
    pass
```

### Usage Example

```python
# Replace current scattered config loading
compiler = UniverseCompiler("configs/L0_0_minimal")

try:
    universe = compiler.compile()
except UniverseCompilationError as e:
    print(f"❌ Cannot start training: {e}")
    sys.exit(1)

# Now safe to use
env = VectorizedHamletEnv(universe["training"].environment, universe["actions"], ...)
```

## Cross-Validation Examples

### Example 1: Catch Dangling Meter Reference

```yaml
# cascades.yaml
- source: "hunger"  # ❌ Typo! Should be "satiation"
  target: "energy"
```

**Compilation Output**:

```
🌍 Compiling universe: L0_0_minimal
  ✅ Stage 1: Bars (8 meters)
  ✅ Stage 2: Actions (6 actions)
  ❌ Stage 3: Cascades - FAILED

❌ UNIVERSE COMPILATION FAILED
Cascade references non-existent source meter: 'hunger'

Available meters: energy, hygiene, satiation, money, mood, social, health, fitness

Did you mean: satiation?
```

### Example 2: Catch Missing INTERACT Action

```yaml
# affordances.yaml
affordances:
  - name: "Bed"
    effects: [...]

# actions.yaml
actions:
  - { id: 0, name: "UP", type: "movement" }
  - { id: 1, name: "WAIT", type: "passive" }
  # Missing INTERACT action!
```

**Compilation Output**:

```
🌍 Compiling universe: broken_config
  ✅ Stage 1: Bars (8 meters)
  ✅ Stage 2: Actions (2 actions)
  ✅ Stage 3: Cascades (11 cascades)
  ❌ Stage 4: Affordances - FAILED

❌ UNIVERSE COMPILATION FAILED
Affordances defined but no 'interaction' action exists!
Agent has no mechanism to interact with affordances.
Add an INTERACT action to actions.yaml.
```

### Example 3: Catch Spatial Impossibility

```yaml
# training.yaml
environment:
  grid_size: 3  # 3×3 = 9 cells
  enabled_affordances: ["Bed", "Hospital", "HomeMeal", "Job", "Shower",
                        "Gym", "Bar", "Park", "Doctor", "Therapist"]  # 10 affordances
```

**Compilation Output**:

```
🌍 Compiling universe: L0_impossible
  ✅ Stage 1: Bars (8 meters)
  ✅ Stage 2: Actions (6 actions)
  ✅ Stage 3: Cascades (11 cascades)
  ✅ Stage 4: Affordances (14 defined)
  ✅ Stage 5: Cues (8 cues)
  ✅ Stage 6: Training (ε=0.99)
  ❌ Stage 7: Assembly - FAILED

❌ UNIVERSE COMPILATION FAILED
Spatial impossibility: 3×3 grid has 9 cells but need 11 (10 affordances + 1 agent).
Increase grid_size to at least 4 or reduce enabled_affordances.
```

## Benefits

1. **Fail Fast**: Invalid universes rejected before training starts
2. **Clear Errors**: Pinpoint exactly what's wrong and where
3. **Dependency Clarity**: Loading order explicit, not implicit
4. **Cross-File Safety**: References validated across configs
5. **Physics Enforcement**: Spatial impossibilities caught
6. **Documentation**: Compiler output shows universe structure

## Success Criteria

- [ ] UniverseCompiler validates all 7 stages in order
- [ ] Dangling meter references caught at compile time
- [ ] Missing INTERACT action detected when affordances exist
- [ ] Spatial impossibilities (grid too small) caught
- [ ] Orphaned definitions warned about
- [ ] Compilation success/failure logged clearly
- [ ] All existing configs compile successfully
- [ ] Invalid test configs fail compilation with clear errors

## Estimated Effort

- **Phase 1** (compiler architecture): 3-4 hours
- **Phase 2** (cross-validators): 4-6 hours
- **Phase 3** (error messages): 2-3 hours
- **Phase 4** (integration): 2-3 hours
- **Total**: 11-16 hours

## Dependencies

- **TASK-000**: Actions in YAML (provides actions_config to validate against)
- **TASK-001**: DTO schemas (provides individual config loaders)

**Recommended order**: TASK-000 → TASK-001 → TASK-002

## Design Principle: Structural Coherence

The compiler enforces **structural coherence** across the universe:

**✅ ENFORCE** (structural/mathematical coherence):

- Meter references resolve (no dangling pointers)
- Grid has enough space for affordances (spatial feasibility)
- INTERACT action exists if affordances defined (mechanism requirement)
- Action IDs contiguous (indexing coherence)

**❌ DON'T ENFORCE** (semantic reasonableness):

- Whether epsilon_decay is "too slow" (performance hint, not error)
- Whether energy costs are "reasonable" (operator's experiment)
- Whether cascade thresholds are "well-tuned" (empirical question)
- Whether grid is "too large" (operator may be testing scale)

The line: **Enforce what makes the universe logically/mathematically possible to execute, not what makes it likely to succeed.**
