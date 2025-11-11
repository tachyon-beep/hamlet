# TASK-004C: Drive As Code (DAC) - Declarative Reward Function Compiler

**Status**: Ready
**Priority**: HIGH
**Estimated Effort**: 30-40 hours
**Dependencies**: TASK-002C (VFS), TASK-004A (Compiler)
**Branch**: `004c-dac-implementation`

---

## AI-Friendly Summary

**For Claude**: This task implements Drive As Code (DAC), a declarative reward function specification system that extracts all reward logic from Python code into composable YAML configurations.

**What This Is**: A reward function compiler that transforms YAML specifications into optimized reward computation graphs. Similar to how UAC compiles universe configs and VFS compiles feature specs, DAC compiles drive specs.

**Why This Matters**: Current reward functions are hardcoded in Python (`RewardStrategy`, `AdaptiveRewardStrategy`). Researchers cannot A/B test reward structures without code changes. DAC makes reward engineering as flexible as universe design.

**Breaking Change**: This is pre-release software with zero users. DAC replaces the legacy reward system entirely - no fallback, no backward compatibility. All configs will be migrated to use `drive_as_code.yaml`.

**Reading Strategy**:
- For implementation: Read Problem Statement → Phase 1 Design → Phase 2 Design → Testing Strategy
- For examples: Jump to "Example Configurations" section (line ~450)
- For architecture: Read Solution Overview → Detailed Design phases
- For acceptance: Jump to Acceptance Criteria (line ~650)

**Key Files**:
- Reference examples: `docs/tasks/TASK-004C-DAC-DEFINITION.md` (626 lines of YAML library)
- Current implementation: `src/townlet/environment/reward_strategy.py` (235 lines - TO BE DELETED)
- VFS integration: `src/townlet/vfs/registry.py` (DAC references VFS variables)

**Note**: This implementation will DELETE the legacy `RewardStrategy` and `AdaptiveRewardStrategy` classes. No backward compatibility.

---

## Problem Statement

### Current State: Hardcoded Reward Functions

Reward functions are currently implemented as Python classes with fixed logic:

```python
# src/townlet/environment/reward_strategy.py (CURRENT)
class RewardStrategy:
    def calculate_rewards(self, step_counts, dones, meters):
        energy = meters[:, self.energy_idx].clamp(min=0.0, max=1.0)
        health = meters[:, self.health_idx].clamp(min=0.0, max=1.0)
        rewards = torch.where(dones, 0.0, health * energy)  # HARDCODED!
        return rewards
```

**Problems**:

1. **No Composability**: Cannot mix strategies (e.g., multiplicative base + threshold bonuses)
2. **Requires Code Changes**: A/B testing reward structures requires editing Python files
3. **No Provenance**: Checkpoints don't track reward function identity
4. **Limited Crisis Handling**: Hardcoded intrinsic weight logic (only `AdaptiveRewardStrategy` has crisis suppression)
5. **No Shaping Support**: Approach rewards, completion bonuses, etc. require custom code
6. **Inconsistent with Design Philosophy**: Universe (UAC), features (VFS), and cognition (BAC) are declarative - rewards should be too
7. **Pre-Release Technical Debt**: Legacy code that should be removed before any users adopt the system

### Why This Is Technical Debt

**Teaching Value**: The "Low Energy Delirium" bug (see `docs/teachable_moments/low_energy_delerium.md`) shows how reward shaping affects agent behavior. But students can't experiment with alternative reward structures without modifying code.

**Research Velocity**: Comparing 5 reward strategies requires:
- Current approach: Write 5 Python classes, modify imports, edit configs
- With DAC: Create 5 YAML files in `configs/L*/drive_as_code.yaml`

**Reproducibility**: Two experiments with "same" config but different hardcoded reward logic are not truly reproducible. DAC adds `drive_hash` to checkpoint provenance.

**Pedagogical Mission**: "Trick students into learning graduate-level RL" requires letting them explore reward engineering as easily as they explore universe design.

---

## Solution Overview

### Design Principle: Separation of Concerns

**VFS** = Feature engineering (observations + derived variables)
**DAC** = Reward engineering (combining features into drive signals)

```yaml
# VFS computes features:
variables:
  - id: energy_urgency
    expression: sigmoid(multiply(subtract(1.0, bar["energy"]), 5))

  - id: worst_physical_need
    expression: min(bar["energy"], bar["health"], bar["satiation"])

# DAC orchestrates them into rewards:
drive_as_code:
  extrinsic:
    base: variable["worst_physical_need"]  # Reference VFS!
    bonuses:
      - variable["energy_urgency"]
```

### Architecture: Reward Function Compiler

DAC follows the same pattern as UAC (Universe Compiler) and VFS:

1. **Parse**: Load `drive_as_code.yaml` → Pydantic DTOs
2. **Validate**: Check meter/variable references, validate ranges
3. **Compile**: Transform declarative spec → optimized computation graph
4. **Execute**: GPU-native vectorized reward calculation
5. **Provenance**: Hash DAC config for checkpoint tracking

### Core Abstractions

**Modifiers**: Range-based multipliers for context-sensitive behavior
```yaml
energy_crisis:
  bar: energy
  ranges:
    - {min: 0.0, max: 0.2, multiplier: 0.0}   # Full suppression
    - {min: 0.2, max: 0.4, multiplier: 0.3}   # Partial suppression
    - {min: 0.4, max: 1.0, multiplier: 1.0}   # No modification
```

**Extrinsic Strategies**: 9 composable reward structures
- constant_base_with_shaped_bonus (RECOMMENDED)
- multiplicative (current - has bug)
- additive_unweighted
- weighted_sum
- polynomial
- threshold_based
- aggregation (min/max/mean)
- vfs_variable (delegate to VFS)
- hybrid (combine multiple)

**Intrinsic Config**: Exploration drive configuration
- RND, ICM, count_based, adaptive_rnd, none
- Modifier application for crisis suppression

**Shaping Bonuses**: 11 behavioral incentive types
- approach_reward, completion_bonus, efficiency_bonus
- state_achievement, streak_bonus, diversity_bonus
- timing_bonus, economic_efficiency, balance_bonus
- crisis_avoidance, vfs_variable

### Composition Formula

```python
total_reward = extrinsic + (intrinsic * effective_intrinsic_weight) + shaping

where:
  extrinsic = computed via extrinsic strategy (with optional modifiers)
  intrinsic = computed via intrinsic strategy (RND/ICM/count/adaptive)
  effective_intrinsic_weight = base_weight * modifier1 * modifier2 * ...
  shaping = sum of all shaping bonuses
```

---

## Detailed Design

### Phase 1: DTO Layer (8-10 hours)

**Goal**: Define Pydantic schemas for all DAC components.

**Location**: `src/townlet/config/drive_as_code.py` (NEW FILE)

**DTOs to Implement**:

1. **Modifier Definitions**
```python
class RangeConfig(BaseModel):
    """Single range in a modifier."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    min: float
    max: float
    multiplier: float

    @model_validator(mode="after")
    def validate_range_bounds(self) -> "RangeConfig":
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self

class ModifierConfig(BaseModel):
    """Range-based modifier for contextual reward adjustment."""
    model_config = ConfigDict(extra="forbid")

    # Source (choose one)
    bar: str | None = None
    variable: str | None = None

    # Range definitions
    ranges: list[RangeConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source(self) -> "ModifierConfig":
        if (self.bar is None and self.variable is None):
            raise ValueError("Must specify either 'bar' or 'variable'")
        if (self.bar is not None and self.variable is not None):
            raise ValueError("Cannot specify both 'bar' and 'variable'")
        return self

    @model_validator(mode="after")
    def validate_ranges_coverage(self) -> "ModifierConfig":
        """Ensure ranges cover [0.0, 1.0] without gaps or overlaps."""
        sorted_ranges = sorted(self.ranges, key=lambda r: r.min)

        # Check coverage starts at 0.0
        if sorted_ranges[0].min != 0.0:
            raise ValueError(f"Ranges must start at 0.0, got {sorted_ranges[0].min}")

        # Check no gaps or overlaps
        for i in range(len(sorted_ranges) - 1):
            current_max = sorted_ranges[i].max
            next_min = sorted_ranges[i + 1].min
            if current_max != next_min:
                raise ValueError(
                    f"Gap or overlap between ranges: "
                    f"{sorted_ranges[i].name} (max={current_max}) and "
                    f"{sorted_ranges[i+1].name} (min={next_min})"
                )

        # Check coverage ends at 1.0
        if sorted_ranges[-1].max != 1.0:
            raise ValueError(f"Ranges must end at 1.0, got {sorted_ranges[-1].max}")

        return self
```

2. **Extrinsic Strategy DTOs** (9 types)
```python
class BarBonusConfig(BaseModel):
    """Bar-based bonus for constant_base_with_shaped_bonus strategy."""
    model_config = ConfigDict(extra="forbid")

    bar: str
    center: float = Field(ge=0.0, le=1.0)  # Neutral point
    scale: float = Field(gt=0.0)  # Magnitude

class VariableBonusConfig(BaseModel):
    """VFS variable-based bonus."""
    model_config = ConfigDict(extra="forbid")

    variable: str
    weight: float

class ExtrinsicStrategyConfig(BaseModel):
    """Base config for extrinsic reward strategies."""
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "constant_base_with_shaped_bonus",
        "multiplicative",
        "additive_unweighted",
        "weighted_sum",
        "polynomial",
        "threshold_based",
        "aggregation",
        "vfs_variable",
        "hybrid",
    ]

    # Fields vary by type (use discriminated union)
    # constant_base_with_shaped_bonus:
    base_reward: float | None = None
    bar_bonuses: list[BarBonusConfig] = Field(default_factory=list)
    variable_bonuses: list[VariableBonusConfig] = Field(default_factory=list)

    # multiplicative:
    base: float | None = None
    bars: list[str] = Field(default_factory=list)

    # ... (other strategy fields)

    apply_modifiers: list[str] = Field(default_factory=list)  # Modifier names to apply
```

3. **Intrinsic Strategy DTOs**
```python
class IntrinsicStrategyConfig(BaseModel):
    """Configuration for intrinsic curiosity."""
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["rnd", "icm", "count_based", "adaptive_rnd", "none"]
    base_weight: float = Field(ge=0.0, le=1.0)
    apply_modifiers: list[str] = Field(default_factory=list)

    # Strategy-specific configs (optional)
    rnd_config: dict[str, Any] | None = None
    icm_config: dict[str, Any] | None = None
    count_config: dict[str, Any] | None = None
    adaptive_config: dict[str, Any] | None = None
```

4. **Shaping Bonus DTOs** (11 types)
```python
class TriggerCondition(BaseModel):
    """Condition for triggering a shaping bonus."""
    model_config = ConfigDict(extra="forbid")

    source: Literal["bar", "variable"]
    name: str
    above: float | None = None
    below: float | None = None

    @model_validator(mode="after")
    def validate_threshold(self) -> "TriggerCondition":
        if self.above is None and self.below is None:
            raise ValueError("Must specify 'above' or 'below'")
        return self

class ApproachRewardConfig(BaseModel):
    """Encourage moving toward goals when needed."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["approach_reward"]
    target_affordance: str
    trigger: TriggerCondition
    bonus: float
    decay_with_distance: bool = True

# ... (10 more shaping bonus types)

ShapingBonusConfig = (
    ApproachRewardConfig
    | CompletionBonusConfig
    | EfficiencyBonusConfig
    | StateAchievementConfig
    | StreakBonusConfig
    | DiversityBonusConfig
    | TimingBonusConfig
    | EconomicEfficiencyConfig
    | BalanceBonusConfig
    | CrisisAvoidanceConfig
    | VFSVariableConfig
)
```

5. **Composition Config**
```python
class CompositionConfig(BaseModel):
    """Reward composition settings."""
    model_config = ConfigDict(extra="forbid")

    normalize: bool = False
    clip: dict[str, float] | None = None  # {"min": -10.0, "max": 100.0}
    log_components: bool = True
    log_modifiers: bool = True
```

6. **Top-Level DAC Config**
```python
class DriveAsCodeConfig(BaseModel):
    """Complete DAC configuration."""
    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1.0")
    modifiers: dict[str, ModifierConfig] = Field(default_factory=dict)
    extrinsic: ExtrinsicStrategyConfig
    intrinsic: IntrinsicStrategyConfig
    shaping: list[ShapingBonusConfig] = Field(default_factory=list)
    composition: CompositionConfig = Field(default_factory=CompositionConfig)

    @model_validator(mode="after")
    def validate_modifier_references(self) -> "DriveAsCodeConfig":
        """Ensure all referenced modifiers exist."""
        defined = set(self.modifiers.keys())

        # Check extrinsic modifiers
        for mod in self.extrinsic.apply_modifiers:
            if mod not in defined:
                raise ValueError(f"Extrinsic references undefined modifier: {mod}")

        # Check intrinsic modifiers
        for mod in self.intrinsic.apply_modifiers:
            if mod not in defined:
                raise ValueError(f"Intrinsic references undefined modifier: {mod}")

        return self

def load_drive_as_code_config(config_dir: Path) -> DriveAsCodeConfig:
    """Load and validate DAC configuration."""
    try:
        data = load_yaml_section(config_dir, "drive_as_code.yaml", "drive_as_code")
        return DriveAsCodeConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "drive_as_code.yaml")) from e
```

**Testing**: 15 unit tests covering:
- RangeConfig validation (gaps, overlaps, bounds)
- ModifierConfig source validation
- ExtrinsicStrategyConfig type-specific fields
- ShapingBonusConfig discriminated unions
- DriveAsCodeConfig modifier reference validation

**Files**:
- `src/townlet/config/drive_as_code.py` (NEW, ~600 lines)
- `tests/test_townlet/unit/config/test_drive_as_code_dto.py` (NEW, ~400 lines)

---

### Phase 2: Compiler Integration (6-8 hours)

**Goal**: Integrate DAC validation into UAC Stage 3 (reference resolution) and Stage 4 (cross-validation).

**Location**: `src/townlet/universe/compiler.py` (MODIFY)

**Stage 3 Extensions** (Line ~808, after affordance validation):

```python
def _validate_dac_references(
    dac_config: DriveAsCodeConfig,
    symbol_table: SymbolTable,
    errors: CompilationErrorCollector,
    formatter: Callable,
) -> None:
    """Validate DAC references to bars, variables, and affordances.

    Checks:
    - Modifiers reference valid bars or VFS variables
    - Extrinsic strategies reference valid bars/variables
    - Shaping bonuses reference valid affordances
    - All VFS variable references exist in symbol table
    """

    # Validate modifier sources
    for mod_name, mod_config in dac_config.modifiers.items():
        if mod_config.bar:
            if mod_config.bar not in symbol_table.bars:
                errors.add(
                    formatter(
                        "DAC-REF-001",
                        f"Modifier '{mod_name}' references undefined bar: {mod_config.bar}",
                        f"drive_as_code.yaml:modifiers.{mod_name}",
                    )
                )
        elif mod_config.variable:
            if mod_config.variable not in symbol_table.vfs_variables:
                errors.add(
                    formatter(
                        "DAC-REF-002",
                        f"Modifier '{mod_name}' references undefined VFS variable: {mod_config.variable}",
                        f"drive_as_code.yaml:modifiers.{mod_name}",
                    )
                )

    # Validate extrinsic strategy references
    if dac_config.extrinsic.type == "multiplicative":
        for bar in dac_config.extrinsic.bars:
            if bar not in symbol_table.bars:
                errors.add(
                    formatter(
                        "DAC-REF-003",
                        f"Extrinsic strategy references undefined bar: {bar}",
                        "drive_as_code.yaml:extrinsic.bars",
                    )
                )

    # ... (similar checks for other extrinsic types)

    # Validate shaping bonus references
    for idx, shaping in enumerate(dac_config.shaping):
        if shaping.type == "approach_reward":
            if shaping.target_affordance not in symbol_table.affordances:
                errors.add(
                    formatter(
                        "DAC-REF-004",
                        f"Shaping bonus references undefined affordance: {shaping.target_affordance}",
                        f"drive_as_code.yaml:shaping[{idx}]",
                    )
                )
```

**Stage 4 Extensions** (Line ~1180, after capability validation):

```python
def _validate_dac_consistency(
    dac_config: DriveAsCodeConfig,
    errors: CompilationErrorCollector,
    formatter: Callable,
) -> None:
    """Validate DAC internal consistency.

    Checks:
    - Extrinsic strategies have required fields for their type
    - Intrinsic strategies have required configs
    - Shaping bonuses have valid trigger conditions
    - Composition clipping ranges are sensible
    """

    # Validate extrinsic strategy completeness
    if dac_config.extrinsic.type == "constant_base_with_shaped_bonus":
        if dac_config.extrinsic.base_reward is None:
            errors.add(
                formatter(
                    "DAC-VAL-001",
                    "constant_base_with_shaped_bonus requires 'base_reward' field",
                    "drive_as_code.yaml:extrinsic",
                )
            )
        if not dac_config.extrinsic.bar_bonuses and not dac_config.extrinsic.variable_bonuses:
            errors.add(
                formatter(
                    "DAC-VAL-002",
                    "constant_base_with_shaped_bonus should have at least one bonus (bar or variable)",
                    "drive_as_code.yaml:extrinsic",
                )
            )

    # ... (similar checks for other extrinsic types)

    # Validate intrinsic strategy completeness
    if dac_config.intrinsic.strategy == "rnd" and not dac_config.intrinsic.rnd_config:
        # Warning, not error - can use defaults
        pass

    # Validate composition clipping
    if dac_config.composition.clip:
        clip_min = dac_config.composition.clip.get("min")
        clip_max = dac_config.composition.clip.get("max")
        if clip_min is not None and clip_max is not None and clip_min >= clip_max:
            errors.add(
                formatter(
                    "DAC-VAL-003",
                    f"Composition clip min ({clip_min}) must be < max ({clip_max})",
                    "drive_as_code.yaml:composition.clip",
                )
            )
```

**Compilation Pipeline Integration**:

```python
def compile_universe(config_dir: Path, ...) -> CompiledUniverse:
    # ... existing stages ...

    # NEW: Load DAC config (REQUIRED)
    dac_config = load_drive_as_code_config(config_dir)  # Raises if missing

    # Stage 3: Validate DAC references
    _validate_dac_references(dac_config, symbol_table, errors, _format_error)

    # Stage 4: Validate DAC consistency
    _validate_dac_consistency(dac_config, errors, _format_error)

    # ... rest of compilation ...

    # Stage 7: Emit compiled universe
    compiled = CompiledUniverse(
        # ... existing fields ...
        dac_config=dac_config,  # NEW FIELD
        drive_hash=_compute_dac_hash(dac_config) if dac_config else None,  # NEW FIELD
    )

    return compiled
```

**Testing**: 12 unit tests covering:
- DAC reference validation (undefined bars, variables, affordances)
- DAC consistency validation (missing required fields, invalid clip ranges)
- Compilation fails gracefully when `drive_as_code.yaml` is missing
- Error messages and locations

**Files**:
- `src/townlet/universe/compiler.py` (MODIFY, +300 lines)
- `src/townlet/universe/compiled_universe.py` (MODIFY, add `dac_config` and `drive_hash` fields)
- `tests/test_townlet/unit/universe/test_dac_compiler_integration.py` (NEW, ~350 lines)

---

### Phase 3: Runtime Execution (10-12 hours)

**Goal**: Implement DAC reward computation engine.

**Location**: `src/townlet/environment/dac_engine.py` (NEW FILE)

**Architecture**:

```python
class DACEngine:
    """Drive As Code reward computation engine.

    Compiles declarative DAC specs into optimized GPU-native computation graphs.

    Design:
    - All operations vectorized across agents (batch dimension)
    - Modifier evaluation uses torch.where for range lookups
    - VFS integration via runtime registry
    - Intrinsic weight modulation for crisis suppression
    """

    def __init__(
        self,
        dac_config: DriveAsCodeConfig,
        vfs_registry: VFSRegistry,
        device: torch.device,
        num_agents: int,
    ):
        self.dac_config = dac_config
        self.vfs_registry = vfs_registry
        self.device = device
        self.num_agents = num_agents

        # Compile modifiers into lookup tables
        self.modifiers = self._compile_modifiers()

        # Compile extrinsic strategy
        self.extrinsic_fn = self._compile_extrinsic()

        # Compile shaping bonuses
        self.shaping_fns = self._compile_shaping()

        # Logging
        self.log_components = dac_config.composition.log_components
        self.log_modifiers = dac_config.composition.log_modifiers

    def _compile_modifiers(self) -> dict[str, Callable]:
        """Compile modifiers into efficient lookup functions."""
        compiled = {}

        for name, config in self.dac_config.modifiers.items():
            # Create range boundaries and multipliers as tensors
            boundaries = torch.tensor(
                [r.min for r in config.ranges] + [config.ranges[-1].max],
                device=self.device,
            )
            multipliers = torch.tensor(
                [r.multiplier for r in config.ranges],
                device=self.device,
            )

            def modifier_fn(value: torch.Tensor) -> torch.Tensor:
                # Vectorized range lookup
                # value: [num_agents]
                # Returns: [num_agents] multipliers

                # Find which range each agent's value falls into
                # digitize: boundaries [0.0, 0.2, 0.4, 1.0] → bins [0, 1, 2]
                bins = torch.searchsorted(boundaries[:-1], value, right=False)
                bins = bins.clamp(max=len(multipliers) - 1)

                return multipliers[bins]

            compiled[name] = modifier_fn

        return compiled

    def _compile_extrinsic(self) -> Callable:
        """Compile extrinsic strategy into computation function."""
        strategy = self.dac_config.extrinsic

        if strategy.type == "constant_base_with_shaped_bonus":
            def compute_extrinsic(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                # Base reward
                base = torch.full((self.num_agents,), strategy.base_reward, device=self.device)

                # Bar bonuses
                for bonus_config in strategy.bar_bonuses:
                    bar_value = self._get_bar_value(bonus_config.bar, meters)
                    bonus = bonus_config.scale * (bar_value - bonus_config.center)
                    base = base + bonus

                # Variable bonuses (from VFS)
                for bonus_config in strategy.variable_bonuses:
                    var_value = self.vfs_registry.get_variable(bonus_config.variable)
                    bonus = bonus_config.weight * var_value
                    base = base + bonus

                # Zero out dead agents
                return torch.where(dones, torch.zeros_like(base), base)

            return compute_extrinsic

        elif strategy.type == "multiplicative":
            def compute_extrinsic(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                result = torch.full((self.num_agents,), strategy.base, device=self.device)

                for bar_name in strategy.bars:
                    bar_value = self._get_bar_value(bar_name, meters)
                    result = result * bar_value

                return torch.where(dones, torch.zeros_like(result), result)

            return compute_extrinsic

        # ... (other strategy types)

        else:
            raise ValueError(f"Unknown extrinsic strategy: {strategy.type}")

    def _compile_shaping(self) -> list[Callable]:
        """Compile shaping bonuses into computation functions."""
        compiled = []

        for shaping_config in self.dac_config.shaping:
            if shaping_config.type == "approach_reward":
                def compute_approach(
                    positions: torch.Tensor,
                    target_pos: torch.Tensor,
                    meters: torch.Tensor,
                ) -> torch.Tensor:
                    # Check trigger condition
                    trigger_value = self._evaluate_trigger(shaping_config.trigger, meters)
                    triggered = trigger_value > 0.5

                    # Compute distance
                    distances = torch.norm(positions - target_pos, dim=1)

                    # Compute bonus (decay with distance if enabled)
                    if shaping_config.decay_with_distance:
                        bonus = shaping_config.bonus / (1.0 + distances)
                    else:
                        bonus = torch.full_like(distances, shaping_config.bonus)

                    # Apply only when triggered
                    return torch.where(triggered, bonus, torch.zeros_like(bonus))

                compiled.append(compute_approach)

            # ... (other shaping types)

        return compiled

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
        meters: torch.Tensor,
        intrinsic_raw: torch.Tensor,
        **kwargs,  # Additional context (positions, affordance states, etc.)
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate total rewards with DAC.

        Args:
            step_counts: [num_agents] current step count
            dones: [num_agents] agent death flags
            meters: [num_agents, meter_count] normalized meter values
            intrinsic_raw: [num_agents] raw intrinsic curiosity values (from RND/ICM)
            **kwargs: Additional context for shaping bonuses

        Returns:
            total_rewards: [num_agents] final rewards
            intrinsic_weights: [num_agents] effective intrinsic weights (for logging)
            components: dict of reward components (extrinsic, intrinsic, shaping)
        """

        # 1. Compute extrinsic rewards
        extrinsic = self.extrinsic_fn(meters, dones)

        # 2. Apply modifiers to extrinsic (optional)
        for mod_name in self.dac_config.extrinsic.apply_modifiers:
            source_value = self._get_modifier_source(mod_name, meters)
            multiplier = self.modifiers[mod_name](source_value)
            extrinsic = extrinsic * multiplier

        # 3. Compute intrinsic weights with modifiers
        intrinsic_weight = torch.full(
            (self.num_agents,),
            self.dac_config.intrinsic.base_weight,
            device=self.device,
        )

        for mod_name in self.dac_config.intrinsic.apply_modifiers:
            source_value = self._get_modifier_source(mod_name, meters)
            multiplier = self.modifiers[mod_name](source_value)
            intrinsic_weight = intrinsic_weight * multiplier

        intrinsic = intrinsic_raw * intrinsic_weight

        # 4. Compute shaping bonuses
        shaping_total = torch.zeros(self.num_agents, device=self.device)
        for shaping_fn in self.shaping_fns:
            shaping_total = shaping_total + shaping_fn(meters=meters, **kwargs)

        # 5. Compose total reward
        total = extrinsic + intrinsic + shaping_total

        # 6. Apply composition rules (clipping, normalization)
        if self.dac_config.composition.clip:
            clip_min = self.dac_config.composition.clip.get("min", -float("inf"))
            clip_max = self.dac_config.composition.clip.get("max", float("inf"))
            total = total.clamp(min=clip_min, max=clip_max)

        if self.dac_config.composition.normalize:
            total = torch.tanh(total)  # Normalize to [-1, 1]

        # 7. Collect components for logging
        components = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "shaping": shaping_total,
        }

        return total, intrinsic_weight, components

    def _get_bar_value(self, bar_name: str, meters: torch.Tensor) -> torch.Tensor:
        """Get bar value from meters tensor."""
        bar_idx = self.vfs_registry.get_bar_index(bar_name)
        return meters[:, bar_idx]

    def _get_modifier_source(self, mod_name: str, meters: torch.Tensor) -> torch.Tensor:
        """Get source value for modifier evaluation."""
        config = self.dac_config.modifiers[mod_name]

        if config.bar:
            return self._get_bar_value(config.bar, meters)
        elif config.variable:
            return self.vfs_registry.get_variable(config.variable)
        else:
            raise ValueError(f"Modifier {mod_name} has no source")

    def _evaluate_trigger(self, trigger: TriggerCondition, meters: torch.Tensor) -> torch.Tensor:
        """Evaluate trigger condition."""
        if trigger.source == "bar":
            value = self._get_bar_value(trigger.name, meters)
        elif trigger.source == "variable":
            value = self.vfs_registry.get_variable(trigger.name)
        else:
            raise ValueError(f"Unknown trigger source: {trigger.source}")

        if trigger.above is not None:
            return (value > trigger.above).float()
        elif trigger.below is not None:
            return (value < trigger.below).float()
        else:
            raise ValueError("Trigger must have 'above' or 'below'")
```

**Integration with Population**:

```python
# src/townlet/population/vectorized.py (MODIFY)

class VectorizedPopulation:
    def __init__(self, ...):
        # ... existing initialization ...

        # DAC integration (REQUIRED - no fallback)
        self.dac_engine = DACEngine(
            dac_config=compiled_universe.dac_config,
            vfs_registry=self.vfs_registry,
            device=self.device,
            num_agents=self.num_agents,
        )

    def _calculate_rewards(self, ...):
        total_rewards, intrinsic_weights, components = self.dac_engine.calculate_rewards(
            step_counts=self.step_counts,
            dones=self.dones,
            meters=self.env.meters,
            intrinsic_raw=self.intrinsic_raw,
            positions=self.env.positions,  # For shaping bonuses
            affordance_states=self.env.affordance_states,
        )

        # Log components if enabled
        if self.dac_engine.log_components:
            self.tensorboard_writer.add_scalars(
                "DAC/reward_components",
                {
                    "extrinsic": components["extrinsic"].mean().item(),
                    "intrinsic": components["intrinsic"].mean().item(),
                    "shaping": components["shaping"].mean().item(),
                },
                self.step_counter,
            )

        return total_rewards, intrinsic_weights
```

**Legacy Code Removal**:

```python
# DELETE these files:
# - src/townlet/environment/reward_strategy.py (entire file)

# REMOVE from training.yaml:
# - reward_strategy field (line 76)
```

**Testing**: 20 unit tests covering:
- Modifier compilation and evaluation
- Extrinsic strategy computation (all 9 types)
- Shaping bonus computation (sample 5 types)
- Composition rules (clipping, normalization)
- VFS integration
- Error handling when DAC config is malformed

**Files**:
- `src/townlet/environment/dac_engine.py` (NEW, ~800 lines)
- `src/townlet/population/vectorized.py` (MODIFY, +50 lines)
- `tests/test_townlet/unit/environment/test_dac_engine.py` (NEW, ~600 lines)

---

### Phase 4: Provenance & Checkpoint Integration (3-4 hours)

**Goal**: Track DAC configuration in checkpoints for reproducibility.

**Location**: Multiple files

**Changes**:

1. **Add `drive_hash` to checkpoints**:
```python
# src/townlet/training/checkpoint.py (MODIFY)

def save_checkpoint(population, compiled_universe, checkpoint_dir, episode):
    checkpoint = {
        "episode": episode,
        "network_state": population.q_network.state_dict(),
        "optimizer_state": population.optimizer.state_dict(),
        # ... existing fields ...

        # NEW: DAC provenance (always present)
        "drive_hash": compiled_universe.drive_hash,
        "dac_config": asdict(compiled_universe.dac_config),
    }

    torch.save(checkpoint, checkpoint_dir / f"checkpoint_ep{episode}.pt")

def load_checkpoint(population, compiled_universe, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # ... existing loading ...

    # Validate DAC consistency
    if checkpoint.get("drive_hash") != compiled_universe.drive_hash:
        warnings.warn(
            f"Checkpoint was trained with different DAC config!\n"
            f"Checkpoint drive_hash: {checkpoint['drive_hash']}\n"
            f"Current drive_hash: {compiled_universe.drive_hash}"
        )
```

2. **Add `drive_hash` to TensorBoard hparams**:
```python
# src/townlet/training/runner.py (MODIFY)

def log_hyperparameters(self):
    self.writer.add_hparams(
        {
            "config_hash": self.compiled_universe.config_hash,
            "drive_hash": self.compiled_universe.drive_hash,  # NEW (always present)
            # ... existing hparams ...
        },
        {},
    )
```

**Testing**: 5 unit tests covering:
- Checkpoint save with DAC config
- Checkpoint load with matching DAC config
- Warning on DAC mismatch
- TensorBoard hparam logging

**Files**:
- `src/townlet/training/checkpoint.py` (MODIFY, +30 lines)
- `src/townlet/training/runner.py` (MODIFY, +5 lines)
- `tests/test_townlet/unit/training/test_checkpoint_dac.py` (NEW, ~150 lines)

---

### Phase 5: Documentation & Examples (4-6 hours)

**Goal**: Document DAC system and provide curriculum examples.

**Deliverables**:

1. **Operator Guide**: `docs/config-schemas/drive_as_code.md`
   - Complete reference for all DAC components
   - Modifier design patterns
   - Extrinsic strategy comparison table
   - Shaping bonus usage guidelines
   - VFS integration examples
   - Debugging tips (component logging, modifier visualization)

2. **Migrate All Existing Configurations**:
   - `configs/L0_0_minimal/drive_as_code.yaml` - Multiplicative (baseline)
   - `configs/L0_5_dual_resource/drive_as_code.yaml` - Fixes "Low Energy Delirium" bug
   - `configs/L1_full_observability/drive_as_code.yaml` - Multi-objective balancing
   - `configs/L2_partial_observability/drive_as_code.yaml` - Same as L1 (POMDP compatible)
   - `configs/L3_temporal_mechanics/drive_as_code.yaml` - Temporal exploration decay
   - Remove `reward_strategy` field from all `training.yaml` files

3. **Migration Reference**: `docs/migration/legacy_reward_to_dac.md`
   - How `RewardStrategy` maps to DAC (multiplicative strategy)
   - How `AdaptiveRewardStrategy` maps to DAC (constant_base_with_shaped_bonus + crisis suppression)
   - Side-by-side comparison for pedagogical value
   - Note: Legacy classes will be deleted

4. **Update CLAUDE.md**:
   - Add DAC quick reference
   - Document `drive_as_code.yaml` location (REQUIRED file)
   - Remove references to legacy reward strategies

**Files**:
- `docs/config-schemas/drive_as_code.md` (NEW, ~800 lines)
- `configs/L0_0_minimal/drive_as_code.yaml` (NEW, ~50 lines - multiplicative baseline)
- `configs/L0_5_dual_resource/drive_as_code.yaml` (NEW, ~80 lines - fixes bug)
- `configs/L1_full_observability/drive_as_code.yaml` (NEW, ~120 lines)
- `configs/L2_partial_observability/drive_as_code.yaml` (NEW, ~120 lines)
- `configs/L3_temporal_mechanics/drive_as_code.yaml` (NEW, ~100 lines)
- `configs/*/training.yaml` (MODIFY all - remove `reward_strategy` field)
- `src/townlet/environment/reward_strategy.py` (DELETE entire file)
- `docs/migration/legacy_reward_to_dac.md` (NEW, ~300 lines)
- `CLAUDE.md` (MODIFY, +30 lines, remove legacy references)

---

## Testing Strategy

### Unit Tests (50 tests total)

1. **DTO Layer** (15 tests)
   - RangeConfig validation (gaps, overlaps, bounds)
   - ModifierConfig source validation
   - ExtrinsicStrategyConfig type-specific fields
   - ShapingBonusConfig discriminated unions
   - DriveAsCodeConfig modifier reference validation

2. **Compiler Integration** (12 tests)
   - DAC reference validation (bars, variables, affordances)
   - DAC consistency validation (required fields, clip ranges)
   - Compilation with/without `drive_as_code.yaml`
   - Error messages and locations

3. **Runtime Execution** (20 tests)
   - Modifier evaluation (range lookups)
   - Extrinsic computation (9 strategies)
   - Shaping computation (sample 5 types)
   - Composition rules (clip, normalize)
   - VFS integration

4. **Provenance** (5 tests)
   - Checkpoint save/load with DAC
   - Drive hash consistency
   - TensorBoard hparam logging

### Integration Tests (8 tests)

1. **End-to-End Workflows**:
   - Load config with `drive_as_code.yaml` → compile → train 100 episodes
   - Missing `drive_as_code.yaml` → compilation fails with clear error message
   - A/B test: L0_0_minimal (multiplicative) vs L0_5_dual_resource (crisis suppression)

2. **Curriculum Configs**:
   - All L0/L1/L2/L3 configs compile and train with DAC
   - L0_5_dual_resource demonstrates bug fix (crisis suppression)
   - L3_temporal_mechanics demonstrates temporal exploration decay

3. **Breaking Change Validation**:
   - Attempting to use old configs without `drive_as_code.yaml` fails gracefully
   - Error message guides user to create DAC config or see migration docs

### Performance Tests (2 benchmarks)

1. **Reward Computation Overhead**:
   - Measure DAC vs legacy RewardStrategy throughput
   - Target: <5% overhead for simple configs, <15% for complex configs

2. **Memory Usage**:
   - Measure DAC engine memory footprint
   - Target: <10MB per population instance

3. **Legacy Code Removal**:
   - Verify all references to `RewardStrategy` and `AdaptiveRewardStrategy` removed
   - Verify `reward_strategy` field removed from all training configs
   - Verify no fallback logic remains in codebase

---

## Example Configurations

### Example 1: Multiplicative Baseline (L0_0_minimal)

**Location**: `configs/L0_0_minimal/drive_as_code.yaml`

```yaml
version: "1.0"

modifiers: {}

extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy, health]

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: []

shaping: []

composition:
  log_components: true
```

**Result**: Equivalent to legacy `RewardStrategy` (multiplicative: health × energy). Has the "Low Energy Delirium" bug.

### Example 2: Fix "Low Energy Delirium" Bug (L0_5_dual_resource)

**Location**: `configs/L0_5_dual_resource/drive_as_code.yaml`

```yaml
version: "1.0"

modifiers:
  energy_health_crisis:
    variable: worst_physical_need  # VFS: min(energy, health)
    ranges:
      - {name: crisis, min: 0.0, max: 0.3, multiplier: 0.0}
      - {name: normal, min: 0.3, max: 1.0, multiplier: 1.0}

extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  bar_bonuses:
    - {bar: energy, center: 0.5, scale: 0.5}
    - {bar: health, center: 0.5, scale: 0.5}
  apply_modifiers: []

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: [energy_health_crisis]  # Suppress exploration in crisis!

shaping: []

composition:
  log_components: true
  log_modifiers: true
```

**Result**: When `energy < 0.3` or `health < 0.3`, intrinsic weight → 0.0, preventing exploration distraction. Fixes the bug in L0_0_minimal.

### Example 3: Advanced Urgency-Based Shaping (L1_full_observability)

**Location**: `configs/L1_full_observability/drive_as_code.yaml`

```yaml
version: "1.0"

modifiers: {}

extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  variable_bonuses:
    - {variable: energy_urgency, weight: 0.5}  # VFS variable
    - {variable: bathroom_emergency, weight: -2.0}

intrinsic:
  strategy: adaptive_rnd
  base_weight: 0.100
  apply_modifiers: []

shaping:
  - type: approach_reward
    target_affordance: Bed
    trigger: {source: variable, name: energy_urgency, above: 0.7}
    bonus: 1.0
    decay_with_distance: true

  - type: approach_reward
    target_affordance: Toilet
    trigger: {source: variable, name: bathroom_emergency, above: 0.5}
    bonus: 5.0  # VERY important!
    decay_with_distance: true

  - type: completion_bonus
    affordances: all
    bonus: 1.0
    scale_with_duration: true

composition:
  log_components: true
  clip: {min: -10.0, max: 100.0}
```

### Example 4: Temporal Exploration Decay (L3_temporal_mechanics)

**Location**: `configs/L3_temporal_mechanics/drive_as_code.yaml`

```yaml
version: "1.0"

modifiers:
  temporal_exploration_decay:
    variable: age  # VFS: step_count / max_steps
    ranges:
      - {name: early, min: 0.0, max: 0.2, multiplier: 5.0}   # High exploration early
      - {name: mid, min: 0.2, max: 0.6, multiplier: 1.0}
      - {name: late, min: 0.6, max: 1.0, multiplier: 0.1}    # Low exploration late

extrinsic:
  type: additive_unweighted
  base: 0.5
  bars: [energy, health, satiation]

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: [temporal_exploration_decay]

shaping: []

composition:
  log_components: true
```

**Result**: Episode starts with 50% intrinsic weight, ends with 1% intrinsic weight.

---

## Acceptance Criteria

### Mandatory (Must Have)

- [ ] All DAC DTOs implemented and validated (Phase 1)
- [ ] DAC config loads from `drive_as_code.yaml` (Phase 1)
- [ ] Compiler validates DAC references (bars, variables, affordances) (Phase 2)
- [ ] Compiler validates DAC consistency (required fields, ranges) (Phase 2)
- [ ] DACEngine computes rewards from declarative spec (Phase 3)
- [ ] All 9 extrinsic strategies implemented (Phase 3)
- [ ] At least 5 shaping bonus types implemented (Phase 3)
- [ ] VFS integration works (modifiers reference VFS variables) (Phase 3)
- [ ] Missing `drive_as_code.yaml` causes clear compilation error (Phase 3)
- [ ] Legacy `RewardStrategy` and `AdaptiveRewardStrategy` classes deleted (Phase 3)
- [ ] `reward_strategy` field removed from all `training.yaml` files (Phase 5)
- [ ] `drive_hash` tracked in checkpoints (Phase 4)
- [ ] All curriculum configs (L0/L1/L2/L3) migrated to DAC (Phase 5)
- [ ] L0_5_dual_resource example demonstrates bug fix (Phase 5)
- [ ] Operator guide documents all components (Phase 5)
- [ ] 50 unit tests pass (all phases)
- [ ] 8 integration tests pass (all phases)

### Optional (Nice to Have)

- [ ] All 11 shaping bonus types implemented (Phase 3 extension)
- [ ] Hierarchical rewards (meta-controller + controller) (Phase 3 extension)
- [ ] Curriculum support (automatic reward annealing) (Phase 3 extension)
- [ ] TensorBoard visualizations for modifier ranges (Phase 4 extension)
- [ ] Interactive DAC config editor (web UI) (Phase 5 extension)

### Performance Targets

- [ ] Reward computation overhead <5% for simple configs
- [ ] Reward computation overhead <15% for complex configs
- [ ] Memory footprint <10MB per population instance

---

## Risk Assessment

### High Risk

**Risk**: DAC adds computational overhead that slows training.
**Mitigation**:
- Compile modifiers into lookup tables (not runtime parsing)
- Vectorize all operations across agent dimension
- Benchmark against equivalent hardcoded logic and optimize hotspots
- Provide performance profiling tools

**Risk**: Complex DAC configs are hard to debug.
**Mitigation**:
- Component logging (extrinsic, intrinsic, shaping separate)
- Modifier visualization (show active ranges per agent)
- Clear error messages with YAML location
- Example configs with comments

### Medium Risk

**Risk**: Breaking changes to reward structure invalidate old checkpoints.
**Mitigation**:
- Track `drive_hash` in checkpoints
- Warn on mismatch at load time
- Allow override for experimentation
- Note: Pre-release software - breaking changes are acceptable

**Risk**: VFS integration is fragile (variable references break easily).
**Mitigation**:
- Validate all VFS references in compiler Stage 3
- Provide clear error messages
- Document VFS→DAC integration patterns

### Low Risk

**Risk**: Users find DAC too complex and don't understand reward configurations.
**Mitigation**:
- Provide simple examples first (L0_0_minimal multiplicative baseline)
- Comprehensive operator guide with pattern library
- Show pedagogical value (L0_5 bug fix example)
- Clear error messages during compilation

---

## Migration Reference (For Pedagogical Value)

**Note**: This section documents how the legacy reward classes map to DAC configurations for educational purposes. The legacy classes (`RewardStrategy`, `AdaptiveRewardStrategy`) will be deleted during implementation.

### How `RewardStrategy` Maps to DAC

**Legacy (Python - TO BE DELETED)**:
```python
class RewardStrategy:
    def calculate_rewards(self, step_counts, dones, meters):
        energy = meters[:, self.energy_idx]
        health = meters[:, self.health_idx]
        rewards = torch.where(dones, 0.0, health * energy)
        return rewards
```

**DAC (YAML)**:
```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy, health]

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: []

shaping: []
```

### How `AdaptiveRewardStrategy` Maps to DAC

**Legacy (Python - TO BE DELETED)**:
```python
class AdaptiveRewardStrategy:
    def calculate_rewards(self, step_counts, dones, meters):
        energy = meters[:, self.energy_idx].clamp(min=0.0, max=1.0)
        health = meters[:, self.health_idx].clamp(min=0.0, max=1.0)

        base = torch.full_like(energy, 1.0)
        health_bonus = 0.5 * (health - 0.5)
        energy_bonus = 0.5 * (energy - 0.5)

        extrinsic_rewards = torch.where(
            dones, torch.zeros_like(energy), base + health_bonus + energy_bonus
        )

        resource_state = torch.maximum(health, energy)
        intrinsic_weights = resource_state

        return extrinsic_rewards, intrinsic_weights
```

**DAC (YAML)**:
```yaml
modifiers:
  resource_crisis:
    variable: worst_resource  # VFS: max(energy, health)
    ranges:
      - {min: 0.0, max: 1.0, multiplier: 1.0}  # Identity (value itself is multiplier)

extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  bar_bonuses:
    - {bar: energy, center: 0.5, scale: 0.5}
    - {bar: health, center: 0.5, scale: 0.5}

intrinsic:
  strategy: rnd
  base_weight: 0.100
  apply_modifiers: [resource_crisis]  # Crisis suppression!

shaping: []
```

---

## Dependencies

### Upstream (Required)

- **TASK-002C (VFS)**: DAC references VFS variables for complex features
- **TASK-004A (Compiler)**: DAC validation integrates into compiler Stage 3/4

### Downstream (Blocked)

None - DAC is a leaf feature.

---

## Implementation Notes

### Design Decisions

1. **Why separate `drive_as_code.yaml` instead of embedding in `training.yaml`?**
   - Reward functions are universe properties (like bars, affordances)
   - Separates "what drives behavior" from "how training works"
   - Allows reusing DAC configs across training configs

2. **Why delete legacy reward classes instead of keeping them for compatibility?**
   - Pre-release software with zero users
   - Fix-on-fail approach - breaking changes acceptable
   - Removes maintenance burden of dual codepaths
   - Forces clean migration of all configs

3. **Why 9 extrinsic strategies instead of just fixing the multiplicative one?**
   - Pedagogical value: Students compare strategies to understand reward shaping
   - Research flexibility: Different domains need different reward structures
   - Composability: Hybrid strategy combines multiple approaches

4. **Why modifiers instead of hardcoded crisis suppression?**
   - Generality: Handles arbitrary bar semantics (high=bad vs low=bad)
   - Composability: Same modifier reusable for extrinsic + intrinsic
   - Temporal dynamics: Can decay exploration over episode

5. **Why 11 shaping bonus types?**
   - Comprehensive library covers common behavioral incentives
   - Students learn reward shaping taxonomy
   - VFS variable type allows arbitrary custom logic

### Alternatives Considered

**Alternative 1**: Embed reward logic in VFS
- **Pros**: No new system, reuse VFS expressions
- **Cons**: VFS is for observations, not rewards; confuses concerns

**Alternative 2**: Python plugin system for reward functions
- **Pros**: Maximum flexibility
- **Cons**: Requires code changes, no provenance, loses pedagogical value

**Alternative 3**: Simplified DAC with only extrinsic + intrinsic (no modifiers/shaping)
- **Pros**: Simpler to implement
- **Cons**: Can't fix "Low Energy Delirium" bug, limited research utility

**Alternative 4**: Keep legacy reward classes for backward compatibility
- **Pros**: No breaking changes
- **Cons**: Dual codepaths, maintenance burden, confusing for students, not aligned with pre-release status

**Decision**: Full DAC with modifiers/shaping, delete legacy code. Pre-release software allows breaking changes.

---

## Glossary

**DAC (Drive As Code)**: Declarative reward function specification system that extracts reward logic from Python into YAML configs.

**Modifier**: Range-based multiplier that adjusts reward components based on context (e.g., crisis suppression).

**Extrinsic Strategy**: Method for computing survival/performance rewards from agent state (9 types).

**Intrinsic Strategy**: Method for computing curiosity/exploration bonuses (RND, ICM, count-based, adaptive).

**Shaping Bonus**: Behavioral incentive added to rewards (approach, completion, diversity, etc.).

**Composition**: Process of combining extrinsic, intrinsic, and shaping into total reward.

**Drive Hash**: Content hash of DAC config for checkpoint provenance.

**Crisis Suppression**: Reducing exploration drive when resources are critically low (via modifiers).

**VFS Integration**: Referencing VFS variables in DAC specs for complex derived features.

---

## References

- **Low Energy Delirium Bug**: `docs/teachable_moments/low_energy_delerium.md`
- **DAC Definition Library**: `docs/tasks/TASK-004C-DAC-DEFINITION.md`
- **VFS Design**: `docs/plans/2025-11-06-variables-and-features-system.md`
- **UAC Architecture**: `docs/architecture/COMPILER_ARCHITECTURE.md`
- **Current Reward Strategies**: `src/townlet/environment/reward_strategy.py`

---

**End of TASK-004C**
