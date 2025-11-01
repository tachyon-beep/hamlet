# ACTION #1: Configurable Cascade Engine - Design Document

**Date**: November 1, 2025  
**Status**: Planning Phase  
**Estimated Duration**: 2-3 weeks  
**Moonshot Priority**: 🚀 CRITICAL (Module B prerequisite)

---

## Executive Summary

**Goal**: Replace hardcoded meter cascades with data-driven YAML configuration system.

**Design Approach**: Use SOFTWARE_DEFINED_WORLD.md as **structural template** (4-file organization), but implement our **validated mathematics** (30% thresholds, gradient penalties, 275 passing tests).

**Why This Matters**: 
- **Moonshot Prerequisite**: Module B (World Model) needs to learn physics from config, not code
- **Pedagogical**: Students can experiment with different cascade strengths
- **Level 3 Ready**: Easy to add 5 new meters without touching code
- **Zero Risk**: Keep proven math, gain SDW structure

**Current State**: All cascade logic hardcoded in `meter_dynamics.py` (270 lines)  
**Target State**: SDW-structured YAML files with our validated gradient cascade math

---

## Current System Analysis

### Meter Architecture (8 meters)

**PRIMARY (Death Conditions):**
- `health` [6]: Are you alive?
- `energy` [0]: Can you move?

**SECONDARY (Strong → Primary):**
- `satiation` [2] → health AND energy (FUNDAMENTAL - affects both!)
- `fitness` [7] → health (via depletion multiplier: 0.5x to 3.0x)
- `mood` [4] → energy

**TERTIARY (Quality of Life):**
- `hygiene` [1] → strong to secondary + weak to primary
- `social` [5] → strong to secondary + weak to primary

**RESOURCE:**
- `money` [3]: Enables affordances (no cascades)

### Current Cascade Effects (13 total)

#### Base Depletions (8 effects)
```python
energy: -0.005 (0.5% per step)
hygiene: -0.003 (0.3%)
satiation: -0.004 (0.4%)
money: 0.0 (no depletion)
mood: -0.001 (0.1%)
social: -0.006 (0.6%)
health: fitness-modulated (0.0005 to 0.003)
fitness: -0.002 (0.2%)
```

#### Secondary → Primary (3 effects)
```python
LOW satiation (<0.3) → health: -0.004 * deficit
LOW satiation (<0.3) → energy: -0.005 * deficit
LOW mood (<0.3) → energy: -0.005 * deficit
```

#### Tertiary → Secondary (4 effects)
```python
LOW hygiene (<0.3) → satiation: -0.002 * deficit
LOW hygiene (<0.3) → fitness: -0.002 * deficit
LOW hygiene (<0.3) → mood: -0.003 * deficit
LOW social (<0.3) → mood: -0.004 * deficit
```

#### Tertiary → Primary (3 effects)
```python
LOW hygiene (<0.3) → health: -0.0005 * deficit
LOW hygiene (<0.3) → energy: -0.0005 * deficit
LOW social (<0.3) → energy: -0.0008 * deficit
```

#### Special: Fitness Modulation (1 effect)
```python
fitness level → health depletion multiplier: 0.5x to 3.0x
```

---

## YAML Configuration Schema

### Design Principles

1. **SDW Structure**: 4-file organization (bars, cascades, affordances, cues) for clean separation
2. **Our Mathematics**: Gradient penalties with 30% thresholds (validated with 275 tests)
3. **Human-Readable**: Students should understand cascade relationships from reading YAML
4. **Type-Safe**: Pydantic validation catches errors at load time
5. **Pedagogically Rich**: Comments explain WHY each cascade exists

### File Structure (SDW-Compliant)

```
configs/
├── bars.yaml         # Meter definitions & base depletion rates
├── cascades.yaml     # Threshold-based cascade effects
├── affordances.yaml  # Action definitions (already exists)
└── cues.yaml         # Social tells (for Module C - future)
```

### Schema 1: `configs/bars.yaml` (Meter Definitions)

```yaml
# bars.yaml - Meter Definitions & Base Depletion Rates
# This defines the meters that exist in the environment and their passive decay.
#
# Hierarchy (following SDW structure):
# - PIVOTAL: Death conditions (health, energy)
# - PRIMARY: Major needs that strongly affect pivotal (satiation, fitness, mood)  
# - SECONDARY: Quality of life needs (hygiene, social)
# - RESOURCE: Enables actions but doesn't cascade (money)

version: "1.0"
description: "Default meter configuration for Level 1.5 (8 meters)"

bars:
  # PIVOTAL: Death if <= 0
  - name: "energy"
    index: 0
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.005  # 0.5% per step
    description: "Can you move? Death if depleted."
    
  - name: "health"
    index: 6
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.001  # Baseline (modulated by fitness)
    description: "Are you alive? Death if depleted."
  
  # PRIMARY: Strong effects on pivotal meters
  - name: "satiation"
    index: 2
    tier: "primary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.004  # 0.4% per step
    description: "Hunger level. Affects BOTH pivotal meters when low."
    
  - name: "fitness"
    index: 7
    tier: "primary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.002  # 0.2% per step
    description: "Physical fitness. Modulates health depletion rate."
    
  - name: "mood"
    index: 4
    tier: "primary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.001  # 0.1% per step
    description: "Mental well-being. Affects energy when low."
  
  # SECONDARY: Quality of life (affect primary meters)
  - name: "hygiene"
    index: 1
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.003  # 0.3% per step
    description: "Cleanliness. Affects appetite, fitness, mood."
    
  - name: "social"
    index: 5
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.006  # 0.6% per step (decays fastest!)
    description: "Social connection. Strongly affects mood."
  
  # RESOURCE: Enables affordances
  - name: "money"
    index: 3
    tier: "resource"
    range: [0.0, 1.0]  # $0-$100 normalized
    initial: 0.5       # Start with $50
    base_depletion: 0.0  # No passive decay
    description: "Currency for purchasing affordances."

# Terminal conditions (death)
terminal_conditions:
  - meter: "health"
    operator: "<="
    value: 0.0
    description: "Agent dies if health depletes"
    
  - meter: "energy"
    operator: "<="
    value: 0.0
    description: "Agent dies if energy depletes"

```

### Schema 2: `configs/cascades.yaml` (Cascade Effects)

```yaml
# cascades.yaml - Threshold-Based Cascade Effects
# This defines how meters affect each other when they fall below thresholds.
#
# Math Approach: GRADIENT PENALTIES (validated with 275 tests)
# When source < threshold:
#   deficit = threshold - source_value
#   penalty = strength * deficit
#   target -= penalty
#
# This gives smooth gradient (unlike multipliers which are binary)

version: "1.0"
description: "Default cascade configuration (gradient penalties, 30% thresholds)"

# Special modulation: Fitness modulates health depletion rate
modulations:
  - name: "fitness_health_modulation"
    description: "Low fitness increases health depletion (gradient approach)"
    source: "fitness"
    target: "health"
    type: "depletion_multiplier"
    
    # Multiplier = base + (range * (1.0 - fitness))
    # fitness=100% → 0.5x depletion (healthy!)
    # fitness=0% → 3.0x depletion (unhealthy!)
    base_multiplier: 0.5
    range: 2.5

# Threshold-based cascades (our validated math)
cascades:
  # SECONDARY → PRIMARY (aggressive effects)
  - name: "satiation_to_health"
    description: "Starvation makes you sick (fundamental need)"
    category: "secondary_to_primary"
    source: "satiation"
    target: "health"
    threshold: 0.3
    strength: 0.004  # penalty = strength * deficit
    
  - name: "satiation_to_energy"
    description: "Hunger makes you exhausted (fundamental need)"
    category: "secondary_to_primary"
    source: "satiation"
    target: "energy"
    threshold: 0.3
    strength: 0.005
  
  - name: "mood_to_energy"
    description: "Depression leads to exhaustion"
    category: "secondary_to_primary"
    source: "mood"
    target: "energy"
    threshold: 0.3
    strength: 0.005
  
  # TERTIARY → SECONDARY (aggressive effects)
  - name: "hygiene_to_satiation"
    description: "Being dirty reduces appetite"
    category: "tertiary_to_secondary"
    source: "hygiene"
    target: "satiation"
    threshold: 0.3
    strength: 0.002
  
  - name: "hygiene_to_fitness"
    description: "Being dirty makes exercise harder"
    category: "tertiary_to_secondary"
    source: "hygiene"
    target: "fitness"
    threshold: 0.3
    strength: 0.002
  
  - name: "hygiene_to_mood"
    description: "Being dirty makes you feel bad"
    category: "tertiary_to_secondary"
    source: "hygiene"
    target: "mood"
    threshold: 0.3
    strength: 0.003
  
  - name: "social_to_mood"
    description: "Loneliness causes depression"
    category: "tertiary_to_secondary"
    source: "social"
    target: "mood"
    threshold: 0.3
    strength: 0.004  # Stronger than hygiene
  
  # TERTIARY → PRIMARY (weak direct effects)
  - name: "hygiene_to_health"
    description: "Poor hygiene weakly affects health directly"
    category: "tertiary_to_primary_weak"
    source: "hygiene"
    target: "health"
    threshold: 0.3
    strength: 0.0005  # Weak
  
  - name: "hygiene_to_energy"
    description: "Poor hygiene weakly affects energy directly"
    category: "tertiary_to_primary_weak"
    source: "hygiene"
    target: "energy"
    threshold: 0.3
    strength: 0.0005  # Weak
  
  - name: "social_to_energy"
    description: "Loneliness weakly affects energy directly"
    category: "tertiary_to_primary_weak"
    source: "social"
    target: "energy"
    threshold: 0.3
    strength: 0.001  # Weak (but stronger than hygiene)
```

### Key Design Decisions

**1. SDW Structure + Our Math = Best of Both Worlds**

- **From SDW**: 4-file organization, clean separation of concerns, Module B ready
- **From Us**: Gradient penalties (smooth), 30% thresholds (tested with 275 tests)
- **Result**: Zero risk (no behavioral change) + SDW compliance

**2. Why Gradient Penalties Over Multipliers?**

```python
# Our approach (gradient):
if satiation < 0.3:
    deficit = 0.3 - satiation
    health -= 0.004 * deficit  # Smooth gradient

# SDW multipliers (alternative):
if satiation < 0.3:
    health_depletion *= 1.5  # Binary threshold effect
```

- **Gradient**: Smooth gameplay, proportional consequences
- **Multipliers**: Simpler model, easier to understand
- **Both valid**: Can add multiplier support in Week 2 for experiments!

**3. Why 30% Threshold?**

- Validated with 275 passing tests
- Gives agents time to respond before cascading
- More forgiving than SDW's 20% (better for learning)
- Can offer 20% as "hard mode" config

**4. File Structure Philosophy**

- `bars.yaml`: "What meters exist?" (ontology)
- `cascades.yaml`: "How do they interact?" (physics)
- `affordances.yaml`: "What can agents do?" (actions)
- `cues.yaml`: "How to communicate?" (Module C future)

Clear separation enables modular development!# Terminal conditions (death triggers)
terminal_conditions:
  - meter: "health"
    condition: "<= 0.0"
    description: "Death if health reaches zero"
  
  - meter: "energy"
    condition: "<= 0.0"
    description: "Death if energy reaches zero"
```

---

## Implementation Plan

**Strategy**: SDW structure (4 files) + Our validated math (gradient penalties, 30% thresholds)  
**Risk**: ZERO - Keep proven behavior, gain clean architecture  
**Timeline**: 2-3 weeks (same as original plan, cleaner result)

### Phase 1: Schema & Validation (Week 1, Days 1-2)

**Tasks:**
1. Create `configs/bars.yaml` - meter definitions (SDW format)
2. Create `configs/cascades.yaml` - cascade effects (our gradient math)
3. Create `src/townlet/environment/cascade_config.py` - configuration loader
4. Define Pydantic models for type-safe YAML validation
5. Write validation tests

**Deliverables:**
- SDW-compliant 4-file structure
- Configuration loader with Pydantic validation
- Configs matching current behavior exactly
- Tests for config loading and validation

### Phase 2: CascadeEngine Implementation (Week 1-2, Days 3-7)

**Tasks:**
1. Create `src/townlet/environment/cascade_engine.py`
2. Implement `CascadeEngine` class that reads config
3. Replace hardcoded logic in `meter_dynamics.py` with engine calls
4. Preserve exact current behavior (zero behavioral changes)

**Key Classes:**

```python
class CascadeEngine:
    """Data-driven cascade engine that applies meter relationships."""
    
    def __init__(self, config: CascadeConfig):
        """Load config and prepare cascade application."""
        self.config = config
        self.meter_indices = self._build_meter_index_map()
        
    def apply_base_depletions(self, meters: torch.Tensor) -> torch.Tensor:
        """Apply base depletion rates from config."""
        
    def apply_modulations(self, meters: torch.Tensor) -> torch.Tensor:
        """Apply special modulation effects (e.g., fitness → health multiplier)."""
        
    def apply_threshold_cascades(
        self, 
        meters: torch.Tensor, 
        categories: List[str]
    ) -> torch.Tensor:
        """Apply threshold-based cascades by category."""
        # categories: ["secondary_to_primary", "tertiary_to_secondary", etc.]
        
    def check_terminal_conditions(
        self, 
        meters: torch.Tensor, 
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Check terminal conditions from config."""
```

**Integration with MeterDynamics:**

```python
class MeterDynamics:
    def __init__(self, num_agents: int, device: torch.device, cascade_config_path: str):
        self.num_agents = num_agents
        self.device = device
        
        # Load and initialize cascade engine
        config = load_cascade_config(cascade_config_path)
        self.engine = CascadeEngine(config, device)
    
    def deplete_meters(self, meters: torch.Tensor) -> torch.Tensor:
        # Apply base depletions from config
        meters = self.engine.apply_base_depletions(meters)
        # Apply modulations (fitness → health)
        meters = self.engine.apply_modulations(meters)
        return meters
    
    def apply_secondary_to_primary_effects(self, meters: torch.Tensor) -> torch.Tensor:
        return self.engine.apply_threshold_cascades(
            meters, 
            categories=["secondary_to_primary"]
        )
    
    # ... similar for other methods
```

**Deliverables:**
- `CascadeEngine` class fully implemented
- `MeterDynamics` refactored to use engine
- All existing tests still passing (zero behavior change)

### Phase 3: Testing & Validation (Week 2-3, Days 8-12)

**Tasks:**
1. **Characterization Tests**: Verify config produces same behavior as hardcoded
2. **Alternative Configs**: Create test configs with different cascade strengths
3. **Pedagogical Tests**: Test "what if satiation was 2x stronger?"
4. **Level 3 Preview**: Create config with 5 new meters (stress, etc.)

**Test Configs to Create:**

1. `configs/cascades/default.yaml` - Current behavior (baseline)
2. `configs/cascades/weak_cascades.yaml` - 50% strength (easier for students)
3. `configs/cascades/strong_cascades.yaml` - 150% strength (harder challenge)
4. `configs/cascades/no_cascades.yaml` - Only base depletions (debugging)
5. `configs/cascades/level_3_preview.yaml` - 13 meters for future (stress, knowledge, etc.)

**Test Structure:**

```python
class TestCascadeEngine:
    def test_default_config_matches_legacy_behavior(self):
        """Config-driven system produces identical results to hardcoded."""
        
    def test_alternative_cascade_strengths(self):
        """Can modify cascade strengths via config."""
        
    def test_pedagogical_experiments(self):
        """Students can experiment with 'what if' scenarios."""
        
    def test_level_3_meter_support(self):
        """Can load config with 13 meters (preview Level 3)."""
```

**Deliverables:**
- 20+ tests for cascade engine
- 5 alternative configs for testing/pedagogy
- Documentation of pedagogical use cases
- Level 3 preview config (not yet used, but validated)

### Phase 4: Documentation & Polish (Week 3, Days 13-15)

**Tasks:**
1. Update AGENTS.md with new cascade engine architecture
2. Create CASCADE_CONFIG_GUIDE.md for students
3. Add inline comments to default.yaml explaining each cascade
4. Update CLEANUP_ROADMAP.md to mark ACTION #1 complete

**Documentation Structure:**

```markdown
# CASCADE_CONFIG_GUIDE.md

## For Students: How to Experiment with Cascades

### Example 1: Make hunger more important
Change satiation cascade strengths from 0.004/0.005 to 0.008/0.010

### Example 2: Remove social effects
Set all social cascade strengths to 0.0

### Example 3: Add a new meter
Define a new meter (e.g., "stress") and its cascades
```

**Deliverables:**
- Complete documentation
- Student-friendly experimentation guide
- Updated project docs

---

## Success Criteria

### Must Have (Required)
- ✅ Config-driven system produces **identical behavior** to hardcoded (zero regression)
- ✅ All 275 existing tests pass without modification
- ✅ New tests achieve 100% coverage on `CascadeEngine`
- ✅ Can define new meter in YAML in <5 minutes
- ✅ Can modify cascade strength in YAML in <1 minute

### Should Have (Important)
- ✅ 5 alternative configs for pedagogical experiments
- ✅ Level 3 preview config with 13 meters
- ✅ Student-friendly documentation
- ✅ Performance: <1% overhead vs hardcoded

### Nice to Have (Bonus)
- ✅ Config validation catches common errors with helpful messages
- ✅ Visualization tool to see cascade graph from config
- ✅ Hot-reload config during training (advanced)

---

## Risk Assessment

### Risk: Performance Degradation

**Likelihood**: Low  
**Impact**: Medium (200 episodes/hour → 180?)  
**Mitigation**:
- Pre-compute indices/masks during initialization
- Keep hot path tight (avoid YAML parsing in loop)
- Benchmark: config-driven should be within 5% of hardcoded

### Risk: Behavioral Divergence

**Likelihood**: Medium (easy to introduce subtle bugs)  
**Impact**: High (breaks existing training)  
**Mitigation**:
- Extensive characterization tests before refactoring
- Test against exact hardcoded outputs (bit-level comparison)
- Keep legacy code temporarily for A/B testing

### Risk: Config Complexity

**Likelihood**: Low (YAML is simple)  
**Impact**: Medium (confusion for students)  
**Mitigation**:
- Extensive comments in default.yaml
- Student guide with examples
- Start simple (can add complexity later)

---

## Moonshot Connection

**Why This Matters for v2.0:**

1. **Module B (World Model)** needs to learn physics:
   - Currently: "When satiation < 0.3, health decreases by 0.004 * deficit"
   - Future: Module B predicts this from observation history
   - **Requirement**: Physics must be in config so Module B can learn the rules

2. **Enables World Model Training**:
   - Config becomes "ground truth" for what Module B should learn
   - Can test: "Does Module B predict cascade effects accurately?"
   - Can vary physics and test generalization

3. **Hierarchical Abstraction**:
   - v1.0: Q-network memorizes `Q(satiation=0.2, health=0.8) = do_X`
   - v2.0: Module B learns `low_satiation → health_decline` as a rule
   - **This refactoring enables that abstraction!**

**From AGENT_MOONSHOT.md:**
> "The hardcoded logic in `vectorized_env.py` (Actions #1, #12) must be refactored to be configuration-driven (e.g., YAML) so the World Model can learn these rules."

✅ **This is THE blocker removal for Module B!**

---

## Timeline Summary

**Week 1**:
- Days 1-2: Schema & validation
- Days 3-5: CascadeEngine implementation
- Days 6-7: MeterDynamics integration

**Week 2**:
- Days 8-10: Testing & validation
- Days 11-12: Alternative configs

**Week 3**:
- Days 13-15: Documentation & polish
- Buffer: Handle unexpected issues

**Total**: 2-3 weeks (15 days estimated, 21 days budgeted)

---

## Next Steps

1. **Approve this design** - Is the YAML schema clear and extensible?
2. **Start Phase 1** - Create cascade_config.py with Pydantic models
3. **Create default.yaml** - Transcribe current hardcoded values
4. **Validate** - Tests confirm config loads correctly

**Ready to start?** Let's build the foundation for Module B! 🚀
