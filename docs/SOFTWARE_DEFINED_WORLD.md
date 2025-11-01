Here is the full "Software Defined World" specification, consolidating our decisions into a single design document for your architect.

-----

### **Hamlet v2.0: "Software Defined World" Specification (v1.0)**

**Document Date:** November 1, 2025
**Status:** **APPROVED FOR IMPLEMENTATION**
**Owner:** Principal Technical Advisor (AI)

### 1. 🚀 Purpose & Core Philosophy

This document defines the data-driven configuration files that will **replace all hardcoded environment logic** (e.g., "god objects," `elif` blocks, hardcoded physics). This "Software Defined World" (SDW) is the foundational prerequisite for the v2.0 "Smart Collection" architecture.

This SDW acts as the **"grammar textbook"** for the agent's "grammar engine":

* **Module B (World Model)** will be pre-trained to learn the rules defined in `bars.yaml`, `cascades.yaml`, and `affordances.yaml`.
* **Module C (Social Model)** will be pre-trained to learn the "dictionary" defined in `cues.yaml`.

This refactor is the *mandatory first step* of the v2.0 migration path.

### 2. 🧠 The "Bar Hierarchy" (Core Design)

The environment's physics are built on a 4-tier hierarchy of agent "bars" (meters). This non-linear, cascading design is the primary source of long-term strategic challenge for the agent.

* **Tier 0: Pivotal Bars (Death)**
  * `energy`, `health`
  * If either hits 0, the agent dies (`set_done: true`).
* **Tier 1: Primary Bars (Gates)**
  * `money`
  * If 0, does not cause death, but "gates" access to most recovery affordances.
* **Tier 2: Secondary Bars (Primary Penalties)**
  * `satiation`, `mood`, `fitness`
  * When low, these bars apply **depletion rate multipliers** to the Pivotal Bars (Energy & Health).
* **Tier 3: Tertiary Bars (Secondary Penalties)**
  * `hygiene`, `social`, `stimulation`
  * When low, these bars apply depletion rate multipliers to the Secondary Bars, creating a slow-burn, cascading "death spiral" if left unmanaged.

-----

### 3. 📄 Specification File 1: `bars.yaml`

**Purpose:** Defines all `M_MAX` bars, their limits, and their default passive depletion rates.
**Replaces:** Hardcoded `self.energy -= 0.1` logic.

**File:** `config/bars.yaml`

```yaml
# Defines all core "bars" or "meters" (M_MAX = 12)

# --- Tier 0: Pivotal ---
- id: "energy"
  initial_value: 100
  min_value: 0
  max_value: 100
  depletion_rate: -0.1 # Base depletion

- id: "health"
  initial_value: 100
  min_value: 0
  max_value: 100
  depletion_rate: -0.05 # Base health depletion (slower)

# --- Tier 1: Primary ---
- id: "money"
  initial_value: 50
  min_value: 0
  max_value: 9999999
  depletion_rate: 0

# --- Tier 2: Secondary ---
- id: "satiation"
  initial_value: 100
  min_value: 0
  max_value: 100
  depletion_rate: -0.2

- id: "mood"
  initial_value: 70
  min_value: 0
  max_value: 100
  depletion_rate: -0.1

- id: "fitness"
  initial_value: 50
  min_value: 0
  max_value: 100
  depletion_rate: 0 # Fitness is passive unless impacted by cascades

# --- Tier 3: Tertiary ---
- id: "hygiene"
  initial_value: 100
  min_value: 0
  max_value: 100
  depletion_rate: -0.5

- id: "social"
  initial_value: 50
  min_value: 0
  max_value: 100
  depletion_rate: -0.2

- id: "stimulation" # The new 'boredom' bar
  initial_value: 50
  min_value: 0
  max_value: 100
  depletion_rate: -0.1
```

-----

### 4. 📄 Specification File 2: `cascades.yaml`

**Purpose:** Defines the complete hierarchical "physics" of how bars interact, including death conditions and multipliers.
**Replaces:** All hardcoded `if self.satiation < 20:` logic.

**File:** `config/cascades.yaml`

```yaml
# Defines the "physics" of how bars interact based on the 4-tier hierarchy.

# --- TIER 0: PIVOTAL (DEATH) CASCADES ---
- id: "death_by_energy"
  condition:
    - { bar: "energy", op: "<=", val: 0 }
  effect:
    - { type: "set_done", value: true, reason: "exhaustion" }

- id: "death_by_health"
  condition:
    - { bar: "health", op: "<=", val: 0 }
  effect:
    - { type: "set_done", value: true, reason: "health_failure" }

# --- TIER 2: SECONDARY CASCADES (Penalty on Primaries) ---
- id: "low_satiation_penalty"
  condition:
    - { bar: "satiation", op: "<", val: 20 }
  effect:
    - { type: "modify_depletion_rate", bar: "health", multiplier: 1.5 }
    - { type: "modify_depletion_rate", bar: "energy", multiplier: 1.5 }

- id: "low_mood_penalty"
  condition:
    - { bar: "mood", op: "<", val: 20 }
  effect:
    - { type: "modify_depletion_rate", bar: "energy", multiplier: 2.0 }

- id: "low_fitness_penalty"
  condition:
    - { bar: "fitness", op: "<", val: 20 }
  effect:
    - { type: "modify_depletion_rate", bar: "health", multiplier: 1.5 }

# --- TIER 3: TERTIARY CASCADES (Penalty on Secondaries) ---
- id: "low_hygiene_penalty"
  condition:
    - { bar: "hygiene", op: "<", val: 20 }
  effect:
    - { type: "modify_depletion_rate", bar: "fitness", multiplier: 1.2 }
    - { type: "modify_depletion_rate", bar: "mood", multiplier: 1.2 }
    - { type: "modify_depletion_rate", bar: "satiation", multiplier: 1.2 }

- id: "low_social_penalty"
  condition:
    - { bar: "social", op: "<", val: 20 }
  effect:
    - { type: "modify_depletion_rate", bar: "mood", multiplier: 1.5 }
    
- id: "low_stimulation_penalty"
  condition:
    - { bar: "stimulation", op: "<", val: 20 }
  effect:
    - { type: "modify_depletion_rate", bar: "fitness", multiplier: 1.5 }
```

-----

### 5. 📄 Specification File 3: `affordances.yaml`

**Purpose:** Defines all `A_MAX` agent-world interactions, including costs, effects, and interaction types.
**Replaces:** The giant, hardcoded `elif` block for affordance interactions.

**File:** `config/affordances.yaml`

```yaml
# Defines all "affordances" (A_MAX) and their effects

# 1. Simple, instant-effect affordance (The "Fridge")
- id: "fridge"
  interaction_type: "instant"
  costs:
    - { bar: "money", change: -5 }
  effects:
    - { bar: "satiation", change: +40 }

# 2. Multi-tick interaction (The "Bed")
- id: "bed"
  interaction_type: "multi_tick"
  duration_ticks: 4
  effects_per_tick:
    - { bar: "energy", change: +25 } # 25% per turn
  completion_bonus:
    - { bar: "energy", change: +25 } # 25% bonus for finishing

# 3. Simple cost/gain affordance (The "Bar")
- id: "bar"
  interaction_type: "instant"
  costs:
    - { bar: "money", change: -20 }
  effects:
    - { bar: "social", change: +30 }
    - { bar: "mood", change: +15 }
    - { bar: "stimulation", change: +20 } # Fights boredom

# 4. Special effect affordance (The "Home Phone")
- id: "home_phone"
  interaction_type: "instant"
  costs:
    - { bar: "money", change: -1000 }
  effects:
    # 'teleport' is a special effect_type the environment 
    # must be coded to handle (set agent.pos = hospital.pos)
    - { effect_type: "teleport", destination: "hospital" } 

# 5. A job affordance (The "Office")
- id: "office_job"
  interaction_type: "multi_tick"
  duration_ticks: 8 # A full "workday"
  effects_per_tick:
    - { bar: "money", change: +10 }
    - { bar:S: "energy", change: -5 }
    - { bar: "stimulation", change: -2 } # Job is boring
  completion_bonus:
    - { bar: "money", change: +40 } # "End of day bonus"
```

-----

### 6. 📄 Specification File 4: `cues.yaml`

**Purpose:** Defines the "social" layer, linking internal bar states to the `public_cues` (tells) that **Module C (Social Model)** will learn from.
**Replaces:** Any hardcoded logic for social "tells."

**File:** `config/cues.yaml`

```yaml
# Defines the 'public_cues' (C_MAX) generated from internal bar states.
# This is the "dictionary" for the Module C "detective".

# 1. Simple Cue (from a Tertiary bar)
- cue_id: "looks_dirty"
  conditions:
    - { bar: "hygiene", op: "<", val: 10 }

# 2. Simple Cue (from a Secondary bar)
- cue_id: "looks_sad"
  conditions:
    - { bar: "mood", op: "<", val: 20 }

# 3. Simple Cue (from a Pivotal bar)
- cue_id: "looks_tired"
  conditions:
    - { bar: "energy", op: "<", val: 20 }

# 4. Compound "AND" Cue (The "Shambling Sob")
- cue_id: "shambling_sob"
  condition_logic: "all_of" # Default, but explicit
  conditions:
    - { bar: "energy", op: "<", val: 20 } # looks_tired
    - { bar: "mood", op: "<", val: 20 } # looks_sad

# 5. Compound "OR" Cue
- cue_id: "looks_unwell"
  condition_logic: "any_of" # "OR" logic
  conditions:
    - { bar: "health", op: "<", val: 30 }
    - { bar: "satiation", op: "<", val: 10 }
```

### 7. 🏗️ Implementation Strategy

#### Phase 1: Configuration Layer (✅ DONE - ACTION #1)

**Goal**: Make meter cascades config-driven

**Deliverables**:
* ✅ `bars.yaml` - Meter definitions with base depletion rates
* ✅ `cascades.yaml` - Cascade effects (modulations + threshold cascades)
* ✅ Type-safe loader with Pydantic validation
* ✅ CascadeEngine that reads and applies config
* ✅ 100% equivalence with hardcoded logic (all tests passing)

**Status**: COMPLETE (Nov 1, 2025)

#### Phase 2: Affordance Layer (🎯 NEXT - ACTION #12)

**Goal**: Make affordance effects config-driven

**Deliverables**:
* `affordances.yaml` - All affordance definitions with effects
* Type-safe AffordanceConfig loader
* AffordanceEngine that processes interactions
* Remove 200+ line elif blocks from environment
* Teaching examples (weak/strong/creative affordances)

**Estimated**: 1-2 weeks

#### Phase 3: Social Layer (📋 FUTURE)

**Goal**: Make social cues config-driven

**Deliverables**:
* `cues.yaml` - Public cue definitions and mappings
* CueGenerator that emits observable signals
* Enable Module C (Social Model) to learn from cues
* Multi-agent opponent modeling foundation

**Estimated**: 2-3 weeks (after Level 4 multi-agent)

### 8. 🎓 Design Principles

#### Principle 1: "Grammar Engine, Not Flashcard Memorizer"

**Problem**: Current v1.0 DQN memorizes `Q(s,a)` values - doesn't understand *why* actions work

**Solution**: Module B learns the *rules* (physics) from config, then *reasons* about consequences

**Example**:
* **Flashcard**: "When energy=20%, go to bed" (memorized)
* **Grammar**: "Bed restores energy by 25% per tick over 4 ticks" (understood rule)
* **Reasoning**: "If energy=30%, I need 2 ticks to recover to safe threshold"

#### Principle 2: "Configuration as Ground Truth"

**Problem**: Code and documentation drift, become inconsistent

**Solution**: YAML is the single source of truth - code *implements* config, never hardcodes

**Example**:

```python
# ❌ BAD (Hardcoded)
if agent.satiation < 0.2:
    agent.health -= 0.004 * ((0.3 - agent.satiation) / 0.3)

# ✅ GOOD (Config-Driven)
cascades = load_cascades_config()
meters = cascade_engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])
```

#### Principle 3: "Interesting Failures Are Features"

**Problem**: Over-tuning creates boring, deterministic gameplay

**Solution**: Expose cascade strengths in config, let students discover "too weak" and "too strong"

**Example**:
* `cascades_weak.yaml` (50% strength) - Agent survives easily, learns affordances
* `cascades.yaml` (100% strength) - Balanced challenge, strategic planning required
* `cascades_strong.yaml` (150% strength) - Death spirals frequent, must prioritize perfectly

**Teaching Value**: Students learn system design trade-offs through experimentation

#### Principle 4: "Zero Behavioral Change (Until We Choose)"

**Problem**: Refactoring introduces subtle bugs that break training

**Solution**: Config replicates exact hardcoded behavior, validated by equivalence tests

**Example**:

```python
def test_equivalence_with_meter_dynamics_low_satiation():
    """CascadeEngine produces IDENTICAL results to hardcoded logic"""
    # Create both systems
    hardcoded = MeterDynamics(use_cascade_engine=False)
    config_driven = MeterDynamics(use_cascade_engine=True)
    
    # Same input
    meters = create_low_satiation_state()
    
    # Same output (within floating point tolerance)
    assert torch.allclose(hardcoded.deplete(meters), config_driven.deplete(meters))
```

### 9. 🧪 Validation & Testing Strategy

#### Equivalence Testing (Critical!)

Every config-driven system MUST pass equivalence tests against hardcoded baseline:

```python
@pytest.mark.parametrize("scenario", [
    "healthy_agent",
    "low_satiation",
    "low_mood",
    "low_hygiene",
    "cascade_combinations",
])
def test_config_matches_hardcoded(scenario):
    """Config-driven produces identical results to hardcoded logic"""
    legacy = create_legacy_system()
    config = create_config_system()
    
    state = load_scenario(scenario)
    
    legacy_result = legacy.step(state)
    config_result = config.step(state)
    
    assert results_match(legacy_result, config_result, tolerance=1e-6)
```

#### Schema Validation (Catch Errors Early)

Use Pydantic for runtime validation:

```python
class CascadeConfig(BaseModel):
    """Threshold-based cascade with gradient penalties"""
    name: str
    source: str
    source_index: int = Field(ge=0, le=7)  # Must be valid meter index
    target: str
    target_index: int = Field(ge=0, le=7)
    threshold: float = Field(gt=0.0, le=1.0)  # Normalized [0,1]
    strength: float = Field(gt=0.0)  # Positive penalty strength
```

**Benefits**:
* Type errors caught at config load time (not runtime)
* Clear error messages: "threshold must be in (0.0, 1.0], got 1.5"
* Auto-generated documentation from field descriptions

#### Teaching Example Validation

Create validation scripts for pedagogical configs:

```bash
$ python scripts/validate_cascade_configs.py

✅ cascades.yaml is VALID (100% strength, balanced)
✅ cascades_weak.yaml is VALID (50% strength, easy mode)
✅ cascades_strong.yaml is VALID (150% strength, hard mode)

🎉 All teaching configs are valid!
```

### 10. 📚 Module B & C Training Strategy

#### Module B (World Model) Pre-Training

**Input**: Config files (`bars.yaml`, `cascades.yaml`, `affordances.yaml`)

**Training Task**: Predict meter changes given actions and current state

```python
# Pseudo-code for Module B training
def train_world_model():
    config = load_environment_config()
    
    # Generate synthetic training data
    for episode in range(10000):
        state = sample_random_state()
        action = sample_random_action()
        
        # Ground truth from config
        next_state_true = config_engine.predict_next_state(state, action)
        
        # Model prediction
        next_state_pred = world_model.forward(state, action)
        
        # Train to match config physics
        loss = mse_loss(next_state_pred, next_state_true)
        loss.backward()
```

**Success Metric**: Model predicts meter changes with <1% error vs config engine

#### Module C (Social Model) Pre-Training

**Input**: `cues.yaml` (mapping from hidden states to public cues)

**Training Task**: Infer hidden bar states from observable cues

```python
# Pseudo-code for Module C training
def train_social_model():
    cues_config = load_cues_config()
    
    # Generate opponent observation data
    for episode in range(10000):
        # Sample hidden opponent state
        opponent_bars = sample_random_bars()
        
        # Generate public cues (ground truth from config)
        public_cues = cues_config.generate_cues(opponent_bars)
        
        # Model inference
        inferred_bars = social_model.infer_from_cues(public_cues)
        
        # Train to recover hidden state
        loss = mse_loss(inferred_bars, opponent_bars)
        loss.backward()
```

**Success Metric**: Model infers opponent bar states with <10% error from cues

### 11. 🚀 Migration Path (v1.0 → v2.0)

#### v1.0 (Current - Monolithic DQN)

```
Observation → DQN(obs) → Q(s,a) → Action
```

**Limitations**:
* Memorizes state-action values
* No understanding of physics
* Can't plan ahead
* No opponent modeling

#### v1.5 (Hybrid - Module A + DQN)

```
Observation → Module A (Perception) → BeliefDistribution → DQN(belief) → Q(belief,a) → Action
```

**Improvements**:
* POMDP solved via belief states
* Keeps proven DQN core
* Validates Module A in isolation

#### v1.7 (Module B Added)

```
Observation → Module A → Belief → Module B (World Model) → ImaginedFutures
                                  ↓
                           DQN(belief + futures) → Q → Action
```

**Improvements**:
* Can imagine consequences
* Plans 1-2 steps ahead
* Still uses DQN for action selection

#### v2.0 (Full Stack - All Modules)

```
Observation → Module A → Belief
              ↓
Public Cues → Module C (Social Model) → OpponentBeliefs
              ↓                          ↓
              Module B (World Model) → ImaginedFutures (self + opponent)
              ↓
              Module D (Hierarchical Policy) → Goal → PrimitiveAction
```

**Improvements**:
* Full opponent modeling
* Multi-step strategic planning
* Hierarchical reasoning (zone → transport → affordance)
* No Q-values - pure model-based RL

### 12. 📋 Implementation Checklist

#### For Each Config File

* [ ] Define YAML schema (structure and fields)
* [ ] Create Pydantic validation models
* [ ] Write type-safe loader function
* [ ] Build processing engine (reads config, applies effects)
* [ ] Write equivalence tests (config vs hardcoded)
* [ ] Create teaching examples (weak/strong variants)
* [ ] Validate all configs load successfully
* [ ] Integrate with environment as default
* [ ] Document config format and examples
* [ ] Mark old hardcoded logic as `LEGACY` (keep for tests)

#### bars.yaml (✅ COMPLETE)

- [x] Schema defined
* [x] Pydantic models (BarConfig, BarsConfig)
* [x] Loader function (load_bars_config)
* [x] CascadeEngine processes base depletions
* [x] Equivalence tests pass
* [x] Teaching examples (same bars for all variants)
* [x] Validation script
* [x] Default in MeterDynamics
* [x] Documentation complete

#### cascades.yaml (✅ COMPLETE)

- [x] Schema defined
* [x] Pydantic models (ModulationConfig, CascadeConfig, CascadesConfig)
* [x] Loader function (load_cascades_config)
* [x] CascadeEngine processes modulations + cascades
* [x] Equivalence tests pass
* [x] Teaching examples (weak 50%, normal 100%, strong 150%)
* [x] Validation script
* [x] Default in MeterDynamics
* [x] Documentation complete

#### affordances.yaml (🎯 NEXT - ACTION #12)

- [ ] Schema defined
* [ ] Pydantic models (AffordanceConfig, AffordancesConfig)
* [ ] Loader function (load_affordances_config)
* [ ] AffordanceEngine processes interactions
* [ ] Equivalence tests pass
* [ ] Teaching examples (creative affordance sets)
* [ ] Validation script
* [ ] Default in Environment
* [ ] Documentation complete

#### cues.yaml (📋 FUTURE)

- [ ] Schema defined (after Level 4 multi-agent)
* [ ] Pydantic models (CueConfig, CuesConfig)
* [ ] Loader function (load_cues_config)
* [ ] CueGenerator emits public signals
* [ ] Equivalence tests pass
* [ ] Teaching examples (subtle vs obvious cues)
* [ ] Validation script
* [ ] Default in Environment
* [ ] Documentation complete

### 13. 🎯 Success Metrics

#### Technical Metrics

* **Equivalence**: Config-driven produces identical results to hardcoded (within 1e-6 tolerance)
* **Test Coverage**: 100% of config-driven code paths tested
* **Performance**: No degradation vs hardcoded baseline (<5% slowdown acceptable)
* **Maintainability**: Config changes don't require code changes

#### Pedagogical Metrics

* **Experimentation**: Students can modify configs without touching code
* **Failure Discovery**: "Too weak" and "too strong" configs teach trade-offs
* **Documentation**: Config files are self-documenting (comments explain intent)
* **Teaching Time**: Reduce onboarding time by 50% (config vs code)

#### Moonshot Metrics

* **Module B Accuracy**: <1% error predicting meter changes from config
* **Module C Accuracy**: <10% error inferring opponent bars from cues
* **Planning Depth**: Agent plans 3-5 steps ahead using world model
* **Win Rate**: v2.0 beats v1.0 DQN in multi-agent competition by 20%+

### 14. 🏁 Conclusion & Next Steps

#### What We've Accomplished

**✅ Phase 1 Complete (ACTION #1)**:
* Meter cascades are now config-driven
* CascadeEngine validated with 44 tests
* Teaching examples (weak/strong) created
* Zero behavioral change confirmed
* **Moonshot Prerequisite #1 achieved**

#### What's Next

**🎯 Phase 2 (ACTION #12) - 1-2 weeks**:
* Move affordance effects to `affordances.yaml`
* Build AffordanceEngine
* Remove 200+ line elif blocks
* Create teaching examples
* **Complete Moonshot Prerequisite #2**

**📋 Phase 3 (Future) - 2-3 weeks**:
* Add `cues.yaml` for social signals
* Enable Module C (Social Model) training
* Foundation for Level 4 multi-agent

#### The Big Picture

This "Software Defined World" is **not just refactoring** - it's the launchpad for v2.0:

1. **Module B (World Model)** can now learn physics from config
2. **Module C (Social Model)** will learn cues from config
3. **Students** can experiment without code changes
4. **Teaching** becomes data-driven and systematic

**We're not cleaning up code - we're building the grammar engine that will replace the flashcard memorizer.** 🚀

---

**Document Status**: ✅ COMPLETE

**Implementation Status**:
* ✅ bars.yaml + cascades.yaml (ACTION #1 - DONE)
* 🎯 affordances.yaml (ACTION #12 - NEXT)
* 📋 cues.yaml (Future - Level 4+)

**Ready to proceed with ACTION #12!**
