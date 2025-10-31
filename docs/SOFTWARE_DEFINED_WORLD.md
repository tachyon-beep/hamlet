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

### 7. Conclusion & Next Steps

This "Software Defined World" spec provides the clean, data-driven, and modular foundation required for the v2.0 refactor. All environment "god objects" and hardcoded logic are now replaced.

The engineering team can now proceed with:

1. Building the new environment "physics engine" that reads and parses these `.yaml` files.
2. Beginning the pre-training work for **Module B (World Model)** and **Module C (Social Model)**, which now have their "textbooks."
3. Proceeding with the **v1.5 Integration (Module A + v1.0 Q-Network)**, as its environment is now stable.
