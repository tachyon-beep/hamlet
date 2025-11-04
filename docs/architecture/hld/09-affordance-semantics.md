# 9. Affordance Semantics in universe_as_code.yaml

**Document Type**: Component Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing UAC engine, world designers creating universes, governance teams auditing world rules
**Technical Level**: Deep Technical (affordance engine semantics, reservation protocols, deterministic contention resolution)
**Estimated Reading Time**: 6 min for skim | 18 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Universe as Code (UAC) affordance system defining world mechanics as declarative YAML (not hardcoded Python). Specifies affordance parameters (capacity, quality, costs, effects_per_tick), engine semantics (reservation, contention resolution, atomic effects), and special effects whitelist.

**Why This Document Exists**:
Establishes UAC as the "world half" of BAC+UAC, making universe rules diffable, teachable, and governable. Transforms "beds make you rested" from hidden code to auditable configuration, enabling world curriculum and forensic reconstruction.

**Who Should Read This**:
- **Must Read**: Engineers implementing UAC engine, world designers creating affordances
- **Should Read**: Governance teams auditing world fairness, instructors teaching world mechanics
- **Optional**: Researchers using existing worlds (high-level understanding sufficient)

**Reading Strategy**:
- **Quick Scan** (6 min): Read §9.1 for affordance YAML examples
- **Partial Read** (12 min): Add §9.2 for engine semantics (reservation, contention, effects)
- **Full Read** (18 min): Add §9.3 for BAC+UAC integration value

---

## Document Scope

**In Scope**:
- **Affordance Declarations**: YAML structure (id, quality, capacity, exclusive, interaction_type, costs, effects_per_tick)
- **Engine Semantics**: Reservation protocol, contention resolution, atomic effects application, interrupts
- **Special Effects Whitelist**: Safe, bounded effect types (teleport, heal, damage)
- **BAC+UAC Integration**: Why declarative world + declarative mind = governance

**Out of Scope**:
- **Specific Affordance Catalog**: See instance-specific docs (Townlet Town affordance library)
- **Spatial Grid Implementation**: See environment engine docs
- **Rendering/Visualization**: See UI implementation docs
- **Bar Mechanics**: See cascades and meter decay specs

**Critical Boundary**:
Affordance system is **framework-level** (works for any universe). Examples show **Townlet Town** affordances (Bed, Hospital, Job, Phone_Ambulance with energy/health bars), but the UAC pattern applies to any universe instance (factory: Assembly Line, Maintenance Station; trading: Trading Desk, Market Data Feed).

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [01-executive-summary.md](01-executive-summary.md) (UAC concept), [08-declarative-goals-termination.md](08-declarative-goals-termination.md) (goals reference bars)
- **Builds On**: UAC philosophy (world as data), bar system (effects_per_tick targets)
- **Related**: [02-brain-as-code.md](02-brain-as-code.md) (BAC+UAC together define SDA)
- **Next**: Section 10 (Success Criteria - future)

**Section Number**: 9 / 12
**Architecture Layer**: Physical + Logical (artifact specification + runtime semantics)

---

## Keywords for Discovery

**Primary Keywords**: affordance semantics, universe_as_code, UAC, capacity, quality, effects_per_tick, reservation, contention resolution
**Secondary Keywords**: interaction_type, interruptible, distance_limit, special effects whitelist, atomic effects, precondition
**Subsystems**: UAC engine, affordance system, reservation protocol, effects applicator
**Design Patterns**: Declarative world mechanics, deterministic contention, bounded expressiveness

**Quick Search Hints**:
- Looking for "how affordances are defined"? → See §9.1 (Affordances Are Declarative)
- Looking for "how contention works"? → See §9.2 (Engine Semantics - Contention Resolution)
- Looking for "special abilities"? → See §9.2 (Special Effects Whitelist)
- Looking for "why UAC+BAC"? → See §9.3 (Why UAC Matters for BAC)

---

## Version History

**Version 1.0** (2025-11-05): Initial UAC affordance specification defining declarative world mechanics

---

## Document Type Specifics

### For Component Specification Documents (Type: Component Spec)

**Component Name**: UAC Affordance System (Declarative World Mechanics)
**Component Type**: Data-Driven Engine (interprets affordance YAML at runtime)
**Location in Codebase**: UAC engine, affordance interpreter, reservation system, effects applicator

**Interface Contract**:
- **Inputs**: universe_as_code.yaml (affordances with parameters), agent actions (affordance requests)
- **Outputs**: Bar effects applied to agents, reservation assignments, contention outcomes
- **Dependencies**: Bar system (targets for effects), spatial grid (distance checks), RNG (deterministic tie-breaking)
- **Guarantees**: Deterministic, replayable, World Model trainable

**Critical Properties**:
- **Declarative**: All affordances defined in YAML, no hidden mechanics in code
- **Deterministic**: Contention resolution reproducible (same inputs → same outcomes)
- **Bounded**: Special effects whitelist prevents arbitrary operations
- **Atomic**: Effects applied simultaneously (no partial update influence)

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 9. Affordance Semantics in universe_as_code.yaml

**Universe as Code (UAC) is the other half of this story.**

- **Brain as Code (BAC)** (Layers 1–3) defines the **mind**
- **Universe as Code (UAC)** defines the **body and the world**

**Framework principle**: UAC is framework-level (works for any universe). The specific affordances (Bed vs Assembly Line) and bars (energy vs machinery_stress) are instance-specific.

Townlet avoids hardcoded rules such as "beds make you rested" embedded throughout the Python code. The world is declared as **affordances with effects on bars**. Beds, jobs, phones, ambulances, hospitals, fridges, and pubs are **entries in the world configuration**.

**Framework benefit**: Declarative world mechanics enable world curriculum ("raise ambulance cost from $300 to $500"), forensic reconstruction ("what were bed healing rates at tick 842?"), and cross-universe comparison ("factory vs town bar recovery dynamics").

---

## 9.1 Affordances Are Declarative

Each actionable thing in the world (Bed, Job, Fridge, Hospital, Phone_Ambulance, etc.) is defined in `universe_as_code.yaml` like so:

**Example: Basic Bed** (Townlet Town instance):

```yaml
- id: "bed_basic"
  quality: 1.0              # scales how effective the rest is
  capacity: 1               # how many agents can use it this tick
  exclusive: true           # if true, only one occupant at a time
  interaction_type: "multi_tick"
  interruptible: true       # can be abandoned mid-sleep
  distance_limit: 0         # must be on the tile
  costs:
    - { bar: "money", change: -0.05 }     # pay rent to crash here
  effects_per_tick:
    - { bar: "energy", change: +0.25, scale_by: "quality" }

  on_interrupt:
    refund_fraction: 0.0    # optional semantics for partial usage
    note: "no refund if you bail early"
```

**Example: Ambulance Call** (Townlet Town instance - "special" affordance):

```yaml
- id: "phone_ambulance"
  interaction_type: "instant"
  distance_limit: 1
  costs:
    - { bar: "money", change: -3.00 }     # normalized cost (e.g. $300)
  effects:
    - { effect_type: "teleport",
        destination_tag: "nearest_hospital",
        precondition: { bar: "health", op: "<=", val: 0.2 } }
```

**Framework pattern**: Affordance YAML structure (id, quality, capacity, costs, effects_per_tick) is framework-level. The specific affordances (Bed, Phone_Ambulance) and bars (energy, health, money) are instance-specific.

**Alternative universe examples**:

**Factory instance**:
```yaml
- id: "assembly_line"
  quality: 1.0
  capacity: 4               # Four workers per line
  exclusive: false
  interaction_type: "multi_tick"
  interruptible: false      # Can't leave mid-shift
  distance_limit: 0
  costs:
    - { bar: "worker_fatigue", change: +0.10 }  # Tiring work
  effects_per_tick:
    - { bar: "production_quota", change: +0.05, scale_by: "quality" }
    - { bar: "money", change: +0.02 }  # Wage per tick

- id: "emergency_shutdown"
  interaction_type: "instant"
  distance_limit: 1
  costs:
    - { bar: "production_quota", change: -0.50 }  # Big loss
  effects:
    - { effect_type: "safety_reset",
        target: "machinery_stress",
        set_value: 0.0,
        precondition: { bar: "machinery_stress", op: ">=", val: 0.8 } }
```

**Trading instance**:
```yaml
- id: "market_data_feed"
  quality: 1.0
  capacity: unlimited       # Many can watch
  exclusive: false
  interaction_type: "instant"
  distance_limit: 0
  costs:
    - { bar: "attention", change: -0.05 }  # Mental load
  effects:
    - { effect_type: "knowledge_update",
        target: "market_information",
        refresh: true }

- id: "execute_trade"
  interaction_type: "instant"
  distance_limit: 0
  costs:
    - { bar: "portfolio_value", change: -0.01 }  # Transaction fee
  effects:
    - { effect_type: "portfolio_action",
        action_type: "buy_or_sell",
        asset: "from_agent_intent" }
```

---

### Key Affordance Properties (Framework-Level)

**There are a few important things to notice**:

### 1. Everything in Terms of Bars and Per-Tick Deltas

**Bed** raises `energy` every tick, costs a bit of `money`, maybe hurts `mood` if it's gross, etc.

**Framework pattern**: Affordances operate on bars (framework-level concept). The specific bars (energy vs worker_fatigue) are instance-specific.

**Example** (Townlet Town):
```yaml
effects_per_tick:
  - { bar: "energy", change: +0.25, scale_by: "quality" }
  - { bar: "mood", change: -0.02 }  # Gross bed lowers mood
```

**Example** (Factory):
```yaml
effects_per_tick:
  - { bar: "production_quota", change: +0.05, scale_by: "quality" }
  - { bar: "worker_fatigue", change: +0.10 }  # Work is tiring
```

### 2. Capacity + Exclusive Model Contention

**capacity + exclusive** let us model resource contention.

- Two agents **can't both occupy** a single-occupancy bed with `capacity: 1, exclusive: true`
- The engine will arbitrate who "wins" this tick in a **deterministic way**

**Framework pattern**: Contention modeling is framework-level. The specific capacity values (1 sleeper vs 4 workers) are instance-specific.

**Example** (Townlet Town):
```yaml
- id: "bed_basic"
  capacity: 1        # One sleeper
  exclusive: true    # Can't share
```

**Example** (Factory):
```yaml
- id: "assembly_line"
  capacity: 4        # Four workers
  exclusive: false   # Shared workspace
```

### 3. Interaction Type Captures Temporal Shape

**interaction_type** captures temporal shape:

- **`multi_tick`**: "stay here over multiple ticks and accumulate `effects_per_tick`"
  - Examples: Sleeping in bed, working at job, assembly line shift
- **`instant`**: "one-shot action now"
  - Examples: Calling ambulance, emergency shutdown, executing trade

**Framework pattern**: Temporal modeling is framework-level. The specific interaction types (multi_tick work vs instant call) are instance-specific.

**Example** (Townlet Town):
```yaml
bed_basic:
  interaction_type: "multi_tick"  # Sleep over time

phone_ambulance:
  interaction_type: "instant"     # One call
```

### 4. Special Abilities Referenced by Name, Not Implemented Ad Hoc

**Special effects** (teleport, heal, damage, etc.) are referenced by name, not implemented ad hoc in YAML.

The YAML is only allowed to invoke a **small whitelist** of engine-side effect handlers (teleport, etc.). That keeps the world spec **expressive but bounded**. You don't get `"nuke_city: true"`.

**Framework pattern**: Special effects whitelist is framework-level security constraint. The specific effects (teleport vs safety_reset) are whitelisted operations.

**Example** (Townlet Town):
```yaml
effects:
  - { effect_type: "teleport",
      destination_tag: "nearest_hospital",
      precondition: { bar: "health", op: "<=", val: 0.2 } }
```

**Framework constraint**: Engine implements `teleport`, `heal`, `damage`, etc. centrally. YAML references them, doesn't define them.

---

## 9.2 Engine Semantics (How the Runtime Interprets Affordances)

To keep the world **deterministic, replayable, and trainable-for-World-Model**, the engine follows strict rules.

**Framework principle**: Engine semantics are framework-level (work for any UAC configuration). The specific affordances and bars are instance-specific.

---

### 1. Reservation

When an agent tries to use an affordance, the engine does a local **"reservation" check**:

**Checks**:
- Is **capacity available**? (How many agents already using this affordance this tick?)
- Are **preconditions met**? (Health low enough, money high enough, distance within limit?)
- If yes, it assigns a **reservation token** to that agent for that tick

**This reservation is not global mutable lore.** It's **per-tick, ephemeral**.

**Why**: We don't create long-lived "ownership" state in random engine globals because that explodes complexity and makes the **World Model's job harder**. World Model needs to predict "if I try Bed next tick, will I get it?" based on observable state, not hidden reservation bookkeeping.

**Framework pattern**: Ephemeral reservation is framework-level discipline. The specific preconditions (health ≤ 0.2 for ambulance) are instance-specific.

**Example** (Townlet Town):
```
Tick 842:
- Agent_001 requests "bed_basic" (capacity=1)
- Agent_002 requests "bed_basic" (capacity=1)
- Engine: capacity=1, two requests → contention
- Contention resolution (see next section)
```

---

### 2. Contention Resolution

If **multiple agents want the same affordance** and **capacity is exceeded**, break ties **deterministically**.

**Typical order**: Sort by distance, then by agent_id.

**Determinism matters** because we want to:
- **Replay the run exactly** (same inputs → same outcomes)
- **Train the World Model** on consistent consequences (World Model learns "if I'm closer, I usually win")

**Framework pattern**: Deterministic contention is framework-level guarantee. The specific tie-breaking rules (distance → agent_id) can be configured but must be reproducible.

**Example** (Townlet Town):
```
Tick 842:
- Agent_001 distance to Bed: 0 (on tile)
- Agent_002 distance to Bed: 1 (adjacent)
- Engine: Sort by distance → Agent_001 wins
- Agent_001 gets reservation token
- Agent_002 action fails (capacity exceeded)
```

**World Model learns**: "If I'm on tile, I'm more likely to get Bed than if I'm adjacent."

**Framework benefit**: Reproducible contention enables World Model training (can learn competition dynamics) and forensic replay (can explain "why Agent_002 didn't get Bed at tick 842").

---

### 3. Effects Application

Once reservations are resolved, all **costs and effects_per_tick** for all active affordances are:

1. **Collected** (per agent)
2. **Summed** (per agent)
3. **Atomically applied** to bars (energy, health, money, etc.)
4. **Clamped** to valid range ([0.0, 1.0] or whatever the world defines)

**Key point**: We don't **partially apply** effects from some affordances and then let those partial updates influence others in the same tick. We apply **atomically at the end of the tick**. This gives **clean training data**.

**Framework pattern**: Atomic effects application is framework-level discipline. The specific bars (energy vs machinery_stress) and clamp ranges ([0.0, 1.0]) are instance-specific.

**Example** (Townlet Town):
```
Tick 842:
- Agent_001 using "bed_basic"
  - effects_per_tick: [{ bar: "energy", change: +0.25 }]
  - costs: [{ bar: "money", change: -0.05 }]
- Agent_001 also has cascade decay: [{ bar: "energy", change: -0.02 }] (natural decay)

Engine collects:
- energy: +0.25 (bed) - 0.02 (decay) = +0.23
- money: -0.05 (bed cost)

Applies atomically:
- energy: 0.55 → 0.78
- money: 0.30 → 0.25

Clamps:
- energy: 0.78 (within [0.0, 1.0], no change)
- money: 0.25 (within [0.0, 1.0], no change)
```

**World Model learns**: "Bed gives +0.25 energy/tick minus natural decay. Net gain ~0.23/tick."

---

### 4. Interrupts

If **`interruptible: true`** and the agent walks off or is forced to bail (panic_controller might decide "leave bed now and call ambulance"), we **stop applying future per-tick effects**.

**`on_interrupt`** can define whether you get any **partial benefit or refund**. That's still **declarative**.

**Framework pattern**: Interrupt semantics are framework-level. The specific refund policies (refund_fraction: 0.0 vs 0.5) are instance-specific.

**Example** (Townlet Town):
```yaml
bed_basic:
  interruptible: true
  on_interrupt:
    refund_fraction: 0.0
    note: "no refund if you bail early"
```

**Scenario**:
```
Tick 840: Agent_001 starts sleeping in "bed_basic"
Tick 841: Agent_001 still sleeping (energy: 0.30 → 0.55)
Tick 842: Panic controller detects health <0.25 → interrupt sleep → call ambulance
Engine: Stop applying bed effects_per_tick, no refund (refund_fraction=0.0)
```

**World Model learns**: "If panic interrupts sleep, I lose remaining benefit."

---

### 5. Special Effects Whitelist

YAML is allowed to reference a **small set of named effect_type operations** (like `teleport`), and the engine implements those **centrally**.

**That way**:
- `"teleport to nearest_hospital"` is a **normal, auditable world affordance**
- **Not** a custom `'if agent.health < X then hack position'` buried in Python

**This whitelist is versioned.** If you add a new special effect, you're extending world semantics globally and that should **change the hash** once it's applied to a snapshot.

**Framework pattern**: Special effects whitelist is framework-level security boundary. The specific effects (teleport, heal, damage, safety_reset, portfolio_action) are centrally implemented and versioned.

**Examples of whitelisted effects**:

**Townlet Town**:
- `teleport`: Move agent to tagged location (e.g., `destination_tag: "nearest_hospital"`)
- `heal`: Restore health bar (e.g., `{ effect_type: "heal", bar: "health", amount: +0.5 }`)
- `damage`: Reduce health bar (e.g., `{ effect_type: "damage", bar: "health", amount: -0.3 }`)

**Factory**:
- `safety_reset`: Zero out machinery_stress (e.g., `{ effect_type: "safety_reset", target: "machinery_stress", set_value: 0.0 }`)
- `quality_boost`: Improve production quality (e.g., `{ effect_type: "quality_boost", target: "product_quality", multiplier: 1.5 }`)

**Trading**:
- `portfolio_action`: Execute buy/sell (e.g., `{ effect_type: "portfolio_action", action_type: "buy_or_sell", asset: "from_agent_intent" }`)
- `knowledge_update`: Refresh market data (e.g., `{ effect_type: "knowledge_update", target: "market_information", refresh: true }`)

**Framework constraint**: No arbitrary operations allowed. Engine refuses `"nuke_city: true"`, `"infinite_money: true"`, etc.

---

## 9.3 Why Universe as Code Matters for BAC

**Universe as Code (UAC) and Brain as Code (BAC) are two halves of the same sentence.**

**Framework principle**: UAC+BAC integration is the foundation of the Townlet Framework. This pattern works for any universe instance.

### UAC: The World Half

**UAC**: The world, bodies, bars, affordances, economy, social cues, ambulance rules, etc., are **all declared in YAML**.

**They are**:
- **Diffable** (show me what changed between world v1 and v2)
- **Teachable** (instructors can point to YAML and explain "ambulance costs $300")
- **Inspectable by non-coders** (governance can read world rules without Python)

**Framework benefit**: Declarative world enables world curriculum, forensic reconstruction, and cross-universe comparison.

**Example world curriculum** (Townlet Town):
```yaml
# Early training: Cheap survival
ambulance_cost: -1.00  # $100

# Later curriculum: Expensive survival
ambulance_cost: -3.00  # $300
```

**Config diff**: `ambulance_cost: -1.00 → -3.00`

**Behavioral shift**: Agent learns to prioritize prevention (maintain health) over reactive spending (expensive ambulance).

---

### BAC: The Mind Half

**BAC**: The mind, panic thresholds, ethics vetoes, planning depth, social reasoning, module architectures, and actual cognition loop are **also declared in YAML**.

**They are**:
- **Diffable** (show me what changed between agent v1 and v2)
- **Teachable** (students can point to YAML and explain "panic triggers at 15% energy")
- **Inspectable by non-coders** (governance can read ethics rules without Python)

**Framework benefit**: Declarative mind enables curriculum, ablations, and governance.

---

### Together: Accountable Simulated Society

When you run a simulation, Townlet snapshots **both halves** into a run folder, stamps them with a **cognitive hash**, and then logs decisions per tick against that identity.

**So instead of**:
> "The AI did something weird overnight and now it's different"

**We can say**:
> "At tick 842, Mind `4f9a7c21`, in World `Nightshift_v3` with `ambulance_cost: -3.00` (normalized $300) and `bed_basic.quality: 1.0`, entered panic because `health < 0.25`.
>
> Panic escalated the action to `call_ambulance`.
>
> EthicsFilter allowed it (ambulance is legal, even if expensive).
>
> Money was deducted (`-3.00` normalized, $300 actual).
>
> Agent teleported to the nearest `hospital` affordance (special effect: `teleport`, `destination_tag: "nearest_hospital"`).
>
> See `veto_reason` in telemetry: it also tried to `STEAL` food two ticks earlier and that was blocked (`ethics_veto_applied: true`, `veto_reason: "forbidden: steal"`)."

**That is the moment where governance stops being hypothetical and becomes screenshot material.**

**Framework value**: This governance narrative works for any universe:

**Factory example**:
> "At tick 1420, Mind `7e2b9d14`, in World `Factory_Floor_v2` with `emergency_shutdown.cost: -0.50` (50% production quota loss) and `assembly_line.quality: 1.0`, entered panic because `machinery_stress >= 0.80`.
>
> Panic escalated the action to `emergency_shutdown`.
>
> EthicsFilter allowed it (shutdown is legal for safety).
>
> Production quota was deducted (`-0.50`).
>
> Machinery stress was reset to `0.0` (special effect: `safety_reset`).
>
> See telemetry: it tried to continue production two ticks earlier and that was blocked by panic (`panic_override_applied: true`, `panic_reason: "machinery_critical"`)."

**Trading example**:
> "At tick 3200, Mind `9a1c5f28`, in World `Trading_Floor_v1` with `execute_trade.cost: -0.01` (1% transaction fee) and `market_volatility: 0.45`, entered panic because `portfolio_value <= 0.70` (lost 30%).
>
> Panic escalated the action to `preserve_capital` (defensive position).
>
> EthicsFilter allowed it (capital preservation is legal).
>
> Portfolio rebalanced to defensive assets (special effect: `portfolio_action`).
>
> See telemetry: it attempted aggressive buy two ticks earlier during crash and that was blocked by panic (`panic_override_applied: true`, `panic_reason: "portfolio_critical"`)."

---

**And that's the point of Townlet**: it's not a toy black box any more. **It's an accountable simulated society with auditable minds.**

**Framework foundation**: BAC+UAC together create **Software Defined Agents in Software Defined Worlds** - fully auditable, reproducible, and governable systems for any domain.

---

**Summary**: The Townlet Framework UAC affordance system provides:

1. **Affordance Declarations** - YAML structures (id, quality, capacity, interaction_type, costs, effects_per_tick)
2. **Engine Semantics** - Reservation protocol, deterministic contention resolution, atomic effects application, interrupt handling
3. **Special Effects Whitelist** - Bounded expressiveness (teleport, heal, damage - no arbitrary operations)
4. **BAC+UAC Integration** - Declarative world + declarative mind = accountable simulated society

**Framework principle**: UAC is framework-level (works for any universe). Specific affordances (Bed vs Assembly Line) and bars (energy vs machinery_stress) are instance-specific.

**Together with BAC**, UAC transforms agents from "black box mystery" to "auditable cognitive system in auditable world."

---
