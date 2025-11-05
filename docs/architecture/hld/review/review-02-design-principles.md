---
document_type: Design Rationale
status: Draft
version: 2.5
related_sections:
  - review-03-architecture-overview.md  # How design principles map to system components
  - review-04-module-specifications.md  # Technical implementation of design patterns
---

## SECTION 2: THE DESIGN PRINCIPLES

### 2.1 The Human Observer Principle

**Statement**: If a human observer standing next to the agent couldn't perceive the information, the agent shouldn't receive it at inference time (with explicit pedagogical exceptions in early curriculum levels).

This principle ensures:

- Learned policies are realistic (no omniscience)
- Social reasoning is genuine (no telepathy)
- Results are defensible (no hidden advantages)

#### Decision Framework

For every piece of information the agent might receive, ask:

```
IF curriculum_level in [Level 0, Level 1]:
    # Pedagogical exception - scaffolding to teach basics
    allowed = True  # give map, full observability

ELSE:
    question = "Can a human observer perceive this?"

    IF answer == YES:
        allowed = True
        examples:
            - "Can I see where I am?" → position
            - "Can I see this person looks tired?" → public_cues
            - "Can I hear my family member's signal?" → family_comm_channel
            - "Can I tell the time?" → time_of_day

    IF answer == NO:
        allowed = False
        examples:
            - "Can I see through walls?" → NO partial observability
            - "Can I read someone's mind?" → NO telepathy
            - "Can I know exact energy=0.234?" → NO (can see "looks tired" only)
            - "Can I know what '123' means without learning?" → NO
```

#### Examples Applied

**✅ GOOD: Location cues in Level 6+**

```yaml
# cues.yaml
- id: "at_hospital"
  trigger:
    current_affordance: "Hospital"
  visibility: "public"
```

**Human observer test**: Can you see someone is at the hospital? YES (they're physically there).

**❌ BAD: Direct bar access in Level 6+**

```python
# Don't do this at inference
observation['other_agents']['bars'] = [0.23, 0.45, ...]  # exact values
```

**Human observer test**: Can you know someone's exact energy is 0.23? NO (that's internal state).

**✅ GOOD: Public cue instead**

```yaml
- id: "looks_tired"
  trigger:
    bar: "energy"
    operator: "<"
    threshold: 0.3
  visibility: "public"
```

**Human observer test**: Can you see someone looks tired? YES (body language, facial expression).

**✅ GOOD (with caveat): Full affordance locations in Level 0-1**

```python
observation['all_affordance_locations'] = {
    'Bed': (1, 1),
    'Job': (6, 6),
    ...
}
```

**Human observer test**: Can you see the whole town layout on day 1? NO, but we're explicitly scaffolding.
**Rationale**: Teaching "bed fixes energy" before teaching "navigate to unknown bed."
**Curriculum rule**: This goes away at Level 4.

#### Training Time Exception: CTDE Labels

The principle applies to **inference time only**. During training, Module C: Social Model receives ground truth labels for supervised learning:

```python
# Training (offline) - ALLOWED
ground_truth_mood = 0.25  # actual internal state
observed_cues = ["looks_sad", "looks_poor"]
loss = train_correlation(cues → predicted_mood, label=ground_truth_mood)

# Inference (deployed policy) - human observer principle enforced
observation = {'public_cues': ["looks_sad", "looks_poor"]}  # no ground truth
predicted_mood = social_model(observation['public_cues'])
```

**Human equivalent**: You ask people "are you tired?" once to learn the correlation, then predict from facial expressions without asking. That's learning, not cheating.

**Cross-reference**: See Section 4.3 (Module C: Social Model) for centralized training details and Section 6.4 for CTDE training methodology.

---

### 2.2 The No Dense Shaping Principle

**Statement**: Agents receive minimal per-tick reward (`r = energy × health`) and discover affordance utility through exploration and long-horizon credit assignment, rather than being guided by per-action reward signals.

#### What Dense Shaping Would Look Like (The Bad Version)

Many RL environments use dense reward shaping:

```python
# Bad: telegraphing the solution
reward = 0.0

if action == "sleep":
    reward += 0.1  # "good job sleeping!"

if action == "eat":
    reward += 0.1  # "good job eating!"

if action == "work":
    reward += 0.05  # "good job working!"

if bars['money'] > prev_money:
    reward += 0.1  # "yay you earned money!"

if bars['energy'] > prev_energy:
    reward += 0.05  # "yay you rested!"
```

**Problems with this approach**:

1. **Not realistic** — humans don't get reward notifications for actions
2. **Teaches wrong thing** — agent learns "maximize action count" not "survive well"
3. **Hides world dynamics** — agent never learns cascades (hunger → health decay)
4. **Removes discovery** — there's no "aha!" moment when agent realizes food prevents starvation
5. **Brittle transfer** — changing world physics requires retuning all the shaping coefficients

#### Townlet Framework's Approach: Sparse + Terminal

```python
# Per-tick: just reflect internal state
r_t = bars[energy] * bars[health]

# Terminal: quality of life at retirement
if age >= 1.0:  # natural retirement
    r_terminal = (
        0.5 * (ticks_survived / max_age) +
        0.3 * mean([health, energy, mood, social, hygiene]) +
        0.2 * sqrt(money)
    )
elif energy <= 0 or health <= 0:  # early death
    r_terminal = 0.1 * (above formula)
```

**Why multiplicative for per-tick**:

- Additive (`energy + health`) allows compensating: 0.1 + 0.9 = 1.0 (seems fine)
- Multiplicative (`energy × health`) requires both: 0.1 × 0.9 = 0.09 (clearly bad)
- Forces balanced survival, not min-maxing one bar

**Why sqrt for terminal wealth**:

- Linear wealth → grind forever (more money always better)
- sqrt wealth → diminishing returns ($0→$25 matters more than $75→$100)
- Prevents "die rich but miserable" attractor

#### What This Forces Agents to Learn

With only sparse rewards, agents must discover:

**Direct effects** (first order):

```
Bed → energy ↑ → r_t ↑ (immediately)
Fridge → satiation ↑ (no immediate reward change)
```

**Cascade effects** (second order):

```
Fridge → satiation ↑ → (cascade) → health decay slows → r_t stays high (delayed)
Shower → hygiene ↑ → (cascade) → mood ↑ → (cascade) → energy decay slows
```

**Economic loops** (third order):

```
Job → money ↑ (no immediate reward)
  → can afford Fridge
  → satiation ↑
  → cascades prevented
  → health maintained over 50+ ticks
  → r_t stays high long-term
```

**Long-horizon planning**:

```
Early life: work hard, accumulate money
Mid life: balance work/rest/social
Late life: reduce work, spend savings, arrive at retirement healthy
Terminal: high lifespan + wellbeing + wealth score
```

#### Credit Assignment is Hard (That's the Point)

Sparse rewards create a genuine credit assignment problem:

- Decision at tick 50: "Go to Job" (gain $10)
- Decision at tick 51: "Skip Fridge" (save $4)
- Tick 100: Satiation drops below threshold
- Tick 101-200: Cascades accelerate health decay
- Tick 210: Health hits 0, agent dies
- Terminal: Low score (early death penalty)

The agent must learn: "The decision to skip Fridge at tick 51 caused death at tick 210." That's 159 ticks of delayed consequence.

**This is what makes the Townlet Framework hard.** And it's why results are interesting — you're not hand-holding the agent to the solution.

**Human equivalent**: You skip meals to save money, feel fine for a week, then get sick and miss work. You must learn "skipping meals → illness 7 days later" without someone telling you "bad job skipping meals!" every time.

#### Why This Matters for Research

Dense shaping produces policies that:

- Overfit to the shaping coefficients
- Break when world physics change
- Don't discover novel strategies

Sparse rewards produce policies that:

- Learn world dynamics (cascades, affordances, temporal patterns)
- Transfer better (understand causality, not arbitrary scores)
- Exhibit genuine emergent behavior (the "aha!" moments are real)

**Cross-reference**: See Section 5.2 for reward computation implementation details and Section 7.3 for curriculum-specific reward tuning.

---

### 2.3 The Configuration Over Code Principle

**Statement**: World physics and agent cognition are expressed as declarative configuration (YAML files) that can be inspected, diffed, and version-controlled, rather than being embedded in imperative Python code.

This enables:

- Non-programmers to design experiments
- Reproducibility (exact configs archived with results)
- Auditability (governance can read YAML, not 10K lines of torch)

#### Universe as Code: Worlds Are YAML Files

The Universe as Code (UAC) system defines world physics through multiple YAML files that compose together:

**bars.yaml** — What variables exist and how they decay:

```yaml
bars:
  - name: "energy"
    index: 0
    tier: "pivotal"
    initial: 1.0
    base_depletion: 0.005  # -0.5% per tick

  - name: "health"
    index: 6
    tier: "pivotal"
    initial: 1.0
    base_depletion: 0.0  # modulated by fitness
```

**cascades.yaml** — How neglect propagates:

```yaml
cascades:
  - name: "low_satiation_hits_health"
    source: "satiation"
    target: "health"
    threshold: 0.2  # triggers below 20%
    strength: 0.010  # -1% health per tick when starving
```

**affordances.yaml** — What actions exist and what they do:

```yaml
affordances:
  - id: "bed"
    interaction_type: "multi_tick"
    required_ticks: 4
    effects_per_tick:
      - { meter: "energy", amount: 0.25 }
    operating_hours: [0, 24]  # always open

  - id: "job"
    interaction_type: "multi_tick"
    required_ticks: 8
    effects_per_tick:
      - { meter: "money", amount: 0.10 }
      - { meter: "mood", amount: -0.02 }
    operating_hours: [9, 18]  # 9am-6pm only
```

**cues.yaml** — What social signals are observable:

```yaml
cues:
  - id: "looks_tired"
    trigger:
      bar: "energy"
      operator: "<"
      threshold: 0.3
    visibility: "public"
```

**Impact**: A researcher can create an "austerity world" (an instance called "Townlet Austerity Town") by:

1. Copy `configs/baseline/` → `configs/austerity/`
2. Edit affordances: lower wages, increase food costs
3. Launch: `townlet train --config configs/austerity/`
4. Compare: diff the YAMLs to see exactly what changed

No Python required.

**Cross-reference**: See Section 8.2 for UAC compilation pipeline and Section 9.1 for universe validation rules.

#### Brain as Code: Minds Are YAML Files

**Layer 1: cognitive_topology.yaml** — Behavior contract (what it's allowed to do):

```yaml
panic_thresholds:
  energy: 0.15  # panic if energy < 15%
  health: 0.25

compliance:
  forbid_actions:
    - "attack"
    - "steal"

personality:
  greed: 0.7
  curiosity: 0.8
  neuroticism: 0.6
```

**Layer 2: agent_architecture.yaml** — Neural blueprints (how it's built):

```yaml
modules:
  perception_encoder:
    core:
      type: "GRU"
      hidden_dim: 512
    optimizer: { type: "Adam", lr: 0.0001 }

  world_model:
    core_network:
      type: "MLP"
      layers: [256, 256]
```

**Layer 3: execution_graph.yaml** — Reasoning loop (step-by-step):

```yaml
steps:
  - name: "perception"
    node: "@modules.perception_encoder"

  - name: "policy"
    node: "@modules.hierarchical_policy"

  - name: "panic"
    node: "@modules.panic_controller"

  - name: "ethics"
    node: "@controllers.ethics_filter"
```

**Impact**: A safety researcher can:

1. Edit Layer 1: change `forbid_actions`, adjust panic thresholds
2. Verify Layer 3: confirm `ethics` runs after `panic` (not before)
3. Launch new run → new cognitive hash
4. Compare: "Same weights, different ethics = different behavior"

This turns safety into auditable configuration, not black-box hopes.

**Cross-reference**: See Section 4 for detailed module specifications and Section 10.3 for cognitive topology compilation.

#### Physics Are Data, Not Logic

Traditional RL simulator:

```python
# Hardcoded in environment.py
def step(self, action):
    if action == "sleep":
        self.energy += 0.25  # magic number
        self.hygiene -= 0.03  # another magic number

    if self.hunger < 0.2:  # arbitrary threshold
        self.health -= 0.01 * (0.2 - self.hunger)  # arbitrary formula
```

**Problems**:

- Magic numbers scattered everywhere
- Cascade logic hidden in if/else soup
- Changing physics requires editing Python
- No diff-able history of world rules

Townlet Framework approach:

```python
# Generic engine reads YAML
cascade_engine = CascadeEngine.from_yaml("cascades.yaml")
affordance_engine = AffordanceEngine.from_yaml("affordances.yaml")

def step(self, action):
    # Apply affordance effects (from YAML)
    affordance_engine.apply(agent, action)

    # Apply cascades (from YAML)
    cascade_engine.apply(agent.bars)

    # Clamp bars
    agent.bars = np.clip(agent.bars, 0.0, 1.0)
```

**Benefits**:

- Physics are version-controlled (Git tracks YAML changes)
- Experiments are reproducible (config snapshot frozen at launch)
- Diff shows exactly what changed between runs
- Non-programmers can tweak parameters

**Cross-reference**: See Section 5.4 for cascade engine implementation and Section 8.1 for affordance engine details.

---

### 2.4 The Provenance By Design Principle

**Statement**: Every run must produce a durable, verifiable artifact that proves which mind, under which world rules, produced which behavior, with tamper protection and complete audit trails.

This is not "nice to have" — it's the difference between "cool demo" and "system we can take to governance."

#### Four Mechanisms

**1. Snapshot Immutability**

On launch, the Townlet Framework:

```python
# Launch creates frozen snapshot
runs/L99_MyWorld__2025-11-03-12-14-22/
  config_snapshot/  # ← byte-for-byte copy
    config.yaml
    bars.yaml
    cascades.yaml
    affordances.yaml
    cues.yaml
    cognitive_topology.yaml
    agent_architecture.yaml
    execution_graph.yaml
```

**Rules**:

- Runtime reads only from `config_snapshot/`, never from live `configs/`
- Prevents "oops I edited the config mid-run"
- Checkpoints embed their own `config_snapshot/` (nested)

**2. Cognitive Hashing**

Every brain gets a deterministic identity:

```python
cognitive_hash = SHA256(
    yaml_files_concatenated +  # all config files
    compiled_execution_graph +  # resolved wiring after @modules.* binding
    instantiated_architectures  # layer types, dims, optimizers
)
```

**Properties**:

- Changing Layer 1 ethics → new hash
- Changing Layer 2 architecture → new hash
- Changing Layer 3 step order → new hash
- Same hash = provably same mind

**Usage**:

```python
# Telemetry every tick
log_entry = {
    'run_id': 'L99_MyWorld__2025-11-03-12-14-22',
    'tick': 1847,
    'cognitive_hash': '4f9a7c21ab...',  # which mind
    'candidate_action': 'steal',
    'panic_override': False,
    'ethics_veto': True,
    'veto_reason': 'steal forbidden by Layer 1',
    'final_action': 'wait'
}
```

**3. Glass-Box Telemetry**

Every tick logs the cognition pipeline:

```
hierarchical_policy → candidate_action
  ↓
panic_controller → panic_adjusted_action (+ panic_reason if overridden)
  ↓
ethics_filter → final_action (+ veto_reason if blocked)
```

**Governance value**:

- "It tried to steal" → logged
- "Panic overrode normal behavior" → logged with reason
- "Ethics blocked it" → logged with which rule
- "Final action was WAIT" → logged

You can answer "why did it do X?" with evidence, not speculation.

**Cross-reference**: See Section 11.2 for telemetry schema and Section 12.4 for audit query examples.

**4. Signed Checkpoints**

Checkpoints include HMAC signature:

```python
checkpoint/
  weights.pt
  optimizers.pt
  rng_state.json
  config_snapshot/
  cognitive_hash.txt
  manifest.txt      # checksums of all files
  signature.txt     # HMAC(manifest + signing_key)
```

**Verification on resume**:

```python
def verify_checkpoint(checkpoint_dir, signing_key):
    manifest = read(checkpoint_dir / 'manifest.txt')
    signature = read(checkpoint_dir / 'signature.txt')

    expected_sig = hmac(signing_key, manifest)

    if signature != expected_sig:
        raise CheckpointTamperedError("Signature invalid")

    # Also verify each file matches manifest checksums
    verify_manifest(checkpoint_dir, manifest)
```

**Security property**: Cannot edit snapshot and claim "same mind."

**Cross-reference**: See Section 13.1 for checkpoint format specification and Section 13.3 for signature verification protocol.

#### Audit Scenario

**Question**: "Why did the agent call an ambulance at 3am when it wasn't an emergency?"

**Townlet Framework answer** (with evidence):

```
Run: L99_AusterityWorld__2025-11-03-12-14-22
Tick: 1847
Cognitive hash: 4f9a7c21ab3d8ef2...

Agent state at tick 1847:
  - energy: 0.82
  - health: 0.18  ← below panic threshold (0.25)
  - money: 0.45

cognitive_topology.yaml (Layer 1) at that tick:
  panic_thresholds:
    health: 0.25  ← triggered

Telemetry line 1847:
  candidate_action: "work"  # policy wanted to continue working
  panic_override: True
  panic_reason: "health_critical"
  panic_adjusted_action: "call_ambulance"
  ethics_veto: False
  veto_reason: null  # ambulance is legal
  final_action: "call_ambulance"

affordances.yaml:
  - id: "call_ambulance"
    costs: [{ meter: "money", amount: 0.50 }]  # -$50
    effects: [{ meter: "health", amount: 0.30 }]  # +30% health
    operating_hours: [0, 24]  # always available

Outcome:
  - Money: 0.45 → -0.05 (went into debt, but health: 0.18 → 0.48)
  - Agent survived (would have died at 0% health)

Conclusion:
  Panic controller correctly triggered emergency response when health fell below
  configured threshold. Action was expensive but legal and life-saving. This is
  intended behavior per Layer 1 configuration.
```

**This level of auditability** is why the Townlet Framework can be deployed in defense/policy contexts, not just research labs.

**Cross-reference**: See Section 12 for complete audit workflow documentation.

---

### 2.5 Design Principle Summary Table

| Principle | Implementation | Benefit | WEP |
|-----------|---------------|---------|-----|
| **Human Observer** | Only human-perceivable info at inference | Realistic policies, no telepathy | 95% |
| **No Dense Shaping** | `r = energy × health` + terminal bonus | Genuine learning, not following breadcrumbs | 95% |
| **Config Over Code** | YAML for worlds (UAC) and minds (BAC) | Non-programmers can experiment | 95% |
| **Provenance By Design** | Snapshots, hashing, telemetry, signatures | Auditable behavior, reproducible results | 90% |

**Emergent property**: These four principles compose to enable **configuration-driven science** — you can test hypotheses by editing YAML, not writing Python.

**Example research workflow**:

1. Hypothesis: "Agents learn better under partial observability"
2. Config A: `observability: full` (Level 0-3 mode)
3. Config B: `observability: partial` (Level 4-5 mode)
4. Launch both → compare learning curves + final performance
5. Publish: "Config diff shows only change, cognitive hashes prove same architecture"

This is tractable because all four principles hold.

**Cross-reference**: See Section 14 for complete experimental methodology using these principles.

---

**End of Section 2**

---
