# Townlet v2.5: Complete Peer Review & Implementation Guide

**Document Version**: 1.0  
**Date**: 3 November 2025  
**Status**: Master Reference Document  
**Author**: Principal Technical Advisor (AI) + Lead Architect Review

---

## SECTION 0: EXECUTIVE SUMMARY

### 0.1 What Townlet Actually Is

Townlet v2.5 is a configuration-driven multi-agent reinforcement learning platform that treats both worlds and minds as auditable, reproducible content rather than opaque code.

**The core insight**: If you can express a problem as meters (state variables), cascades (coupling dynamics), and affordances (available actions), you can simulate it. The learning agent discovers survival strategies through sparse rewards and long-horizon planning, without dense reward shaping or privileged information access.

Townlet provides three coordinated layers:

1. **Universe as Code (UAC)**: World physics defined in YAML
   - `bars.yaml` — survival meters (energy, health, money, mood, etc.)
   - `cascades.yaml` — how neglect propagates (hunger → health decay)
   - `affordances.yaml` — available actions (Bed, Job, Hospital, etc.)
   - `cues.yaml` — social observability (visible body language, not telepathy)

2. **Brain as Code (BAC)**: Agent cognition defined in YAML
   - Layer 1: `cognitive_topology.yaml` — behavior contract (what it's allowed to do)
   - Layer 2: `agent_architecture.yaml` — neural blueprints (how it's built)
   - Layer 3: `execution_graph.yaml` — reasoning loop (step-by-step cognition)

3. **Provenance by Design**: Every run is a frozen, auditable artifact
   - Snapshot immutability (configs copied on launch, never re-read)
   - Cognitive hashing (mind fingerprint proving "which brain did what")
   - Glass-box telemetry (candidate action → panic override → ethics veto → final action)
   - Signed checkpoints (tamper protection)

The platform ships with an 8-level curriculum (L0–L8) that progressively removes scaffolding: from god-mode survival (full observability) to emergent family communication (coordinated signaling without pre-shared semantics).

**Words of estimative probability (WEP)** that this summary reflects system intent: very high (~95%).

---

### 0.2 The Core Innovation: No Cheating

Most RL simulators give agents information humans wouldn't have (omniscience, telepathy, dense action rewards). Townlet enforces a strict **"human observer principle"**: if a human standing next to the agent couldn't perceive it, the agent shouldn't get it at inference time.

This manifests in four design choices:

#### 1. **Sparse Reward: `r = energy × health`**

Per-tick reward is just the product of two survival meters. That's it.

```python
r_t = bars[energy] * bars[health]  # range [0, 1]
```

**What this forces**:

- No dense shaping ("good job eating!", "nice work sleeping!")
- Agents must discover that Fridge → satiation ↑ → cascade prevented → health maintained
- Credit assignment over 50+ ticks (Job → money → Fridge → survival)

**Human equivalent**: "I feel good when I'm energized and healthy." You don't get a reward notification for eating; you just stop feeling hungry.

#### 2. **Terminal Retirement Bonus**

Episodes end naturally when agents reach max age (the "age" bar hits 1.0, representing retirement). Terminal reward depends on:

- Lifespan completion (did you make it to retirement?)
- Wellbeing at retirement (health, energy, mood, social, hygiene)
- Wealth at retirement (diminishing returns via sqrt)

```python
if age >= 1.0:  # retirement
    terminal_reward = (
        0.5 * (ticks_survived / max_age) +      # lifespan
        0.3 * mean([health, energy, mood, social, hygiene]) +  # wellbeing
        0.2 * sqrt(money)                        # wealth
    )
elif energy <= 0 or health <= 0:  # early death
    terminal_reward = 0.1 * (above formula)  # 90% penalty
```

**What this teaches**:

- Surviving to retirement is worth 10× dying early
- Quality of life matters (can't just grind money and die miserable)
- There's a sweet spot between hoarding and living well

**Human equivalent**: "Did you live a good life?" — judged at the end, not tick-by-tick.

#### 3. **Human-Perceivable Observations Only**

Curriculum progression removes information access:

- **L0-1 (pedagogical exception)**: Full observability, all affordance locations given (like being handed a map)
- **L2-3**: Still full observability (mastering economics before navigation)
- **L4-5**: Partial observability (5×5 vision on 8×8 grid — must explore and remember)
- **L6-7**: Social cues only (see "looks_tired", not exact energy=0.23 — no telepathy)
- **L8**: Communication signals without semantics (hear "123", must learn meaning via correlation)

**Human observer test** for every design decision:

- ✅ "Can a human see someone looks tired?" YES → observable cue
- ❌ "Can a human know their exact energy value?" NO → hidden
- ✅ "Can a human hear a signal?" YES → family_comm_channel
- ❌ "Does a human know what '123' means innately?" NO → must learn

#### 4. **CTDE via Observable Cues**

The only "cheat" is offline supervised learning for social reasoning:

**Training time** (using logged episodes):

```python
# Ground truth (privileged labels)
agent_a_actual_mood = 0.25

# Observable signals (from cues.yaml)
agent_a_cues = ["looks_sad", "looks_poor"]

# Supervised learning
predicted_mood = social_model(agent_a_cues)
loss = mse(predicted_mood, agent_a_actual_mood)
```

**Inference time** (deployed policy):

```python
# Only sees cues, not ground truth
observation = {'other_agents': {'public_cues': [['looks_tired', 'at_job']]}}
predicted_state = social_model(observation['other_agents']['public_cues'])
```

**Human equivalent**: You learn "droopy eyes + slow movement = tired" by asking people once, then predicting without asking. That's not cheating, that's learning.

**WEP** that this architecture avoids runtime telepathy: very high (~95%).

---

### 0.3 What Makes This Different

#### vs. Other RL Simulators (OpenAI Gym, MuJoCo, etc.)

| Feature | Typical RL Sim | Townlet |
|---------|---------------|---------|
| **World physics** | Hardcoded in Python | Configured in YAML |
| **Reward function** | Dense shaping per action | Sparse (`r = energy × health`) |
| **Observability** | Often full state | Curriculum-staged, human-realistic |
| **Provenance** | "Trust me" | Cognitive hash + signed checkpoints |
| **Reproducibility** | Config drift common | Snapshot immutability |
| **Auditability** | Black box | Glass-box telemetry |

#### vs. Other Multi-Agent Systems (SMAC, Hanabi, etc.)

| Feature | Typical Multi-Agent | Townlet |
|---------|---------------------|---------|
| **Social info** | Full state or blind | Observable cues only |
| **Communication** | Pre-grounded or none | Emergent via correlation |
| **Cooperation** | Shared reward | Families form via breeding |
| **Competition** | Teams fixed at init | Dynamic (rivals, remarriage, churn) |

#### vs. Other Survival Sims (The Sims, RimWorld, etc.)

| Feature | Typical Survival Sim | Townlet |
|---------|---------------------|----------|
| **Agent learning** | Scripted AI | Deep RL from scratch |
| **Physics** | Game balance tweaks | Scientific modeling via cascades |
| **Reward** | Player satisfaction | Minimalist (`r = energy × health`) |
| **Purpose** | Entertainment | Research + pedagogy |

**The unique combination**: Scientific rigor (RL research) + configuration flexibility (game mods) + governance auditability (defense/policy applications).

---

### 0.4 Who This Is For

#### Researchers

**Value proposition**: "I have a hypothesis about X. Can I test it without writing Python?"

**If X involves**:

- Resource management under scarcity
- Multi-objective optimization
- Temporal planning (time-of-day constraints)
- Social coordination (families, competition, emergent communication)
- Long-horizon credit assignment

**Then YES** — edit YAML, launch runs, compare results with provenance.

**Example**: "Does dynasty inheritance produce better coordination than meritocratic churn?"

- Edit `population_genetics.yaml` → two configs
- Launch both → compare family stability, communication diversity, wealth distribution
- Publish with cognitive hashes proving which rules produced which outcomes

#### Educators

**Value proposition**: "I want students to experiment without coding."

**Curriculum levels map to learning objectives**:

- L0-1: "What is a policy?" (learn affordance semantics)
- L2-3: "How do I balance resources?" (economic loops)
- L4-5: "How do I navigate under uncertainty?" (exploration + memory)
- L6-7: "How do I reason about others?" (theory of mind via cues)
- L8: "How does language emerge?" (communication without pre-shared meaning)

**Students can**:

- Tweak world parameters (make food cheaper, jobs scarcer)
- Observe behavioral changes
- Read glass-box telemetry ("it tried to steal, ethics blocked it")
- Understand *why* policies emerged

**No code required** — just YAML editing and running `townlet train --config configs/student_world/`.

#### Policy / Governance People

**Value proposition**: "Can you prove this is safe?"

**Townlet provides**:

- Explicit ethics rules in Layer 1 (`cognitive_topology.yaml`)
- Deterministic EthicsFilter (no learned safety — just rule enforcement)
- Cognitive hashing (prove which mind, with which rules, did what)
- Telemetry showing vetoes ("attempted STEAL, blocked, reason: forbidden by Layer 1")
- Signed checkpoints (tamper protection)

**Audit question**: "Why did it call an ambulance at 3am?"
**Townlet answer**:

- "At tick 1847, agent hash `4f9a7c21` had health=0.18, panic threshold=0.25"
- "Panic controller overrode normal policy with `call_ambulance`"
- "EthicsFilter allowed (ambulance is legal even when expensive)"
- "See telemetry line 1847, veto_reason=null, panic_reason='health_critical'"

This is evidence, not anecdote.

---

### 0.5 Document Roadmap

This document is organized as:

**Understanding (Sections 0-2)**: What Townlet is, why it's designed this way, and the reward architecture

**Technical Detail (Sections 3-5)**: Curriculum, observation space, and cues system

**Implementation (Sections 6-7)**: Critical blockers to fix and missing specifications to write

**Research Directions (Sections 8-9)**: Population genetics, inheritance experiments, and the broader platform vision

**Action Plan (Sections 10-11)**: What to do next and how to pitch it

**Appendices**: Configuration templates, success criteria checklists, glossary, related work

**How to use this document**:

- If you're **fixing bugs**: Start with Section 6 (Critical Blockers)
- If you're **understanding the design**: Read Sections 0-2 in order
- If you're **implementing features**: Sections 4-5, then 7
- If you're **doing research**: Sections 8-9
- If you're **writing docs**: Use appendices as templates

---

## SECTION 1: THE DESIGN PRINCIPLES

### 1.1 The Human Observer Principle

**Statement**: If a human observer standing next to the agent couldn't perceive the information, the agent shouldn't receive it at inference time (with explicit pedagogical exceptions in early curriculum levels).

This principle ensures:

- Learned policies are realistic (no omniscience)
- Social reasoning is genuine (no telepathy)
- Results are defensible (no hidden advantages)

#### Decision Framework

For every piece of information the agent might receive, ask:

```
IF curriculum_level in [L0, L1]:
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

**✅ GOOD: Location cues in L6+**

```yaml
# cues.yaml
- id: "at_hospital"
  trigger:
    current_affordance: "Hospital"
  visibility: "public"
```

**Human observer test**: Can you see someone is at the hospital? YES (they're physically there).

**❌ BAD: Direct bar access in L6+**

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

**✅ GOOD (with caveat): Full affordance locations in L0-1**

```python
observation['all_affordance_locations'] = {
    'Bed': (1, 1),
    'Job': (6, 6),
    ...
}
```

**Human observer test**: Can you see the whole town layout on day 1? NO, but we're explicitly scaffolding.
**Rationale**: Teaching "bed fixes energy" before teaching "navigate to unknown bed."
**Curriculum rule**: This goes away at L4.

#### Training Time Exception: CTDE Labels

The principle applies to **inference time only**. During training, Module C (Social Model) receives ground truth labels for supervised learning:

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

**WEP** that this exception is philosophically sound: high (~90%). The alternative (unsupervised clustering of cues) would be slower but equally valid.

---

### 1.2 The No Dense Shaping Principle

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

#### Townlet's Approach: Sparse + Terminal

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

**This is what makes Townlet hard.** And it's why results are interesting — you're not hand-holding the agent to the solution.

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

**WEP** that sparse rewards are essential for genuine learning: very high (~95%).

---

### 1.3 The Configuration Over Code Principle

**Statement**: World physics and agent cognition are expressed as declarative configuration (YAML files) that can be inspected, diffed, and version-controlled, rather than being embedded in imperative Python code.

This enables:

- Non-programmers to design experiments
- Reproducibility (exact configs archived with results)
- Auditability (governance can read YAML, not 10K lines of torch)

#### Universe as Code: Worlds Are YAML Files

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

**Impact**: A researcher can create an "austerity world" by:

1. Copy `universe_baseline.yaml` → `universe_austerity.yaml`
2. Edit affordances: lower wages, increase food costs
3. Launch: `townlet train --config configs/austerity/`
4. Compare: diff the YAMLs to see exactly what changed

No Python required.

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
    node: "@modules.EthicsFilter"
```

**Impact**: A safety researcher can:

1. Edit Layer 1: change `forbid_actions`, adjust panic thresholds
2. Verify Layer 3: confirm `ethics` runs after `panic` (not before)
3. Launch new run → new cognitive hash
4. Compare: "Same weights, different ethics = different behavior"

This turns safety into auditable configuration, not black-box hopes.

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

Townlet approach:

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

**WEP** that config-driven design is essential for research velocity: very high (~95%).

---

### 1.4 The Provenance By Design Principle

**Statement**: Every run must produce a durable, verifiable artifact that proves which mind, under which world rules, produced which behavior, with tamper protection and complete audit trails.

This is not "nice to have" — it's the difference between "cool demo" and "system we can take to governance."

#### Four Mechanisms

**1. Snapshot Immutability**

On launch, Townlet:

```python
# Launch creates frozen snapshot
runs/L99_MyWorld__2025-11-03-12-14-22/
  config_snapshot/  # ← byte-for-byte copy
    config.yaml
    universe_as_code.yaml
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
    yaml_files_concatenated +  # all 5 config files
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
EthicsFilter → final_action (+ veto_reason if blocked)
```

**Governance value**:

- "It tried to steal" → logged
- "Panic overrode normal behavior" → logged with reason
- "Ethics blocked it" → logged with which rule
- "Final action was WAIT" → logged

You can answer "why did it do X?" with evidence, not speculation.

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

#### Audit Scenario

**Question**: "Why did the agent call an ambulance at 3am when it wasn't an emergency?"

**Townlet answer** (with evidence):

```
Run: L99_AusterityWorld__2025-11-03-12-14-22
Tick: 1847
Cognitive hash: 4f9a7c21ab3d8ef2...

Agent state at tick 1847:
  - energy: 0.82
  - health: 0.18  ← below panic threshold (0.25)
  - money: 0.45

Cognitive_topology.yaml (Layer 1) at that tick:
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

Affordances.yaml:
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

**This level of auditability** is why Townlet can be deployed in defense/policy contexts, not just research labs.

**WEP** that provenance architecture enables governance deployment: high (~90%).

---

### 1.5 Design Principle Summary Table

| Principle | Implementation | Benefit | WEP |
|-----------|---------------|---------|-----|
| **Human Observer** | Only human-perceivable info at inference | Realistic policies, no telepathy | 95% |
| **No Dense Shaping** | `r = energy × health` + terminal bonus | Genuine learning, not following breadcrumbs | 95% |
| **Config Over Code** | YAML for worlds (UAC) and minds (BAC) | Non-programmers can experiment | 95% |
| **Provenance By Design** | Snapshots, hashing, telemetry, signatures | Auditable behavior, reproducible results | 90% |

**Emergent property**: These four principles compose to enable **configuration-driven science** — you can test hypotheses by editing YAML, not writing Python.

**Example research workflow**:

1. Hypothesis: "Agents learn better under partial observability"
2. Config A: `observability: full` (L0-3 mode)
3. Config B: `observability: partial` (L4-5 mode)
4. Launch both → compare learning curves + final performance
5. Publish: "Config diff shows only change, cognitive hashes prove same architecture"

This is tractable because all four principles hold.

---

**End of Sections 0-1**

---

## SECTION 2: THE REWARD ARCHITECTURE

### 2.1 Per-Tick Reward: `r = energy × health`

The per-tick reward is deliberately minimal:

```python
r_t = bars[energy] * bars[health]
```

Where both values are normalized [0.0, 1.0], producing reward range [0.0, 1.0].

#### Why These Two Bars Specifically?

**Energy and health are the pivotal survival meters** — the only two that cause immediate death when they hit zero.

From `bars.yaml`:

```yaml
terminal_conditions:
  - meter: "energy"
    operator: "<="
    value: 0.0
    description: "Death by exhaustion"
    
  - meter: "health"
    operator: "<="
    value: 0.0
    description: "Death by health failure"
```

**Rationale**: The reward reflects "am I in immediate danger of dying?" Not "am I happy" or "am I rich" — those matter for long-term quality of life (captured in terminal reward), but moment-to-moment, survival dominates.

**Human equivalent**: A starving person doesn't care about their mood or social connections *right now* — they care about not dying. Once survival is assured, other concerns emerge. This is Maslow's hierarchy in a reward function.

#### Why Multiplicative, Not Additive?

**Additive reward** `r = energy + health` allows compensating:

```python
# Scenario A: balanced
energy = 0.5, health = 0.5
r = 0.5 + 0.5 = 1.0  ✓

# Scenario B: unbalanced
energy = 0.1, health = 0.9
r = 0.1 + 0.9 = 1.0  ✓ (same reward!)
```

But scenario B is dangerous — you're one more energy tick from death. The reward shouldn't suggest these are equivalent.

**Multiplicative reward** `r = energy × health` requires both:

```python
# Scenario A: balanced
energy = 0.5, health = 0.5
r = 0.5 × 0.5 = 0.25

# Scenario B: unbalanced
energy = 0.1, health = 0.9
r = 0.1 × 0.9 = 0.09  ✗ (much worse!)
```

The agent learns: "Having one bar very low is catastrophic, even if the other is high."

**Alternative considered: minimum**

```python
r = min(energy, health)  # "weakest link"
```

This also captures the danger of imbalance, but it's discontinuous (derivatives are undefined at energy=health), which can cause training instability. Multiplication gives smooth gradients.

**WEP** that multiplicative is optimal for this use case: high (~85%). Minimum would work but is harder to train through.

#### What This Doesn't Tell Agents

The reward provides no guidance about:

**❌ Which affordances exist**

- No reward bump for discovering "Bed"
- No reward bump for first time using "Job"

**❌ Which affordances to use**

- No `+0.1` for sleeping
- No `+0.1` for eating
- No `+0.05` for working

**❌ When to use affordances**

- No temporal shaping ("good job working during business hours!")
- No economic shaping ("good job saving money!")

**❌ How bars are coupled**

- No hint that satiation affects health via cascades
- No hint that hygiene affects mood affects energy
- Must discover cascade dynamics through experience

**❌ Long-term strategy**

- No guidance about balancing work/rest
- No guidance about retirement planning
- No guidance about money management

**This is intentional.** The agent must learn world dynamics through exploration and credit assignment, not by following reward breadcrumbs.

#### What Agents Must Discover

With only `r = energy × health`, agents must learn through trial, error, and long-horizon reasoning:

**Direct relationships** (observable within 1-5 ticks):

```
Action: go to Bed
  → energy increases (direct effect from affordances.yaml)
  → r_t increases (immediate feedback)
  → Learning: "Bed is good when energy is low"

Action: go to Doctor
  → health increases (direct effect)
  → r_t increases (immediate feedback)
  → Learning: "Doctor is good when health is low"
```

**Indirect relationships via cascades** (observable over 10-50 ticks):

```
Action: ignore Fridge
  → satiation drops (base_depletion from bars.yaml)
  → satiation < 0.2 threshold (cascades.yaml)
  → cascade: low_satiation_hits_health
  → health drops faster than base rate
  → r_t drops gradually
  → Learning: "Not eating causes slow health decay"

Action: go to Fridge
  → satiation increases (direct effect)
  → satiation > 0.2 threshold
  → cascade prevented
  → health decay returns to normal
  → r_t stabilizes
  → Learning: "Eating prevents health decay"
```

**Economic loops** (observable over 50-200 ticks):

```
Action: go to Job repeatedly
  → money increases (direct effect)
  → energy decreases (cost per tick)
  → mood decreases (cost per tick)
  → r_t drops during work
  → BUT: money enables Fridge, Bed, etc
  → Using money on recovery → health/energy maintained
  → r_t recovers and stays high long-term
  → Learning: "Work is bad short-term but enables survival long-term"
```

**Temporal patterns** (observable over 100+ ticks):

```
Observation: Job is masked at night
  → Trying to work at 3am fails
  → Energy wasted walking to Job
  → r_t drops
  → Learning: "Job has operating hours, plan around them"

Strategy: arrive at Job at 7am, WAIT until 8am
  → Minimal energy waste
  → Can work full shift when it opens
  → More efficient money accumulation
  → Better r_t over episode
  → Learning: "Temporal planning matters"
```

**Social competition** (L6+, observable over entire episodes):

```
Observation: rival agent took the Job
  → I can't work this shift
  → Money doesn't increase
  → Can't afford food later
  → Health drops due to starvation
  → r_t plummets
  → Learning: "Other agents compete for resources"

Strategy: predict when rival will go to Job
  → Arrive earlier, or choose alternate income (Labor)
  → Money still flows
  → Survival maintained
  → Better r_t over episode
  → Learning: "Social reasoning enables resource access"
```

#### Why This Is Hard (And Why That's Good)

The credit assignment problem is severe:

**Example: The delayed starvation death**

```
Tick 50: Agent chooses to skip Fridge (saves $4)
Tick 51-99: Satiation drops slowly
Tick 100: Satiation < 0.2, cascade activates
Tick 101-200: Health decays at 2× normal rate
Tick 210: Health = 0, agent dies
Terminal reward: 0.1 × (low score) due to early death penalty

Credit assignment challenge:
  The decision at tick 50 killed the agent at tick 210.
  That's 160 ticks of delayed consequence.
  No intermediate rewards pointed to the mistake.
  Agent must learn: "skip meal → death 160 ticks later"
```

**This requires**:

- Value function learning (predict long-term outcomes)
- World model learning (understand cascade dynamics)
- Temporal reasoning (connect distant cause and effect)

**Comparison to dense shaping**:

```python
# Dense shaping (easy mode)
if action == "eat":
    r += 0.1  # immediate feedback, no credit assignment needed

# Sparse (hard mode)
# No feedback at all until health drops 100 ticks later
```

**Why hard mode is better for research**:

- Tests genuine learning, not reward following
- Produces policies that understand causality
- Transfers better when world physics change
- Exhibits real emergent behavior (discovery moments)

**WEP** that this credit assignment difficulty is a feature, not a bug: very high (~95%).

---

### 2.2 Terminal Reward: Retirement Scoring

When an episode ends, the agent receives a terminal bonus based on how well they lived.

#### The Formula

```python
def compute_terminal_reward(bars, money, ticks_survived, max_age, outcome):
    """
    Terminal reward depends on:
    - Did you complete your full lifespan? (retirement vs early death)
    - Quality of life at the end (wellbeing)
    - Financial security (wealth)
    """
    
    # Component 1: Lifespan completion
    lifespan_score = ticks_survived / max_age
    
    # Component 2: Wellbeing at terminal state
    wellbeing_bars = [
        bars['health'],
        bars['energy'],
        bars['mood'],
        bars['social'],
        bars['hygiene']
    ]
    wellbeing_score = mean(wellbeing_bars)
    
    # Component 3: Wealth (diminishing returns)
    wealth_score = sqrt(money)  # or linear, log - configurable
    
    # Weighted combination
    raw_score = (
        config.retirement.weights.lifespan * lifespan_score +
        config.retirement.weights.wellbeing * wellbeing_score +
        config.retirement.weights.wealth * wealth_score
    )
    
    # Apply outcome multiplier
    if outcome == "retirement":  # age >= 1.0
        return raw_score
    elif outcome == "death":  # energy=0 or health=0
        return config.retirement.early_death_multiplier * raw_score
```

**Default configuration**:

```yaml
retirement:
  max_age_ticks: 1200
  
  weights:
    lifespan: 0.5   # 50% - did you complete your life?
    wellbeing: 0.3  # 30% - were you healthy/happy at the end?
    wealth: 0.2     # 20% - did you have financial security?
  
  wealth_curve: "sqrt"  # diminishing returns on money
  early_death_multiplier: 0.1  # 90% penalty for dying early
  
  wellbeing_bars:
    - health
    - energy
    - mood
    - social
    - hygiene
```

#### Component Breakdown

**Lifespan (50% weight by default)**

```python
lifespan_score = ticks_survived / max_age
```

**Interpretation**:

- Die at tick 600 / 1200 = 0.5 lifespan score
- Retire at tick 1200 / 1200 = 1.0 lifespan score

**Why this matters**:

- Surviving longer is inherently valuable
- Distinguishes "scraped by to retirement" from "died young"
- Even if you retire broke, completing your life counts for something

**Wellbeing (30% weight by default)**

```python
wellbeing_score = mean([health, energy, mood, social, hygiene])
```

**Interpretation**:

- Retiring with all bars at 0.8+ = 0.8 wellbeing score (comfortable)
- Retiring with mixed bars (0.9, 0.3, 0.5, 0.4, 0.6) = 0.54 wellbeing score (struggled)
- Retiring at death's door (0.1, 0.1, 0.1, 0.0, 0.2) = 0.10 wellbeing score (miserable)

**Why this matters**:

- Prevents "die rich but filthy and alone" strategies
- Rewards quality of life, not just survival
- Makes mood/social/hygiene instrumentally valuable (they affect terminal score)

**Note**: Fitness is excluded because it's already instrumentally valuable (modulates health decay via cascades). Including it would double-count its importance.

**Wealth (20% weight by default)**

```python
wealth_score = sqrt(money)  # money normalized [0, 1] where 1.0 = $100
```

**Interpretation**:

```python
money = 0.00 → sqrt(0.00) = 0.00  # broke
money = 0.25 → sqrt(0.25) = 0.50  # $25 saved
money = 0.64 → sqrt(0.64) = 0.80  # $64 saved
money = 1.00 → sqrt(1.00) = 1.00  # $100 saved (cap)
```

**Why sqrt (diminishing returns)**:

Linear wealth would incentivize endless grinding:

```python
# Linear (bad)
money = 0.90 → score = 0.90
money = 1.00 → score = 1.00  # +10% money = +10% score
# Incentive: work until last possible moment
```

Sqrt wealth has diminishing returns:

```python
# Sqrt (good)
money = 0.00 → score = 0.00
money = 0.25 → score = 0.50  # first $25 is very valuable
money = 0.64 → score = 0.80  # next $39 is moderately valuable
money = 1.00 → score = 1.00  # last $36 is marginally valuable
```

**Effect on behavior**:

- Saving some money matters a lot (0 → $25)
- Hoarding beyond "comfortable nest egg" has less marginal benefit
- Prevents late-life grinding (working when you should be enjoying retirement)

**Alternative curves** (configurable):

```yaml
wealth_curve: "linear"  # every dollar matters equally
wealth_curve: "log"     # even more aggressive diminishing returns
wealth_curve: "sqrt"    # balanced (default)
```

**WEP** that sqrt is the right default: high (~80%). Linear encourages pathological hoarding, log may undervalue savings too much.

#### Retirement vs Early Death (The 10× Multiplier)

The outcome of an episode drastically changes the terminal reward:

**Retirement** (age >= 1.0):

```python
terminal_reward = raw_score  # full value
```

**Early Death** (energy=0 or health=0 before max_age):

```python
terminal_reward = 0.1 * raw_score  # 90% penalty
```

**Example**:

```python
# Agent A: Dies at tick 600
ticks_survived = 600
lifespan_score = 600 / 1200 = 0.5
wellbeing_score = 0.6
wealth_score = sqrt(0.36) = 0.6

raw_score = 0.5*0.5 + 0.3*0.6 + 0.2*0.6 = 0.25 + 0.18 + 0.12 = 0.55
terminal_reward = 0.1 * 0.55 = 0.055  # early death penalty

# Agent B: Retires at tick 1200
ticks_survived = 1200
lifespan_score = 1200 / 1200 = 1.0
wellbeing_score = 0.6
wealth_score = sqrt(0.36) = 0.6

raw_score = 0.5*1.0 + 0.3*0.6 + 0.2*0.6 = 0.50 + 0.18 + 0.12 = 0.80
terminal_reward = 0.80  # no penalty

# Agent B earned 14.5× more terminal reward despite similar wellbeing/wealth
```

**Why this extreme multiplier**:

- Making it to retirement should be the primary goal
- Early death is a catastrophic failure, regardless of how much money you saved
- Teaches agents to prioritize long-term survival over short-term gains

**Human equivalent**: "You worked yourself to death at 40 with $100k in the bank" is not a success story, even though you were wealthy. "You retired at 65 with $10k, healthy and happy" is better.

---

### 2.3 The Age Bar: Natural Horizon Without Cheating

Most RL environments use arbitrary episode length:

```python
# Typical approach (external to world)
if steps >= max_steps:
    done = True  # "time's up!"
```

**Problems**:

- Not in-world (agent can't observe it approaching)
- Can't be explained ("why did my life suddenly end?")
- Can't be planned for (agent doesn't know episode length)

**Townlet's approach**: Age is a visible meter that advances naturally.

#### Age Bar Configuration

From `bars.yaml`:

```yaml
- name: "age"
  index: 8  # the 9th bar (adding to the standard 8)
  tier: "lifecycle"
  range: [0.0, 1.0]
  initial: 0.0
  base_depletion: 0.0  # doesn't passively decay
  
  terminal_condition:
    operator: ">="
    value: 1.0
    outcome: "retirement"
    
  description: "Life progression toward retirement"
```

#### How Age Advances

In the environment step function:

```python
def step(self, action):
    # Execute action, apply affordances, cascades, etc
    # ...
    
    # Advance age
    age_increment = 1.0 / self.config.retirement.max_age_ticks
    self.bars['age'] += age_increment
    
    # Check terminal conditions
    if self.bars['age'] >= 1.0:
        done = True
        outcome = "retirement"
    elif self.bars['energy'] <= 0 or self.bars['health'] <= 0:
        done = True
        outcome = "death"
```

**Example with max_age_ticks=1200**:

```python
age_increment = 1.0 / 1200 = 0.000833...

tick 0:    age = 0.000
tick 600:  age = 0.500  # halfway through life
tick 1199: age = 0.999
tick 1200: age = 1.000  # retirement!
```

#### Why This Works

**✅ Observable by the agent**:

```python
observation['bars'][8] = age  # agent can see they're aging
```

**✅ Allows planning**:

- Agent at age=0.3 can predict: "I have ~840 ticks left"
- Can adjust strategy: "accumulate savings early, reduce work late"

**✅ In-world narrative**:

- Not "episode ended arbitrarily"
- But "you lived your full natural lifespan"

**✅ Curriculum adjustable**:

```yaml
# L0-1: Short episodes for rapid iteration
retirement:
  max_age_ticks: 200

# L2-3: Medium episodes for economic learning
retirement:
  max_age_ticks: 500

# L4-8: Full episodes for long-horizon planning
retirement:
  max_age_ticks: 1200
```

**Human observer test**:

- ✅ Can a human know they're aging? YES (visible passage of time)
- ✅ Can a human plan for retirement? YES (they know it's coming)
- ❌ Does time suddenly stop for no reason? NO (age is in-world physics)

**WEP** that visible age bar is superior to hidden episode limits: very high (~95%).

---

### 2.4 Value Systems as Configuration

The terminal reward formula encodes a value system: what constitutes a "good life"?

By making weights configurable, Townlet lets researchers explore different moral frameworks.

#### Capitalist World (Wealth Priority)

```yaml
retirement:
  weights:
    lifespan: 0.3   # completing life matters less
    wellbeing: 0.1  # quality of life matters less
    wealth: 0.6     # money is 60% of score
  
  wealth_curve: "linear"  # every dollar matters equally
```

**Expected agent behavior**:

- Work constantly, even when tired/sad
- Minimize spending on recreation, social, comfort
- Tolerate low mood, poor hygiene, loneliness
- Accumulate maximum money for terminal payout
- "Die rich but miserable" becomes rational strategy

**Research questions**:

- Does the agent learn to exploit itself?
- What's the equilibrium wealth at retirement?
- Does personality (greed slider) matter more in this world?

#### Balanced World (Default)

```yaml
retirement:
  weights:
    lifespan: 0.5   # survival to retirement is primary
    wellbeing: 0.3  # quality of life matters
    wealth: 0.2     # financial security matters
  
  wealth_curve: "sqrt"  # diminishing returns
```

**Expected agent behavior**:

- Sustainable work-rest cycles
- Balanced spending on necessities + some comfort
- Moderate savings, not obsessive hoarding
- "Comfortable middle-class retirement" strategy

**Research questions**:

- What's the optimal work/leisure balance?
- How much money is "enough"?
- Does the agent discover preventative healthcare?

#### Hedonist World (Experience Priority)

```yaml
retirement:
  weights:
    lifespan: 0.2   # making it to old age is less important
    wellbeing: 0.7  # quality of life is 70% of score
    wealth: 0.1     # money barely matters
  
  wealth_curve: "sqrt"
```

**Expected agent behavior**:

- Minimal work, maximum leisure
- High spending on Bar, Recreation, social activities
- Live paycheck-to-paycheck
- Risk of death due to insufficient emergency funds
- "Live fast, die happy" strategy

**Research questions**:

- Does the agent still survive to retirement?
- What's the minimum viable income?
- Does social/mood optimization emerge?

#### Comparing Worlds

Researchers can run the same agent architecture in different value systems:

```bash
# Same brain, different value systems
townlet train --config configs/capitalist/ --seed 42
townlet train --config configs/balanced/ --seed 42
townlet train --config configs/hedonist/ --seed 42

# Compare outcomes
townlet analyze --runs runs/capitalist_* runs/balanced_* runs/hedonist_*
```

**Metrics to compare**:

- Mean terminal reward (which world scores highest?)
- Retirement rate (which world has more early deaths?)
- Wellbeing at retirement (which world produces happier agents?)
- Wealth distribution (which world has more inequality?)
- Time allocation (work vs leisure hours)

**Expected finding**: Agents learn different personalities/strategies under different scoring systems, even with identical initial conditions.

**WEP** that value systems meaningfully affect learned behavior: high (~85%).

---

### 2.5 Learning Dynamics: Credit Assignment Over Multiple Time Scales

With sparse per-tick reward and delayed terminal bonus, learning happens at multiple time scales.

#### First-Order Relationships (1-5 ticks)

These are directly observable cause-and-effect pairs:

```
Action: Bed
  Tick 0: energy = 0.3, health = 0.8, r = 0.24
  Tick 1: energy = 0.55 (direct effect +0.25)
  Tick 1: r = 0.55 × 0.8 = 0.44
  
  Learning signal: r increased immediately
  Lesson: "Bed → energy ↑ → reward ↑"
  Credit assignment: trivial (1 tick delay)
```

**Agents learn these quickly** because the feedback loop is tight. By episode 10-20, agents reliably use Bed when energy is low.

#### Second-Order Relationships (10-50 ticks)

These involve cascades — indirect effects that take time to manifest:

```
Action: Skip Fridge (save $4)
  Tick 0: satiation = 0.25, health = 0.8
  Tick 1-9: satiation drops to 0.19 (base depletion)
  Tick 10: satiation < 0.2 threshold
  Tick 10: cascade activates: low_satiation_hits_health
  Tick 11-30: health drops 0.010 per tick (cascade strength)
  Tick 30: health = 0.60 (lost 0.20 health)
  Tick 30: r drops from 0.60 to 0.45 (energy also dropped)
  
  Learning signal: r decreased gradually, 30 ticks after decision
  Lesson: "Not eating → health decay → reward decay"
  Credit assignment: moderate difficulty (30 tick delay)
```

**Agents learn these by mid-training** (episodes 50-200) once they've experienced enough starvation events to correlate "skip Fridge" → "feel bad later".

**World model helps**: If the agent trains a world model (Module B), it can predict "if I skip Fridge, my satiation will drop below 0.2, which I've observed triggers health decay."

#### Third-Order Relationships (50-200 ticks)

These involve multi-step economic loops:

```
Strategy: Work → Save Money → Use Money Later

  Tick 0-80: Go to Job repeatedly
    - Direct effect: money ↑ (0.0 → 0.8)
    - Direct cost: energy ↓, mood ↓
    - Immediate r: drops (working is tiring)
  
  Tick 80: Stop working, have $80 saved
  
  Tick 81-120: Use savings on recovery
    - Fridge (cost $4) → satiation maintained
    - Bed (free) → energy recovered
    - Shower (cost $2) → hygiene maintained
    - Recreation (cost $5) → mood recovered
  
  Tick 120-200: Maintain high bars without working
    - Energy high (from rest)
    - Health high (from food + prevented cascades)
    - Mood high (from recreation)
    - r stays high for 80 ticks
  
  Terminal: High lifespan, high wellbeing, moderate wealth
  Terminal reward: 0.75
  
  Learning signal: short-term pain (working) enables long-term gain (comfortable life)
  Lesson: "Money is instrumentally valuable for survival"
  Credit assignment: hard (80-200 tick delayed benefit)
```

**Agents learn these late in training** (episodes 500+) after experiencing many full episodes and learning the value of savings.

**This is why terminal reward matters**: Without the delayed terminal bonus, agents would just optimize immediate `r = energy × health`, never learning that work (which lowers immediate reward) enables better terminal outcomes.

#### Fourth-Order: Temporal Planning (100-500 ticks)

These involve scheduling around world constraints:

```
Observation: Job has operating_hours [9, 18]

  Naive strategy: Try to work whenever money is low
    - Wake up at 3am, energy low
    - Walk to Job
    - Try INTERACT
    - Action is masked (Job closed)
    - Energy wasted on commute
    - Can't work, money doesn't increase
    - Result: death by starvation
  
  Learned strategy: Plan work around operating hours
    - Wake up at 7am
    - Arrive at Job at 7:45am
    - WAIT until 8am (minimal energy cost)
    - Job opens at 9am
    - INTERACT for full 8-hour shift
    - Earn $80 efficiently
    - Result: survival with savings
  
  Learning signal: terminal reward is much higher with temporal planning
  Lesson: "Schedule matters as much as action choice"
  Credit assignment: very hard (must connect schedule across entire episode)
```

**Agents learn this gradually** (episodes 1000+) by experiencing failed work attempts and eventually correlating "time of day" with "action success."

**This is why the LSTM is required at L4+**: Temporal planning requires memory of when things worked before.

#### Fifth-Order: Social Coordination (Full Episode Scope in L6+)

These involve predicting and responding to other agents:

```
Scenario: Two agents need the same Job (capacity=1)

  Naive strategy: Both agents try to work simultaneously
    - Agent A: goes to Job
    - Agent B: goes to Job (same tick)
    - Contention resolution: Agent A wins (closer)
    - Agent B: wasted energy commuting, earns $0
    - Agent B: can't buy food
    - Agent B: dies of starvation
  
  Learned strategy: Agent B predicts Agent A's behavior
    - Module C (Social Model): sees Agent A "looks_poor" + "moving toward Job"
    - Predicts: Agent A will work this shift
    - Module D (Policy): selects alternate goal: go to Labor instead
    - Agent B: earns $48 at Labor (less than Job, but > $0)
    - Agent B: survives
  
  Terminal: Agent B gets lower score than Agent A, but > early death penalty
  
  Learning signal: social reasoning prevents catastrophic failure
  Lesson: "Predict competitors, choose non-contested resources"
  Credit assignment: extremely hard (requires correlating cues → intent → resource contention → terminal outcome)
```

**Agents learn this very late** (episodes 5000+, often requires CTDE pretraining for Module C).

**This is why L6+ is the research frontier**: Multi-agent credit assignment over full episode lengths with partial observability is an unsolved hard problem in RL.

---

### 2.6 Why This Reward Architecture Enables Real Research

The combination of sparse per-tick + rich terminal produces three research-valuable properties:

#### 1. Agents Learn World Dynamics, Not Reward Hacking

**Dense shaping version** (what we're avoiding):

```python
r = 0.0
if action == "eat": r += 0.1
if action == "sleep": r += 0.1
# etc...
```

**What the agent learns**: "Maximize action counts"
**What the agent doesn't learn**: "Food prevents starvation cascades"

**Townlet version**:

```python
r = energy × health  # just reflect internal state
```

**What the agent learns**: "Cascades are real, I must prevent them"
**Emergent understanding**: Agent discovers the causal graph (Fridge → satiation → health)

#### 2. Policies Transfer Across World Variants

**Dense shaping**: If you change world physics (make food more expensive), you must retune all the shaping coefficients or the policy breaks.

**Sparse reward**: If you change world physics, the reward function stays the same (`r = energy × health`). The agent must re-learn strategies, but the objective is still "survive well."

**Example**:

```yaml
# Baseline world
affordances:
  - id: "fridge"
    costs: [{ meter: "money", amount: 0.04 }]  # $4

# Austerity world (just change one line)
affordances:
  - id: "fridge"
    costs: [{ meter: "money", amount: 0.08 }]  # $8 (doubled)
```

With sparse rewards, you can directly compare:

- How much more does the agent work?
- Does it skip comfort affordances?
- Does the starvation rate increase?

The reward function didn't change, so behavioral changes are due to world physics, not reward tuning artifacts.

#### 3. Terminal Bonus Enables Value Alignment Research

The configurable terminal reward is essentially a **learned value alignment** mechanism:

```yaml
# What does this society value?
retirement:
  weights:
    lifespan: 0.5
    wellbeing: 0.3
    wealth: 0.2
```

Agents internalize these values through RL. Change the weights, and agents learn different priorities.

**This is actual alignment research** — not "stop the AI from saying bad words" but "can the AI learn to value what we configure it to value?"

**Research question**: Do agents trained under different value systems generalize their values to new situations? Or do they just memorize "maximize terminal score" without understanding the underlying values?

**WEP** that this reward architecture enables genuine scientific discoveries: very high (~90%).

---

**End of Section 2**

---

## SECTION 3: CURRICULUM DESIGN

### 3.1 Curriculum Philosophy

The Townlet curriculum (L0–L8) progressively removes assumptions, taking agents from god-mode tutorial to fully realistic multi-agent coordination.

**Core principle**: Each level removes exactly one simplifying assumption, forcing agents to develop one new capability.

**Pedagogical scaffold**: Early levels (L0-3) provide "training wheels" — full observability, perfect knowledge — to teach fundamental relationships before adding complexity.

**Realism progression**: Later levels (L4-8) remove scaffolding piece by piece until agents operate under human-realistic constraints.

#### The Assumption Removal Ladder

| Level | Removed Assumption | New Capability Required | Architecture |
|-------|-------------------|------------------------|--------------|
| **L0** | None (tutorial) | "Affordances fix bars" | SimpleQNetwork |
| **L1** | None | "Money enables survival" | SimpleQNetwork |
| **L2** | None | "Multi-resource loops" | SimpleQNetwork |
| **L3** | None | "Optimize quality of life" | SimpleQNetwork |
| **L4** | Perfect geography | Exploration + memory | LSTM required |
| **L5** | — | Temporal planning | LSTM + time input |
| **L6** | Omniscience about others | Social reasoning via cues | Module A-D |
| **L7** | — | Rich social prediction | Module A-D |
| **L8** | Pre-shared language | Emergent communication | Module A-D + channel |

**Design insight**: You can't learn social reasoning (L6) until you can navigate (L4), and you can't navigate without knowing what affordances do (L0-3). The order is pedagogically necessary, not arbitrary.

---

### 3.2 Level-by-Level Breakdown

#### L0: The "Two Unsolvable Problems"

**World configuration**:

```yaml
grid_size: [5, 5]
affordances: ["Bed"]  # only Bed exists
observability: "full"  # agent sees everything
initial_money: 0.50  # $50
max_age_ticks: 200   # short episodes for rapid learning
```

**What the agent sees**:

```python
observation = {
    'bars': [1.0, 1.0, 1.0, 0.5, 0.7, 1.0, 1.0, 0.5],  # all 8 bars visible
    'position': [2, 2],
    'all_affordance_locations': {'Bed': [1, 1]},  # perfect map
    'grid_size': [5, 5]
}
```

**The learning experience**:

1. Energy drops (base_depletion = 0.005 per tick)
2. `r = energy × health` drops
3. RND (curiosity) drives agent to explore
4. Agent discovers Bed at (1,1)
5. Tries INTERACT → energy restored → `r` jumps up
6. Massive positive TD error
7. Agent learns: "Bed → energy ↑"

8. Agent hammers this new policy: Bed, Bed, Bed
9. After 10 uses: money = $0 (Bed costs $5 per use)
10. Can't afford Bed anymore
11. Energy drops, agent dies
12. Terminal: early death penalty, very low score

13. Simultaneously: satiation dropping
14. Satiation < 0.2 → cascade activates
15. Health decaying faster
16. No way to stop it (Fridge doesn't exist yet)
17. Agent dies of starvation even if it has energy

**The "Aha!" moment**: "I've solved energy, but now I face two new problems: I'll either die of starvation (no Fridge) or die when I run out of money (can't afford Bed)."

**Key learning**:

- `V(low_energy)` is terrible
- `V(low_money)` is terrible
- `V(low_satiation)` is terrible
- But only one solution (Bed) exists
- This is unsolvable by design

**Why it's L0**: Teaches the bare minimum — "affordances affect bars, bars affect reward."

**Human observer exception**: Full observability is scaffolding. A human learning a new town would be given a map. This is pedagogically honest.

**WEP** that agents learn affordance semantics by episode 20: very high (~95%).

---

#### L1: The "Economic Problem"

**World configuration**:

```yaml
grid_size: [5, 5]
affordances: ["Bed", "Job"]  # ← Job added
observability: "full"
initial_money: 0.50
max_age_ticks: 200
```

**What's new**: Job affordance provides income

```yaml
- id: "job"
  interaction_type: "multi_tick"
  required_ticks: 8
  effects_per_tick:
    - { meter: "money", amount: 0.10 }  # earn $10/tick
    - { meter: "energy", amount: -0.05 }  # costs energy
  operating_hours: [9, 18]
```

**The learning experience**:

1. Agent inherits L0 knowledge: "Bed fixes energy"
2. Agent runs out of money (same as L0)
3. Now in `low_money` state (previously hopeless)
4. RND drives exploration again
5. Agent discovers Job at (6,6)
6. Tries INTERACT → money increases!
7. Agent learns second sub-policy: "Job → money ↑"

8. Agent creates stable loop: Job → earn money → Bed → restore energy → Job
9. This is economically stable (can pay for Bed indefinitely)
10. Agent still dies of starvation (satiation still dropping)
11. But survives much longer than L0

**The "Aha!" moment**: "The Job solves the money problem! I can now afford the Bed forever. But I still can't solve the starvation problem."

**Key learning**:

- Money is instrumentally valuable (enables affordance access)
- Work-rest cycles are necessary
- But still can't reach retirement (satiation cascade kills them)

**Why it's L1**: Isolates economic reasoning. Agent masters money/energy loop before tackling food.

**Expected behavior by episode 100**: Reliable work-rest cycles, but 100% death by starvation.

---

#### L2: The "First Stable Loop"

**World configuration**:

```yaml
grid_size: [5, 5]
affordances: ["Bed", "Job", "Fridge"]  # ← Fridge added
observability: "full"
initial_money: 0.50
max_age_ticks: 500  # longer episodes now survivable
```

**What's new**: Fridge prevents starvation

```yaml
- id: "fridge"
  interaction_type: "instant"
  costs:
    - { meter: "money", amount: 0.04 }  # $4
  effects:
    - { meter: "satiation", amount: 0.40 }  # +40% satiation
```

**The learning experience**:

1. Agent has L1 knowledge: reliable Job ↔ Bed loop
2. Satiation drops as always
3. Cascade activates, health starts decaying
4. RND drives agent to new affordance: Fridge
5. Tries INTERACT → satiation jumps up
6. Cascade stops → health stabilizes
7. Agent learns third sub-policy: "Fridge → satiation ↑ → health stable"

8. Agent combines all three sub-policies:
   - Low energy → Bed
   - Low money → Job
   - Low satiation → Fridge

9. This three-way loop is infinitely stable
10. Agent can now reach retirement (age = 1.0)
11. Terminal reward: moderate score (survived, but minimal wellbeing)

**The "Aha!" moment**: "The Fridge solves the final problem! I can now survive indefinitely."

**Key learning**:

- Food prevents health decay (second-order relationship)
- Must balance three resources: energy, money, satiation
- Stable survival loop: Job → Fridge → Bed → repeat

**Why it's L2**: This is the minimum viable life. A "competent but miserable" agent can survive here indefinitely.

**Expected behavior by episode 200**: 70%+ retirement rate, but low terminal scores (wellbeing suffers).

**This is the baseline**: Any agent that can't master L2 is not viable for advanced levels.

---

#### L3: The "Full Simulation (Small Grid)"

**World configuration**:

```yaml
grid_size: [5, 5]  # still small, fully visible
affordances: ["Bed", "Job", "Fridge", "Shower", "Bar", "Recreation", 
              "Doctor", "Hospital", "Gym", "Labor", "Park", ...]  # all 15
observability: "full"
initial_money: 0.50
max_age_ticks: 1000  # full-length episodes
```

**What's new**: All tertiary affordances (mood, social, hygiene, fitness)

**The learning experience**:

1. Agent has L2 mastery: Job-Fridge-Bed loop
2. But retirement scores are still low
3. Why? Tertiary cascades are active:
   - Low hygiene → mood drops → energy drops faster
   - Low social → mood drops → energy drops faster
   - Low fitness → health drops faster (via modulation)

4. Agent notices: "I'm going to Bed more often than I used to"
5. World model learning: "Something is making energy decay faster"
6. Exploration: tries Shower → hygiene ↑
7. Observation: energy decay rate returns to normal
8. Learning: "Hygiene maintains energy efficiency"

9. Similar discovery for Bar (social), Gym (fitness), Recreation (mood)
10. Agent learns optimization: "These affordances don't directly give reward, but they prevent cascades, which makes survival cheaper"

**The "Aha!" moment**: "Going to the Bar isn't pointless fun — it's instrumentally useful because it prevents mood decay, which prevents energy decay, which means I need to work less."

**Key learning**:

- Tertiary meters matter (hygiene, social, fitness, mood)
- They affect survival through cascades (third-order relationships)
- There's an optimal allocation of time/money across all affordances
- Quality of life and survival efficiency are linked

**Why it's L3**: Teaches optimization and multi-objective reasoning before adding navigation/social complexity.

**Expected behavior by episode 500**:

- 80%+ retirement rate
- Higher terminal scores (wellbeing component improves)
- Agents discover "balanced life" strategies

**Graduation criteria**: Agent can reliably reach retirement with terminal score > 0.65.

---

#### L4: The "Fog of War" (Partial Observability)

**World configuration**:

```yaml
grid_size: [8, 8]  # ← larger grid
visibility_radius: 2  # ← 5×5 window (agent can see 2 tiles in each direction)
affordances: [all 15]
observability: "partial"  # ← KEY CHANGE
initial_money: 0.50
max_age_ticks: 1200
```

**What the agent sees NOW**:

```python
observation = {
    'bars': [energy, health, ...],  # still see own state
    'position': [x, y],  # still know where they are
    'visible_grid': np.array([5, 5, N_AFFORDANCES]),  # ← only 5×5 window
    'time_of_day': hour,
    # ❌ NO all_affordance_locations (removed!)
}
```

**The learning experience**:

1. Agent tries L3 policy: "Go to Bed when energy low"
2. But: can't see Bed (it's outside the 5×5 window)
3. Policy fails: wanders randomly
4. Energy drops to 0 → dies
5. Terminal: massive negative reward (early death)

6. RND drives exploration (random curiosity-driven movement)
7. Eventually stumbles upon Bed at (2,1)
8. LSTM records: "At tick 50, I saw Bed at (2,1)"
9. Memory persists in hidden state

10. Later (tick 150): energy is low again
11. Agent is at position (7,6), can't see Bed
12. But LSTM's hidden state contains "Bed is at (2,1)"
13. Policy uses memory: "Navigate toward remembered location (2,1)"
14. Arrives at Bed, survives

**The "Aha!" moment**: "My policy is useless without memory. I must explore to build a mental map, then navigate from memory."

**Key learning**:

- Exploration is essential (RND-driven curiosity)
- LSTM is required (spatial memory)
- Mental map must be maintained across hundreds of ticks
- Navigation from memory is a new skill

**Architecture requirement**: SimpleQNetwork can no longer solve this. Must use RecurrentSpatialQNetwork (LSTM core).

**Why it's L4**: First removal of scaffolding. Agents must now discover the world themselves.

**Expected behavior**:

- Episodes 0-500: Terrible performance (learning to explore)
- Episodes 500-2000: Improving (building reliable mental maps)
- Episodes 2000+: Near-L3 performance (memory compensates for limited vision)

**Graduation criteria**: Agent can navigate to any affordance from any starting position, 80%+ success rate.

**Human observer validation**: "Can I see the whole town from any location?" NO (5×5 vision is realistic).

---

#### L5: The "9-to-5" (Temporal Constraints)

**World configuration**:

```yaml
grid_size: [8, 8]
visibility_radius: 2
affordances: [all 15, with operating_hours enforced]  # ← KEY CHANGE
observability: "partial"
temporal_mechanics: true  # ← enable time-of-day
initial_money: 0.50
max_age_ticks: 1200
```

**What's new**: Affordances have operating hours, and they're enforced

```yaml
- id: "job"
  operating_hours: [9, 18]  # 9am-6pm only

- id: "bar"
  operating_hours: [18, 28]  # 6pm-4am (wraps around)
```

**What the agent sees NOW**:

```python
observation = {
    'bars': [...],
    'position': [x, y],
    'visible_grid': [...],
    'time_of_day': 14,  # ← NEW: current hour [0-23]
}
```

**The learning experience**:

1. Agent has L4 mastery: can navigate from memory
2. At 3am: money is low, energy is moderate
3. Agent navigates to Job (from memory)
4. Tries INTERACT
5. Action is masked! (Job is closed at 3am)
6. Agent tries again... and again... stuck in loop
7. Wasting energy on failed interactions
8. Eventually runs out of energy and dies

9. Learning: "Job only works during certain hours"
10. Must correlate time_of_day with action success
11. Policy learns: "Don't go to Job at 3am, wait until 9am"

12. New strategy emerges: arrive at Job at 7am, WAIT until it opens at 9am
13. WAIT costs minimal energy (0.001 vs 0.005 for moving)
14. Efficient scheduling learned

**The "Aha!" moment**: "The WAIT action is strategic. Arriving early and waiting is better than arriving exactly at open (which requires perfect timing) or arriving late (which wastes work hours)."

**Key learning**:

- Time-of-day matters for action success
- Scheduling is as important as action selection
- WAIT is not useless, it's an efficiency tool
- Policy becomes: f(state, memory, time)

**Architecture requirement**: Still RecurrentSpatialQNetwork, but now time_of_day must be input.

**Why it's L5**: Teaches temporal planning without adding social complexity.

**Expected behavior**:

- Episodes 0-500: Many deaths due to bad timing
- Episodes 500-1500: Learning to check time before acting
- Episodes 1500+: Efficient scheduling (arrive early, wait, maximize work hours)

**Graduation criteria**: Agent can reliably schedule around operating hours, 75%+ retirement rate.

**Human observer validation**: "Can I know what time it is?" YES (clocks exist).

---

#### L6: The "Social Game" (Multi-Agent, Cues Only)

**World configuration**:

```yaml
grid_size: [8, 8]
visibility_radius: 2
affordances: [all 15, with capacity: 1 for contested resources]  # ← KEY CHANGE
num_agents: 2+  # ← multiple agents
observability: "partial + social"  # ← cues visible
temporal_mechanics: true
max_age_ticks: 1200
```

**What's new**: Other agents exist, you can see limited info about them

**What the agent sees NOW**:

```python
observation = {
    'bars': [...],
    'position': [x, y],
    'visible_grid': [...],
    'time_of_day': 14,
    
    # ← NEW: social information
    'other_agents_in_window': {
        'positions': [[4, 5], [6, 3]],  # where they are
        'public_cues': [
            ['looks_tired', 'at_job'],  # agent 2's cues
            ['looks_poor', 'looks_sad']  # agent 3's cues
        ],
        'recent_actions': [[2, 2, 4], [1, 1, 1]]  # action history
    }
}
```

**What the agent does NOT see**:

```python
# ❌ NO direct access to other agents' bars
# ❌ NO direct access to other agents' goals
# ❌ NO direct access to other agents' intentions
```

**Cues configuration** (from cues.yaml):

```yaml
cues:
  - id: "looks_tired"
    trigger: { bar: "energy", operator: "<", threshold: 0.3 }
    
  - id: "looks_poor"
    trigger: { bar: "money", operator: "<", threshold: 0.2 }
    
  - id: "at_job"
    trigger: { current_affordance: "Job" }
```

**The learning experience**:

**Scenario: The Race to the Job**

1. Agent A (you): L5 mastery, knows to go to Job at 9am
2. Agent B (rival): also learned to go to Job at 9am
3. Both agents arrive at Job at 8:59am
4. Both agents try INTERACT at 9am
5. Contention resolution: Agent B is 1 tile closer → Agent B wins
6. Agent A: INTERACT fails (Job is occupied, capacity=1)
7. Agent A: tries again... fails again... stuck
8. Agent A: can't earn money this shift
9. Agent A: can't afford food later
10. Agent A: dies of starvation
11. Terminal: massive penalty (early death)

**Retry with social reasoning:**

1. Agent A observes (at 8am):
   - Agent B position: (5, 5)
   - Agent B cues: ['looks_poor', 'moving_toward_industrial_zone']

2. Module C (Social Model) prediction:

   ```python
   # Trained via CTDE to predict: cues → likely goal
   predicted_goal_dist = social_model(['looks_poor', 'at_industrial_zone'])
   # Output: {"go_to_job": 0.85, "go_to_labor": 0.10, "go_to_bed": 0.05}
   ```

3. Module D (Hierarchical Policy) reasoning:

   ```python
   # Meta-controller:
   my_goal = "EARN_MONEY"
   
   # Evaluate options:
   option_1 = "go_to_job"
   world_model_prediction = "Agent B will also go to Job (85% confidence)"
   world_model_prediction += "I will lose contention (Agent B is closer)"
   expected_reward = -100  # early death if I can't earn money
   
   option_2 = "go_to_labor"
   world_model_prediction = "Labor is uncontested"
   expected_reward = +15  # earn less money, but > $0
   
   # Choose option 2
   final_action = "navigate_to_labor"
   ```

4. Agent A goes to Labor instead
5. Earns $48 (less than Job's $80, but > dying)
6. Survives

**The "Aha!" moment**: "I must reason about what other agents will do, and choose non-contested resources when I'll lose the race."

**Key learning**:

- Other agents are strategic competitors
- Public cues → predicted intentions (Module C)
- Predicted intentions → strategic response (Module D)
- Policy becomes: f(state, memory, time, belief_about_others)

**Architecture requirement**: Full Module A-D stack

- Module A: Perception (process cues into belief)
- Module B: World Model (predict outcomes)
- Module C: Social Model (infer intent from cues)
- Module D: Hierarchical Policy (meta-controller picks goals, controller picks actions)

**Why it's L6**: First genuine social reasoning. Theory of Mind emerges from cue interpretation.

**Expected behavior**:

- Episodes 0-1000: Frequent deaths due to contention (no social reasoning)
- Episodes 1000-3000: Learning cue correlations (CTDE pretraining helps)
- Episodes 3000+: Strategic resource selection (avoid contested affordances)

**Graduation criteria**: In contested scenarios, agent chooses alternate resources 60%+ of the time.

**Human observer validation**:

- ✅ "Can I see someone looks poor?" YES (visible clothing, demeanor)
- ❌ "Can I know their exact money value?" NO (internal state)

---

#### L7: Rich Social Reasoning (More Cues)

**World configuration**: Same as L6, but richer cue set

```yaml
cues:  # expanded from L6
  - "looks_tired"
  - "looks_energetic"
  - "looks_sad"
  - "looks_happy"
  - "looks_poor"
  - "looks_wealthy"
  - "looks_sick"
  - "looks_healthy"
  - "at_job"
  - "at_hospital"
  - "at_bar"
  - "carrying_items"  # behavioral cues
  - "rushing"
  - "resting"
```

**What's new**: Agents can infer more detailed state from richer cue vocabulary

**The learning experience**:

1. Agent A sees Agent B: ['looks_sick', 'at_hospital', 'looks_poor']
2. Module C prediction:

   ```python
   predicted_state = {
       'health': ~0.25,  # "looks_sick"
       'money': ~0.15,   # "looks_poor"
       'current_goal': 'SURVIVAL'  # "at_hospital" while sick
   }
   ```

3. Strategic reasoning:
   - "Agent B is in survival crisis"
   - "They will prioritize hospital over work"
   - "Job will likely be available"
   - Decision: go to Job confidently

**Key learning**: More cues → better predictions → better strategy

**Why it's L7**: Incremental improvement over L6, not a new capability. Validates that cue richness scales.

**Expected behavior**: Higher retirement rates than L6 (better predictions → better resource allocation).

---

#### L8: Emergent Communication (Family Channel)

**World configuration**:

```yaml
grid_size: [8, 8]
visibility_radius: 2
affordances: [all 15, capacity: 1]
num_agents: 2+
observability: "partial + social"
temporal_mechanics: true
family_formation: true  # ← NEW: agents can form families
family_comm_channel: true  # ← NEW: in-group signaling
max_age_ticks: 1200
```

**What's new**: Families can communicate via abstract signals

**What the agent sees NOW**:

```python
observation = {
    'bars': [...],
    'position': [...],
    'visible_grid': [...],
    'time_of_day': 14,
    'other_agents_in_window': {...},  # same as L6/L7
    
    # ← NEW: family communication
    'family_comm_channel': [123, 0, 0],  # int64 signals from family members
    'family_id': 'family_42',
    
    # ❌ NO semantic dictionary provided
    # ❌ NO pre-shared meaning of signals
}
```

**Available actions NOW**:

```python
actions = [
    UP, DOWN, LEFT, RIGHT,
    INTERACT,
    WAIT,
    SET_COMM_CHANNEL(value)  # ← NEW: broadcast int64 signal to family
]
```

**The learning experience**:

**Scenario: The Coordination Problem**

1. Parent (Agent A) is at Job, money is full
2. Child (Agent C) is at home, money is low
3. No way to tell child "don't waste energy walking here, Job is taken"
4. Child walks to Job (wastes energy)
5. Job is occupied (Parent is using it)
6. Child can't work
7. Child doesn't earn money
8. Child dies of starvation later

**Learning emergent communication (over 1000s of episodes):**

1. Parent's Module D learns (via random exploration):
   - "When my money is full, try action: SET_COMM_CHANNEL(123)"
   - This action has high correlation with child surviving
   - (Because it happened to correlate with successful episodes during exploration)

2. Child's Module C learns (via CTDE pretraining, then online refinement):
   - Input: family_comm_channel = [123, 0]
   - Ground truth (during training): Parent's money = high
   - Supervised learning: "signal 123 → parent money high"

3. Child's Module D learns:
   - "If signal=123, don't select goal: go_to_job"
   - "Instead, select goal: go_to_recreation (free time)"
   - Child's survival rate increases when respecting signal

4. Over many episodes:
   - Parent reliably sends 123 when Job is taken
   - Child reliably interprets 123 as "Job unavailable"
   - Coordination success rate increases
   - Family terminal reward > solo agent terminal reward

**The "Aha!" moment**: "The signal '123' has negotiated meaning. It wasn't given by designers, it emerged from correlation learning."

**Key learning**:

- Emergent communication via grounded correlation
- Signals start arbitrary (123 could have been 456)
- Meaning emerges from: signal → correlated state → coordinated action → better outcomes
- This is proto-language

**Architecture requirement**:

- Module A: process family_comm_channel as input
- Module C: predict state from signals (CTDE training)
- Module D: controller action space includes SET_COMM_CHANNEL

**Why it's L8**: This is the research frontier. Emergent communication without pre-shared semantics is an unsolved problem in multi-agent RL.

**Expected behavior**:

- Episodes 0-5000: Random signals, no coordination
- Episodes 5000-15000: Some signals start correlating with outcomes
- Episodes 15000+: Stable protocols emerge (e.g., 123="job taken", 456="danger", 789="food available")

**Graduation criteria**: Family coordination gain > 20% (family agents outperform solo agents with statistical significance).

**Evaluation metrics**:

- Signal diversity (number of unique signals used)
- Signal stability (P(same signal → same outcome))
- Semantic alignment (mutual information I(signal; state))
- Family coordination gain (terminal reward family vs solo)

**Human observer validation**:

- ✅ "Can I hear my family member's signal?" YES (acoustic/visual signal)
- ❌ "Do I know what '123' means without learning?" NO (must learn via correlation)

**Open research questions**:

- Do protocols transfer when families churn?
- Do different families develop different "dialects"?
- Can we decode learned semantics via causal interventions?

**WEP** that true emergent communication will emerge by episode 20000: moderate (~60%). This is genuinely hard.

---

### 3.3 Observability Progression Table

| Level | Grid | Observability | Scaffolding | Human Equivalent |
|-------|------|--------------|-------------|------------------|
| **L0-3** | 5×5 | Full | All affordances visible | "Here's a map of town" |
| **L4-5** | 8×8 | Partial (5×5 window) | None | "Explore and remember" |
| **L6-7** | 8×8 | Partial + social cues | None | "Read body language" |
| **L8** | 8×8 | Partial + cues + channel | None | "Learn language" |

---

### 3.4 Architecture Requirements Table

| Level | Minimum Architecture | Why |
|-------|---------------------|-----|
| **L0-3** | SimpleQNetwork (feedforward) | Full observability, reactive policies sufficient |
| **L4-5** | RecurrentSpatialQNetwork (LSTM) | Partial obs requires memory for navigation |
| **L6-7** | Full Module A-D stack | Social reasoning requires cue interpretation |
| **L8** | Module A-D + comm channel | Emergent communication requires signal grounding |

**Module breakdown**:

- **Module A (Perception)**: Builds belief state from partial observations
- **Module B (World Model)**: Predicts outcomes of candidate actions
- **Module C (Social Model)**: Infers other agents' states/goals from cues
- **Module D (Hierarchical Policy)**: Meta-controller (pick goal) + controller (pick action)

---

## SECTION 4: OBSERVATION SPACE SPECIFICATION

### 4.1 Design Principle Recap: Human Observer Test

Every element of the observation tensor must pass the human observer test:

**At inference time**:

- ✅ Can a human see/know this? → Include in observation
- ❌ Would this require telepathy/omniscience? → Exclude from observation

**At training time** (CTDE exception):

- Module C can receive ground truth labels for supervised learning
- But at inference, Module C only sees public cues

**Curriculum progression**:

- L0-3: Pedagogical scaffolding (full observability, perfect map)
- L4-5: Realistic vision (partial observability, memory required)
- L6-7: Realistic social perception (cues only, no telepathy)
- L8: Realistic communication (signals without semantics)

---

### 4.2 Observation Space by Curriculum Level

| Level | Spatial | Meters | Temporal | Social | Comm | Observation Dim |
|-------|---------|--------|----------|--------|------|-----------------|
| **L0-3** | One-hot grid | 8 bars | - | - | - | grid² + 8 + (types+1) |
| **L4-5** | 5×5 window + pos | 8 bars | sin/cos time + progress | - | - | 25 + 2 + 8 + (types+1) + 3 |
| **L6** | 5×5 window + pos | 8 bars | sin/cos time + progress | Positions | - | 25 + 2 + 8 + (types+1) + 3 + 2M |
| **L7** | 5×5 window + pos | 8 bars | sin/cos time + progress | Pos + cues | - | 25 + 2 + 8 + (types+1) + 3 + (2+K)M |
| **L8** | 5×5 window + pos | 8 bars | sin/cos time + progress | Pos + cues | Family channel | 25 + 2 + 8 + (types+1) + 3 + (2+K)M + F |

Where:

- `grid`: grid_size (e.g., 5 for L0-3, 8 for L4+)
- `types`: num_affordance_types (typically 15)
- `M`: max_visible_agents (e.g., 5)
- `K`: num_cue_types (e.g., 12 for L7)
- `F`: max_family_size (typically 2-3)

---

### 4.3 Full Observability (L0-3)

**Used in**: Curriculum levels 0-3 (small 5×5 grid, tutorial mode)

**Tensor structure**:

```python
observation = torch.cat([
    grid_encoding,         # [num_agents, grid_size²]  - one-hot position
    meters,               # [num_agents, 8]            - all bars
    affordance_encoding,  # [num_agents, types + 1]    - current affordance
], dim=1)

# Total dimension: grid_size² + 8 + (num_affordance_types + 1)
# Example (5×5 grid, 15 affordances): 25 + 8 + 16 = 49
```

#### Component Breakdown

**Grid Encoding** `[num_agents, grid_size²]`

```python
# One-hot encoding of agent position in flattened grid
# Example: agent at position (2, 3) in 5×5 grid
flat_index = y * grid_size + x = 3 * 5 + 2 = 17
grid_encoding = [0, 0, ..., 0, 1, 0, ..., 0]  # 1 at index 17
#                           ↑ position 17
```

**Purpose**: Agent knows exactly where it is on the grid
**Human equivalent**: "I'm at coordinates (2, 3)"

**Meters** `[num_agents, 8]`

```python
meters = [
    energy,      # index 0, range [0, 1]
    satiation,   # index 1
    mood,        # index 2
    hygiene,     # index 3
    social,      # index 4
    fitness,     # index 5
    health,      # index 6
    money,       # index 7 (normalized: 0.50 = $50)
]
```

**Purpose**: Agent has full introspection of internal state
**Human equivalent**: "I know how tired, hungry, happy I am"

**Affordance Encoding** `[num_agents, num_affordance_types + 1]`

```python
# One-hot encoding of which affordance (if any) agent is standing on
# Example: agent at Job location
affordance_encoding = [
    0,  # Bed
    0,  # Fridge
    1,  # Job ← agent is here
    0,  # Hospital
    ...,
    0,  # (num_affordance_types entries)
    0   # "none" (not on any affordance)
]

# Example: agent not on any affordance
affordance_encoding = [0, 0, 0, ..., 0, 1]  # last element = 1
```

**Purpose**: Agent knows what affordance it's standing on
**Human equivalent**: "I'm currently at the Job building"

**Scaffolding note**: In L0-3, agents also receive perfect knowledge of affordance locations (via `affordances` dict), but this is NOT in the observation tensor itself. This is why SimpleQNetwork suffices — the agent can directly see where everything is without needing memory.

---

### 4.4 Partial Observability (L4-5)

**Used in**: Curriculum levels 4-5 (8×8 grid, fog of war)

**Tensor structure**:

```python
observation = torch.cat([
    local_grid,           # [num_agents, window²]      - 5×5 local view
    normalized_position,  # [num_agents, 2]            - where am I?
    meters,              # [num_agents, 8]            - all bars
    affordance_encoding, # [num_agents, types + 1]    - current affordance
    time_sin,            # [num_agents, 1]            - sin(2π * hour/24)
    time_cos,            # [num_agents, 1]            - cos(2π * hour/24)
    progress,            # [num_agents, 1]            - interaction ticks / 10
], dim=1)

# Total dimension: 25 + 2 + 8 + 16 + 1 + 1 + 1 = 54 (for 15 affordance types)
```

#### Component Breakdown

**Local Grid** `[num_agents, window_size²]`

```python
# 5×5 window centered on agent (vision_range = 2)
window_size = 2 * vision_range + 1 = 5

# Grid is binary: 1 = affordance visible, 0 = empty or out of bounds
# Example: agent at (4, 4) in 8×8 world
#
#   World coordinates:        Local window:
#   (2,2) (3,2) (4,2) (5,2) (6,2)    [0 0 1 0 0]  ← row 0 (world y=2)
#   (2,3) (3,3) (4,3) (5,3) (6,3)    [0 0 0 0 0]  ← row 1
#   (2,4) (3,4) (4,4) (5,4) (6,4)    [0 0 * 0 1]  ← row 2 (* = agent position)
#   (2,5) (3,5) (4,5) (5,5) (6,5)    [0 1 0 0 0]  ← row 3
#   (2,6) (3,6) (4,6) (5,6) (6,6)    [0 0 0 0 0]  ← row 4
#
# Flattened: [0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,1, 0,1,0,0,0, 0,0,0,0,0]
```

**Construction algorithm**:

```python
for dy in range(-vision_range, vision_range + 1):  # -2 to +2
    for dx in range(-vision_range, vision_range + 1):
        world_x = agent_pos[0] + dx
        world_y = agent_pos[1] + dy
        
        # Check bounds
        if 0 <= world_x < grid_size and 0 <= world_y < grid_size:
            # Check if affordance exists at (world_x, world_y)
            has_affordance = any(
                aff_pos == [world_x, world_y] 
                for aff_pos in affordances.values()
            )
            
            if has_affordance:
                local_idx = (dy + vision_range) * window_size + (dx + vision_range)
                local_grid[local_idx] = 1.0
```

**Purpose**: Agent sees only immediate vicinity, must explore to discover world
**Human equivalent**: "I can see 2 tiles in each direction, rest is unknown"

**Normalized Position** `[num_agents, 2]`

```python
# Absolute position scaled to [0, 1]
normalized_position = [
    x / (grid_size - 1),  # e.g., 4 / 7 = 0.571 in 8×8 grid
    y / (grid_size - 1)   # e.g., 4 / 7 = 0.571
]
```

**Purpose**: Agent knows where it is in absolute coordinates (has a compass)
**Human equivalent**: "I'm at coordinates (4, 4) in this 8×8 town"
**Why needed**: With partial observability, agent can't infer absolute position from local grid alone

**Temporal Features** `[num_agents, 3]`

```python
# Time of day (cyclical encoding)
angle = (time_of_day / 24.0) * 2 * π
time_sin = sin(angle)  # e.g., sin(2π * 14/24) for 2pm
time_cos = cos(angle)  # e.g., cos(2π * 14/24)

# Interaction progress (if doing multi-tick action)
progress = ticks_completed / 10.0  # e.g., 3/10 = 0.3 if 3 ticks into Job
```

**Why sin/cos encoding**:

```python
# Bad: linear time
time = 23  # 11pm
# Network can't know that 23 is close to 0 (midnight)

# Good: cyclical encoding
sin(2π * 23/24) ≈ -0.26
cos(2π * 23/24) ≈  0.97
sin(2π * 0/24)  =  0.00
cos(2π * 0/24)  =  1.00
# These are close in embedding space!
```

**Purpose**: Agent can plan around operating hours, understand progress through multi-tick actions
**Human equivalent**: "It's 2pm, and I'm 30% done with this 8-hour work shift"

**Architecture requirement**: LSTM or GRU to maintain spatial memory across ticks

---

### 4.5 Social Observability (L6-7) — NEEDS IMPLEMENTATION

**Used in**: Curriculum levels 6-7 (multi-agent, competition/cooperation)

**Tensor structure**:

```python
observation = torch.cat([
    # All L4-5 features
    local_grid,           # [num_agents, 25]
    normalized_position,  # [num_agents, 2]
    meters,              # [num_agents, 8]
    affordance_encoding, # [num_agents, 16]
    time_sin,            # [num_agents, 1]
    time_cos,            # [num_agents, 1]
    progress,            # [num_agents, 1]
    
    # NEW: Social features
    other_agent_positions,  # [num_agents, max_visible * 2]
    other_agent_cues,       # [num_agents, max_visible * num_cue_types]
], dim=1)

# Total dimension: 54 + (max_visible * 2) + (max_visible * num_cue_types)
# Example (max_visible=5, num_cue_types=12): 54 + 10 + 60 = 124
```

#### New Components for L6-7

**Other Agent Positions** `[num_agents, max_visible_agents * 2]`

```python
# Positions of other agents within vision_range, relative to observer
# Padded with zeros if fewer than max_visible agents seen

# Example: agent at (4, 4) sees two others
other_positions = [
    # Agent 1 (relative position)
    1.0,   # dx = +1 (agent is at 5, 4)
    -2.0,  # dy = -2 (agent is at 4, 2)
    
    # Agent 2 (relative position)
    -1.0,  # dx = -1 (agent is at 3, 4)
    0.0,   # dy =  0 (agent is at 3, 4)
    
    # Padding (no more agents visible)
    0.0, 0.0,  # agent 3 (not present)
    0.0, 0.0,  # agent 4 (not present)
    0.0, 0.0,  # agent 5 (not present)
]
```

**Construction**:

```python
def get_visible_agents(observer_pos, all_agents, vision_range, max_visible):
    """Find other agents within vision_range of observer."""
    visible = []
    
    for agent in all_agents:
        if agent.id == observer.id:
            continue  # Don't observe self
        
        distance = manhattan_distance(observer_pos, agent.position)
        if distance <= vision_range:
            relative_pos = agent.position - observer_pos
            visible.append(relative_pos)
    
    # Sort by distance (closest first)
    visible.sort(key=lambda pos: abs(pos[0]) + abs(pos[1]))
    
    # Take top max_visible, pad if needed
    visible = visible[:max_visible]
    while len(visible) < max_visible:
        visible.append([0.0, 0.0])
    
    return torch.tensor(visible).flatten()
```

**Purpose**: Agent knows where other agents are (spatial awareness)
**Human equivalent**: "I see two people nearby, one to my right, one ahead of me"

**Other Agent Cues** `[num_agents, max_visible_agents * num_cue_types]`

```python
# Binary matrix: which cues each visible agent is emitting
# Each agent emits up to 3 cues (from cues.yaml, by priority)

# Example: two visible agents, 12 cue types
other_cues = [
    # Agent 1 cues (emits: looks_tired, at_job)
    1,  # looks_tired
    0,  # looks_energetic
    0,  # looks_sick
    0,  # looks_healthy
    0,  # looks_sad
    0,  # looks_happy
    0,  # looks_poor
    0,  # looks_wealthy
    0,  # looks_dirty
    1,  # at_job
    0,  # at_hospital
    0,  # at_bar
    
    # Agent 2 cues (emits: looks_sick, looks_poor, at_hospital)
    0,  # looks_tired
    0,  # looks_energetic
    1,  # looks_sick
    0,  # looks_healthy
    0,  # looks_sad
    0,  # looks_happy
    1,  # looks_poor
    0,  # looks_wealthy
    0,  # looks_dirty
    0,  # at_job
    1,  # at_hospital
    0,  # at_bar
    
    # Agents 3-5 (padding, all zeros)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # agent 3
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # agent 4
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # agent 5
]
```

**Cue ordering** (must be deterministic):

```python
CUE_INDICES = {
    'looks_tired': 0,
    'looks_energetic': 1,
    'looks_sick': 2,
    'looks_healthy': 3,
    'looks_sad': 4,
    'looks_happy': 5,
    'looks_poor': 6,
    'looks_wealthy': 7,
    'looks_dirty': 8,
    'at_job': 9,
    'at_hospital': 10,
    'at_bar': 11,
}
```

**Purpose**: Agent sees body language / emotional state of others
**Human equivalent**: "That person looks tired and they're at the Job. That other person looks sick and poor and they're at the Hospital."

**Level 6 vs Level 7 difference**:

```yaml
# L6: Minimal cues (just location)
cues:
  - at_job
  - at_hospital
  - at_bar
# → num_cue_types = 3

# L7: Rich cues (physical + emotional + socioeconomic + location)
cues:
  - looks_tired
  - looks_energetic
  - looks_sick
  - looks_healthy
  - looks_sad
  - looks_happy
  - looks_poor
  - looks_wealthy
  - looks_dirty
  - at_job
  - at_hospital
  - at_bar
# → num_cue_types = 12
```

**Implementation requirements**:

```python
# Add to ObservationBuilder.__init__
self.max_visible_agents = config.get('max_visible_agents', 5)
self.num_cue_types = len(config.cues.cues)

# Add to build_observations (for L6+)
if self.curriculum_level >= 6:
    # Get visible agents
    visible_positions = self._get_visible_agent_positions(positions)
    visible_cues = self._get_visible_agent_cues(positions)
    
    obs = torch.cat([obs, visible_positions, visible_cues], dim=1)
```

---

### 4.6 Communication Channel (L8) — NEEDS IMPLEMENTATION

**Used in**: Curriculum level 8 (family communication)

**Tensor structure**:

```python
observation = torch.cat([
    # All L6-7 features
    local_grid,              # [num_agents, 25]
    normalized_position,     # [num_agents, 2]
    meters,                  # [num_agents, 8]
    affordance_encoding,     # [num_agents, 16]
    time_sin,                # [num_agents, 1]
    time_cos,                # [num_agents, 1]
    progress,                # [num_agents, 1]
    other_agent_positions,   # [num_agents, max_visible * 2]
    other_agent_cues,        # [num_agents, max_visible * num_cue_types]
    
    # NEW: Family communication
    family_comm_channel,     # [num_agents, max_family_size]
    family_member_ids,       # [num_agents, max_family_size]  (optional)
], dim=1)

# Total dimension: 124 + max_family_size + max_family_size
# Example (max_family_size=3): 124 + 3 + 3 = 130
```

#### New Components for L8

**Family Comm Channel** `[num_agents, max_family_size]`

```python
# Integer signals from family members, normalized to [0, 1]
# Each family member can broadcast one int64 value per tick

# Example: agent in 3-person family (self + 2 others)
family_comm_channel = [
    0.123,  # signal from family member 1 (encoded as 123 / 1000)
    0.456,  # signal from family member 2 (encoded as 456 / 1000)
    0.000,  # unused slot (family size < max_family_size)
]

# If agent is NOT in a family:
family_comm_channel = [0.0, 0.0, 0.0]  # all zeros
```

**Signal encoding**:

```python
# Agents can set their signal via action
action_space.add('SET_COMM_CHANNEL', param_range=[0, 999])

# On execution:
agent.current_signal = action_param  # int in [0, 999]

# In observation builder:
for member_id in agent.family_members:
    member = agents[member_id]
    normalized_signal = member.current_signal / 1000.0
    family_comm_channel.append(normalized_signal)
```

**Purpose**: Family members can coordinate via abstract signals
**Human equivalent**: "My spouse is signaling '123' to me" (meaning negotiated through learning)

**Key constraint: NO SEMANTIC BOOTSTRAPPING**

```python
# ❌ BAD: Providing signal meanings
signal_meanings = {
    0: "all_clear",
    1: "job_taken",
    2: "need_help",
}
# This defeats the research purpose!

# ✅ GOOD: Signals start meaningless
# Agents must learn correlations:
# - Parent sends 123 when at Job
# - Child observes parent at Job correlates with signal 123
# - Child learns: "signal 123 probably means Job is occupied"
# - Child's Module C is trained via CTDE to predict this
```

**Family Member IDs** `[num_agents, max_family_size]` (optional)

```python
# Agent IDs of family members (for tracking who sent which signal)
# Normalized by population size

# Example: agent in family with members [42, 73]
family_member_ids = [
    0.42,  # member 1 is agent_42 (42 / 100 in population of 100)
    0.73,  # member 2 is agent_73
    0.00,  # unused slot
]
```

**Why this might be useful**: Agent can learn "signal from member 42 means X, signal from member 73 means Y" (personalized protocols).

**Why this might not be needed**: If all family members learn the same protocol, identity doesn't matter.

**Recommended**: Start without `family_member_ids`, add if experiments show heterogeneous protocols within families.

---

### 4.7 Complete Dimension Calculations

**Formula by curriculum level**:

```python
# L0-3: Full observability
dim_L0_L3 = (
    grid_size ** 2 +           # one-hot position
    8 +                         # meters
    (num_affordance_types + 1)  # current affordance (+ "none")
)
# Example (5×5 grid, 15 affordances): 25 + 8 + 16 = 49

# L4-5: Partial observability + temporal
window_size = 2 * vision_range + 1  # e.g., 2*2+1 = 5
dim_L4_L5 = (
    window_size ** 2 +         # local grid (5×5 = 25)
    2 +                         # normalized position
    8 +                         # meters
    (num_affordance_types + 1) +  # current affordance
    3                           # time_sin, time_cos, progress
)
# Example: 25 + 2 + 8 + 16 + 3 = 54

# L6: Social (sparse cues)
dim_L6 = dim_L4_L5 + (
    max_visible_agents * 2 +              # positions
    max_visible_agents * num_cue_types_L6  # cues (3 types in L6)
)
# Example (max_visible=5, 3 cues): 54 + 10 + 15 = 79

# L7: Social (rich cues)
dim_L7 = dim_L4_L5 + (
    max_visible_agents * 2 +              # positions
    max_visible_agents * num_cue_types_L7  # cues (12 types in L7)
)
# Example (max_visible=5, 12 cues): 54 + 10 + 60 = 124

# L8: Family communication
dim_L8 = dim_L7 + (
    max_family_size              # comm channel
    # + max_family_size          # member IDs (optional)
)
# Example (family_size=3): 124 + 3 = 127
# Or with IDs: 124 + 3 + 3 = 130
```

**Configuration**:

```yaml
# config.yaml
observation:
  grid_size: 8
  vision_range: 2  # 5×5 window
  num_affordance_types: 15
  
  # Social features (L6+)
  max_visible_agents: 5
  num_cue_types: 12  # full cue set for L7
  
  # Family features (L8)
  max_family_size: 3  # typical: 2 parents + 1 child
  include_family_ids: false  # start without
```

---

### 4.8 Implementation Notes

#### Current State (as of code review)

**What's implemented**:

```python
# observation_builder.py contains:
- Full observability (L0-3) ✓
- Partial observability (L4-5) ✓
- Temporal mechanics (time_of_day, interaction_progress) ✓
```

**What needs to be added**:

```python
# For L6-7:
- _get_visible_agent_positions()
- _get_visible_agent_cues()
- Integration with cue_engine

# For L8:
- family_comm_channel handling
- SET_COMM_CHANNEL action
- Family membership tracking
```

#### Implementation Guide: Adding Social Observability

**Step 1: Add to ObservationBuilder.**init****

```python
def __init__(self, ..., curriculum_level=0):
    # ... existing fields ...
    self.curriculum_level = curriculum_level
    
    if curriculum_level >= 6:
        self.max_visible_agents = config.get('max_visible_agents', 5)
        self.cue_engine = CueEngine(config.cues)
```

**Step 2: Add helper methods**

```python
def _get_visible_agent_positions(
    self,
    observer_positions: torch.Tensor,
    all_agent_positions: torch.Tensor,
) -> torch.Tensor:
    """Get relative positions of visible agents.
    
    Returns:
        [num_agents, max_visible_agents * 2]
    """
    batch_size = observer_positions.shape[0]
    output = torch.zeros(
        batch_size, 
        self.max_visible_agents * 2, 
        device=self.device
    )
    
    for agent_idx in range(batch_size):
        observer_pos = observer_positions[agent_idx]
        visible = []
        
        # Find agents within vision_range
        for other_idx in range(batch_size):
            if other_idx == agent_idx:
                continue
            
            other_pos = all_agent_positions[other_idx]
            distance = torch.abs(observer_pos - other_pos).sum()
            
            if distance <= self.vision_range:
                relative_pos = other_pos - observer_pos
                visible.append((distance, relative_pos))
        
        # Sort by distance, take top max_visible
        visible.sort(key=lambda x: x[0])
        visible = visible[:self.max_visible_agents]
        
        # Fill output tensor
        for i, (_, rel_pos) in enumerate(visible):
            output[agent_idx, i*2:(i+1)*2] = rel_pos
    
    return output

def _get_visible_agent_cues(
    self,
    observer_positions: torch.Tensor,
    all_agent_positions: torch.Tensor,
    all_agent_cues: list[list[str]],  # from cue_engine
) -> torch.Tensor:
    """Get cues of visible agents.
    
    Returns:
        [num_agents, max_visible_agents * num_cue_types]
    """
    batch_size = observer_positions.shape[0]
    output = torch.zeros(
        batch_size,
        self.max_visible_agents * self.num_cue_types,
        device=self.device
    )
    
    for agent_idx in range(batch_size):
        observer_pos = observer_positions[agent_idx]
        visible = []
        
        # Find agents within vision_range (same as positions)
        for other_idx in range(batch_size):
            if other_idx == agent_idx:
                continue
            
            other_pos = all_agent_positions[other_idx]
            distance = torch.abs(observer_pos - other_pos).sum()
            
            if distance <= self.vision_range:
                cues = all_agent_cues[other_idx]  # list of cue strings
                visible.append((distance, other_idx, cues))
        
        # Sort by distance, take top max_visible
        visible.sort(key=lambda x: x[0])
        visible = visible[:self.max_visible_agents]
        
        # Encode cues as binary vector
        for i, (_, other_idx, cues) in enumerate(visible):
            for cue_str in cues:
                cue_idx = self.cue_engine.cue_to_index[cue_str]
                flat_idx = i * self.num_cue_types + cue_idx
                output[agent_idx, flat_idx] = 1.0
    
    return output
```

**Step 3: Update build_observations**

```python
def build_observations(self, ...):
    # Build base observation (L0-5)
    obs = ...  # existing code
    
    # Add social features (L6+)
    if self.curriculum_level >= 6:
        positions_social = self._get_visible_agent_positions(
            positions, positions  # observer and all agents
        )
        cues_social = self._get_visible_agent_cues(
            positions, positions, all_agent_cues
        )
        obs = torch.cat([obs, positions_social, cues_social], dim=1)
    
    # Add family comm (L8)
    if self.curriculum_level >= 8:
        family_signals = self._get_family_comm_channel(
            agent_ids, family_data
        )
        obs = torch.cat([obs, family_signals], dim=1)
    
    return obs
```

#### Implementation Guide: Adding Family Communication

**Step 1: Extend action space**

```python
# In environment setup
self.action_space = [
    'UP', 'DOWN', 'LEFT', 'RIGHT',  # movement
    'INTERACT',                      # use affordance
    'WAIT',                          # do nothing
    'SET_COMM_CHANNEL',              # L8 only, param: [0, 999]
]
```

**Step 2: Track family communication state**

```python
# In agent state
class AgentState:
    def __init__(self):
        self.family_id = None
        self.family_members = []  # list of agent_ids
        self.current_signal = 0   # int [0, 999]
```

**Step 3: Process SET_COMM_CHANNEL action**

```python
def step(self, actions):
    for agent_idx, action in enumerate(actions):
        if action == Action.SET_COMM_CHANNEL:
            # Get parameter (signal value)
            signal_value = action_params[agent_idx]  # int [0, 999]
            self.agents[agent_idx].current_signal = signal_value
```

**Step 4: Build family_comm_channel in observation**

```python
def _get_family_comm_channel(
    self,
    agent_ids: torch.Tensor,
    family_data: dict,
) -> torch.Tensor:
    """Get signals from family members.
    
    Returns:
        [num_agents, max_family_size]
    """
    output = torch.zeros(
        len(agent_ids),
        self.max_family_size,
        device=self.device
    )
    
    for agent_idx, agent_id in enumerate(agent_ids):
        family_id = self.agents[agent_idx].family_id
        
        if family_id is None:
            continue  # not in a family, leave as zeros
        
        family_members = self.agents[agent_idx].family_members
        
        for i, member_id in enumerate(family_members):
            if i >= self.max_family_size:
                break
            
            # Get member's current signal
            signal = self.agents[member_id].current_signal
            normalized = signal / 1000.0
            output[agent_idx, i] = normalized
    
    return output
```

---

### 4.9 Observation Space Summary Table

| Component | L0-3 | L4-5 | L6 | L7 | L8 | Dim | Human Equivalent |
|-----------|------|------|----|----|----|----|------------------|
| **Spatial** | | | | | | | |
| Grid (one-hot) | ✓ | - | - | - | - | grid² | "I'm at (x,y)" |
| Local window | - | ✓ | ✓ | ✓ | ✓ | 25 | "I see 5×5 around me" |
| Position (abs) | - | ✓ | ✓ | ✓ | ✓ | 2 | "My coordinates" |
| **Internal** | | | | | | | |
| Meters (8 bars) | ✓ | ✓ | ✓ | ✓ | ✓ | 8 | "How I feel" |
| Current affordance | ✓ | ✓ | ✓ | ✓ | ✓ | types+1 | "Where I'm standing" |
| **Temporal** | | | | | | | |
| Time (sin/cos) | - | ✓ | ✓ | ✓ | ✓ | 2 | "What time is it" |
| Progress | - | ✓ | ✓ | ✓ | ✓ | 1 | "% done with action" |
| **Social** | | | | | | | |
| Other positions | - | - | ✓ | ✓ | ✓ | M×2 | "Where are others" |
| Other cues | - | - | ✓ | ✓ | ✓ | M×K | "How do they look" |
| **Communication** | | | | | | | |
| Family signals | - | - | - | - | ✓ | F | "What did they say" |

**Total dimensions**:

- L0-3: 49 (5×5 grid, 15 affordances)
- L4-5: 54
- L6: 79 (5 visible, 3 cues)
- L7: 124 (5 visible, 12 cues)
- L8: 127 (family size 3)

---

### 4.10 Validation Checklist

Before claiming observation space is correctly implemented:

**L0-3 (Full Observability)**:

- [ ] Agent can determine its exact position from observation
- [ ] Agent knows all 8 meter values
- [ ] Agent knows which affordance (if any) it's on
- [ ] Observation dimension = grid² + 8 + (types+1)

**L4-5 (Partial + Temporal)**:

- [ ] Agent sees only 5×5 local window
- [ ] Out-of-bounds tiles are encoded as 0
- [ ] Agent knows absolute position via normalized coordinates
- [ ] Time is cyclically encoded (sin/cos)
- [ ] Interaction progress is normalized [0, 1]
- [ ] Observation dimension = 25 + 2 + 8 + (types+1) + 3

**L6-7 (Social)**:

- [ ] Agent sees positions of other agents within vision_range
- [ ] Positions are relative (not absolute)
- [ ] Closest agents appear first (sorted by distance)
- [ ] Cues are binary vectors (1 = cue active)
- [ ] Max visible agents enforced (padding with zeros)
- [ ] No telepathy (only public cues, not internal bars)

**L8 (Family Comm)**:

- [ ] Family members' signals are in observation
- [ ] Non-family agents receive all zeros
- [ ] Signals are normalized [0, 1]
- [ ] SET_COMM_CHANNEL action updates agent's outgoing signal
- [ ] No semantic meaning provided (agents must learn)

**General**:

- [ ] All observations are deterministic (same state → same observation)
- [ ] All values are normalized or one-hot (no raw coordinates > grid_size)
- [ ] Batch dimension is first (for vectorized environments)
- [ ] Device handling is correct (CPU/GPU)

---

## SECTION 5: THE CUES SYSTEM

### 5.1 Cues as Universe as Code

Social observability in Townlet is explicitly configured, not hardcoded. The **cues system** defines which internal states are publicly visible and under what conditions.

**Design principle**: If an agent can observe something about another agent, that observability must be declared in `cues.yaml` as part of the world configuration.

**Why this matters**:

- **Auditable social transparency**: You can inspect exactly what agents can perceive about each other
- **Configurable realism**: Different worlds can model different levels of social observability
- **Research flexibility**: Test hypotheses about information asymmetry by editing YAML
- **No telepathy**: Agents only see what the cue system explicitly broadcasts

**Location in config hierarchy**:

```
configs/my_world/
  universe_as_code.yaml
  bars.yaml
  cascades.yaml
  affordances.yaml
  cues.yaml  # ← social observability layer
```

---

### 5.2 Cue Definition Schema

A cue is a publicly observable signal triggered by internal state or behavior.

**Basic structure**:

```yaml
cues:
  - id: "looks_tired"
    description: "Visible exhaustion from low energy"
    trigger:
      type: "bar_threshold"
      bar: "energy"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 2
```

**Field definitions**:

**`id` (required)**: Unique identifier for the cue

- Used in telemetry and Module C training
- Agents receive these as strings (not semantic embeddings)

**`description` (optional)**: Human-readable explanation

- For documentation and teaching
- Not visible to agents

**`trigger` (required)**: Condition that activates this cue

**Trigger types**:

**Type 1: Bar threshold**

```yaml
trigger:
  type: "bar_threshold"
  bar: "energy"  # which bar to check
  operator: "<"   # <, <=, >, >=, ==
  threshold: 0.3  # normalized [0, 1]
```

**Type 2: Location-based**

```yaml
trigger:
  type: "location"
  current_affordance: "Hospital"
```

**Type 3: Behavioral**

```yaml
trigger:
  type: "action_pattern"
  recent_actions: ["INTERACT", "INTERACT", "INTERACT"]  # repeated interactions
  window: 3  # check last 3 ticks
```

**Type 4: Composite** (future extension)

```yaml
trigger:
  type: "composite"
  operator: "AND"
  conditions:
    - { bar: "health", operator: "<", threshold: 0.3 }
    - { current_affordance: "Hospital" }
  # Meaning: "at hospital while sick"
```

**`visibility` (required)**: Who can see this cue

- `"public"`: Anyone within visual range
- `"family_only"`: Only family members (L8)
- `"private"`: Not observable (future: internal reasoning only)

**`priority` (required)**: Salience ranking

- Higher priority cues are more noticeable
- Used when multiple cues fire simultaneously
- Range: 1-5 (5 = most salient)

**`distance_limit` (optional)**: Maximum visibility radius

```yaml
distance_limit: 5  # only visible within 5 tiles
```

- Default: inherits from environment visibility_radius
- Can be shorter than vision (e.g., subtle cues vs obvious ones)

---

### 5.3 Cue Emission Rules

**Maximum broadcast limit**:

```yaml
visibility_rules:
  max_cues_broadcast: 3  # agent emits top 3 cues by priority
```

**Why limit**:

- Prevents information overload (agents would see 10+ cues per other agent)
- Forces prioritization (only most salient signals visible)
- Realistic (humans can't track 10 simultaneous body language signals)

**Emission algorithm**:

```python
def compute_public_cues(agent, cue_config):
    """
    Evaluate all cue triggers, emit top-k by priority.
    """
    active_cues = []
    
    # Evaluate each cue definition
    for cue_def in cue_config.cues:
        if evaluate_trigger(agent, cue_def.trigger):
            active_cues.append({
                'id': cue_def.id,
                'priority': cue_def.priority,
                'visibility': cue_def.visibility
            })
    
    # Filter by visibility (L6-7: public only, L8: public + family)
    if not agent.in_family or not config.family_channel_enabled:
        active_cues = [c for c in active_cues if c['visibility'] == 'public']
    
    # Sort by priority (high first)
    active_cues.sort(key=lambda c: -c['priority'])
    
    # Take top max_cues_broadcast
    max_cues = cue_config.visibility_rules.max_cues_broadcast
    emitted = active_cues[:max_cues]
    
    return [c['id'] for c in emitted]  # return just the IDs
```

**Example**:

```python
# Agent state
bars = {
    'energy': 0.22,    # triggers "looks_tired" (threshold 0.3)
    'health': 0.18,    # triggers "looks_sick" (threshold 0.3)
    'money': 0.15,     # triggers "looks_poor" (threshold 0.2)
    'mood': 0.35       # triggers "looks_sad" (threshold 0.4)
}
current_affordance = "Hospital"  # triggers "at_hospital"

# Active cues (priority in parentheses)
active = [
    "looks_sick" (4),
    "looks_sad" (3),
    "at_hospital" (3),
    "looks_tired" (2),
    "looks_poor" (1)
]

# After sorting by priority and taking top 3
emitted = ["looks_sick", "looks_sad", "at_hospital"]
# OR (if there's a tie): ["looks_sick", "at_hospital", "looks_sad"]
```

**Tie-breaking**: If multiple cues have same priority, deterministic ordering by `id` (alphabetical).

---

### 5.4 Example Cue Pack: Baseline World

```yaml
# cues.yaml for baseline world
version: "1.0"
description: "Standard social observability for L6-L8"

visibility_rules:
  max_cues_broadcast: 3
  distance_limit: 5  # default visibility radius

cues:
  # Energy-based (physical fatigue)
  - id: "looks_tired"
    description: "Visible exhaustion, droopy posture"
    trigger:
      type: "bar_threshold"
      bar: "energy"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 2
    
  - id: "looks_energetic"
    description: "Visible vitality, alert"
    trigger:
      type: "bar_threshold"
      bar: "energy"
      operator: ">"
      threshold: 0.8
    visibility: "public"
    priority: 1
  
  # Health-based (physical condition)
  - id: "looks_sick"
    description: "Pale, coughing, limping"
    trigger:
      type: "bar_threshold"
      bar: "health"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 4  # highly salient
    
  - id: "looks_healthy"
    description: "Robust, good color"
    trigger:
      type: "bar_threshold"
      bar: "health"
      operator: ">"
      threshold: 0.8
    visibility: "public"
    priority: 1
  
  # Mood-based (emotional state)
  - id: "looks_sad"
    description: "Downcast, withdrawn"
    trigger:
      type: "bar_threshold"
      bar: "mood"
      operator: "<"
      threshold: 0.4
    visibility: "public"
    priority: 3
    
  - id: "looks_happy"
    description: "Smiling, animated"
    trigger:
      type: "bar_threshold"
      bar: "mood"
      operator: ">"
      threshold: 0.7
    visibility: "public"
    priority: 2
  
  # Wealth-based (socioeconomic signals)
  - id: "looks_poor"
    description: "Shabby clothing, worn items"
    trigger:
      type: "bar_threshold"
      bar: "money"
      operator: "<"
      threshold: 0.2
    visibility: "public"
    priority: 1
    
  - id: "looks_wealthy"
    description: "Well-dressed, quality items"
    trigger:
      type: "bar_threshold"
      bar: "money"
      operator: ">"
      threshold: 0.8
    visibility: "public"
    priority: 1
  
  # Location-based (intent signals)
  - id: "at_job"
    description: "Located at workplace"
    trigger:
      type: "location"
      current_affordance: "Job"
    visibility: "public"
    priority: 2
    
  - id: "at_hospital"
    description: "Located at hospital (medical need)"
    trigger:
      type: "location"
      current_affordance: "Hospital"
    visibility: "public"
    priority: 3
    
  - id: "at_bar"
    description: "Located at bar (socializing)"
    trigger:
      type: "location"
      current_affordance: "Bar"
    visibility: "public"
    priority: 2
    
  - id: "at_home"
    description: "Located at bed/residence"
    trigger:
      type: "location"
      current_affordance: "Bed"
    visibility: "public"
    priority: 1
  
  # Hygiene-based (social acceptability)
  - id: "looks_dirty"
    description: "Unkempt, odorous"
    trigger:
      type: "bar_threshold"
      bar: "hygiene"
      operator: "<"
      threshold: 0.3
    visibility: "public"
    priority: 2
    distance_limit: 2  # only visible up close
```

**Design notes**:

- Health cues have highest priority (life-threatening states are most salient)
- Mood cues are mid-priority (emotionally important but not urgent)
- Wealth/hygiene cues are lower priority (social but not survival-critical)
- Location cues provide intent signals without telepathy

---

### 5.5 Module C Training: CTDE via Cues

The Social Model (Module C) learns to predict internal state from observable cues using Centralized Training, Decentralized Execution (CTDE).

**Training phase** (offline, using logged episodes):

```python
# Training loop (centralized)
for episode in training_dataset:
    for tick in episode:
        for agent_id in other_agents:
            # Ground truth (privileged - training only)
            true_bars = episode.agents[agent_id].bars
            true_goal = episode.agents[agent_id].current_goal
            
            # Observable inputs (available at inference)
            emitted_cues = episode.agents[agent_id].public_cues
            # e.g., ['looks_tired', 'at_job']
            
            # Supervised learning
            predicted_bars = social_model.predict_state(emitted_cues)
            predicted_goal = social_model.predict_goal(emitted_cues)
            
            # Losses
            state_loss = mse(predicted_bars, true_bars)
            goal_loss = cross_entropy(predicted_goal, true_goal)
            
            # Backward pass
            total_loss = state_loss + goal_loss
            total_loss.backward()
            optimizer.step()
```

**What Module C learns**:

- `['looks_tired', 'at_job']` → likely bars: `{energy: ~0.25, ...}`
- `['looks_sick', 'at_hospital']` → likely goal: `SURVIVAL`
- `['looks_happy', 'at_bar']` → likely goal: `SOCIAL`

**Inference phase** (deployed policy, decentralized):

```python
# Inference (no ground truth available)
observation = {
    'other_agents_in_window': {
        'public_cues': [
            ['looks_tired', 'at_job'],  # agent 2
            ['looks_sick', 'at_hospital']  # agent 3
        ]
    }
}

# Module C predictions (from learned correlations)
for i, cues in enumerate(observation['other_agents_in_window']['public_cues']):
    predicted_state = social_model.predict_state(cues)
    # Output: {energy: 0.28, health: 0.75, ...}
    
    predicted_goal = social_model.predict_goal(cues)
    # Output: {SURVIVAL: 0.15, THRIVING: 0.70, SOCIAL: 0.15}
```

**Human equivalent**:

- Training: "When I see droopy eyes, I ask 'are you tired?' They say 'yes.' I learn the correlation."
- Inference: "I see droopy eyes now. I predict they're tired without asking."

**Why this isn't cheating**:

- At inference, Module C only sees `public_cues` (same as human observer)
- Ground truth was only used during training (offline)
- The learned model generalizes from labels to real-time prediction
- This is standard supervised learning, not runtime telepathy

**WEP** that CTDE is the correct approach for social reasoning: very high (~95%).

---

### 5.6 Curriculum Staging via Cue Richness

Different curriculum levels can use different cue sets to stage social reasoning difficulty.

**Level 6: Sparse Cues (Basics)**

```yaml
# L6_cues.yaml (minimal information)
cues:
  - "at_job"
  - "at_hospital"
  - "at_bar"
  # Location only - agents must infer state from behavior
```

**What agents learn at L6**:

- "Agent at Job probably needs money"
- "Agent at Hospital probably has low health"
- Basic intent inference from location

**Level 7: Rich Cues (Full Observability)**

```yaml
# L7_cues.yaml (detailed state information)
cues:
  - "looks_tired"
  - "looks_energetic"
  - "looks_sick"
  - "looks_healthy"
  - "looks_sad"
  - "looks_happy"
  - "looks_poor"
  - "looks_wealthy"
  - "looks_dirty"
  - "at_job"
  - "at_hospital"
  - "at_bar"
  # Physical + emotional + socioeconomic + location
```

**What agents learn at L7**:

- Fine-grained state inference
- "Looks sick + at hospital + looks poor" → survival crisis, no money for treatment
- Strategic predictions: "They'll stay at hospital, Job is free"

**Level 8: Behavioral Cues (Advanced)**

```yaml
# L8_cues.yaml (includes behavioral patterns)
cues:
  [all L7 cues] +
  - "carrying_food"          # just left Fridge
  - "rushing"                # moving quickly (high energy)
  - "lingering"              # staying in one place (waiting)
  - "avoiding_others"        # spatial pattern
```

**What agents learn at L8**:

- Behavior prediction from micro-actions
- "Carrying food + moving toward home" → feeding family
- Coordination: "They're lingering at Job entrance → waiting for it to open"

---

### 5.7 World Design via Cue Configuration

Cue systems enable research on information asymmetry and social transparency.

**High-Transparency World** (trust-based society)

```yaml
cues:
  [rich cue set with 15+ cues]
  max_cues_broadcast: 5  # lots of information visible
  distance_limit: 8       # see far
```

**Expected outcomes**:

- Agents can accurately predict others' needs
- Cooperation emerges easily
- Low coordination failures
- But: privacy is minimal (everyone's state is public)

**Low-Transparency World** (privacy-preserving society)

```yaml
cues:
  - "at_job"  # only location visible
  max_cues_broadcast: 1
  distance_limit: 2  # must be very close
```

**Expected outcomes**:

- Agents must infer from sparse signals
- More coordination failures (misunderstandings)
- Strategic behavior harder to predict
- But: privacy is high (internal state mostly hidden)

**Asymmetric Transparency** (class-based society)

```yaml
cues:
  - id: "looks_poor"
    visibility: "public"  # poverty is visible
    priority: 3
    
  - id: "looks_wealthy"
    visibility: "private"  # wealth is concealable
    priority: 1
```

**Expected outcomes**:

- Disadvantage is broadcast, advantage is hidden
- "Rich agents can pretend to be poor, poor agents can't pretend to be rich"
- Models real-world information asymmetries
- Research question: Does this create exploitation dynamics?

**Research protocol**:

```bash
# Run same agent in three transparency worlds
townlet train --config configs/high_transparency/ --seed 42
townlet train --config configs/low_transparency/ --seed 42
townlet train --config configs/asymmetric/ --seed 42

# Compare:
# - Coordination success rate
# - Social prediction accuracy (Module C)
# - Terminal reward distribution
# - Emergent strategies (trust vs suspicion)
```

---

### 5.8 Implementation Notes

**Where cues are computed**: `VectorizedTownletEnv`

```python
# In environment step() function
def step(self, actions):
    # ... execute actions, apply affordances, cascades ...
    
    # Compute public cues for all agents
    for agent_id, agent in enumerate(self.agents):
        agent.public_cues = self.cue_engine.compute_cues(
            agent=agent,
            cue_config=self.config.cues
        )
    
    # Build observations
    for agent_id in range(self.num_agents):
        # Get other agents in visibility window
        visible_agents = self.get_agents_in_window(agent_id, radius=5)
        
        # Extract their public cues
        other_cues = [
            self.agents[other_id].public_cues
            for other_id in visible_agents
        ]
        
        observations[agent_id] = {
            'bars': self.agents[agent_id].bars,
            'position': self.agents[agent_id].position,
            'other_agents_in_window': {
                'positions': [self.agents[i].position for i in visible_agents],
                'public_cues': other_cues,  # ← cues inserted here
            }
        }
```

**How Module C receives cues**: As part of observation

```python
# Module C forward pass
def forward(self, observation):
    # Extract public cues from observation
    other_cues = observation['other_agents_in_window']['public_cues']
    # List[List[str]], e.g. [['looks_tired', 'at_job'], ['looks_sick']]
    
    # Embed each cue string
    cue_embeddings = [self.cue_embedding(cue) for cue_list in other_cues 
                      for cue in cue_list]
    # Learned embedding: "looks_tired" → 128-dim vector
    
    # Process through GRU (for temporal patterns)
    h_t = self.gru(cue_embeddings, h_prev)
    
    # Predict state and goal
    predicted_bars = self.state_head(h_t)  # → [energy, health, ...]
    predicted_goal_dist = self.goal_head(h_t)  # → [P(SURVIVAL), P(THRIVING), ...]
    
    return predicted_bars, predicted_goal_dist
```

**Validation**: Cues are validated on load

```python
# cue_validator.py
def validate_cues_config(cues_yaml):
    for cue in cues_yaml['cues']:
        # Check required fields
        assert 'id' in cue
        assert 'trigger' in cue
        assert 'visibility' in cue
        assert 'priority' in cue
        
        # Validate trigger
        if cue['trigger']['type'] == 'bar_threshold':
            assert cue['trigger']['bar'] in VALID_BARS
            assert 0.0 <= cue['trigger']['threshold'] <= 1.0
        
        # Validate priority
        assert 1 <= cue['priority'] <= 5
```

---

### 5.9 Cues System Summary

**What it is**: A declarative configuration layer that defines social observability

**What it does**:

- Specifies which internal states are publicly visible
- Controls information flow in multi-agent scenarios
- Enables CTDE training for Module C (Social Model)

**Why it matters**:

- No hardcoded telepathy (all social information is explicit)
- Auditable (you can see exactly what agents perceive)
- Configurable (research on transparency levels)
- Human-realistic (body language, not mind-reading)

**Key files**:

- `cues.yaml` — world configuration
- `cue_engine.py` — runtime emission
- Module C in `agent_architecture.yaml` — prediction

**WEP** that cues system is correct architectural choice: very high (~95%).

---

**End of Section 5**

---

## SECTION 6: CRITICAL BLOCKERS

These are showstoppers. They must be fixed before any production deployment, before publishing results, and ideally before significant further development.

**Priority order**: Fix in this sequence (1 → 2 → 3).

---

### 6.1 BLOCKER 1: EthicsFilter Architecture Ambiguity (GOVERNANCE)

**Severity**: CRITICAL  
**Affected systems**: Brain as Code, checkpoints, governance claims  
**Risk if unfixed**: Audit failure, safety cannot be proven

#### The Problem

Your documentation contradicts itself about EthicsFilter:

**Brain as Code doc (Section 2.2) says**:

```yaml
modules:
  # ... other modules ...
  EthicsFilter:
    # Implies it's a learned module with weights
```

**Brain as Code doc (Section 6.4) says**:
> "EthicsFilter reads Layer 1 rules deterministically"

**Checkpoint doc (Section 4.1) says**:

```
weights.pt includes:
  - all module weights (including EthicsFilter)
```

**These cannot all be true.**

If EthicsFilter has weights, it's a learned module. If it's learned:

- How do you prove the weights implement the declared policy?
- What if it learns to approve forbidden actions during training?
- How do you audit "does this brain respect `forbid_actions: ['steal']`"?

**Governance implication**: You cannot take this to IRAP/ISM assessment if you can't prove safety rules are enforced.

#### The Solution: Make EthicsFilter Deterministic

EthicsFilter must be a **pure rule evaluator**, not a learned module.

**Implementation**:

```python
# agent/ethics_filter.py

class EthicsFilter:
    """
    Deterministic rule enforcement. No learning. No weights.
    This is the governance boundary.
    """
    
    def __init__(self, config: Layer1Config):
        """
        Load ethics rules from Layer 1 (cognitive_topology.yaml).
        No nn.Module inheritance, no optimizers, no weights.
        """
        self.forbid_actions = set(config.compliance.forbid_actions)
        self.penalize_actions = {
            p['action']: p['penalty'] 
            for p in config.compliance.penalize_actions
        }
        # NO self.parameters(), NO self.optimizer
        
    def filter(self, 
               candidate_action: int, 
               action_names: List[str],
               agent_state: dict) -> dict:
        """
        Apply compliance rules to candidate action.
        
        Returns:
            {
                'final_action': int,
                'veto_applied': bool,
                'veto_reason': str | None,
                'shaping_penalty': float
            }
        """
        action_name = action_names[candidate_action]
        
        # Hard veto (forbid_actions)
        if action_name in self.forbid_actions:
            return {
                'final_action': self._substitute_safe_action(action_names),
                'veto_applied': True,
                'veto_reason': f'{action_name} forbidden by Layer 1 compliance.forbid_actions',
                'shaping_penalty': 0.0
            }
        
        # Soft penalty (penalize_actions)
        penalty = self.penalize_actions.get(action_name, 0.0)
        
        return {
            'final_action': candidate_action,
            'veto_applied': False,
            'veto_reason': None,
            'shaping_penalty': penalty  # RL can use this in reward calculation
        }
    
    def _substitute_safe_action(self, action_names: List[str]) -> int:
        """
        When vetoing, substitute with a safe fallback.
        Default: WAIT (always legal).
        """
        wait_action_idx = action_names.index('WAIT')
        return wait_action_idx
```

**Key properties**:

- ✅ No `nn.Module` inheritance
- ✅ No learnable parameters
- ✅ No optimizer
- ✅ Pure function: same inputs → same outputs
- ✅ Rules come from Layer 1 YAML (auditable)

**Integration in execution_graph.yaml**:

```yaml
steps:
  # ... earlier steps ...
  
  - name: "panic_adjustment"
    node: "@modules.panic_controller"
    inputs:
      - "@steps.candidate_action"
      - "@config.L1.panic_thresholds"
    outputs:
      - "panic_action"
      - "panic_reason"
  
  - name: "final_action"
    node: "@controllers.ethics_filter"  # ← NOT @modules (not learned)
    inputs:
      - "@steps.panic_adjustment.panic_action"
      - "@config.L1.compliance"
    outputs:
      - "action"
      - "veto_reason"
      - "shaping_penalty"

outputs:
  final_action: "@steps.final_action.action"
  veto_applied: "@steps.final_action.veto_applied"
  veto_reason: "@steps.final_action.veto_reason"
```

#### Documentation Updates Required

**1. Brain as Code doc (Section 2.2)**: Remove EthicsFilter from `modules:` list

```yaml
modules:
  perception_encoder: {...}
  world_model: {...}
  social_model: {...}
  hierarchical_policy: {...}
  # ❌ REMOVE: EthicsFilter (not a learned module)
```

**2. Brain as Code doc (Section 6.4)**: Clarify architecture

```markdown
### 6.4 EthicsFilter

EthicsFilter is a **deterministic rule executor**, not a learned module.

**Inputs**: candidate_action (from panic_controller), compliance rules (from Layer 1)
**Outputs**: final_action, veto_applied, veto_reason
**Implementation**: Pure Python class, no torch parameters

This design ensures ethics rules are provably enforced:
- Rules are explicit in cognitive_topology.yaml
- No learning can subvert them
- Audit can verify implementation matches specification
```

**3. Checkpoint doc (Section 4.1)**: Remove EthicsFilter from weights

```markdown
### 4.1 weights.pt

This checkpoint file contains the neural state of learned modules:
- perception_encoder
- world_model
- social_model
- hierarchical_policy
- panic_controller (if learned)

**Not included**: EthicsFilter (deterministic, no weights)
```

**4. Add to High-Level Design doc (Section 2)**: Governance design choice

```markdown
## 2.X EthicsFilter as Governance Boundary

EthicsFilter is intentionally **not learned** to maintain governance auditability.

**Design rationale**:
- Learned safety is probabilistic ("probably won't do bad things")
- Rule-based safety is deterministic ("cannot do bad things by construction")
- Governance requires the latter

**Trade-off**: 
- Pro: Provable safety, auditable
- Con: Cannot adapt to novel situations not covered by rules
- Mitigation: Rules can be updated in Layer 1 for new deployments
```

#### Verification

After fixing, you should be able to answer audit questions:

**Q**: "Can this agent steal?"  
**A**: "No. See `cognitive_topology.yaml` line 47: `forbid_actions: ['steal']`. EthicsFilter (line 123 of `ethics_filter.py`) checks this list and substitutes WAIT if steal is attempted. This is deterministic Python code, not learned behavior. We can prove it by inspection."

**Q**: "What if panic overrides normal behavior?"  
**A**: "Panic can escalate actions for survival (see `execution_graph.yaml` line 82), but EthicsFilter runs after panic (line 95) and has final authority. Even if panic proposes 'steal', EthicsFilter will veto it."

**Q**: "Could training change this?"  
**A**: "No. EthicsFilter has no learnable parameters. See `weights.pt` contents: EthicsFilter is not included. Only the policy, world model, and perception are trained."

**WEP** that deterministic EthicsFilter is the correct choice: very high (~98%).

---

### 6.2 BLOCKER 2: World Model Training vs Curriculum Changes (CORRECTNESS)

**Severity**: HIGH  
**Affected systems**: Module B (World Model), curriculum, reproducibility  
**Risk if unfixed**: Agents learn outdated world dynamics, experiments are confounded

#### The Problem

You train Module B (World Model) on `uac_ground_truth_logs` which contains:

- Exact affordance costs (Fridge costs $4)
- Exact cascade dynamics (satiation < 0.2 → health -0.01/tick)
- Exact terminal conditions

But you also allow curriculum to change these mid-run (Section 13.2):

- "Austerity curriculum: make food more expensive"
- "Hardship curriculum: accelerate cascade strengths"

**Consequence**:

```python
# Agent trained on baseline world
world_model.predict(action='go_to_fridge') 
  → predicted_cost: -0.04  # expects $4

# Curriculum changes to austerity
actual_cost: -0.08  # now costs $8

# Agent's world model is now wrong
# It will make suboptimal decisions based on outdated predictions
```

**Why this breaks experiments**:

- Behavioral changes could be due to curriculum OR due to stale world model
- Can't distinguish "agent adapted to scarcity" from "agent is confused"
- Reproducibility fails (world model accuracy degrades over time)

#### The Solution: Curriculum Forks Require New Hash

**Rule**: Changing affordance semantics or cascade dynamics is a **world fork**, not curriculum pressure.

**What's allowed as "curriculum pressure"** (no fork):

- Spawning additional instances of existing affordances
  - "Add a second Job location" (same costs, same effects)
- Adjusting operating hours within existing ranges
  - "Job closes at 5pm instead of 6pm"
- Changing initial agent spawn conditions
  - "Start with $25 instead of $50"
- Adding/removing agents
  - "Introduce a competitor at tick 10k"

**What requires a fork** (new hash, new run_id):

- Changing affordance costs
  - "Fridge now costs $8" → fork
- Changing affordance effects
  - "Bed now restores 0.30 energy instead of 0.25" → fork
- Changing cascade strengths
  - "low_satiation_hits_health strength: 0.020 instead of 0.010" → fork
- Changing terminal conditions
  - "Health death threshold: 0.1 instead of 0.0" → fork
- Adding new affordance types
  - "Introduce 'Ambulance' affordance" → fork

**Implementation**:

**1. Add world_config_hash to observations**:

```python
# When building observations
world_config_hash = compute_config_hash(
    affordances_yaml,
    cascades_yaml,
    bars_yaml
)

observation = {
    'bars': [...],
    'position': [...],
    'world_config_hash': world_config_hash,  # ← NEW
    # ... other fields
}
```

**2. Module B can condition on world_config_hash**:

```yaml
# agent_architecture.yaml
modules:
  world_model:
    inputs:
      - belief_state  # from Module A
      - candidate_action
      - world_config_hash  # ← NEW: lets model know which world
    
    core_network:
      type: "MLP"
      layers: [256, 256]
      world_conditioning: "concat"  # concatenate hash to input
```

**3. Curriculum config makes forks explicit**:

```yaml
# config.yaml
curriculum:
  enabled: true
  
  stages:
    - stage_id: "baseline"
      duration_ticks: 50000
      world_config: "universe_baseline.yaml"
      
    - stage_id: "austerity"
      duration_ticks: 50000
      world_config: "universe_austerity.yaml"  # ← DIFFERENT FILE
      fork_required: true  # ← EXPLICIT MARKER
      note: "Affordance costs changed, world model must adapt"
    
    - stage_id: "boom"
      duration_ticks: 50000
      world_config: "universe_boom.yaml"
      fork_required: true
```

**4. Launcher enforces fork semantics**:

```python
def launch_curriculum_stage(stage_config):
    if stage_config.fork_required:
        # This is a new world, requires new run
        new_run_id = f"{base_name}_stage_{stage_config.stage_id}_{timestamp}"
        
        # Snapshot new world config
        snapshot_dir = f"runs/{new_run_id}/config_snapshot/"
        copy_yaml(stage_config.world_config, snapshot_dir)
        
        # Recompute cognitive hash (world changed)
        new_hash = compute_cognitive_hash(snapshot_dir)
        
        # Load weights from previous stage (transfer learning)
        if stage_config.get('resume_from'):
            load_weights(prev_checkpoint)
        
        # But this is a NEW MIND in a NEW WORLD
        log.info(f"Forked to new run: {new_run_id}, hash: {new_hash}")
    else:
        # Curriculum pressure (spawn more affordances, etc)
        # Same run, same hash
        apply_pressure(stage_config.pressure_params)
```

#### Alternative: Retrain World Model on Curriculum Change

If you want to keep the same run_id across curriculum changes, you could retrain Module B:

```python
def on_curriculum_stage_change(new_world_config):
    # Collect new ground truth logs (100-1000 episodes)
    logs = collect_logs_in_new_world(new_world_config, num_episodes=1000)
    
    # Retrain Module B only (freeze other modules)
    for module in ['perception', 'social_model', 'policy']:
        freeze(module)
    
    # Fine-tune world model
    for epoch in range(50):
        train_world_model(logs)
    
    # Unfreeze everything
    for module in all_modules:
        unfreeze(module)
    
    # Continue training
```

**Trade-offs**:

- Pro: Keeps same run_id, avoids confusion
- Con: Expensive (1000 episodes + 50 epochs retraining)
- Con: Introduces training gap (world model accuracy dips during transition)

**Recommended**: Use fork approach. It's cleaner for science.

#### Documentation Updates Required

**1. High-Level Design (Section 13.2)**: Resolve open question

```markdown
## 13.2 Curriculum Pressure vs World Forks (RESOLVED)

**Allowed as curriculum pressure** (no fork):
- Spawn additional affordance instances
- Adjust operating hours
- Change initial conditions
- Add/remove agents

**Requires fork** (new hash, new run_id):
- Change affordance costs or effects
- Change cascade strengths
- Change bar dynamics
- Add new affordance types

**Rationale**: Module B (World Model) learns world physics. Changing physics creates a new world, which is a different experimental condition and must be tracked as such.

**Implementation**: `world_config_hash` in observation space, curriculum stages explicitly marked with `fork_required: true`.
```

**2. Observation Space (Section 4)**: Add world_config_hash

```python
observation = {
    'bars': torch.Tensor([8]),
    'position': torch.Tensor([2]),
    'visible_grid': torch.Tensor([5, 5, N_AFFORDANCES]),
    'time_of_day': torch.Tensor([1]),
    'world_config_hash': torch.Tensor([1]),  # ← NEW
    # ... other fields
}
```

**3. Brain as Code (Section 2.2)**: Update world_model inputs

```yaml
modules:
  world_model:
    inputs:
      - belief_state
      - candidate_action
      - world_config_hash  # ← NEW
```

**WEP** that fork approach is correct: high (~85%). The retraining approach might work but adds complexity.

---

### 6.3 BLOCKER 3: Checkpoint Integrity (SECURITY)

**Severity**: HIGH  
**Affected systems**: Checkpoints, resume, provenance, audit  
**Risk if unfixed**: Tampering is undetectable, provenance is worthless

#### The Problem

Checkpoints store:

```
runs/.../checkpoints/step_500/
  weights.pt
  optimizers.pt
  config_snapshot/
    cognitive_topology.yaml  # ← editable!
    # ... other YAMLs
  cognitive_hash.txt
```

**Attack vector**:

```bash
# Malicious actor (or accidental edit)
vim runs/.../checkpoints/step_500/config_snapshot/cognitive_topology.yaml

# Change:
forbid_actions: ["steal"]
# to:
forbid_actions: []  # allow stealing

# Recompute hash
python scripts/recompute_hash.py --checkpoint step_500

# Resume training
townlet resume --checkpoint step_500
# Now claims "same mind" but ethics rules changed
```

**You cannot detect this** because:

- Snapshots are mutable directories
- Hashes are recomputable
- No tamper protection

**Governance impact**: Chain of custody is broken. You can't prove a checkpoint hasn't been modified.

#### The Solution: Sign Checkpoints with HMAC

Add cryptographic signatures to checkpoints to detect tampering.

**Implementation**:

```python
# checkpointing/secure_checkpoint.py

import hashlib
import hmac
from pathlib import Path
import json

class SecureCheckpointWriter:
    """
    Writes checkpoints with HMAC signatures for tamper detection.
    """
    
    def __init__(self, signing_key: bytes):
        """
        Args:
            signing_key: Secret key for HMAC (stored securely, not in repo)
        """
        self.signing_key = signing_key
    
    def write_checkpoint(self, checkpoint_dir: Path, payload: dict):
        """
        Write checkpoint with all components + signature.
        
        Args:
            checkpoint_dir: Where to write checkpoint
            payload: {
                'weights': state_dicts,
                'optimizers': optimizer_states,
                'rng_state': rng_state,
                'config_snapshot': {yaml_files},
                'cognitive_hash': str
            }
        """
        # Create directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Write all standard files
        torch.save(payload['weights'], checkpoint_dir / 'weights.pt')
        torch.save(payload['optimizers'], checkpoint_dir / 'optimizers.pt')
        json.dump(payload['rng_state'], open(checkpoint_dir / 'rng_state.json', 'w'))
        
        # Write config snapshot
        snapshot_dir = checkpoint_dir / 'config_snapshot'
        snapshot_dir.mkdir(exist_ok=True)
        for filename, content in payload['config_snapshot'].items():
            (snapshot_dir / filename).write_text(content)
        
        # Write cognitive hash
        (checkpoint_dir / 'cognitive_hash.txt').write_text(payload['cognitive_hash'])
        
        # Compute manifest (checksums of all files)
        manifest = self._compute_manifest(checkpoint_dir)
        (checkpoint_dir / 'manifest.txt').write_text(manifest)
        
        # Sign the manifest
        signature = hmac.new(
            self.signing_key,
            manifest.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        (checkpoint_dir / 'signature.txt').write_text(signature)
        
        print(f"✓ Checkpoint written and signed: {checkpoint_dir}")
    
    def _compute_manifest(self, checkpoint_dir: Path) -> str:
        """
        Compute SHA256 checksum for each file in checkpoint.
        
        Returns:
            Manifest string: "filename: checksum\n" for each file
        """
        files_to_check = [
            'weights.pt',
            'optimizers.pt',
            'rng_state.json',
            'cognitive_hash.txt',
            'config_snapshot/config.yaml',
            'config_snapshot/universe_as_code.yaml',
            'config_snapshot/cognitive_topology.yaml',
            'config_snapshot/agent_architecture.yaml',
            'config_snapshot/execution_graph.yaml',
        ]
        
        manifest_lines = []
        for rel_path in files_to_check:
            file_path = checkpoint_dir / rel_path
            if not file_path.exists():
                raise ValueError(f"Missing file: {rel_path}")
            
            # Compute SHA256
            file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            manifest_lines.append(f"{rel_path}: {file_hash}")
        
        return "\n".join(manifest_lines)
    
    def verify_checkpoint(self, checkpoint_dir: Path) -> bool:
        """
        Verify checkpoint hasn't been tampered with.
        
        Returns:
            True if valid, raises exception if invalid
        
        Raises:
            CheckpointTamperedError: If signature doesn't match
            CheckpointCorruptedError: If files are missing or checksums wrong
        """
        # Read manifest and signature
        manifest_path = checkpoint_dir / 'manifest.txt'
        signature_path = checkpoint_dir / 'signature.txt'
        
        if not manifest_path.exists() or not signature_path.exists():
            raise CheckpointTamperedError("Missing manifest or signature")
        
        manifest = manifest_path.read_text()
        signature = signature_path.read_text().strip()
        
        # Verify signature
        expected_signature = hmac.new(
            self.signing_key,
            manifest.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if signature != expected_signature:
            raise CheckpointTamperedError(
                f"Signature mismatch in {checkpoint_dir}\n"
                f"Expected: {expected_signature}\n"
                f"Got: {signature}\n"
                "Checkpoint may have been tampered with."
            )
        
        # Verify each file's checksum matches manifest
        for line in manifest.strip().split('\n'):
            rel_path, expected_hash = line.split(': ')
            file_path = checkpoint_dir / rel_path
            
            if not file_path.exists():
                raise CheckpointCorruptedError(f"Missing file: {rel_path}")
            
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            
            if actual_hash != expected_hash:
                raise CheckpointCorruptedError(
                    f"Checksum mismatch: {rel_path}\n"
                    f"Expected: {expected_hash}\n"
                    f"Got: {actual_hash}"
                )
        
        print(f"✓ Checkpoint verified: {checkpoint_dir}")
        return True


class CheckpointTamperedError(Exception):
    """Raised when checkpoint signature is invalid."""
    pass


class CheckpointCorruptedError(Exception):
    """Raised when checkpoint files are missing or corrupted."""
    pass
```

**Usage**:

```python
# During training: write signed checkpoints
signer = SecureCheckpointWriter(signing_key=load_signing_key())

checkpoint_payload = {
    'weights': {name: module.state_dict() for name, module in agent.modules.items()},
    'optimizers': {...},
    'rng_state': {...},
    'config_snapshot': {...},
    'cognitive_hash': full_cognitive_hash
}

signer.write_checkpoint(
    checkpoint_dir=Path(f"runs/{run_id}/checkpoints/step_{step}/"),
    payload=checkpoint_payload
)

# On resume: verify before loading
try:
    signer.verify_checkpoint(checkpoint_dir)
    # If verification passes, safe to load
    load_checkpoint(checkpoint_dir)
except CheckpointTamperedError as e:
    print(f"❌ SECURITY ALERT: {e}")
    print("Refusing to resume from tampered checkpoint.")
    sys.exit(1)
```

**Key management**:

```python
# keys/signing_key.py (NOT committed to git)

def load_signing_key() -> bytes:
    """
    Load HMAC signing key from secure location.
    
    Options:
    1. Environment variable (for CI/production)
    2. Encrypted keyfile (for development)
    3. Hardware security module (for deployment)
    """
    key_source = os.getenv('TOWNLET_SIGNING_KEY_SOURCE', 'keyfile')
    
    if key_source == 'env':
        key_hex = os.getenv('TOWNLET_SIGNING_KEY')
        if not key_hex:
            raise ValueError("TOWNLET_SIGNING_KEY not set")
        return bytes.fromhex(key_hex)
    
    elif key_source == 'keyfile':
        keyfile = Path.home() / '.townlet' / 'signing_key.bin'
        if not keyfile.exists():
            # Generate new key on first run
            keyfile.parent.mkdir(exist_ok=True)
            new_key = secrets.token_bytes(32)  # 256-bit key
            keyfile.write_bytes(new_key)
            keyfile.chmod(0o600)  # read/write by owner only
            print(f"Generated new signing key: {keyfile}")
        return keyfile.read_bytes()
    
    else:
        raise ValueError(f"Unknown key source: {key_source}")
```

#### Documentation Updates Required

**1. Checkpoint doc (Section 4)**: Add subsection on security

```markdown
### 4.8 Checkpoint Security and Tamper Protection

Every checkpoint includes an HMAC signature to detect tampering.

**Files in checkpoint**:
- `weights.pt`, `optimizers.pt`, `rng_state.json` — standard checkpoint data
- `config_snapshot/` — frozen configuration
- `cognitive_hash.txt` — mind identity
- `manifest.txt` — SHA256 checksums of all files
- `signature.txt` — HMAC signature of manifest

**Verification on resume**:
1. Recompute manifest from current file contents
2. Verify HMAC signature matches
3. If mismatch → refuse to load, log security alert

**Key management**:
- Signing key stored in `~/.townlet/signing_key.bin` (not in repo)
- Production: use environment variable or HSM
- Key must be kept secret (grants ability to forge signatures)

**Security properties**:
- Cannot edit config_snapshot without detection
- Cannot recompute cognitive_hash and claim "same mind"
- Chain of custody is cryptographically enforced
```

**2. High-Level Design (Section 1.4)**: Update provenance section

```markdown
### 1.4 Provenance By Design

... existing content ...

**4. Signed Checkpoints**

Checkpoints include HMAC signatures for tamper detection:
- Manifest contains SHA256 of every file
- Signature = HMAC(manifest, signing_key)
- On resume: verify signature before loading

**Security property**: Cannot modify checkpoint without detection, ensuring provenance chain integrity.
```

**3. Implementation guide (Section 10)**: Add to Week 1

```markdown
### 10.1 Week 1: Fix Blockers

- [ ] Make EthicsFilter deterministic
- [ ] Add checkpoint signatures  # ← NEW
  - Implement SecureCheckpointWriter
  - Generate signing key
  - Update checkpoint writer/loader to use signatures
  - Test tampering detection (modify file, verify fails)
- [ ] Add world_config_hash to observations
- [ ] Update docs
```

**WEP** that HMAC signatures are sufficient for security: high (~90%). For high-security deployments, consider adding timestamping or blockchain anchoring.

---

### 6.4 Blocker Summary & Recommended Fix Order

| Blocker | Severity | Effort | Fix Order | Why This Order |
|---------|----------|--------|-----------|----------------|
| **1. EthicsFilter** | CRITICAL | Low | FIRST | Blocks all governance claims |
| **2. Checkpoint Integrity** | HIGH | Medium | SECOND | Enables secure development |
| **3. World Model + Curriculum** | HIGH | Medium | THIRD | Affects experiment validity |

**Week 1 schedule**:

- **Day 1-2**: Fix EthicsFilter (pure function, no weights)
- **Day 3-4**: Implement checkpoint signatures
- **Day 5**: Add world_config_hash to observations, update Module B
- **Day 6-7**: Update all documentation, run validation tests

**Validation tests** (write these):

```python
# tests/test_blockers.py

def test_ethics_filter_has_no_weights():
    """Blocker 1: EthicsFilter must be deterministic."""
    ethics_filter = EthicsFilter(config)
    assert not hasattr(ethics_filter, 'parameters')
    assert not isinstance(ethics_filter, nn.Module)

def test_checkpoint_detects_tampering():
    """Blocker 3: Checkpoint signatures must work."""
    signer = SecureCheckpointWriter(test_key)
    signer.write_checkpoint(checkpoint_dir, payload)
    
    # Tamper with file
    (checkpoint_dir / 'config_snapshot/cognitive_topology.yaml').write_text("# hacked")
    
    # Verification should fail
    with pytest.raises(CheckpointTamperedError):
        signer.verify_checkpoint(checkpoint_dir)

def test_world_config_hash_in_observation():
    """Blocker 2: World config changes must be observable."""
    env = VectorizedTownletEnv(config)
    obs = env.reset()
    
    assert 'world_config_hash' in obs
    
    # Change world config (fork)
    env.load_new_config(austerity_config)
    obs2 = env.step(action)
    
    assert obs['world_config_hash'] != obs2['world_config_hash']
```

After these fixes, you can credibly claim:

- ✅ "Ethics are provably enforced" (Blocker 1)
- ✅ "Checkpoints are tamper-proof" (Blocker 3)
- ✅ "World model adapts to curriculum" (Blocker 2)

**WEP** that these three blockers are the only critical issues: high (~85%). There may be other bugs, but these three block any serious deployment or publication.

---

## SECTION 7: MISSING SPECIFICATIONS

These are not bugs or blockers — they're underspecified behaviors that need explicit documentation so implementations are deterministic and reproducible.

**Priority**: Fix after blockers, before any multi-agent (L6+) or family (L8) experiments.

---

### 7.1 Multi-Agent Contention Resolution

**Status**: UNDERSPECIFIED  
**Affects**: L6-L8 (multi-agent levels)  
**Current state**: High-Level Design mentions "deterministic tie-breaking" but doesn't specify the algorithm

#### The Problem

When multiple agents try to use the same affordance simultaneously:

```python
# Tick 842
agent_2.action = INTERACT  # at Job position (6,6)
agent_5.action = INTERACT  # also at Job position (6,6)

# Job has capacity: 1 (only one agent can use it)
# Who wins?
```

**Your documentation says**:
> "Tie-breaking is deterministic (not random)"

**But doesn't specify**:

- What's the tie-breaking rule?
- Distance? Agent ID? First-come-first-served?
- What happens to the loser?

**Why this matters**:

- Non-deterministic resolution → experiments aren't reproducible
- Unclear rules → agents can't learn optimal strategies
- Different implementations → results can't be compared across codebases

#### The Solution: Distance-First, Then Agent ID

**Rule**: When multiple agents attempt to INTERACT with the same capacity-1 affordance on the same tick, resolve by:

1. **Distance**: Closest agent wins
2. **Tie-breaker**: If equidistant, lowest agent_id wins

**Implementation**:

```python
# environment/vectorized_env.py

def resolve_contention(self, affordance_id: str, tick: int):
    """
    Resolve contention for capacity-1 affordances.
    
    Returns:
        winner_id: agent_id that gets access
        losers: list of agent_ids that were blocked
    """
    affordance = self.affordances[affordance_id]
    
    # Find all agents trying to INTERACT with this affordance
    candidates = []
    for agent_id, agent in enumerate(self.agents):
        if agent.current_action == Action.INTERACT:
            if agent.position == affordance.position:
                candidates.append(agent_id)
    
    if len(candidates) <= affordance.capacity:
        # No contention, everyone succeeds
        return candidates, []
    
    # Contention: rank by distance, then agent_id
    def rank_key(agent_id):
        agent = self.agents[agent_id]
        distance = manhattan_distance(agent.position, affordance.position)
        return (distance, agent_id)  # tuple sorts by distance first, then ID
    
    candidates.sort(key=rank_key)
    
    winners = candidates[:affordance.capacity]
    losers = candidates[affordance.capacity:]
    
    # Log contention for telemetry
    self.log_contention(
        tick=tick,
        affordance=affordance_id,
        winners=winners,
        losers=losers
    )
    
    return winners, losers

def manhattan_distance(pos1, pos2):
    """Manhattan distance in grid world."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

**Example scenarios**:

**Scenario 1: Different distances**

```python
# Job at (6, 6), capacity: 1
agent_2.position = (6, 5)  # distance = 1
agent_5.position = (6, 4)  # distance = 2

# Both try INTERACT
# Winner: agent_2 (closer)
# Result: agent_2 works, agent_5's action fails
```

**Scenario 2: Same distance**

```python
# Job at (6, 6), capacity: 1
agent_2.position = (5, 6)  # distance = 1
agent_5.position = (6, 5)  # distance = 1

# Both equidistant
# Winner: agent_2 (lower agent_id)
# Result: agent_2 works, agent_5's action fails
```

**Scenario 3: Capacity > 1**

```python
# Hospital at (3, 8), capacity: 2
agent_1.position = (3, 7)  # distance = 1
agent_3.position = (3, 6)  # distance = 2
agent_7.position = (2, 8)  # distance = 1

# Three agents try INTERACT, capacity = 2
# Ranked: [agent_1 (d=1, id=1), agent_7 (d=1, id=7), agent_3 (d=2, id=3)]
# Winners: agent_1, agent_7 (top 2 by distance, tie-break by ID)
# Loser: agent_3
```

**What happens to losers**:

```python
# After contention resolution
for loser_id in losers:
    # Action fails, agent stays in current position
    self.agents[loser_id].interaction_failed = True
    
    # Agent receives observation flag
    self.observations[loser_id]['action_failed'] = True
    self.observations[loser_id]['failure_reason'] = 'contention'
    
    # Agent can learn: "I tried Job, it was occupied"
    # Next time: check for other agents before attempting
```

**Why this rule**:

✅ **Deterministic**: Same positions → same outcome  
✅ **Intuitive**: Closer agents win (realistic spatial reasoning)  
✅ **Fair**: When equidistant, lowest ID wins (arbitrary but consistent)  
✅ **Learnable**: Agents can predict outcomes and plan around contention  

**Alternative considered: First-come-first-served**

```python
# Track when each agent arrived at affordance
rank_key = lambda agent_id: (arrival_tick[agent_id], agent_id)
```

**Rejected because**:

- Requires tracking arrival history (more state)
- "Camping" strategy dominates (arrive early, wait)
- Doesn't reward spatial planning as much

**WEP** that distance-first is optimal: high (~80%). First-come might be equally valid but distance feels more intuitive.

#### Documentation Updates Required

**1. Add to Universe as Code (affordances.yaml docs)**:

```yaml
# affordances.yaml

affordances:
  - id: "job"
    # ... other fields ...
    capacity: 1
    
    # Contention resolution (when capacity is exceeded):
    # 1. Closest agent(s) win (Manhattan distance)
    # 2. Tie-breaker: lowest agent_id
    # 3. Losers receive action_failed=True in observation
```

**2. Add new doc: `docs/multi_agent_mechanics.md`**:

```markdown
## Multi-Agent Contention Resolution

### Affordance Capacity

Each affordance has a `capacity` field (default: 1):
- `capacity: 1` — only one agent can use it per tick
- `capacity: 2+` — multiple agents can use simultaneously
- `capacity: inf` — unlimited (e.g., Park)

### Resolution Algorithm

When more agents attempt INTERACT than capacity allows:

1. **Distance ranking**: Compute Manhattan distance from each agent to affordance
2. **Select winners**: Take top N agents by distance (N = capacity)
3. **Tie-breaking**: If multiple agents equidistant, sort by agent_id (ascending)
4. **Notify losers**: Set `action_failed=True` in their observation

### Example Code

```python
def rank_key(agent_id):
    distance = manhattan_distance(agent.position, affordance.position)
    return (distance, agent_id)

candidates.sort(key=rank_key)
winners = candidates[:capacity]
```

### Learnable Signals

Agents can learn to:

- Predict contention by observing other agents' positions
- Choose alternate affordances when contention is likely
- Arrive early to minimize distance (spatial planning)

### Implementation

See: `environment/vectorized_env.py::resolve_contention()`

```

**3. Add to High-Level Design (Section 11)**:

```markdown
### 11.X Multi-Agent Contention

When multiple agents compete for limited-capacity affordances:
- Distance-based resolution (closest wins)
- Tie-breaker: agent_id
- Losers receive explicit failure signal
- This creates strategic resource allocation problems for agents to solve
```

---

### 7.2 Family Lifecycle State Machine

**Status**: UNDERSPECIFIED  
**Affects**: L8 (family communication), population genetics  
**Current state**: High-Level Design mentions families but doesn't specify all transitions

#### The Problem

The family system has many edge cases:

**Scenario 1: Child leaves home**

- Parents had one child (breeding locked)
- Child reaches maturity age (age > 0.3?)
- Child disconnects from family
- **Question**: Can parents breed again immediately? What's the eligibility check?

**Scenario 2: One parent dies**

- Family had 2 parents, 1 child
- Parent A dies (health=0)
- **Question**: Does family persist? Can Parent B remarry? Is child orphaned or still in single-parent family?

**Scenario 3: Child dies before maturity**

- Family had 2 parents, 1 child
- Child dies at age=0.1 (early death)
- **Question**: Can parents breed again? What's the cooldown?

**Scenario 4: Both parents die**

- Family had 2 parents, 1 child
- Both parents die
- **Question**: Is child orphaned? Becomes solo agent? Can still use family channel?

**Scenario 5: Remarriage eligibility**

- Agent A was in family_1 (spouse died)
- Agent A is now solo
- Agent B is also solo (never married)
- **Question**: Can A and B form family_2? What are the rules?

**Why this matters**:

- Population dynamics depend on precise rules
- Family communication experiments need stable families
- Genetics experiments need predictable inheritance

#### The Solution: Complete State Machine

**Family states**:

```python
class FamilyState(Enum):
    NO_FAMILY = 0         # solo agent, never been in family
    MARRIED_NO_CHILD = 1  # two parents, no child yet
    MARRIED_WITH_CHILD = 2  # two parents, one child (breeding locked)
    SINGLE_PARENT = 3     # one parent, one child (other parent died)
    ORPHAN = 4            # child, both parents died
    DIVORCED = 5          # was married, child left, now solo (can remarry)
```

**Transitions**:

```python
# Complete state machine for family lifecycle

TRANSITIONS = {
    # From NO_FAMILY (solo agent)
    NO_FAMILY: {
        'meet_partner': MARRIED_NO_CHILD,  # eligibility check passes
    },
    
    # From MARRIED_NO_CHILD (married, no kids yet)
    MARRIED_NO_CHILD: {
        'breed': MARRIED_WITH_CHILD,  # child spawned
        'partner_dies': NO_FAMILY,    # back to solo (can remarry)
    },
    
    # From MARRIED_WITH_CHILD (locked, raising child)
    MARRIED_WITH_CHILD: {
        'child_leaves': MARRIED_NO_CHILD,   # child matured, can breed again
        'child_dies': MARRIED_NO_CHILD,     # child died, can breed again
        'one_parent_dies': SINGLE_PARENT,   # now single parent with child
        'both_parents_die': ORPHAN,         # child becomes orphan
    },
    
    # From SINGLE_PARENT (one parent, one child)
    SINGLE_PARENT: {
        'child_leaves': NO_FAMILY,   # back to solo, can remarry
        'child_dies': NO_FAMILY,     # back to solo, can remarry
        'parent_dies': ORPHAN,       # child becomes orphan
    },
    
    # From ORPHAN (child with no parents)
    ORPHAN: {
        'reach_adulthood': NO_FAMILY,  # mature, become independent
        'die': None,                   # removed from population
    },
    
    # From DIVORCED (was in family, child left, now solo)
    DIVORCED: {
        'meet_partner': MARRIED_NO_CHILD,  # can remarry
    },
}
```

**Breeding eligibility rules**:

```python
def check_breeding_eligibility(agent_a, agent_b, population):
    """
    Two agents can breed if:
    1. Both are in MARRIED_NO_CHILD state (same family)
    2. Neither has existing child
    3. Population has space (current_pop < max_pop)
    4. Both agents are adults (age > maturity_threshold)
    5. High-performer criteria met (optional, for meritocratic selection)
    """
    # Must be married to each other
    if agent_a.family_id != agent_b.family_id:
        return False, "not in same family"
    
    # Must be in MARRIED_NO_CHILD state
    if agent_a.family_state != FamilyState.MARRIED_NO_CHILD:
        return False, f"agent_a state is {agent_a.family_state}"
    if agent_b.family_state != FamilyState.MARRIED_NO_CHILD:
        return False, f"agent_b state is {agent_b.family_state}"
    
    # Population cap
    if len(population) >= population.max_size:
        return False, "population at capacity"
    
    # Maturity check
    maturity_age = 0.2  # agents must be at least 20% through life
    if agent_a.bars['age'] < maturity_age:
        return False, "agent_a too young"
    if agent_b.bars['age'] < maturity_age:
        return False, "agent_b too young"
    
    # High-performer check (for meritocratic mode)
    if population.breeding_mode == "meritocratic":
        performance_threshold = 0.6  # top 60% of population
        if not is_high_performer(agent_a, population):
            return False, "agent_a performance below threshold"
        if not is_high_performer(agent_b, population):
            return False, "agent_b performance below threshold"
    
    return True, "eligible"
```

**Child maturity rules**:

```python
def check_child_maturity(child_agent):
    """
    Child leaves family when:
    - age > 0.3 (30% through life)
    OR
    - terminal score > 0.5 on their first "test episode"
    
    This creates graduation mechanics.
    """
    if child_agent.bars['age'] > 0.3:
        return True, "reached maturity age"
    
    # Optional: competency-based graduation
    if hasattr(child_agent, 'test_score'):
        if child_agent.test_score > 0.5:
            return True, "passed competency test"
    
    return False, None
```

**Remarriage rules**:

```python
def check_remarriage_eligibility(agent_a, agent_b):
    """
    Two solo agents can marry if:
    1. Both are in NO_FAMILY or DIVORCED state
    2. Both are adults (age > maturity_threshold)
    3. Not related (no parent-child relationship)
    """
    valid_states = {FamilyState.NO_FAMILY, FamilyState.DIVORCED}
    
    if agent_a.family_state not in valid_states:
        return False, f"agent_a state is {agent_a.family_state}"
    if agent_b.family_state not in valid_states:
        return False, f"agent_b state is {agent_b.family_state}"
    
    # Maturity check
    maturity_age = 0.2
    if agent_a.bars['age'] < maturity_age:
        return False, "agent_a too young"
    if agent_b.bars['age'] < maturity_age:
        return False, "agent_b too young"
    
    # Prevent incest (parent can't marry their child)
    if agent_a.agent_id in agent_b.lineage or agent_b.agent_id in agent_a.lineage:
        return False, "agents are related"
    
    return True, "eligible"
```

**Example scenarios resolved**:

**Scenario 1: Child leaves**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.family_state = CHILD  # (not shown in enum, implicit)

# Trigger: child.age > 0.3
child.family_state = NO_FAMILY
child.family_id = None

parent_a.family_state = MARRIED_NO_CHILD  # can breed again
parent_b.family_state = MARRIED_NO_CHILD
```

**Scenario 2: One parent dies**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.family_state = CHILD

# Trigger: parent_a.health = 0 (dies)
parent_a.remove_from_population()

parent_b.family_state = SINGLE_PARENT  # still raising child
child.family_state = CHILD  # still in family with parent_b

# Later: child leaves
child.family_state = NO_FAMILY
parent_b.family_state = DIVORCED  # can remarry
```

**Scenario 3: Child dies early**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.age = 0.1

# Trigger: child.health = 0 (dies)
child.remove_from_population()

parent_a.family_state = MARRIED_NO_CHILD  # immediately eligible to breed again
parent_b.family_state = MARRIED_NO_CHILD
```

**Scenario 4: Both parents die**

```python
# Initial state
parent_a.family_state = MARRIED_WITH_CHILD
parent_b.family_state = MARRIED_WITH_CHILD
child.family_state = CHILD

# Trigger: both parents die
parent_a.remove_from_population()
parent_b.remove_from_population()

child.family_state = ORPHAN
child.family_id = None  # no longer has family channel access

# Orphan continues as solo agent until:
# - Reaches adulthood (age > 0.3) → NO_FAMILY state
# - Dies (removed from population)
```

**Scenario 5: Remarriage**

```python
# agent_a was in family_1 (spouse died), now DIVORCED
# agent_b was never married, now NO_FAMILY

# Check eligibility
eligible, reason = check_remarriage_eligibility(agent_a, agent_b)
# → True, "eligible"

# Create new family
new_family_id = generate_family_id()  # e.g., "family_137"
agent_a.family_id = new_family_id
agent_a.family_state = MARRIED_NO_CHILD
agent_b.family_id = new_family_id
agent_b.family_state = MARRIED_NO_CHILD

# They can now breed (once eligibility is checked)
```

#### Documentation Updates Required

**1. Create new doc: `docs/family_lifecycle.md`**:

```markdown
## Family Lifecycle State Machine

### States

- **NO_FAMILY**: Solo agent, never been in a family
- **MARRIED_NO_CHILD**: Two parents, no child yet (can breed)
- **MARRIED_WITH_CHILD**: Two parents, one child (breeding locked)
- **SINGLE_PARENT**: One parent, one child (other parent died)
- **ORPHAN**: Child with no parents
- **DIVORCED**: Was married, child left, now solo (can remarry)

### State Transitions

[Include state machine diagram or table]

### Breeding Eligibility

Two agents can breed if:
1. Both in MARRIED_NO_CHILD state (same family_id)
2. Population < max_size
3. Both age > 0.2 (maturity threshold)
4. (Optional) Both are high-performers (meritocratic mode)

### Child Maturity

Child leaves family when:
- age > 0.3 (standard)
- OR passes competency test (optional)

On leaving:
- Child → NO_FAMILY state
- Parents → MARRIED_NO_CHILD (can breed again)

### Death Handling

**One parent dies**:
- Remaining parent → SINGLE_PARENT
- Child stays in family
- When child leaves later, parent → DIVORCED (can remarry)

**Both parents die**:
- Child → ORPHAN (loses family channel access)
- At age > 0.3, orphan → NO_FAMILY

**Child dies**:
- Parents → MARRIED_NO_CHILD (can breed again immediately)

### Remarriage

Solo agents can form new families if:
- Both in NO_FAMILY or DIVORCED state
- Both age > 0.2
- Not related (no parent-child lineage)

### Implementation

See: `population/dynasty_manager.py`
```

**2. Add to High-Level Design (Section 14)**:

```markdown
## 14. Population Genetics and Family Dynamics

### 14.3 Family State Machine

[Include diagram and rules]

### 14.4 Population Equilibrium

With these rules:
- Initial population: 100
- Max population: 100
- Expected equilibrium: 70-80 agents in families, 20-30 solo
- Churn rate: ~5% per 1000 ticks (deaths + births)
```

**3. Add configuration schema to `population_genetics.yaml`**:

```yaml
population_genetics:
  enabled: true
  
  population:
    initial_size: 100
    max_size: 100
    breeding_mode: "meritocratic"  # or "random", "arranged"
  
  family_rules:
    maturity_age: 0.3      # when child can leave family
    min_breeding_age: 0.2  # parents must be this old
    one_child_policy: true # lock breeding while child exists
    
  remarriage:
    enabled: true
    prevent_incest: true
    
  death_replacement:
    enabled: true
    cull_policy: "worst_performers"  # replace deaths by removing worst agents
```

**WEP** that this state machine is complete: high (~85%). May need minor adjustments for edge cases.

---

### 7.3 Child Initialization

**Status**: UNDERSPECIFIED  
**Affects**: L8 (inheritance), population genetics  
**Current state**: Mentioned but weights/DNA initialization not specified

#### The Problem

When a new child is spawned from two parents:

**Question 1: Weights initialization**

- Random initialization (start from scratch)?
- Clone one parent (inherit full policy)?
- Average parents' weights (literal parameter averaging)?
- Pretrained baseline (from L7 curriculum)?

**Question 2: DNA initialization**

- Clone one parent's DNA?
- Crossover (mix both parents' DNA)?
- Mutation applied?

**Question 3: Training**

- Does child train from scratch?
- Transfer learning from parents?
- Frozen weights, or trainable?

**Why this matters**:

- Random init → no inheritance (defeats genetics experiments)
- Pure clone → no diversity (population converges)
- Crossover without mutation → limited exploration
- Pretrained → faster learning but less "natural" evolution

#### The Solution: Configurable Inheritance Modes

**Default mode: Crossover + Mutation**

```python
# population/child_initialization.py

def initialize_child(parent_a, parent_b, config):
    """
    Initialize child agent from two parents.
    
    Returns:
        child_agent: new agent with inherited weights and DNA
    """
    child_agent = Agent(
        agent_id=generate_agent_id(),
        config=config
    )
    
    # 1. DNA crossover + mutation
    child_agent.dna = crossover_dna(
        parent_a.dna,
        parent_b.dna,
        crossover_rate=config.genetics.crossover_rate
    )
    
    child_agent.dna = mutate_dna(
        child_agent.dna,
        mutation_rate=config.genetics.mutation_rate
    )
    
    # 2. Weights initialization (configurable)
    if config.genetics.weight_init == "crossover":
        # Average parents' weights
        child_agent.weights = average_weights(
            parent_a.weights,
            parent_b.weights
        )
    
    elif config.genetics.weight_init == "clone_best":
        # Clone better parent
        better_parent = parent_a if parent_a.lifetime_reward > parent_b.lifetime_reward else parent_b
        child_agent.weights = copy.deepcopy(better_parent.weights)
    
    elif config.genetics.weight_init == "pretrained":
        # Start from L7 curriculum baseline
        checkpoint = load_checkpoint(config.genetics.pretrained_path)
        child_agent.weights = checkpoint['weights']
    
    elif config.genetics.weight_init == "random":
        # Start from scratch (no inheritance)
        child_agent.weights = initialize_random_weights()
    
    # 3. Training mode
    if config.genetics.child_training == "frozen":
        # Weights are inherited but not updated
        for param in child_agent.parameters():
            param.requires_grad = False
    
    elif config.genetics.child_training == "finetune":
        # Start with inherited weights, continue training with lower LR
        for param in child_agent.parameters():
            param.requires_grad = True
        child_agent.learning_rate = config.genetics.finetune_lr  # e.g., 0.1× parent LR
    
    elif config.genetics.child_training == "full":
        # Start with inherited weights, train normally
        for param in child_agent.parameters():
            param.requires_grad = True
        child_agent.learning_rate = config.genetics.learning_rate
    
    return child_agent
```

**DNA crossover**:

```python
def crossover_dna(dna_a, dna_b, crossover_rate=0.5):
    """
    Combine two parent DNAs with genetic crossover.
    
    DNA structure (from cognitive_topology.yaml):
    {
        'personality': {
            'greed': float,
            'curiosity': float,
            'neuroticism': float,
            'agreeableness': float,
        },
        'panic_thresholds': {
            'energy': float,
            'health': float,
        },
        'compliance': {
            'forbid_actions': list,
            'penalize_actions': list,
        }
    }
    """
    child_dna = {}
    
    for key in dna_a.keys():
        if isinstance(dna_a[key], dict):
            # Recurse for nested dicts
            child_dna[key] = crossover_dna(dna_a[key], dna_b[key], crossover_rate)
        
        elif isinstance(dna_a[key], float):
            # Crossover continuous values
            if random.random() < crossover_rate:
                # Take from parent_a
                child_dna[key] = dna_a[key]
            else:
                # Take from parent_b
                child_dna[key] = dna_b[key]
        
        elif isinstance(dna_a[key], list):
            # For lists (like forbid_actions), take union or intersection
            if key == 'forbid_actions':
                # Child inherits stricter rules (union of both parents)
                child_dna[key] = list(set(dna_a[key]) | set(dna_b[key]))
            else:
                # For other lists, random choice
                child_dna[key] = dna_a[key] if random.random() < 0.5 else dna_b[key]
    
    return child_dna
```

**DNA mutation**:

```python
def mutate_dna(dna, mutation_rate=0.05):
    """
    Apply random mutations to DNA.
    
    For each gene:
    - With probability mutation_rate, perturb the value
    - Perturbation is small (±10% for continuous values)
    """
    mutated = copy.deepcopy(dna)
    
    for key, value in mutated.items():
        if isinstance(value, dict):
            # Recurse for nested dicts
            mutated[key] = mutate_dna(value, mutation_rate)
        
        elif isinstance(value, float):
            if random.random() < mutation_rate:
                # Add Gaussian noise (σ = 0.1 of current value)
                noise = random.gauss(0, 0.1 * abs(value))
                mutated[key] = np.clip(value + noise, 0.0, 1.0)
        
        # Lists (like forbid_actions) don't mutate by default
        # Could add/remove actions with very low probability if desired
    
    return mutated
```

**Weight averaging** (for crossover mode):

```python
def average_weights(weights_a, weights_b):
    """
    Average two sets of neural network weights.
    
    Returns:
        averaged_weights: parameter-wise mean
    """
    averaged = {}
    
    for name in weights_a.keys():
        if isinstance(weights_a[name], torch.Tensor):
            averaged[name] = (weights_a[name] + weights_b[name]) / 2.0
        else:
            # For non-tensor values, pick randomly
            averaged[name] = weights_a[name] if random.random() < 0.5 else weights_b[name]
    
    return averaged
```

**Configuration example**:

```yaml
# population_genetics.yaml

genetics:
  # DNA inheritance
  crossover_rate: 0.5      # 50% genes from each parent
  mutation_rate: 0.05      # 5% chance per gene
  
  # Weight initialization
  weight_init: "crossover"  # options: crossover, clone_best, pretrained, random
  pretrained_path: "checkpoints/L7_baseline/step_10000/"  # if using pretrained
  
  # Training mode
  child_training: "finetune"  # options: frozen, finetune, full
  finetune_lr: 0.0001         # learning rate for children (if finetune)
  learning_rate: 0.001        # learning rate for adults
```

**Research modes**:

**Mode 1: Pure Genetics (no learning)**

```yaml
weight_init: "crossover"
child_training: "frozen"
```

- Tests: Can good policies evolve through selection alone?
- No individual learning, only population-level evolution

**Mode 2: Lamarckian (learning + inheritance)**

```yaml
weight_init: "crossover"
child_training: "full"
```

- Tests: Does learned behavior + inheritance accelerate adaptation?
- Children start with parents' knowledge, improve via learning

**Mode 3: Tabula Rasa + DNA**

```yaml
weight_init: "random"
child_training: "full"
```

- Tests: Does personality (DNA) matter if weights reset?
- Children inherit personality traits but not skills

**Mode 4: Pretrained Baseline**

```yaml
weight_init: "pretrained"
child_training: "finetune"
```

- Tests: Fastest path to competent population
- All children start from L7 curriculum, specialize via DNA

#### Documentation Updates Required

**1. Add to `docs/population_genetics.md`**:

```markdown
## Child Initialization

When parents breed, the child is initialized with:

### 1. DNA Inheritance

DNA (personality, thresholds, compliance rules) is combined via:
- **Crossover**: 50% genes from each parent (configurable)
- **Mutation**: 5% chance of perturbation per gene (configurable)

Example:
- parent_a.greed = 0.8
- parent_b.greed = 0.4
- child.greed = 0.8 (50% chance) or 0.4 (50% chance)
- mutation: child.greed = 0.83 (if mutation occurs)

### 2. Weight Initialization

Neural network weights are initialized via (configurable):

**Crossover** (default):
- Average both parents' weights parameter-wise
- Produces intermediate policy between parents

**Clone Best**:
- Copy better-performing parent's weights
- Faster learning, less diversity

**Pretrained**:
- Load from L7 curriculum checkpoint
- Most competent starting point

**Random**:
- Initialize from scratch
- No inheritance (pure learning)

### 3. Training Mode

**Frozen**:
- Weights inherited but not updated
- Pure genetic evolution (no individual learning)

**Finetune** (default):
- Start with inherited weights
- Continue training with lower learning rate
- Lamarckian evolution (learning + inheritance)

**Full**:
- Start with inherited weights
- Train normally
- Maximum plasticity

### Configuration

See: `population_genetics.yaml`
```

**2. Add to High-Level Design (Section 14.5)**:

```markdown
### 14.5 Child Initialization Modes

[Table of modes and research questions]

**Recommended default**: Crossover weights + finetune training
- Balances inheritance and learning
- Produces diverse population
- Enables both genetic and behavioral evolution
```

**WEP** that crossover + finetune is best default: moderate (~70%). Could argue for pretrained for faster experiments.

---

### 7.4 Missing Specifications Summary & Where They Belong

| Specification | Priority | Goes In | Estimated Effort |
|---------------|----------|---------|------------------|
| **Contention resolution** | HIGH | `docs/multi_agent_mechanics.md` + affordances.yaml | 1 day |
| **Family lifecycle** | HIGH | `docs/family_lifecycle.md` + population_genetics.yaml | 2 days |
| **Child initialization** | MEDIUM | `docs/population_genetics.md` + population_genetics.yaml | 1 day |
| Operating hours enforcement | LOW | Already specified in affordances.yaml | - |
| Panic override rules | LOW | Already in cognitive_topology.yaml | - |
| Cascade evaluation order | LOW | cascades.yaml (add `priority` field) | 0.5 days |

**Week 2 schedule** (after blockers fixed):

**Day 1**: Write multi-agent mechanics docs

- Document contention algorithm
- Add to affordances.yaml schema
- Write tests for tie-breaking

**Day 2**: Write family lifecycle docs

- Complete state machine
- Add all edge case handlers
- Document in family_lifecycle.md

**Day 3**: Write population genetics docs

- Child initialization modes
- Configuration options
- Research mode guide

**Day 4**: Implement validators

```python
# tests/test_specifications.py

def test_contention_is_deterministic():
    """Same positions → same winner."""
    env = VectorizedTownletEnv(config)
    
    # Scenario: two agents, one affordance
    agents = [
        Agent(position=(6, 5)),  # agent_0
        Agent(position=(5, 6))   # agent_1
    ]
    
    affordance = Affordance(id="job", position=(6, 6), capacity=1)
    
    # Both try INTERACT
    actions = [Action.INTERACT, Action.INTERACT]
    
    # Resolve 100 times (should be identical)
    winners = [env.resolve_contention("job", tick=i)[0] for i in range(100)]
    
    # All should be same winner
    assert len(set(winners)) == 1  # only one unique winner across all runs
    assert winners[0] == [0]  # agent_0 wins (equidistant, lower ID)

def test_family_state_transitions():
    """Verify all state machine transitions."""
    family = Family(parent_a, parent_b)
    
    # Initial state
    assert family.state == FamilyState.MARRIED_NO_CHILD
    
    # Breed
    child = family.spawn_child()
    assert family.state == FamilyState.MARRIED_WITH_CHILD
    
    # Child leaves
    child.age = 0.4  # past maturity
    family.process_maturity()
    assert family.state == FamilyState.MARRIED_NO_CHILD
    
    # One parent dies
    parent_a.health = 0
    family.process_deaths()
    assert family.state == FamilyState.SINGLE_PARENT
    
    # [test remaining transitions...]

def test_child_inherits_dna():
    """Child DNA is crossover of parents."""
    parent_a = Agent(dna={'greed': 0.8, 'curiosity': 0.3})
    parent_b = Agent(dna={'greed': 0.4, 'curiosity': 0.7})
    
    child = initialize_child(parent_a, parent_b, config)
    
    # Child's greed should be one of parent values (or mutated nearby)
    assert child.dna['greed'] in [0.8, 0.4] or abs(child.dna['greed'] - 0.8) < 0.2
    
    # Crossover should mix traits
    # (Test with multiple children to check distribution)
```

**Day 5**: Add to QUICKSTART.md

- How to configure families
- How to interpret contention logs
- How to set inheritance modes

**Validation criteria**:

- ✅ All state machine transitions documented
- ✅ All edge cases handled
- ✅ Configuration schemas validated
- ✅ Tests pass for determinism
- ✅ Examples in docs

After Week 2, you can credibly claim:

- "Multi-agent interactions are fully specified"
- "Family dynamics are completely defined"
- "Population genetics are configurable and reproducible"

**WEP** that these specs are sufficient for L6-L8 experiments: high (~85%).

---

## SECTION 8: POPULATION GENETICS & FAMILIES

### 8.1 Official Rules: Meritocratic Churn

The baseline population genetics system implements **meritocratic churn with one-child families**.

**Core principles**:

1. **One-child policy**: Families can have exactly one child at a time
2. **Breeding locked while raising**: Cannot breed again until child leaves
3. **High-performer selection**: Only successful agents breed
4. **Constant population**: Deaths are replaced by culling worst performers
5. **Natural churn**: Families dissolve and reform as children mature

**Configuration**:

```yaml
# population_genetics.yaml (baseline mode)

population:
  initial_size: 100
  max_size: 100  # constant population
  replacement_policy: "cull_worst"  # when someone dies, remove worst performer
  
family_formation:
  mode: "meritocratic"
  eligibility:
    min_age: 0.2  # must be 20% through life
    performance_threshold: 0.6  # top 60% of population
    family_state: ["NO_FAMILY", "DIVORCED"]
  
  pairing_strategy: "complementary_dna"  # match compatible personalities
  
breeding:
  one_child_policy: true
  child_leaves_at_age: 0.3  # child matures at 30% through life
  
child_initialization:
  weight_init: "crossover"  # average parents' neural weights
  training_mode: "finetune"  # continue learning from inherited weights
  
  dna_inheritance:
    crossover_rate: 0.5  # 50% genes from each parent
    mutation_rate: 0.05  # 5% chance per gene
    
death_and_replacement:
  natural_death_age: 1.0  # retirement
  early_death: ["energy <= 0", "health <= 0"]
  
  on_death:
    - remove_from_population
    - identify_worst_performer  # by lifetime reward
    - cull_worst_to_maintain_population_cap
```

**Population dynamics**:

```python
# At initialization (tick 0)
population = create_random_agents(100)
# All agents start as NO_FAMILY, random weights, random DNA

# At tick 10,000 (agents have learned survival)
evaluate_all_agents()  # compute lifetime rewards
high_performers = top_60_percent(population)

# Form initial families
for _ in range(20):  # form 20 families (40 agents paired)
    parent_a, parent_b = select_complementary_pair(high_performers)
    create_family(parent_a, parent_b)
    # Both parents: NO_FAMILY → MARRIED_NO_CHILD

# Families breed
for family in families:
    if family.state == MARRIED_NO_CHILD:
        child = spawn_child(family.parent_a, family.parent_b)
        family.state = MARRIED_WITH_CHILD
        population.append(child)  # population = 100 → 120

# 20 worst performers are culled to maintain cap
worst_20 = bottom_20_by_lifetime_reward(population)
for agent in worst_20:
    remove_from_population(agent)
# population = 120 → 100 (back to cap)

# At tick 20,000 (children are maturing)
for family in families:
    if family.child.age > 0.3:
        # Child leaves home
        family.child.family_id = None
        family.child.family_state = NO_FAMILY
        
        # Parents can breed again
        family.parent_a.family_state = MARRIED_NO_CHILD
        family.parent_b.family_state = MARRIED_NO_CHILD
        
# If parent dies while child is young
parent_a.health = 0  # death
family.state = SINGLE_PARENT
# Child continues with one parent
# When child leaves, remaining parent → DIVORCED, can remarry
```

**Expected equilibrium** (after 50k ticks):

- 70-80 agents in families (35-40 families)
- 20-30 solo agents (divorced, orphaned, or never married)
- Constant churn: ~5% births/deaths per 1000 ticks
- High performers dominate breeding pool
- Low performers get culled before reproducing

**Why meritocratic as baseline**:

- Tests selection pressure (do good strategies propagate?)
- Prevents population collapse (only competent agents breed)
- Creates competition (agents must perform to pass on genes)
- Reflects real-world success → reproduction correlation

**WEP** that meritocratic mode is stable: high (~85%). May need tuning of performance_threshold.

---

### 8.2 The Complete Family Lifecycle (Reference)

See **Section 7.2** for full state machine specification.

**Quick reference**:

```
States:
NO_FAMILY → [meet partner] → MARRIED_NO_CHILD
MARRIED_NO_CHILD → [breed] → MARRIED_WITH_CHILD
MARRIED_WITH_CHILD → [child leaves] → MARRIED_NO_CHILD
                   → [one parent dies] → SINGLE_PARENT
                   → [child dies] → MARRIED_NO_CHILD
SINGLE_PARENT → [child leaves] → DIVORCED
              → [child dies] → DIVORCED
DIVORCED → [remarry] → MARRIED_NO_CHILD
ORPHAN → [reach adulthood] → NO_FAMILY
```

**Key transitions for population experiments**:

- Child maturity triggers eligibility reset (parents can breed again)
- Death of parent creates single-parent families (not immediate dissolution)
- Orphans can survive and become independent adults
- Remarriage allows multi-generation dynasties

---

### 8.3 Alternative Inheritance Systems

The meritocratic baseline is just one configuration. Alternative modes enable different research questions.

#### Mode 1: Dynasty (Primogeniture + Inheritance)

**Concept**: Families persist across generations, child inherits parents' resources

**Configuration**:

```yaml
# population_genetics_dynasty.yaml

family_formation:
  mode: "dynasty"
  
breeding:
  one_child_policy: false  # ← KEY CHANGE: multiple children allowed
  primogeniture: true      # first child is heir
  
inheritance:
  on_parent_death:
    transfer_wealth: true   # child gets parents' money
    transfer_affordances: false  # (for future: private property)
    keep_family_id: true    # dynasty persists across generations
  
  heir_selection: "oldest"  # or "most_successful"
  
child_initialization:
  weight_init: "clone_best"  # ← inherit best parent's full policy
  training_mode: "frozen"    # ← no learning, pure genetic selection
  
  dna_inheritance:
    crossover_rate: 0.7   # more parental influence
    mutation_rate: 0.02   # less mutation (preserve dynasty traits)
```

**Behavioral differences from baseline**:

```python
# Baseline (meritocratic): Child leaves, becomes independent
child.age = 0.3
child.family_id = None
child.money = 0.50  # starts with initial endowment

# Dynasty mode: Child inherits
parent_a.health = 0  # parent dies
child.family_id = parent_a.family_id  # keeps dynasty name
child.money += parent_a.money  # inherits wealth
# Child becomes head of dynasty, can have own children
```

**Expected population dynamics**:

- 5-10 dynasty lineages emerge (named families)
- Wealth concentration (rich dynasties stay rich)
- Behavioral dynasties (coordinated families develop shared protocols)
- "Aristocracy" vs "peasants" stratification

**Research questions**:

1. **Wealth inequality**: How quickly does wealth concentrate? Does Gini coefficient increase over time?
2. **Coordination advantage**: Do dynasties with shared communication protocols outperform solo agents?
3. **Dynasty collapse**: What causes a dynasty to fail (bad heir, accumulated bad mutations)?
4. **Speciation**: Do different dynasties evolve distinct survival strategies?

**Hypothesis**: Dynasty mode produces more coordinated families but higher inequality.

**WEP** that dynasties will emerge and persist: moderate (~65%). Depends on whether wealth inheritance provides enough advantage.

---

#### Mode 2: Polygamy (Multiple Simultaneous Families)

**Concept**: Agents can belong to multiple family units simultaneously

**Configuration**:

```yaml
# population_genetics_polygamy.yaml

family_formation:
  mode: "polygamous"
  max_families_per_agent: 2  # can be in 2 families at once
  
breeding:
  one_child_policy: true  # per family
  # Agent can have 2 children (one per family) simultaneously
  
child_initialization:
  weight_init: "crossover"
  training_mode: "full"  # children learn independently
```

**Behavioral differences**:

```python
# Agent A is in family_1 with Agent B (child C)
# Agent A is also in family_2 with Agent D (child E)

agent_a.families = [family_1, family_2]
agent_a.family_comm_channels = {
    family_1: [signal_from_b, signal_from_c],
    family_2: [signal_from_d, signal_from_e],
}

# Agent A must manage two separate communication protocols
# Children C and E are half-siblings (share parent A, different co-parents)
```

**Expected population dynamics**:

- Complex kinship networks (overlapping families)
- More communication channels per agent
- Potential for information brokerage (Agent A can coordinate across families)

**Research questions**:

1. **Communication complexity**: Can agents maintain distinct protocols per family?
2. **Strategic alliances**: Do high performers form multiple families to maximize offspring?
3. **Network effects**: Does family overlap create more coordinated populations?
4. **Cognitive load**: Does managing multiple families hurt individual performance?

**Hypothesis**: Polygamy increases coordination complexity; performance depends on Module C capacity.

**WEP** that polygamy is implementable: high (~85%). Trainable: moderate (~60%, depends on architecture).

---

#### Mode 3: Arranged Marriage (DNA Complementarity)

**Concept**: Pairing prioritizes genetic diversity, not just performance

**Configuration**:

```yaml
# population_genetics_arranged.yaml

family_formation:
  mode: "arranged"
  
  pairing_strategy: "maximize_diversity"
  diversity_metric:
    - personality_distance  # pair high-greed with low-greed
    - skill_complementarity  # pair good-at-X with good-at-Y
  
  eligibility:
    min_age: 0.2
    performance_threshold: 0.4  # ← lower bar (60% → 40%)
    # Even mediocre agents can breed if genetically diverse
```

**Pairing algorithm**:

```python
def select_complementary_pair(eligible_agents):
    """
    Find pair with maximum genetic distance.
    
    Diversity score = 
        |greed_a - greed_b| + 
        |curiosity_a - curiosity_b| + 
        |neuroticism_a - neuroticism_b| +
        |agreeableness_a - agreeableness_b|
    """
    max_diversity = -inf
    best_pair = None
    
    for agent_a in eligible_agents:
        for agent_b in eligible_agents:
            if agent_a == agent_b:
                continue
            
            diversity = compute_dna_distance(agent_a.dna, agent_b.dna)
            
            if diversity > max_diversity:
                max_diversity = diversity
                best_pair = (agent_a, agent_b)
    
    return best_pair

def compute_dna_distance(dna_a, dna_b):
    """Euclidean distance in personality space."""
    return sqrt(
        (dna_a['greed'] - dna_b['greed'])**2 +
        (dna_a['curiosity'] - dna_b['curiosity'])**2 +
        (dna_a['neuroticism'] - dna_b['neuroticism'])**2 +
        (dna_a['agreeableness'] - dna_b['agreeableness'])**2
    )
```

**Expected population dynamics**:

- High genetic diversity maintained (no convergence to single personality type)
- Children have more "hybrid vigor" (diverse traits)
- Population covers broader strategy space

**Research questions**:

1. **Exploration-exploitation trade-off**: Does diversity improve population adaptability?
2. **Hybrid vigor**: Do diverse children outperform same-personality children?
3. **Speciation prevention**: Does forced mixing prevent dynasties from diverging?
4. **Optimal diversity level**: Is there a sweet spot (too much diversity = poor coordination)?

**Hypothesis**: Arranged marriage maintains exploration, prevents premature convergence.

**WEP** that arranged marriage increases diversity: very high (~90%). That it improves performance: moderate (~60%).

---

### 8.4 Research Questions Enabled by Population Genetics

#### Category 1: Evolutionary Dynamics

**Q1.1: Does natural selection work in RL?**

- Setup: Meritocratic mode, 50k ticks
- Measure: Mean terminal reward of population over time
- Hypothesis: Population performance increases as good strategies propagate

**Q1.2: Lamarckian vs Darwinian evolution**

- Setup: Compare training_mode "frozen" (pure genetic) vs "full" (learning + inheritance)
- Measure: Convergence speed, final performance
- Hypothesis: Lamarckian (learning + inheritance) converges faster

**Q1.3: Mutation rate tuning**

- Setup: Vary mutation_rate from 0.01 to 0.20
- Measure: Population diversity, performance stability
- Hypothesis: Sweet spot around 0.05 (enough exploration, not too noisy)

#### Category 2: Social Coordination

**Q2.1: Emergent communication protocols**

- Setup: L8 (family channel), track signal usage over time
- Measure:
  - Signal diversity (# unique signals used)
  - Signal stability (P(same signal → same state))
  - Coordination gain (family performance vs solo)
- Hypothesis: Stable protocols emerge by episode 20k

**Q2.2: Protocol transfer across generations**

- Setup: Dynasty mode, track child's signal usage vs parents'
- Measure: Protocol similarity (mutual information I(parent_signals; child_signals))
- Hypothesis: Children learn parents' protocols (cultural transmission)

**Q2.3: Dialect formation**

- Setup: Multiple dynasties, L8
- Measure: Inter-family signal similarity vs intra-family
- Hypothesis: Different families develop different "dialects"

#### Category 3: Inequality & Fairness

**Q3.1: Wealth concentration**

- Setup: Dynasty mode (inheritance enabled)
- Measure: Gini coefficient over time
- Hypothesis: Gini increases (wealth concentrates in successful dynasties)

**Q3.2: Mobility**

- Setup: Dynasty vs meritocratic mode
- Measure: P(child in bottom quartile → adult in top quartile)
- Hypothesis: Meritocratic mode has higher mobility

**Q3.3: Intervention effectiveness**

- Setup: Dynasty mode + wealth redistribution (e.g., tax wealthy dynasties)
- Measure: Gini, population performance
- Hypothesis: Redistribution reduces inequality without harming total performance

#### Category 4: Family Structure

**Q4.1: Optimal family size**

- Setup: Vary max_family_size (2, 3, 4)
- Measure: Coordination effectiveness, individual performance
- Hypothesis: Size 3 (2 parents + 1 child) is optimal balance

**Q4.2: Single-parent outcomes**

- Setup: Track children from SINGLE_PARENT families
- Measure: Adult performance vs two-parent children
- Hypothesis: Single-parent children have lower performance (less coordination training)

**Q4.3: Remarriage stability**

- Setup: Track DIVORCED agents who remarry
- Measure: Success rate of second families vs first families
- Hypothesis: Second families are more stable (learned from experience)

---

### 8.5 Expected Results & Predictions

Based on analogous evolutionary RL experiments (e.g., Neural MMO, Pommerman with population):

**High confidence predictions** (>80%):

1. **Selection works**: Mean population performance increases over 50k ticks in meritocratic mode
2. **Learning helps**: Lamarckian (training_mode="full") converges faster than pure genetic (training_mode="frozen")
3. **Diversity matters**: Some mutation (0.03-0.10) outperforms zero mutation or high mutation (0.20+)
4. **Families coordinate**: L8 families outperform solo agents by 10-30% in contested resource scenarios

**Medium confidence predictions** (50-80%):

5. **Protocols emerge**: Stable signal meanings develop by episode 15k-25k in L8
6. **Dynasties stratify**: Dynasty mode produces 2-3 dominant lineages controlling 50%+ of population by 100k ticks
7. **Wealth concentrates**: Gini coefficient increases from ~0.3 (random) to ~0.6 (high inequality) in dynasty mode
8. **Arranged marriage maintains diversity**: Personality variance stays >0.2 (vs <0.1 in pure meritocratic)

**Low confidence predictions** (30-50%):

9. **Protocol transfer**: Children use >60% of parents' signal vocabulary (cultural transmission)
10. **Dialect formation**: Inter-family signal overlap <30% (distinct family languages)
11. **Hybrid vigor**: Arranged marriage produces higher-performing children than same-personality pairs
12. **Polygamy is trainable**: Agents successfully manage 2+ families without performance collapse

**Open questions** (too uncertain for prediction):

- Do dynasties speciate into distinct strategies (e.g., "farmer dynasties" vs "trader dynasties")?
- Can we decode emergent protocols via causal intervention?
- What's the relationship between personality DNA and communication protocol choice?
- Do orphans develop different survival strategies than family-raised agents?

---

### 8.6 Implementation Overview

**File structure**:

```
townlet/
  population/
    dynasty_manager.py        # family lifecycle state machine
    breeding_selector.py      # eligibility checks, pairing logic
    child_initializer.py      # weight/DNA inheritance
    population_controller.py  # maintains population cap, culling
    genetics_config.py        # dataclasses for config
```

**Key classes**:

```python
# dynasty_manager.py

class Family:
    """Represents a family unit."""
    def __init__(self, parent_a, parent_b):
        self.family_id = generate_family_id()
        self.parent_a = parent_a
        self.parent_b = parent_b
        self.child = None
        self.state = FamilyState.MARRIED_NO_CHILD
        self.creation_tick = current_tick
        
    def can_breed(self) -> bool:
        """Check if family can spawn a child."""
        return self.state == FamilyState.MARRIED_NO_CHILD
    
    def spawn_child(self, config) -> Agent:
        """Create child from parents."""
        child = initialize_child(
            self.parent_a,
            self.parent_b,
            config
        )
        self.child = child
        self.state = FamilyState.MARRIED_WITH_CHILD
        return child
    
    def process_maturity(self):
        """Check if child has matured."""
        if self.child and self.child.age > config.child_leaves_at_age:
            self.child.family_id = None
            self.child.family_state = FamilyState.NO_FAMILY
            self.child = None
            self.state = FamilyState.MARRIED_NO_CHILD
    
    def process_deaths(self):
        """Handle parent or child death."""
        if self.parent_a.health <= 0:
            if self.parent_b.health <= 0:
                # Both parents dead
                if self.child:
                    self.child.family_state = FamilyState.ORPHAN
            else:
                # One parent dead
                self.state = FamilyState.SINGLE_PARENT
```

```python
# breeding_selector.py

class BreedingSelector:
    """Selects which agents form families and breed."""
    
    def select_breeding_pairs(
        self,
        population: list[Agent],
        mode: str,
        config: GeneticsConfig,
    ) -> list[tuple[Agent, Agent]]:
        """
        Select compatible pairs for breeding.
        
        Args:
            population: All agents
            mode: "meritocratic", "arranged", "polygamous"
            config: Genetics configuration
        
        Returns:
            List of (agent_a, agent_b) pairs to form families
        """
        eligible = self.filter_eligible(population, config)
        
        if mode == "meritocratic":
            return self._select_meritocratic(eligible, config)
        elif mode == "arranged":
            return self._select_arranged(eligible, config)
        elif mode == "polygamous":
            return self._select_polygamous(eligible, config)
    
    def filter_eligible(self, population, config):
        """Filter agents who can breed."""
        eligible = []
        for agent in population:
            # Check age
            if agent.bars['age'] < config.min_breeding_age:
                continue
            
            # Check family state
            if agent.family_state not in [FamilyState.NO_FAMILY, FamilyState.DIVORCED]:
                continue
            
            # Check performance (if meritocratic)
            if config.mode == "meritocratic":
                if agent.lifetime_reward < self.performance_threshold:
                    continue
            
            eligible.append(agent)
        
        return eligible
    
    def _select_meritocratic(self, eligible, config):
        """Pair highest performers."""
        # Sort by lifetime reward
        eligible.sort(key=lambda a: -a.lifetime_reward)
        
        pairs = []
        for i in range(0, len(eligible) - 1, 2):
            pairs.append((eligible[i], eligible[i+1]))
        
        return pairs
    
    def _select_arranged(self, eligible, config):
        """Pair maximally diverse agents."""
        pairs = []
        remaining = set(eligible)
        
        while len(remaining) >= 2:
            # Find most diverse pair
            best_pair = None
            max_diversity = -float('inf')
            
            for agent_a in remaining:
                for agent_b in remaining:
                    if agent_a == agent_b:
                        continue
                    
                    diversity = compute_dna_distance(agent_a.dna, agent_b.dna)
                    if diversity > max_diversity:
                        max_diversity = diversity
                        best_pair = (agent_a, agent_b)
            
            pairs.append(best_pair)
            remaining.remove(best_pair[0])
            remaining.remove(best_pair[1])
        
        return pairs
```

```python
# population_controller.py

class PopulationController:
    """Maintains population size and composition."""
    
    def maintain_population_cap(
        self,
        population: list[Agent],
        max_size: int,
        replacement_policy: str,
    ):
        """
        If population exceeds cap, remove worst performers.
        
        Args:
            population: Current population
            max_size: Maximum allowed population
            replacement_policy: "cull_worst", "random", "oldest"
        """
        if len(population) <= max_size:
            return  # under cap, nothing to do
        
        excess = len(population) - max_size
        
        if replacement_policy == "cull_worst":
            # Sort by lifetime reward, remove bottom N
            population.sort(key=lambda a: a.lifetime_reward)
            to_remove = population[:excess]
        
        elif replacement_policy == "oldest":
            # Remove agents closest to natural death
            population.sort(key=lambda a: -a.bars['age'])
            to_remove = population[:excess]
        
        elif replacement_policy == "random":
            import random
            to_remove = random.sample(population, excess)
        
        for agent in to_remove:
            self.remove_agent(agent, population)
    
    def remove_agent(self, agent, population):
        """Remove agent and update family structures."""
        # If agent is in a family, update family state
        if agent.family_id:
            family = self.families[agent.family_id]
            
            if agent in [family.parent_a, family.parent_b]:
                # Parent died
                family.process_deaths()
            elif agent == family.child:
                # Child died
                family.child = None
                family.state = FamilyState.MARRIED_NO_CHILD
        
        # Remove from population
        population.remove(agent)
```

---

### 8.7 Configuration Examples

**Baseline (Meritocratic)**:

```yaml
# configs/population_baseline.yaml
population_genetics:
  enabled: true
  mode: "meritocratic"
  
  population:
    initial_size: 100
    max_size: 100
    replacement_policy: "cull_worst"
  
  family_formation:
    min_age: 0.2
    performance_threshold: 0.6
    pairing_strategy: "highest_performers"
  
  breeding:
    one_child_policy: true
    child_leaves_at_age: 0.3
  
  child_initialization:
    weight_init: "crossover"
    training_mode: "finetune"
    dna_crossover_rate: 0.5
    dna_mutation_rate: 0.05
```

**Dynasty Mode**:

```yaml
# configs/population_dynasty.yaml
population_genetics:
  enabled: true
  mode: "dynasty"
  
  inheritance:
    transfer_wealth: true
    primogeniture: true
    keep_family_id: true
  
  child_initialization:
    weight_init: "clone_best"
    training_mode: "frozen"
    dna_crossover_rate: 0.7
    dna_mutation_rate: 0.02
```

**Arranged Marriage**:

```yaml
# configs/population_arranged.yaml
population_genetics:
  enabled: true
  mode: "arranged"
  
  family_formation:
    performance_threshold: 0.4  # lower bar
    pairing_strategy: "maximize_diversity"
    diversity_metric: "personality_distance"
```

---

**End of Section 8**

---

## SECTION 9: THE BIGGER VISION

### 9.1 The Core Abstraction

Townlet is not about towns. It's about **bars, cascades, and affordances**.

**The minimal viable simulation**:

```yaml
# Any world can be expressed as:

bars:
  - state variables that change over time
  
cascades:
  - how neglecting one bar affects others
  
affordances:
  - actions that change bars
  - with costs, effects, and constraints
```

**Why this is general**:

- **Bars** = any continuous state variable
- **Cascades** = any coupling between variables
- **Affordances** = any controllable action

Towns just happen to be one instantiation:

- Bars = survival meters (energy, health, money)
- Cascades = biological needs (hunger → health decay)
- Affordances = locations with services (Bed, Job, Hospital)

But this works for *any domain* where:

1. State evolves over time
2. Variables are coupled
3. Actions have effects and costs

---

### 9.2 Example Application 1: Economic Simulation

**Domain**: Monetary policy and central banking

#### Bar Configuration

```yaml
# bars_economy.yaml
bars:
  - name: "gdp"
    description: "Gross Domestic Product"
    initial: 1.0
    base_depletion: 0.0  # modulated by policies
    
  - name: "inflation"
    description: "Price level change rate"
    initial: 0.02  # 2% baseline
    base_depletion: 0.0
    
  - name: "unemployment"
    description: "Labor market slack"
    initial: 0.05  # 5% baseline
    base_depletion: 0.0
    
  - name: "debt"
    description: "Government debt as % of GDP"
    initial: 0.60  # 60% debt/GDP ratio
    base_depletion: 0.01  # grows slightly (deficit spending)
    
  - name: "consumer_confidence"
    description: "Economic sentiment"
    initial: 0.70
    base_depletion: -0.002  # slowly improves by default
    
  - name: "asset_prices"
    description: "Stock/housing market levels"
    initial: 1.0
    base_depletion: 0.0
    
  - name: "exchange_rate"
    description: "Currency value (higher = stronger)"
    initial: 1.0
    base_depletion: 0.0
    
  - name: "credit_availability"
    description: "Ease of borrowing"
    initial: 0.70
    base_depletion: 0.0
```

#### Cascade Configuration

```yaml
# cascades_economy.yaml
cascades:
  - name: "high_inflation_hurts_confidence"
    source: "inflation"
    target: "consumer_confidence"
    threshold: 0.04  # >4% inflation
    strength: 0.020  # -2% confidence per tick
    
  - name: "low_confidence_reduces_gdp"
    source: "consumer_confidence"
    target: "gdp"
    threshold: 0.50  # <50% confidence
    strength: 0.010  # -1% GDP per tick
    
  - name: "high_unemployment_lowers_confidence"
    source: "unemployment"
    target: "consumer_confidence"
    threshold: 0.08  # >8% unemployment
    strength: 0.015
    
  - name: "recession_increases_unemployment"
    source: "gdp"
    target: "unemployment"
    threshold: 0.95  # GDP <95% of baseline
    strength: 0.005
    operator: "<"
    
  - name: "high_debt_limits_spending"
    source: "debt"
    target: "gdp"
    threshold: 0.90  # >90% debt/GDP
    strength: 0.008
    
  - name: "loose_credit_inflates_assets"
    source: "credit_availability"
    target: "asset_prices"
    threshold: 0.80  # very loose credit
    strength: 0.010
    
  - name: "asset_bubble_pops"
    source: "asset_prices"
    target: "consumer_confidence"
    threshold: 1.50  # >150% of baseline
    strength: 0.030  # severe confidence shock when bubble pops
    operator: ">"
```

#### Affordance Configuration

```yaml
# affordances_economy.yaml
affordances:
  - id: "raise_interest_rates"
    description: "Federal Reserve increases rates"
    interaction_type: "instant"
    costs: []  # policy actions are "free" (no money cost)
    effects:
      - { meter: "inflation", amount: -0.02 }      # cool inflation
      - { meter: "credit_availability", amount: -0.10 }  # tighten credit
      - { meter: "unemployment", amount: 0.01 }    # slight job loss
      - { meter: "exchange_rate", amount: 0.05 }   # strengthen currency
    cooldown: 3  # can't raise rates every tick
    
  - id: "lower_interest_rates"
    description: "Federal Reserve decreases rates"
    interaction_type: "instant"
    effects:
      - { meter: "inflation", amount: 0.01 }       # risk inflation
      - { meter: "credit_availability", amount: 0.10 }  # loosen credit
      - { meter: "gdp", amount: 0.02 }             # stimulate economy
      - { meter: "unemployment", amount: -0.01 }   # more jobs
      - { meter: "exchange_rate", amount: -0.05 }  # weaken currency
    cooldown: 3
    
  - id: "government_spending"
    description: "Fiscal stimulus"
    interaction_type: "instant"
    costs:
      - { meter: "debt", amount: 0.05 }  # increases debt
    effects:
      - { meter: "gdp", amount: 0.03 }
      - { meter: "unemployment", amount: -0.02 }
      - { meter: "consumer_confidence", amount: 0.02 }
    
  - id: "austerity"
    description: "Reduce government spending"
    interaction_type: "instant"
    effects:
      - { meter: "debt", amount: -0.03 }  # reduce debt
      - { meter: "gdp", amount: -0.02 }   # contractionary
      - { meter: "unemployment", amount: 0.02 }
      - { meter: "consumer_confidence", amount: -0.01 }
    
  - id: "quantitative_easing"
    description: "Central bank buys assets"
    interaction_type: "multi_tick"
    required_ticks: 10  # long-term program
    effects_per_tick:
      - { meter: "asset_prices", amount: 0.03 }
      - { meter: "credit_availability", amount: 0.02 }
      - { meter: "inflation", amount: 0.005 }
    
  - id: "wait"
    description: "Do nothing (let market stabilize)"
    interaction_type: "instant"
    effects: []
```

#### Research Questions

**Can RL learn monetary policy?**

- Train agent on economy environment
- Reward = `r = gdp * (1 - unemployment) * (1 / max(inflation, 0.01))`
- Test: Does agent learn to stabilize cycles?

**Does it rediscover the Taylor rule?**

- [Taylor rule: r = natural_rate + 1.5*(inflation - target) + 0.5*(output_gap)]
- After training, analyze policy: do rate decisions correlate with inflation + unemployment?

**Can it handle shocks?**

- Introduce external shocks (e.g., sudden inflation spike from energy crisis)
- Measure: recovery time, stability

**Human observer test**:

- ✅ "Can a central banker know GDP, inflation, unemployment?" YES (public data)
- ✅ "Can they control interest rates?" YES (policy lever)
- ✅ "Do actions have delayed effects?" YES (cascades model this)

**WEP** that RL can learn basic monetary policy: moderate (~60%). Complex enough to be interesting, simple enough to be tractable.

---

### 9.3 Example Application 2: Ecosystem Simulation

**Domain**: Predator-prey dynamics and trophic cascades

#### Bar Configuration

```yaml
# bars_ecosystem.yaml
bars:
  # Species populations (normalized [0, 1] where 1.0 = carrying capacity)
  - name: "vegetation"
    initial: 0.80
    base_depletion: -0.01  # grows naturally
    
  - name: "herbivore_pop"
    initial: 0.50
    base_depletion: 0.005  # slight natural decline (hunger)
    
  - name: "predator_pop"
    initial: 0.30
    base_depletion: 0.010  # higher natural decline
    
  # Environmental factors
  - name: "water_availability"
    initial: 1.0
    base_depletion: 0.002  # droughts
    
  - name: "soil_quality"
    initial: 0.80
    base_depletion: 0.001  # degradation
    
  # Agent-specific (if modeling individual animals)
  - name: "energy"  # for the agent (e.g., a predator)
    initial: 0.70
    base_depletion: 0.010
    
  - name: "health"
    initial: 1.0
    base_depletion: 0.0
```

#### Cascade Configuration

```yaml
# cascades_ecosystem.yaml
cascades:
  - name: "herbivores_eat_vegetation"
    source: "herbivore_pop"
    target: "vegetation"
    threshold: 0.40  # if herbivores > 40% of capacity
    strength: 0.020  # they consume vegetation
    
  - name: "overgrazing_damages_soil"
    source: "vegetation"
    target: "soil_quality"
    threshold: 0.30  # if vegetation < 30%
    strength: 0.005
    operator: "<"
    
  - name: "poor_soil_limits_vegetation"
    source: "soil_quality"
    target: "vegetation"
    threshold: 0.50
    strength: 0.010
    operator: "<"
    
  - name: "predators_eat_herbivores"
    source: "predator_pop"
    target: "herbivore_pop"
    threshold: 0.25
    strength: 0.015
    
  - name: "low_herbivores_starve_predators"
    source: "herbivore_pop"
    target: "predator_pop"
    threshold: 0.20
    strength: 0.020
    operator: "<"
    
  - name: "drought_kills_vegetation"
    source: "water_availability"
    target: "vegetation"
    threshold: 0.50
    strength: 0.015
    operator: "<"
```

#### Affordance Configuration

```yaml
# affordances_ecosystem.yaml
affordances:
  # For individual predator agent
  - id: "hunt"
    description: "Hunt herbivores"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs:
      - { meter: "energy", amount: 0.05 }  # per tick
    effects:
      - { meter: "energy", amount: 0.30 }  # gain from kill (final tick)
      - { meter: "herbivore_pop", amount: -0.02 }  # reduce prey population
    success_probability: 0.60  # hunts can fail
    
  - id: "graze"  # for herbivore agent
    description: "Eat vegetation"
    interaction_type: "instant"
    costs:
      - { meter: "vegetation", amount: -0.03 }
    effects:
      - { meter: "energy", amount: 0.20 }
    
  - id: "rest"
    description: "Rest and recover"
    interaction_type: "multi_tick"
    required_ticks: 2
    effects_per_tick:
      - { meter: "energy", amount: 0.10 }
      - { meter: "health", amount: 0.05 }
    
  - id: "migrate"
    description: "Move to better territory"
    interaction_type: "instant"
    costs:
      - { meter: "energy", amount: 0.15 }
    effects:
      - { meter: "water_availability", amount: 0.10 }  # find better water
      - { meter: "vegetation", amount: 0.05 }  # find better grazing
```

#### Research Questions

**Do predator-prey cycles emerge?**

- Train agents in ecosystem environment
- Reward = `r = energy * health` (individual survival)
- Measure: population oscillations (Lotka-Volterra dynamics)

**Does overgrazing lead to collapse?**

- If herbivores overgraze (vegetation < 0.2), does ecosystem crash?
- Can predators "learn" to regulate herbivore population?

**Group selection**:

- If some predators "restrain" from overhunting, do their populations persist longer?
- Does altruism emerge?

**Human observer test**:

- ✅ "Can a predator know its own energy?" YES (internal state)
- ✅ "Can it see prey nearby?" YES (partial observability)
- ✅ "Does hunting affect global prey population?" YES (affordance effects)

**WEP** that Lotka-Volterra cycles emerge: moderate (~55%). Requires careful tuning of cascade strengths.

---

### 9.4 Example Application 3: Mental Health Treatment

**Domain**: Modeling anxiety/depression treatment strategies

**Disclaimer**: This is an illustrative example for research purposes, not a clinical tool.

#### Bar Configuration

```yaml
# bars_mentalhealth.yaml
bars:
  - name: "anxiety"
    initial: 0.60
    base_depletion: 0.005  # increases over time
    
  - name: "depression"
    initial: 0.50
    base_depletion: 0.003
    
  - name: "sleep_quality"
    initial: 0.60
    base_depletion: -0.002  # improves slightly naturally
    
  - name: "medication_level"
    initial: 0.0
    base_depletion: 0.01  # medication wears off
    
  - name: "therapy_progress"
    initial: 0.0
    base_depletion: 0.0  # cumulative, doesn't decay
    
  - name: "social_support"
    initial: 0.50
    base_depletion: 0.002  # relationships require maintenance
    
  - name: "physical_health"
    initial: 0.80
    base_depletion: 0.001
    
  - name: "daily_functioning"
    initial: 0.70
    base_depletion: 0.0  # modulated by other bars
```

#### Cascade Configuration

```yaml
# cascades_mentalhealth.yaml
cascades:
  - name: "anxiety_disrupts_sleep"
    source: "anxiety"
    target: "sleep_quality"
    threshold: 0.60
    strength: 0.015
    
  - name: "poor_sleep_worsens_depression"
    source: "sleep_quality"
    target: "depression"
    threshold: 0.40
    strength: 0.010
    operator: "<"
    
  - name: "depression_reduces_social"
    source: "depression"
    target: "social_support"
    threshold: 0.50
    strength: 0.012
    
  - name: "low_social_worsens_depression"
    source: "social_support"
    target: "depression"
    threshold: 0.30
    strength: 0.015
    operator: "<"
    
  - name: "anxiety_impairs_function"
    source: "anxiety"
    target: "daily_functioning"
    threshold: 0.70
    strength: 0.020
    
  - name: "exercise_improves_mood"
    source: "physical_health"
    target: "depression"
    threshold: 0.70
    strength: -0.008  # negative = reduces depression
    operator: ">"
```

#### Affordance Configuration

```yaml
# affordances_mentalhealth.yaml
affordances:
  - id: "therapy_session"
    description: "Cognitive behavioral therapy"
    interaction_type: "multi_tick"
    required_ticks: 4  # 1-hour session
    costs:
      - { meter: "daily_functioning", amount: 0.05 }  # time commitment
    effects:
      - { meter: "therapy_progress", amount: 0.10 }
      - { meter: "anxiety", amount: -0.05 }
      - { meter: "depression", amount: -0.03 }
    cooldown: 24  # weekly sessions
    
  - id: "medication"
    description: "Take prescribed medication"
    interaction_type: "instant"
    effects:
      - { meter: "medication_level", amount: 0.30 }
      - { meter: "anxiety", amount: -0.10 }  # fast relief
    side_effects:
      - { meter: "sleep_quality", amount: -0.05 }  # some meds affect sleep
    
  - id: "exercise"
    description: "Physical activity"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs:
      - { meter: "daily_functioning", amount: 0.08 }
    effects:
      - { meter: "physical_health", amount: 0.10 }
      - { meter: "depression", amount: -0.05 }
      - { meter: "sleep_quality", amount: 0.05 }
    
  - id: "social_activity"
    description: "Spend time with friends"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs:
      - { meter: "anxiety", amount: 0.03 }  # social anxiety
    effects:
      - { meter: "social_support", amount: 0.15 }
      - { meter: "depression", amount: -0.08 }
    
  - id: "rest"
    description: "Rest and recover"
    interaction_type: "instant"
    effects:
      - { meter: "daily_functioning", amount: 0.05 }
      - { meter: "sleep_quality", amount: 0.03 }
```

#### Research Questions

**Can RL discover treatment strategies?**

- Reward = `r = daily_functioning * (1 - anxiety) * (1 - depression)`
- Does agent learn: therapy + medication + exercise > medication alone?

**Optimal treatment sequencing**:

- Does agent learn to stabilize acute symptoms (medication) before building long-term strategies (therapy)?

**Side effect management**:

- Can agent balance benefit (anxiety reduction) vs cost (sleep disruption from meds)?

**Relapse prevention**:

- After recovery (bars high), does agent continue maintenance (exercise, social)?

**Human observer test**:

- ✅ "Can someone know their anxiety/depression level?" YES (self-awareness)
- ✅ "Can they choose therapy or medication?" YES (treatment decisions)
- ✅ "Do symptoms interact?" YES (cascades model this)

**WEP** that RL can model treatment strategies: moderate (~50%). Controversial topic, requires careful validation.

---

### 9.5 Example Application 4: Supply Chain Management

**Domain**: Inventory control and logistics optimization

#### Bar Configuration

```yaml
# bars_supply_chain.yaml
bars:
  - name: "inventory_level"
    initial: 0.60
    base_depletion: 0.02  # daily sales
    
  - name: "customer_satisfaction"
    initial: 0.80
    base_depletion: 0.0
    
  - name: "cash_reserves"
    initial: 1.0
    base_depletion: 0.01  # operating costs
    
  - name: "supplier_reliability"
    initial: 0.70
    base_depletion: 0.001
    
  - name: "shipping_capacity"
    initial: 0.80
    base_depletion: 0.0
    
  - name: "warehouse_capacity"
    initial: 0.90
    base_depletion: 0.0
    
  - name: "demand_forecast"
    initial: 0.50  # medium demand
    base_depletion: 0.0  # fluctuates
```

#### Cascades

```yaml
cascades:
  - name: "stockout_hurts_satisfaction"
    source: "inventory_level"
    target: "customer_satisfaction"
    threshold: 0.20  # low stock
    strength: 0.030
    operator: "<"
    
  - name: "low_satisfaction_reduces_demand"
    source: "customer_satisfaction"
    target: "demand_forecast"
    threshold: 0.50
    strength: 0.010
    operator: "<"
    
  - name: "overstock_ties_up_cash"
    source: "inventory_level"
    target: "cash_reserves"
    threshold: 0.80  # excess inventory
    strength: 0.015
    operator: ">"
```

#### Affordances

```yaml
affordances:
  - id: "order_inventory"
    description: "Order from supplier"
    costs:
      - { meter: "cash_reserves", amount: 0.30 }
    effects:
      - { meter: "inventory_level", amount: 0.40 }  # after lead time
    lead_time: 5  # 5 ticks delay
    
  - id: "expedited_shipping"
    costs:
      - { meter: "cash_reserves", amount: 0.50 }  # expensive
    effects:
      - { meter: "inventory_level", amount: 0.40 }
    lead_time: 1  # fast delivery
    
  - id: "promote_sale"
    costs:
      - { meter: "cash_reserves", amount: 0.10 }
    effects:
      - { meter: "inventory_level", amount: -0.20 }  # move inventory
      - { meter: "customer_satisfaction", amount: 0.10 }
```

**Research questions**: Can RL learn (s,S) inventory policies? Does it discover just-in-time strategies?

---

### 9.6 Platform Value Proposition by Audience

#### For Researchers

**Value**: Hypothesis testing without Python

**Workflow**:

1. Formulate hypothesis: "Does X affect Y?"
2. Design world: encode X and Y as bars, add cascade
3. Launch experiment: `townlet train --config hypothesis_X/`
4. Analyze results: compare treatment vs control

**Example**:

- Hypothesis: "High initial wealth reduces work ethic"
- World A: `initial_money: 0.20`
- World B: `initial_money: 0.80`
- Measure: Hours worked over lifetime
- Publish: Config diff + results + provenance

**Why this matters**: Configuration-driven experiments are:

- Reproducible (exact config archived)
- Auditable (cognitive hash proves which brain)
- Comparable (structured diffs across experiments)

#### For Educators

**Value**: Students experiment without coding

**Pedagogy**:

1. Provide baseline world config
2. Students modify parameters (e.g., "make food twice as expensive")
3. Run experiment, observe behavioral changes
4. Write report explaining why behavior changed

**Example assignment**:

```
Assignment: The Scarcity Experiment

1. Run baseline world (50 episodes)
2. Create "scarcity world": double Fridge cost
3. Compare:
   - Average lifetime money at retirement
   - Frequency of starvation deaths
   - Time spent at Job vs Recreation
4. Explain: Why did agent behavior change?
```

**Why this matters**: Students learn:

- Systems thinking (cascades create feedback loops)
- Policy design (affordance costs affect behavior)
- Emergent phenomena (policies not explicitly programmed)

#### For Policy Analysts

**Value**: Auditable simulations for governance

**Use case**: Testing policy interventions

**Workflow**:

1. Model current system (e.g., welfare policy)
2. Propose intervention (e.g., universal basic income)
3. Encode as affordance change (e.g., passive income bar)
4. Run simulation, measure outcomes
5. Present to stakeholders with provenance

**Example**:

```yaml
# Current welfare system
affordances:
  - id: "welfare_office"
    eligibility: { money < 0.10 }  # means-tested
    effects: { money: +0.05 }

# Proposed UBI
affordances:
  - id: "ubi_payment"
    eligibility: { always }  # universal
    effects: { money: +0.03 }
    operating_hours: [0, 24]  # automatic
```

**Compare**:

- Work incentives (does UBI reduce labor?)
- Poverty rates (does UBI prevent starvation?)
- Cost (total money distributed)

**Why this matters**: Policy debates become empirical

- Not "I think UBI will work"
- But "When we ran the simulation, we observed X"
- With provenance: "Config hash ABC123, cognitive hash DEF456"

---

### 9.7 What Makes This Generalizable

**Three properties enable cross-domain transfer**:

#### 1. Declarative Physics

```yaml
# Physics are data, not code
# This means:
- Non-programmers can design worlds
- Worlds are diff-able (structured comparison)
- Worlds are version-controlled (Git tracks changes)
- Worlds are reproducible (config snapshot frozen)
```

**Alternative (bad)**: Physics embedded in environment.step()

- Requires Python expertise
- Difficult to compare across experiments
- Hard to audit ("what exactly changed?")
- Brittle to refactoring

#### 2. Minimal Assumptions

**Townlet only assumes**:

- State evolves over time (bars with depletion)
- Variables couple (cascades)
- Actions have effects (affordances)

**Townlet does NOT assume**:

- Grid worlds (could be continuous space)
- Agents (could model single controller)
- Survival (could maximize profit, not survival)
- Spatial navigation (could be pure resource management)

This means you can model:

- Economic systems (no grid, pure resource flows)
- Ecosystems (spatial but not grid-based)
- Mental health (no space, just internal dynamics)
- Supply chains (network topology, not grid)

#### 3. Human Observer Principle

The same observability constraints that make Townlet realistic for agents make it realistic for other domains.

**Economic policy**:

- ✅ Can a central banker observe GDP, inflation? YES
- ❌ Can they know exact future GDP? NO
- ✅ Can they control interest rates? YES

**Ecosystem management**:

- ✅ Can a ranger observe population counts? YES
- ❌ Can they know exact future weather? NO
- ✅ Can they introduce species? YES

**Mental health treatment**:

- ✅ Can a patient know their anxiety level? YES
- ❌ Can they instantly cure depression? NO
- ✅ Can they choose therapy? YES

The human observer test generalizes across domains.

---

### 9.8 Limitations & Non-Applications

**Townlet is NOT suitable for**:

**1. Fast-reaction games**

- Fighting games, racing, real-time shooters
- Why: Townlet is turn-based, focuses on long-horizon planning
- Use instead: Game-specific simulators (e.g., OpenAI Gym Atari)

**2. Perfect-information games**

- Chess, Go
- Why: Partial observability and cascades add no value here
- Use instead: AlphaZero-style tree search

**3. Pure text domains**

- Language modeling, dialogue
- Why: Bars/cascades don't map to linguistic structure
- Use instead: Transformer-based LLMs

**4. Continuous control**

- Robotics, drones, manipulation
- Why: Discrete affordances don't capture continuous dynamics well
- Use instead: MuJoCo, PyBullet, Isaac Gym

**5. Purely spatial problems**

- Pathfinding, maze solving
- Why: Townlet's value is in temporal reasoning and resource allocation
- Use instead: Grid-world simulators without survival mechanics

**The sweet spot**:

- Long-horizon planning (100+ steps)
- Multi-objective optimization (balance multiple meters)
- Resource allocation under scarcity
- Coupled dynamics (cascades matter)
- Partial observability (memory required)
- Social coordination (multi-agent with communication)

---

### 9.9 Future Extensions

**What could be added to make Townlet more general**:

#### Extension 1: Continuous Space

```yaml
# Instead of grid_size: 8
space:
  type: "continuous"
  bounds: [[0, 10], [0, 10]]  # 2D continuous
  
affordances:
  - id: "job"
    position: [3.5, 7.2]  # continuous coordinates
    interaction_radius: 0.5  # fuzzy boundary
```

**Use case**: Model ecosystems with actual spatial movement, not grid-constrained

#### Extension 2: Relational Networks

```yaml
# Instead of affordances at positions
entities:
  - id: "company_a"
    relations:
      - { target: "company_b", type: "supplier" }
      - { target: "company_c", type: "competitor" }
```

**Use case**: Supply chains, social networks, financial systems

#### Extension 3: Stochastic Events

```yaml
# Random shocks to bars
events:
  - id: "recession"
    probability: 0.02  # 2% per 100 ticks
    effects:
      - { meter: "gdp", amount: -0.20 }
      - { meter: "unemployment", amount: 0.05 }
```

**Use case**: Test robustness to black swan events

#### Extension 4: Partial Affordance Observability

```yaml
affordances:
  - id: "hidden_treasure"
    visible: false  # agent must discover via exploration
    discovery_condition: { position_within: 1.0 }
```

**Use case**: Exploration-driven domains (R&D, scientific discovery)

---

### 9.10 The Long-Term Vision

**Townlet becomes a platform for configuration-driven science**:

**Year 1**: Towns (survival simulation)

- L0-L8 curriculum proven
- Population genetics validated
- Emergent communication demonstrated

**Year 2**: Economics (monetary policy)

- RL learns basic Taylor rule
- Validated against macro models
- Published in econ/CS conferences

**Year 3**: Ecosystems (conservation)

- Predator-prey dynamics emerge
- Used by ecology researchers
- Published in Nature/Science

**Year 5**: General adoption

- 100+ research papers using Townlet
- Standard benchmark (like Atari, MuJoCo)
- Taught in RL courses
- Used by policymakers for scenario planning

**The ultimate success criterion**:
> "Can a domain expert with zero Python knowledge create a Townlet world and run meaningful experiments?"

If yes, we've built a truly general platform.

**WEP** that Townlet becomes widely adopted: moderate (~50%). Depends on documentation quality, community support, and killer applications.

---

## SECTION 10: ACTIONABLE NEXT STEPS

### 10.1 The Critical Path: First 30 Days

**Goal**: Fix blockers, complete specifications, achieve L0-L5 stability

**Assumption**: One full-time developer, already familiar with codebase

---

#### Week 1: Fix Critical Blockers

**Objective**: Address all three blockers from Section 6

**Monday-Tuesday: EthicsFilter Refactor**

```python
# Tasks:
[ ] Remove EthicsFilter from modules dict in agent_architecture.yaml
[ ] Implement EthicsFilter as pure Python class (no nn.Module)
[ ] Update execution_graph.yaml (ethics is controller, not module)
[ ] Remove EthicsFilter from weights.pt saving/loading
[ ] Write test: assert not hasattr(ethics_filter, 'parameters')
[ ] Update Brain as Code doc (Section 6.4)
[ ] Update Checkpoint doc (Section 4.1)

# Validation:
python -m pytest tests/test_ethics_filter.py
# Expected: EthicsFilter has no learnable parameters
# Expected: Vetoes are deterministic (same input → same output)
```

**Estimated time**: 2 days  
**Risk**: Low (straightforward refactor)

---

**Wednesday-Thursday: Checkpoint Signatures**

```python
# Tasks:
[ ] Implement SecureCheckpointWriter class (from Section 6.3)
[ ] Generate signing key (~/.townlet/signing_key.bin)
[ ] Update checkpoint writer to compute manifest + signature
[ ] Update checkpoint loader to verify before loading
[ ] Write test: tamper with file, verify detection
[ ] Add CheckpointTamperedError exception
[ ] Update Checkpoint doc (Section 4.8)

# Validation:
python -m pytest tests/test_checkpoint_security.py
# Expected: Signature verification passes on valid checkpoint
# Expected: Tampering raises CheckpointTamperedError
```

**Estimated time**: 2 days  
**Risk**: Low (standard cryptographic pattern)

---

**Friday: World Config Hash**

```python
# Tasks:
[ ] Implement compute_world_config_hash() function
[ ] Add world_config_hash to observation tensor (Section 4)
[ ] Update ObservationBuilder to include hash
[ ] Update Module B to accept world_config_hash as input
[ ] Write test: different configs → different hashes
[ ] Update High-Level Design doc (Section 13.2)

# Validation:
python -m pytest tests/test_world_config_hash.py
# Expected: Hash changes when affordance costs change
# Expected: Hash stable across runs with same config
```

**Estimated time**: 1 day  
**Risk**: Low (hash computation is straightforward)

---

**Week 1 Deliverable**: All three blockers fixed, tests passing

**Checkpoint**: Can you answer these questions?

- ✅ "Is EthicsFilter learned?" → NO (deterministic rules)
- ✅ "Can checkpoints be tampered with?" → NO (signatures detect tampering)
- ✅ "Can world model adapt to curriculum changes?" → YES (hash conditioning)

---

#### Week 2: Complete Missing Specifications

**Objective**: Document and implement all underspecified behaviors from Section 7

**Monday-Tuesday: Multi-Agent Contention**

```python
# Tasks:
[ ] Implement resolve_contention() (distance-first, then agent_id)
[ ] Add action_failed flag to observation
[ ] Write contention resolution tests
[ ] Document in docs/multi_agent_mechanics.md
[ ] Update affordances.yaml schema to include capacity field
[ ] Add telemetry logging for contention events

# Validation:
python -m pytest tests/test_contention.py
# Expected: Same positions → deterministic winner
# Expected: Closer agent wins
# Expected: Loser receives action_failed=True
```

**Estimated time**: 2 days  
**Risk**: Low (clear specification)

---

**Wednesday-Thursday: Family Lifecycle**

```python
# Tasks:
[ ] Implement Family class with full state machine
[ ] Implement all state transitions (from Section 7.2)
[ ] Add family_state to Agent class
[ ] Write transition tests (all edge cases)
[ ] Document in docs/family_lifecycle.md
[ ] Create population_genetics.yaml schema

# Validation:
python -m pytest tests/test_family_lifecycle.py
# Expected: All transitions work correctly
# Expected: Child maturity triggers parent eligibility
# Expected: Death updates family state correctly
```

**Estimated time**: 2 days  
**Risk**: Medium (complex state machine, many edge cases)

---

**Friday: Child Initialization**

```python
# Tasks:
[ ] Implement initialize_child() with all modes
[ ] Implement crossover_dna() and mutate_dna()
[ ] Implement average_weights() for crossover mode
[ ] Write child initialization tests
[ ] Document in docs/population_genetics.md
[ ] Add genetics config to population_genetics.yaml

# Validation:
python -m pytest tests/test_child_initialization.py
# Expected: Crossover produces intermediate DNA
# Expected: Mutation perturbs genes slightly
# Expected: Weight inheritance works for all modes
```

**Estimated time**: 1 day  
**Risk**: Low (clear algorithms)

---

**Week 2 Deliverable**: All missing specs documented and implemented

**Checkpoint**: Can you answer these questions?

- ✅ "What happens when 2 agents try to use Job?" → Closest wins, loser gets action_failed
- ✅ "What happens when a parent dies?" → Family transitions to SINGLE_PARENT
- ✅ "How does a child inherit weights?" → Configurable (crossover/clone/pretrained/random)

---

#### Week 3: Social Observability (L6-7)

**Objective**: Implement observation space for multi-agent levels

**Monday-Tuesday: Cue Engine**

```python
# Tasks:
[ ] Implement CueEngine class
[ ] Load cues from cues.yaml
[ ] Implement compute_cues() (evaluate triggers, emit top-k)
[ ] Write cue computation tests
[ ] Create baseline cue pack (12 cues from Section 5.4)
[ ] Validate cues.yaml schema

# Validation:
python -m pytest tests/test_cue_engine.py
# Expected: Triggers evaluate correctly
# Expected: Max 3 cues emitted per agent
# Expected: Priority ordering works
```

**Estimated time**: 2 days  
**Risk**: Low (clear specification in Section 5)

---

**Wednesday-Thursday: Social Observation Builder**

```python
# Tasks:
[ ] Add _get_visible_agent_positions() to ObservationBuilder
[ ] Add _get_visible_agent_cues() to ObservationBuilder
[ ] Update build_observations() for curriculum_level >= 6
[ ] Write observation tests (L6-7)
[ ] Verify observation dimensions match Section 4.5

# Implementation (from Section 4.8):
def _get_visible_agent_positions(self, observer_positions, all_positions):
    # Return [num_agents, max_visible * 2]
    # Relative positions, sorted by distance, padded

def _get_visible_agent_cues(self, observer_positions, all_positions, all_cues):
    # Return [num_agents, max_visible * num_cue_types]
    # Binary matrix of active cues

# Validation:
python -m pytest tests/test_social_observations.py
# Expected: Visible agents sorted by distance
# Expected: Cues encoded as binary vectors
# Expected: Padding works correctly
```

**Estimated time**: 2 days  
**Risk**: Medium (careful indexing, padding edge cases)

---

**Friday: Module C Integration**

```python
# Tasks:
[ ] Update Module C to accept cue inputs
[ ] Implement cue embedding layer
[ ] Test Module C forward pass with cues
[ ] Verify Module C can predict state from cues (placeholder training)

# Validation:
# Manual test: feed cues ['looks_tired', 'at_job']
# Expected: Module C outputs prediction (even if untrained)
```

**Estimated time**: 1 day  
**Risk**: Low (architecture already defined, just connecting)

---

**Week 3 Deliverable**: L6-7 observation space working

**Checkpoint**: Can you run a multi-agent episode with social observations?

```bash
python scripts/run_episode.py --level L6 --num_agents 5
# Expected: Episode completes, observations include other_agents_in_window
# Expected: Cues appear in telemetry logs
```

---

#### Week 4: Communication Channel (L8)

**Objective**: Implement family communication system

**Monday-Tuesday: Family Communication Infrastructure**

```python
# Tasks:
[ ] Add SET_COMM_CHANNEL action to action space
[ ] Add current_signal field to Agent class
[ ] Add family_id and family_members fields
[ ] Process SET_COMM_CHANNEL in environment step
[ ] Write action processing tests

# Validation:
python -m pytest tests/test_communication_action.py
# Expected: SET_COMM_CHANNEL updates agent.current_signal
# Expected: Signal persists across ticks until changed
```

**Estimated time**: 2 days  
**Risk**: Low (straightforward state management)

---

**Wednesday-Thursday: Family Communication Observations**

```python
# Tasks:
[ ] Add _get_family_comm_channel() to ObservationBuilder
[ ] Update build_observations() for curriculum_level >= 8
[ ] Normalize signals [0, 999] → [0, 1]
[ ] Handle agents not in families (all zeros)
[ ] Write observation tests (L8)

# Implementation (from Section 4.8):
def _get_family_comm_channel(self, agent_ids, family_data):
    # Return [num_agents, max_family_size]
    # Normalized signals from family members

# Validation:
python -m pytest tests/test_family_observations.py
# Expected: Family members' signals appear in observation
# Expected: Non-family agents receive zeros
```

**Estimated time**: 2 days  
**Risk**: Low (similar to social observations)

---

**Friday: End-to-End L8 Test**

```python
# Tasks:
[ ] Create test scenario: 2-agent family
[ ] Parent sets signal, verify child receives it
[ ] Track signal over 100 ticks
[ ] Verify observation dimensions match Section 4.6

# Validation:
python scripts/run_episode.py --level L8 --num_agents 3 --families 1
# Expected: Episode completes
# Expected: Family channel in observations
# Expected: SET_COMM_CHANNEL actions in telemetry
```

**Estimated time**: 1 day  
**Risk**: Low (integration test)

---

**Week 4 Deliverable**: L8 communication working end-to-end

**Checkpoint**: Can you run a family episode with communication?

```bash
python scripts/run_family_episode.py
# Expected: Parents and child can exchange signals
# Expected: Signals visible in telemetry
# Expected: No semantic meaning (agents must learn)
```

---

### 10.2 Months 2-3: Training & Validation

**Goal**: Train agents L0-L8, validate curriculum progression, run pilot experiments

---

#### Month 2, Week 1: L0-L3 Training

```python
# Tasks:
[ ] Train baseline agents on L0-L3
[ ] Verify learning curves (from Section 3.2)
[ ] Measure graduation criteria:
    - L0: Learns "Bed fixes energy" by episode 20
    - L1: Establishes Job-Bed loop by episode 100
    - L2: Reaches retirement by episode 200
    - L3: Terminal score > 0.65 by episode 500
[ ] Document hyperparameters (learning rate, batch size, etc)
[ ] Save trained checkpoints

# Validation:
- L0: 80%+ agents use Bed when energy < 0.3
- L1: Stable income from Job by episode 100
- L2: 70%+ retirement rate
- L3: 80%+ retirement rate, mean score > 0.65
```

**Estimated time**: 1 week (parallel training on GPU)  
**Risk**: Medium (may need hyperparameter tuning)

---

#### Month 2, Week 2: L4-L5 Training (LSTM Required)

```python
# Tasks:
[ ] Implement RecurrentSpatialQNetwork architecture
[ ] Train on L4-L5 with partial observability
[ ] Verify RND-driven exploration works
[ ] Measure spatial memory performance
[ ] Compare to L3 baseline (should be similar after training)

# Validation:
- L4: Agents build mental maps by episode 2000
- L4: Can navigate to unseen affordances from memory
- L5: Respects operating hours (doesn't spam Job at 3am)
- L5: Uses WAIT strategically (arrive early, wait for open)
```

**Estimated time**: 1 week  
**Risk**: High (LSTM training is finicky, may need architecture tuning)

---

#### Month 2, Week 3: Module C Pretraining (CTDE)

```python
# Tasks:
[ ] Collect logged episodes from L4-5 training
[ ] Extract (cues, ground_truth_state) pairs
[ ] Train Module C via supervised learning
[ ] Evaluate prediction accuracy on held-out set
[ ] Save pretrained Module C checkpoint

# Validation:
- Module C predicts energy with MAE < 0.15
- Module C predicts health with MAE < 0.15
- Module C goal prediction accuracy > 60%
- Cue embeddings are learned (not random)
```

**Estimated time**: 1 week  
**Risk**: Medium (depends on data quality)

---

#### Month 2, Week 4: L6-L7 Training (Social Reasoning)

```python
# Tasks:
[ ] Load pretrained Module C
[ ] Train full Module A-D stack on L6
[ ] Measure contention resolution behavior
[ ] Train on L7 with rich cues
[ ] Compare L6 vs L7 performance

# Validation:
- L6: Agents choose alternate affordances when contention detected
- L6: Strategic resource selection > 60% success rate
- L7: Better predictions than L6 (more cues → better inference)
- L7: Retirement rate matches L5 (no performance degradation)
```

**Estimated time**: 1 week  
**Risk**: High (social reasoning is complex, may need architecture changes)

---

#### Month 3, Week 1-2: L8 Training (Emergent Communication)

```python
# Tasks:
[ ] Train families on L8 with communication channel
[ ] Track signal usage over 20k episodes
[ ] Measure coordination metrics:
    - Signal diversity
    - Signal stability
    - Coordination gain (family vs solo)
[ ] Analyze emergent protocols
[ ] Document learned signal meanings (post-hoc)

# Validation:
- Families use 3-10 unique signals by episode 20k
- Signal stability > 0.5 (same signal → same context 50%+ of time)
- Family coordination gain > 10% over solo agents
```

**Estimated time**: 2 weeks (long training, extensive analysis)  
**Risk**: Very High (emergent communication may not emerge reliably)

---

#### Month 3, Week 3: Curriculum Validation

```python
# Tasks:
[ ] Train agents from scratch through full L0-L8 curriculum
[ ] Measure transfer learning (does L7 help L8?)
[ ] Compare to agents trained only on L8 (no curriculum)
[ ] Document curriculum benefits
[ ] Write curriculum_results.md

# Validation:
- Curriculum agents outperform scratch agents on L8
- Each level provides measurable skill (ablation study)
- Training time: curriculum < 2× scratch training
```

**Estimated time**: 1 week  
**Risk**: Medium (may not see curriculum benefit)

---

#### Month 3, Week 4: Pilot Experiments

```python
# Tasks:
[ ] Run 3 pilot experiments from Section 8.4:
    1. Q1.1: Does natural selection work? (meritocratic mode)
    2. Q2.1: Emergent communication protocols (L8 families)
    3. Q3.1: Wealth concentration (dynasty mode)
[ ] Collect data, create visualizations
[ ] Write experiment_results.md
[ ] Prepare for publication/demo

# Validation:
- Each experiment has clear hypothesis + result
- Visualizations are publication-ready
- Code + configs are reproducible
```

**Estimated time**: 1 week  
**Risk**: Low (exploratory, no specific target)

---

### 10.3 Month 4+: Documentation, Release, Research

**Goal**: Prepare for public release, enable external researchers

---

#### Month 4, Week 1-2: Documentation

```python
# Tasks:
[ ] Write QUICKSTART.md (5-minute tutorial)
[ ] Write TUTORIAL.md (30-minute walkthrough)
[ ] Write API_REFERENCE.md (complete spec)
[ ] Write CONTRIBUTING.md (for contributors)
[ ] Polish all existing docs (Sections 0-9 from this review)
[ ] Create video tutorials (optional)

# Validation:
- New user can run first experiment in 10 minutes
- All public APIs are documented
- Examples cover L0-L8 + all genetics modes
```

**Estimated time**: 2 weeks  
**Risk**: Low (tedious but straightforward)

---

#### Month 4, Week 3: Schema Validation & Tooling

```python
# Tasks:
[ ] Write JSON schemas for all YAML configs
[ ] Implement config validator (pre-flight checks)
[ ] Write schema tests (reject invalid configs)
[ ] Create config templates for common scenarios
[ ] Implement townlet validate command

# Example:
townlet validate --config configs/my_world/
# Expected: Reports errors in YAML before launching

# Validation:
- All invalid configs are caught by validator
- Error messages are helpful ("missing required field: cascade.target")
- Templates cover 80% of use cases
```

**Estimated time**: 1 week  
**Risk**: Low

---

#### Month 4, Week 4: Testing & CI

```python
# Tasks:
[ ] Achieve 80%+ code coverage
[ ] Set up GitHub Actions (pytest, linting, type checking)
[ ] Write integration tests (L0-L8 smoke tests)
[ ] Write regression tests (prevent performance degradation)
[ ] Document testing strategy in tests/README.md

# Validation:
- pytest tests/ passes on main branch
- CI runs on every PR
- No critical bugs in issue tracker
```

**Estimated time**: 1 week  
**Risk**: Medium (achieving good coverage is time-consuming)

---

#### Month 5+: Public Release

```python
# Tasks:
[ ] Create GitHub repo (github.com/tachyon-beep/townlet)
[ ] Write README.md (from Section 11 of this doc)
[ ] Tag v1.0.0 release
[ ] Announce on Twitter, Reddit, HN
[ ] Submit to RL/AI conferences (NeurIPS, ICLR)
[ ] Write blog post (architecture deep dive)
[ ] Create Discord/Slack community (optional)

# Success criteria:
- 100+ GitHub stars in first month
- 5+ external contributors
- 1+ research paper using Townlet (not by you)
- Featured in RL newsletter/podcast
```

**Estimated time**: Ongoing  
**Risk**: High (adoption is unpredictable)

---

### 10.4 Validation Criteria & Success Metrics

**How do you know when each stage is "done"?**

#### Week 1 Success: Blockers Fixed

```python
✅ EthicsFilter has no learnable parameters
✅ Checkpoint tampering raises CheckpointTamperedError
✅ World config changes are observable
✅ All tests pass: pytest tests/test_blockers.py
```

#### Week 2 Success: Specifications Complete

```python
✅ Contention resolution is deterministic
✅ All family state transitions work
✅ Child initialization supports 4 modes
✅ All tests pass: pytest tests/test_specifications.py
```

#### Week 3 Success: L6-7 Working

```python
✅ Cue engine emits correct cues
✅ Social observations include positions + cues
✅ Module C accepts cue inputs
✅ Multi-agent episode completes without errors
```

#### Week 4 Success: L8 Working

```python
✅ SET_COMM_CHANNEL action works
✅ Family channel in observations
✅ Family episode runs end-to-end
✅ Telemetry logs show signals
```

#### Month 2 Success: L0-L5 Trained

```python
✅ L0-L3: Agents reach retirement reliably
✅ L4-L5: LSTM agents navigate from memory
✅ Learning curves match expectations
✅ Saved checkpoints load correctly
```

#### Month 3 Success: L6-L8 Trained

```python
✅ Module C predicts state from cues
✅ L6-L7: Social reasoning works
✅ L8: Signals are used (even if protocol is unclear)
✅ Pilot experiments produce results
```

#### Month 4 Success: Release-Ready

```python
✅ Documentation is complete
✅ New user can run experiment in 10 minutes
✅ Tests pass, coverage > 80%
✅ No critical bugs
```

---

### 10.5 Resource Requirements

**What do you need to execute this plan?**

#### Human Resources

**Minimum (solo developer)**:

- 1 full-time developer (4 months)
- Familiar with: PyTorch, RL, YAML, testing
- Can learn: Townlet codebase (1 week ramp-up)

**Optimal (small team)**:

- 1 lead engineer (architecture, curriculum L6-L8)
- 1 infrastructure engineer (testing, CI, docs)
- 1 researcher (experiments, analysis, papers)
- 4 months total

**Consulting support**:

- 1 RL expert (review architecture, advise on training)
- 5 hours/month × 4 months = 20 hours

#### Compute Resources

**Week 1-4 (implementation)**:

- Local dev machine (CPU only)
- Total cost: $0

**Month 2-3 (training)**:

- GPU instance (e.g., AWS p3.2xlarge with V100)
- Training time:
  - L0-L3: 24 hours
  - L4-L5: 48 hours (LSTM)
  - L6-L7: 72 hours (Module C + multi-agent)
  - L8: 120 hours (long-horizon communication)
- Total: ~260 GPU-hours
- Cost: ~$3/hour × 260 = **~$780**

**Month 3-4 (experiments)**:

- Same GPU instance
- Pilot experiments: 50 GPU-hours
- Ablations: 30 GPU-hours
- Total: 80 GPU-hours
- Cost: ~$3/hour × 80 = **~$240**

**Total compute cost**: ~$1,020 (very manageable)

**Alternative (free tier)**:

- Google Colab Pro+ ($50/month)
- 4 months × $50 = $200
- Trade-off: Slower, but much cheaper

#### Software Infrastructure

**Required**:

- GitHub (free for public repos)
- PyTorch (free)
- Pytest (free)
- YAML libraries (free)

**Optional**:

- Weights & Biases (experiment tracking, free tier sufficient)
- Discord (community, free)
- Notion/GitHub Wiki (docs, free)

**Total software cost**: $0 (using free tiers)

---

### 10.6 Risk Mitigation

**What could go wrong? How do you handle it?**

#### Risk 1: LSTM Training Fails (L4-L5)

**Probability**: Medium (~40%)  
**Impact**: High (blocks L6-L8)

**Mitigation**:

- Week 2 fallback: Use simpler RNN instead of LSTM
- Week 3 fallback: Keep full observability, skip partial obs curriculum
- Month 2 fallback: Use pretrained spatial encoder from other RL work
- Acceptance: L4-L5 may need more tuning than expected (add 2 weeks buffer)

---

#### Risk 2: Emergent Communication Doesn't Emerge (L8)

**Probability**: High (~60%)  
**Impact**: Medium (L8 is research frontier, not core platform)

**Mitigation**:

- Month 3 fallback: Use hand-crafted signal semantics as baseline
- Month 3 alternative: Focus on L6-L7 (social reasoning still valuable)
- Month 4 pivot: Reframe as "communication infrastructure" not "emergent protocols"
- Acceptance: L8 is aspirational; failure is publishable result

---

#### Risk 3: Multi-Agent Training is Unstable (L6-L7)

**Probability**: Medium (~50%)  
**Impact**: High (blocks social reasoning)

**Mitigation**:

- Month 2 fallback: Train agents against frozen opponents (not co-adapting)
- Month 2 alternative: Use league training (like AlphaStar)
- Month 3 pivot: Focus on single-agent with simulated competitors
- Acceptance: Multi-agent RL is hard; may need architecture research

---

#### Risk 4: Performance Degradation Across Curriculum

**Probability**: Low (~20%)  
**Impact**: Medium (curriculum value unclear)

**Mitigation**:

- Month 3 validation: Measure transfer vs scratch
- Month 3 alternative: Offer both curriculum and direct training
- Acceptance: Curriculum may be pedagogical (for humans) not necessary (for agents)

---

#### Risk 5: Adoption Fails (Post-Release)

**Probability**: Medium (~50%)  
**Impact**: Low (research value exists regardless)

**Mitigation**:

- Month 4 preparation: Create killer demo (video, interactive)
- Month 5 outreach: Target specific research groups (multi-agent RL, evolutionary)
- Month 6 pivot: Use internally for your own research (still valuable)
- Acceptance: Building good tools is inherently useful, even with small community

---

### 10.7 Checkpoint Meetings (Self-Review)

**Weekly check-ins to stay on track**

#### Week 1 Friday: Blockers Review

```
Questions:
1. Are all three blockers fixed? (Yes/No)
2. Do tests pass? (pytest tests/test_blockers.py)
3. Is documentation updated? (Sections 6.1-6.3)

If NO to any:
- Identify bottleneck (technical? time?)
- Adjust plan (cut scope? extend deadline?)
- Document decision (why, what changes)
```

#### Week 2 Friday: Specifications Review

```
Questions:
1. Is contention deterministic? (run test 100 times)
2. Do all family transitions work? (edge case tests)
3. Can you initialize a child? (test all 4 modes)

If NO to any:
- Extend Week 2 into Week 3 (push social obs back)
- Or: Punt complex specs to Month 4 (focus on core curriculum)
```

#### Week 3 Friday: Social Observations Review

```
Questions:
1. Can you run L6 multi-agent episode? (3+ agents)
2. Are cues in observations? (inspect tensor)
3. Does Module C accept cues? (forward pass test)

If NO to any:
- Debug observation builder (print tensors)
- Simplify cues (use only 3 types, not 12)
- Defer Module C training (use random predictions)
```

#### Week 4 Friday: Communication Review

```
Questions:
1. Can family members exchange signals? (end-to-end test)
2. Is SET_COMM_CHANNEL action working? (telemetry check)
3. Are signals in observations? (inspect tensor)

If NO to any:
- Review family lifecycle (is family_id set?)
- Check observation builder (is family_comm_channel populated?)
- Test with 2 agents only (simplify)
```

#### Month 2 End: Training Review

```
Questions:
1. Did L0-L5 agents train successfully? (learning curves)
2. Are graduation criteria met? (retirement rates)
3. Are checkpoints saved? (can load and resume)

If NO to any:
- Extend training (add 1 week)
- Tune hyperparameters (learning rate, batch size)
- Accept lower performance (update success criteria)
```

#### Month 3 End: Curriculum Review

```
Questions:
1. Did L6-L8 training complete? (even if imperfect)
2. Are pilot experiments done? (3 results)
3. Is anything publishable? (write draft)

If NO to any:
- Focus on best result (L6 or L7, skip L8)
- Simplify experiments (1 strong result > 3 weak)
- Accept partial success (L0-L7 is still valuable)
```

#### Month 4 End: Release Review

```
Questions:
1. Is documentation complete? (QUICKSTART exists)
2. Can external user run experiment? (user testing)
3. Are tests passing? (CI green)

If NO to any:
- Delay release (better to launch well than fast)
- Recruit beta testers (friendly users)
- Cut scope (release L0-L5 first, L6-L8 later)
```

---

### 10.8 Contingency Plans

**If timeline slips, what gets cut?**

#### Priority Tiers

**Tier 0 (Must Have)**:

- Week 1: Blocker fixes
- Week 2-4: L0-L5 implementation
- Month 2: L0-L5 training
- Month 4: Basic documentation

**Tier 1 (Should Have)**:

- Week 3-4: L6-L7 social observations
- Month 3: L6-L7 training
- Month 4: Schema validation
- Month 4: Testing (80% coverage)

**Tier 2 (Nice to Have)**:

- Week 4: L8 communication
- Month 3: L8 training
- Month 3: Pilot experiments
- Month 4: Video tutorials

**Tier 3 (Future Work)**:

- Population genetics experiments
- Alternative inheritance modes
- Dynasty experiments
- External applications (economy, ecosystem)

#### Cut Order (if 4 months becomes 3 months)

**Week 1**: No cuts (blockers are critical)  
**Week 2**: No cuts (specs are foundational)  
**Week 3**: Cut L7 (keep L6 with sparse cues)  
**Week 4**: Cut L8 entirely (defer to future work)  
**Month 2**: No cuts (training is essential)  
**Month 3**: Cut L8 training, simplify experiments to 1 pilot  
**Month 4**: Cut video tutorials, reduce test coverage to 60%  

**Result**: 3-month plan focuses on L0-L6, solid foundation for future work

#### Cut Order (if 4 months becomes 2 months)

**Week 1-2**: Combine (fix blockers + critical specs only)  
**Week 3-4**: Skip social obs, keep L0-L5 only  
**Month 2**: Train L0-L5 only  
**Month 3-4**: Documentation + release (L0-L5 only)  

**Result**: 2-month plan is "Townlet Core" (single-agent, no social)

---

### 10.9 Success Criteria (Final)

**What does "done" look like?**

#### Minimum Viable Product (3 months)

```python
✅ L0-L5 curriculum implemented and working
✅ Agents learn survival strategies from sparse rewards
✅ Partial observability (LSTM) works
✅ All three blockers fixed
✅ Basic documentation (QUICKSTART + API reference)
✅ Tests pass, coverage > 60%
✅ Can demo to external researcher
```

**Outcome**: Research-grade single-agent survival simulator

---

#### Full Product (4 months)

```python
✅ L0-L8 curriculum implemented and working
✅ Multi-agent social reasoning (L6-L7) works
✅ Family communication infrastructure (L8) exists (even if protocols unclear)
✅ Population genetics system implemented
✅ 3+ pilot experiments completed
✅ Comprehensive documentation (tutorials, videos)
✅ Tests pass, coverage > 80%
✅ Ready for public release
```

**Outcome**: Research platform for multi-agent, social, evolutionary RL

---

#### Stretch Goals (6 months)

```python
✅ All of Full Product, plus:
✅ Emergent communication demonstrated (L8)
✅ 5+ published experiments (selection, dynasties, protocols)
✅ 10+ external users / contributors
✅ 1+ paper submitted to conference
✅ Alternative applications (economy or ecosystem) implemented
✅ Featured in RL newsletter / podcast
```

**Outcome**: Established research platform, community forming

---

### 10.10 Deliverable Checklist

**Concrete artifacts to produce**

#### Code

- [ ] `ethics_filter.py` (deterministic, no weights)
- [ ] `secure_checkpoint.py` (HMAC signatures)
- [ ] `observation_builder.py` (L6-L8 support)
- [ ] `cue_engine.py` (cue computation)
- [ ] `family.py` (lifecycle state machine)
- [ ] `breeding_selector.py` (pairing logic)
- [ ] `child_initializer.py` (inheritance modes)
- [ ] `population_controller.py` (cap maintenance)

#### Configs

- [ ] `cues.yaml` (baseline cue pack, 12 cues)
- [ ] `population_genetics.yaml` (all inheritance modes)
- [ ] `curriculum_configs/` (L0-L8 directories)
- [ ] Config templates (meritocratic, dynasty, arranged)

#### Tests

- [ ] `test_blockers.py` (Week 1 validation)
- [ ] `test_specifications.py` (Week 2 validation)
- [ ] `test_social_observations.py` (Week 3 validation)
- [ ] `test_family_communication.py` (Week 4 validation)
- [ ] `test_curriculum.py` (integration tests)
- [ ] Coverage report (pytest-cov)

#### Documentation

- [ ] `QUICKSTART.md` (10-minute tutorial)
- [ ] `TUTORIAL.md` (30-minute walkthrough)
- [ ] `API_REFERENCE.md` (complete spec)
- [ ] `CONTRIBUTING.md` (for contributors)
- [ ] `docs/multi_agent_mechanics.md` (contention resolution)
- [ ] `docs/family_lifecycle.md` (state machine)
- [ ] `docs/population_genetics.md` (inheritance modes)
- [ ] `docs/cues.md` (social observability)
- [ ] This master document (Sections 0-11)

#### Trained Models

- [ ] `checkpoints/L0_baseline/` (survival basics)
- [ ] `checkpoints/L3_baseline/` (full small-grid)
- [ ] `checkpoints/L5_baseline/` (LSTM + temporal)
- [ ] `checkpoints/L7_baseline/` (social reasoning)
- [ ] `checkpoints/L8_baseline/` (communication, even if imperfect)
- [ ] `checkpoints/module_c_pretrained/` (CTDE social model)

#### Experimental Results

- [ ] `results/Q1_1_selection_works.md` (natural selection)
- [ ] `results/Q2_1_emergent_comm.md` (protocols, even if negative result)
- [ ] `results/Q3_1_wealth_concentration.md` (dynasties)
- [ ] Learning curves (all L0-L8)
- [ ] Ablation studies (curriculum vs scratch)
- [ ] Visualizations (population dynamics, signal usage)

#### Release Materials

- [ ] `README.md` (from Section 11)
- [ ] GitHub repo setup
- [ ] v1.0.0 release tag
- [ ] Announcement blog post
- [ ] Demo video (5 minutes)
- [ ] Twitter/Reddit posts

---

### 10.11 Final Timeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│ MONTH 1: CORE INFRASTRUCTURE                                │
├─────────────────────────────────────────────────────────────┤
│ Week 1: Fix blockers (ethics, checkpoints, hash)            │
│ Week 2: Complete specs (contention, families, children)     │
│ Week 3: L6-L7 social observations                           │
│ Week 4: L8 communication channel                            │
│                                                              │
│ Deliverable: L0-L8 implemented (not trained)                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 2: TRAINING (L0-L7)                                   │
├─────────────────────────────────────────────────────────────┤
│ Week 1: L0-L3 (SimpleQNetwork)                              │
│ Week 2: L4-L5 (LSTM)                                        │
│ Week 3: Module C pretraining (CTDE)                         │
│ Week 4: L6-L7 (social reasoning)                            │
│                                                              │
│ Deliverable: Trained agents L0-L7                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 3: ADVANCED TRAINING & EXPERIMENTS                    │
├─────────────────────────────────────────────────────────────┤
│ Week 1-2: L8 training (communication)                       │
│ Week 3: Curriculum validation                               │
│ Week 4: Pilot experiments (3 results)                       │
│                                                              │
│ Deliverable: Complete curriculum + pilot results            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 4: POLISH & RELEASE                                   │
├─────────────────────────────────────────────────────────────┤
│ Week 1-2: Documentation                                     │
│ Week 3: Schema validation & tooling                         │
│ Week 4: Testing & CI                                        │
│                                                              │
│ Deliverable: v1.0.0 public release                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 5+: COMMUNITY & RESEARCH                              │
├─────────────────────────────────────────────────────────────┤
│ Public release, user support, paper submissions             │
│ External experiments, community growth                       │
└─────────────────────────────────────────────────────────────┘
```

---

**Critical path**: Blockers → Specs → Training → Release  
**Buffer**: 2 weeks (can absorb delays in Month 2-3)  
**Total time**: 4 months to v1.0.0  
**Total cost**: ~$1,000 compute + 4 months salary  

**Next action**: Start Week 1, Monday, with EthicsFilter refactor.

---

## SECTION 11: THE PITCH

**This is your README.md** — the first thing researchers see when they visit your repo.

---

# Townlet

> **Configuration-driven multi-agent RL for long-horizon planning, social reasoning, and emergent coordination**

Townlet is a research platform where you design worlds and minds as YAML files, not Python code. Agents learn survival strategies from sparse rewards, reason about competitors via observable cues, and coordinate through emergent communication — all without dense reward shaping or privileged information access.

**The core insight**: Most RL simulators hardcode their physics and give agents omniscience. Townlet makes worlds auditable (YAML configs), observations realistic (human observer principle), and results reproducible (cryptographic provenance).

---

## Quick Example

**Create a world** (configure physics as YAML):

```yaml
# universe_my_world.yaml
bars:
  - name: energy
    initial: 1.0
    base_depletion: 0.005  # drops 0.5% per tick

cascades:
  - source: satiation
    target: health
    threshold: 0.2  # starving
    strength: 0.010  # -1% health per tick

affordances:
  - id: bed
    effects:
      - { meter: energy, amount: 0.25 }
    costs:
      - { meter: money, amount: 0.05 }
```

**Train an agent** (no dense shaping, just survival):

```bash
townlet train --config configs/my_world/ --level L4
```

**Result**: Agent learns to work → earn money → buy food → avoid starvation, discovering multi-step strategies from sparse reward (`r = energy × health`).

**No hand-crafted rewards like**:

```python
# ❌ Dense shaping (what we DON'T do)
if action == "eat": reward += 0.1
if action == "sleep": reward += 0.1
```

**Just natural consequences**:

```python
# ✅ Sparse rewards (what we DO)
reward = energy * health  # feel good when healthy and energized
terminal_bonus = quality_of_life_at_retirement  # long-horizon planning
```

---

## Why This Exists

Most multi-agent RL simulators:

- **Hardcode physics** in Python → impossible to compare across experiments
- **Give agents omniscience** → unrealistic policies that break on deployment  
- **Use dense reward shaping** → agents follow breadcrumbs, don't discover strategies
- **Black-box checkpoints** → can't audit "which brain did what"

Townlet enforces:

1. **Configuration over code** — worlds are YAML files (auditable, version-controlled)
2. **Human observer principle** — agents only see what humans could see (no telepathy)
3. **Sparse rewards** — agents discover strategies via exploration, not shaping
4. **Provenance by design** — every run is cryptographically signed (brain identity + tamper protection)

**Result**: Experiments that are reproducible, auditable, and scientifically rigorous.

---

## Key Features

### 🎯 **Sparse Rewards + Long Horizons**

- Per-tick: `r = energy × health` (no action-specific shaping)
- Terminal: Retirement bonus based on lifespan, wellbeing, wealth
- Agents learn 50+ tick credit assignment (Job → money → food → survival)

### 🧠 **8-Level Curriculum (L0-L8)**

- **L0-3**: Learn survival basics (full observability)
- **L4-5**: Navigate under uncertainty (partial observability, LSTM required)
- **L6-7**: Compete with others (social reasoning via observable cues)
- **L8**: Coordinate via emergent communication (signals without pre-shared semantics)

### 👥 **Multi-Agent Social Reasoning**

- Agents observe competitors via **public cues** (body language, location)
- No telepathy: can't see others' internal state directly
- Module C learns: `['looks_tired', 'at_job'] → predicted_state`
- Strategic resource allocation (avoid contested affordances)

### 🧬 **Population Genetics**

- Families form, breed, and pass on learned strategies
- Child initialization: DNA crossover + weight inheritance
- Three modes: meritocratic (baseline), dynasty (inheritance), arranged (diversity)
- Research questions: Does selection work? Do protocols transfer across generations?

### 🔒 **Governance-Ready**

- Deterministic EthicsFilter (provably enforces rules)
- Cognitive hashing (unique ID per brain configuration)
- Signed checkpoints (HMAC tamper detection)
- Glass-box telemetry (candidate action → panic → ethics → final action)

### 📊 **Configuration-Driven Science**

- Hypothesis: "Does scarcity affect behavior?"
- Config A: `fridge.cost: 0.04`
- Config B: `fridge.cost: 0.08`
- Compare results with exact config diffs
- Publish with provenance (cognitive hash proves brain identity)

---

## What You Can Build

### Research Applications

**Multi-Agent Coordination**

- Do agents learn Theory of Mind from cues?
- Can emergent communication protocols evolve?
- How does wealth inequality emerge in dynasties?

**Long-Horizon RL**

- Test credit assignment over 100+ steps
- Validate curriculum learning benefits
- Compare sparse vs dense reward shaping

**Evolutionary Dynamics**

- Does natural selection work in RL populations?
- Lamarckian vs Darwinian evolution
- Optimal mutation rates

### Educational Use Cases

**RL Pedagogy**

- Students modify YAML configs (no coding required)
- Observe behavioral changes from parameter tweaks
- Learn systems thinking (cascades create feedback loops)

**Example Assignment**:
> "Double food cost. Explain why agent behavior changed. Submit config diff + analysis."

### Policy Simulations

**Testable Interventions**

- Model welfare systems, UBI, tax policies
- Measure behavioral outcomes (work incentives, poverty rates)
- Present results with provenance (auditable)

---

## Beyond Towns: The General Platform

Townlet's core abstraction (**bars + cascades + affordances**) works for any domain where:

- State evolves over time
- Variables are coupled
- Actions have effects

**Example applications** (not yet implemented, but enabled by architecture):

**Economic Simulation**

- Bars: GDP, inflation, unemployment, debt
- Affordances: raise_interest_rates, government_spending
- Research question: Can RL learn monetary policy?

**Ecosystem Management**

- Bars: predator_pop, herbivore_pop, vegetation
- Affordances: hunt, graze, migrate
- Research question: Do Lotka-Volterra cycles emerge?

**Supply Chain Optimization**

- Bars: inventory, customer_satisfaction, cash
- Affordances: order_inventory, expedited_shipping
- Research question: Can RL discover (s,S) policies?

See [Section 9: The Bigger Vision](docs/bigger_vision.md) for details.

---

## Quick Start

### Installation

```bash
git clone https://github.com/tachyon-beep/townlet.git
cd townlet
pip install -e .
```

### Run Your First Experiment (5 minutes)

```bash
# Train agent on baseline survival world (Level 0)
townlet train --config configs/baseline/ --level L0

# Agent learns: "Bed fixes energy" by episode 20
# Watch training: tensorboard --logdir runs/
```

### Customize a World (10 minutes)

```bash
# Copy baseline config
cp -r configs/baseline/ configs/my_world/

# Edit physics
vim configs/my_world/affordances.yaml
# Change: fridge.cost from 0.04 to 0.08 (make food expensive)

# Train on modified world
townlet train --config configs/my_world/ --level L2

# Compare results
townlet compare --baseline runs/baseline_L2_* --treatment runs/my_world_L2_*
# See: agents work more, hoard less, higher starvation rate
```

### Next Steps

- **Tutorial**: [30-minute walkthrough](docs/TUTORIAL.md)
- **Curriculum Guide**: [Understanding L0-L8](docs/curriculum.md)
- **Configuration Reference**: [Complete YAML spec](docs/configuration.md)
- **Research Examples**: [Population genetics experiments](docs/experiments.md)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIVERSE AS CODE                          │
│  bars.yaml     cascades.yaml    affordances.yaml  cues.yaml  │
│  (what exists) (how coupled)    (what actions)   (visible)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   VECTORIZED ENVIRONMENT                     │
│  • Reads world physics from YAML                             │
│  • Enforces human observer principle                         │
│  • Computes sparse rewards (r = energy × health)             │
│  • Emits cryptographically signed telemetry                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      BRAIN AS CODE                           │
│  Layer 1: Cognitive Topology (behavior rules)                │
│  Layer 2: Agent Architecture (neural blueprints)             │
│  Layer 3: Execution Graph (reasoning pipeline)               │
│                                                              │
│  Modules: Perception → World Model → Social Model → Policy  │
│  Controllers: Panic (survival override) → Ethics (veto)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       PROVENANCE                             │
│  • Cognitive hash (brain fingerprint)                        │
│  • Snapshot immutability (config frozen at launch)           │
│  • Signed checkpoints (HMAC tamper detection)                │
│  • Glass-box telemetry (full reasoning trace)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Curriculum Levels at a Glance

| Level | Challenge | Skills Learned | Architecture |
|-------|-----------|---------------|--------------|
| **L0-1** | Survival basics | Affordance semantics, economic loops | SimpleQNetwork |
| **L2-3** | Multi-resource balance | Cascade management, optimization | SimpleQNetwork |
| **L4-5** | Fog of war | Exploration, spatial memory, temporal planning | LSTM required |
| **L6-7** | Competition | Social reasoning, strategic resource allocation | Module A-D |
| **L8** | Coordination | Emergent communication, family protocols | Module A-D + channel |

**Pedagogical principle**: Each level removes one assumption

- L0-3: Full observability (scaffolding)
- L4-5: Partial observability (must explore and remember)
- L6-7: Social observability (cues only, no telepathy)
- L8: Communication (signals without semantics)

---

## Example: Emergent Communication (L8)

**Setup**: Parents and child can broadcast integer signals [0, 999]

**No semantic bootstrapping**:

```python
# ❌ We DON'T provide meanings
signal_meanings = {
    123: "job_taken",
    456: "danger",
}

# ✅ We DO provide raw channel
family_comm_channel = [0.123, 0.456, 0.0]  # what family members broadcast
# Agents must learn correlations via CTDE
```

**Learning process**:

1. Parent at Job broadcasts signal `123` (exploration)
2. Child observes: parent at Job correlates with signal `123`
3. Module C learns via supervised learning: `signal=123 → parent_at_job`
4. Policy learns: "When I see `123`, don't go to Job (it's occupied)"
5. Coordination emerges: family avoids resource conflicts

**Research result**: Do stable protocols emerge by episode 20k?

---

## Documentation

### Core Docs

- **[Quick Start](docs/QUICKSTART.md)** — 10-minute tutorial
- **[Tutorial](docs/TUTORIAL.md)** — 30-minute walkthrough  
- **[Configuration Reference](docs/configuration.md)** — Complete YAML spec
- **[Curriculum Guide](docs/curriculum.md)** — L0-L8 progression
- **[API Reference](docs/API_REFERENCE.md)** — Python API

### Design Docs

- **[Master Review](docs/master_review.md)** — Complete technical specification (this document)
- **[Design Principles](docs/design_principles.md)** — Human observer, sparse rewards, config-driven
- **[Reward Architecture](docs/reward_architecture.md)** — Why `r = energy × health` works
- **[Observation Space](docs/observation_space.md)** — Tensor specifications by level
- **[The Cues System](docs/cues_system.md)** — Social observability implementation

### Research Guides

- **[Population Genetics](docs/population_genetics.md)** — Family dynamics, inheritance modes
- **[Multi-Agent Mechanics](docs/multi_agent_mechanics.md)** — Contention resolution, social reasoning
- **[Emergent Communication](docs/emergent_communication.md)** — L8 protocols, analysis methods
- **[The Bigger Vision](docs/bigger_vision.md)** — Beyond towns (economy, ecosystems, etc.)

### Governance Docs

- **[Provenance System](docs/provenance.md)** — Cognitive hashing, checkpoint signing
- **[Ethics Architecture](docs/ethics.md)** — Deterministic compliance enforcement
- **[Audit Guide](docs/audit.md)** — How to verify "which brain did what"

---

## Performance

**Training times** (on single V100 GPU):

| Level | Episodes | Training Time | Graduation Rate |
|-------|----------|---------------|-----------------|
| L0-1  | 100      | ~2 hours      | 95%+ learn affordances |
| L2-3  | 500      | ~12 hours     | 80%+ reach retirement |
| L4-5  | 2,000    | ~48 hours     | 75%+ navigate from memory |
| L6-7  | 5,000    | ~72 hours     | 60%+ strategic resource selection |
| L8    | 20,000   | ~120 hours    | TBD (research frontier) |

**Compute costs**: ~$1,000 for full L0-L8 curriculum (AWS p3.2xlarge)

**Scalability**: Vectorized environments support 100+ parallel agents

---

## Validation & Testing

```bash
# Run all tests
pytest tests/

# Validate config
townlet validate --config configs/my_world/

# Check provenance
townlet verify --checkpoint runs/my_run/checkpoints/step_1000/
# Output: ✓ Signature valid, brain hash: 4f9a7c21ab...

# Compare experiments
townlet compare --baseline baseline_run --treatment treatment_run
# Output: Config diff, performance metrics, statistical significance
```

**Test coverage**: 80%+ (target)

---

## Research Using Townlet

**Papers** (to be added as they're published):

- *Coming soon* — Submit yours!

**Experiments**:

- Meritocratic selection in multi-agent RL populations
- Emergent communication protocols in families
- Wealth concentration in dynasty mode
- Curriculum learning for long-horizon tasks

**Open questions**:

- Can we decode emergent protocols via causal intervention?
- Do different dynasties evolve distinct strategies?
- What's the relationship between personality DNA and communication style?
- Can RL discover optimal monetary policy in economic simulations?

---

## Community & Contributing

**Discussions**: [GitHub Discussions](https://github.com/tachyon-beep/townlet/discussions)

**Issues**: [Bug reports & feature requests](https://github.com/tachyon-beep/townlet/issues)

**Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

- We welcome configs (new worlds!)
- We welcome experiments (research results!)
- We welcome docs (tutorials, examples!)
- We welcome code (features, bug fixes!)

**Code of Conduct**: Be respectful, collaborative, and curious

---

## Roadmap

### ✅ v1.0 (Current)

- L0-L8 curriculum
- Single-agent + multi-agent
- Population genetics (baseline)
- Provenance system
- Documentation

### 🚧 v1.1 (Next 3 months)

- Improved L8 training (communication)
- Dynasty experiments validated
- Schema validation tooling
- Video tutorials
- Community examples

### 🔮 v2.0 (6-12 months)

- Alternative applications (economy, ecosystems)
- Continuous action spaces
- Relational networks (beyond grids)
- Web-based visualization dashboard
- Benchmark suite (standardized tasks)

---

## Citation

If you use Townlet in your research, please cite:

```bibtex
@software{townlet2025,
  title = {Townlet: Configuration-Driven Multi-Agent RL for Long-Horizon Planning and Social Reasoning},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/tachyon-beep/townlet},
  version = {1.0.0}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

Built with:

- PyTorch (deep learning)
- Gymnasium (RL interface)
- YAML (configuration)
- pytest (testing)

Inspired by:

- Neural MMO (multi-agent survival)
- Melting Pot (social dilemmas)
- AlphaStar (emergent coordination)
- Open-Endedness literature (evolutionary RL)

Special thanks to the RL research community for feedback and ideas.

---

## Contact

- **GitHub**: [@tachyon-beep](https://github.com/tachyon-beep)
- **Email**: [your-email@domain.com]
- **Issues**: [Report bugs or request features](https://github.com/tachyon-beep/townlet/issues)

---

**Ready to build worlds?**

```bash
pip install townlet
townlet create --name my_first_world
townlet train --config configs/my_first_world/ --level L0
```

**Welcome to Townlet.** 🏘️

---
