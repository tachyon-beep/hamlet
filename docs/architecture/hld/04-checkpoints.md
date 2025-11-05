# 4. Checkpoints

**Document Type**: Component Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing checkpoint system, governance auditors verifying provenance, researchers replaying experiments
**Technical Level**: Deep Technical (file formats, serialization, causality preservation)
**Estimated Reading Time**: 6 min for skim | 15 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Checkpoint specification - not "saved weights lol" but a frozen moment of a specific mind, in a specific world, under specific rules, at a specific instant. Defines the five components (weights.pt, optimizers.pt, rng_state.json, config_snapshot/, cognitive hash) and their legal significance for audit.

**Why This Document Exists**:
Establishes checkpoints as evidence artifacts that enable honest training resume, exact behavioral replay, and provenance proof. Turns anecdotes about agent behavior into auditable evidence trails.

**Who Should Read This**:
- **Must Read**: Engineers implementing checkpoint save/load, governance teams investigating incidents
- **Should Read**: Researchers reproducing experiments, safety auditors verifying claims
- **Optional**: Operators running training (high-level understanding sufficient)

**Reading Strategy**:
- **Quick Scan** (6 min): Read §4.1-§4.5 for checkpoint component purposes
- **Partial Read** (12 min): Focus on §4.4-§4.5 for config snapshot embedding and hash computation
- **Full Read** (15 min): Add §4.6 for legal/governance significance

---

## Document Scope

**In Scope**:
- **Checkpoint Components**: weights.pt, optimizers.pt, rng_state.json, config_snapshot/, cognitive hash
- **Why Each Component**: Purpose and what it preserves
- **Honest Resume**: Why optimizer state and RNG matter
- **Evidence Value**: How checkpoints eliminate plausible deniability

**Out of Scope**:
- **Serialization format details**: See implementation docs
- **Checkpoint frequency policy**: See training configuration
- **Storage backend**: See infrastructure docs
- **Resume algorithm**: See §5 (Resume Semantics) and §3.3 (Checkpoints and Resume)

**Critical Boundary**:
Checkpoint structure is **framework-level** (works for any SDA). Examples show **Townlet Town** run names (L99_AusterityNightshift) and affordances (STEAL actions, ambulance costs), but the five-component structure and hash-based identity apply to any BAC/UAC configuration.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [03-run-bundles-provenance.md](03-run-bundles-provenance.md) (provenance system overview), [02-brain-as-code.md](02-brain-as-code.md) (what gets saved in weights)
- **Builds On**: Configuration snapshot concept (§3.2)
- **Related**: [05-resume-semantics.md](05-resume-semantics.md) (how checkpoints are loaded), [06-runtime-engine-components.md](06-runtime-engine-components.md) (module registry)
- **Next**: [05-resume-semantics.md](05-resume-semantics.md) (resume vs fork semantics)

**Section Number**: 4 / 12
**Architecture Layer**: Physical (artifact specification and serialization)

---

## Keywords for Discovery

**Primary Keywords**: checkpoint, weights.pt, optimizers.pt, rng_state.json, config snapshot embedding, evidence trail
**Secondary Keywords**: module registry, optimizer state, RNG preservation, honest resume, plausible deniability
**Subsystems**: checkpoint system, module serialization, provenance
**Design Patterns**: Evidence artifact, embedded snapshot, reproducible causality

**Quick Search Hints**:
- Looking for "what's in a checkpoint"? → See §4.1-§4.5 (component breakdown)
- Looking for "why save optimizer state"? → See §4.2 (Optimizers)
- Looking for "why save RNG"? → See §4.3 (RNG State)
- Looking for "why embed config snapshot"? → See §4.4 (Config Snapshot)
- Looking for "legal significance"? → See §4.6 (Why Legally Interesting)

---

## Version History

**Version 1.0** (2025-11-05): Initial checkpoint specification defining five-component evidence artifact structure

---

## Document Type Specifics

### For Component Specification Documents (Type: Component Spec)

**Component Name**: Checkpoint (Evidence Artifact)
**Component Type**: Serialized Artifact (filesystem structure)
**Location in Codebase**: Checkpoint writer/loader, module registry

**Interface Contract**:
- **Inputs**: Running SDA (weights + optimizers + RNG + config_snapshot + hash)
- **Outputs**: Checkpoint directory (step_NNNNNN/) with five components
- **Dependencies**: PyTorch serialization, module registry, factory (hash computation)
- **Guarantees**: Reproducible replay, honest resume, provenance proof

**Critical Properties**:
- **Complete**: Contains everything needed for exact resume/replay
- **Self-Contained**: Embedded config_snapshot (not pointer to mutable configs)
- **Auditable**: Cognitive hash proves exact mind+world identity
- **Honest**: Optimizer state + RNG preserve learning trajectory and stochastic causality

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 4. Checkpoints

A checkpoint is not "saved weights lol". It's a **frozen moment of a specific mind, in a specific world, under specific rules, at a specific instant in time**.

The Townlet Framework treats every checkpoint as evidence. A checkpoint must include everything required to:

- **Pick up training honestly** (continue learning trajectory, not restart with different momentum)
- **Replay behavior honestly** (reproduce stochastic outcomes, not approximate them)
- **Prove provenance** (which exact cognitive configuration produced which exact action)

When the framework writes a checkpoint for a run, it creates:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22/   # (Townlet Town run)
    checkpoints/
      step_000500/                                # Checkpoint at tick 500
        weights.pt                                # Neural state
        optimizers.pt                             # Learning state
        rng_state.json                            # Causality state
        config_snapshot/                          # Embedded frozen configs
          config.yaml
          universe_as_code.yaml
          cognitive_topology.yaml
          agent_architecture.yaml
          execution_graph.yaml
        full_cognitive_hash.txt                   # Identity proof
```

**Let's unpack what those five components actually mean.**

---

## 4.1 weights.pt

This is the **live neural state of the brain** at that tick.

**Contains**:
- Perception module weights
- World model weights
- Social model weights
- Hierarchical policy weights
- Panic controller weights (if learned/parameterized)
- EthicsFilter weights (if learned/parameterized)
- Anything else registered in the **module registry**

**Framework pattern**: Module registry is populated from Layer 2 (`agent_architecture.yaml`). All modules declared there get saved together.

**Why all modules, not just policy?**: In v1 (old Hamlet), these components lived in one giant black-box DQN. In the Townlet Framework, they are separate submodules declared in Layer 2 and wired by Layer 3. We save them together because, for audit, **"the brain" encompasses the entire SDA module set**, not only the action head.

**Framework note**: The specific modules saved (perception, world_model, social_model, etc.) are framework-level patterns. Whether a universe instance uses all modules or disables some (e.g., `social_model.enabled: false` in Layer 1) affects what weights exist, but the save-all-registered-modules pattern is universal.

---

## 4.2 optimizers.pt

The framework logs both **parameters and optimizer state** (e.g., Adam moments) for each trainable module.

**Why?** Because **"resume training" must mean "continue the same mind's learning process"**, not "respawn something with the same weights but different momentum and call it continuous".

**Honest resume requirement**: If you've ever done RL, you know that quietly dropping optimizer state can absolutely change learning behavior. The framework refuses to pretend that's irrelevant. We store it.

**Framework pattern**: Optimizer state preservation is framework-level (required for honest resume). The specific optimizers (Adam, RMSprop, SGD) and their hyperparameters (lr=0.0001) are defined in Layer 2, but the principle "save optimizer state for continuity" is universal.

**Example**: Adam optimizer state includes first and second moment estimates for every parameter. Dropping this means the next training step has no momentum history, changing the learning trajectory.

---

## 4.3 rng_state.json

**Randomness is part of causality.**

The framework stores the **RNG states** that matter:
- Environment RNG (affordance tie-breaks, spawn locations - Townlet Town)
- Agent RNG (PyTorch generators for exploration noise, stochastic policy sampling)
- Any other source affecting rollout sampling, exploration, or decision-making

**Why?** This allows us to **re-run tick 501 and observe the same stochastic outcomes**.

**Replay value**: When someone asks, "would it always have chosen STEAL here?", we can answer: "Under this exact random sequence, here is what occurred," and reproduce the evidence without speculation.

**Framework pattern**: RNG preservation is framework-level (reproducible causality). The specific sources of randomness (environment contention, exploration strategy, policy sampling) vary by universe instance, but the principle "preserve RNG for replay" is universal.

**Example (Townlet Town)**: If two agents want the same Bed (capacity=1), environment RNG determines who wins. Preserving RNG lets us replay and see the same winner.

---

## 4.4 config_snapshot/

**This is critical.**

Inside every checkpoint, the framework embeds a **fresh copy** of the exact `config_snapshot/` that the run is using at that moment.

**That snapshot contains**:
- `config.yaml` (runtime envelope: tick rate, max ticks, curriculum step, etc.)
- `universe_as_code.yaml` (UAC: meters, affordances, costs, social cues, ambulance behavior, bed quality, etc. - Townlet Town vocabulary)
- `cognitive_topology.yaml` (BAC Layer 1: panic thresholds, ethics rules, personality - Townlet Town: greed=0.7, forbid STEAL)
- `agent_architecture.yaml` (BAC Layer 2: module shapes, learning rates, pretraining origins, interface dims)
- `execution_graph.yaml` (BAC Layer 3: think loop ordering - panic before ethics)

**This is not a pointer. It's an embedded copy at that checkpoint tick.**

**Why embed it every time?**

**Curriculum evolution**: Curriculum might change some parts of the world over time (e.g., add new competition, raise prices, close the hospital at night - Townlet Town examples). If that's allowed under policy, those changes will appear in `universe_as_code.yaml` at tick 10,000 that didn't exist at tick 500. **Checkpoint 500 needs to show what the world rules were then, not now.**

**Governance audit**: "Panic thresholds" and "forbid_actions" in `cognitive_topology.yaml` are part of that snapshot. When someone asks "did you allow it to steal at tick 842?", we don't argue philosophy. **We open the checkpoint around that time and read the file.**

**Framework pattern**: Embedded config snapshot is framework-level (prevents time-dependent ambiguity). Whether curriculum changes UAC (Townlet Town: prices rise) or BAC (some universe allows panic_threshold adjustments) doesn't matter - the embedding pattern ensures checkpoint shows the rules at that tick.

**Townlet Town example**: If curriculum raises ambulance cost from $300 to $500 at tick 5000, checkpoint at step_000500 shows `ambulance_cost: 300`, checkpoint at step_010000 shows `ambulance_cost: 500`.

---

## 4.5 full_cognitive_hash.txt

**This is the mind's ID badge.**

The hash is **deterministic** over:

1. **Exact text bytes of the 5 YAMLs** in the snapshot (config.yaml, universe_as_code.yaml, cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml)

2. **Compiled execution graph** after resolution. Not the pretty YAML, but the actual ordered list of steps the agent is running after we bind `@modules.*` symbolic bindings to real modules. If someone sneaks in "panic after ethics" instead of "panic before ethics", the hash changes.

3. **Constructed module architectures**. Types, hidden sizes, optimizer settings, interface dims. Not just "GRU exists", but "GRU with hidden_dim=512 paired with Adam lr=1e-4".

**That means**:

**Ethics changes detected**:
- If you fiddle the EthicsFilter to quietly allow STEAL under panic → hash changes

**Architecture changes detected**:
- If you widen the GRU and try to pretend it's the same mind → hash changes

**World changes detected**:
- If you reduce ambulance cost in the world (Townlet Town) → hash changes (because `universe_as_code.yaml` changed)

**We're basically tattooing "this exact mind in this exact world with this exact cognition loop" into the checkpoint.**

**Framework pattern**: Cognitive hash computation is framework-level. The hash includes both BAC (mind) and UAC (world), making it the complete cognitive+environmental identity for any universe instance.

---

## 4.6 Why Checkpoints Are Legally Interesting (Not Just Technically Interesting)

**Because they kill plausible deniability.**

**Example claims** (Townlet Town context):

Someone claims:
- "Oh, it only stole because it was desperate"
  **or**
- "Ethics must have bugged out at 2am"
  **or**
- "We didn't change anything important, we just tuned panic a little"

**You can respond with evidence**:
- "Here's the checkpoint from tick 800. Panic thresholds are documented: `energy: 0.15, health: 0.25`. Ethics still forbids STEAL: `forbid_actions: ["attack", "steal"]`. Hash `9af3c2e1` says it's the same mind before and after 2am. Telemetry shows the agent attempted STEAL at tick 842, EthicsFilter vetoed it (`veto_reason: "forbidden"`), and final_action was WAIT. So no, it wasn't allowed to steal. It attempted to anyway and EthicsFilter stopped it."

**In other words, checkpoints turn anecdotes about behavior into evidence trails.**

**Framework benefit**: This pattern works for any universe. The specific affordances (STEAL in Townlet Town, SHUTDOWN in factory), bars (energy vs machinery_stress), and ethics rules differ, but the evidence mechanism (checkpoint + hash + telemetry → proof) is universal.

**Governance value**:
- **Honest resume**: Can't quietly change rules mid-training and claim continuity
- **Reproducible replay**: Can re-run critical moments with same stochastic outcomes
- **Audit trail**: Config snapshot at each checkpoint proves what rules were active when

---

**Summary**: A Townlet Framework checkpoint is a five-component evidence artifact:
1. **weights.pt**: Neural state of all SDA modules
2. **optimizers.pt**: Learning state (Adam moments, etc.) for honest training continuation
3. **rng_state.json**: RNG state for reproducible stochastic causality
4. **config_snapshot/**: Embedded frozen configs showing rules at that tick
5. **full_cognitive_hash.txt**: Mind+world identity proof

Together, these transform checkpoints from "saved model" into "legal evidence for what mind, under what rules, did what."

---
