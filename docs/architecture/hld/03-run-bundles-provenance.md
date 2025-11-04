# 3. Run Bundles and Provenance

**Document Type**: Component Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing runtime system, governance teams auditing deployments, researchers reproducing experiments
**Technical Level**: Deep Technical (filesystem structure, hashing mechanisms, provenance protocols)
**Estimated Reading Time**: 8 min for skim | 20 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
The framework's provenance system that transforms executions from "cool AI demo" to "auditable artifact with identity and chain of custody." Defines run bundles (pre-launch configs), configuration snapshots (frozen at launch), cognitive hash computation, and honest resume semantics.

**Why This Document Exists**:
Establishes the provenance guarantees that enable governance audit, scientific reproducibility, and forensic reconstruction. Makes "which mind, under which rules, did what and why" answerable with evidence, not guesses.

**Who Should Read This**:
- **Must Read**: Engineers implementing launcher/checkpoint system, governance auditors verifying provenance claims
- **Should Read**: Researchers comparing experimental runs, safety teams investigating incidents
- **Optional**: Operators running experiments (high-level understanding sufficient)

**Reading Strategy**:
- **Quick Scan** (8 min): Read §3.1-§3.2 for run bundle structure and snapshot creation
- **Partial Read** (15 min): Add §3.3 for checkpoint/resume semantics
- **Full Read** (20 min): Add §3.4 for governance significance

---

## Document Scope

**In Scope**:
- **Run Bundle**: Pre-launch configuration directory structure
- **Configuration Snapshot**: Launch-time freeze of all YAMLs
- **Cognitive Hash**: Identity computation from configs + compiled graph + architecture
- **Telemetry Structure**: Per-tick provenance logging
- **Resume vs Fork**: Honest continuation semantics
- **Forensics**: Behavioral reconstruction from logs

**Out of Scope**:
- **Checkpoint format internals**: See implementation docs
- **Telemetry transport**: See infrastructure docs
- **Database schema**: See persistence layer docs
- **UI visualization**: See §7 (Telemetry UI Surfacing)

**Critical Boundary**:
Provenance system is **framework-level** (works for any universe instance). Examples show **Townlet Town** run names (L99_AusterityNightshift) and affordances (Bed/Hospital/PhoneAmbulance), but the snapshot/hash/resume pattern applies to any BAC/UAC configuration.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [02-brain-as-code.md](02-brain-as-code.md) (BAC layers that get snapshotted), [01-executive-summary.md](01-executive-summary.md) (cognitive hash concept)
- **Builds On**: BAC and UAC specifications (what gets frozen)
- **Related**: [04-checkpoints.md](04-checkpoints.md) (checkpoint format details), [06-runtime-engine-components.md](06-runtime-engine-components.md) (factory hash computation)
- **Next**: [04-checkpoints.md](04-checkpoints.md) (detailed checkpoint specification)

**Section Number**: 3 / 12
**Architecture Layer**: Physical (artifact structure and provenance protocols)

---

## Keywords for Discovery

**Primary Keywords**: run bundle, configuration snapshot, cognitive hash, provenance, resume, fork, telemetry
**Secondary Keywords**: run ID, tick index, forensics, config diff, runtime envelope, checkpoint
**Subsystems**: launcher, factory, telemetry, checkpoint system
**Design Patterns**: Snapshot+Hash pattern, immutable artifact, chain of custody, honest resume

**Quick Search Hints**:
- Looking for "how runs are structured"? → See §3.1 (Run Bundle)
- Looking for "what gets frozen at launch"? → See §3.2 (Configuration Snapshot)
- Looking for "how hash is computed"? → See §3.2 (Cognitive Hash section)
- Looking for "resume vs fork"? → See §3.3 (Checkpoints and Resume)
- Looking for "why provenance matters"? → See §3.4 (Non-Negotiable Provenance)

---

## Version History

**Version 1.0** (2025-11-05): Initial provenance specification establishing snapshot/hash/resume protocols

---

## Document Type Specifics

### For Component Specification Documents (Type: Component Spec)

**Component Name**: Provenance System (Run Bundles + Snapshots + Hash + Telemetry)
**Component Type**: Lifecycle Management System
**Location in Codebase**: Launcher, factory (hash computation), telemetry logger

**Interface Contract**:
- **Inputs**: Run bundle (configs/ directory), launch command
- **Outputs**: Run folder (runs/<run_id>/) with snapshot, hash, telemetry logs
- **Dependencies**: BAC/UAC configs, factory (for hash computation), filesystem
- **Guarantees**: Immutable snapshot, reproducible hash, traceable resume/fork

**Critical Properties**:
- **Immutable**: Config snapshot never changes after launch
- **Traceable**: Every telemetry entry links to run_id + cognitive_hash
- **Honest**: Resume recomputes hash; changes create fork (new identity)
- **Forensic**: Can reconstruct "why" from logs + snapshot

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 3. Run Bundles and Provenance

The Townlet Framework doesn't "run an agent". It mints an artifact with identity, provenance, and chain of custody. That's the difference between "cool AI demo" and "system we can take in front of governance without sweating through our shirt."

---

## 3.1 The Run Bundle

Before a run starts, you prepare a **run bundle** (configuration directory) under `configs/<run_name>/`:

```text
configs/
  L99_AusterityNightshift/              # (Townlet Town run name)
    config.yaml                         # Runtime envelope: tick rate, duration, curriculum, seed
    universe_as_code.yaml               # UAC: The world (bars, affordances, prices, cues)
    cognitive_topology.yaml             # BAC Layer 1 (behavior contract and safety knobs)
    agent_architecture.yaml             # BAC Layer 2 (module blueprints and interfaces)
    execution_graph.yaml                # BAC Layer 3 (think loop + panic/ethics chain)
```

**What each file contains**:

**universe_as_code.yaml** (UAC):
The world specification. Defines:
- Bars (energy, health, money - Townlet Town vocabulary; other instances use different bars)
- Affordances (Bed, Job, Hospital, PhoneAmbulance - Townlet Town; other instances define different affordances)
- Per-tick effects and costs
- Capacity limits, interrupt rules
- Whitelisted special effects (e.g., `teleport_to:hospital`)
- Public cues other agents can see ("looks_tired", "bleeding", "panicking" - instance-specific)

**The three BAC layers** (L1/L2/L3):
The mind specification (see §2 for details).

**config.yaml** (Runtime envelope):
Execution parameters:
- How long to run (num_ticks or max_episodes)
- Tick rate (ticks per second)
- Number of agents
- Curriculum schedule (e.g., "start alone, introduce food-scarcity rival after 10k ticks" - Townlet Town)
- Random seed for reproducibility

**Framework note**: The five-file bundle structure is framework-level. The example name "L99_AusterityNightshift" and contents (energy/health bars, Bed/Hospital affordances) are Townlet Town instance-specific. A factory simulation might use "F03_MachineryStress" with machinery_health bars and Assembly Line affordances.

**This bundle is what we claim we are about to run.**

---

## 3.2 Launching a Run

When we actually launch, **we don't execute the live bundle**. We snapshot it.

The launcher creates a **run folder**:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22/   # Run ID: <name>__<timestamp>
    config_snapshot/                              # Frozen configs
      config.yaml
      universe_as_code.yaml
      cognitive_topology.yaml
      agent_architecture.yaml
      execution_graph.yaml
      full_cognitive_hash.txt                     # Computed identity
    checkpoints/                                  # Periodic saves
    telemetry/                                    # Per-tick logs
    logs/                                         # System logs
```

**Critical details**:

### Configuration Snapshot (Frozen Configs)

**`config_snapshot/`** is a byte-for-byte copy of the five YAMLs at launch time.

**Immutability guarantee**: After launch, the runtime simulator reads **only** from this snapshot, never from the mutable `configs/` directory. This prevents untracked hotpatches to ethics during a run.

**Why this matters**: Governance cannot be assured if the "forbid_actions" list can change mid-run without leaving evidence. The snapshot makes configuration immutable.

### Cognitive Hash Computation

During agent instantiation (via factory.py), the framework computes **`full_cognitive_hash.txt`** from:

1. **Exact text of the five snapshot YAMLs** (byte-for-byte content)
2. **Compiled execution graph** (post-resolution: real step order after resolving @modules.* symbolic bindings)
3. **Instantiated module architectures** (types, hidden dims, optimizer hyperparameters from Layer 2)

**That hash is this mind's identity**. It's the "brain fingerprint plus declared world."

**Hash properties**:
- **Deterministic**: Same configs → same hash
- **Sensitive**: Any change (panic threshold, ethics rule, optimizer LR) → different hash
- **Provenance**: Links telemetry to exact configuration that produced it

**Framework note**: Cognitive hash computation is framework-level. The hash includes both BAC (mind) and UAC (world), making it the "complete cognitive+environmental identity."

### Telemetry Logging

The framework starts ticking. **Every tick** we log telemetry with:

- **run_id** (e.g., `L99_AusterityNightshift__2025-11-03-12-14-22`)
- **tick_index** (0, 1, 2, ...)
- **full_cognitive_hash** (links to exact config snapshot)
- **current_goal** (engine ground truth - Townlet Town: SURVIVAL/THRIVING/SOCIAL)
- **agent_claimed_reason** (what it says it's doing, if introspection enabled)
- **panic_state** and any panic override (was panic active this tick?)
- **candidate_action** (what policy wanted to do)
- **final_action** (what actually happened after panic + ethics)
- **ethics_veto_applied** and **veto_reason** (was action blocked? why?)
- **planning_depth** (world_model.rollout_depth - how far ahead it planned)
- **social_model.enabled** (was social reasoning active?)
- **Prediction summaries** from world_model and social_model (what it expected to happen)

**That is now evidence**. If someone later asks "why didn't the agent eat even though it was starving?", we don't guess. We read the log:
- Tick 842: `candidate_action=EAT_FRIDGE`, `final_action=WAIT`, `veto_reason="insufficient money"`, `panic_state=false` (hadn't hit threshold yet)

**Framework benefit**: Telemetry structure is framework-level. The specific fields (current_goal values, affordance names in actions) are instance-specific, but the provenance pattern (run_id + tick_index + hash + decision chain) applies to any universe.

---

## 3.3 Checkpoints and Resume

During the run, the framework periodically checkpoints to:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22/
    checkpoints/
      step_000500/
        weights.pt                      # Neural network weights
        optimizers.pt                   # Optimizer state
        rng_state.json                  # Random number generator state
        config_snapshot/                # Config snapshot (redundant copy)
          config.yaml
          universe_as_code.yaml
          cognitive_topology.yaml
          agent_architecture.yaml
          execution_graph.yaml
          full_cognitive_hash.txt
```

**Each checkpoint is effectively "a frozen moment of mind + world + RNG".**

### What Checkpoints Enable

**1. Honest Resume**

To resume, the framework:
1. **Loads from the checkpoint's `config_snapshot/`**, not from `configs/` (prevents stealth edits)
2. Writes out a new run folder: `L99_AusterityNightshift__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/`
3. **Recomputes the cognitive hash** from the loaded snapshot

**Honest continuation**:
- If the snapshot is unchanged → hash matches → we can legitimately say "this is a continuation of the same mind"
- If we touch anything cognitive or world-rules (panic thresholds, forbid_actions, ambulance cost, bed healing rate, module architecture) → hash changes → **that is now a fork, not a continuation**

**Governance integrity**: You cannot stealth-edit survival rules and claim it's still the same agent. The hash proves otherwise.

**Framework pattern**: Resume vs fork semantics are framework-level. Any BAC or UAC change creates a fork.

**2. Forensics**

We can go back to tick 842 (or any tick) and reconstruct:
- What body state the agent believed it was in (from telemetry: bar values, belief_distribution)
- What goal it claimed (SURVIVAL/THRIVING/SOCIAL - Townlet Town)
- Whether panic took over (panic_state=true, panic_reason logged)
- Whether EthicsFilter stopped something illegal (ethics_veto_applied=true, veto_reason="forbidden: steal")
- What world rules and costs it was operating under (from config_snapshot: ambulance price, bed healing rate, etc.)

**Framework benefit**: Forensics work for any universe. The specific bars and affordances differ, but the reconstruction pattern (telemetry + snapshot → "why") is universal.

**3. Curriculum / Scientific Comparison**

We can diff two runs and say:
> "The only change was that we turned off the Social Model and raised panic aggressiveness (panic_thresholds.energy: 0.15 → 0.10). Here's how behavior shifted (survival rate 45% → 62%, social interactions dropped 90%)."

**It's not anecdote, it's a config diff plus a new hash.**

**Framework pattern**: Config diff enables controlled experiments: change one variable (Layer 1 setting, Layer 2 architecture, UAC affordance), measure behavioral shift.

---

## 3.4 Why Provenance Is Non-Negotiable

Without this provenance model, the Townlet Framework would revert to a generic agent-in-a-box demonstration, forcing governance to rely on trust rather than evidence.

**With this provenance model**:

**Governance audit**:
- We can prove at audit time which ethics rules were live (read forbid_actions from config_snapshot)
- We can prove panic never bypassed ethics unless someone explicitly allowed that in Layer 3 (and if they did, the hash changed, creating evidence of the modification)

**Incident investigation**:
- We can replay any behavior clip and show both "what happened" (telemetry) and "which mind, under which declared rules, proposed, attempted, and was vetoed" (config_snapshot + hash)

**Scientific rigor**:
- Reproducibility: Same config_snapshot → same hash → same mind (modulo RNG if not seeded)
- Experimental control: Config diff shows exactly what changed between runs

**Deployment readiness**: This capability enables deployment beyond laboratory settings. Regulators and safety teams can audit based on evidence, not promises.

**Framework foundation**: Provenance is what transforms "interesting research artifact" into "governable AI system."

---

**Summary**: The Townlet Framework provenance system works as follows:
1. **Prepare run bundle** (`configs/<run_name>/` with 5 YAMLs)
2. **Snapshot at launch** (byte-for-byte copy → `runs/<run_id>/config_snapshot/`)
3. **Compute cognitive hash** (frozen configs + compiled graph + architectures)
4. **Log telemetry** (every tick: run_id + tick_index + hash + decision chain)
5. **Checkpoint periodically** (snapshot + weights + RNG for resume)
6. **Resume honestly** (recompute hash; changes create fork, not continuation)

This is the framework's identity and accountability mechanism. BAC/UAC define the mind and world. Provenance proves which mind, under which world, did what.

---
