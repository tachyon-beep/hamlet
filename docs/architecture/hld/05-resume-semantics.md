# 5. Resume Semantics

**Document Type**: Design Rationale
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing resume logic, governance teams auditing continuity claims, researchers conducting ablation studies
**Technical Level**: Deep Technical (resume protocol, hash verification, fork detection)
**Estimated Reading Time**: 5 min for skim | 12 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Resume semantics - the forensic procedure for continuing training from checkpoints. Defines "the checkpoint snapshot is law" (restore from frozen configs, not live directory), hash-based continuity verification, and explicit fork vs continuation distinction.

**Why This Document Exists**:
Establishes resume as a governance primitive, not just technical convenience. Makes continuity of mind provable (same hash) vs fork (new hash), enabling honest ablation studies and eliminating untracked configuration drift.

**Who Should Read This**:
- **Must Read**: Engineers implementing resume/load, governance teams verifying continuity claims
- **Should Read**: Researchers doing ablations, safety auditors investigating "when did rules change"
- **Optional**: Operators resuming training (follow protocol, understand rationale less critical)

**Reading Strategy**:
- **Quick Scan** (5 min): Read §5.1-§5.2 for "checkpoint snapshot is law" rule
- **Partial Read** (8 min): Add §5.3 for fork vs continuation examples
- **Full Read** (12 min): Add §5.4 for governance significance

---

## Document Scope

**In Scope**:
- **Resume Rule**: Restore from checkpoint's config_snapshot/, not live configs/
- **Hash Verification**: Recompute hash to prove continuity (same) or fork (different)
- **Fork Examples**: What changes create new identity (ethics, panic, world rules, architecture)
- **Governance Value**: Why resume is primitive (not convenience), prevents drift

**Out of Scope**:
- **Resume implementation**: See runtime engine docs
- **Checkpoint format**: See §4 (Checkpoints)
- **Ablation experimental design**: See research methodology docs
- **Hardware failure recovery**: See infrastructure docs

**Critical Boundary**:
Resume semantics are **framework-level** (work for any BAC/UAC configuration). Examples show **Townlet Town** specifics (panic_thresholds.energy, forbid STEAL, ambulance costs), but the principle "checkpoint snapshot is law + hash proves continuity" applies to any universe instance.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [04-checkpoints.md](04-checkpoints.md) (what gets saved), [03-run-bundles-provenance.md](03-run-bundles-provenance.md) (cognitive hash concept)
- **Builds On**: Checkpoint structure (§4), configuration snapshot (§3.2)
- **Related**: [02-brain-as-code.md](02-brain-as-code.md) (what changes trigger new hash)
- **Next**: [06-runtime-engine-components.md](06-runtime-engine-components.md) (factory hash computation)

**Section Number**: 5 / 12
**Architecture Layer**: Logical (protocol specification and governance semantics)

---

## Keywords for Discovery

**Primary Keywords**: resume semantics, checkpoint snapshot is law, continuity of mind, fork vs continuation, hash verification
**Secondary Keywords**: rehydration, ablation study, configuration drift, governance primitive, honest resume
**Subsystems**: resume loader, factory (hash recomputation), checkpoint system
**Design Patterns**: Forensic procedure, hash-based identity, explicit fork

**Quick Search Hints**:
- Looking for "how to resume"? → See §5.1 (The Rule)
- Looking for "resume vs fork"? → See §5.3 (Forking vs Continuing)
- Looking for "what changes create fork"? → See §5.3 (Examples of forking)
- Looking for "why this matters"? → See §5.4 (Why Resume Semantics Matter)

---

## Version History

**Version 1.0** (2025-11-05): Initial resume semantics specification defining forensic resume protocol

---

## Document Type Specifics

### For Design Rationale Documents (Type: Design Rationale)

**Design Question Addressed**:
"How should the framework distinguish honest training continuation (same agent paused/resumed) from experimental fork (modified agent)?"

**Alternatives Considered**:
1. **Trust-based**: Assume user knows if they changed configs → **Rejected** (no provenance, unauditable)
2. **Pointer-based**: Resume from live configs/ directory → **Rejected** (allows untracked drift)
3. **Hash-based snapshot** (checkpoint snapshot is law) → **Chosen** (provable continuity)

**Key Trade-offs**:
- **Chosen**: Provable continuity, explicit forks, governance auditability
- **Sacrificed**: Convenience (can't just "tweak config and resume"), requires new run folder per resume

**Decision Drivers**:
- **Governance requirement**: Must prove agent didn't change during pause
- **Scientific rigor**: Ablations must be explicit, reviewable operations
- **Audit trail**: "When did this rule enter the system?" must be answerable

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 5. Resume Semantics

Resume operations must do more than reload weights; they are part of the **audit chain**.

If we can't prove continuity of mind across pauses, we can't claim continuity of behavior for governance, and we can't do serious ablation science.

**The framework defines resume like a forensic procedure.**

---

## 5.1 The Rule: The Checkpoint Snapshot Is Law

When you resume from a checkpoint, you **must** restore from the checkpoint's own `config_snapshot/`, not from whatever is currently sitting in `configs/<run_name>/` in your working tree.

**That means**:

**Restore exact cognitive_topology.yaml** from checkpoint:
- Same ethics (`forbid_actions: ["attack", "steal"]` - Townlet Town)
- Same panic thresholds (`energy: 0.15, health: 0.25` - Townlet Town bars)
- Same personality sliders (`greed: 0.7, curiosity: 0.8` - Townlet Town)

**Restore exact universe_as_code.yaml** from checkpoint:
- Same ambulance cost (e.g., `$300` - Townlet Town)
- Same bed healing effects (e.g., `+0.25 energy per tick` - Townlet Town)
- Same wage rates (e.g., Job pays `$22.5` - Townlet Town)

**Restore exact execution_graph.yaml** from checkpoint:
- Same panic-then-ethics ordering (panic_controller before EthicsFilter - framework pattern)

**Restore optimizer state and RNG** from checkpoint:
- Same Adam moments (framework-level: honest training continuation)
- Same random number generator states (framework-level: reproducible outcomes)

**You do not "reconstruct" the agent from the latest code and hope it's approximately right.** You **rehydrate** that specific mind in that specific world with that specific internal loop.

**Framework principle**: Checkpoint snapshot is law (framework-level). The specific contents (Townlet Town affordances, bars, goals) vary, but the rule "restore from checkpoint snapshot, not live configs" is universal.

---

## 5.2 Where the Resumed Run Lives

Resuming from:

```text
runs/L99_AusterityNightshift__2025-11-03-12-14-22/checkpoints/step_000500/
```

creates a **fresh new run folder**, for example:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/
    config_snapshot/          # Copied from checkpoint, byte-for-byte
    checkpoints/              # New checkpoints for resumed run
    telemetry/                # New telemetry logs
    logs/                     # New system logs
```

**Important details**:

**New run, new timeline**:
- We do **not** keep writing into the old run folder
- Each resume creates distinct timeline with its own provenance

**Hash recomputation**:
- The framework recomputes the cognitive hash from the checkpoint snapshot
- **If you have not changed anything** → hash matches → proves it's the same mind continuing
- **If snapshot was modified** → hash differs → proves it's a fork (new agent)

**Telemetry continuity**:
- Telemetry in the resumed run logs the **same hash** (if unchanged)
- Audit can verify: "This is truly the same mind, same ethics, same world, just continued later"

**Framework pattern**: New run folder per resume (framework-level). The naming convention (`_resume_<timestamp>`) and hash verification apply to any universe instance.

---

## 5.3 Forking vs Continuing

**Now the governance-critical part.**

If, before resuming, you **edit that copied snapshot, even slightly**, you are **not continuing**. You are **forking**.

**Examples of forking** (Townlet Town context):

**BAC Layer 1 changes** (behavior contract):
- Lower `panic_thresholds.energy` from `0.15` to `0.05` (agent panics earlier)
- Turn off `social_model.enabled` (disable social reasoning)
- Remove `"steal"` from `forbid_actions` (ethics now allows stealing)
- Change personality: `greed: 0.7 → 0.3` (less money-driven)

**UAC changes** (world rules):
- Change ambulance cost in `universe_as_code.yaml` (`$300 → $150`)
- Modify bed healing rate (`+0.25 energy/tick → +0.50 energy/tick`)
- Add new affordance or remove existing one

**BAC Layer 2 changes** (architecture):
- Widen GRU hidden dimension (`hidden_dim: 512 → 1024`)
- Change optimizer learning rate (`lr: 0.0001 → 0.0005`)

**BAC Layer 3 changes** (execution ordering):
- Reorder execution graph (panic_controller runs **after** EthicsFilter instead of before)

**Any of those changes produce a new cognitive hash.**

**Result**: New run, new identity, **not legally/experimentally the same agent**.

**That's a feature, not a bug.** It's how we make "do an ablation" an **explicit, reviewable act** instead of "I tweaked it a bit and ran five more hours overnight, trust me it's comparable".

**Framework principle**: Any BAC or UAC change creates fork (framework-level). The specific changes (panic thresholds vs machinery_stress thresholds, STEAL vs SHUTDOWN actions) are instance-specific, but the principle "config change → new hash → provable fork" is universal.

---

## 5.4 Why Resume Semantics Matter

**Three governance reasons.**

### 1. Long Training on Flaky Hardware

**Use case**: Training gets pre-empted at 3am due to hardware failure or scheduled maintenance.

**Framework guarantee**: You can resume later without inventing a "different" agent.

**Proof**: Same hash, same mind, same optimizer state, same RNG continuation.

**Benefit**: No "we think it's approximately the same agent but can't prove it."

**Framework pattern**: Hardware recovery (framework-level). Works for any universe instance running on any infrastructure.

---

### 2. Honest Ablations

**Use case**: Researcher wants to measure impact of social reasoning on survival rate.

**Framework enforcement**:
1. Resume from checkpoint
2. Edit snapshot: `social_model.enabled: true → false`
3. Framework detects change → new hash (`9af3c2e1 → 7bc4d5f3`)
4. Creates fork (new run folder with new hash)

**Scientific value**: You can state, "This is the same mind except the Social Model is disabled," and **substantiate it** with:
- Configuration diff (only `social_model.enabled` changed)
- New hash (proves distinct agent identity)
- Behavioral comparison (survival rate 45% vs 62%)

**Prevents**: "I tweaked some things overnight, behavior changed, not sure exactly what I modified."

**Framework pattern**: Ablation study protocol (framework-level). Whether ablating social_model (Townlet Town), risk_assessment (factory), or market_prediction (trading) doesn't matter - the explicit-fork pattern works universally.

---

### 3. Audit Trail

**Use case**: Safety auditor questions a decision: "Why did you let panic override normal reasoning here?"

**Framework answer**: We can show **exactly when that rule entered the snapshot**.

**Proof mechanism**:
- Checkpoint at step_000500: `panic_thresholds.energy: 0.15` (hash `9af3c2e1`)
- Checkpoint at step_010000: `panic_thresholds.energy: 0.10` (hash `7bc4d5f3`)
- Fork detected at resume timestamp `2025-11-03-13-40-09`
- Config diff shows threshold change
- Telemetry after fork uses new hash

**No "it drifted over time"**: Drift is now a **recorded fork** with timestamp, config diff, and new hash.

**Framework guarantee**: Configuration drift is impossible without creating detectable fork. Any change (BAC or UAC) in any universe instance creates new hash.

---

**Summary: Resume is now a governance primitive, not a convenience function.**

**The framework enforces**:
1. **Checkpoint snapshot is law** (restore from frozen snapshot, not live configs)
2. **Hash verification** (recompute hash to prove continuity or detect fork)
3. **Explicit forks** (any config change creates new hash, new run folder)
4. **Audit trail** (every fork is timestamped with config diff)

This transforms resume from "reload some weights and hope it works" into "provable continuity or explicit fork with governance visibility."

---
