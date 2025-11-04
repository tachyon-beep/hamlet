## 5. Resume semantics

---

Resume operations must do more than reload weights; they are part of the audit chain.

If we can't prove continuity of mind across pauses, we can't claim continuity of behaviour for governance, and we can't do serious ablation science.

So we define resume like a forensic procedure.

### 5.1 The Rule: The Checkpoint Snapshot Is Law

When you resume from a checkpoint, you must restore from the checkpoint's own `config_snapshot/`, not from whatever is currently sitting in `configs/<run_name>/` in your working tree.

That means:

- You bring back the exact cognitive_topology.yaml from that checkpoint (same ethics, same panic thresholds, same greed sliders).
- You bring back the exact universe_as_code.yaml from that checkpoint (same ambulance cost, same bed effects, same wage rates).
- You bring back the exact execution_graph.yaml (same panic-then-ethics ordering).
- You bring back the optimiser state and RNG.

You do not "reconstruct" the agent from the latest code and hope it's approximately right. You rehydrate that specific mind in that specific world with that specific internal loop.

### 5.2 Where the Resumed Run Lives

Resuming from:

```text
runs/L99_AusterityNightshift__2025-11-03-12-14-22/checkpoints/step_000500/
```

creates a fresh new run folder, for example:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/
    config_snapshot/          # copied from the checkpoint, byte-for-byte
    checkpoints/
    telemetry/
    logs/
```

Important bits:

- We do not keep writing into the old run folder. New run, new timeline.
- We recompute the cognitive hash from the checkpoint snapshot. If you have not changed anything, the hash will match. That proves it's the same mind continuing.
- Telemetry in the resumed run now logs the same hash, so audit can say: "this is truly the same mind, same ethics, same world, just continued later".

### 5.3 Forking vs Continuing

Now the fun part.

If, before resuming, you edit that copied snapshot, even slightly, you are not continuing. You are forking.

Examples of forking:

- You lower `panic_thresholds.energy` from 0.15 to 0.05 so it doesn't bug out early.
- You turn off `social_model.enabled`.
- You remove `"steal"` from `forbid_actions`.
- You change ambulance cost in `universe_as_code.yaml`.
- You reorder the execution graph so panic_controller runs after EthicsFilter instead of before.

Any of those changes produce a new cognitive hash.

Result: new run, new identity, not legally/experimentally the same agent.

That's a feature, not a bug. It's how we make "do an ablation" an explicit, reviewable act instead of "I tweaked it a bit and ran five more hours overnight, trust me it's comparable".

### 5.4 Why Resume Semantics Matter

Three reasons.

1. Long training on flaky hardware
   If training gets pre-empted at 3 am, you can resume later without inventing a "different" agent. Same hash, same mind, same optimiser, same RNG continuation.

2. Honest ablations
   You can state, "this is the same mind except the Social Model is disabled," and substantiate it with the configuration diff plus the new hash. Behavioural comparisons remain well-defined.

3. Audit trail
   If someone questions a safety decision ("why did you let panic override normal reasoning here?"), you can show exactly when that rule entered the snapshot. There's no "it drifted over time"; drift is now a recorded fork.

Resume is now a governance primitive, not a convenience function.

---
