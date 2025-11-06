# Variable and Feature System

Treat the Variable & Feature System (VFS) — i.e., custom variables, derived features, privacy, lifetimes, and observation exposure — as an **intrinsic subsystem of UAC**. It defines *universe semantics* (what exists, who can see it, who can write it, and for how long). That sits squarely in UAC’s remit, just like affordances, rewards, time, and agent lifecycle.

Then:

- UAC(VFS) → emits variables/features + which are exposed as observations
- ACS → consumes those declarations to compile masks/constraints and action read/write sets
- Engine → hosts the runtime store and executes effects deterministically
- BAC → binds to the exposed observations and action index

# Why not a separate top-level service?

**Cons**

- **Semantic split brain.** “What the universe *is*” ends up in two owners (UAC and a new service). You’ll fight consistency bugs (who wins on schema, access control, privacy?).
- **Versioning pain.** You’d need yet another IR and compatibility matrix; every change requires tri-version coordination (UAC/VFS/ACS).
- **Latency & complexity.** If masks/validation depend on variable reads, an extra hop for VFS state adds tail latency or forces cache duplication in ACS/Engine.
- **Safety drift.** Admission and policy checks become cross-service. Easier to miss that an action writes a private var it shouldn’t.

**Pros (when separate can make sense)**

- If you want a **shared cross-universe blackboard** with independent scaling and storage backends (e.g., long-horizon persistence across many experiments), a separate “state service” can help — but that’s a niche need, and you can still back UAC’s VFS with a pluggable store later.

Net: the downsides outweigh the upsides at your current stage.

# Build order (what depends on what)

**Stratum → UAC(VFS) → ACS → BAC → Engine/Orchestrator**

- **Stratum** must run first to define substrate references VFS expressions may use.
- **UAC(VFS)** compiles variables, features, privacy, lifetimes, observation exposure, and action effects that read/write them.
- **ACS** compiles masks/constraints using VFS declarations (dependency indexing for incremental recompute).
- **BAC** binds inputs (exposed observations) and outputs (actions.idx).
- **Engine** loads all artefacts and hosts the runtime variable store.

# Minimum contracts to add in UAC (so ACS/BAC/Engine stay happy)

- **UIRM additions**: `variables[]`, `features[]`, `exposed_observations[]`, each with scope, lifetime, type, access rules.
- **Actions**: per-action `reads[]`/`writes[]` sets and declarative effects (the same place you already declare affordances).
- **Observation spec**: a selector over variables/features with privacy baked in.

ACS then treats these as **inputs** (not owners) and emits:

- `constraints.cns.json` with variable-scoped predicate kernels
- `actions.asm.yaml` with dependency metadata
- `actions.idx.json` with shapes/factors

# If you still prefer “separate system”, when and how?

If you *must* split it out for organisational reasons, do it as a **UAC-adjacent sub-compiler** that:

- lives in the UAC tree,
- shares the UIRM version,
- runs **immediately after** UAC parsing and **before** ACS,
- emits a **single, merged UIRM** (UAC+VFS) so downstream consumers still see one contract.

Don’t put it **after** ACS — ACS needs VFS to compile masks and read/write sets correctly.

# Migration note

- Start by expressing your current “teleport home”, stacks/queues, and similar patterns as UAC(VFS) declarations.
- Keep the Engine’s existing ad-hoc fields, but map them 1:1 to VFS variables; delete the ad-hoc paths once parity is green.
- Add admission checks in the Control Plane for variable access (privacy, lifetimes, write permissions) alongside your current UAC validators.

# TL;DR

- **Ownership:** VFS belongs **in UAC**.
- **Order:** Run **before ACS**.
- **Effect:** One source of truth for universe state; ACS and Engine become faster, safer, and simpler; BAC just binds to shapes.

=====

# A) TASK-003B — Strict schemas + no defaults (the contract freeze)

## A1. Schema registry and versioning

- **One registry, many IRs.** Create a single `schemas/` folder containing versioned JSON Schemas for:

  - `SIRM` (Stratum IR)
  - `UIRM` (Universe IR; includes Variables & Features, observations, action effects/contracts)
  - `ASM` (Action Space Manifest; ACS outputs)
  - `BIRM` (Brain IR; BAC outputs)
  - `OBSV`, `RWD`, `CNS` (observations, rewards, constraints)
  - `RUN-MANIFEST` (control plane provenance)
- **Semantic versions.** Start at `0.9.0` for UIRM and `0.7.0` for ASM. Bump **minor** on additive fields; bump **major** on behaviour-affecting changes.
- **Compatibility matrix.** A small `schemas/compatibility.yaml` that declares permitted tuples, e.g.:

  ```yaml
  matrix:
    - uirm: ">=0.9.0 <1.0.0"
      asm:  ">=0.7.0 <0.8.0"
      sirm: ">=1.3.0 <2.0.0"
      engine: ">=1.12.0"
      acs: ">=0.8.0"
  ```

  The **Admission Controller** reads this.

## A2. No-defaults principle, enforced

- **Rule:** every behavioural parameter must be explicit in config or compiler output. Accept **zero** behavioural defaults in code.
- **Mechanics:**

  - Pydantic models: `model_config = {"extra":"forbid"}` and **no default** values for behaviour fields (use `...` required).
  - JSON Schema: mark fields `"default"` only for *non-behavioural* metadata (e.g., labels), never for gameplay/economics/physics.
  - Admission: reject any compiled IR that’s missing required fields; reject any runtime attempt to load a non-signed, non-validated artefact.

## A3. Strict typing and units

- **Numeric units.** Where a number represents a quantity, define a unit or scale (e.g., “ticks”, “normalised [0..1]”). Include a `unit` or `scale` field in schemas.
- **Enums over strings.** Any “mode”/“type” is an enum in schema; no free-text sentinel values.
- **Shapes over vibes.** Observations specify `shape`, `dtype`, `packing` (flat/struct), and `normalisation` rules.
- **IDs are canonical.** `id` fields are `^[a-zA-Z][A-Za-z0-9_]{2,63}$`; stable across runs; no spaces.

## A4. Deterministic ordering and hashing

- All compiler outputs must:

  - Sort maps by key before serialisation.
  - Emit stable IDs and indices (e.g., action tensor slots).
  - Include `content_hash` (blake3 of the canonical serialisation) in the IR header.

## A5. Schema slices you’ll likely need (illustrative)

**UIRM core (JSON Schema excerpt)** — includes Variables & Features (VFS) natively:

```json
{
  "$id": "uirm-0.9.0.json",
  "type": "object",
  "required": ["version","time","lifecycle","variables","features","observations","actions"],
  "properties": {
    "version": {"const":"0.9.0"},
    "time": {"type":"object","required":["tickHz"],"properties":{"tickHz":{"type":"integer","minimum":1}}},
    "lifecycle": {"type":"object","required":["agent"],"properties":{"agent":{"type":"object","required":["spawn","retire"],"properties":{"spawn":{"type":"string"},"retire":{"type":"string"}}}}},
    "variables": {
      "type":"array",
      "items": {
        "type":"object",
        "required":["id","scope","type","lifetime","readable_by","writable_by"],
        "properties":{
          "id":{"type":"string","pattern":"^[A-Za-z][A-Za-z0-9_]{2,63}$"},
          "scope":{"enum":["global","stratum","entity","agent","agent_private","episode"]},
          "type":{"enum":["bool","i32","i64","f32","f64","vec2i","vec3i","tensor","enum","stack<i8>","stack<f32>"]},
          "lifetime":{"enum":["tick","episode","run","persistent"]},
          "readable_by":{"type":"array","items":{"enum":["engine","acs","agent","none"]}},
          "writable_by":{"type":"array","items":{"enum":["engine","actions","none"]}},
          "default": {}
        }
      }
    },
    "features": {
      "type":"array",
      "items": {"type":"object","required":["id","type","lifetime","expr"],"properties":{
        "id":{"type":"string","pattern":"^[A-Za-z][A-Za-z0-9_]{2,63}$"},
        "type":{"enum":["bool","i32","f32"]},
        "lifetime":{"enum":["tick"]},
        "expr":{"type":"string"}
      }}
    },
    "observations": {
      "type":"array",
      "items":{"type":"object","required":["id","source","to","shape","dtype"],"properties":{
        "id":{"type":"string"},
        "source":{"type":"string"},  // variable or feature id
        "to":{"type":"array","items":{"enum":["agent","evaluator"]}},
        "shape":{"type":"array","items":{"type":"integer","minimum":-1}},
        "dtype":{"enum":["bool","i32","f32"]},
        "normalisation":{"type":"object"}
      }}
    },
    "actions": {
      "type":"array",
      "items":{"type":"object","required":["id","params","pre","effect"],"properties":{
        "id":{"type":"string"},
        "params":{"type":"array"},
        "pre":{"type":"array","items":{"type":"string"}},
        "effect":{"type":"array","items":{"type":"object"}},
        "reads":{"type":"array","items":{"type":"string"}},
        "writes":{"type":"array","items":{"type":"string"}}
      }}
    }
  }
}
```

**ASM (ACS output) essentials:**

- `actions[]` with `tensor_slot`, `mask_rule_id`, `reads[]`, `writes[]`
- `param_families[]` (for mixed/parametric actions)
- `dependencies{var_id:[rule_ids...]}` for incremental masks

**BIRM (BAC output) essentials:**

- `inputs[]` with `sourceRef` and resolved `shape/dtype`
- `heads[]` with `actionRef` ⇢ `tensor_slot(s)`
- No learnable hyperparams defaulted; all explicit

## A6. “Rip the defaults out” checklist (where to look)

Kill any behaviour implied by code when a field is missing. Common culprits:

- **Observation dim calc**: must derive from UIRM OBSV spec; remove baked values like “+ 8 meters”.
- **Action count**: cannot be 6 by default; must come from ASM/IDX.
- **Meter dynamics**: base depletions/cascades are data; no silent fallbacks.
- **Operating hours**: never “24/7” by default unless explicitly configured.
- **Cooldowns**: zero unless declared.
- **Partial vs full observability**: explicit. No “if vision_range missing, assume full grid”.
- **Random seeds**: declared in run manifest; engine never invents one.

## A7. Admission rules (control plane)

On `Admit`, validate:

- JSON Schema compliance (all IRs)
- Compatibility matrix satisfied
- No use of deprecated fields (and no auto-migration at **admission**; migrations only happen **at compile time** with warnings)
- Privacy rules (`agent_private` exposed only to that agent)
- `reads/writes` sets are legal against `readable_by/writable_by`
- All observation shapes are concrete (no -1 except batch dim at runtime)

## A8. Developer ergonomics

- **`make validate`**: runs JSON Schema validation on all IRs in a config pack.
- **`make nuke-defaults`**: lints repo for disallowed defaults in Pydantic models and engine paths (simple AST check).
- **Golden samples**: `samples/` with tiny universes that compile and run under CI; one sample per tricky construct (multi-tick, cooldown, privacy).

---

# B) TASK-004A — Universe Compiler implementation (how it uses strict schemas)

Hook your excellent plan into the stricter world:

## B1. Stage-by-stage deltas

- **Stage 1 (Parse)**
  Parse YAML → Pydantic DTOs → **immediately** validate against JSON Schema (yes, both: Pydantic for Python type comfort; JSON Schema for language-agnostic contracts). Fail if either rejects.
- **Stage 2 (Symbols)**
  Register *everything addressable*: meters, affordances, variables, features, actions, observation IDs.
- **Stage 3 (Reference resolution)**
  Resolve all cross-refs (including VFS sources, feature dependency graph). Enforce access control (no action writes `agent_private` unless scope allows).
- **Stage 4 (Cross-validation)**
  Keep your checks (spatial, economic, circularity, temporal) and add:

  - **Privacy**: no observation exposes `agent_private` to other agents.
  - **VFS lifetimes**: any `tick`-lifetime feature cannot be used as a persistent write target.
  - **Reads/writes closure**: every action’s `reads[]/writes[]` is minimal and sound (no undeclared side effects).
- **Stage 5 (Metadata)**
  Observation dim is computed **only** from the OBSV section (count fields, flatten shapes when `packing=flat`). Action count comes **only** from ASM/IDX (if you keep a temporary bridge, guard it with a hard warning and a kill-switch flag).
- **Stage 6 (Optimise)**
  Build the feature dependency DAG; emit topo-ordered kernels; pre-index ACS constraints by variable ID to drive incremental masks.
- **Stage 7 (Emit)**
  Emit **UIRM** and **OBSV/RWD** with version pins; sign them; write `content_hash`.

## B2. Outputs & signatures

- Every compiler stage emits a `stage_report` (counts, timings, warnings) and the IR chunk it produced **plus** a `content_hash`. The final `CompiledUniverse` includes:

  - `uirm.json`, `obsv.yaml`, `rwd.yaml`
  - `indexes/` folder with maps (`name ↔ index`)
  - `meta.json` (compiler version, wall-clock, seed hints)
  - `content_hash` at root

## B3. Zero-default enforcement

At **emit** time, run a “default audit” that asserts:

- No prohibited fields carry defaults
- Every numeric under a behavioural key exists
- No missing “hours”, “cooldown_ticks”, etc.

If audit fails, **hard error** with a pointer to the field path.

---

# C) TASK-004B — Capabilities, masking, pipelines (aligned with schemas)

Your DTOs are on point. A few glue rules so 004B remains schema-tight:

## C1. Capability composition rules (compiler-enforced)

- Mutual exclusions: `instant` vs `multi_tick`, or any pair you’ve declared in a `capability_rules.yaml`.
- Dependencies: `resumable` ⇒ `multi_tick`; `probabilistic` ⇒ both `on_completion` and `on_failure` effects present.
- Operating hours wrap: allowed, but pre-expand into a 24-bit mask table at compile time for ACS/Engine.

## C2. Availability and policy masks

- `availability[]` is **purely declarative**; ACS compiles it to `CNS` kernels.
- Curriculum/policy packs live in ASM as **overlays**. The Control Plane may flip overlays per stage (no schema mutation mid-run).

## C3. Legacy migration

- Auto-migrate `effects{}` → `effect_pipeline.on_completion[]` **at compile time** with a single warning. Store a `migration_log` in the compiled pack for audit.

---

# D) Concrete deliverables (checklist you can track in Jira)

## D1. Repos & files

- `schemas/` with `uirm-0.9.0.json`, `asm-0.7.0.json`, `birm-0.6.0.json`, `sirm-1.3.0.json`, `obs-0.4.0.json`, `rwd-0.4.0.json`, `cns-0.3.0.json`, `run-manifest-1.0.0.json`, `compatibility.yaml`
- `tools/schema_check.py` (validates IRs and prints diffs)
- `townlet/universe/compiler/` (stages, reports, audit)
- `townlet/acs/` (reads/writes dependency index, mask kernels)
- `townlet/engine/` (load/store for variables, topo feature kernels)
- `docs/contracts/*.md` (one pager per IR with examples)

## D2. Lints & gates

- CI job: “**Schema Gate**” — reject PR if any IR sample or compiler output fails JSON Schema or default audit.
- CI job: “**Golden Parity**” — Hamlet parity runs (fixed seed) pass within tolerance, step-time p95 below target.

## D3. Acceptance tests (minimum)

- Compile fails on: missing hours; undefined meter refs; privacy breach; action writes to read-only var; cyclical features; missing observation shapes; undeclared action reads/writes.
- Compile succeeds on: tiny VFS demo; multi-tick + cooldown; probabilistic with failure path; mode-switch hours wrap.
- Runtime: ACS mask recompute p95 < target; Engine step p95 < target; determinism across two launches with same run manifest.

---

# E) Order of operations (practical day-by-day)

**Day 1–2**

- Land schema registry + compatibility matrix + validators.
- Wire Admission to call the validators (even before 004A is finished).
- Add the default-audit pass (red bar until green).

**Day 3–4**

- Update 004A Stage 1–3 to emit schema-compliant chunks; turn on “extra: forbid” in DTOs; bulk-remove code defaults.

**Day 5–6**

- Finish Stage 4–7 with feature DAG + ACS dependency index; sign artefacts; produce `content_hash`.

**Day 7–8**

- Parity + perf baselines, lock SLOs, enable control-plane gates for schema versions.

---

# F) Red flags to watch (so you can head them off)

- **Implicit 6-action bias** lurking in old training loops — make the Orchestrator consume `get_spaces()` only.
- **Observation packing drift** between compiler and engine — treat OBSV as **source of truth** for order and normalisation.
- **“Helpful” defaults sneaking back** via convenience constructors — forbid in lint; allow only in tests with explicit override.

---

# G) What “done” looks like

- A config pack that previously “worked by accident” will now **fail at compile** with precise errors and hints.
- The engine and trainer have **zero** knowledge of meters, hours, cooldowns, action counts — they read them from IRs.
- The Control Plane can safely **canary** new UIRM/ASM versions because schemas are explicit, signed, and compatibility-checked.
