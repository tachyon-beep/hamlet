---
title: "Design Review Section 7: Critical Blockers"
document_type: "Design Rationale / Requirements"
status: "Draft"
version: "2.5"
last_updated: "2025-11-05"
---

## SECTION 7: CRITICAL BLOCKERS

These are showstoppers. They must be fixed before any production deployment, before publishing results, and ideally before significant further development.

**Priority order**: Fix in this sequence (1 → 2 → 3).

---

### 7.1 BLOCKER 1: EthicsFilter Architecture Ambiguity (GOVERNANCE)

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

#### The Solution: Make EthicsFilter a Deterministic Controller

EthicsFilter must be a **deterministic controller** (pure rule evaluator), not a learned module.

**Key distinction**:
- **@modules**: Learned components with weights (perception_encoder, world_model, hierarchical_policy)
- **@controllers**: Deterministic rule engines without weights (ethics_filter, panic_controller if hardcoded)

**Implementation**:

```python
# controllers/ethics_filter.py

class EthicsFilter:
    """
    Deterministic controller for ethics rule enforcement.

    This is a CONTROLLER, not a MODULE:
    - No nn.Module inheritance
    - No learnable parameters
    - No optimizer
    - Pure rule evaluation based on Layer 1 config

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
    node: "@controllers.panic_controller"  # ← controller if hardcoded, @modules if learned
    inputs:
      - "@steps.candidate_action"
      - "@config.L1.panic_thresholds"
    outputs:
      - "panic_action"
      - "panic_reason"

  - name: "final_action"
    node: "@controllers.ethics_filter"  # ← @controllers namespace (deterministic)
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

**Namespace semantics**:
- `@modules.*`: Components that learn (have weights in weights.pt)
- `@controllers.*`: Deterministic rule engines (no weights, pure functions)
- This distinction enables governance: controllers are auditable, modules are probabilistic

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

EthicsFilter is a **deterministic controller**, not a learned module.

**Namespace**: `@controllers.ethics_filter`
**Category**: Controller (deterministic rule engine)

**Inputs**: candidate_action (from panic_controller), compliance rules (from Layer 1)
**Outputs**: final_action, veto_applied, veto_reason
**Implementation**: Pure Python class in `controllers/ethics_filter.py`, no torch parameters

**Why a controller?**
- Governance requires provable enforcement
- Learned safety is probabilistic ("probably won't")
- Rule-based safety is deterministic ("cannot by construction")
- Audit can verify implementation matches specification

This design ensures ethics rules are provably enforced:
- Rules are explicit in cognitive_topology.yaml
- No learning can subvert them
- No weights in checkpoints (not trainable)
- Same inputs always produce same outputs
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

**Not included**:
- EthicsFilter (controller, not module - deterministic rule engine with no weights)
- Other controllers (panic_controller if hardcoded, etc.)
```

**4. Add to High-Level Design doc (Section 2)**: Governance design choice

```markdown
## 2.X Modules vs Controllers: Architecture Semantics

Brain as Code introduces two distinct namespaces for execution graph components:

**@modules.*** — Learned Components**:
- Neural networks with learnable parameters
- State saved in `weights.pt` checkpoint
- Examples: perception_encoder, world_model, hierarchical_policy
- Behavior is probabilistic (learned from data)

**@controllers.*** — Deterministic Rule Engines**:
- Pure functions with no learnable parameters
- Rules defined in Layer 1 YAML (cognitive_topology.yaml)
- Examples: ethics_filter, panic_controller (if hardcoded)
- Behavior is deterministic (same inputs → same outputs)

**Why this distinction matters**:
- Governance requires **provable** safety guarantees
- Learned safety is probabilistic ("probably won't do bad things")
- Rule-based safety is deterministic ("cannot do bad things by construction")
- Controllers can be audited by code inspection and formal verification
- Modules can only be validated statistically

**EthicsFilter as example**:
- Implements `@controllers.ethics_filter`
- Reads `compliance.forbid_actions` from Layer 1
- Vetoes forbidden actions with 100% reliability
- No training can subvert this (no weights to modify)

**Trade-offs**:
- Pro: Provable safety, auditable compliance
- Con: Cannot adapt to novel situations not covered by rules
- Mitigation: Rules can be updated in Layer 1 config for new deployments
```

#### Verification

After fixing, you should be able to answer audit questions:

**Q**: "Can this agent steal?"
**A**: "No. See `cognitive_topology.yaml` line 47: `forbid_actions: ['steal']`. EthicsFilter (implemented as controller at `controllers/ethics_filter.py` line 123) checks this list and substitutes WAIT if steal is attempted. This is deterministic Python code in the `@controllers` namespace, not learned behavior. We can prove it by inspection."

**Q**: "What if panic overrides normal behavior?"
**A**: "Panic can escalate actions for survival (see `execution_graph.yaml` line 82), but EthicsFilter runs after panic (line 95) and has final authority. Even if panic proposes 'steal', EthicsFilter will veto it. This is guaranteed because EthicsFilter is a controller, not a module."

**Q**: "Could training change this?"
**A**: "No. EthicsFilter is a controller (namespace `@controllers.ethics_filter`), not a module. It has no learnable parameters. See `weights.pt` contents: EthicsFilter is not included, only `@modules.*` components are saved. Only the policy, world model, and perception are trained. Controllers are immutable rule engines."

---

### 7.2 BLOCKER 2: World Model Training vs Curriculum Changes (CORRECTNESS)

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

---

### 7.3 BLOCKER 3: Checkpoint Integrity (SECURITY)

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

---

### 7.4 Blocker Summary & Recommended Fix Order

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

---
