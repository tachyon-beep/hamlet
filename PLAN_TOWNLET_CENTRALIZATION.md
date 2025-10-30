# Townlet Centralization Plan

## Current State Analysis

### Problem
The codebase is split between two systems:
1. **hamlet** (src/hamlet/) - OBSOLETE legacy training system
2. **townlet** (src/townlet/) - CURRENT GPU-native vectorized system

This creates confusion because:
- Entry point (`runner.py`) lives in `src/hamlet/demo/` but orchestrates townlet
- Documentation inconsistently refers to both systems
- It's unclear which code is active vs obsolete

### Current Structure

**Townlet (CURRENT - Keep)**:
```
src/townlet/
├── agent/
│   ├── networks.py          # SimpleQNetwork, RecurrentSpatialQNetwork
│   └── __init__.py
├── environment/
│   ├── vectorized_env.py    # VectorizedHamletEnv (GPU-native)
│   └── __init__.py
├── population/
│   ├── vectorized.py        # VectorizedPopulation (batched training)
│   ├── base.py
│   └── __init__.py
├── curriculum/
│   ├── adversarial.py       # AdversarialCurriculum
│   ├── static.py
│   ├── base.py
│   └── __init__.py
├── exploration/
│   ├── adaptive_intrinsic.py # AdaptiveIntrinsicExploration + RND
│   ├── rnd.py
│   ├── epsilon_greedy.py
│   ├── base.py
│   └── __init__.py
└── training/
    ├── state.py             # BatchedAgentState, PopulationCheckpoint
    ├── replay_buffer.py     # ReplayBuffer
    └── __init__.py
```

**Hamlet (OBSOLETE - But contains needed components)**:
```
src/hamlet/
├── demo/
│   ├── runner.py            # ← ENTRY POINT (orchestrates townlet!)
│   ├── database.py          # ← NEEDED (stores training state)
│   ├── live_inference.py    # ← NEEDED (inference server)
│   ├── viz_server.py        # ← NEEDED (visualization server)
│   └── snapshot_daemon.py   # ← NEEDED (periodic snapshots)
├── environment/
│   ├── hamlet_env.py        # OBSOLETE (replaced by townlet vectorized)
│   ├── entities.py          # OBSOLETE
│   ├── meters.py            # OBSOLETE
│   ├── affordances.py       # OBSOLETE
│   └── ...
├── agent/
│   ├── drl_agent.py         # OBSOLETE (replaced by townlet population)
│   ├── networks.py          # OBSOLETE (replaced by townlet agent/networks)
│   └── ...
└── training/
    ├── trainer.py           # OBSOLETE (replaced by townlet runner)
    ├── config.py            # OBSOLETE
    └── ...
```

---

## Centralization Plan

### Phase 1: Move Active Components to Townlet

**Move these from `src/hamlet/demo/` → `src/townlet/`**:

1. **runner.py** → `src/townlet/training/runner.py`
   - Main training entry point
   - Orchestrates: VectorizedHamletEnv, VectorizedPopulation, AdversarialCurriculum, AdaptiveIntrinsicExploration
   - Updates: Change all imports, update CLI invocation

2. **database.py** → `src/townlet/training/database.py`
   - SQLite database for training state
   - Used by runner for checkpointing

3. **live_inference.py** → `src/townlet/inference/server.py`
   - WebSocket inference server
   - Loads checkpoints, runs trained agents
   - Streams state to frontend

4. **viz_server.py** → `src/townlet/visualization/server.py`
   - Visualization server for live training
   - WebSocket + FastAPI

5. **snapshot_daemon.py** → `src/townlet/training/snapshot_daemon.py`
   - Periodic checkpoint snapshots

**New structure**:
```
src/townlet/
├── agent/              [existing]
├── environment/        [existing]
├── population/         [existing]
├── curriculum/         [existing]
├── exploration/        [existing]
├── training/
│   ├── state.py        [existing]
│   ├── replay_buffer.py [existing]
│   ├── runner.py       [NEW - from hamlet/demo/runner.py]
│   ├── database.py     [NEW - from hamlet/demo/database.py]
│   ├── snapshot_daemon.py [NEW - from hamlet/demo/snapshot_daemon.py]
│   └── __init__.py
├── inference/
│   ├── server.py       [NEW - from hamlet/demo/live_inference.py]
│   └── __init__.py
└── visualization/
    ├── server.py       [NEW - from hamlet/demo/viz_server.py]
    └── __init__.py
```

### Phase 2: Update Entry Points

**Old command**:
```bash
python -m hamlet.demo.runner configs/townlet_level_2_pomdp.yaml demo.db checkpoints 10000
```

**New command**:
```bash
python -m townlet.training.runner configs/townlet_level_2_pomdp.yaml demo.db checkpoints 10000
```

**Or create a top-level script**:
```bash
uv run townlet-train configs/townlet_level_2_pomdp.yaml demo.db checkpoints 10000
```

### Phase 3: Delete Obsolete Hamlet Code

**Delete all obsolete hamlet code immediately after moving active files:**

```bash
# After moving active files from hamlet/demo/, delete obsolete directories
git rm -r src/hamlet/environment
git rm -r src/hamlet/agent
git rm -r src/hamlet/training
git rm -r src/hamlet/demo  # After moving runner.py, database.py, etc.
git rm -r src/hamlet/web    # If not used by frontend

# Commit deletion
git commit -m "Delete obsolete hamlet training system

All functionality moved to townlet GPU-native system.
Legacy code removed to prevent confusion.
History preserved in git."
```

**Rationale**: Code is in git history, no need to keep obsolete code around.

### Phase 4: Update All Documentation

**Files to update**:

1. **CLAUDE.md** (most critical - guides AI behavior)
   - Remove all hamlet references
   - Replace with townlet architecture
   - Update command examples
   - Update file paths

2. **README.md**
   - Update quick start
   - Change entry point commands
   - Update architecture diagram

3. **docs/ARCHITECTURE_DESIGN.md**
   - Clarify townlet is the active system
   - Update Level 2 POMDP implementation status

4. **docs/townlet/*.md**
   - Update paths in all phase verification docs
   - Consolidate with main docs

5. **LEVEL_2_TRAINING.md**
   - Update commands to use townlet entry point
   - Update file paths

---

## Execution Steps

### Step 1: Create townlet subdirectories
```bash
mkdir -p src/townlet/inference
mkdir -p src/townlet/visualization
```

### Step 2: Move files with git mv (preserves history)
```bash
git mv src/hamlet/demo/runner.py src/townlet/training/runner.py
git mv src/hamlet/demo/database.py src/townlet/training/database.py
git mv src/hamlet/demo/snapshot_daemon.py src/townlet/training/snapshot_daemon.py
git mv src/hamlet/demo/live_inference.py src/townlet/inference/server.py
git mv src/hamlet/demo/viz_server.py src/townlet/visualization/server.py
```

### Step 3: Update imports in moved files
```python
# In src/townlet/training/runner.py
# OLD:
from hamlet.demo.database import DemoDatabase

# NEW:
from townlet.training.database import DemoDatabase
```

### Step 4: Update __init__.py files
Add exports to `src/townlet/training/__init__.py`:
```python
from townlet.training.runner import DemoRunner
from townlet.training.database import DemoDatabase
```

### Step 5: Create CLI entry point (optional)
In `pyproject.toml`:
```toml
[project.scripts]
townlet-train = "townlet.training.runner:main"
townlet-infer = "townlet.inference.server:main"
```

### Step 6: Update CLAUDE.md
Complete rewrite focusing on townlet as the only active system.

### Step 7: Update all other docs
Batch update file paths and commands.

### Step 8: Delete obsolete hamlet code
```bash
git rm -r src/hamlet/environment
git rm -r src/hamlet/agent
git rm -r src/hamlet/training
git rm -r src/hamlet/demo  # After moving active files
```

Keep only:
- `src/hamlet/web/` (if still used by frontend)
- Frontend directory (separate concern)

---

## Documentation Update Priority

### Critical (Do First):
1. **CLAUDE.md** - AI guidance system (prevents future mistakes)
2. **README.md** - User entry point
3. **LEVEL_2_TRAINING.md** - Immediate training guide

### Important (Do Second):
4. **docs/ARCHITECTURE_DESIGN.md** - System overview
5. **docs/townlet/*.md** - Consolidate phase docs
6. **docs/TRAINING_SYSTEM.md** - Update paths

### Nice to Have (Do Third):
7. **docs/scraps/*.md** - Historical notes (low priority)
8. **Other docs/** - Update as needed

---

## Benefits

1. **Clear entry point**: `python -m townlet.training.runner`
2. **No confusion**: Only one system exists
3. **Clean imports**: All townlet imports
4. **Better organization**: Logical directory structure
5. **AI won't get confused**: CLAUDE.md clearly states townlet is the only system

---

## Risks & Mitigation

**Risk**: Breaking existing commands
**Mitigation**: Update all docs simultaneously, provide migration guide

**Risk**: Breaking frontend connection
**Mitigation**: Test WebSocket servers after move, update frontend config if needed

**Risk**: Lost git history
**Mitigation**: Use `git mv` (preserves history), don't delete until after move

---

## Timeline

- **Phase 1** (Move files): 30 minutes
- **Phase 2** (Update imports): 30 minutes
- **Phase 3** (Mark obsolete): 5 minutes
- **Phase 4** (Update docs): 1-2 hours

**Total**: ~3 hours of careful work

---

## Success Criteria

✅ All training runs via: `python -m townlet.training.runner`
✅ No imports from `hamlet` (except web server if needed)
✅ CLAUDE.md only references townlet
✅ README shows correct commands
✅ All tests pass
✅ Frontend still connects
✅ Inference server works

---

## Next Actions

1. **Immediate**: Update CLAUDE.md to clearly mark hamlet as obsolete
2. **Short-term**: Execute Phase 1 (move files)
3. **Medium-term**: Execute Phases 2-4 (update all references)
4. **Long-term**: Delete obsolete hamlet code entirely
