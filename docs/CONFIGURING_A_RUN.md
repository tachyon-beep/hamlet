# Training Run Organization - Quick Reference

## ✅ What Changed

**Before:**

- Config, DB, and checkpoints scattered in project root
- Hard to track which files belong together
- Messy git status with training artifacts

**After:**

- Everything organized in `runs/` subdirectories
- Each run is self-contained and reproducible
- Clean project root, gitignored automatically

## 🚀 How to Start a Run

### Option 1: Helper Script (Recommended)

```bash
# Automatically creates: runs/L2_pomdp/{config.yaml, checkpoints/, metrics.db}
# Episodes configured in the YAML file (training.max_episodes)
python scripts/start_training_run.py L2_pomdp configs/townlet_level_2_pomdp.yaml
```

### Option 2: Manual Command

```bash
# You handle directory creation and read max_episodes from config
mkdir -p runs/my_run/checkpoints
python -m townlet.demo.runner \
    configs/townlet_level_2_pomdp.yaml \
    runs/my_run/metrics.db \
    runs/my_run/checkpoints \
    10000
```

## 📁 Directory Structure

```
runs/
├── README.md                  # Documentation
├── L2_pomdp/                  # Example run
│   ├── townlet_level_2_pomdp.yaml  # Config snapshot
│   ├── checkpoints/           # Model checkpoints
│   │   ├── checkpoint_ep00100.pt
│   │   ├── checkpoint_ep00200.pt
│   │   └── ...
│   └── metrics.db             # Training metrics
└── L2_5_temporal/             # Another run
    ├── townlet_level_2_5_temporal.yaml
    ├── checkpoints/
    └── metrics.db
```

## 🎯 For Your Current Validation

See `docs/TRAINING_LEVELS.md` for complete level specifications.

```bash
# Level 1: Full observability baseline
# Runs for 5000 episodes (configured in level_1_full_observability.yaml)
python scripts/start_training_run.py L1_baseline configs/level_1_full_observability.yaml

# Level 2: POMDP (tests LSTM memory fix from ACTION #9)
# Runs for 10000 episodes (configured in level_2_pomdp.yaml)
python scripts/start_training_run.py L2_validation configs/level_2_pomdp.yaml

# Level 3: Temporal mechanics
# Runs for 10000 episodes (configured in level_3_temporal.yaml)
python scripts/start_training_run.py L3_validation configs/level_3_temporal.yaml
```

## 📝 Git Status

- ✅ `runs/` is already in `.gitignore` (line 69)
- ✅ `*.db` files are already ignored (line 93)
- ✅ `*.pt` checkpoint files are already ignored (line 77)
- ✅ Only `runs/README.md` is tracked (documentation)

## 🧹 Cleanup Old Files

If you want to move existing DB files into runs:

```bash
# Create archive directory for old runs
mkdir -p runs/archive

# Move old DB files
mv demo_*.db metrics*.db runs/archive/ 2>/dev/null || true

# Clean up later when safe
# rm -rf runs/archive
```
