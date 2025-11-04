# Multi-Day Tech Demo Design (Phase 3.5)

> **Date:** 2025-10-30
>
> **Status:** Design Complete, Ready for Implementation
>
> **Goal:** Validate Phase 3 system over 48+ hours, generate teaching materials, stream to YouTube

---

## Executive Summary

Phase 3.5 runs a multi-day demonstration of the Hamlet DRL system to validate stability, capture exploration→exploitation transitions, and generate rich teaching materials. The system will run unattended for 2-3 days, streaming live to remote viewers while capturing data for post-analysis.

**Key Principles:**

- **For but not with containers** - Design for easy containerization but run as simple processes
- **SQLite as single source of truth** - Shared state, no complex orchestration
- **Minimal viable features** - Ship working demo, avoid overengineering
- **Teaching-focused outputs** - Multi-layer visualizations showing emergent behaviors

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Multi-Day Demo System                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │  Training Process │────────▶│   Shared State DB    │  │
│  │  (demo_runner.py) │         │   (SQLite)           │  │
│  │                   │         │                       │  │
│  │ • 10K episodes    │         │ • Episode metrics    │  │
│  │ • Checkpoint/100  │         │ • Position heatmaps  │  │
│  │ • Auto-resume     │         │ • Affordance visits  │  │
│  │ • systemd restart │         │ • System state       │  │
│  └──────────────────┘         └──────────┬───────────┘  │
│           │                              │               │
│           │                              ▼               │
│           │                  ┌──────────────────────┐    │
│           │                  │  Visualization Server │    │
│           │                  │  (viz_server.py)      │    │
│           │                  │                       │    │
│           │                  │ • WebSocket :8765     │    │
│           │                  │ • Serve frontend      │    │
│           │                  │ • Stream SQLite data  │    │
│           │                  └──────────┬───────────┘    │
│           │                             │                │
│           ▼                             ▼                │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │  Checkpoint Files │         │  Remote Viewers      │  │
│  │  (checkpoints/)   │         │  (YouTube/Browser)   │  │
│  └──────────────────┘         └──────────────────────┘  │
│           │                                              │
│           ▼                                              │
│  ┌──────────────────────────────────────┐               │
│  │  Snapshot Daemon (snapshot_daemon.py) │               │
│  │  • Screenshots every 5 min            │               │
│  │  • Heatmap GIFs every 50 episodes     │               │
│  │  • CSV exports hourly                 │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Training Runner (`demo_runner.py`)

**Purpose:** Orchestrates 10K episode training with checkpointing, metrics logging, and auto-recovery.

**Interface (Containerization-Ready):**

- **Input:** `HAMLET_CONFIG` env var → `configs/townlet/sparse_adaptive.yaml`
- **Output:** Checkpoints to `HAMLET_CHECKPOINT_DIR`, metrics to `HAMLET_DB_PATH`
- **Ports:** None (write-only)
- **State:** SQLite at `$HAMLET_DB_PATH/demo_state.db`

**Features:**

```python
class DemoRunner:
    def __init__(self, config_path: str, db_path: str, checkpoint_dir: str):
        """Initialize from config, load checkpoint if exists."""

    def run(self):
        """Main training loop with auto-recovery."""
        # Load latest checkpoint or start fresh
        # For each episode:
        #   - Run training
        #   - Write metrics to SQLite
        #   - Checkpoint every 100 episodes
        #   - Heartbeat log every 10 episodes
        #   - At episode 5000: randomize affordances (generalization test)
        # Graceful shutdown on SIGTERM

    def save_checkpoint(self, episode: int):
        """Save PyTorch checkpoint with episode number."""

    def load_checkpoint(self) -> Optional[int]:
        """Load latest checkpoint, return episode number or None."""
```

**Metrics Written Per Episode:**

- `episode_id`, `timestamp`, `survival_time`
- `total_reward`, `extrinsic_reward`, `intrinsic_reward`
- `intrinsic_weight`, `curriculum_stage`, `epsilon`

**Checkpoints:**

- Filename: `checkpoint_ep{episode:05d}.pt`
- Contents: Q-network, target network, optimizer, RND predictor, exploration state, episode number
- Frequency: Every 100 episodes
- Retention: Keep all (disk is cheap, enables time-travel debugging)

**Logging:**

```
[2025-10-30 14:23:45] Episode 4230/10000 | Survival: 156 steps | Reward: 42.3 | Intrinsic Weight: 0.34 | Stage: 4/5
[2025-10-30 14:23:50] Checkpoint saved: checkpoint_ep04200.pt
[2025-10-30 15:00:00] GENERALIZATION TEST: Randomizing affordance positions at episode 5000
```

---

### 2. Shared State Database (`demo_state.db`)

**Purpose:** Single source of truth for all demo state, readable by training and visualization.

**Schema:**

```sql
-- Episode metrics (time series)
CREATE TABLE episodes (
    episode_id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    survival_time INTEGER NOT NULL,
    total_reward REAL NOT NULL,
    extrinsic_reward REAL NOT NULL,
    intrinsic_reward REAL NOT NULL,
    intrinsic_weight REAL NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    epsilon REAL NOT NULL
);
CREATE INDEX idx_episodes_timestamp ON episodes(timestamp);

-- Affordance visitation transitions (for "garden path" visualization)
CREATE TABLE affordance_visits (
    episode_id INTEGER NOT NULL,
    from_affordance TEXT NOT NULL,
    to_affordance TEXT NOT NULL,
    visit_count INTEGER NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);
CREATE INDEX idx_visits_episode ON affordance_visits(episode_id);

-- Grid position heatmap (rolling window: last 100 episodes)
CREATE TABLE position_heatmap (
    episode_id INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    visit_count INTEGER NOT NULL,
    novelty_value REAL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);
CREATE INDEX idx_heatmap_episode ON position_heatmap(episode_id);

-- System state (singleton, key-value)
CREATE TABLE system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Keys: 'current_episode', 'last_checkpoint', 'training_status',
--       'start_time', 'affordance_randomization_episode'
```

**Why SQLite:**

- Single file, easy backup/copy/mount as Docker volume
- ACID transactions = safe concurrent reads during writes
- No server, no ports, no authentication
- Standard SQL tools for debugging: `sqlite3 demo_state.db "SELECT COUNT(*) FROM episodes"`
- Trivial CSV export: `sqlite3 demo_state.db ".mode csv" ".output data.csv" "SELECT * FROM episodes"`

**Size Estimation:**

- 10K episodes × ~200 bytes/episode ≈ 2 MB (tiny)
- Position heatmap (rolling 100 episodes × 64 cells) ≈ 50 KB
- Total DB size: <10 MB

---

### 3. Visualization Server (`viz_server.py`)

**Purpose:** Serve frontend and stream real-time updates from SQLite to browsers.

**Minimal WebSocket Server:**

```python
class VizServer:
    def __init__(self, db_path: str, frontend_dir: str, port: int = 8765):
        """Initialize with SQLite connection and static file serving."""

    async def run(self):
        """Start WebSocket server and static file server."""
        # Serve frontend/ directory on HTTP
        # WebSocket on ws://hostname:8765/ws
        # Every 1 second:
        #   - Read latest episodes row
        #   - Read current position_heatmap (last 100 episodes avg)
        #   - Read affordance_visits for garden path
        #   - Send as JSON via WebSocket

    async def broadcast_update(self):
        """Query SQLite and send to all connected clients."""
        data = {
            'type': 'state_update',
            'episode': latest_episode,
            'metrics': {...},
            'position_heatmap': [[...]],  # 8x8 grid
            'affordance_graph': {...},     # garden path data
        }
```

**What We're NOT Adding:**

- ❌ Authentication (demo is public)
- ❌ Rate limiting (low viewer count)
- ❌ Complex state machine (just read and send)
- ❌ Caching (SQLite is fast enough)
- ❌ API versioning (no API, just WebSocket)

**Containerization-Ready:**

- `HAMLET_VIZ_PORT` env var (default 8765)
- `HAMLET_DB_PATH` env var
- `HAMLET_FRONTEND_DIR` env var (default `./frontend/dist`)

**Estimated LOC:** ~100 lines (reuses Phase 3 WebSocket infrastructure)

---

### 4. Snapshot Daemon (`snapshot_daemon.py`)

**Purpose:** Periodically capture screenshots, GIFs, and CSV exports for teaching materials.

**Cron-Like Daemon:**

```python
class SnapshotDaemon:
    def __init__(self, db_path: str, output_dir: str, browser_url: str):
        """Initialize with output directory for captures."""

    async def run(self):
        """Periodic capture loop."""
        while True:
            # Every 5 minutes:
            await self.capture_screenshot()

            # Every 50 episodes (check DB):
            if self.should_generate_gif():
                await self.generate_heatmap_gif()

            # Every hour:
            if self.should_export_csv():
                await self.export_metrics_csv()

            await asyncio.sleep(60)  # Check every minute

    async def capture_screenshot(self):
        """Headless Chrome screenshot of visualization."""
        # Uses selenium or playwright
        # Saves to: snapshots/screenshot_{timestamp}.png

    async def generate_heatmap_gif(self):
        """Compile last 50 episodes of novelty heatmaps into GIF."""
        # Query position_heatmap table for last 50 episodes
        # Generate 50 frames of 8x8 heatmap
        # Compile to: snapshots/novelty_ep{start}-{end}.gif

    async def export_metrics_csv(self):
        """Export episodes table to CSV."""
        # sqlite3 .mode csv .output
        # Saves to: exports/metrics_{timestamp}.csv
```

**Outputs:**

- `snapshots/screenshot_{timestamp}.png` - Every 5 minutes
- `snapshots/novelty_ep{start}-{end}.gif` - Every 50 episodes (~8 sec GIF, 25fps)
- `exports/metrics_{timestamp}.csv` - Hourly
- `exports/final_metrics.csv` - On completion

**Dependencies:**

- `selenium` or `playwright` for browser automation
- `Pillow` for GIF generation
- Headless Chrome

**Containerization-Ready:**

- `HAMLET_SNAPSHOT_DIR` env var
- `HAMLET_BROWSER_URL` env var (default `http://localhost:5173`)

---

### 5. Frontend Extensions (Vue Components)

**Reuses Phase 3 Visualization** with 4 new layers:

**Layer 1: Position Heatmap** (already exists from Phase 3)

- NoveltyHeatmap.vue component
- 8×8 grid colored by visit frequency or novelty
- Updates every episode

**Layer 2: "Garden Path" Affordance Transitions** (NEW)

```vue
<!-- components/AffordanceGraph.vue -->
<template>
  <div class="affordance-graph">
    <svg :width="width" :height="height">
      <!-- Nodes: Bed, Job, Shower, Fridge -->
      <!-- Edges: Transition frequency (thickness) and recency (color) -->
    </svg>
  </div>
</template>
```

- Sankey diagram or force-directed graph
- Shows Bed→Job→Fridge→Shower patterns
- Edge thickness = transition frequency
- Edge color = time period (early=blue, late=red)

**Layer 3: Temporal Novelty GIF** (generated offline by snapshot daemon)

- Displayed as static image that updates every 50 episodes
- Shows "boredom spreading" across the grid

**Layer 4: Reward Decomposition Timeline** (extend existing IntrinsicRewardChart)

- Stacked area chart showing extrinsic (bottom) + intrinsic (top)
- X-axis: Episode number
- Y-axis: Reward magnitude
- Shows intrinsic shrinking, extrinsic growing

---

## Data Flow

### Startup Sequence

```bash
# Terminal 1: Training (or systemd unit)
export HAMLET_CONFIG=configs/townlet/sparse_adaptive.yaml
export HAMLET_DB_PATH=$PWD/demo_state.db
export HAMLET_CHECKPOINT_DIR=$PWD/checkpoints
python demo_runner.py

# Terminal 2: Visualization Server
export HAMLET_DB_PATH=$PWD/demo_state.db
export HAMLET_VIZ_PORT=8765
python viz_server.py

# Terminal 3: Snapshot Daemon
export HAMLET_DB_PATH=$PWD/demo_state.db
export HAMLET_SNAPSHOT_DIR=$PWD/snapshots
export HAMLET_BROWSER_URL=http://localhost:5173
python snapshot_daemon.py

# Terminal 4: Frontend Dev Server
cd frontend && npm run dev
# Production: npm run build && serve dist/
```

### Runtime Flow

1. **Training writes** episode metrics → SQLite (1 transaction per episode, ~1 sec)
2. **Viz server polls** SQLite every 1 sec → broadcasts to WebSocket clients
3. **Browsers receive** updates → Vue components re-render (reactive)
4. **Snapshot daemon wakes** every 5 min → screenshots, GIFs, CSVs
5. **Checkpoints saved** every 100 episodes → `checkpoints/` directory
6. **On crash:** systemd restarts training → resumes from latest checkpoint

**No complex orchestration:** Just processes reading/writing a shared file.

---

## Auto-Restart Strategy

**Using systemd (Ubuntu 24.04, RunPod):**

```ini
# /etc/systemd/system/hamlet-demo.service
[Unit]
Description=Hamlet Multi-Day Demo Training
After=network.target

[Service]
Type=simple
User=john
WorkingDirectory=/home/john/hamlet
ExecStart=/home/john/hamlet/.venv/bin/python demo_runner.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment variables
Environment="HAMLET_CONFIG=/home/john/hamlet/configs/townlet/sparse_adaptive.yaml"
Environment="HAMLET_DB_PATH=/home/john/hamlet/demo_state.db"
Environment="HAMLET_CHECKPOINT_DIR=/home/john/hamlet/checkpoints"

[Install]
WantedBy=multi-user.target
```

**Commands:**

```bash
# Enable and start
sudo systemctl enable hamlet-demo
sudo systemctl start hamlet-demo

# Monitor
sudo systemctl status hamlet-demo
sudo journalctl -u hamlet-demo -f

# Restart if needed
sudo systemctl restart hamlet-demo
```

**Why systemd:**

- Native to Ubuntu 24.04 and RunPod
- Logs to journalctl (persistent, queryable)
- Auto-restart on crashes
- Survives reboots (if enabled)
- Standard Linux tooling

---

## Generalization Test

**At episode 5000, test if agent generalizes when affordances move:**

```python
# In demo_runner.py
if episode == 5000:
    logger.info("=" * 60)
    logger.info("GENERALIZATION TEST: Randomizing affordance positions")
    logger.info("=" * 60)

    # Store old positions in DB for comparison
    old_positions = env.get_affordance_positions()

    # Randomize
    env.randomize_affordance_positions()

    # Store new positions
    new_positions = env.get_affordance_positions()

    # Mark in system_state table
    db.execute(
        "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
        ("affordance_randomization_episode", "5000")
    )
    db.execute(
        "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
        ("old_affordance_positions", json.dumps(old_positions))
    )
    db.execute(
        "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
        ("new_affordance_positions", json.dumps(new_positions))
    )
    db.commit()
```

**Expected Behavior:**

- Survival time dips (agent searches for moved affordances)
- RND novelty spikes (familiar locations now empty, new locations occupied)
- Within 200-500 episodes: survival time recovers as agent re-learns
- Position heatmap shifts to new affordance locations

**Visualization:**

- Vertical line on survival trend chart at episode 5000
- Annotation: "Affordances Randomized"
- Shows the "disruption → recovery" arc

---

## Success Criteria

### Quantitative Metrics

- [ ] **Runtime:** 48+ hours without manual intervention
- [ ] **Episodes:** Complete 5,000+ episodes (10K is stretch goal)
- [ ] **Intrinsic Weight:** 1.0 → <0.3 (exploration → exploitation)
- [ ] **Survival Time:** ~115 steps baseline → 150+ steps
- [ ] **Curriculum:** Progress to stage 4 or 5
- [ ] **Generalization:** After episode 5000 randomization, recovery within 500 episodes
- [ ] **Stability:** No memory leaks, no NaN losses, no crashes requiring manual restart

### Qualitative (Teaching Materials)

- [ ] **Data Captured:** 10K rows in episodes table, position heatmaps, affordance graphs
- [ ] **Visualizations:** 200+ screenshots, 100+ novelty GIFs, complete CSV export
- [ ] **Transitions Visible:** Clear exploration→exploitation in charts
- [ ] **Garden Paths:** Affordance transition graph shows learned routines
- [ ] **Interesting Failures:** At least one documented emergent behavior (reward hacking, etc.)

### Deliverables

1. **SQLite Database:** `demo_state.db` with full training history
2. **Checkpoints:** `checkpoints/` directory with all episode checkpoints
3. **Screenshots:** `snapshots/` directory with 5-minute captures
4. **Novelty GIFs:** 100+ GIFs showing temporal heatmap evolution
5. **CSV Exports:** Hourly snapshots + final complete export
6. **YouTube Stream:** OBS recording of full demo (time-lapse or live)
7. **Analysis Notebook:** Jupyter notebook with plots and commentary

---

## Teaching Materials Output

### Immediate Use (Blog Post/Video)

**Title:** "What Happens When an RL Agent Learns for 3 Days Straight"

**Content:**

- Time-lapse video showing novelty heatmap fading over hours
- Survival time chart showing steady improvement
- Garden path diagram: "The agent learned this routine"
- Generalization test: "Watch it adapt when we move the affordances"
- Intrinsic weight decay: "Curiosity fades as mastery grows"

### Course Materials

**Lecture 1: Exploration vs Exploitation**

- Use intrinsic reward chart to show trade-off
- "Early: curious, late: greedy"

**Lecture 2: Curriculum Learning**

- Show 5-stage progression from demo data
- "Complexity increases as agent masters basics"

**Lecture 3: Generalization**

- Episode 5000 disruption as case study
- "True learning means adapting to change"

**Lecture 4: Emergent Behavior**

- Document any reward hacking observed
- "Agents optimize what you measure, not what you mean"

---

## Future Containerization Path

**When needed (H200 scaling or distribution), each process becomes a container:**

```yaml
# docker-compose.yml (future)
services:
  training:
    build: .
    command: python demo_runner.py
    environment:
      - HAMLET_CONFIG=/config/sparse_adaptive.yaml
      - HAMLET_DB_PATH=/data/demo_state.db
    volumes:
      - ./checkpoints:/checkpoints
      - ./data:/data
    restart: always

  viz:
    build: .
    command: python viz_server.py
    ports:
      - "8765:8765"
    environment:
      - HAMLET_DB_PATH=/data/demo_state.db
    volumes:
      - ./data:/data
    depends_on:
      - training

  snapshots:
    build: .
    command: python snapshot_daemon.py
    environment:
      - HAMLET_DB_PATH=/data/demo_state.db
      - HAMLET_BROWSER_URL=http://viz:8765
    volumes:
      - ./data:/data
      - ./snapshots:/snapshots
    depends_on:
      - viz
```

**The current design is already container-ready** (env vars, ports, file-based state). No refactoring needed.

---

## Timeline & Effort Estimate

**Implementation:** 2-3 days

- Day 1: demo_runner.py, SQLite schema, systemd setup
- Day 2: viz_server.py extensions, snapshot_daemon.py, frontend tweaks
- Day 3: Testing, debugging, deployment to server

**Demo Runtime:** 2-3 days

- 10K episodes at ~100-150 episodes/hour = 70-100 hours
- Realistically: 48-72 hours is sufficient for teaching materials

**Post-Demo Analysis:** 1 day

- Generate final visualizations
- Write blog post or documentation
- Create Jupyter notebook with analysis

**Total:** 1 week end-to-end

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Training crashes repeatedly** | Demo fails, no data | systemd auto-restart, checkpoint every 100 episodes |
| **Memory leak over 48+ hours** | Process OOM, restart loop | Profile before demo, add memory monitoring alerts |
| **SQLite write contention** | Slow training, corrupted DB | Use WAL mode, single writer (training), batch writes |
| **Visualization freezes browsers** | Poor viewer experience | Throttle WebSocket updates to 1/sec, limit data size |
| **Snapshot daemon fills disk** | Crash due to no space | Monitor disk usage, limit snapshot retention |
| **Frontend build fails on server** | Can't serve viz | Test deployment process beforehand, use `npm run build` |

---

## Design Validation Checklist

- [x] Architecture supports "for but not with containers"
- [x] Auto-restart mechanism (systemd) is robust
- [x] SQLite schema captures all needed metrics
- [x] Visualization server is minimal (no overengineering)
- [x] Snapshot daemon outputs useful teaching materials
- [x] Generalization test validates learning (not just memorization)
- [x] Success criteria are measurable
- [x] Timeline is realistic (1 week total)
- [x] Risks are identified and mitigated

---

## Next Steps

1. **Phase 5: Worktree Setup** - Create isolated workspace (or work on main if single-branch)
2. **Phase 6: Planning Handoff** - Use writing-plans skill to create detailed implementation plan
3. **Implementation** - Build components following plan
4. **Deployment** - Set up systemd, test auto-restart
5. **Demo Run** - Execute 48-72 hour demonstration
6. **Analysis** - Generate teaching materials and document findings

---

## References

- Phase 3 Verification: `docs/townlet/PHASE3_VERIFICATION.md`
- Roadmap: `docs/plans/ROADMAP.md`
- Phase 3 Design: `docs/plans/2025-10-30-townlet-phase3-intrinsic-exploration.md`
- North Star Vision: `docs/plans/ROADMAP.md#north-star-vision-social-hamlet`
