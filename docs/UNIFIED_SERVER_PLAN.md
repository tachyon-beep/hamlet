# Unified Server Implementation Plan

**Created:** November 2, 2025  
**Methodology:** Red-Green-Refactor  
**Goal:** Single-command demo experience (`python run_demo.py`)

---

## Executive Summary

Replace three-terminal workflow with unified server that coordinates:

1. **Training** (background thread)
2. **Inference Server** (FastAPI WebSocket)
3. **Frontend** (Vue + Vite subprocess)

**User Experience Target:**

```bash
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 10000
# Browser opens automatically to http://localhost:5173
# Training starts in background
# Ctrl+C gracefully shuts down all components
```

---

## File Inventory

### New Files (2)

1. **`run_demo.py`** (entry point, ~100 lines)
   - CLI argument parsing
   - UnifiedServer instantiation
   - Signal handler registration

2. **`src/townlet/demo/unified_server.py`** (~300 lines)
   - UnifiedServer class
   - Thread management for training + inference
   - Subprocess management for frontend
   - Graceful shutdown coordination

### Modified Files (2-3)

3. **`src/townlet/demo/runner.py`** (minor modifications)
   - Add `run_single_episode()` method (extract from `run()`)
   - Make `run()` check `self.should_shutdown` more frequently
   - Ensure thread-safe checkpoint access

4. **`src/townlet/demo/live_inference.py`** (minor modifications)
   - Expose `start()` and `stop()` methods
   - Already async, just needs better lifecycle hooks

5. **`frontend/package.json`** (optional)
   - Add `"demo": "vite --open"` script for auto-browser-open

**Risk Assessment:** Low-Medium (mostly orchestration, minimal changes to existing systems)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ run_demo.py (Main Process)                                  │
│  ├─ Parse args (config, episodes, checkpoint_dir)           │
│  ├─ Create UnifiedServer                                    │
│  ├─ Register signal handlers (SIGINT/SIGTERM)               │
│  └─ Call server.start() → blocks until shutdown             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ UnifiedServer (Orchestrator)                                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Training Thread                                    │    │
│  │  - DemoRunner.run() modified to be stoppable      │    │
│  │  - Saves checkpoints every 100 episodes           │    │
│  │  - Checks self.should_shutdown flag               │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Inference Server Thread                            │    │
│  │  - uvicorn running FastAPI app                     │    │
│  │  - WebSocket on port 8766                          │    │
│  │  - Polls latest checkpoint                         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Frontend Subprocess                                │    │
│  │  - subprocess.Popen(["npm", "run", "dev"])        │    │
│  │  - Runs in frontend/ directory                     │    │
│  │  - HTTP server on port 5173                        │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Shutdown Sequence:                                         │
│  1. Set training.should_shutdown = True                     │
│  2. Wait for training thread to finish episode             │
│  3. Stop inference server (FastAPI shutdown)                │
│  4. Send SIGTERM to frontend subprocess                     │
│  5. Wait for clean exits (with timeout)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## RED Phase: Define the Interface

### Command Line Interface

```bash
python run_demo.py \
  --config configs/level_1_full_observability.yaml \
  --episodes 10000 \
  --checkpoint-dir runs/L1_full_observability/2025-11-02_HHMMSS/checkpoints \
  [--port 8766] \
  [--frontend-port 5173] \
  [--no-browser]
```

### Expected Output

```
[2025-11-02 12:34:56] Unified Demo Server Starting
[2025-11-02 12:34:56] Config: configs/level_1_full_observability.yaml
[2025-11-02 12:34:56] Episodes: 10000
[2025-11-02 12:34:56] Checkpoints: runs/L1_full_observability/2025-11-02_123456/checkpoints
[2025-11-02 12:34:56] ─────────────────────────────────────────────────
[2025-11-02 12:34:56] [Training] Starting background training...
[2025-11-02 12:34:57] [Training] Loaded checkpoint from episode 500
[2025-11-02 12:34:58] [Inference] Starting WebSocket server on port 8766...
[2025-11-02 12:34:59] [Inference] Server ready at ws://localhost:8766
[2025-11-02 12:35:00] [Frontend] Starting Vue dev server...
[2025-11-02 12:35:02] [Frontend] Dev server ready at http://localhost:5173
[2025-11-02 12:35:03] Opening browser to http://localhost:5173...
[2025-11-02 12:35:03] ─────────────────────────────────────────────────
[2025-11-02 12:35:03] All systems operational. Press Ctrl+C to stop.
[2025-11-02 12:35:03] ─────────────────────────────────────────────────
[2025-11-02 12:35:10] [Training] Episode 501 | Survival: 87 steps | Reward: -12.34
[2025-11-02 12:35:15] [Training] Episode 502 | Survival: 123 steps | Reward: +5.67
...
^C
[2025-11-02 12:45:00] Shutdown signal received. Gracefully stopping...
[2025-11-02 12:45:00] [Training] Finishing current episode...
[2025-11-02 12:45:02] [Training] Saving checkpoint at episode 550...
[2025-11-02 12:45:03] [Training] Thread stopped.
[2025-11-02 12:45:03] [Inference] Stopping WebSocket server...
[2025-11-02 12:45:04] [Inference] Server stopped.
[2025-11-02 12:45:04] [Frontend] Sending SIGTERM to subprocess...
[2025-11-02 12:45:05] [Frontend] Subprocess terminated.
[2025-11-02 12:45:05] ─────────────────────────────────────────────────
[2025-11-02 12:45:05] All systems stopped. Goodbye!
```

---

## GREEN Phase: Minimal Implementation

### Phase 1: Basic Structure (Day 1 - 2 hours)

**Goal:** Create entry point and orchestrator skeleton.

**Tasks:**

1. Create `run_demo.py`:
   - Argparse CLI (config, episodes, checkpoint_dir, port, no-browser flag)
   - Import UnifiedServer
   - Signal handler registration (SIGINT, SIGTERM)
   - Call `server.start()` and block until shutdown

2. Create `src/townlet/demo/unified_server.py`:
   - UnifiedServer class skeleton
   - `__init__(config_path, episodes, checkpoint_dir, port, open_browser)`
   - `start()` method (placeholder - just log "Starting...")
   - `stop()` method (placeholder - just log "Stopping...")
   - Basic logging setup

**Test:** Run `python run_demo.py --help` and verify argument parsing.

**Success Criteria:**

- ✅ CLI parses arguments correctly
- ✅ UnifiedServer instantiates without errors
- ✅ Ctrl+C triggers stop() method

---

### Phase 2: Non-Blocking Training (Day 1-2 - 3 hours)

**Goal:** Make training run in background thread.

**Tasks:**

1. Modify `src/townlet/demo/runner.py`:
   - Extract `run_single_episode()` method from main `run()` loop
   - Make `run()` check `self.should_shutdown` after every episode
   - No breaking changes to checkpoint/TensorBoard logic

2. Update `unified_server.py`:
   - Import DemoRunner
   - Create training thread in `start()`:

     ```python
     self.training_thread = threading.Thread(
         target=self._run_training,
         daemon=False
     )
     self.training_thread.start()
     ```

   - `_run_training()` method: instantiate DemoRunner and call `run()`
   - `stop()` sets `runner.should_shutdown = True` and waits for thread

**Test:** Run demo, verify training proceeds in background, Ctrl+C stops cleanly.

**Success Criteria:**

- ✅ Training runs in separate thread
- ✅ Checkpoints saved correctly
- ✅ Graceful shutdown after current episode completes
- ✅ No zombie threads

---

### Phase 3: Subprocess Management (Day 2 - 3 hours)

**Goal:** Start frontend and inference server alongside training.

**Tasks:**

1. Update `unified_server.py` - Frontend subprocess:

   ```python
   def _start_frontend(self):
       self.frontend_process = subprocess.Popen(
           ["npm", "run", "dev"],
           cwd=Path(__file__).parent.parent.parent.parent / "frontend",
           stdout=subprocess.PIPE,
           stderr=subprocess.PIPE,
           text=True
       )
       # Wait for "ready" message in stdout (timeout 30s)
       # Optionally open browser with webbrowser.open()
   
   def _stop_frontend(self):
       if self.frontend_process:
           self.frontend_process.terminate()
           self.frontend_process.wait(timeout=5)
   ```

2. Update `unified_server.py` - Inference server:
   - Import LiveInferenceServer
   - Create inference thread similar to training
   - Use uvicorn.run() in thread with shutdown mechanism

3. Update `src/townlet/demo/live_inference.py`:
   - Add `stop()` method that sets shutdown flag
   - Ensure FastAPI shutdown is clean

**Test:** Run demo, verify all three components start, browser opens, Ctrl+C stops all.

**Success Criteria:**

- ✅ Frontend subprocess starts and serves on port 5173
- ✅ Inference server starts and accepts WebSocket connections
- ✅ Browser opens automatically (unless --no-browser)
- ✅ All components stop gracefully on Ctrl+C
- ✅ No orphaned processes after shutdown

---

### Phase 4: Integration & Polish (Day 3 - 2 hours)

**Goal:** End-to-end testing and edge case handling.

**Tasks:**

1. Test full startup sequence:
   - Fresh run (no checkpoints)
   - Resume from checkpoint
   - Multiple Ctrl+C scenarios (during episode, between episodes)

2. Add logging prefixes:
   - `[Training]` for DemoRunner logs
   - `[Inference]` for LiveInferenceServer logs
   - `[Frontend]` for npm subprocess logs

3. Error handling:
   - Port already in use → retry with +1 or fail gracefully
   - Frontend npm not found → clear error message
   - Config file not found → fail fast with helpful message

4. Documentation:
   - Update `DEMO_README.md` with new workflow
   - Add `--help` text examples
   - Update `AGENTS.md` with new entry point

**Test:** Run 10-episode demo end-to-end, verify all systems work together.

**Success Criteria:**

- ✅ Checkpoint resume works correctly
- ✅ Logs are clearly attributed to each component
- ✅ Port conflicts handled gracefully
- ✅ Documentation updated
- ✅ 10-episode run completes without issues

---

## REFACTOR Phase: Make It Shine

### Polish Features (Optional - Day 4)

**Progress Bars:**

```python
from tqdm import tqdm
# Add episode progress bar in training thread
```

**Health Checks:**

```python
# Wait for inference server to be healthy before opening browser
# Retry frontend startup if initial connection fails
```

**Colored Output:**

```python
from colorama import Fore, Style
# [Training] in green, [Inference] in blue, [Frontend] in yellow
```

**Configuration:**

```python
# Add unified_server section to YAML configs
unified_server:
  inference_port: 8766
  frontend_port: 5173
  open_browser: true
  health_check_timeout: 30
```

**Monitoring:**

```python
# Optional: Print aggregate stats every 100 episodes
# Episodes completed, avg survival time, checkpoint size, etc.
```

---

## Success Criteria Matrix

### Must-Have (Phase 1-4)

- ✅ Single command starts all systems
- ✅ Training runs in background without blocking
- ✅ Inference server accepts WebSocket connections
- ✅ Frontend serves Vue app on port 5173
- ✅ Browser opens automatically
- ✅ Ctrl+C gracefully stops all components
- ✅ Checkpoints save/load correctly
- ✅ No orphaned processes after shutdown
- ✅ Clear error messages for common failures
- ✅ Documentation updated

### Nice-to-Have (Refactor Phase)

- ⭐ Progress bars for visual feedback
- ⭐ Colored log output for readability
- ⭐ Health checks before browser open
- ⭐ Port conflict auto-recovery
- ⭐ Aggregate statistics display
- ⭐ Docker support (future)

---

## Timeline Estimate

**Total Time:** 10-12 hours over 3-4 days

- **Phase 1:** 2 hours (basic structure)
- **Phase 2:** 3 hours (non-blocking training)
- **Phase 3:** 3 hours (subprocess management)
- **Phase 4:** 2 hours (integration & testing)
- **Refactor:** 2-4 hours (optional polish)

**Risk Buffers:**

- Threading bugs: +1-2 hours
- Port conflicts: +0.5 hour
- Frontend startup timing: +1 hour
- Documentation: +1 hour

---

## Testing Strategy

### Unit Tests (Optional)

- `test_unified_server.py`: Test thread lifecycle, shutdown sequence
- Mock subprocess.Popen and threading.Thread for isolated testing

### Integration Tests (Required)

1. **Fresh Start Test**: Run with no checkpoints, verify all systems start
2. **Resume Test**: Run with existing checkpoint, verify continuation
3. **Shutdown Test**: Ctrl+C at various points, verify graceful exit
4. **Port Conflict Test**: Start inference on 8766, verify error handling
5. **Long Run Test**: 100 episodes, verify no memory leaks or crashes

### Manual Testing Checklist

- [ ] `python run_demo.py --help` shows all options
- [ ] Fresh run creates checkpoint directory and starts training
- [ ] Browser opens to <http://localhost:5173>
- [ ] Frontend shows live agent visualization
- [ ] Training logs appear with [Training] prefix
- [ ] Ctrl+C during episode waits for completion before stopping
- [ ] Ctrl+C between episodes stops immediately
- [ ] All processes terminate cleanly (verify with `ps aux | grep python`)
- [ ] Resume from checkpoint continues from correct episode number
- [ ] TensorBoard logs continue in same directory

---

## Known Risks & Mitigations

### Risk 1: Race Condition on Checkpoint Access

**Scenario:** Training thread writes checkpoint while inference thread reads it.

**Mitigation:**

- Atomic writes: Write to temp file, then rename
- Inference polls every N seconds (current behavior)
- No file locking needed (inference reads, training writes)

### Risk 2: Frontend Startup Timing

**Scenario:** Browser opens before Vite dev server is ready.

**Mitigation:**

- Parse stdout for "Local: <http://localhost:5173>"
- Wait up to 30 seconds before opening browser
- Add --no-browser flag for manual control

### Risk 3: Orphaned Processes

**Scenario:** Ctrl+C doesn't cleanly stop all components.

**Mitigation:**

- Use daemon=False for threads (explicit join)
- Use subprocess.terminate() with timeout
- Register atexit handlers as backup
- Test on Linux (current environment)

### Risk 4: Port Conflicts

**Scenario:** Ports 8766 or 5173 already in use.

**Mitigation:**

- Try to bind socket before starting servers
- Fail fast with clear error message
- Optional: Auto-increment port and retry
- Document port requirements in README

---

## Rollback Plan

If unified server causes issues:

1. **Keep legacy workflow available**:
   - Don't delete `start_demo.sh` or manual instructions
   - Add note in README: "For manual 3-terminal setup, see LEGACY_DEMO.md"

2. **Feature flag**:
   - Add `--unified` flag to existing scripts
   - Default to legacy behavior until proven stable

3. **Incremental rollout**:
   - Phase 1-2: Just unified training + inference
   - Phase 3: Add frontend after validation
   - Phase 4: Make default after 1 week testing

---

## Future Enhancements (Post-MVP)

### WebSocket Dashboard Integration

- Real-time training metrics in frontend
- Episode survival graph updates live
- Curriculum stage visualization

### Docker Support

```dockerfile
FROM python:3.12
RUN apt-get install -y nodejs npm
COPY . /app
WORKDIR /app
RUN uv pip install -e .
RUN cd frontend && npm install
ENTRYPOINT ["python", "run_demo.py"]
```

### Cloud Deployment

- AWS EC2 instance with screen/tmux
- Persist checkpoints to S3
- CloudWatch logs integration

### Multi-Run Management

```bash
python run_demo.py --experiment-name "test_rnd_annealing" \
  --runs 5 --parallel 3
# Runs 5 independent training runs, 3 at a time
# Aggregates results for statistical significance
```

---

## Questions for Future Consideration

1. Should inference server auto-restart if it crashes?
2. Should we add a "pause training" API endpoint?
3. Should frontend detect new checkpoints and auto-refresh visualization?
4. Should we add Prometheus metrics export for monitoring?
5. Should we support remote training (SSH tunnel to inference server)?

---

**Status:** Planning Complete ✅  
**Next Action:** Execute Phase 1 (Basic Structure)  
**Estimated Completion:** November 5, 2025
