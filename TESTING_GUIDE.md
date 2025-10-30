# Testing Guide - Auto-Play UI Fix

## Quick Start

### 1. Create Test Checkpoint (One-time setup)
```bash
uv run python create_test_checkpoint.py
```

This creates `checkpoints/checkpoint_ep00001.pt` with a minimal trained network.

### 2. Start Backend Server
```bash
uv run python -c "from hamlet.demo.live_inference import run_server; run_server(checkpoint_dir='checkpoints', port=8766, step_delay=0.2, total_episodes=5000)"
```

Expected output:
```
[INFO] Starting live inference server on port 8766
[INFO] Checkpoint directory: checkpoints
[INFO] Loading checkpoint: checkpoint_ep00001.pt (episode 1)
[INFO] Loaded Q-network weights
INFO:     Uvicorn running on http://0.0.0.0:8766
```

### 3. Start Frontend (New terminal)
```bash
cd frontend
npm run dev
```

Expected output:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### 4. Test Auto-Play

1. Open browser to `http://localhost:5173`
2. Click the "Connect" button
3. **Expected behavior**:
   - Connection indicator turns green
   - Grid appears immediately (8×8 cells)
   - Agent (green dot) visible on grid
   - 14 affordances visible (icons: 🛏️🚿🍴💼👥🎭🏋️)
   - Meters panel shows 8 meters with values
   - Agent starts moving automatically
   - Episode counter increments

4. **If it works**: Grid is animated, meters update in real-time, no manual play needed
5. **If it fails**: Grid stays empty, meters show "No data" - check browser console

### 5. Test Controls

- **Speed slider**: Change simulation speed (0.1x to 5x)
- **Disconnect button**: Stop streaming, grid should clear
- **Connect again**: Auto-play resumes immediately

## Automated Testing

Run the automated test (backend must be running):

```bash
uv run python test_auto_play.py
```

Expected output:
```
✓ Connected successfully
✓ Received: connected
✓ Sent play command
✓ Episode 1 started
✓ First state update received
✓ Test passed!
```

## Troubleshooting

### Backend won't start
**Error**: `No checkpoints found in checkpoints/`
**Fix**: Run `uv run python create_test_checkpoint.py`

### Frontend won't connect
**Error**: Browser console shows `WebSocket connection failed`
**Checks**:
1. Is backend running? Check `ps aux | grep live_inference`
2. Correct port? Backend uses 8766, frontend connects to same
3. Check backend logs: `tail -20 /tmp/backend_server.log`

### Connection succeeds but no data
**Error**: Grid stays empty after connecting
**Checks**:
1. Open browser DevTools → Network → WS tab
2. Click on the WebSocket connection
3. Look for messages:
   - `connected` (initial)
   - `episode_start` (should appear within 1 second)
   - `state_update` (continuous stream)
4. If you only see `connected`, the auto-play fix didn't work

### Grid appears but agent doesn't move
**Error**: Grid and affordances visible, but agent is frozen
**Checks**:
1. Check browser console for JavaScript errors
2. Look for `state_update` messages in DevTools WS tab
3. Backend might have crashed - check process is still running

## Development Workflow

### Making Frontend Changes

1. Edit Vue components in `frontend/src/`
2. Vite hot-reloads automatically
3. No need to restart backend

### Making Backend Changes

1. Stop backend (Ctrl+C)
2. Edit files in `src/hamlet/demo/`
3. Restart backend
4. Frontend will auto-reconnect

## Clean Shutdown

1. Frontend: Ctrl+C in terminal
2. Backend: Ctrl+C in terminal
3. Or kill all: `pkill -f live_inference`

## Next Steps

After verifying the fix works:
1. Test with real trained checkpoints (run full training)
2. Test model switching (load different checkpoints while running)
3. Test multi-hour sessions (backend stability)
4. Test with multiple clients (multiple browser tabs)
