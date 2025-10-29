# Watch Training Live on the UI! 🎮

## Quick Start

### Terminal 1: Start the Training Server
```bash
cd /home/john/hamlet
uv run python src/hamlet/web/training_server.py
```

You should see:
```
============================================================
HAMLET TRAINING SERVER
============================================================
WebSocket endpoint: ws://localhost:8765/ws/training
Frontend should connect to this endpoint
============================================================
```

### Terminal 2: Start the Frontend
```bash
cd /home/john/hamlet/frontend
npm run dev
```

You should see:
```
  ➜  Local:   http://localhost:5173/
```

### Browser: Open and Connect

1. Open `http://localhost:5173` in your browser
2. In the **Controls** panel on the right:
   - Click the **Training** mode button
   - Click **Connect**
3. Configure training parameters:
   - **Episodes**: 100 (or your desired number)
   - **Batch Size**: 32
   - **Buffer Capacity**: 10000
   - **Show Every N Episodes**: 1 (shows EVERY episode - change to 5 or 10 for faster training)
   - **Step Delay**: 0.2 seconds (time between steps when visualizing)
4. Click **Start Training**

**Note:** The agent will move visibly on the grid during training! With the default settings (Show Every = 1), you'll see every single episode. The Step Delay controls how fast it moves (0.2s = 5 steps/second, smooth for human viewing).

## What You'll See

### In the Browser
- **Live grid updates** every 5 episodes (configurable)
- **Agent moving** around the grid 🤖
- **Meter bars** showing energy, hygiene, satiation, money, mood, social
- **Affordances**: Bed 🛏️, Shower 🚿, HomeMeal 🥘, FastFood 🍔, Job 💼, Recreation 🎮, Bar 🍺

### In the UI (Controls Panel)
You'll see:
- **Progress bar** showing Episode X / Total
- **Training Metrics**:
  - Avg Reward (last 5 episodes)
  - Avg Length (last 5 episodes)
  - Avg Loss (last 5 episodes)
  - Epsilon (exploration rate)
  - Buffer Size

### In the Console (Browser F12)
```
Episode 5 complete: 104 steps, reward: -107.37
Episode 10 complete: 143 steps, reward: -90.16
Episode 15 complete: 170 steps, reward: -81.15
...
```

### In the Terminal (Training Server)
```
Episode   5/100
  Avg Reward (last 5):  -107.37
  Avg Length (last 5):    104.6  ← Agent surviving longer!
  Avg Loss (last 5):    55.3484  ← Network learning
  Epsilon:                0.975  ← Exploration decreasing
  Buffer size:              523
```

## Parameters You Can Tweak

Use the UI form fields in the Controls panel:

- **Episodes** (Default: 100) - How many episodes to train
- **Batch Size** (Default: 32) - Learning batch size
- **Buffer Capacity** (Default: 10000) - Replay buffer size
- **Show Every N Episodes** (Default: 1) - 1 = show all episodes, 5 = show every 5th
- **Step Delay** (Default: 0.2s) - Time between visualization steps (0.2s = smooth)

**Examples:**

**Quick test (watch closely):**
- Episodes: 20
- Batch Size: 32
- Buffer Capacity: 5000
- Show Every: 1 (see every episode)
- Step Delay: 0.3s (slower, easier to follow)

**Long training (faster):**
- Episodes: 500
- Batch Size: 64
- Buffer Capacity: 50000
- Show Every: 10
- Step Delay: 0.1s (faster visualization)

**Ultra-fast training (minimal visualization):**
- Episodes: 1000
- Batch Size: 64
- Buffer Capacity: 50000
- Show Every: 50 (only show occasionally)
- Step Delay: 0.05s (very fast steps)

## What the RelationalQNetwork Should Learn

Watch the **Avg Length** (survival time) increase:

| Episodes | Expected Survival | What Agent Learned |
|----------|-------------------|-------------------|
| 0-20 | ~100 steps | Random wandering, frequent deaths |
| 20-50 | ~130 steps | Bed when tired, Shower when dirty |
| 50-100 | ~180 steps | Job for money, HomeMeal for food |
| 100-200 | ~230 steps | Starting to discover job penalty |
| 200-400 | ~280 steps | **Job penalty discovered!** (works when healthy) |
| 400-600 | ~320 steps | Spatial food choices (HomeMeal at home, FastFood at work) |
| 600+ | ~400+ steps | **Near-optimal!** Multi-step Bar planning |

## Key Metrics to Watch

### Avg Length (Survival Time)
- **Good**: Steadily increasing
- **Bad**: Stuck at ~100 steps (not learning)

### Avg Loss
- **Good**: Decreasing from ~50 to ~5
- **Bad**: Staying high or exploding

### Epsilon (Exploration)
- **Good**: Smoothly decreasing from 1.0 to 0.05
- **Bad**: N/A (this is automatic)

## Comparing Architectures

Want to see if the RelationalQNetwork really is better? Train both!

### Train the "Potato" (Basic MLP)
Modify `src/hamlet/web/training_server.py` line 46:
```python
network_type="qnetwork",  # Basic MLP (potato)
```

Restart the training server, then train for 500 episodes.

### Train the Relational Network
Use the default:
```python
network_type="relational",  # Attention network
```

**Expected Result**: Relational network should reach 400+ step survival 2-3× faster!

## Troubleshooting

### "WebSocket not connected"
- Make sure training server is running (`uv run python src/hamlet/web/training_server.py`)
- Make sure you selected **Training** mode before clicking **Connect**
- Check browser console (F12) for error messages

### "Can't see the agent moving"
- The agent only shows every N episodes (default: 5)
- Try setting **Show Every N Episodes** to 1 to see every episode
- Check browser console for errors

### "Training is too slow"
- Increase **Show Every N Episodes** to 10 or higher
- The visualization slows down training (adds 50ms delay per step when showing)

### "Want to pause/resume"
- Pause/resume functionality is not yet implemented in the UI
- You can stop training by clicking **Disconnect** and refreshing the browser

## Advanced: Watch Attention Weights (Future)

In a future update, we could add attention weight visualization:
- See which meters the network focuses on for each decision
- "When deciding to work, attention weights: Energy 45%, Hygiene 35%, Money 10%..."

This would show the network learning cross-meter dependencies!

## Files Created

- **Backend**: `src/hamlet/web/training_server.py` (WebSocket server for training)
- **Frontend**: Modified `src/hamlet/stores/simulation.js` (added training mode)
- **Guide**: `TRAINING_UI_GUIDE.md` (this file)

## Next Steps

1. ✅ ~~Add a "Training Mode" button in the UI (no console needed)~~
2. ✅ ~~Show training metrics in the UI~~
3. Add training history charts (survival time, loss, epsilon over all episodes)
4. Add pause/resume controls for training
5. Visualize attention weights (see what the network is "thinking")
6. Compare multiple architectures side-by-side

---

**Enjoy watching your agent learn! 🎓🤖**

The best part is watching the survival time increase as the network discovers:
- Basic survival (Bed, Shower, Food)
- Economic planning (Job when low money)
- **The job penalty!** (Work only when healthy)
- **Spatial trade-offs!** (HomeMeal vs FastFood based on location)
- **Multi-step planning!** (Bar → Bed → Shower → Work sequence)

This is **deep reinforcement learning in action!**
