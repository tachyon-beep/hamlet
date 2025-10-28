# Hamlet Visualization System

Real-time web visualization for the Hamlet DRL agent using WebSocket streaming.

## Overview

The visualization system consists of:
- **Backend**: FastAPI server (Python) streaming simulation state via WebSocket
- **Frontend**: Vue 3 SPA with SVG grid rendering and real-time updates

## Quick Start

### 1. Start the Backend Server

```bash
# From project root
uv run python demo_visualization.py
```

The server will start on **http://localhost:8765**

### 2. Start the Frontend (Development)

In a separate terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at **http://localhost:5173**

### 3. Open in Browser

Navigate to http://localhost:5173 and you should see:
- 8x8 grid with agent and affordances
- Real-time meter displays
- Playback controls (play/pause/step/reset/speed)
- Episode statistics and performance history

## Architecture

### Backend Components

**`src/hamlet/web/server.py`**
- FastAPI application with WebSocket endpoint
- Serves static frontend files (production)
- REST API endpoints for status and models

**`src/hamlet/web/websocket.py`**
- WebSocket connection manager
- Broadcasts simulation updates to clients
- Handles control commands

**`src/hamlet/web/simulation_runner.py`**
- Async simulation orchestrator
- Runs agent in environment
- Yields state updates for streaming

**`src/hamlet/environment/renderer.py`**
- Serializes environment state to JSON
- Prepares data for WebSocket transmission

### Frontend Components

**`frontend/src/App.vue`**
- Root component with layout
- Manages WebSocket connection

**`frontend/src/stores/simulation.js`**
- Pinia store for state management
- WebSocket integration
- Control command dispatch

**`frontend/src/components/Grid.vue`**
- SVG 8x8 grid rendering
- Agents and affordances visualization
- Smooth animations

**`frontend/src/components/MeterPanel.vue`**
- Progress bars for energy/hygiene/satiation/money
- Color-coded meter levels
- Critical warnings (pulse animation)

**`frontend/src/components/Controls.vue`**
- Play/Pause/Step/Reset buttons
- Speed control slider (0.5x - 10x)
- Model selector

**`frontend/src/components/StatsPanel.vue`**
- Episode number, steps, reward
- Last action taken
- Performance history line chart
- Average survival time

## WebSocket Protocol

### Server → Client Messages

```json
// Connection acknowledgment
{
  "type": "connected",
  "available_models": ["trained_agent.pt"]
}

// Episode start
{
  "type": "episode_start",
  "episode": 42,
  "model_name": "trained_agent.pt"
}

// State update (every step)
{
  "type": "state_update",
  "step": 127,
  "grid": {
    "width": 8,
    "height": 8,
    "agents": [{"id": "agent_0", "x": 4, "y": 3, "color": "#3b82f6"}],
    "affordances": [
      {"type": "Bed", "x": 0, "y": 0},
      {"type": "Shower", "x": 7, "y": 0},
      {"type": "Fridge", "x": 0, "y": 7},
      {"type": "Job", "x": 7, "y": 7}
    ]
  },
  "agents": {
    "agent_0": {
      "meters": {"energy": 0.65, "hygiene": 0.42, "satiation": 0.88, "money": 45.0},
      "last_action": "move_up",
      "reward": -0.5
    }
  },
  "cumulative_reward": 123.4
}

// Episode end
{
  "type": "episode_end",
  "episode": 42,
  "steps": 234,
  "total_reward": 145.2,
  "reason": "meter_depleted"
}
```

### Client → Server Messages

```json
// Control commands
{"type": "control", "command": "play"}
{"type": "control", "command": "pause"}
{"type": "control", "command": "step"}
{"type": "control", "command": "reset"}
{"type": "control", "command": "set_speed", "speed": 2.0}
{"type": "control", "command": "load_model", "model": "agent_v2.pt"}
```

## Production Build

### Build Frontend

```bash
cd frontend
npm run build
```

This creates `frontend/dist/` with optimized static files.

### Run Production Server

```bash
uv run python demo_visualization.py
```

The server will automatically serve the built frontend from `frontend/dist/` at http://localhost:8765

## Configuration

### Change Port

Edit `src/hamlet/web/server.py`:

```python
def main():
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

Also update `frontend/vite.config.js`:

```javascript
server: {
  proxy: {
    '/ws': { target: 'ws://localhost:8765', ws: true },
    '/api': { target: 'http://localhost:8765' }
  }
}
```

### Adjust Simulation Speed

Edit `src/hamlet/web/simulation_runner.py`:

```python
def __init__(self, ..., base_delay: float = 0.1):
    # 0.1 = 10 steps/second at 1x speed
```

## Troubleshooting

### Frontend Can't Connect to Backend

- Ensure backend is running on port 8765
- Check browser console for WebSocket errors
- Verify CORS settings in `server.py`

### Agent Not Moving

- Click "Play" button (starts paused by default)
- Check that trained agent model exists in `models/` directory
- Backend will use untrained agent if model not found

### Slow Performance

- Reduce simulation speed
- Check network tab for large WebSocket messages
- Consider reducing update rate in `simulation_runner.py`

## Streaming to Twitch/YouTube

For streaming the visualization:

1. Open visualization in browser (http://localhost:5173)
2. Use OBS Studio to capture browser window
3. Add as "Browser Source" or "Window Capture"
4. Recommended resolution: 1280x720 (fits grid + controls)
5. Hide cursor in OBS settings for cleaner look

## Future Enhancements

- Multiple agent support (colors, labels)
- Replay system (save/load episodes)
- Training mode integration (watch agent learn in real-time)
- Customizable affordances and grid sizes
- WebRTC for lower latency streaming
- Mobile-responsive layout
