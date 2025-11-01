# Metaserver Usage Guide

## Quick Start

The **metaserver** is the recommended way to run Hamlet visualization. It provides both inference and training modes on a single port.

### Start the Metaserver

```bash
# Terminal 1: Start metaserver
uv run python demo_metaserver.py
```

### Start the Frontend

```bash
# Terminal 2: Start frontend
cd frontend && npm run dev
```

### Open Browser

Navigate to `http://localhost:5173`

**Both mode buttons will now be enabled!** Switch between inference and training without restarting the server.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Metaserver (Port 8765)                          │
├─────────────────────────────────────────────────────────────┤
│  HTTP Endpoints:                                             │
│  • GET  /           - Health check and status               │
│  • GET  /api/models - List available trained models         │
│                                                               │
│  WebSocket Endpoints:                                        │
│  • /ws              - Inference mode (multi-session)        │
│  • /ws/training     - Training mode (broadcast)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Inference Mode (`/ws`)

**Multi-Session Architecture**: Each connected client gets an independent session.

### Features
- Load different trained models per session
- Multiple users can watch different agents simultaneously
- Real-time playback controls (play/pause/step/reset)
- Adjustable playback speed
- Position heat map visualization
- Episode statistics and performance charts

### Use Case
Perfect for demonstrating trained agents to students. Each student can:
1. Select a different checkpoint (ep50, ep455, ep1000)
2. Watch at their own pace
3. Compare learning stages
4. See reward hacking behaviors

---

## Training Mode (`/ws/training`)

**Singleton Broadcast Architecture**: One training session broadcasts to all clients.

### Features
- Watch agent learn in real-time
- Configurable training parameters (episodes, batch size, etc.)
- Live metrics: reward, loss, epsilon, buffer size
- Position heat map tracks exploration
- Pause/resume during training
- Trained model saved automatically

### Use Case
Perfect for live demonstrations:
1. Students watch agent explore randomly (high epsilon)
2. Gradually learn better strategies (decreasing loss)
3. Converge to exploitation (low epsilon)
4. Observe "interesting failures" like reward hacking
5. Multiple students can observe same training run

### Parameters
```python
{
  "num_episodes": 100,      # Total episodes to train
  "batch_size": 32,         # Batch size for learning
  "buffer_capacity": 10000, # Replay buffer size
  "show_every": 5,          # Visualize every Nth episode
  "step_delay": 0.2         # Delay between steps (seconds)
}
```

---

## API Endpoints

### GET /
Health check and server status.

**Response:**
```json
{
  "status": "running",
  "service": "Hamlet Metaserver",
  "version": "2.0.0",
  "endpoints": {
    "inference": "/ws",
    "training": "/ws/training"
  },
  "active_sessions": {
    "inference": 2,
    "training": 1
  },
  "training_active": true
}
```

### GET /api/models
List available trained models.

**Response:**
```json
{
  "models": [
    "trained_agent.pt",
    "checkpoint_ep50.pt",
    "checkpoint_ep455.pt"
  ],
  "count": 3
}
```

---

## Comparison with Old Approach

### Before (Separate Servers)
```bash
# Could only run ONE at a time

# Option 1: Inference only
uv run python demo_visualization.py

# Option 2: Training only
uv run python demo_training_visualization.py

# Problem: Must stop one to start the other!
```

### After (Unified Metaserver)
```bash
# Run BOTH simultaneously
uv run python demo_metaserver.py

# ✅ Inference and training both available
# ✅ Switch modes without restarting
# ✅ Multiple users, different modes
```

---

## Advanced Usage

### Running Multiple Concurrent Inference Sessions

```python
# Client 1: Watching early checkpoint
ws1 = WebSocket("ws://localhost:8765/ws")
ws1.send({"command": "load_model", "model": "checkpoint_ep50.pt"})
ws1.send({"command": "play"})

# Client 2: Watching trained agent
ws2 = WebSocket("ws://localhost:8765/ws")
ws2.send({"command": "load_model", "model": "trained_agent.pt"})
ws2.send({"command": "play"})

# Both run independently!
```

### Starting Training While Others Watch Inference

```python
# Some students watching inference
inference_clients = [...]

# Teacher starts training demo
training_client = WebSocket("ws://localhost:8765/ws/training")
training_client.send({
    "command": "start_training",
    "num_episodes": 50,
    "show_every": 1  # Show every episode
})

# Inference sessions continue uninterrupted!
```

---

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8765
lsof -i :8765

# Kill old processes
pkill -f demo_visualization
pkill -f demo_training_visualization

# Then restart metaserver
uv run python demo_metaserver.py
```

### Frontend Can't Connect
1. Verify metaserver is running: `curl http://localhost:8765`
2. Check browser console for errors
3. Ensure frontend is using correct WebSocket URLs
4. Try clearing browser cache

### Training Won't Start
- Check if training is already running: `curl http://localhost:8765`
- Look for `"training_active": true` in response
- Only one training session at a time (by design)
- Wait for current training to complete, or restart server

---

## Performance Considerations

### Inference
- **Lightweight**: Each session uses minimal resources
- **Scalable**: Supports 10+ concurrent sessions easily
- **Responsive**: AsyncIO ensures non-blocking operation

### Training
- **CPU-Intensive**: Uses one CPU core actively
- **Memory**: ~500MB for replay buffer and networks
- **Broadcast Efficient**: All clients share same data stream

### Recommendations
- For classrooms (20+ students): Run metaserver on decent CPU
- For large classes: Consider multiple metaserver instances
- For production: Scale horizontally with load balancer

---

## Development vs Production

### Development (Current)
```python
# CORS: Allow all origins
allow_origins=["*"]

# Host: All interfaces
host="0.0.0.0"

# Logging: Info level
log_level="info"
```

### Production (Recommended)
```python
# CORS: Specific frontend domain
allow_origins=["https://hamlet.yourdomain.com"]

# Host: Behind reverse proxy (nginx)
host="127.0.0.1"

# Logging: Warning level
log_level="warning"

# Add: Rate limiting, authentication, HTTPS
```

---

## Migration Guide

If you have existing scripts using old servers:

### Update Imports
```python
# Old
from hamlet.web.server import main

# New
from hamlet.web.metaserver import main
```

### Update Demo Scripts
```bash
# Old way (deprecated but still works)
uv run python demo_visualization.py
uv run python demo_training_visualization.py

# New way (recommended)
uv run python demo_metaserver.py
```

### Frontend Changes
No changes needed! Frontend already supports both endpoints.
The server availability detection automatically finds both modes.

---

## Future Enhancements

### Planned Features
- [ ] WebRTC for peer-to-peer student collaboration
- [ ] Persistent session storage (Redis)
- [ ] Checkpoint auto-saving during training
- [ ] Multi-agent training visualization
- [ ] Export training replay as video
- [ ] Distributed training across multiple servers

### Community Contributions Welcome!
See `CONTRIBUTING.md` for guidelines.

---

## Support

**Issues**: https://github.com/yourusername/hamlet/issues
**Documentation**: https://hamlet-drl.readthedocs.io
**Discord**: https://discord.gg/hamlet-drl
