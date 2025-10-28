# Metaserver Architecture Design

## Problem Statement

Currently, Hamlet has two separate servers:
- **Inference Server** (`server.py`): Runs trained agents, handles `/ws` endpoint
- **Training Server** (`training_server.py`): Live training visualization, handles `/ws/training` endpoint

Both use port 8765, so only one can run at a time. This creates a poor UX where users must manually stop one server and start the other to switch modes.

## Design Goals

1. **Single Port, Multiple Modes**: One server handling both inference and training
2. **Concurrent Sessions**: Multiple users can run different modes simultaneously
3. **Resource Isolation**: Training doesn't block inference and vice versa
4. **Clean Architecture**: Maintainable, testable, follows SOLID principles
5. **Production-Ready**: Proper error handling, graceful shutdown, logging
6. **Pedagogical Focus**: Simple enough for students to understand

## Architectural Options Considered

### Option 1: Unified Server with AsyncIO Task Management ✅ RECOMMENDED

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Metaserver (Port 8765)           │
├─────────────────────────────────────────────────────────────┤
│  HTTP Endpoints                                              │
│  ├─ GET  /                  Health check                    │
│  ├─ GET  /api/status        Server status & active sessions │
│  └─ GET  /api/models        Available trained models        │
├─────────────────────────────────────────────────────────────┤
│  WebSocket Endpoints                                         │
│  ├─ /ws                     Inference mode (per-session)    │
│  └─ /ws/training            Training mode (singleton)       │
├─────────────────────────────────────────────────────────────┤
│  Session Management                                          │
│  ├─ InferenceSessionManager  Multiple concurrent sessions   │
│  │   └─ Each session: independent env, agent, state        │
│  └─ TrainingBroadcaster      Single shared training session │
│      └─ Broadcasts to all /ws/training clients             │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

1. **Inference = Multi-Session**
   - Each WebSocket connection gets its own environment instance
   - Users can watch different trained models simultaneously
   - Isolated state prevents interference
   - Resource cleanup on disconnect

2. **Training = Singleton Broadcast**
   - One training task running at a time (compute-intensive)
   - All connected clients receive the same broadcast
   - Additional clients can "tune in" to ongoing training
   - First client starts training, others observe

3. **AsyncIO for Concurrency**
   - Python's native async/await for I/O-bound operations
   - FastAPI's built-in async support
   - Training runs as background AsyncIO task
   - Non-blocking: inference sessions don't wait for training

4. **Clean Separation via Modules**
   ```
   hamlet/web/
   ├── metaserver.py          # Main unified server
   ├── inference_manager.py   # Inference session management
   ├── training_manager.py    # Training broadcast management
   ├── websocket.py           # Shared WebSocket utilities
   └── simulation_runner.py   # Async simulation runner (reusable)
   ```

**Pros:**
- ✅ Simple deployment (one process, one port)
- ✅ Efficient resource usage (async I/O)
- ✅ Clean code separation
- ✅ Python best practices (AsyncIO, FastAPI)
- ✅ Easy to test and maintain
- ✅ Perfect for pedagogical tool

**Cons:**
- ⚠️ Training and inference share same Python process
- ⚠️ CPU-bound training could impact inference responsiveness (mitigated by AsyncIO)

---

### Option 2: Metaserver with Child Process Spawning

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│              Metaserver (Port 8765) - Orchestrator           │
│  ├─ Routes /ws → Inference child process                    │
│  └─ Routes /ws/training → Training child process            │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────────┐        ┌──────────────────────────┐
│ Inference Process    │        │ Training Process         │
│ (multiprocessing)    │        │ (multiprocessing)        │
│ - Separate CPU core  │        │ - Separate CPU core      │
│ - Independent memory │        │ - Independent memory     │
└──────────────────────┘        └──────────────────────────┘
```

**Pros:**
- ✅ True process isolation
- ✅ CPU cores allocated separately
- ✅ Training can't impact inference performance
- ✅ Fault isolation (one crash doesn't kill the other)

**Cons:**
- ❌ More complex IPC (inter-process communication)
- ❌ Higher memory overhead (duplicate imports)
- ❌ Process management complexity
- ❌ Harder to debug
- ❌ Overkill for pedagogical use case

---

### Option 3: Microservices with Reverse Proxy

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│           Nginx/Traefik Reverse Proxy (Port 8765)           │
│  ├─ /ws          → Inference Service (Port 8766)           │
│  └─ /ws/training → Training Service (Port 8767)            │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- ✅ True microservices architecture
- ✅ Independently scalable
- ✅ Production-grade deployment
- ✅ Can use different tech stacks

**Cons:**
- ❌ Requires additional infrastructure (nginx/traefik)
- ❌ Docker/Kubernetes for proper deployment
- ❌ Much more complex for students to run
- ❌ Overkill for single-machine pedagogical tool

---

## Recommended Solution: Option 1 (Unified AsyncIO Server)

For Hamlet's pedagogical use case, **Option 1** is the clear winner:

### Implementation Strategy

1. **Create `metaserver.py`**
   - Combines both endpoint handlers
   - Routes `/ws` to inference manager
   - Routes `/ws/training` to training broadcaster
   - Single `uvicorn.run()` call

2. **Refactor Existing Code**
   - Extract `InferenceSessionManager` from `server.py`
   - Extract `TrainingBroadcaster` from `training_server.py`
   - Keep `WebSocketManager` as shared utility
   - Reuse `SimulationRunner` logic

3. **Session Management**
   ```python
   # Inference: per-connection state
   class InferenceSessionManager:
       sessions: Dict[str, InferenceSession]

       async def create_session(websocket) -> str:
           session_id = uuid4()
           session = InferenceSession(env, agent, websocket)
           sessions[session_id] = session
           return session_id

   # Training: singleton broadcast
   class TrainingBroadcaster:
       connections: Set[WebSocket]
       training_task: Optional[Task] = None

       async def start_training(...):
           if self.training_task:
               return  # Already running
           self.training_task = asyncio.create_task(...)
   ```

4. **Graceful Degradation**
   - If training is running, inference still works
   - If training is requested while running, return status
   - Clean shutdown cancels all tasks

### Benefits for Pedagogy

1. **Students see both modes in one codebase**
2. **AsyncIO patterns are industry-standard**
3. **Clean separation of concerns**
4. **Easy to run and debug**
5. **Production-quality architecture without over-engineering**

### Migration Path

If performance becomes an issue (unlikely for pedagogical use):
1. Profile with `py-spy` or `cProfile`
2. Move training to separate process if needed (Option 2)
3. Add process pool for multiple training sessions
4. Eventually scale to Option 3 for production SaaS

## Implementation Checklist

- [ ] Create `src/hamlet/web/metaserver.py`
- [ ] Extract `InferenceSessionManager` class
- [ ] Extract `TrainingBroadcaster` class (already exists)
- [ ] Add session ID tracking for inference
- [ ] Add health check endpoint with status
- [ ] Add `/api/models` endpoint for available models
- [ ] Update `demo_visualization.py` to use metaserver
- [ ] Add comprehensive error handling
- [ ] Add graceful shutdown for all sessions
- [ ] Write integration tests
- [ ] Update documentation

## Success Criteria

- ✅ Single server runs both inference and training
- ✅ Multiple users can watch different trained agents simultaneously
- ✅ One user can start training while others watch inference
- ✅ Frontend automatically detects both endpoints available
- ✅ Clean shutdown with no resource leaks
- ✅ Comprehensive logging for debugging
