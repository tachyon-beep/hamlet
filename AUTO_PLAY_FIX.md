# Auto-Play Fix Documentation

## Problem

After removing the play/pause buttons from the UI (to create an auto-playing "TV station" experience), the visualization completely broke:
- WebSocket connection succeeded
- But no grid, meters, or affordances appeared
- No data was flowing from backend to frontend

## Root Cause

The backend `live_inference.py` server requires an explicit `play` command to start the inference loop:

```python
async def _handle_command(self, websocket: WebSocket, data: dict):
    command = data.get('command') or data.get('type')
    if command == 'play':
        if not self.is_running:
            self.is_running = True
            asyncio.create_task(self._run_inference_loop())
```

**Before the fix:**
1. Frontend connects → Backend sends `connected` message
2. User clicks Play button → Frontend sends `{type: 'control', command: 'play'}`
3. Backend starts `_run_inference_loop()` → Data flows

**After removing play button (broken):**
1. Frontend connects → Backend sends `connected` message
2. **Nothing sends play command** → Backend never starts loop
3. No data flows → Empty UI

## Solution

Add auto-play logic to the frontend store's `connect()` method. After the WebSocket connection is established, automatically send the `play` command:

```javascript
ws.value.onopen = () => {
  console.log('WebSocket connected')
  isConnected.value = true
  isConnecting.value = false
  reconnectAttempts.value = 0

  // Auto-start simulation (no manual play button needed)
  // Wait for next tick to ensure WebSocket is fully ready
  setTimeout(() => {
    console.log('Auto-starting simulation...')
    sendCommand('play')
  }, 100)
}
```

The 100ms delay ensures the WebSocket is fully initialized before sending commands.

## Files Modified

- **`frontend/src/stores/simulation.js`**: Added auto-play logic in `ws.value.onopen` handler

## Testing

Created test infrastructure to verify the fix:

1. **`create_test_checkpoint.py`**: Creates minimal checkpoint for UI testing without full training
2. **`test_auto_play.py`**: WebSocket test client that verifies:
   - Connection succeeds
   - Play command is sent
   - Episode starts
   - State updates flow correctly

Test results:
```
✓ Connected successfully
✓ Received: connected
✓ Sent play command
✓ Episode 1 started
✓ First state update received
  - Step: 1
  - Grid: 8x8
  - Agents: 1
  - Affordances: 14
✓ Received 49 state updates
✓ Test passed!
```

## User Experience Flow (After Fix)

1. User opens frontend in browser
2. User clicks "Connect" button
3. **Automatically**:
   - WebSocket connects to backend
   - `play` command sent immediately
   - Simulation starts running
   - Grid, meters, affordances appear
   - Updates flow in real-time
4. User can still control:
   - Speed (slider)
   - Disconnect (button)

No manual play/pause needed - it's a continuous "TV station" stream.

## Design Philosophy

This aligns with the "TV station" mental model:
- Turning on a TV (connecting) immediately shows content (auto-play)
- You can change the channel (disconnect/reconnect)
- You can adjust volume (speed)
- But you don't "play" or "pause" a live broadcast

The simulation is always running on the backend; connecting to it should immediately show what's happening.
