"""Test script to verify auto-play functionality after connection."""

import asyncio
import json
import websockets
import sys

async def test_auto_play():
    """Test that backend auto-starts after connection."""

    uri = "ws://localhost:8766/ws"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully")

            # Receive connection message
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(msg)
            print(f"✓ Received: {data['type']}")

            if data['type'] == 'connected':
                print(f"  - Checkpoint: {data.get('checkpoint', 'None')}")
                print(f"  - Episode: {data.get('checkpoint_episode', 0)}")
                print(f"  - Epsilon: {data.get('epsilon', 0.0)}")

            # Send play command (simulating what frontend auto-play will do)
            play_cmd = {
                'type': 'control',
                'command': 'play'
            }
            await websocket.send(json.dumps(play_cmd))
            print("✓ Sent play command")

            # Wait for messages
            print("\nWaiting for state updates (10 seconds)...")
            message_count = 0
            episode_started = False
            state_updates = 0

            try:
                while message_count < 50:  # Limit to 50 messages
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(msg)
                    message_count += 1

                    if data['type'] == 'episode_start':
                        episode_started = True
                        print(f"✓ Episode {data['episode']} started")

                    elif data['type'] == 'state_update':
                        state_updates += 1
                        if state_updates == 1:
                            print(f"✓ First state update received")
                            print(f"  - Step: {data.get('step', 0)}")
                            print(f"  - Grid: {data.get('grid', {}).get('width', 0)}x{data.get('grid', {}).get('height', 0)}")
                            print(f"  - Agents: {len(data.get('grid', {}).get('agents', []))}")
                            print(f"  - Affordances: {len(data.get('grid', {}).get('affordances', []))}")
                        elif state_updates % 10 == 0:
                            print(f"✓ Received {state_updates} state updates...")

                    elif data['type'] == 'episode_end':
                        print(f"✓ Episode {data['episode']} ended")
                        print(f"  - Steps: {data.get('steps', 0)}")
                        print(f"  - Reward: {data.get('total_reward', 0.0):.2f}")
                        break

            except asyncio.TimeoutError:
                print("✗ Timeout waiting for messages")
                return False

            print(f"\n✓ Test passed!")
            print(f"  - Total messages: {message_count}")
            print(f"  - Episode started: {episode_started}")
            print(f"  - State updates: {state_updates}")

            return episode_started and state_updates > 0

    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_auto_play())
    sys.exit(0 if success else 1)
