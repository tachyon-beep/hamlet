"""Polling utilities for async/threading tests.

This module provides utilities for waiting on conditions in tests,
particularly useful for:
- Testing threaded code (EpisodeRecorder, RecordingWriter)
- Waiting for async operations to complete
- Any scenario requiring polling for state changes

See Also:
    - builders.py: Test data construction
    - assertions.py: Test assertion helpers
"""

import time


def wait_for_condition(condition_fn, timeout=2.0, poll_interval=0.01):
    """Wait for a condition to become true with timeout.

    This is a proper way to wait for background thread processing in tests,
    avoiding flaky time.sleep() calls.

    Args:
        condition_fn: Callable that returns True when condition is met
        timeout: Maximum time to wait in seconds (default: 2.0)
        poll_interval: How often to check the condition in seconds (default: 0.01)

    Returns:
        True if condition was met, False if timeout occurred

    Example:
        # Wait for background thread to process item
        assert wait_for_condition(lambda: len(writer.episode_buffer) == 1, timeout=2.0)

        # Wait for queue to be empty
        success = wait_for_condition(lambda: recorder.queue.empty(), timeout=1.0)
        assert success, "Queue did not empty within timeout"

    Use Cases:
        - Testing threaded code (EpisodeRecorder, RecordingWriter)
        - Waiting for async operations to complete
        - Any scenario where you need to poll for a state change

    Rationale:
        Arbitrary time.sleep() calls in tests are antipatterns:
        - Tests are flaky (sleep too short → false failures)
        - Tests are slow (sleep too long → wasted time)
        - This function polls intelligently and exits early when condition is met
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_fn():
            return True
        time.sleep(poll_interval)
    return False
