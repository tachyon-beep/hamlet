"""Temporal mechanics utilities.

Canonical implementation of time-based logic for affordance operating hours.
This module provides the single source of truth for temporal semantics.

IMPORTANT: All time-based logic MUST use these functions. Do not re-implement
operating hours logic elsewhere - it creates drift and bugs (see JANK-09).
"""


def is_affordance_open(time_of_day: int, operating_hours: tuple[int, int]) -> bool:
    """Check if affordance is open at given time (CANONICAL IMPLEMENTATION).

    This is the single source of truth for operating hours logic.
    All other implementations (AffordanceEngine, compiler, env) must delegate to this.

    Supports three notations:
    - Normal hours: [8, 18] = 8am to 6pm (open_hour < close_hour)
    - Wraparound (modulo): [18, 28] = 6pm to 4am (close_hour > 24)
    - Wraparound (negative): [22, 6] = 10pm to 6am (open_hour > close_hour)
    - 24/7: [0, 24] or any interval spanning 24 hours

    Examples:
        >>> is_affordance_open(12, (8, 18))  # Noon, normal hours
        True
        >>> is_affordance_open(20, (18, 28))  # 8pm, bar open 6pm-4am
        True
        >>> is_affordance_open(2, (18, 28))   # 2am, bar still open
        True
        >>> is_affordance_open(23, (22, 6))   # 11pm, late night
        True
        >>> is_affordance_open(1, (22, 6))    # 1am, late night
        True
        >>> is_affordance_open(12, (0, 24))   # 24/7 operation
        True

    Args:
        time_of_day: Current hour [0-23]
        operating_hours: Tuple of (open_hour, close_hour)

    Returns:
        True if affordance is open, False if closed

    Implementation notes:
        - Uses modulo arithmetic to handle all wraparound cases uniformly
        - Detects 24/7 operation: (close - open) % 24 == 0
        - Normalizes all hours to [0-23] range before comparison
        - Handles both [18, 28] and [22, 6] notations correctly

    History:
        - 2025-11-14: Extracted from UniverseCompiler._is_open() to fix JANK-09
        - Previously had 3 implementations with inconsistencies:
          * affordance_config.is_affordance_open() - failed on [18, 28]
          * AffordanceEngine.is_affordance_open() - failed on [22, 6]
          * UniverseCompiler._is_open() - correct (became this function)
    """
    open_hour, close_hour = operating_hours

    # Normalize time_of_day to [0-23]
    hour = time_of_day % 24

    # Normalize open/close hours to [0-23]
    open_mod = open_hour % 24
    close_mod = close_hour % 24

    # 24/7 operation: interval spans full day
    if (close_hour - open_hour) % 24 == 0:
        return True

    # Normal vs wraparound hours
    if open_mod < close_mod:
        # Normal hours: 8-18, 0-12, etc.
        return open_mod <= hour < close_mod
    else:
        # Wraparound: 22-6 or 18-4 (after modulo normalization)
        return hour >= open_mod or hour < close_mod
