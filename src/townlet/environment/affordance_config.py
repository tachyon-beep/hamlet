# src/townlet/environment/affordance_config.py
"""
Affordance configuration for multi-interaction and time-based mechanics.

Each affordance specifies:
- required_ticks: Number of consecutive INTERACTs for full benefit
- cost_per_tick: Money charged per tick (normalized [0, 1])
- operating_hours: (open_tick, close_tick) in [0, 23]
- benefits:
  - linear: 75% of total benefit, distributed per tick
  - completion: 25% bonus on full completion
"""

from typing import Dict, Any, Tuple


AffordanceConfig = Dict[str, Any]


AFFORDANCE_CONFIGS: Dict[str, AffordanceConfig] = {
    # === Static Affordances (24/7) ===

    'Bed': {
        'required_ticks': 5,
        'cost_per_tick': 0.01,  # $1 per tick ($5 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'energy': +0.075,  # Per tick: (50% * 0.75) / 5
            },
            'completion': {
                'energy': +0.125,  # 50% * 0.25
                'health': +0.02,
            }
        }
    },

    'LuxuryBed': {
        'required_ticks': 5,
        'cost_per_tick': 0.022,  # $2.20 per tick ($11 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'energy': +0.1125,  # Per tick: (75% * 0.75) / 5
            },
            'completion': {
                'energy': +0.1875,  # 75% * 0.25
                'health': +0.05,
            }
        }
    },

    'Shower': {
        'required_ticks': 3,
        'cost_per_tick': 0.01,  # $1 per tick ($3 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'hygiene': +0.10,  # Per tick: (40% * 0.75) / 3
            },
            'completion': {
                'hygiene': +0.10,  # 40% * 0.25
            }
        }
    },

    'HomeMeal': {
        'required_ticks': 2,
        'cost_per_tick': 0.015,  # $1.50 per tick ($3 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'satiation': +0.16875,  # Per tick: (45% * 0.75) / 2
            },
            'completion': {
                'satiation': +0.1125,  # 45% * 0.25
                'health': +0.03,
            }
        }
    },

    'Hospital': {
        'required_ticks': 3,
        'cost_per_tick': 0.05,  # $5 per tick ($15 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'health': +0.225,  # Per tick: (60% * 0.75) / 3
            },
            'completion': {
                'health': +0.15,  # 60% * 0.25
            }
        }
    },

    'Gym': {
        'required_ticks': 4,
        'cost_per_tick': 0.02,  # $2 per tick ($8 total)
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'fitness': +0.1125,  # Per tick: (30% * 0.75) / 4
                'energy': -0.03,
            },
            'completion': {
                'fitness': +0.075,  # 30% * 0.25
                'mood': +0.05,
            }
        }
    },

    'FastFood': {
        'required_ticks': 1,
        'cost_per_tick': 0.10,  # $10
        'operating_hours': (0, 24),
        'benefits': {
            'linear': {
                'satiation': +0.3375,  # (45% * 0.75) / 1
                'energy': +0.1125,
            },
            'completion': {
                'satiation': +0.1125,  # 45% * 0.25
                'energy': +0.0375,
                'fitness': -0.03,
                'health': -0.02,
            }
        }
    },

    # === Business Hours Affordances (8am-6pm) ===

    'Job': {
        'required_ticks': 4,
        'cost_per_tick': 0.0,
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'money': +0.140625,  # Per tick: ($22.5 * 0.75) / 4
                'energy': -0.0375,
            },
            'completion': {
                'money': +0.05625,  # $22.5 * 0.25
                'social': +0.02,
                'health': -0.03,
            }
        }
    },

    'Labor': {
        'required_ticks': 4,
        'cost_per_tick': 0.0,
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'money': +0.1875,  # Per tick: ($30 * 0.75) / 4
                'energy': -0.05,
            },
            'completion': {
                'money': +0.075,  # $30 * 0.25
                'fitness': -0.05,
                'health': -0.05,
                'social': +0.01,
            }
        }
    },

    'Doctor': {
        'required_ticks': 2,
        'cost_per_tick': 0.04,  # $4 per tick ($8 total)
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'health': +0.1125,  # Per tick: (30% * 0.75) / 2
            },
            'completion': {
                'health': +0.075,  # 30% * 0.25
            }
        }
    },

    'Therapist': {
        'required_ticks': 3,
        'cost_per_tick': 0.05,  # $5 per tick ($15 total)
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'mood': +0.15,  # Per tick: (40% * 0.75) / 3
            },
            'completion': {
                'mood': +0.10,  # 40% * 0.25
                'social': +0.05,
            }
        }
    },

    'Recreation': {
        'required_ticks': 2,
        'cost_per_tick': 0.03,  # $3 per tick ($6 total)
        'operating_hours': (8, 22),
        'benefits': {
            'linear': {
                'mood': +0.1125,  # Per tick: (30% * 0.75) / 2
                'social': +0.075,
            },
            'completion': {
                'mood': +0.075,  # 30% * 0.25
                'social': +0.05,
            }
        }
    },

    # === Dynamic Affordances (Time-Dependent) ===

    'CoffeeShop': {
        'required_ticks': 1,
        'cost_per_tick': 0.02,  # $2
        'operating_hours': (8, 18),
        'benefits': {
            'linear': {
                'energy': +0.1125,  # (15% * 0.75) / 1
                'mood': +0.0375,
                'social': +0.045,
            },
            'completion': {
                'energy': +0.0375,  # 15% * 0.25
                'mood': +0.0125,
                'social': +0.015,
            }
        }
    },

    'Bar': {
        'required_ticks': 2,
        'cost_per_tick': 0.075,  # $7.50 per round ($15 total)
        'operating_hours': (18, 4),  # Wraps midnight
        'benefits': {
            'linear': {
                'mood': +0.075,  # Per tick: (20% * 0.75) / 2
                'social': +0.05625,
                'health': -0.01875,
            },
            'completion': {
                'mood': +0.05,  # 20% * 0.25
                'social': +0.0375,
                'health': -0.0125,
            }
        }
    },

    'Park': {
        'required_ticks': 2,
        'cost_per_tick': 0.0,
        'operating_hours': (6, 22),
        'benefits': {
            'linear': {
                'mood': +0.0975,  # Per tick: (26% * 0.75) / 2
                'social': +0.0375,
            },
            'completion': {
                'mood': +0.065,  # 26% * 0.25
                'social': +0.025,
                'fitness': +0.02,
            }
        }
    },
}


# Meter name to index mapping
METER_NAME_TO_IDX = {
    'energy': 0,
    'hygiene': 1,
    'satiation': 2,
    'money': 3,
    'mood': 4,
    'social': 5,
    'health': 6,
    'fitness': 7,
}


def is_affordance_open(time_of_day: int, operating_hours: Tuple[int, int]) -> bool:
    """
    Check if affordance is open at given time.

    Handles midnight wraparound (e.g., Bar: 18-4 means 6pm to 4am).

    Args:
        time_of_day: Current tick [0-23]
        operating_hours: (open_tick, close_tick)

    Returns:
        True if open, False if closed
    """
    open_tick, close_tick = operating_hours

    if open_tick < close_tick:
        # Normal hours (e.g., 8-18)
        return open_tick <= time_of_day < close_tick
    else:
        # Wraparound hours (e.g., 18-4 = 6pm to 4am)
        return time_of_day >= open_tick or time_of_day < close_tick
