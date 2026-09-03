import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

ROLLING_WINDOWS = ['1h', '3h', '6h']
SENSORS = ['temperature', 'pressure', 'humidity']

# A small delta to consider values "unchanged" for the frozen sensor detection
# We'll use 0.0 as exact match is required for purely frozen, but we'll also compute a near-unchanged count
UNCHANGED_TOLERANCES = {
    'temperature': 0.05,
    'pressure': 0.05,
    'humidity': 0.1
}

# Expected baseline interval (in seconds)
EXPECTED_INTERVAL_SEC = 600 # 10 mins
