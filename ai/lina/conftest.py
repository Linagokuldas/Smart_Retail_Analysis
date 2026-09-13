"""
pytest configuration for Lina's analytics module tests.

Adds the ai/lina root to sys.path so that `from src.xxx import yyy`
works correctly when pytest is run from the ai/lina directory.
"""

import sys
from pathlib import Path

# Ensure the lina root is on the path for all test files
LINA_ROOT = Path(__file__).resolve().parent
if str(LINA_ROOT) not in sys.path:
    sys.path.insert(0, str(LINA_ROOT))
