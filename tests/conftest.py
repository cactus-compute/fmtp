"""
Pytest configuration for nanochat tests.
Adds the project root to sys.path so that nanochat module can be imported.
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
