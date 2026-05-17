"""Root conftest — ensures the workspace root is on sys.path so 'import src' works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
