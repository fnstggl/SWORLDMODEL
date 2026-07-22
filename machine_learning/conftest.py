"""Ensure the repository root is importable so `import machine_learning` works whether
pytest is run from the repo root or from inside machine_learning/."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
