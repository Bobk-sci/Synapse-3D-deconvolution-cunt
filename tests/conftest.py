import sys
from pathlib import Path

# Make the package and the fixture helpers importable without installing.
ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)
