"""Run optional IsaacLab demos in a separate Isaac environment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from resync_isaac.cli import main

    main()
