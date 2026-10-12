"""Runtime asset paths (configured by the CLI before importing task modules)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(
    os.environ.get("RESYNC_ISAAC_ASSETS", ROOT / "inference_assets/isaac")
)
