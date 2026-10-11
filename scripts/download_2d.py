"""Download and verify the Blocked Stacking inference assets."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(Path(__file__).with_name("download_release_assets.py")),
                "--domains",
                "blocked_stacking",
                *sys.argv[1:],
            ]
        )
    )
